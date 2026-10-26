"""strange — Doctor Strange sling-ring portal ending (pre-glow 69.99 s, EV['strange'] 70.05 s -> end).

Choreography (from a frame-by-frame study of his R hand, tracking + footage):
  69.99-70.10  pre-glow: a flickering ember on his two raised fingertips (anticipation of the ignition)
  70.10        ignition flash (short, tilted glint + halo); a white-hot head orbits the fingertips and DRAWS
               the ring (rider sparks lag the head, so the trail closes into a circle by ~70.45)
  70.62-72.26  small sparkler ring (R 44 -> 80) at the fingertips, IN FRONT of him. What is under the ring is
               darkened a little so it reads over the white captions. A tight hot light at the ring flickers
               on his hand and face with the crackle bursts (word onsets).
  72.27        depth switch in ONE frame, at the measured minimum overlap of the band with his face. From then
               on the ring, its riders and every spark born over his silhouette live in the BACKGROUND
               (behind captions and him); only ~15 % of sparks born outside him fly in front.
  72.36-73.30  growth 80 -> 370 px (damped spring: lag, overshoot, settle); the centre migrates behind his head
  72.56-72.98  the aperture BURNS open from the centre outward (noisy burn front with a hot gold edge + sparks)
               onto the original bathroom (plate 0)
  72.85-74.12  his circling fingertips stir the rim: hot clump at the hand's phase, spin nudged by its speed
  73.60-73.82  breath-in: the portal contracts 12 %, dims, riders/sparks are sucked inward
  73.82        release: energy pulse + spark burst, accelerating expansion; the ring leaves the frame ~74.42
  73.82-74.15  reality front from the portal centre: the raw footage (relit by the ring) replaces everything;
               a gold burn line + embers on the suit only (thin and dim on face/hands)
  74.70-end    raw footage (exact: inverse-tonemapped, bloom/vignette/grain faded); a few hot embers shed where
               the rim left the frame fade by 74.95

Light: tight hot light on the hand for the small ring; for the portal a few weak lights at the ring points
nearest his silhouette (smoothly weighted, small rim) plus a crisp full-res LIGHT WRAP (blurred ring + bathroom
behind him, applied on a 9 px inner band of the matte) done here in post(); the raw person restored by the wipe
is relit with the same lights (core._relight on a scratch scene).

Everything is an analytic function of t: spark emission times / initial states are precomputed once in
__init__ from a seeded RNG and the (smoothed) tracking; positions are evaluated in closed form (drag + gravity).
Rendering: GLSL procedural hot ring band, GLSL burn-through aperture, instanced tapered streak sprites.
"""
import types
from functools import lru_cache

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter1d, median_filter

from .. import config as C
from .. import assets as A
from .. import core
from ..core import Light
from ..glfx import gl

EV = C.EV
W, H = C.W, C.H
T0 = EV['strange'][0]                  # 70.05
T_PRE = 69.99                          # ignition pre-glow starts 2 frames before the module window
TW0, TW1 = EV['strange_wipe']          # 73.60, 74.70
T_IGN = 70.08                          # first spark
T_GROW0, T_GROW1 = 72.45, 73.30        # portal growth while he circles
T_SWITCH = 72.26                       # ring goes behind him (measured: band/face overlap minimum)
T_AP0, T_AP1 = 72.56, 72.98            # aperture burns open
T_REL = 73.82                          # wipe release (after the breath-in TW0..T_REL)
T_FRONT1 = 74.15                       # reality front reaches the rim
T_STIR0, T_STIR1 = 72.85, 74.12        # his circling hand stirs the big ring
CF = np.array([460.0, 930.0])          # final portal centre (behind his head, slightly frame-left)
R_SMALL0, R_SMALL1, R_OPEN = 44.0, 80.0, 370.0
GRAV = 1900.0                          # px/s^2
SHUTTER = 0.030                        # streak length in seconds of motion
WOB_A = np.array([0.020, 0.015, 0.009, 0.006])      # rim wobble amplitudes (fraction of R)
WOB_K = np.array([2.0, 3.0, 5.0, 7.0])
WOB_NU = np.array([1.7, -2.3, 3.1, -4.4])           # rad/s
WOB_P0 = np.array([0.3, 2.1, 4.0, 1.2])
WORD_PULSES = (70.10, 71.06, 71.86, 72.10, 72.28, 72.66)
GOLD = (1.0, 0.50, 0.15)
# emission gains (linear HDR)
K_RING = 0.36      # GLSL ring band
K_RIDER = 0.12     # sparks riding the band
K_SPARK = 0.85     # flying sparks
K_WRAP_RING = 0.8  # light wrap from the ring behind him
K_WRAP_PLATE = 0.16  # light wrap from the bright bathroom seen through the portal
DT_TAB = 1.0 / 1200.0


def _sm(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def _ss(a, b, x):
    return _sm((np.asarray(x, np.float64) - a) / (b - a))


def _pulse(t, t0, decay):
    t = np.asarray(t, np.float64)
    return _ss(t0 - 0.035, t0, t) * np.exp(-np.maximum(t - t0, 0.0) / decay)


def _gather(t):
    """Breath-in before the wipe burst: 0 -> 1 over TW0..T_REL (ease in-out), snaps back after."""
    t = np.asarray(t, np.float64)
    u = np.clip((t - TW0) / (T_REL - TW0), 0, 1)
    g = np.where(u < 0.5, 4 * u ** 3, 1 - (-2 * u + 2) ** 3 / 2)
    return np.where(t < T_REL, g, np.exp(-np.maximum(t - T_REL, 0.0) / 0.035))


def _front_R(t, R):
    """Radius of the reality front (raw footage inside) during the wipe."""
    t = np.asarray(t, np.float64)
    return np.where(t >= T_REL, R * _ss(T_REL, T_FRONT1, t) ** 0.85, 0.0)


def _pre(t):
    """Ignition pre-glow envelope (0..1)."""
    return float(_ss(T_PRE, 70.09, t) * (1 - _ss(70.10, 70.15, t)))


def _hash(ids, frame, salt=0):
    """Deterministic per-(id, frame) uniform in [0,1)."""
    x = (np.asarray(ids, np.uint64) * np.uint64(0x9E3779B1) + np.uint64(frame * 0x85EBCA77 + salt * 0xC2B2AE3D + 12345))
    x ^= x >> np.uint64(15)
    x *= np.uint64(0x2C1B3C6D)
    x ^= x >> np.uint64(12)
    x *= np.uint64(0x297A2D39)
    x ^= x >> np.uint64(15)
    return (x & np.uint64(0xFFFFFF)).astype(np.float64) / float(0x1000000)


def _vnoise1(x, salt=0):
    """Smooth 1D value noise in [0,1] (numpy)."""
    x = np.asarray(x, np.float64)
    i0 = np.floor(x)
    f = x - i0
    f = f * f * (3 - 2 * f)
    ii = (i0.astype(np.int64) % 100003).astype(np.uint64)
    a = _hash(ii, 0, salt + 11)
    b = _hash(ii + np.uint64(1), 0, salt + 11)
    return a + (b - a) * f


@lru_cache(maxsize=512)
def _alpha_lo(i):
    """Low-res (360x480) matte, used for cheap birth-point occlusion tests in __init__."""
    i = int(np.clip(i, 0, C.N_FRAMES - 1))
    a = cv2.imread(f'{C.ALPHA_LO}/{i:04d}.png', cv2.IMREAD_GRAYSCALE)
    if a is None:
        return np.zeros((C.SRC_H, C.SRC_W), np.float32)
    return a.astype(np.float32) / 255.0


def _sample_lo(te, p, fn):
    """Sample a per-frame low-res mask fn(i) at frame-px positions p (frame of each te)."""
    out = np.zeros(len(te))
    if len(te) == 0:
        return out
    fr = np.rint(np.asarray(te) * C.FPS).astype(int)
    for fi in np.unique(fr):
        m = fr == fi
        a = fn(int(fi))
        q = p[m] / C.SCALE
        ok = (q[:, 0] >= 0) & (q[:, 0] < C.SRC_W) & (q[:, 1] >= 0) & (q[:, 1] < C.SRC_H)
        xi = np.clip(q[:, 0].astype(int), 0, C.SRC_W - 1)
        yi = np.clip(q[:, 1].astype(int), 0, C.SRC_H - 1)
        out[np.nonzero(m)[0]] = np.where(ok, a[yi, xi], 0.0)
    return out


def _skin_polys(i, off=(0.0, 0.0)):
    """Face oval (widened ~15 %, incl. ears/hairline) and hand hulls, frame px (+ offset)."""
    polys = []
    fo = A.face_oval(i).astype(np.float64)
    c = fo.mean(0)
    polys.append((c + (fo - c) * np.array([1.18, 1.12])) + off)
    # hair cap (the suit never covered his hair): ellipse from the top of the hair down to the brow
    hx0, htop, hx1, _ = A.head_box(i)
    brow = fo[:, 1].min()
    ec = (int(round((hx0 + hx1) / 2)), int(round((htop + brow) / 2 + 10)))
    ax = (int(round((hx1 - hx0) / 2 * 1.05)), int(round(max((brow - htop) / 2 + 28, 20))))
    polys.append(np.asarray(cv2.ellipse2Poly(ec, ax, 0, 0, 360, 10), np.float64) + off)
    for side in 'RL':
        h = A.hand(i, side)
        if h is not None:
            hull = cv2.convexHull(h[:, :2].astype(np.float32)).reshape(-1, 2).astype(np.float64)
            hc = hull.mean(0)
            polys.append(hc + (hull - hc) * 1.25 + off)
    return polys


@lru_cache(maxsize=64)
def _suit_lo(i):
    """Low-res mask of the suit region (matte minus dilated face and hands)."""
    a = _alpha_lo(i)
    m = (a > 0.6).astype(np.uint8)
    for poly in _skin_polys(i):
        cv2.fillPoly(m, [np.round(poly / C.SCALE).astype(np.int32)], 0)
    return m.astype(np.float32)


# ------------------------------------------------------------------ scalar timeline
def _R_scalar(t):
    if t < T_IGN:
        return 0.0
    if t < 70.62:
        return R_SMALL0 * (1 - (1 - (t - T_IGN) / 0.54) ** 3)
    if t < T_GROW0:
        u = (t - 70.62) / (T_GROW0 - 70.62)
        breath = 2.5 * np.sin(2 * np.pi * 1.6 * (t - 70.62)) * (1 - _ss(72.15, T_GROW0, t))
        return R_SMALL0 + (R_SMALL1 - R_SMALL0) * float(_sm(u)) + float(breath)
    if t < T_GROW1:
        u = np.clip((t - 72.36) / 0.86, 0, 1)
        dip = 9.0 * np.sin(np.pi * np.clip((t - 72.36) / 0.20, 0, 1)) ** 2
        return R_SMALL1 + (R_OPEN - R_SMALL1) * float(u * u * (3 - 2 * u)) - float(dip)
    r = R_OPEN + 5.0 * np.sin(2 * np.pi * 1.3 * (t - T_GROW1)) * (1 - float(_ss(TW0 - 0.12, TW0, t)))
    # wipe: breath-in (12 %), then a snap + accelerating expansion past the frame edges
    r *= 1 - 0.12 * float(_gather(t))
    ue = t - T_REL
    if ue > 0:
        r += 60.0 * (1 - np.exp(-ue / 0.05)) + 1050.0 * (ue / 0.66) ** 2.2
    return float(r)


def _I_scalar(t):
    i = 0.95 * float(_ss(T_IGN - 0.01, 70.24, t))
    i += 0.35 * float(_ss(T_GROW0, 73.20, t))
    i *= 1 - 0.25 * float(_gather(t))                   # dims while it gathers
    i += 0.55 * float(_pulse(t, T_REL, 0.12))          # energy release as the wipe bursts outward
    for k, tp in enumerate(WORD_PULSES):
        i += (1.4 if k == 0 else 0.45) * float(_pulse(t, tp, 0.09 if k == 0 else 0.14))
    i *= 1 - float(_ss(74.42, 74.62, t))
    return i


def _E_scalar(t, R):
    """Spark emission rate (1/s)."""
    e = 700.0 * float(_ss(T_IGN, 70.30, t)) + 220.0 * float(_ss(70.6, 72.4, t))
    e += 12.5 * max(R - R_SMALL1, 0.0) * (1.0 if t < TW0 else 0.8)
    pul = sum(float(_pulse(t, tp, 0.10)) for tp in WORD_PULSES[1:])
    e *= 1 + 1.6 * pul
    if t < T_REL:
        e *= 1 - 0.35 * float(_gather(t))
    e *= 1 - float(_ss(74.48, 74.62, t))
    return max(e, 0.0)


class FX(core.Effect):
    name = 'strange'

    def __init__(self):
        self.rng = np.random.default_rng(70051)
        self._build_track()
        self._build_tables()
        self._build_sparks()
        self._build_riders()
        self._prog = None
        self._prog_ap = None

    # ---------------------------------------------------------------- tracking
    def _build_track(self):
        i0, i1 = C.i_of(69.2), C.N_FRAMES
        idx = np.arange(i0, i1)
        pen = np.zeros((len(idx), 2))
        for k, i in enumerate(idx):
            h = A.hand(i, 'R')
            if h is not None:
                tip = (h[8, :2] + h[12, :2]) * 0.5
                base = (h[5, :2] + h[9, :2]) * 0.5
                d = tip - base
                d = d / (np.linalg.norm(d) + 1e-6)
                pen[k] = tip + d * 10.0
            else:
                p = A.palm(i, 'R')
                up = A.palm_normal_up(i, 'R')
                pen[k] = p + up * 0.7 * A.hand_size(i, 'R')
        pen = median_filter(pen, size=(5, 1), mode='nearest')
        pen_s = gaussian_filter1d(pen, 1.6, axis=0, mode='nearest')
        # ring follows the fingertips with a slight lag (causal one-pole, precomputed)
        lag = np.zeros_like(pen_s)
        lag[0] = pen_s[0]
        aa = 1 - np.exp(-(1 / C.FPS) / 0.055)
        for k in range(1, len(pen_s)):
            lag[k] = lag[k - 1] + (pen_s[k] - lag[k - 1]) * aa
        circ = gaussian_filter1d(pen, 9.0, axis=0, mode='nearest')   # centre of the circling motion
        ang = np.unwrap(np.arctan2(pen_s[:, 1] - circ[:, 1], pen_s[:, 0] - circ[:, 0]))
        ang = gaussian_filter1d(ang, 1.5, mode='nearest')
        angv = gaussian_filter1d(np.gradient(ang) * C.FPS, 2.0, mode='nearest')
        self.tr_t = idx / C.FPS
        self.tr_pen, self.tr_lag, self.tr_circ, self.tr_ang, self.tr_angv = pen_s, lag, circ, ang, angv

    def _trk(self, arr, t):
        t = np.asarray(t, np.float64)
        if arr.ndim == 1:
            return np.interp(t, self.tr_t, arr)
        return np.stack([np.interp(t, self.tr_t, arr[:, 0]), np.interp(t, self.tr_t, arr[:, 1])], -1)

    # ---------------------------------------------------------------- timeline tables
    def _build_tables(self):
        tg = np.arange(69.9, 76.2, DT_TAB)
        R = np.array([_R_scalar(t) for t in tg])
        I = np.array([_I_scalar(t) for t in tg])
        # portal growth: a damped spring chases the target radius -> natural lag, overshoot, settle
        j0, j1 = np.searchsorted(tg, 72.30), np.searchsorted(tg, 73.62)
        x, v = R[j0], 0.0
        w0, zeta = 2 * np.pi * 2.6, 0.50
        Rs = R.copy()
        for j in range(j0, j1):
            acc = w0 * w0 * (R[j] - x) - 2 * zeta * w0 * v
            v += acc * DT_TAB
            x += v * DT_TAB
            Rs[j] = x
        bl = _ss(73.42, 73.60, tg)
        R = np.where((tg >= tg[j0]) & (tg < tg[j1]), Rs * (1 - bl) + R * bl, R)
        dR = gaussian_filter1d(np.gradient(R, DT_TAB), 12, mode='nearest')
        self.tdR = dR
        # stir: his circling fingertips (72.85-74.12) drive a hot clump and nudge the spin rate
        stir = _ss(T_STIR0, T_STIR0 + 0.15, tg) * (1 - _ss(T_STIR1 - 0.10, T_STIR1 + 0.06, tg))
        hav = np.abs(self._trk(self.tr_angv, tg))
        nudge = 1 + 0.20 * np.clip((hav - 8.5) / 4.0, -1, 1) * stir
        rev = (0.85 + 1.45 * np.clip((400.0 - R) / 340.0, 0, 1)) * nudge
        rev = np.where(tg < 70.62, rev + 0.9 * (1 - _ss(70.45, 70.62, tg)), rev)   # fast orbiting head at ignition
        omega = -2 * np.pi * rev                         # negative = counter-clockwise on screen (y down)
        phi = np.cumsum(omega) * DT_TAB
        c_hand = self._trk(self.tr_lag, tg)
        c_circ = self._trk(self.tr_circ, tg)
        wcirc = _ss(72.10, 72.60, tg)[:, None]
        ch = c_hand * (1 - wcirc) + c_circ * wcirc
        wF = _ss(95.0, 330.0, R)
        cen = ch * (1 - wF[:, None]) + CF[None, :] * wF[:, None]
        vel = np.gradient(cen, DT_TAB, axis=0)
        vel = gaussian_filter1d(vel, 20, axis=0, mode='nearest')
        E = np.array([_E_scalar(t, r) for t, r in zip(tg, R)])
        self.tg, self.tR, self.tI, self.tphi, self.tomega = tg, R, I, phi, omega
        self.tcen, self.tvel, self.tE, self.twF = cen, vel, E, wF
        self.tb = (tg >= T_SWITCH).astype(np.float64)    # ring behind him (instant switch)
        self.tap = _ss(T_AP0, T_AP1, tg)                  # aperture burn progress
        self.twidth = 3.6 + 0.021 * R                     # band half-width px
        self.tgather = _gather(tg)
        # hotspot 1: ignition = orbiting head; 72.3-72.75 = the hand that feeds the growing ring
        hand_ang = self._trk(self.tr_ang, tg)
        self.thot_ang = np.where(tg < 71.0, phi + 0.0, hand_ang)
        self.thot_w = np.maximum(1 - _ss(70.40, 70.62, tg), 0) * (tg >= T_IGN) \
            + 0.42 * _ss(72.30, 72.50, tg) * (1 - _ss(150.0, 320.0, R))
        self.thot_g = np.where(tg < 71.0, 1.1, 0.6)       # gain (the hand stretch is capped at x0.6)
        # hotspot 2: the stirring hand on the big ring
        self.thot2_ang = hand_ang
        self.thot2_w = 0.25 * stir
        # orbit trail span (radians) during ignition: the ring is DRAWN by the orbiting head
        i_ign = np.searchsorted(tg, T_IGN + 0.02)
        span = np.clip(np.abs(phi - phi[i_ign]), 0, None) * 0.85 + 0.35
        span = np.where(tg < T_IGN + 0.02, 0.35, span)
        self.tspan = np.where(tg > 70.62, 7.0, np.minimum(span, 7.0))

    def _tab(self, arr, t):
        t = np.asarray(t, np.float64)
        if arr.ndim == 1:
            return np.interp(t, self.tg, arr)
        return np.stack([np.interp(t, self.tg, arr[:, k]) for k in range(arr.shape[1])], -1)

    def rim_factor(self, th, t):
        """1 + wobble(th, t) (numpy, broadcasting)."""
        th = np.asarray(th, np.float64)
        t = np.asarray(t, np.float64)
        f = np.ones(np.broadcast(th, t).shape)
        for a, k, nu, p0 in zip(WOB_A, WOB_K, WOB_NU, WOB_P0):
            f = f + a * np.sin(k * th + p0 + nu * t)
        return f

    # ---------------------------------------------------------------- particles
    def _emit_angles(self, te, rng):
        """Emission angle on the ring for each emission time."""
        n = len(te)
        th = rng.uniform(-np.pi, np.pi, n)
        hw = np.clip(self._tab(self.thot_w, te), 0, 0.92)
        hw2 = np.clip(self._tab(self.thot2_w, te), 0, 0.5)
        span = self._tab(self.tspan, te)
        phi = self._tab(self.tphi, te)
        u = rng.random(n)
        use_hot = u < hw
        use_hot2 = (~use_hot) & (u < hw + hw2)
        th = np.where(use_hot, self._tab(self.thot_ang, te) + rng.normal(0, 0.45, n), th)
        th = np.where(use_hot2, self._tab(self.thot2_ang, te) + rng.normal(0, 0.30, n), th)
        # during the ignition orbit, non-hot sparks only come from the drawn arc (behind the head)
        arc = span < 6.28
        th_arc = phi + rng.random(n) * span
        th = np.where(arc & ~use_hot, th_arc, th)
        return th

    @staticmethod
    def _group(n, **kw):
        g = dict(kw)
        g.setdefault('emb', np.zeros(n, bool))
        g.setdefault('cool', np.ones(n))
        g.setdefault('shut', np.ones(n))
        g.setdefault('halo', np.zeros(n, bool))
        g.setdefault('behind', np.zeros(n, bool))
        return g

    def _build_sparks(self):
        rng = self.rng
        tg, E = self.tg, self.tE
        groups = []
        # ---- main emission from the ring
        cum = np.cumsum(E) * DT_TAB
        n = int(cum[-1])
        u = (np.arange(n) + rng.random(n)) / max(n, 1) * cum[-1]
        te = np.interp(u, cum, tg)
        # explicit bursts: ignition flash + crackles on word onsets + wipe release
        bursts = [(70.10, 170, 1.6), (71.06, 90, 1.3), (71.86, 70, 1.2), (72.28, 70, 1.2),
                  (72.66, 160, 1.3), (73.30, 220, 1.2), (T_REL, 520, 1.5)]
        tb, sb = [te], [np.ones(n)]
        for (t0, m, spd) in bursts:
            tb.append(t0 + rng.random(m) * 0.06)
            sb.append(np.full(m, spd))
        te = np.concatenate(tb)
        spd_mul = np.concatenate(sb)
        n = len(te)
        th = self._emit_angles(te, rng)
        R = self._tab(self.tR, te)
        cen = self._tab(self.tcen, te)
        vel = self._tab(self.tvel, te)
        om = self._tab(self.tomega, te)
        wid = self._tab(self.twidth, te)
        rf = self.rim_factor(th, te)
        rr = R * rf + rng.normal(0, 0.35, n) * wid
        cs, sn = np.cos(th), np.sin(th)
        p0 = cen + np.stack([cs, sn], -1) * rr[:, None]
        tan = np.stack([sn, -cs], -1)          # direction of travel for decreasing theta
        rad = np.stack([cs, sn], -1)
        vrim = np.abs(om) * R
        size_k = 0.55 + np.clip(R / 420.0, 0, 1.6)
        st = (vrim * rng.uniform(0.25, 0.85, n) + rng.uniform(70, 300, n) * size_k) * spd_mul
        sr = st * rng.uniform(-0.12, 0.55, n)
        # breath-in: sparks emitted while the portal gathers are sucked inward
        ga = np.where(te < T_REL, self._tab(self.tgather, te), 0.0)
        inward = rng.random(n) < 0.6 * ga
        sr = np.where(inward, -np.abs(sr) * 1.2 - 160.0 * ga, sr)
        st = np.where(inward, st * 0.55, st)
        dR = self._tab(self.tdR, te)
        v0 = tan * st[:, None] + rad * (sr + 0.85 * dR)[:, None] + vel * 0.55 + rng.normal(0, 35, (n, 2))
        life = np.exp(rng.normal(np.log(0.36), 0.50, n)).clip(0.08, 1.2)
        life *= np.where(R < 80, 0.8, 1.0)
        k = rng.uniform(2.0, 4.6, n)
        g = np.full(n, GRAV) * rng.uniform(0.8, 1.15, n)
        size = rng.uniform(0.8, 1.6, n)
        temp0 = rng.uniform(0.85, 1.35, n)
        shut = np.exp(rng.normal(0.0, 0.40, n)).clip(0.45, 2.2)      # wider streak-length distribution
        # embers: slow, floaty, long-lived, dimmer, a bit larger, a few with a soft halo
        emb = rng.random(n) < 0.07
        life = np.where(emb, rng.uniform(0.9, 2.0, n), life)
        k = np.where(emb, rng.uniform(4.5, 7.0, n), k)
        g = np.where(emb, rng.uniform(250, 600, n), g)
        size = np.where(emb, rng.uniform(1.3, 2.2, n), size)
        temp0 = np.where(emb, rng.uniform(0.6, 0.9, n), temp0)
        cool = np.where(emb, 1.8, 1.0)
        halo = emb & (rng.random(n) < 0.3)
        # depth: before the switch everything is in front of him; after it a spark is BEHIND him if it is
        # born over his silhouette, and only ~15 % of those born outside him fly in front (toward camera)
        a_birth = _sample_lo(te, p0, _alpha_lo)
        behind = (te >= T_SWITCH) & ((a_birth > 0.3) | (rng.random(n) >= 0.15) | emb)
        front_late = (te >= T_SWITCH) & ~behind
        # slow embers drifting in front of his face read as freckles: front embers die out quickly
        life = np.where(emb & ~behind, np.minimum(life, 0.8), life)
        size = np.where(front_late, size * 1.3, size)                # closer to camera -> a bit larger
        main = self._group(n, te=te, p0=p0, v0=v0, life=life, k=k, g=g, size=size, temp0=temp0,
                           emb=emb, cool=cool, shut=shut, halo=halo, behind=behind)
        # ---- metal-spark splits: ~4 % of sparks pop mid-life into 3-6 short hot children
        par = np.nonzero((~emb) & (life > 0.22) & (rng.random(n) < 0.04))[0]
        if len(par):
            f = rng.uniform(0.35, 0.65, len(par))
            a_s = f * life[par]
            kk = k[par][:, None]
            vt = np.stack([np.zeros(len(par)), g[par]], -1) / kk
            ps = p0[par] + vt * a_s[:, None] + (v0[par] - vt) * (1 - np.exp(-kk * a_s[:, None])) / kk
            vs = vt + (v0[par] - vt) * np.exp(-kk * a_s[:, None])
            Ts = temp0[par] * np.exp(-a_s / (0.42 * life[par] + 0.04))
            main['life'][par] = a_s                                  # parent dies at the split
            nch = rng.integers(3, 7, len(par))
            rep = np.repeat(np.arange(len(par)), nch)
            m = len(rep)
            ang = rng.uniform(-np.pi, np.pi, m)
            spd = rng.uniform(180, 520, m)
            groups.append(self._group(
                m, te=te[par][rep] + a_s[rep], p0=ps[rep],
                v0=vs[rep] * 0.45 + np.stack([np.cos(ang), np.sin(ang)], -1) * spd[:, None],
                life=rng.uniform(0.05, 0.20, m), k=rng.uniform(3.0, 5.0, m), g=g[par][rep] * 0.9,
                size=size[par][rep] * rng.uniform(0.55, 0.85, m), temp0=np.minimum(Ts[rep] * 1.1 + 0.2, 1.45),
                cool=np.full(m, 0.8), shut=rng.uniform(0.6, 1.3, m), behind=behind[par][rep]))
            # the pop itself: a brief glint
            mp = len(par)
            groups.append(self._group(
                mp, te=te[par] + a_s, p0=ps, v0=vs * 0.2, life=np.full(mp, 0.045), k=np.full(mp, 4.0),
                g=np.zeros(mp), size=size[par] * 1.8, temp0=np.full(mp, 1.35), behind=behind[par]))
        groups.insert(0, main)
        # ---- sparks thrown off the aperture's burn front (in the portal plane, behind him)
        na = 260
        te_a = rng.uniform(T_AP0 + 0.05, T_AP1 - 0.03, na)
        fr = 1.25 * self._tab(self.tap, te_a) - 0.15
        q = fr / 0.85
        Ra = self._tab(self.tR, te_a) - self._tab(self.twidth, te_a) * 0.35
        ca = self._tab(self.tcen, te_a)
        oa = np.array([self._burn_origin(tt, cc, rr) for tt, cc, rr in zip(te_a, ca, Ra)])
        rmax = Ra + np.linalg.norm(oa - ca, axis=1)
        th_a = rng.uniform(-np.pi, np.pi, na)
        rd = np.stack([np.cos(th_a), np.sin(th_a)], -1)
        p_a = oa + rd * (q * rmax)[:, None]
        ok = (q > 0.03) & (np.linalg.norm(p_a - ca, axis=1) < 0.95 * Ra)
        te_a, rd, p_a = te_a[ok], rd[ok], p_a[ok]
        na = len(te_a)
        v_a = rd * rng.uniform(30, 150, na)[:, None] + np.stack([-rd[:, 1], rd[:, 0]], -1) * rng.uniform(-70, 70, na)[:, None] \
            + rng.normal(0, 40, (na, 2))
        groups.append(self._group(na, te=te_a, p0=p_a, v0=v_a, life=rng.uniform(0.15, 0.40, na),
                                  k=rng.uniform(3, 5, na), g=rng.uniform(500, 1200, na),
                                  size=rng.uniform(0.7, 1.15, na), temp0=rng.uniform(1.0, 1.3, na),
                                  behind=np.ones(na, bool)))
        # ---- pre-glow: a few tiny sparks fizzing off the fingertips before the flash
        npg = 5
        te_p = np.sort(rng.uniform(70.015, 70.095, npg))
        p_p = self._trk(self.tr_pen, te_p)
        ang = rng.uniform(-np.pi, 0.2, npg)
        groups.append(self._group(npg, te=te_p, p0=p_p,
                                  v0=np.stack([np.cos(ang), np.sin(ang)], -1) * rng.uniform(80, 200, npg)[:, None],
                                  life=rng.uniform(0.10, 0.22, npg), k=np.full(npg, 3.5), g=np.full(npg, 900.0),
                                  size=np.full(npg, 0.8), temp0=rng.uniform(0.95, 1.15, npg)))
        # ---- last embers: shed only where the rim is leaving the frame (within ~80 px of the edge it crosses)
        nc = 9000
        te_e = rng.uniform(73.98, 74.46, nc)
        th_e = rng.uniform(-np.pi, np.pi, nc)
        R_e = self._tab(self.tR, te_e) * self.rim_factor(th_e, te_e)
        rd_e = np.stack([np.cos(th_e), np.sin(th_e)], -1)
        p_e = CF + rd_e * R_e[:, None]
        inside = (p_e[:, 0] > 2) & (p_e[:, 0] < W - 2) & (p_e[:, 1] > 2) & (p_e[:, 1] < H - 2)
        dedge = np.minimum(np.minimum(p_e[:, 0], W - p_e[:, 0]), np.minimum(p_e[:, 1], H - p_e[:, 1]))
        sel = np.nonzero(inside & (dedge < 80))[0]
        sel = sel[rng.permutation(len(sel))[:340]]
        te_e, rd_e, p_e = te_e[sel], rd_e[sel], p_e[sel]
        ne = len(sel)
        tan_e = np.stack([rd_e[:, 1], -rd_e[:, 0]], -1)
        inw = np.stack([W / 2 - p_e[:, 0], H * 0.45 - p_e[:, 1]], -1)
        inw /= np.linalg.norm(inw, axis=1, keepdims=True) + 1e-6
        v_e = rd_e * (self._tab(self.tdR, te_e) * rng.uniform(0.02, 0.12, ne))[:, None] \
            + tan_e * rng.uniform(-120, 220, ne)[:, None] + inw * rng.uniform(60, 320, ne)[:, None] \
            + rng.normal(0, 60, (ne, 2))
        life_e = np.minimum(rng.uniform(0.35, 0.85, ne), 74.95 - te_e)
        uu = rng.random(ne)
        groups.append(self._group(ne, te=te_e, p0=p_e, v0=v_e, life=life_e, k=rng.uniform(2.2, 3.8, ne),
                                  g=rng.uniform(500, 1100, ne), size=1.0 + 2.5 * uu ** 2.2,
                                  temp0=rng.uniform(1.0, 1.3, ne), cool=np.full(ne, 2.6),
                                  shut=rng.uniform(0.5, 1.0, ne), emb=rng.random(ne) < 0.5,
                                  halo=uu > 0.80))
        # ---- the suit burns off into embers where the reality front crosses it (suit only: no face/hands)
        nb = 900
        te_b = np.sort(rng.uniform(T_REL + 0.01, T_FRONT1 + 0.04, nb))
        th_b = rng.uniform(-np.pi, np.pi, nb)
        Rp_b = _front_R(te_b, self._tab(self.tR, te_b))
        rd_b = np.stack([np.cos(th_b), np.sin(th_b)], -1)
        p_b = CF + rd_b * (Rp_b * self.rim_factor(th_b, te_b))[:, None]
        on = _sample_lo(te_b, p_b, _suit_lo) > 0.5
        te_b, rd_b, p_b = te_b[on], rd_b[on], p_b[on]
        nb = len(te_b)
        v_b = rd_b * rng.uniform(40, 200, nb)[:, None] + np.stack([np.zeros(nb), -rng.uniform(60, 260, nb)], -1) \
            + rng.normal(0, 45, (nb, 2))
        groups.append(self._group(nb, te=te_b, p0=p_b, v0=v_b, life=rng.uniform(0.18, 0.5, nb),
                                  k=rng.uniform(3.0, 5.0, nb), g=rng.uniform(250, 650, nb),
                                  size=rng.uniform(0.8, 1.3, nb), temp0=rng.uniform(0.8, 1.1, nb),
                                  cool=np.full(nb, 1.3)))
        sp = {kk: np.concatenate([gg[kk] for gg in groups]) for kk in groups[0]}
        # the last sparks fade out (their envelope) by ~75.35 so the final frames are the clean footage
        sp['life'] = np.minimum(sp['life'], 75.35 - sp['te'])
        # cull sparks born far outside the frame (big wipe ring)
        p0 = sp['p0']
        keep = (p0[:, 0] > -350) & (p0[:, 0] < W + 350) & (p0[:, 1] > -350) & (p0[:, 1] < H + 250) \
            & (sp['life'] > 0.03)
        sel = np.nonzero(keep)[0]
        sel = sel[np.argsort(sp['te'][sel], kind='stable')]
        self.sp = {kk: v[sel] for kk, v in sp.items()}
        self.sp['id'] = np.arange(len(sel), dtype=np.uint64)
        self.sp_maxlife = float(self.sp['life'].max()) if len(sel) else 0.0

    def _build_riders(self):
        """Sparks that ride the ring band (the dense spinning ring texture)."""
        rng = np.random.default_rng(70052)
        tg = self.tg
        circ = 2 * np.pi * np.maximum(self.tR, 1.0)
        Eb = (4.4 * circ + 500.0 * (self.tR > 1)) * np.clip(self.tI, 0, 1.6)
        Eb = Eb * (1 - _ss(74.42, 74.60, tg))
        Eb = np.where(tg < T_IGN, 0, Eb)
        cum = np.cumsum(Eb) * DT_TAB
        n = int(cum[-1])
        u = (np.arange(n) + rng.random(n)) / max(n, 1) * cum[-1]
        te = np.interp(u, cum, tg)
        R = self._tab(self.tR, te)
        th = self._emit_angles(te, rng)
        span = self._tab(self.tspan, te)
        ign = span < 6.28
        m = np.where(ign, rng.uniform(0.55, 0.95, n), rng.uniform(0.85, 1.40, n))
        life = np.where(ign, rng.uniform(0.12, 0.40, n), rng.uniform(0.07, 0.22, n))
        dr = rng.normal(0, 0.55, n)
        # cull riders that are outside the frame during the wipe (approximate, by emission pos)
        cen = self._tab(self.tcen, te)
        p = cen + np.stack([np.cos(th), np.sin(th)], -1) * R[:, None]
        keep = (p[:, 0] > -150) & (p[:, 0] < W + 150) & (p[:, 1] > -150) & (p[:, 1] < H + 150)
        sel = np.nonzero(keep)[0]
        self.rb = dict(te=te[sel], th0=th[sel], m=m[sel], life=life[sel], dr=dr[sel],
                       phi0=self._tab(self.tphi, te[sel]), size=rng.uniform(0.8, 1.5, n)[sel],
                       suck=rng.random(n)[sel], shut=rng.uniform(1.0, 1.8, n)[sel],
                       id=np.arange(len(sel), dtype=np.uint64) + np.uint64(10 ** 7))

    # ---------------------------------------------------------------- per-frame state
    def state(self, S):
        t = S.t
        st = getattr(S, '_strange_state', None)
        if st is not None:
            return st
        R = float(self._tab(self.tR, t))
        wF = float(self._tab(self.twF, t))
        off = np.asarray(S.person_offset, np.float64) * (1 - wF)
        cen = self._tab(self.tcen, t) + off
        st = dict(t=t, R=R, I=float(self._tab(self.tI, t)), cen=cen, off=off, wF=wF,
                  dR=float(self._tab(self.tdR, t)), vel=self._tab(self.tvel, t),
                  phi=float(self._tab(self.tphi, t)), b=float(self._tab(self.tb, t)),
                  ap=float(self._tab(self.tap, t)), width=float(self._tab(self.twidth, t)),
                  span=float(self._tab(self.tspan, t)), hot_ang=float(self._tab(self.thot_ang, t)),
                  hot_w=float(self._tab(self.thot_w, t)), hot_g=float(self._tab(self.thot_g, t)),
                  hot2_ang=float(self._tab(self.thot2_ang, t)), hot2_w=float(self._tab(self.thot2_w, t)),
                  gather=float(_gather(t)) if t < T_REL else 0.0)
        # wipe: reality front (person + everything) radius, follows from the portal centre
        st['Rp'] = float(_front_R(t, R))
        # flicker of the whole ring (subtle, never strobing) + crackle (word bursts) for the hand light
        fi = S.i
        st['flick'] = 0.92 + 0.08 * np.sin(fi * 2.1) * np.sin(fi * 0.77 + 1.3) + 0.05 * (_hash([7], fi)[0] - 0.5)
        st['crack'] = 1.0 + 0.7 * sum(float(_pulse(t, tp, 0.12)) for tp in WORD_PULSES[1:]) \
            + 0.18 * (_hash([11], fi)[0] - 0.5)
        S._strange_state = st
        return st

    def active(self, t):
        return t >= T_PRE - 1e-6

    def _done(self, st):
        return st['Rp'] > 1450

    # ---------------------------------------------------------------- rim polygon
    def rim_poly(self, cen, R, t, scale=1.0, n=360):
        th = np.linspace(-np.pi, np.pi, n, endpoint=False)
        r = R * scale * self.rim_factor(th, t)
        return np.stack([cen[0] + r * np.cos(th), cen[1] + r * np.sin(th)], -1)

    def disk_mask(self, cen, R, t, shape=(H, W), box=(0, 0), inset=0.0, soft=1.5):
        """Anti-aliased filled rim polygon (float 0..1) in a crop."""
        h, w = shape
        poly = self.rim_poly(cen, max(R - inset, 0.5), t, n=max(64, min(720, int(R * 1.2))))
        poly = (poly - np.array(box, np.float64)) * 8.0
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 255, lineType=cv2.LINE_AA, shift=3)
        m = m.astype(np.float32) / 255.0
        if soft > 0.6:
            m = cv2.GaussianBlur(m, (0, 0), soft)
        return m

    # ---------------------------------------------------------------- hooks
    def background(self, S):
        st = self.state(S)
        S.meta['strange'] = dict(cen=(float(st['cen'][0]), float(st['cen'][1])), R=st['R'], b=st['b'],
                                 front_c=(float(CF[0]), float(CF[1])), front_R=st['Rp'],
                                 wipe=float(_ss(T_REL, 74.5, st['t'])))
        S._strange_store = store = dict(emit=[], plate=None)
        if self._done(st) or st['b'] < 0.5:
            return
        self._portal_interior(S, st, store)
        # the ring band, its riders and the sparks born behind him are part of the BACKGROUND plane now:
        # captions and the person are composited over them by the compositor
        sp = self._spark_sprites(st, S.i)
        bgroups = [self._rider_sprites(st, S.i)]
        if sp is not None:
            bgroups.append(self._subset(sp, sp['behind']))
        for lay, dark in ((self._ring(st), 0.0), (self._splat(bgroups), 0.35)):
            if lay is None:
                continue
            rgb, x0, y0 = lay
            h, w = rgb.shape[:2]
            _add_emissive(S.bg[y0:y0 + h, x0:x0 + w], rgb, dark)
            store['emit'].append((rgb, x0, y0, dark))

    def _portal_interior(self, S, st, store):
        if st['ap'] <= 0.001 or st['R'] < 20:
            return
        if st['Rp'] >= st['R'] - 2:          # everything inside is replaced by the raw footage in post
            return
        R, cen, t = st['R'], st['cen'], st['t']
        Rm = R * 1.06 + 8
        x0, y0 = int(max(0, cen[0] - Rm)), int(max(0, cen[1] - Rm))
        x1, y1 = int(min(W, cen[0] + Rm + 1)), int(min(H, cen[1] + Rm + 1))
        if x1 <= x0 or y1 <= y0:
            return
        inset = st['width'] * 0.35
        edge = None
        if st['ap'] < 0.999:
            m, edge = self._aperture(st, (x0, y0, x1, y1), R - inset)
        else:
            m = self.disk_mask(cen, R, t, (y1 - y0, x1 - x0), (x0, y0), inset=inset, soft=1.2)
        view = A.plate(0)[y0:y1, x0:x1]
        # warm spill from the ring onto the other side + slight darkening at the very rim
        X, Y = core.grid(y1 - y0, x1 - x0, (x0, y0))
        rho = np.sqrt((X - cen[0]) ** 2 + (Y - cen[1]) ** 2)
        ed = np.clip(1.0 - rho / max(R, 1.0), 0, 1)
        spill = np.exp(-ed * R / (0.10 * R + 12.0)).astype(np.float32)
        warm = np.array([1.0, 0.55, 0.20], np.float32)
        view = view * (1.0 - 0.22 * spill)[..., None] + spill[..., None] * warm * (0.12 * st['I'])
        # opening flare: the other side is over-exposed for a moment as the aperture completes
        flare = float(_pulse(t, 72.92, 0.16)) * 0.20
        if flare > 0.01:
            view = view * (1 + flare) + flare * 0.12
        bg = S.bg[y0:y1, x0:x1]
        bg *= (1 - m)[..., None]
        pv = (view * m[..., None]).astype(np.float32)
        bg += pv
        store['plate'] = (pv, x0, y0)
        if edge is not None and edge.max() > 0.01:
            # the burn front: hot gold line with a soft glow on the not-yet-open side
            ecol = (edge[..., None] * np.array([2.6, 1.25, 0.36], np.float32) * (0.6 + 0.4 * min(st['I'], 1.4)))
            bg += ecol
            store['emit'].append((ecol, x0, y0, 0.0))

    APERTURE = r'''
uniform vec2 uC; uniform vec2 uO; uniform float uRi; uniform float uRmax; uniform vec4 uWa; uniform vec4 uWp;
uniform float uFront; uniform float uT;
float rimR(float th){
  return uRi * (1.0 + uWa.x*sin(2.0*th + uWp.x) + uWa.y*sin(3.0*th + uWp.y)
                    + uWa.z*sin(5.0*th + uWp.z) + uWa.w*sin(7.0*th + uWp.w));
}
// burn value: distance from the burn origin (behind his head) + noise; the hole is where v < uFront
float burnV(vec2 pf){
  vec2 p = pf - uC;
  float q = length(pf - uO) / uRmax;
  float n = fbm3(vec3(p * 0.011, uT * 0.6), 4);
  float n2 = snoise(vec3(p * 0.05, uT * 1.7 + 3.0));
  return q * 0.85 + 0.34 * n + 0.035 * n2;
}
void main(){
  vec2 pf = fpx();
  vec2 p = pf - uC;
  float rho = length(p);
  float th = atan(p.y, p.x);
  float rr = rimR(th);
  float disk = smoothstep(rr + 1.2, rr - 1.2, rho);
  float v = burnV(pf);
  // per-pixel gradient by finite differences (fwidth works on 2x2 quads -> blocky)
  float gx = burnV(pf + vec2(1.0, 0.0)) - v, gy = burnV(pf + vec2(0.0, 1.0)) - v;
  float fw = max(length(vec2(gx, gy)), 1e-5);
  float dv = (v - uFront) / fw;                       // ~ signed distance to the burn front in px
  float open = smoothstep(1.2, -1.2, dv);
  float live = smoothstep(1.10, 0.96, uFront);
  float n3 = snoise(vec3(pf * 0.03, uT * 2.3));
  float edge = exp(-dv * dv / 3.2) + 0.28 * exp(-max(dv, 0.0) / 3.5) * step(0.0, dv);
  edge *= (0.7 + 0.6 * saturate(0.5 + 0.5 * n3)) * live;
  fragColor = vec4(open * disk, edge * disk, 0.0, 1.0);
}
'''

    def _burn_origin(self, t, cen, Ri):
        """Where the aperture starts burning: behind his forehead (hidden by his head), kept inside the ring."""
        i = C.i_of(t)
        fo = A.face_oval(i)
        c = fo.mean(0)
        hgt = fo[:, 1].max() - fo[:, 1].min()
        o = np.array([c[0], c[1] - 0.22 * hgt], np.float64)
        d = o - cen
        dl = np.linalg.norm(d)
        lim = 0.7 * max(Ri, 1.0)
        if dl > lim:
            o = cen + d / dl * lim
        return o

    def _aperture(self, st, box, Ri):
        """Burn-through opening mask + hot edge (GLSL) for the crop box."""
        x0, y0, x1, y1 = box
        if self._prog_ap is None:
            self._prog_ap = gl.program(self.APERTURE)
        t = st['t']
        wp = WOB_P0 + WOB_NU * t
        o = self._burn_origin(t, st['cen'], Ri) + st['off']
        img = gl.render(self._prog_ap, (x1 - x0, y1 - y0), offset=(x0, y0), uniforms={
            'uC': (float(st['cen'][0]), float(st['cen'][1])), 'uRi': float(max(Ri, 1.0)),
            'uO': (float(o[0]), float(o[1])), 'uRmax': float(max(Ri, 1.0) + np.linalg.norm(o - st['cen'])),
            'uWa': tuple(float(v) for v in WOB_A), 'uWp': tuple(float(v) for v in wp),
            'uFront': float(1.25 * st['ap'] - 0.15), 'uT': float(t)})
        return np.ascontiguousarray(img[..., 0]), np.ascontiguousarray(img[..., 1])

    def behind(self, S):
        """Heat haze around the rim. Done here (after the person hooks and the float offset) so it can be
        masked out of his final silhouette: the shimmer is behind him, not on his body."""
        st = self.state(S)
        if self._done(st) or st['b'] < 0.5 or st['R'] < 30 or st['Rp'] >= st['R'] - 2:
            return []
        R, cen, t = st['R'], st['cen'], st['t']
        wdt = st['width']
        reach = wdt * 9.0 + 40
        x0, y0 = int(max(0, cen[0] - R - reach)), int(max(0, cen[1] - R - reach))
        x1, y1 = int(min(W, cen[0] + R + reach)), int(min(H, cen[1] + R + reach))
        if x1 - x0 < 8 or y1 - y0 < 8:
            return []
        q = 2                                                     # smooth field: compute at 1/2 res
        hh, ww = (y1 - y0) // q, (x1 - x0) // q
        X, Y = core.grid(hh, ww)
        X = X * q + x0 + q / 2
        Y = Y * q + y0 + q / 2
        rho = np.sqrt((X - cen[0]) ** 2 + (Y - cen[1]) ** 2)
        d = rho - R
        band = np.exp(-(d / (wdt * 3.0 + 14.0)) ** 2) * min(1.0, st['I'])
        if band.max() < 0.02:
            return []
        th = np.arctan2(Y - cen[1], X - cen[0])
        amp = 1.8 * band * np.sin(d * 0.22 - t * 23.0 + 2.5 * np.sin(th * 9.0 + t * 4.0))
        a = cv2.resize(S.alpha[y0:y0 + hh * q, x0:x0 + ww * q], (ww, hh), interpolation=cv2.INTER_AREA)
        a = cv2.GaussianBlur(cv2.dilate(a, np.ones((5, 5), np.uint8)), (0, 0), 2.0)
        amp = amp * (1.0 - np.clip(a * 1.3, 0, 1))
        nx, ny = (X - cen[0]) / (rho + 1e-3), (Y - cen[1]) / (rho + 1e-3)
        dx = cv2.resize((nx * amp).astype(np.float32), (ww * q, hh * q), interpolation=cv2.INTER_LINEAR)
        dy = cv2.resize((ny * amp).astype(np.float32), (ww * q, hh * q), interpolation=cv2.INTER_LINEAR)
        S.distort.append((x0, y0, dx, dy))
        return []

    def lights(self, S):
        return self._light_list(S, self.state(S))

    def _light_list(self, S, st):
        if 'lights' in st:
            return st['lights']
        out = []
        t = st['t']
        pre = _pre(t)
        if pre > 0.01:
            pp = self._trk(self.tr_pen, t) + st['off']
            out.append(Light(pp, (1.0, 0.55, 0.2), intensity=0.45 * pre * st['crack'], radius=90, rim=0.3,
                             diffuse=0.8))
        if self._done(st) or st['I'] < 0.01:
            st['lights'] = out
            return out
        I, R, cen, wF = st['I'] * st['flick'], st['R'], st['cen'], st['wF']
        small = 1.0 - wF
        if small > 0.01:
            # tight hot light right at the sparkler ring (hand, fingers, forehead) flickering with the crackle
            out.append(Light(cen, (1.0, 0.56, 0.22), intensity=2.2 * I * small * st['crack'], radius=110,
                             rim=0.4, diffuse=1.0, bg=0.03))
            out.append(Light(cen, GOLD, intensity=0.30 * I * small, radius=300 + R, rim=0.15, diffuse=0.35,
                             bg=0.02))
        if wF > 0.01 and R < 1700:
            # a few weak lights at the ring points nearest his silhouette edge (smooth weights, no popping)
            # distance (px) from each ring point to his silhouette; points behind him or close to it count
            a8 = cv2.resize(S.alpha, (W // 8, H // 8), interpolation=cv2.INTER_AREA)
            dist = cv2.distanceTransform((a8 < 0.5).astype(np.uint8), cv2.DIST_L2, 3) * 8.0
            nC = 12
            ang = st['phi'] * 0.1 + 2 * np.pi * np.arange(nC) / nC
            pts = cen[None, :] + R * np.stack([np.cos(ang), np.sin(ang)], -1)
            xi = np.clip((pts[:, 0] / 8).astype(int), 0, W // 8 - 1)
            yi = np.clip((pts[:, 1] / 8).astype(int), 0, H // 8 - 1)
            ox = np.maximum(np.maximum(-pts[:, 0], pts[:, 0] - W), 0) + np.maximum(np.maximum(-pts[:, 1], pts[:, 1] - H), 0)
            dk = dist[yi, xi] + ox
            wk = np.exp(-dk / 110.0)
            tot = 1.3 * I * wF / max(wk.sum(), 3.0)
            for kq in np.nonzero(wk > 0.08)[0]:
                out.append(Light(pts[kq], GOLD, intensity=float(tot * wk[kq]), radius=min(0.35 * R + 110, 420),
                                 rim=0.22, diffuse=0.12, bg=0.02))
        # the hand that feeds the growing ring (72.3-72.75)
        if st['hot_w'] > 0.05 and t > 72.3:
            hp = self._trk(self.tr_pen, t) + st['off']
            out.append(Light(hp, (1.0, 0.62, 0.28), intensity=0.7 * st['hot_w'] * I, radius=110, rim=0.3,
                             diffuse=0.5, bg=0.01))
        # the clump his stirring hand drives around the big ring
        if st['hot2_w'] > 0.02 and R < 1200:
            a2 = st['hot2_ang']
            p2 = cen + R * np.array([np.cos(a2), np.sin(a2)])
            out.append(Light(p2, GOLD, intensity=1.6 * st['hot2_w'] * I, radius=0.3 * R + 90, rim=0.2,
                             diffuse=0.10, bg=0.02))
        st['lights'] = out
        return out

    # ---------------------------------------------------------------- ring band (GLSL)
    FRAG = r'''
uniform vec2 uC; uniform float uR; uniform float uPhase; uniform float uI; uniform float uW;
uniform vec4 uWa; uniform vec4 uWp; uniform float uSpan; uniform float uHead;
uniform float uHotA; uniform float uHotW; uniform float uHot2A; uniform float uHot2W;
uniform float uT; uniform float uBlur; uniform vec2 uSm; uniform int uNs; uniform float uGather;
float rimR(float th){
  return uR * (1.0 + uWa.x*sin(2.0*th + uWp.x) + uWa.y*sin(3.0*th + uWp.y)
                   + uWa.z*sin(5.0*th + uWp.z) + uWa.w*sin(7.0*th + uWp.w));
}
float angd(float a, float b){ float d = mod(a - b + PI, TAU) - PI; return d; }
vec3 ringAt(vec2 p){
  float rho = length(p);
  float th = atan(p.y, p.x);
  float rr = rimR(th);
  float d = rho - rr;
  float Wd = uW;
  // drawn-arc mask (ignition): visible behind the orbiting head only
  float arcm = 1.0;
  if (uSpan < 6.28) {
    float behind = mod(th - uHead, TAU);
    arcm = smoothstep(uSpan, uSpan * 0.55, behind) * smoothstep(-0.05, 0.12, behind);
    float ah = angd(th, uHead) / 0.18; arcm += exp(-ah * ah) * 1.2;
  }
  float hd = angd(th, uHotA) / 0.55; float hd2 = angd(th, uHot2A) / 0.45;
  float hh = uHotW * exp(-hd * hd) + uHot2W * exp(-hd2 * hd2);
  // hot stretches are tinted toward deep gold instead of clipping to white
  vec3 hot = vec3(1.0) + hh * vec3(1.0, 0.70, 0.35);
  // broad haze glow
  float gs = 0.09 * uR + 10.0;
  float glo = exp(-abs(d) / gs) * (d < 0.0 ? 0.6 : 1.0);
  vec3 col = vec3(0.60, 0.22, 0.05) * glo * 0.30;
  float reach = Wd * 9.0 + 30.0;
  if (abs(d) < reach) {
    float thr = th - uPhase;
    vec2 cs = vec2(cos(thr), sin(thr));
    float fa = max(uR, 25.0) / 40.0;
    // three filament layers rotating at different rates (parallax of spark sheets)
    float n1 = snoise(vec3(cs * fa, d / (Wd * 0.55) + 1.7));
    vec2 cs2 = vec2(cos(th - uPhase * 1.45), sin(th - uPhase * 1.45));
    float n2 = snoise(vec3(cs2 * fa * 1.9 + 7.1, d / (Wd * 0.33) - uT * 2.0));
    vec2 cs3 = vec2(cos(th - uPhase * 0.7), sin(th - uPhase * 0.7));
    float n3 = snoise(vec3(cs3 * fa * 1.3 + 3.3, d / (Wd * 0.45) + 11.0));
    float f1 = pow(saturate(1.0 - abs(n1)), 10.0);
    float f2 = pow(saturate(1.0 - abs(n2)), 14.0);
    float f3 = pow(saturate(1.0 - abs(n3)), 12.0);
    float fil = f1 + 0.9 * f2 + 0.8 * f3;
    // clumps: brighter / dimmer stretches of the rim racing around
    float clump = 0.5 + 0.5 * snoise(vec3(cs * 2.2, uT * 1.1 + 5.0));
    float clump2 = 0.5 + 0.5 * snoise(vec3(cs2 * 3.5, uT * 1.7 + 9.0));
    float thick0 = Wd * (0.65 + 0.7 * clump);
    // radial motion blur (fast expansion): widen profiles, conserve energy
    float thick = sqrt(thick0 * thick0 + uBlur * uBlur);
    float wc0 = Wd * 0.20 + 0.6; float wc = sqrt(wc0 * wc0 + uBlur * uBlur);
    float dcen = d - uBlur * 0.5;
    float dc = dcen / wc; float core = exp(-dc * dc) * (wc0 / wc);
    float db = dcen / thick; float band = exp(-db * db) * mix(1.0, thick0 / thick, 0.7);
    // jagged spray: radial spikes outward (sparks leaving the rim), shorter inward; while the portal
    // gathers (breath-in) the spray turns inward
    float spk = snoise(vec3(cs * fa * 2.6 + 1.3, uT * 5.0));
    float spk0 = max(spk, 0.0); float spikes = spk0 * spk0 * spk0;
    float sprayO = d > 0.0 ? exp(-d / (thick * (1.2 + 5.0 * spikes * (1.0 - 0.7 * uGather))))
                           : exp(d / (thick * (0.7 + 4.0 * spikes * uGather)));
    float bright = 0.45 + 0.75 * clump * clump2 + 0.3 * clump;
    // the smooth core is kept low: the band must read as a wheel of discrete streaks, not a neon tube
    col += vec3(4.0, 2.9, 1.5) * core * (0.17 + 1.1 * fil) * bright;
    col += vec3(3.3, 1.35, 0.26) * band * (0.08 + 1.6 * fil) * bright;
    col += vec3(1.9, 0.62, 0.10) * sprayO * (0.05 + 1.2 * fil) * bright * 0.8;
  }
  return col * arcm * hot;
}
void main(){
  vec2 p = fpx() - uC;
  vec3 col = vec3(0.0);
  if (uNs <= 1) {
    col = ringAt(p);
  } else {
    // smear along the centre velocity (fast hand moves): average a few sub-frame positions
    for (int k = 0; k < 5; k++) {
      if (k >= uNs) break;
      float f = float(k) / float(uNs - 1) - 0.5;
      col += ringAt(p + uSm * f);
    }
    col /= float(uNs);
  }
  fragColor = vec4(col * uI, 1.0);
}
'''

    def _ring(self, st):
        """Render the hot ring band. Returns (img, x0, y0) or None."""
        R, cen = st['R'], st['cen']
        if R < 1.0 or st['I'] < 0.005:
            return None
        sm = np.asarray(st['vel'], np.float64) * SHUTTER
        sml = float(np.linalg.norm(sm))
        ns = 1 if sml < 9.0 else (3 if sml < 25 else 5)
        m = R * 1.08 + st['width'] * 9 + 0.35 * R + 40 + sml * 0.5
        x0, y0 = int(max(0, cen[0] - m)), int(max(0, cen[1] - m))
        x1, y1 = int(min(W, cen[0] + m + 1)), int(min(H, cen[1] + m + 1))
        if x1 <= x0 or y1 <= y0:
            return None
        # skip if the frame lies entirely inside the ring (big wipe ring)
        corners = np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]], np.float64)
        if np.sqrt(((corners - cen) ** 2).sum(1)).max() < R * 0.9 - 200:
            return None
        if self._prog is None:
            self._prog = gl.program(self.FRAG)
        t = st['t']
        wp = WOB_P0 + WOB_NU * t
        img = gl.render(self._prog, (x1 - x0, y1 - y0), offset=(x0, y0), uniforms={
            'uC': (float(cen[0]), float(cen[1])), 'uR': float(R), 'uPhase': float(st['phi']),
            'uI': float(st['I'] * st['flick'] * K_RING), 'uW': float(st['width']),
            'uWa': tuple(float(v) for v in WOB_A), 'uWp': tuple(float(v) for v in wp),
            'uSpan': float(st['span']), 'uHead': float(st['phi']),
            'uHotA': float(st['hot_ang']), 'uHotW': float(st['hot_w'] * st['hot_g']),
            'uHot2A': float(st['hot2_ang']), 'uHot2W': float(st['hot2_w'] * 2.4),
            'uT': float(t), 'uBlur': float(abs(st['dR']) * 0.6 / C.FPS),
            'uSm': (float(sm[0]), float(sm[1])), 'uNs': int(ns), 'uGather': float(st['gather'])})
        return img[..., :3], x0, y0

    # ---------------------------------------------------------------- sparks
    @staticmethod
    def _temp_color(T):
        """Blackbody-ish HDR ramp for spark temperature T (0..1.5)."""
        xs = np.array([0.0, 0.15, 0.40, 0.70, 1.00, 1.50])
        rs = np.array([0.0, 0.70, 2.70, 5.00, 6.20, 7.20])
        gs = np.array([0.0, 0.07, 0.62, 2.00, 3.50, 5.40])
        bs = np.array([0.0, 0.01, 0.06, 0.32, 1.05, 2.70])
        T = np.clip(T, 0, 1.5)
        return np.stack([np.interp(T, xs, rs), np.interp(T, xs, gs), np.interp(T, xs, bs)], -1)

    @staticmethod
    def _subset(g, m):
        return {kk: v[m] for kk, v in g.items()}

    def _spark_sprites(self, st, i):
        """Tapered 3-segment streak sprites of all live sparks (cached per frame).
        Returns dict(head, tail, rad, col, behind) with one row per sprite."""
        if 'sparks' in st:
            return st['sparks']
        st['sparks'] = None
        sp = self.sp
        t = st['t']
        lo = np.searchsorted(sp['te'], t - self.sp_maxlife)
        hi = np.searchsorted(sp['te'], t, side='right')
        if hi <= lo:
            return None
        a = t - sp['te'][lo:hi]
        idx = np.nonzero(a < sp['life'][lo:hi])[0] + lo
        if len(idx) == 0:
            return None
        a = t - sp['te'][idx]
        k = sp['k'][idx][:, None]
        g = sp['g'][idx]
        vt = np.stack([np.zeros_like(g), g], -1) / k
        p0, v0 = sp['p0'][idx] + st['off'], sp['v0'][idx]

        def pos(aa):
            aa = np.maximum(aa, 0.0)[:, None]
            return p0 + vt * aa + (v0 - vt) * (1 - np.exp(-k * aa)) / k

        s = SHUTTER * sp['shut'][idx]
        P = [pos(a), pos(a - s / 3), pos(a - 2 * s / 3), pos(a - s)]
        life = sp['life'][idx]
        T = sp['temp0'][idx] * np.exp(-a / (sp['cool'][idx] * (0.42 * life + 0.04)))
        env = (0.12 + 0.88 * _sm(a / 0.08)) * (1 - _sm((a - 0.65 * life) / (0.35 * life)))
        fl = _hash(sp['id'][idx], i)
        flick = 0.45 + 0.55 * fl + 1.6 * (fl > 0.95)
        emb = sp['emb'][idx]
        flick = np.where(emb, 0.7 + 0.3 * np.sin(sp['id'][idx].astype(np.float64) * 1.7 + t * 9.0), flick)
        L = np.linalg.norm(P[0] - P[3], axis=1)
        amp = env * flick * K_SPARK * (1.6 / (1.0 + L / 60.0))
        rad = sp['size'][idx] * (0.80 + 0.35 * np.clip(T, 0, 1.2))
        inb = (P[0][:, 0] > -40) & (P[0][:, 0] < W + 40) & (P[0][:, 1] > -40) & (P[0][:, 1] < H + 40)
        beh = sp['behind'][idx]
        heads, tails, rads, cols, behs = [], [], [], [], []
        # head -> tail: hotter, thicker head; cooler, thinner, dimmer tail
        tk, rk, bk = (1.0, 0.84, 0.68), (1.0, 0.78, 0.55), (1.0, 0.82, 0.58)
        for j in range(3):
            pa, pb = P[j], P[j + 1]
            dvec = pb - pa
            dl = np.linalg.norm(dvec, axis=1, keepdims=True)
            u = dvec / (dl + 1e-6)
            rj = rad * rk[j]
            pull = np.minimum(0.35 * rj[:, None], 0.3 * dl)
            if j > 0:
                pa = pa + u * pull
            if j < 2:
                pb = pb - u * pull
            heads.append(pa[inb])
            tails.append(pb[inb])
            rads.append(rj[inb])
            cols.append((self._temp_color(T * tk[j]) * (amp * bk[j])[:, None])[inb])
            behs.append(beh[inb])
        # bright glints on hot heads, soft halos on a few big embers
        hm = inb & (T > 0.55)
        heads.append(P[0][hm]); tails.append(P[0][hm]); rads.append(rad[hm] * 1.5)
        cols.append(self._temp_color(T[hm]) * (amp[hm] * 0.55)[:, None]); behs.append(beh[hm])
        ha = inb & sp['halo'][idx]
        heads.append(P[0][ha]); tails.append(P[0][ha]); rads.append(rad[ha] * 4.5)
        cols.append(self._temp_color(T[ha] * 0.9) * (amp[ha] * 0.10)[:, None]); behs.append(beh[ha])
        out = dict(head=np.concatenate(heads), tail=np.concatenate(tails), rad=np.concatenate(rads),
                   col=np.concatenate(cols), behind=np.concatenate(behs))
        st['sparks'] = out
        return out

    def _rider_sprites(self, st, i):
        rb = self.rb
        t = st['t']
        a = t - rb['te']
        alive = np.nonzero((a >= 0) & (a < rb['life']))[0]
        if len(alive) == 0:
            return None
        a = a[alive]
        phi_t = float(st['phi'])
        shut = rb['shut'][alive]
        phi_tail = self._tab(self.tphi, t - SHUTTER * shut)
        m = rb['m'][alive]
        th = rb['th0'][alive] + m * (phi_t - rb['phi0'][alive])
        # tail = where the rider was one shutter ago (or its birth point if younger than that)
        th_t = rb['th0'][alive] + m * (np.where(a > SHUTTER * shut, phi_tail, rb['phi0'][alive]) - rb['phi0'][alive])
        R, cen, wd = st['R'], st['cen'], st['width']
        suck = st['gather'] * (rb['suck'][alive] < 0.5) * rb['suck'][alive] * 2 * (14.0 + 0.06 * R)
        r = R * self.rim_factor(th, t) + rb['dr'][alive] * wd - suck
        r_t = R * self.rim_factor(th_t, t) + rb['dr'][alive] * wd - suck * 0.6
        head = cen + np.stack([np.cos(th), np.sin(th)], -1) * r[:, None]
        tail = cen + np.stack([np.cos(th_t), np.sin(th_t)], -1) * r_t[:, None]
        life = rb['life'][alive]
        env = _sm(a / 0.02) * (1 - _sm((a - 0.6 * life) / (0.4 * life)))
        fl = _hash(rb['id'][alive], i, 3)
        T = 0.85 + 0.55 * fl - 0.5 * np.abs(rb['dr'][alive]) / 1.5
        col = self._temp_color(T) * (env * (0.55 + 0.45 * fl) * st['I'] * K_RIDER)[:, None]
        L = np.linalg.norm(head - tail, axis=1)
        col *= (1.0 / (1.0 + L / 80.0))[:, None] * 1.4
        inb = (head[:, 0] > -40) & (head[:, 0] < W + 40) & (head[:, 1] > -40) & (head[:, 1] < H + 40)
        return dict(head=head[inb], tail=tail[inb], col=col[inb], rad=rb['size'][alive][inb])

    @staticmethod
    def _splat(groups):
        """Render a list of sprite dicts additively into a crop (rgb, x0, y0)."""
        p0, p1, rad, col = [], [], [], []
        for gdict in groups:
            if gdict is None or len(gdict['head']) == 0:
                continue
            p0.append(gdict['head'])
            p1.append(gdict['tail'])
            rad.append(gdict['rad'])
            col.append(gdict['col'])
        if not p0:
            return None
        p0 = np.concatenate(p0)
        p1 = np.concatenate(p1)
        rad = np.concatenate(rad)
        col = np.concatenate(col)
        pad = 12 + 2 * float(rad.max())
        lo = np.minimum(p0, p1).min(0) - pad
        hi = np.maximum(p0, p1).max(0) + pad
        x0, y0 = int(max(0, lo[0])), int(max(0, lo[1]))
        x1, y1 = int(min(W, hi[0])), int(min(H, hi[1]))
        if x1 <= x0 or y1 <= y0:
            return None
        c4 = np.concatenate([col, np.ones((len(col), 1))], 1)
        img = gl.sprites((x1 - x0, y1 - y0), p0, p1, radius=rad, color=c4, offset=(x0, y0), shape=2)
        return img[..., :3], x0, y0

    # ---------------------------------------------------------------- post: wipe, light wrap, front layers
    def post(self, S, img):
        st = self.state(S)
        t, i = st['t'], S.i
        store = getattr(S, '_strange_store', None) or dict(emit=[], plate=None)
        done = self._done(st)
        occ = S.alpha
        mfront = None
        araw = None
        # ---- wipe back to the raw footage (relit by the ring)
        if t >= T_REL:
            Rp = st['Rp']
            raw = _RAW_LUT()[A.frame_u8(i)]
            araw = A.alpha(i)
            if done:
                img = raw.copy()
                occ = araw
            elif Rp > 1.0:
                mfront = self.disk_mask(CF, Rp, t, soft=2.0)
                rawp = self._relight_raw(S, st, raw, araw)
                img = img * (1 - mfront)[..., None] + rawp * mfront[..., None]
                occ = S.alpha * (1 - mfront) + araw * mfront
                # the background-plane emission (ring, riders, behind sparks) inside the front sits over the
                # raw footage now, still behind him
                for (rgb, x0, y0, dark) in store['emit']:
                    h, w = rgb.shape[:2]
                    mm = mfront[y0:y0 + h, x0:x0 + w] * (1 - araw[y0:y0 + h, x0:x0 + w])
                    _add_emissive(img[y0:y0 + h, x0:x0 + w], rgb, dark, mm)
                self._front_line(S, st, img)
        if t >= TW0:
            # fade the global post so the very end is the untouched footage
            k = float(_ss(TW0, 74.65, t))
            S.vignette *= 1 - k
            S.grain *= 1 - k
            S.bloom *= 1 - 0.65 * float(_ss(74.4, 74.95, t)) - 0.35 * float(_ss(75.0, 75.5, t))
        # ---- light wrap: the ring + bright bathroom behind him bleed over his silhouette edges
        if st['b'] > 0.5 and not done:
            self._light_wrap(S, st, img, occ, store, mfront)
        # ---- front layers
        b = st['b']
        sp = self._spark_sprites(st, i)
        layers = []
        if b < 0.5 and not done:
            layers.append((self._ring(st), 0.55))
            fg = [self._rider_sprites(st, i)]
            if sp is not None:
                fg.append(sp)
            layers.append((self._splat(fg), 0.40))
        else:
            if sp is not None:
                layers.append((self._splat([self._subset(sp, ~sp['behind'])]), 0.40))
            if done and sp is not None:
                # leftover sparks born behind him, now over the raw footage (still behind him)
                lay = self._splat([self._subset(sp, sp['behind'])])
                if lay is not None:
                    rgb, x0, y0 = lay
                    h, w = rgb.shape[:2]
                    rgb = rgb * (1 - occ[y0:y0 + h, x0:x0 + w])[..., None]
                    layers.append(((rgb, x0, y0), 0.35))
        pg = self._preglow(st, i)
        if pg is not None:
            layers.append((pg, 0.0))
        for lay, dark in layers:
            if lay is None:
                continue
            rgb, x0, y0 = lay
            h, w = rgb.shape[:2]
            _add_emissive(img[y0:y0 + h, x0:x0 + w], rgb, dark)
        self._ignition_flare(st, img)
        return img

    def _relight_raw(self, S, st, raw, araw):
        """The person restored by the wipe keeps reacting to the ring: relight the raw person with the same
        light list (core._relight on a scratch scene; nothing in core is modified)."""
        L = self._light_list(S, st)
        if not L or st['I'] < 0.01:
            return raw
        S2 = types.SimpleNamespace(fg=raw.copy(), alpha=araw, bg=np.zeros((8, 8, 3), np.float32))
        Lfg = [dict(l, bg=0.0) for l in L]
        core._relight(S2, Lfg)
        return raw + (S2.fg - raw) * araw[..., None]

    def _front_line(self, S, st, img):
        """Gold energy line where the reality front crosses his body. Full on the suit (it burns off to skin),
        thin and dim on face and hands (nothing changes there)."""
        t, Rp, i = st['t'], st['Rp'], S.i
        fade = 1 - float(_ss(74.0, 74.3, t))
        if not (2 < Rp < 1300) or fade < 0.01:
            return
        rm = Rp * (1 + WOB_A.sum()) + 60
        pa = np.maximum(S.alpha, A.alpha(i))
        ys, xs = np.nonzero(pa[::8, ::8] > 0.05)
        if not len(xs):
            return
        bx0, by0 = max(int(CF[0] - rm), xs.min() * 8 - 8, 0), max(int(CF[1] - rm), ys.min() * 8 - 8, 0)
        bx1, by1 = min(int(CF[0] + rm), xs.max() * 8 + 16, W), min(int(CF[1] + rm), ys.max() * 8 + 16, H)
        if bx1 <= bx0 or by1 <= by0:
            return
        X, Y = core.grid(by1 - by0, bx1 - bx0, (bx0, by0))
        rho = np.sqrt((X - CF[0]) ** 2 + (Y - CF[1]) ** 2)
        th = np.arctan2(Y - CF[1], X - CF[0])
        dd = rho - Rp * self.rim_factor(th, t)
        near = np.abs(dd) < 80
        if not near.any():
            return
        sk = np.zeros((by1 - by0, bx1 - bx0), np.uint8)
        for poly in _skin_polys(i):
            cv2.fillPoly(sk, [np.round(poly - (bx0, by0)).astype(np.int32)], 255)
        sk = cv2.GaussianBlur(sk.astype(np.float32) / 255.0, (0, 0), 5.0)
        # sparkle as noise along the arc length (no radial spokes near the origin)
        sparkle = 0.70 + 0.45 * _vnoise1(th * max(Rp, 20.0) / 40.0 + t * 6.0)
        e_suit = (0.55 * np.exp(-(dd / 24.0) ** 2) + 0.45 * np.exp(-(dd / 6.0) ** 2)) * sparkle
        e_skin = 0.28 * np.exp(-(dd / 4.0) ** 2)
        e = (e_suit * (1 - sk) + e_skin * sk) * pa[by0:by1, bx0:bx1] * np.clip(Rp / 60.0, 0, 1)
        img[by0:by1, bx0:bx1] += (e[..., None] * np.array([1.9, 0.95, 0.25])).astype(np.float32) * fade

    def _light_wrap(self, S, st, img, occ, store, mfront):
        """Crisp light wrap: blur what glows behind him (ring band, riders, behind sparks, the bathroom seen
        through the portal) and add it on a ~9 px inner band of his matte (face interior excluded)."""
        srcs = [(rgb, x0, y0, K_WRAP_RING) for (rgb, x0, y0, _) in store['emit']]
        if store['plate'] is not None:
            srcs.append(store['plate'] + (K_WRAP_PLATE,))
        if not srcs:
            return
        a8 = occ[::8, ::8]
        ys, xs = np.nonzero(a8 > 0.05)
        if not len(xs):
            return
        rx0, ry0 = max(0, xs.min() * 8 - 32), max(0, ys.min() * 8 - 32)
        rx1, ry1 = min(W, xs.max() * 8 + 40), min(H, ys.max() * 8 + 40)
        hh, ww = (ry1 - ry0) // 2 * 2, (rx1 - rx0) // 2 * 2
        ry1, rx1 = ry0 + hh, rx0 + ww
        if hh < 8 or ww < 8:
            return
        E = np.zeros((hh, ww, 3), np.float32)
        anyv = False
        for (rgb, x0, y0, kw) in srcs:
            h, w = rgb.shape[:2]
            ix0, iy0 = max(rx0, x0), max(ry0, y0)
            ix1, iy1 = min(rx1, x0 + w), min(ry1, y0 + h)
            if ix1 <= ix0 or iy1 <= iy0:
                continue
            src = rgb[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0]
            if mfront is not None:            # inside the front the raw footage carries its own real light
                src = src * (1 - mfront[iy0:iy1, ix0:ix1])[..., None] if kw == K_WRAP_PLATE else src
            E[iy0 - ry0:iy1 - ry0, ix0 - rx0:ix1 - rx0] += src * kw
            anyv = True
        if not anyv:
            return
        Eh = cv2.resize(E, (ww // 2, hh // 2), interpolation=cv2.INTER_AREA)
        Eh = cv2.GaussianBlur(Eh, (0, 0), 7.0)
        Ef = cv2.resize(Eh, (ww, hh), interpolation=cv2.INTER_LINEAR)
        a = np.ascontiguousarray(occ[ry0:ry1, rx0:rx1])
        er = cv2.erode(a, np.ones((15, 15), np.uint8))
        edge = cv2.GaussianBlur(a * (1 - er), (0, 0), 1.2)
        fm = np.zeros((hh, ww), np.uint8)
        fo = S.tx(A.face_oval(S.i)).astype(np.float64)
        cv2.fillPoly(fm, [np.round(fo - (rx0, ry0)).astype(np.int32)], 255)
        fm = cv2.GaussianBlur(fm.astype(np.float32) / 255.0, (0, 0), 6.0)
        edge *= 1 - 0.7 * fm
        img[ry0:ry1, rx0:rx1] += Ef * (edge * st['b'])[..., None]

    def _preglow(self, st, i):
        """Flickering ember on the fingertips before the ignition flash."""
        pre = _pre(st['t'])
        if pre < 0.01:
            return None
        pp = self._trk(self.tr_pen, st['t']) + st['off']
        fl = 0.55 + 0.45 * _hash([3], i)[0]
        c_core = self._temp_color(1.3) * 0.55 * pre * fl
        c_halo = self._temp_color(0.9) * 0.10 * pre * fl
        p = np.array([pp, pp])
        col = np.concatenate([np.stack([c_core, c_halo]), np.ones((2, 1))], 1)
        x0, y0 = int(pp[0] - 60), int(pp[1] - 60)
        im = gl.sprites((120, 120), p, None, radius=np.array([3.0, 16.0]), color=col, offset=(x0, y0), shape=2)
        return im[..., :3], x0, y0

    def _ignition_flare(self, st, img):
        """Short tilted glint + soft flash through the white-hot spark at ignition (70.10)."""
        k = float(_pulse(st['t'], 70.105, 0.08))
        if k < 0.01:
            return
        hp = self._tab(self.tcen, st['t']) + st['off']
        x0, x1 = int(max(0, hp[0] - 300)), int(min(W, hp[0] + 301))
        y0, y1 = int(max(0, hp[1] - 150)), int(min(H, hp[1] + 151))
        if y1 <= y0 or x1 <= x0:
            return
        X, Y = core.grid(y1 - y0, x1 - x0, (x0, y0))
        dx, dy = X - hp[0], Y - hp[1]
        ca, sa = np.cos(np.radians(5.0)), np.sin(np.radians(5.0))     # tilted away from the black-hole disk
        u = dx * ca + dy * sa
        v = -dx * sa + dy * ca
        au = np.abs(u)
        streak = np.exp(-(v / 2.0) ** 2) * (np.exp(-au / 65.0) + 0.6 * np.exp(-au / 22.0)) * _ss(155.0, 80.0, au)
        halo = np.exp(-(dx * dx + dy * dy) / (2 * 24.0 ** 2))
        col_s = np.array([1.0, 0.62, 0.30], np.float32)
        col_h = np.array([1.6, 1.15, 0.7], np.float32)
        img[y0:y1, x0:x1] += (streak[..., None] * col_s * 1.3 + halo[..., None] * col_h * 1.2).astype(np.float32) * k


# ------------------------------------------------------------------ helpers
def _add_emissive(d, rgb, dark, mask=None):
    """d += rgb (in place). With dark > 0, bright emitters over bright plates (white captions, tiles) get a
    soft dark surround first so they still read (white + orange would otherwise just be white)."""
    if dark > 0:
        lum = rgb[..., 0] * 0.3 + rgb[..., 1] * 0.6 + rgb[..., 2] * 0.1
        cov = np.clip(cv2.GaussianBlur(lum, (0, 0), 3.0) / 0.6, 0, 1) * dark
        if mask is not None:
            cov = cov * mask
        d *= (1.0 - cov)[..., None]
    d += rgb if mask is None else rgb * mask[..., None]


def _inv_tonemap(y):
    """Inverse of core._tonemap's shoulder so that the final output equals the raw footage."""
    k = 0.8
    y = np.clip(y, 0, 0.996)
    over = np.maximum(y - k, 0) / (1 - k)
    x = np.where(y > k, k - (1 - k) * np.log(np.maximum(1 - over, 1e-4)), y)
    return x.astype(np.float32)


_RL = None


def _RAW_LUT():
    """uint8 sRGB -> linear value that the compositor's tonemap maps back to the same sRGB."""
    global _RL
    if _RL is None:
        _RL = _inv_tonemap(A.srgb_to_lin(np.arange(256) / 255.0)).astype(np.float32)
    return _RL
