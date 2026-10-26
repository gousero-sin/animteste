"""strange — Doctor Strange sling-ring portal ending (EV['strange'][0] = 70.05 s -> end).

Choreography (from frame-by-frame study of his R hand, tracking + footage):
  70.08-70.60  ignition: a white-hot spark orbits his raised two fingertips and DRAWS a ring
               (rider sparks lag behind the orbiting head -> the trail closes into a circle)
  70.60-72.45  small sparkler ring (R 36 -> 62) spinning at the fingertips, in FRONT of him,
               showering gravity-bent streaks over his arm/face, crackle bursts on word onsets;
               follows the hand (with lag) when it drops 71.0-71.2 and moves to the chest 72.0-72.4
  72.45-73.30  he circles the hand (counter-clockwise on screen): the ring grows 62 -> 395 px with an
               overshoot/settle, migrates up/back to a halo position behind his head, passes BEHIND
               him (occlusion ramps in), and the aperture opens onto the original bathroom (plate 0)
  73.30-73.60  open portal, raging, wobbling rim
  73.60-74.70  wipe: anticipation contraction, then accelerating expansion past the frame edges;
               a reality front from the portal centre turns everything (him included) back into the
               raw footage A.frame(i) (gold edge on his body), sparks ride the expanding rim.
  74.70-end    raw footage (exactly: inverse-tonemapped, bloom/vignette/grain faded), last sparks fall.

Everything is an analytic function of t: spark emission times / initial states are precomputed
once in __init__ from a seeded RNG and the (smoothed) tracking; positions are evaluated in closed
form (drag + gravity). Rendering: GLSL procedural hot ring band + instanced streak sprites.
"""
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
TW0, TW1 = EV['strange_wipe']          # 73.60, 74.70
T_IGN = 70.08                          # first spark
T_GROW0, T_GROW1 = 72.45, 73.30        # portal growth while he circles
CF = np.array([490.0, 890.0])          # final portal centre (behind his head, slightly frame-left)
R_SMALL0, R_SMALL1, R_OPEN = 36.0, 62.0, 395.0
GRAV = 1900.0                          # px/s^2
SHUTTER = 0.030                        # streak length in seconds of motion
WOB_A = np.array([0.020, 0.015, 0.009, 0.006])      # rim wobble amplitudes (fraction of R)
WOB_K = np.array([2.0, 3.0, 5.0, 7.0])
WOB_NU = np.array([1.7, -2.3, 3.1, -4.4])           # rad/s
WOB_P0 = np.array([0.3, 2.1, 4.0, 1.2])
WORD_PULSES = (70.10, 71.06, 71.86, 72.10, 72.28, 72.66)
# emission gains (linear HDR)
K_RING = 0.36      # GLSL ring band
K_RIDER = 0.13     # sparks riding the band
K_SPARK = 0.85     # flying sparks
DT_TAB = 1.0 / 1200.0


def _sm(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def _ss(a, b, x):
    return _sm((np.asarray(x, np.float64) - a) / (b - a))


def _pulse(t, t0, decay):
    t = np.asarray(t, np.float64)
    return _ss(t0 - 0.035, t0, t) * np.exp(-np.maximum(t - t0, 0.0) / decay)


def _ease_out_back(x, s=1.25):
    x = np.clip(x, 0, 1) - 1
    return x * x * ((s + 1) * x + s) + 1


def _hash(ids, frame, salt=0):
    """Deterministic per-(id, frame) uniform in [0,1)."""
    x = (np.asarray(ids, np.uint64) * np.uint64(0x9E3779B1) + np.uint64(frame * 0x85EBCA77 + salt * 0xC2B2AE3D + 12345))
    x ^= x >> np.uint64(15)
    x *= np.uint64(0x2C1B3C6D)
    x ^= x >> np.uint64(12)
    x *= np.uint64(0x297A2D39)
    x ^= x >> np.uint64(15)
    return (x & np.uint64(0xFFFFFF)).astype(np.float64) / float(0x1000000)


# ------------------------------------------------------------------ scalar timeline
def _grow_curve(u):
    """0 -> 1 S-curve with a small anticipation dip (~-3.5%) and an overshoot (+4% at u=0.85),
    zero slope at both ends (quintic fit)."""
    u = float(np.clip(u, 0, 1))
    f = -7.797 * u ** 2 + 39.586 * u ** 3 - 50.783 * u ** 4 + 19.993 * u ** 5
    return f * 0.5 if f < 0 else f


def _R_scalar(t):
    if t < T_IGN:
        return 0.0
    if t < 70.62:
        return R_SMALL0 * (1 - (1 - (t - T_IGN) / 0.54) ** 3)
    if t < T_GROW0:
        u = (t - 70.62) / (T_GROW0 - 70.62)
        breath = 2.5 * np.sin(2 * np.pi * 1.6 * (t - 70.62)) * (1 - _ss(72.15, T_GROW0, t))
        return R_SMALL0 + (R_SMALL1 - R_SMALL0) * float(_sm(u)) + float(breath)
    breath = 5.0 * np.sin(2 * np.pi * 1.3 * (t - T_GROW1)) if t > T_GROW1 else 0.0
    if t < T_GROW1:
        u = (t - T_GROW0) / (T_GROW1 - T_GROW0)
        return R_SMALL1 + (R_OPEN - R_SMALL1) * _grow_curve(u)
    r = R_OPEN + breath
    # wipe: anticipation contraction, then accelerating expansion past the frame
    ud = np.clip((t - TW0) / 0.20, 0, 1)
    r -= 18.0 * np.sin(np.pi * ud) ** 2
    ue = (t - 73.70) / 0.85
    if ue > 0:
        r += 1000.0 * min(ue, 1.0) ** 2 + max(ue - 1.0, 0) * 2000.0
    return float(r)


def _I_scalar(t):
    i = 0.95 * float(_ss(T_IGN - 0.01, 70.24, t))
    i += 0.35 * float(_ss(T_GROW0, 73.20, t)) + 0.25 * float(_ss(TW0, 73.85, t))
    for k, tp in enumerate(WORD_PULSES):
        i += (1.4 if k == 0 else 0.45) * float(_pulse(t, tp, 0.09 if k == 0 else 0.14))
    i *= 1 - float(_ss(74.42, 74.62, t))
    return i


def _E_scalar(t, R):
    """Spark emission rate (1/s)."""
    e = 750.0 * float(_ss(T_IGN, 70.30, t)) + 380.0 * float(_ss(70.6, 72.4, t))
    e += 12.5 * max(R - R_SMALL1, 0.0) * (1.0 if t < TW0 else 0.8)
    pul = sum(float(_pulse(t, tp, 0.10)) for tp in WORD_PULSES[1:])
    e *= 1 + 1.6 * pul
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

    # ---------------------------------------------------------------- tracking
    def _build_track(self):
        i0, i1 = C.i_of(69.2), C.N_FRAMES
        idx = np.arange(i0, i1)
        pen = np.zeros((len(idx), 2))
        fdir = np.zeros((len(idx), 2))
        for k, i in enumerate(idx):
            h = A.hand(i, 'R')
            if h is not None:
                tip = (h[8, :2] + h[12, :2]) * 0.5
                base = (h[5, :2] + h[9, :2]) * 0.5
                d = tip - base
                d = d / (np.linalg.norm(d) + 1e-6)
                pen[k] = tip + d * 10.0
                fdir[k] = d
            else:
                p = A.palm(i, 'R')
                up = A.palm_normal_up(i, 'R')
                pen[k] = p + up * 0.7 * A.hand_size(i, 'R')
                fdir[k] = up
        pen = median_filter(pen, size=(5, 1), mode='nearest')
        pen_s = gaussian_filter1d(pen, 1.6, axis=0, mode='nearest')
        fdir = gaussian_filter1d(fdir, 2.0, axis=0, mode='nearest')
        fdir /= np.linalg.norm(fdir, axis=1, keepdims=True) + 1e-6
        # ring follows the fingertips with a slight lag (causal one-pole, precomputed)
        lag = np.zeros_like(pen_s)
        lag[0] = pen_s[0]
        aa = 1 - np.exp(-(1 / C.FPS) / 0.055)
        for k in range(1, len(pen_s)):
            lag[k] = lag[k - 1] + (pen_s[k] - lag[k - 1]) * aa
        circ = gaussian_filter1d(pen, 9.0, axis=0, mode='nearest')   # centre of the circling motion
        ang = np.unwrap(np.arctan2(pen_s[:, 1] - circ[:, 1], pen_s[:, 0] - circ[:, 0]))
        ang = gaussian_filter1d(ang, 1.5, mode='nearest')
        self.tr_t = idx / C.FPS
        self.tr_pen, self.tr_lag, self.tr_circ, self.tr_ang = pen_s, lag, circ, ang

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
        rev = 0.85 + 1.45 * np.clip((400.0 - R) / 340.0, 0, 1)
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
        self.tb = _ss(72.42, 72.88, tg)                 # ring goes behind him
        self.tap = _ss(72.56, 72.98, tg)                # aperture opens
        self.twidth = 3.6 + 0.021 * R                   # band half-width px
        # hotspot: ignition = orbiting head; growth = the circling hand
        hand_ang = self._trk(self.tr_ang, tg)
        self.thot_ang = np.where(tg < 71.0, phi + 0.0, hand_ang)
        self.thot_w = np.maximum(1 - _ss(70.40, 70.62, tg), 0) * (tg >= T_IGN) \
            + 0.8 * _ss(72.30, 72.50, tg) * (1 - _ss(140.0, 300.0, R))
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
    def _emit_angles(self, te, n_rng):
        """Emission angle on the ring for each emission time."""
        rng = n_rng
        n = len(te)
        th = rng.uniform(-np.pi, np.pi, n)
        hw = self._tab(self.thot_w, te)
        span = self._tab(self.tspan, te)
        phi = self._tab(self.tphi, te)
        hot_ang = self._tab(self.thot_ang, te)
        use_hot = rng.random(n) < np.clip(hw, 0, 0.92)
        th = np.where(use_hot, hot_ang + rng.normal(0, 0.28, n), th)
        # during the ignition orbit, non-hot sparks only come from the drawn arc (behind the head)
        arc = span < 6.28
        th_arc = phi + rng.random(n) * span
        th = np.where(arc & ~use_hot, th_arc, th)
        return th

    def _build_sparks(self):
        rng = self.rng
        tg, E = self.tg, self.tE
        cum = np.cumsum(E) * DT_TAB
        n = int(cum[-1])
        u = (np.arange(n) + rng.random(n)) / max(n, 1) * cum[-1]
        te = np.interp(u, cum, tg)
        # explicit bursts: ignition flash + crackles on word onsets + wipe release
        bursts = [(70.10, 170, 1.6), (71.06, 90, 1.3), (71.86, 70, 1.2), (72.28, 70, 1.2),
                  (72.66, 160, 1.3), (73.30, 260, 1.2), (73.74, 520, 1.5)]
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
        v0 = tan * st[:, None] + rad * sr[:, None] + vel * 0.55 + rng.normal(0, 35, (n, 2))
        life = np.exp(rng.normal(np.log(0.36), 0.45, n)).clip(0.10, 1.1)
        life *= np.where(R < 80, 0.8, 1.0)
        k = rng.uniform(2.0, 4.6, n)
        g = np.full(n, GRAV) * rng.uniform(0.8, 1.15, n)
        size = rng.uniform(0.85, 1.55, n)
        temp0 = rng.uniform(0.85, 1.35, n)
        # embers: slow, floaty, long-lived, dimmer, a bit larger
        emb = rng.random(n) < 0.07
        life = np.where(emb, rng.uniform(0.9, 2.0, n), life)
        k = np.where(emb, rng.uniform(4.5, 7.0, n), k)
        g = np.where(emb, rng.uniform(250, 600, n), g)
        size = np.where(emb, rng.uniform(1.3, 2.0, n), size)
        temp0 = np.where(emb, rng.uniform(0.55, 0.8, n), temp0)
        b = self._tab(self.tb, te)
        behind = rng.random(n) < 0.70 * b
        # cull sparks born far outside the frame (big wipe ring)
        keep = (p0[:, 0] > -350) & (p0[:, 0] < W + 350) & (p0[:, 1] > -350) & (p0[:, 1] < H + 250)
        sel = np.nonzero(keep)[0]
        order = np.argsort(te[sel], kind='stable')
        sel = sel[order]
        self.sp = dict(te=te[sel], p0=p0[sel], v0=v0[sel], life=life[sel], k=k[sel], g=g[sel],
                       size=size[sel], temp0=temp0[sel], emb=emb[sel], behind=behind[sel],
                       id=np.arange(len(sel), dtype=np.uint64))
        self.sp_maxlife = float(self.sp['life'].max()) if len(sel) else 0.0

    def _build_riders(self):
        """Sparks that ride the ring band (the dense spinning ring texture)."""
        rng = np.random.default_rng(70052)
        tg = self.tg
        circ = 2 * np.pi * np.maximum(self.tR, 1.0)
        Eb = (3.0 * circ + 400.0 * (self.tR > 1)) * np.clip(self.tI, 0, 1.6)
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
        m = np.where(ign, rng.uniform(0.55, 0.95, n), rng.uniform(0.85, 1.35, n))
        life = np.where(ign, rng.uniform(0.12, 0.40, n), rng.uniform(0.06, 0.17, n))
        dr = rng.normal(0, 0.55, n)
        # cull riders that are outside the frame during the wipe (approximate, by emission pos)
        cen = self._tab(self.tcen, te)
        p = cen + np.stack([np.cos(th), np.sin(th)], -1) * R[:, None]
        keep = (p[:, 0] > -150) & (p[:, 0] < W + 150) & (p[:, 1] > -150) & (p[:, 1] < H + 150)
        sel = np.nonzero(keep)[0]
        self.rb = dict(te=te[sel], th0=th[sel], m=m[sel], life=life[sel], dr=dr[sel],
                       phi0=self._tab(self.tphi, te[sel]), size=rng.uniform(0.8, 1.5, n)[sel],
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
                  phi=float(self._tab(self.tphi, t)), b=float(self._tab(self.tb, t)),
                  ap=float(self._tab(self.tap, t)), width=float(self._tab(self.twidth, t)),
                  span=float(self._tab(self.tspan, t)), hot_ang=float(self._tab(self.thot_ang, t)),
                  hot_w=float(self._tab(self.thot_w, t)))
        # wipe: reality front (person + everything) radius, follows from the portal centre
        if t >= TW0:
            st['Rp'] = R * float(_ss(TW0 + 0.02, 74.06, t)) ** 0.85
        else:
            st['Rp'] = 0.0
        # flicker of the whole ring (subtle, never strobing)
        fi = S.i
        st['flick'] = 0.92 + 0.08 * np.sin(fi * 2.1) * np.sin(fi * 0.77 + 1.3) + 0.05 * (_hash([7], fi)[0] - 0.5)
        S._strange_state = st
        return st

    def active(self, t):
        return t >= T0 - 1e-6

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
        if self._done(st) or st['ap'] <= 0.001 or st['R'] < 20:
            return
        R, cen, t = st['R'], st['cen'], st['t']
        Rm = R * 1.06 + 8
        x0, y0 = int(max(0, cen[0] - Rm)), int(max(0, cen[1] - Rm))
        x1, y1 = int(min(W, cen[0] + Rm + 1)), int(min(H, cen[1] + Rm + 1))
        if x1 <= x0 or y1 <= y0:
            return
        m = self.disk_mask(cen, R, t, (y1 - y0, x1 - x0), (x0, y0), inset=st['width'] * 0.35, soft=1.2)
        m *= st['ap']
        view = A.plate(0)[y0:y1, x0:x1]
        # warm spill from the ring onto the other side + slight darkening at the very rim
        X, Y = core.grid(y1 - y0, x1 - x0, (x0, y0))
        rho = np.sqrt((X - cen[0]) ** 2 + (Y - cen[1]) ** 2)
        edge = np.clip(1.0 - rho / max(R, 1.0), 0, 1)
        spill = np.exp(-edge * R / (0.10 * R + 12.0)).astype(np.float32)
        warm = np.array([1.0, 0.55, 0.20], np.float32)
        view = view * (1.0 - 0.22 * spill)[..., None] + spill[..., None] * warm * (0.12 * st['I'])
        # opening flare: the other side is over-exposed for a moment as the aperture opens
        flare = float(_pulse(t, 72.92, 0.16)) * 0.25
        if flare > 0.01:
            view = view * (1 + flare) + flare * 0.15
        bg = S.bg[y0:y1, x0:x1]
        bg *= (1 - m)[..., None]
        bg += view * m[..., None]
        # heat haze around the rim (background only)
        self._haze(S, st, x0, y0, x1, y1, rho, X, Y)

    def _haze(self, S, st, x0, y0, x1, y1, rho, X, Y):
        R, cen, t = st['R'], st['cen'], st['t']
        wdt = st['width']
        d = rho - R
        band = np.exp(-(d / (wdt * 3.0 + 14.0)) ** 2) * min(1.0, st['I'])
        if band.max() < 0.02:
            return
        th = np.arctan2(Y - cen[1], X - cen[0])
        amp = 2.2 * band * np.sin(d * 0.22 - t * 23.0 + 2.5 * np.sin(th * 9.0 + t * 4.0))
        nx, ny = (X - cen[0]) / (rho + 1e-3), (Y - cen[1]) / (rho + 1e-3)
        S.distort.append((x0, y0, (nx * amp).astype(np.float32), (ny * amp).astype(np.float32)))

    def lights(self, S):
        st = self.state(S)
        if self._done(st) or st['I'] < 0.01:
            return []
        I, R, cen, wF = st['I'] * st['flick'], st['R'], st['cen'], st['wF']
        gold = (1.0, 0.50, 0.15)
        out = []
        small = 1.0 - wF
        if small > 0.01:
            out.append(Light(cen, gold, intensity=1.25 * I * small, radius=150 + R * 1.5, rim=1.3,
                             diffuse=0.85, bg=0.05))
        if wF > 0.01 and R < 1600:
            n = 10
            for k in range(n):
                a = st['phi'] * 0.15 + 2 * np.pi * k / n
                p = cen + R * np.array([np.cos(a), np.sin(a)])
                if -400 < p[0] < W + 400 and -400 < p[1] < H + 400:
                    out.append(Light(p, gold, intensity=0.42 * I * wF, radius=0.45 * R + 120, rim=1.7,
                                     diffuse=0.45, bg=0.03))
            # the bright bathroom behind him back-lights his silhouette
            if st['ap'] > 0.01:
                out.append(Light(cen + np.array([0.0, -0.25 * R]), (0.85, 0.88, 1.0),
                                 intensity=0.55 * st['ap'] * min(1.0, R / R_OPEN), radius=0.9 * R,
                                 rim=1.4, diffuse=0.10, bg=0.0))
        # the hand that draws the ring
        if st['hot_w'] > 0.05 and st['t'] > 72.3:
            hp = self._trk(self.tr_pen, st['t']) + st['off']
            out.append(Light(hp, gold, intensity=0.7 * st['hot_w'] * I, radius=160, rim=0.8, diffuse=0.9, bg=0.02))
        return out

    # ---------------------------------------------------------------- ring band (GLSL)
    FRAG = r'''
uniform vec2 uC; uniform float uR; uniform float uPhase; uniform float uI; uniform float uW;
uniform vec4 uWa; uniform vec4 uWp; uniform float uSpan; uniform float uHead;
uniform float uHotA; uniform float uHotW; uniform float uT;
float rimR(float th){
  return uR * (1.0 + uWa.x*sin(2.0*th + uWp.x) + uWa.y*sin(3.0*th + uWp.y)
                   + uWa.z*sin(5.0*th + uWp.z) + uWa.w*sin(7.0*th + uWp.w));
}
float angd(float a, float b){ float d = mod(a - b + PI, TAU) - PI; return d; }
void main(){
  vec2 p = fpx() - uC;
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
  float hd = angd(th, uHotA) / 0.45; float hot = 1.0 + uHotW * 2.2 * exp(-hd * hd);
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
    float thick = Wd * (0.65 + 0.7 * clump);
    float dc = d / (Wd * 0.20 + 0.6); float core = exp(-dc * dc);
    float db = d / thick; float band = exp(-db * db);
    // jagged spray: radial spikes outward (sparks leaving the rim), shorter inward
    float spk = snoise(vec3(cs * fa * 2.6 + 1.3, uT * 5.0));
    float spk0 = max(spk, 0.0); float spikes = spk0 * spk0 * spk0;
    float sprayO = d > 0.0 ? exp(-d / (thick * (1.2 + 5.0 * spikes))) : exp(d / (thick * 0.7));
    float bright = 0.45 + 0.75 * clump * clump2 + 0.3 * clump;
    col += vec3(4.6, 3.3, 1.7) * core * (0.30 + 1.1 * fil) * bright;
    col += vec3(3.3, 1.35, 0.26) * band * (0.10 + 1.6 * fil) * bright;
    col += vec3(1.9, 0.62, 0.10) * sprayO * (0.06 + 1.2 * fil) * bright * 0.8;
  }
  col *= uI * arcm * hot;
  fragColor = vec4(col, 1.0);
}
'''

    def _ring(self, st):
        """Render the hot ring band. Returns (img, x0, y0) or None."""
        R, cen = st['R'], st['cen']
        if R < 1.0 or st['I'] < 0.005:
            return None
        m = R * 1.08 + st['width'] * 9 + 0.35 * R + 40
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
            'uHotA': float(st['hot_ang']), 'uHotW': float(st['hot_w']), 'uT': float(t)})
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

    def _spark_sprites(self, st, i):
        sp = self.sp
        t = st['t']
        lo = np.searchsorted(sp['te'], t - self.sp_maxlife)
        hi = np.searchsorted(sp['te'], t, side='right')
        if hi <= lo:
            return None
        s = slice(lo, hi)
        a = t - sp['te'][s]
        alive = a < sp['life'][s]
        idx = np.nonzero(alive)[0] + lo
        if len(idx) == 0:
            return None
        a = t - sp['te'][idx]
        k = sp['k'][idx][:, None]
        g = sp['g'][idx]
        vt = np.stack([np.zeros_like(g), g], -1) / k
        p0, v0 = sp['p0'][idx] + st['off'], sp['v0'][idx]

        def pos(aa):
            aa = aa[:, None]
            return p0 + vt * aa + (v0 - vt) * (1 - np.exp(-k * aa)) / k

        head = pos(a)
        tail = pos(np.maximum(a - SHUTTER, 0.0))
        life = sp['life'][idx]
        T = sp['temp0'][idx] * np.exp(-a / (0.42 * life + 0.04))
        env = (0.25 + 0.75 * _sm(a / 0.06)) * (1 - _sm((a - 0.65 * life) / (0.35 * life)))
        fl = _hash(sp['id'][idx], i)
        flick = 0.45 + 0.55 * fl + 1.6 * (fl > 0.95)
        emb = sp['emb'][idx]
        flick = np.where(emb, 0.7 + 0.3 * np.sin(sp['id'][idx].astype(np.float64) * 1.7 + t * 9.0), flick)
        col = self._temp_color(T) * (env * flick)[:, None] * K_SPARK
        # a streak's energy is spread over its length: keep brightness per pixel plausible
        L = np.linalg.norm(head - tail, axis=1)
        col *= (1.0 / (1.0 + L / 60.0))[:, None] * 1.6
        rad = sp['size'][idx] * (0.80 + 0.35 * np.clip(T, 0, 1.2))
        inb = (head[:, 0] > -40) & (head[:, 0] < W + 40) & (head[:, 1] > -40) & (head[:, 1] < H + 40)
        return dict(head=head[inb], tail=tail[inb], col=col[inb], rad=rad[inb], T=T[inb],
                    behind=sp['behind'][idx][inb])

    def _rider_sprites(self, st, i):
        rb = self.rb
        t = st['t']
        a = t - rb['te']
        alive = np.nonzero((a >= 0) & (a < rb['life']))[0]
        if len(alive) == 0:
            return None
        a = a[alive]
        phi_t = float(st['phi'])
        phi_tail = float(self._tab(self.tphi, t - SHUTTER * 0.8))
        m = rb['m'][alive]
        th = rb['th0'][alive] + m * (phi_t - rb['phi0'][alive])
        tail_phi = np.maximum(phi_tail, rb['phi0'][alive]) if False else phi_tail
        th_t = rb['th0'][alive] + m * (np.where(a > SHUTTER * 0.8, tail_phi, rb['phi0'][alive]) - rb['phi0'][alive])
        R, cen, wd = st['R'], st['cen'], st['width']
        r = R * self.rim_factor(th, t) + rb['dr'][alive] * wd
        r_t = R * self.rim_factor(th_t, t) + rb['dr'][alive] * wd
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
        """Render a list of sprite dicts additively into a full frame (rgb)."""
        p0, p1, rad, col = [], [], [], []
        for gdict in groups:
            if gdict is None or len(gdict['head']) == 0:
                continue
            p0.append(gdict['head'])
            p1.append(gdict['tail'])
            rad.append(gdict['rad'])
            col.append(gdict['col'])
            # bright heads (glints) on hot sparks
            if 'T' in gdict:
                hm = gdict['T'] > 0.55
                if hm.any():
                    p0.append(gdict['head'][hm])
                    p1.append(gdict['head'][hm])
                    rad.append(gdict['rad'][hm] * 1.5)
                    col.append(gdict['col'][hm] * 0.55)
        if not p0:
            return None
        p0 = np.concatenate(p0)
        p1 = np.concatenate(p1)
        rad = np.concatenate(rad)
        col = np.concatenate(col)
        lo = np.minimum(p0, p1).min(0) - 12
        hi = np.maximum(p0, p1).max(0) + 12
        x0, y0 = int(max(0, lo[0])), int(max(0, lo[1]))
        x1, y1 = int(min(W, hi[0])), int(min(H, hi[1]))
        if x1 <= x0 or y1 <= y0:
            return None
        c4 = np.concatenate([col, np.ones((len(col), 1))], 1)
        img = gl.sprites((x1 - x0, y1 - y0), p0, p1, radius=rad, color=c4, offset=(x0, y0), shape=2)
        return img[..., :3], x0, y0

    # ---------------------------------------------------------------- post: sparks + wipe
    def post(self, S, img):
        st = self.state(S)
        t, i = st['t'], S.i
        occ = S.alpha
        # ---- wipe back to the raw footage
        if t >= TW0:
            Rp = st['Rp']
            raw = _inv_tonemap(A.frame(i))
            if self._done(st):
                img = raw
                occ = A.alpha(i)
                m = None
            else:
                m = self.disk_mask(CF, Rp, t, soft=2.0) if Rp > 2 else np.zeros((H, W), np.float32)
                img = img * (1 - m)[..., None] + raw * m[..., None]
                occ = S.alpha * (1 - m) + A.alpha(i) * m
                # gold energy edge washing over his body at the reality front
                if 2 < Rp < 1300:
                    X, Y = _grid_full()
                    rho = np.sqrt((X - CF[0]) ** 2 + (Y - CF[1]) ** 2)
                    th = np.arctan2(Y - CF[1], X - CF[0])
                    rr = Rp * self.rim_factor(th, t)
                    e = np.exp(-((rho - rr) / 16.0) ** 2) * (rho < rr + 50)
                    e = (e * np.maximum(S.alpha, A.alpha(i))).astype(np.float32)
                    fade = 1 - float(_ss(74.0, 74.3, t))
                    img += e[..., None] * np.array([2.6, 1.3, 0.35], np.float32) * fade
            # fade the global post so the very end is the untouched footage
            k = float(_ss(TW0, 74.65, t))
            S.vignette *= 1 - k
            S.grain *= 1 - k
            S.bloom *= 1 - 0.65 * float(_ss(74.4, 74.95, t)) - 0.35 * float(_ss(75.0, 75.5, t))
        # ---- emissive ring + sparks
        b = st['b']
        sparks = self._spark_sprites(st, i)
        behind_groups, front_groups = [], []
        if sparks is not None:
            bm = sparks['behind']
            behind_groups.append({kk: v[bm] for kk, v in sparks.items() if kk != 'behind'})
            front_groups.append({kk: v[~bm] for kk, v in sparks.items() if kk != 'behind'})
        if not self._done(st):
            behind_groups.append(self._rider_sprites(st, i))
        occ_w = None
        ring = None if self._done(st) else self._ring(st)
        for layer, is_behind in ((ring, True), (self._splat(behind_groups), True), (self._splat(front_groups), False)):
            if layer is None:
                continue
            rgb, x0, y0 = layer
            h, w = rgb.shape[:2]
            if is_behind and b > 0.001:
                o = occ[y0:y0 + h, x0:x0 + w]
                rgb = rgb * (1.0 - b * o)[..., None]
            img[y0:y0 + h, x0:x0 + w] += rgb
        return img


# ------------------------------------------------------------------ helpers
def _inv_tonemap(y):
    """Inverse of core._tonemap's shoulder so that the final output equals the raw footage."""
    k = 0.8
    y = np.clip(y, 0, 0.996)
    over = np.maximum(y - k, 0) / (1 - k)
    x = np.where(y > k, k - (1 - k) * np.log(np.maximum(1 - over, 1e-4)), y)
    return x.astype(np.float32)


_GF = None


def _grid_full():
    global _GF
    if _GF is None:
        _GF = core.grid()
    return _GF
