"""ceiling_portal — portal on the ceiling, suction, wormhole tunnel, arrival (43.5 .. 54.6 s).

Timeline (seconds, synced to config.EV and the spoken words):
  44.6-45.9   charge: faint haze swirls on the ceiling, motes drift up, glimmer at the spot
  45.9-47.8   EV ceil_portal  ("portal" 45.96): ignition pinpoint -> shock ring -> vortex opens
              (spring with overshoot, peak ~46.9, settled ~47.5), interior + god rays + dust
  47.8-49.3   sustained vortex, pull grows from "e" (48.4); hair/head start to strain upward
  49.3-50.55  EV suck ("me suga"): anticipation dip -> yank: he is stretched up into the
              mouth (twist, thinning, strands, energy streaks, chromatic smear), gone by 50.5
  50.55-51.1  EV ceil_portal_close: implosion to a point, spark burst, white-violet flash
  51.0-53.55  EV tunnel: full-frame wormhole flight, accelerating, exit light grows, white-out
  53.55-54.6  EV arrive: small portal at the top spits him out (inverse suction), he lands with
              a squash and settles, the portal closes with a pop.

Person handling: the suction/arrival warp is applied in lights(S) - i.e. AFTER every person()
hook (suit) and after person_offset - so the suit/space float are warped together with him.
The person is fully hidden (alpha = 0) from 50.50 until he starts to emerge (> 53.55).
"""
import numpy as np
import cv2

from .. import config as C
from .. import core
from ..core import Layer, Light, smoothstep
from ..glfx import gl
from . import ceiling_portal_glsl as SH

W, H = C.W, C.H
EV = C.EV
DT = 1.0 / C.FPS

T_ON, T_OFF = 43.5, 54.6

# ------------------------------------------------------------------ geometry
# ceiling portal: centre (px, yc), perspective plane with horizon yh and strength k
CEIL = dict(px=520.0, yc=290.0, yh=640.0, k=0.25, rx0=300.0)
# arrival portal (smaller, top of frame)
ARR = dict(px=575.0, yc=150.0, yh=640.0, k=0.12, rx0=190.0)
for _g in (CEIL, ARR):
    _g['A'] = _g['yh'] - _g['yc']
    _top = _g['yh'] - _g['A'] / (1 - _g['k'])
    _bot = _g['yh'] - _g['A'] / (1 + _g['k'])
    _g['ry'] = (_bot - _top) / 2
    _g['ymid'] = (_bot + _top) / 2

# timeline constants
SUCK0, SUCK1 = 49.55, 50.50          # yank (main stretch) window
HIDE0, HIDE1 = 50.50, 53.57          # person invisible
ARR0, ARR1 = 53.58, 54.25            # emergence
COL0, COL1 = EV['ceil_portal_close']  # 50.55, 51.10
TUN0, TUN1 = EV['tunnel']             # 51.00, 53.55
LAM = 380.0                           # stretch fall-off length (px)


def sstep(a, b, x):
    return float(smoothstep(a, b, x))


def lin(a, b, x):
    return float(np.clip((x - a) / (b - a), 0.0, 1.0))


def spring(x, zeta=0.62, wn=4.6):
    """Damped spring step response (0 -> 1 with overshoot), x in seconds."""
    if x <= 0:
        return 0.0
    wd = wn * np.sqrt(1 - zeta * zeta)
    return float(1 - np.exp(-zeta * wn * x) * (np.cos(wd * x) + zeta / np.sqrt(1 - zeta * zeta) * np.sin(wd * x)))


def integrate(f, t0, t, n=1200):
    if t <= t0:
        return 0.0
    ts = np.linspace(t0, t, n)
    v = f(ts)
    return float(np.sum((v[1:] + v[:-1]) * 0.5) * (ts[1] - ts[0]))


def vsm(a, b, x):
    x = np.clip((np.asarray(x, np.float64) - a) / (b - a), 0, 1)
    return x * x * (3 - 2 * x)


def disk_to_screen(g, u, v):
    q = 1 + g['k'] * v
    return g['px'] + g['rx0'] * u / q, g['yh'] - g['A'] / q


def hash01(*xs):
    """Deterministic hash -> [0,1) for numpy arrays (vectorised)."""
    h = np.zeros_like(np.asarray(xs[0], np.float64))
    for k, x in enumerate(xs):
        h = h + np.asarray(x, np.float64) * (12.9898 + 41.233 * k)
    return np.modf(np.abs(np.sin(h) * 43758.5453))[0]


# ------------------------------------------------------------------ ceiling portal timeline
def ceil_omega(t):
    t = np.asarray(t, np.float64)
    return (vsm(45.9, 46.1, t) * (1.3 + 2.6 * (1 - vsm(46.1, 47.3, t)))
            + 1.4 * vsm(48.3, 49.5, t) + 4.0 * vsm(49.3, 50.3, t) + 14.0 * vsm(50.5, 50.85, t))


def ceil_state(t):
    st = {}
    op = spring(t - 45.98, zeta=0.6, wn=4.4)
    op *= 1 + 0.012 * np.sin(TAU * 0.55 * t) * sstep(47.3, 47.8, t)
    op *= 1 + 0.10 * sstep(49.25, 49.75, t)                     # mouth widens for the intake
    op *= (1 - lin(COL0, 50.84, t)) ** 1.6 if t > COL0 else 1.0  # implosion
    st['open'] = max(op, 0.0)
    st['spin'] = integrate(ceil_omega, 45.5, t)
    pw = 0.55 * sstep(45.95, 46.9, t) + 0.15 * sstep(46.9, 47.6, t) + 0.15 * sstep(48.3, 49.3, t)
    pw += 0.45 * sstep(49.3, 49.8, t) + 0.5 * sstep(50.45, 50.84, t)
    pw += 0.25 * np.exp(-((t - 46.02) / 0.12) ** 2)              # ignition flare
    st['pow'] = pw
    st['inner'] = sstep(46.15, 47.2, t)
    # shock rings
    shR, shI = 0.0, 0.0
    if 45.96 < t < 46.8:
        x = lin(45.96, 46.8, t)
        shR, shI = 0.08 + 2.6 * (1 - (1 - x) ** 2.5), 1.6 * (1 - x) ** 1.5
    if 50.83 < t < 51.2:
        x = lin(50.83, 51.2, t)
        shR, shI = 0.05 + 3.6 * (1 - (1 - x) ** 2.2), 3.0 * (1 - x) ** 1.3
    st['shockR'], st['shockI'] = shR, shI
    # pinpoint: glimmer during charge, ignition at 45.96, implosion point at the collapse
    core_ = 0.0
    if 44.9 < t < 46.4:
        flick = 0.6 + 0.4 * np.sin(t * 53.0) * np.sin(t * 31.0)
        core_ += 0.18 * sstep(44.9, 45.8, t) * flick * (1 - sstep(46.0, 46.3, t))
        core_ += 1.5 * np.exp(-((t - 45.96) / 0.07) ** 2) + 0.5 * sstep(45.75, 45.95, t) * (1 - sstep(45.98, 46.35, t))
    if 50.6 < t < 51.05:
        core_ += 2.2 * sstep(50.62, 50.84, t) * (1 - sstep(50.86, 51.0, t))
    st['core'] = core_
    st['haze'] = 0.5 * sstep(44.6, 46.4, t) + 0.6 * sstep(45.95, 47.0, t) + 0.5 * sstep(49.2, 50.0, t)
    st['haze'] *= 1 - sstep(50.6, 50.9, t)
    st['rays'] = (0.22 * sstep(46.3, 47.5, t) + 0.08 * sstep(48.4, 49.4, t) + 0.15 * sstep(49.3, 49.9, t)) * (1 - sstep(50.45, 50.8, t))
    st['light'] = (1.1 * sstep(45.9, 46.9, t) + 0.25 * sstep(48.4, 49.3, t) + 1.4 * sstep(49.3, 49.9, t)) * (1 - sstep(50.5, 50.75, t)) \
        + 1.2 * np.exp(-((t - 46.0) / 0.1) ** 2) + 0.15 * sstep(45.0, 45.9, t) * (1 - sstep(45.9, 46.1, t))
    st['pull'] = 0.25 * sstep(47.2, 48.0, t) + 0.5 * sstep(48.4, 49.3, t) + 1.6 * sstep(49.3, 50.2, t)
    return st


def arr_omega(t):
    t = np.asarray(t, np.float64)
    return 3.2 - 1.6 * vsm(53.6, 54.2, t) + 10.0 * vsm(54.3, 54.55, t)


def arr_state(t):
    st = {}
    op = spring(t - 53.42, zeta=0.55, wn=9.0)
    if t > 54.3:
        op *= (1 - lin(54.3, 54.53, t)) ** 1.7
    st['open'] = max(op, 0.0)
    st['spin'] = integrate(arr_omega, 53.3, t)
    st['pow'] = 1.0 + 0.4 * sstep(53.55, 53.7, t) - 0.5 * sstep(53.9, 54.25, t) + 0.6 * sstep(54.3, 54.52, t)
    st['inner'] = sstep(53.45, 53.7, t)
    shR, shI = 0.0, 0.0
    if 54.5 < t < 54.6 + 0.3:
        x = lin(54.5, 54.85, t)
        shR, shI = 0.05 + 2.4 * (1 - (1 - x) ** 2.0), 2.0 * (1 - x) ** 1.4
    st['shockR'], st['shockI'] = shR, shI
    st['core'] = 1.6 * sstep(54.38, 54.52, t) * (1 - sstep(54.53, 54.6, t))
    st['haze'] = 0.0
    st['rays'] = 0.14 * sstep(53.5, 53.8, t) * (1 - sstep(54.2, 54.5, t))
    st['light'] = (1.3 * sstep(53.45, 53.65, t) * (1 - 0.55 * sstep(54.0, 54.4, t))) * (1 - sstep(54.42, 54.56, t)) \
        + 1.0 * np.exp(-((t - 54.52) / 0.05) ** 2)
    return st


TAU = 2 * np.pi


# ------------------------------------------------------------------ suction parameters
def suck_params(t):
    """Return (mode, s, ant, prepull) for the person warp, or None when untouched.
    s: 0 = intact, 1 = completely inside the portal.  ant: >0 squash (anticipation / landing),
    <0 stretch.  prepull: small absorption depth (px) - hair straining toward the portal."""
    if 48.3 <= t < HIDE0:
        pre = 55.0 * sstep(48.4, 49.25, t) * (1 - sstep(49.3, 49.6, t))
        ant = np.sin(np.pi * lin(49.27, 49.72, t)) * (1 if t < 49.72 else 0)
        s = lin(SUCK0, SUCK1, t)
        return 'ceil', s, float(ant), pre
    if ARR0 - 0.02 <= t < 54.6:
        tau = lin(ARR0, ARR1, t)
        s = (1 - tau)
        ant = 0.0
        if t > ARR1 - 0.04:
            x = t - (ARR1 - 0.04)
            ant = 1.25 * np.sin(x / 0.30 * TAU * 0.75) * np.exp(-x / 0.13)
        return 'arr', s, float(ant), 0.0
    return None


def absorb_depth(s, mode):
    if mode == 'ceil':
        return 1380.0 * s ** 1.7
    return 1380.0 * s ** 1.25


class FX(core.Effect):
    name = 'ceiling_portal'

    def __init__(self):
        rng = np.random.default_rng(90210)
        self._progs = {}
        # tileable fbm noise texture (for strands) ---------------------------------
        n = np.zeros((256, 256), np.float32)
        amp = 1.0
        for oc in (8, 16, 32, 64):
            r = rng.random((oc, oc)).astype(np.float32)
            r = np.tile(r, (3, 3))
            big = cv2.resize(r, (256 * 3, 256 * 3), interpolation=cv2.INTER_CUBIC)[256:512, 256:512]
            n += big * amp
            amp *= 0.5
        n = (n - n.min()) / (n.max() - n.min())
        self.noise = n.astype(np.float32)
        # dust motes ---------------------------------------------------------------
        M = 560
        self.dN = M
        self.d_ph = rng.random(M)
        self.d_L = rng.uniform(1.8, 3.4, M)
        self.d_size = rng.uniform(0.9, 2.4, M) * np.where(rng.random(M) < 0.12, 2.2, 1.0)
        self.d_b = rng.uniform(0.25, 1.0, M) ** 1.5
        self.d_hue = rng.random(M)
        self.d_order = rng.permutation(M) / M
        # suction energy fragments ------------------------------------------------
        K = 420
        self.fK = K
        self.f_tb = np.sort(rng.uniform(SUCK0 - 0.05, SUCK1 - 0.05, K))
        self.f_L = rng.uniform(0.22, 0.5, K)
        self.f_u = rng.random(K)          # where on the body
        self.f_v = rng.random(K)
        self.f_ang = rng.uniform(0, TAU, K)
        self.f_b = rng.uniform(0.4, 1.0, K)
        self.f_hue = rng.random(K)
        # arrival fragments (time-reversed suction + burst)
        self.a_tb = np.sort(rng.uniform(ARR0 - 0.02, ARR1 - 0.15, K))
        self.a_L = rng.uniform(0.2, 0.45, K)
        self.a_u = rng.random(K)
        self.a_v = rng.random(K)
        self.a_ang = rng.uniform(0, TAU, K)
        self.a_b = rng.uniform(0.4, 1.0, K)
        # collapse / close spark bursts (disk-plane radial)
        B = 320
        self.bN = B
        self.b_ang = rng.uniform(0, TAU, B)
        self.b_v = rng.uniform(1.0, 4.2, B) ** 1.2
        self.b_drag = rng.uniform(2.5, 6.0, B)
        self.b_b = rng.uniform(0.4, 1.0, B)
        self.b_lift = rng.normal(0, 0.12, B)
        # tunnel stars ---------------------------------------------------------------
        S = 1500
        self.sN = S
        self.s_a = rng.uniform(0, TAU, S)
        self.s_r = np.sqrt(rng.uniform(0.02, 0.85, S))
        self.s_z = rng.uniform(0, 40.0, S)
        self.s_b = rng.uniform(0.3, 1.0, S) ** 2
        self.s_hue = rng.random(S)

    # ------------------------------------------------------------------ hooks
    def active(self, t):
        return T_ON <= t <= T_OFF

    def _prog(self, key):
        if key not in self._progs:
            self._progs[key] = gl.program(getattr(SH, key))
        return self._progs[key]

    def background(self, S):
        t = S.t
        # camera shake: suction rumble, collapse kick, tunnel turbulence, landing bump
        amp = 0.0
        amp += 3.5 * sstep(49.35, 50.3, t) * (1 - sstep(50.45, 50.6, t))
        amp += 9.0 * np.exp(-max(t - 50.84, 0) / 0.18) * (t > 50.84) * (t < 51.4)
        if TUN0 <= t < TUN1:
            amp += 2.0 + 4.5 * lin(TUN0, TUN1, t) ** 2 + 6.0 * np.exp(-((t - 52.3) / 0.12) ** 2)
        amp += 3.0 * np.exp(-max(t - 54.18, 0) / 0.15) * (t > 54.18)
        if amp > 0.05:
            ph = t * 37.0
            dx = amp * (np.sin(ph * 1.7) * 0.6 + np.sin(ph * 3.1 + 1.3) * 0.4)
            dy = amp * (np.sin(ph * 2.3 + 0.7) * 0.6 + np.sin(ph * 4.1 + 2.1) * 0.4)
            S.shake.append((float(dx), float(dy), float(amp * 0.05 * np.sin(ph * 1.3))))
        if 45.8 < t < 51.3 or TUN0 <= t:
            S.bloom = max(S.bloom, 1.25)
        S.meta['portal_center'] = (CEIL['px'], CEIL['yc']) if t < 52 else (ARR['px'], ARR['yc'])

    def person(self, S):
        if HIDE0 <= S.t < HIDE1:
            S.alpha[:] = 0.0
            S.meta['person_hidden'] = True

    def lights(self, S):
        t = S.t
        self._warp_person(S)
        out = []
        if t < 51.0:
            st = ceil_state(t)
            I = st['light']
            if I > 0.01:
                P = (CEIL['px'], CEIL['yc'] + 40)
                out.append(Light(P, (0.42, 0.50, 1.0), intensity=0.6 * I, radius=650, rim=1.6, diffuse=0.30, bg=0.035))
                out.append(Light((CEIL['px'] + 220, CEIL['yc'] + 20), (0.15, 0.85, 0.9), intensity=0.22 * I,
                                 radius=420, rim=1.0, diffuse=0.10, bg=0.01))
                out.append(Light((CEIL['px'] - 230, CEIL['yc'] + 20), (0.6, 0.2, 1.0), intensity=0.22 * I,
                                 radius=420, rim=1.0, diffuse=0.10, bg=0.01))
            # collapse flash
            fI = 4.0 * sstep(50.7, 50.84, t) * (1 - sstep(50.86, 51.0, t))
            if fI > 0.01:
                out.append(Light((CEIL['px'], CEIL['yc']), (0.7, 0.75, 1.0), intensity=fI, radius=800, rim=1.0, diffuse=0.4, bg=0.08))
        if t >= 53.4:
            st = arr_state(t)
            I = st['light']
            if I > 0.01:
                P = (ARR['px'], ARR['yc'] + 30)
                out.append(Light(P, (0.45, 0.5, 1.0), intensity=0.7 * I, radius=600, rim=1.8, diffuse=0.30, bg=0.02))
                out.append(Light((ARR['px'] + 160, ARR['yc']), (0.15, 0.85, 0.9), intensity=0.25 * I, radius=380, rim=1.0, diffuse=0.10))
        return out

    # ------------------------------------------------------------------ person warp
    def _axis(self, alpha):
        q = 4
        a = cv2.resize(alpha, (W // q, H // q), interpolation=cv2.INTER_AREA)
        xs = np.arange(W // q, dtype=np.float32) * q + q / 2
        ys = np.arange(H // q, dtype=np.float32) * q + q / 2
        rs = a.sum(1)
        cx = (a * xs[None, :]).sum(1) / np.maximum(rs, 1e-6)
        ok = rs > 4
        if ok.sum() < 4:
            return ys, np.full_like(ys, W / 2)
        cxv = np.interp(ys, ys[ok], cx[ok]).astype(np.float32)
        cxv = cv2.GaussianBlur(cxv[:, None], (1, 0), 7).ravel()
        return ys, cxv

    def _maps(self, t, mode, s, ant, pre, ax):
        """Inverse warp maps at half resolution: returns map_x, map_y, mask, dp (half res)."""
        g = CEIL if mode == 'ceil' else ARR
        dirn = 1.0 if mode == 'ceil' else -1.0
        Px = g['px']
        Py = g['ymid'] + (8.0 if mode == 'ceil' else 4.0)
        rx = g['rx0'] * 0.95
        ry = g['ry'] * 1.1
        q = 2
        h, w = H // q, W // q
        X = np.broadcast_to((np.arange(w, dtype=np.float32) * q + 0.5)[None, :], (h, w))
        Y = np.broadcast_to((np.arange(h, dtype=np.float32) * q + 0.5)[:, None], (h, w))
        # swirl around the mouth
        phim = 1.5 * sstep(0.12, 0.85, s) * dirn
        if abs(phim) > 1e-4:
            xn = (X - Px) / rx
            yn = (Y - Py) / ry
            rr = np.sqrt(xn * xn + yn * yn)
            phi = phim * np.exp(-np.maximum(rr - 0.6, 0.0) ** 2 / 1.4)
            c, sn = np.cos(phi), np.sin(phi)
            X1 = Px + (xn * c + yn * sn) * rx
            Y1 = Py + (-xn * sn + yn * c) * ry
        else:
            X1, Y1 = X, Y
        dp = Y1 - Py
        # vertical absorption / stretch
        D = absorb_depth(s, mode) + pre
        S2 = D * 0.35 * sstep(0.4, 1.0, s)
        S1 = max(D - S2, 0.0) * np.exp(D / LAM)
        dg = np.linspace(-800.0, 2400.0, 4000)
        Fg = dg - S1 * np.exp(-np.maximum(dg, 0) / LAM) - S2
        d = np.interp(dp.ravel(), Fg, dg).reshape(h, w).astype(np.float32)
        ysrc = Py + d
        N = sstep(0.02, 0.5, s)
        wf = 0.045 + 0.955 * smoothstep(0.0, 720.0, dp) ** 0.7
        wv = (1.0 - N * (1.0 - wf)) * (1.0 - 0.25 * sstep(0.6, 1.0, s))
        ays, axx = ax
        cxs = np.interp(ysrc, ays, axx).astype(np.float32)
        conv = smoothstep(0.0, 650.0, dp)
        Atw = 42.0 * sstep(0.08, 0.5, s) * (1 - 0.5 * sstep(0.85, 1.0, s))
        twist = Atw * np.sin(dp / 72.0 - 13.0 * t * dirn) * smoothstep(0.0, 120.0, dp) * (1 - smoothstep(380.0, 950.0, dp))
        cxo = cxs + N * (Px - cxs) * (1 - conv) + twist * N
        xsrc = cxs + (X1 - cxo) / np.maximum(wv, 1e-3)
        if abs(ant) > 1e-4:
            k = 0.022 * ant
            ysrc = H - (H - ysrc) / (1 - k)
            xsrc = cxs + (xsrc - cxs) / (1 + 0.55 * k)
        m = smoothstep(-8.0, 75.0, dp)
        if S2 > 0:
            m = m * smoothstep(H + 4.0, H - 320.0 * min(1.0, S2 / 220.0), ysrc)
        thr = N * 0.95 * (1 - smoothstep(25.0, 560.0, dp))
        if N > 0.01:
            nu = np.mod(xsrc / 9.0, 256.0).astype(np.float32)
            nv = np.mod(dp / 120.0 - t * 2.5 * dirn + 50.0, 256.0).astype(np.float32)
            nz = cv2.remap(self.noise, nu, nv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
            m = m * smoothstep(thr - 0.13, thr + 0.13, nz)
        return xsrc.astype(np.float32), ysrc.astype(np.float32), m.astype(np.float32), dp.astype(np.float32)

    def _warp_person(self, S):
        t = S.t
        if HIDE0 <= t < HIDE1:
            S.alpha = np.zeros_like(S.alpha)
            S.fg = np.zeros_like(S.fg)
            S.meta['person_hidden'] = True
            return
        sp = suck_params(t)
        if sp is None:
            return
        mode, s, ant, pre = sp
        if s <= 0 and abs(ant) < 1e-4 and pre < 0.5:
            return
        if s >= 0.999:
            S.alpha = np.zeros_like(S.alpha)
            S.fg = np.zeros_like(S.fg)
            S.meta['person_hidden'] = True
            return
        ax = self._axis(S.alpha)
        src = np.dstack([S.fg * S.alpha[..., None], S.alpha]).astype(np.float32)
        # temporal super-sampling of the warp (180 deg shutter) = true motion blur
        moving = 0.0 < s < 1.0
        offs = (-0.21, -0.07, 0.07, 0.21) if moving else (0.0,)
        acc = None
        dp_c = None
        m_c = None
        for o in offs:
            tt = t + o * DT
            sp2 = suck_params(tt)
            if sp2 is None:
                sp2 = sp
            _, s2, ant2, pre2 = sp2
            mx, my, mm, dp = self._maps(tt, mode, s2, ant2, pre2, ax)
            mx = cv2.resize(mx, (W, H), interpolation=cv2.INTER_LINEAR)
            my = cv2.resize(my, (W, H), interpolation=cv2.INTER_LINEAR)
            mm = cv2.resize(mm, (W, H), interpolation=cv2.INTER_LINEAR)
            wv = cv2.remap(src, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            wv *= mm[..., None]
            acc = wv if acc is None else acc + wv
            if dp_c is None or abs(o) < 0.1:
                dp_c, m_c = dp, mm
        acc /= len(offs)
        a = np.clip(acc[..., 3], 0, 1)
        S.alpha = a
        S.fg = acc[..., :3] / np.maximum(acc[..., 3:4], 1e-4)
        S.meta['_cp_warp'] = dict(mode=mode, s=s, dp=dp_c)

    # ------------------------------------------------------------------ portal rendering
    def _portal_layer(self, g, st, t, y0, y1, seed, dirn, scale=1.0):
        prog = self._prog('PORTAL')
        w, h = int(W * scale), int((y1 - y0) * scale)
        img = gl.render(prog, (w, h), offset=(0, y0), time=t, uniforms=dict(
            uP=(g['px'], g['yc']), uRx=g['rx0'], uA=g['A'], uYh=g['yh'], uK=g['k'],
            uOpen=float(st['open']), uSpin=float(st['spin']), uPow=float(st['pow']), uInner=float(st['inner']),
            uShockR=float(st['shockR']), uShockI=float(st['shockI']), uCore=float(st['core']), uHaze=float(st['haze']),
            uSeed=float(seed), uScale=1.0 / scale, uDir=float(dirn)))
        if scale != 1.0:
            img = cv2.resize(img, (W, y1 - y0), interpolation=cv2.INTER_LINEAR)
        rgb = np.ascontiguousarray(img[..., :3])
        a = np.ascontiguousarray(np.clip(img[..., 3], 0, 1))
        return Layer('over', rgb, a, box=(0, y0))

    def _rays_img(self, g, I, t, y0, seed):
        prog = self._prog('RAYS')
        q = 3
        h = (H - y0) // q
        img = gl.render(prog, (W // q, h), offset=(0, y0), time=t, uniforms=dict(
            uP=(g['px'], g['ymid']), uRx=g['rx0'] * 1.0, uRy=g['ry'], uI=float(I), uScale=float(q), uSeed=float(seed)))
        return cv2.resize(img[..., :3], (W, h * q), interpolation=cv2.INTER_LINEAR)

    # ------------------------------------------------------------------ particles
    def _dust(self, t, st):
        """Studio dust pulled up into the ceiling portal.  Returns (behind_img, front_img) or None."""
        g = CEIL
        dens = 0.18 * sstep(44.6, 45.9, t) + 0.55 * sstep(46.2, 47.6, t) + 0.27 * sstep(48.4, 49.5, t)
        dens *= 1 - sstep(50.55, 50.8, t)
        if dens <= 0.002:
            return None
        clock = lambda tt: integrate(lambda x: 1.0 + 0.35 * vsm(47.0, 48.0, x) + 0.6 * vsm(48.4, 49.4, x) + 2.2 * vsm(49.3, 50.2, x), 43.0, tt)
        k = np.arange(self.dN)
        Px, Py = g['px'], g['yc'] + 10

        def pos(T):
            tt = T / self.d_L + self.d_ph
            cyc = np.floor(tt)
            tau = tt - cyc
            r0 = 120 + 560 * hash01(k, cyc, 1.0) ** 0.8
            phi0 = TAU * hash01(k, cyc, 2.0)
            y0 = 520 + 980 * hash01(k, cyc, 3.0)
            e = tau ** 2.3
            r = r0 * (1 - e) ** 1.15 + 6
            phi = phi0 + 1.3 * tau + 5.5 * e
            y = y0 + (Py - y0) * (0.3 * tau + 0.7 * e)
            flat = 0.06 + 0.26 * np.clip((900 - y) / 620, 0, 1)
            x = Px + r * np.cos(phi)
            yy = y + r * np.sin(phi) * flat
            return x, yy, tau, np.sin(phi)

        Tn = clock(t)
        x, y, tau, depth = pos(Tn)
        xp, yp, taup, _ = pos(Tn - 0.6 * DT * (1.0 + 0.35 + 0.6 + 2.2 * sstep(49.3, 50.2, t)))
        wrap = taup > tau
        xp = np.where(wrap, x, xp)
        yp = np.where(wrap, y, yp)
        vis = np.clip((dens - self.d_order) * self.dN / 40.0, 0, 1)
        fade = vsm(0.0, 0.12, tau) * (1 - vsm(0.9, 1.0, tau))
        dist = np.hypot(x - Px, (y - Py) * 1.6)
        lit = 0.18 + 2.4 * np.exp(-dist / 230.0) + 0.5 * np.exp(-dist / 600.0)
        b = self.d_b * vis * fade * lit * (0.5 + 0.6 * st['pow'])
        hue = self.d_hue
        col = np.stack([0.55 + 0.4 * hue, 0.65 + 0.2 * (1 - hue), np.full_like(hue, 1.15)], 1)
        size = self.d_size * (1 + 0.9 * np.clip(-depth, 0, 1) * (y > 800))
        rgba = np.concatenate([col * b[:, None] * 0.9, b[:, None]], 1).astype(np.float32)
        p0 = np.stack([x, y], 1)
        p1 = np.stack([xp, yp], 1)
        front = depth < 0
        res = []
        for msk in (~front, front):
            sel = msk & (b > 0.004)
            res.append(gl.sprites((W, H), p0[sel], p1[sel], size[sel], rgba[sel], shape=2)[..., :3])
        return res

    def _body_point(self, i, u, v):
        """Point on the body (frame px) from pose at frame i: u horizontal 0..1, v 0 = head .. 1 = waist."""
        p = core_pose(i)
        sl, sr = p[11, :2], p[12, :2]
        nose = p[0, :2]
        sw = np.linalg.norm(sl - sr) + 1e-3
        top = nose + np.array([0, -0.55 * sw])
        midsh = (sl + sr) / 2
        yb = np.minimum(midsh[1] + 0.9 * sw, H + 40)
        ishead = v < 0.3
        vv = np.where(ishead, v / 0.3, (v - 0.3) / 0.7)
        hx = nose[0] + (u - 0.5) * 0.55 * sw
        hy = top[1] + vv * (midsh[1] - top[1]) * 0.9
        bx = midsh[0] + (u - 0.5) * sw * (1.0 + 0.15 * vv)
        by = midsh[1] - 0.05 * sw + vv * (yb - midsh[1])
        return np.where(ishead, hx, bx), np.where(ishead, hy, by)

    def _fwd(self, mode, s, t, x, y, axx):
        """Forward warp of body points (approx., matches _maps) -> output positions + dp."""
        g = CEIL if mode == 'ceil' else ARR
        Px = g['px']
        Py = g['ymid'] + (8.0 if mode == 'ceil' else 4.0)
        D = absorb_depth(s, mode)
        S2 = D * 0.35 * sstep(0.4, 1.0, s)
        S1 = max(D - S2, 0.0) * np.exp(D / LAM)
        d = y - Py
        dp = d - S1 * np.exp(-np.maximum(d, 0) / LAM) - S2
        N = sstep(0.02, 0.5, s)
        wf = 0.045 + 0.955 * smoothstep(0.0, 720.0, dp) ** 0.7
        wv = (1.0 - N * (1.0 - wf)) * (1.0 - 0.25 * sstep(0.6, 1.0, s))
        conv = smoothstep(0.0, 650.0, dp)
        cxo = axx + N * (Px - axx) * (1 - conv)
        return cxo + (x - axx) * wv, Py + dp, dp

    def _fragments(self, t, mode):
        """Energy fragments peeling off the body and spiralling into the mouth (or out of it)."""
        if mode == 'ceil':
            g, tb, L, u, v, ang, bb = CEIL, self.f_tb, self.f_L, self.f_u, self.f_v, self.f_ang, self.f_b
        else:
            g, tb, L, u, v, ang, bb = ARR, self.a_tb, self.a_L, self.a_u, self.a_v, self.a_ang, self.a_b
        Px = g['px']
        Py = g['ymid'] + 8
        res_p0, res_p1, res_b = [], [], []
        for dt_s in (0.0, -0.5 * DT):
            tt = t + dt_s
            if mode == 'ceil':
                age = tt - tb
                tau = age / L
            else:
                # reversed: fragments travel from the mouth to the body, arriving at tb + L
                age = tt - tb
                tau = 1 - age / L
            alive = (age >= 0) & (age <= L)
            res_p0.append((tau, alive))
        tau, alive = res_p0[0]
        taup, alivep = res_p0[1]
        sel = alive & alivep
        if not sel.any():
            return None
        idx = np.where(sel)[0]
        # origin on the (warped) body at the emission time
        te = tb[idx] if mode == 'ceil' else (tb[idx] + L[idx])
        i_e = np.clip(np.round(te * C.FPS).astype(int), 0, C.N_FRAMES - 1)
        ox = np.zeros(len(idx))
        oy = np.zeros(len(idx))
        for j, (ii, uu, vv, tte) in enumerate(zip(i_e, u[idx], v[idx], te)):
            bx, by = self._body_point(ii, uu, vv)
            sp = suck_params(tte)
            s_e = sp[1] if sp is not None else (1.0 if mode == 'ceil' else 0.0)
            p = core_pose(ii)
            axx = 0.5 * (p[11, 0] + p[12, 0]) if vv >= 0.3 else p[0, 0]
            fx, fy, dp = self._fwd(mode, s_e, tte, bx, by, axx)
            ox[j], oy[j] = fx, fy

        def path(tau_):
            e = np.clip(tau_, 0, 1) ** 1.8
            r = 1 - e
            spin = ang[idx] + (1 if mode == 'ceil' else -1) * (2.5 * e + 1.0 * tau_)
            cx = Px + (ox - Px) * r + np.cos(spin) * 60 * r * e * 2
            cy = Py + (oy - Py) * r + np.sin(spin) * 20 * r * e * 2
            return cx, cy

        x0, y0 = path(tau[idx])
        x1, y1 = path(taup[idx])
        fade = np.sin(np.pi * np.clip(tau[idx], 0, 1)) ** 0.6
        b = bb[idx] * fade
        hue = self.f_hue[idx]
        col = np.stack([0.5 + 0.6 * hue, 0.75 + 0.3 * (1 - hue), np.full_like(hue, 1.4)], 1) * 2.2
        rgba = np.concatenate([col * b[:, None], b[:, None]], 1).astype(np.float32)
        size = 1.2 + 1.6 * bb[idx]
        img = gl.sprites((W, H), np.stack([x0, y0], 1), np.stack([x1, y1], 1), size, rgba, shape=2)
        return img[..., :3]

    def _burst(self, t, g, t0, dur, scale=1.0, inward=False):
        """Sparks bursting radially along the portal plane at t0 (collapse / close)."""
        if not (t0 <= t <= t0 + dur):
            return None
        out = []
        for dts in (0.0, -0.6 * DT):
            tau = max(t + dts - t0, 0.0)
            R = self.b_v / self.b_drag * (1 - np.exp(-self.b_drag * tau)) * scale + 0.02
            u = R * np.cos(self.b_ang)
            v = R * np.sin(self.b_ang)
            x, y = disk_to_screen(g, u, v)
            y = y + self.b_lift * 400 * tau ** 1.5 * scale + 120 * tau ** 2 * scale
            out.append((x, y))
        (x0, y0), (x1, y1) = out
        tau = t - t0
        fade = np.exp(-tau * (1.5 + 2.5 * (1 - self.b_b))) * (1 - lin(t0 + dur * 0.7, t0 + dur, t))
        b = self.b_b * fade
        col = np.stack([0.7 + 0.3 * self.b_b, 0.75 + 0.2 * self.b_b, np.full(self.bN, 1.4)], 1) * 3.0
        rgba = np.concatenate([col * b[:, None], b[:, None]], 1).astype(np.float32)
        img = gl.sprites((W, H), np.stack([x0, y0], 1), np.stack([x1, y1], 1), 1.0 + 1.2 * self.b_b, rgba, shape=2)
        return img[..., :3]

    # ------------------------------------------------------------------ tunnel
    def _tunnel_speed(self, t):
        t = np.asarray(t, np.float64)
        tau = np.clip((t - TUN0) / (TUN1 - TUN0), 0, 1)
        return 1.6 + 13.0 * tau ** 2.3 + 3.0 * np.exp(-((t - 52.3) / 0.15) ** 2)

    def _tunnel_cam(self, t):
        tau = lin(TUN0, TUN1, t)
        cx = W / 2 + 55 * np.sin(0.9 * t + 1.0) + 22 * np.sin(2.3 * t) * (0.5 + tau)
        cy = 690 + 45 * np.sin(1.1 * t + 0.3) + 18 * np.sin(2.9 * t + 1.0) * (0.5 + tau)
        roll = 0.35 * np.sin(0.8 * t) + 1.3 * tau ** 2.0
        return np.array([cx, cy]), roll

    def _tunnel(self, t):
        prog = self._prog('TUNNEL')
        dist = integrate(self._tunnel_speed, TUN0 - 0.2, t) + 7.0
        v = float(self._tunnel_speed(t))
        c, roll = self._tunnel_cam(t)
        foc = 540.0
        bend = 0.32
        if t > 52.95:
            ex = 0.012 * np.exp((t - 52.95) / 0.115)
        else:
            ex = 0.0
        q = 2
        img = gl.render(prog, (W // q, H // q), time=t, uniforms=dict(
            uC=(float(c[0]), float(c[1])), uRoll=float(roll), uDist=float(dist), uSpeed=v, uExit=float(min(ex, 4.0)),
            uScale=float(q), uFoc=foc, uBend=bend, uBright=1.0, uSeed=3.0))
        img = cv2.resize(img[..., :3], (W, H), interpolation=cv2.INTER_LINEAR)
        # star streaks (full res)
        zc = np.mod(self.s_z - dist, 40.0) + 0.18
        zp = zc + v * DT * 0.6 * 1.6
        b0 = np.array([np.sin(dist * 0.21) + 0.5 * np.sin(dist * 0.47 + 1.3),
                       np.cos(dist * 0.17 + 0.4) + 0.4 * np.sin(dist * 0.39)])

        def proj(z):
            Z = dist + z
            bx = (np.sin(Z * 0.21) + 0.5 * np.sin(Z * 0.47 + 1.3) - b0[0]) * bend
            by = (np.cos(Z * 0.17 + 0.4) + 0.4 * np.sin(Z * 0.39) - b0[1]) * bend
            lx = (self.s_r * np.cos(self.s_a) + bx) / z
            ly = (self.s_r * np.sin(self.s_a) + by) / z
            cr, sr = np.cos(roll), np.sin(roll)
            # inverse of GLSL rot2(roll): d = R s  ->  s = R^T d
            sx = cr * lx - sr * ly
            sy = sr * lx + cr * ly
            return c[0] + sx * foc, c[1] + sy * foc

        x0, y0 = proj(zc)
        x1, y1 = proj(zp)
        fresh = zc < 39.5
        bright = self.s_b * np.clip(2.2 / zc, 0, 3.0) * vsm(40.0, 28.0, zc) * fresh
        hue = self.s_hue
        col = np.stack([0.65 + 0.45 * hue, 0.75 + 0.15 * np.sin(hue * 9), np.full_like(hue, 1.3)], 1)
        rgba = np.concatenate([col * bright[:, None] * 1.4, bright[:, None]], 1).astype(np.float32)
        onscr = (x0 > -50) & (x0 < W + 50) & (y0 > -50) & (y0 < H + 50) & (bright > 0.01)
        st = gl.sprites((W, H), np.stack([x0, y0], 1)[onscr], np.stack([x1, y1], 1)[onscr],
                        (0.8 + 1.4 * np.clip(1.0 / zc, 0, 2))[onscr], rgba[onscr], shape=2)
        img += st[..., :3]
        # radial chromatic aberration + zoom blur around the vanishing point
        ca = 0.004 + 0.010 * lin(TUN0, TUN1, t) ** 1.5
        X, Y = core.grid()
        dx, dy = X - c[0], Y - c[1]
        out = img.copy()
        for ch, sc in ((0, 1 + ca), (2, 1 - ca)):
            out[..., ch] = cv2.remap(np.ascontiguousarray(img[..., ch]), (c[0] + dx / sc).astype(np.float32),
                                     (c[1] + dy / sc).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        zb = 0.012 + 0.03 * lin(TUN0, TUN1, t) ** 2
        acc = out.copy()
        for k in (1, 2):
            sc = 1 + zb * k
            M = np.float32([[1 / sc, 0, c[0] * (1 - 1 / sc)], [0, 1 / sc, c[1] * (1 - 1 / sc)]])
            acc += cv2.warpAffine(out, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        out = acc / 3.0
        # entry: the tunnel resolves out of the collapse flash
        return out

    # ------------------------------------------------------------------ layer hooks
    def behind(self, S):
        t = S.t
        L = []
        if t < 51.25:
            st = ceil_state(t)
            if st['open'] > 0.001 or st['core'] > 0.001 or st['haze'] > 0.001 or st['shockI'] > 0.001:
                L.append(self._portal_layer(CEIL, st, t, 0, 600, 1.0, 1.0))
            if st['rays'] > 0.01:
                y0 = 240
                r = self._rays_img(CEIL, st['rays'], t, y0, 2.0)
                L.append(Layer('add', r, box=(0, y0)))
                S.meta['_cp_rays'] = (r, y0)
            d = self._dust(t, st)
            if d is not None:
                L.append(Layer('add', d[0]))
                S.meta['_cp_dust_front'] = d[1]
        if TUN0 <= t < 53.62:
            img = self._tunnel(t)
            fin = sstep(TUN0, TUN0 + 0.08, t)
            L.append(Layer('over', img * fin, np.full((H, W), fin, np.float32)))
        if t >= 53.4:
            st = arr_state(t)
            if st['open'] > 0.001 or st['core'] > 0.001 or st['shockI'] > 0.001:
                L.append(self._portal_layer(ARR, st, t, 0, 420, 7.0, -1.0))
            if st['rays'] > 0.01:
                y0 = 120
                r = self._rays_img(ARR, st['rays'], t, y0, 5.0)
                L.append(Layer('add', r, box=(0, y0)))
        return L

    def front(self, S):
        t = S.t
        L = []
        d = S.meta.pop('_cp_dust_front', None)
        if d is not None:
            L.append(Layer('add', d))
        r = S.meta.pop('_cp_rays', None)
        if r is not None:
            L.append(Layer('add', r[0] * 0.22, box=(0, r[1])))
        wp = S.meta.get('_cp_warp')
        if wp is not None and 0 < wp['s'] < 1:
            L += self._stream_glow(S, wp)
        if SUCK0 - 0.05 <= t <= SUCK1 + 0.05:
            f = self._fragments(t, 'ceil')
            if f is not None:
                L.append(Layer('add', f))
        if ARR0 - 0.05 <= t <= ARR1 + 0.1:
            f = self._fragments(t, 'arr')
            if f is not None:
                L.append(Layer('add', f))
        b = self._burst(t, CEIL, 50.84, 0.55, scale=1.0)
        if b is not None:
            L.append(Layer('add', b))
        b = self._burst(t, ARR, 54.52, 0.5, scale=0.7)
        if b is not None:
            L.append(Layer('add', b))
        return L

    def _stream_glow(self, S, wp):
        """Energy streaks on the stretched body: bright near the mouth, smeared along the flow."""
        s = wp['s']
        N = sstep(0.02, 0.5, s)
        if N < 0.01:
            return []
        dp = cv2.resize(wp['dp'], (W, H), interpolation=cv2.INTER_LINEAR)
        a = S.alpha
        e = a * (1 - smoothstep(0.0, 620.0, dp)) * N
        if e.max() < 1e-3:
            return []
        ys, xs = np.where(e > 0.01)
        x0, x1 = max(0, xs.min() - 40), min(W, xs.max() + 40)
        y0, y1 = max(0, ys.min() - 120), min(H, ys.max() + 60)
        ec = e[y0:y1, x0:x1]
        dpc = dp[y0:y1, x0:x1]
        k = np.zeros((61, 1), np.float32)
        k[:, 0] = np.exp(-np.abs(np.arange(61) - 30) / 9.0)
        k[:30] *= 1.6                                      # trail extends below (behind the motion)
        k /= k.sum()
        sm = cv2.filter2D(ec, -1, k, borderType=cv2.BORDER_CONSTANT)
        g2 = core.glow(ec, 6.0)
        hot = np.clip(1 - dpc / 260.0, 0, 1)
        col_a = np.array([0.25, 0.75, 1.2], np.float32)     # teal
        col_b = np.array([0.75, 0.35, 1.3], np.float32)     # violet
        mixc = col_a[None, None] * (1 - hot[..., None]) + col_b[None, None] * hot[..., None]
        img = mixc * (sm[..., None] * 1.6 + g2[..., None] * 0.6) + np.array([1.0, 1.0, 1.2], np.float32) * (ec * hot ** 2)[..., None] * 1.2
        # hot spot where he disappears into the mouth
        g = CEIL if wp['mode'] == 'ceil' else ARR
        flare = np.zeros((H, W, 3), np.float32)
        return [Layer('add', img.astype(np.float32), box=(x0, y0))]

    def post(self, S, img):
        t = S.t
        # chromatic smear on the stretched body
        wp = S.meta.get('_cp_warp')
        if wp is not None and 0.05 < wp['s'] < 1:
            N = sstep(0.05, 0.5, wp['s'])
            m = core.glow(S.alpha, 5.0) * N
            ys, xs = np.where(m > 0.02)
            if len(ys):
                x0, x1 = max(0, xs.min() - 20), min(W, xs.max() + 20)
                y0, y1 = max(0, ys.min() - 30), min(H, ys.max() + 30)
                mc = m[y0:y1, x0:x1]
                X, Y = core.grid(y1 - y0, x1 - x0)
                crop = img[y0:y1, x0:x1].copy()
                sh = 9.0 * mc
                img[y0:y1, x0:x1, 0] = cv2.remap(np.ascontiguousarray(crop[..., 0]), X, (Y + sh).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                img[y0:y1, x0:x1, 2] = cv2.remap(np.ascontiguousarray(crop[..., 2]), X, (Y - sh).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        # flashes: collapse (bridges into the tunnel) and tunnel exit (into space)
        f = 0.0
        f += sstep(50.76, 50.87, t) * (1 - sstep(51.02, 51.28, t)) * 1.0
        f += sstep(53.33, 53.55, t) ** 2 * (1 - sstep(53.555, 53.76, t)) ** 1.5
        if f > 0.002:
            tint = np.array([0.85, 0.85, 1.0], np.float32) if t < 52 else np.array([0.95, 0.97, 1.0], np.float32)
            img = img * (1 - 0.75 * f) + tint * (3.2 * f)
        return img


def core_pose(i):
    from .. import assets as A
    return A.pose(int(np.clip(i, 0, C.N_FRAMES - 1)))
