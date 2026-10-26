"""ceiling_portal — portal on the ceiling, suction, wormhole tunnel, arrival (43.5 .. 54.6 s).

Timeline (seconds, synced to config.EV and the spoken words):
  44.6-45.9   charge: faint haze swirls on the ceiling, motes drift up, glimmer at the spot
  45.9-47.8   EV ceil_portal  ("portal" 45.96): ignition pinpoint -> vortex opens (spring with
              overshoot, settled ~47.4); a thin heat-ripple shock ring launches once the rim has
              passed half size; interior throat + god rays + dust catching the shafts
  47.8-49.2   sustained vortex; from "e" (48.4) the pull grows: energy wisps peel off his hair and
              shoulders and his upper body strains toward the mouth
  49.2-49.7   anticipation on "me": he dips/squashes ~9 %, the mouth flares and widens
  49.45-50.5  EV suck ("suga"): yank - stretched up into the mouth (rope twist, thinning, the part
              near the lip turns into energy), directional motion blur along the flow, gone by 50.55
  50.55-50.86 EV ceil_portal_close: implosion to a point, white-violet flash (peak 50.855),
              3-D spark burst; under the flash the camera pushes into the ceiling and the tunnel
              iris opens from the collapse point (50.86-51.2)
  51.0-53.55  EV tunnel: full-frame wormhole flight, accelerating, exit light grows, short
              white-out (peak 53.45-53.50) that dissolves into space
  53.52-54.6  EV arrive: small portal at the top springs open and spits him out (inverse
              suction), he lands with a squash and settles, the portal closes with a pop.

Person handling: the warp runs in person() (before suit) with the CENTRE shutter sample only (so
the body stays opaque) and publishes the maps in S.meta['person_warp'] with every pixel where
the body is gone poisoned to -100, so the suit's remapped seams vanish with the body.  The
motion blur is a directional (vector) blur along the warp flow, applied in lights() - i.e. after
every person() hook, so the suit colour and its seams (S.meta['suit_emis']) are blurred with the
same flow.  The person is fully hidden (alpha = 0) from 50.55 (EV suck end) until 53.57.
"""
import numpy as np
import cv2

from .. import config as C
from .. import core
from .. import assets as A
from ..core import Layer, Light, smoothstep
from ..glfx import gl
from . import ceiling_portal_glsl as SH

W, H = C.W, C.H
EV = C.EV
DT = 1.0 / C.FPS

T_ON, T_OFF = 43.5, 54.6

# ------------------------------------------------------------------ geometry
# ceiling portal: centre (px, yc), perspective plane with horizon yh and strength k
# (px follows his pointing finger: the index ray meets the ceiling at x ~ 600-625 during 46.6-47.4)
CEIL = dict(px=588.0, yc=290.0, yh=640.0, k=0.25, rx0=300.0)
# arrival portal (smaller, top of frame)
ARR = dict(px=578.0, yc=150.0, yh=640.0, k=0.12, rx0=190.0)
for _g in (CEIL, ARR):
    _g['A'] = _g['yh'] - _g['yc']
    _top = _g['yh'] - _g['A'] / (1 - _g['k'])
    _bot = _g['yh'] - _g['A'] / (1 + _g['k'])
    _g['ry'] = (_bot - _top) / 2
    _g['ymid'] = (_bot + _top) / 2

# timeline constants
SUCK0, SUCK1 = 49.45, 50.55          # yank (main stretch) window
HIDE0, HIDE1 = 50.55, 53.57          # person invisible
ARR0, ARR1 = 53.58, 54.25            # emergence
ARR_OPEN = 53.50                      # arrival portal spring start (just after the exit flash peak)
IRIS0, IRIS1 = 50.86, 51.20           # tunnel iris opens under the collapse flash
PUSH0, PUSH1 = 50.70, 51.24           # camera push into the ceiling
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
    op *= 1 + 0.12 * sstep(49.30, 49.70, t)                     # mouth widens for the intake ("me")
    op *= (1 - lin(COL0, 50.84, t)) ** 1.6 if t > COL0 else 1.0  # implosion
    st['open'] = max(op, 0.0)
    st['spin'] = integrate(ceil_omega, 45.5, t)
    pw = 0.55 * sstep(45.95, 46.9, t) + 0.15 * sstep(46.9, 47.6, t) + 0.15 * sstep(48.3, 49.3, t)
    pw += 0.45 * sstep(49.3, 49.8, t) + 0.5 * sstep(50.45, 50.84, t)
    pw += 0.25 * np.exp(-((t - 46.02) / 0.12) ** 2)              # ignition flare
    pw += 0.45 * np.exp(-((t - 49.43) / 0.09) ** 2)              # inhale flare on "me"
    st['pow'] = pw
    st['inner'] = sstep(46.15, 47.2, t)
    # shock rings: thin hot leading edge (shader) + refraction (shockD, px) that carries the read.
    # The opening ring launches only once the rim has passed half size, so it never reads as the rim.
    shR, shI, shD = 0.0, 0.0, 0.0
    if 46.28 < t < 46.95:
        x = lin(46.28, 46.95, t)
        shR, shI, shD = 0.55 + 1.9 * (1 - (1 - x) ** 2.2), 0.30 * (1 - x) ** 3, 6.0 * (1 - x) ** 2
    if 50.83 < t < 51.2:
        x = lin(50.83, 51.2, t)
        shR, shI, shD = 0.05 + 3.2 * (1 - (1 - x) ** 2.2), 0.60 * (1 - x) ** 3, 9.0 * (1 - x) ** 2
    st['shockR'], st['shockI'], st['shockD'] = shR, shI, shD
    # pinpoint: glimmer during charge, ignition at 45.96, implosion point at the collapse
    core_ = 0.0
    if 44.9 < t < 46.4:
        flick = 0.6 + 0.4 * np.sin(t * 53.0) * np.sin(t * 31.0)
        core_ += 0.18 * sstep(44.9, 45.8, t) * flick * (1 - sstep(46.0, 46.3, t))
        core_ += 1.5 * np.exp(-((t - 45.96) / 0.07) ** 2) + 0.5 * sstep(45.75, 45.95, t) * (1 - sstep(45.98, 46.35, t))
    if 50.6 < t < 51.05:
        core_ += 2.2 * sstep(50.62, 50.84, t) * (1 - sstep(50.86, 51.0, t))
    st['core'] = core_
    st['haze'] = 1.6 * sstep(44.7, 45.8, t) * (1 - 0.6 * sstep(46.0, 46.8, t)) + 0.5 * sstep(45.95, 47.0, t) + 0.5 * sstep(49.2, 50.0, t)
    st['haze'] *= 1 - sstep(50.6, 50.9, t)
    st['rays'] = (0.13 * sstep(46.3, 47.5, t) + 0.05 * sstep(48.4, 49.4, t) + 0.08 * sstep(49.3, 49.9, t)
                  + 0.06 * np.exp(-((t - 49.45) / 0.12) ** 2)) * (1 - sstep(50.45, 50.8, t))
    st['light'] = (1.1 * sstep(45.9, 46.9, t) + 0.25 * sstep(48.4, 49.3, t) + 0.7 * sstep(49.3, 49.9, t)) * (1 - sstep(50.5, 50.75, t)) \
        + 1.2 * np.exp(-((t - 46.0) / 0.1) ** 2) + 0.15 * sstep(45.0, 45.9, t) * (1 - sstep(45.9, 46.1, t)) \
        + 0.5 * np.exp(-((t - 49.43) / 0.09) ** 2)
    st['pull'] = 0.25 * sstep(47.2, 48.0, t) + 0.5 * sstep(48.4, 49.3, t) + 1.6 * sstep(49.3, 50.2, t)
    return st


def arr_omega(t):
    t = np.asarray(t, np.float64)
    return 3.2 - 1.6 * vsm(53.6, 54.1, t) + 10.0 * vsm(54.2, 54.43, t)


def arr_state(t):
    st = {}
    op = spring(t - ARR_OPEN, zeta=0.55, wn=11.0)
    if t > 54.2:
        op *= (1 - lin(54.2, 54.43, t)) ** 1.7
    st['open'] = max(op, 0.0)
    st['spin'] = integrate(arr_omega, 53.3, t)
    st['pow'] = 1.0 + 0.4 * sstep(53.55, 53.7, t) - 0.5 * sstep(53.9, 54.15, t) + 0.6 * sstep(54.2, 54.42, t)
    st['inner'] = sstep(ARR_OPEN + 0.02, 53.72, t)
    shR, shI, shD = 0.0, 0.0, 0.0
    if 54.43 < t < 54.62:
        x = lin(54.43, 54.6, t)
        shR, shI, shD = 0.05 + 2.0 * (1 - (1 - x) ** 2.0), 0.50 * (1 - x) ** 3, 6.0 * (1 - x) ** 2
    st['shockR'], st['shockI'], st['shockD'] = shR, shI, shD
    st['core'] = 1.6 * sstep(54.3, 54.43, t) * (1 - sstep(54.44, 54.56, t)) \
        + 0.9 * np.exp(-((t - (ARR_OPEN + 0.02)) / 0.05) ** 2)
    st['haze'] = 0.0
    st['rays'] = 0.09 * sstep(53.52, 53.8, t) * (1 - sstep(54.1, 54.4, t))
    st['light'] = (1.3 * sstep(ARR_OPEN, 53.66, t) * (1 - 0.55 * sstep(54.0, 54.3, t))) * (1 - sstep(54.32, 54.45, t)) \
        + 1.0 * np.exp(-((t - 54.44) / 0.05) ** 2)
    return st


TAU = 2 * np.pi


# ------------------------------------------------------------------ suction parameters
def suck_params(t):
    """Return (mode, s, ant, prepull) for the person warp, or None when untouched.
    s: 0 = intact, 1 = completely inside the portal.  ant: >0 squash (anticipation / landing),
    <0 stretch.  prepull: small absorption depth (px) - upper body straining toward the portal."""
    if 48.3 <= t < HIDE0:
        s = lin(SUCK0, SUCK1, t)
        # strain toward the mouth from "e" (48.4), relaxes into the dip
        pre = 120.0 * sstep(48.5, 49.2, t) * (1 - sstep(49.20, 49.36, t))
        # anticipation dip on "me": ~9 % squash down, released into the yank
        ant = float(np.sin(np.pi * lin(49.20, 49.64, t))) if t < 49.64 else 0.0
        return 'ceil', s, ant, pre
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
        return 1380.0 * s ** 1.5
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
        M = 280
        self.dN = M
        self.d_ph = rng.random(M)
        self.d_L = rng.uniform(1.8, 3.4, M)
        self.d_size = rng.uniform(0.7, 1.6, M)
        self.d_b = rng.uniform(0.3, 1.0, M) ** 1.5
        self.d_hue = rng.random(M)
        self.d_order = rng.permutation(M) / M
        self.d_bokeh = rng.random(M) < 0.06          # big defocused near-camera motes
        # wisps peeling off his hair/shoulders before and during the yank ---------
        Kw = 150
        self.wN = Kw
        self.w_tb = np.sort(48.55 + 1.55 * rng.random(Kw) ** 0.8)
        self.w_L = rng.uniform(0.35, 0.65, Kw)
        self.w_u = rng.random(Kw)
        self.w_ang = rng.uniform(0, TAU, Kw)
        self.w_b = rng.uniform(0.35, 1.0, Kw)
        self.w_hue = rng.random(Kw)
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
        self.a_tb = np.sort(rng.uniform(ARR0 - 0.02, ARR1 - 0.35, K))
        self.a_L = rng.uniform(0.2, 0.45, K)
        self.a_u = rng.random(K)
        self.a_v = rng.random(K)
        self.a_ang = rng.uniform(0, TAU, K)
        self.a_b = rng.uniform(0.4, 1.0, K)
        # collapse / close spark bursts: 3-D velocities (mostly down/toward the camera), drag, gravity
        B = 120
        self.bN = B
        az = rng.uniform(0, TAU, B)
        hor = rng.uniform(0.35, 1.0, B)
        down = rng.uniform(0.15, 1.0, B) ** 0.7
        toward = rng.uniform(0.0, 0.9, B)
        V = np.stack([np.cos(az) * hor, down, np.sin(az) * hor * 0.6 - toward], 1)
        V /= np.linalg.norm(V, axis=1, keepdims=True)
        self.b_V = V * (rng.uniform(1.2, 4.8, B) ** 1.15)[:, None]
        self.b_drag = rng.uniform(1.6, 4.2, B)
        self.b_b = rng.uniform(0.3, 1.0, B) ** 1.3
        self.b_life = rng.uniform(0.18, 0.55, B)
        self.b_w = rng.uniform(0.5, 1.6, B)
        # tunnel stars ---------------------------------------------------------------
        S = 900
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
        self._shake(S, t)
        # glow strength eased in/out (no pop when the module switches on/off)
        S.bloom = max(S.bloom, 1.0 + 0.25 * sstep(45.5, 46.0, t) * (1 - sstep(54.3, 54.6, t)))
        S.meta['portal_center'] = (CEIL['px'], CEIL['yc']) if t < 52 else (ARR['px'], ARR['yc'])
        # camera push into the ceiling toward the collapse point (the studio is drawn by cleanup
        # before this hook): zoom with radial motion blur
        if PUSH0 < t < PUSH1 + 0.05 and S.bg is not None and S.bg.any():
            z, zr = self._push(t)
            if z > 1.002:
                S.bg = self._zoom_blur(S.bg, (CEIL['px'], CEIL['yc']), z, zr)

    @staticmethod
    def _push(t):
        """Camera push zoom factor and its rate (zoom per half frame) for the radial blur."""
        f = lambda x: 1.0 + 0.55 * sstep(PUSH0, PUSH1, x) ** 1.7
        z = f(t)
        zr = abs(f(t + 0.25 * DT) - f(t - 0.25 * DT))
        return z, zr

    @staticmethod
    def _zoom_blur(img, c, z, zr):
        n = int(np.clip(np.ceil(zr * 700), 1, 7))
        acc = None
        for k in range(n):
            zk = z * (1.0 + zr * (k / max(n - 1, 1) - 0.5)) if n > 1 else z
            M = np.float32([[zk, 0, c[0] * (1 - zk)], [0, zk, c[1] * (1 - zk)]])
            r = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            acc = r if acc is None else acc + r
        return acc / n

    @staticmethod
    def _lfn(t, seed):
        """Smooth band-limited (2-6 Hz) noise of t, roughly in [-1, 1]."""
        fr = (2.1, 3.3, 4.6, 5.7)
        wt = (0.42, 0.30, 0.18, 0.10)
        return sum(w * np.sin(TAU * f * t + 1.7 * seed + 2.3 * k * seed + k)
                   for k, (f, w) in enumerate(zip(fr, wt))) * 1.35

    def _shake(self, S, t):
        """Camera rumble: band-limited turbulence + short high-frequency kicks (collapse/landing)."""
        amp = 0.0
        amp += 3.2 * sstep(49.35, 50.3, t) * (1 - sstep(50.45, 50.62, t))
        amp += 4.0 * np.exp(-max(t - 50.84, 0) / 0.25) * (t > 50.84) * (t < 51.6)
        if TUN0 <= t < TUN1:
            amp += 1.6 + 3.6 * lin(TUN0, TUN1, t) ** 2 + 4.5 * np.exp(-((t - 52.3) / 0.16) ** 2)
        amp += 1.5 * np.exp(-max(t - 54.2, 0) / 0.2) * (t > 54.2)
        dx = dy = rot = 0.0
        if amp > 0.05:
            dx = amp * self._lfn(t, 1.0)
            dy = amp * self._lfn(t, 2.0) * 0.85
            rot = float(np.clip(amp * 0.035 * self._lfn(t, 3.0), -0.28, 0.28))
        # high-frequency kicks: only the first 2-3 frames of the collapse and of the landing
        for t0, A_, ax_ in ((50.845, 9.0, (0.45, 1.0)), (54.21, 4.5, (0.2, 1.0))):
            x = t - t0
            if 0 <= x < 0.12:
                k = A_ * np.exp(-x / 0.045) * np.cos(TAU * 8.0 * x)
                dx += k * ax_[0]
                dy += k * ax_[1]
        if abs(dx) + abs(dy) + abs(rot) > 0.02:
            S.shake.append((float(dx), float(dy), float(rot)))

    def person(self, S):
        # runs before suit's person(): suit detects the warped person and transfers itself through
        # S.meta['person_warp'] (centre warp maps), so the suit + its emissive seams stretch with him
        self._warp_person(S)

    def lights(self, S):
        t = S.t
        # all person() hooks (suit included) have run: blur the warped person along the flow
        vel = S.meta.pop('_cp_vel', None)
        if vel is not None:
            self._motion_blur(S, vel)
        out = []
        rims = []
        if t < 51.0:
            st = ceil_state(t)
            I = st['light']
            if I > 0.01:
                P = (CEIL['px'], CEIL['yc'] + 40)
                # soft coloured fill from the compositor (its quarter-res rim fogs dark hair, so
                # it is kept low); the crisp rim is drawn by _rim() below
                out.append(Light(P, (0.42, 0.50, 1.0), intensity=0.6 * I, radius=650, rim=0.12, diffuse=0.45, bg=0.012))
                out.append(Light((CEIL['px'] + 220, CEIL['yc'] + 20), (0.15, 0.85, 0.9), intensity=0.22 * I,
                                 radius=420, rim=0.1, diffuse=0.14, bg=0.004))
                out.append(Light((CEIL['px'] - 230, CEIL['yc'] + 20), (0.6, 0.2, 1.0), intensity=0.22 * I,
                                 radius=420, rim=0.1, diffuse=0.14, bg=0.004))
                rims.append(((CEIL['px'], CEIL['yc'] + 20), np.float32([0.50, 0.62, 1.25]), 0.7 * I, 700.0))
            # collapse flash
            fI = 3.0 * sstep(50.76, 50.85, t) * (1 - sstep(50.86, 50.95, t))
            if fI > 0.01:
                out.append(Light((CEIL['px'], CEIL['yc']), (0.7, 0.75, 1.0), intensity=fI, radius=800, rim=0.5, diffuse=0.4, bg=0.0))
        if t >= ARR_OPEN:
            st = arr_state(t)
            I = st['light']
            if I > 0.01:
                P = (ARR['px'], ARR['yc'] + 30)
                out.append(Light(P, (0.45, 0.5, 1.0), intensity=0.7 * I, radius=600, rim=0.15, diffuse=0.40, bg=0.01))
                out.append(Light((ARR['px'] + 160, ARR['yc']), (0.15, 0.85, 0.9), intensity=0.25 * I, radius=380, rim=0.12, diffuse=0.12))
                rims.append(((ARR['px'], ARR['yc'] + 20), np.float32([0.50, 0.62, 1.25]), 0.6 * I, 650.0))
        if rims:
            self._rim(S, rims)
        return out

    def _rim(self, S, rims):
        """Crisp thin rim (3-5 px band on the inner matte edge) facing the portal, scaled by the
        local albedo so dark hair gets a sheen instead of a milky cap."""
        a = S.alpha
        a8 = a[::8, ::8]
        ys, xs = np.where(a8 > 0.05)
        if len(ys) == 0:
            return
        x0, x1 = max(0, xs.min() * 8 - 16) // 2 * 2, min(W, xs.max() * 8 + 24) // 2 * 2
        y0, y1 = max(0, ys.min() * 8 - 16) // 2 * 2, min(H, ys.max() * 8 + 24) // 2 * 2
        ac = a[y0:y1, x0:x1]
        h2, w2 = (y1 - y0) // 2, (x1 - x0) // 2
        a2 = cv2.resize(ac, (w2, h2), interpolation=cv2.INTER_AREA)
        ab = cv2.GaussianBlur(a2, (0, 0), 2.2)
        band = np.clip((a2 - ab - 0.015) * 3.0, 0, 1) * smoothstep(0.6, 0.95, a2)
        if band.max() <= 0:
            return
        an = cv2.GaussianBlur(a2, (0, 0), 5.0)
        gx = cv2.Sobel(an, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(an, cv2.CV_32F, 0, 1, ksize=3)
        gm = np.sqrt(gx * gx + gy * gy) + 1e-6
        nx, ny = -gx / gm, -gy / gm
        X, Y = core.grid(h2, w2)
        X = X * 2 + x0 + 1
        Y = Y * 2 + y0 + 1
        field = np.zeros((h2, w2, 3), np.float32)
        for pos, col, I, rad in rims:
            dx, dy = pos[0] - X, pos[1] - Y
            d = np.sqrt(dx * dx + dy * dy) + 1e-3
            fall = I / (1.0 + (d / rad) ** 2)
            facing = np.clip((nx * dx + ny * dy) / d, 0, 1) ** 2
            field += (facing * fall)[..., None] * col
        lum = (cv2.resize(S.fg[y0:y1, x0:x1], (w2, h2), interpolation=cv2.INTER_AREA) * np.float32([0.2126, 0.7152, 0.0722])).sum(-1)
        acc = field * (band * (0.10 + 1.6 * np.clip(lum, 0, 0.4)))[..., None]
        S.fg = S.fg.copy() if S.fg is S.raw_fg else S.fg
        S.fg[y0:y1, x0:x1] += cv2.resize(acc, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)

    # ------------------------------------------------------------------ person warp
    def _axis(self, alpha):
        """Body axis and half-width per row (quarter-res rows) from the row's actual silhouette
        extent (left-most .. right-most matte pixel), so the twisted rope wraps the WHOLE row
        (arms + torso + the gaps between them) and nothing outside it has to be cut away."""
        q = 4
        a = cv2.resize(alpha, (W // q, H // q), interpolation=cv2.INTER_AREA)
        xs = np.arange(W // q, dtype=np.float32) * q + q / 2
        ys = np.arange(H // q, dtype=np.float32) * q + q / 2
        on = a > 0.35
        ok = on.sum(1) > 1
        if ok.sum() < 4:
            return ys, np.full_like(ys, W / 2), np.full_like(ys, 200.0)
        xl = np.where(ok, xs[np.argmax(on, axis=1)], 0.0)
        xr = np.where(ok, xs[W // q - 1 - np.argmax(on[:, ::-1], axis=1)], 0.0)
        cx = 0.5 * (xl + xr)
        hw = 0.5 * (xr - xl) + q
        cxv = np.interp(ys, ys[ok], cx[ok]).astype(np.float32)
        hwv = np.interp(ys, ys[ok], hw[ok]).astype(np.float32)
        cxv = cv2.GaussianBlur(cxv[:, None], (1, 0), 4).ravel()
        hwv = cv2.GaussianBlur(hwv[:, None], (1, 0), 2.5).ravel()
        return ys, cxv, np.maximum(hwv, 2.0)

    def _maps(self, t, mode, s, ant, pre, ax, off=(0.0, 0.0), chin=1100.0):
        """Inverse warp maps at half resolution (tracking coordinates: the screen-fixed portal is
        shifted by -person_offset): returns map_x, map_y, mask, dp, twist phase (half res)."""
        g = CEIL if mode == 'ceil' else ARR
        dirn = 1.0 if mode == 'ceil' else -1.0
        Px = g['px'] - off[0]
        Py = g['ymid'] + (8.0 if mode == 'ceil' else 4.0) - off[1]
        rx = g['rx0'] * 0.95
        ry = g['ry'] * 1.1
        q = 2
        h, w = H // q, W // q
        X = np.broadcast_to((np.arange(w, dtype=np.float32) * q + 0.5)[None, :], (h, w))
        Y = np.broadcast_to((np.arange(h, dtype=np.float32) * q + 0.5)[:, None], (h, w))
        # swirl around the mouth (localised to the lip)
        phim = 0.9 * sstep(0.15, 0.9, s) * dirn
        if abs(phim) > 1e-4:
            xn = (X - Px) / rx
            yn = (Y - Py) / ry
            rr = np.sqrt(xn * xn + yn * yn)
            phi = phim * np.exp(-np.maximum(rr - 0.5, 0.0) ** 2 / 1.2)
            c, sn = np.cos(phi), np.sin(phi)
            X1 = Px + (xn * c + yn * sn) * rx
            Y1 = Py + (-xn * sn + yn * c) * ry
        else:
            X1, Y1 = X, Y
        dp = Y1 - Py
        # vertical absorption / exponential stretch (the waist goes last)
        D = absorb_depth(s, mode) + pre
        S1 = D * np.exp(D / LAM)
        dB = H - Py
        lift0 = S1 * np.exp(-dB / LAM)
        anchor = (1.0 - sstep(0.68, 0.93, s)) if mode == 'ceil' else (1.0 - sstep(0.62, 0.95, s))
        dg = np.linspace(-800.0, 2400.0, 4000)
        Fg = dg - S1 * np.exp(-np.maximum(dg, 0) / LAM) + lift0 * anchor * (np.maximum(dg, 0) / dB) ** 4
        d = np.interp(dp.ravel(), Fg, dg).reshape(h, w).astype(np.float32)
        ysrc = Py + d
        lift_bot = lift0 * (1.0 - anchor)
        # funnel: narrow near the mouth, whole body thins as it goes
        N = sstep(0.02, 0.45, s)
        Dw = 420.0 + 520.0 * s
        wf = 0.12 + 0.88 * smoothstep(0.0, Dw, dp) ** 0.8
        wv = (1.0 - N * (1.0 - wf)) * (1.0 - 0.35 * sstep(0.45, 1.0, s))
        ays, axx, ahw = ax
        cxs = np.interp(ysrc, ays, axx).astype(np.float32)
        hws = np.interp(ysrc, ays, ahw).astype(np.float32)
        conv = smoothstep(0.0, 600.0, dp)
        wave = 16.0 * N * np.sin(dp / 85.0 - 11.0 * t * dirn) * smoothstep(0.0, 150.0, dp) * (1 - smoothstep(400.0, 900.0, dp))
        cxo = cxs + N * (Px - cxs) * (1 - conv) + wave
        # rope twist: rotate the body "cylinder" around the stream axis, faster near the mouth
        u = (X1 - cxo) / np.maximum(hws * wv, 2.0)
        Om = (2.0 * TAU * s ** 2.4) * dirn
        atw = N * Om * np.exp(-np.maximum(dp, 0.0) / 380.0)
        # keep the face readable: damp the twist on the head rows until it is deep in the stretch
        # (by then the energy wash covers it)
        hd = (1.0 - smoothstep(chin - 40.0, chin + 80.0, ysrc)) * (1.0 - sstep(0.45, 0.72, s))
        atw = atw * (1.0 - 0.6 * hd)
        au = np.abs(u)
        inside = au < 1.0
        ph = np.arcsin(np.clip(u, -1, 1)) + atw
        up = np.where(inside, np.sin(ph), u)
        xsrc = cxs + up * hws
        twon = np.clip(np.abs(atw) / 0.4, 0, 1)
        if abs(ant) > 1e-4:
            k = (0.09 if mode == 'ceil' else 0.045) * ant
            ysrc = H - (H - ysrc) / (1 - k)
            xsrc = cxs + (xsrc - cxs) / (1 + 0.55 * k)
        # strand noise (x in source px, y in OUTPUT px along the flow -> streaks along the stream)
        nu = np.mod(xsrc / 9.0, 256.0).astype(np.float32)
        nv = np.mod(dp / 120.0 - t * 2.5 * dirn + 50.0, 256.0).astype(np.float32)
        nz = cv2.remap(self.noise, nu, nv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        # rope: |u| < 1 spans the whole source row, so the twisted body keeps its own silhouette;
        # only a thin anti-aliased edge at the rope boundary
        m = 1.0 - twon * (1.0 - smoothstep(1.04, 0.95, au))
        m = m * smoothstep(-0.1 * ry, 0.75 * ry, dp)
        thr = N * 0.7 * (1 - smoothstep(0.0, 320.0, dp)) * sstep(0.2, 0.6, s)
        if lift_bot > 30.0:
            # the lower edge leaves the frame bottom: a ragged edge ~20 px wide in OUTPUT space
            sy = np.abs(np.gradient(ysrc, axis=0)) / 2.0
            dist_out = (H - 3.0 - ysrc) / np.maximum(sy, 0.05)
            rag = (nz - 0.5) * 70.0
            m = m * smoothstep(0.0, 22.0, dist_out + rag * min(1.0, lift_bot / 200.0))
            tb_ = 0.6 if mode == 'ceil' else 0.85
            thr = np.maximum(thr, tb_ * smoothstep(160.0, 0.0, dist_out) * min(1.0, (lift_bot - 30.0) / 60.0))
        if thr.max() > 0.01:
            m = m * smoothstep(thr - 0.12, thr + 0.12, nz)
        return xsrc.astype(np.float32), ysrc.astype(np.float32), m.astype(np.float32), dp.astype(np.float32), ph.astype(np.float32)

    def _warp_person(self, S):
        """Warp the person with the CENTRE shutter sample only (the body stays opaque); the flow
        field for the directional motion blur (applied in lights(), after the suit) comes from the
        maps at t +- a quarter frame."""
        t = S.t
        if HIDE0 <= t < HIDE1:
            S.alpha[:] = 0.0
            S.meta['person_hidden'] = True
            return
        sp = suck_params(t)
        if sp is None:
            return
        mode, s, ant, pre = sp
        if s <= 0 and abs(ant) < 1e-4 and pre < 0.5:
            return
        if s >= 0.999:
            S.alpha[:] = 0.0
            S.meta['person_hidden'] = True
            return
        off = (float(S.person_offset[0]), float(S.person_offset[1]))   # portal is screen-fixed
        ax = self._axis(S.alpha)
        S.meta['_cp_axis_x'] = float(np.median(ax[1][(ax[0] > 900) & (ax[0] < 1300)])) + float(S.person_offset[0])
        S.meta['_cp_hw'] = float(np.median(ax[2][(ax[0] > 1100) & (ax[0] < 1400)]))
        try:
            chin = float(A.face_oval(S.i)[:, 1].max())
        except Exception:
            chin = 1100.0
        xs_c, ys_c, mm, dp_c, ph_c = self._maps(t, mode, s, ant, pre, ax, off, chin)
        # flow (screen px per half frame = 180 deg shutter) from the maps at t -+ DT/4
        flows = []
        shut = 0.25 * (1.0 - 0.5 * sstep(0.5, 0.75, s))      # 180 deg shutter, 90 deg in the fastest part
        for o in (-shut, shut):
            sp2 = suck_params(t + o * DT) or sp
            xs2, ys2, _, _, _ = self._maps(t + o * DT, mode, sp2[1], sp2[2], sp2[3], ax, off, chin)
            flows.append((xs2, ys2))
        vx, vy = self._flow(xs_c, ys_c, flows[0], flows[1])
        mx = cv2.resize(xs_c, (W, H), interpolation=cv2.INTER_LINEAR)
        my = np.minimum(cv2.resize(ys_c, (W, H), interpolation=cv2.INTER_LINEAR), H - 1.0)
        m = cv2.resize(mm, (W, H), interpolation=cv2.INTER_LINEAR)
        src = np.dstack([S.fg * S.alpha[..., None], S.alpha]).astype(np.float32)
        wv = cv2.remap(src, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        a = np.clip(wv[..., 3] * m, 0, 1)
        S.alpha = a
        S.fg = wv[..., :3] / np.maximum(wv[..., 3:4], 1e-4)
        # publish the warp: poison every pixel where the body is gone so attached layers that
        # remap through it (suit colour + emissive seams) vanish with the body
        vis = cv2.dilate((a > 0.003).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        mxp, myp = mx.copy(), my.copy()
        mxp[~vis] = -100.0
        myp[~vis] = -100.0
        S.meta['person_warp'] = (mxp, myp)
        S.meta['_cp_warp'] = dict(mode=mode, s=s, dp=dp_c, ph=ph_c, off=off)
        S.meta['_cp_vel'] = (vx, vy)

    @staticmethod
    def _flow(xs, ys, m0, m1, vmax=64.0):
        """Screen-space displacement over the shutter (half res field, full-res px) of the source
        point shown at each output pixel: v = J^-1 (M(t-) - M(t+)), J = dM/dx of the centre map."""
        q = 2.0
        a_ = np.gradient(xs, axis=1) / q
        b_ = np.gradient(xs, axis=0) / q
        c_ = np.gradient(ys, axis=1) / q
        d_ = np.gradient(ys, axis=0) / q
        det = a_ * d_ - b_ * c_
        det = np.where(np.abs(det) < 1e-3, np.sign(det + 1e-12) * 1e-3, det)
        ex = m0[0] - m1[0]
        ey = m0[1] - m1[1]
        vx = (d_ * ex - b_ * ey) / det
        vy = (-c_ * ex + a_ * ey) / det
        mag = np.sqrt(vx * vx + vy * vy)
        k = np.minimum(1.0, vmax / np.maximum(mag, 1e-6))
        vx, vy = (vx * k).astype(np.float32), (vy * k).astype(np.float32)
        # the field is only used on the body: smooth out fold singularities
        vx = cv2.GaussianBlur(vx, (0, 0), 2.0)
        vy = cv2.GaussianBlur(vy, (0, 0), 2.0)
        return vx, vy

    def _motion_blur(self, S, vel):
        """Directional (vector) blur of the warped person along the warp flow: one crisp warp,
        smeared over a 180-degree shutter.  Runs after every person() hook (suit included); the
        suit's seams (S.meta['suit_emis']) get the same blur and the body visibility."""
        vx_h, vy_h = vel
        mag = np.sqrt(vx_h * vx_h + vy_h * vy_h)
        a = S.alpha
        a8 = a[::8, ::8]
        ys, xs = np.where(a8 > 0.002)
        if len(ys) == 0:
            return
        vmax = float(mag.max())
        if vmax < 1.0:
            return
        pad = int(min(vmax, 90.0) * 0.5) + 6
        x0, x1 = max(0, xs.min() * 8 - pad), min(W, xs.max() * 8 + 8 + pad)
        y0, y1 = max(0, ys.min() * 8 - pad), min(H, ys.max() * 8 + 8 + pad)
        ox, oy = float(S.person_offset[0]), float(S.person_offset[1])

        def field(xa, ya, xb, yb, shift):
            # full-res crop [xa,xb)x[ya,yb) (screen coords = tracking + shift) of the half-res field;
            # half-res sample j sits at full-res pixel 2 j + 0.5
            M = np.float32([[0.5, 0, (xa - shift[0] - 0.5) * 0.5], [0, 0.5, (ya - shift[1] - 0.5) * 0.5]])
            fl = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
            fx = cv2.warpAffine(vx_h, M, (xb - xa, yb - ya), flags=fl, borderMode=cv2.BORDER_REPLICATE)
            fy = cv2.warpAffine(vy_h, M, (xb - xa, yb - ya), flags=fl, borderMode=cv2.BORDER_REPLICATE)
            return fx, fy

        def blur(img, fx, fy):
            h, w = img.shape[:2]
            X, Y = core.grid(h, w)
            vm = float(np.sqrt(fx * fx + fy * fy).max())
            n = int(np.clip(np.ceil(vm / 5.0), 2, 12))
            acc = None
            for o in np.linspace(-0.5, 0.5, n).tolist():
                r = cv2.remap(img, (X + fx * o).astype(np.float32), (Y + fy * o).astype(np.float32),
                              cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                acc = r if acc is None else acc + r
            return acc / n

        fx, fy = field(x0, y0, x1, y1, (ox, oy))
        ac = a[y0:y1, x0:x1]
        pm = np.dstack([S.fg[y0:y1, x0:x1] * ac[..., None], ac]).astype(np.float32)
        out = blur(pm, fx, fy)
        na = np.clip(out[..., 3], 0, 1)
        S.alpha = S.alpha.copy() if S.alpha is S.raw_alpha else S.alpha
        S.fg = S.fg.copy() if S.fg is S.raw_fg else S.fg
        S.alpha[y0:y1, x0:x1] = na
        S.fg[y0:y1, x0:x1] = np.where(na[..., None] > 1e-4, out[..., :3] / np.maximum(out[..., 3:4], 1e-4),
                                     S.fg[y0:y1, x0:x1])
        # suit seams: same flow, hidden where the (blurred) body is gone
        em = S.meta.get('suit_emis')
        if em is not None:
            E, (ex0, ey0) = em
            eh, ew = E.shape[:2]
            if eh > 0 and ew > 0:
                fxe, fye = field(ex0, ey0, ex0 + ew, ey0 + eh, (0.0, 0.0))
                Eb = blur(np.ascontiguousarray(E, dtype=np.float32), fxe, fye)
                # body visibility in tracking coords (undo the float offset)
                M = np.float32([[1, 0, -ox], [0, 1, -oy]])
                at = cv2.warpAffine(S.alpha, M, (W, H), flags=cv2.INTER_LINEAR) if abs(ox) + abs(oy) > 0.01 else S.alpha
                ea = np.zeros((eh, ew), np.float32)
                cx0, cy0 = max(0, ex0), max(0, ey0)
                cx1, cy1 = min(W, ex0 + ew), min(H, ey0 + eh)
                if cx1 > cx0 and cy1 > cy0:
                    ea[cy0 - ey0:cy1 - ey0, cx0 - ex0:cx1 - ex0] = at[cy0:cy1, cx0:cx1]
                S.meta['suit_emis'] = ((Eb * smoothstep(0.05, 0.6, ea)).astype(np.float32), (ex0, ey0))

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
    @staticmethod
    def _dust_rate(x):
        return 1.0 + 0.3 * vsm(47.0, 48.0, x) + 0.4 * vsm(48.4, 49.4, x) + 1.0 * vsm(49.3, 50.2, x)

    def _dust(self, t, st, rays=None):
        """Studio dust pulled up into the ceiling portal, glowing only where it catches the god
        rays or the mouth's light.  Soft points with a hot core + faint halo; motion streaks only
        close to the mouth; a few big defocused motes near the camera.
        Returns (behind_img, front_img) or None."""
        g = CEIL
        dens = 0.15 * sstep(44.6, 45.9, t) + 0.55 * sstep(46.2, 47.6, t) + 0.30 * sstep(48.4, 49.5, t)
        dens *= 1 - sstep(50.5, 50.8, t)
        if dens <= 0.002:
            return None
        k = np.arange(self.dN)
        Px, Py = g['px'], g['yc'] + 10

        def pos(T):
            tt = T / self.d_L + self.d_ph
            cyc = np.floor(tt)
            tau = tt - cyc
            r0 = 90 + 600 * hash01(k, cyc, 1.0) ** 0.9
            phi0 = TAU * hash01(k, cyc, 2.0)
            y0 = 480 + 1050 * hash01(k, cyc, 3.0)
            e = tau ** 2.6
            r = r0 * (1 - e) ** 1.3 + 4
            phi = phi0 + 0.7 * tau + 3.2 * e
            y = y0 + (Py - y0) * (0.25 * tau + 0.75 * e)
            flat = 0.06 + 0.26 * np.clip((900 - y) / 620, 0, 1)
            x = Px + r * np.cos(phi)
            yy = y + r * np.sin(phi) * flat
            return x, yy, tau, np.sin(phi)

        rate = float(self._dust_rate(t))
        Tn = integrate(self._dust_rate, 43.0, t)
        x, y, tau, depth = pos(Tn)
        xp, yp, taup, _ = pos(Tn - 0.5 * DT * rate)
        wrap = taup > tau
        xp = np.where(wrap, x, xp)
        yp = np.where(wrap, y, yp)
        vis = np.clip((dens - self.d_order) * self.dN / 30.0, 0, 1)
        fade = vsm(0.0, 0.15, tau) * (1 - vsm(0.9, 1.0, tau))
        dist = np.hypot(x - Px, (y - Py) * 1.6)
        # light: god-ray shafts (sampled from the rays image) + the mouth's own glow
        shaft = np.zeros(self.dN)
        if rays is not None:
            rimg, ry0 = rays
            lum = rimg.mean(-1)
            xi = np.clip(x.astype(int), 0, W - 1)
            yi = np.clip(y.astype(int) - ry0, 0, lum.shape[0] - 1)
            ok = (y >= ry0) & (x >= 0) & (x < W)
            shaft = np.where(ok, lum[yi, xi], 0.0) / max(float(st['rays']), 1e-3)
            shaft = np.clip(shaft * 2.2, 0, 2.0)
        mouth = 1.4 * np.exp(-dist / 140.0) + 0.12 * np.exp(-dist / 420.0)
        lit = 0.015 + shaft + mouth
        b = self.d_b * vis * fade * lit * (0.35 + 0.3 * st['pow'])
        hue = self.d_hue
        col = np.stack([0.62 + 0.35 * hue, 0.70 + 0.2 * (1 - hue), np.full_like(hue, 1.15)], 1)
        bokeh = self.d_bokeh
        # streak only close to the mouth, length = velocity over the shutter
        near = np.exp(-np.maximum(dist - 40.0, 0.0) / 150.0)
        xp = x + (xp - x) * near
        yp = y + (yp - y) * near
        p0 = np.stack([x, y], 1)
        p1 = np.stack([xp, yp], 1)
        front = (depth < 0) & ~bokeh
        res = []
        for msk in (~front & ~bokeh, front):
            img = np.zeros((H, W, 3), np.float32)
            sel = msk & (b > 0.004)
            if sel.any():
                rad = self.d_size[sel]
                c4 = np.concatenate([col[sel] * b[sel, None], b[sel, None]], 1).astype(np.float32)
                img += gl.sprites((W, H), p0[sel], p1[sel], rad, c4 * 1.6, shape=2)[..., :3]
                img += gl.sprites((W, H), p0[sel], p1[sel], rad * 4.0, c4 * 0.10, shape=0)[..., :3]
            res.append(img)
        # big out-of-focus motes drifting in front of him (no streaks, very soft)
        sel = bokeh & (vis > 0)
        if sel.any():
            bb = (0.010 + 0.022 * self.d_b[sel]) * vis[sel] * fade[sel] * (0.25 + 0.75 * np.clip(shaft[sel] + mouth[sel], 0, 1.5))
            c4 = np.concatenate([col[sel] * np.float32([0.8, 0.9, 1.2]) * bb[:, None], bb[:, None]], 1).astype(np.float32)
            res[1] += gl.sprites((W, H), p0[sel], None, 12.0 + 10.0 * self.d_b[sel], c4, shape=0)[..., :3]
        return res

    def _stream_geom(self, mode, s):
        g = CEIL if mode == 'ceil' else ARR
        return g['px'], g['ymid'] + (8.0 if mode == 'ceil' else 4.0)

    def _fragments(self, t, mode, axis_x):
        """Energy fragments peeling off the stretched body near the mouth and spiralling in
        (mode 'ceil'), or shooting out of the arrival portal down onto the forming body ('arr')."""
        if mode == 'ceil':
            tb, L, u, v, ang, bb = self.f_tb, self.f_L, self.f_u, self.f_v, self.f_ang, self.f_b
        else:
            tb, L, u, v, ang, bb = self.a_tb, self.a_L, self.a_u, self.a_v, self.a_ang, self.a_b
        Px, Py = self._stream_geom(mode, 0)
        dirn = 1.0 if mode == 'ceil' else -1.0

        def taus(tt):
            age = tt - tb
            tau = age / L if mode == 'ceil' else 1 - age / L
            return tau, (age >= 0) & (age <= L)

        tau, al = taus(t)
        taup, alp = taus(t - 0.35 * DT)
        sel = al & alp
        if not sel.any():
            return None
        idx = np.where(sel)[0]
        te = tb[idx] if mode == 'ceil' else tb[idx] + L[idx]
        # origin ON the body at the emission time: between the (stretched) head top and ~450 px
        # below it, so nothing is emitted from empty air before the body reaches the mouth
        sp_e = [suck_params(x) or ('', 0.0, 0.0, 0.0) for x in te]
        s_e = np.array([q[1] for q in sp_e])
        pre_e = np.array([q[3] for q in sp_e])
        D_e = np.array([absorb_depth(x, mode) for x in s_e]) + pre_e
        S1_e = D_e * np.exp(D_e / LAM)
        dH = 595.0 if mode == 'ceil' else 700.0
        dp_head = np.maximum(dH - S1_e * np.exp(-dH / LAM), 25.0)
        dp0 = dp_head + 20 + 300 * v[idx] ** 1.4
        gate = vsm(0.22, 0.38, s_e) if mode == 'ceil' else vsm(0.1, 0.3, s_e)
        N = vsm(0.02, 0.5, s_e)
        wf = 0.045 + 0.955 * vsm(0.0, 720.0, dp0) ** 0.7
        halfw = 230.0 * (1.0 - N * (1.0 - wf))
        conv = vsm(0.0, 650.0, dp0)
        cx = axis_x + N * (Px - axis_x) * (1 - conv)
        ox = cx + (u[idx] - 0.5) * 2 * halfw
        oy = Py + dp0

        def path(tau_):
            e = np.clip(tau_, 0, 1) ** 1.7
            r = 1 - e
            spin = ang[idx] + dirn * (1.2 * e + 0.4 * tau_)
            orb = np.sin(np.pi * np.clip(tau_, 0, 1)) * (8 + 14 * bb[idx])
            x = Px + (ox - Px) * r + np.cos(spin) * orb
            y = Py + (oy - Py) * r + np.sin(spin) * orb * 0.35
            return x, y

        x0, y0 = path(tau[idx])
        x1, y1 = path(taup[idx])
        fade = np.sin(np.pi * np.clip(tau[idx], 0, 1)) ** 0.7
        b = bb[idx] * fade * 0.22 * gate * (bb[idx] > (0.75 if mode == 'ceil' else 0.82))
        hue = self.f_hue[idx]
        col = np.stack([0.30 + 0.55 * hue, 0.62 + 0.3 * (1 - hue), np.full_like(hue, 1.45)], 1) * 1.3
        rgba = np.concatenate([col * b[:, None], b[:, None]], 1).astype(np.float32)
        size = 0.6 + 0.6 * bb[idx]
        img = gl.sprites((W, H), np.stack([x0, y0], 1), np.stack([x1, y1], 1), size, rgba, shape=2)
        img += gl.sprites((W, H), np.stack([x0, y0], 1), np.stack([x1, y1], 1), size * 5.0, rgba * 0.08, shape=0)
        return img[..., :3]

    def _burst(self, t, g, t0, dur, scale=1.0):
        """3-D spark burst from the portal centre at t0: sparks fly out and down / toward the
        camera with drag and gravity, are projected with perspective (near ones grow and move
        faster), vary in length/width/brightness and cool white -> violet -> out."""
        if not (t0 <= t <= t0 + dur):
            return None
        f, cx, cy, Z0, G = 1000.0, W / 2.0, H / 2.0, 4.0, 3.2
        P0 = np.array([(g['px'] - cx) * Z0 / f, (g['yc'] - cy) * Z0 / f, Z0])

        def P(tau):
            tau = np.maximum(tau, 0.0)
            e = (1 - np.exp(-self.b_drag * tau)) / self.b_drag
            p = P0[None, :] + self.b_V * scale * e[:, None]
            p[:, 1] += 0.5 * G * scale * tau ** 2
            return p

        tau = np.full(self.bN, t - t0)
        pa = P(tau)
        pb = P(tau - 0.9 * DT)
        Za = np.maximum(pa[:, 2], 0.35)
        Zb = np.maximum(pb[:, 2], 0.35)
        xa, ya = cx + f * pa[:, 0] / Za, cy + f * pa[:, 1] / Za
        xb, yb = cx + f * pb[:, 0] / Zb, cy + f * pb[:, 1] / Zb
        u = np.clip(tau / self.b_life, 0, 1)
        alive = (u < 1) & (pa[:, 2] > 0.5)
        if not alive.any():
            return None
        near = np.clip(Z0 / Za, 0.5, 4.0)
        b = self.b_b * (1 - u) ** 1.6 * 3.2 * alive * vsm(0.5, 1.2, pa[:, 2])
        hot = np.array([1.6, 1.55, 2.0])
        mid = np.array([0.80, 0.45, 1.65])
        cool = np.array([0.38, 0.18, 0.95])
        k1 = vsm(0.0, 0.3, u)[:, None]
        k2 = vsm(0.3, 1.0, u)[:, None]
        col = (hot * (1 - k1) + mid * k1) * (1 - k2) + cool * k2
        rgba = np.concatenate([col * b[:, None], b[:, None]], 1).astype(np.float32)
        rad = (0.45 + 0.55 * self.b_w) * near
        sel = alive & (b > 0.01)
        p0 = np.stack([xa, ya], 1)[sel]
        p1 = np.stack([xb, yb], 1)[sel]
        img = gl.sprites((W, H), p0, p1, rad[sel], rgba[sel], shape=2)
        img += gl.sprites((W, H), p0, p1, rad[sel] * 4.5, rgba[sel] * 0.07, shape=0)
        return img[..., :3]

    # ------------------------------------------------------------------ lightning
    def _bolts(self, t, g, pw, open_, seed, rate=13.0):
        """Jagged lightning arcs crawling out of the rim along the portal plane (time-quantised
        flicker, each bolt lives 2-3 frames).  Returns an additive image or None."""
        if open_ < 0.15 or pw <= 0.05:
            return None
        P0, P1, RAD, COL = [], [], [], []
        slot = int(np.floor(t * rate))
        for sl in (slot - 1, slot):
            rng = np.random.default_rng(int(seed * 7919 + sl * 104729) & 0x7fffffff)
            nb = rng.poisson(1.2 + 2.2 * pw)
            for _ in range(nb):
                th = rng.uniform(0, TAU)
                L = rng.uniform(0.10, 0.22 + 0.25 * pw)
                outward = rng.random() < 0.75
                # path in (along-rim angle, radial) with midpoint displacement
                n = 17
                pts = np.zeros((n, 2))
                pts[:, 1] = np.linspace(0, 1, n)
                disp = np.zeros(n)
                step = n - 1
                amp = 0.35
                while step > 1:
                    half = step // 2
                    for i0 in range(0, n - 1, step):
                        disp[i0 + half] = (disp[i0] + disp[i0 + step]) / 2 + rng.normal(0, amp)
                    step = half
                    amp *= 0.55
                rr = 1.0 + (1 if outward else -0.6) * pts[:, 1] * L
                ang = th + disp * L * 0.9 + (rng.uniform(-1, 1) * 0.25) * pts[:, 1]
                u = rr * np.cos(ang) * open_
                v = rr * np.sin(ang) * open_
                x, y = disk_to_screen(g, u, v)
                life = 1.0 if sl == slot else 0.45
                br = rng.uniform(0.6, 1.0) * life * (0.6 + 0.6 * pw)
                taper = np.linspace(1.0, 0.25, n - 1)
                P0.append(np.stack([x[:-1], y[:-1]], 1))
                P1.append(np.stack([x[1:], y[1:]], 1))
                RAD.append(0.7 + 0.8 * taper)
                COL.append(br * taper)
                # one or two branches
                for _br in range(2):
                    if rng.random() >= (0.55 if _br == 0 else 0.35):
                        break
                    k0 = rng.integers(3, 12)
                    m = 7
                    bang = ang[k0] + np.cumsum(rng.normal(0, 0.05, m)) + rng.choice([-1, 1]) * 0.08
                    brr = rr[k0] + np.linspace(0, 0.5 * L, m) * (1 if outward else -0.6)
                    bx, by = disk_to_screen(g, brr * np.cos(bang) * open_, brr * np.sin(bang) * open_)
                    P0.append(np.stack([bx[:-1], by[:-1]], 1))
                    P1.append(np.stack([bx[1:], by[1:]], 1))
                    RAD.append(np.full(m - 1, 0.6))
                    COL.append(np.full(m - 1, br * 0.5))
        if not P0:
            return None
        p0 = np.concatenate(P0)
        p1 = np.concatenate(P1)
        rad = np.concatenate(RAD)
        c = np.concatenate(COL)
        core_c = np.stack([0.75 * c, 0.85 * c, 1.6 * c, c], 1).astype(np.float32) * 2.2
        glow_c = np.stack([0.25 * c, 0.35 * c, 1.0 * c, c], 1).astype(np.float32) * 0.22
        img = gl.sprites((W, H), p0, p1, rad, core_c, shape=2)[..., :3]
        img += gl.sprites((W, H), p0, p1, rad * 7.0, glow_c, shape=0)[..., :3]
        return img

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
        # while the iris opens, the vanishing point sits in the collapse point on the ceiling
        e = sstep(IRIS0, IRIS1 + 0.18, t)
        cx = CEIL['px'] * (1 - e) + cx * e
        cy = CEIL['yc'] * (1 - e) + cy * e
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
        img = np.ascontiguousarray(img[..., :3])
        # zoom (radial) motion blur at half res around the vanishing point
        zb = 0.004 + 0.010 * lin(TUN0, TUN1, t) ** 2
        ch, cw = c[1] / q, c[0] / q
        acc = img.copy()
        ntap = 8
        for k in range(1, ntap):
            sc = 1 + zb * k
            M = np.float32([[sc, 0, cw * (1 - sc)], [0, sc, ch * (1 - sc)]])
            acc += cv2.warpAffine(img, M, (W // q, H // q), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        img = cv2.resize(acc / ntap, (W, H), interpolation=cv2.INTER_LINEAR)
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
        rgba = np.concatenate([col * bright[:, None] * 0.9, bright[:, None]], 1).astype(np.float32)
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

        # entry: the tunnel resolves out of the collapse flash
        return out

    # ------------------------------------------------------------------ refraction
    def _lens(self, S, g, st, dirn, y1, amp):
        """Refractive swirl around the rim + shock-ring refraction (S.distort, quarter res)."""
        op = st['open']
        if amp <= 0.01 or (op < 0.02 and st.get('shockD', 0.0) < 0.05):
            return
        q = 4
        h, w = y1 // q, W // q
        X, Y = core.grid(h, w)
        X = X * q + q / 2
        Y = Y * q + q / 2
        rx, ry = g['rx0'], g['ry']
        xn = (X - g['px']) / rx
        yn = (Y - g['ymid']) / ry
        rn = np.sqrt(xn * xn + yn * yn) + 1e-3
        tx, ty = -yn / rn * rx, xn / rn * ry          # tangential (screen)
        tn = np.sqrt(tx * tx + ty * ty) + 1e-3
        tx, ty = tx / tn, ty / tn
        rxv, ryv = xn / rn * rx, yn / rn * ry          # radial (screen)
        rnn = np.sqrt(rxv * rxv + ryv * ryv) + 1e-3
        rxv, ryv = rxv / rnn, ryv / rnn
        sw = 0.0
        if op > 0.02:
            rr = rn / max(op, 1e-3)
            sw = amp * op * np.exp(-((rr - 1.18) / 0.32) ** 2) * dirn
            sw = sw * (0.8 + 0.2 * np.sin(rr * 9.0 - S.t * 6.0))
        dx = tx * sw
        dy = ty * sw
        if st.get('shockD', 0.0) > 0.05:
            R = st['shockR']
            dd = (rn - R) / (0.05 + 0.04 * R)
            ring = -2 * dd * np.exp(-dd * dd) * st['shockD']
            dx = dx + rxv * ring
            dy = dy + ryv * ring
        # never bend the studio below the ceiling by more than ~3 px (LED bars would jog)
        mag = np.sqrt(dx * dx + dy * dy) + 1e-6
        cap = 3.0 + 40.0 * (1 - smoothstep(260.0, 360.0, Y))
        k = np.minimum(1.0, cap / mag)
        dx = cv2.resize((dx * k).astype(np.float32), (W, y1), interpolation=cv2.INTER_LINEAR)
        dy = cv2.resize((dy * k).astype(np.float32), (W, y1), interpolation=cv2.INTER_LINEAR)
        fade = smoothstep(float(y1), y1 - 80.0, np.arange(y1, dtype=np.float32))[:, None]
        S.distort.append((0, 0, dx * fade, dy * fade))

    # ------------------------------------------------------------------ layer hooks
    def behind(self, S):
        t = S.t
        L = []
        if t < 51.25:
            st = ceil_state(t)
            if st['open'] > 0.001 or st['core'] > 0.001 or st['haze'] > 0.001 or st['shockI'] > 0.001:
                L.append(self._portal_layer(CEIL, st, t, 0, 600, 1.0, 1.0))
                self._lens(S, CEIL, st, 1.0, 600, 5.0 + 6.0 * sstep(49.3, 50.2, t))
            bl = self._bolts(t, CEIL, st['pow'], st['open'], 11.0)
            if bl is not None:
                L.append(Layer('add', bl))
            rays = None
            if st['rays'] > 0.005:
                y0 = 240
                r = self._rays_img(CEIL, st['rays'], t, y0, 2.0)
                L.append(Layer('add', r, box=(0, y0)))
                rays = (r, y0)
                S.meta['_cp_rays'] = rays
            d = self._dust(t, st, rays)
            if d is not None:
                L.append(Layer('add', d[0]))
                S.meta['_cp_dust_front'] = d[1]
        if IRIS0 <= t < 53.56:
            img = self._tunnel(t)
            e = sstep(IRIS0, IRIS1, t)
            if e < 0.999:
                # iris: opens from the collapse point in the ceiling plane (portal ellipse) and
                # rounds out into a full-frame circle as it engulfs the camera
                cen = np.array(self._tunnel_cam(t)[0])
                asp = CEIL['ry'] / CEIL['rx0'] + (1 - CEIL['ry'] / CEIL['rx0']) * e ** 0.8
                R = 18 + 1550 * e ** 1.5
                X, Y = core.grid()
                rr = np.sqrt((X - cen[0]) ** 2 + ((Y - cen[1]) / asp) ** 2)
                soft = 6.0 + 50.0 * e
                msk = smoothstep(R, R - soft, rr).astype(np.float32)
                wd = 3.0 + 10.0 * e
                e_out = np.exp(-((rr - R + 1.0) / wd) ** 2).astype(np.float32)
                e_in = np.exp(-((rr - R + 2.0 + 2.4 * wd) / (wd * 1.6)) ** 2).astype(np.float32)
                gain = 1.6 * (1 - e) + 0.3
                img = img * msk[..., None] + (e_out[..., None] * np.array([0.75, 0.55, 1.5], np.float32)
                                              + e_in[..., None] * np.array([0.2, 0.75, 1.1], np.float32) * 0.6) * gain
                L.append(Layer('over', img, np.maximum(msk, np.clip(e_out * gain, 0, 1))))
            else:
                fo = 1.0 - sstep(53.45, 53.535, t)          # dissolve into space under the exit flash
                L.append(Layer('over', img * fo, np.full((H, W), fo, np.float32)))
        if t >= ARR_OPEN - 0.02:
            st = arr_state(t)
            if st['open'] > 0.001 or st['core'] > 0.001 or st['shockI'] > 0.001:
                L.append(self._portal_layer(ARR, st, t, 0, 420, 7.0, -1.0))
                self._lens(S, ARR, st, -1.0, 420, 6.0)
            bl = self._bolts(t, ARR, st['pow'] * 0.7, st['open'], 23.0)
            if bl is not None:
                L.append(Layer('add', bl))
            if st['rays'] > 0.005:
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
            img, y0 = r
            veil = (1.0 - 0.85 * S.alpha[y0:y0 + img.shape[0]])[..., None]
            L.append(Layer('add', img * 0.06 * veil, box=(0, y0)))
        wi = self._wisps(S, t)
        if wi is not None:
            L.append(Layer('add', wi))
        wp = S.meta.get('_cp_warp')
        if wp is not None and 0 < wp['s'] < 1:
            L += self._stream_glow(S, wp)
        axis_x = S.meta.get('_cp_axis_x', W / 2)
        if SUCK0 - 0.05 <= t <= SUCK1 + 0.05:
            f = self._fragments(t, 'ceil', axis_x)
            if f is not None:
                L.append(Layer('add', f))
        if ARR0 - 0.05 <= t <= ARR1 + 0.1:
            f = self._fragments(t, 'arr', axis_x)
            if f is not None:
                L.append(Layer('add', f))
        b = self._burst(t, CEIL, 50.84, 0.6, scale=1.0)
        if b is not None:
            L.append(Layer('add', b))
        b = self._burst(t, ARR, 54.43, 0.17, scale=0.55)
        if b is not None:
            L.append(Layer('add', b))
        return L

    def _wisps(self, S, t):
        """Energy wisps peeling off the top of his silhouette (hair, shoulders) and streaming up
        into the mouth: visible build-up from "e" (48.4) through "me" into the yank."""
        if not (48.55 <= t <= 50.35):
            return None
        age = t - self.w_tb
        live = (age >= 0) & (age <= self.w_L)
        if not live.any():
            return None
        ramp = (0.35 * sstep(48.55, 49.2, t) + 0.65 * sstep(49.25, 49.6, t)) * (1 - sstep(50.1, 50.35, t))
        if ramp < 0.01:
            return None
        a4 = S.alpha[::4, ::4]
        on = a4 > 0.5
        cols = np.where(on.any(0))[0]
        if len(cols) < 3:
            return None
        top = np.argmax(on, axis=0).astype(np.float32) * 4
        ymin = top[cols].min()
        uc = cols[top[cols] < ymin + 70]
        if len(uc) < 2:
            return None
        idx = np.where(live)[0]
        xo = (uc.min() + self.w_u[idx] * (uc.max() - uc.min())) * 4.0
        yo = np.interp(xo / 4.0, uc.astype(np.float32), top[uc]) + 8.0
        Px, Py = CEIL['px'], CEIL['ymid'] + 6.0
        L = self.w_L[idx]
        bb = self.w_b[idx]
        ang = self.w_ang[idx]

        def path(tau_):
            tau_ = np.clip(tau_, 0, 1)
            e = tau_ ** 1.6
            sway = np.sin(np.pi * tau_) * (16 + 30 * bb) * np.sin(ang + 9.0 * tau_)
            spin = ang + 2.2 * e
            orb = np.sin(np.pi * tau_) ** 2 * (10 + 30 * bb) * e
            x = xo + (Px - xo) * e + sway + np.cos(spin) * orb
            y = yo + (Py - yo) * e + np.sin(spin) * orb * 0.35
            return x, y

        tau = age[idx] / L
        P0, P1, C4, R = [], [], [], []
        nseg = 6
        for k in range(nseg):
            ta = tau - k * 0.022
            tb = tau - (k + 1) * 0.022
            xa, ya = path(ta)
            xb, yb = path(tb)
            life = vsm(0.0, 0.12, tau) * (1 - vsm(0.45, 0.8, tau))
            fade = life * (1 - k / nseg) ** 1.4 * (ta > 0)
            b = bb * fade * ramp * 0.6
            col = np.stack([0.45 + 0.35 * self.w_hue[idx], 0.75 + 0.15 * (1 - self.w_hue[idx]), np.full(len(idx), 1.45)], 1)
            P0.append(np.stack([xa, ya], 1))
            P1.append(np.stack([xb, yb], 1))
            C4.append(np.concatenate([col * b[:, None], b[:, None]], 1))
            R.append(0.55 + 0.5 * bb * (1 - k / nseg))
        p0 = np.concatenate(P0).astype(np.float32)
        p1 = np.concatenate(P1).astype(np.float32)
        c4 = np.concatenate(C4).astype(np.float32)
        rad = np.concatenate(R).astype(np.float32)
        sel = c4[:, 3] > 0.004
        if not sel.any():
            return None
        img = gl.sprites((W, H), p0[sel], p1[sel], rad[sel], c4[sel] * 1.8, shape=2)
        img += gl.sprites((W, H), p0[sel], p1[sel], rad[sel] * 5.0, c4[sel] * 0.07, shape=0)
        return img[..., :3]

    def _stream_glow(self, S, wp):
        """Energy on the stretched body: the part near the mouth turns into luminous energy (an
        opaque 'over' layer that also hides the distorted features), teal/violet streaks smeared
        along the flow, helical bands following the twist, chromatic fringes."""
        s = wp['s']
        N = sstep(0.0, 0.3, s)
        if N < 0.01:
            return []
        a = S.alpha
        a4 = a[::4, ::4]
        ys, xs = np.where(a4 > 0.02)
        if len(ys) == 0:
            return []
        x0, x1 = max(0, xs.min() * 4 - 40) // 2 * 2, min(W, xs.max() * 4 + 44)
        y0, y1 = max(0, ys.min() * 4 - 120) // 2 * 2, min(H, ys.max() * 4 + 64)
        cw, chh = x1 - x0, y1 - y0
        hw_, hh_ = (cw + 1) // 2, (chh + 1) // 2

        ox_, oy_ = wp.get('off', (0.0, 0.0))
        sx, sy = int(round(ox_ / 2)), int(round(oy_ / 2))

        def up2(m):                    # half-res crop (tracking coords) -> full-res crop (screen)
            m = np.pad(m, ((abs(sy), abs(sy)), (abs(sx), abs(sx))), mode='edge')
            yy, xx = y0 // 2 - sy + abs(sy), x0 // 2 - sx + abs(sx)
            c = m[yy:yy + hh_, xx:xx + hw_]
            return cv2.resize(np.ascontiguousarray(c), (c.shape[1] * 2, c.shape[0] * 2), interpolation=cv2.INTER_LINEAR)[:chh, :cw]

        ac = a[y0:y1, x0:x1]
        dpc = up2(wp['dp'])
        ac = ac[:dpc.shape[0], :dpc.shape[1]]
        y1, x1 = y0 + ac.shape[0], x0 + ac.shape[1]
        prox = 1 - smoothstep(0.0, 560.0, dpc)
        e = ac * prox * N
        # flow smear + glow at half res
        eh = cv2.resize(e, (max(1, e.shape[1] // 2), max(1, e.shape[0] // 2)), interpolation=cv2.INTER_AREA)
        k = np.zeros((31, 1), np.float32)
        k[:, 0] = np.exp(-np.abs(np.arange(31) - 15) / 5.0)
        k[16:] *= 0.35                                     # smear trails below each feature
        k /= k.sum()
        smh = cv2.filter2D(eh, -1, k[::-1], borderType=cv2.BORDER_CONSTANT)
        g2h = core.glow(eh, 4.0)
        sm = cv2.resize(smh, (e.shape[1], e.shape[0]), interpolation=cv2.INTER_LINEAR)
        g2 = cv2.resize(g2h, (e.shape[1], e.shape[0]), interpolation=cv2.INTER_LINEAR)
        hot = np.clip(1 - dpc / 240.0, 0, 1)
        col_a = np.array([0.20, 0.70, 1.10], np.float32)     # teal
        col_b = np.array([0.70, 0.35, 1.25], np.float32)     # violet
        mixc = col_a[None, None] * (1 - hot[..., None]) + col_b[None, None] * hot[..., None]
        ph = up2(wp['ph'])[:ac.shape[0], :ac.shape[1]]
        dirn = 1.0 if wp['mode'] == 'ceil' else -1.0
        band = (0.5 + 0.5 * np.cos(2.0 * ph - dpc / 38.0 + 24.0 * S.t * dirn)) ** 10
        eb = ac * band * N * (1 - smoothstep(0.0, 760.0, dpc)) * sstep(0.12, 0.45, s)
        eb = cv2.GaussianBlur(eb, (0, 0), 1.2)
        img = mixc * (sm[..., None] * 0.7 + g2[..., None] * 0.35)
        img += (mixc * 0.6 + 0.5) * eb[..., None] * 1.4
        img += np.array([0.9, 0.95, 1.2], np.float32) * (e * hot ** 3)[..., None] * 0.9
        # chromatic fringes on the silhouette (subtle; none on the frame-bottom edge)
        sh = int(round(2 + 4 * N))
        up = np.zeros_like(ac)
        dn = np.zeros_like(ac)
        up[:-sh] = ac[sh:]
        dn[sh:] = ac[:-sh]
        lum = (S.fg[y0:y1, x0:x1] * np.array([0.3, 0.5, 0.2], np.float32)).sum(-1)
        fr = np.clip(up - ac, 0, 1) * (0.2 + lum) * N * 0.6
        fb = np.clip(dn - ac, 0, 1) * (0.2 + lum) * N * 0.6
        fade_b = smoothstep(H - 2.0, H - 40.0, np.arange(y0, y1, dtype=np.float32))[:, None] * (1 - smoothstep(450.0, 750.0, dpc))
        fr = fr * fade_b * sstep(0.15, 0.4, s)
        fb = fb * fade_b
        img += fr[..., None] * np.array([0.8, 0.12, 0.45], np.float32) + fb[..., None] * np.array([0.1, 0.25, 1.1], np.float32)
        # energy conversion near the lip: opaque luminous plasma that replaces the body
        Dc = 110.0 + 420.0 * sstep(0.08, 0.6, s)
        cov = ac * (1.0 - smoothstep(Dc * 0.35, Dc, dpc)) * sstep(0.05, 0.25, s)
        cov = cv2.GaussianBlur(cov, (0, 0), 2.0) * 0.95
        smn = np.clip(sm / (sm.max() + 1e-4), 0, 1)
        ecol = mixc * (0.35 + 0.9 * band[..., None] + 0.5 * smn[..., None]) + np.float32([0.6, 0.65, 0.9]) * (hot ** 2)[..., None]
        layers = []
        if cov.max() > 0.01:
            layers.append(Layer('over', (ecol * cov[..., None]).astype(np.float32), cov.astype(np.float32), box=(x0, y0)))
        layers.append(Layer('add', img.astype(np.float32), box=(x0, y0)))
        return layers

    def post(self, S, img):
        t = S.t
        # flashes: collapse (bridges into the tunnel) and tunnel exit (into space); they radiate
        # from their source and only white out the whole frame at the peak
        f1 = sstep(50.78, 50.855, t) * (1 - sstep(50.86, 50.96, t)) ** 2.0
        f2 = sstep(53.34, 53.47, t) ** 2 * (1 - sstep(53.48, 53.62, t)) ** 1.5
        for f, cen, tint in ((f1, (CEIL['px'], CEIL['yc']), (0.85, 0.85, 1.0)),
                             (f2, tuple(self._tunnel_cam(min(t, TUN1))[0]), (0.95, 0.97, 1.0))):
            if f <= 0.002:
                continue
            q = 8
            X, Y = core.grid(H // q, W // q)
            r = np.sqrt((X * q - cen[0]) ** 2 + (Y * q - cen[1]) ** 2)
            k = f ** 1.5
            prof = k + (1 - k) * (np.exp(-r / (120 + 900 * f)) * 0.8 + np.exp(-r / 45.0) * 0.6)
            prof = cv2.resize(prof.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
            g = (3.2 * f * prof)[..., None]
            img = img * (1 - 0.75 * f * prof[..., None]) + np.asarray(tint, np.float32) * g
        return img


def core_pose(i):
    from .. import assets as A
    return A.pose(int(np.clip(i, 0, C.N_FRAMES - 1)))
