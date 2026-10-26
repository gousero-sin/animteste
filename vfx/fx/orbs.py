"""orbs — fire ball (R hand, frame-left), ice orb (L hand, frame-right), fusion ring in front
of the face, plasma charge above the head and the finale detonation.

Active EV['anticipation'][0] (19.70 s) ... EV['finale_end'] (34.40 s).

Everything is a pure function of the frame time: hand tracks are pre-filtered once in
__init__ (temporal smoothing + a damped spring for lag/overshoot), particles are evaluated
analytically from seeded birth tables.
"""
import numpy as np
import cv2

from .. import config as C
from .. import assets as A
from .. import core
from ..core import Layer, Light, smoothstep, ease_out_cubic, ease_out_back, ease_in_out
from ..glfx import gl
from . import orbs_shaders as SH

W, H = C.W, C.H
FPS = C.FPS
EV = C.EV

T_START = EV['anticipation'][0]          # 19.70
T_FIRE = EV['fire_on']                   # 22.86
T_ICE = EV['ice_on']                     # 26.56
T_FUS0, T_FUS1 = EV['fusion']            # 28.70, 31.00
T_REL = EV['release']                    # 31.00
T_BOOM = EV['finale_boom']               # 32.62
T_END = EV['finale_end']                 # 34.40

# fusion choreography: as the hands close in on the face the orbs shrink and slide outward
# (off his eyes), then fly into the finger ring and unroll into the two helix strands
T_PUSH0, T_PUSH1 = 28.00, 28.42          # shrink + push away from the face
T_FLY0, T_FLY1 = 28.38, 28.92            # palm -> ring entry
T_UNR0, T_UNR1 = 28.74, 29.45            # strands unroll into full loops
T_HEAD0 = 28.74                          # strand heads start orbiting
SHUTTER = 0.5 / FPS                      # motion-blur shutter for streak sprites (s)
# palm occlusion of the fire's lower hemisphere by the fingers. 0: in this footage the R palm faces
# the camera with the fingers up, so the ball sits IN FRONT of the hand (seated by the palm light)
FIRE_OCC = 0.0

FIRE_COL = np.array([1.0, 0.43, 0.12], np.float32)
ICE_COL = np.array([0.35, 0.72, 1.0], np.float32)
VIOLET = np.array([0.72, 0.42, 1.0], np.float32)


# ------------------------------------------------------------------ small helpers
def sstep(a, b, x):
    return float(smoothstep(a, b, x))


def lerp(a, b, k):
    return a + (b - a) * k


def flicker(t, seed=0.0, speed=1.0):
    """smooth deterministic 1D noise in ~[-1, 1]"""
    t = t * speed
    return (0.5 * np.sin(t * 13.1 + seed) + 0.3 * np.sin(t * 23.7 + 1.3 * seed + 0.7)
            + 0.2 * np.sin(t * 41.3 + 2.1 * seed + 2.1))


def gauss_smooth(x, sigma):
    """temporal gaussian smoothing along axis 0 (edge-padded)."""
    if sigma <= 0:
        return x.copy()
    r = int(3 * sigma + 1)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.concatenate([np.repeat(x[:1], r, 0), x, np.repeat(x[-1:], r, 0)], 0)
    out = np.zeros_like(x, dtype=np.float64)
    for j, kv in enumerate(k):
        out += kv * pad[j:j + len(x)]
    return out.astype(np.float32)


def spring(target, f=3.2, zeta=0.55, sub=6):
    """critically/under-damped spring follow of a (N, D) target sequence (one row per frame)."""
    w = 2 * np.pi * f
    dt = 1.0 / FPS / sub
    x = target[0].astype(np.float64).copy()
    v = np.zeros_like(x)
    pos = np.zeros_like(target, dtype=np.float64)
    vel = np.zeros_like(target, dtype=np.float64)
    for n in range(len(target)):
        tg = target[n]
        for _ in range(sub):
            a = w * w * (tg - x) - 2 * zeta * w * v
            v += a * dt
            x += v * dt
        pos[n] = x
        vel[n] = v
    return pos.astype(np.float32), vel.astype(np.float32)


def bbody_np(x):
    """CPU twin of the GLSL bbody() ramp (x: (N,) temperature) -> (N,3)."""
    x = np.maximum(np.asarray(x, np.float32), 0)[:, None]
    def ss(a, b):
        return np.clip((x - a) / (b - a), 0, 1) ** 2 * (3 - 2 * np.clip((x - a) / (b - a), 0, 1))
    c = np.array([0.55, 0.06, 0.012]) * ss(0.02, 0.22)
    c = c + np.array([1.05, 0.30, 0.035]) * ss(0.12, 0.45)
    c = c + np.array([0.75, 0.55, 0.12]) * ss(0.38, 0.78)
    c = c + np.array([1.10, 1.05, 0.85]) * ss(0.72, 1.15)
    return c.astype(np.float32)


class Births:
    """Seeded table of particle births in [t0, t1] at a given rate (analytic particles)."""

    def __init__(self, rng, t0, t1, rate, life, nrand=10):
        n = max(1, int((t1 - t0) * rate))
        self.tb = (t0 + (np.arange(n) + rng.random(n)) / n * (t1 - t0)).astype(np.float32)
        self.life = rng.uniform(life[0], life[1], n).astype(np.float32)
        self.u = rng.random((n, nrand)).astype(np.float32)
        self.t0, self.t1 = t0, t1

    def alive(self, t, extra=0.0):
        m = (self.tb <= t) & (t < self.tb + self.life + extra)
        idx = np.nonzero(m)[0]
        return idx, (t - self.tb[idx]).astype(np.float32)


# ------------------------------------------------------------------ the effect
class FX(core.Effect):
    name = 'orbs'

    def __init__(self):
        self.rng = np.random.default_rng(20240622)
        self._frame_cache = (None, None)
        self._build_tracks()
        self._build_particles()
        self._build_haze_tex()
        self._build_ice_geom()
        self.p_fire = gl.program(SH.FIRE)
        self.p_glow = gl.program(SH.GLOW)
        self.p_ice = gl.program(SH.ICE)
        self.p_mist = gl.program(SH.MIST)
        self.p_plasma = gl.program(SH.PLASMA)
        self.p_boom = gl.program(SH.BOOM)

    # .............................................................. tracking
    def _build_tracks(self):
        self.i0 = C.i_of(T_START) - 40
        self.i1 = min(C.N_FRAMES - 1, C.i_of(T_END) + 10)
        fr = np.arange(self.i0, self.i1 + 1)
        n = len(fr)
        anch = {'R': np.zeros((n, 2), np.float32), 'L': np.zeros((n, 2), np.float32)}
        hsz = {'R': np.zeros(n, np.float32), 'L': np.zeros(n, np.float32)}
        palm = {'R': np.zeros((n, 2), np.float32), 'L': np.zeros((n, 2), np.float32)}
        tips = {'R': np.zeros((n, 2), np.float32), 'L': np.zeros((n, 2), np.float32)}
        ring = np.zeros((n, 3), np.float32)
        head = np.zeros((n, 3), np.float32)   # cx, top y, face width
        fcen = np.zeros((n, 2), np.float32)   # face centre
        prev_ring = np.array([590, 1065, 90], np.float32)
        for k, i in enumerate(fr):
            for s in 'RL':
                p = A.palm(i, s)
                up = A.palm_normal_up(i, s)
                hs = float(np.clip(A.hand_size(i, s), 120, 240))
                d = 0.45 * up + 0.55 * np.array([0, -1], np.float32)
                d /= np.linalg.norm(d) + 1e-6
                anch[s][k] = p + d * 0.40 * hs
                hsz[s][k] = hs
                palm[s][k] = p
                h = A.hand(i, s)
                tips[s][k] = h[[4, 8, 12], :2].mean(0) if h is not None else p
            hr, hl = A.hand(i, 'R'), A.hand(i, 'L')
            if hr is not None and hl is not None:
                idx = [3, 4, 7, 8, 11, 12]
                pts = np.r_[hr[idx, :2], hl[idx, :2]].astype(np.float64)
                x, y = pts[:, 0], pts[:, 1]
                M = np.c_[2 * x, 2 * y, np.ones_like(x)]
                cc = np.linalg.lstsq(M, x * x + y * y, rcond=None)[0]
                r = np.sqrt(max(cc[2] + cc[0] ** 2 + cc[1] ** 2, 1.0))
                if 60 < r < 140 and abs(cc[0] - 590) < 80 and abs(cc[1] - 1065) < 90:
                    prev_ring = np.array([cc[0], cc[1], r], np.float32)
            ring[k] = prev_ring
            fo = A.face_oval(i)
            fw = float(fo[:, 0].max() - fo[:, 0].min())
            head[k] = (fo[:, 0].mean(), fo[:, 1].min() - 0.42 * fw, fw)
            fcen[k] = fo.mean(0)
        self.fr = fr
        self.raw_anchor = anch
        for s in 'RL':
            sm = gauss_smooth(anch[s], 1.2)
            anch[s], _ = sm, None
        self.anchor = {}
        self.anchor_vel = {}
        for s in 'RL':
            pos, vel = spring(anch[s], f=3.4, zeta=0.52)
            self.anchor[s] = pos
            self.anchor_vel[s] = vel
        self.hsize = {s: gauss_smooth(hsz[s][:, None], 4.0)[:, 0] for s in 'RL'}
        self.palm = {s: gauss_smooth(palm[s], 1.0) for s in 'RL'}
        self.tips = {s: gauss_smooth(tips[s], 1.0) for s in 'RL'}
        self.ring = gauss_smooth(ring, 3.0)
        hd = gauss_smooth(head, 4.0)
        self.head = hd
        self.fcen = gauss_smooth(fcen, 3.0)
        # first time the ice hand is tracked in frame (its anticipation idles at 30 % until then)
        self.t_rdet = T_FIRE - 1.0
        for i in range(C.i_of(T_START), C.i_of(T_FIRE)):
            if A.hand_detected(i, 'R'):
                self.t_rdet = C.t_of(i)
                break
        self.t_ldet = T_ICE - 1.0
        for i in range(C.i_of(24.0), C.i_of(T_ICE)):
            if A.hand_detected(i, 'L'):
                self.t_ldet = C.t_of(i)
                break
        # plasma orb anchor well above the head: above the captions band that sits on the crown
        tgt = np.c_[hd[:, 0], hd[:, 1] - 300.0].astype(np.float32)
        self.over_head, _ = spring(gauss_smooth(tgt, 3.0), f=1.2, zeta=0.8)

    def _ix(self, t):
        """fractional index into track arrays for time t"""
        return np.clip(t * FPS - self.i0, 0, len(self.fr) - 1)

    def _interp(self, arr, t):
        x = self._ix(np.asarray(t, np.float32))
        j = np.floor(x).astype(int)
        j1 = np.minimum(j + 1, len(self.fr) - 1)
        f = (x - j)
        if arr.ndim == 1:
            return arr[j] * (1 - f) + arr[j1] * f
        return arr[j] * (1 - f)[..., None] + arr[j1] * f[..., None]

    # .............................................................. timeline: fire
    def fire_state(self, t):
        """None or dict(c, r, grow, amp, lean, core, amount)."""
        if t < 22.30 or t > 29.12:
            return None
        c = self._interp(self.anchor['R'], t)
        v = self._interp(self.anchor_vel['R'], t)
        hs = float(self._interp(self.hsize['R'], t))
        r = float(np.clip(0.47 * hs, 70, 100))
        st = dict(c=c.astype(np.float32), v=v, r=r, seed=3.7)
        if t < T_FIRE:
            # seed pearl: grows, then contracts right before ignition (anticipation)
            g = sstep(22.30, 22.72, t)
            squash = sstep(22.70, T_FIRE, t)
            st.update(kind='seed', grow=0.20 * g * (1 - 0.55 * squash), amp=0.7 + 1.6 * squash, amount=g * 0.6)
            return st
        a = t - T_FIRE
        grow = float(ease_out_back(min(a / 0.42, 1.0), 2.4))
        grow = max(grow, 0.0)
        amp = 1.0 + 1.3 * np.exp(-a / 0.12)
        st.update(kind='fire', grow=grow, amp=amp, amount=sstep(0, 0.12, a), core=0.45 + 0.55 * np.exp(-a / 0.2))
        # fusion: shrink + slide off his eyes, then fly into the finger ring
        if t > T_PUSH0:
            pos, sc, k = self._fusion_move('R', c, r, t)
            st['c'] = pos
            st['grow'] = grow * sc
            st['amount'] *= 1 - sstep(28.86, 29.10, t)
            st['v'] = v * (1 - k)
            st['fly'] = k
        return st

    def _fusion_move(self, side, c, r, t):
        """(pos, scale, fly) of an orb while the hands close in on the face (T_PUSH*) and while it
        flies off the palm into the ring entry point (T_FLY*)."""
        fc = self._interp(self.fcen, t)
        rc = self._interp(self.ring, t)
        push = sstep(T_PUSH0, T_PUSH1, t)
        scale = 1.0 - 0.36 * push
        d = c - fc
        dist = float(np.hypot(d[0], d[1])) + 1e-6
        dmin = rc[2] + r * scale + 36.0
        if dist < dmin:
            c = lerp(c, fc + d / dist * dmin, push)
        k = 0.0
        if t > T_FLY0:
            k = sstep(T_FLY0, T_FLY1, t)
            ent = self.ring_entry(side, t)
            c = self._spiral(c, ent, k, rc, +1 if side == 'R' else -1)
            scale *= 1 - 0.72 * k
        return np.asarray(c, np.float32), float(scale), float(k)

    def ring_entry(self, side, t):
        rc = self._interp(self.ring, t)
        th = self.stream_head_angle(side, t)
        return np.array([rc[0] + rc[2] * np.cos(th), rc[1] + rc[2] * np.sin(th)], np.float32)

    def stream_head_angle(self, side, t):
        # fire enters from the left (angle pi), ice from the right (0); angular speed ramps up
        base = np.pi if side == 'R' else 0.0
        a = max(t - T_HEAD0, 0.0)
        return base + (2.4 * a + 1.7 * a * a)

    def _spiral(self, p0, p1, k, rc, sgn):
        """point on a curved path from p0 to p1 (bulging around the ring centre)."""
        e = k * k * (3 - 2 * k)
        mid = (p0 + p1) / 2
        d = p1 - p0
        nrm = np.array([-d[1], d[0]], np.float32) * 0.25 * sgn
        return ((1 - e) ** 2 * p0 + 2 * (1 - e) * e * (mid + nrm) + e * e * p1).astype(np.float32)


    # .............................................................. ice geometry / timeline
    def _build_ice_geom(self):
        rng = np.random.default_rng(909)
        n = 26
        k = np.arange(n) + 0.5
        phi = np.arccos(1 - 2 * k / n)
        th = np.pi * (1 + 5 ** 0.5) * k
        N = np.c_[np.cos(th) * np.sin(phi), np.cos(phi), np.sin(th) * np.sin(phi)]
        N += rng.normal(0, 0.10, N.shape)
        N /= np.linalg.norm(N, axis=1, keepdims=True)
        D = 1.0 + rng.uniform(-0.035, 0.035, n)
        self.ice_planes = np.c_[N, D].astype(np.float32)
        ns = 10
        axes = []
        while len(axes) < ns:
            v = rng.normal(0, 1, 3)
            v /= np.linalg.norm(v)
            if v[1] < -0.35:          # not into the palm
                continue
            if abs(v[2]) > 0.8:       # not straight at / away from camera
                continue
            if any(np.dot(v, a) > 0.80 for a in axes):
                continue
            axes.append(v)
        axes = np.array(axes)
        perp = np.cross(axes, np.array([0.3, 0.2, 0.93]))
        perp /= np.linalg.norm(perp, axis=1, keepdims=True)
        self.sh_axis = axes.astype(np.float32)
        self.sh_perp = perp.astype(np.float32)
        self.sh_full = rng.uniform(1.55, 2.15, ns).astype(np.float32)
        self.sh_final = rng.uniform(1.30, 1.72, ns).astype(np.float32)
        self.sh_rad = rng.uniform(0.17, 0.27, ns).astype(np.float32)
        self.sh_t = (26.20 + 0.30 * rng.permutation(ns) / ns).astype(np.float32)

    def ice_state(self, t):
        if t < 26.15 or t > 29.15:
            return None
        c = self._interp(self.anchor['L'], t)
        v = self._interp(self.anchor_vel['L'], t)
        hs = float(self._interp(self.hsize['L'], t))
        r = float(np.clip(0.30 * hs, 52, 70))
        # shards: burst out, then lock (retract into a compact crystal crown)
        g = np.clip((t - self.sh_t) / 0.28, 0, 1)
        grow = np.array([ease_out_back(x, 1.8) for x in g], np.float32) * (g > 0)
        lock = sstep(T_ICE, T_ICE + 0.22, t)
        L = grow * (self.sh_full * (1 - lock) + self.sh_final * lock)
        L += np.sin(np.clip((t - T_ICE) / 0.35, 0, 1) * np.pi) * 0.06 * (t > T_ICE)
        corek = float(ease_out_back(np.clip((t - 26.40) / 0.30, 0, 1), 2.2)) if t > 26.40 else 0.0
        flash = 0.0
        if t > T_ICE - 0.04:
            flash = 0.38 * np.exp(-max(t - (T_ICE + 0.02), 0) / 0.06) * sstep(T_ICE - 0.04, T_ICE + 0.02, t)
        # orientation: fast spin during formation that decelerates and locks, then slow drift
        spin = 0.35 * t + 2.2 * (1 - np.exp(-max(t - 26.2, 0) / 0.22))
        tilt = 0.32 + 0.12 * np.sin(0.7 * t)
        st = dict(c=c.astype(np.float32), v=v, r=r, L=L.astype(np.float32), core=corek, flash=flash,
                  spin=spin, tilt=tilt, opacity=1.0, glow=1.0 + 0.10 * flicker(t, 7.0, 0.5), scale=1.0,
                  amount=sstep(26.25, 26.65, t))
        if t > T_PUSH0:
            pos, sc, k = self._fusion_move('L', c, r, t)
            st['c'] = pos
            st['scale'] = sc
            st['opacity'] = 1 - sstep(28.86, 29.12, t)
            st['glow'] *= 1 + 1.5 * k
            st['amount'] *= st['opacity']
        return st

    def _rot(self, spin, tilt):
        cy, sy = np.cos(spin), np.sin(spin)
        cx, sx = np.cos(tilt), np.sin(tilt)
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], np.float32)
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], np.float32)
        return (Rx @ Ry).astype(np.float32)     # object->view would be R^T; we pass R (view->object)

    def _ice_post(self, S, img, st, t):
        c = S.tx(st['c'])
        r = st['r'] * st['scale']
        if r < 3 or st['opacity'] <= 0.002:
            return img
        ext = max(st['core'] * 1.08, float(st['L'].max()) + 0.08, 0.2) * r + 6
        x0, y0 = int(c[0] - ext), int(c[1] - ext)
        x1, y1 = int(c[0] + ext) + 1, int(c[1] + ext) + 1
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        if x1 - x0 < 4 or y1 - y0 < 4:
            return img
        m = int(0.9 * r) + 8
        tx0, ty0, tx1, ty1 = max(0, x0 - m), max(0, y0 - m), min(W, x1 + m), min(H, y1 + m)
        tex = np.ascontiguousarray(img[ty0:ty1, tx0:tx1])
        R = self._rot(st['spin'], st['tilt'])
        fst = self.fire_state(t)
        fdir = (0.0, 0.0, 0.0)
        if fst is not None and fst['kind'] == 'fire':
            fc = S.tx(fst['c'])
            d = np.array([fc[0] - c[0], -(fc[1] - c[1]), 0.35 * np.hypot(*(fc - c))], np.float32)
            d = d / (np.linalg.norm(d) + 1e-6) * float(fst['amount']) * 1.0
            fdir = tuple(map(float, d))
        ss = 2
        w, h = x1 - x0, y1 - y0
        uni = dict(uC=tuple(map(float, c)), uR=float(r), uSS=float(ss), uT=float(t), uCore=float(st['core']),
                   uPl=self.ice_planes, uSh=np.c_[self.sh_axis, st['L']].astype(np.float32),
                   uShB=np.c_[self.sh_perp, self.sh_rad].astype(np.float32),
                   uRx=tuple(map(float, R[0])), uRy=tuple(map(float, R[1])), uRz=tuple(map(float, R[2])),
                   uTexOff=(float(tx0), float(ty0)), uTexSize=(float(tx1 - tx0), float(ty1 - ty0)),
                   uFireDir=fdir, uGlow=float(st['glow']), uFlash=float(st['flash']), uOpacity=float(st['opacity']))
        out = gl.render(self.p_ice, (w * ss, h * ss), uniforms=uni, textures={'uScene': tex}, offset=(x0, y0))
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_AREA)
        a = out[..., 3:4]
        img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - a) + out[..., :3]
        return img

    def _mist_layer(self, S, t, c, r, amt):
        if amt <= 0.01:
            return []
        R = r * 1.0
        box = (c[0] - 3.2 * R, c[1] - 2.4 * R, c[0] + 3.2 * R, c[1] + 4.8 * R)
        img, b = self._gl_layer(self.p_mist, box, dict(uC=tuple(map(float, c)), uR=float(R), uT=float(t),
                                                         uAmt=float(amt), uCol=(0.42, 0.60, 0.85)), scale=0.5)
        if img is None:
            return []
        return [Layer('add', img[..., :3], box=b)]


    # .............................................................. fusion / release / plasma geometry
    def orb_pos(self, t):
        p = self._interp(self.over_head, t)
        return np.array([p[0], p[1] + 6 * np.sin(t * 3.1)], np.float32)

    def ring_geom(self, t):
        rc = self._interp(self.ring, min(t, 30.92))
        c = rc[:2].astype(np.float32)
        R = float(rc[2])
        if t > T_REL:
            e = float(ease_in_out(np.clip((t - T_REL) / 0.55, 0, 1)))
            c = lerp(c, self.orb_pos(t), e)
            R = lerp(R, 30.0, float(ease_in_out(np.clip((t - T_REL) / 0.45, 0, 1))))
        return c, R

    def stream_pts(self, side, t, n=220):
        c, R = self.ring_geom(t)
        th_h = self.stream_head_angle(side, t)
        s = np.linspace(0, 1, n).astype(np.float32)
        th = th_h - self.unroll(t) * s
        rho, z = self._rho(side, th, t, R)
        P = np.c_[c[0] + rho * np.cos(th), c[1] + rho * np.sin(th)].astype(np.float32)
        return P, z.astype(np.float32), s, th

    def plasma_state(self, t):
        if t < 31.20 or t > T_BOOM + 0.01:
            return None
        c, _ = self.ring_geom(t)
        born = sstep(31.22, 31.55, t)
        charge = sstep(31.55, 32.40, t)
        r = 60.0 * float(ease_out_back(np.clip((t - 31.22) / 0.4, 0, 1), 1.6)) * (1 + 0.035 * np.sin(t * 23) * charge)
        k = sstep(32.22, 32.50, t)                 # squeeze: contracts while still building
        r *= 1 - 0.40 * k
        inhale = sstep(32.40, 32.54, t)            # last beat: energy sucked in, dims to 40 %
        amp = (1.0 + 0.3 * charge + 0.6 * k * k) * (0.88 + 0.12 * flicker(t, 3.0, 2.0)) * (1 - 0.6 * inhale)
        if t > 32.565:
            r *= max(0.0, 1 - (t - 32.565) / 0.055)
        return dict(c=c, r=r, born=born, charge=charge, squeeze=k, amp=amp, inhale=inhale)

    def boom_center(self):
        c, _ = self.ring_geom(T_BOOM)
        return c

    # .............................................................. lightning
    @staticmethod
    def _bolt(pts, rng, disp=0.22, depth=4, decay=0.62):
        P = np.asarray(pts, np.float32)
        for lvl in range(depth):
            d = P[1:] - P[:-1]
            L = np.linalg.norm(d, axis=1, keepdims=True) + 1e-6
            perp = np.c_[-d[:, 1], d[:, 0]] / L
            mids = (P[1:] + P[:-1]) / 2 + perp * (rng.normal(0, disp, (len(d), 1)) * L)
            Q = np.empty((len(P) + len(mids), 2), np.float32)
            Q[0::2] = P
            Q[1::2] = mids
            P = Q
            disp *= decay
        return P

    def _draw_bolt(self, P, inten, out, core=(2.2, 2.0, 3.0), glow=None, width=1.1):
        glow = VIOLET if glow is None else np.asarray(glow, np.float32)
        n = len(P) - 1
        if n < 1:
            return
        taper = np.linspace(1.0, 0.55, n)
        col = np.asarray(core, np.float32)[None] * (inten * taper)[:, None]
        out.append((P[:-1], P[1:], width * (0.6 + 0.4 * taper), np.c_[col, np.ones(n)], 2))
        gcol = glow[None] * (inten * 0.22 * taper)[:, None]
        out.append((P[:-1:2], P[1::2][:len(P[:-1:2])], 7.0, np.c_[gcol[::2][:len(P[:-1:2])], np.ones(len(P[:-1:2]))], 0))

    def _arcs_fusion(self, S, t, out):
        if t < 29.35 or t > 31.15:
            return
        build = sstep(29.35, 30.9, t)
        slot = int(t * FPS / 2)
        rng = np.random.default_rng(slot * 7919 + 101)
        c, R = self.ring_geom(t)
        n = int(rng.poisson(0.6 + 2.4 * build))
        fade = 1 - sstep(31.0, 31.15, t)
        for a in range(min(n, 6)):
            kind = rng.random()
            inten = (1.2 + 2.2 * rng.random()) * fade * (0.5 + 0.7 * build)
            if kind < 0.55:          # crawl along the ring
                th0 = rng.uniform(0, 2 * np.pi)
                span = rng.uniform(0.5, 1.5) * rng.choice([-1, 1])
                th = th0 + np.linspace(0, span, 7)
                rr = R + rng.normal(0, 5, 7)
                base = np.c_[c[0] + rr * np.cos(th), c[1] + rr * np.sin(th)]
                P = self._bolt(base, rng, 0.16, 3)
                inten *= 0.7
            elif kind < 0.85:        # arch between the hands over the top of the ring
                hr, hl = A.hand(S.i, 'R'), A.hand(S.i, 'L')
                if hr is None or hl is None:
                    continue
                pa = hr[rng.choice([5, 6, 9, 10]), :2]
                pb = hl[rng.choice([5, 6, 9, 10]), :2]
                top = np.array([(pa[0] + pb[0]) / 2, min(pa[1], pb[1]) - rng.uniform(40, 95)])
                tt = np.linspace(0, 1, 7)[:, None]
                base = (1 - tt) ** 2 * pa + 2 * (1 - tt) * tt * top + tt ** 2 * pb
                P = self._bolt(base, rng, 0.14, 3)
            else:                    # short sparks at the finger contacts (top / bottom of the ring)
                th0 = -np.pi / 2 if rng.random() < 0.6 else np.pi / 2
                p0 = c + R * np.array([np.cos(th0), np.sin(th0)])
                p1 = p0 + rng.normal(0, 1, 2) * 30 + np.array([np.cos(th0), np.sin(th0)]) * 25
                P = self._bolt([p0, p1], rng, 0.3, 3)
            self._draw_bolt(P.astype(np.float32), inten, out)
            if rng.random() < 0.5 and len(P) > 6:   # branch
                j = rng.integers(2, len(P) - 3)
                d = P[j + 2] - P[j]
                ang = np.arctan2(d[1], d[0]) + rng.choice([-1, 1]) * rng.uniform(0.4, 1.0)
                L = rng.uniform(15, 45)
                Pb = self._bolt([P[j], P[j] + L * np.array([np.cos(ang), np.sin(ang)])], rng, 0.3, 2)
                self._draw_bolt(Pb.astype(np.float32), inten * 0.5, out, width=0.8)

    def _arcs_plasma(self, t, st, out):
        slot = int(t * FPS / 2)
        rng = np.random.default_rng(slot * 104729 + 7)
        n = int(rng.poisson(0.6 + 3.5 * st['charge'] + 2.5 * st['squeeze']))
        c, r = st['c'], max(st['r'], 8)
        dim = 1 - 0.6 * st['inhale']
        for a in range(min(n, 6)):
            ang = rng.uniform(0, 2 * np.pi)
            if np.sin(ang) > 0.35:      # never lash down into the captions band under the orb
                ang = -ang
            p0 = c + r * 0.9 * np.array([np.cos(ang), np.sin(ang)])
            inten = (1.3 + 2.0 * rng.random()) * (0.6 + 0.6 * st['charge']) * (0.45 + 0.75 * rng.random()) * dim
            if rng.random() < 0.55:     # loop back onto the orb
                ang2 = ang + rng.uniform(0.6, 1.6) * rng.choice([-1, 1])
                p2 = c + r * 0.9 * np.array([np.cos(ang2), np.sin(ang2)])
                mid = c + r * rng.uniform(1.5, 2.3) * np.array([np.cos((ang + ang2) / 2), np.sin((ang + ang2) / 2)])
                tt = np.linspace(0, 1, 5)[:, None]
                base = (1 - tt) ** 2 * p0 + 2 * (1 - tt) * tt * mid + tt ** 2 * p2
                P = self._bolt(base, rng, 0.2, 3)
                self._draw_bolt(P.astype(np.float32), inten, out)
            else:                       # lash outward: jagged, always forked
                L = rng.uniform(60, 170)
                base = [p0, c + (r + L) * np.array([np.cos(ang), np.sin(ang)]) + rng.normal(0, 18, 2)]
                P = self._bolt(base, rng, 0.35, 4, 0.55)
                self._draw_bolt(P.astype(np.float32), inten, out)
                j = int(rng.integers(len(P) // 4, max(len(P) // 4 + 1, 3 * len(P) // 4)))
                d = P[min(j + 2, len(P) - 1)] - P[j]
                ang_b = np.arctan2(d[1], d[0]) + rng.choice([-1, 1]) * rng.uniform(0.4, 1.0)
                Lb = rng.uniform(0.3, 0.6) * L
                Pb = self._bolt([P[j], P[j] + Lb * np.array([np.cos(ang_b), np.sin(ang_b)])], rng, 0.35, 3, 0.55)
                self._draw_bolt(Pb.astype(np.float32), inten * 0.55, out, width=0.8)

    # .............................................................. fusion streams
    def _parts_shed(self, t, out):
        idx, age = self.shed.alive(t)
        if not len(idx):
            return
        tb = self.shed.tb[idx]
        u = self.shed.u[idx]
        life = self.shed.life[idx]
        env = np.array([sstep(T_UNR0 + 0.05, T_UNR0 + 0.35, x) * (1 - sstep(31.3, 31.55, x)) for x in tb], np.float32)
        keep = u[:, 9] < env
        idx, age, tb, u, life = idx[keep], age[keep], tb[keep], u[keep], life[keep]
        if not len(idx):
            return
        isfire = u[:, 0] < 0.5
        P0s = np.zeros((len(idx), 2), np.float32)
        V = np.zeros((len(idx), 2), np.float32)
        for j in range(len(idx)):
            side = 'R' if isfire[j] else 'L'
            c, R = self.ring_geom(float(tb[j]))
            lam = self.unroll(float(tb[j]))
            th = self.stream_head_angle(side, float(tb[j])) - lam * u[j, 1] ** 2
            P0s[j] = c + R * np.array([np.cos(th), np.sin(th)])
            rad = np.array([np.cos(th), np.sin(th)])
            tan = np.array([-np.sin(th), np.cos(th)])
            V[j] = rad * (60 + 220 * u[j, 2]) + tan * (150 + 300 * u[j, 3])
        grav = np.where(isfire, -90.0, 110.0)

        def pos(tau):
            e = (1 - np.exp(-2.5 * tau)) / 2.5
            return P0s + V * e[:, None] + np.c_[np.zeros_like(tau), grav * tau * tau]
        k = age / life
        colf = bbody_np(1.2 - k) * 2.6
        coli = np.array([0.7, 0.9, 1.0])[None] * (1.6 + 2.0 * (0.5 + 0.5 * np.sin(age * 14 + u[:, 4] * 9)) ** 4)[:, None]
        col = np.where(isfire[:, None], colf, coli) * (1 - k[:, None]) ** 1.2 * (0.5 + 0.8 * u[:, 6:7])
        self._emit(out, pos, age, self._psize(u[:, 5], 0.55, 1.8), col)

    def _parts_charge(self, t, st, out):
        idx, age = self.charge.alive(t)
        if not len(idx):
            return
        u = self.charge.u[idx]
        life = self.charge.life[idx]
        c = st['c']
        ang = u[:, 0] * 2 * np.pi
        D0 = 90 + 200 * u[:, 1]
        spin = np.where(u[:, 5] < 0.5, -1.0, 1.0) * (1.5 + 2.0 * u[:, 2])
        speed = 1.0 + 1.0 * st['inhale']           # pulled in twice as fast on the inhale

        def pos(tau):
            s = np.clip(tau / life * speed, 0, 1)
            Dd = D0 * (1 - s) ** 1.25 + st['r'] * 0.8
            a = ang + spin * s ** 1.5
            return np.c_[c[0] + Dd * np.cos(a), c[1] + Dd * np.sin(a)]
        s = np.clip(age / life * speed, 0, 1)
        pal = np.array([[2.2, 0.9, 0.25], [0.6, 1.3, 2.4], [1.5, 1.0, 2.6]], np.float32)
        col = pal[(u[:, 3] * 3).astype(int).clip(0, 2)] * ((0.15 + 1.2 * s ** 2) * (0.6 + 0.8 * st['charge']))[:, None]
        col = col * (s < 0.995)[:, None]
        self._emit(out, pos, age, self._psize(u[:, 4], 0.6, 1.6), col, shutter=SHUTTER * 1.8)

    def _plasma_layers(self, S, t, st):
        if st['r'] < 1.0:
            return []
        c = S.tx(st['c'])
        R = st['r']
        ext = R * 3.2 + 10
        img, b = self._gl_layer(self.p_plasma, (c[0] - ext, c[1] - ext, c[0] + ext, c[1] + ext),
                                dict(uC=tuple(map(float, c)), uR=float(R), uT=float(t), uAmp=float(0.30 * st['amp'] * st['born']),
                                     uCharge=float(st['charge']), uCorona=float(0.8 + 0.6 * st['charge'])))
        if img is None:
            return []
        return [Layer('add', img[..., :3], box=b)]

    # .............................................................. detonation
    def _boom_cloud(self, S, t):
        """volumetric detonation cloud, drawn behind him: white-hot pockets inside dark,
        absorbing smoke that billows out, rolls, rises, cools and thins out."""
        tau = t - T_BOOM
        if tau < 0 or tau > 1.5:
            return []
        heat = 1.5 * np.exp(-tau / 0.33) + 0.3 * np.exp(-tau / 1.0)
        smoke = sstep(0.06, 0.50, tau)
        fade = 1 - sstep(0.55, 1.45, tau)
        if fade * max(heat, 0.3 * smoke) < 0.04:
            return []
        c = S.tx(self.boom_center())
        Rb = 70 + 400 * (1 - np.exp(-tau / 0.17)) + 45 * tau
        ext = min(Rb * 1.55, 760.0)
        n = 24 if tau < 0.6 else (16 if tau < 1.0 else 11)
        sc = 0.5
        x0, y0, x1, y1 = [int(round(v)) for v in (c[0] - ext, c[1] - ext, c[0] + ext, c[1] + ext)]
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        if x1 - x0 < 4 or y1 - y0 < 4:
            return []
        w, h = x1 - x0, y1 - y0
        ws, hs = max(2, int(w * sc)), max(2, int(h * sc))
        uni = dict(uC=(float(c[0] * sc), float(c[1] * sc)), uR=float(Rb * sc), uT=float(t), uAge=float(tau),
                   uHeat=float(heat), uSmoke=float(smoke), uFade=float(fade), uN=int(n))
        img = gl.render(self.p_boom, (ws, hs), uniforms=uni, offset=(x0 * sc, y0 * sc))
        img = cv2.blur(img, (3, 3))                         # remove the (static) dither pattern
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
        rgb = np.maximum(img[..., :3], 0)
        a = np.clip(img[..., 3], 0, 1)
        # keep the captions band (on the crown of his head) readable during the hottest part
        ys = np.arange(y0, y1, dtype=np.float32)
        band = smoothstep(650, 720, ys) * (1 - smoothstep(890, 960, ys))
        rgb *= (1 - 0.45 * band * (1 - sstep(0.45, 0.75, tau)))[:, None, None]
        return [Layer('over', rgb, a, box=(x0, y0))]

    def _parts_boom(self, t, out_front, out_back):
        """detonation debris. ~1/3 of the sparks fly toward the camera and are drawn in front
        of him (bigger, more blurred); the rest are drawn behind him (his head hides them)."""
        tau = t - T_BOOM
        if tau < 0:
            return
        c = self.boom_center()
        for B, isfire in ((self.bfire, True), (self.bice, False)):
            idx, age = B.alive(t)
            if not len(idx):
                continue
            u = B.u[idx]
            life = B.life[idx]
            # half of the long-lived sparks fade out early: no "rain curtain" later on
            cull = np.where(u[:, 9] < 0.5, 1.0, 1.0 - smoothstep(0.35, 0.60, age))
            front = u[:, 8] < 0.33
            ang = u[:, 0] * 2 * np.pi
            sp = (260 + 3000 * u[:, 1] ** 2.5) * (1.0 if isfire else 0.85) * np.where(front, 1.2, 1.0)
            dirv = np.c_[np.cos(ang), np.sin(ang) * 0.9]
            # born throughout the young fireball, not at one point (no hollow spark ring)
            a2 = u[:, 6] * 2 * np.pi
            off = np.c_[np.cos(a2), np.sin(a2)] * (75 * np.sqrt(np.mod(u[:, 6] * 13.7 + u[:, 1] * 5.3, 1.0)))[:, None]
            drag = (1.4 + 3.0 * u[:, 7]) if isfire else (1.2 + 2.4 * u[:, 7])
            g = 380.0 if isfire else 520.0
            k = age / life
            if isfire:
                fl = 0.75 + 0.25 * np.sin(age * (9 + 8 * u[:, 4]) + 6.28 * u[:, 5])
                col = bbody_np(1.30 - 1.15 * k) * (2.2 + 2.6 * u[:, 2:3] ** 2) * ((1 - k) ** 1.3 * fl)[:, None]
                rad = self._psize(u[:, 3], 0.6, 2.4)
            else:
                tw = 0.5 + 0.5 * np.sin(age * (8 + 7 * u[:, 4]) + 6.28 * u[:, 5])
                col = np.array([0.75, 0.9, 1.05])[None] * ((1.2 + 3.5 * tw ** 4) * (1 - k) ** 1.2)[:, None]
                rad = self._psize(u[:, 3], 0.6, 2.6)
            col = col * cull[:, None]
            for m, out in ((front, out_front), (~front, out_back)):
                if not m.any():
                    continue
                dm, sm, drm, om = dirv[m], sp[m], drag[m], off[m]

                def pos(tau_, dm=dm, sm=sm, drm=drm, om=om):
                    e = (1 - np.exp(-drm * tau_)) / drm
                    return c + om + dm * (sm * e)[:, None] + np.c_[np.zeros_like(tau_), g * tau_ * tau_]
                if out is out_front:
                    self._emit(out, pos, age[m], rad[m] * (1.5 + 1.8 * age[m]), col[m] * 0.85, shutter=SHUTTER * 1.5)
                else:
                    self._emit(out, pos, age[m], rad[m], col[m])
        # aftermath: embers + snow drifting down (some in front, most behind)
        idx, age = self.after.alive(t)
        if len(idx):
            u = self.after.u[idx]
            life = self.after.life[idx]
            ang = u[:, 0] * 2 * np.pi
            rad0 = 520 * np.sqrt(u[:, 1])
            p0 = c + np.c_[np.cos(ang) * rad0 * 1.1, np.sin(ang) * rad0 * 0.75 + 160]
            isem = u[:, 2] < 0.5
            k = age / life
            envl = np.sin(np.pi * np.clip(k, 0, 1)) * (1 - sstep(T_END - 0.55, T_END, t))
            fl = 0.6 + 0.4 * np.sin(age * (4 + 6 * u[:, 7]) + u[:, 8] * 6.28)
            ce = bbody_np(0.55 + 0.4 * u[:, 7]) * 1.8 * fl[:, None]
            cs = np.array([0.75, 0.85, 1.0])[None] * (0.55 + 1.6 * fl[:, None] ** 6)
            col = np.where(isem[:, None], ce, cs) * envl[:, None]
            rad = np.where(isem, 0.7, 0.9) + 1.3 * u[:, 9] ** 3
            fr = u[:, 9] > 0.7
            for m, out in ((fr, out_front), (~fr, out_back)):
                if not m.any():
                    continue
                um, pm, im = u[m], p0[m], isem[m]

                def pos(tau_, um=um, pm=pm, im=im):
                    vy = np.where(im, 18 + 40 * um[:, 3], 45 + 70 * um[:, 3])
                    x = pm[:, 0] + 22 * np.sin(tau_ * (1.2 + 1.8 * um[:, 4]) + 6.28 * um[:, 5]) + (um[:, 6] - 0.5) * 40 * tau_
                    y = pm[:, 1] + vy * tau_
                    return np.c_[x, y]
                self._emit(out, pos, age[m], rad[m] * (1.3 if out is out_front else 1.0), col[m])

    def _shockwave(self, S, img, t):
        tau = t - T_BOOM
        if tau < 0 or tau > 0.9:
            return img
        c = S.tx(self.boom_center())
        Rs = 30 + 1650 * (1 - np.exp(-tau / 0.24))
        wdt = 28 + 60 * tau
        amp = 28 * np.exp(-tau / 0.30) * sstep(0.0, 0.03, tau)
        q = 4
        h, w = H // q, W // q
        X, Y = np.meshgrid(np.arange(w, dtype=np.float32) * q + q / 2, np.arange(h, dtype=np.float32) * q + q / 2)
        dx, dy = X - c[0], Y - c[1]
        d = np.sqrt(dx * dx + dy * dy) + 1e-3
        x = (d - Rs) / wdt
        prof = -x * np.exp(-x * x * 1.5) * 2.0          # lens profile (derivative of a gaussian)
        disp = prof * amp
        ux, uy = dx / d, dy / d
        mx = cv2.resize((ux * disp).astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
        my = cv2.resize((uy * disp).astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
        XX, YY = core.grid()
        out = np.empty_like(img)
        for ch, kk in enumerate((1.12, 1.0, 0.88)):      # chromatic split
            out[..., ch] = cv2.remap(np.ascontiguousarray(img[..., ch]), (XX + mx * kk).astype(np.float32), (YY + my * kk).astype(np.float32),
                                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        # thin bright chromatic leading edge (red outside, blue inside), broken up angularly
        ei = 1.1 * np.exp(-tau / 0.16) * sstep(0.0, 0.02, tau)
        if ei > 0.01:
            q2 = 2
            X2, Y2 = np.meshgrid(np.arange(W // q2, dtype=np.float32) * q2 + 1, np.arange(H // q2, dtype=np.float32) * q2 + 1)
            dx2, dy2 = X2 - c[0], Y2 - c[1]
            d2 = np.sqrt(dx2 * dx2 + dy2 * dy2)
            ang = np.arctan2(dy2, dx2)
            brk = np.clip(0.55 + 0.45 * np.sin(ang * 7 + 1.3) * np.sin(ang * 13 - tau * 9 + 0.4)
                          + 0.25 * np.sin(ang * 31 + 2.0), 0, 1.2)
            ew = 3.5 + 7 * tau
            lead = Rs + 0.55 * wdt
            edge = np.empty((H // q2, W // q2, 3), np.float32)
            for ch, off in enumerate((3.5, 0.0, -3.5)):
                edge[..., ch] = np.exp(-((d2 - lead - off * (1 + 2 * tau)) / ew) ** 2)
            edge *= (brk * ei)[..., None] * np.array([0.9, 0.85, 1.05], np.float32)
            edge += np.exp(-((d2 - Rs) / (wdt * 0.9)) ** 2)[..., None] * (0.06 * ei)       # faint inner sheen
            out += cv2.resize(edge, (W, H), interpolation=cv2.INTER_LINEAR)
        return out

    # .............................................................. particles
    def _build_particles(self):
        rng = self.rng
        self.emb = Births(rng, T_FIRE - 0.05, 29.4, 55, (0.7, 1.7))
        self.ign = Births(rng, T_FIRE, T_FIRE + 0.06, 1500, (0.35, 0.9))
        self.wispR = Births(rng, T_START, T_FIRE, 64, (0.7, 1.3))
        self.wispL = Births(rng, T_START, T_ICE, 36, (0.7, 1.3))
        self.snow = Births(rng, 26.30, 29.40, 34, (1.1, 2.2))
        self.iceburst = Births(rng, T_ICE, T_ICE + 0.05, 1100, (0.5, 1.1))
        self.shed = Births(rng, 29.15, 31.50, 260, (0.30, 0.85))
        self.charge = Births(rng, 31.45, T_BOOM, 170, (0.55, 1.05))
        self.bfire = Births(rng, T_BOOM, T_BOOM + 0.06, 6000, (0.35, 1.5))
        self.bice = Births(rng, T_BOOM, T_BOOM + 0.06, 3500, (0.5, 1.6))
        self.after = Births(rng, T_BOOM + 0.05, T_END - 0.25, 150, (1.2, 2.4))

    def _parts_fire(self, t, out):
        """embers rising from the fire (world space after birth) + ignition burst."""
        idx, age = self.emb.alive(t)
        if len(idx):
            tb = self.emb.tb[idx]
            u = self.emb.u[idx]
            life = self.emb.life[idx]
            amt = np.array([self._fire_amount(x) for x in tb], np.float32)
            keep = u[:, 0] < amt
            idx, age, tb, u, life = idx[keep], age[keep], tb[keep], u[keep], life[keep]
        if len(idx):
            c0 = np.array([self.fire_state(float(x))['c'] for x in tb], np.float32)
            v0 = self._interp(self.anchor_vel['R'], tb)
            r = 70.0
            ang = -np.pi / 2 + (u[:, 1] - 0.5) * 2.4
            rad = r * (0.25 + 0.75 * u[:, 2])
            p0 = c0 + np.c_[np.cos(ang), np.sin(ang)] * rad[:, None]

            def pos(tau):
                vx = (u[:, 3] - 0.5) * 110 + v0[:, 0] * 0.35
                vy = -(140 + 220 * u[:, 4]) + v0[:, 1] * 0.25
                drag = 1.3
                e = (1 - np.exp(-drag * tau)) / drag
                x = p0[:, 0] + vx * e + 22 * np.sin(tau * (3 + 4 * u[:, 5]) + 6.28 * u[:, 6]) * tau
                y = p0[:, 1] + vy * e - 40 * tau * tau
                return np.c_[x, y]
            k = age / life
            temp = (1.10 - 1.0 * k) * (0.75 + 0.35 * u[:, 7])
            fl = 0.6 + 0.4 * np.sin(age * (6 + 9 * u[:, 9]) + u[:, 9] * 20)
            sz = np.mod(u[:, 8] * 7.13 + u[:, 2] * 3.7, 1.0)
            col = bbody_np(temp) * (1.8 + 2.6 * u[:, 8:9] ** 2) * (1 - k[:, None] ** 2) * fl[:, None]
            self._emit(out, pos, age, self._psize(sz, 0.55, 2.3), col)
        # ignition burst sparks (radial, fast, drag + gravity)
        idx, age = self.ign.alive(t)
        if len(idx):
            u = self.ign.u[idx]
            life = self.ign.life[idx]
            c0 = self._interp(self.anchor['R'], T_FIRE)
            ang = u[:, 0] * 2 * np.pi
            sp = 380 + 1500 * u[:, 1] ** 2.2
            dirv = np.c_[np.cos(ang), np.sin(ang) * 0.85 - 0.25]

            def pos(tau):
                drag = 3.5
                e = (1 - np.exp(-drag * tau)) / drag
                return c0 + dirv * (sp * e)[:, None] + np.c_[np.zeros_like(tau), 260 * tau * tau]
            k = age / life
            fl = 0.7 + 0.3 * np.sin(age * (20 + 15 * u[:, 4]) + 6.28 * u[:, 5])
            col = bbody_np(1.25 - 1.1 * k) * (2.6 + 3.0 * u[:, 2:3] ** 2) * ((1 - k) * fl)[:, None]
            self._emit(out, pos, age, self._psize(u[:, 3], 0.55, 2.2), col)

    def _parts_ice(self, t, out):
        idx, age = self.snow.alive(t)
        if len(idx):
            tb = self.snow.tb[idx]
            u = self.snow.u[idx]
            life = self.snow.life[idx]
            amt = np.array([(lambda st: 0.0 if st is None else st['amount'])(self.ice_state(float(x))) for x in tb], np.float32)
            keep = u[:, 0] < amt
            idx, age, tb, u, life = idx[keep], age[keep], tb[keep], u[keep], life[keep]
        if len(idx):
            c0 = np.array([self.ice_state(float(x))['c'] for x in tb], np.float32)
            ang = u[:, 1] * 2 * np.pi
            rad = 60 * (0.3 + 0.9 * u[:, 2])
            p0 = c0 + np.c_[np.cos(ang), np.sin(ang) * 0.8] * rad[:, None]

            def pos(tau):
                x = p0[:, 0] + (u[:, 3] - 0.5) * 70 * tau + 14 * np.sin(tau * (2 + 3 * u[:, 4]) + 6.28 * u[:, 5])
                y = p0[:, 1] + (35 + 80 * u[:, 6]) * tau + 25 * tau * tau
                return np.c_[x, y]
            k = age / life
            tw = 0.5 + 0.5 * np.sin(age * (5 + 8 * u[:, 7]) + 6.28 * u[:, 8])
            inten = (0.45 + 2.6 * tw ** 6) * np.sin(np.pi * np.clip(k, 0, 1)) ** 0.7
            col = np.array([0.72, 0.88, 1.0])[None] * inten[:, None]
            self._emit(out, pos, age, self._psize(u[:, 9], 0.55, 2.0), col)
        idx, age = self.iceburst.alive(t)
        if len(idx):
            u = self.iceburst.u[idx]
            life = self.iceburst.life[idx]
            c0 = self._interp(self.anchor['L'], T_ICE)
            ang = u[:, 0] * 2 * np.pi
            sp = 260 + 950 * u[:, 1] ** 2
            dirv = np.c_[np.cos(ang), np.sin(ang)]

            def pos(tau):
                drag = 4.5
                e = (1 - np.exp(-drag * tau)) / drag
                return c0 + dirv * (sp * e)[:, None] + np.c_[np.zeros_like(tau), 90 * tau * tau]
            k = age / life
            tw = 0.5 + 0.5 * np.sin(age * (10 + 8 * u[:, 4]) + 6.28 * u[:, 2])
            col = np.array([0.75, 0.9, 1.0])[None] * ((1.3 + 3.0 * tw ** 4) * (1 - k) ** 1.5)[:, None]
            self._emit(out, pos, age, self._psize(u[:, 3], 0.55, 2.0), col)

    def _fire_amount(self, t):
        st = self.fire_state(float(t))
        if st is None or st['kind'] != 'fire':
            return 0.0
        return float(st['amount'])

    def _antic_glows(self, t):
        """soft energy glows gathering at each hand before its element appears:
        (pos, colour, strength, hand_below_frame)"""
        res = []
        for side, t_on, col in (('R', T_FIRE, FIRE_COL), ('L', T_ICE, ICE_COL)):
            if t > t_on:
                continue
            k = sstep(T_START, T_START + 0.6, t) * (0.6 + 0.4 * sstep(t_on - 1.6, t_on, t)) * self._side_ramp(side, t)
            below, c = self._below(side, t)
            c = np.array([c[0], min(c[1], H - 12)], np.float32)
            res.append((c, col, k * (0.85 + 0.15 * flicker(t, 11.0 if side == 'R' else 4.0, 1.4)), below))
        return res

    def _parts_wisps(self, t, out):
        """anticipation: energy wisps spiralling in to each hand. While the hand is still under the
        frame they converge on a point just below the bottom edge (light pools there)."""
        for side, B, t_on, col0 in (('R', self.wispR, T_FIRE, FIRE_COL), ('L', self.wispL, T_ICE, ICE_COL)):
            if t > t_on + 0.1:
                continue
            idx, age = B.alive(t)
            if not len(idx):
                continue
            u = B.u[idx]
            life = B.life[idx]
            tb = B.tb[idx]
            ramp = (0.5 + 0.5 * np.clip((tb - B.t0) / (t_on - B.t0), 0, 1) ** 1.5) * self._side_ramp(side, t)
            keep = u[:, 0] < ramp
            idx, age, u, life, tb = idx[keep], age[keep], u[keep], life[keep], tb[keep]
            if not len(idx):
                continue
            below, c = self._below(side, t)
            base = col0[None] * 1.2 + 0.25
            spin = 1 if side == 'R' else -1
            s = age / life
            inten = (np.sin(np.pi * s) ** 1.5) * (0.45 + 1.1 * s) * (0.8 + 0.8 * u[:, 4])
            if below:      # converge on the hand just under the frame edge (wider, brighter)
                inten = inten * 1.5
                c = np.array([np.clip(c[0], 80, W - 80), H + 25.0], np.float32)
                R0 = 150 + 320 * u[:, 1]
            else:
                c = np.array([c[0], min(c[1], H - 30)], np.float32)
                R0 = 110 + 170 * u[:, 1]

            def pos(tau, below=below, c=c, R0=R0):
                ss = np.clip(tau / life, 0, 1)
                R = R0 * (1 - ss) ** 1.6 + 6
                if below:     # upper half-plane only: they curl down into the frame edge
                    th = u[:, 2] * np.pi + spin * (1.2 + 1.0 * u[:, 3]) * ss ** 1.3
                    return np.c_[c[0] + R * np.cos(th), c[1] - np.abs(np.sin(th)) * R * 0.85]
                th = u[:, 2] * 6.283 + spin * (2.5 + 2 * u[:, 3]) * ss ** 1.3
                return np.c_[c[0] + R * np.cos(th), c[1] + R * np.sin(th) * 0.85]
            nseg = 5
            for j in range(nseg):
                a0 = np.maximum(age - 0.018 * j, 0)
                a1 = np.maximum(age - 0.018 * (j + 1), 0)
                fade = (1 - j / nseg) ** 1.5
                col = base * (inten * fade)[:, None] * 0.9
                out.append((pos(a0), pos(a1), (0.7 + 0.6 * u[:, 5]) * (1 - 0.5 * j / nseg), np.c_[col, np.ones(len(col))], 2))
                if j == 0:
                    out.append((pos(a0), pos(a1), 4.0 + 3 * u[:, 5], np.c_[base * (inten * 0.05)[:, None], np.ones(len(col))], 0))

    # .............................................................. haze
    def _build_haze_tex(self):
        rng = np.random.default_rng(77)
        n = 256
        tex = np.zeros((n, n, 2), np.float32)
        for ch in range(2):
            z = rng.standard_normal((n, n)).astype(np.float32)
            Fz = np.fft.fft2(z)
            fy = np.fft.fftfreq(n)[:, None]
            fx = np.fft.fftfreq(n)[None, :]
            f = np.sqrt(fx * fx + fy * fy)
            filt = np.exp(-(f / 0.035) ** 2) * (f > 0)
            v = np.real(np.fft.ifft2(Fz * filt))
            tex[..., ch] = v / (v.std() + 1e-6)
        self.haze_tex = tex

    def _haze(self, S, c, r, strength, t):
        """heat-shimmer displacement above a flame (S.distort crop): bends the studio behind the
        flame, kept off the person so his cheek/ear do not wobble."""
        w = int(4.4 * r)
        h = int(6.5 * r)
        x0 = int(c[0] - w / 2)
        y0 = int(c[1] - 5.2 * r)
        X, Y, dx, dy = self._haze_field(w, h, x0, y0, t)
        u = (X - w / 2) / (w / 2)
        v = (Y - h * (5.2 / 6.5)) / r          # 0 at centre, negative upward
        mask = np.exp(-u * u * 2.5) * np.clip(-v / 1.0, 0, 1) * np.exp(-np.maximum(-v - 1.5, 0) / 2.2)
        mask *= np.clip((v + 4.8) * 0.8, 0, 1)
        fx0, fy0, fx1, fy1 = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
        if fx1 > fx0 and fy1 > fy0:
            al = np.zeros((h, w), np.float32)
            al[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0] = S.alpha[fy0:fy1, fx0:fx1]
            mask = mask * (1 - 0.85 * cv2.GaussianBlur(al, (0, 0), 6))
        amp = 4.0 * strength
        S.distort.append((x0, y0, (dx * mask * amp).astype(np.float32), (dy * mask * amp * 1.4).astype(np.float32)))

    def _haze_field(self, w, h, x0, y0, t):
        n = self.haze_tex.shape[0]
        X, Y = np.meshgrid(np.arange(w), np.arange(h))
        sx = (X * 0.9 + x0).astype(int) % n
        sy = (Y * 0.9 + y0 + int(t * 140)).astype(int) % n
        return X, Y, self.haze_tex[sy, sx, 0], self.haze_tex[(sy + 97) % n, (sx + 31) % n, 1]

    def _haze_remap(self, img, off, c, r, strength, t):
        """shimmer the flame crop: strong in the plume above the ball, faint in the ball."""
        h, w = img.shape[:2]
        if strength < 0.02 or r < 4:
            return img
        X, Y, dx, dy = self._haze_field(w, h, off[0], off[1], t)
        v = (Y + off[1] - c[1]) / r                  # 0 at the centre, negative upward
        m = 0.25 + 0.75 * np.clip(-v / 1.6, 0, 1)
        amp = 4.5 * strength
        mx = (X + dx * m * amp).astype(np.float32)
        my = (Y + dy * m * amp * 1.3).astype(np.float32)
        return cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def _hand_mask(self, S, side, box):
        """soft coverage of one hand (landmark skeleton ∩ matte) inside box=(x0,y0,x1,y1)."""
        x0, y0, x1, y1 = box
        m = np.zeros((y1 - y0, x1 - x0), np.float32)
        hd = A.hand(S.i, side)
        if hd is None:
            return m
        pts = S.tx(hd[:, :2]) - np.array([x0, y0], np.float32)
        hs = float(np.clip(A.hand_size(S.i, side), 90, 260))
        th = max(3, int(0.12 * hs))
        hull = cv2.convexHull(pts[[0, 1, 2, 5, 9, 13, 17]].astype(np.int32))
        cv2.fillConvexPoly(m, hull, 1.0)
        for ch in ((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)):
            cv2.polylines(m, [pts[list(ch)].astype(np.int32)], False, 1.0, th)
        m = cv2.GaussianBlur(m, (0, 0), 2.5)
        return np.ascontiguousarray(m * S.alpha[y0:y1, x0:x1])

    # .............................................................. render helpers
    def _gl_layer(self, prog, box, uniforms, scale=1.0):
        """render prog into frame-space crop box=(x0,y0,x1,y1) (clipped), optional sub-res."""
        x0, y0, x1, y1 = [int(round(v)) for v in box]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(W, x1), min(H, y1)
        if x1 - x0 < 2 or y1 - y0 < 2:
            return None, (x0, y0)
        w, h = x1 - x0, y1 - y0
        if scale != 1.0:
            ws, hs = max(2, int(w * scale)), max(2, int(h * scale))
            uni = dict(uniforms)
            # render a scaled copy: coordinates shrink with the target
            if 'uC' in uni:
                uni['uC'] = (uni['uC'][0] * scale, uni['uC'][1] * scale)
            if 'uR' in uni:
                uni['uR'] = uni['uR'] * scale
            img = gl.render(prog, (ws, hs), uniforms=uni, offset=(x0 * scale, y0 * scale))
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
            return img, (x0, y0)
        img = gl.render(prog, (w, h), uniforms=uniforms, offset=(x0, y0))
        return img, (x0, y0)

    def _splat(self, batches, pad=12):
        """render a list of sprite batches into one crop; returns Layer list (add)."""
        layers = []
        allp = [np.r_[b[0], b[1]] for b in batches if len(b[0])]
        if not allp:
            return layers
        P = np.concatenate(allp, 0)
        P = P[np.isfinite(P).all(1)]
        if not len(P):
            return layers
        x0 = int(max(0, np.floor(P[:, 0].min()) - pad))
        y0 = int(max(0, np.floor(P[:, 1].min()) - pad))
        x1 = int(min(W, np.ceil(P[:, 0].max()) + pad))
        y1 = int(min(H, np.ceil(P[:, 1].max()) + pad))
        if x1 - x0 < 2 or y1 - y0 < 2:
            return layers
        acc = None
        for shape in (0, 1, 2):
            bs = [b for b in batches if b[4] == shape and len(b[0])]
            if not bs:
                continue
            p0 = np.concatenate([b[0] for b in bs], 0)
            p1 = np.concatenate([b[1] for b in bs], 0)
            rad = np.concatenate([np.broadcast_to(np.asarray(b[2], np.float32), (len(b[0]),)) for b in bs], 0)
            col = np.concatenate([b[3] for b in bs], 0)
            im = gl.sprites((x1 - x0, y1 - y0), p0, p1, rad, col, offset=(x0, y0), shape=shape)
            acc = im if acc is None else acc + im
        if acc is not None:
            layers.append(Layer('add', acc[..., :3], box=(x0, y0)))
        return layers

    @staticmethod
    def _emit(out, pos, age, rad, col, shutter=SHUTTER, shape=2):
        """Motion-blurred particle sprites: the path over the shutter interval [age - shutter, age]
        is drawn as 3 tapered sub-segments (head bright + wide, tail dim + thin). Slow particles
        collapse to small round glowing points, only fast ones streak. Long streaks are dimmed a
        little (their energy is spread along the path)."""
        n = len(age)
        if n == 0:
            return
        sh = np.broadcast_to(np.asarray(shutter, np.float32), (n,))
        P = [pos(np.maximum(age - sh * f, 0)) for f in (0.0, 1 / 3, 2 / 3, 1.0)]
        rad = np.broadcast_to(np.asarray(rad, np.float32), (n,))
        col = np.asarray(col, np.float32).reshape(n, 3)
        L = np.hypot(P[0][:, 0] - P[3][:, 0], P[0][:, 1] - P[3][:, 1])
        col = col * np.clip((12.0 / (L + 1e-3)) ** 0.3, 0.45, 1.0)[:, None]
        one = np.ones((n, 1), np.float32)
        for j, (ki, kr) in enumerate(((1.0, 1.0), (0.5, 0.75), (0.2, 0.5))):
            out.append((P[j], P[j + 1], rad * kr, np.c_[col * ki, one], shape))

    @staticmethod
    def _psize(u, lo=0.55, hi=2.4, p=3.5):
        """power-law sprite radius: many tiny, few large."""
        return lo + (hi - lo) * np.asarray(u, np.float32) ** p

    def unroll(self, t):
        """angular length (rad) of the unrolled strands."""
        return 2 * np.pi * 0.97 * sstep(T_UNR0, T_UNR1, t)

    def _rho(self, side, th, t, R):
        """radius of strand `side` at absolute ring angle th. Different wobble frequencies per
        strand (crossings travel round the ring) plus a slow organic radial noise."""
        A = 11.0 * R / 90.0
        if side == 'R':
            fw, ft, ph, sd = 6.0, 7.0, 0.0, 0.0
        else:
            fw, ft, ph, sd = 7.0, -5.0, np.pi, 2.3
        wob = fw * th + ft * t + ph
        nz = 0.35 * (0.6 * np.sin(3 * th + 1.3 * t + sd) + 0.4 * np.sin(5.3 * th - 2.1 * t + 2.0 * sd))
        return R + A * (np.cos(wob) + nz), np.sin(wob)

    def _streams_layer(self, S, t):
        """the intertwined fire + ice helix strands orbiting the finger ring, their glow band,
        the crossing flares and the violet build-up halo (one crop layer)."""
        if t < T_UNR0 or t > 31.62:
            return []
        env = sstep(T_UNR0, T_UNR0 + 0.30, t) * (1 - sstep(31.32, 31.60, t))
        if env <= 0.002:
            return []
        build = sstep(29.4, 31.0, t)
        rel = sstep(T_REL, 31.4, t)
        c, R = self.ring_geom(t)
        c = S.tx(c)
        cores, sheath = [], []
        for side, cc, cg, sd in (('R', (1.9, 0.62, 0.13), (1.0, 0.30, 0.05), 0.0),
                                 ('L', (0.32, 1.0, 2.0), (0.10, 0.42, 1.0), 2.0)):
            P, z, s, th = self.stream_pts(side, t, 260)
            P = P + S.person_offset
            cc = np.asarray(cc, np.float32) * (1 - 0.12 * build) + np.array([1.5, 1.4, 1.6], np.float32) * 0.12 * build
            I = (0.55 + 0.9 * np.exp(-s * 5)) * (0.72 + 0.35 * z) * (1 - 0.5 * s ** 2)
            I *= 0.78 + 0.22 * np.sin(s * 23 + t * 6.0 + sd * 3) * np.sin(s * 7 - t * 2.3 + sd)
            I += 0.6 * np.maximum(0, np.sin(2 * np.pi * (s * 4 - t * 3.2 + 0.37 * sd))) ** 8
            I *= env * (1 + 0.25 * build + 0.4 * rel) * (0.92 + 0.08 * flicker(t, 3 + sd, 1.3))
            idx = np.nonzero(I[:-1] > 0.01)[0]
            if not len(idx):
                continue
            w = (1.0 + 0.9 * (z[idx] + 1) / 2) * (0.85 + 0.3 * np.sin(s[idx] * 11 + t * 4 + sd))
            one = np.ones((len(idx), 1), np.float32)
            cores.append((P[idx], P[idx + 1], w, np.c_[cc[None] * I[idx, None], one], 2))
            sheath.append((P[idx], P[idx + 1], w * 3.0, np.c_[np.asarray(cg)[None] * (I[idx, None] * 0.10), one], 0))
            # the orb arrives as a bright comet head that leads its strand round the ring
            hk = sstep(T_UNR0, T_UNR0 + 0.12, t) * (1 - sstep(29.25, 29.7, t)) * env
            if hk > 0.01:
                ph = P[:1]
                cores.append((ph, ph, 4.0, np.c_[cc[None] * 1.6 * hk, np.ones((1, 1))], 2))
                sheath.append((ph, ph, 15.0, np.c_[np.asarray(cg)[None] * 0.35 * hk, np.ones((1, 1))], 0))
        # crossings of the two strands (both unrolled there): violet-white flares
        lam = self.unroll(t)
        thg = np.linspace(0, 2 * np.pi, 540, endpoint=False)
        rR, _ = self._rho('R', thg, t, R)
        rL, _ = self._rho('L', thg, t, R)
        cov = np.ones_like(thg, bool)
        for side in 'RL':
            cov &= np.mod(self.stream_head_angle(side, t) - thg, 2 * np.pi) < lam
        dd = rR - rL
        xi = np.nonzero((np.sign(dd) != np.sign(np.roll(dd, -1))) & cov & np.roll(cov, -1))[0]
        if len(xi):
            thc = thg[xi]
            rc_ = 0.5 * (rR[xi] + rL[xi])
            Pc = np.c_[c[0] + rc_ * np.cos(thc), c[1] + rc_ * np.sin(thc)].astype(np.float32)
            fl = (0.45 + 0.8 * build) * env * (0.7 + 0.3 * np.sin(t * 17 + thc * 5))
            one = np.ones((len(Pc), 1), np.float32)
            cores.append((Pc, Pc, 2.6, np.c_[np.array([1.7, 1.4, 2.5])[None] * fl[:, None], one], 2))
            sheath.append((Pc, Pc, 11.0, np.c_[VIOLET[None] * fl[:, None] * 0.10, one], 0))
        m = int(R + 70)
        x0, y0 = int(c[0] - m), int(c[1] - m)
        box = (x0, y0, x0 + 2 * m, y0 + 2 * m)
        core_img = self._splat_box(cores, box)
        if core_img is None:
            return []
        sh_img = self._splat_box(sheath, box)
        img = core_img + (sh_img if sh_img is not None else 0)
        # soft glow band (blurred copy of the strands at low res) + violet build-up halo
        h, w = img.shape[:2]
        lo = cv2.resize(core_img, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
        g = cv2.GaussianBlur(lo, (0, 0), 2.0) * 0.40 + cv2.GaussianBlur(lo, (0, 0), 5.0) * 0.35
        if build > 0.01:
            ring = np.zeros(lo.shape[:2], np.float32)
            cv2.circle(ring, (int(round((c[0] - x0) / 4)), int(round((c[1] - y0) / 4))), int(round(R / 4)), 1.0, 2)
            ring = cv2.GaussianBlur(ring, (0, 0), 3.5)
            g += ring[..., None] * VIOLET[None, None] * (0.28 * build * env * (1 + 0.5 * rel))
        img = img + cv2.resize(g, (w, h), interpolation=cv2.INTER_LINEAR)
        return [Layer('add', img, box=(x0, y0))]

    def _splat_box(self, batches, box):
        """additive sprite render of batches into a fixed crop box -> rgb or None."""
        bs = [b for b in batches if len(b[0])]
        if not bs:
            return None
        x0, y0, x1, y1 = box
        acc = None
        for shape in (0, 1, 2):
            bb = [b for b in bs if b[4] == shape]
            if not bb:
                continue
            p0 = np.concatenate([b[0] for b in bb], 0)
            p1 = np.concatenate([b[1] for b in bb], 0)
            rad = np.concatenate([np.broadcast_to(np.asarray(b[2], np.float32), (len(b[0]),)) for b in bb], 0)
            col = np.concatenate([b[3] for b in bb], 0)
            im = gl.sprites((x1 - x0, y1 - y0), p0, p1, rad, col, offset=(x0, y0), shape=shape)
            acc = im if acc is None else acc + im
        return None if acc is None else acc[..., :3]

    def _boom_flash_bg(self, S, t):
        """the detonation flash on the environment behind him (he reads as a silhouette)."""
        tau = t - T_BOOM
        if tau < 0 or tau > 0.25:
            return []
        f = np.exp(-max(tau - 0.5 / FPS, 0) / 0.022)
        if f < 0.01:
            return []
        c = S.tx(self.boom_center())
        q = 8
        X, Y = np.meshgrid(np.arange(W // q, dtype=np.float32) * q, np.arange(H // q, dtype=np.float32) * q)
        d2 = ((X - c[0]) ** 2 + (Y - c[1]) ** 2) / (280.0 ** 2)
        m = (0.04 + 2.4 / (1 + d2) ** 1.5) * f
        m = cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
        return [Layer('add', m[..., None] * np.array([1.0, 0.9, 0.95], np.float32))]

    def _side_ramp(self, side, t):
        """1 once the hand is tracked in frame; the ice side idles at 30 % before that."""
        if side == 'R':
            return 1.0
        return 0.3 + 0.7 * sstep(self.t_ldet, self.t_ldet + 0.8, t)

    def _below(self, side, t):
        """hand still under the frame (or not yet tracked: the pose fallback is unreliable)"""
        c = self._interp(self.anchor[side], t)
        det = t >= (self.t_rdet if side == 'R' else self.t_ldet)
        return (c[1] > H - 40) or not det, c

    # .............................................................. hooks
    def active(self, t):
        return T_START - 0.02 <= t <= T_END + 0.02

    def lights(self, S):
        """Interactive light. Rim terms are kept low on the big lights (the compositor adds rim
        flat on top of the colour, which would paint untextured bands over hair/skin); the light
        is carried by the multiplicative diffuse term. Strong rim only on the tiny palm lights."""
        t = S.t
        L = []
        for c, col, k, below in self._antic_glows(t):
            L.append(Light(S.tx(c), col, (0.45 if not below else 0.75) * k, radius=150 if not below else 260, rim=0.45, diffuse=0.8, bg=0.02))
        st = self.fire_state(t)
        if st is not None:
            c = S.tx(st['c'])
            fl = 1.0 + 0.16 * flicker(t, 1.0) + 0.06 * flicker(t, 4.0, 2.3)
            if st['kind'] == 'seed':
                inten = 0.5 * st['amount']
            else:
                inten = (1.35 * st['amount'] * min(st['grow'], 1.2) + 2.2 * np.exp(-(t - T_FIRE) / 0.10)) * fl
            L.append(Light(c + np.array([0, -0.3 * st['r']]), FIRE_COL, inten, radius=240, rim=0.5, diffuse=1.1, bg=0.05))
            # close, hot light on the palm itself
            L.append(Light(c + np.array([0, 0.6 * st['r']]), np.array([1.0, 0.55, 0.25]), inten * 0.9 * (1 - st.get('fly', 0.0)),
                           radius=90, rim=0.9, diffuse=1.2, bg=0.0))
        ist = self.ice_state(t)
        if ist is not None:
            c = S.tx(ist['c'])
            inten = 1.05 * ist['amount'] * ist['glow'] * (0.6 + 0.4 * ist['scale']) + 2.2 * ist['flash']
            L.append(Light(c, ICE_COL, inten, radius=230, rim=0.6, diffuse=1.0, bg=0.04))
            L.append(Light(c + np.array([0, 0.7 * ist['r']]), np.array([0.55, 0.8, 1.0]), inten * 0.7, radius=90, rim=0.8, diffuse=1.0, bg=0.0))
        if T_UNR0 <= t <= 31.6:
            c, R = self.ring_geom(t)
            c = S.tx(c)
            env = sstep(T_UNR0, T_UNR0 + 0.35, t) * (1 - sstep(31.3, 31.6, t))
            build = sstep(29.4, 31.0, t)
            fl = 1 + 0.12 * flicker(t, 2.0, 1.5)
            L.append(Light(c + np.array([-R, 0]), FIRE_COL, 0.85 * env * fl, radius=170, rim=0.35, diffuse=0.7, bg=0.03))
            L.append(Light(c + np.array([R, 0]), ICE_COL, 0.85 * env * fl, radius=170, rim=0.35, diffuse=0.7, bg=0.03))
            L.append(Light(c, VIOLET, (0.15 + 1.0 * build) * env * fl, radius=200, rim=0.3, diffuse=0.55, bg=0.03))
            # lightning strobes
            slot = int(t * FPS / 2)
            if build > 0.05 and np.random.default_rng(slot * 7919 + 101).random() < 0.55:
                L.append(Light(c + np.array([0, -R]), np.array([0.8, 0.7, 1.0]), 0.6 * build, radius=220, rim=0.4, diffuse=0.45, bg=0.03))
        pst = self.plasma_state(t)
        if pst is not None:
            c = S.tx(pst['c'])
            inten = (0.8 + 1.4 * pst['charge'] + 1.0 * pst['squeeze']) * pst['born'] * (0.9 + 0.1 * flicker(t, 5.0, 2.0))
            inten *= 1 - 0.6 * pst['inhale']
            L.append(Light(c, VIOLET * 0.8 + 0.2, inten, radius=280, rim=0.35, diffuse=0.9, bg=0.035))
        tau = t - T_BOOM
        if 0 <= tau < 1.9:
            c = S.tx(self.boom_center())
            flash = 2.4 * np.exp(-tau / 0.05)
            fire = 2.4 * np.exp(-tau / 0.40) * sstep(0.0, 0.04, tau) * (1 - sstep(T_END - 0.6, T_END, t))
            L.append(Light(c, np.array([1.0, 0.92, 1.0]), flash, radius=560, rim=0.8, diffuse=0.35, bg=0.03))
            L.append(Light(c + np.array([0, -60]), np.array([1.0, 0.5, 0.2]), fire, radius=500, rim=0.7, diffuse=0.75, bg=0.04))
            L.append(Light(c + np.array([120, 0]), ICE_COL, fire * 0.45, radius=420, rim=0.5, diffuse=0.45, bg=0.03))
        return L

    def behind(self, S):
        t = S.t
        layers = []
        st = self.fire_state(t)
        if st is not None and st['kind'] == 'fire':
            self._haze(S, S.tx(st['c']), max(st['r'] * st['grow'], 20), st['amount'], t)
        if T_BOOM <= t:
            layers += self._boom_flash_bg(S, t)
            layers += self._boom_cloud(S, t)
            back = []
            self._parts_boom(t, [], back)
            layers += self._splat(back)
        return layers

    def front(self, S):
        t = S.t
        layers = []
        sprites = []
        st = self.fire_state(t)
        if st is not None:
            layers += self._fire_layers(S, st, t)
        ist = self.ice_state(t)
        if ist is not None:
            layers += self._mist_layer(S, t, S.tx(ist['c']), ist['r'] * max(ist['scale'], 0.4), 1.25 * sstep(26.0, 26.7, t) * ist['opacity'])
        for c, col, k, below in self._antic_glows(t):
            out_c = np.asarray(S.tx(c), np.float32)[None]
            if below:     # hand still under the frame: light pooling up from the bottom edge
                for dx in (-70.0, 0.0, 70.0):
                    pc = out_c + np.array([[dx, 10.0]], np.float32)
                    sprites.append((pc, pc, 80.0, np.c_[(col * 0.15 * k)[None], np.ones(1)], 0))
                sprites.append((out_c + 6, out_c + 6, 22.0, np.c_[((col + 0.3) * 0.25 * k)[None], np.ones(1)], 0))
            else:
                sprites.append((out_c, out_c, 38.0, np.c_[(col * 0.10 * k)[None], np.ones(1)], 0))
                sprites.append((out_c, out_c, 9.0, np.c_[((col + 0.4) * 0.35 * k)[None], np.ones(1)], 0))
        self._parts_wisps(t, sprites)
        self._parts_fire(t, sprites)
        self._parts_ice(t, sprites)
        layers += self._streams_layer(S, t)
        self._arcs_fusion(S, t, sprites)
        self._parts_shed(t, sprites)
        pst = self.plasma_state(t)
        if pst is not None:
            layers += self._plasma_layers(S, t, pst)
            self._parts_charge(t, pst, sprites)
            if pst['born'] > 0.5:
                self._arcs_plasma(t, pst, sprites)
            # imploded to a white-hot pinpoint right before the detonation
            pin = sstep(32.54, T_BOOM, t)
            if pin > 0.01:
                pc = np.asarray(S.tx(pst['c']), np.float32)[None]
                sprites.append((pc, pc, 3.0 + 3.0 * pin, np.c_[np.array([[3.0, 2.8, 3.4]]) * (3.0 * pin), np.ones(1)], 2))
                sprites.append((pc, pc, 18.0, np.c_[np.array([[0.8, 0.6, 1.2]]) * (0.5 * pin), np.ones(1)], 0))
        if t >= T_BOOM:
            self._parts_boom(t, sprites, [])
        tau = t - T_BOOM
        if 0 <= tau < 1.0:
            # jolt away from the blast then a damped 5-7 Hz settle (all components < 12 Hz)
            e = sstep(0.0, 0.02, tau)
            dy = -20.0 * np.exp(-tau / 0.22) * np.sin(2 * np.pi * 5.5 * tau) * e
            dx = 8.0 * np.exp(-tau / 0.25) * np.sin(2 * np.pi * 7.0 * tau + 0.8) * e
            lf = np.exp(-tau / 0.45) * (2.5 * np.sin(2 * np.pi * 1.7 * tau + 0.3) + 1.5 * np.sin(2 * np.pi * 2.9 * tau + 1.1)) * e
            rot = 0.30 * np.exp(-tau / 0.25) * np.sin(2 * np.pi * 4.0 * tau + 0.4) * e
            S.shake.append((dx + lf, dy + 0.6 * lf, rot))
        layers += self._splat(sprites)
        return layers

    def post(self, S, img):
        t = S.t
        ist = self.ice_state(t)
        if ist is not None:
            img = self._ice_post(S, img, ist, t)
        tau = t - T_BOOM
        if 0 <= tau < 0.9:
            img = self._shockwave(S, img, t)
            # veiling glare of the flash over everything (he still reads as a silhouette)
            f = np.exp(-max(tau - 0.5 / FPS, 0) / 0.025)
            if f > 0.003:
                c = S.tx(self.boom_center())
                q = 8
                X, Y = np.meshgrid(np.arange(W // q, dtype=np.float32) * q, np.arange(H // q, dtype=np.float32) * q)
                d2 = ((X - c[0]) ** 2 + (Y - c[1]) ** 2) / (380.0 ** 2)
                m = cv2.resize((0.02 + 0.45 / (1 + d2)).astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
                img = img + (m * f)[..., None] * np.array([1.0, 0.93, 1.05], np.float32)
        return img

    def _fire_layers(self, S, st, t):
        out = []
        c = S.tx(st['c'])
        r = st['r']
        g = max(st['grow'], 0.0)
        if st['kind'] == 'seed':
            # pre-ignition pearl: hot point with soft halo
            rr = max(r * g, 2.0)
            img, box = self._gl_layer(self.p_glow, (c[0] - 12 * rr - 30, c[1] - 12 * rr - 30, c[0] + 12 * rr + 30, c[1] + 12 * rr + 30),
                                      dict(uC=tuple(map(float, c)), uR=float(rr * 1.2), uCol=tuple(map(float, np.array([1.6, 0.8, 0.35]) * st['amp'] * 2.2)), uPow=2.0))
            if img is not None:
                out.append(Layer('add', img[..., :3], box=box))
            return out
        plume = 2.5
        rr = r * g
        box = (c[0] - 2.1 * rr - 4, c[1] - (plume * 1.35 + 1.0) * rr - 4, c[0] + 2.1 * rr + 4, c[1] + 1.8 * rr + 4)
        bx0, by0, bx1, by1 = [int(round(v)) for v in box]
        bx0, by0, bx1, by1 = max(0, bx0), max(0, by0), min(W, bx1), min(H, by1)
        if bx1 - bx0 >= 4 and by1 - by0 >= 4:
            lean = -np.asarray(st['v'], np.float32) / max(r, 1) * 0.10
            lean = np.clip(lean, -2.0, 2.0)
            if FIRE_OCC > 0:
                hm = self._hand_mask(S, 'R', (bx0, by0, bx1, by1)) * (1 - st.get('fly', 0.0))
            else:
                hm = np.zeros((2, 2), np.float32)
            uni = dict(uC=tuple(map(float, c)), uR=float(r), uT=float(t), uAmp=float(st['amp'] * 1.1),
                       uGrow=float(g), uLean=(float(lean[0]), 0.0), uSeed=float(st['seed']),
                       uCore=float(st.get('core', 1.0)), uPlume=plume, uSmoke=1.0, uOcc=float(FIRE_OCC))
            img = gl.render(self.p_fire, (bx1 - bx0, by1 - by0), uniforms=uni, textures={'uHand': hm}, offset=(bx0, by0))
            # heat shimmer on the flame itself (the flame is drawn after S.distort)
            img = self._haze_remap(img, (bx0, by0), c, rr, st['amount'], t)
            img = cv2.GaussianBlur(img, (0, 0), 0.45)
            a = img[..., 3]
            smoke = np.array([0.060, 0.034, 0.022], np.float32)
            out.append(Layer('over', a[..., None] * smoke, a, box=(bx0, by0)))
            out.append(Layer('add', img[..., :3] * st['amount'], box=(bx0, by0)))
        # ignition flash halo
        a = t - T_FIRE
        if 0 <= a < 0.5:
            k = np.exp(-a / 0.09)
            R2 = r * (1.2 + 3.0 * ease_out_cubic(a / 0.5))
            img, box = self._gl_layer(self.p_glow, (c[0] - 3 * R2, c[1] - 3 * R2, c[0] + 3 * R2, c[1] + 3 * R2),
                                      dict(uC=tuple(map(float, c)), uR=float(R2), uCol=tuple(map(float, np.array([2.4, 1.3, 0.6]) * k)), uPow=2.0))
            if img is not None:
                out.append(Layer('add', img[..., :3], box=box))
        return out
