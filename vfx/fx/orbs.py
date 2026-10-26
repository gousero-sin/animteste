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
from ..core import Layer, Light, smoothstep, ease_out_cubic, ease_out_back
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
            hb = A.head_box(i)
            head[k] = ((hb[0] + hb[2]) / 2, hb[1], hb[2] - hb[0])
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
        # plasma orb anchor above the head (heavily lagged)
        tgt = np.c_[hd[:, 0], hd[:, 1] - 150.0].astype(np.float32)
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
        if t < 22.30 or t > 29.55:
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
        amp = 1.0 + 1.6 * np.exp(-a / 0.12)
        st.update(kind='fire', grow=grow, amp=amp, amount=sstep(0, 0.12, a), core=1.15 + 0.8 * np.exp(-a / 0.2))
        # fusion: leave the palm, spiral into the ring
        if t > T_FUS0:
            k = sstep(T_FUS0, 29.30, t)
            ent = self.ring_entry('R', t)
            path = self._spiral(c, ent, k, self._interp(self.ring, t), +1)
            st['c'] = path
            st['grow'] = grow * (1 - 0.70 * k)
            st['amount'] *= 1 - sstep(29.20, 29.50, t)
            st['v'] = v * (1 - k)
        return st

    def ring_entry(self, side, t):
        rc = self._interp(self.ring, t)
        th = self.stream_head_angle(side, t)
        return np.array([rc[0] + rc[2] * np.cos(th), rc[1] + rc[2] * np.sin(th)], np.float32)

    def stream_head_angle(self, side, t):
        # fire enters from the left (angle pi), ice from the right (0); angular speed ramps up
        base = np.pi if side == 'R' else 0.0
        a = max(t - 29.05, 0.0)
        return base + (2.4 * a + 2.2 * a * a)

    def _spiral(self, p0, p1, k, rc, sgn):
        """point on a curved path from p0 to p1 (bulging around the ring centre)."""
        e = float(ease_out_cubic(k)) if k < 1 else 1.0
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
        self.sh_final = rng.uniform(1.18, 1.50, ns).astype(np.float32)
        self.sh_rad = rng.uniform(0.17, 0.27, ns).astype(np.float32)
        self.sh_t = (26.20 + 0.30 * rng.permutation(ns) / ns).astype(np.float32)

    def ice_state(self, t):
        if t < 26.15 or t > 29.50:
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
        if t > T_FUS0:
            k = sstep(T_FUS0, 29.30, t)
            ent = self.ring_entry('L', t)
            st['c'] = self._spiral(c, ent, k, self._interp(self.ring, t), -1)
            st['scale'] = 1 - 0.68 * k
            st['opacity'] = 1 - sstep(29.15, 29.45, t)
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
                                                         uAmt=float(amt), uCol=(0.30, 0.48, 0.70)), scale=0.5)
        if img is None:
            return []
        return [Layer('add', img[..., :3], box=b)]

    # .............................................................. particles
    def _build_particles(self):
        rng = self.rng
        self.emb = Births(rng, T_FIRE - 0.05, 29.4, 55, (0.7, 1.7))
        self.ign = Births(rng, T_FIRE, T_FIRE + 0.06, 1500, (0.35, 0.9))
        self.wispR = Births(rng, T_START, T_FIRE, 34, (0.7, 1.3))
        self.wispL = Births(rng, T_START + 0.4, T_ICE, 26, (0.7, 1.3))
        self.snow = Births(rng, 26.30, 29.40, 34, (1.1, 2.2))
        self.iceburst = Births(rng, T_ICE, T_ICE + 0.05, 1100, (0.5, 1.1))

    def _parts_fire(self, t, out):
        """embers rising from the fire (world space after birth)."""
        idx, age = self.emb.alive(t)
        if len(idx):
            tb = self.emb.tb[idx]
            u = self.emb.u[idx]
            life = self.emb.life[idx]
            amt = np.array([self._fire_amount(x) for x in tb], np.float32)
            keep = u[:, 0] < amt
            idx, age, tb, u, life = idx[keep], age[keep], tb[keep], u[keep], life[keep]
        if len(idx):
            c0 = self._interp(self.anchor['R'], tb)
            v0 = self._interp(self.anchor_vel['R'], tb)
            r = 75.0
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
            P0 = pos(age)
            P1 = pos(np.maximum(age - 0.035, 0))
            k = age / life
            temp = (1.15 - 1.05 * k) * (0.75 + 0.35 * u[:, 7])
            col = bbody_np(temp) * (2.2 + 2.0 * u[:, 8:9]) * (1 - k[:, None] ** 2)
            fl = 0.65 + 0.35 * np.sin(age * 37 + u[:, 9] * 20)
            col = col * fl[:, None]
            rad = 1.1 + 1.6 * u[:, 2]
            out.append((P0, P1, rad, np.c_[col, np.ones(len(col))], 2))
        # ignition burst sparks (radial, fast, drag + gravity)
        idx, age = self.ign.alive(t)
        if len(idx):
            u = self.ign.u[idx]
            life = self.ign.life[idx]
            c0 = self._interp(self.anchor['R'], T_FIRE)
            ang = u[:, 0] * 2 * np.pi
            sp = 500 + 1300 * u[:, 1] ** 2
            dirv = np.c_[np.cos(ang), np.sin(ang) * 0.85 - 0.25]

            def pos(tau):
                drag = 3.5
                e = (1 - np.exp(-drag * tau)) / drag
                return c0 + dirv * (sp * e)[:, None] + np.c_[np.zeros_like(tau), 260 * tau * tau]
            P0 = pos(age)
            P1 = pos(np.maximum(age - 0.03, 0))
            k = age / life
            col = bbody_np(1.25 - 1.1 * k) * (3.0 + 3 * u[:, 2:3]) * (1 - k[:, None])
            out.append((P0, P1, 1.0 + 1.4 * u[:, 3], np.c_[col, np.ones(len(col))], 2))

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
            c0 = self._interp(self.anchor['L'], tb)
            ang = u[:, 1] * 2 * np.pi
            rad = 60 * (0.3 + 0.9 * u[:, 2])
            p0 = c0 + np.c_[np.cos(ang), np.sin(ang) * 0.8] * rad[:, None]

            def pos(tau):
                x = p0[:, 0] + (u[:, 3] - 0.5) * 70 * tau + 14 * np.sin(tau * (2 + 3 * u[:, 4]) + 6.28 * u[:, 5])
                y = p0[:, 1] + (35 + 80 * u[:, 6]) * tau + 25 * tau * tau
                return np.c_[x, y]
            P0 = pos(age)
            P1 = pos(np.maximum(age - 0.03, 0))
            k = age / life
            tw = 0.5 + 0.5 * np.sin(age * (9 + 14 * u[:, 7]) + 6.28 * u[:, 8])
            inten = (0.5 + 2.6 * tw ** 6) * np.sin(np.pi * np.clip(k, 0, 1)) ** 0.7
            col = np.array([0.72, 0.88, 1.0])[None] * inten[:, None]
            out.append((P0, P1, 0.9 + 1.4 * u[:, 9], np.c_[col, np.ones(len(col))], 2))
        idx, age = self.iceburst.alive(t)
        if len(idx):
            u = self.iceburst.u[idx]
            life = self.iceburst.life[idx]
            c0 = self._interp(self.anchor['L'], T_ICE)
            ang = u[:, 0] * 2 * np.pi
            sp = 300 + 900 * u[:, 1] ** 2
            dirv = np.c_[np.cos(ang), np.sin(ang)]

            def pos(tau):
                drag = 4.5
                e = (1 - np.exp(-drag * tau)) / drag
                return c0 + dirv * (sp * e)[:, None] + np.c_[np.zeros_like(tau), 90 * tau * tau]
            P0 = pos(age)
            P1 = pos(np.maximum(age - 0.03, 0))
            k = age / life
            tw = 0.5 + 0.5 * np.sin(age * 30 + 6.28 * u[:, 2])
            col = np.array([0.75, 0.9, 1.0])[None] * ((1.5 + 3.0 * tw ** 4) * (1 - k) ** 1.5)[:, None]
            out.append((P0, P1, 0.9 + 1.2 * u[:, 3], np.c_[col, np.ones(len(col))], 2))

    def _fire_amount(self, t):
        st = self.fire_state(float(t))
        if st is None or st['kind'] != 'fire':
            return 0.0
        return float(st['amount'])

    def _parts_wisps(self, t, out):
        """anticipation: faint energy wisps spiralling into each hand."""
        for side, B, t_on, col0 in (('R', self.wispR, T_FIRE, FIRE_COL), ('L', self.wispL, T_ICE, ICE_COL)):
            if t > t_on + 0.1:
                continue
            idx, age = B.alive(t)
            if not len(idx):
                continue
            u = B.u[idx]
            life = B.life[idx]
            tb = B.tb[idx]
            # density ramps up toward the ignition
            ramp = 0.25 + 0.75 * np.clip((tb - B.t0) / (t_on - B.t0), 0, 1) ** 1.5
            keep = u[:, 0] < ramp
            idx, age, u, life = idx[keep], age[keep], u[keep], life[keep]
            if not len(idx):
                continue
            c = self._interp(self.anchor[side], t)
            c = np.array([c[0], min(c[1], H - 30)], np.float32)
            spin = 1 if side == 'R' else -1

            def pos(tau):
                s = np.clip(tau / life, 0, 1)
                R = (110 + 170 * u[:, 1]) * (1 - s) ** 1.6 + 6
                th = u[:, 2] * 6.283 + spin * (2.5 + 2 * u[:, 3]) * s ** 1.3
                return np.c_[c[0] + R * np.cos(th), c[1] + R * np.sin(th) * 0.85]
            s = age / life
            inten = (np.sin(np.pi * s) ** 1.5) * (0.30 + 0.9 * s) * (0.8 + 0.8 * u[:, 4])
            base = col0[None] * 1.2 + 0.25
            nseg = 5
            for j in range(nseg):
                a0 = np.maximum(age - 0.018 * j, 0)
                a1 = np.maximum(age - 0.018 * (j + 1), 0)
                fade = (1 - j / nseg) ** 1.5
                col = base * (inten * fade)[:, None] * 0.9
                out.append((pos(a0), pos(a1), 0.7 + 0.6 * u[:, 5], np.c_[col, np.ones(len(col))], 2))
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
        """heat-shimmer displacement above a flame (S.distort crop)."""
        w = int(4.4 * r)
        h = int(6.5 * r)
        x0 = int(c[0] - w / 2)
        y0 = int(c[1] - 5.2 * r)
        X, Y = np.meshgrid(np.arange(w), np.arange(h))
        n = self.haze_tex.shape[0]
        sx = (X * 0.9 + x0).astype(int) % n
        sy = (Y * 0.9 + y0 + int(t * 140)).astype(int) % n
        dx = self.haze_tex[sy, sx, 0]
        dy = self.haze_tex[(sy + 97) % n, (sx + 31) % n, 1]
        u = (X - w / 2) / (w / 2)
        v = (Y - h * (5.2 / 6.5)) / r          # 0 at centre, negative upward
        mask = np.exp(-u * u * 2.5) * np.clip(-v / 1.0, 0, 1) * np.exp(-np.maximum(-v - 1.5, 0) / 2.2)
        mask *= np.clip((v + 4.8) * 0.8, 0, 1) * 1.0
        amp = 3.2 * strength
        S.distort.append((x0, y0, (dx * mask * amp).astype(np.float32), (dy * mask * amp * 1.4).astype(np.float32)))

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

    # .............................................................. hooks
    def active(self, t):
        return T_START - 0.02 <= t <= T_END + 0.02

    def lights(self, S):
        t = S.t
        L = []
        st = self.fire_state(t)
        if st is not None:
            c = S.tx(st['c'])
            fl = 1.0 + 0.16 * flicker(t, 1.0) + 0.06 * flicker(t, 4.0, 2.3)
            if st['kind'] == 'seed':
                inten = 0.5 * st['amount']
            else:
                inten = (1.35 * st['amount'] * min(st['grow'], 1.2) + 2.5 * np.exp(-(t - T_FIRE) / 0.10)) * fl
            L.append(Light(c + np.array([0, -0.3 * st['r']]), FIRE_COL, inten, radius=240, rim=1.8, diffuse=0.85, bg=0.06))
            # close, hot light on the palm itself
            L.append(Light(c + np.array([0, 0.6 * st['r']]), np.array([1.0, 0.55, 0.25]), inten * 0.9, radius=95, rim=0.6, diffuse=1.2, bg=0.0))
        ist = self.ice_state(t)
        if ist is not None:
            c = S.tx(ist['c'])
            inten = 1.05 * ist['amount'] * ist['glow'] * (0.6 + 0.4 * ist['scale']) + 2.2 * ist['flash']
            L.append(Light(c, ICE_COL, inten, radius=230, rim=1.8, diffuse=0.75, bg=0.05))
            L.append(Light(c + np.array([0, 0.7 * ist['r']]), np.array([0.55, 0.8, 1.0]), inten * 0.7, radius=90, rim=0.5, diffuse=1.0, bg=0.0))
        return L

    def behind(self, S):
        t = S.t
        st = self.fire_state(t)
        if st is not None and st['kind'] == 'fire':
            self._haze(S, S.tx(st['c']), max(st['r'] * st['grow'], 20), st['amount'], t)
        return []

    def front(self, S):
        t = S.t
        layers = []
        sprites = []
        st = self.fire_state(t)
        if st is not None:
            layers += self._fire_layers(S, st, t)
        ist = self.ice_state(t)
        if ist is not None:
            layers += self._mist_layer(S, t, S.tx(ist['c']), ist['r'] * max(ist['scale'], 0.4), 0.9 * sstep(26.0, 26.7, t) * ist['opacity'])
        self._parts_wisps(t, sprites)
        self._parts_fire(t, sprites)
        self._parts_ice(t, sprites)
        layers += self._splat(sprites)
        return layers

    def post(self, S, img):
        t = S.t
        ist = self.ice_state(t)
        if ist is not None:
            img = self._ice_post(S, img, ist, t)
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
        lean = -np.asarray(st['v'], np.float32) / max(r, 1) * 0.10
        lean = np.clip(lean, -2.0, 2.0)
        uni = dict(uC=tuple(map(float, c)), uR=float(r), uT=float(t), uAmp=float(st['amp'] * 1.9),
                   uGrow=float(g), uLean=(float(lean[0]), 0.0), uSeed=float(st['seed']),
                   uCore=float(st.get('core', 1.0)), uPlume=plume, uSmoke=1.0)
        img, box0 = self._gl_layer(self.p_fire, box, uni)
        if img is not None:
            img = cv2.GaussianBlur(img, (0, 0), 0.85)
            a = img[..., 3]
            smoke = np.array([0.075, 0.042, 0.025], np.float32)
            out.append(Layer('over', a[..., None] * smoke, a, box=box0))
            out.append(Layer('add', img[..., :3] * st['amount'], box=box0))
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
