"""The black studio: near-black seamless backdrop, a soft pool of light behind him, two rim
lights (+ a faint hair light) whose beams catch a slowly drifting haze, sparse dust motes lit by
every light in the scene, and a dark top for the ceiling portal.

Everything is computed at quarter resolution (smooth content) and upsampled once.
"""
import numpy as np
import cv2
from .. import config as C
from ..glfx import gl
from .cleanup_util import sstep, sstep_f, tile_noise, sample_wrap, pulse

W, H = C.W, C.H
Q = 4
WQ, HQ = W // Q, H // Q

T_RIM_L = 14.32
T_RIM_R = 14.44
T_POOL = (14.45, 15.05)
PORTAL = np.float32([540.0, 300.0])

# Two short vertical LED light bars standing on the floor far behind him (out of focus: soft,
# wide core, modest peak). They motivate the rim lights. Set STUDIO_BARS = False for a pure
# black void (the rim lights stay).
STUDIO_BARS = True
BAR_L = dict(x=158.0, top=660.0, color=(0.72, 0.86, 1.00), t0=14.32, ph=0.0)
BAR_R = dict(x=922.0, top=675.0, color=(1.00, 0.84, 0.66), t0=14.44, ph=0.37)
RIM_L = dict(pos=(158.0, 980.0), color=(0.70, 0.84, 1.00), intensity=1.15, radius=700.0, rim=2.0, diffuse=0.03)
RIM_R = dict(pos=(922.0, 1000.0), color=(1.00, 0.86, 0.70), intensity=0.90, radius=700.0, rim=2.0, diffuse=0.02)
HAIR = dict(pos=(590.0, 560.0), color=(0.95, 0.96, 1.00), intensity=0.55, radius=420.0, rim=1.6, diffuse=0.0)

GRADE = dict(exposure=0.74, contrast=1.12, saturation=0.90, tint=(0.96, 0.99, 1.04))


def power_off(t):
    """The studio dies with the portal: after the suction flicker the lights cut out at the
    portal-close flash (50.6), sputter once (50.76) and stay off for the tunnel."""
    if t < 50.55:
        return 1.0
    return max(0.0, (1.0 - sstep_f(50.58, 50.66, t)) + 0.55 * pulse(t, 50.76, 0.016) * (t < 50.9))


def portal_flicker(t, ph=0.0):
    """The ceiling portal drains the studio lights: a dip when it opens, erratic during suction
    (ph decorrelates the two bars' flicker only), then they die at the portal close."""
    f = 1.0 - 0.45 * pulse(t, 46.02, 0.025) - 0.30 * pulse(t, 46.12, 0.02)
    sk = sstep_f(49.25, 49.6, t) * (1.0 - sstep_f(50.45, 50.6, t))
    if sk > 0:
        tp = t + ph
        f -= sk * 0.55 * (0.5 + 0.5 * np.sin(tp * 53.0)) * (0.5 + 0.5 * np.sin(tp * 31.0 + 1.0))
    return max(f, 0.0) * power_off(t)


def strike(t, t0):
    """Studio light power-on: fast ramp, overshoot, one flicker, settle."""
    k = sstep_f(t0, t0 + 0.05, t)
    k *= 1.0 + 0.8 * pulse(t, t0 + 0.045, 0.025)
    k *= 1.0 - 0.45 * pulse(t, t0 + 0.13, 0.018)
    return k


class Studio:
    def __init__(self, seed=77):
        self.seed = seed
        self._built = False

    def build(self):
        if self._built:
            return
        X, Y = np.meshgrid((np.arange(WQ) + 0.5) * Q, (np.arange(HQ) + 0.5) * Q)
        X = X.astype(np.float32); Y = Y.astype(np.float32)
        self.X, self.Y = X, Y
        top = sstep(60, 820, Y)[..., None]
        self.base = (np.float32([0.0010, 0.0011, 0.0014]) * (0.45 + 0.55 * top)).astype(np.float32)
        cx, cy = 585.0, 1090.0
        pool = np.exp(-0.5 * (((X - cx) / 360.0) ** 2 + ((Y - cy) / 430.0) ** 2))[..., None]
        wide = np.exp(-0.5 * (((X - cx) / 720.0) ** 2 + ((Y - cy) / 820.0) ** 2))[..., None]
        self.pool = ((pool * 0.0037 * np.float32([0.92, 0.96, 1.0]) + wide * 0.0008 * np.float32([0.85, 0.92, 1.0]))
                     * (0.25 + 0.75 * top)).astype(np.float32)
        self.pool_l = (pool[..., 0] ** 1.5 * sstep(300, 800, Y)).astype(np.float32)   # illumination for motes

        def beam(px, py, sx, sy, amp, col):
            g = np.exp(-0.5 * (((X - px) / sx) ** 2 + ((Y - py) / sy) ** 2))
            return (g[..., None] * amp * np.float32(col)).astype(np.float32), g.astype(np.float32)
        self.beamL, self.beamL_l = beam(-40, 960, 240, 330, 0.0016, RIM_L['color'])
        self.beamR, self.beamR_l = beam(1120, 1000, 230, 320, 0.0013, RIM_R['color'])
        # light bars: wide haze glow (quarter res) + illumination for the motes
        self.bars = []
        for B in (BAR_L, BAR_R):
            vm = sstep(B['top'] - 40, B['top'] + 90, Y)
            g = np.exp(-0.5 * ((X - B['x']) / 120.0) ** 2) * vm
            amp = 0.0045 if STUDIO_BARS else 0.0
            self.bars.append(((g[..., None] * amp * np.float32(B['color'])).astype(np.float32), g.astype(np.float32)))
        self.ys = np.arange(H, dtype=np.float32)
        self.n1 = tile_noise(256, [4, 8, 16, 32], self.seed)
        self.n2 = tile_noise(256, [3, 6, 12, 24], self.seed + 1)
        # dust motes
        rng = np.random.default_rng(self.seed + 5)
        n_s, n_b = 60, 22
        n = n_s + n_b
        self.m_big = np.r_[np.zeros(n_s, bool), np.ones(n_b, bool)]
        self.m_x0 = rng.uniform(0, W, n).astype(np.float32)
        self.m_y0 = rng.uniform(220, H, n).astype(np.float32)
        self.m_vx = rng.uniform(-7, 7, n).astype(np.float32)
        self.m_vy = rng.uniform(-6, 2.5, n).astype(np.float32)
        self.m_a = rng.uniform(8, 22, n).astype(np.float32)
        self.m_f = rng.uniform(0.05, 0.22, n).astype(np.float32)
        self.m_p = rng.uniform(0, 2 * np.pi, n).astype(np.float32)
        self.m_tw = rng.uniform(0.3, 1.2, n).astype(np.float32)
        self.m_r = np.where(self.m_big, rng.uniform(5, 13, n), rng.uniform(1.3, 2.4, n)).astype(np.float32)
        self.m_i = np.where(self.m_big, rng.uniform(0.006, 0.018, n), rng.uniform(0.10, 0.32, n)).astype(np.float32)
        self.m_pull = rng.uniform(0.6, 1.0, n).astype(np.float32)
        self._built = True

    # ------------------------------------------------------------------ timing
    def levels(self, t):
        self.build()
        on_l = strike(t, BAR_L['t0']) * portal_flicker(t, BAR_L['ph'])
        on_r = strike(t, BAR_R['t0']) * portal_flicker(t, BAR_R['ph'])
        on_p = sstep_f(T_POOL[0], T_POOL[1], t) * power_off(t)
        return on_l, on_r, on_p

    def haze_q(self, t):
        """Quarter-res haze density in [0,1], slowly drifting."""
        self.build()
        u = self.X / Q
        v = self.Y / Q
        h1 = sample_wrap(self.n1, u * 0.9 + t * 2.2, v * 0.9 - t * 1.4)
        h2 = sample_wrap(self.n2, u * 0.55 - t * 1.1 + 40, v * 0.55 - t * 2.0 + 17)
        h = 0.6 * h1 + 0.4 * h2
        return np.clip((h - 0.5) * 1.5 + 0.5, 0, 1).astype(np.float32)

    def backdrop(self, t):
        """Full-res studio background (linear)."""
        self.build()
        on_l, on_r, on_p = self.levels(t)
        img = self.base.copy()
        if on_l + on_r + on_p > 1e-4:
            hz = self.haze_q(t)[..., None]
            img += self.pool * (on_p * (0.70 + 0.60 * hz))
            img += self.beamL * (on_l * (0.35 + 1.1 * hz))
            img += self.beamR * (on_r * (0.35 + 1.1 * hz))
            img += self.bars[0][0] * (on_l * (0.35 + 1.1 * hz))
            img += self.bars[1][0] * (on_r * (0.35 + 1.1 * hz))
        out = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
        for B, lev in ((BAR_L, on_l), (BAR_R, on_r)):
            if lev > 1e-4 and STUDIO_BARS:
                self._draw_bar(out, B, lev, t)
        return out

    def _draw_bar(self, out, B, lev, t):
        """Out-of-focus emissive LED tube far behind him (soft wide core + halo, modest peak so
        it blooms only a little), lighting up bottom-to-top when it strikes."""
        x = B['x']
        x0, x1 = int(max(0, x - 110)), int(min(W, x + 110))
        dx = np.arange(x0, x1, dtype=np.float32) - x
        prof = (np.exp(-(dx / 6.5) ** 2) * 0.80 + np.exp(-(dx / 15.0) ** 2) * 0.10
                + np.exp(-(dx / 45.0) ** 2) * 0.018)
        sweep = sstep_f(B['t0'] - 0.01, B['t0'] + 0.07, t)
        y_on = H + 20 - (H + 20 - B['top']) * sweep
        vm = sstep(B['top'] - 14, B['top'] + 22, self.ys) * sstep(y_on - 14, y_on + 14, self.ys)
        # slight brightness variation along the tube (diffuser), static
        vm = vm * (0.92 + 0.08 * np.cos(self.ys * 0.004 + x))
        out[:, x0:x1] += (vm[:, None] * prof[None, :] * lev)[..., None] * np.float32(B['color'])

    def lights(self, t, dim=1.0):
        on_l, on_r, on_p = self.levels(t)
        out = []
        for L, k in ((RIM_L, on_l), (RIM_R, on_r), (HAIR, 0.5 * (on_l + on_r))):
            if k * dim > 1e-3:
                d = dict(L)
                d['intensity'] = L['intensity'] * k * dim
                out.append(d)
        return out

    # ------------------------------------------------------------------ behind(): scattering + motes
    def scatter(self, t, lights):
        """Haze in-scattering of every light in the scene (quarter res -> full res add layer)."""
        self.build()
        on = sstep_f(14.35, 15.0, t)
        if on <= 0 or not lights:
            return None
        hz = self.haze_q(t)
        acc = np.zeros((HQ, WQ, 3), np.float32)
        any_ = False
        for L in lights:
            I = float(L['intensity'])
            if I < 0.02:
                continue
            k = 0.0014 + 0.035 * float(L.get('bg', 0.0))
            p = np.asarray(L['pos'], np.float32)
            r = float(L['radius']) * 0.65
            fall = I / (1.0 + ((self.X - p[0]) ** 2 + (self.Y - p[1]) ** 2) / (r * r)) ** 2
            acc += (fall * k)[..., None] * np.asarray(L['color'], np.float32)
            any_ = True
        if not any_:
            return None
        acc *= (0.30 + 1.1 * hz)[..., None] * on
        return cv2.resize(acc, (W, H), interpolation=cv2.INTER_LINEAR)

    def motes(self, t, lights):
        """Dust motes (additive full-frame layer) lit by the pool, the beams and all lights."""
        self.build()
        on_l, on_r, on_p = self.levels(t)
        vis = sstep_f(14.4, 15.2, t)
        if vis <= 0:
            return None

        def pos(tt):
            x = self.m_x0 + self.m_vx * (tt - 14.0) + self.m_a * np.sin(2 * np.pi * self.m_f * tt + self.m_p)
            y = self.m_y0 + self.m_vy * (tt - 14.0) + 0.6 * self.m_a * np.cos(2 * np.pi * self.m_f * 0.8 * tt + self.m_p)
            x = np.mod(x + 40, W + 80) - 40
            y = np.mod(y - 180, H - 140) + 180
            # the ceiling portal draws the dust up into it
            pull = sstep(46.3, 50.6, tt) ** 2 * self.m_pull
            sw = pull * 0.6
            dx, dy = PORTAL[0] - x, PORTAL[1] - y
            x2 = x + dx * pull - dy * sw * 0.25
            y2 = y + dy * pull + dx * sw * 0.25
            return np.stack([x2, y2], 1).astype(np.float32), pull
        p1, pull = pos(t)
        p0, _ = pos(t - 1.0 / 50)
        # illumination at the motes
        xi = np.clip((p1[:, 0] / Q).astype(int), 0, WQ - 1)
        yi = np.clip((p1[:, 1] / Q).astype(int), 0, HQ - 1)
        illum = np.zeros((len(p1), 3), np.float32)
        illum += (self.pool_l[yi, xi] * on_p * 0.9)[:, None] * np.float32([0.9, 0.95, 1.0])
        illum += (self.beamL_l[yi, xi] * on_l * 1.4)[:, None] * np.float32(RIM_L['color'])
        illum += (self.beamR_l[yi, xi] * on_r * 1.2)[:, None] * np.float32(RIM_R['color'])
        illum += (self.bars[0][1][yi, xi] * on_l * 1.0)[:, None] * np.float32(BAR_L['color'])
        illum += (self.bars[1][1][yi, xi] * on_r * 1.0)[:, None] * np.float32(BAR_R['color'])
        own = [np.float32(RIM_L['pos']), np.float32(RIM_R['pos']), np.float32(HAIR['pos'])]
        for L in lights or []:
            I = float(L['intensity'])
            if I < 0.02:
                continue
            pp = np.asarray(L['pos'], np.float32)
            if any(np.abs(pp - o).max() < 0.5 for o in own):
                continue          # studio lights light the dust through their beam maps
            r = float(L['radius'])
            d2 = ((p1 - pp) ** 2).sum(1)
            illum += (I / (1.0 + d2 / (r * r)) * 0.8)[:, None] * np.asarray(L['color'], np.float32)
        tw = 0.45 + 0.55 * (0.5 + 0.5 * np.sin(2 * np.pi * self.m_tw * t + 3 * self.m_p)) ** 3
        fade_portal = 1.0 - sstep(0.8, 1.0, pull)
        inten = self.m_i * tw * vis * fade_portal
        col = illum * inten[:, None]
        c4 = np.concatenate([col, np.ones((len(col), 1), np.float32)], 1)
        keep = inten > 1e-4
        if not keep.any():
            return None
        return gl.sprites((W, H), p0[keep], p1[keep], radius=self.m_r[keep], color=c4[keep], shape=0)[..., :3]
