"""Compositor core: scene state, effect interface, layer blending, lighting and post.

Frame pipeline (all in float32 LINEAR light, HDR allowed):

  1. S = Scene(i)          S.fg/S.alpha = decontaminated person + matte, S.bg = black
  2. fx.background(S)      environment modules write/modify S.bg (bathroom, studio, tunnel, space)
  3. fx.person(S)          modify the person: suit (S.fg), warps (S.fg + S.alpha), float offset
  4. fx.lights(S)          return Light dicts -> compositor relights the person (diffuse + rim)
  5. fx.behind(S)          layers drawn between background and person (captions, bg FX)
  6.                       person composited over;  S.distort applied to that base image
  7. fx.front(S)           layers drawn over the person (orbs, particles, sparks)
  8. fx.post(S, img)       full-frame modifications (wipes, flashes, colour)
  9. bloom, camera shake, tonemap, vignette, grain -> sRGB uint8

Every effect must be a PURE FUNCTION OF THE FRAME INDEX / TIME (no state carried from
previous frames) so the video can be rendered in parallel chunks in any order. Use
deterministic seeds (e.g. np.random.default_rng(1234)) created once in __init__ for particle
systems and evaluate particle positions analytically from t.
"""
import numpy as np
import cv2
from . import config as C
from . import assets as A

W, H = C.W, C.H


# ---------------------------------------------------------------- layers
class Layer:
    """A blendable image layer, optionally a crop placed at box=(x0, y0).

    mode 'add'    : dst += rgb                     (emissive light, glows, sparks)
    mode 'over'   : dst = rgb + dst * (1 - a)       (rgb must be PREMULTIPLIED by a)
    mode 'screen' : dst = 1 - (1 - dst)(1 - rgb)    (LDR-ish soft light)
    mode 'mul'    : dst *= rgb                     (darkening / tinting)
    """
    __slots__ = ('mode', 'rgb', 'a', 'box')

    def __init__(self, mode, rgb, a=None, box=(0, 0)):
        self.mode, self.rgb, self.a, self.box = mode, rgb, a, (int(box[0]), int(box[1]))


def blend(dst, L):
    """Blend layer L into full-frame dst (H, W, 3) in place, clipping crops to the frame."""
    h, w = L.rgb.shape[:2]
    x0, y0 = L.box
    fx0, fy0, fx1, fy1 = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
    if fx1 <= fx0 or fy1 <= fy0:
        return dst
    sx0, sy0 = fx0 - x0, fy0 - y0
    src = L.rgb[sy0:sy0 + (fy1 - fy0), sx0:sx0 + (fx1 - fx0), :3]
    d = dst[fy0:fy1, fx0:fx1]
    if L.mode == 'add':
        d += src
    elif L.mode == 'over':
        a = L.a[sy0:sy0 + (fy1 - fy0), sx0:sx0 + (fx1 - fx0)]
        d *= (1.0 - a)[..., None]
        d += src
    elif L.mode == 'screen':
        d[:] = 1.0 - (1.0 - np.clip(d, 0, 1)) * (1.0 - np.clip(src, 0, 1)) + np.maximum(d - 1, 0)
    elif L.mode == 'mul':
        d *= src
    else:
        raise ValueError(L.mode)
    return dst


def Light(pos, color, intensity=1.0, radius=300.0, rim=1.0, diffuse=1.0, bg=0.0):
    """A 2D point light used to relight the person.
    pos (x, y) frame px; color linear rgb; falloff = intensity / (1 + (d / radius)^2).
    rim: strength of edge/rim lighting on matte edges facing the light.
    diffuse: strength of the soft wash over the body.
    bg: strength of a soft glow added to the BACKGROUND around the light (floor/wall bounce)."""
    return dict(pos=np.asarray(pos, np.float32), color=np.asarray(color, np.float32),
                intensity=float(intensity), radius=float(radius), rim=float(rim),
                diffuse=float(diffuse), bg=float(bg))


# ---------------------------------------------------------------- scene state
class Scene:
    def __init__(self, i):
        self.i = i
        self.t = C.t_of(i)
        self.env = C.environment(self.t)
        self.fg = A.fgr(i).copy()                  # person colour (linear, straight alpha)
        self.alpha = A.alpha(i).copy()             # person matte
        self.raw_fg = None                         # untouched copy (set by compositor)
        self.bg = np.zeros((H, W, 3), np.float32)  # background (linear)
        self.lights = []
        self.behind_layers = []
        self.front_layers = []
        self.distort = []        # list of (x0, y0, dx, dy) crops, displacement in px (sample offset)
        self.shake = []          # list of (dx, dy, rot_deg)
        self.person_grade = dict(exposure=1.0, contrast=1.0, saturation=1.0, tint=(1, 1, 1), ambient_rgb=(0, 0, 0))
        self.person_offset = np.zeros(2, np.float32)  # translation applied to the person (space float)
        self.bloom = 1.0         # global bloom strength multiplier
        self.vignette = 0.25
        self.grain = 0.010
        self.meta = {}           # free-form cross-module info (avoid relying on it)

    # helpers ---------------------------------------------------------
    def tx(self, pt):
        """Map a tracked point (frame px of the original footage) to where the person is
        drawn this frame (adds person_offset). Use this for anything attached to the body."""
        return np.asarray(pt, np.float32) + self.person_offset


class Effect:
    """Base class. Override what you need. All hooks receive the Scene `S`."""
    name = 'effect'

    def active(self, t):
        return True

    def background(self, S):
        pass

    def person(self, S):
        pass

    def lights(self, S):
        return []

    def behind(self, S):
        return []

    def front(self, S):
        return []

    def post(self, S, img):
        return img


# ---------------------------------------------------------------- utilities for modules
def smoothstep(a, b, x):
    x = np.clip((np.asarray(x, np.float32) - a) / (b - a + 1e-9), 0.0, 1.0)
    return x * x * (3 - 2 * x)


def ease_out_cubic(x):
    x = np.clip(x, 0, 1)
    return 1 - (1 - x) ** 3


def ease_in_out(x):
    x = np.clip(x, 0, 1)
    return np.where(x < 0.5, 4 * x ** 3, 1 - (-2 * x + 2) ** 3 / 2)


def ease_out_back(x, s=1.70158):
    x = np.clip(x, 0, 1) - 1
    return x * x * ((s + 1) * x + s) + 1


def glow(img, sigma):
    """Gaussian blur helper for glows (works on crops)."""
    return cv2.GaussianBlur(img, (0, 0), sigma)


def grid(h=H, w=W, offset=(0, 0)):
    """Pixel coordinate grids (X, Y) float32 for a crop."""
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    return xs + offset[0], ys + offset[1]


def warp_person(S, map_x, map_y):
    """Resample S.fg and S.alpha with full-frame float32 maps (cv2.remap convention:
    output(x,y) = input(map_x(x,y), map_y(x,y)))."""
    S.fg = cv2.remap(S.fg, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    S.alpha = cv2.remap(S.alpha, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


# ---------------------------------------------------------------- compositor
def _relight(S, lights):
    """Diffuse + rim relighting of the person from 2D point lights (computed at 1/4 res)."""
    if not lights:
        return
    q = 4
    h, w = H // q, W // q
    X, Y = grid(h, w)
    X = X * q + q / 2
    Y = Y * q + q / 2
    a_s = cv2.resize(S.alpha, (w, h), interpolation=cv2.INTER_AREA)
    a_b = cv2.GaussianBlur(a_s, (0, 0), 3.0)
    gx = cv2.Sobel(a_b, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(a_b, cv2.CV_32F, 0, 1, ksize=3)
    gm = np.sqrt(gx * gx + gy * gy) + 1e-6
    nx, ny = -gx / gm, -gy / gm                       # outward normal of the silhouette
    edge = np.clip(gm * 6.0, 0, 1) * np.clip(a_b * 2.2, 0, 1)
    # inner falloff so rim spreads a little into the body
    inner = cv2.GaussianBlur(edge, (0, 0), 2.0)
    diff = np.zeros((h, w, 3), np.float32)
    rim = np.zeros((h, w, 3), np.float32)
    bgl = np.zeros((h, w, 3), np.float32)
    for L in lights:
        dx, dy = L['pos'][0] - X, L['pos'][1] - Y
        d = np.sqrt(dx * dx + dy * dy) + 1e-3
        fall = L['intensity'] / (1.0 + (d / L['radius']) ** 2)
        col = L['color'][None, None, :]
        if L['diffuse'] > 0:
            diff += (fall * L['diffuse'])[..., None] * col
        if L['rim'] > 0:
            facing = np.clip((nx * dx + ny * dy) / d, 0, 1) ** 1.5
            rim += (fall * L['rim'] * facing * inner)[..., None] * col
        if L['bg'] > 0:
            bgl += (fall * L['bg'])[..., None] * col
    diff = cv2.resize(diff, (W, H), interpolation=cv2.INTER_LINEAR)
    rim = cv2.resize(rim, (W, H), interpolation=cv2.INTER_LINEAR)
    S.fg = S.fg * (1.0 + diff) + rim * 0.6 + diff * 0.04
    if bgl.any():
        S.bg += cv2.resize(bgl, (W, H), interpolation=cv2.INTER_LINEAR) * (1.0 - S.alpha[..., None])


def _grade_person(S):
    g = S.person_grade
    fg = S.fg
    if g['exposure'] != 1.0:
        fg = fg * g['exposure']
    if g['contrast'] != 1.0:
        mid = 0.18
        fg = mid * np.power(np.maximum(fg, 0) / mid, g['contrast'])
    if g['saturation'] != 1.0:
        lum = (fg * np.array([0.2126, 0.7152, 0.0722], np.float32)).sum(-1, keepdims=True)
        fg = lum + (fg - lum) * g['saturation']
    tint = np.asarray(g['tint'], np.float32)
    if (tint != 1).any():
        fg = fg * tint
    amb = np.asarray(g['ambient_rgb'], np.float32)
    if amb.any():
        fg = fg + amb
    S.fg = fg.astype(np.float32)


def _bloom(img, strength):
    if strength <= 0:
        return img
    lum = img.max(-1)
    bright = img * (np.clip(lum - 0.85, 0, None) / (lum + 1e-4))[..., None]
    acc = np.zeros_like(img)
    cur = bright
    weights = (0.30, 0.35, 0.40, 0.45, 0.50)
    sizes = []
    for k, wgt in enumerate(weights):
        hh, ww = cur.shape[0] // 2, cur.shape[1] // 2
        cur = cv2.resize(cur, (ww, hh), interpolation=cv2.INTER_AREA)
        b = cv2.GaussianBlur(cur, (0, 0), 1.6)
        acc += cv2.resize(b, (W, H), interpolation=cv2.INTER_LINEAR) * wgt
    return img + acc * strength * 0.55


def _tonemap(x):
    # identity below 0.8, smooth shoulder above (keeps the plate/person untouched)
    k = 0.8
    over = np.maximum(x - k, 0)
    y = np.where(x > k, k + (1 - k) * (1 - np.exp(-over / (1 - k))), x)
    # gentle desaturation of very hot pixels (white-hot cores)
    m = x.max(-1, keepdims=True)
    hot = np.clip((m - 1.2) / 3.0, 0, 0.6)
    lum = y.mean(-1, keepdims=True)
    return y * (1 - hot) + lum * hot + (np.clip(m - 1.0, 0, None) * 0.0)


class Compositor:
    def __init__(self, effects):
        self.effects = effects

    def render(self, i, return_linear=False):
        S = Scene(i)
        S.raw_fg = S.fg
        S.raw_alpha = S.alpha
        fx = [e for e in self.effects if e.active(S.t)]
        for e in fx:
            e.background(S)
        for e in fx:
            e.person(S)
        # person_offset (e.g. zero-g float in space) is applied once, after every person()
        # hook, so person hooks always work in original tracking coordinates.
        ox, oy = float(S.person_offset[0]), float(S.person_offset[1])
        if abs(ox) + abs(oy) > 0.01:
            M = np.float32([[1, 0, ox], [0, 1, oy]])
            S.fg = cv2.warpAffine(S.fg, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            S.alpha = cv2.warpAffine(S.alpha, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            if oy < 0:   # the body is cut by the frame bottom: keep it cut there when floating up
                S.alpha[int(H + oy) - 1:, :] = S.alpha[int(H + oy) - 2:int(H + oy) - 1, :]
        lights = []
        for e in fx:
            lights += e.lights(S) or []
        _grade_person(S)
        _relight(S, lights)
        img = S.bg.copy()
        for e in fx:
            for L in e.behind(S) or []:
                blend(img, L)
        a = S.alpha[..., None]
        img = img * (1 - a) + np.maximum(S.fg, 0) * a
        if S.distort:
            dxm = np.zeros((H, W), np.float32)
            dym = np.zeros((H, W), np.float32)
            for (x0, y0, dx, dy) in S.distort:
                h, w = dx.shape
                fx0, fy0, fx1, fy1 = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
                if fx1 > fx0 and fy1 > fy0:
                    dxm[fy0:fy1, fx0:fx1] += dx[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0]
                    dym[fy0:fy1, fx0:fx1] += dy[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0]
            X, Y = grid()
            img = cv2.remap(img, X + dxm, Y + dym, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        for e in fx:
            for L in e.front(S) or []:
                blend(img, L)
        for e in fx:
            img = e.post(S, img)
        img = _bloom(img, S.bloom)
        if S.shake:
            dx = sum(s[0] for s in S.shake)
            dy = sum(s[1] for s in S.shake)
            rot = sum(s[2] for s in S.shake)
            if abs(dx) + abs(dy) + abs(rot) > 0.05:
                zoom = 1.0 + min(0.08, (abs(dx) + abs(dy)) / W * 2.2 + abs(rot) * 0.01)
                M = cv2.getRotationMatrix2D((W / 2, H / 2), rot, zoom)
                M[:, 2] += (dx, dy)
                img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        if return_linear:
            return img, S
        img = _tonemap(img)
        if S.vignette > 0:
            X, Y = _vig_grid()
            img = img * (1 - S.vignette * Y)[..., None]
        out = A.lin_to_srgb(np.clip(img, 0, 1))
        if S.grain > 0:
            rng = np.random.default_rng(i * 7919 + 13)
            g = rng.standard_normal((H // 2, W // 2)).astype(np.float32)
            g = cv2.resize(g, (W, H), interpolation=cv2.INTER_LINEAR)
            out = out + g[..., None] * S.grain
        return (np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8), S


_VG = None


def _vig_grid():
    global _VG
    if _VG is None:
        X, Y = grid()
        r = np.sqrt(((X - W / 2) / (W / 2)) ** 2 * 0.8 + ((Y - H / 2) / (H / 2)) ** 2)
        v = np.clip((r - 0.55) / 0.8, 0, 1) ** 1.6
        _VG = (None, v.astype(np.float32))
    return _VG
