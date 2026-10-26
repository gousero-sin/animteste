"""space — outer space + Gargantua-style black hole (active t >= EV['space'][0] - 0.1).

Background (linear HDR):
  * far canvas  (nebula, galactic band, dust lanes, distant galaxies)            slow drift
  * ~48k faint far stars splatted per frame at exact float positions (energy-normalised
    gaussian footprints -> no coherent "breathing" from resampling a baked starfield)
  * near canvas (foreground dust absorption + glowing wisps)                     faster drift
  * ~520 twinkling mid stars (gl.sprites, motion-blurred on arrival)            medium drift
  * 6 hero stars with diffraction spikes (spikes are a camera artefact -> drawn unlensed)
  * arrival: hyperspace-exit zoom that decelerates and settles (starts after the exit flash)
Black hole (seed 61.62, grows 62.0 -> 62.70, ignites 62.74 on 'NEgro', settles, then spins):
  * Schwarzschild geodesics ray-traced once (space_bh.py, cached): exact 1-D background
    lensing table + per-super-sample disk-plane crossings (3 orders -> lensed far side,
    lower image, photon ring), Keplerian disk with gravitational + Doppler shift.
  * per frame: full-frame lens remap of the sky (numpy) + GLSL shader for the box around the
    hole (2x2 super-sampled disk, shadow, mip-filtered strongly-lensed sky, photon ring).
  * formation: lens mass grows (space visibly warps + co-rotating swirl), gas spirals in
    along two log-spiral arms IN the disk plane (ray-traced, so they arch over / under the
    shadow like the disk), the disk forms from the outside in behind a hot shock front and
    contracts to its final size, ignition flash on the approaching ISCO limb + refractive
    gravitational-wave ripple, overshoot + settle.
  * protect-black: post() pre-compensates the compositor bloom inside the event horizon so the
    shadow stays black next to the HDR disk.
Person: zero-g float via S.person_offset, cool space grade, nebula fill, soft warm wash from the
hole (compositor light) + a crisp full-res warm rim on the hole-facing silhouette (front()).
"""
import numpy as np
import cv2
from .. import config as C
from .. import core
from .. import assets as A
from . import space_sky as K
from .space_bh import BHData

W, H = C.W, C.H
T0 = C.EV['space'][0]                 # 53.55
T_ON = T0 - 0.1
BH_C = np.array([846.0, 650.0], np.float32)     # black hole centre (cv2 pixel-centre coords)
SHADOW_PX = 76.0
INCL = 4.0
ROLL = -13.0                          # disk roll (deg): right end up, left (approaching) end down
BOX = (560, 400)
SS = 2
_rl = np.radians(ROLL)
APPROACH = np.array([-np.cos(_rl), -np.sin(_rl)], np.float32)   # screen dir of the approaching limb

V_FAR = np.array([-3.2, 1.3], np.float32)       # px / s drift of the far layer
V_NEAR = V_FAR * 2.4
V_MID = V_FAR * 1.5
V_HERO = V_FAR * 1.7
ZC = np.array([540.0, 620.0], np.float32)       # arrival zoom centre
ZOOM_T = T0 + 0.25                              # hyperspace-exit deceleration starts after the flash
ZOOM_DUR = 1.25
ZOOM_Z0 = 0.80
T_PEAK = 62.70                                  # end of growth (size overshoot peak)
T_IGNITE = 62.74                                # ignition on the stressed syllable of 'NEgro'
LENS_SOFT = (150.0, 340.0, 0.38)               # weaken bending beyond ~2x shadow (less 'twirl')


# ------------------------------------------------------------------ timeline helpers
def _ss(a, b, x):
    return float(core.smoothstep(a, b, x))


def arrival_zoom(t):
    """Hyperspace exit: the view expands (z < 1 -> 1) and decelerates (ease-out quint) over
    ZOOM_T .. ZOOM_T + ZOOM_DUR; before ZOOM_T (under the portal's white exit flash) it keeps the
    initial speed, softly floored."""
    u = (t - ZOOM_T) / ZOOM_DUR
    if u >= 1.0:
        return 1.0
    e = 1.0 - (1.0 - u) ** 5 if u >= 0 else 5.0 * u
    z = ZOOM_Z0 + (1.0 - ZOOM_Z0) * e
    zf = 0.74
    if z < zf + 0.04:                     # smooth floor
        z = zf + 0.04 * np.exp((z - zf - 0.04) / 0.04)
    return float(z)


def bh_state(t):
    st = dict(s=0.0, form=0.0, gain=0.0, flash=0.0, twist=0.0, gw=(0.0, 0.0, 70.0), seed=0.0,
              tdisk=max(0.0, t - 61.8))
    if t < 61.70:
        return st
    # lens scale (mass): seed -> growth -> overshoot -> critically-shaped damped settle
    if t < 62.0:
        s = 0.16 * _ss(61.70, 62.0, t)
    elif t < T_PEAK:
        u = (t - 62.0) / (T_PEAK - 62.0)
        e = 0.5 * (1.0 - (1.0 - u) ** 2.2) + 0.5 * float(core.ease_in_out(u))
        s = 0.16 + (1.045 - 0.16) * e
    else:
        dt = t - T_PEAK
        s = 1.0 + 0.045 * np.exp(-dt * 6.5) * (np.cos(dt * 15.0) + 6.5 / 15.0 * np.sin(dt * 15.0))
    st['s'] = float(s)
    st['seed'] = _ss(61.62, 61.92, t) * (1.0 - _ss(62.08, 62.32, t))
    st['form'] = float(core.ease_in_out((t - 61.95) / 0.85))
    g = _ss(62.0, T_IGNITE, t)
    st['gain'] = (0.35 * g + 0.65 * g * g) * (1.0 + 0.035 * np.sin(t * 1.7) + 0.02 * np.sin(t * 4.3 + 1.0))
    if t > T_IGNITE - 0.05:
        dt = t - T_IGNITE
        st['flash'] = float(_ss(-0.05, 0.015, dt) * np.exp(-max(dt, 0.0) / 0.13))
    # frame-dragging swirl of the sky, CO-ROTATING with the disk (counter-clockwise on screen);
    # relaxes slowly after ignition
    st['twist'] = 1.1 * _ss(61.7, 62.4, t) * (1.0 - _ss(62.65, 63.9, t))
    if t > T_IGNITE:
        dt = t - T_IGNITE
        st['gw'] = (14.0 * np.exp(-dt * 2.4), 70.0 + 1100.0 * dt, 95.0)
    return st


def float_offset(t):
    """Zero-g drift of the person: smooth incommensurate sines, eased in after the arrival and
    eased out before the Dr. Strange wipe reveals the raw footage."""
    env = _ss(54.1, 56.4, t) * (1.0 - _ss(72.5, 73.55, t))
    if env <= 0:
        return np.zeros(2, np.float32)
    x = 8.0 * np.sin(2 * np.pi * t / 6.7 + 0.6) + 4.0 * np.sin(2 * np.pi * t / 3.9 + 2.1)
    y = 7.0 * np.sin(2 * np.pi * t / 5.3 + 1.3) + 3.5 * np.sin(2 * np.pi * t / 3.1 + 0.2)
    return (np.array([x, y], np.float32) * env).astype(np.float32)


def splat_stars(p, rgb, sigma, k=4):
    """Energy-normalised gaussian splat of point sources at float positions (cv2 pixel-centre
    convention). Each star's footprint weights sum to 1, so its flux is exactly constant while
    it drifts across the pixel grid (no sub-pixel 'breathing')."""
    x, y = p[:, 0], p[:, 1]
    ix = np.floor(x).astype(np.int32) - (k // 2 - 1)
    iy = np.floor(y).astype(np.int32) - (k // 2 - 1)
    offs = np.arange(k, dtype=np.int32)
    gx = ix[:, None] + offs[None]
    gy = iy[:, None] + offs[None]
    s2 = (2.0 * sigma * sigma)[:, None]
    wx = np.exp(-(gx - x[:, None]) ** 2 / s2)
    wy = np.exp(-(gy - y[:, None]) ** 2 / s2)
    wx /= wx.sum(1, keepdims=True)
    wy /= wy.sum(1, keepdims=True)
    w = (wy[:, :, None] * wx[:, None, :]).astype(np.float32)
    X = np.broadcast_to(gx[:, None, :], w.shape)
    Y = np.broadcast_to(gy[:, :, None], w.shape)
    ok = (X >= 0) & (X < W) & (Y >= 0) & (Y < H)
    idx = (Y * W + X)[ok]
    ww = w[ok]
    out = np.empty((H * W, 3), np.float32)
    for c in range(3):
        cc = np.broadcast_to(rgb[:, c][:, None, None], w.shape)[ok]
        out[:, c] = np.bincount(idx, weights=ww * cc, minlength=H * W)
    return out.reshape(H, W, 3)


def bloom_crop(img, box):
    """Exactly the compositor's bloom (core._bloom, before its strength factor), evaluated only on
    the crop box=(x0, y0, w, h): the pyramid is built full-frame, the INTER_LINEAR upsample is
    reproduced with remap on the crop pixels only."""
    x0, y0, w, h = box
    lum = img.max(-1)
    bright = img * (np.clip(lum - 0.85, 0, None) / (lum + 1e-4))[..., None]
    acc = np.zeros((h, w, 3), np.float32)
    cur = bright
    xs = np.arange(x0, x0 + w, dtype=np.float32) + 0.5
    ys = np.arange(y0, y0 + h, dtype=np.float32) + 0.5
    for wgt in (0.30, 0.35, 0.40, 0.45, 0.50):
        hh, ww = cur.shape[0] // 2, cur.shape[1] // 2
        cur = cv2.resize(cur, (ww, hh), interpolation=cv2.INTER_AREA)
        b = cv2.GaussianBlur(cur, (0, 0), 1.6)
        mx, my = np.meshgrid(xs * (ww / W) - 0.5, ys * (hh / H) - 0.5)
        acc += cv2.remap(b, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE) * wgt
    return acc


# ------------------------------------------------------------------ GLSL
BH_FRAG = r'''
uniform sampler2D uSky;  uniform vec4 uSkyRect;  uniform float uSkyMaxLod;
uniform sampler2D uH0;   uniform sampler2D uH1;  uniform sampler2D uH2;
uniform vec2 uMapSize;   uniform vec2 uMapHalf;  uniform float uSS;
uniform sampler2D uLut;  uniform float uLutStep; uniform float uLutN;
uniform float uTailRho;  uniform float uTailK;
uniform sampler2D uDisk; uniform sampler2D uBB;  uniform vec2 uBBRange;
uniform vec2 uC;  uniform float uS;  uniform float uThCrit;
uniform float uT; uniform float uW0; uniform float uForm; uniform float uGain;
uniform float uFlash; uniform float uFlashK; uniform float uDop; uniform float uTpk;
uniform float uTwist; uniform vec3 uGW; uniform vec2 uApp; uniform float uRing;
uniform float uKnee; uniform float uCeil;
uniform vec2 uProfF; uniform vec2 uProfA;   // outer fade (r0, r1) while forming / when formed

vec3 sky(vec2 fp, float lod){
    vec2 uv = (fp - uSkyRect.xy) / uSkyRect.zw;
    uv = 1.0 - abs(1.0 - mod(uv, 2.0));
    return textureLod(uSky, uv, lod).rgb;
}
// (delta, d beta / d rho) at scale-free radius th
vec2 lutAt(float th){
    if (th >= uTailRho) return vec2(uTailK / th, 1.0 + uTailK / (th * th));
    return texture(uLut, vec2((th / uLutStep + 0.5) / uLutN, 0.5)).rg;
}
float softclip(float x){
    return x < uKnee ? x : uKnee + (x - uKnee) / (1.0 + (x - uKnee) / (uCeil - uKnee));
}
vec4 diskShade(vec4 h){
    if (h.w < -0.5) return vec4(0.0);
    float r = h.x, phi = h.y, g = h.z;
    float om = uW0 * pow(6.0 / r, 1.5);
    float ph = phi - om * uT;
    float u = (r - 4.6) / 13.9;
    vec4 tx = textureLod(uDisk, vec2(ph / TAU, u), h.w + 0.3);
    // formation: infalling gas front + two spiral streams that wind up into the disk
    float armMix = smoothstep(0.72, 1.0, uForm);
    float prof = smoothstep(5.55, 6.5, r) * (1.0 - smoothstep(mix(uProfF.x, uProfA.x, armMix),
                                                               mix(uProfF.y, uProfA.y, armMix), r));
    float rf = mix(uProfF.y, 4.5, uForm);
    float front = smoothstep(rf - 0.3, rf + 2.2, r);
    float arms = 0.5 + 0.5 * cos(2.0 * ph + 4.5 * log(r));
    float a4 = arms * arms; a4 *= a4;
    float dens = tx.r * prof * front * mix(2.6 * a4 + 0.04, 1.0, armMix);
    float shock = exp(-pow((r - rf) / 0.9, 2.0)) * (1.0 - armMix) * 2.5 * prof * (0.3 + a4);
    // Novikov-Thorne-like temperature profile (peak ~ r 8), Doppler + gravitational shift
    float tem = uTpk * pow(8.2 / r, 0.75) * pow(max(1.0 - sqrt(4.8 / r), 0.0), 0.25) / 0.6157;
    tem *= 0.85 + 0.3 * tx.g;
    float Tobs = tem * g;
    vec3 col = texture(uBB, vec2(clamp((Tobs - uBBRange.x) / (uBBRange.y - uBBRange.x), 0.0, 1.0), 0.5)).rgb;
    // smooth brightness envelope (temperature, beaming, ignition) is soft-clipped BEFORE the
    // filament texture modulates it, so the Doppler-boosted limb keeps its texture
    float env = pow(tem / uTpk, 2.4) * pow(g, uDop);
    env *= 1.0 + uFlash * uFlashK * exp(-(r - 6.0) / 1.8);
    env = softclip(env * uGain);
    float em = env * (0.55 + 0.65 * tx.g) * (0.85 + 0.3 * tx.b);
    float a = clamp(dens * 1.35, 0.0, 0.985);
    vec3 c = col * (em * a + uGain * shock * dens);
    return vec4(c, a);
}
vec4 shadeSample(vec2 fp){
    vec2 d = fp - uC;
    float rr = length(d);
    vec2 dir = d / max(rr, 1e-4);
    float thc = uThCrit * uS;
    vec3 acc = vec3(0.0);
    float T = 1.0;
    vec2 mt = (d / uS + uMapHalf) * uSS;
    if (uGain > 0.0 && mt.x >= 0.0 && mt.y >= 0.0 && mt.x < uMapSize.x && mt.y < uMapSize.y){
        vec2 uv = (floor(mt) + 0.5) / uMapSize;
        vec4 c;
        c = diskShade(texture(uH0, uv)); acc += T * c.rgb; T *= 1.0 - c.a;
        c = diskShade(texture(uH1, uv)); acc += T * c.rgb * 0.85; T *= 1.0 - c.a;
        c = diskShade(texture(uH2, uv)); acc += T * c.rgb * 0.55; T *= 1.0 - c.a;
    }
    float cov = smoothstep(thc - 0.8, thc + 0.8, rr);
    vec3 bg = vec3(0.0);
    if (cov > 0.0){
        float th = max(rr / uS, uThCrit + 0.02);
        vec2 L = lutAt(th);
        float beta = th * uS - uS * L.x;
        float ang = uTwist * exp(-rr / 150.0);
        vec2 dd = rot2(-ang) * dir;
        float x = (rr - uGW.y) / uGW.z;
        beta += uGW.x * sin(x * TAU) * exp(-x * x * 0.6);
        // sky footprint of one sub-sample: radial magnification, tangential stretch, swirl shear.
        // Near the critical curve this diverges -> the prefiltered (mip) average, not noise.
        float foot = max(abs(L.y), abs(beta) / max(rr, 1.0));
        foot = max(foot, abs(beta) * abs(uTwist) / 150.0 * exp(-rr / 150.0));
        float lod = clamp(log2(max(foot * 0.7, 1.0)), 0.0, uSkyMaxLod);
        bg = sky(uC + dd * beta, lod) * cov;
    }
    // analytic photon ring: the n >= 2 images of the disk piled up on the critical curve,
    // Doppler-brighter on the approaching side
    float pr = exp(-pow((rr - thc * 1.012) / 0.9, 2.0)) * uRing * max(1.0 + 0.6 * dot(dir, uApp), 0.2);
    vec3 ring = vec3(1.0, 0.80, 0.58) * pr;
    return vec4(acc + T * (bg + ring), T * (1.0 - cov));
}
void main(){
    vec2 p = fpx();
    vec4 c = shadeSample(p + vec2(-0.25, -0.25)) + shadeSample(p + vec2(0.25, -0.25))
           + shadeSample(p + vec2(-0.25, 0.25)) + shadeSample(p + vec2(0.25, 0.25));
    fragColor = c * 0.25;
}
'''


class FX(core.Effect):
    name = 'space'

    def __init__(self):
        self.far, self.near, fs = K.load_layers(tuple(float(v) for v in BH_C))
        self.fs_xy, self.fs_rgb, self.fs_sig = fs['xy'], fs['rgb'], fs['sig']
        self.bh = BHData(shadow_px=SHADOW_PX, D=25.0, incl_deg=INCL, roll_deg=ROLL, box=BOX, ss=SS)
        self._prep_lens()
        (self.mid_xy, self.mid_flux, self.mid_col,
         self.tw_a, self.tw_f, self.tw_p) = K.mid_star_catalog()
        hero = np.array(K.HERO, np.float32)
        self.hero_xy = hero[:, :2]
        self.hero_flux = hero[:, 2]
        self.hero_col = K.blackbody_rgb(hero[:, 3])
        rng = np.random.default_rng(77)
        self.hero_ph = rng.uniform(0, 2 * np.pi, len(hero)).astype(np.float32)
        self.hero_ang = np.radians(rng.normal(0, 2.0, len(hero)) + 12.0).astype(np.float32)
        # out-of-focus dust motes in front of him (depth)
        m = 26
        r3 = np.random.default_rng(55)
        self.mote_xy = r3.uniform([-100, -100], [W + 100, H + 100], (m, 2)).astype(np.float32)
        self.mote_v = (r3.normal(0, 1, (m, 2)) * 4.0 + V_FAR * 6.0).astype(np.float32)
        self.mote_r = r3.uniform(5, 26, m).astype(np.float32)
        self.mote_b = r3.uniform(0.3, 1.0, m).astype(np.float32)
        self.mote_ph = r3.uniform(0, 2 * np.pi, m).astype(np.float32)
        self._gl = None
        self._static_maps = None
        X, Y = core.grid()
        dx, dy = X - BH_C[0], Y - BH_C[1]
        self._rr = np.sqrt(dx * dx + dy * dy).astype(np.float32)
        self._dir = (np.stack([dx, dy], -1) / np.maximum(self._rr, 1e-3)[..., None]).astype(np.float32)

    # ------------------------------------------------------------------ lens profile
    def _prep_lens(self):
        """Scale-free lens table delta(th) (th = screen radius / lens scale), softened beyond ~2x
        the shadow so the far field reads less like a twirl filter (the shadow, arch, photon ring
        and Einstein ring stay physical), + d beta / d rho for sky prefiltering, + the inverse
        table (primary image radius vs unlensed radius) for point sources."""
        b = self.bh
        rho = b.lut_rho.astype(np.float64)
        d = b.lut_delta.astype(np.float64)
        cap = d < -0.5
        r0, r1, k = LENS_SOFT
        wgt = 1.0 - k * np.clip((rho - r0) / (r1 - r0), 0, 1) ** 2 * (3 - 2 * np.clip((rho - r0) / (r1 - r0), 0, 1))
        dm = np.where(cap, -1.0, d * wgt)
        self.tail_rho = float(rho[-1])
        self.tail_K = float(dm[-1] * rho[-1])
        dd = np.gradient(np.where(cap, np.nan, dm), rho)
        dbdr = np.where(cap | ~np.isfinite(dd), 1e4, 1.0 - dd)
        # one-sided at the first valid sample (critical curve): derivative diverges there
        ic = np.where(cap)[0].max() + 1 if cap.any() else 0
        dbdr[ic] = 1e4
        self.lut_delta = dm.astype(np.float32)
        self.lut_dbdr = np.abs(dbdr).astype(np.float32)
        self.rho_crit = float(rho[ic])
        beta = rho - np.where(cap, 0.0, dm)
        k_ = np.arange(np.where(~(np.isfinite(b.lut_beta) & (beta > 0)))[0].max() + 1, len(rho))
        self._inv_beta, self._inv_rho = beta[k_], rho[k_]
        self._lut_rho32 = rho.astype(np.float32)

    def delta_np(self, th):
        """Vectorised softened delta(th) for unit lens scale, captured -> nan."""
        th = np.asarray(th, np.float32)
        d = np.interp(th, self._lut_rho32, self.lut_delta)
        d = np.where(th > self.tail_rho, self.tail_K / np.maximum(th, 1.0), d)
        return np.where(th < self.rho_crit, np.nan, d)

    def active(self, t):
        return t >= T_ON

    # ------------------------------------------------------------------ sky
    def _warp_layer(self, canvas, margin, off, z):
        # dst(x) = canvas((x - ZC) / z + ZC - off + margin)
        a = 1.0 / z
        M = np.float32([[a, 0, (1 - a) * ZC[0] - off[0] + margin[0]],
                        [0, a, (1 - a) * ZC[1] - off[1] + margin[1]]])
        return cv2.warpAffine(canvas, M, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_REFLECT)

    def _sky(self, t):
        dt = t - T0
        z = arrival_zoom(t)
        zprev = arrival_zoom(t - 0.5 / C.FPS)
        # radial motion blur length at the frame corner (px over half a frame)
        blur_px = abs(z - zprev) / max(z, 1e-3) * 800.0
        moving = blur_px > 0.35
        if blur_px > 1.5:   # radial motion blur of the far nebula during the hyperspace exit
            n = int(min(6, 2 + blur_px / 4))
            acc = np.zeros((H, W, 3), np.float32)
            for k in range(n):
                acc += self._warp_layer(self.far, K.FAR_M, V_FAR * dt, z + (zprev - z) * k / (n - 1))
            far = acc * (1.0 / n)
        else:
            far = self._warp_layer(self.far, K.FAR_M, V_FAR * dt, z)
        # faint far stars: exact float positions, motion-blurred by sub-sampling the streak
        q = self.fs_xy + V_FAR * dt
        nsub = int(np.clip(np.ceil(blur_px * 1.2), 1, 8)) if moving else 1
        for k in range(nsub):
            zk = z + (zprev - z) * (k / max(nsub - 1, 1)) if nsub > 1 else z
            far += splat_stars((q - ZC) * zk + ZC, self.fs_rgb * (1.0 / nsub), self.fs_sig)
        near = self._warp_layer(self.near, K.NEAR_M, V_NEAR * dt, z)
        U = far * (1.0 - near[..., 3:4]) + near[..., :3]
        # mid stars (lensable), twinkle, streaks while zooming
        from ..glfx import gl
        zm = 1.0 + (z - 1.0) * 1.3
        zmp = 1.0 + (zprev - 1.0) * 1.3
        p = (self.mid_xy + V_MID * dt - ZC) * zm + ZC
        pp = (self.mid_xy + V_MID * (dt - 0.5 / C.FPS) - ZC) * zmp + ZC
        tw = 1.0 + self.tw_a * np.sin(t * self.tw_f * 2 * np.pi + self.tw_p) \
            + 0.5 * self.tw_a * np.sin(t * self.tw_f * 5.3 + 2 * self.tw_p)
        L = np.linalg.norm(p - pp, axis=1) + 1.0
        f = self.mid_flux * tw / np.sqrt(L)
        col = np.concatenate([self.mid_col * f[:, None], np.ones((len(p), 1), np.float32)], 1)
        rad = (1.6 + 0.35 * np.clip(self.mid_flux, 0, 2)).astype(np.float32)
        spr = gl.sprites((W, H), p, pp if moving else None, rad, col, shape=0)
        U += spr[..., :3]
        return U

    # ------------------------------------------------------------------ lensing
    def _lens_maps(self, st):
        s = st['s']
        rr = self._rr
        d = np.nan_to_num(self.delta_np(rr / s), nan=0.0)
        beta = rr - s * d
        tw = st['twist']
        if tw != 0:   # same convention as the shader: sample direction rotated by +ang
            ang = tw * np.exp(-rr / 150.0)
            c, sn = np.cos(ang), np.sin(ang)
            dx = self._dir[..., 0] * c - self._dir[..., 1] * sn
            dy = self._dir[..., 0] * sn + self._dir[..., 1] * c
        else:
            dx, dy = self._dir[..., 0], self._dir[..., 1]
        amp, rad, wid = st['gw']
        if amp > 0.05:
            x = (rr - rad) / wid
            beta = beta + amp * np.sin(x * 2 * np.pi) * np.exp(-x * x * 0.6)
        mx = (BH_C[0] + dx * beta).astype(np.float32)
        my = (BH_C[1] + dy * beta).astype(np.float32)
        return mx, my

    def _lens_full(self, U, st):
        static = abs(st['s'] - 1.0) < 1e-4 and st['twist'] == 0 and st['gw'][0] <= 0.05
        if static:
            if self._static_maps is None:
                mx, my = self._lens_maps(st)
                self._static_maps = cv2.convertMaps(mx, my, cv2.CV_16SC2)
            m1, m2 = self._static_maps
            return cv2.remap(U, m1, m2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        mx, my = self._lens_maps(st)
        return cv2.remap(U, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    def _gl_init(self):
        from ..glfx import gl
        b = self.bh
        hm = b.hits                                     # (h, w, 3 hits, 3)
        texs = []
        for k in range(3):
            arr = np.concatenate([hm[:, :, k, :], b.lods[:, :, k:k + 1]], -1)
            texs.append(gl.texture(np.ascontiguousarray(arr.astype(np.float32)), linear=False))
        lut = np.stack([self.lut_delta, np.minimum(self.lut_dbdr, 1e4)], -1).astype(np.float32)[None]
        tl = gl.texture(lut, linear=True)
        td = gl.texture(b.disk, linear=True, repeat=True)
        td.build_mipmaps()
        tb = gl.texture(np.ascontiguousarray(b.bb[None, :, :].astype(np.float32)), linear=True)
        self._gl = dict(prog=gl.program(BH_FRAG), h=texs, lut=tl, disk=td, bb=tb,
                        lutN=float(lut.shape[1]), bbR=(float(b.bbT[0]), float(b.bbT[-1])),
                        msize=(float(hm.shape[1]), float(hm.shape[0])))

    def _bh_box(self, U, st, t):
        from ..glfx import gl
        import moderngl
        if self._gl is None:
            self._gl_init()
        G = self._gl
        bw, bh = BOX
        x0 = int(max(0, BH_C[0] - bw / 2)); x1 = int(min(W, BH_C[0] + bw / 2))
        y0 = int(max(0, BH_C[1] - bh / 2)); y1 = int(min(H, BH_C[1] + bh / 2))
        m = 160
        sx0, sy0 = max(0, x0 - m), max(0, y0 - m)
        sx1, sy1 = min(W, x1 + m), min(H, y1 + m)
        sky = gl.texture(np.ascontiguousarray(U[sy0:sy1, sx0:sx1]), linear=True)
        sky.build_mipmaps()
        sky.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        unis = {
            'uSkyRect': (float(sx0), float(sy0), float(sx1 - sx0), float(sy1 - sy0)),
            'uSkyMaxLod': float(np.floor(np.log2(max(sx1 - sx0, sy1 - sy0)))) - 1.0,
            'uMapSize': G['msize'], 'uMapHalf': (bw / 2.0, bh / 2.0), 'uSS': float(SS),
            'uLutStep': float(self.bh.lut_step), 'uLutN': G['lutN'],
            'uTailRho': float(self.tail_rho), 'uTailK': float(self.tail_K),
            'uBBRange': G['bbR'],
            'uC': (float(BH_C[0] + 0.5), float(BH_C[1] + 0.5)), 'uS': float(st['s']),
            'uThCrit': float(self.rho_crit),
            'uT': float(st['tdisk']), 'uW0': float(2 * np.pi / 3.4), 'uForm': float(st['form']),
            'uGain': float(st['gain']) * 1.05, 'uFlash': float(st['flash']), 'uFlashK': 1.8,
            'uDop': 2.8, 'uTpk': 4700.0, 'uTwist': float(st['twist']),
            'uGW': (float(st['gw'][0]), float(st['gw'][1]), float(st['gw'][2])),
            'uApp': (float(APPROACH[0]), float(APPROACH[1])),
            'uRing': float(st['gain']) * 0.55 * (1.0 + 0.8 * st['flash']),
            'uKnee': 0.9, 'uCeil': 2.6,
            'uProfF': (13.5, 17.0), 'uProfA': (9.0, 14.0),
        }
        tex = {'uSky': sky, 'uH0': G['h'][0], 'uH1': G['h'][1], 'uH2': G['h'][2],
               'uLut': G['lut'], 'uDisk': G['disk'], 'uBB': G['bb']}
        try:
            out = gl.render(G['prog'], (x1 - x0, y1 - y0), uniforms=unis, textures=tex, offset=(x0, y0))
        finally:
            sky.release()
        return out[..., :3], out[..., 3], (x0, y0, x1, y1)

    # ------------------------------------------------------------------ hero stars
    def _hero(self, t, st, z):
        from ..glfx import gl
        dt = t - T0
        zm = 1.0 + (z - 1.0) * 1.5
        p = (self.hero_xy + V_HERO * dt - ZC) * zm + ZC
        s = st['s']
        if s > 0:   # move the star to its primary lensed image (spikes stay unlensed)
            v = p - BH_C
            b = np.linalg.norm(v, axis=1)
            th = s * np.interp(b / s, self._inv_beta, self._inv_rho, right=np.nan)
            far = ~np.isfinite(th)
            K_ = self.tail_K * s * s
            th_far = 0.5 * (b + np.sqrt(b * b + 4 * K_))
            th = np.where(far, th_far, th)
            p = BH_C + v * (th / np.maximum(b, 1e-3))[:, None]
        tw = 1.0 + 0.10 * np.sin(t * 1.3 + self.hero_ph) + 0.05 * np.sin(t * 3.7 + 2 * self.hero_ph)
        P0, P1, R, Cc = [], [], [], []
        for k in range(len(p)):
            f = self.hero_flux[k] * tw[k]
            c = self.hero_col[k]
            P0.append(p[k]); P1.append(p[k]); R.append(2.4); Cc.append([*(c * f), 1])
            P0.append(p[k]); P1.append(p[k]); R.append(8.0); Cc.append([*(c * f * 0.03), 1])
            P0.append(p[k]); P1.append(p[k]); R.append(24.0); Cc.append([*(c * f * 0.004), 1])
            a = self.hero_ang[k]
            for j in range(4):
                ang = a + j * np.pi / 2
                dv = np.array([np.cos(ang), np.sin(ang)], np.float32)
                for L, wgt in ((8.0 * f ** 0.5, 0.10), (18.0 * f ** 0.5, 0.03), (34.0 * f ** 0.5, 0.008)):
                    P0.append(p[k]); P1.append(p[k] + dv * L); R.append(0.9)
                    Cc.append([*(c * f * wgt), 1])
        img = gl.sprites((W, H), np.array(P0), np.array(P1), np.array(R, np.float32),
                         np.array(Cc, np.float32), shape=0)
        return img[..., :3]

    # ------------------------------------------------------------------ hooks
    def background(self, S):
        t = S.t
        st = bh_state(t)
        z = arrival_zoom(t)
        U = self._sky(t)
        if st['s'] > 0:
            L = self._lens_full(U, st)
            box, vis, (x0, y0, x1, y1) = self._bh_box(U, st, t)
            # feathered blend of the GLSL box into the full-frame lens remap
            h, w = box.shape[:2]
            fy = np.minimum(np.arange(h), np.arange(h)[::-1]).astype(np.float32)
            fx = np.minimum(np.arange(w), np.arange(w)[::-1]).astype(np.float32)
            if x0 == 0: fx[:w // 2] = 1e3
            if x1 == W: fx[w // 2:] = 1e3
            if y0 == 0: fy[:h // 2] = 1e3
            if y1 == H: fy[h // 2:] = 1e3
            wgt = np.clip(np.minimum(fy[:, None], fx[None, :]) / 24.0, 0, 1)[..., None]
            L[y0:y1, x0:x1] = L[y0:y1, x0:x1] * (1 - wgt) + box * wgt
            U = L
            # shadow visibility crop for the protect-black bloom compensation in post()
            rs = int(np.ceil(self.rho_crit * st['s'] + 4))
            cx, cy = int(BH_C[0]), int(BH_C[1])
            bx0, by0 = cx - rs, cy - rs
            S.meta['_space_shadow'] = (np.ascontiguousarray(vis[by0 - y0:by0 - y0 + 2 * rs + 1,
                                                                bx0 - x0:bx0 - x0 + 2 * rs + 1]), (bx0, by0))
            S.meta['space_bh'] = dict(center=(float(BH_C[0]), float(BH_C[1])),
                                      shadow_r=float(self.rho_crit * st['s']),
                                      lens_r=float(2.35 * self.rho_crit * st['s']),
                                      strength=float(min(1.0, st['s'])))
        U += self._hero(t, st, z)
        self._glow(U, st, t)
        S.bg = S.bg + U
        S.person_offset = S.person_offset + float_offset(t)
        # cool space grade on the person
        S.person_grade.update(exposure=0.62, contrast=1.10, saturation=0.82,
                              tint=(0.88, 0.95, 1.10), ambient_rgb=(0.0015, 0.0022, 0.0045))
        S.vignette = max(S.vignette, 0.34)

    def _glow(self, U, st, t):
        """Seed glimmer while the hole nucleates, warm scattered haze around the disk, ignition
        flash on the approaching ISCO limb + anamorphic streak. Smooth fields -> 1/4 res. Nothing
        is ever added inside the event horizon."""
        lum = st['gain'] + 2.0 * st['flash']
        if st['seed'] <= 0.01 and lum <= 0.01:
            return
        q = 4
        X, Y = core.grid(H // q, W // q)
        Xf, Yf = X * q + q / 2, Y * q + q / 2
        dx, dy = Xf - BH_C[0], Yf - BH_C[1]
        r2 = dx * dx + dy * dy
        add = np.zeros((H // q, W // q, 3), np.float32)
        if st['seed'] > 0.01:
            pul = 0.8 + 0.2 * np.sin(t * 23.0)
            g = (np.exp(-r2 / (2 * 14.0 ** 2)) * 0.35 + np.exp(-r2 / (2 * 45.0 ** 2)) * 0.06
                 + np.exp(-r2 / (2 * 140.0 ** 2)) * 0.012) * st['seed'] * pul
            add += g[..., None] * np.array([0.55, 0.62, 1.0], np.float32)
        if lum > 0.01:
            c, s_ = np.cos(_rl), np.sin(_rl)
            u = dx * c + dy * s_
            v = -dx * s_ + dy * c
            haze = np.exp(-((u / 260.0) ** 2 + (v / 105.0) ** 2) * 2.0) * 0.014 \
                + np.exp(-((u / 480.0) ** 2 + (v / 320.0) ** 2) * 2.0) * 0.004
            add += (haze * st['gain'])[..., None] * np.array([1.0, 0.60, 0.32], np.float32)
            if st['flash'] > 0.01:
                f = st['flash']
                fp = BH_C + APPROACH * (95.0 * SHADOW_PX / 80.0)
                fx_, fy_ = Xf - fp[0], Yf - fp[1]
                f2 = fx_ * fx_ + fy_ * fy_
                fl = f * (np.exp(-f2 / (2 * 55.0 ** 2)) * 0.9 + np.exp(-f2 / (2 * 220.0 ** 2)) * 0.04
                          + np.exp(-f2 / (2 * 600.0 ** 2)) * 0.004)
                add += fl[..., None] * np.array([1.0, 0.86, 0.70], np.float32)
                # anamorphic streak (a lens artefact -> horizontal on screen), cool tint
                streak = f * np.exp(-(fy_ / 3.5) ** 2) * (np.exp(-np.abs(fx_) / 120.0) * 0.40
                                                          + np.exp(-np.abs(fx_) / 300.0) * 0.012)
                add += streak[..., None] * np.array([0.70, 0.82, 1.0], np.float32)
        add = cv2.resize(add, (W, H), interpolation=cv2.INTER_LINEAR)
        # keep the shadow pitch black: no scattered glow inside the event-horizon disk (mask at
        # full res, only over the crop around the hole)
        sh = self.rho_crit * max(st['s'], 1e-3)
        rs = int(np.ceil(sh * 1.2 + 3))
        cx, cy = int(BH_C[0]), int(BH_C[1])
        sl = (slice(cy - rs, cy + rs + 1), slice(cx - rs, cx + rs + 1))
        add[sl] *= core.smoothstep(sh * 1.0, sh * 1.2, self._rr[sl])[..., None]
        U += add

    def _rim(self, S, st):
        """Crisp warm rim light from the hole on the silhouette edges that face it: a ~3 px band
        (plus a soft 10 px falloff) inside the final, float-offset matte, per-pixel light
        direction, modulated by the surface. Full res over the person's bounding box only."""
        lum = st['gain'] + 0.4 * min(st['flash'], 1.0)
        if lum < 0.02:
            return None
        a_full = S.alpha
        y_lo = 640
        sub = a_full[y_lo::8, ::8]
        ys, xs = np.where(sub > 0.05)
        if len(ys) == 0:
            return None
        pad = 16
        xa, xb = int(max(0, xs.min() * 8 - pad)), int(min(W, xs.max() * 8 + 8 + pad))
        ya, yb = int(max(0, y_lo + ys.min() * 8 - pad)), H
        a = np.ascontiguousarray(a_full[ya:yb, xa:xb])
        X, Y = core.grid(yb - ya, xb - xa, (xa, ya))
        lp = BH_C + APPROACH * 60.0
        lx, ly = lp[0] - X, lp[1] - Y
        d = np.sqrt(lx * lx + ly * ly) + 1e-3
        lx, ly = lx / d, ly / d
        Xl, Yl = X - xa, Y - ya
        band = np.zeros_like(a)
        for k, wgt, blur in ((3.0, 1.0, 0.8), (10.0, 0.30, 2.5)):
            sh = cv2.remap(a, (Xl + k * lx).astype(np.float32), (Yl + k * ly).astype(np.float32),
                           cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            band += cv2.GaussianBlur(np.clip(a - sh, 0, 1), (0, 0), blur) * wgt
        band *= a
        fall = 1.0 / (1.0 + (d / 650.0) ** 2)
        fg = S.fg[ya:yb, xa:xb]
        sl = fg[..., 0] * 0.2126 + fg[..., 1] * 0.7152 + fg[..., 2] * 0.0722
        k_int = 0.42 * lum
        amt = band * fall * (0.35 + np.minimum(sl, 0.6)) * k_int
        rgb = amt[..., None] * np.array([1.0, 0.60, 0.30], np.float32)
        return core.Layer('add', rgb.astype(np.float32), box=(xa, ya))

    def front(self, S):
        """Defocused dust motes drifting in front of him (depth cue), catching the cool nebula light
        and, once it exists, the warm light of the hole; + the crisp warm rim from the hole."""
        t = S.t
        out = []
        st = bh_state(t)
        rim = self._rim(S, st)
        if rim is not None:
            out.append(rim)
        if t < T0 + 0.3:
            return out
        from ..glfx import gl
        dt = t - T0
        span = np.array([W + 200, H + 200], np.float32)
        p = (self.mote_xy + self.mote_v * dt + 100) % span - 100
        warm = min(1.0, st['gain'] + st['flash'])
        col = np.array([0.45, 0.65, 1.0], np.float32) * (1 - warm) + np.array([1.0, 0.68, 0.40], np.float32) * warm
        d = np.linalg.norm(p - BH_C, axis=1)
        lit = 0.35 + 0.65 * warm * np.exp(-d / 700.0) + 0.25 * (1 - warm)
        tw = 0.8 + 0.2 * np.sin(t * 0.9 + self.mote_ph)
        I = 0.020 * self.mote_b * lit * tw * _ss(T0 + 0.3, T0 + 2.0, t)
        c4 = np.concatenate([col[None] * I[:, None], np.ones((len(p), 1), np.float32)], 1)
        img = gl.sprites((W, H), p, None, self.mote_r, c4, shape=1)[..., :3]
        out.append(core.Layer('add', img))
        return out

    def post(self, S, img):
        """Protect black: the compositor blooms the HDR disk into the shadow, lifting the event
        horizon to grey. Pre-subtract exactly that bloom (computed with the compositor's own
        algorithm) wherever the bare shadow is visible, so after bloom the horizon stays black."""
        sv = S.meta.pop('_space_shadow', None)
        if sv is None or S.bloom <= 0:
            return img
        vis, (bx0, by0) = sv
        h, w = vis.shape
        if vis.max() < 0.02:
            return img
        reg = img[by0:by0 + h, bx0:bx0 + w]
        # only where the pixel is still the (near-black) shadow: if another module has drawn
        # over it since background() (portal plate, caption text), leave its bloom alone
        dark = np.clip(1.0 - reg.max(-1) / 0.04, 0.0, 1.0)
        m = vis * dark
        if m.max() < 0.02:
            return img
        pred = bloom_crop(img, (bx0, by0, w, h)) * (S.bloom * 0.55)
        reg -= pred * m[..., None]
        return img

    def lights(self, S):
        t = S.t
        st = bh_state(t)
        L = [
            # nebula / starlight: cool teal key from the upper left, magenta kick from the right
            core.Light((-150, 380), (0.28, 0.60, 1.0), intensity=0.55, radius=900, rim=0.5, diffuse=0.30),
            core.Light((1250, 1100), (0.55, 0.35, 1.0), intensity=0.20, radius=800, rim=0.5, diffuse=0.06),
        ]
        lum = st['gain'] + 0.4 * st['flash']
        if lum > 0.01:
            # soft warm wash only (the crisp rim is drawn at full res in front())
            lp = BH_C + APPROACH * 60.0
            L.append(core.Light((lp[0], lp[1]), (1.0, 0.58, 0.28), intensity=0.65 * lum,
                                radius=650, rim=0.10, diffuse=0.08, bg=0.0))
        if st['seed'] > 0.01:
            L.append(core.Light(tuple(BH_C), (0.55, 0.6, 1.0), intensity=0.4 * st['seed'], radius=400,
                                rim=0.6, diffuse=0.1))
        return L
