"""Static deep-space layers for the `space` module (generated once, cached).

far canvas  : base gradient + galactic band + emission nebula + dust + distant galaxies +
              ~11k faint stars, linear RGB, frame + margin (drifts slowly).
near canvas : sharper nebula wisps (rgb emission) + foreground dust (alpha = absorption),
              drifts faster (parallax).
Star catalogs for the per-frame twinkling layers are generated from fixed seeds.
"""
import os
import numpy as np
import cv2
from .. import config as C

CACHE = os.path.join(C.DATA, 'cache', 'space')
W, H = C.W, C.H

FAR_M = (140, 150)        # canvas margin (x, y) around the frame
NEAR_M = (260, 170)

# galactic band: line through A with direction DIR (frame px); leads from lower-left (behind
# him) up to the black hole on the upper right.
BAND_A = np.array([0.0, 1180.0])
BAND_DIR = np.array([1080.0, -760.0]) / np.hypot(1080.0, 760.0)
BAND_N = np.array([-BAND_DIR[1], BAND_DIR[0]])

NEB_FAR = r'''
uniform vec2 uOff;
uniform vec2 uBandA; uniform vec2 uBandN;
uniform vec2 uGal;     // main galaxy centre (frame px)
float bandD(vec2 p){ return dot(p - uBandA, uBandN); }
vec3 galaxy(vec2 p, vec2 c, float R, float incl, float pa, float seed, float bright){
    vec2 g = rot2(pa) * (p - c);
    g.y /= cos(incl);
    float r = length(g) / R;
    if (r > 2.4) return vec3(0.0);
    float th = atan(g.y, g.x);
    float pitch = 0.36;
    float sp = th - log(r + 0.03) / tan(pitch);
    float wob = 0.35 * fbm3(vec3(g / R * 3.0, seed + 1.0), 3);
    float arm = pow(0.5 + 0.5 * cos(2.0 * sp + wob), 1.6);
    float lane = pow(0.5 + 0.5 * cos(2.0 * sp - 0.8 + wob), 10.0);
    float knots = smoothstep(0.58, 0.85, fbm3(vec3(g / R * 8.0, seed), 4) * 0.5 + 0.5);
    float disk = exp(-r / 0.32);
    float bulge = exp(-pow(r / 0.06, 0.75)) * 1.4 + exp(-r / 0.12) * 0.3;
    float fine = 0.7 + 0.6 * (fbm3(vec3(g / R * 16.0, seed + 3.0), 3) * 0.5 + 0.5);
    vec3 col = vec3(1.0, 0.80, 0.55) * bulge
             + vec3(0.60, 0.72, 1.0) * disk * (0.35 + 0.8 * arm) * fine
             + vec3(1.0, 0.40, 0.65) * disk * arm * knots * 1.2;
    col *= 1.0 - 0.65 * lane * smoothstep(0.06, 0.25, r) * exp(-r / 0.9);
    return col * bright;
}
void main(){
    vec2 p = px() + uOff;
    vec2 q = p / 1000.0;
    float d = bandD(p);
    float bandW = exp(-pow(d / 400.0, 2.0));
    float bandC = exp(-pow(d / 170.0, 2.0));
    float lo = fbm3(vec3(q * 1.1, 0.7), 3) * 0.5 + 0.5;
    float pres = clamp(bandW * 1.05 + smoothstep(0.55, 0.75, lo) * 0.55, 0.0, 1.3);
    vec3 P = vec3(q * 1.6, 0.37);
    vec2 w = vec2(fbm3(P, 4), fbm3(P + vec3(5.2, 1.3, 7.7), 4));
    vec3 Q = vec3(q * 2.5 + 0.55 * w, 3.1);
    float c1 = fbm3(Q, 7) * 0.5 + 0.5;
    float b1 = 1.0 - abs(fbm3(Q * 1.35 + vec3(4.0, 1.0, 2.0), 6));
    float dens = pres * pow(smoothstep(0.30, 0.86, c1), 1.5) * (0.5 + 0.7 * b1);
    float cm = fbm3(vec3(q * 1.3 + 0.3 * w, 11.0), 4) * 0.5 + 0.5;
    vec3 ha = vec3(1.0, 0.13, 0.40);
    vec3 o3 = vec3(0.06, 0.72, 0.85);
    vec3 bl = vec3(0.20, 0.30, 1.0);
    vec3 neb = mix(o3, ha, smoothstep(0.40, 0.62, cm));
    neb = mix(neb, vec3(0.55, 0.95, 1.0), smoothstep(0.8, 1.2, dens) * 0.45);
    vec3 col = neb * pow(dens, 1.9) * 0.20;
    col += bl * pow(lo, 3.0) * 0.010 * (0.3 + bandW);
    float fil = ridged3(vec3(q * 3.6 + 0.6 * w, 9.0), 6);
    col += mix(o3, ha, smoothstep(0.40, 0.62, cm)) * pow(fil, 8.0) * pres * 0.06;
    col += vec3(0.95, 0.88, 0.78) * bandC * 0.030 * (0.5 + 0.7 * (fbm3(vec3(q * 7.0, 21.0), 4) * 0.5 + 0.5));
    float dn = fbm3(vec3(q * 2.4 + 0.5 * w.yx, 13.0), 7) * 0.5 + 0.5;
    float dust = smoothstep(0.53, 0.66, dn) * (0.30 + 0.70 * bandW);
    float rim = smoothstep(0.47, 0.53, dn) * (1.0 - smoothstep(0.53, 0.58, dn));
    col *= 1.0 - 0.92 * dust;
    col += vec3(1.0, 0.62, 0.45) * rim * dens * 0.035;
    vec3 base = mix(vec3(0.0012, 0.0013, 0.0034), vec3(0.003, 0.0028, 0.007), bandW);
    col += base;
    col += galaxy(p, uGal, 95.0, 1.08, 0.62, 2.0, 0.42) * (1.0 - 0.6 * dust);
    col += galaxy(p, vec2(985.0, 210.0), 20.0, 1.30, -0.4, 5.0, 0.20);
    col += galaxy(p, vec2(78.0, 760.0), 15.0, 0.6, 1.2, 7.0, 0.16);
    col += galaxy(p, vec2(640.0, 140.0), 11.0, 1.4, 0.2, 9.0, 0.15);
    fragColor = vec4(col, 1.0);
}
'''

NEB_NEAR = r'''
uniform vec2 uOff;
uniform vec2 uBandA; uniform vec2 uBandN;
void main(){
    vec2 p = px() + uOff;
    vec2 q = p / 1000.0;
    float d = dot(p - uBandA, uBandN);
    vec3 P = vec3(q * 1.9, 41.0);
    vec2 w = vec2(fbm3(P, 5), fbm3(P + vec3(3.3, 7.1, 2.2), 5));
    float n = fbm3(vec3(q * 3.0 + w * 1.1, 44.0), 7) * 0.5 + 0.5;
    float fil = ridged3(vec3(q * 4.2 + w * 1.3, 47.0), 6);
    // foreground dust: framing clouds on the left and right edges, low centre
    float side = smoothstep(260.0, 0.0, p.x) + smoothstep(820.0, 1150.0, p.x) * smoothstep(1500.0, 780.0, p.y) * 0.8
               + smoothstep(950.0, 1500.0, p.y) * 0.7;
    float keep = 1.0 - exp(-pow(length((p - vec2(840.0, 640.0)) / vec2(330.0, 250.0)), 4.0) * 0.0) ;
    float dust = smoothstep(0.50, 0.74, n) * clamp(side, 0.0, 1.0) * 0.82;
    // thin glowing wisps (in front of everything far), stronger near the band
    float bw = exp(-pow(d / 420.0, 2.0));
    float wisp = pow(fil, 7.0) * smoothstep(0.35, 0.65, n) * (0.25 + bw);
    vec3 wc = mix(vec3(0.2, 0.75, 0.95), vec3(0.95, 0.35, 0.75), smoothstep(0.4, 0.7, fbm3(vec3(q * 1.3, 49.0), 3) * 0.5 + 0.5));
    vec3 col = wc * wisp * 0.05;
    // dust edges catch a little light
    col += vec3(0.9, 0.5, 0.35) * smoothstep(0.44, 0.52, n) * (1.0 - smoothstep(0.52, 0.6, n)) * clamp(side, 0.0, 1.0) * 0.004;
    fragColor = vec4(col, dust);
}
'''


def blackbody_rgb(T):
    """Approximate linear-sRGB colour of a blackbody, normalised (max = 1). T in K (array)."""
    T = np.asarray(T, np.float64) / 100.0
    r = np.where(T <= 66, 255.0, 329.698727446 * np.power(np.maximum(T - 60, 1e-3), -0.1332047592))
    g = np.where(T <= 66, 99.4708025861 * np.log(np.maximum(T, 1e-3)) - 161.1195681661,
                 288.1221695283 * np.power(np.maximum(T - 60, 1e-3), -0.0755148492))
    b = np.where(T >= 66, 255.0, np.where(T <= 19, 0.0, 138.5177312231 * np.log(np.maximum(T - 10, 1e-3)) - 305.0447927307))
    rgb = np.clip(np.stack([r, g, b], -1) / 255.0, 0, 1)
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return (lin / np.maximum(lin.max(-1, keepdims=True), 1e-6)).astype(np.float32)


def _star_temps(rng, n):
    T = 10 ** rng.normal(3.78, 0.16, n)
    return np.clip(T, 2700, 22000)


def _band_density(xy):
    d = (xy - BAND_A) @ BAND_N
    return np.exp(-(d / 260.0) ** 2)


def faint_star_catalog(seed=101):
    rng = np.random.default_rng(seed)
    cw, ch = W + 2 * FAR_M[0], H + 2 * FAR_M[1]
    n_uni, n_band, n_dust = 9000, 9000, 30000
    xy_u = rng.uniform([0, 0], [cw, ch], (n_uni, 2)) - FAR_M
    # band stars: sample along the band with gaussian across
    s = rng.uniform(-300, 1700, n_band)
    a = rng.normal(0, 190, n_band)
    xy_b = BAND_A + s[:, None] * BAND_DIR + a[:, None] * BAND_N
    # two loose open clusters
    cl = [np.array([330.0, 640.0]), np.array([960.0, 980.0])]
    xy_c = np.concatenate([c + rng.normal(0, 38, (260, 2)) for c in cl])
    # star dust: very faint unresolved stars, denser in the band
    sd = rng.uniform(-300, 1700, n_dust // 2)
    ad = rng.normal(0, 260, n_dust // 2)
    xy_d = np.concatenate([BAND_A + sd[:, None] * BAND_DIR + ad[:, None] * BAND_N,
                           rng.uniform([0, 0], [cw, ch], (n_dust // 2, 2)) - FAR_M])
    xy = np.concatenate([xy_u, xy_b, xy_c, xy_d])
    n = len(xy)
    u = rng.uniform(0, 1, n)
    flux = 0.016 * (1 - u * 0.9995) ** (-1 / 1.5)
    flux = np.minimum(flux, 0.7)
    nd = len(xy_d)
    flux[-nd:] = rng.uniform(0.004, 0.016, nd)
    T = _star_temps(rng, n)
    col = blackbody_rgb(T)
    sig = rng.uniform(0.48, 0.66, n) + 0.3 * np.clip(flux / 0.6, 0, 1)
    return xy.astype(np.float32), flux.astype(np.float32), col, sig.astype(np.float32)


def mid_star_catalog(seed=202, n=520):
    rng = np.random.default_rng(seed)
    m = 320
    xy = rng.uniform([-m, -m], [W + m, H + m], (n, 2)).astype(np.float32)
    u = rng.uniform(0, 1, n)
    flux = np.minimum(0.22 * (1 - u * 0.999) ** (-1 / 1.45), 2.2).astype(np.float32)
    T = _star_temps(rng, n)
    col = blackbody_rgb(T)
    tw_a = rng.uniform(0.05, 0.22, n).astype(np.float32)
    tw_f = rng.uniform(0.25, 1.6, n).astype(np.float32)
    tw_p = rng.uniform(0, 2 * np.pi, n).astype(np.float32)
    return xy, flux, col, tw_a, tw_f, tw_p


# hero stars (unlensed, with diffraction spikes): hand-placed away from the black hole and
# from his face; (x, y, flux, temperature)
HERO = [(132.0, 118.0, 5.5, 11000), (470.0, 300.0, 3.2, 4200), (1010.0, 95.0, 3.8, 15000),
        (60.0, 1010.0, 3.0, 7000), (300.0, 560.0, 2.4, 9000), (1040.0, 1130.0, 3.2, 5200),
        (585.0, 470.0, 2.0, 18000), (205.0, 1290.0, 2.6, 13000), (1000.0, 300.0, 2.2, 6000)]


def _render_stars(size, offset, xy, flux, col, sig):
    from ..glfx import gl
    radius = sig * 2.0
    c4 = np.concatenate([col * flux[:, None], np.ones((len(xy), 1), np.float32)], 1)
    # peak of the sprite gaussian exp(-2 d2) with d in units of radius: value = color at centre
    img = gl.sprites(size, xy, None, radius, c4, offset=offset, shape=0)
    return img[..., :3]


def build_far(gal=(190.0, 300.0)):
    from ..glfx import gl
    cw, ch = W + 2 * FAR_M[0], H + 2 * FAR_M[1]
    prog = gl.program(NEB_FAR)
    off = (-FAR_M[0], -FAR_M[1])
    img = gl.render(prog, (cw, ch), uniforms={'uOff': off, 'uBandA': tuple(BAND_A), 'uBandN': tuple(BAND_N),
                                              'uGal': gal})[..., :3]
    xy, flux, col, sig = faint_star_catalog()
    # stars are dimmed where dust is thick: sample the nebula luminance as a proxy is not
    # possible (dust is multiplicative), so render stars then apply a dust-only pass
    stars = _render_stars((cw, ch), off, xy, flux, col, sig)
    dust = _dust_mask((cw, ch), off)
    img = img + stars * (1.0 - 0.9 * dust[..., None])
    return img.astype(np.float32)


DUST_ONLY = r'''
uniform vec2 uOff; uniform vec2 uBandA; uniform vec2 uBandN;
void main(){
    vec2 p = px() + uOff; vec2 q = p / 1000.0;
    float d = dot(p - uBandA, uBandN);
    float bandW = exp(-pow(d / 400.0, 2.0));
    vec3 P = vec3(q * 1.6, 0.37);
    vec2 w = vec2(fbm3(P, 4), fbm3(P + vec3(5.2, 1.3, 7.7), 4));
    float dn = fbm3(vec3(q * 2.4 + 0.5 * w.yx, 13.0), 7) * 0.5 + 0.5;
    float dust = smoothstep(0.53, 0.66, dn) * (0.30 + 0.70 * bandW);
    fragColor = vec4(dust, 0, 0, 1);
}
'''


def _dust_mask(size, off):
    from ..glfx import gl
    prog = gl.program(DUST_ONLY)
    return gl.render(prog, size, uniforms={'uOff': off, 'uBandA': tuple(BAND_A), 'uBandN': tuple(BAND_N)})[..., 0]


def build_near():
    from ..glfx import gl
    cw, ch = W + 2 * NEAR_M[0], H + 2 * NEAR_M[1]
    prog = gl.program(NEB_NEAR)
    img = gl.render(prog, (cw, ch), uniforms={'uOff': (-NEAR_M[0], -NEAR_M[1]), 'uBandA': tuple(BAND_A),
                                              'uBandN': tuple(BAND_N)})
    return img.astype(np.float32)


def load_layers(version=5, verbose=True):
    os.makedirs(CACHE, exist_ok=True)
    pf = os.path.join(CACHE, f'far_v{version}.npy')
    pn = os.path.join(CACHE, f'near_v{version}.npy')
    if os.path.exists(pf) and os.path.exists(pn):
        return np.load(pf).astype(np.float32), np.load(pn).astype(np.float32)
    if verbose:
        print('[space] building nebula layers ...', flush=True)
    far = build_far()
    near = build_near()
    np.save(pf + '.tmp.npy', far.astype(np.float16))
    os.replace(pf + '.tmp.npy', pf)
    np.save(pn + '.tmp.npy', near.astype(np.float16))
    os.replace(pn + '.tmp.npy', pn)
    return far, near
