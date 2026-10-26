"""Black-hole precomputation for the `space` module (cached to <DATA>/cache/space/).

Physics (units G = c = M = 1, horizon r = 2, photon sphere r = 3, ISCO r = 6):
null geodesics of the Schwarzschild metric integrated in Cartesian form
    x'' = -3 h^2 x / r^5,      h = |x x v| (conserved)
which reproduces the exact photon orbit equation u'' + u = 3 M u^2.

Camera: static pinhole-ish camera at distance D looking at the hole. Screen offsets are mapped
to view angles with an equidistant (angle-proportional) projection psi = rho / f.

Outputs
  * lut      : 1-D background lensing table. Because the hole is spherically symmetric, the
               sky direction seen through screen radius rho only depends on rho:
               the unlensed screen radius is beta(rho) = rho - delta(rho)  (signed, may wrap).
  * hits     : per super-sample of a box around the hole, the first 3 crossings of the
               (tilted) equatorial disk plane: radius, azimuth in the disk frame and the
               redshift factor g (gravitational + Doppler for a Keplerian disk).
  * disk tex : polar turbulence texture (phi x r) for the accretion disk.
  * bb lut   : blackbody colour table (linear sRGB chromaticity) vs temperature.
"""
import os, json, hashlib, time
import numpy as np
from .. import config as C

CACHE = os.path.join(C.DATA, 'cache', 'space')

R_HOR = 2.0
B_CRIT = 3.0 * np.sqrt(3.0)          # critical impact parameter 5.196 M


def _acc(x, h2):
    r2 = (x * x).sum(1)
    r5 = r2 * r2 * np.sqrt(r2)
    return x * (-3.0 * h2 / r5)[:, None]


def _rk4(x, v, h2, dt):
    d = dt[:, None]
    k1x, k1v = v, _acc(x, h2)
    k2x, k2v = v + 0.5 * d * k1v, _acc(x + 0.5 * d * k1x, h2)
    k3x, k3v = v + 0.5 * d * k2v, _acc(x + 0.5 * d * k2x, h2)
    k4x, k4v = v + d * k3v, _acc(x + d * k3x, h2)
    return (x + d / 6.0 * (k1x + 2 * k2x + 2 * k3x + k4x),
            v + d / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v))


def camera_dirs(dx, dy, f):
    """Screen offsets (px, y down) -> unit ray directions (camera looks +z), equidistant."""
    rho = np.sqrt(dx * dx + dy * dy)
    psi = rho / f
    s = np.where(rho > 1e-9, np.sin(psi) / np.maximum(rho, 1e-9), 1.0 / f)
    return np.stack([dx * s, dy * s, np.cos(psi)], 1)


def trace_disk(D, dirs, n, e1, e2, r_rec=(4.6, 18.5), max_hits=3, r_stop=21.0,
               dt_k=0.035, max_steps=5000, log=None):
    """Trace rays from (0,0,-D) along `dirs`; record up to `max_hits` disk-plane crossings."""
    N = len(dirs)
    x = np.zeros((N, 3)); x[:, 2] = -D
    v = dirs.astype(np.float64).copy()
    L = np.cross(x, v)
    h2 = (L * L).sum(1)
    Lphys = -(L @ n)                         # photon angular momentum about the disk axis
    hits = np.zeros((N, max_hits, 3), np.float32)
    nh = np.zeros(N, np.int32)
    sprev = x @ n
    idx = np.arange(N)
    t0 = time.time()
    for step in range(max_steps):
        if len(idx) == 0:
            break
        xi, vi = x[idx], v[idx]
        r = np.sqrt((xi * xi).sum(1))
        dt = np.clip(dt_k * (r - 1.2), 0.004, 1.5)
        xn, vn = _rk4(xi, vi, h2[idx], dt)
        sp = sprev[idx]
        sn = xn @ n
        cr = np.where(sp * sn < 0)[0]
        if len(cr):
            tc = sp[cr] / (sp[cr] - sn[cr])
            xc = xi[cr] + tc[:, None] * (xn[cr] - xi[cr])
            rc = np.sqrt((xc * xc).sum(1))
            ok = (rc >= r_rec[0]) & (rc <= r_rec[1])
            cr, xc, rc = cr[ok], xc[ok], rc[ok]
            gi = idx[cr]
            k = nh[gi]
            sel = k < max_hits
            gi, k, xc, rc = gi[sel], k[sel], xc[sel], rc[sel]
            phi = np.arctan2(xc @ e2, xc @ e1)
            om = rc ** -1.5
            g = np.sqrt(np.maximum(1 - 3.0 / rc, 1e-6)) / (1.0 - om * Lphys[gi])
            hits[gi, k, 0] = rc
            hits[gi, k, 1] = phi
            hits[gi, k, 2] = g
            nh[gi] += 1
        x[idx], v[idx], sprev[idx] = xn, vn, sn
        rn = np.sqrt((xn * xn).sum(1))
        out = (rn > r_stop) & ((xn * vn).sum(1) > 0)
        done = (rn < R_HOR * 1.0005) | out | (nh[idx] >= max_hits)
        idx = idx[~done]
        if log and step % 200 == 0:
            log(f'  disk trace step {step} active {len(idx)} ({time.time() - t0:.1f}s)')
    return hits


def trace_lut(D, f, rho_max, step_px):
    """1-D background table: screen radius rho (px) -> signed unlensed radius beta (px).
    Rays escaping to large r keep their asymptotic direction; captured rays -> nan."""
    rho = np.arange(0, rho_max + step_px, step_px, dtype=np.float64)
    dirs = camera_dirs(rho, np.zeros_like(rho), f)
    N = len(rho)
    x = np.zeros((N, 3)); x[:, 2] = -D
    v = dirs.copy()
    L = np.cross(x, v)
    h2 = (L * L).sum(1)
    cap = np.zeros(N, bool)
    idx = np.arange(N)
    for step in range(200000):
        if len(idx) == 0:
            break
        xi, vi = x[idx], v[idx]
        r = np.sqrt((xi * xi).sum(1))
        dt = np.clip(0.01 * (r - 1.5), 0.001, 200.0)
        xn, vn = _rk4(xi, vi, h2[idx], dt)
        x[idx], v[idx] = xn, vn
        rn = np.sqrt((xn * xn).sum(1))
        c = rn < R_HOR * 1.0005
        cap[idx[c]] = True
        done = c | ((rn > 4e4) & ((xn * vn).sum(1) > 0))
        idx = idx[~done]
    # remaining deflection beyond r_end (weak field, ~2 b / r per side) is < 1e-4 rad: ignore
    theta = np.arctan2(v[:, 0], v[:, 2])     # signed final angle from the optical axis
    beta = f * theta
    beta[cap] = np.nan
    return rho, beta


def blackbody_lut(n=512, t0=1000.0, t1=40000.0):
    """Linear-sRGB chromaticity of a blackbody (normalised so max channel = 1), plus the
    relative luminance curve (not used for brightness, which is art-directed)."""
    T = np.linspace(t0, t1, n)
    lam = np.linspace(380e-9, 780e-9, 81)
    # CIE 1931 2deg approximations (Wyman, Sloan & Shirley 2013 multi-lobe fit)
    l_nm = lam * 1e9

    def g(x, mu, s1, s2):
        return np.exp(-0.5 * ((x - mu) / np.where(x < mu, s1, s2)) ** 2)
    xb = 1.056 * g(l_nm, 599.8, 37.9, 31.0) + 0.362 * g(l_nm, 442.0, 16.0, 26.7) - 0.065 * g(l_nm, 501.1, 20.4, 26.2)
    yb = 0.821 * g(l_nm, 568.8, 46.9, 40.5) + 0.286 * g(l_nm, 530.9, 16.3, 31.1)
    zb = 1.217 * g(l_nm, 437.0, 11.8, 36.0) + 0.681 * g(l_nm, 459.0, 26.0, 13.8)
    h, c, k = 6.626e-34, 2.998e8, 1.381e-23
    out = np.zeros((n, 3), np.float32)
    for j, t in enumerate(T):
        B = 1.0 / (lam ** 5 * (np.exp(h * c / (lam * k * t)) - 1.0))
        X, Y, Z = (B * xb).sum(), (B * yb).sum(), (B * zb).sum()
        rgb = np.array([3.2406 * X - 1.5372 * Y - 0.4986 * Z,
                        -0.9689 * X + 1.8758 * Y + 0.0415 * Z,
                        0.0557 * X - 0.2040 * Y + 1.0570 * Z])
        rgb = np.maximum(rgb, 0)
        out[j] = rgb / rgb.max()
    return T.astype(np.float32), out


def _fft_noise(shape, rng, spec):
    """Periodic gaussian noise with a custom amplitude spectrum spec(ky, kx) (cycles/axis)."""
    h, w = shape
    ky = np.fft.fftfreq(h)[:, None] * h
    kx = np.fft.fftfreq(w)[None, :] * w
    A = spec(ky, kx)
    A[0, 0] = 0
    ph = rng.standard_normal((h, w)) + 1j * rng.standard_normal((h, w))
    n = np.real(np.fft.ifft2(A * ph))
    return (n / (n.std() + 1e-9)).astype(np.float32)


def disk_texture(seed=7, w=2048, h=512):
    """Polar disk texture: x = azimuth (periodic, 0..2pi), y = radius (r 4.6 .. 18.5).
    R: density (optical depth-ish), G: hot filaments, B: fine streak detail."""
    rng = np.random.default_rng(seed)
    # wavelengths: x in px of 2048/rev, y in px of 512 per 13.9 M (~ isotropic at r ~ 9)
    rings = _fft_noise((h, 1), rng, lambda ky, kx: (1 + (np.abs(ky) / 18.0) ** 2) ** -0.9 * (np.abs(ky) > 1.5))
    rings = np.repeat(rings, w, 1)
    streak = _fft_noise((h, w), rng, lambda ky, kx: (1 + (ky / 22.0) ** 2 + (kx / 4.0) ** 2) ** -1.05)
    fine = _fft_noise((h, w), rng, lambda ky, kx: (1 + (ky / 90.0) ** 2 + (kx / 14.0) ** 2) ** -1.1)
    clump = _fft_noise((h, w), rng, lambda ky, kx: (1 + (ky / 30.0) ** 2 + (kx / 12.0) ** 2) ** -1.4)
    fil = _fft_noise((h, w), rng, lambda ky, kx: (1 + (ky / 60.0) ** 2 + (kx / 6.0) ** 2) ** -1.0)
    dens = 0.62 + 0.20 * rings + 0.22 * streak + 0.10 * fine + 0.10 * clump
    # thin dark gaps (lanes) from the ring field
    gaps = np.clip(1.0 - 2.5 * np.maximum(np.abs(rings) - 1.6, 0), 0.25, 1.0)
    dens = np.clip(dens * gaps, 0.02, 1.4)
    hot = np.clip(0.5 + 0.35 * fil + 0.25 * clump, 0, 1.5) ** 1.6
    tex = np.stack([dens, hot, 0.5 + 0.5 * np.tanh(fine), np.ones_like(dens)], -1).astype(np.float32)
    return tex


def hit_lods(hits, ss, tex_w=2048, tex_h=512, r_span=13.9):
    """Per super-sample texture LOD for the disk lookups: screen-space footprint of one OUTPUT
    pixel in disk-texture texels (from finite differences of r and phi between neighbouring
    samples of the same image order). Returns (h, w, 3) float32, -1 where no hit."""
    h, w = hits.shape[:2]
    out = np.full((h, w, hits.shape[2]), -1.0, np.float32)
    for k in range(hits.shape[2]):
        r = hits[:, :, k, 0].astype(np.float32)
        ph = hits[:, :, k, 1].astype(np.float32)
        v = r > 0
        fr = np.zeros((h, w), np.float32)
        fp = np.zeros((h, w), np.float32)
        for ax in (0, 1):
            for sh in (1, -1):
                r2 = np.roll(r, sh, ax)
                p2 = np.roll(ph, sh, ax)
                v2 = np.roll(v, sh, ax) & v
                dr = np.abs(r2 - r)
                dp = np.abs(np.angle(np.exp(1j * (p2 - ph)))).astype(np.float32)
                cont = v2 & (dr < 1.5) & (dp < 0.6)
                fr = np.maximum(fr, np.where(cont, dr, 0))
                fp = np.maximum(fp, np.where(cont, dp, 0))
        fr_t = fr * ss * tex_h / r_span
        fp_t = fp * ss * tex_w / (2 * np.pi)
        lod = np.log2(np.maximum(np.maximum(fr_t, fp_t * 0.5), 1.0))
        out[:, :, k] = np.where(v, lod, -1.0)
    return out


class BHData:
    """Loads (or builds and caches) everything the black-hole shader needs."""

    def __init__(self, shadow_px=84.0, D=25.0, incl_deg=7.0, roll_deg=-9.0,
                 box=(600, 460), ss=2, verbose=True):
        self.params = dict(shadow_px=shadow_px, D=D, incl_deg=incl_deg, roll_deg=roll_deg,
                           box=list(box), ss=ss, v=6)
        key = hashlib.md5(json.dumps(self.params, sort_keys=True).encode()).hexdigest()[:10]
        os.makedirs(CACHE, exist_ok=True)
        path = os.path.join(CACHE, f'bh_{key}.npz')
        psi_c = np.arcsin(B_CRIT / D)                 # starless-model capture angle
        self.f = shadow_px / psi_c
        self.theta_crit = shadow_px
        log = print if verbose else None
        if os.path.exists(path):
            z = np.load(path)
            self.hits = z['hits'].astype(np.float32)
            self.lut_rho, self.lut_beta = z['lut_rho'], z['lut_beta']
            self.disk = z['disk'].astype(np.float32)
            self.bbT, self.bb = z['bbT'], z['bb']
        else:
            t0 = time.time()
            i, rl = np.radians(incl_deg), np.radians(roll_deg)
            n = np.array([0.0, -np.cos(i), -np.sin(i)])        # disk normal: up + toward camera
            Rz = np.array([[np.cos(rl), -np.sin(rl), 0], [np.sin(rl), np.cos(rl), 0], [0, 0, 1]])
            n = Rz @ n
            e1 = Rz @ np.array([1.0, 0.0, 0.0])
            e1 = e1 - n * (e1 @ n); e1 /= np.linalg.norm(e1)
            e2 = np.cross(n, e1)
            bw, bh = box
            xs = (np.arange(bw * ss) + 0.5) / ss - bw / 2
            ys = (np.arange(bh * ss) + 0.5) / ss - bh / 2
            X, Y = np.meshgrid(xs, ys)
            dirs = camera_dirs(X.ravel(), Y.ravel(), self.f)
            if log:
                log(f'[space] tracing {len(dirs)} disk rays (f={self.f:.1f})')
            hits = trace_disk(D, dirs, n, e1, e2, log=log)
            self.hits = hits.reshape(bh * ss, bw * ss, 3, 3)
            if log:
                log(f'[space] disk rays done {time.time() - t0:.1f}s; tracing LUT')
            self.lut_rho, self.lut_beta = trace_lut(D, self.f, 420.0, 0.1)
            self.disk = disk_texture()
            self.bbT, self.bb = blackbody_lut()
            np.savez(path, hits=self.hits.astype(np.float32), lut_rho=self.lut_rho, lut_beta=self.lut_beta,
                     disk=self.disk.astype(np.float16), bbT=self.bbT, bb=self.bb)
            if log:
                log(f'[space] BH precompute total {time.time() - t0:.1f}s -> {path}')
        lp = path.replace('.npz', '_lod.npy')
        if os.path.exists(lp):
            self.lods = np.load(lp)
        else:
            self.lods = hit_lods(self.hits, ss)
            np.save(lp, self.lods)
        self._prep_lut()

    def _prep_lut(self):
        """delta(rho) = rho - beta(rho) on a uniform grid, captured -> -1 (flag).
        Beyond the traced range: weak-field tail delta ~ K / rho."""
        rho, beta = self.lut_rho, self.lut_beta
        delta = rho - beta
        cap = ~np.isfinite(beta)
        self.lut_step = float(rho[1] - rho[0])
        self.lut_delta = np.where(cap, -1.0, delta).astype(np.float32)
        # tail: match at the last traced sample, scaled to a paraxial 1/rho fall-off
        self.tail_rho = float(rho[-1])
        self.tail_K = float(delta[-1] * rho[-1])
        # rho at which the lens map is captured (shadow edge) from the trace
        self.rho_crit = float(rho[np.where(cap)[0].max() + 1]) if cap.any() else 0.0

    def delta_np(self, rho):
        """Vectorised delta(rho) for s = 1 (numpy), captured -> nan."""
        rho = np.asarray(rho, np.float32)
        d = np.interp(rho, self.lut_rho.astype(np.float32), self.lut_delta)
        d = np.where(rho > self.tail_rho, self.tail_K / np.maximum(rho, 1.0), d)
        d = np.where(rho < self.rho_crit, np.nan, d)
        return d
