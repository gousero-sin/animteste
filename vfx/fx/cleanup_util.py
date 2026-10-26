"""Small numpy helpers shared by the cleanup module (noise, ramps, sampling)."""
import numpy as np
import cv2


def sstep(a, b, x):
    x = np.clip((np.asarray(x, np.float32) - a) / (b - a + 1e-9), 0.0, 1.0)
    return x * x * (3 - 2 * x)


def sstep_f(a, b, x):
    """scalar smoothstep (python float)."""
    x = min(max((x - a) / (b - a + 1e-9), 0.0), 1.0)
    return x * x * (3 - 2 * x)


def value_fbm(h, w, cell, octaves, seed, gain=0.55):
    """Deterministic value-noise fbm in [0,1] (approximately mean 0.5)."""
    rng = np.random.default_rng(seed)
    out = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    c = float(cell)
    for _ in range(octaves):
        ci = max(2, int(round(c)))
        gh, gw = h // ci + 4, w // ci + 4
        g = rng.random((gh, gw)).astype(np.float32)
        up = cv2.resize(g, (gw * ci, gh * ci), interpolation=cv2.INTER_CUBIC)
        oy, ox = ci, ci
        out += amp * up[oy:oy + h, ox:ox + w]
        tot += amp
        amp *= gain
        c *= 0.5
    out /= tot
    # stretch to roughly [0,1]
    lo, hi = np.percentile(out, 1), np.percentile(out, 99)
    return np.clip((out - lo) / (hi - lo + 1e-6), 0, 1).astype(np.float32)


def tile_noise(n, cells, seed, gain=0.55):
    """Periodic (tileable) value-noise fbm of size n x n in [0,1]. cells: list of grid sizes."""
    rng = np.random.default_rng(seed)
    out = np.zeros((n, n), np.float32)
    amp, tot = 1.0, 0.0
    for g in cells:
        grid = rng.random((g, g)).astype(np.float32)
        big = np.tile(grid, (3, 3))
        up = cv2.resize(big, (3 * n, 3 * n), interpolation=cv2.INTER_CUBIC)
        out += amp * up[n:2 * n, n:2 * n]
        tot += amp
        amp *= gain
    out /= tot
    lo, hi = np.percentile(out, 1), np.percentile(out, 99)
    return np.clip((out - lo) / (hi - lo + 1e-6), 0, 1).astype(np.float32)


def sample_wrap(tex, mx, my):
    """Bilinear wrap-around sampling of a 2D texture at float maps (cv2.remap)."""
    return cv2.remap(tex, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_WRAP)


def heat_ramp(h):
    """Incandescent colour ramp for a 'heat' value h in [0,1] (1 = white hot). Linear RGB, ~unit."""
    h = np.asarray(h, np.float32)[..., None]
    c1 = np.array([1.00, 0.20, 0.03], np.float32)   # deep orange-red
    c2 = np.array([0.00, 0.42, 0.16], np.float32)   # -> amber
    c3 = np.array([0.35, 0.45, 0.60], np.float32)   # -> white-hot
    return (c1 * sstep(0.0, 0.40, h) + c2 * sstep(0.30, 0.75, h) + c3 * sstep(0.70, 1.0, h)).astype(np.float32)


def heat_ramp_s(h):
    return heat_ramp(np.array([h], np.float32))[0]


def pulse(t, c, w):
    return float(np.exp(-((t - c) / w) ** 2))
