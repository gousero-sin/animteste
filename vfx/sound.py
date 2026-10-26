"""Sound design + music bed + mix for the VFX edit.

    python3 -m vfx.sound            # build DATA/audio/mix.wav (+ sfx_only.wav, stems, report)
    python3 -m vfx.sound --no-report

Everything is synthesised here with numpy/scipy (no samples):
  * voice  : SRC_VIDEO audio stream 0:a:0 -> HPF, spectral denoise, EQ, compressor, leveler,
             gentle gap expander.  Mono-centred, kept dry and on top.
  * sfx    : one designed, layered sound per visual event in config.EV (whooshes = STFT-shaped
             noise sweeps, shimmers/chimes = FM + additive, crackle = granular, booms = pitch-
             enveloped saturated sub + blast), spatialised (equal-power pan, tracking-driven for
             the hand orbs), sent to a synthetic convolution reverb of the current "space"
             (bathroom / black studio / tunnel / outer space).
  * music  : subtle cinematic bed in D, one colour per section (bathroom curious -> studio dark ->
             powers tense/rising -> portal mystery -> space awe -> black hole dread -> Strange
             mystic -> warm D major ending), own hall reverb, tape-stop at the blackout,
             gravitational pitch-sag when the black hole forms.
  * mix    : 3-band side-chain ducking of music+sfx under the voice (envelope follower with
             look-ahead + adaptive unmasking to keep voice >= ~14 dB over the background),
             loudness normalisation to -14 LUFS (BS.1770), 4x-oversampled true-peak limiter at
             -1.2 dBTP.  Output is exactly config.DURATION long, 48 kHz stereo.
  * report : spectrogram / waveform / loudness / voice-to-background plots with event markers
             (DATA/audio/report_*.png) + ffmpeg ebur128 cross-check.
"""
import os
import sys
import json
import time
import argparse
import subprocess

import numpy as np
import scipy.signal as ss
import scipy.io.wavfile as wavfile
from scipy.interpolate import CubicSpline
from scipy.ndimage import minimum_filter1d, maximum_filter1d, uniform_filter1d, uniform_filter

from . import config as C

SR = 48000
N = int(round(C.DURATION * SR))
AUDIO = os.path.join(C.DATA, 'audio')
STEMS = os.path.join(AUDIO, 'stems')
EV = C.EV
TAU = 2 * np.pi


# =============================================================================================
# basic helpers
# =============================================================================================
def ns(d):
    return max(1, int(round(d * SR)))


def tvec(n):
    return np.arange(n, dtype=np.float64) / SR


def rng(seed):
    return np.random.default_rng(seed)


def db(x):
    return 20 * np.log10(np.maximum(np.abs(x), 1e-12))


def undb(x):
    return 10.0 ** (np.asarray(x, np.float64) / 20.0)


def smooth01(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def env_pts(t, pts):
    """Piecewise-linear envelope through (time, value) points."""
    tp = [p[0] for p in pts]
    vp = [p[1] for p in pts]
    return np.interp(t, tp, vp)


def env_ar(t, a, tau, shape=1.0):
    """Smooth attack over `a` seconds then exponential decay (time constant tau)."""
    up = smooth01(t / max(a, 1e-4)) ** shape
    return up * np.exp(-np.maximum(t - a, 0) / tau)


def fade(x, fin=0.003, fout=0.003):
    n = len(x)
    a, b = min(n // 2, ns(fin)), min(n // 2, ns(fout))
    w = np.ones(n)
    if a > 0:
        w[:a] = np.sin(np.linspace(0, np.pi / 2, a)) ** 2
    if b > 0:
        w[-b:] = np.cos(np.linspace(0, np.pi / 2, b)) ** 2
    return x * (w[:, None] if x.ndim == 2 else w)


def norm_peak(x, peak=1.0):
    m = np.abs(x).max()
    return x * (peak / m) if m > 0 else x


def colored(n, seed, slope=0.0, fmin=20.0):
    """Gaussian noise with power spectrum ~ f^slope (0 white, -1 pink, -2 brown), unit std."""
    x = rng(seed).standard_normal(n)
    if slope != 0.0:
        X = np.fft.rfft(x)
        f = np.fft.rfftfreq(n, 1 / SR)
        X *= np.maximum(f, fmin) ** (slope / 2)
        X[0] = 0
        x = np.fft.irfft(X, n)
    return x / (x.std() + 1e-12)


def noise2(n, seed, slope=-1.0, corr=0.3):
    """Stereo noise with inter-channel correlation `corr`."""
    a = colored(n, seed, slope)
    b = colored(n, seed + 7, slope)
    c = colored(n, seed + 13, slope)
    k1, k2 = np.sqrt(corr), np.sqrt(1 - corr)
    return np.stack([k1 * a + k2 * b, k1 * a + k2 * c], 1)


def lfo_noise(n, rate, seed, lo=None):
    """Smooth random modulation signal (approx unit std) varying at ~rate Hz."""
    k = int(n / SR * rate) + 4
    pts = rng(seed).standard_normal(k)
    xs = np.linspace(0, n - 1, k)
    return CubicSpline(xs, pts)(np.arange(n))


def pan2(x, p):
    """Equal-power pan of mono x (centre = unity per channel); balance for stereo x."""
    p = np.clip(np.asarray(p, np.float64), -1, 1)
    a = (p + 1) * np.pi / 4
    gl, gr = np.cos(a) * np.sqrt(2), np.sin(a) * np.sqrt(2)
    if x.ndim == 1:
        return np.stack([x * gl, x * gr], 1)
    return np.stack([x[:, 0] * np.minimum(gl, 1), x[:, 1] * np.minimum(gr, 1)], 1)


def widen(x, w):
    """Mid/side width scale (w can be an array)."""
    m = (x[:, 0] + x[:, 1]) * 0.5
    s = (x[:, 0] - x[:, 1]) * 0.5 * w
    return np.stack([m + s, m - s], 1)


def sat(x, drive=2.0):
    return np.tanh(drive * x) / np.tanh(drive)


def track_pan(side, scale=2.0):
    """Function t(abs s) -> pan following a tracked palm's x position."""
    from . import assets as A
    ts = np.array([C.t_of(i) for i in range(C.N_FRAMES)])
    xs = np.array([A.palm(i, side)[0] for i in range(C.N_FRAMES)], np.float64)
    xs = np.clip(xs, 0, C.W)
    xs = uniform_filter1d(xs, 9, mode='nearest')
    pans = np.clip((xs / C.W - 0.5) * scale, -0.85, 0.85)
    return lambda t: np.interp(t, ts, pans)


# =============================================================================================
# filters
# =============================================================================================
def spec(x, fn, nfft=2048, hop=512):
    """Time-varying spectral filter: fn(f[:,None] Hz, t[None,:] s from start) -> gain."""
    x = np.asarray(x, np.float64)
    if x.ndim == 2:
        return np.stack([spec(x[:, c], fn, nfft, hop) for c in range(x.shape[1])], 1)
    n = len(x)
    if n < nfft:
        x = np.pad(x, (0, nfft - n))
    f, tt, Z = ss.stft(x, SR, window='hann', nperseg=nfft, noverlap=nfft - hop)
    G = np.broadcast_to(fn(f[:, None], tt[None, :]), Z.shape)
    _, y = ss.istft(Z * G, SR, window='hann', nperseg=nfft, noverlap=nfft - hop)
    y = y[:n]
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    return y


def g_bp(f, fc, bw):
    """log-gaussian band-pass, bw = std in octaves."""
    return np.exp(-0.5 * (np.log2(np.maximum(f, 1.0) / fc) / bw) ** 2)


def g_lp(f, fc, order=2):
    return 1 / np.sqrt(1 + (f / fc) ** (2 * order))


def g_hp(f, fc, order=2):
    return 1 / np.sqrt(1 + (fc / np.maximum(f, 1e-3)) ** (2 * order))


def butter(x, kind, fc, order=4, zero_phase=False):
    sos = ss.butter(order, fc, btype=kind, fs=SR, output='sos')
    if zero_phase:
        return ss.sosfiltfilt(sos, x, axis=0)
    return ss.sosfilt(sos, x, axis=0)


def rbj(kind, f0, gain_db=0.0, q=0.707):
    """RBJ cookbook biquad -> sos row."""
    A = 10 ** (gain_db / 40)
    w0 = TAU * f0 / SR
    al = np.sin(w0) / (2 * q)
    cw = np.cos(w0)
    if kind == 'peak':
        b = [1 + al * A, -2 * cw, 1 - al * A]
        a = [1 + al / A, -2 * cw, 1 - al / A]
    elif kind == 'hshelf':
        sq = 2 * np.sqrt(A) * al
        b = [A * ((A + 1) + (A - 1) * cw + sq), -2 * A * ((A - 1) + (A + 1) * cw), A * ((A + 1) + (A - 1) * cw - sq)]
        a = [(A + 1) - (A - 1) * cw + sq, 2 * ((A - 1) - (A + 1) * cw), (A + 1) - (A - 1) * cw - sq]
    elif kind == 'lshelf':
        sq = 2 * np.sqrt(A) * al
        b = [A * ((A + 1) - (A - 1) * cw + sq), 2 * A * ((A - 1) - (A + 1) * cw), A * ((A + 1) - (A - 1) * cw - sq)]
        a = [(A + 1) + (A - 1) * cw + sq, -2 * ((A - 1) + (A + 1) * cw), (A + 1) + (A - 1) * cw - sq]
    else:
        raise ValueError(kind)
    b, a = np.array(b) / a[0], np.array(a) / a[0]
    return np.concatenate([b, a])[None, :]


def vdelay(x, d):
    """x(n - d(n)) with fractional, time-varying delay d (samples)."""
    idx = np.arange(len(x)) - d
    return np.interp(idx, np.arange(len(x)), x, left=0.0, right=0.0)


def flanger(x, base_ms, depth_ms, rate, mix=0.7, seed=0, taps=2):
    """Feed-forward multi-tap flanger / jet; rate may be an array (Hz)."""
    n = len(x)
    rate = np.broadcast_to(np.asarray(rate, np.float64), (n,))
    ph = np.cumsum(rate) / SR
    y = x.copy()
    r = rng(seed)
    for k in range(taps):
        o = r.uniform(0, 1)
        d = (base_ms + depth_ms * 0.5 * (1 + np.sin(TAU * (ph * (1 + 0.37 * k) + o)))) * SR / 1000
        y += mix * (0.8 ** k) * vdelay(x, d)
    return y / (1 + mix)


def resample_speed(x, speed):
    """Variable-speed playback (tape): speed array (n_out,) of read-rate multipliers."""
    pos = np.concatenate([[0.0], np.cumsum(speed)[:-1]])
    if x.ndim == 2:
        return np.stack([np.interp(pos, np.arange(len(x)), x[:, c], right=0.0) for c in range(2)], 1)
    return np.interp(pos, np.arange(len(x)), x, right=0.0)


# =============================================================================================
# oscillators / elementary sounds
# =============================================================================================
def phase(f, n):
    f = np.broadcast_to(np.asarray(f, np.float64), (n,))
    return np.cumsum(f) / SR


def sine(f, n, ph=0.0):
    return np.sin(TAU * (phase(f, n) + ph))


def saw(f, n, ph=0.0):
    """Band-limited (polyBLEP) sawtooth."""
    f = np.broadcast_to(np.asarray(f, np.float64), (n,))
    dt = np.maximum(f / SR, 1e-9)
    p = (np.cumsum(dt) + ph) % 1.0
    y = 2 * p - 1
    m1 = p < dt
    t1 = p[m1] / dt[m1]
    y[m1] -= t1 + t1 - t1 * t1 - 1
    m2 = p > 1 - dt
    t2 = (p[m2] - 1) / dt[m2]
    y[m2] -= t2 * t2 + t2 + t2 + 1
    return y


def damped(f, tau, dur=None, ph=0.0, attack=0.0005):
    dur = dur or min(6 * tau, 4.0)
    t = tvec(ns(dur))
    return np.sin(TAU * f * t + ph) * np.exp(-t / tau) * (1 - np.exp(-t / attack))


def bell(f, dur=1.5, ratio=3.5, index=3.0, tau=0.7, tau_i=0.2, attack=0.0015, glass=0.3):
    """FM bell / crystal (inharmonic ratio)."""
    n = ns(dur)
    t = tvec(n)
    I = index * np.exp(-t / tau_i)
    y = np.sin(TAU * f * t + I * np.sin(TAU * f * ratio * t))
    if glass > 0:
        y += glass * np.sin(TAU * f * 2.756 * t) * np.exp(-t / (tau * 0.35))
        y += glass * 0.5 * np.sin(TAU * f * 5.404 * t) * np.exp(-t / (tau * 0.18))
    return y * (1 - np.exp(-t / attack)) * np.exp(-t / tau)


def click(r, fc, tau=0.004, modes=2, noise_ms=0.6):
    """Short resonant click: noise micro-burst + a few damped modes around fc."""
    dur = min(0.25, 7 * tau + 0.002)
    n = ns(dur)
    t = tvec(n)
    y = np.zeros(n)
    for m in range(modes):
        fm = fc * (1.0 if m == 0 else r.uniform(1.3, 2.9))
        y += np.sin(TAU * fm * t + r.uniform(0, TAU)) * np.exp(-t / (tau * (1 if m == 0 else 0.6))) / (1 + m)
    k = ns(noise_ms / 1000)
    y[:k] += r.standard_normal(k) * np.exp(-np.arange(k) / max(1, k / 3)) * 0.8
    return y


def grains(dur, times, make, pans=None, seed=0):
    """Scatter mono grains (make(k, r) -> 1D array) at `times` (s) into a stereo buffer."""
    n = ns(dur)
    out = np.zeros((n, 2))
    r = rng(seed)
    for k, tk in enumerate(times):
        g = make(k, r)
        i0 = int(tk * SR)
        if i0 >= n or i0 < 0:
            continue
        g = g[:n - i0]
        p = 0.0 if pans is None else float(pans[k])
        a = (np.clip(p, -1, 1) + 1) * np.pi / 4
        out[i0:i0 + len(g), 0] += g * np.cos(a) * np.sqrt(2)
        out[i0:i0 + len(g), 1] += g * np.sin(a) * np.sqrt(2)
    return out


def poisson_times(dur, rate_fn, seed, rmax=None):
    """Inhomogeneous Poisson process (thinning). rate_fn(t) in events/s."""
    r = rng(seed)
    tt = np.linspace(0, dur, 400)
    rmax = rmax or max(1e-3, float(np.max(rate_fn(tt))) * 1.05)
    m = r.poisson(rmax * dur * 1.2) + 10
    cand = np.sort(r.uniform(0, dur, m))
    keep = r.uniform(0, 1, m) < rate_fn(cand) / rmax
    return cand[keep]


def shepard(dur, rate_oct, f_center=700, n_oct=7, up=True, seed=0):
    """Shepard-Risset glissando; rate_oct (oct/s) may be array."""
    n = ns(dur)
    rate_oct = np.broadcast_to(np.asarray(rate_oct, np.float64), (n,))
    pos = np.cumsum(rate_oct) / SR * (1 if up else -1)
    y = np.zeros(n)
    r = rng(seed)
    lo = np.log2(f_center) - n_oct / 2
    for k in range(n_oct):
        o = (k + pos) % n_oct
        f = 2 ** (lo + o)
        amp = np.exp(-0.5 * ((o - n_oct / 2) / (n_oct / 5)) ** 2)
        y += amp * np.sin(TAU * np.cumsum(f) / SR + r.uniform(0, TAU))
    return y / n_oct * 2


def whoosh(dur, fc_pts, amp_pts, bw=0.55, seed=0, corr=0.4, slope=-1.0, bw_pts=None, tone=None):
    """Band-pass noise sweep: fc_pts/amp_pts/bw_pts = [(t, value)...]."""
    n = ns(dur)
    x = noise2(n, seed, slope, corr)

    def fn(f, t):
        fc = np.interp(t, *zip(*fc_pts))
        b = bw if bw_pts is None else np.interp(t, *zip(*bw_pts))
        return g_bp(f, fc, b)
    y = spec(x, fn)
    y *= env_pts(tvec(n), amp_pts)[:, None]
    if tone is not None:   # (f_pts, level): a low tonal body following the sweep
        fp, lvl = tone
        f = np.interp(tvec(n), *zip(*fp))
        y += (lvl * sine(f, n) * env_pts(tvec(n), amp_pts))[:, None]
    return y


def sub_drop(dur, f0, f1, tau_f, tau_a, drive=2.5, attack=0.004, h2=0.25):
    n = ns(dur)
    t = tvec(n)
    f = f1 + (f0 - f1) * np.exp(-t / tau_f)
    ph = phase(f, n)
    y = np.sin(TAU * ph) + h2 * np.sin(2 * TAU * ph)
    y = sat(y * 0.9, drive)                   # harmonics so phones "hear" the sub
    y = butter(y, 'low', 900, 2)
    return y * (1 - np.exp(-t / attack)) * np.exp(-t / tau_a)


def zap(r, dur=0.12, bright=1.0):
    """Electric arc: gated HP noise + descending FM chirp."""
    n = ns(dur)
    t = tvec(n)
    gate = np.zeros(n)
    i = 0
    while i < n:
        seg = int(r.uniform(0.0015, 0.008) * SR)
        if r.uniform() < 0.62:
            gate[i:i + seg] = r.uniform(0.4, 1.0)
        i += seg
    gate = uniform_filter1d(gate, 24)
    nz = r.standard_normal(n)
    nz = butter(nz, 'high', 1800 * bright, 2)
    f = r.uniform(2000, 3800) * np.exp(-t / (dur * 0.6)) + 250
    ch = saw(f * (1 + 0.15 * np.sin(TAU * 90 * t)), n) * 0.35
    ch = butter(ch, 'high', 400, 2)
    env = (1 - np.exp(-t / 0.002)) * np.exp(-t / (dur * 0.5))
    return (nz + ch) * gate * env


def crackle(dur, rate_fn, f_lo, f_hi, tau=(0.002, 0.012), seed=0, pan_fn=None, pan_spread=0.3,
            alpha=1.6, modes=2):
    """Granular crackle: Poisson resonant clicks with power-law amplitudes."""
    times = poisson_times(dur, rate_fn, seed)
    r = rng(seed + 1)
    amps = np.minimum(r.pareto(alpha, len(times)) + 0.3, 6.0) / 3.0
    fcs = np.exp(r.uniform(np.log(f_lo), np.log(f_hi), len(times)))
    taus = np.exp(r.uniform(np.log(tau[0]), np.log(tau[1]), len(times)))
    pans = (pan_fn(times) if pan_fn else 0.0) + r.uniform(-pan_spread, pan_spread, len(times))
    return grains(dur, times, lambda k, rr: click(rr, fcs[k], taus[k], modes) * amps[k], pans, seed + 2)


def sparkle(dur, times, freqs, seed=0, pan_fn=None, spread=0.4, tau=(0.08, 0.35), amp=1.0):
    r = rng(seed)
    taus = r.uniform(tau[0], tau[1], len(times))
    pans = (pan_fn(np.asarray(times)) if pan_fn else 0.0) + r.uniform(-spread, spread, len(times))
    fr = [freqs[int(r.integers(len(freqs)))] * (2 if r.uniform() < 0.3 else 1) for _ in times]
    return grains(dur, times, lambda k, rr: amp * rr.uniform(0.4, 1.0) * bell(
        fr[k], dur=taus[k] * 5, ratio=rr.choice([2.0, 3.0, 1.41]), index=rr.uniform(0.8, 2.5),
        tau=taus[k], tau_i=taus[k] * 0.3, glass=0.25), pans, seed + 3)


# =============================================================================================
# reverbs (synthetic impulse responses)
# =============================================================================================
def make_ir(rt_lo, rt_hi, dur, predelay=0.01, er=10, er_span=0.05, damp=9000, seed=0, flutter=0.0):
    n = ns(dur)
    r = rng(seed)
    out = np.zeros((n, 2))
    for c in range(2):
        x = r.standard_normal(n)

        def fn(f, t):
            lf = np.clip(np.log2(np.maximum(f, 1) / 150) / np.log2(10000 / 150), 0, 1)
            rt = rt_lo * (1 - lf) + rt_hi * lf
            return np.exp(-6.91 * t / rt) * g_lp(f, damp, 1)
        y = spec(x, fn, 1024, 256)
        t = tvec(n)
        y *= 1 - np.exp(-t / 0.015)
        if flutter > 0:   # metallic flutter echo (tunnel)
            y = y + flutter * vdelay(y, np.full(n, 0.0123 * SR)) + flutter * 0.6 * vdelay(y, np.full(n, 0.0291 * SR))
        pd = ns(predelay)
        y = np.concatenate([np.zeros(pd), y])[:n]
        for k in range(er):
            tk = predelay * 0.4 + r.uniform(0.002, er_span)
            i = ns(tk)
            if i < n:
                y[i] += r.choice([-1, 1]) * r.uniform(0.4, 1.0) * np.exp(-tk / (er_span * 1.5)) * 3.0
        out[:, c] = y
    return out / np.sqrt((out ** 2).sum() / 2)


def convolve(x, ir):
    y = np.stack([ss.oaconvolve(x[:, c], ir[:, c])[:len(x)] for c in range(2)], 1)
    return y


# =============================================================================================
# loudness / peaks (ITU-R BS.1770-4)
# =============================================================================================
_KB1 = [1.53512485958697, -2.69169618940638, 1.19839281085285]
_KA1 = [1.0, -1.69065929318241, 0.73248077421585]
_KB2 = [1.0, -2.0, 1.0]
_KA2 = [1.0, -1.99004745483398, 0.99007225036621]


def kweight(x):
    y = ss.lfilter(_KB1, _KA1, x, axis=0)
    return ss.lfilter(_KB2, _KA2, y, axis=0)


def loud_curve(x, win=0.4, hop=0.01, weighted=None):
    """Momentary (win=0.4) / short-term (win=3) loudness curve, LUFS, frame times = centres."""
    y = weighted if weighted is not None else kweight(x if x.ndim == 2 else x[:, None])
    p = (y ** 2).sum(1) if y.ndim == 2 else y ** 2
    w = ns(win)
    h = ns(hop)
    cs = np.concatenate([[0], np.cumsum(p)])
    starts = np.arange(0, len(p) - w + 1, h)
    ms = (cs[starts + w] - cs[starts]) / w
    tc = (starts + w / 2) / SR
    return tc, -0.691 + 10 * np.log10(ms + 1e-15)


def integrated(x):
    tc, l = loud_curve(x, 0.4, 0.1)
    p = 10 ** ((l + 0.691) / 10)
    g = l > -70
    if not g.any():
        return -99.0
    rel = -0.691 + 10 * np.log10(p[g].mean()) - 10
    g2 = g & (l > rel)
    return -0.691 + 10 * np.log10(p[g2].mean())


def true_peak(x):
    up = ss.resample_poly(x, 4, 1, axis=0)
    return db(np.abs(up).max())


# =============================================================================================
# voice
# =============================================================================================
def load_voice():
    os.makedirs(AUDIO, exist_ok=True)
    p = os.path.join(AUDIO, 'voice_src.wav')
    if not os.path.exists(p):
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', C.SRC_VIDEO, '-map', '0:a:0', '-ac', '2',
                        '-ar', str(SR), '-c:a', 'pcm_f32le', p], check=True)
    sr, x = wavfile.read(p)
    assert sr == SR
    x = x.astype(np.float64)
    if len(x) < N:
        x = np.pad(x, ((0, N - len(x)), (0, 0)))
    return x[:N]


def frame_db(x, hop=480, win=960):
    p = np.concatenate([[0], np.cumsum(x ** 2)])
    starts = np.arange(0, len(x), hop)
    ends = np.minimum(starts + win, len(x))
    ms = (p[ends] - p[starts]) / np.maximum(ends - starts, 1)
    return 10 * np.log10(ms + 1e-14)


def follow(x, att, rel, hop_s):
    """Asymmetric one-pole follower on a frame-rate signal (rising uses att, falling rel)."""
    a_att = np.exp(-hop_s / max(att, 1e-6))
    a_rel = np.exp(-hop_s / max(rel, 1e-6))
    y = np.empty_like(x)
    cur = x[0]
    for k in range(len(x)):
        v = x[k]
        a = a_att if v > cur else a_rel
        cur = a * cur + (1 - a) * v
        y[k] = cur
    return y


def to_samples(frames, hop):
    tf = (np.arange(len(frames)) + 0.5) * hop
    return np.interp(np.arange(N), tf, frames)


def voice_vad(m):
    """10 ms frames voice-activity (bool) + raw level (dB) for the mono voice m."""
    lv = frame_db(butter(m, 'band', [150, 4000], 2), 480, 960)
    floor = np.percentile(lv, 10)
    thr = floor + 13.0
    act = lv > thr
    # hangover 220 ms, pre-roll 60 ms, drop blips < 50 ms
    act = maximum_filter1d(act.astype(np.uint8), 5).astype(bool)
    act = minimum_filter1d(act.astype(np.uint8), 5).astype(bool)
    act = minimum_filter1d(act.astype(np.uint8), 5).astype(bool)
    act = maximum_filter1d(act.astype(np.uint8), 5).astype(bool)
    hang = np.zeros_like(act)
    for k in range(len(act)):
        if act[max(0, k - 22):k + 7].any():
            hang[k] = True
    return hang, lv, thr


def denoise(x, floor_db=-10.0, alpha=1.35):
    f, tt, Z = ss.stft(x, SR, window='hann', nperseg=2048, noverlap=1536)
    P = np.abs(Z) ** 2
    fe = P.sum(0)
    quiet = fe <= np.percentile(fe, 12)
    Np = np.mean(P[:, quiet], axis=1)[:, None]
    G = np.maximum(1 - alpha * Np / (P + 1e-20), 10 ** (floor_db / 10))
    G = uniform_filter(G, size=(3, 3))
    _, y = ss.istft(Z * np.sqrt(G), SR, window='hann', nperseg=2048, noverlap=1536)
    return y[:len(x)]


def process_voice(raw):
    """-> processed stereo voice (n,2), vad frames (10 ms), info dict."""
    mid = raw.mean(1)
    side = (raw[:, 0] - raw[:, 1]) * 0.5
    mid = butter(mid, 'high', 75, 4)
    side = butter(side, 'high', 150, 2)
    mid = denoise(mid)
    side = denoise(side, -14, 1.6)
    eq = np.concatenate([rbj('peak', 290, -2.5, 1.0), rbj('peak', 3300, 3.5, 0.8),
                         rbj('hshelf', 9000, 2.5, 0.7), rbj('peak', 160, 1.0, 0.9)])
    mid = ss.sosfilt(eq, mid)
    side = ss.sosfilt(eq, side)
    vad, lv, thr = voice_vad(mid)
    # --- compressor (1 ms frames, RMS 5 ms)
    hop = 48
    e = frame_db(mid, hop, 240)
    vad1 = np.repeat(vad, 10)[:len(e)]
    vad1 = np.pad(vad1, (0, len(e) - len(vad1)))
    th = np.median(e[vad1]) + 2.0
    ratio, knee = 3.0, 8.0
    over = e - th
    gr = np.where(over <= -knee / 2, 0.0,
                  np.where(over >= knee / 2, over * (1 / ratio - 1),
                           (1 / ratio - 1) * (over + knee / 2) ** 2 / (2 * knee)))
    gs = -follow(-gr, 0.006, 0.12, hop / SR)
    g = undb(to_samples(gs, hop))
    mid *= g
    side *= g
    # --- slow leveler during speech (+-5 dB toward the median phrase loudness)
    tc, lm = loud_curve(mid[:, None], 0.8, 0.01)
    vadc = np.interp(tc, (np.arange(len(vad)) + 0.5) * 0.01, vad.astype(float)) > 0.5
    target = np.median(lm[vadc])
    want = np.clip(target - lm, -4.0, 5.0)
    want[~vadc] = np.nan
    idx = np.where(~np.isnan(want), np.arange(len(want)), 0)
    np.maximum.accumulate(idx, out=idx)
    want = want[idx]
    want = np.nan_to_num(want)
    b, a = ss.butter(1, 1.0 / (0.5 * 100), 'low')       # ~1 Hz smoothing (frames at 100 Hz)
    want = ss.filtfilt(b, a, want)
    g = undb(np.interp(np.arange(N) / SR, tc, want))
    mid *= g
    side *= g
    # --- gentle expander in the gaps (-7 dB) with look-ahead
    vs = vad.astype(float)
    open_ = follow(maximum_filter1d(vs, 7, origin=-3), 0.015, 0.25, 0.01)
    gexp = undb(-7.0 * (1 - np.clip(open_, 0, 1)))
    gexp = to_samples(gexp, 480)
    mid *= gexp
    side *= gexp
    v = np.stack([mid + 0.18 * side, mid - 0.18 * side], 1)
    # normalise voice to -20 LUFS internal reference
    li = integrated(v)
    v *= undb(-20.0 - li)
    return v, vad, dict(voice_lufs_raw=li, comp_thr=th)


# =============================================================================================
# the mix buses
# =============================================================================================
SPACE_RT = dict(  # name: (rt_lo, rt_hi, dur, predelay, er, er_span, damp, flutter)
    bath=(0.75, 0.45, 1.4, 0.004, 14, 0.025, 11000, 0.0),
    studio=(2.4, 1.3, 3.5, 0.02, 8, 0.07, 6500, 0.0),
    tunnel=(3.0, 1.8, 4.0, 0.012, 10, 0.04, 9000, 0.35),
    space=(7.5, 5.0, 9.0, 0.06, 4, 0.12, 12000, 0.0),
    hall=(4.2, 2.6, 6.0, 0.03, 10, 0.08, 8000, 0.0),
)


def space_at(t):
    if t < EV['studio'][0] or t > EV['strange_wipe'][1]:
        return 'bath'
    if t < EV['tunnel'][0]:
        return 'studio'
    if t < EV['tunnel'][1]:
        return 'tunnel'
    return 'space'


class Bus:
    def __init__(self, sends):
        self.dry = np.zeros((N, 2), np.float64)
        self.send = {k: np.zeros((N, 2), np.float64) for k in sends}
        self.log = []

    def add(self, x, t0, gain_db=0.0, pan=0.0, rev=0.0, space=None, name='', dry=1.0):
        x = np.asarray(x, np.float64)
        if x.ndim == 1 or np.ndim(pan) > 0 or pan != 0.0:
            x = pan2(x, pan)
        x = fade(x, 0.002, 0.004) * undb(gain_db)
        i0 = int(round(t0 * SR))
        a0 = max(0, -i0)
        i0c = max(0, i0)
        i1 = min(N, i0 + len(x))
        if i1 <= i0c:
            return
        seg = x[a0:a0 + (i1 - i0c)]
        self.dry[i0c:i1] += seg * dry
        if rev > 0:
            sp = space or space_at(t0 + 0.05)
            self.send[sp][i0c:i1] += seg * rev
        if name:
            self.log.append((name, t0, t0 + len(x) / SR))


# =============================================================================================
# SFX library: designed composite sounds
# =============================================================================================
def sfx_eraser(d, base, size, seed):
    """Magic eraser: scan laser + outline shimmer -> burn crackle + sparkles + rising air."""
    tail = 1.6
    T = d + tail
    n = ns(T)
    t = tvec(n)
    out = np.zeros((n, 2))
    r = rng(seed)
    # 1. scan: thin gliding laser tone with a bright "shhk" band
    ts = 0.38 * d
    f = base * 0.5 * 3.0 ** smooth01(t / ts)
    las = sine(f * (1 + 0.004 * np.sin(TAU * 7 * t)), n) + 0.3 * sine(f * 2.01, n)
    las *= env_pts(t, [(0, 0), (0.03, 1), (ts * 0.8, 0.8), (ts + 0.12, 0)])
    out += pan2(las * 0.28, 0.25 * np.sin(TAU * t / max(ts, 0.1)))
    sh = whoosh(T, [(0, 3500), (ts, 7500), (T, 6000)], [(0, 0), (0.05, 0.6), (ts, 0.35), (ts + 0.2, 0)],
                bw=0.35, seed=seed, slope=0.0)
    out += sh * 0.5
    # 2. outline shimmer: harmonic partials of base, staggered onsets
    for k, mult in enumerate([1, 1.5, 2, 3, 4, 6]):
        on = r.uniform(0.0, ts)
        g = np.clip(t - on, 0, None)
        p = sine(base * mult * (1 + r.uniform(-0.002, 0.002)), n) * (g > 0)
        p *= (1 - np.exp(-g / 0.03)) * np.exp(-g / (0.5 + 0.25 * size)) / (1 + 0.5 * k)
        out += pan2(p * 0.22, r.uniform(-0.4, 0.4))
    # 3. burn crackle (dissolve front)
    t0b = 0.25 * d
    dissolve = lambda tt: 160 * size * np.exp(-0.5 * ((tt - (t0b + 0.45 * d)) / (0.32 * d + 0.05)) ** 2) + 4
    cr = crackle(T, dissolve, 2500, 9000, (0.0012, 0.006), seed + 11, None, 0.5, 1.8)
    out += cr * 0.55
    # 4. sparkles (particles)
    st = np.sort(t0b + np.abs(r.normal(0.35 * d, 0.3 * d + 0.1, int(26 * size + 6))))
    pent = [base * 2 ** (s / 12) for s in (0, 2, 4, 7, 9, 12, 14, 16)]
    out += sparkle(T, st, pent, seed + 21, None, 0.55, (0.08, 0.4), 0.30)
    # 5. rising airy whoosh (particles float up)
    air = whoosh(T, [(0, 700), (t0b, 900), (d, 3500), (T, 7000)],
                 [(0, 0), (t0b, 0.05), (0.7 * d, 0.55), (d + 0.6, 0.25), (T, 0)], bw=0.6, seed=seed + 31)
    out += air * 0.55
    # 6. soft warm bloom on reveal
    bl = sine(base / 8, n) * env_pts(t, [(0, 0), (0.55 * d, 0), (0.8 * d, 1), (T, 0)]) ** 2
    out += pan2(bl * 0.25, 0)
    return out


def sfx_studio():
    """Blackout: power-off clunk + dying hum, ink wave whoosh, tile disintegration, sub drop."""
    T = 4.2
    n = ns(T)
    t = tvec(n)
    out = np.zeros((n, 2))
    r = rng(400)
    # relay clunk at 0
    cl = np.zeros(n)
    k = click(r, 1800, 0.006, 3) * 0.6
    cl[:len(k)] += k
    th = sub_drop(0.4, 140, 70, 0.05, 0.08, 1.5)
    cl[:len(th)] += th * 0.9
    out += pan2(cl, 0)
    # mains hum dying with pitch dive
    fh = 100 * (1 - 0.55 * smooth01(t / 0.7))
    hum = sum(sine(fh * h, n) / h ** 1.2 for h in (1, 2, 3, 5, 7))
    hum = sat(hum * 1.2, 2.0) * env_pts(t, [(0, 0), (0.01, 0.6), (0.15, 0.5), (0.75, 0)])
    out += pan2(hum * 0.35, 0)
    # ink wave whoosh radiating outward (stereo widening), 0.05 .. 1.5
    w = whoosh(T, [(0, 120), (0.55, 700), (1.3, 260), (T, 120)],
               [(0, 0), (0.08, 0.35), (0.55, 1.0), (1.4, 0.3), (2.2, 0)], bw=0.7, seed=401, corr=0.2,
               tone=([(0, 110), (1.5, 45)], 0.25))
    w = widen(w, np.interp(t, [0, 0.6, 1.5], [0.2, 1.4, 1.8]))
    out += w * 0.9
    # disintegration: ceramic crumble grains, peak ~0.75 s, widening
    rate = lambda tt: 380 * np.exp(-0.5 * ((tt - 0.8) / 0.35) ** 2) + 20 * np.exp(-tt / 2)
    times = poisson_times(T, rate, 402)
    rr = rng(403)
    pans = np.clip(rr.normal(0, 1, len(times)) * np.interp(times, [0, 1.2], [0.2, 0.9]), -1, 1)
    fcs = np.exp(rr.uniform(np.log(1500), np.log(6500), len(times)))
    amps = rr.uniform(0.2, 1.0, len(times))
    deb = grains(T, times, lambda k, q: click(q, fcs[k], q.uniform(0.0015, 0.007), 3) * amps[k], pans, 404)
    out += deb * 0.5
    # "sucked away": reverse whoosh 1.15 -> 1.7 then cut
    sw = whoosh(0.6, [(0, 400), (0.6, 4000)], [(0, 0), (0.5, 0.8), (0.6, 0)], bw=0.5, seed=405)
    i0 = ns(1.15)
    out[i0:i0 + len(sw)] += sw * 0.55
    # sub drop
    sd = sub_drop(3.0, 85, 27, 0.6, 1.1, 2.6)
    i0 = ns(0.15)
    out[i0:i0 + len(sd)] += pan2(sd, 0) * 0.95
    # darkness settles: deep soft hit at 1.75
    hit = sub_drop(2.0, 70, 38, 0.15, 0.6, 2.0) * 0.8
    body = butter(colored(ns(2.0), 406, -2), 'low', 260, 2) * env_ar(tvec(ns(2.0)), 0.01, 0.35) * 0.35
    i0 = ns(1.75)
    out[i0:i0 + len(hit)] += pan2(hit + body, 0)
    return out


def sfx_suit(d):
    """Nanotech assembly: granular metallic ticks spreading out + servo whine + riser + lock."""
    T = d + 1.6
    n = ns(T)
    t = tvec(n)
    out = np.zeros((n, 2))
    rate = lambda tt: np.where(tt < d, 40 + 520 * np.sin(np.pi * np.clip(tt / d, 0, 1)) ** 1.5 * smooth01(tt / (0.3 * d)), 0)
    times = poisson_times(d, rate, 500)
    r = rng(501)
    spread = np.interp(times, [0, d], [0.1, 0.95])
    pans = np.clip(r.normal(0, 0.6, len(times)) * spread, -1, 1)
    fcs = np.exp(r.uniform(np.log(2500), np.log(9500), len(times)))
    amps = r.uniform(0.25, 1.0, len(times)) * np.interp(times, [0, d * 0.3, d], [0.5, 1.0, 0.8])
    ticks = grains(T, times, lambda k, q: click(q, fcs[k], q.uniform(0.0008, 0.004), 3) * amps[k], pans, 502)
    out += ticks * 0.42
    # hex cell snaps (bigger "tk")
    times2 = poisson_times(d, lambda tt: 9 + 14 * (tt / d), 503)
    pans2 = r.uniform(-0.8, 0.8, len(times2))
    snaps = grains(T, times2, lambda k, q: click(q, q.uniform(1100, 2600), 0.006, 3) * q.uniform(0.5, 1), pans2, 504)
    out += snaps * 0.35
    # servo whine (two voices)
    for j, (fa, fb, pn) in enumerate([(520, 2300, -0.25), (780, 3100, 0.25)]):
        f = fa * (fb / fa) ** smooth01(t / d) * (1 + 0.006 * np.sin(TAU * (5.5 + j) * t))
        sv = sine(f, n) + 0.25 * saw(f * 0.5, n) * 0.4
        sv = butter(sv, 'band', [300, 6000], 2)
        sv *= env_pts(t, [(0, 0), (0.25, 0.3), (d * 0.9, 0.6), (d, 0)])
        out += pan2(sv * 0.10, pn)
    # riser
    ri = whoosh(T, [(0, 900), (d, 7500), (T, 7500)], [(0, 0), (d * 0.97, 0.7), (d + 0.02, 0)], bw=0.45, seed=505)
    out += ri * 0.45
    # lock: "shk-chunk" at d
    i0 = ns(d)
    m = ns(1.5)
    tl = tvec(m)
    shk = butter(rng(506).standard_normal(m), 'high', 3000, 2) * np.exp(-tl / 0.018) * 0.8
    thud = sub_drop(1.5, 110, 55, 0.04, 0.18, 2.0) * 0.9
    ring = sum(np.sin(TAU * fr * tl) * np.exp(-tl / tc) for fr, tc in ((1830, 0.35), (2910, 0.22), (4370, 0.14)))
    lock = shk + thud + ring * 0.12 * (1 - np.exp(-tl / 0.001))
    out[i0:i0 + m] += pan2(lock, 0)[:n - i0]
    # power-up seams: cool soft tone
    i0 = ns(d + 0.06)
    m = ns(1.4)
    tl = tvec(m)
    pw = (sine(hz('E5'), m) + 0.5 * sine(hz('B5'), m) + 0.25 * sine(hz('E6') * 1.002, m))
    pw *= env_pts(tl, [(0, 0), (0.12, 1), (0.5, 0.5), (1.4, 0)])
    out[i0:i0 + m] += pan2(pw * 0.12, 0)[:n - i0]
    return out


def sfx_anticipation(d, panR, panL, t_abs0):
    T = d + 0.2
    n = ns(T)
    t = tvec(n)
    out = np.zeros((n, 2))
    grow = smooth01(t / d) ** 1.4 * (t < d + 0.02)
    for j, (pf, sd) in enumerate([(panR, 600), (panL, 610)]):
        x = colored(n, sd, -1)
        fc = 1800 + 900 * lfo_noise(n, 0.7, sd + 1)

        def fn(f, tt, fc=fc):
            c = np.interp(tt, t, fc)
            return g_bp(f, c, 0.35)
        w = spec(x, fn) * (0.6 + 0.4 * np.sin(TAU * (1.3 + 0.4 * j) * t + j)) * grow
        out += pan2(w * 0.35, pf(t_abs0 + t))
        cr = crackle(T, lambda tt: 1.5 + 9 * smooth01(tt / d), 2500, 8000, (0.0008, 0.003), sd + 2,
                     lambda tt: pf(t_abs0 + tt), 0.1, 1.8, 1)
        out += cr * 0.25 * grow[:, None]
    # low riser into the ignition
    f = 48 * 2 ** (smooth01(t / d) * 0.6)
    lr = sat(sine(f, n) + 0.3 * sine(2 * f, n), 1.5) * smooth01(t / d) ** 2.5 * (t < d)
    out += pan2(lr * 0.30, 0)
    rs = whoosh(T, [(0, 500), (d, 3000), (T, 3000)], [(0, 0), (d * 0.6, 0.1), (d - 0.02, 0.5), (d, 0)], bw=0.6, seed=607)
    out += rs * 0.4
    return out


def sfx_fire_ignite():
    T = 1.6
    n = ns(T)
    t = tvec(n)
    out = np.zeros((n, 2))
    # pre-suck 0..0.16 (reverse) then burst at 0.16
    pre = whoosh(0.16, [(0, 1500), (0.16, 400)], [(0, 0), (0.15, 0.5), (0.16, 0)], bw=0.6, seed=700)
    out[:len(pre)] += pre * 0.5
    b0 = 0.16
    tb = np.clip(t - b0, 0, None)
    bu = whoosh(T, [(0, 250), (b0, 250), (b0 + 0.12, 1800), (b0 + 0.6, 500), (T, 300)],
                [(0, 0), (b0, 0), (b0 + 0.008, 1.0), (b0 + 0.3, 0.45), (T, 0)], bw=0.9, seed=701, corr=0.5)
    out += bu * 1.0
    fl = noise2(n, 702, 0.0, 0.5)
    fl = butter(fl, 'high', 3500, 2) * (env_ar(tb, 0.005, 0.12) * (t > b0))[:, None]
    out += fl * 0.35
    th = sub_drop(1.2, 95, 50, 0.08, 0.3, 2.2)
    out[ns(b0):ns(b0) + len(th)] += pan2(th, 0)[:n - ns(b0)] * 0.85
    cr = crackle(T, lambda tt: np.where(tt > b0, 220 * np.exp(-(tt - b0) / 0.25), 0) + 1, 800, 5000,
                 (0.002, 0.01), 703, None, 0.4, 1.4)
    out += cr * 0.6
    return out


def fire_bed(T, pan_fn, t_abs0, seed=710):
    n = ns(T)
    t = tvec(n)
    flick = 1 + 0.45 * lfo_noise(n, 4.0, seed) + 0.15 * lfo_noise(n, 17, seed + 1)
    roar = noise2(n, seed + 2, -2.0, 0.6)
    roar = spec(roar, lambda f, tt: g_bp(f, 260, 1.1) * g_hp(f, 60, 2))
    roar *= np.clip(flick, 0.2, None)[:, None]
    mid = noise2(n, seed + 3, -1.0, 0.4)
    mid = spec(mid, lambda f, tt: g_bp(f, 1100, 0.6))
    mid *= (0.6 + 0.4 * np.sin(TAU * 21 * t + 3 * lfo_noise(n, 2, seed + 4)))[:, None] * np.clip(flick, 0.2, None)[:, None]
    hiss = butter(noise2(n, seed + 5, 0.0, 0.3), 'high', 5000, 2) * np.clip(flick, 0, None)[:, None] ** 2
    cr = crackle(T, lambda tt: 22 + 14 * np.sin(tt * 1.7) ** 2, 700, 4500, (0.002, 0.012), seed + 6,
                 lambda tt: pan_fn(t_abs0 + tt), 0.15, 1.3)
    out = roar * 0.9 + mid * 0.25 + hiss * 0.06 + cr * 0.55
    return pan2(out, pan_fn(t_abs0 + t))


def sfx_ice_form():
    T = 2.6
    n = ns(T)
    t = tvec(n)
    out = np.zeros((n, 2))
    r = rng(800)
    # frost "shhh" swell (-0.25 .. 0.6)
    fr = whoosh(T, [(0, 6000), (0.5, 8000), (T, 9000)], [(0, 0), (0.25, 0.4), (0.55, 0.25), (1.6, 0)], bw=0.4, seed=801, slope=0.0)
    out += fr * 0.45
    # crystal growth: ascending accelerating bells 0.25 .. 0.75
    k = np.arange(18)
    times = 0.25 + 0.5 * (1 - (1 - k / 17) ** 1.8)
    scale = [hz(x) for x in ('E6', 'F#6', 'A6', 'B6', 'C#7', 'E7', 'F#7', 'A7')]
    freqs = [scale[min(len(scale) - 1, int(i * len(scale) / 18))] for i in k]
    pans = r.uniform(-0.35, 0.35, len(k))
    g = grains(T, times, lambda j, q: bell(freqs[j] * q.uniform(0.997, 1.003), 1.2, ratio=q.choice([3.5, 2.76, 4.1]),
                                            index=1.6, tau=0.35, tau_i=0.05, glass=0.35) * (0.5 + 0.5 * j / 17), pans, 802)
    out += g * 0.32
    # ice cracks
    cr = crackle(T, lambda tt: 45 * np.exp(-0.5 * ((tt - 0.5) / 0.22) ** 2) + 1.5, 2200, 7500, (0.003, 0.02), 803,
                 None, 0.4, 1.5, 3)
    out += cr * 0.55
    # low "thunk" of ice forming at 0.28
    th = sub_drop(0.6, 180, 95, 0.05, 0.12, 1.8) * 0.7
    i0 = ns(0.28)
    out[i0:i0 + len(th)] += pan2(th, 0)
    # lock-in "ting" at 0.78 + crystalline tail
    i0 = ns(0.78)
    m = n - i0
    tl = tvec(m)
    ting = sum(a * np.sin(TAU * f * tl) * np.exp(-tl / tc)
               for f, a, tc in ((hz('B6'), 1.0, 1.1), (hz('B6') * 2.756, 0.4, 0.4), (hz('F#7'), 0.5, 0.8), (hz('B6') * 5.4, 0.2, 0.2)))
    ting *= 1 - np.exp(-tl / 0.0008)
    out[i0:] += pan2(ting * 0.2, 0.1)
    hit = sub_drop(0.8, 120, 70, 0.03, 0.15, 1.6) * 0.5
    out[i0:i0 + len(hit)] += pan2(hit, 0)
    return out


def ice_bed(T, pan_fn, t_abs0, seed=820):
    n = ns(T)
    t = tvec(n)
    out = np.zeros((n, 2))
    r = rng(seed)
    for k, f in enumerate([hz('B6'), hz('E7'), hz('F#7') * 1.003, hz('A7'), hz('E8') * 0.998]):
        a = 0.5 + 0.5 * np.sin(TAU * r.uniform(0.15, 0.6) * t + r.uniform(0, TAU))
        out += pan2(sine(f, n) * a * 0.05 / (1 + 0.3 * k), r.uniform(-0.3, 0.3))
    fr = butter(noise2(n, seed + 1, 0.0, 0.2), 'high', 6500, 2) * (0.6 + 0.4 * lfo_noise(n, 0.8, seed + 2))[:, None]
    out += fr * 0.05
    times = poisson_times(T, lambda tt: 3.0 + 0 * tt, seed + 3)
    gl = sparkle(T, times, [hz('E7'), hz('F#7'), hz('B7')], seed + 4, None, 0.3, (0.03, 0.12), 0.25)
    out += gl
    cr = crackle(T, lambda tt: 3 + 0 * tt, 2500, 7000, (0.002, 0.01), seed + 5, None, 0.3, 1.6, 2)
    out += cr * 0.3
    return pan2(out, pan_fn(t_abs0 + t))


def sfx_fusion(d, t_abs0, panR, panL):
    """Orbs spiral into the ring: circling fire+ice sources, rising electric hum, arcs."""
    T = d
    n = ns(T)
    t = tvec(n)
    out = np.zeros((n, 2))
    u = t / d
    rot = 0.6 + 2.6 * u ** 1.5
    ph = TAU * np.cumsum(rot) / SR
    rad = np.interp(u, [0, 0.35, 1], [1.0, 0.45, 0.25])
    cR, cL = panR(t_abs0), panL(t_abs0)
    p_fire = cR * rad * 0.8 * (1 - u) + 0.45 * rad * np.sin(ph)
    p_ice = cL * rad * 0.8 * (1 - u) - 0.45 * rad * np.sin(ph)
    depth = 1 + 0.35 * np.cos(ph)
    fb = fire_bed(T, lambda tt: 0 * tt, 0, 900)[:, 0]
    out += pan2(fb * depth * 0.9, p_fire)
    ib = ice_bed(T, lambda tt: 0 * tt, 0, 910)[:, 0]
    out += pan2(ib * (2 - depth) * 1.2, p_ice)
    # electric hum rising
    f0 = hz('D2') * 2 ** (3 / 12 * u ** 2)
    trem = 0.65 + 0.35 * np.sin(TAU * np.cumsum(6 + 12 * u) / SR)
    hm = saw(f0, n) + 0.7 * saw(f0 * 1.5 * 1.003, n) + 0.5 * saw(f0 * 2 * 0.997, n)
    hm = spec(hm, lambda f, tt: g_bp(f, np.interp(tt, [0, d], [500, 1800]), 1.0))
    hm *= trem * smooth01(u * 1.6) * (0.5 + 0.5 * u)
    out += pan2(hm * 0.35, 0) + widen(pan2(hm * 0.12, 0), 0)
    # arcs between the hands
    times = poisson_times(T, lambda tt: 3 + 11 * (tt / d) ** 1.3, 920)
    r = rng(921)
    pans = np.where(np.arange(len(times)) % 2 == 0, -0.3, 0.3) + r.uniform(-0.15, 0.15, len(times))
    arcs = grains(T, times, lambda k, q: zap(q, q.uniform(0.04, 0.2)) * q.uniform(0.4, 1.0), pans, 922)
    out += arcs * 0.42
    # helix swirl (jet-flanged noise)
    sw = colored(n, 923, -1)
    sw = flanger(sw, 0.6, 4.0, 0.4 + 1.2 * u, 0.85, 924, 3)
    sw = spec(sw, lambda f, tt: g_bp(f, np.interp(tt, [0, d], [900, 2600]), 0.8))
    out += pan2(sw * smooth01(u) * 0.35, 0.5 * np.sin(ph * 0.5))
    # violet riser in the last 60 %
    ri = whoosh(T, [(0, 600), (d * 0.4, 600), (d, 4000)], [(0, 0), (d * 0.4, 0), (d, 0.55)], bw=0.6, seed=925)
    out += ri
    return out


def sfx_release_charge(t_rel, t_boom):
    """Hands spread (whoosh) -> plasma orb rises and charges -> inhale; ends exactly at boom."""
    d = t_boom - t_rel
    n = ns(d)
    t = tvec(n)
    u = t / d
    out = np.zeros((n, 2))
    # spread whoosh, outward
    w = whoosh(0.7, [(0, 500), (0.25, 1800), (0.7, 700)], [(0, 0), (0.08, 0.8), (0.7, 0)], bw=0.6, seed=950, corr=0.1)
    w = widen(w, np.interp(tvec(len(w)), [0, 0.5], [0.5, 1.8]))
    out[:len(w)] += w * 0.8
    # condense thrum at 0.25
    i0 = ns(0.22)
    m = ns(0.5)
    th = sub_drop(0.5, 70, 110, 0.15, 0.25, 2.0) * 0.6
    out[i0:i0 + m] += pan2(th, 0)
    # tonal charge riser
    f = hz('D3') * 2 ** (1.0 * smooth01(u) ** 1.3)
    trem = 0.6 + 0.4 * np.sin(TAU * np.cumsum(4 + 26 * u ** 2) / SR)
    tone = saw(f, n) + 0.6 * saw(f * 1.5 * 1.002, n) + 0.4 * saw(f * 2.003, n)
    tone = spec(tone, lambda ff, tt: g_lp(ff, np.interp(tt, [0, d], [700, 5000]), 2))
    out += pan2(tone * trem * smooth01(u * 3) * (0.25 + 0.75 * u ** 1.5) * 0.30, 0)
    # noise riser
    ri = whoosh(d, [(0, 400), (d, 7000)], [(0, 0), (d * 0.98, 0.9), (d, 0)], bw=0.5, seed=951)
    out += ri * 0.6
    # shepard up
    sh = shepard(d, 0.6 + 2.5 * u, 900, 7, True, 952) * smooth01(u * 2) * (0.3 + 0.7 * u)
    out += pan2(sh * 0.25, 0)
    # arcs, contracting
    times = poisson_times(d, lambda tt: 5 + 40 * (tt / d) ** 2, 953)
    r = rng(954)
    arcs = grains(d, times, lambda k, q: zap(q, q.uniform(0.03, 0.12), 1.2) * q.uniform(0.3, 1.0),
                  r.uniform(-0.4, 0.4, len(times)) * (1 - np.asarray(times) / d * 0.7), 955)
    out += arcs * 0.35
    # inhale: last 0.12 s rise, then hard stop 15 ms before the boom (pre-impact silence)
    end = np.ones(n)
    ks = ns(d - 0.022)
    end[ks:] = np.linspace(1, 0, n - ks)
    out *= end[:, None]
    return out


def sfx_boom():
    T = 5.0
    n = ns(T)
    t = tvec(n)
    out = np.zeros((n, 2))
    r = rng(1000)
    # crack
    k = ns(0.006)
    cr = r.standard_normal((k, 2)) * np.exp(-np.arange(k) / (k / 4))[:, None]
    out[:k] += cr * 1.4
    # blast (LP sweep down)
    bl = noise2(n, 1001, -0.6, 0.35)
    bl = spec(bl, lambda f, tt: g_lp(f, 250 + 14000 * np.exp(-tt / 0.22), 2) * g_hp(f, 40, 2))
    bl *= env_ar(t, 0.003, 0.55)[:, None] * (1 + 0.25 * lfo_noise(n, 25, 1002))[:, None]
    out += bl * 1.2
    # body thump
    bo = butter(colored(n, 1003, -2), 'low', 320, 2) * env_ar(t, 0.004, 0.35)
    out += pan2(bo * 1.0, 0)
    # sub
    sd = sub_drop(T, 75, 26, 0.35, 1.3, 2.8, 0.003)
    out += pan2(sd * 1.15, 0)
    # shockwave whoosh outward
    sw = whoosh(1.4, [(0, 2500), (0.6, 600), (1.4, 250)], [(0, 0.0), (0.04, 1), (1.4, 0)], bw=0.7, seed=1004, corr=0.0)
    sw = widen(sw, np.interp(tvec(len(sw)), [0, 0.8], [0.6, 2.0]))
    out[:len(sw)] += sw * 0.7
    # debris: ice tinkles + fire embers falling
    times = np.sort(0.08 + r.exponential(0.55, 40))
    times = times[times < T - 1]
    ice_f = [hz(x) for x in ('E6', 'B6', 'F#7', 'A6', 'C#7')]
    out += sparkle(T, times, ice_f, 1005, None, 0.8, (0.05, 0.25), 0.22)
    emb = crackle(T, lambda tt: 160 * np.exp(-tt / 0.35) + 12 * np.exp(-tt / 1.6), 900, 6000, (0.002, 0.01), 1006,
                  None, 0.85, 1.4)
    out += emb * 0.45
    # rumble tail (camera shake)
    ru = noise2(n, 1007, -2.0, 0.7)
    ru = butter(ru, 'low', 140, 2) * env_ar(t, 0.05, 1.0)[:, None]
    out += ru * 0.6
    return out


def sfx_portal_open(T_sus, seed=1100):
    """Portal opening swell (peak at 0.4) + sustained vortex for T_sus seconds (increasing)."""
    T = 0.4 + T_sus
    n = ns(T)
    t = tvec(n)
    out = np.zeros((n, 2))
    # reverse swell into the open
    rv = whoosh(0.4, [(0, 300), (0.4, 2500)], [(0, 0), (0.39, 1.0), (0.4, 0)], bw=0.6, seed=seed, corr=0.3)
    out[:len(rv)] += rv * 0.7
    # open impact: deep soft boom + crackle burst + shimmer
    i0 = ns(0.4)
    m = ns(2.5)
    hit = sub_drop(2.5, 60, 32, 0.3, 0.9, 2.2)
    out[i0:i0 + m] += pan2(hit * 0.85, 0)
    cb = crackle(2.5, lambda tt: 120 * np.exp(-tt / 0.2) + 1, 1500, 6000, (0.0015, 0.006), seed + 1, None, 0.7, 1.6)
    out[i0:i0 + m] += cb * 0.4
    sh = sum(sine(hz(x) * (1 + 0.003 * j), m) / (1 + j) for j, x in enumerate(('D6', 'A6', 'E7', 'F#7')))
    out[i0:i0 + m] += pan2(sh * env_ar(tvec(m), 0.05, 0.8) * 0.12, 0)
    # vortex sustain
    grow = np.interp(t, [0, 0.4, 1.0, T], [0, 0.0, 0.7, 1.0])
    rot = 0.4 + 0.25 * t / T
    rph = TAU * np.cumsum(rot) / SR
    ru = butter(noise2(n, seed + 2, -2.0, 0.7), 'low', 130, 2)
    ru += pan2(sat(sine(hz('D1'), n) + 0.5 * sine(hz('D1') * 1.01, n), 1.5) * 0.4, 0)
    out += ru * (grow * (0.8 + 0.2 * np.sin(rph)))[:, None] * 0.55
    wn = colored(n, seed + 3, -1)
    fc = 800 * 2 ** (1.0 * np.sin(rph * 0.7))
    wn = spec(wn, lambda f, tt: g_bp(f, np.interp(tt, t, fc), 0.5))
    wn = flanger(wn, 1.0, 5.0, 0.25, 0.7, seed + 4, 2)
    out += pan2(wn * grow * 0.40, 0.6 * np.sin(rph))
    times = poisson_times(T, lambda tt: np.where(tt > 0.4, 4 + 3 * tt / T, 0), seed + 5)
    zr = rng(seed + 6)
    zp = grains(T, times, lambda k, q: zap(q, q.uniform(0.03, 0.1), 1.3) * q.uniform(0.2, 0.7),
                0.7 * np.sin(np.interp(times, t, rph) + zr.uniform(-0.5, 0.5, len(times))), seed + 7)
    out += zp * 0.25
    st = poisson_times(T, lambda tt: np.where(tt > 0.6, 1.4, 0), seed + 8)
    out += sparkle(T, st, [hz(x) for x in ('D7', 'A6', 'E7', 'F#6')], seed + 9, None, 0.6, (0.2, 0.6), 0.10)
    return out


def sfx_suck(d):
    T = d + 0.2
    n = ns(T)
    t = tvec(n)
    u = np.clip(t / d, 0, 1)
    out = np.zeros((n, 2))
    ri = whoosh(T, [(0, 300), (d * 0.85, 5000), (d, 7000), (T, 7000)], [(0, 0.05), (d * 0.8, 0.8), (d, 0.2), (T, 0)],
                bw=0.5, seed=1200)
    out += ri * 0.7
    sh = shepard(T, 0.8 + 3 * u, 800, 7, True, 1201) * smooth01(u * 2) * (t < d)
    out += pan2(sh * 0.22, 0)
    f = 220 * 2 ** (2 * u ** 1.5) * (1 + 0.03 * u * np.sin(TAU * (4 + 10 * u) * t))
    st = saw(f, n) * 0.3 + sine(f, n)
    st = butter(st, 'low', 3000, 2) * smooth01(u * 3) * (1 - smooth01((t - d + 0.1) / 0.1))
    out += pan2(st * 0.16, 0)
    # big whoosh up/away (Doppler falling) peaking at 0.75 d
    pk = 0.75 * d
    wu = whoosh(T, [(0, 600), (pk, 2600), (T, 400)], [(0, 0), (pk - 0.25, 0.4), (pk, 1.0), (d, 0.25), (T, 0)],
                bw=0.6, seed=1202, corr=0.3, tone=([(0, 160), (pk, 120), (T, 60)], 0.3))
    out += wu * 0.8
    sd = sub_drop(T, 45, 75, 0.5, 2.0, 2.0) * smooth01(u * 1.2) * (1 - smooth01((t - d) / 0.15))
    out += pan2(sd * 0.6, 0)
    return out


def sfx_portal_close():
    """Implosion: reverse-reverb swell into a punchy 'tchoom' + flash at 0.45 s."""
    T = 2.4
    n = ns(T)
    out = np.zeros((n, 2))
    pk = 0.45
    burst = noise2(ns(0.05), 1300, -0.5, 0.5) * np.exp(-tvec(ns(0.05)) / 0.01)[:, None]
    ir = make_ir(2.0, 1.4, 1.2, 0.0, 0, 0.01, 8000, 1301)
    rv = convolve(np.pad(burst, ((0, ns(1.2)), (0, 0))), ir)[::-1][-ns(pk):]
    rv = norm_peak(rv) * np.linspace(0.2, 1, len(rv))[:, None] ** 2
    out[:len(rv)] += rv * 0.8
    i0 = ns(pk)
    m = n - i0
    tl = tvec(m)
    hit = sub_drop(T - pk, 90, 38, 0.06, 0.4, 2.5)
    fl = butter(noise2(m, 1302, 0.0, 0.5), 'high', 2500, 2) * env_ar(tl, 0.002, 0.09)[:, None]
    sh = sum(sine(hz(x), m) for x in ('D6', 'A6', 'D7')) * env_ar(tl, 0.003, 0.5) * 0.15
    out[i0:] += pan2(hit, 0) + fl * 0.6 + pan2(sh, 0)
    return out


def sfx_tunnel(d):
    T = d
    n = ns(T)
    t = tvec(n)
    u = t / d
    out = np.zeros((n, 2))
    wn = noise2(n, 1400, -1.0, 0.35)
    wn = np.stack([flanger(wn[:, c], 0.5, 6.0, 0.4 + 1.6 * u, 0.8, 1401 + c, 3) for c in range(2)], 1)
    wn = spec(wn, lambda f, tt: g_bp(f, np.interp(tt, [0, d], [500, 2400]), np.interp(tt, [0, d], [0.9, 0.6])))
    out += wn * (0.3 + 0.7 * smooth01(u * 1.5))[:, None] * 0.75
    # ring pass-bys, accelerating
    tk, cur, k = [], 0.12, 0
    while cur < d - 0.15:
        tk.append(cur)
        cur += np.interp(cur, [0, d], [0.42, 0.09])
    r = rng(1402)
    for k, tc in enumerate(tk):
        dd = 0.24
        w = whoosh(dd, [(0, 2800), (dd * 0.5, 1400), (dd, 500)], [(0, 0), (dd * 0.5, 1), (dd, 0)], bw=0.5,
                   seed=1403 + k, corr=0.5)
        side = 0.65 if k % 2 == 0 else -0.65
        w = pan2(w, np.linspace(side * 0.3, side, len(w)))
        i0 = ns(tc)
        out[i0:i0 + len(w)] += w[:n - i0] * (0.3 + 0.5 * k / max(1, len(tk))) * 0.9
    sh = shepard(T, 0.5 + 2.0 * u, 700, 7, True, 1450) * (0.4 + 0.6 * u)
    out += pan2(sh * 0.22, 0)
    ru = butter(noise2(n, 1451, -2.0, 0.7), 'low', 110, 2) * (0.4 + 0.6 * u)[:, None]
    out += ru * 0.55
    # reverse cymbal into the exit flash (last 0.38 s)
    rc = whoosh(0.38, [(0, 5000), (0.38, 9000)], [(0, 0), (0.37, 1), (0.38, 0)], bw=0.6, seed=1452, slope=0.0)
    out[n - len(rc):] += rc * 0.8
    return out


def sfx_exit_flash():
    T = 4.0
    n = ns(T)
    t = tvec(n)
    out = np.zeros((n, 2))
    sd = sub_drop(T, 70, 30, 0.3, 1.2, 2.2)
    out += pan2(sd * 0.75, 0)
    bl = noise2(n, 1500, -0.8, 0.2)
    bl = spec(bl, lambda f, tt: g_lp(f, 400 + 9000 * np.exp(-tt / 0.3), 2))
    out += bl * env_ar(t, 0.004, 0.6)[:, None] * 0.65
    sh = sum(sine(hz(x) * (1 + 0.002 * j), n) / (1 + 0.4 * j) for j, x in enumerate(('D5', 'A5', 'D6', 'F#6', 'A6', 'E7')))
    out += widen(pan2(sh * env_ar(t, 0.02, 1.4) * 0.12, 0), 1.0)
    return out


def sfx_arrive():
    T = 2.0
    n = ns(T)
    t = tvec(n)
    out = np.zeros((n, 2))
    w = whoosh(0.45, [(0, 4500), (0.4, 500), (0.45, 400)], [(0, 0), (0.2, 0.8), (0.45, 0.1)], bw=0.6, seed=1600,
               tone=([(0, 700), (0.45, 180)], 0.15))
    out[:len(w)] += w * 0.75
    i0 = ns(0.40)
    th = sub_drop(1.0, 80, 45, 0.06, 0.25, 1.8) * 0.75
    out[i0:i0 + len(th)] += pan2(th, 0)
    bo = butter(colored(ns(0.6), 1601, -1), 'band', [120, 700], 2) * env_ar(tvec(ns(0.6)), 0.004, 0.08)
    out[i0:i0 + len(bo)] += pan2(bo * 0.5, 0)
    # close "shhp" + sparkle at 0.8
    i1 = ns(0.78)
    cs = whoosh(0.12, [(0, 2000), (0.12, 8000)], [(0, 0), (0.11, 0.6), (0.12, 0)], bw=0.5, seed=1602, slope=0.0)
    out[i1:i1 + len(cs)] += cs
    st = 0.9 + np.sort(rng(1603).exponential(0.15, 7))
    out += sparkle(T, st, [hz(x) for x in ('A6', 'D7', 'E7')], 1604, None, 0.4, (0.1, 0.3), 0.18)
    return out


def space_amb(T, seed=1700):
    n = ns(T)
    t = tvec(n)
    dr = butter(noise2(n, seed, -2.0, 0.2), 'band', [28, 110], 2) * (0.7 + 0.3 * lfo_noise(n, 0.08, seed + 1))[:, None]
    wind = noise2(n, seed + 2, -1.0, 0.1)
    c1 = 500 * 2 ** (0.8 * lfo_noise(n, 0.05, seed + 3))
    c2 = 1500 * 2 ** (0.6 * lfo_noise(n, 0.04, seed + 4))
    wind = spec(wind, lambda f, tt: 0.7 * g_bp(f, np.interp(tt, t, c1), 0.12) + 0.4 * g_bp(f, np.interp(tt, t, c2), 0.1),
                4096, 1024)
    times = poisson_times(T, lambda tt: 1.3 + 0 * tt, seed + 5)
    tw = sparkle(T, times, [hz(x) for x in ('D7', 'E7', 'F#7', 'A7', 'B7', 'D8')], seed + 6, None, 0.9, (0.15, 0.5), 0.10)
    return dr * 0.9 + wind * 0.45 + tw


def sfx_blackhole_form():
    T = 3.2
    n = ns(T)
    t = tvec(n)
    out = np.zeros((n, 2))
    # deep downward sweep cluster
    f = 40 + 150 * np.exp(-t / 0.5)
    cl = saw(f, n) + saw(f * 1.5 * 1.004, n) * 0.6 + saw(f * 0.5, n) * 0.6
    cl = spec(cl, lambda ff, tt: g_lp(ff, 1200 * np.exp(-tt / 0.8) + 150, 2))
    cl *= env_pts(t, [(0, 0), (0.15, 0.8), (1.4, 0.6), (T, 0)])
    out += pan2(cl * 0.35, 0.35)
    # grains pulled in toward the hole (pan converging right, pitch dropping)
    times = poisson_times(1.6, lambda tt: 90 * np.exp(-0.5 * ((tt - 0.6) / 0.35) ** 2) + 5, 1800)
    r = rng(1801)
    p0 = r.uniform(-0.9, 0.9, len(times))
    pans = p0 + (0.55 - p0) * np.clip(np.asarray(times) / 1.4, 0, 1)
    gr = grains(T, times, lambda k, q: click(q, q.uniform(800, 3500) * (1 - 0.4 * times[k] / 1.6), 0.008, 2)
                * q.uniform(0.3, 1), pans, 1802)
    out += gr * 0.35
    # reverse swell into settle thud at 1.3
    rv = whoosh(1.3, [(0, 3000), (1.3, 300)], [(0, 0), (1.25, 0.7), (1.3, 0)], bw=0.7, seed=1803)
    out[:len(rv)] += pan2(rv, 0.4) * 0.6
    i0 = ns(1.3)
    hit = sub_drop(T - 1.3, 55, 28, 0.25, 0.8, 2.6)
    out[i0:] += pan2(hit * 0.85, 0.3)
    return out


def blackhole_bed(T, seed=1850):
    n = ns(T)
    t = tvec(n)
    pulse = 0.55 + 0.45 * np.sin(TAU * 0.7 * t) ** 2
    sub = sat(sine(33, n) + 0.6 * sine(66.3, n) + 0.3 * sine(99.2, n), 1.6) * pulse
    sub = butter(sub, 'low', 400, 2)
    ru = butter(colored(n, seed, -2.0), 'band', [40, 160], 2) * pulse
    rot = TAU * 0.45 * t
    disk = colored(n, seed + 1, -1.0)
    disk = spec(disk, lambda f, tt: g_bp(f, 1400 * 2 ** (0.5 * np.sin(TAU * 0.45 * tt)), 0.6), 2048, 512)
    disk *= 0.6 + 0.4 * np.sin(rot)
    hiss = butter(colored(n, seed + 2, 0.0), 'high', 6000, 2) * (0.5 + 0.5 * np.sin(rot + 1.0))
    out = pan2(sub * 0.65 + ru * 0.5, 0.25) + pan2(disk * 0.22, 0.55 + 0.25 * np.sin(rot)) + pan2(hiss * 0.04, 0.6)
    return out


def sfx_strange(d, pan_fn, t_abs0, tail=1.5):
    """Sling-ring: dense spark sizzle spinning on a growing circle, whir, crackle, portal roar."""
    T = d + tail
    n = ns(T)
    t = tvec(n)
    u = np.clip(t / d, 0, 1)
    out = np.zeros((n, 2))
    inten = smooth01(t / 0.6) * (0.45 + 0.55 * u) * np.where(t < d, 1, np.exp(-(t - d) / 0.5))
    spin = 7.0 - 3.5 * u
    sph = TAU * np.cumsum(spin) / SR
    # sizzle bed (dense micro-sparks) via AM noise
    sz = noise2(n, 1900, 0.0, 0.3)
    sz = spec(sz, lambda f, tt: g_hp(f, 3500, 2) * g_lp(f, 14000, 2))
    micro = np.clip(1 + 1.6 * colored(n, 1901, 0.0), 0, None) ** 3
    micro = uniform_filter1d(micro, 30)
    am = 0.55 + 0.45 * np.sin(sph)
    out += sz * (micro * am * inten)[:, None] * 0.11
    # spark crackle
    cr = crackle(T, lambda tt: np.interp(tt, [0, 0.5, d, d + 0.4, T], [0, 70, 160, 20, 0]), 1500, 8000,
                 (0.0008, 0.005), 1902, lambda tt: pan_fn(t_abs0 + tt) + 0.35 * np.sin(np.interp(tt, t, sph)), 0.25, 1.6)
    out += cr * 0.5
    # whir: band noise whose centre spins
    wh = colored(n, 1903, -1.0)
    fc = 1200 * 2 ** (0.5 * np.sin(sph)) * (1 - 0.35 * u)
    wh = spec(wh, lambda f, tt: g_bp(f, np.interp(tt, t, fc), 0.35))
    out += pan2(wh * inten * 0.35, pan_fn(t_abs0 + t) + 0.3 * np.sin(sph))
    # warm portal roar growing
    ro = noise2(n, 1904, -1.6, 0.5)
    ro = spec(ro, lambda f, tt: g_bp(f, 350, 1.0))
    out += ro * (inten * u)[:, None] * 0.55
    # shimmering ring tones
    rt = sum(sine(hz(x) * (1 + 0.004 * np.sin(TAU * 0.3 * t + j)), n) / (1 + j) for j, x in enumerate(('A5', 'E6', 'A6')))
    out += pan2(rt * inten * u * 0.05, 0)
    return out


def sfx_wipe(d):
    T = d + 1.0
    n = ns(T)
    t = tvec(n)
    out = np.zeros((n, 2))
    pk = 0.55 * d
    w = whoosh(T, [(0, 300), (pk, 2800), (d, 600), (T, 300)], [(0, 0.1), (pk, 1.0), (d, 0.15), (d + 0.3, 0)],
               bw=0.7, seed=2000, corr=0.1, tone=([(0, 90), (pk, 140), (T, 50)], 0.25))
    w = widen(w, np.interp(t, [0, pk, d], [0.6, 1.8, 2.0]))
    out += w * 0.9
    sz = noise2(n, 2001, 0.0, 0.0)
    sz = spec(sz, lambda f, tt: g_hp(f, 3500, 2))
    micro = uniform_filter1d(np.clip(1 + 1.6 * colored(n, 2002, 0.0), 0, None) ** 3, 30)
    edge = env_pts(t, [(0, 0.5), (pk, 1.0), (d, 0.6), (d + 0.35, 0)])
    out += widen(sz * (micro * edge)[:, None] * 0.10, np.interp(t, [0, d], [0.5, 2.0]))
    cr = crackle(T, lambda tt: np.interp(tt, [0, pk, d, d + 0.8], [150, 220, 60, 0]), 1500, 8000, (0.0008, 0.005), 2003,
                 None, 0.95, 1.6)
    out += cr * 0.4
    # reality snap at d - 0.05
    i0 = ns(d - 0.05)
    m = ns(0.6)
    pop = butter(rng(2004).standard_normal(m), 'band', [600, 5000], 2) * np.exp(-tvec(m) / 0.012)
    pop += sub_drop(0.6, 120, 60, 0.03, 0.08, 1.5) * 0.6
    out[i0:i0 + m] += pan2(pop * 0.6, 0)
    return out


def sfx_claude_accent():
    T = 2.5
    n = ns(T)
    t = tvec(n)
    y = sum(sine(hz(x) * (1 + 0.002 * j), n) / (1 + 0.6 * j) for j, x in enumerate(('D5', 'F#5', 'A5', 'E6')))
    y *= env_pts(t, [(0, 0), (0.25, 1), (0.9, 0.6), (T, 0)])
    st = 0.05 + np.sort(rng(2100).uniform(0, 0.7, 6))
    sp = sparkle(T, st, [hz(x) for x in ('A6', 'D7', 'F#7')], 2101, None, 0.5, (0.1, 0.3), 0.6)
    return widen(pan2(y * 0.25, 0), 1.0) + sp * 0.5


# =============================================================================================
# SFX timeline
# =============================================================================================
def build_sfx():
    bus = Bus(['bath', 'studio', 'tunnel', 'space'])
    panR = track_pan('R')
    panL = track_pan('L')

    # intro accent on "Claude"
    bus.add(sfx_claude_accent(), 2.80, -24, 0, 0.3, name='claude_accent')

    # removals: magic eraser (scan -> burn -> particles), pitched up a step each time
    for key, base, pan, size, sd in (('clean_clutter', hz('A5'), 0.5, 1.0, 100),
                                    ('clean_flush', hz('C#6'), 0.12, 0.55, 120),
                                    ('clean_shower', hz('E6'), -0.6, 0.9, 140)):
        a, b = EV[key]
        bus.add(sfx_eraser(b - a, base, size, sd), a, -9, pan, 0.35, name=key)

    # studio blackout
    bus.add(sfx_studio(), EV['studio'][0] - 0.05, -2, 0, 0.35, name='studio')
    # studio room tone (very faint air) through to the tunnel
    t0, t1 = EV['studio'][0] + 1.2, EV['tunnel'][0] + 0.05
    rt = butter(noise2(ns(t1 - t0), 300, -2.0, 0.3), 'band', [40, 600], 2)
    rt *= env_pts(tvec(len(rt)), [(0, 0), (1.5, 1), (t1 - t0 - 0.3, 1), (t1 - t0, 0)])[:, None]
    bus.add(rt, t0, -46, 0, 0.0, name='studio_tone')

    # suit
    a, b = EV['suit']
    bus.add(sfx_suit(b - a), a, -8, 0, 0.3, name='suit')

    # anticipation
    a, b = EV['anticipation']
    bus.add(sfx_anticipation(EV['fire_on'] - a, panR, panL, a), a, -10, 0, 0.25, name='anticipation')

    # fire ignition + bed (until the fusion)
    tf = EV['fire_on']
    bus.add(sfx_fire_ignite(), tf - 0.16, -3, panR(tf), 0.3, name='fire_ignite')
    T = EV['fusion'][0] + 0.4 - (tf + 0.1)
    fb = fire_bed(T, panR, tf + 0.1)
    fb *= env_pts(tvec(len(fb)), [(0, 0), (0.25, 1), (T - 0.5, 1), (T, 0)])[:, None]
    bus.add(fb, tf + 0.1, -13, 0, 0.2, name='fire_bed')

    # ice formation + bed
    ti = EV['ice_on']
    bus.add(sfx_ice_form(), ti - 0.25, -5, panL(ti), 0.35, name='ice_form')
    T = EV['fusion'][0] + 0.4 - (ti + 0.55)
    ib = ice_bed(T, panL, ti + 0.55)
    ib *= env_pts(tvec(len(ib)), [(0, 0), (0.3, 1), (T - 0.5, 1), (T, 0)])[:, None]
    bus.add(ib, ti + 0.55, -9, 0, 0.3, name='ice_bed')

    # fusion
    a, b = EV['fusion']
    fu = sfx_fusion(b - a, a, panR, panL)
    fu *= env_pts(tvec(len(fu)), [(0, 0), (0.35, 1), (b - a, 1)])[:, None]
    bus.add(fu, a, -8, 0, 0.3, name='fusion')

    # release + charge (ends 22 ms before the boom)
    bus.add(sfx_release_charge(EV['release'], EV['finale_boom']), EV['release'], -6, 0, 0.3, name='charge')
    # finale boom + debris tail
    bus.add(sfx_boom(), EV['finale_boom'], 0, 0, 0.45, name='finale_boom')

    # ceiling portal: opening swell peaks at ceil_portal start, vortex sustains until the suck
    a, b = EV['ceil_portal']
    T_sus = EV['suck'][1] - a
    po = sfx_portal_open(T_sus)
    po *= env_pts(tvec(len(po)), [(0, 1), (0.4 + T_sus - 0.25, 1), (0.4 + T_sus, 0)])[:, None]
    bus.add(po, a - 0.4, -8, 0, 0.35, name='ceil_portal')
    # suction
    a, b = EV['suck']
    bus.add(sfx_suck(b - a), a, -5, 0, 0.3, name='suck')
    # collapse + flash (flash lands at close + 0.45)
    a, b = EV['ceil_portal_close']
    bus.add(sfx_portal_close(), a, -4, 0, 0.35, space='studio', name='portal_close')
    # tunnel
    a, b = EV['tunnel']
    tu = sfx_tunnel(b - a)
    tu *= env_pts(tvec(len(tu)), [(0, 0), (0.3, 1), (b - a, 1)])[:, None]
    bus.add(tu, a, -6, 0, 0.3, space='tunnel', name='tunnel')
    bus.add(sfx_exit_flash(), b, -4, 0, 0.5, space='space', name='exit_flash')
    # arrival
    bus.add(sfx_arrive(), EV['arrive'][0] + 0.05, -6, 0, 0.45, space='space', name='arrive')
    # space ambience until the wipe
    t0, t1 = EV['space'][0], EV['strange_wipe'][1]
    sa = space_amb(t1 - t0)
    sa *= env_pts(tvec(len(sa)), [(0, 0), (1.5, 1), (t1 - t0 - 1.0, 1), (t1 - t0, 0)])[:, None]
    bus.add(sa, t0, -20, 0, 0.35, space='space', name='space_amb')
    # black hole
    a, b = EV['blackhole']
    bus.add(sfx_blackhole_form(), a, -5, 0, 0.4, space='space', name='blackhole_form')
    t1 = EV['strange_wipe'][0] + 0.4
    T = t1 - (a + 0.9)
    bh = blackhole_bed(T)
    bh *= env_pts(tvec(len(bh)), [(0, 0), (0.8, 1), (EV['strange'][0] - a - 0.9, 1), (T - 0.4, 0.45), (T, 0)])[:, None]
    bus.add(bh, a + 0.9, -13, 0, 0.3, space='space', name='blackhole_bed')
    # Doctor Strange sling-ring
    a, b = EV['strange']
    pan_s = lambda tt: np.where(tt < 72.45, np.clip(panR(tt), -0.6, 0.6) * 0.8, np.clip(panR(tt), -0.6, 0.6) * 0.3)
    bus.add(sfx_strange(EV['strange_wipe'][0] - a, pan_s, a, 1.6), a, -8, 0, 0.3, space='space', name='strange')
    # wipe
    a, b = EV['strange_wipe']
    bus.add(sfx_wipe(b - a), a, -5, 0, 0.3, space='space', name='strange_wipe')
    return bus


# =============================================================================================
# music bed
# =============================================================================================
_NOTE = {'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
         'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11}


def hz(name):
    k = 2 if len(name) > 2 and name[1] in '#b' else 1
    midi = 12 * (int(name[k:]) + 1) + _NOTE[name[:k]]
    return 440.0 * 2 ** ((midi - 69) / 12)


def pad_tone(notes, d, seed, bright=2200, voices=3, detune=9.0, harm=7, roll=1.6, width=0.7, amp=1.0):
    n = ns(d)
    r = rng(seed)
    out = np.zeros((n, 2))
    for j, nm in enumerate(notes):
        f = hz(nm) if isinstance(nm, str) else nm
        drift = 1 + 0.0012 * lfo_noise(n, 0.25, seed + 31 * j)
        for v in range(voices):
            c = (v - (voices - 1) / 2) * detune + r.normal(0, 1.5)
            ph = phase(f * 2 ** (c / 1200) * drift, n) + r.uniform(0, 1)
            y = np.zeros(n)
            for k in range(1, harm + 1):
                if f * k > 9000:
                    break
                y += np.sin(TAU * k * ph + r.uniform(0, TAU)) / k ** roll
            p = ((v - (voices - 1) / 2) / max(1, (voices - 1) / 2)) * width * (1 if j % 2 else -1)
            out += pan2(y * (0.8 + 0.2 * np.sin(TAU * r.uniform(0.05, 0.2) * tvec(n) + r.uniform(0, TAU)))[:, None][:, 0], p)
    out = spec(out, lambda ff, tt: g_lp(ff, bright, 2) * g_hp(ff, 50, 2), 4096, 1024)
    return out * amp / (len(notes) * voices) ** 0.5


def pad_air(notes, d, seed, bw=0.006, harm=6, roll=1.0, lp=3500, amp=1.0, formant=None):
    """Airy 'breath/choir' pad: pink noise through a harmonic comb at the chord's notes."""
    n = ns(d)
    out = np.zeros((n, 2))
    nfft, hop = 16384, 4096
    for c in range(2):
        x = colored(n + nfft, seed + c, -1.0)
        f, tt, Z = ss.stft(x, SR, window='hann', nperseg=nfft, noverlap=nfft - hop)
        mask = np.zeros_like(f)
        for nm in notes:
            f0 = hz(nm) if isinstance(nm, str) else nm
            for k in range(1, harm + 1):
                fk = f0 * k
                if fk > 12000:
                    break
                sig = max(2.2, fk * bw)
                mask += np.exp(-0.5 * ((f - fk) / sig) ** 2) / k ** roll
        mask *= g_lp(f, lp, 2) * g_hp(f, 45, 2)
        if formant is not None:
            mask *= 0.35 + sum(g_bp(f, fc, 0.25) for fc in formant)
        _, y = ss.istft(Z * mask[:, None], SR, window='hann', nperseg=nfft, noverlap=nfft - hop)
        out[:, c] = y[nfft // 2:nfft // 2 + n] if len(y) >= n + nfft // 2 else np.pad(y, (0, n))[:n]
    out /= (np.sqrt((out ** 2).mean()) + 1e-12)
    return out * amp * 0.1


def pluck(f, dur=1.8, bright=1.0, seed=0):
    n = ns(dur)
    t = tvec(n)
    r = rng(seed)
    base_tau = 0.9 * (440 / f) ** 0.35
    y = np.zeros(n)
    for ratio, a, tf in ((1, 1, 1.0), (2.0, 0.30, 0.5), (3.0, 0.12, 0.33), (4.07, 0.10 * bright, 0.2),
                         (5.2, 0.05 * bright, 0.14), (6.8, 0.03 * bright, 0.09)):
        if f * ratio > 16000:
            continue
        y += a * np.sin(TAU * f * ratio * t + r.uniform(0, TAU)) * np.exp(-t / (base_tau * tf))
    k = ns(0.004)
    y[:k] += butter(r.standard_normal(k), 'low', 3000, 2) * np.linspace(1, 0, k) * 0.25
    return y * (1 - np.exp(-t / 0.0015))


def tape_stop(x, t_stop, dur):
    n = len(x)
    t = tvec(n)
    u = np.clip((t - t_stop) / dur, 0, 1)
    speed = np.where(t < t_stop, 1.0, (1 - u) ** 2)
    y = resample_speed(x, speed)
    amp = np.where(t < t_stop, 1.0, np.sqrt(np.clip(1 - u, 0, 1)))
    return y * amp[:, None]


class Music:
    def __init__(self):
        self.bus = np.zeros((N, 2))
        self.send = np.zeros((N, 2))

    def put(self, x, t0, gain_db=0.0, rev=0.35, buf=None, fin=0.01, fout=0.05):
        x = fade(np.asarray(x, np.float64), fin, fout) * undb(gain_db)
        i0 = int(round(t0 * SR))
        a0 = max(0, -i0)
        i0 = max(0, i0)
        i1 = min(N, i0 + len(x) - a0)
        if i1 <= i0:
            return
        tgt = self.bus if buf is None else buf
        tgt[i0:i1] += x[a0:a0 + i1 - i0]
        if rev > 0 and buf is None:
            self.send[i0:i1] += x[a0:a0 + i1 - i0] * rev


def chord_env(d, att, rel):
    t = tvec(ns(d))
    return np.minimum(smooth01(t / att), smooth01((d - t) / rel))[:, None]


def build_music():
    M = Music()
    # ------------------------------------------------ A. bathroom: light / curious (0 .. 13.05)
    bath = np.zeros((N, 2))
    chords = [(0.0, ['D3', 'A3', 'E4', 'F#4']), (3.25, ['G2', 'D3', 'F#3', 'B3', 'C#4']),
              (6.5, ['B2', 'F#3', 'A3', 'D4', 'E4']), (9.75, ['G2', 'D3', 'A3', 'B3', 'E4'])]
    for k, (t0, ch) in enumerate(chords):
        t1 = chords[k + 1][0] if k + 1 < len(chords) else 13.6
        d = t1 - t0 + 1.2
        x = pad_tone(ch, d, 3000 + k, bright=2600, amp=1.0) * chord_env(d, 0.9 if k else 1.6, 1.2)
        x += pad_air([c[:-1] + str(int(c[-1]) + 1) for c in ch], d, 3100 + k, lp=6000, amp=0.5) * chord_env(d, 1.0, 1.2)
        M.put(x, t0 - (0.0 if k == 0 else 0.6), -21, buf=bath)
    # curious glass plucks, 8th grid at 92 bpm
    e8 = 60 / 92 / 2
    pat = [0, None, 2, 1, None, 3, 2, None, 0, None, 3, None, 2, 1, None, None]
    r = rng(3200)
    tt = 0.35
    step = 0
    while tt < 12.95:
        ch = [c for t0, c in chords if t0 <= tt][-1]
        sel = pat[step % len(pat)]
        if sel is not None:
            nm = ch[(sel + (step // 16)) % len(ch)]
            f = hz(nm) * (4 if hz(nm) < 200 else 2)
            v = r.uniform(0.6, 1.0) * (0.75 if step % 2 else 1.0)
            M.put(pluck(f, 1.8, 0.8, 3300 + step) * v, tt + r.normal(0, 0.006), -27 + 3 * smooth01(tt / 3), buf=bath,
                  fin=0.0005, fout=0.2)
        tt += e8
        step += 1
    bath = tape_stop(bath, EV['studio'][0], 0.55)
    M.bus += bath
    M.send += bath * 0.45

    # ------------------------------------------------ B. studio: dark (13.3 .. 19.9)
    a = EV['studio'][0] + 0.2
    d = EV['anticipation'][0] + 0.4 - a
    x = pad_air(['D2', 'A2', 'D3', 'F3', 'A3'], d, 3400, lp=900, amp=1.0) * chord_env(d, 2.0, 0.8)
    x += pad_tone(['D2', 'A2', 'F3'], d, 3401, bright=600, amp=0.8) * chord_env(d, 2.5, 0.8)
    dr = sat(sine(hz('D1'), ns(d)) + 0.5 * sine(hz('D2'), ns(d)), 1.4)
    x += pan2(dr * 0.18, 0) * chord_env(d, 2.0, 0.8)
    M.put(x, a, -20, rev=0.3)
    a, b = EV['suit']
    d = b - a + 2.2
    x = pad_air(['E5', 'A5', 'C6'], d, 3410, lp=9000, bw=0.003, amp=0.8) * chord_env(d, 1.2, 1.2)
    M.put(x, a + 0.2, -27, rev=0.5)

    # ------------------------------------------------ C. powers: tense / rising (19.7 .. 32.62)
    B = [0, 6, 13, 17, 22, 26]
    Tm = [EV['anticipation'][0], EV['fire_on'], 26.56 if EV['ice_on'] == 26.56 else EV['ice_on'], EV['fusion'][0],
          EV['release'], EV['finale_boom']]
    btime = lambda b: np.interp(b, B, Tm)
    roots = lambda b: hz('D2') if b < 6 else hz('A#1') if b < 13 else hz('G1') if b < 17 else hz('A1') if b < 22 else hz('D2')
    off = [0, 0, 12, 0, 7, 0, 12, 0]
    ost = np.zeros((N, 2))
    bb, k = 0.0, 0
    while bb < 25.9:
        tb = btime(bb)
        nb = bb + (0.5 if bb < 22 else 0.25)
        dur = (btime(nb) - tb) * 0.9
        f = roots(bb) * 2 ** (off[k % 8] / 12)
        n = ns(dur + 0.08)
        t = tvec(n)
        y = saw(f, n) + 0.6 * saw(f * 1.004, n) + 0.8 * sine(f / 2, n)
        e = (1 - np.exp(-t / 0.004)) * np.exp(-t / (dur * 0.6))
        acc = 1.15 if k % 8 == 0 else 1.0
        M.put(pan2(y * e * acc, 0.15 * (1 if k % 2 else -1)), tb, 0, buf=ost, fin=0.001, fout=0.02)
        bb = nb
        k += 1
    t0o, t1o = Tm[0], Tm[-1]
    ost = spec(ost, lambda ff, tt: g_lp(ff, np.interp(tt, [0, t0o, Tm[1], Tm[3], t1o], [250, 250, 500, 1000, 2600]), 3),
               2048, 512)
    lvl = env_pts(tvec(N), [(t0o, 0), (t0o + 1.5, 0.35), (Tm[1], 0.55), (Tm[3], 0.8), (t1o - 0.12, 1.0), (t1o - 0.1, 0)])
    M.bus += ost * lvl[:, None] * undb(-25)
    M.send += ost * lvl[:, None] * undb(-25) * 0.2
    # sustained "string" chords
    sc = [(Tm[0], ['D3', 'F3', 'A3']), (Tm[1], ['A#2', 'F3', 'D4', 'C4']), (Tm[2], ['G2', 'D3', 'A#3', 'A4']),
          (Tm[3], ['A2', 'D3', 'E3', 'G3', 'E4']), (Tm[3] + 1.2, ['A2', 'C#3', 'E3', 'G3', 'C#4']),
          (Tm[4], ['D3', 'A3', 'D4', 'E4', 'A4'])]
    for k, (t0, ch) in enumerate(sc):
        t1 = sc[k + 1][0] if k + 1 < len(sc) else t1o - 0.1
        d = t1 - t0 + (0.6 if k + 1 < len(sc) else 0.0)
        x = pad_tone(ch, d, 3500 + k, bright=1200 + 350 * k, voices=3, harm=9, roll=1.1, amp=1.0)
        x *= chord_env(d, 0.6 if k else 2.0, 0.6 if k + 1 < len(sc) else 0.02)
        M.put(x, t0 - (0.3 if k else 0), -24 + 1.4 * k, rev=0.3, fout=0.01)
    # low cinematic hits on the section changes + 16th ticks in the fusion, roll into the boom
    for t0, g in ((Tm[1], -13), (Tm[2], -14), (Tm[3], -13), (Tm[4], -12)):
        n = ns(1.5)
        tl = tvec(n)
        hit = sub_drop(1.5, 110, 48, 0.05, 0.35, 2.2) + butter(colored(n, int(t0 * 10), -1), 'low', 500, 2) * env_ar(tl, 0.002, 0.08) * 0.6
        M.put(pan2(hit, 0), t0, g, rev=0.25, fin=0.0005)
    bb = 17.0
    k = 0
    while bb < 25.92:
        tb = btime(bb)
        n = ns(0.05)
        tk = butter(colored(n, 3600 + k, 0.0), 'high', 6000, 2) * np.exp(-tvec(n) / 0.008)
        acc = 1.0 if k % 4 == 0 else 0.55
        g = -36 + 10 * smooth01((bb - 17) / 9)
        M.put(pan2(tk * acc, 0.3 * np.sin(k)), tb, g, rev=0.15, fin=0.0003, fout=0.01)
        bb += 0.25 if bb < 22 else 0.125
        k += 1

    # ------------------------------------------------ D. calm after the boom (34.0 .. 43.6)
    sc = [(34.0, ['D3', 'A3', 'E4', 'F4'], 4.8), (38.4, ['A#2', 'F3', 'A3', 'D4'], 5.4)]
    for k, (t0, ch, d) in enumerate(sc):
        x = pad_tone(ch, d + 0.8, 3700 + k, bright=1100, amp=1.0) * chord_env(d + 0.8, 2.5 if k == 0 else 1.0, 1.0)
        x += pad_air(ch, d + 0.8, 3710 + k, lp=2500, amp=0.7) * chord_env(d + 0.8, 2.5 if k == 0 else 1.0, 1.0)
        M.put(x, t0, -24, rev=0.4)
    d = 43.6 - 34.6
    dr = sat(sine(hz('D1'), ns(d)), 1.3) * chord_env(d, 3.0, 1.0)[:, 0]
    M.put(pan2(dr * 0.6, 0), 34.6, -28, rev=0.0)

    # ------------------------------------------------ E. portal: mystery -> wonder (43.4 .. 50.55)
    a = 43.4
    d = EV['ceil_portal'][0] - a + 0.6
    x = pad_air(['A#2', 'D3', 'G3', 'A3', 'D4'], d, 3800, lp=3000, amp=1.0) * chord_env(d, 1.5, 0.6)
    x += pad_tone(['A#2', 'D3', 'G3', 'A3'], d, 3801, bright=1500, amp=0.7) * chord_env(d, 1.5, 0.6)
    M.put(x, a, -24, rev=0.45)
    a = EV['ceil_portal'][0] - 0.1
    b = EV['ceil_portal_close'][0]
    d = b - a
    ch = ['D#2', 'A#2', 'G3', 'D4', 'A4', 'D5']
    x = pad_air(ch, d, 3810, lp=5000, amp=1.0, formant=(750, 1150)) * chord_env(d, 1.2, 0.05)
    x += pad_tone(ch[:4], d, 3811, bright=1800, amp=0.8) * chord_env(d, 1.0, 0.05)
    lvl = env_pts(tvec(ns(d)), [(0, 0.7), (EV['suck'][0] - a, 0.85), (d, 1.0)])
    x *= lvl[:, None]
    # pitch rises during the suck (variable speed read-ahead)
    ts = EV['suck'][0] - a
    tl = tvec(ns(d))
    speed = 1 + 0.26 * smooth01((tl - ts) / (d - ts)) ** 1.5
    x = resample_speed(np.pad(x, ((0, ns(1.0)), (0, 0))), speed)
    M.put(x, a, -22, rev=0.45, fout=0.03)

    # ------------------------------------------------ F. tunnel: Bb/D -> C/D -> D (51.0 .. 53.55)
    a, b = EV['tunnel']
    mid = (a + b) / 2
    for k, (t0, t1, ch) in enumerate(((a, mid, ['D2', 'A#2', 'F3', 'D4', 'F4']), (mid, b, ['D2', 'C3', 'G3', 'E4', 'G4']))):
        d = t1 - t0 + 0.25
        x = pad_tone(ch, d, 3900 + k, bright=2500, voices=4, harm=9, roll=1.0, amp=1.0)
        x *= chord_env(d, 0.35, 0.25) * np.interp(tvec(ns(d)), [0, d], [0.7, 1.0])[:, None]
        M.put(x, t0, -21 + k, rev=0.35)

    # ------------------------------------------------ G. space: vast / awe (53.55 .. 62.2)
    sc = [(EV['space'][0], 57.6, ['D2', 'A2', 'F#3', 'E4', 'A4', 'C#5']),
          (57.4, 60.0, ['D2', 'B2', 'F#3', 'C#4', 'D4', 'F#4']),
          (59.8, EV['blackhole'][0] + 0.3, ['D2', 'G2', 'D3', 'F#3', 'B3', 'C#4'])]
    for k, (t0, t1, ch) in enumerate(sc):
        d = t1 - t0 + 0.6
        x = pad_air(ch, d, 4000 + k, lp=7000, amp=1.2) * chord_env(d, 1.8 if k == 0 else 1.0, 0.8)
        x += pad_tone(ch, d, 4010 + k, bright=2400, voices=4, amp=0.9) * chord_env(d, 1.5 if k == 0 else 1.0, 0.8)
        x += pad_air([c[:-1] + str(int(c[-1]) + 2) for c in ch[2:]], d, 4020 + k, lp=12000, bw=0.003, amp=0.35) * chord_env(d, 2.0, 0.8)
        M.put(widen(x, 1.4), t0, -21, rev=0.6)
    # celestial chimes
    r = rng(4100)
    penta = ['D5', 'E5', 'F#5', 'A5', 'B5', 'D6', 'E6', 'F#6']
    for tt in poisson_times(EV['strange'][0] - 54.6, lambda q: 0.55 + 0 * q, 4101) + 54.6:
        if EV['blackhole'][0] - 0.3 < tt < EV['blackhole'][0] + 2.0:
            continue
        nm = penta[int(r.integers(len(penta)))]
        b_ = bell(hz(nm), 3.0, ratio=r.choice([1.0, 2.0, 1.41]), index=r.uniform(0.6, 1.5), tau=1.1, tau_i=0.4, glass=0.15)
        M.put(pan2(b_, r.uniform(-0.8, 0.8)), tt, -33 + r.uniform(-3, 2), rev=0.9, fin=0.001, fout=0.3)

    # ------------------------------------------------ H. black hole: dread (62.0 .. 70.2)
    sc = [(EV['blackhole'][0], 65.6, ['D2', 'D#3', 'G3', 'A#3']), (65.4, 68.0, ['D2', 'G2', 'A#2', 'D3', 'G3']),
          (67.8, EV['strange'][0] + 0.2, ['D2', 'A#2', 'F3', 'A3', 'D4'])]
    for k, (t0, t1, ch) in enumerate(sc):
        d = t1 - t0 + 0.6
        x = pad_air(ch, d, 4200 + k, lp=2200 + 600 * k, amp=1.1) * chord_env(d, 1.2, 0.8)
        x += pad_tone(ch, d, 4210 + k, bright=900 + 300 * k, amp=0.9) * chord_env(d, 1.2, 0.8)
        M.put(widen(x, 1.3), t0, -21, rev=0.55)
    d = EV['strange'][0] + 0.5 - EV['blackhole'][0]
    dr = sat(sine(hz('D1'), ns(d)) + 0.4 * sine(hz('D#1') * 1.0, ns(d)) * 0.5, 1.4) * chord_env(d, 1.0, 0.6)[:, 0]
    M.put(pan2(dr, 0), EV['blackhole'][0], -27, rev=0.2)

    # ------------------------------------------------ I. Strange: mystic -> warm D major ending
    a = EV['strange'][0]
    w0 = EV['strange_wipe'][0]
    sc = [(a, 72.0, ['D2', 'A#2', 'F3', 'C4', 'E4']), (71.8, w0 + 0.3, ['A1', 'A2', 'D3', 'E3', 'G3', 'D4'])]
    for k, (t0, t1, ch) in enumerate(sc):
        d = t1 - t0 + 0.4
        x = pad_tone(ch, d, 4300 + k, bright=1600 + 900 * k, voices=3, harm=9, roll=1.2, amp=1.0)
        x += pad_air(ch, d, 4310 + k, lp=4000, amp=0.8)
        x *= chord_env(d, 0.8, 0.4) * np.interp(tvec(ns(d)), [0, d], [0.8, 1.1])[:, None]
        M.put(x, t0, -22, rev=0.4)
    t0 = w0 + 0.15
    d = N / SR - t0
    ch = ['D2', 'A2', 'D3', 'F#3', 'A3', 'E4', 'F#4']
    x = pad_tone(ch, d, 4400, bright=1900, voices=4, amp=1.0) + pad_air(ch, d, 4401, lp=4500, amp=0.8)
    x *= np.minimum(smooth01(tvec(ns(d)) / 0.6), smooth01((d - tvec(ns(d))) / 0.9))[:, None]
    M.put(widen(x, 1.3), t0, -20, rev=0.45)
    tb = EV['strange_wipe'][1] - 0.05
    M.put(pan2(bell(hz('D6'), 1.5, 2.0, 1.0, 0.6, 0.2, glass=0.1), 0), tb, -28, rev=0.6, fin=0.001, fout=0.2)
    return M


# =============================================================================================
# mix
# =============================================================================================
def black_hole_warp(x):
    """Gravitational pitch-sag of the background while the black hole forms, then cross-fade back."""
    a = EV['blackhole'][0]
    t = tvec(N)
    dmax = 0.10
    u = np.clip((t - a) / 1.25, 0, 1)
    dly = dmax * smooth01(u) * SR
    xf = smooth01((t - (a + 1.25)) / 0.45)   # 0 -> 1 back to the dry
    i0, i1 = ns(a - 0.05), ns(a + 1.9)
    y = x.copy()
    for c in range(2):
        seg = x[:, c]
        wv = vdelay(seg[i0 - ns(0.2):i1], dly[i0 - ns(0.2):i1])[ns(0.2):]
        y[i0:i1, c] = wv * (1 - xf[i0:i1]) + seg[i0:i1] * xf[i0:i1]
    return y


def split3(x, f_lo=220.0, f_hi=5000.0):
    lo = butter(x, 'low', f_lo, 4, zero_phase=True)
    hi = butter(x, 'high', f_hi, 4, zero_phase=True)
    return lo, x - lo - hi, hi


def duck(bg, voice, vad, depth, adapt=True, target=15.0, protect=None):
    """3-band side-chain ducking. depth = (low, mid, high) dB at full voice activity."""
    hop = 0.01
    vs = vad.astype(float)
    # look-ahead 50 ms, attack 25 ms, release 350 ms
    act = follow(maximum_filter1d(vs, 11, origin=-5), 0.025, 0.35, hop)
    red = np.zeros((len(act), 3))
    for b in range(3):
        red[:, b] = depth[b] * act
    bands = split3(bg)
    kv = kweight(voice)
    if adapt:
        for it in range(3):
            cur = sum(bands[b] * undb(-to_samples(red[:, b], 480))[:, None] for b in range(3))
            tcv, lv = loud_curve(None, 0.2, hop, weighted=kv)
            tcb, lb = loud_curve(None, 0.2, hop, weighted=kweight(cur))
            need = np.zeros(len(act))
            k = min(len(lv), len(need))
            off = int(round((0.1) / hop))       # curve frame j is centred at j*hop + 0.1 s
            need_c = np.clip(lb[:k] - (lv[:k] - target), 0, 14)
            need[off:off + k] = need_c[:len(need) - off]
            need *= act
            need = maximum_filter1d(need, 9)
            need = follow(need, 0.02, 0.3, hop)
            if protect is not None:
                need *= protect
            red[:, 1] += need * 0.9
            red[:, 2] += need * 0.7
            red[:, 0] += need * 0.35
    red[:, 0] = np.minimum(red[:, 0], 12)
    red[:, 1] = np.minimum(red[:, 1], 24)
    red[:, 2] = np.minimum(red[:, 2], 18)
    out = sum(bands[b] * undb(-to_samples(red[:, b], 480))[:, None] for b in range(3))
    return out, red


def tp_limit(x, ceil_db=-1.2, look_ms=1.5, rel_ms=80.0):
    up = ss.resample_poly(x, 4, 1, axis=0)
    pk = np.abs(up).max(1)
    n = len(x)
    pk = pk[:n * 4].reshape(n, 4).max(1)
    pk = np.maximum(pk, np.abs(x).max(1))
    g = np.minimum(1.0, undb(ceil_db) / (pk + 1e-12))
    L = ns(look_ms / 1000)
    g = minimum_filter1d(g, 2 * L + 1)
    blk = 16
    nb = (n + blk - 1) // blk
    gp = np.pad(g, (0, nb * blk - n), constant_values=1.0).reshape(nb, blk).min(1)
    a = np.exp(-blk / (rel_ms / 1000 * SR))
    out = np.empty(nb)
    cur = 1.0
    for k in range(nb):
        v = gp[k]
        cur = v if v < cur else a * cur + (1 - a) * v
        out[k] = cur
    gs = np.repeat(out, blk)[:n]
    gs = np.minimum(gs, g)
    gs = uniform_filter1d(gs, L + 1)
    return x * gs[:, None], gs


def write_wav(path, x, pcm16=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if pcm16:
        r = rng(77)
        d = (r.uniform(-0.5, 0.5, x.shape) + r.uniform(-0.5, 0.5, x.shape))   # TPDF dither
        y = np.clip(np.round(x * 32767 + d), -32768, 32767).astype(np.int16)
        wavfile.write(path, SR, y)
    else:
        wavfile.write(path, SR, x.astype(np.float32))


def build(report=True):
    T0 = time.time()
    os.makedirs(STEMS, exist_ok=True)
    print('[sound] voice ...', flush=True)
    raw = load_voice()
    voice, vad, vinfo = process_voice(raw)
    print(f'[sound]   voice processed ({time.time() - T0:.1f}s)', flush=True)

    print('[sound] sfx ...', flush=True)
    sb = build_sfx()
    irs = {k: make_ir(*SPACE_RT[k][:7], seed=50 + j, flutter=SPACE_RT[k][7]) for j, k in enumerate(SPACE_RT)}
    ret = dict(bath=0.55, studio=0.6, tunnel=0.7, space=0.75)
    sfx = sb.dry.copy()
    for k, s in sb.send.items():
        if np.abs(s).max() > 0:
            sfx += convolve(s, irs[k]) * ret[k]
    print(f'[sound]   sfx done ({time.time() - T0:.1f}s)', flush=True)

    print('[sound] music ...', flush=True)
    M = build_music()
    music = M.bus + convolve(M.send, irs['hall']) * 0.55
    music = butter(music, 'high', 35, 2)
    print(f'[sound]   music done ({time.time() - T0:.1f}s)', flush=True)

    # gravitational warp of the background (music + space bed) as the black hole forms
    music = black_hole_warp(music)
    # pre-impact silence: hard-duck music around the boom, let it return softly
    tb = EV['finale_boom']
    gm = env_pts(tvec(N), [(0, 1), (tb - 0.14, 1), (tb - 0.03, 0), (tb + 1.2, 0), (34.2, 1), (80, 1)])
    music *= gm[:, None]

    # protection window: keep the boom's punch even though it lands on "finalização"
    tf = np.arange(int(N / 480) + 1) * 0.01
    protect = 1 - 0.75 * env_pts(tf, [(0, 0), (tb - 0.05, 0), (tb, 1), (tb + 0.35, 1), (tb + 0.7, 0), (999, 0)])
    vad_n = vad[:len(tf)] if len(vad) >= len(tf) else np.pad(vad, (0, len(tf) - len(vad)))
    musd, mred = duck(music, voice, vad_n, (5.0, 11.0, 8.0), True, 18.0)
    sfxd, sred = duck(sfx, voice, vad_n, (3.0, 8.0, 6.0), True, 15.0, protect)
    bg = musd + sfxd

    mix = voice + bg
    # master: normalise to -14 LUFS, true-peak limit, iterate
    g = undb(-14.0 - integrated(mix))
    for it in range(4):
        lim, gs = tp_limit(mix * g, -1.25)
        li = integrated(lim)
        if abs(li + 14.0) < 0.1:
            break
        g *= undb(-14.0 - li)
    tp = true_peak(lim)
    # fade the last 150 ms to digital silence (clean end)
    lim *= np.minimum(1, (N - 1 - np.arange(N)) / ns(0.15))[:, None]
    G = (g * gs)[:, None]
    print(f'[sound] master: I={li:.2f} LUFS  TP={tp:.2f} dBTP  max GR={-db(gs.min()):.2f} dB  ({time.time() - T0:.1f}s)')

    write_wav(os.path.join(AUDIO, 'mix.wav'), lim, pcm16=True)
    write_wav(os.path.join(AUDIO, 'sfx_only.wav'), bg * G, pcm16=True)
    write_wav(os.path.join(STEMS, 'voice.wav'), voice * G)
    write_wav(os.path.join(STEMS, 'sfx.wav'), sfxd * G)
    write_wav(os.path.join(STEMS, 'music.wav'), musd * G)
    info = dict(integrated_lufs=float(li), true_peak_dbtp=float(tp), limiter_max_gr_db=float(-db(gs.min())),
                master_gain_db=float(db(g)), samples=int(N), duration=N / SR, voice=vinfo,
                events=[dict(name=nm, t0=round(a, 3), t1=round(b, 3)) for nm, a, b in sb.log])
    if report:
        info.update(make_report(voice * G, bg * G, lim, vad_n, sfxd * G, musd * G, mred, sred))
    with open(os.path.join(AUDIO, 'mix_info.json'), 'w') as fh:
        json.dump(info, fh, indent=1)
    print(f'[sound] done in {time.time() - T0:.1f}s')
    return info


# =============================================================================================
# report: plots + measurements
# =============================================================================================
EV_MARKS = [
    (EV['clean_clutter'][0], 'clutter'), (EV['clean_flush'][0], 'flush'), (EV['clean_shower'][0], 'shower'),
    (EV['studio'][0], 'studio'), (EV['suit'][0], 'suit'), (EV['suit'][1], 'lock'), (EV['anticipation'][0], 'antic'),
    (EV['fire_on'], 'FIRE'), (EV['ice_on'], 'ICE'), (EV['fusion'][0], 'fusion'), (EV['release'], 'release'),
    (EV['finale_boom'], 'BOOM'), (EV['ceil_portal'][0], 'portal'), (EV['suck'][0], 'suck'),
    (EV['ceil_portal_close'][0], 'close'), (EV['tunnel'][0], 'tunnel'), (EV['arrive'][0], 'arrive/space'),
    (EV['blackhole'][0], 'blackhole'), (EV['strange'][0], 'strange'), (EV['strange_wipe'][0], 'wipe'),
    (EV['strange_wipe'][1], 'raw'),
]


def make_report(voice, bg, mix, vad, sfx, music, mred, sred):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    hop = 0.01
    tcv, lv = loud_curve(voice, 0.4, hop)
    tcb, lb = loud_curve(bg, 0.4, hop)
    tcm, lm = loud_curve(mix, 0.4, hop)
    _, lst = loud_curve(mix, 3.0, hop)
    tst = np.arange(len(lst)) * hop + 1.5
    # speech frames: >= 70 % VAD inside the 400 ms window
    vf = np.convolve(vad.astype(float), np.ones(40) / 40, 'valid')
    k = min(len(vf), len(lv))
    sp = vf[:k] >= 0.7
    ratio = (lv[:k] - lb[:k])
    rs = ratio[sp]
    stats = dict(vbr_median_db=float(np.median(rs)), vbr_p10_db=float(np.percentile(rs, 10)),
                 vbr_p05_db=float(np.percentile(rs, 5)), vbr_frac_ge12=float((rs >= 12).mean()),
                 bg_peak_momentary_lufs=float(lb.max()))
    # worst moments
    worst = []
    order = np.argsort(rs)
    tsp = tcv[:k][sp]
    for j in order:
        if all(abs(tsp[j] - w) > 0.5 for w in worst):
            worst.append(float(tsp[j]))
        if len(worst) >= 6:
            break
    stats['vbr_worst_times'] = [round(w, 2) for w in worst]
    stats['vbr_worst_values'] = [round(float(ratio[:k][np.argmin(abs(tcv[:k] - w))]), 1) for w in worst]

    def spectro(x, t0, t1, nfft=4096, hop_s=512):
        i0, i1 = ns(t0), ns(t1)
        m = x[i0:i1].mean(1)
        f, tt, Z = ss.stft(m, SR, nperseg=nfft, noverlap=nfft - hop_s)
        P = 20 * np.log10(np.abs(Z) + 1e-9)
        # log-frequency resample 30 Hz .. 20 kHz
        fl = np.geomspace(30, 20000, 360)
        Pl = np.stack([np.interp(fl, f, P[:, j]) for j in range(P.shape[1])], 1)
        return tt + t0, fl, Pl

    paths = []
    sections = [(0, C.DURATION, 'full'), (0, 20, 'a_0-20'), (19, 36, 'b_19-36'), (40, 56, 'c_40-56'),
                (55, C.DURATION, 'd_55-end')]
    for t0, t1, tag in sections:
        fig, ax = plt.subplots(5, 1, figsize=(26 if tag == 'full' else 22, 15), sharex=True,
                               gridspec_kw=dict(height_ratios=[1.2, 2.2, 2.2, 1.6, 1.0]))
        fig.patch.set_facecolor('#111')
        for a in ax:
            a.set_facecolor('#151515')
            a.tick_params(colors='#ccc', labelsize=9)
            for s in a.spines.values():
                s.set_color('#444')
        i0, i1 = ns(t0), ns(t1)
        step = max(1, (i1 - i0) // 6000)
        tt = np.arange(i0, i1, step) / SR
        mm = mix[i0:i1:step].mean(1)
        vv = voice[i0:i1:step].mean(1)
        bb = bg[i0:i1:step].mean(1)
        ax[0].plot(tt, mm, color='#bbb', lw=0.4)
        ax[0].plot(tt, bb, color='#ff9a3c', lw=0.4, alpha=0.8)
        ax[0].plot(tt, vv, color='#4fc3f7', lw=0.4, alpha=0.8)
        ax[0].set_ylim(-1, 1)
        ax[0].set_ylabel('wave\nmix/bg/voice', color='#ccc')
        st_, fl, P = spectro(bg, t0, t1)
        vmax = P.max()
        ax[1].pcolormesh(st_, fl, P, shading='auto', cmap='magma', vmin=vmax - 90, vmax=vmax)
        ax[1].set_yscale('log')
        ax[1].set_ylabel('BG (sfx+music)\nspectrogram Hz', color='#ccc')
        st_, fl, P = spectro(mix, t0, t1)
        vmax = P.max()
        ax[2].pcolormesh(st_, fl, P, shading='auto', cmap='magma', vmin=vmax - 90, vmax=vmax)
        ax[2].set_yscale('log')
        ax[2].set_ylabel('MIX\nspectrogram Hz', color='#ccc')
        sel = (tcv >= t0) & (tcv <= t1)
        ax[3].plot(tcm[sel], lm[sel], color='#eee', lw=0.8, label='mix M')
        ax[3].plot(tcv[sel], lv[sel], color='#4fc3f7', lw=0.8, label='voice M')
        ax[3].plot(tcb[sel], lb[sel], color='#ff9a3c', lw=0.8, label='bg M')
        s2 = (tst >= t0) & (tst <= t1)
        ax[3].plot(tst[s2], lst[s2], color='#9ccc65', lw=1.4, label='mix S (3s)')
        ax[3].axhline(-14, color='#666', ls='--', lw=0.8)
        ax[3].set_ylim(-60, 0)
        ax[3].set_ylabel('LUFS', color='#ccc')
        ax[3].legend(loc='lower left', fontsize=8, ncol=4, facecolor='#222', labelcolor='#ddd')
        rr = np.where(sp, ratio[:k], np.nan)
        s3 = (tcv[:k] >= t0) & (tcv[:k] <= t1)
        ax[4].plot(tcv[:k][s3], rr[s3], color='#ce93d8', lw=0.9)
        ax[4].axhline(12, color='#e57373', ls='--', lw=0.8)
        ax[4].set_ylim(0, 40)
        ax[4].set_ylabel('voice - bg\n(speech) dB', color='#ccc')
        # speech shading + event markers
        vt = np.arange(len(vad)) * 0.01
        ax[3].fill_between(vt, -60, 0, where=vad, color='#4fc3f7', alpha=0.07, step='mid')
        ax[4].fill_between(vt, 0, 40, where=vad, color='#4fc3f7', alpha=0.07, step='mid')
        for te, lab in EV_MARKS:
            if t0 <= te <= t1:
                for a in ax:
                    a.axvline(te, color='#00e5ff', lw=0.7, alpha=0.55)
                ax[0].text(te, 0.92, lab, color='#00e5ff', fontsize=9, rotation=0, va='top', ha='left')
        ax[4].set_xlim(t0, t1)
        ax[4].set_xlabel('time (s)', color='#ccc')
        fig.suptitle(f'sound report [{tag}]  I={integrated(mix):.1f} LUFS  TP={true_peak(mix):.1f} dBTP  '
                     f'voice/bg in speech: median {stats["vbr_median_db"]:.1f} dB, p10 {stats["vbr_p10_db"]:.1f} dB, '
                     f'>=12dB {100 * stats["vbr_frac_ge12"]:.0f}%', color='#eee', fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        p = os.path.join(AUDIO, f'report_{tag}.png')
        fig.savefig(p, dpi=70, facecolor=fig.get_facecolor())
        plt.close(fig)
        paths.append(p)
    stats['report_pngs'] = paths
    print('[sound] voice/bg during speech:', {k_: v for k_, v in stats.items() if k_.startswith('vbr')})
    return stats


def ffmpeg_check(path):
    r = subprocess.run(['ffmpeg', '-hide_banner', '-nostats', '-i', path, '-af', 'ebur128=peak=true', '-f', 'null', '-'],
                       capture_output=True, text=True)
    lines = [l for l in r.stderr.splitlines() if l.strip().startswith(('I:', 'Peak:', 'LRA:'))]
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-report', action='store_true')
    a = ap.parse_args()
    info = build(report=not a.no_report)
    print('[sound] ffmpeg ebur128 check:', ffmpeg_check(os.path.join(AUDIO, 'mix.wav')))


if __name__ == '__main__':
    main()
