"""Sound design + music bed + mix for the VFX edit.

    python3 -m vfx.sound            # build DATA/audio/mix.wav (+ sfx_only.wav, stems, report)
    python3 -m vfx.sound --no-report

Everything is synthesised here with numpy/scipy (no samples):
  * beats  : the visual beat times are READ FROM THE EFFECT MODULES (cleanup / cleanup_shatter /
             cleanup_studio / suit / orbs / ceiling_portal / space / strange) by load_beats(), with
             logged fallbacks, so a picture change only needs a re-run of this script.
  * voice  : SRC_VIDEO audio stream 0:a:0 -> HPF, spectral denoise, EQ, compressor, leveler,
             gentle gap expander, non-speech transient tamer. Mono-centred, kept dry and on top.
  * sfx    : one designed, layered sound per visual beat (whooshes = STFT-shaped noise sweeps,
             shimmers/chimes = FM + additive, crackle/shatter = granular, booms = pitch-enveloped sub
             with a parallel harmonic layer + a 150-600 Hz body + 1-3 kHz attack so every hit also
             reads on a phone speaker), spatialised (equal-power pan, tracking-driven for the hand
             orbs), sent to a synthetic convolution reverb of the current space (bathroom / black
             studio / tunnel / outer space). Two buses: short 'hit's and sustained 'bed's.
  * music  : subtle cinematic bed in D, one colour per section (bathroom curious -> studio dark ->
             powers tense/rising -> portal mystery -> space awe -> black hole dread -> Strange
             mystic -> warm D major ending), own hall reverb, tape-stop on the lamp flicker,
             gravitational pitch-sag into the black-hole ignition, dips before the finale and wipe.
  * mix    : 4-band side-chain ducking (<220 / 220-1k / 1-4k / >4k) keyed on the voice: VAD
             look-ahead envelope, a voice-presence-keyed dynamic EQ on 1-4 kHz, adaptive unmasking
             (speech band + broadband targets); beds use slow, capped ducking (no pumping); short
             transient-only protection windows on hits that land on their cue words. Lows <120 Hz
             summed to mono, BS.1770 normalisation to -14 LUFS, 4x-oversampled true-peak limiter
             at -1.5 dBTP. Output is exactly config.DURATION long, 48 kHz stereo.
  * report : spectrogram / waveform / loudness / voice-to-background plots with config.EV markers
             and the visual beats (DATA/audio/report_*.png) + ffmpeg ebur128 cross-check.
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


def noise2(n, seed, slope=-1.0, corr=0.5):
    """Stereo noise with inter-channel correlation `corr`."""
    a = colored(n, seed, slope)
    b = colored(n, seed + 7, slope)
    c = colored(n, seed + 13, slope)
    k1, k2 = np.sqrt(corr), np.sqrt(1 - corr)
    return np.stack([k1 * a + k2 * b, k1 * a + k2 * c], 1)


def lfo_noise(n, rate, seed):
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
    """Mid/side width scale (w can be an array). Capped at 1.4 so phones (mono) lose little."""
    w = np.minimum(np.asarray(w, np.float64), 1.4)
    m = (x[:, 0] + x[:, 1]) * 0.5
    s = (x[:, 0] - x[:, 1]) * 0.5 * w
    return np.stack([m + s, m - s], 1)


def sat(x, drive=2.0):
    return np.tanh(drive * x) / np.tanh(drive)


def put(out, x, t, gain=1.0, pan=None):
    """Mix x (mono or stereo) into the stereo buffer `out` at time t (s, may be negative)."""
    x = np.asarray(x, np.float64)
    if x.ndim == 1:
        x = pan2(x, 0.0 if pan is None else pan)
    elif pan is not None:
        x = pan2(x, pan)
    i0 = int(round(t * SR))
    a0 = max(0, -i0)
    i0 = max(0, i0)
    m = min(len(x) - a0, len(out) - i0)
    if m > 0:
        out[i0:i0 + m] += x[a0:a0 + m] * gain


def rms_max(x, win=0.05):
    """Max short-window RMS (mono sum)."""
    m = x.mean(1) if x.ndim == 2 else x
    w = max(1, ns(win))
    if len(m) < w:
        return float(np.sqrt((m ** 2).mean()))
    return float(np.sqrt(uniform_filter1d(m ** 2, w).max()))


def track_pan(side, scale=2.0):
    """Function t(abs s) -> pan following a tracked palm's x position."""
    try:
        from . import assets as A
        ts = np.array([C.t_of(i) for i in range(C.N_FRAMES)])
        xs = np.array([A.palm(i, side)[0] for i in range(C.N_FRAMES)], np.float64)
        xs = np.clip(xs, 0, C.W)
        xs = uniform_filter1d(xs, 9, mode='nearest')
        pans = np.clip((xs / C.W - 0.5) * scale, -0.85, 0.85)
        return lambda t: np.interp(t, ts, pans)
    except Exception as e:                          # tracking unavailable: fixed sides
        print(f'[sound] track_pan({side}) fallback: {e}')
        v = -0.4 if side == 'R' else 0.45
        return lambda t: np.full(np.shape(t), v) if np.ndim(t) else v


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


def mono_low(x, fc=120.0):
    """Sum everything below fc to mono (phones / mono fold-down; no decorrelated sub)."""
    lo = butter(x, 'low', fc, 4, zero_phase=True)
    return x - lo + lo.mean(1, keepdims=True)


# =============================================================================================
# visual beats: read from the effect modules (fallback = values measured when this was written)
# =============================================================================================
BEAT_SRC = {}


def _fxmod(name):
    try:
        import importlib
        return importlib.import_module(f'vfx.fx.{name}')
    except Exception as e:
        print(f'[sound] beats: vfx.fx.{name} not importable ({type(e).__name__}: {e}) -> fallbacks', flush=True)
        return None


def _attr(B, key, mod, name, default):
    v = getattr(mod, name, None) if mod is not None else None
    if v is None:
        B[key] = default
        BEAT_SRC[key] = 'fallback'
    else:
        B[key] = v
        BEAT_SRC[key] = f'{mod.__name__.rsplit(".", 1)[-1]}.{name}'
    return B[key]


def _lit(B, key, value, where):
    """A beat that is a literal inside a module function (documented where it lives)."""
    B[key] = value
    BEAT_SRC[key] = 'literal: ' + where


def _first_cross(t, y, thr):
    k = np.flatnonzero(np.asarray(y) >= thr)
    return float(t[k[0]]) if len(k) else float(t[-1])


def load_beats():
    B = {}
    # ---------------------------------------------------------------- cleanup: magic erasers
    fb = dict(clutter=(5.25, 0.26, 5.55, 6.70, 0.0), flush=(10.58, 0.16, 10.78, 11.06, 1.0),
              shower=(11.53, 0.22, 11.77, 12.53, 0.0))
    ers = {}
    cl = _fxmod('cleanup')
    try:
        for e in cl.FX().erasers:
            ers[e.name] = (float(e.burn_t0 - e.trace_lead), float(e.trace_dur), float(e.burn_t0), float(e.burn_t1),
                           float(getattr(e, 'pop_gain', 0.0)))
    except Exception as ex:
        print('[sound] beats: eraser parameters unavailable:', ex)
    for k, v in fb.items():
        B['er_' + k] = ers.get(k, v)                 # (trace start, trace dur, burn t0, burn t1, pop)
        BEAT_SRC['er_' + k] = 'cleanup.FX().erasers[...]' if k in ers else 'fallback'
    # ---------------------------------------------------------------- studio transformation
    cs = _fxmod('cleanup_shatter')
    _attr(B, 'st_ant', cs, 'T_ANT', 13.00)
    _attr(B, 'st_tw0', cs, 'TW0', 13.26)
    _attr(B, 'st_tw1', cs, 'TW1', 14.20)
    _lit(B, 'st_flicker', (13.06, 13.12, 13.17), 'cleanup_shatter.Shatter.room_light pulses')
    _lit(B, 'st_core', 13.27, 'cleanup_shatter.Shatter.core_amp pulse (front launch)')
    try:
        tt = np.linspace(B['st_tw0'], B['st_tw1'], 4000)
        rr = cs.front_R(tt)
        d = np.asarray(cs.LAMP, np.float64) - np.asarray(cs.ORIGIN, np.float64)
        B['st_lamp'] = float(np.interp(np.hypot(*d) / cs.front_mod(np.arctan2(d[1], d[0])), rr, tt))
        BEAT_SRC['st_lamp'] = 'cleanup_shatter: front_R reaches LAMP'
        front_R, front_mod, org, rmax = cs.front_R, cs.front_mod, np.asarray(cs.ORIGIN, np.float64), float(cs.R_MAX)
        B['lamp_pan'] = float(np.clip((cs.LAMP[0] / C.W - 0.5) * 1.6, -0.8, 0.8))
    except Exception as ex:
        print('[sound] beats: shatter front unavailable:', ex)
        tw0, tw1 = B['st_tw0'], B['st_tw1']

        def front_R(t):
            u = np.clip((np.asarray(t, np.float64) - tw0) / (tw1 - tw0), 0.0, 1.0)
            return np.where(np.asarray(t) < tw0, 0.0, 40.0 + 1520.0 * (0.22 * u + 0.78 * u * u))

        def front_mod(th):
            return 1.0 + 0.0 * th
        org, rmax = np.array([580.0, 1100.0]), 1560.0
        B['st_lamp'] = 13.91
        BEAT_SRC['st_lamp'] = 'fallback'
        B['lamp_pan'] = 0.7
    # front activity = visible length of the front x its speed (how much wall breaks per second)
    th = np.linspace(-np.pi, np.pi, 720, endpoint=False)
    fm = front_mod(th)
    tg = np.linspace(B['st_tw0'] - 0.02, B['st_tw1'] + 0.12, 700)
    R = np.asarray(front_R(tg), np.float64)
    vis = []
    for Rk in R:
        x = org[0] + Rk * fm * np.cos(th)
        y = org[1] + Rk * fm * np.sin(th)
        vis.append(((x >= 0) & (x < C.W) & (y >= 0) & (y < C.H)).mean() * TAU * Rk)
    act = np.array(vis) * np.maximum(np.gradient(R, tg), 0)
    B['st_front_t'], B['st_front_a'], B['st_front_r'] = tg, act / max(act.max(), 1e-9), np.clip(R / rmax, 0, 1)
    B['st_front_peak'] = float(tg[int(np.argmax(act))])
    cst = _fxmod('cleanup_studio')
    _attr(B, 'st_rim_l', cst, 'T_RIM_L', 14.32)
    _attr(B, 'st_rim_r', cst, 'T_RIM_R', 14.44)
    try:
        B['pan_bar_l'] = float(np.clip((cst.BAR_L['x'] / C.W - 0.5) * 2 * 0.85, -0.75, 0.75))
        B['pan_bar_r'] = float(np.clip((cst.BAR_R['x'] / C.W - 0.5) * 2 * 0.85, -0.75, 0.75))
    except Exception:
        B['pan_bar_l'], B['pan_bar_r'] = -0.6, 0.6
    # ---------------------------------------------------------------- suit
    su = _fxmod('suit')
    _attr(B, 'su_t0', su, 'T0', EV['suit'][0])
    _attr(B, 'su_t1', su, 'T1', EV['suit'][1])
    _attr(B, 'su_ign', su, 'T_IGN', EV['suit'][0] + 0.18)
    # ---------------------------------------------------------------- orbs
    ob = _fxmod('orbs')
    _attr(B, 'antic', ob, 'T_START', EV['anticipation'][0])
    _attr(B, 'fire', ob, 'T_FIRE', EV['fire_on'])
    _attr(B, 'ice', ob, 'T_ICE', EV['ice_on'])
    _lit(B, 'ice_sh0', 26.20, 'orbs.FX ice shards sh_t = 26.20 + 0.30 * k / n')
    _lit(B, 'ice_sh1', 26.50, 'orbs.FX ice shards (last burst)')
    B['ice_lock'] = B['ice'] + 0.02                  # flash peak (orbs.ice_state: exp after T_ICE + 0.02)
    BEAT_SRC['ice_lock'] = 'orbs.T_ICE + 0.02 (flash peak)'
    _attr(B, 'push0', ob, 'T_PUSH0', 28.00)
    _attr(B, 'push1', ob, 'T_PUSH1', 28.42)
    _attr(B, 'fl0', ob, 'T_FLY0', 28.38)
    _attr(B, 'fl1', ob, 'T_FLY1', 28.92)
    _attr(B, 'unr0', ob, 'T_UNR0', 28.74)
    _attr(B, 'unr1', ob, 'T_UNR1', 29.45)
    _attr(B, 'rel', ob, 'T_REL', EV['release'])
    _attr(B, 'boom', ob, 'T_BOOM', EV['finale_boom'])
    _attr(B, 'fin_end', ob, 'T_END', EV['finale_end'])
    _lit(B, 'pl_born', 31.22, 'orbs.plasma_state born')
    _lit(B, 'pl_charge1', 32.40, 'orbs.plasma_state charge end')
    _lit(B, 'pl_in0', 32.40, 'orbs.plasma_state inhale start')
    _lit(B, 'pl_in1', 32.54, 'orbs.plasma_state inhale end')
    _lit(B, 'pl_col', 32.565, 'orbs.plasma_state collapse')
    # ---------------------------------------------------------------- ceiling portal / tunnel / arrival
    cp = _fxmod('ceiling_portal')
    _lit(B, 'po_glim0', 44.9, 'ceiling_portal.ceil_state pinpoint glimmer')
    _lit(B, 'po_open', 45.98, 'ceiling_portal.ceil_state spring(t - 45.98)')
    _lit(B, 'po_flare', 46.02, 'ceiling_portal.ceil_state ignition flare')
    _attr(B, 'col0', cp, 'COL0', EV['ceil_portal_close'][0])
    _lit(B, 'col_flash', 50.84, 'ceiling_portal collapse flash sstep(50.76,50.85) + shake 50.84')
    _attr(B, 'tun0', cp, 'TUN0', EV['tunnel'][0])
    _attr(B, 'tun1', cp, 'TUN1', EV['tunnel'][1])
    _lit(B, 'jolt', 52.30, 'ceiling_portal tunnel turbulence exp(-((t-52.3)/0.12)^2)')
    _attr(B, 'arr0', cp, 'ARR0', 53.58)
    _attr(B, 'arr1', cp, 'ARR1', 54.25)
    _lit(B, 'land', 54.19, 'ceiling_portal landing bump 54.18 / squash 54.21')
    _lit(B, 'arr_pop', 54.43, 'ceiling_portal.arr_state close pop (core 54.3-54.43, light 54.44)')
    # ---------------------------------------------------------------- space / black hole
    sp = _fxmod('space')
    _attr(B, 'bh_ign', sp, 'T_IGNITE', 62.74)
    _attr(B, 'bh_peak', sp, 'T_PEAK', 62.70)
    _lit(B, 'bh_seed', 61.62, 'space.bh_state seed glint')
    try:
        B['pan_bh'] = float(np.clip((sp.BH_C[0] / C.W - 0.5) * 1.6, -0.8, 0.8))
    except Exception:
        B['pan_bh'] = 0.5
    # ---------------------------------------------------------------- Doctor Strange
    sg = _fxmod('strange')
    _attr(B, 'sg_pre', sg, 'T_PRE', 69.99)
    _attr(B, 'sg_ign', sg, 'T_IGN', 70.08)
    _attr(B, 'sg_pulses', sg, 'WORD_PULSES', (70.10, 71.06, 71.86, 72.10, 72.28, 72.66))
    _attr(B, 'sg_switch', sg, 'T_SWITCH', 72.26)
    _attr(B, 'sg_grow0', sg, 'T_GROW0', 72.45)
    _attr(B, 'sg_grow1', sg, 'T_GROW1', 73.30)
    _attr(B, 'sg_ap0', sg, 'T_AP0', 72.56)
    _attr(B, 'sg_ap1', sg, 'T_AP1', 72.98)
    _attr(B, 'sg_tw0', sg, 'TW0', EV['strange_wipe'][0])
    _attr(B, 'sg_tw1', sg, 'TW1', EV['strange_wipe'][1])
    _attr(B, 'sg_rel', sg, 'T_REL', 73.82)
    _attr(B, 'sg_front1', sg, 'T_FRONT1', 74.15)
    cf = getattr(sg, 'CF', None) if sg is not None else None
    cf = np.array([460.0, 930.0]) if cf is None else np.asarray(cf, np.float64)
    B['pan_cf'] = float(np.clip((cf[0] / C.W - 0.5) * 1.6, -0.8, 0.8))
    tgs = np.arange(69.9, C.DURATION + 1e-6, 0.002)
    try:
        Rs = np.array([sg._R_scalar(t) for t in tgs])
        Es = np.array([sg._E_scalar(t, r) for t, r in zip(tgs, Rs)])
        Is = np.array([sg._I_scalar(t) for t in tgs])
        BEAT_SRC['sg_curves'] = 'strange._R_scalar/_E_scalar/_I_scalar'
    except Exception as ex:
        print('[sound] beats: strange curves unavailable:', ex)
        Rs = np.interp(tgs, [70.08, 70.62, 72.45, 73.30, B['sg_rel'], B['sg_rel'] + 0.6], [0, 44, 80, 370, 326, 1200])
        Es = np.interp(tgs, [70.08, 70.3, 72.4, 73.3, 74.4, 74.62], [0, 700, 920, 4000, 9000, 0])
        Is = np.interp(tgs, [70.08, 70.24, 72.45, 73.2, 74.42, 74.62], [0, 0.95, 0.95, 1.3, 1.3, 0])
        BEAT_SRC['sg_curves'] = 'fallback'
    B['sg_t'], B['sg_R'], B['sg_E'], B['sg_I'] = tgs, Rs, Es, Is
    corners = np.array([[0, 0], [C.W, 0], [0, C.H], [C.W, C.H]], np.float64)
    r_out = float(np.max(np.hypot(*(corners - cf).T)))
    r_mid = 0.5 * (min(cf[0], C.W - cf[0], cf[1], C.H - cf[1]) + r_out)
    after = tgs >= B['sg_rel']
    B['sg_edge'] = _first_cross(tgs[after], Rs[after], r_mid)      # rim sweeping across the frame edges
    B['sg_out'] = _first_cross(tgs[after], Rs[after], r_out)       # rim has left the frame entirely
    return B


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


def tinkle(r, f, tau=None):
    """Glass / glazed-ceramic shard ping: inharmonic modes + micro click."""
    tau = tau or r.uniform(0.012, 0.06)
    n = ns(min(0.4, 6 * tau + 0.005))
    t = tvec(n)
    y = np.zeros(n)
    for k, (m, a) in enumerate(((1.0, 1.0), (2.32, 0.6), (4.25, 0.35), (6.63, 0.2))):
        fm = f * m * r.uniform(0.98, 1.02)
        if fm > 18000:
            break
        y += a * np.sin(TAU * fm * t + r.uniform(0, TAU)) * np.exp(-t / (tau / (1 + 0.6 * k)))
    k = ns(0.0006)
    y[:k] += r.standard_normal(k) * 0.6
    return y * (1 - np.exp(-t / 0.0003))


def shatter_burst(r, dur=0.6):
    """Broadband glass crash transient: HP noise burst (fast + slow decay) + dense glassy modes."""
    n = ns(dur)
    t = tvec(n)
    nz = butter(r.standard_normal(n), 'high', 900, 2)
    nz /= np.abs(nz).max() + 1e-9
    env = (1 - np.exp(-t / 0.0006)) * (0.7 * np.exp(-t / 0.016) + 0.3 * np.exp(-t / 0.11))
    y = nz * env
    for k in range(16):
        f = np.exp(r.uniform(np.log(2200), np.log(11500)))
        y += 0.12 * np.sin(TAU * f * t + r.uniform(0, TAU)) * np.exp(-t / r.uniform(0.02, 0.16)) * r.uniform(0.3, 1)
    lo = butter(r.standard_normal(n), 'band', [300, 1400], 2)
    y += 0.5 * lo / (np.abs(lo).max() + 1e-9) * (1 - np.exp(-t / 0.001)) * np.exp(-t / 0.025)
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
        p = 0.0 if pans is None else float(np.asarray(pans)[k])
        a = (np.clip(p, -1, 1) + 1) * np.pi / 4
        out[i0:i0 + len(g), 0] += g * np.cos(a) * np.sqrt(2)
        out[i0:i0 + len(g), 1] += g * np.sin(a) * np.sqrt(2)
    return out


def poisson_times(dur, rate_fn, seed, rmax=None):
    """Inhomogeneous Poisson process (thinning). rate_fn(t) in events/s."""
    r = rng(seed)
    tt = np.linspace(0, dur, 800)
    rmax = rmax or max(1e-3, float(np.max(rate_fn(tt))) * 1.05)
    m = r.poisson(rmax * dur * 1.2) + 10
    cand = np.sort(r.uniform(0, dur, m))
    keep = r.uniform(0, 1, m) < np.asarray(rate_fn(cand)) / rmax
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


def whoosh(dur, fc_pts, amp_pts, bw=0.55, seed=0, corr=0.5, slope=-1.0, bw_pts=None, tone=None):
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


def sub_drop(dur, f0, f1, tau_f, tau_a, drive=2.5, attack=0.004, h2=0.25, harm=0.45):
    """Pitch-enveloped sub hit. `harm`: parallel 2nd-5th harmonic layer (90-700 Hz) that carries the
    same pitch envelope so small speakers hear the drop."""
    n = ns(dur)
    t = tvec(n)
    f = f1 + (f0 - f1) * np.exp(-t / tau_f)
    ph = phase(f, n)
    y = np.sin(TAU * ph) + h2 * np.sin(2 * TAU * ph)
    y = sat(y * 0.9, drive)
    y = butter(y, 'low', 900, 2)
    if harm > 0:
        hl = (np.sin(2 * TAU * ph + 0.4) + 0.75 * np.sin(3 * TAU * ph + 1.1) + 0.5 * np.sin(4 * TAU * ph + 2.0)
              + 0.3 * np.sin(5 * TAU * ph + 0.3))
        hl = butter(sat(hl * 0.5, 1.6), 'band', [90.0, 700.0], 2)
        y = y + harm * hl * np.exp(-t / max(tau_a * 0.55, 0.02))
    tail = smooth01((n - np.arange(n)) / max(1.0, 0.25 * n))
    return y * (1 - np.exp(-t / attack)) * np.exp(-t / tau_a) * tail


def impact(r, f_hi=240.0, f_lo=110.0, tau=0.09, click_fc=2200.0, body=1.0, noise=0.6, click_amt=0.5, dur=0.6,
           nb=(150.0, 600.0)):
    """Phone-translating impact layers (mono, peak ~1): a pitched body thump gliding f_hi -> f_lo, a
    band-passed noise burst (150-600 Hz) and a 1-3 kHz attack click."""
    n = ns(dur)
    t = tvec(n)
    f = f_lo + (f_hi - f_lo) * np.exp(-t / (tau * 0.5))
    ph = phase(f, n)
    th = (np.sin(TAU * ph) + 0.35 * np.sin(2 * TAU * ph + 0.6)) * (1 - np.exp(-t / 0.0012)) * np.exp(-t / tau)
    z = butter(r.standard_normal(n), 'band', list(nb), 2)
    z = z / (np.abs(z).max() + 1e-9) * (1 - np.exp(-t / 0.0008)) * np.exp(-t / (tau * 0.5))
    m = min(n, ns(0.02))
    tc = tvec(m)
    ck = butter(r.standard_normal(m), 'band', [click_fc * 0.6, min(click_fc * 1.7, 20000)], 2)
    ck = ck / (np.abs(ck).max() + 1e-9) * np.exp(-tc / 0.0022)
    ck += 0.5 * np.sin(TAU * click_fc * tc) * np.exp(-tc / 0.004)
    y = body * th / 1.3 + noise * z
    y[:m] += click_amt * ck
    return y


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
            alpha=1.6, modes=2, decay=None, clamp_after=None, t_cut=None):
    """Granular crackle: Poisson resonant clicks with power-law amplitudes.
    decay: amplitudes *= exp(-t/decay); clamp_after: clamp the Pareto tail to 1.0 after that time;
    t_cut: no grains after that time."""
    times = poisson_times(dur, rate_fn, seed)
    r = rng(seed + 1)
    amps = np.minimum(r.pareto(alpha, len(times)) + 0.3, 6.0) / 3.0
    fcs = np.exp(r.uniform(np.log(f_lo), np.log(f_hi), len(times)))
    taus = np.exp(r.uniform(np.log(tau[0]), np.log(tau[1]), len(times)))
    pans = (pan_fn(times) if pan_fn else 0.0) + r.uniform(-pan_spread, pan_spread, len(times))
    if decay is not None:
        amps = amps * np.exp(-times / decay)
    if clamp_after is not None:
        amps = np.where(times > clamp_after, np.minimum(amps, 1.0), amps)
    if t_cut is not None:
        keep = times < t_cut
        times, amps, fcs, taus, pans = times[keep], amps[keep], fcs[keep], taus[keep], np.broadcast_to(pans, keep.shape)[keep]
    return grains(dur, times, lambda k, rr: click(rr, fcs[k], taus[k], modes) * amps[k], pans, seed + 2)


def sparkle(dur, times, freqs, seed=0, pan_fn=None, spread=0.4, tau=(0.08, 0.35), amp=1.0, amps=None):
    r = rng(seed)
    times = np.asarray(times, np.float64)
    taus = r.uniform(tau[0], tau[1], len(times))
    pans = (pan_fn(times) if pan_fn else 0.0) + r.uniform(-spread, spread, len(times))
    fr = [freqs[int(r.integers(len(freqs)))] * (2 if r.uniform() < 0.3 else 1) for _ in times]
    av = np.ones(len(times)) if amps is None else np.asarray(amps)
    return grains(dur, times, lambda k, rr: amp * av[k] * rr.uniform(0.4, 1.0) * bell(
        fr[k], dur=taus[k] * 5, ratio=rr.choice([2.0, 3.0, 1.41]), index=rr.uniform(0.8, 2.5),
        tau=taus[k], tau_i=taus[k] * 0.3, glass=0.25), pans, seed + 3)


def buzz(n, f=120.0, seed=0, lo=180.0, hi=3500.0):
    """Mains / ballast buzz (Brazil: 60 Hz mains -> 120 Hz), rich and slightly unstable."""
    t = tvec(n)
    y = saw(f * (1 + 0.002 * np.sin(TAU * 3.1 * t) + 0.0015 * lfo_noise(n, 2.0, seed)), n) + 0.5 * saw(f * 2.003, n)
    y = butter(y, 'band', [lo, hi], 2)
    return sat(y * 1.5, 2.0)


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
    return np.stack([ss.oaconvolve(x[:, c], ir[:, c])[:len(x)] for c in range(2)], 1)


def conv_into(y, x, ir, gain):
    """y += (x * ir) * gain, convolving only the non-silent span of x (full-length buffers)."""
    nz = np.flatnonzero(np.abs(x).max(1) > 0)
    if len(nz) == 0:
        return
    i0, i1 = int(nz[0]), int(nz[-1]) + 1
    seg = np.asarray(x[i0:i1], np.float64)
    for c in range(2):
        r = ss.oaconvolve(seg[:, c], ir[:, c])
        m = min(len(r), len(y) - i0)
        y[i0:i0 + m, c] += r[:m] * gain


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


def max_momentary(x):
    x = x if x.ndim == 2 else x[:, None]
    if len(x) < ns(0.45):
        x = np.pad(x, ((0, ns(0.45) - len(x)), (0, 0)))
    return float(loud_curve(x, 0.4, 0.02)[1].max())


def os_peak(x):
    """Per-sample 4x-oversampled absolute peak (max over channels), computed channel by channel."""
    x = x if x.ndim == 2 else x[:, None]
    n = len(x)
    pk = np.abs(x).max(1).astype(np.float32)
    for c in range(x.shape[1]):
        up = np.abs(ss.resample_poly(x[:, c].astype(np.float32), 4, 1))
        m = up[:n * 4].reshape(-1, 4).max(1)
        pk[:len(m)] = np.maximum(pk[:len(m)], m)
        del up
    return pk


def true_peak(x):
    return db(float(os_peak(x).max()))


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
    """10 ms frames voice-activity (bool, with hangover) + raw speech cores + level (dB)."""
    lv = frame_db(butter(m, 'band', [150, 4000], 2), 480, 960)
    floor = np.percentile(lv, 10)
    thr = max(floor + 20.0, np.percentile(lv, 95) - 20.0)
    act = lv > thr
    core = act.copy()
    # hangover 220 ms, pre-roll 60 ms, drop blips < 50 ms
    act = maximum_filter1d(act.astype(np.uint8), 5).astype(bool)
    act = minimum_filter1d(act.astype(np.uint8), 5).astype(bool)
    act = minimum_filter1d(act.astype(np.uint8), 5).astype(bool)
    act = maximum_filter1d(act.astype(np.uint8), 5).astype(bool)
    hang = maximum_filter1d(act.astype(np.uint8), 29, origin=-7).astype(bool)   # [-60 ms, +220 ms]
    return hang, core, lv, thr


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


def tame_transients(v, core, thr_db=15.0, red_db=9.0):
    """Non-speech impulsive noises (hand slap at 48.48 s ...): where the 5 ms peak exceeds the 300 ms
    RMS by > thr_db outside the speech cores, pull the voice down red_db (5 ms attack incl. look-ahead,
    150 ms release). Returns (v, list of (t, crest dB))."""
    m = np.abs(v).max(1)
    pk = maximum_filter1d(m, 240)
    rms = np.sqrt(uniform_filter1d((v ** 2).mean(1), 14400) + 1e-14)
    cr = 20 * np.log10(pk / rms + 1e-12)
    hop = 48                                                        # 1 ms frames
    nf = len(v) // hop + 1
    crf = maximum_filter1d(np.pad(cr, (0, nf * hop - len(cr)), constant_values=0).reshape(nf, hop).max(1), 3)
    pkf = np.pad(pk, (0, nf * hop - len(pk))).reshape(nf, hop).max(1)
    coref = np.repeat(maximum_filter1d(core.astype(np.uint8), 7).astype(bool), 10)
    coref = np.pad(coref, (0, max(0, nf - len(coref))))[:nf]
    det = (crf > thr_db) & ~coref & (pkf > np.abs(v).max() * undb(-30))
    red = np.where(det, red_db, 0.0)
    red = maximum_filter1d(red, 11, origin=-5)                       # 5 ms look-ahead
    red = follow(red, 0.0015, 0.15, 0.001)
    g = undb(-np.interp(np.arange(len(v)), (np.arange(nf) + 0.5) * hop, red))
    hits = []
    for j in np.flatnonzero(det):
        tj = j * 0.001
        if not hits or tj - hits[-1][0] > 0.2:
            hits.append((round(tj, 3), round(float(crf[j]), 1)))
    return v * g[:, None], hits


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
    vad, core, lv, thr = voice_vad(mid)
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
    # --- non-speech transients (source hand slap etc.)
    v, hits = tame_transients(v, core)
    # normalise voice to -20 LUFS internal reference
    li = integrated(v)
    v *= undb(-20.0 - li)
    return v, vad, dict(voice_lufs_raw=li, comp_thr=th, tamed_transients=hits)


# =============================================================================================
# the SFX buses
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
    """Two kinds: 'hit' (transients; fast ducking, protectable) and 'bed' (sustained; slow capped
    ducking, no pumping). Each kind has a dry buffer and one reverb send per space."""
    KINDS = ('hit', 'bed')

    def __init__(self, spaces):
        self.dry = {k: np.zeros((N, 2), np.float32) for k in self.KINDS}
        self.send = {k: {s: np.zeros((N, 2), np.float32) for s in spaces} for k in self.KINDS}
        self.log = []

    def add(self, x, t0, gain_db=0.0, pan=0.0, rev=0.0, space=None, name='', dry=1.0, lufs=None, kind='hit'):
        """lufs: normalise the sound's max momentary loudness to this (internal scale, voice = -20
        LUFS integrated) before applying gain_db."""
        x = np.asarray(x, np.float64)
        if x.ndim == 1 or np.ndim(pan) > 0 or pan != 0.0:
            x = pan2(x, pan)
        if lufs is not None:
            gain_db = gain_db + lufs - max_momentary(x)
        x = fade(x, 0.002, 0.004) * undb(gain_db)
        i0 = int(round(t0 * SR))
        a0 = max(0, -i0)
        i0c = max(0, i0)
        i1 = min(N, i0 + len(x))
        if i1 <= i0c:
            return
        seg = x[a0:a0 + (i1 - i0c)]
        self.dry[kind][i0c:i1] += seg * dry
        if rev > 0:
            sp = space or space_at(t0 + 0.05)
            self.send[kind][sp][i0c:i1] += seg * rev
        if name:
            self.log.append((name, kind, t0, t0 + len(x) / SR))

    def render(self, irs, ret):
        out = {}
        for k in self.KINDS:
            y = self.dry[k].astype(np.float64)
            for s, snd in self.send[k].items():
                conv_into(y, snd, irs[s], ret[s])
            out[k] = y
            self.dry[k] = None
            self.send[k] = None
        return out


# =============================================================================================
# SFX library: designed composite sounds (each returns (stereo array, absolute start time))
# =============================================================================================
def sfx_claude(B):
    """'Claude': a tiny >5 kHz sparkle on the caption pop (2.72, under the word) and a warm shimmer
    bloom in the breath after the word (4.1 .. 5.2), with sparkle partials above 5 kHz."""
    t0 = 2.70
    T = 2.55
    n = ns(T)
    t = tvec(n)
    out = np.zeros((n, 2))
    st = 0.02 + np.sort(rng(2100).uniform(0, 0.22, 5))
    out += sparkle(T, st, [hz(x) for x in ('E8', 'F#8', 'A8')], 2101, None, 0.5, (0.05, 0.16), 0.22)
    a = 4.08 - t0
    y = sum(sine(hz(x) * (1 + 0.002 * j), n) / (1 + 0.6 * j) for j, x in enumerate(('D5', 'F#5', 'A5', 'E6', 'A6')))
    y *= env_pts(t, [(0, 0), (a, 0), (a + 0.28, 1), (a + 0.65, 0.55), (T - 0.12, 0), (T, 0)])
    out += widen(pan2(y * 0.20, 0) + np.stack([0.06 * sine(hz('A5') * 1.004, n), -0.06 * sine(hz('A5') * 1.004, n)], 1)
                 * env_pts(t, [(0, 0), (a, 0), (a + 0.3, 1), (T - 0.12, 0), (T, 0)])[:, None], 1.3)
    sp_t = a + 0.04 + np.sort(rng(2102).uniform(0, 0.75, 10))
    out += sparkle(T, sp_t, [hz(x) for x in ('D8', 'E8', 'F#8', 'A8')], 2103, None, 0.6, (0.08, 0.3), 0.5)
    air = whoosh(T, [(0, 5000), (a, 5000), (a + 0.4, 8000), (T, 9000)],
                 [(0, 0), (a, 0), (a + 0.3, 0.35), (T - 0.15, 0), (T, 0)], bw=0.4, seed=2104, slope=0.0, corr=0.5)
    out += air * 0.4
    return out, t0


def sfx_eraser(er, base, size, seed):
    """Magic eraser synced to cleanup's Eraser: lock-on pings + laser trace (two comet heads racing
    round the outline) from the trace start for trace_dur, ignition 'fft' + incandescent burn
    crackle/sizzle (4-12 kHz) over the burn window, embers (sparkles), rising air, reveal shimmer.
    er = (trace start, trace dur, burn t0, burn t1, pop gain)."""
    tr0, trd, b0, b1, pop = er
    s0 = tr0 - 0.03
    T = (b1 - s0) + 1.9
    n = ns(T)
    t = tvec(n)
    out = np.zeros((n, 2))
    r = rng(seed)
    a = tr0 - s0
    sd = max(trd * 1.4, 0.2)
    bb0, bb1 = b0 - s0, b1 - s0
    bd = max(bb1 - bb0, 0.15)
    # 0. lock-on: two quick bright pings (selection)
    for k, (dt, f) in enumerate(((0.0, base * 4), (0.05, base * 6))):
        m = ns(0.25)
        tl = tvec(m)
        p = (np.sin(TAU * f * tl) + 0.3 * np.sin(TAU * f * 2.76 * tl) * np.exp(-tl / 0.02)) * np.exp(-tl / 0.045)
        p *= 1 - np.exp(-tl / 0.0006)
        put(out, p * 0.25, a + dt, pan=(-0.2, 0.2)[k])
    # 1. scan: two comet heads (opposite directions) -> two gliding laser tones, scan ripple
    u = np.clip((t - a) / sd, 0, 1)
    env = smooth01((t - a) / 0.015) * (1 - smooth01((t - a - sd) / 0.07))
    for sgn in (-1, 1):
        f = base * 2 * 2 ** smooth01(u) * (1 + 0.004 * sgn)
        las = sine(f, n) + 0.3 * sine(f * 2.003, n) + 0.12 * sine(f * 3.01, n)
        las *= 1 + 0.25 * np.sin(TAU * (28 + 6 * sgn) * t)
        out += pan2(las * env * 0.12, sgn * (0.1 + 0.5 * smooth01(u)))
    shk = whoosh(T, [(0, 6500), (a, 6500), (a + sd, 11000), (T, 9000)],
                 [(0, 0), (a, 0), (a + 0.02, 0.8), (a + sd, 0.5), (a + sd + 0.08, 0), (T, 0)], bw=0.3, seed=seed,
                 slope=0.0, corr=0.5)
    out += shk * 0.5
    # 2. ignition 'fft' at the burn start (hand-over from the trace)
    m = ns(0.4)
    tl = tvec(m)
    ig = butter(r.standard_normal(m), 'high', 3500, 2) * (1 - np.exp(-tl / 0.002)) * np.exp(-tl / 0.05)
    ig += 0.5 * butter(r.standard_normal(m), 'band', [700, 2400], 2) * (1 - np.exp(-tl / 0.002)) * np.exp(-tl / 0.03)
    put(out, ig * 0.9, bb0)

    # 3. burn: crackle at the incandescent front (4-10 kHz), density follows the burn progress
    def rate(tt):
        tt = np.asarray(tt, np.float64)
        uu = np.clip((tt - bb0) / bd, 0, 1)
        during = (tt > bb0) & (tt < bb1)
        after = tt >= bb1
        return (during * (15 + 210 * size * np.sin(np.pi * uu) ** 0.6)
                + after * 30 * size * np.exp(-np.maximum(tt - bb1, 0) / 0.3))
    cr = crackle(T, rate, 4000, 10000, (0.0008, 0.004), seed + 11, None, 0.5, 1.8)
    out += cr * 0.5
    # 4. sizzle of the burning edge (5-12 kHz), micro-modulated
    sz = noise2(n, seed + 5, 0.0, 0.5)
    sz = spec(sz, lambda f, tt: g_hp(f, 5000, 2) * g_lp(f, 12000, 2))
    micro = uniform_filter1d(np.clip(1 + 1.5 * colored(n, seed + 6, 0.0), 0, None) ** 3, 40)
    burn_env = smooth01((t - bb0) / 0.04) * (1 - smooth01((t - bb1) / 0.35))
    out += sz * (micro / (micro.mean() + 1e-9) * burn_env)[:, None] * 0.06
    # 5. embers: sparkles an octave up (identity above the voice)
    st = np.sort(bb0 + np.abs(r.normal(0.45 * bd, 0.4 * bd + 0.15, int(30 * size + 8))))
    pent = [base * 2 * 2 ** (s / 12) for s in (0, 2, 4, 7, 9, 12, 14, 16)]
    out += sparkle(T, st, pent, seed + 21, None, 0.55, (0.06, 0.3), 0.32)
    # 6. rising air (ash floats up)
    air = whoosh(T, [(0, 1500), (bb0, 1800), (bb1, 5000), (T, 8000)],
                 [(0, 0), (bb0, 0.0), (bb0 + 0.5 * bd, 0.4), (bb1 + 0.4, 0.3), (T, 0)], bw=0.5, seed=seed + 31, corr=0.5)
    out += air * 0.45
    # 7. reveal shimmer (clean wall)
    m = n - ns(bb1)
    tl = tvec(m)
    sh = sum(sine(base * mult * (1 + 0.002 * j), m) / (1 + 0.5 * j) for j, mult in enumerate((2, 3, 4, 6)))
    sh *= env_pts(tl, [(0, 0), (0.12, 1), (0.5, 0.45), (1.4, 0)])
    put(out, widen(pan2(sh * 0.08, 0), 1.2), bb1)
    # 8. light pop (flush: warm pop on his arm)
    if pop > 0:
        m = ns(0.5)
        tl = tvec(m)
        pp = butter(r.standard_normal(m), 'band', [1800, 6000], 2) * np.exp(-tl / 0.006) * 3.0
        pp += bell(base * 3, 0.5, 2.0, 1.2, 0.18, 0.04, glass=0.2)[:m] * 0.5
        put(out, pp * 0.35 * pop, bb0 + 0.05)
    return out, s0


def light_strike(r, dur=1.7):
    """Studio light bar striking on: contactor 'CHUNK' (bounces + 150-600 Hz body + housing ring) and
    a ballast hum swell that flickers on strike and settles low."""
    n = ns(dur)
    t = tvec(n)
    y = np.zeros(n)
    for dt, a in ((0.0, 1.0), (0.011, 0.45), (0.024, 0.2)):
        ck = click(r, r.uniform(1600, 2600), 0.004, 3) * a
        put_m = ns(dt)
        y[put_m:put_m + len(ck)] += ck[:n - put_m]
    imp = impact(r, 330, 150, 0.07, 1900, body=1.0, noise=0.8, click_amt=0.4, dur=0.5)
    y[:len(imp)] += imp * 0.9
    sd = sub_drop(0.6, 95, 55, 0.04, 0.12, 2.0, harm=0.5)
    y[:len(sd)] += sd * 0.35
    ring = sum(np.sin(TAU * f * t) * np.exp(-t / tc) for f, tc in ((612, 0.25), (1377, 0.14), (2290, 0.08)))
    y += ring * 0.06 * (1 - np.exp(-t / 0.001))
    hb = buzz(n, 120.0, int(r.integers(1 << 20)), 150, 2600)
    fl = 1 - 0.6 * (np.sin(TAU * 31 * t) > 0.3) * np.exp(-t / 0.09)
    hb *= env_pts(t, [(0, 0), (0.025, 1.0), (0.25, 0.5), (0.9, 0.18), (dur, 0)]) * fl
    y += hb * 0.16
    y += sine(9800 + 300 * np.exp(-t / 0.2), n) * env_pts(t, [(0, 0), (0.05, 1), (0.6, 0.3), (dur, 0)]) * 0.012
    return y


def sfx_studio(B):
    """Bathroom -> black studio, synced to cleanup_shatter/cleanup_studio:
    lamp flicker ticks + mains buzz (13.06/13.12/13.17) -> light drawn into the core (inhale) ->
    launch WHOMP at the core pulse (13.27) -> electric front 13.26-14.20 (zaps + front roar whose
    density follows the visible front length x speed) cracking tiles ahead of it and SHATTERING the
    walls (glass/ceramic crash bursts + dense tinkle shower panned outward, peak ~13.85-14.0) ->
    lamp pop (glass pop + fizz + arc) when the front reaches it -> shards tumble into the void ->
    two light bars strike on (L 14.32, R 14.44)."""
    s0 = B['st_ant'] - 0.03
    T = (B['st_rim_r'] - s0) + 1.9
    n = ns(T)
    t = tvec(n)
    out = np.zeros((n, 2))
    r = rng(400)

    def rel(x):
        return x - s0
    lp_pan = B['lamp_pan']
    tw0, tw1, core = rel(B['st_tw0']), rel(B['st_tw1']), rel(B['st_core'])
    fa = lambda tt: np.interp(np.asarray(tt) + s0, B['st_front_t'], B['st_front_a'], left=0.0, right=0.0)
    fr = lambda tt: np.interp(np.asarray(tt) + s0, B['st_front_t'], B['st_front_r'], left=0.0, right=1.0)
    # 1. room electrics: mains buzz that stutters with the lamp flickers, then is drawn into the core
    bz = buzz(n, 120.0, 410)
    gate = np.ones(n)
    for tf, dep, w in zip(B['st_flicker'], (0.9, 0.7, 1.0), (0.018, 0.012, 0.02)):
        gate -= dep * np.exp(-0.5 * ((t - rel(tf)) / w) ** 2)
    gate = np.clip(gate, 0, None)
    benv = env_pts(t, [(0, 0), (0.03, 0.3), (core - 0.12, 0.85), (core, 0.0), (T, 0)])
    out += pan2(bz * gate * benv * 0.09, lp_pan * 0.5)
    for k, tf in enumerate(B['st_flicker']):
        ck = click(r, 2600, 0.003, 3) * 0.8
        zp = zap(r, 0.05, 1.0) * 0.6
        put(out, ck * 0.6, rel(tf) - 0.003, pan=lp_pan)
        put(out, zp * 0.5, rel(tf) - 0.002, pan=lp_pan)
    # 2. the room light is drawn into the energy core behind him: rising inhale into the launch
    ci0 = rel(13.08)
    L = core - ci0
    iw = whoosh(L, [(0, 300), (L, 2400)], [(0, 0), (L - 0.015, 1.0), (L, 0)], bw=0.45, seed=401, corr=0.6)
    iw *= (np.linspace(0, 1, len(iw)) ** 2.2)[:, None]
    put(out, iw * 0.55, ci0)
    m = ns(L)
    tl = tvec(m)
    tn = sat(sine(55 * 2 ** (0.7 * tl / L), m) + 0.4 * sine(110 * 2 ** (0.7 * tl / L), m), 1.6) * (tl / L) ** 2
    put(out, tn * 0.35, ci0)
    # 3. launch WHOMP at the core pulse (the front starts here)
    imp = impact(r, 210, 85, 0.10, 2400, body=1.0, noise=0.7, click_amt=0.6, dur=0.6)
    sd = sub_drop(1.2, 80, 34, 0.2, 0.32, 2.4, harm=0.5)
    cz = zap(r, 0.16, 1.2)
    put(out, imp * 0.5, core)
    put(out, sd * 0.45, core)
    put(out, cz * 0.6, core)
    # 4. electric front: zap grains + front roar following the activity, panned outward with the radius
    zt = poisson_times(T, lambda tt: 3 + 70 * fa(tt), 402)
    zr = rng(403)
    zp = np.clip(zr.choice([-1, 1], len(zt)) * (0.15 + 0.7 * fr(zt)) + zr.normal(0, 0.12, len(zt)), -0.95, 0.95)
    out += grains(T, zt, lambda k, q: zap(q, q.uniform(0.02, 0.07), 1.3) * q.uniform(0.3, 1.0), zp, 404) * 0.38
    fro = noise2(n, 405, -1.0, 0.55)
    frr = fr(t)
    fro = spec(fro, lambda f, tt: g_bp(f, 500 * 2 ** (2.3 * np.interp(tt, t, frr)), 0.7))
    frenv = np.clip(fa(t), 0, 1) ** 0.7 * smooth01((t - tw0) / 0.05)
    out += widen(fro * frenv[:, None], 0.6 + 0.8 * frr) * 0.5
    # 5. tiles crack just ahead of the front (ceramic 'tk' clicks)
    ct = poisson_times(T, lambda tt: np.where(np.asarray(tt) > core - 0.03, 20 + 240 * fa(np.asarray(tt) + 0.08), 0), 406)
    cq = rng(407)
    cpan = np.clip(cq.normal(0, 1, len(ct)) * (0.2 + 0.7 * fr(ct + 0.08)), -0.95, 0.95)
    cfs = np.exp(cq.uniform(np.log(1200), np.log(5000), len(ct)))
    out += grains(T, ct, lambda k, q: click(q, cfs[k], q.uniform(0.0015, 0.006), 3) * q.uniform(0.25, 1.0), cpan, 408) * 0.35
    # 6. walls SHATTER: crash bursts at the activity quantiles, the biggest one (with a body and a
    #    sub) as the front reaches the lamp corner (climax, 13.85-13.95), + dense tinkle shower
    cum = np.cumsum(B['st_front_a'])
    cum /= cum[-1]
    t_main = B['st_lamp'] - 0.015
    crash_t = [float(np.interp(q, cum, B['st_front_t'])) for q in (0.12, 0.32, 0.5, 0.93)]
    for k, tc in enumerate(crash_t):
        side = (-1) ** k * (0.25 + 0.45 * float(fr(rel(tc))))
        put(out, shatter_burst(rng(420 + k), 0.6) * (0.45 + 0.08 * k), rel(tc), pan=side)
    put(out, shatter_burst(rng(425), 0.8) * 0.9, rel(t_main), pan=-0.35)
    put(out, shatter_burst(rng(426), 0.8) * 0.8, rel(t_main) + 0.012, pan=0.45)
    put(out, impact(r, 190, 85, 0.16, 2000, body=0.9, noise=1.0, click_amt=0.5, dur=0.9) * 0.75, rel(t_main))
    put(out, sub_drop(1.6, 70, 30, 0.25, 0.5, 2.4, harm=0.55) * 0.6, rel(t_main))
    t1 = tw1

    def sh_rate(tt):
        tt = np.asarray(tt, np.float64)
        return 650 * fa(tt - 0.03) ** 1.2 + 140 * np.exp(-np.maximum(tt - t1, 0) / 0.4) * (tt > t1) + 0.5
    st = poisson_times(T, sh_rate, 430)
    sq = rng(431)
    spn = np.clip(sq.normal(0, 1, len(st)) * (0.25 + 0.75 * fr(st)), -1, 1)
    sfq = np.exp(sq.uniform(np.log(2600), np.log(11000), len(st)))
    sam = sq.uniform(0.15, 1.0, len(st)) * np.exp(-np.maximum(st - t1, 0) / 0.5)
    out += grains(T, st, lambda k, q: tinkle(q, sfq[k]) * sam[k], spn, 432) * 0.3
    # 7. lamp pop: glass pop + filament fizz + arc + a few falling shards
    lpt = rel(B['st_lamp'])
    m = ns(0.7)
    tl = tvec(m)
    pop = butter(r.standard_normal(m), 'high', 1000, 2) * (1 - np.exp(-tl / 0.0004)) * np.exp(-tl / 0.004) * 2.0
    pop += impact(r, 420, 220, 0.03, 3000, body=0.5, noise=0.6, click_amt=0.8, dur=0.7) * 0.6
    fz = butter(r.standard_normal(m), 'band', [3000, 9000], 2)
    fz *= np.clip(1 + 2.5 * colored(m, 433, 0.0), 0, None) ** 2 * np.exp(-tl / 0.16) * 0.12
    put(out, (pop + fz) * 0.55, lpt, pan=lp_pan)
    put(out, zap(r, 0.12, 1.1) * 0.4, lpt + 0.005, pan=lp_pan)
    pt = lpt + 0.03 + np.sort(r.exponential(0.12, 10))
    out += grains(T, pt, lambda k, q: tinkle(q, q.uniform(3000, 9000)) * q.uniform(0.3, 0.8),
                  np.clip(lp_pan + r.normal(0, 0.15, len(pt)), -1, 1), 434) * 0.3
    # 8. shards tumble back into the void: receding whoosh (LP falling), low void swell
    vw = whoosh(T, [(0, 2500), (rel(B['st_front_peak']), 2500), (tw1 + 0.7, 350), (T, 250)],
                [(0, 0), (rel(B['st_front_peak']) - 0.1, 0), (tw1 - 0.05, 0.8), (tw1 + 0.8, 0.0), (T, 0)],
                bw=0.75, seed=440, corr=0.55)
    out += widen(vw, 1.3) * 0.45
    # 9. the two light bars strike on
    for tt_, pn, sd_ in ((B['st_rim_l'], B['pan_bar_l'], 450), (B['st_rim_r'], B['pan_bar_r'], 451)):
        put(out, light_strike(rng(sd_)) * 0.85, rel(tt_), pan=pn)
    return out, s0


def sfx_studio_tone(B):
    """Very faint studio air (room tone) from after the light strikes to the tunnel."""
    t0, t1 = B['st_rim_r'] + 0.6, EV['tunnel'][0] + 0.05
    rt = butter(noise2(ns(t1 - t0), 300, -2.0, 0.6), 'band', [40, 600], 2)
    rt *= env_pts(tvec(len(rt)), [(0, 0), (1.5, 1), (t1 - t0 - 0.3, 1), (t1 - t0, 0)])[:, None]
    return rt, t0


def sfx_suit_hits(B):
    """Ignition at suit.T_IGN (core flash + shockwave): cyan power-on zap, 1-3 kHz click, soft sub tick
    and body, shockwave 'fwip'; lock 'shk-chunk' at the end of the window + cool power-up tone."""
    ti, t1 = B['su_ign'], B['su_t1']
    s0 = ti - 0.2
    T = (t1 - s0) + 1.8
    n = ns(T)
    out = np.zeros((n, 2))
    r = rng(520)
    # charge-up whine into the ignition (0.18 s)
    m = ns(0.2)
    tl = tvec(m)
    wh = sine(1800 * 2 ** (1.2 * tl / 0.2), m) * (tl / 0.2) ** 2 * 0.25
    put(out, wh, 0.0)
    # ignition
    a = ti - s0
    m = ns(1.0)
    tl = tvec(m)
    f = 3800 * np.exp(-tl / 0.05) + 900
    ch = np.sin(TAU * np.cumsum(f) / SR + 2.5 * np.sin(TAU * np.cumsum(f * 1.5) / SR)) * (1 - np.exp(-tl / 0.001)) * np.exp(-tl / 0.07)
    nz = butter(r.standard_normal(m), 'high', 2500, 2) * (1 - np.exp(-tl / 0.0008)) * np.exp(-tl / 0.025)
    imp = impact(r, 260, 120, 0.06, 2200, body=0.7, noise=0.4, click_amt=0.9, dur=0.5)
    sd = sub_drop(0.7, 110, 55, 0.05, 0.14, 2.0, harm=0.5)
    put(out, ch * 0.35 + nz * 0.5, a)
    put(out, imp * 0.8, a)
    put(out, sd * 0.5, a)
    sw = whoosh(0.35, [(0, 1500), (0.35, 6000)], [(0, 0), (0.02, 1), (0.35, 0)], bw=0.5, seed=521, corr=0.4)
    put(out, widen(sw, np.interp(tvec(len(sw)), [0, 0.3], [0.6, 1.4])) * 0.45, a)
    # lock at t1: 'shk-chunk' + housing ring
    a = t1 - s0
    m = ns(1.5)
    tl = tvec(m)
    shk = butter(rng(506).standard_normal(m), 'high', 3000, 2) * np.exp(-tl / 0.018) * 0.8
    thud = sub_drop(1.5, 110, 55, 0.04, 0.18, 2.0, harm=0.5) * 0.8
    ring = sum(np.sin(TAU * fr * tl) * np.exp(-tl / tc) for fr, tc in ((1830, 0.35), (2910, 0.22), (4370, 0.14)))
    lock = shk + thud + ring * 0.12 * (1 - np.exp(-tl / 0.001))
    lock[:ns(0.5)] += impact(r, 300, 140, 0.05, 2000, body=0.8, noise=0.5, click_amt=0.6, dur=0.5) * 0.6
    put(out, lock, a)
    m = ns(1.4)
    tl = tvec(m)
    pw = (sine(hz('E5'), m) + 0.5 * sine(hz('B5'), m) + 0.25 * sine(hz('E6') * 1.002, m))
    pw *= env_pts(tl, [(0, 0), (0.12, 1), (0.5, 0.5), (1.4, 0)])
    put(out, pw * 0.12, a + 0.06)
    return out, s0


def sfx_suit_bed(B):
    """Nanotech assembly from the ignition to the end of the window: granular metallic ticks spreading
    outward from the sternum, hex snaps, servo whine, riser into the lock."""
    ti, t1 = B['su_ign'], B['su_t1']
    d = t1 - ti
    T = d + 0.1
    n = ns(T)
    t = tvec(n)
    out = np.zeros((n, 2))
    rate = lambda tt: np.where(tt < d, 40 + 520 * np.sin(np.pi * np.clip(tt / d, 0, 1)) ** 1.5 * smooth01(tt / (0.25 * d)), 0)
    times = poisson_times(d, rate, 500)
    r = rng(501)
    spread = np.interp(times, [0, d], [0.1, 0.9])
    pans = np.clip(r.normal(0, 0.6, len(times)) * spread, -1, 1)
    fcs = np.exp(r.uniform(np.log(2500), np.log(9500), len(times)))
    amps = r.uniform(0.25, 1.0, len(times)) * np.interp(times, [0, d * 0.3, d], [0.6, 1.0, 0.8])
    out += grains(T, times, lambda k, q: click(q, fcs[k], q.uniform(0.0008, 0.004), 3) * amps[k], pans, 502) * 0.42
    times2 = poisson_times(d, lambda tt: 9 + 14 * (tt / d), 503)
    pans2 = r.uniform(-0.8, 0.8, len(times2))
    out += grains(T, times2, lambda k, q: click(q, q.uniform(1100, 2600), 0.006, 3) * q.uniform(0.5, 1), pans2, 504) * 0.32
    for j, (fa_, fb_, pn) in enumerate([(520, 2300, -0.25), (780, 3100, 0.25)]):
        f = fa_ * (fb_ / fa_) ** smooth01(t / d) * (1 + 0.006 * np.sin(TAU * (5.5 + j) * t))
        sv = sine(f, n) + 0.25 * saw(f * 0.5, n) * 0.4
        sv = butter(sv, 'band', [300, 6000], 2)
        sv *= env_pts(t, [(0, 0), (0.25, 0.3), (d * 0.9, 0.6), (d, 0), (T, 0)])
        out += pan2(sv * 0.10, pn)
    ri = whoosh(T, [(0, 900), (d, 7500), (T, 7500)], [(0, 0), (d * 0.97, 0.7), (d + 0.02, 0), (T, 0)], bw=0.45, seed=505)
    out += ri * 0.45
    return out, ti


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
    f = 48 * 2 ** (smooth01(t / d) * 0.6)
    lr = sat(sine(f, n) + 0.3 * sine(2 * f, n), 1.5) * smooth01(t / d) ** 2.5 * (t < d)
    out += pan2(lr * 0.30, 0)
    rs = whoosh(T, [(0, 500), (d, 3000), (T, 3000)], [(0, 0), (d * 0.6, 0.1), (d - 0.02, 0.5), (d, 0), (T, 0)], bw=0.6, seed=607)
    out += rs * 0.4
    return out


def sfx_fire_ignite():
    T = 1.6
    n = ns(T)
    t = tvec(n)
    out = np.zeros((n, 2))
    r = rng(704)
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
    th = sub_drop(1.2, 95, 50, 0.08, 0.3, 2.2, harm=0.5)
    put(out, th * 0.8, b0)
    put(out, impact(r, 260, 130, 0.08, 1800, body=0.8, noise=0.9, click_amt=0.5, dur=0.6) * 0.6, b0)
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
    mid = noise2(n, seed + 3, -1.0, 0.5)
    mid = spec(mid, lambda f, tt: g_bp(f, 1100, 0.6))
    mid *= (0.6 + 0.4 * np.sin(TAU * 21 * t + 3 * lfo_noise(n, 2, seed + 4)))[:, None] * np.clip(flick, 0.2, None)[:, None]
    hiss = butter(noise2(n, seed + 5, 0.0, 0.5), 'high', 5000, 2) * np.clip(flick, 0, None)[:, None] ** 2
    cr = crackle(T, lambda tt: 22 + 14 * np.sin(tt * 1.7) ** 2, 700, 4500, (0.002, 0.012), seed + 6,
                 lambda tt: pan_fn(t_abs0 + tt), 0.15, 1.3)
    out = roar * 0.9 + mid * 0.25 + hiss * 0.06 + cr * 0.55
    return pan2(out, pan_fn(t_abs0 + t))


def sfx_ice_form(B):
    """Ice orb synced to orbs.ice_state: frost breath -> shards burst out (26.20-26.50) as ascending
    crystal 'tsing' grains + cracks -> lock 'ting' + snap + small body at the flash (T_ICE + 0.02).
    Crystal identity kept above ~3-4 kHz so it sits under 'gelo' without masking it."""
    sh0, sh1, tl_ = B['ice_sh0'], B['ice_sh1'], B['ice_lock']
    s0 = sh0 - 0.4
    T = (tl_ - s0) + 2.2
    n = ns(T)
    t = tvec(n)
    out = np.zeros((n, 2))
    r = rng(800)
    a, b, L = sh0 - s0, sh1 - s0, tl_ - s0
    fr = whoosh(T, [(0, 5000), (a, 7500), (T, 9000)], [(0, 0), (a - 0.02, 0.55), (L, 0.3), (L + 0.8, 0), (T, 0)],
                bw=0.4, seed=801, slope=0.0, corr=0.6)
    out += fr * 0.45
    nsh = 14
    tsh = a + (b - a) * np.arange(nsh) / (nsh - 1)
    scale = [hz(x) for x in ('E7', 'F#7', 'A7', 'B7', 'C#8', 'E8', 'F#8')]
    freqs = [scale[min(len(scale) - 1, int(i * len(scale) / nsh))] for i in range(nsh)]
    pans = r.uniform(-0.35, 0.35, nsh)
    out += grains(T, tsh, lambda j, q: bell(freqs[j] * q.uniform(0.997, 1.003), 0.9, ratio=q.choice([2.76, 3.5, 4.1]),
                                            index=1.4, tau=0.22, tau_i=0.04, glass=0.35) * (0.55 + 0.45 * j / (nsh - 1)),
                  pans, 802) * 0.24
    # each shard's growth 'tss' (ease-out-back over 0.28 s): short HP noise swells
    for j, tj in enumerate(tsh):
        m = ns(0.3)
        tl = tvec(m)
        g = butter(rng(810 + j).standard_normal(m), 'high', 4500, 2) * (np.sin(np.pi * np.clip(tl / 0.28, 0, 1)) ** 2)
        put(out, g * 0.05 * np.exp(-tl / 0.12), tj, pan=float(pans[j]))
    cr = crackle(T, lambda tt: 50 * np.exp(-0.5 * ((tt - (a + b) / 2) / 0.18) ** 2) + 10 * np.exp(-0.5 * ((tt - L) / 0.05) ** 2) + 1.5,
                 3000, 9000, (0.002, 0.012), 803, None, 0.4, 1.5, 3)
    out += cr * 0.5
    # lock: ting (B7 + partials) + crystalline snap + small body (phones) at the flash
    m = n - ns(L)
    tl = tvec(m)
    ting = sum(a_ * np.sin(TAU * f * tl) * np.exp(-tl / tc)
               for f, a_, tc in ((hz('B7'), 1.0, 0.9), (hz('B7') * 2.756, 0.35, 0.3), (hz('F#8'), 0.5, 0.6), (hz('B7') * 1.5, 0.3, 0.5)))
    ting *= 1 - np.exp(-tl / 0.0008)
    put(out, ting * 0.34, L, pan=0.1)
    m2 = ns(0.3)
    snap = butter(r.standard_normal(m2), 'high', 4000, 2) * np.exp(-tvec(m2) / 0.005) * 1.5
    put(out, snap * 0.8, L)
    put(out, impact(r, 320, 190, 0.05, 3000, body=0.6, noise=0.5, click_amt=0.7, dur=0.35) * 0.55, L)
    put(out, sub_drop(0.6, 120, 70, 0.03, 0.12, 1.6, harm=0.5) * 0.35, L)
    return out, s0


def ice_bed(T, pan_fn, t_abs0, seed=820):
    n = ns(T)
    t = tvec(n)
    out = np.zeros((n, 2))
    r = rng(seed)
    for k, f in enumerate([hz('B6'), hz('E7'), hz('F#7') * 1.003, hz('A7'), hz('E8') * 0.998]):
        a = 0.5 + 0.5 * np.sin(TAU * r.uniform(0.15, 0.6) * t + r.uniform(0, TAU))
        out += pan2(sine(f, n) * a * 0.05 / (1 + 0.3 * k), r.uniform(-0.3, 0.3))
    fr = butter(noise2(n, seed + 1, 0.0, 0.5), 'high', 6500, 2) * (0.6 + 0.4 * lfo_noise(n, 0.8, seed + 2))[:, None]
    out += fr * 0.05
    times = poisson_times(T, lambda tt: 3.0 + 0 * tt, seed + 3)
    out += sparkle(T, times, [hz('E7'), hz('F#7'), hz('B7')], seed + 4, None, 0.3, (0.03, 0.12), 0.25)
    cr = crackle(T, lambda tt: 3 + 0 * tt, 2500, 7000, (0.002, 0.01), seed + 5, None, 0.3, 1.6, 2)
    out += cr * 0.3
    return pan2(out, pan_fn(t_abs0 + t))


def sfx_fusion_converge(B, panR, panL):
    """Orbs leave the palms and fly into the finger ring (orbs.T_FLY0 .. T_FLY1): a fire whoosh from the
    R palm and an ice whoosh from the L palm glide to the centre with a Doppler pitch-up, ending in a
    soft lock 'whump' + violet shimmer at the ring entry. Plus the shrink/push-aside 'fwip'."""
    p0, f0, f1 = B['push0'], B['fl0'], B['fl1']
    s0 = p0 - 0.02
    T = (f1 - s0) + 1.4
    n = ns(T)
    t = tvec(n)
    out = np.zeros((n, 2))
    r = rng(930)
    a, b = f0 - s0, f1 - s0
    k = smooth01(np.clip((t - a) / (b - a), 0, 1))
    # push aside (orbs shrink and slide outward off his eyes)
    pw = whoosh(T, [(0, 2500), (B['push1'] - s0, 1200), (T, 1200)],
                [(0, 0), (0.12, 0.35), (B['push1'] - s0, 0.0), (T, 0)], bw=0.5, seed=931, corr=0.5)
    out += widen(pw, 1.3) * 0.3
    amp = [(0, 0), (a, 0), (a + 0.08, 0.2), (b - 0.04, 1.0), (b + 0.02, 0.25), (b + 0.35, 0), (T, 0)]
    fw = whoosh(T, [(0, 450), (a, 450), (b, 1700), (T, 900)], amp, bw=0.8, seed=932, corr=0.6,
                tone=([(0, 150), (a, 150), (b, 330), (T, 200)], 0.25))
    fw *= (1 + 0.35 * lfo_noise(n, 18, 933))[:, None]
    out += pan2(fw * 0.7, float(panR(f0)) * (1 - k))
    iw = whoosh(T, [(0, 2600), (a, 2600), (b, 7500), (T, 7000)], amp, bw=0.45, seed=934, corr=0.6, slope=0.0)
    out += pan2(iw * 0.55, float(panL(f0)) * (1 - k))
    gl = a + np.sort(r.uniform(0, b - a, 12))
    out += sparkle(T, gl, [hz(x) for x in ('E7', 'B7', 'F#7')], 935,
                   lambda tt: float(panL(f0)) * (1 - smooth01(np.clip((tt - a) / (b - a), 0, 1))), 0.15, (0.03, 0.1), 0.25)
    # ring entry: soft whump + violet shimmer
    put(out, impact(r, 230, 120, 0.09, 1600, body=0.9, noise=0.6, click_amt=0.3, dur=0.6) * 0.6, b)
    put(out, sub_drop(0.9, 90, 48, 0.08, 0.25, 2.0, harm=0.5) * 0.5, b)
    m = n - ns(b)
    tl = tvec(m)
    sh = sum(sine(hz(x) * (1 + 0.003 * j), m) * env_ar(tl - 0.03 * j, 0.01, 0.5) * (tl > 0.03 * j) / (1 + 0.4 * j)
             for j, x in enumerate(('D6', 'A6', 'E7', 'F#7')))
    put(out, widen(pan2(sh * 0.12, 0), 1.3), b)
    return out, s0


def sfx_fusion(B):
    """Helix around the finger ring (orbs.T_UNR0 .. T_REL): circling fire + ice sources, rising electric
    hum, arcs between the hands, helix jet-swirl unrolling, violet riser. No fade-in dip."""
    a, b = B['unr0'], B['rel']
    d = b - a
    T = d + 0.25
    n = ns(T)
    t = tvec(n)
    out = np.zeros((n, 2))
    u = np.clip(t / d, 0, 1)
    rot = 1.0 + 2.4 * u ** 1.3
    ph = TAU * np.cumsum(rot) / SR
    rad = 0.55 - 0.25 * u
    p_fire = rad * np.sin(ph)
    p_ice = -rad * np.sin(ph)
    depth = 1 + 0.35 * np.cos(ph)
    fb = fire_bed(T, lambda tt: 0 * tt, 0, 900)[:, 0]
    out += pan2(fb * depth * 0.42, p_fire)
    ib = ice_bed(T, lambda tt: 0 * tt, 0, 910)[:, 0]
    out += pan2(ib * (2 - depth) * 0.9, p_ice)
    f0 = hz('D2') * 2 ** (3 / 12 * u ** 2)
    trem = 0.65 + 0.35 * np.sin(TAU * np.cumsum(6 + 12 * u) / SR)
    hm = saw(f0, n) + 0.7 * saw(f0 * 1.5 * 1.003, n) + 0.5 * saw(f0 * 2 * 0.997, n)
    hm = spec(hm, lambda f, tt: g_bp(f, np.interp(tt, [0, d], [500, 1800]), 1.0))
    hm *= trem * smooth01(u * 2.2) * (0.35 + 0.65 * u)
    out += pan2(hm * 0.6, 0)
    times = poisson_times(d, lambda tt: 3 + 11 * (tt / d) ** 1.3, 920)
    r = rng(921)
    pans = np.where(np.arange(len(times)) % 2 == 0, -0.3, 0.3) + r.uniform(-0.15, 0.15, len(times))
    out += grains(T, times, lambda k, q: zap(q, q.uniform(0.04, 0.2)) * q.uniform(0.4, 1.0), pans, 922) * 0.42
    unr = (B['unr1'] - a)
    sw = colored(n, 923, -1)
    sw = flanger(sw, 0.6, 4.0, 0.4 + 1.2 * u, 0.85, 924, 3)
    sw = spec(sw, lambda f, tt: g_bp(f, np.interp(tt, [0, d], [900, 2600]), 0.8))
    out += pan2(sw * (0.5 * smooth01(t / unr) + 0.5 * smooth01(u)) * 0.35, 0.5 * np.sin(ph * 0.5))
    ri = whoosh(T, [(0, 600), (d * 0.4, 600), (d, 4000), (T, 4000)], [(0, 0), (d * 0.4, 0), (d, 0.55), (T, 0)], bw=0.6, seed=925)
    out += ri
    out *= (smooth01(t / 0.12) * (1 - smooth01((t - d) / 0.25)))[:, None]
    return out, a


def sfx_release_charge(B):
    """Hands spread (whoosh) -> plasma born thrum -> charge crescendo -> INHALE (orbs: 32.40-32.54 the
    orb dims): the charge drops ~12 dB and the D1 swell stops, a reverse 'suck' rises into the
    collapse (32.565), then near-silence until the boom."""
    t_rel, t_boom = B['rel'], B['boom']
    t_in0, t_in1, t_col = B['pl_in0'], B['pl_in1'], B['pl_col']
    d = t_boom - t_rel
    n = ns(d)
    t = tvec(n)
    u = t / d
    out = np.zeros((n, 2))
    w = whoosh(0.7, [(0, 500), (0.25, 1800), (0.7, 700)], [(0, 0), (0.08, 0.8), (0.7, 0)], bw=0.6, seed=950, corr=0.3)
    w = widen(w, np.interp(tvec(len(w)), [0, 0.5], [0.5, 1.4]))
    put(out, w * 0.35, 0.0)
    th = sub_drop(0.5, 70, 110, 0.15, 0.25, 2.0, harm=0.5) * 0.3
    put(out, th, B['pl_born'] - t_rel)
    put(out, impact(rng(956), 180, 240, 0.08, 1500, body=0.5, noise=0.4, click_amt=0.2, dur=0.4) * 0.25, B['pl_born'] - t_rel)
    f = hz('D3') * 2 ** (1.0 * smooth01(u) ** 1.3)
    trem = 0.6 + 0.4 * np.sin(TAU * np.cumsum(4 + 26 * u ** 2) / SR)
    tone = saw(f, n) + 0.6 * saw(f * 1.5 * 1.002, n) + 0.4 * saw(f * 2.003, n)
    tone = spec(tone, lambda ff, tt: g_lp(ff, np.interp(tt, [0, d], [700, 5000]), 2))
    charge = pan2(tone * trem * smooth01(u * 3) * (0.1 + 0.9 * u ** 2) * 0.45, 0)
    ri = whoosh(d, [(0, 500), (d * 0.6, 2500), (d, 8000)], [(0, 0), (d * 0.98, 1.0), (d, 0)], bw=0.5, seed=951)
    charge += ri * (0.15 + 0.85 * u ** 2.2)[:, None] * 1.2
    sh = shepard(d, 0.6 + 2.5 * u, 1400, 7, True, 952) * smooth01(u * 2) * (0.1 + 0.9 * u ** 2)
    charge += pan2(sh * 0.35, 0)
    times = poisson_times(d, lambda tt: 5 + 40 * (tt / d) ** 2, 953)
    r = rng(954)
    charge += grains(d, times, lambda k, q: zap(q, q.uniform(0.03, 0.12), 1.2) * q.uniform(0.3, 1.0),
                     r.uniform(-0.4, 0.4, len(times)) * (1 - np.asarray(times) / d * 0.7), 955) * 0.35
    a_in, b_in, c_col = t_in0 - t_rel, t_in1 - t_rel, t_col - t_rel
    gin = env_pts(t, [(0, 1), (a_in, 1), (a_in + 0.10, undb(-12)), (c_col, undb(-15)), (c_col + 0.01, 0), (d, 0)])
    out += charge * gin[:, None]
    swl = sat(sine(hz('D1') * 2 ** (u * 0.5), n), 1.5) * u ** 2
    swl *= env_pts(t, [(0, 1), (a_in, 1), (a_in + 0.10, 0), (d, 0)])
    out += pan2(swl * 0.45, 0)
    # reverse 'suck' into the collapse: rising, narrowing whoosh + rising whistle, hard stop at c_col
    L = c_col - a_in
    sk = whoosh(L, [(0, 900), (L, 6500)], [(0, 0), (L - 0.008, 1.0), (L, 0)], bw=0.5, seed=957, corr=0.6,
                bw_pts=[(0, 0.7), (L, 0.25)])
    sk *= (np.linspace(0, 1, len(sk)) ** 2.5)[:, None]
    m = len(sk)
    tl = tvec(m)
    wh = sine(700 * 2 ** (1.8 * tl / L), m) * (tl / L) ** 3 * 0.35
    put(out, (sk + pan2(wh, 0)[:m]) * 0.5, a_in)
    # collapse glint (tiny, high) then near-silence until the boom
    m = ns(0.05)
    gl = np.sin(TAU * 6200 * tvec(m)) * np.exp(-tvec(m) / 0.008)
    put(out, gl * 0.12, c_col)
    out *= (1 - smooth01((t - (d - 0.006)) / 0.006))[:, None]
    return out, t_rel


def sfx_boom(B):
    """Finale detonation: crack (broadband + 1-3 kHz snap), LP-swept blast, 150-600 Hz body, sub with
    harmonic layer, shockwave, debris (decaying, cut at finale_end + 0.3 s), rumble tail; saturated glue."""
    T = 4.6
    n = ns(T)
    t = tvec(n)
    out = np.zeros((n, 2))
    r = rng(1000)
    k = ns(0.006)
    cr = r.standard_normal((k, 2)) * np.exp(-np.arange(k) / (k / 4))[:, None]
    out[:k] += cr * 1.4
    m = ns(0.06)
    snap = butter(r.standard_normal(m), 'band', [1000, 3500], 2)
    snap = snap / np.abs(snap).max() * np.exp(-tvec(m) / 0.009)
    put(out, snap * 0.9, 0.0)
    bl = noise2(n, 1001, -0.6, 0.5)
    bl = spec(bl, lambda f, tt: g_lp(f, 250 + 14000 * np.exp(-tt / 0.22), 2) * g_hp(f, 40, 2))
    bl *= env_ar(t, 0.003, 0.55)[:, None] * (1 + 0.25 * lfo_noise(n, 25, 1002))[:, None]
    out += bl * 1.2
    put(out, impact(r, 300, 90, 0.22, 2000, body=1.0, noise=0.9, click_amt=0.6, dur=1.4) * 0.9, 0.0)
    bo = butter(colored(n, 1003, -2), 'low', 320, 2) * env_ar(t, 0.004, 0.35)
    out += pan2(bo * 0.8, 0)
    sd = sub_drop(T, 75, 26, 0.35, 1.3, 2.8, 0.003, harm=0.55)
    out += pan2(sd * 1.15, 0)
    sw = whoosh(1.4, [(0, 2500), (0.6, 600), (1.4, 250)], [(0, 0.0), (0.04, 1), (1.4, 0)], bw=0.7, seed=1004, corr=0.3)
    sw = widen(sw, np.interp(tvec(len(sw)), [0, 0.8], [0.6, 1.4]))
    out[:len(sw)] += sw * 0.7
    t_cut = B['fin_end'] + 0.3 - B['boom']
    times = np.sort(0.08 + r.exponential(0.5, 40))
    times = times[times < t_cut]
    ice_f = [hz(x) for x in ('E6', 'B6', 'F#7', 'A6', 'C#7')]
    out += sparkle(T, times, ice_f, 1005, None, 0.8, (0.05, 0.25), 0.22, amps=np.exp(-times / 0.8))
    emb = crackle(T, lambda tt: 160 * np.exp(-tt / 0.35) + 12 * np.exp(-tt / 1.6), 900, 6000, (0.002, 0.01), 1006,
                  None, 0.85, 1.4, decay=0.8, clamp_after=0.5, t_cut=t_cut)
    out += emb * 0.45
    ru = noise2(n, 1007, -2.0, 0.8)
    ru = butter(ru, 'low', 140, 2) * env_ar(t, 0.05, 0.9)[:, None]
    out += ru * 0.6
    out = norm_peak(out)
    return sat(out, 2.6), B['boom']


def sfx_portal_open_hit(B):
    """Ceiling portal ignition: pinpoint glimmer (44.9-45.9, flickering like the visual), reverse swell
    into the open, impact at the spring start 45.98 (body + click + sub w/ harmonics), crackle burst,
    shimmer, shock-ring whoosh expanding to 46.8."""
    tg, to = B['po_glim0'], B['po_open']
    s0 = tg
    T = (to - s0) + 2.8
    n = ns(T)
    t = tvec(n)
    out = np.zeros((n, 2))
    r = rng(1100)
    a = to - s0
    flick = 0.6 + 0.4 * np.sin(t * 53.0 + s0 * 53.0) * np.sin(t * 31.0 + s0 * 31.0)
    gle = env_pts(t, [(0, 0), (0.9, 0.6), (a - 0.05, 1.0), (a, 0), (T, 0)]) * flick
    gls = sum(sine(hz(x) * (1 + 0.002 * j), n) / (1 + j) for j, x in enumerate(('A6', 'D7', 'E7')))
    out += pan2(gls * gle * 0.05, -0.05)
    out += whoosh(T, [(0, 6000), (a, 8000), (T, 8000)], [(0, 0), (a - 0.1, 0.2), (a, 0), (T, 0)], bw=0.4, seed=1101, slope=0.0) * gle[:, None] * 0.3
    rv = whoosh(0.4, [(0, 300), (0.4, 2500)], [(0, 0), (0.39, 1.0), (0.4, 0)], bw=0.6, seed=1102, corr=0.5)
    put(out, rv * 0.7, a - 0.4)
    put(out, impact(r, 230, 100, 0.14, 2200, body=1.0, noise=0.8, click_amt=0.6, dur=1.0) * 0.8, a)
    put(out, sub_drop(2.5, 60, 32, 0.3, 0.9, 2.2, harm=0.55) * 0.6, a)
    cb = crackle(2.5, lambda tt: 120 * np.exp(-tt / 0.2) + 1, 900, 6000, (0.0015, 0.008), 1103, None, 0.7, 1.6, 3)
    put(out, cb * 0.55, a)
    m = ns(2.5)
    sh = sum(sine(hz(x) * (1 + 0.003 * j), m) / (1 + j) for j, x in enumerate(('D6', 'A6', 'E7', 'F#7')))
    put(out, sh * env_ar(tvec(m), 0.05, 0.8) * 0.12, a)
    rw = whoosh(0.9, [(0, 2500), (0.9, 500)], [(0, 0), (0.03, 1.0), (0.9, 0)], bw=0.6, seed=1104, corr=0.4)
    put(out, widen(rw, np.interp(tvec(len(rw)), [0, 0.6], [0.7, 1.4])) * 0.75, a)
    return out, s0


def portal_vortex_bed(B):
    """Sustained ceiling vortex from the opening to the collapse: sub rumble (mono), rotating flanged
    wind, a 300 Hz-2 kHz swirling flanged layer (reads on phones), rim zaps, star chimes. Intensity
    follows ceiling_portal (spring open, settle, pull growing into the suction)."""
    to, t1 = B['po_open'], B['col0']
    T = t1 - to + 0.1
    n = ns(T)
    t = tvec(n)
    ta = t + to
    out = np.zeros((n, 2))
    sd = 1150
    grow = np.interp(ta, [to, to + 0.92, to + 1.52, 48.4, 49.3, 50.2, t1, t1 + 0.1], [0, 1.0, 0.82, 0.88, 1.05, 1.35, 1.4, 0])
    rot = 0.4 + 0.35 * np.clip((ta - 48.4) / 2.0, 0, 1)
    rph = TAU * np.cumsum(rot) / SR
    ru = butter(colored(n, sd + 2, -2.0), 'low', 130, 2)
    ru += sat(sine(hz('D1'), n) + 0.5 * sine(hz('D1') * 1.01, n), 1.5) * 0.4
    out += pan2(ru * grow * (0.8 + 0.2 * np.sin(rph)) * 0.55, 0)
    wn = colored(n, sd + 3, -1)
    fc = 800 * 2 ** (1.0 * np.sin(rph * 0.7))
    wn = spec(wn, lambda f, tt: g_bp(f, np.interp(tt, t, fc), 0.5))
    wn = flanger(wn, 1.0, 5.0, 0.25, 0.7, sd + 4, 2)
    out += pan2(wn * grow * 0.38, 0.55 * np.sin(rph))
    sw = colored(n, sd + 10, -1)
    fc2 = 750 * 2 ** (0.9 * np.sin(rph + 1.7))
    sw = spec(sw, lambda f, tt: g_bp(f, np.interp(tt, t, fc2), 0.6) * g_hp(f, 300, 2) * g_lp(f, 2000, 2))
    sw = flanger(sw, 0.8, 3.0, 0.6 + 0.4 * rot, 0.8, sd + 11, 3)
    out += pan2(sw * grow * 0.45, -0.45 * np.sin(rph + 0.8))
    times = poisson_times(T, lambda tt: 4 + 3 * tt / T, sd + 5)
    zr = rng(sd + 6)
    out += grains(T, times, lambda k, q: zap(q, q.uniform(0.03, 0.1), 1.3) * q.uniform(0.2, 0.7),
                  0.7 * np.sin(np.interp(times, t, rph) + zr.uniform(-0.5, 0.5, len(times))), sd + 7) * 0.22 * np.interp(np.arange(n), [0, n], [0.6, 1.0])[:, None]
    st = poisson_times(T, lambda tt: np.where(tt > 0.3, 1.4, 0), sd + 8)
    out += sparkle(T, st, [hz(x) for x in ('D7', 'A6', 'E7', 'F#6')], sd + 9, None, 0.6, (0.2, 0.6), 0.10)
    out *= (1 - smooth01((t - (T - 0.25)) / 0.25))[:, None]
    return out, to


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
    pk = 0.75 * d
    wu = whoosh(T, [(0, 600), (pk, 2600), (T, 400)], [(0, 0), (pk - 0.25, 0.4), (pk, 1.0), (d, 0.25), (T, 0)],
                bw=0.6, seed=1202, corr=0.4, tone=([(0, 160), (pk, 120), (T, 60)], 0.3))
    out += wu * 0.8
    sd = sub_drop(T, 45, 75, 0.5, 2.0, 2.0, harm=0.3) * smooth01(u * 1.2) * (1 - smooth01((t - d) / 0.15))
    out += pan2(sd * 0.6, 0)
    return out


def sfx_portal_close(B):
    """Implosion (ceiling_portal: 50.55 -> point at 50.84): reverse-reverb swell + rising implosion tone
    normalised to reach -6 dB re the hit, then the flash hit at 50.84 (impact + sub + HP flash +
    shimmer + shock-ring whoosh) and the spark burst."""
    c0, tf = B['col0'], B['col_flash']
    pk = tf - c0
    T = 2.4
    n = ns(T)
    out = np.zeros((n, 2))
    r = rng(1300)
    i0 = ns(pk)
    m = n - i0
    tl = tvec(m)
    hit = np.zeros((m, 2))
    put(hit, sub_drop(T - pk, 90, 38, 0.06, 0.4, 2.5, harm=0.55), 0.0)
    put(hit, impact(r, 260, 110, 0.10, 2500, body=1.0, noise=0.8, click_amt=0.7, dur=0.8) * 0.8, 0.0)
    hit += butter(noise2(m, 1302, 0.0, 0.5), 'high', 2500, 2) * env_ar(tl, 0.002, 0.09)[:, None] * 0.6
    hit += pan2(sum(sine(hz(x), m) for x in ('D6', 'A6', 'D7')) * env_ar(tl, 0.003, 0.5) * 0.15, 0)
    rw = whoosh(0.8, [(0, 3000), (0.8, 500)], [(0, 0), (0.02, 1.0), (0.8, 0)], bw=0.6, seed=1303, corr=0.4)
    put(hit, widen(rw, np.interp(tvec(len(rw)), [0, 0.5], [0.7, 1.4])) * 0.45, 0.0)
    sp = crackle(1.2, lambda tt: 160 * np.exp(-tt / 0.15) + 1, 2000, 8000, (0.001, 0.005), 1304, None, 0.8, 1.6)
    put(hit, sp * 0.4, 0.02)
    burst = noise2(ns(0.05), 1305, -0.5, 0.5) * np.exp(-tvec(ns(0.05)) / 0.01)[:, None]
    ir = make_ir(2.0, 1.4, 1.2, 0.0, 0, 0.01, 8000, 1301)
    rv = convolve(np.pad(burst, ((0, ns(1.2)), (0, 0))), ir)[::-1][-i0:]
    rv = norm_peak(rv) * (np.linspace(0, 1, len(rv)) ** 1.6)[:, None]
    mt = len(rv)
    tt_ = tvec(mt)
    rt = sine(200 * 2 ** (2.2 * (tt_ / pk) ** 1.5), mt) * (tt_ / pk) ** 2
    rv = rv + pan2(rt * 0.5, 0)
    rv *= 0.5 * rms_max(hit, 0.05) / max(rms_max(rv[-ns(0.05):], 0.05), 1e-9)     # -6 dB re hit
    out[:i0] += rv[-i0:]
    out[i0:] += hit
    return out, c0


def tunnel_speed(B, t):
    """ceiling_portal._tunnel_speed (formula, jolt time from the beats)."""
    tau = np.clip((t - B['tun0']) / (B['tun1'] - B['tun0']), 0, 1)
    return 1.6 + 13.0 * tau ** 2.3 + 3.0 * np.exp(-((t - B['jolt']) / 0.15) ** 2)


def sfx_tunnel_bed(B):
    a, b = B['tun0'], B['tun1']
    d = b - a
    n = ns(d)
    t = tvec(n)
    u = t / d
    out = np.zeros((n, 2))
    wn = noise2(n, 1400, -1.0, 0.5)
    wn = np.stack([flanger(wn[:, c], 0.5, 6.0, 0.4 + 1.6 * u, 0.8, 1401 + c, 3) for c in range(2)], 1)
    wn = spec(wn, lambda f, tt: g_bp(f, np.interp(tt, [0, d], [500, 2400]), np.interp(tt, [0, d], [0.9, 0.6])))
    out += wn * (0.3 + 0.7 * smooth01(u * 1.5))[:, None] * 0.75
    sh = shepard(d, 0.5 + 2.0 * u, 700, 7, True, 1450) * (0.4 + 0.6 * u)
    out += pan2(sh * 0.22, 0)
    ru = butter(colored(n, 1451, -2.0), 'low', 110, 2) * (0.4 + 0.6 * u)
    out += pan2(ru * 0.55, 0)
    out *= smooth01(t / 0.3)[:, None]
    return out, a


def sfx_tunnel_hits(B):
    """Ring pass-bys spaced by distance travelled (speed curve of ceiling_portal), the turbulence jolt
    at 52.30 (thump + close pass-by + rattle) and the reverse cymbal into the exit flash."""
    a, b = B['tun0'], B['tun1']
    d = b - a
    n = ns(d)
    out = np.zeros((n, 2))
    tg = np.linspace(a, b, 2000)
    D = np.concatenate([[0], np.cumsum(tunnel_speed(B, tg)[1:] * np.diff(tg))])
    k = 0
    for Dk in np.arange(1.0, D[-1] - 0.5, 1.35):
        tc = float(np.interp(Dk, D, tg))
        if abs(tc - B['jolt']) < 0.12:
            continue
        v = float(tunnel_speed(B, tc))
        dd = float(np.clip(1.0 / v, 0.08, 0.3))
        w = whoosh(dd, [(0, 2800), (dd * 0.5, 1400), (dd, 500)], [(0, 0), (dd * 0.5, 1), (dd, 0)], bw=0.5,
                   seed=1403 + k, corr=0.5)
        side = 0.65 if k % 2 == 0 else -0.65
        w = pan2(w, np.linspace(side * 0.3, side, len(w)))
        put(out, w * (0.3 + 0.5 * (tc - a) / d) * 0.9, tc - a - dd * 0.5)
        k += 1
    r = rng(1460)
    tj = B['jolt'] - a
    put(out, impact(r, 190, 70, 0.12, 1500, body=1.0, noise=0.8, click_amt=0.5, dur=0.8) * 0.8, tj)
    put(out, sub_drop(1.0, 70, 35, 0.1, 0.35, 2.2, harm=0.5) * 0.7, tj)
    dd = 0.3
    w = whoosh(dd, [(0, 3200), (dd * 0.45, 1500), (dd, 450)], [(0, 0), (dd * 0.45, 1), (dd, 0)], bw=0.55, seed=1461, corr=0.5)
    put(out, pan2(w, np.linspace(-0.8, 0.8, len(w))) * 1.0, tj - dd * 0.45)
    rt = crackle(0.5, lambda tt: 200 * np.exp(-tt / 0.1), 800, 3000, (0.002, 0.008), 1462, None, 0.6, 1.5, 3)
    put(out, rt * 0.4, tj)
    rc = whoosh(0.38, [(0, 5000), (0.38, 9000)], [(0, 0), (0.37, 1), (0.38, 0)], bw=0.6, seed=1452, slope=0.0)
    put(out, rc * 0.8, d - 0.38)
    return out, a


def sfx_exit_flash():
    T = 4.0
    n = ns(T)
    t = tvec(n)
    out = np.zeros((n, 2))
    r = rng(1501)
    sd = sub_drop(T, 70, 30, 0.3, 1.2, 2.2, harm=0.5)
    out += pan2(sd * 0.75, 0)
    put(out, impact(r, 240, 100, 0.12, 2400, body=0.8, noise=0.8, click_amt=0.5, dur=0.9) * 0.55, 0.0)
    bl = noise2(n, 1500, -0.8, 0.5)
    bl = spec(bl, lambda f, tt: g_lp(f, 400 + 9000 * np.exp(-tt / 0.3), 2))
    out += bl * env_ar(t, 0.004, 0.6)[:, None] * 0.65
    sh = sum(sine(hz(x) * (1 + 0.002 * j), n) / (1 + 0.4 * j) for j, x in enumerate(('D5', 'A5', 'D6', 'F#6', 'A6', 'E7')))
    out += pan2(sh * env_ar(t, 0.02, 1.4) * 0.12, 0)
    return out


def sfx_arrive(B):
    """Arrival: inverse-suction down-whoosh during the emergence, landing at 54.19 (camera bump 54.18 /
    squash 54.21): thump + 200-600 Hz body + soft cloth/skin foley; portal close 'shhp' + pop 54.43."""
    a0, tl_, tp = B['arr0'], B['land'], B['arr_pop']
    s0 = a0 - 0.03
    T = (tp - s0) + 1.4
    n = ns(T)
    out = np.zeros((n, 2))
    r = rng(1600)
    L = tl_ - s0
    w = whoosh(L + 0.05, [(0, 4500), (L - 0.05, 500), (L + 0.05, 400)], [(0, 0), (L * 0.55, 0.8), (L, 0.15), (L + 0.05, 0)],
               bw=0.6, seed=1600, tone=([(0, 700), (L, 180), (L + 0.05, 160)], 0.15))
    put(out, w * 0.75, 0.0)
    put(out, sub_drop(1.0, 80, 45, 0.06, 0.25, 1.8, harm=0.5) * 0.6, L)
    put(out, impact(r, 380, 200, 0.06, 2600, body=0.9, noise=0.9, click_amt=0.4, dur=0.5) * 0.6, L, pan=0.05)
    m = ns(0.12)
    fo = butter(r.standard_normal(m), 'band', [900, 4000], 2) * np.exp(-tvec(m) / 0.018)
    fo /= np.abs(fo).max() + 1e-9
    put(out, fo * 0.35, L + 0.004, pan=0.1)
    P = tp - s0
    cs = whoosh(0.12, [(0, 2000), (0.12, 8000)], [(0, 0), (0.11, 0.6), (0.12, 0)], bw=0.5, seed=1602, slope=0.0)
    put(out, cs, P - 0.12)
    m = ns(0.3)
    pp = butter(r.standard_normal(m), 'band', [1500, 6000], 2) * np.exp(-tvec(m) / 0.006)
    put(out, pp / (np.abs(pp).max() + 1e-9) * 0.35, P)
    st = P + 0.04 + np.sort(rng(1603).exponential(0.15, 7))
    out += sparkle(T, st, [hz(x) for x in ('A6', 'D7', 'E7')], 1604, None, 0.4, (0.1, 0.3), 0.18)
    return out, s0


def space_amb(T, seed=1700):
    n = ns(T)
    dr = butter(noise2(n, seed, -2.0, 0.7), 'band', [28, 110], 2) * (0.7 + 0.3 * lfo_noise(n, 0.08, seed + 1))[:, None]
    wind = noise2(n, seed + 2, -1.0, 0.5)
    t = tvec(n)
    c1 = 500 * 2 ** (0.8 * lfo_noise(n, 0.05, seed + 3))
    c2 = 1500 * 2 ** (0.6 * lfo_noise(n, 0.04, seed + 4))
    wind = spec(wind, lambda f, tt: 0.7 * g_bp(f, np.interp(tt, t, c1), 0.12) + 0.4 * g_bp(f, np.interp(tt, t, c2), 0.1),
                4096, 1024)
    times = poisson_times(T, lambda tt: 1.3 + 0 * tt, seed + 5)
    tw = sparkle(T, times, [hz(x) for x in ('D7', 'E7', 'F#7', 'A7', 'B7', 'D8')], seed + 6, None, 0.9, (0.15, 0.5), 0.10)
    return dr * 0.9 + wind * 0.45 + tw


def sfx_blackhole_form(B):
    """Black hole (space.py: seed 61.62, grows 62.0 -> 62.70, IGNITES 62.74 on 'NEgro'):
    seed glint, gravity inhale + in-falling grains converging on the hole + downward sweep cluster,
    then at the ignition: bright limb-flash crack/shimmer (2-8 kHz, ~60 ms), sub + body, and a
    pitch-bent low 'whomp' (80-300 Hz) riding the gravitational-wave ripple; small settle."""
    tsd, ti = B['bh_seed'], B['bh_ign']
    pb = B['pan_bh']
    s0 = tsd
    T = (ti - s0) + 2.6
    n = ns(T)
    t = tvec(n)
    out = np.zeros((n, 2))
    r = rng(1800)
    I = ti - s0
    gl = sine(hz('A7') * (1 + 0.004 * np.sin(TAU * 6 * t)), n) * env_pts(t, [(0, 0), (0.08, 1), (0.3, 0.4), (0.7, 0), (T, 0)])
    out += pan2(gl * 0.06, pb)
    g0 = 62.0 - s0
    f = 40 + 150 * np.exp(-np.maximum(t - g0, 0) / 0.5)
    cl = saw(f, n) + saw(f * 1.5 * 1.004, n) * 0.6 + saw(f * 0.5, n) * 0.6
    cl = spec(cl, lambda ff, tt: g_lp(ff, 1200 * np.exp(-np.maximum(tt - g0, 0) / 0.8) + 150, 2))
    cl *= env_pts(t, [(0, 0), (g0 - 0.15, 0), (g0 + 0.15, 0.7), (I, 0.8), (I + 0.4, 0.3), (T, 0)])
    out += pan2(cl * 0.3, pb * 0.6)
    fall0 = 61.70 - s0
    times = poisson_times(I - fall0, lambda tt: 25 + 120 * (tt / (I - fall0)) ** 1.5, 1801) + fall0
    rr = rng(1802)
    p0 = rr.uniform(-0.9, 0.9, len(times))
    kk = np.clip((times - fall0) / (I - fall0), 0, 1)
    pans = p0 + (pb - p0) * kk
    out += grains(T, times, lambda k, q: click(q, q.uniform(1200, 4500) * (1 - 0.45 * kk[k]), 0.008, 2) * q.uniform(0.3, 1), pans, 1803) * 0.32
    # gravity inhale (reverse swell) into the ignition: from the start of the growth
    L = I - g0
    rv = whoosh(L, [(0, 3000), (L, 280)], [(0, 0), (L - 0.02, 0.9), (L, 0)], bw=0.7, seed=1804, corr=0.5)
    rv *= (np.linspace(0, 1, len(rv)) ** 1.8)[:, None]
    put(out, pan2(rv, pb * 0.7) * 0.6, g0)
    # ignition
    m = ns(0.12)
    tl = tvec(m)
    fl = butter(r.standard_normal(m), 'band', [2000, 8000], 2)
    fl = fl / np.abs(fl).max() * (1 - np.exp(-tl / 0.0008)) * np.exp(-tl / 0.022)
    sh = sum(np.sin(TAU * fq * tl + r.uniform(0, TAU)) * np.exp(-tl / 0.05) for fq in (2900, 4300, 6100, 7700)) * 0.2
    put(out, (fl + sh) * 0.7, I, pan=pb)
    put(out, impact(r, 260, 110, 0.10, 2400, body=1.0, noise=0.8, click_amt=0.4, dur=0.8) * 0.7, I, pan=pb * 0.5)
    put(out, sub_drop(2.2, 60, 28, 0.25, 0.75, 2.6, harm=0.55) * 0.65, I)
    gw = whoosh(1.0, [(0, 2600), (0.35, 900), (1.0, 400)], [(0, 0), (0.015, 1.0), (0.3, 0.45), (1.0, 0)], bw=0.6,
                seed=1805, corr=0.5, slope=-0.5)                                            # GW shock front
    put(out, widen(gw, np.interp(tvec(len(gw)), [0, 0.5], [0.7, 1.4])) * 0.6, I, pan=pb * 0.5)
    cz = crackle(0.8, lambda tt: 140 * np.exp(-tt / 0.12) + 1, 800, 3500, (0.002, 0.01), 1806, None, 0.6, 1.5, 3)
    put(out, cz * 0.35, I, pan=pb * 0.5)
    m = ns(1.3)
    tl = tvec(m)
    fw = 210 * np.exp(-tl / 0.25) + 85 + 18 * np.sin(TAU * 3.2 * tl) * np.exp(-tl / 0.45)   # GW ripple wobble
    wp = sine(fw, m) + 0.45 * sine(fw * 2.01, m) + 0.2 * sine(fw * 3.0, m)
    wp = sat(wp * 0.8, 1.4) * env_pts(tl, [(0, 0), (0.025, 1), (0.35, 0.55), (1.3, 0)])
    put(out, wp * 0.32, I + 0.01, pan=pb * 0.4)
    put(out, sub_drop(1.4, 55, 32, 0.15, 0.4, 2.0, harm=0.3) * 0.2, I + 0.55)
    return out, s0


def blackhole_bed(T, seed=1850, pan=0.5):
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
    out = pan2(sub * 0.65 + ru * 0.5, 0.0) + pan2(disk * 0.22, pan + 0.2 * np.sin(rot)) + pan2(hiss * 0.04, pan)
    return out


def _sg(B, key, tt):
    return np.interp(tt, B['sg_t'], B['sg_' + key])


def sfx_strange_bed(B, pan_fn):
    """Sling-ring bed driven by strange.py's own curves: spark sizzle and crackle density follow the
    emission rate E(t) (word-onset bursts included), width and portal roar follow the radius R(t),
    whir spinning, aperture burn-through (72.56-72.98) with a pressure 'whoomp', ring tones."""
    a = B['sg_pre']
    end = B['sg_out'] + 0.25
    T = end - a + 0.8
    n = ns(T)
    t = tvec(n)
    ta = t + a
    out = np.zeros((n, 2))
    E = _sg(B, 'E', ta)
    R = _sg(B, 'R', ta)
    I = _sg(B, 'I', ta)
    tw0 = B['sg_tw0']
    e_n = np.clip(E / 2500.0, 0, 1.6) ** 0.6 * np.where(ta > tw0, undb(-6), 1.0)
    rn = np.clip((R - 80) / 290.0, 0, 1)
    pan = pan_fn(ta)
    spin = 7.0 - 3.0 * np.clip((ta - B['sg_grow0']) / 1.0, 0, 1)
    sph = TAU * np.cumsum(spin) / SR
    sz = noise2(n, 1900, 0.0, 0.5)
    sz = spec(sz, lambda f, tt: g_hp(f, 3500, 2) * g_lp(f, 12000, 4))
    micro = uniform_filter1d(np.clip(1 + 1.6 * colored(n, 1901, 0.0), 0, None) ** 3, 30)
    micro /= micro.mean() + 1e-9
    am = 0.6 + 0.4 * np.sin(sph)
    out += pan2(sz.mean(1) * micro * am * e_n * 0.05, pan) * 0.5 + sz * (micro * am * e_n)[:, None] * 0.025
    cr = crackle(T, lambda tt: np.minimum(_sg(B, 'E', np.asarray(tt) + a) * 0.11, 260), 1500, 9000,
                 (0.0008, 0.005), 1902, lambda tt: pan_fn(np.asarray(tt) + a) + 0.3 * np.sin(np.interp(tt, t, sph)) * (0.3 + 0.7 * np.interp(tt, t, rn)), 0.2, 1.6)
    out += cr * 0.42
    for k, tp in enumerate(B['sg_pulses'][1:]):          # fizz bursts on the word onsets
        m = ns(0.35)
        tl = tvec(m)
        fz = butter(rng(1910 + k).standard_normal(m), 'band', [3000, 9000], 2)
        fz *= np.clip(1 + 2.0 * colored(m, 1920 + k, 0.0), 0, None) ** 2 * env_ar(tl, 0.01, 0.1)
        put(out, fz * 0.06, tp - a, pan=float(pan_fn(tp)))
    wh = colored(n, 1903, -1.0)
    fc = 1200 * 2 ** (0.5 * np.sin(sph)) * (1 - 0.3 * rn)
    wh = spec(wh, lambda f, tt: g_bp(f, np.interp(tt, t, fc), 0.35))
    out += pan2(wh * np.clip(I, 0, 1.5) * 0.25, pan + 0.3 * np.sin(sph) * (0.3 + 0.7 * rn))
    ro = noise2(n, 1904, -1.6, 0.6)
    ro = spec(ro, lambda f, tt: g_bp(f, 330, 1.0))
    out += widen(ro * (rn * np.clip(I, 0, 1.5))[:, None], 0.8 + 0.6 * rn) * 0.5
    ap0, ap1 = B['sg_ap0'] - a, B['sg_ap1'] - a
    bc = crackle(T, lambda tt: 160 * np.exp(-0.5 * ((tt - (ap0 + ap1) / 2) / ((ap1 - ap0) / 3)) ** 2), 2500, 9000,
                 (0.001, 0.005), 1905, None, 0.6, 1.7)
    out += bc * 0.35
    put(out, impact(rng(1906), 200, 120, 0.15, 1400, body=0.6, noise=0.7, click_amt=0.1, dur=0.8) * 0.3, ap0)
    room = butter(noise2(n, 1907, -1.5, 0.6), 'band', [200, 3000], 2)
    out += room * env_pts(ta, [(0, 0), (B['sg_ap0'], 0), (B['sg_ap1'], 0.6), (tw0, 0.6), (B['sg_rel'], 0.0), (99, 0)])[:, None] * 0.05
    rt = sum(sine(hz(x) * (1 + 0.004 * np.sin(TAU * 0.3 * t + j)), n) / (1 + j) for j, x in enumerate(('A5', 'E6', 'A6')))
    out += pan2(rt * np.clip(I, 0, 1.5) * rn * 0.04, B['pan_cf'])
    out *= smooth01(t / 0.02)[:, None] * (1 - smooth01((ta - (end - 0.3)) / 0.3))[:, None]
    return out, a


def sfx_strange_ign(B, pan):
    """Sling-ring ignition at strange.T_IGN: pre-glow flicker, then a spark-strike (bright noise burst +
    fizz tail + click + a little body)."""
    tp, ti = B['sg_pre'], B['sg_ign']
    s0 = tp
    T = (ti - s0) + 0.9
    n = ns(T)
    t = tvec(n)
    out = np.zeros((n, 2))
    r = rng(1950)
    cr = crackle(T, lambda tt: np.where(tt < ti - s0, 60, 0), 3000, 9000, (0.0006, 0.002), 1951, None, 0.1, 1.8)
    out += cr * 0.25
    a = ti - s0
    m = ns(0.6)
    tl = tvec(m)
    nb = butter(r.standard_normal(m), 'high', 2500, 2)
    nb = nb / np.abs(nb).max() * (1 - np.exp(-tl / 0.0006)) * np.exp(-tl / 0.03) * 1.2
    fz = butter(r.standard_normal(m), 'band', [3500, 11000], 2)
    fz *= np.clip(1 + 2.2 * colored(m, 1952, 0.0), 0, None) ** 2 * np.exp(-tl / 0.18) * 0.12
    put(out, nb + fz, a)
    put(out, impact(r, 400, 220, 0.04, 2600, body=0.4, noise=0.4, click_amt=0.9, dur=0.3) * 0.5, a)
    return pan2(out, pan), s0


def sfx_wipe(B):
    """Strange wipe, synced to strange.py: breath-in TW0 -> T_REL (reverse whoosh, everything sucked
    in), BURST at T_REL (transient + spark fizz + sub), Doppler whoosh rising as the rim sweeps
    across the frame edges and falling as it passes, reality SNAP when the rim has left the frame.
    Lows mono, width <= 1.4."""
    tw0, trel, tedge, tout = B['sg_tw0'], B['sg_rel'], B['sg_edge'], B['sg_out']
    s0 = tw0 - 0.02
    T = (tout - s0) + 1.6
    n = ns(T)
    t = tvec(n)
    out = np.zeros((n, 2))
    r = rng(2000)
    a, bpt, e, o = tw0 - s0, trel - s0, tedge - s0, tout - s0
    L = bpt - a
    inh = whoosh(L, [(0, 4500), (L, 900)], [(0, 0), (L - 0.012, 1.0), (L, 0)], bw=0.5, seed=2001, corr=0.6,
                 bw_pts=[(0, 0.3), (L, 0.7)])
    inh *= (np.linspace(0, 1, len(inh)) ** 2)[:, None]
    put(out, inh * 0.55, a)
    m = len(inh)
    tl = tvec(m)
    put(out, sine(900 * 2 ** (-1.0 * tl / L), m) * (tl / L) ** 2 * 0.12, a)
    put(out, impact(r, 280, 110, 0.08, 2200, body=1.0, noise=0.8, click_amt=0.8, dur=0.6) * 0.8, bpt)
    put(out, sub_drop(0.9, 75, 34, 0.1, 0.22, 2.4, harm=0.55) * 0.6, bpt)
    m = ns(0.08)
    ck = butter(r.standard_normal((m, 2)), 'high', 1500, 2) * np.exp(-tvec(m) / 0.006)[:, None] * 1.6
    put(out, ck, bpt)
    m = ns(0.9)
    tl = tvec(m)
    fz = butter(r.standard_normal((m, 2)), 'band', [3000, 11000], 2)
    fz *= (np.clip(1 + 2.0 * colored(m, 2002, 0.0), 0, None) ** 2 * env_ar(tl, 0.004, 0.22))[:, None] * 0.1
    put(out, fz, bpt)
    sp = crackle(1.2, lambda tt: 300 * np.exp(-tt / 0.18) + 2, 2500, 9000, (0.0008, 0.004), 2003, None, 0.8, 1.6)
    put(out, sp * 0.45, bpt)
    dw = whoosh(T, [(0, 500), (bpt, 600), (e, 3200), (o + 0.15, 700), (T, 350)],
                [(0, 0), (bpt, 0.0), (bpt + 0.08, 0.12), (e - 0.12, 0.6), (e, 1.0), (o, 0.55), (o + 0.45, 0), (T, 0)],
                bw=0.65, seed=2004, corr=0.55, tone=([(0, 90), (e, 150), (T, 60)], 0.22))
    out += widen(dw, np.interp(t, [bpt, e, o], [0.8, 1.4, 1.4])) * 1.5
    out *= (1 - 0.65 * np.exp(-0.5 * ((t - (o - 0.03)) / 0.022) ** 2))[:, None]      # pre-snap suck
    m = ns(0.7)
    tl = tvec(m)
    snap = butter(r.standard_normal(m), 'band', [1200, 5000], 2) * np.exp(-tl / 0.007)
    snap = snap / np.abs(snap).max()
    snap += impact(r, 360, 180, 0.05, 3000, body=0.7, noise=0.5, click_amt=0.6, dur=0.7) * 0.6
    snap += sub_drop(0.7, 120, 60, 0.03, 0.09, 1.5, harm=0.4) * 0.4
    put(out, snap * 0.9, o)
    return out, s0


# =============================================================================================
# SFX timeline
# =============================================================================================
# max momentary loudness targets (internal scale: voice integrated = -20 LUFS, master adds ~+6 dB)
LV = dict(claude=-28, eraser=-21, studio=-18, studio_tone=-54, suit_bed=-27, suit_hit=-22, antic=-28,
          fire_ignite=-20, fire_bed=-27, ice_form=-20, ice_bed=-30, converge=-22, fusion=-24, charge=-20,
          boom=-11, portal_hit=-20, vortex=-29, suck=-21, close=-18, tunnel=-23, tunnel_hits=-22, exit=-20,
          arrive=-23, space=-36, bh_form=-18, bh_bed=-30, strange_ign=-24, strange=-25, wipe=-16)


def build_sfx(B):
    bus = Bus(['bath', 'studio', 'tunnel', 'space'])
    panR = track_pan('R')
    panL = track_pan('L')

    x, t0 = sfx_claude(B)
    bus.add(x, t0, 0, 0, 0.3, name='claude', lufs=LV['claude'])

    # removals: magic eraser (trace -> burn -> embers), pitched up a step each time
    for key, base, pan, size, sd, off in (('clutter', hz('A5'), 0.5, 1.0, 100, 0), ('flush', hz('C#6'), 0.12, 0.55, 120, -2),
                                         ('shower', hz('E6'), -0.6, 0.9, 140, 0)):
        x, t0 = sfx_eraser(B['er_' + key], base, size, sd)
        bus.add(x, t0, off, pan, 0.35, name='eraser_' + key, lufs=LV['eraser'])

    x, t0 = sfx_studio(B)
    bus.add(x, t0, 0, 0, 0.35, space='studio', name='studio', lufs=LV['studio'])
    x, t0 = sfx_studio_tone(B)
    bus.add(x, t0, 0, 0, 0.0, name='studio_tone', lufs=LV['studio_tone'], kind='bed')

    x, t0 = sfx_suit_bed(B)
    bus.add(x, t0, 0, 0, 0.3, name='suit_bed', lufs=LV['suit_bed'], kind='bed')
    x, t0 = sfx_suit_hits(B)
    bus.add(x, t0, 0, 0, 0.3, name='suit_hits', lufs=LV['suit_hit'])

    a = B['antic']
    bus.add(sfx_anticipation(B['fire'] - a, panR, panL, a), a, 0, 0, 0.25, name='anticipation', lufs=LV['antic'], kind='bed')

    # fire ignition + palm bed (fades while the orbs leave the palms)
    tf = B['fire']
    bus.add(sfx_fire_ignite(), tf - 0.16, 0, panR(tf), 0.3, name='fire_ignite', lufs=LV['fire_ignite'])
    s = tf + 0.1
    T = B['unr0'] - s
    fb = fire_bed(T, panR, s)
    fb *= env_pts(tvec(len(fb)) + s, [(s, 0), (s + 0.25, 1), (B['push0'], 1), (B['push1'], 0.7), (B['fl0'], 0.65),
                                      (B['unr0'] - 0.04, 0), (99, 0)])[:, None]
    bus.add(fb, s, 0, 0, 0.2, name='fire_bed', lufs=LV['fire_bed'], kind='bed')

    # ice formation + bed
    x, t0 = sfx_ice_form(B)
    bus.add(x, t0, 0, panL(B['ice']), 0.35, name='ice_form', lufs=LV['ice_form'])
    s = B['ice'] + 0.55
    T = B['unr0'] - s
    ib = ice_bed(T, panL, s)
    ib *= env_pts(tvec(len(ib)) + s, [(s, 0), (s + 0.3, 1), (B['push0'], 1), (B['push1'], 0.7), (B['fl0'], 0.65),
                                      (B['unr0'] - 0.04, 0), (99, 0)])[:, None]
    bus.add(ib, s, 0, 0, 0.3, name='ice_bed', lufs=LV['ice_bed'], kind='bed')

    # fusion: converge into the ring, then the helix
    x, t0 = sfx_fusion_converge(B, panR, panL)
    bus.add(x, t0, 0, 0, 0.3, name='fusion_converge', lufs=LV['converge'])
    x, t0 = sfx_fusion(B)
    bus.add(x, t0, 0, 0, 0.3, name='fusion', lufs=LV['fusion'], kind='bed')

    # release + charge (inhale, suck, silence) + boom
    x, t0 = sfx_release_charge(B)
    bus.add(x, t0, 0, 0, 0.3, name='charge', lufs=LV['charge'])
    x, t0 = sfx_boom(B)
    bus.add(x, t0, 0, 0, 0.45, name='finale_boom', lufs=LV['boom'])

    # ceiling portal: glimmer + ignition hit, vortex bed until the collapse
    x, t0 = sfx_portal_open_hit(B)
    bus.add(x, t0, 0, 0, 0.35, space='studio', name='portal_open', lufs=LV['portal_hit'])
    x, t0 = portal_vortex_bed(B)
    bus.add(x, t0, 0, 0, 0.35, space='studio', name='portal_vortex', lufs=LV['vortex'], kind='bed')
    a, b = EV['suck']
    bus.add(sfx_suck(b - a), a, 0, 0, 0.3, space='studio', name='suck', lufs=LV['suck'])
    x, t0 = sfx_portal_close(B)
    bus.add(x, t0, 0, 0, 0.35, space='studio', name='portal_close', lufs=LV['close'])
    # tunnel
    x, t0 = sfx_tunnel_bed(B)
    bus.add(x, t0, 0, 0, 0.3, space='tunnel', name='tunnel_bed', lufs=LV['tunnel'], kind='bed')
    x, t0 = sfx_tunnel_hits(B)
    bus.add(x, t0, 0, 0, 0.3, space='tunnel', name='tunnel_hits', lufs=LV['tunnel_hits'])
    bus.add(sfx_exit_flash(), B['tun1'], 0, 0, 0.5, space='space', name='exit_flash', lufs=LV['exit'])
    x, t0 = sfx_arrive(B)
    bus.add(x, t0, 0, 0, 0.45, space='space', name='arrive', lufs=LV['arrive'])
    # space ambience until the wipe
    t0, t1 = EV['space'][0], EV['strange_wipe'][1]
    sa = space_amb(t1 - t0)
    sa *= env_pts(tvec(len(sa)), [(0, 0), (1.5, 1), (t1 - t0 - 1.0, 1), (t1 - t0, 0)])[:, None]
    bus.add(sa, t0, 0, 0, 0.35, space='space', name='space_amb', lufs=LV['space'], kind='bed')
    # black hole
    x, t0 = sfx_blackhole_form(B)
    bus.add(x, t0, 0, 0, 0.4, space='space', name='blackhole_form', lufs=LV['bh_form'])
    s = B['bh_ign'] + 0.15
    t1 = B['sg_tw0'] + 0.4
    T = t1 - s
    bh = blackhole_bed(T, pan=B['pan_bh'])
    bh *= env_pts(tvec(len(bh)) + s, [(s, 0), (s + 0.8, 1), (B['sg_pre'], 1), (t1 - 0.4, 0.45), (t1, 0), (99, 0)])[:, None]
    bus.add(bh, s, 0, 0, 0.3, space='space', name='blackhole_bed', lufs=LV['bh_bed'], kind='bed')
    # Doctor Strange: ignition hit, ring bed (curves from strange.py), wipe
    sw = B['sg_switch']
    pan_s = lambda tt: np.where(np.asarray(tt) < sw, np.clip(panR(tt), -0.6, 0.6) * 0.8,
                                np.clip(panR(tt), -0.6, 0.6) * 0.8 * (1 - smooth01((np.asarray(tt) - sw) / 1.0))
                                + B['pan_cf'] * smooth01((np.asarray(tt) - sw) / 1.0))
    x, t0 = sfx_strange_ign(B, float(pan_s(B['sg_ign'])))
    bus.add(x, t0, 0, 0, 0.3, space='space', name='strange_ign', lufs=LV['strange_ign'])
    x, t0 = sfx_strange_bed(B, pan_s)
    bus.add(x, t0, 0, 0, 0.3, space='space', name='strange_bed', lufs=LV['strange'], kind='bed')
    x, t0 = sfx_wipe(B)
    bus.add(x, t0, 0, 0, 0.3, space='space', name='strange_wipe', lufs=LV['wipe'])
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
            out += pan2(y * (0.8 + 0.2 * np.sin(TAU * r.uniform(0.05, 0.2) * tvec(n) + r.uniform(0, TAU))), p)
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

    def put(self, x, t0, gain_db=0.0, rev=0.35, buf=None, fin=0.01, fout=0.05, lufs=None, peak=False):
        """lufs: normalise integrated loudness of the element (or max momentary if peak=True)."""
        x = np.asarray(x, np.float64)
        if x.ndim == 1:
            x = pan2(x, 0.0)
        if lufs is not None:
            ref = max_momentary(x) if peak else integrated(x if len(x) > ns(0.5) else np.pad(x, ((0, ns(0.5)), (0, 0))))
            gain_db = gain_db + lufs - ref
        x = fade(x, fin, fout) * undb(gain_db)
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


# music loudness targets (integrated over each element, internal scale; peak=True -> max momentary)
MV = dict(bath_pad=-33, bath_pluck=-34, studio=-32, suit=-38, ost=-33, strings=-33, hit=-28, ticks=-41,
          calm=-34, drone=-41, portal1=-32, portal2=-31, tunnel=-29, space=-29, chime=-39, bh=-31,
          strange=-30, final=-28, bell=-33)


def build_music(B):
    M = Music()
    # ------------------------------------------------ A. bathroom: light / curious (0 .. lamp flicker)
    bath = np.zeros((N, 2))
    chords = [(0.0, ['D3', 'A3', 'E4', 'F#4']), (3.25, ['G2', 'D3', 'F#3', 'B3', 'C#4']),
              (6.5, ['B2', 'F#3', 'A3', 'D4', 'E4']), (9.75, ['G2', 'D3', 'A3', 'B3', 'E4'])]
    for k, (t0, ch) in enumerate(chords):
        t1 = chords[k + 1][0] if k + 1 < len(chords) else 13.6
        d = t1 - t0 + 1.2
        x = pad_tone(ch, d, 3000 + k, bright=2600, amp=1.0) * chord_env(d, 0.9 if k else 1.6, 1.2)
        x += pad_air([c[:-1] + str(int(c[-1]) + 1) for c in ch], d, 3100 + k, lp=6000, amp=0.5) * chord_env(d, 1.0, 1.2)
        M.put(x, t0 - (0.0 if k == 0 else 0.6), 0, buf=bath, lufs=MV['bath_pad'])
    plk = np.zeros((N, 2))
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
            M.put(pan2(pluck(f, 1.8, 0.8, 3300 + step) * v, r.uniform(-0.35, 0.35)), tt + r.normal(0, 0.006),
                  3 * smooth01(tt / 3), buf=plk, fin=0.0005, fout=0.2)
        tt += e8
        step += 1
    M.put(plk, 0, 0, buf=bath, fin=0, fout=0, lufs=MV['bath_pluck'])
    bath = tape_stop(bath, B['st_flicker'][0], 0.4)          # the music dies with the lamp
    M.bus += bath
    M.send += bath * 0.45

    # ------------------------------------------------ B. studio: dark (front launch .. anticipation)
    a = B['st_core'] - 0.02
    d = B['antic'] + 0.4 - a
    x = pad_air(['D2', 'A2', 'D3', 'F3', 'A3'], d, 3400, lp=900, amp=1.0) * chord_env(d, 2.0, 0.8)
    x += pad_tone(['D2', 'A2', 'F3'], d, 3401, bright=600, amp=0.8) * chord_env(d, 2.5, 0.8)
    dr = sat(sine(hz('D1'), ns(d)) + 0.5 * sine(hz('D2'), ns(d)), 1.4)
    x += pan2(dr * 0.18, 0) * chord_env(d, 2.0, 0.8)
    M.put(x, a, 0, rev=0.3, lufs=MV['studio'])
    a, b = B['su_t0'], B['su_t1']
    d = b - a + 2.2
    x = pad_air(['E5', 'A5', 'C6'], d, 3410, lp=9000, bw=0.003, amp=0.8) * chord_env(d, 1.2, 1.2)
    M.put(x, a + 0.2, 0, rev=0.5, lufs=MV['suit'])

    # ------------------------------------------------ C. powers: tense / rising (19.7 .. 32.62)
    Bt = [0, 6, 13, 17, 22, 26]
    Tm = [B['antic'], B['fire'], B['ice'], EV['fusion'][0], B['rel'], B['boom']]
    btime = lambda b: np.interp(b, Bt, Tm)
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
    i0, i1 = ns(t0o - 0.2), ns(t1o + 0.3)
    seg = ost[i0:i1]
    seg = spec(seg, lambda ff, tt: g_lp(ff, np.interp(tt + (t0o - 0.2), [t0o, Tm[1], Tm[3], t1o], [250, 500, 1000, 2600]), 3),
               2048, 512)
    lvl = env_pts(tvec(len(seg)) + (t0o - 0.2), [(t0o, 0), (t0o + 1.5, 0.35), (Tm[1], 0.55), (Tm[3], 0.8), (t1o - 0.12, 1.0),
                                                (t1o - 0.1, 0)])
    seg *= lvl[:, None]
    seg *= undb(MV['ost'] - integrated(seg))
    M.bus[i0:i1] += seg
    M.send[i0:i1] += seg * 0.2
    del ost, seg
    sc = [(Tm[0], ['D3', 'F3', 'A3']), (Tm[1], ['A#2', 'F3', 'D4', 'C4']), (Tm[2], ['G2', 'D3', 'A#3', 'A4']),
          (Tm[3], ['A2', 'D3', 'E3', 'G3', 'E4']), (Tm[3] + 1.2, ['A2', 'C#3', 'E3', 'G3', 'C#4']),
          (Tm[4], ['D3', 'A3', 'D4', 'E4', 'A4'])]
    for k, (t0, ch) in enumerate(sc):
        t1 = sc[k + 1][0] if k + 1 < len(sc) else t1o - 0.1
        d = t1 - t0 + (0.6 if k + 1 < len(sc) else 0.0)
        x = pad_tone(ch, d, 3500 + k, bright=1200 + 350 * k, voices=3, harm=9, roll=1.1, amp=1.0)
        x *= chord_env(d, 0.6 if k else 2.0, 0.6 if k + 1 < len(sc) else 0.02)
        M.put(x, t0 - (0.3 if k else 0), 1.0 * k, rev=0.3, fout=0.01, lufs=MV['strings'])
    for t0, g in ((Tm[1], 0), (Tm[2], -1), (Tm[3], 0), (Tm[4], 1)):
        n = ns(1.5)
        tl = tvec(n)
        hit = sub_drop(1.5, 110, 48, 0.05, 0.35, 2.2) + butter(colored(n, int(t0 * 10), -1), 'low', 500, 2) * env_ar(tl, 0.002, 0.08) * 0.6
        M.put(pan2(hit, 0), t0, g, rev=0.25, fin=0.0005, lufs=MV['hit'], peak=True)
    tks = np.zeros((N, 2))
    bb = 17.0
    k = 0
    while bb < 25.92:
        tb = btime(bb)
        n = ns(0.05)
        tk = butter(colored(n, 3600 + k, 0.0), 'high', 6000, 2) * np.exp(-tvec(n) / 0.008)
        acc = 1.0 if k % 4 == 0 else 0.55
        g = -10 + 10 * smooth01((bb - 17) / 9)
        M.put(pan2(tk * acc, 0.3 * np.sin(k)), tb, g, buf=tks, fin=0.0003, fout=0.01)
        bb += 0.25 if bb < 22 else 0.125
        k += 1
    M.put(tks, 0, 0, rev=0.15, fin=0, fout=0, lufs=MV['ticks'])

    # ------------------------------------------------ D. calm after the boom (34.0 .. 43.6)
    sc = [(34.0, ['D3', 'A3', 'E4', 'F4'], 4.8), (38.4, ['A#2', 'F3', 'A3', 'D4'], 5.4)]
    for k, (t0, ch, d) in enumerate(sc):
        x = pad_tone(ch, d + 0.8, 3700 + k, bright=1100, amp=1.0) * chord_env(d + 0.8, 2.5 if k == 0 else 1.0, 1.0)
        x += pad_air(ch, d + 0.8, 3710 + k, lp=2500, amp=0.7) * chord_env(d + 0.8, 2.5 if k == 0 else 1.0, 1.0)
        M.put(x, t0, 0, rev=0.4, lufs=MV['calm'])
    d = 43.6 - 34.6
    dr = sat(sine(hz('D1'), ns(d)), 1.3) * chord_env(d, 3.0, 1.0)[:, 0]
    M.put(pan2(dr * 0.6, 0), 34.6, 0, rev=0.0, lufs=MV['drone'])

    # ------------------------------------------------ E. portal: mystery -> wonder (43.4 .. 50.55)
    a = 43.4
    d = B['po_open'] - a + 0.6
    x = pad_air(['A#2', 'D3', 'G3', 'A3', 'D4'], d, 3800, lp=3000, amp=1.0) * chord_env(d, 1.5, 0.6)
    x += pad_tone(['A#2', 'D3', 'G3', 'A3'], d, 3801, bright=1500, amp=0.7) * chord_env(d, 1.5, 0.6)
    M.put(x, a, 0, rev=0.45, lufs=MV['portal1'])
    a = B['po_open'] - 0.15
    b = B['col0']
    d = b - a
    ch = ['D#2', 'A#2', 'G3', 'D4', 'A4', 'D5']
    x = pad_air(ch, d, 3810, lp=5000, amp=1.0, formant=(750, 1150)) * chord_env(d, 1.2, 0.05)
    x += pad_tone(ch[:4], d, 3811, bright=1800, amp=0.8) * chord_env(d, 1.0, 0.05)
    lvl = env_pts(tvec(ns(d)), [(0, 0.7), (EV['suck'][0] - a, 0.85), (d, 1.0)])
    x *= lvl[:, None]
    ts = EV['suck'][0] - a
    tl = tvec(ns(d))
    speed = 1 + 0.26 * smooth01((tl - ts) / (d - ts)) ** 1.5
    x = resample_speed(np.pad(x, ((0, ns(1.0)), (0, 0))), speed)
    M.put(x, a, 0, rev=0.45, fout=0.03, lufs=MV['portal2'])

    # ------------------------------------------------ F. tunnel: Bb/D -> C/D -> D (51.0 .. 53.55)
    a, b = B['tun0'], B['tun1']
    mid = (a + b) / 2
    for k, (t0, t1, ch) in enumerate(((a, mid, ['D2', 'A#2', 'F3', 'D4', 'F4']), (mid, b, ['D2', 'C3', 'G3', 'E4', 'G4']))):
        d = t1 - t0 + 0.25
        x = pad_tone(ch, d, 3900 + k, bright=2500, voices=4, harm=9, roll=1.0, amp=1.0)
        x *= chord_env(d, 0.35, 0.25) * np.interp(tvec(ns(d)), [0, d], [0.7, 1.0])[:, None]
        M.put(x, t0, 1.0 * k, rev=0.35, lufs=MV['tunnel'])

    # ------------------------------------------------ G. space: vast / awe (53.55 .. 62.2)
    sc = [(EV['space'][0], 57.6, ['D2', 'A2', 'F#3', 'E4', 'A4', 'C#5']),
          (57.4, 60.0, ['D2', 'B2', 'F#3', 'C#4', 'D4', 'F#4']),
          (59.8, EV['blackhole'][0] + 0.3, ['D2', 'G2', 'D3', 'F#3', 'B3', 'C#4'])]
    for k, (t0, t1, ch) in enumerate(sc):
        d = t1 - t0 + 0.6
        x = pad_air(ch, d, 4000 + k, lp=7000, amp=1.2) * chord_env(d, 1.8 if k == 0 else 1.0, 0.8)
        x += pad_tone(ch, d, 4010 + k, bright=2400, voices=4, amp=0.9) * chord_env(d, 1.5 if k == 0 else 1.0, 0.8)
        x += pad_air([c[:-1] + str(int(c[-1]) + 2) for c in ch[2:]], d, 4020 + k, lp=12000, bw=0.003, amp=0.35) * chord_env(d, 2.0, 0.8)
        M.put(widen(x, 1.3), t0, 0, rev=0.6, lufs=MV['space'])
    r = rng(4100)
    penta = ['D5', 'E5', 'F#5', 'A5', 'B5', 'D6', 'E6', 'F#6']
    for tt in poisson_times(EV['strange'][0] - 54.6, lambda q: 0.55 + 0 * q, 4101) + 54.6:
        if B['bh_ign'] - 1.0 < tt < B['bh_ign'] + 1.3:
            continue
        nm = penta[int(r.integers(len(penta)))]
        b_ = bell(hz(nm), 3.0, ratio=r.choice([1.0, 2.0, 1.41]), index=r.uniform(0.6, 1.5), tau=1.1, tau_i=0.4, glass=0.15)
        M.put(pan2(b_, r.uniform(-0.7, 0.7)), tt, r.uniform(-3, 2), rev=0.9, fin=0.001, fout=0.3, lufs=MV['chime'], peak=True)

    # ------------------------------------------------ H. black hole: dread (62.0 .. 70.2)
    sc = [(EV['blackhole'][0], 65.6, ['D2', 'D#3', 'G3', 'A#3']), (65.4, 68.0, ['D2', 'G2', 'A#2', 'D3', 'G3']),
          (67.8, EV['strange'][0] + 0.2, ['D2', 'A#2', 'F3', 'A3', 'D4'])]
    for k, (t0, t1, ch) in enumerate(sc):
        d = t1 - t0 + 0.6
        x = pad_air(ch, d, 4200 + k, lp=2200 + 600 * k, amp=1.1) * chord_env(d, 1.2, 0.8)
        x += pad_tone(ch, d, 4210 + k, bright=900 + 300 * k, amp=0.9) * chord_env(d, 1.2, 0.8)
        M.put(widen(x, 1.3), t0, 0, rev=0.55, lufs=MV['bh'])
    d = EV['strange'][0] + 0.5 - EV['blackhole'][0]
    dr = sat(sine(hz('D1'), ns(d)) + 0.4 * sine(hz('D#1') * 1.0, ns(d)) * 0.5, 1.4) * chord_env(d, 1.0, 0.6)[:, 0]
    M.put(pan2(dr, 0), EV['blackhole'][0], 0, rev=0.2, lufs=MV['drone'] + 2)

    # ------------------------------------------------ I. Strange: mystic -> warm D major ending
    a = EV['strange'][0]
    w0 = B['sg_tw0']
    sc = [(a, 72.0, ['D2', 'A#2', 'F3', 'C4', 'E4']), (71.8, w0 + 0.3, ['A1', 'A2', 'D3', 'E3', 'G3', 'D4'])]
    for k, (t0, t1, ch) in enumerate(sc):
        d = t1 - t0 + 0.4
        x = pad_tone(ch, d, 4300 + k, bright=1600 + 900 * k, voices=3, harm=9, roll=1.2, amp=1.0)
        x += pad_air(ch, d, 4310 + k, lp=4000, amp=0.8)
        x *= chord_env(d, 0.8, 0.4) * np.interp(tvec(ns(d)), [0, d], [0.8, 1.1])[:, None]
        M.put(x, t0, 1.0 * k, rev=0.4, lufs=MV['strange'])
    t0 = B['sg_rel'] - 0.03
    d = N / SR - t0
    ch = ['D2', 'A2', 'D3', 'F#3', 'A3', 'E4', 'F#4']
    x = pad_tone(ch, d, 4400, bright=1900, voices=4, amp=1.0) + pad_air(ch, d, 4401, lp=4500, amp=0.8)
    sw_ = B['sg_out'] - t0
    x *= np.minimum(smooth01(tvec(ns(d)) / max(sw_, 0.3)), smooth01((d - tvec(ns(d))) / 0.9))[:, None]
    M.put(widen(x, 1.3), t0, 0, rev=0.45, lufs=MV['final'])
    M.put(pan2(bell(hz('D6'), 1.5, 2.0, 1.0, 0.6, 0.2, glass=0.1), 0), B['sg_out'], 0, rev=0.6, fin=0.001, fout=0.2,
          lufs=MV['bell'], peak=True)
    return M


# =============================================================================================
# mix
# =============================================================================================
def black_hole_warp(x, B):
    """Gravitational pitch-sag of the music while the black hole grows (62.0 -> ignition), then a
    cross-fade back to the dry signal after the ignition."""
    a, ti = EV['blackhole'][0], B['bh_ign']
    L = max(ti - a, 0.3)
    t = tvec(N)
    dmax = 0.075
    u = np.clip((t - a) / L, 0, 1)
    dly = dmax * smooth01(u) * SR
    xf = smooth01((t - (ti + 0.05)) / 0.45)
    i0, i1 = ns(a - 0.05), ns(ti + 0.6)
    y = x.copy()
    for c in range(2):
        seg = x[:, c]
        wv = vdelay(seg[i0 - ns(0.2):i1], dly[i0 - ns(0.2):i1])[ns(0.2):]
        y[i0:i1, c] = wv * (1 - xf[i0:i1]) + seg[i0:i1] * xf[i0:i1]
    return y


def split4(x):
    """Complementary zero-phase split: <220, 220-1k, 1k-4k (voice presence), >4k."""
    a = butter(x, 'low', 220.0, 4, zero_phase=True)
    b = butter(x, 'low', 1000.0, 4, zero_phase=True)
    c = butter(x, 'low', 4000.0, 4, zero_phase=True)
    return [a.astype(np.float32), (b - a).astype(np.float32), (c - b).astype(np.float32), (x - c).astype(np.float32)]


def _align(lv, nf, off=10, fill=-70.0):
    out = np.full(nf, fill)
    k = min(len(lv), nf - off)
    out[off:off + k] = lv[:k]
    return out


def voice_keys(voice, vad):
    """Side-chain keys from the processed voice (10 ms frames)."""
    hop = 0.01
    nf = len(vad)
    vs = vad.astype(float)
    m = voice.mean(1)
    pres = butter(m, 'band', [1000.0, 4000.0], 2)
    lp = frame_db(pres, 480, 960)
    lp = np.pad(lp, (0, max(0, nf - len(lp))), constant_values=-140)[:nf]
    ref = np.median(lp[vad])
    key = np.clip((lp - (ref - 16.0)) / 10.0, 0, 1)
    key = follow(maximum_filter1d(key, 3, origin=-1), 0.012, 0.12, hop)
    _, lbb = loud_curve(None, 0.2, hop, weighted=kweight(voice))
    _, lsp = loud_curve(None, 0.2, hop, weighted=butter(m, 'band', [250.0, 4500.0], 2)[:, None])
    lbb, lsp = _align(lbb, nf), _align(lsp, nf)
    rb, rs = np.median(lbb[vs > 0]), np.median(lsp[vs > 0])
    return dict(vs=vs, pres=key,
                act=follow(maximum_filter1d(vs, 15, origin=-7), 0.05, 0.35, hop),        # look-ahead 140 ms
                act_bed=follow(maximum_filter1d(vs, 25, origin=-10), 0.12, 0.6, hop),   # slow, no pumping
                lbb=np.maximum(lbb, rb - 12.0), lsp=np.maximum(lsp, rs - 12.0),         # don't chase word tails
                vmask=maximum_filter1d(vs, 15))


def duck(bg, K, depth, pres_dyn, tgt_bb, tgt_sp, bed=False, protect=None, iters=3):
    """4-band side-chain ducking of bg under the voice.
    depth: static reduction per band (dB) at full voice activity; pres_dyn: extra 1-4 kHz reduction
    keyed on the voice's own presence envelope (dynamic EQ); adaptive unmasking keeps the voice
    >= tgt_sp dB over bg in the speech band (250-4500 Hz) and >= tgt_bb dB broadband (K-weighted,
    200 ms). bed=True: 120 ms attack / 600 ms release, low-band adaptive cut capped at 6 dB.
    protect: (4, n_frames) multipliers 0..1 of the reduction (hit windows)."""
    hop = 0.01
    nf = len(K['vs'])
    act = K['act_bed'] if bed else K['act']
    P = np.ones((4, nf)) if protect is None else protect
    red = np.zeros((nf, 4))
    for b in range(4):
        red[:, b] = depth[b] * act * P[b]
    red[:, 2] += pres_dyn * K['pres'] * P[2]
    bands = split4(bg)
    w_sp = np.array([0.45, 1.0, 1.0, 0.3])
    w_bb = np.array([0.8, 0.8, 0.8, 1.0])
    att, rel = (0.12, 0.6) if bed else (0.045, 0.28)
    la = (25, -8) if bed else (15, -5)
    adapt = np.zeros((nf, 4))
    for it in range(iters):
        g = red + adapt
        cur = sum(bands[b] * undb(-to_samples(g[:, b], 480)).astype(np.float32)[:, None] for b in range(4))
        _, lb = loud_curve(None, 0.2, hop, weighted=kweight(cur))
        _, ls = loud_curve(None, 0.2, hop, weighted=butter(cur.mean(1), 'band', [250.0, 4500.0], 2)[:, None])
        del cur
        lb, ls = _align(lb, nf), _align(ls, nf)
        needs = []
        for lvl, ref, tgt in ((lb, K['lbb'], tgt_bb), (ls, K['lsp'], tgt_sp)):
            nd = np.clip(lvl - (ref - tgt), 0, 12) * K['vmask']
            nd = maximum_filter1d(nd, la[0], origin=la[1])
            needs.append(follow(nd, att, rel, hop))
        adapt += needs[1][:, None] * w_sp[None, :] + needs[0][:, None] * w_bb[None, :]
        adapt *= P.T
        adapt = np.minimum(adapt, 15.0)
        if bed:
            adapt[:, 0] = np.minimum(adapt[:, 0], 6.0)
    red += adapt
    red = np.minimum(red, np.array([15.0, 24.0, 24.0, 18.0])[None, :])
    out = sum(bands[b].astype(np.float64) * undb(-to_samples(red[:, b], 480))[:, None] for b in range(4))
    return out, red


def protect_windows(B, nf):
    """Hit protection (fraction of the ducking removed per band) for the first ~100 ms of the hits that
    land ON their cue words; the low band can be held longer (sub + rumble of the boom)."""
    tf = np.arange(nf) * 0.01
    wm = np.zeros(nf)
    wl = np.zeros(nf)
    ent = [(B['st_core'], 0.3, 0.12, None), (B['st_lamp'], 0.35, 0.10, None),
           (B['su_ign'], 0.35, 0.10, None), (B['fire'], 0.5, 0.12, 0.3), (B['ice_lock'], 0.45, 0.12, None),
           (B['fl1'], 0.3, 0.10, None), (B['po_open'], 0.4, 0.12, 0.3), (B['bh_ign'], 0.4, 0.10, 0.3),
           (B['sg_ign'], 0.3, 0.10, None), (B['boom'], 0.85, 0.12, 0.9)]
    for k in ('clutter', 'flush', 'shower'):
        tr0, trd, b0, b1, pop = B['er_' + k]
        ent += [(tr0, 0.4, 0.12, None), (b0, 0.4, 0.12, None)]
    for th, s, hold, lo_hold in ent:
        wm = np.maximum(wm, s * env_pts(tf, [(0, 0), (th - 0.06, 0), (th - 0.02, 1), (th + hold, 1), (th + hold + 0.1, 0), (999, 0)]))
        lh = hold if lo_hold is None else lo_hold
        wl = np.maximum(wl, s * env_pts(tf, [(0, 0), (th - 0.06, 0), (th - 0.02, 1), (th + lh, 1), (th + lh + 0.25, 0), (999, 0)]))
    return np.stack([1 - np.minimum(1, wl * 1.1), 1 - wm, 1 - wm * 0.9, 1 - wm * 0.8])


def tp_limit(x, ceil_db=-1.2, look_ms=1.5, rel_ms=80.0):
    n = len(x)
    pk = os_peak(x).astype(np.float64)
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


def _mem():
    try:
        import resource
        return f'{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f}MB'
    except Exception:
        return '?'


def build(report=True):
    T0 = time.time()
    os.makedirs(STEMS, exist_ok=True)
    print('[sound] beats ...', flush=True)
    B = load_beats()
    fb = sorted(k for k, v in BEAT_SRC.items() if v == 'fallback')
    print(f'[sound]   {len(BEAT_SRC)} beat entries, fallbacks: {fb or "none"}', flush=True)
    print('[sound] voice ...', flush=True)
    raw = load_voice()
    voice, vad, vinfo = process_voice(raw)
    del raw
    print(f'[sound]   voice processed ({time.time() - T0:.1f}s, peak {_mem()}), tamed {vinfo["tamed_transients"]}', flush=True)

    print('[sound] sfx ...', flush=True)
    sb = build_sfx(B)
    irs = {k: make_ir(*SPACE_RT[k][:7], seed=50 + j, flutter=SPACE_RT[k][7]) for j, k in enumerate(SPACE_RT)}
    ret = dict(bath=0.55, studio=0.6, tunnel=0.7, space=0.75)
    sx = sb.render(irs, ret)
    log = sb.log
    del sb
    print(f'[sound]   sfx done ({time.time() - T0:.1f}s, peak {_mem()})', flush=True)

    print('[sound] music ...', flush=True)
    M = build_music(B)
    music = M.bus + convolve(M.send, irs['hall']) * 0.55
    del M
    music = butter(music, 'high', 35, 2)
    music = black_hole_warp(music, B)
    # finale: dip on the plasma inhale, silence from the collapse, soft return after the boom
    tb = B['boom']
    gm = env_pts(tvec(N), [(0, 1), (B['pl_in0'] - 0.02, 1), (B['pl_in0'] + 0.1, 0.45), (B['pl_col'] - 0.005, 0.25),
                           (B['pl_col'] + 0.01, 0.0), (tb + 1.2, 0), (34.2, 1),
                           (B['sg_tw0'], 1), (B['sg_rel'] - 0.02, 0.5), (B['sg_rel'] + 0.03, 1), (99, 1)])
    music *= gm[:, None]
    music = mono_low(music)
    print(f'[sound]   music done ({time.time() - T0:.1f}s, peak {_mem()})', flush=True)

    vad_n = vad[:N // 480 + 1] if len(vad) >= N // 480 + 1 else np.pad(vad, (0, N // 480 + 1 - len(vad)))
    K = voice_keys(voice, vad_n)
    prot = protect_windows(B, len(vad_n))
    musd, _ = duck(music, K, (5.0, 8.0, 9.0, 6.0), 4.0, 19.0, 21.0, bed=True)
    del music
    hit = mono_low(sx['hit'])
    hitd, _ = duck(hit, K, (3.0, 5.0, 6.0, 3.0), 7.0, 12.5, 17.0, bed=False, protect=prot)
    del hit
    bed = mono_low(sx['bed'])
    bedd, _ = duck(bed, K, (4.0, 6.0, 7.0, 4.0), 5.0, 16.0, 19.0, bed=True)
    del bed, sx
    sfxd = hitd + bedd
    bg = musd + sfxd
    print(f'[sound]   ducking done ({time.time() - T0:.1f}s, peak {_mem()})', flush=True)

    mix = voice + bg
    g = undb(-14.0 - integrated(mix))
    for it in range(4):
        lim, gs = tp_limit(mix * g, -1.5)
        li = integrated(lim)
        if abs(li + 14.0) < 0.1:
            break
        g *= undb(-14.0 - li)
    tp = true_peak(lim)
    lim *= np.minimum(1, (N - 1 - np.arange(N)) / ns(0.15))[:, None]
    G = (g * gs)[:, None]
    print(f'[sound] master: I={li:.2f} LUFS  TP={tp:.2f} dBTP  max GR={-db(gs.min()):.2f} dB  ({time.time() - T0:.1f}s, peak {_mem()})')
    grf = -db(gs[:len(gs) // 480 * 480].reshape(-1, 480).min(1))
    top = []
    for j in np.argsort(-grf):
        if grf[j] < 1.0 or len(top) >= 8:
            break
        if all(abs(j * 0.01 - q) > 0.4 for q, _ in top):
            top.append((round(j * 0.01, 2), round(float(grf[j]), 1)))
    print('[sound] limiter GR peaks (t, dB):', top)

    write_wav(os.path.join(AUDIO, 'mix.wav'), lim, pcm16=True)
    write_wav(os.path.join(AUDIO, 'sfx_only.wav'), bg * G, pcm16=True)
    write_wav(os.path.join(STEMS, 'voice.wav'), voice * G)
    write_wav(os.path.join(STEMS, 'sfx.wav'), sfxd * G)
    write_wav(os.path.join(STEMS, 'sfx_hits.wav'), hitd * G)
    write_wav(os.path.join(STEMS, 'sfx_beds.wav'), bedd * G)
    write_wav(os.path.join(STEMS, 'music.wav'), musd * G)
    beats = {k: (v if isinstance(v, (int, float, str)) else list(np.round(np.asarray(v, np.float64), 3)))
             for k, v in B.items() if k not in ('sg_t', 'sg_R', 'sg_E', 'sg_I', 'st_front_t', 'st_front_a', 'st_front_r')}
    info = dict(integrated_lufs=float(li), true_peak_dbtp=float(tp), limiter_max_gr_db=float(-db(gs.min())),
                master_gain_db=float(db(g)), samples=int(N), duration=N / SR, voice=vinfo, limiter_peaks=top,
                beats=beats, beat_sources=BEAT_SRC,
                events=[dict(name=nm, kind=kd, t0=round(a, 3), t1=round(b, 3)) for nm, kd, a, b in log])
    if report:
        del hitd, bedd, musd, sfxd, mix, irs
        info.update(make_report((voice * G).astype(np.float32), (bg * G).astype(np.float32), lim.astype(np.float32),
                                vad_n, B))
    with open(os.path.join(AUDIO, 'mix_info.json'), 'w') as fh:
        json.dump(info, fh, indent=1, default=float)
    print(f'[sound] done in {time.time() - T0:.1f}s (peak {_mem()})')
    return info


# =============================================================================================
# report: plots + measurements
# =============================================================================================
def ev_marks():
    return [
        (EV['clean_clutter'][0], 'clutter'), (EV['clean_flush'][0], 'flush'), (EV['clean_shower'][0], 'shower'),
        (EV['studio'][0], 'studio'), (EV['suit'][0], 'suit'), (EV['suit'][1], 'lock'), (EV['anticipation'][0], 'antic'),
        (EV['fire_on'], 'FIRE'), (EV['ice_on'], 'ICE'), (EV['fusion'][0], 'fusion'), (EV['release'], 'release'),
        (EV['finale_boom'], 'BOOM'), (EV['ceil_portal'][0], 'portal'), (EV['suck'][0], 'suck'),
        (EV['ceil_portal_close'][0], 'close'), (EV['tunnel'][0], 'tunnel'), (EV['arrive'][0], 'arrive/space'),
        (EV['blackhole'][0], 'blackhole'), (EV['strange'][0], 'strange'), (EV['strange_wipe'][0], 'wipe'),
        (EV['strange_wipe'][1], 'raw')]


def visual_beats(B):
    vb = []
    for k in ('clutter', 'flush', 'shower'):
        tr0, trd, b0, b1, pop = B['er_' + k]
        vb += [(tr0, 'v:trace'), (b0, 'v:burn')]
    vb += [(t, 'v:flicker') for t in B['st_flicker']]
    vb += [(B['st_core'], 'v:front'), (B['st_lamp'], 'v:lamp pop'), (B['st_tw1'], 'v:front out'),
           (B['st_rim_l'], 'v:light L'), (B['st_rim_r'], 'v:light R'), (B['su_ign'], 'v:suit ign'), (B['su_t1'], 'v:settle'),
           (B['fire'], 'v:fire ign'), (B['ice_sh0'], 'v:ice shards'), (B['ice_lock'], 'v:ice lock'), (B['fl0'], 'v:orbs fly'),
           (B['fl1'], 'v:ring entry'), (B['pl_born'], 'v:plasma'), (B['pl_in0'], 'v:inhale'), (B['pl_col'], 'v:collapse'),
           (B['boom'], 'v:BOOM'), (B['po_open'], 'v:portal opens'), (B['col_flash'], 'v:collapse FLASH'), (B['jolt'], 'v:jolt'),
           (B['tun1'], 'v:exit flash'), (B['land'], 'v:landing'), (B['arr_pop'], 'v:pop'), (B['bh_ign'], 'v:BH IGNITE'),
           (B['sg_ign'], 'v:spark ign'), (B['sg_rel'], 'v:wipe burst'), (B['sg_edge'], 'v:rim at edges'), (B['sg_out'], 'v:rim out')]
    return vb


def make_report(voice, bg, mix, vad, B):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    hop = 0.01
    tcv, lv = loud_curve(voice, 0.4, hop)
    tcb, lb = loud_curve(bg, 0.4, hop)
    tcm, lm = loud_curve(mix, 0.4, hop)
    _, lst = loud_curve(mix, 3.0, hop)
    tst = np.arange(len(lst)) * hop + 1.5
    vf = np.convolve(vad.astype(float), np.ones(40) / 40, 'valid')
    k = min(len(vf), len(lv))
    sp = vf[:k] >= 0.7
    ratio = (lv[:k] - lb[:k])
    rs = ratio[sp]
    vb = butter(voice.mean(1), 'band', [300, 5000], 4, zero_phase=True)
    bb_ = butter(bg.mean(1), 'band', [300, 5000], 4, zero_phase=True)
    _, lvb = loud_curve(vb, 0.4, hop)
    _, lbb = loud_curve(bb_, 0.4, hop)
    rsb = (lvb[:k] - lbb[:k])[sp]
    stats = dict(vbr_median_db=float(np.median(rs)), vbr_p10_db=float(np.percentile(rs, 10)),
                 vbr_band300_5k_median_db=float(np.median(rsb)), vbr_band300_5k_p10_db=float(np.percentile(rsb, 10)),
                 vbr_band300_5k_frac_ge12=float((rsb >= 12).mean()),
                 vbr_p05_db=float(np.percentile(rs, 5)), vbr_frac_ge12=float((rs >= 12).mean()),
                 bg_peak_momentary_lufs=float(lb.max()))
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

    def spectro(x, t0, t1, nfft=4096, hop_s=None):
        i0, i1 = ns(t0), ns(t1)
        hop_s = hop_s or max(256, int((i1 - i0) / 2400))
        m = x[i0:i1].mean(1).astype(np.float32)
        f, tt, Z = ss.stft(m, SR, nperseg=nfft, noverlap=nfft - hop_s)
        P = 20 * np.log10(np.abs(Z) + 1e-9)
        fl = np.geomspace(30, 20000, 360)
        Pl = np.stack([np.interp(fl, f, P[:, j]) for j in range(P.shape[1])], 1)
        return tt + t0, fl, Pl

    paths = []
    I_mix, TP_mix = integrated(mix), true_peak(mix)
    VB = visual_beats(B)
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
        ax[0].plot(tt, mix[i0:i1:step].mean(1), color='#bbb', lw=0.4)
        ax[0].plot(tt, bg[i0:i1:step].mean(1), color='#ff9a3c', lw=0.4, alpha=0.8)
        ax[0].plot(tt, voice[i0:i1:step].mean(1), color='#4fc3f7', lw=0.4, alpha=0.8)
        ax[0].set_ylim(-1, 1)
        ax[0].set_ylabel('wave\nmix/bg/voice', color='#ccc')
        for row, x, lab in ((1, bg, 'BG (sfx+music)\nspectrogram Hz'), (2, mix, 'MIX\nspectrogram Hz')):
            st_, fl, P = spectro(x, t0, t1)
            vmax = P.max()
            ax[row].pcolormesh(st_, fl, P, shading='auto', cmap='magma', vmin=vmax - 90, vmax=vmax)
            ax[row].set_yscale('log')
            ax[row].set_ylabel(lab, color='#ccc')
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
        vt = np.arange(len(vad)) * 0.01
        ax[3].fill_between(vt, -60, 0, where=vad, color='#4fc3f7', alpha=0.07, step='mid')
        ax[4].fill_between(vt, 0, 40, where=vad, color='#4fc3f7', alpha=0.07, step='mid')
        for te, lab in ev_marks():
            if t0 <= te <= t1:
                for a in ax:
                    a.axvline(te, color='#00e5ff', lw=0.7, alpha=0.5)
                ax[0].text(te, 0.92, lab, color='#00e5ff', fontsize=9, va='top', ha='left')
        if tag != 'full':
            for te, lab in VB:
                if t0 <= te <= t1:
                    for a in ax[1:]:
                        a.axvline(te, color='#ff4fd8', lw=0.8, ls='--', alpha=0.6)
                    ax[1].text(te, 15000, lab, color='#ff4fd8', fontsize=8, va='top', ha='left', rotation=90)
        ax[4].set_xlim(t0, t1)
        ax[4].set_xlabel('time (s)   cyan = config.EV   magenta dashed = visual beat read from the module code', color='#ccc')
        fig.suptitle(f'sound report [{tag}]  I={I_mix:.1f} LUFS  TP={TP_mix:.1f} dBTP  '
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
    build(report=not a.no_report)
    print('[sound] ffmpeg ebur128 check:', ffmpeg_check(os.path.join(AUDIO, 'mix.wav')))


if __name__ == '__main__':
    main()
