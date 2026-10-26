"""Per-frame asset access. All images returned as float32 LINEAR light, (H, W, C), row 0 = top,
at output resolution 1080x1440 unless stated otherwise. Coordinates are output pixels
(x = column, y = row), i.e. source pixels * 3.

Hi-res matte / SR frames are produced by the preprocessing pipeline; while they are still
being generated, the loaders transparently fall back to upscaled low-res versions.
"""
import os
from functools import lru_cache
import numpy as np
import cv2
from . import config as C


# ---------------------------------------------------------------- colour
def srgb_to_lin(x):
    x = np.asarray(x, np.float32)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def lin_to_srgb(x):
    x = np.clip(np.asarray(x, np.float32), 0, None)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055).astype(np.float32)


_LUT = srgb_to_lin(np.arange(256) / 255.0).astype(np.float32)


def u8_to_lin(img_u8):
    """uint8 sRGB BGR/RGB image -> float32 linear (fast LUT)."""
    return _LUT[img_u8]


def _read_rgb(path):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        return None
    return im[:, :, ::-1]


# ---------------------------------------------------------------- frames
@lru_cache(maxsize=8)
def frame_u8(i):
    """Super-resolved source frame (RGB uint8, 1440x1080)."""
    p = f'{C.FRAMES_HYB}/{i:04d}.jpg'
    im = _read_rgb(p) if os.path.exists(p) else None
    if im is None:
        im = _read_rgb(f'{C.FRAMES_SRC}/{i:04d}.png')
        im = cv2.resize(im, (C.W, C.H), interpolation=cv2.INTER_LANCZOS4)
    return np.ascontiguousarray(im)


def frame(i):
    """Linear float RGB of the (super-resolved) original frame."""
    return u8_to_lin(frame_u8(i))


@lru_cache(maxsize=8)
def alpha(i):
    """Person matte, float32 (1440,1080) in [0,1]."""
    p = f'{C.ALPHA_HI}/{i:04d}.png'
    if os.path.exists(p):
        a = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if a is not None:
            return a.astype(np.float32) / 255.0
    a = cv2.imread(f'{C.ALPHA_LO}/{i:04d}.png', cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    return cv2.resize(a, (C.W, C.H), interpolation=cv2.INTER_CUBIC).clip(0, 1)


@lru_cache(maxsize=8)
def fgr(i):
    """Person foreground colour with background spill removed (linear RGB). Use with alpha()."""
    p = f'{C.FGR_HI}/{i:04d}.jpg'
    if os.path.exists(p):
        im = _read_rgb(p)
        if im is not None:
            return u8_to_lin(im)
    return frame(i)


@lru_cache(maxsize=8)
def plate(k):
    """Clean background plates (linear RGB 1440x1080):
    0 = original bathroom without the person
    1 = + sink/towel/bottles/mirror/holder/outlet/toilet removed   ("limpa o fundo aqui")
    2 = + beige flush plate removed                                ("descarga")
    3 = + shower panel/head/hose/valve removed                     ("chuveiro")
    """
    return u8_to_lin(_read_rgb(f'{C.PLATES}/plate{k}.png'))


@lru_cache(maxsize=8)
def plate_mask(name):
    """Object masks (float 0..1, 1440x1080) for the removals:
    'clutter', 'flush', 'shower' (the object itself), 'shower_region' (object + shadow strip
    that changes between plate2 and plate3)."""
    return cv2.imread(f'{C.PLATES}/mask_{name}.png', cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0


# ---------------------------------------------------------------- tracking
@lru_cache(maxsize=1)
def _track():
    return dict(np.load(C.TRACK))


def pose(i):
    """MediaPipe pose (33, 4): x, y (output px), z, visibility. Smoothed.
    Key indices: 0 nose, 11 L shoulder, 12 R shoulder, 13 L elbow, 14 R elbow,
    15 L wrist, 16 R wrist, 23/24 hips. 'R' = the person's right = FRAME-LEFT side."""
    return _track()['pose'][i]


def hand(i, side):
    """21 hand landmarks (x, y, z) in output px for side 'R' or 'L', or None.
    R = person's right hand (appears on frame-left) = FIRE hand.
    L = person's left hand (appears on frame-right) = ICE hand.
    Short detection gaps are interpolated; may still be None when the hand is out of view."""
    h = _track()['hand' + side][i]
    if np.isnan(h[0, 0]):
        return None
    return h


def hand_detected(i, side):
    return bool(_track()['hand%s_det' % side][i])


def palm(i, side):
    """Palm centre (x, y) in output px. Falls back to pose wrist/index/pinky when the hand
    landmarks are missing. Always returns a point (may be outside the frame)."""
    h = hand(i, side)
    if h is not None:
        return h[[0, 5, 9, 13, 17], :2].mean(0)
    p = pose(i)
    idx = (16, 18, 20, 22) if side == 'R' else (15, 17, 19, 21)
    return p[list(idx), :2].mean(0)


def palm_normal_up(i, side):
    """Rough 2D 'up' direction out of the palm (unit vector, image coords): from wrist toward
    middle-finger MCP. Useful to place an orb slightly above the palm."""
    h = hand(i, side)
    if h is None:
        return np.array([0.0, -1.0], np.float32)
    v = h[9, :2] - h[0, :2]
    n = np.linalg.norm(v) + 1e-6
    return (v / n).astype(np.float32)


def hand_size(i, side):
    """Approx hand length in px (wrist->middle fingertip), fallback 110."""
    h = hand(i, side)
    if h is None:
        return 110.0
    return float(np.linalg.norm(h[12, :2] - h[0, :2]) + 1e-3)


def face(i):
    """478 face-mesh points (x, y) in output px (gap-filled)."""
    return _track()['face'][i]


def face_oval(i):
    """36-point face outline polygon (x, y), output px."""
    t = _track()
    return t['face'][i][t['face_oval_idx']]


@lru_cache(maxsize=4096)
def head_box(i):
    """(x0, y0, x1, y1) head bounding box in output px. y0 = top of hair (from the matte),
    y1 = chin; x from the face oval widened to include hair/ears."""
    fo = face_oval(i)
    x0, x1 = fo[:, 0].min(), fo[:, 0].max()
    cx = (x0 + x1) / 2
    wdt = (x1 - x0)
    chin = fo[:, 1].max()
    brow = fo[:, 1].min()
    a = alpha(i)
    xa, xb = int(max(0, cx - wdt * 0.35)), int(min(C.W, cx + wdt * 0.35))
    ya = int(max(0, brow - wdt * 0.9))
    col = a[ya:int(max(ya + 1, brow)), xa:xb].mean(1)
    ys = np.where(col > 0.5)[0]
    top = float(ys.min() + ya) if len(ys) else brow - wdt * 0.4
    top = min(top, brow)
    return (float(cx - wdt * 0.62), top, float(cx + wdt * 0.62), float(chin))
