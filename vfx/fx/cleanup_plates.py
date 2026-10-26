"""Clean-plate access for the cleanup module, with a repaired plate 3.

plate3.png has a smeared vertical band on the far-left wall (x 0..140, y > 520) where the glass
screen / shower panel was removed: the horizontal grout lines stop at the x~144 vertical grout
and there is no tile structure. It is visible right when the shower vanishes, so we synthesise
that strip once (cached under DATA/cache/cleanup/):

  * the wall's horizontal grout lines converge to a vanishing point V (fitted from plate 3 in the
    intact tile column x 150..335). Every point gets a projective row coordinate
    eta = Vy + (y - Vy) * (Vx - 150) / (Vx - x)  (constant along each grout line).
  * target strip pixels copy tile texture from the intact column (mirrored + scaled, and shifted
    by one tile row to decorrelate it from the neighbouring tile) with the SAME fraction inside
    the tile row, so the grout lines continue exactly through the strip in perspective.
  * a low-frequency gain keeps the plate's original shading; the seam hides inside the vertical
    grout line at x~144 and just above the top grout line.
"""
import os
import numpy as np
import cv2
from .. import config as C
from .. import assets as A

CACHE = os.path.join(C.DATA, 'cache', 'cleanup')
VERSION = 6

_P = {}


def _fit_lines(S3):
    """Horizontal grout lines in the intact column: returns V=(Vx,Vy) and line y at x=150."""
    xs = np.arange(152, 334, 6)
    seeds = [558.0, 831.0, 1129.0, 1430.0]
    pts = {k: [] for k in range(len(seeds))}
    for x in xs:
        col = S3[:, x - 2:x + 3].mean(1)
        base = cv2.blur(col.reshape(-1, 1), (1, 41)).ravel()
        d = base - col
        for k, s in enumerate(seeds):
            # follow the line: predicted position from previous points
            if len(pts[k]) >= 2:
                (xa, ya), (xb, yb) = pts[k][0], pts[k][-1]
                yp = yb + (ya - yb) * 0 + (yb - ya) / max(xb - xa, 1) * (x - xb)
            elif pts[k]:
                yp = pts[k][-1][1]
            else:
                yp = s
            lo, hi = int(max(0, yp - 14)), int(min(len(d) - 1, yp + 14))
            if hi - lo < 5:
                continue
            j = lo + int(np.argmax(d[lo:hi]))
            if d[j] > 0.05:
                # sub-pixel refine
                if 0 < j < len(d) - 1:
                    a, b, c = d[j - 1], d[j], d[j + 1]
                    den = a - 2 * b + c
                    j = j + (0.5 * (a - c) / den if abs(den) > 1e-6 else 0.0)
                pts[k].append((float(x), float(j)))
    lines = []
    for k in range(len(seeds)):
        p = np.array(pts[k], np.float64)
        if len(p) < 6:
            lines.append(None)
            continue
        m, b = np.polyfit(p[:, 0], p[:, 1], 1)
        lines.append((m, b))
    good = [l for l in lines if l is not None]
    # vanishing point = least-squares intersection of the fitted lines
    Am = np.array([[-m, 1.0] for m, b in good])
    bm = np.array([b for m, b in good])
    V = np.linalg.lstsq(Am, bm, rcond=None)[0]       # (x, y)
    eta = [None if l is None else l[0] * 150.0 + l[1] for l in lines]
    return V, eta


def _vgrout(S3, xa, xb, y0=540, y1=1120):
    """Fit x = a + b*y of a (slightly tilted) vertical grout line searched in [xa, xb)."""
    ys, xs = [], []
    for y in range(y0, y1, 20):
        prof = S3[y:y + 20, xa:xb].mean(0)
        j = int(np.argmin(prof))
        if prof.max() - prof[j] > 0.15:
            ys.append(y + 10.0)
            xs.append(xa + j)
    b, a = np.polyfit(ys, xs, 1)
    return float(a), float(b)


def _repair_plate3():
    P3 = A.plate(3)
    S3 = A.lin_to_srgb(P3).mean(-1)
    V, eta = _fit_lines(S3)
    Vx, Vy = float(V[0]), float(V[1])
    eA, eB, eC, eD = eta
    if eD is None:
        eD = eC + (eC - eB) * 1.02
    ga, gb = _vgrout(S3, 120, 175)          # vertical grout at the strip's right edge (~x 144)
    ha, hb = _vgrout(S3, 320, 372)          # next vertical grout (~x 345): source of the band
    H, W = S3.shape
    x1 = 180
    ys, xs = np.mgrid[0:H, 0:x1].astype(np.float64)
    xg = ga + gb * ys                      # grout centre per row
    et = Vy + (ys - Vy) * (Vx - 150.0) / (Vx - xs)    # projective row coordinate of target px

    def eta_to_y(e, x):
        return Vy + (e - Vy) * (Vx - x) / (Vx - 150.0)

    # 1) strip interior: mirrored + scaled tile texture taken from a DIFFERENT tile row/column
    #    than the neighbouring tile, same fraction inside the tile row. Only tile INTERIORS are
    #    sampled (source grout excluded); the horizontal grout lines are then drawn
    #    synthetically with the cross-profile measured on the real grout (single, crisp lines).
    #    target row A-B <- row B-C (translated);  B-C <- row C-D (mirrored);
    #    C-D (and below) <- row B-C (mirrored);  above A <- above B (mirrored)
    g = 5.0
    # (only the column the vanishing point was fitted on is used: column 2's grout lines
    #  deviate a few px from the extrapolated lines)
    xsrc = np.where((et >= eA) & (et < eB), 178.0 + xs * 0.82, 326.0 - xs * 0.80)
    spec = [  # (target lo, target hi, source lo, source hi)
        (-1e9, eA, eB - (eA - (-1e9)), eB),
        (eA, eB, eB, eC),
        (eB, eC, eC, eD),
        (eC, eD + 2000.0, eB, eC + (eC - eB) * 2000.0 / (eD - eC)),
    ]
    es = np.zeros_like(et)
    for (t0, t1, s0, s1) in spec:
        m = (et >= t0) & (et < t1)
        if not m.any():
            continue
        if t0 < -1e8:      # above A: plain translation of the row above line B
            es[m] = np.minimum(eB - g, et[m] - eA + eB)
            continue
        if t1 > eD + 1000:  # last row: same scale as row C-D, continues past the frame bottom
            sc = (eC - eB) / (eD - eC)
            es[m] = np.clip(eB + (et[m] - eC) * sc, eB + g, eC - g)
            continue
        f = (et[m] - t0 - g) / (t1 - t0 - 2 * g)
        es[m] = np.clip(s0 + g + f * (s1 - s0 - 2 * g), s0 + g, s1 - g)
    ysrc = eta_to_y(es, xsrc)
    synth = cv2.remap(P3, xsrc.astype(np.float32), ysrc.astype(np.float32), cv2.INTER_CUBIC,
                      borderMode=cv2.BORDER_REFLECT)
    # synthetic horizontal grout lines B and C (measured cross-profile of lines B, C)
    prof = []
    for e in (eB, eC):
        rows = []
        for d in range(-8, 9):
            xx = np.arange(165, 325, 2)
            yy = np.round(eta_to_y(e, xx) + d).astype(int)
            rows.append(P3[yy, xx].mean(0))
        rows = np.array(rows)
        prof.append(rows / ((rows[0] + rows[-1]) / 2.0))
    prof = np.clip(np.mean(prof, 0), 0, 1.0)        # (17, 3) ratio by offset -8..8
    for e in (eB, eC):
        d = ys - eta_to_y(e, xs)
        near = np.abs(d) < 8
        for c in range(3):
            r = np.interp(d[near], np.arange(-8, 9), prof[:, c])
            synth[..., c][near] *= r
    # 2) grout band: a real vertical grout line copied from x~345 (same row coordinate)
    xb_src = xs - xg + (ha + hb * ys)
    yb_src = eta_to_y(et, xb_src)
    band = cv2.remap(P3, xb_src.astype(np.float32), yb_src.astype(np.float32), cv2.INTER_CUBIC,
                     borderMode=cv2.BORDER_REFLECT)
    dxg = xs - xg
    # plate 3 keeps a crisp grout of its own down to y~1000; below that it fades, so there we
    # use the copied grout band (narrow: the source tile has a highlight next to its grout)
    wb = np.clip((dxg + 3.0) / 1.5, 0, 1) * np.clip((ys - 950.0) / 100.0, 0, 1)
    synth = synth * (1 - wb[..., None]) + band * wb[..., None]
    synth = np.clip(synth, 0, None)
    # 3) low-frequency gain: keep the plate's shading, add tile structure (grout excluded)
    sl = A.lin_to_srgb(synth).mean(-1)
    loc = cv2.blur(sl, (31, 31))
    tile = (sl > loc - 0.05).astype(np.float32)
    lum_s = synth.mean(-1)
    num = cv2.GaussianBlur(lum_s * tile, (0, 0), 24.0)
    den = cv2.GaussianBlur(tile, (0, 0), 24.0) + 1e-4
    sp = A.lin_to_srgb(P3[:, :x1 + 40]).mean(-1)
    tref = (sp > cv2.blur(sp, (31, 31)) - 0.05).astype(np.float32)
    ref = (cv2.GaussianBlur(P3[:, :x1 + 40].mean(-1) * tref, (0, 0), 24.0)
           / (cv2.GaussianBlur(tref, (0, 0), 24.0) + 1e-4))[:, :x1]
    gain = np.clip(ref / (num / den + 1e-4), 0.6, 1.6)
    synth = synth * gain[..., None]
    # 4) blend: below the top grout line, up to just right of the vertical grout
    yA = eta_to_y(eA, xs)
    wy = np.clip((ys - (yA + 4.0)) / 5.0, 0, 1)          # plate 3 keeps its own top grout line
    wtop = np.clip((xg - 3.0 - xs) / 2.0, 0, 1)          # ... and its own vertical grout ...
    wbot = np.clip((xg + 6.5 - xs) / 2.5, 0, 1)          # ... except low down (copied band)
    wx = wtop + (wbot - wtop) * np.clip((ys - 950.0) / 100.0, 0, 1)
    w = (wy * wx).astype(np.float32)
    w = w * w * (3 - 2 * w)
    out = P3.copy()
    out[:, :x1] = P3[:, :x1] * (1 - w[..., None]) + synth * w[..., None]
    return out.astype(np.float32)


def plate(k):
    """A.plate(k), except plate 3 whose smeared far-left strip is repaired (cached on disk)."""
    if k != 3:
        return A.plate(k)
    if 3 in _P:
        return _P[3]
    path = os.path.join(CACHE, f'plate3_fix_v{VERSION}.png')
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED) if os.path.exists(path) else None
    if im is not None and im.shape[:2] == (C.H, C.W) and im.dtype == np.uint16:
        lin = A.srgb_to_lin(im[:, :, ::-1].astype(np.float32) / 65535.0)
    else:
        lin = _repair_plate3()
        os.makedirs(CACHE, exist_ok=True)
        s = (np.clip(A.lin_to_srgb(lin), 0, 1) * 65535.0 + 0.5).astype(np.uint16)
        tmp = path + f'.{os.getpid()}.tmp.png'
        cv2.imwrite(tmp, s[:, :, ::-1])
        os.replace(tmp, path)
        # re-read so every process uses bit-identical data (quantised like the cache)
        lin = A.srgb_to_lin(s.astype(np.float32) / 65535.0)
    _P[3] = np.ascontiguousarray(lin, np.float32)
    return _P[3]
