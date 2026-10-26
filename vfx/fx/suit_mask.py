"""Garment-mask construction for the `suit` module.

garment = (1 - head * (1 - arm)) * (1 - hands)       (alpha is applied by the compositor)

head  : convex head prior (face oval dilated + beard, cranium/hair ellipse, ears)
arm   : tapered capsules along upper arm / forearm; an arm crossing in front of the head
        (arms raised, forearm across the face) wins over the head prior
hands : finger capsules + palm polygon from the 21 landmarks (pose fallback)

Everything is rasterised at half resolution with anti-aliasing and feathered, then
upsampled, so the boundary is smooth and sub-pixel stable from frame to frame.
"""
import numpy as np
import cv2

SH = 4                       # fixed-point bits for cv2 drawing
Q = 2                        # rasterisation downscale


def _pts(p, x0, y0):
    p = (np.asarray(p, np.float64) - (x0, y0)) / Q
    return np.round(p * (1 << SH)).astype(np.int32)


def _capsule(img, a, b, ra, rb, x0, y0, val=1.0):
    """Tapered capsule (round caps) from a (radius ra) to b (radius rb)."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    d = b - a
    L = np.linalg.norm(d)
    if L < 1e-3:
        cv2.circle(img, tuple(_pts(a, x0, y0)), int(max(ra, rb) / Q * (1 << SH)), val, -1, cv2.LINE_AA, SH)
        return
    n = np.array([-d[1], d[0]]) / L
    poly = np.array([a + n * ra, b + n * rb, b - n * rb, a - n * ra])
    cv2.fillPoly(img, [_pts(poly, x0, y0)], val, cv2.LINE_AA, SH)
    cv2.circle(img, tuple(_pts(a, x0, y0)), int(ra / Q * (1 << SH)), val, -1, cv2.LINE_AA, SH)
    cv2.circle(img, tuple(_pts(b, x0, y0)), int(rb / Q * (1 << SH)), val, -1, cv2.LINE_AA, SH)


def body_scale(geo, i):
    sR, sL = geo.shoulders(i)
    sw = float(np.linalg.norm(sL - sR))
    fh = float(geo.fh_ref[i])
    # shoulder width collapses when he turns sideways: never go below ~1.45 face heights
    return max(sw, 1.45 * fh)


def build(geo, i, crop, fg=None):
    """Returns dict of float32 masks at FULL resolution of the crop (h, w):
    head, hands, arm, garment (coverage, alpha not applied), plus geometry info."""
    x0, y0, x1, y1 = crop
    w, h = x1 - x0, y1 - y0
    wq, hq = (w + Q - 1) // Q, (h + Q - 1) // Q
    sw = body_scale(geo, i)

    head = np.zeros((hq, wq), np.float32)
    polys, ff = geo.head_shapes(i)
    for pl in polys:
        if len(pl) >= 3:
            cv2.fillPoly(head, [_pts(pl, x0, y0)], 1.0, cv2.LINE_AA, SH)
    # skull disc: when the face mesh is reliable it only claims hair-dark pixels (so a lit arm
    # beside / in front of the head becomes suit); when unreliable (profile) it claims all.
    skull = np.zeros((hq, wq), np.float32)
    if len(ff['skull']) >= 3:
        cv2.fillPoly(skull, [_pts(ff['skull'], x0, y0)], 1.0, cv2.LINE_AA, SH)
    conf = float(ff['conf'])
    if fg is not None:
        v = cv2.resize(fg.max(-1), (wq, hq), interpolation=cv2.INTER_AREA)
        v = cv2.GaussianBlur(v, (0, 0), 1.5)
        hair = np.clip((0.16 - v) / (0.16 - 0.06), 0, 1)
        hair = hair * hair * (3 - 2 * hair)
        hair = cv2.dilate(hair, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        skull *= conf * hair + (1 - conf)
        big = np.zeros((hq, wq), np.float32)
        if len(ff['skull_big']) >= 3:
            cv2.fillPoly(big, [_pts(ff['skull_big'], x0, y0)], 1.0, cv2.LINE_AA, SH)
        skull = np.maximum(skull, big * hair)
    head = np.maximum(head, skull)
    core = np.zeros((hq, wq), np.float32)
    cv2.fillPoly(core, [_pts(ff['face_core'], x0, y0)], float(ff['conf']), cv2.LINE_AA, SH)

    # ---------------- arms (only used to override the head prior) ----------------------
    # An arm in front of / beside the head wins over the head prior, but only where the pose
    # is confident (hidden elbows get hallucinated onto the face).  Inside the face core the
    # override uses thinner capsules so a slightly-off arm never eats the face.
    arm = np.zeros((hq, wq), np.float32)
    arm_thin = np.zeros((hq, wq), np.float32)
    arm_info = {}
    P4 = geo.pose[i]
    for side in ('R', 'L'):
        shp, el, wr = geo.arm(i, side)
        ish, iel, iwr = (12, 14, 16) if side == 'R' else (11, 13, 15)
        v_el = float(P4[iel, 3])
        v_wr = float(P4[iwr, 3])
        hc = float(geo.hand_conf[side][i]) if geo.have_hand[side][i] else 0.0
        f_ua = float(np.clip((v_el - 0.40) / 0.30, 0, 1))
        f_fa = f_ua * max(float(np.clip((v_wr - 0.35) / 0.30, 0, 1)), hc)
        if geo.have_hand[side][i]:
            hh = geo.hand[side][i]
            pw = float(np.linalg.norm(hh[5] - hh[17]))
            r_wr = np.clip(0.42 * pw, 0.075 * sw, 0.13 * sw)
        else:
            r_wr = 0.10 * sw
        r_el = max(r_wr * 1.25, 0.12 * sw)
        r_sh = 0.165 * sw
        if f_ua > 0.01:
            _capsule(arm, shp, el, r_sh, r_el, x0, y0, f_ua)
            _capsule(arm_thin, shp, el, r_sh * 0.7, r_el * 0.7, x0, y0, f_ua)
        if f_fa > 0.01:
            _capsule(arm, el, wr, r_el, r_wr, x0, y0, f_fa)
            _capsule(arm_thin, el, wr, r_el * 0.72, r_wr * 0.72, x0, y0, f_fa)
        arm_info[side] = (shp, el, wr, r_sh, r_el, r_wr, f_ua, f_fa)

    # ---------------- hands -------------------------------------------------------------
    hands = np.zeros((hq, wq), np.float32)
    for side in ('R', 'L'):
        hs = geo.hand_shapes(i, side, sw)
        if hs is None:
            continue
        kind, data = hs
        if kind == 'lm':
            p, rf, size = data
            wl = float(geo.hand_w[side][i])
            p = p[:, :2]
            hb = np.zeros_like(hands)
            palm = cv2.convexHull(p[list(PALM_IDX)].astype(np.float32)).reshape(-1, 2)
            cv2.fillPoly(hb, [_pts(palm, x0, y0)], 1.0, cv2.LINE_AA, SH)
            for k in range(len(palm)):
                _capsule(hb, palm[k], palm[(k + 1) % len(palm)], rf * 1.15, rf * 1.15, x0, y0)
            for chain in FINGERS:
                pts = [p[k] for k in chain]
                for a, b in zip(pts[:-1], pts[1:]):
                    _capsule(hb, a, b, rf, rf * 0.92, x0, y0)
                a, b = pts[-2], pts[-1]
                d = b - a
                d = d / (np.linalg.norm(d) + 1e-6)
                _capsule(hb, b, b + d * rf * 0.6, rf * 0.9, rf * 0.8, x0, y0)
            hands = np.maximum(hands, hb * wl)
            if wl < 0.98:
                pk = geo.hand_shapes_pose(i, side, sw)
                if pk is not None:
                    _pose_hand(hands, pk[1], x0, y0, 1.0 - wl)
        else:
            _pose_hand(hands, data, x0, y0, 1.0)

    # feathering (at half res): head edge soft-ish, hands a bit softer (tracking error)
    head = cv2.GaussianBlur(head, (0, 0), 2.2)
    core = cv2.GaussianBlur(core, (0, 0), 3.0)
    arm = cv2.GaussianBlur(arm, (0, 0), 3.0)
    arm_thin = cv2.GaussianBlur(arm_thin, (0, 0), 3.0)
    hands = cv2.GaussianBlur(hands, (0, 0), 2.0)
    arm_ovr = arm * (1.0 - core) + arm_thin * core
    head_eff = head * (1.0 - arm_ovr)
    garment = (1.0 - head_eff) * (1.0 - hands)

    up = lambda m: cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)[:h, :w]
    return dict(garment=up(garment), head=up(head_eff), hands=up(hands), arm=up(arm),
                polys=polys, face=ff, arms=arm_info, sw=sw)


PALM_IDX = (0, 1, 2, 5, 9, 13, 17)
FINGERS = ((2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))


def _pose_hand(dst, data, x0, y0, wgt):
    poly, m, vis = data
    wgt = wgt * vis
    if wgt < 0.01:
        return
    hb = np.zeros_like(dst)
    hullp = cv2.convexHull(poly.astype(np.float32)).reshape(-1, 2)
    cv2.fillPoly(hb, [_pts(hullp, x0, y0)], 1.0, cv2.LINE_AA, SH)
    for k in range(len(hullp)):
        _capsule(hb, hullp[k], hullp[(k + 1) % len(hullp)], m, m, x0, y0)
    np.maximum(dst, hb * wgt, out=dst)
