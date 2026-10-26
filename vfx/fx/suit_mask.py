"""Garment-mask construction for the `suit` module (pure function of the frame index).

    garment = (1 - head_eff) * (1 - hands)          (the matte is applied by the compositor)
    head_eff = max(head * (1 - arm_override), protect)

head      : convex head prior (face oval dilated + beard, cranium/hair ellipse, ears, pose jaw)
arm ovr.  : an arm in front of / beside the head wins over the head prior:
              * pose capsules where the pose is confident
              * matte protrusions connected to a hand (pointing / raised arms sticking out of the
                head+torso silhouette) - independent of the (often hallucinated) pose elbow
              * a forearm stub from the hand wrist along the matte (elbow not trusted)
protect   : dark beard / hair pixels inside the pose-jaw hull always stay head (no arm capsule,
            right or wrong, may paint the suit over the beard)
hands     : finger capsules + palm hull from validated / interpolated landmarks, swept along
            the hand velocity over the shutter interval (motion-blurred hands stay skin), plus a
            flat cuff cut across the wrist so the sleeve ends in a straight line
limbFront : forearms + hands that are in front of the torso (the shader suppresses the torso
            decorations there, so chevron/node never paint over a crossing forearm or hand)

Rasterised at half resolution with anti-aliasing; head feathered softly, hands crisply.
"""
import numpy as np
import cv2

SH = 4                       # fixed-point bits for cv2 drawing
Q = 2                        # rasterisation downscale

PALM_IDX = (0, 1, 2, 5, 9, 13, 17)
FINGERS = ((2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))
POSE_ARM = {'R': (12, 14, 16), 'L': (11, 13, 15)}


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


def _quad(img, pts, x0, y0, val=1.0):
    cv2.fillPoly(img, [_pts(np.asarray(pts), x0, y0)], val, cv2.LINE_AA, SH)


def body_scale(geo, i):
    return float(geo.sw[i])


def _ss(a, b, x):
    x = np.clip((x - a) / (b - a), 0.0, 1.0)
    return x * x * (3 - 2 * x)


def _hand_hull(dst, p, rf, x0, y0, val=1.0):
    """Palm hull + finger capsules for 21 landmarks p (21,2)."""
    palm = cv2.convexHull(p[list(PALM_IDX)].astype(np.float32)).reshape(-1, 2)
    cv2.fillPoly(dst, [_pts(palm, x0, y0)], val, cv2.LINE_AA, SH)
    for k in range(len(palm)):
        _capsule(dst, palm[k], palm[(k + 1) % len(palm)], rf * 1.15, rf * 1.15, x0, y0, val)
    for chain in FINGERS:
        pts = [p[k] for k in chain]
        for a, b in zip(pts[:-1], pts[1:]):
            _capsule(dst, a, b, rf, rf * 0.92, x0, y0, val)
        a, b = pts[-2], pts[-1]
        d = b - a
        d = d / (np.linalg.norm(d) + 1e-6)
        _capsule(dst, b, b + d * rf * 0.6, rf * 0.9, rf * 0.8, x0, y0, val)


def _ray_runs(a4, origin4, angs, maxlen4, start4):
    """For each angle: distance (quarter-res px) a ray from origin stays inside the matte
    (a4 = 'inside' field; values <= 0.5 stop the ray: matte exit or dark beard / hair)."""
    n = max(4, int(maxlen4 / 1.5))
    r = np.linspace(start4, maxlen4, n)
    dx, dy = np.cos(angs), np.sin(angs)
    X = origin4[0] + dx[:, None] * r[None, :]
    Y = origin4[1] + dy[:, None] * r[None, :]
    h4, w4 = a4.shape
    xi = np.clip(np.round(X).astype(int), 0, w4 - 1)
    yi = np.clip(np.round(Y).astype(int), 0, h4 - 1)
    inside = (a4[yi, xi] > 0.5) & (X >= 0) & (X < w4) & (Y >= 0) & (Y < h4)
    # first exit
    out = ~inside
    first = np.where(out.any(1), out.argmax(1), n)
    return np.where(first > 0, r[np.maximum(first - 1, 0)], 0.0)


def forearm_from_matte(a4, x0, y0, wr, pref, sw, maxlen_sw=0.75, fan=80.0):
    """Direction (unit) + visible length of the forearm leaving the wrist `wr`, found by ray
    casting in the quarter-res matte around the preferred direction `pref` (toward elbow /
    shoulder).  Returns (dir, length_px) or (None, 0)."""
    q4 = 2 * Q
    o4 = ((wr[0] - x0) / q4, (wr[1] - y0) / q4)
    base = np.arctan2(pref[1], pref[0])
    angs = base + np.radians(np.linspace(-fan, fan, 33))
    runs = _ray_runs(a4, o4, angs, maxlen_sw * sw / q4, 0.05 * sw / q4) * q4
    score = np.minimum(runs, 0.6 * sw) - 0.22 * sw * (np.abs(angs - base) / np.radians(fan)) ** 2
    k = int(np.argmax(score))
    if runs[k] < 0.12 * sw:
        return None, 0.0
    return np.array([np.cos(angs[k]), np.sin(angs[k])], np.float32), float(runs[k])


def build(geo, i, crop, fg=None, alpha=None, dbg=None):
    """Returns dict of float32 masks at FULL resolution of the crop (h, w):
    garment, head, head_trim, hands, arm, limb_front (coverage, alpha not applied) + info."""
    x0, y0, x1, y1 = crop
    w, h = x1 - x0, y1 - y0
    wq, hq = (w + Q - 1) // Q, (h + Q - 1) // Q
    sw = body_scale(geo, i)
    fhr = float(geo.fh_ref[i])
    aq = cv2.resize(alpha, (wq, hq), interpolation=cv2.INTER_AREA) if alpha is not None else np.ones((hq, wq), np.float32)
    a4 = cv2.resize(aq, (max(1, wq // 2), max(1, hq // 2)), interpolation=cv2.INTER_AREA)
    s4 = (max(1, wq // 2), max(1, hq // 2))

    # ---------------------------------------------------------------- head prior
    polys, ff = geo.head_shapes(i)
    conf = float(ff['conf'])
    jaw_poly = polys[-1]
    face_hull = np.zeros((hq, wq), np.float32)
    cv2.fillPoly(face_hull, [_pts(polys[0], x0, y0)], 1.0, cv2.LINE_AA, SH)
    ears = np.zeros((hq, wq), np.float32)
    for pl in polys[1:-1]:
        if len(pl) >= 3:
            cv2.fillPoly(ears, [_pts(pl, x0, y0)], 1.0, cv2.LINE_AA, SH)
    jaw = np.zeros((hq, wq), np.float32)
    cv2.fillPoly(jaw, [_pts(jaw_poly, x0, y0)], 1.0, cv2.LINE_AA, SH)
    face_core = np.zeros((hq, wq), np.float32)
    cv2.fillPoly(face_core, [_pts(ff['face_core'], x0, y0)], 1.0, cv2.LINE_AA, SH)
    skull = np.zeros((hq, wq), np.float32)
    if len(ff['skull']) >= 3:
        cv2.fillPoly(skull, [_pts(ff['skull'], x0, y0)], 1.0, cv2.LINE_AA, SH)
    if fg is not None:
        v = cv2.resize(fg.max(-1), (wq, hq), interpolation=cv2.INTER_AREA)
        v = cv2.GaussianBlur(v, (0, 0), 1.5)
        hair = _ss(0.16, 0.06, v)
        hair = cv2.dilate(hair, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        # beard darkness relative to his own skin (exposure / grade independent): reference =
        # the body skin just under the face (always lit skin, also in back / profile views)
        m = (face_core > 0.5) & (aq > 0.9) & (hair < 0.5)
        ref = float(np.percentile(v[m], 60)) if m.sum() > 50 else 0.25
        ref = float(np.clip(ref, 0.05, 0.6))
        beard = _ss(0.55 * ref, 0.30 * ref, v)
        beard_d = cv2.dilate(beard, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    else:
        hair = np.zeros_like(face_hull)
        beard = np.zeros_like(face_hull)
        beard_d = beard
    # ray field for the forearm search: matte, stopped by dark hair / beard
    ray4 = a4 * (1.0 - cv2.resize(np.maximum(hair, beard), s4, interpolation=cv2.INTER_AREA))
    skull *= conf * hair + (1 - conf)
    big = np.zeros((hq, wq), np.float32)
    if len(ff['skull_big']) >= 3:
        cv2.fillPoly(big, [_pts(ff['skull_big'], x0, y0)], 1.0, cv2.LINE_AA, SH)
    skull = np.maximum(skull, big * hair)
    # face hull: everything when the mesh is reliable; when synthesised (profile / back view)
    # it is often far off, so it then claims only the beard (the skull covers above the ears).
    # pose-jaw hull: beard only (the pose drags it onto raised forearms / the neck).
    head = np.maximum.reduce([face_hull * (conf + (1 - conf) * beard_d), ears, skull, jaw * beard_d])
    core = face_core * conf

    # ---------------------------------------------------------------- hands
    hands = np.zeros((hq, wq), np.float32)
    hand_core = np.zeros((hq, wq), np.float32)      # un-swept, for protrusion seeding
    hand_info = {}
    for side in ('R', 'L'):
        hs = geo.hand_shapes(i, side, sw)
        if hs is None:
            continue
        kind, data = hs
        if kind == 'lm':
            p, rf, size, wl, vel = data
            hb = np.zeros_like(hands)
            # shutter sweep: hull at t-1/2 .. t+1/2 frame along the landmark velocity
            vn = float(np.linalg.norm(vel[0]))
            nsub = 1 if vn < 2.0 else (3 if vn < 25 else 5)
            for o in np.linspace(-0.5, 0.5, nsub) if nsub > 1 else (0.0,):
                _hand_hull(hb, p + vel * o, rf, x0, y0)
            hc = np.zeros_like(hands)
            _hand_hull(hc, p, rf, x0, y0)
            # flat cuff cut across the wrist (sleeve ends in a straight line, not a round cap)
            fdir = _forearm_dir(geo, i, side, p, sw)
            r_wr = float(np.clip(0.42 * np.linalg.norm(p[5] - p[17]), 0.07 * sw, 0.12 * sw)) * 1.05
            r_wr = max(r_wr, rf * 2.2)
            nrm = np.array([-fdir[1], fdir[0]])
            c0 = p[0] + fdir * (0.25 * rf)
            c1 = p[0] - fdir * (0.05 * sw)
            _quad(hb, [c0 + nrm * r_wr, c1 + nrm * r_wr, c1 - nrm * r_wr, c0 - nrm * r_wr], x0, y0)
            np.maximum(hands, hb * wl, out=hands)
            np.maximum(hand_core, hc * wl, out=hand_core)
            hand_info[side] = dict(p=p, wr=p[0].copy(), fdir=fdir, r_wr=r_wr, w=wl, cuff=c1)
        else:
            poly, mrg, wgt = data
            hb = np.zeros_like(hands)
            hullp = cv2.convexHull(poly.astype(np.float32)).reshape(-1, 2)
            cv2.fillPoly(hb, [_pts(hullp, x0, y0)], 1.0, cv2.LINE_AA, SH)
            for k in range(len(hullp)):
                _capsule(hb, hullp[k], hullp[(k + 1) % len(hullp)], mrg, mrg, x0, y0)
            # a pose hand must sit on the matte (else it is hallucinated)
            sup = float((hb * aq).sum() / max(hb.sum(), 1.0))
            if sup > 0.6:
                np.maximum(hands, hb * wgt, out=hands)
                np.maximum(hand_core, hb * wgt, out=hand_core)

    # ---------------------------------------------------------------- arm override
    arm = np.zeros((hq, wq), np.float32)
    arm_thin = np.zeros((hq, wq), np.float32)
    stub = np.zeros((hq, wq), np.float32)
    limb = np.zeros((hq, wq), np.float32)
    arm_info = {}
    for side in ('R', 'L'):
        shp, el, wr = geo.arm(i, side)
        ish, iel, iwr = POSE_ARM[side]
        v_el = float(geo.vis_s[i, iel])
        v_wr = float(geo.vis_s[i, iwr])
        hi = hand_info.get(side)
        att = hi is not None and geo.hand_attached(i, side)
        hcw = hi['w'] if att else 0.0
        f_ua = float(np.clip((v_el - 0.40) / 0.30, 0, 1))
        f_fa = f_ua * max(float(np.clip((v_wr - 0.35) / 0.30, 0, 1)), hcw)
        r_wr = hi['r_wr'] if hi is not None else 0.10 * sw
        r_el = max(r_wr * 1.25, 0.12 * sw)
        r_sh = 0.165 * sw
        if f_ua > 0.01:
            _capsule(arm, shp, el, r_sh, r_el, x0, y0, f_ua)
            _capsule(arm_thin, shp, el, r_sh * 0.7, r_el * 0.7, x0, y0, f_ua)
        if f_fa > 0.01:
            _capsule(arm, el, wr, r_el, r_wr, x0, y0, f_fa)
            _capsule(arm_thin, el, wr, r_el * 0.72, r_wr * 0.72, x0, y0, f_fa)
        fa_geo = None
        if hi is not None:
            # forearm leaving the hand wrist, found in the matte (prior: toward the elbow if
            # trusted, else toward the shoulder) - opaque, independent of a hallucinated elbow
            pref = (el - hi['wr']) if (v_el > 0.6 and att) else (shp - hi['wr'])
            if np.linalg.norm(pref) < 1e-3:
                pref = -hi['fdir']
            d, run = forearm_from_matte(ray4, x0, y0, hi['wr'], pref, sw, maxlen_sw=0.8)
            if d is not None:
                L = min(run, 0.75 * sw)
                fa_geo = (hi['wr'], hi['wr'] + d * L)
                f_st = hi['w'] * (1.0 - f_fa)
                if f_st > 0.02:
                    _capsule(stub, hi['wr'], hi['wr'] + d * L, r_wr * 1.0, r_wr * 1.18, x0, y0, f_st)
        # limbFront (forearm in front of the torso)
        if f_fa > 0.5 or (v_el > 0.6 and hi is None and v_wr > 0.6):
            _capsule(limb, el, wr, r_el * 0.95, r_wr, x0, y0, 1.0)
        elif fa_geo is not None:
            _capsule(limb, fa_geo[0], fa_geo[1], r_wr, r_wr * 1.18, x0, y0, hi['w'])
        arm_info[side] = (shp, el, wr, r_sh, r_el, r_wr, f_ua, f_fa)

    sk = ff['skull']
    if stub.any():
        # the stub never cuts through the inner skull or a reliably tracked face
        if len(sk) >= 3:
            cen_ = sk.mean(0)
            core_sk = np.zeros((hq, wq), np.float32)
            cv2.fillPoly(core_sk, [_pts(cen_ + (sk - cen_) * 0.78, x0, y0)], 1.0, cv2.LINE_AA, SH)
            core_sk = cv2.GaussianBlur(core_sk, (0, 0), 3.0)
            # reliable mesh: the whole inner skull is protected; profile / back view: only the
            # hair (the ear / neck skin there is indistinguishable from the arm anyway)
            stub *= 1.0 - core_sk * np.maximum(hair, conf)
        core_f = cv2.GaussianBlur(core, (0, 0), 4.0)
        stub *= np.clip(1.0 - 1.6 * core_f, 0, 1)
        # ... nor the face hull (also when synthesised): the matte ray from the wrist toward
        # the shoulder often runs through the cheek once the forearm joins the head silhouette
        stub *= 1.0 - cv2.GaussianBlur(face_hull, (0, 0), 2.0)
        np.maximum(arm, stub, out=arm)

    # matte protrusions attached to a hand: thin structures sticking out of the head+torso
    # silhouette (opening removes them) that touch a hand -> forearm / raised arm
    prot = np.zeros((hq, wq), np.float32)
    if hand_core.max() > 0.5 and alpha is not None:
        r_open = max(2, int(0.18 * sw / (2 * Q)))
        ab = (a4 > 0.5).astype(np.uint8)
        k_ = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_open + 1, 2 * r_open + 1))
        opened = cv2.morphologyEx(ab, cv2.MORPH_OPEN, k_)
        P = (ab & (1 - opened)).astype(np.uint8)
        hc4 = cv2.resize(hand_core, (a4.shape[1], a4.shape[0]), interpolation=cv2.INTER_AREA) > 0.3
        hc4 = cv2.dilate(hc4.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        n, lab = cv2.connectedComponents(P, connectivity=8)
        if n > 1:
            touch = np.unique(lab[(hc4 > 0) & (P > 0)])
            touch = touch[touch > 0]
            if len(touch):
                keep = np.isin(lab, touch).astype(np.float32)
                keep = cv2.resize(keep, (wq, hq), interpolation=cv2.INTER_LINEAR)
                prot = np.clip(keep * 1.4 - 0.2, 0, 1)
                prot *= 1.0 - np.clip(hand_core * 1.5, 0, 1)
                # protrusions never claim the reliably tracked face core
                prot *= 1.0 - core

    # ---------------------------------------------------------------- feathering + combine
    head = cv2.GaussianBlur(head, (0, 0), 2.0)
    core_b = cv2.GaussianBlur(core, (0, 0), 3.0)
    arm = cv2.GaussianBlur(arm, (0, 0), 2.5)
    arm_thin = cv2.GaussianBlur(arm_thin, (0, 0), 2.5)
    prot = cv2.GaussianBlur(prot, (0, 0), 1.5)
    hands = cv2.GaussianBlur(hands, (0, 0), 0.9)
    arm_ovr = arm * (1.0 - core_b) + arm_thin * core_b
    arm_ovr = np.maximum(arm_ovr, prot)
    head_eff = head * (1.0 - arm_ovr)
    # dark beard / hair inside the jaw hull is always head (an arm capsule, right or wrong,
    # never paints the suit over his beard); protrusions are outside the head silhouette
    jaw_d = cv2.dilate(jaw, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    protect = cv2.GaussianBlur(jaw_d * beard * (1.0 - prot), (0, 0), 1.5)
    head_eff = np.maximum(head_eff, protect)
    garment = (1.0 - head_eff) * (1.0 - hands)
    limb = np.maximum(limb, cv2.dilate(hands, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))))
    limb = cv2.GaussianBlur(limb, (0, 0), 3.0)

    up = lambda m: cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)[:h, :w]
    if dbg is not None:
        dbg.update(head=head, jaw=jaw, skull=skull, ears=ears, face_hull=face_hull, beard_d=beard_d, prot=prot, stub=stub, arm=arm, protect=protect,
                   hands=hands, core=core, beard=beard, hair=hair)
    # field for the collar trim: no trim along boundaries carved by an arm override
    head_trim = head_eff * (1.0 - np.clip(arm_ovr * 2.5, 0, 1))
    return dict(garment=up(garment), head=up(head_eff), head_trim=up(head_trim), hands=up(hands), arm=up(arm_ovr),
                limb_front=up(limb), polys=polys, face=ff, arms=arm_info, hands_info=hand_info, sw=sw)


def _forearm_dir(geo, i, side, p, sw):
    """Unit vector pointing from the elbow side toward the wrist (forearm axis at the cuff)."""
    ish, iel, iwr = POSE_ARM[side]
    hax = p[9] - p[0]
    hax = hax / (np.linalg.norm(hax) + 1e-6)
    if geo.vis_s[i, iel] > 0.6 and geo.hand_attached(i, side):
        el = geo.pose_s[i, iel]
        d = p[0] - el
        n = np.linalg.norm(d)
        if n > 0.15 * sw:
            d = d / n
            # wrist flexion: blend a little toward the hand axis, but never flip
            m = d * 0.75 + hax * 0.25
            return (m / (np.linalg.norm(m) + 1e-6)).astype(np.float64)
    return hax.astype(np.float64)
