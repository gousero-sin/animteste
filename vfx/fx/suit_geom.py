"""Geometry priors for the `suit` module (pure functions of the frame index).

Everything here is derived from the precomputed tracking arrays only, so the masks are a
deterministic function of i.  Temporal smoothing is done over the *tracking data* of the
neighbouring frames (not over rendered frames), which keeps every frame independent while
removing landmark jitter / detection drop-outs.

Coordinates: output pixels (1080x1440), x = column, y = row.
R = the person's right = FRAME-LEFT, L = the person's left = FRAME-RIGHT.
"""
import numpy as np
import cv2
from .. import config as C
from .. import assets as A

# face-oval index positions inside track['face_oval_idx'] (36 pts)
OV_TOP, OV_LEFTEAR, OV_CHIN, OV_RIGHTEAR = 0, 28, 18, 8      # 10, 234, 152, 454

FINGERS = ((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))
PALM = (0, 1, 2, 5, 9, 13, 17)

POSE_HAND = {'R': (16, 18, 20, 22), 'L': (15, 17, 19, 21)}
POSE_ARM = {'R': (12, 14, 16), 'L': (11, 13, 15)}          # shoulder, elbow, wrist


def _gauss_smooth_series(x, valid, sigma):
    """Gaussian temporal smoothing of an (N, ...) series using only valid samples."""
    N = len(x)
    r = int(np.ceil(sigma * 3))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2).astype(np.float64)
    flat = x.reshape(N, -1).astype(np.float64)
    v = valid.astype(np.float64)
    num = np.zeros_like(flat)
    den = np.zeros(N)
    for o, w in zip(range(-r, r + 1), k):
        src = np.clip(np.arange(N) + o, 0, N - 1)
        num += w * v[src, None] * flat[src]
        den += w * v[src]
    out = np.where(den[:, None] > 1e-6, num / np.maximum(den[:, None], 1e-9), flat)
    return out.reshape(x.shape).astype(np.float32)


def _smooth1d(x, sigma):
    r = int(np.ceil(sigma * 3))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    xp = np.pad(x.astype(np.float64), r, mode='edge')
    return np.convolve(xp, k, mode='valid').astype(np.float32)


def _rolling_median(x, valid, half):
    N = len(x)
    out = np.zeros(N, np.float32)
    for i in range(N):
        a, b = max(0, i - half), min(N, i + half + 1)
        vv = x[a:b][valid[a:b]]
        out[i] = np.median(vv) if len(vv) else np.nan
    # fill nans
    m = np.isnan(out)
    if m.any():
        idx = np.arange(N)
        out[m] = np.interp(idx[m], idx[~m], out[~m])
    return out


def _runs(mask):
    """Inclusive (start, end) index pairs of the True runs of a 1-D bool array."""
    m = np.r_[False, np.asarray(mask, bool), False]
    d = np.diff(m.astype(np.int8))
    st = np.where(d == 1)[0]
    en = np.where(d == -1)[0] - 1
    return list(zip(st.tolist(), en.tolist()))


class Geometry:
    """Cleaned, temporally smoothed body geometry for every frame."""

    def __init__(self):
        T = A._track()
        self.N = N = C.N_FRAMES
        self.pose = T['pose'].astype(np.float32)
        oidx = T['face_oval_idx']
        face = T['face'].astype(np.float32)
        oval = face[:, oidx]                                   # (N,36,2)
        fd = T['face_det'].astype(bool)

        # ---------------- face validity -------------------------------------------------
        fh = np.linalg.norm(oval[:, OV_TOP] - oval[:, OV_CHIN], axis=1)
        fw = np.linalg.norm(oval[:, OV_RIGHTEAR] - oval[:, OV_LEFTEAR], axis=1)
        med_h = _rolling_median(fh, fd, 60)
        med_w = _rolling_median(fw, fd, 60)
        P = self.pose
        head_c = P[:, 0:11, :2].mean(1)                        # pose head centre (robust to turns)
        oc = oval.mean(1)
        off = oc - head_c
        med_off = np.stack([_rolling_median(off[:, k], fd, 60) for k in range(2)], 1)
        dev = np.linalg.norm(off - med_off, axis=1)
        ratio = fw / np.maximum(fh, 1)
        # leaning toward the camera makes the face legitimately bigger -> generous upper bound;
        # garbage meshes (head turned away) are small and/or have a wrong aspect ratio
        valid = (fd & (fh / med_h > 0.78) & (fh / med_h < 1.9) & (ratio > 0.62) & (ratio < 0.95)
                 & (dev < 0.30 * fh))
        # erode validity by 1 frame on each side (frames next to a failure are suspicious)
        v2 = valid.copy()
        v2[1:] &= valid[:-1]
        v2[:-1] &= valid[1:]
        valid = v2
        self.face_valid = valid

        # ---------------- synthesise ovals for invalid frames ---------------------------
        # (nearest valid ovals on both sides, carried by the pose head-centre motion)
        syn = oval.copy()
        vidx = np.where(valid)[0]
        for i in np.where(~valid)[0]:
            k = np.searchsorted(vidx, i)
            cands = []
            if k > 0:
                cands.append(vidx[k - 1])
            if k < len(vidx):
                cands.append(vidx[k])
            if not cands:
                continue
            acc = np.zeros((36, 2), np.float64)
            wsum = 0.0
            for j in cands:
                w = 1.0 / (abs(i - j) + 1e-3)
                acc += w * (oval[j] + (head_c[i] - head_c[j]) * 0.85)
                wsum += w
            syn[i] = (acc / wsum).astype(np.float32)
        self.oval = _gauss_smooth_series(syn, np.ones(N, bool), 1.2)
        # nose tip relative to the oval centre (for yaw); synthesised frames use the pose nose
        nose = face[:, 1].copy()
        nose[~valid] = self.pose[~valid, 0, :2] - head_c[~valid] + syn[~valid].mean(1)
        self.nose = _gauss_smooth_series(nose, np.ones(N, bool), 1.2)
        # confidence (1 = good face mesh around this time)
        self.face_conf = np.clip(_smooth1d(valid.astype(np.float32), 3.0), 0, 1)
        # distance (frames) to the nearest valid mesh: a synthesised oval is reliable a few
        # frames from a real one, garbage far from it (back of the head)
        vi = np.where(valid)[0]
        ii = np.arange(N)
        if len(vi):
            k = np.clip(np.searchsorted(vi, ii), 0, len(vi) - 1)
            km = np.clip(k - 1, 0, len(vi) - 1)
            self.face_dist = np.minimum(np.abs(ii - vi[k]), np.abs(ii - vi[km])).astype(np.float32)
        else:
            self.face_dist = np.full(N, 999.0, np.float32)
        # face size reference: interpolated over valid frames (follows lean-ins), smoothed
        idx = np.arange(N)
        self.fh_ref = _smooth1d(np.interp(idx, idx[valid], fh[valid]).astype(np.float32), 3.0)
        self.fw_ref = _smooth1d(np.interp(idx, idx[valid], fw[valid]).astype(np.float32), 3.0)

        # ---------------- pose smoothing (light) -----------------------------------------
        self.pose_s = _gauss_smooth_series(P[:, :, :2], np.ones(N, bool), 1.0)
        self.vis_s = _gauss_smooth_series(P[:, :, 3], np.ones(N, bool), 1.5)
        # frontal-ness from the pose: nose inside the ear span and ears well apart.  A
        # synthesised (frontal) face oval is trustworthy when frontal (hand over the face),
        # garbage in profile / back views.
        ps_ = self.pose_s
        e78 = ps_[:, 7] - ps_[:, 8]
        ed_ = np.linalg.norm(e78, axis=1)
        ax_ = e78 / np.maximum(ed_, 1.0)[:, None]
        no_ = np.abs(((ps_[:, 0] - 0.5 * (ps_[:, 7] + ps_[:, 8])) * ax_).sum(1)) / np.maximum(ed_ * 0.5, 1.0)
        fr_ = np.clip((1.6 - no_) / 0.5, 0, 1) * np.clip((ed_ / np.maximum(self.fw_ref, 1) - 0.40) / 0.15, 0, 1)
        self.frontal = np.clip(_smooth1d(fr_.astype(np.float32), 1.5), 0, 1)
        shd = _smooth1d(np.linalg.norm(self.pose_s[:, 11] - self.pose_s[:, 12], axis=1), 2.0)
        self.sw = np.maximum(shd, 1.45 * self.fh_ref).astype(np.float32)

        # ---------------- hands -------------------------------------------------------------
        # The raw track gap-fills missing hands by interpolating between detections, including
        # false ones (forearm detected as a hand, mirrored hands) - garbage that cuts holes in the
        # suit.  We re-derive everything from validated detections only:
        #   1. per-frame / per-run validation by temporal self-consistency (size, axis angle)
        #   2. gaps between valid detections: linear interpolation when plausible
        #   3. run ends: constant-velocity extrapolation for a few frames (motion-blurred exits)
        #   4. pose fallback only when the pose hand is clearly trustworthy
        self.hand, self.hand_w, self.hand_vel, self.hand_kind, self.det_ok = {}, {}, {}, {}, {}
        self.hand_conf = {}
        self.fb_ok = {}
        for s in ('R', 'L'):
            self._build_hand(T, s)
        self.have_hand = {s: self.hand_w[s] > 0.02 for s in ('R', 'L')}

    def _build_hand(self, T, s):
        N = self.N
        P = self.pose
        sw = self.sw
        h = T['hand' + s][:, :, :2].astype(np.float32)
        det = T['hand%s_det' % s].astype(bool) & ~np.isnan(h[:, 0, 0])
        hn = np.nan_to_num(h)
        L12 = np.linalg.norm(hn[:, 12] - hn[:, 0], axis=1)
        L9 = np.linalg.norm(hn[:, 9] - hn[:, 0], axis=1)
        PW = np.linalg.norm(hn[:, 5] - hn[:, 17], axis=1)
        size = np.maximum.reduce([L12, 1.9 * L9, 1.6 * PW])
        ax = hn[:, 9] - hn[:, 0]
        ang = np.arctan2(ax[:, 1], ax[:, 0])

        def adiff(a, b):
            return np.abs((a - b + np.pi) % (2 * np.pi) - np.pi)

        # (a) degenerate frames
        det &= size > 0.10 * sw
        # (b) frames inconsistent with their own run (size collapse / axis flip)
        for a, b in _runs(det):
            if b - a + 1 >= 3:
                med = np.median(size[a:b + 1])
                for k in range(a, b + 1):
                    lo, hi = max(a, k - 3), min(b, k + 3)
                    am = np.median(ang[lo:hi + 1])
                    if size[k] < 0.5 * med or adiff(ang[k], am) > np.radians(70):
                        det[k] = False
        # (c) short runs must agree with the long ("anchor") runs around them
        runs = _runs(det)
        anchors = [r for r in runs if r[1] - r[0] + 1 >= 5]
        for a, b in runs:
            n = b - a + 1
            if n >= 5:
                continue
            ra = np.median(ang[a:b + 1])
            rs = np.median(size[a:b + 1])
            votes = []
            prev = [r for r in anchors if r[1] < a and a - r[1] <= 45]
            nxt = [r for r in anchors if r[0] > b and r[0] - b <= 45]
            if prev:
                e = prev[-1][1]
                votes.append(adiff(ra, ang[e]) < np.radians(100) and 0.4 < rs / size[e] < 2.3)
            if nxt:
                e = nxt[0][0]
                votes.append(adiff(ra, ang[e]) < np.radians(100) and 0.4 < rs / size[e] < 2.3)
            if votes:
                ok = any(votes)
            else:
                # isolated short run: keep only when it sits on a confident pose wrist
                wi = POSE_ARM[s][2]
                dpw = np.linalg.norm(hn[a:b + 1, 0] - P[a:b + 1, wi, :2], axis=1) / sw[a:b + 1]
                ok = bool(np.median(P[a:b + 1, wi, 3]) > 0.6 and np.median(dpw) < 0.6)
            if not ok:
                det[a:b + 1] = False
        self.det_ok[s] = det.copy()
        # smoothing inside runs (valid samples only)
        hs = h.copy()
        hs[~det] = 0
        hs = _gauss_smooth_series(hs, det, 0.8)
        out = np.full((N, 21, 2), np.nan, np.float32)
        out[det] = hs[det]
        w = np.zeros(N, np.float32)
        w[det] = 1.0
        kind = np.zeros(N, np.int8)              # 0 none, 1 det, 2 interp, 3 extrap, 4 pose
        kind[det] = 1
        vel = np.zeros((N, 21, 2), np.float32)
        didx = np.where(det)[0]
        # gaps
        ext = []                                  # (frame, base frame, direction)
        for j in range(len(didx) - 1):
            a, b = didx[j], didx[j + 1]
            gap = b - a - 1
            if gap <= 0:
                continue
            disp = np.linalg.norm(out[b, 0] - out[a, 0]) / sw[a]
            if (gap <= 24 and disp <= 0.7) or (gap <= 45 and disp <= 0.35):
                # pose-guided interpolation: the deviation of the pose palm from its own
                # straight-line path is added (hands circling / waving between detections)
                wi = POSE_ARM[s][2]
                pp = self.pose_s[:, list(POSE_HAND[s])].mean(1)
                for k in range(a + 1, b):
                    u = (k - a) / (b - a)
                    out[k] = out[a] * (1 - u) + out[b] * u
                    if gap > 4:
                        dev = pp[k] - (pp[a] * (1 - u) + pp[b] * u)
                        dn = float(np.linalg.norm(dev))
                        lim = 0.35 * sw[k]
                        if dn > lim:
                            dev = dev * (lim / dn)
                        beta = float(np.clip((self.vis_s[k, wi] - 0.45) / 0.2, 0, 1)) * min(1.0, 4.0 * min(u, 1 - u))
                        out[k] = out[k] + beta * dev
                    w[k] = 1.0
                    kind[k] = 2
            else:
                ext.append((a, +1))
                ext.append((b, -1))
        if len(didx):
            ext.append((didx[0], -1))
            ext.append((didx[-1], +1))
        # velocities (px / frame) for known frames
        known = w > 0
        for k in np.where(known)[0]:
            a = k - 1 if k > 0 and known[k - 1] else k
            b = k + 1 if k + 1 < N and known[k + 1] else k
            if b > a:
                vel[k] = (out[b] - out[a]) / (b - a)
        # run-end extrapolation (motion-blurred exits / entries): opaque for 2 frames, then
        # a quick fade (stuck extrapolations would otherwise cut holes in the torso)
        EXW = (1.0, 1.0, 0.7, 0.35)
        for base, dr in ext:
            k2 = base - dr * 2
            if 0 <= k2 < N and known[k2] and kind[k2] in (1, 2):
                v = (out[base] - out[k2]) / 2.0 * dr
            elif 0 <= base - dr < N and known[base - dr]:
                v = (out[base] - out[base - dr]) * dr
            else:
                v = np.zeros((21, 2), np.float32)
            vmax = 0.35 * sw[base]
            vn = float(np.linalg.norm(v[0]))
            if vn > vmax:
                v = v * (vmax / vn)
            for k in range(1, len(EXW) + 1):
                f = base + dr * k
                if not (0 <= f < N) or kind[f] in (1, 2):
                    break
                wk = EXW[k - 1]
                if wk <= w[f]:
                    continue
                out[f] = out[base] + v * (k * dr)
                w[f] = wk
                kind[f] = 3
                vel[f] = v * dr
        # pose fallback: strictly gated (hallucinated pose hands cut holes in the suit)
        wi, ei, si = POSE_ARM[s][2], POSE_ARM[s][1], POSE_ARM[s][0]
        ps = self.pose_s
        pidx = list(POSE_HAND[s])
        palm = ps[:, pidx].mean(1)
        jump = np.zeros(N, np.float32)
        jump[1:] = np.linalg.norm(palm[1:] - palm[:-1], axis=1)
        jump = np.maximum(jump, np.r_[jump[1:], 0.0]) / sw
        ua = np.linalg.norm(ps[:, ei] - ps[:, si], axis=1)
        fa = np.linalg.norm(ps[:, wi] - ps[:, ei], axis=1)
        idx = np.arange(N)
        if len(didx):
            kk = np.clip(np.searchsorted(didx, idx), 0, len(didx) - 1)
            kp = np.clip(kk - 1, 0, len(didx) - 1)
            near = np.where(np.abs(idx - didx[kk]) < np.abs(idx - didx[kp]), didx[kk], didx[kp])
            dnear = np.abs(idx - near)
            dwr = np.linalg.norm(ps[:, wi] - out[near, 0], axis=1) / sw
        else:
            dnear = np.full(N, 999)
            dwr = np.full(N, 9.0)
        fb = ((w < 0.5) & (self.vis_s[:, wi] >= 0.75) & (dnear <= 8) & (dwr <= 0.4)
              & (fa > 0.4 * ua) & (fa < 1.3 * ua) & (jump < 0.25)
              & (ps[:, wi, 0] > 0) & (ps[:, wi, 0] < C.W) & (ps[:, wi, 1] < C.H - 10))
        self.fb_ok[s] = fb
        self.hand[s] = out
        self.hand_w[s] = w
        self.hand_vel[s] = vel
        self.hand_kind[s] = kind
        self.hand_conf[s] = np.clip(_smooth1d(det.astype(np.float32), 3.0), 0, 1)

    # ------------------------------------------------------------------ queries
    def shoulders(self, i):
        p = self.pose_s[i]
        return p[12].copy(), p[11].copy()          # frame-left (R), frame-right (L)

    def arm(self, i, side):
        """(shoulder, elbow, wrist) points; wrist from the hand landmarks when the hand is
        known AND plausibly attached to this arm (MediaPipe sometimes swaps handedness)."""
        p = self.pose_s[i]
        a, b, c = POSE_ARM[side]
        sh, el, wr = p[a].copy(), p[b].copy(), p[c].copy()
        w = float(self.hand_w[side][i])
        if w > 0.02 and self.hand_attached(i, side):
            wr = w * self.hand[side][i][0] + (1 - w) * wr
        return sh, el, wr

    def hand_attached(self, i, side):
        if self.hand_w[side][i] <= 0.02:
            return False
        p = self.pose_s[i]
        a, b, c = POSE_ARM[side]
        sw = float(self.sw[i])
        hw = self.hand[side][i][0]
        ua = max(float(np.linalg.norm(p[b] - p[a])), 0.45 * sw)
        if self.vis_s[i, b] > 0.5:
            return float(np.linalg.norm(hw - p[b])) < 1.6 * ua
        return float(np.linalg.norm(hw - p[a])) < 2.6 * ua

    def face_frame(self, i):
        o = self.oval[i]
        c = o.mean(0)
        top, chin = o[OV_TOP], o[OV_CHIN]
        le, re = o[OV_LEFTEAR], o[OV_RIGHTEAR]
        fh = float(np.linalg.norm(top - chin))
        fw = float(np.linalg.norm(re - le))
        conf = float(self.face_conf[i])
        # when the mesh is synthesised, sizes come from the robust reference
        fh = conf * fh + (1 - conf) * float(self.fh_ref[i])
        fw = conf * fw + (1 - conf) * float(self.fw_ref[i])
        ey = top - chin
        ey = ey / (np.linalg.norm(ey) + 1e-6)                 # face 'up' (image coords)
        ex = np.array([-ey[1], ey[0]], np.float32)             # perpendicular, frame-right
        if np.dot(ex, re - le) < 0:
            ex = -ex
        return c, ex, ey, fw, fh, conf

    def head_shapes(self, i):
        """Polygons whose UNION covers the whole head: face (+beard, glasses), ears, hair and
        the back of the head at any yaw.  Bottom boundary = jaw line (collar goes up to it).

        * face hull  : dilated (smoothed / synthesised) face oval + ear discs
        * skull      : ellipse centred near the pose ear midpoint (= skull centre in 2D at any
                       yaw), cut by a half-plane under the ears (lower when turned sideways)
        * pose jaw   : hull of pose ears / nose / mouth / estimated chin, dilated (profile beard)
        """
        o = self.oval[i]
        c, ex, ey, fw, fh, conf = self.face_frame(i)
        unc = 1.0 - conf
        polys = []
        # ---- face hull
        rel = o - c
        lo = np.clip(-(rel @ ey) / (0.5 * fh), 0, 1)
        m = 0.07 + 0.10 * unc
        face = c + rel * (1 + m) - ey[None, :] * (fh * (0.055 + 0.04 * unc) * lo[:, None] ** 2)
        le, re = o[OV_LEFTEAR], o[OV_RIGHTEAR]
        # yaw from the nose tip: the far ear is hidden behind the head in 3/4 view
        yaw_f = float(np.clip(np.dot(self.nose[i] - c, ex) / (0.5 * fw), -1, 1))
        polys.append(cv2.convexHull(face.astype(np.float32)).reshape(-1, 2))
        for p, sgn in ((le, -1), (re, 1)):
            vis = float(np.clip(1.0 - (sgn * yaw_f - 0.12) / 0.30, 0.0, 1.0))
            vis = max(vis, unc)
            if vis < 0.05:
                continue
            q = p + sgn * ex * (0.12 * fw * vis) + ey * 0.03 * fh
            ear = [q + (np.cos(a) * ex * 0.13 * fh * (0.4 + 0.6 * vis) + np.sin(a) * ey * 0.17 * fh)
                   for a in np.linspace(0, 2 * np.pi, 14, endpoint=False)]
            # bridge the ear to the face edge (temples / glasses arms)
            ear += [p + ey * 0.12 * fh, p - ey * 0.10 * fh]
            polys.append(cv2.convexHull(np.array(ear, np.float32)).reshape(-1, 2))
        # ---- skull ellipse
        P = self.pose_s[i]
        em = (P[7] + P[8]) * 0.5
        nose = P[0]
        fhr = float(self.fh_ref[i])
        # head 'up': face up when the mesh is good, else from the pose (shoulders -> ears)
        sm = (P[11] + P[12]) * 0.5
        up_p = em - sm
        up_p = up_p / (np.linalg.norm(up_p) + 1e-6)
        up_p = 0.5 * up_p + 0.5 * np.array([0, -1], np.float32)
        up = conf * ey + (1 - conf) * up_p
        up = up / (np.linalg.norm(up) + 1e-6)
        rt = np.array([-up[1], up[0]], np.float32)
        side = float(np.dot(nose - em, rt))                  # + : nose toward frame-right
        yaw = float(np.clip(abs(side) / (0.40 * fhr), 0, 1))
        sgn = np.sign(side) if abs(side) > 1e-3 else 0.0
        cc = em + up * (0.17 * fhr) - rt * sgn * (0.13 * fhr * yaw)
        ax = (0.72 + 0.14 * yaw) * fhr
        ay = (0.72 + 0.04 * yaw) * fhr
        th = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        ell = cc[None, :] + np.cos(th)[:, None] * ax * rt[None, :] + np.sin(th)[:, None] * ay * up[None, :]
        cut_p = em - up * ((0.06 + 0.24 * yaw) * fhr)
        skull = _clip_halfplane(ell, cut_p, up)
        ell2 = cc[None, :] + np.cos(th)[:, None] * ax * 1.3 * rt[None, :] + np.sin(th)[:, None] * ay * 1.3 * up[None, :]
        skull_big = _clip_halfplane(ell2, em - up * ((0.0 + 0.24 * yaw) * fhr), up)
        # ---- pose jaw hull (covers the beard when the mesh is unreliable / profile)
        eyes_c = (P[2] + P[5]) * 0.5
        mouth = (P[9] + P[10]) * 0.5
        chin = mouth + (mouth - eyes_c) * 0.85
        core = np.array([P[7], P[8], nose, P[9], P[10], chin], np.float32)
        r = (0.09 + 0.05 * unc) * fhr
        jp = [q + r * np.array([np.cos(a), np.sin(a)]) for q in core for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)]
        polys.append(cv2.convexHull(np.array(jp, np.float32)).reshape(-1, 2))
        info = dict(c=c, ex=ex, ey=ey, fw=fw, fh=fh, conf=conf, face_core=c + rel * 1.0, skull=skull, skull_big=skull_big, up=up)
        return polys, info

    def hand_shapes(self, i, side, sw):
        """Returns (kind, data) for the hand mask:
           ('lm', (pts21, finger_radius, size, weight, velocity)) or ('pose', (polygon, margin, w))
           or None."""
        if self.have_hand[side][i]:
            h = self.hand[side][i]
            hs = float(np.linalg.norm(h[12] - h[0]))
            pw = float(np.linalg.norm(h[5] - h[17]))
            size = max(hs, 1.6 * pw, 0.25 * sw)
            conf = float(self.hand_conf[side][i])
            rf = 0.075 * size + 3.0 + 0.02 * size * (1 - conf)
            return 'lm', (h, rf, size, float(self.hand_w[side][i]), self.hand_vel[side][i])
        if self.fb_ok[side][i]:
            return self.hand_shapes_pose(i, side, sw)
        return None

    def hand_shapes_pose(self, i, side, sw):
        p = self.pose_s[i]
        idx = POSE_HAND[side]
        q = p[list(idx)]
        wr = q[0]
        knu = (q[1] + q[2]) * 0.5
        d = knu - wr
        L = np.linalg.norm(d) + 1e-6
        tip = knu + d / L * max(L * 1.3, 0.18 * sw)
        poly = np.array([wr, q[1], q[2], q[3], tip + (q[1] - knu) * 0.8, tip + (q[2] - knu) * 0.8], np.float32)
        return ('pose', (poly, 0.08 * sw, 1.0))

def _clip_halfplane(poly, p0, n):
    """Keep the part of a convex polygon where (x - p0) . n >= 0 (Sutherland-Hodgman)."""
    out = []
    k = len(poly)
    d = (poly - p0) @ n
    for j in range(k):
        a, b = poly[j], poly[(j + 1) % k]
        da, db = d[j], d[(j + 1) % k]
        if da >= 0:
            out.append(a)
        if (da >= 0) != (db >= 0):
            t = da / (da - db)
            out.append(a + (b - a) * t)
    if len(out) < 3:
        return np.zeros((0, 2), np.float32)
    return np.array(out, np.float32)
