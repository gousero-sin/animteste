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
        # confidence (1 = good face mesh around this time)
        self.face_conf = np.clip(_smooth1d(valid.astype(np.float32), 3.0), 0, 1)
        # face size reference: interpolated over valid frames (follows lean-ins), smoothed
        idx = np.arange(N)
        self.fh_ref = _smooth1d(np.interp(idx, idx[valid], fh[valid]).astype(np.float32), 3.0)
        self.fw_ref = _smooth1d(np.interp(idx, idx[valid], fw[valid]).astype(np.float32), 3.0)

        # ---------------- pose smoothing (light) -----------------------------------------
        self.pose_s = _gauss_smooth_series(P[:, :, :2], np.ones(N, bool), 1.0)

        # ---------------- hands -------------------------------------------------------------
        # Non-detected landmarks in the track are extrapolations that often stay stuck on the
        # chest after the hand has left: trust them only close (in time) to a real detection.
        self.hand = {}
        self.hand_conf = {}
        self.hand_w = {}
        idx = np.arange(N)
        for s in ('R', 'L'):
            h = T['hand' + s].astype(np.float32)
            have = ~np.isnan(h[:, 0, 0])
            det = T['hand%s_det' % s].astype(bool)
            hs = h.copy()
            hs[~have] = 0
            hs = _gauss_smooth_series(hs[:, :, :2], have, 0.8)
            hs[~have] = np.nan
            self.hand[s] = hs
            self.hand_conf[s] = np.clip(_smooth1d(det.astype(np.float32), 3.0), 0, 1)
            didx = np.where(det)[0]
            if len(didx):
                k = np.clip(np.searchsorted(didx, idx), 1, len(didx) - 1) if len(didx) > 1 else np.zeros(N, int)
                dist = np.minimum(np.abs(idx - didx[k]), np.abs(idx - didx[np.maximum(k - 1, 0)]))
            else:
                dist = np.full(N, 999)
            w = np.exp(-np.maximum(dist - 1, 0) / 2.5).astype(np.float32)
            w[~have] = 0
            self.hand_w[s] = w
        self.have_hand = {s: self.hand_w[s] > 0.02 for s in ('R', 'L')}

    # ------------------------------------------------------------------ queries
    def shoulders(self, i):
        p = self.pose_s[i]
        return p[12].copy(), p[11].copy()          # frame-left (R), frame-right (L)

    def arm(self, i, side):
        """(shoulder, elbow, wrist) points; wrist from the hand landmarks when trusted."""
        p = self.pose_s[i]
        a, b, c = POSE_ARM[side]
        sh, el, wr = p[a].copy(), p[b].copy(), p[c].copy()
        w = float(self.hand_w[side][i])
        if w > 0.02:
            wr = w * self.hand[side][i][0] + (1 - w) * wr
        return sh, el, wr

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
        ears = []
        for p, sgn in ((le, -1), (re, 1)):
            q = p + sgn * ex * (0.12 * fw) + ey * 0.03 * fh
            for a in np.linspace(0, 2 * np.pi, 10, endpoint=False):
                ears.append(q + (np.cos(a) * ex * 0.13 * fh + np.sin(a) * ey * 0.17 * fh))
        pts = np.concatenate([face, np.array(ears)], 0).astype(np.float32)
        polys.append(cv2.convexHull(pts).reshape(-1, 2))
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
           ('lm', (pts21, finger_radius, size)) or ('pose', (polygon, margin)) or None."""
        if self.have_hand[side][i]:
            h = self.hand[side][i]
            hs = float(np.linalg.norm(h[12] - h[0]))
            pw = float(np.linalg.norm(h[5] - h[17]))
            size = max(hs, 1.6 * pw, 0.25 * sw)
            conf = float(self.hand_conf[side][i])
            rf = 0.075 * size + 3.0 + 0.03 * size * (1 - conf)
            return 'lm', (h, rf, size)
        return self.hand_shapes_pose(i, side, sw)

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
        vis = float(np.clip((self.pose[i, idx[0], 3] - 0.25) / 0.3, 0, 1))
        return ('pose', (poly, 0.10 * sw, vis))

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
