"""'Magic eraser' removal of one object group from the bathroom plate.

Phases (all analytic in t):
  1. selection  : a laser outline races around the object's real silhouette from the point
                  nearest to where he points (two comet heads), a scan ripple runs through it.
  2. burn       : a noise-driven incandescent front spreads from the ignition point; just ahead
                  of it the surface chars, at the front it glows white-hot -> amber -> red, and
                  behind it the clean plate is revealed. Light spills onto the surrounding tiles.
  3. disintegrate: the burnt material leaves as embers (emissive) and object-coloured dust
                  (premultiplied 'over') that rise, swirl and fade.
"""
import numpy as np
import cv2
from .. import config as C
from .. import assets as A
from ..glfx import gl
from .cleanup_util import sstep, value_fbm, heat_ramp

W, H = C.W, C.H


def _smooth_contour(cnt, step=3.0, sigma=2.0):
    """Resample a closed contour at ~step px and smooth it (wrap-around gaussian)."""
    p = cnt.reshape(-1, 2).astype(np.float32)
    if len(p) < 8:
        return None
    seg = np.linalg.norm(np.diff(np.vstack([p, p[:1]]), axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    P = s[-1]
    n = max(16, int(P / step))
    ss = np.linspace(0, P, n, endpoint=False)
    pp = np.vstack([p, p[:1]])
    x = np.interp(ss, s, pp[:, 0])
    y = np.interp(ss, s, pp[:, 1])
    k = int(3 * sigma) + 1
    ker = np.exp(-0.5 * (np.arange(-k, k + 1) / sigma) ** 2)
    ker /= ker.sum()
    xw = np.concatenate([x[-k:], x, x[:k]])
    yw = np.concatenate([y[-k:], y, y[:k]])
    x = np.convolve(xw, ker, 'valid')
    y = np.convolve(yw, ker, 'valid')
    return np.stack([x, y], 1).astype(np.float32), float(P)


class Eraser:
    def __init__(self, name, k_old, k_new, region, objmask, ign_t, ign_side, fallback_ign,
                 burn_t0, burn_t1, trace_lead=0.30, trace_dur=0.24, stagger=0.45,
                 edge_w=0.05, n_part=2400, ember_frac=0.55, seed=1, noise_amp=0.20,
                 light_gain=1.0):
        self.name = name
        self.k_old, self.k_new = k_old, k_new
        self.region, self.objmask = region, objmask
        self.ign_t, self.ign_side, self.fallback_ign = ign_t, ign_side, np.float32(fallback_ign)
        self.burn_t0, self.burn_t1 = burn_t0, burn_t1
        self.trace_lead, self.trace_dur = trace_lead, trace_dur
        self.stagger, self.edge_w = stagger, edge_w
        self.n_part, self.ember_frac, self.seed = n_part, ember_frac, seed
        self.noise_amp = noise_amp
        self.light_gain = light_gain
        self.t_first = burn_t0 - trace_lead - 0.05     # refined in build()
        self.t_done = burn_t1 + 0.02                  # crop fully revealed after this
        self.t_part_end = burn_t1 + 2.4
        self._built = False

    # ------------------------------------------------------------------ build (lazy)
    def _ign_point(self):
        i = C.i_of(self.ign_t)
        best = None
        for di in (0, -1, 1, -2, 2, -3, 3, -5, 5):
            h = A.hand(i + di, self.ign_side)
            if h is not None:
                best = h[8, :2]
                break
        if best is None or not (0 <= best[0] < W and 0 <= best[1] < H):
            best = self.fallback_ign
        return np.float32(best)

    def build(self):
        if self._built:
            return
        P0, P1 = A.plate(self.k_old), A.plate(self.k_new)
        S0 = A.lin_to_srgb(P0)
        S1 = A.lin_to_srgb(P1)
        d = np.abs(S0 - S1).max(-1)
        reg = A.plate_mask(self.region)
        sup = (d > 0.004) | (reg > 0.01)
        ys, xs = np.where(sup)
        m = 44
        x0, x1 = max(0, xs.min() - m), min(W, xs.max() + m + 1)
        y0, y1 = max(0, ys.min() - m), min(H, ys.max() + m + 1)
        self.box = (x0, y0, x1, y1)
        h, w = y1 - y0, x1 - x0
        self.h, self.w = h, w
        self.old = np.ascontiguousarray(P0[y0:y1, x0:x1])
        self.new = np.ascontiguousarray(P1[y0:y1, x0:x1])
        dc = cv2.GaussianBlur(d[y0:y1, x0:x1], (0, 0), 1.0)
        regc = reg[y0:y1, x0:x1]
        objm = A.plate_mask(self.objmask)[y0:y1, x0:x1]
        regd = cv2.dilate((regc > 0.01).astype(np.uint8), np.ones((15, 15), np.uint8))
        O = sstep(0.035, 0.12, dc) * regd
        O = cv2.morphologyEx(O, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        O = cv2.GaussianBlur(O, (0, 0), 0.8)
        self.O = O.astype(np.float32)
        Og = cv2.dilate(O, np.ones((7, 7), np.uint8))
        self.Og = cv2.GaussianBlur(Og, (0, 0), 2.0).astype(np.float32)

        objd = cv2.dilate((objm > 0.01).astype(np.uint8), np.ones((9, 9), np.uint8))
        objbin = ((O > 0.4) & (objd > 0)).astype(np.uint8)
        closed = cv2.morphologyEx(objbin, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
        closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        self.objbin = objbin

        # ---- components and ignition points
        grp = cv2.dilate(closed, np.ones((25, 25), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(grp)
        ign = self._ign_point() - np.float32([x0, y0])
        self.ign_main = ign + np.float32([x0, y0])
        comps = []
        for k in range(1, n):
            msk = (lab == k) & (objbin > 0)
            if msk.sum() < 120:
                continue
            py, px = np.where(msk)
            dd = (px - ign[0]) ** 2 + (py - ign[1]) ** 2
            j = int(np.argmin(dd))
            comps.append(dict(label=k, ign=np.float32([px[j], py[j]]), dist=float(np.sqrt(dd[j]))))
        if not comps:
            comps = [dict(label=-1, ign=ign, dist=0.0)]
        comps.sort(key=lambda c: c['dist'])
        dmin = comps[0]['dist']
        Y, X = np.mgrid[0:h, 0:w].astype(np.float32)
        T = np.full((h, w), 1e9, np.float32)
        best = np.zeros((h, w), np.int32)
        for ci, c in enumerate(comps):
            c['delay'] = self.stagger * (c['dist'] - dmin)
            Tc = c['delay'] + np.sqrt((X - c['ign'][0]) ** 2 + (Y - c['ign'][1]) ** 2)
            better = Tc < T
            T = np.where(better, Tc, T)
            best = np.where(better, ci, best)
        size = np.sqrt(max(1.0, float(objbin.sum())) / max(1, len(comps)))
        nz = value_fbm(h, w, max(12, size * 0.35), 4, self.seed) - 0.5
        T = T + nz * 2 * self.noise_amp * max(60.0, size)
        ob = objbin > 0
        tmin = float(T[ob].min()) if ob.any() else float(T.min())
        tmax = float(np.percentile(T[ob], 99.7)) if ob.any() else float(T.max())
        Tn = np.clip((T - tmin) / (tmax - tmin + 1e-6), -0.05, 1.0)
        self.T = (self.burn_t0 + Tn * (self.burn_t1 - self.burn_t0)).astype(np.float32)
        # selection start per pixel (by component) and scan distance
        self.comps = comps
        tsel = np.zeros((h, w), np.float32)
        dsc = np.zeros((h, w), np.float32)
        for ci, c in enumerate(comps):
            msk = best == ci
            if c['label'] > 0:
                cm = (lab == c['label']) & ob
            else:
                cm = ob
            tb = float(self.T[cm].min()) if cm.any() else self.burn_t0
            c['t_burn'] = tb
            c['t_sel'] = tb - self.trace_lead
            dist = np.sqrt((X - c['ign'][0]) ** 2 + (Y - c['ign'][1]) ** 2)
            c['dmax'] = float(dist[cm].max()) if cm.any() else 100.0
            tsel[msk] = c['t_sel']
            dsc[msk] = dist[msk] / max(c['dmax'], 1.0)
        self.tsel = tsel
        self.dscan = dsc
        self.t_first = min(c['t_sel'] for c in comps) - 0.02

        # ---- trace contours (real silhouettes)
        cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        self.traces = []
        for cnt in cnts:
            if cv2.contourArea(cnt) < 220:
                continue
            r = _smooth_contour(cnt)
            if r is None:
                continue
            pts, P = r
            # T at contour points (min in a small neighbourhood: contour is on the boundary)
            Tmin_img = cv2.erode(self.T, np.ones((7, 7), np.uint8))
            xi = np.clip(pts[:, 0].round().astype(int), 0, w - 1)
            yi = np.clip(pts[:, 1].round().astype(int), 0, h - 1)
            Tp = Tmin_img[yi, xi]
            st = int(np.argmin(Tp))
            pts = np.roll(pts, -st, axis=0)
            Tp = np.roll(Tp, -st)
            seg = np.linalg.norm(np.diff(np.vstack([pts, pts[:1]]), axis=0), axis=1)
            s = np.concatenate([[0], np.cumsum(seg)[:-1]])
            Pl = float(seg.sum())
            frac = np.minimum(s, Pl - s) / (Pl / 2)
            t0 = float(Tp[0]) - self.trace_lead
            dur = self.trace_dur * (0.75 + 0.25 * min(1.0, Pl / 1500.0))
            self.traces.append(dict(pts=pts + np.float32([x0, y0]), s=s, P=Pl, frac=frac,
                                    t0=t0, dur=dur, Tp=Tp))
            self.t_first = min(self.t_first, t0 - 0.02)

        # ---- particles
        rng = np.random.default_rng(self.seed * 7 + 3)
        wts = (self.O * (objbin > 0)).ravel().astype(np.float64)
        if wts.sum() <= 0:
            wts = np.ones_like(wts)
        wts /= wts.sum()
        idx = rng.choice(len(wts), size=self.n_part, p=wts)
        py, px = np.divmod(idx, w)
        jit = rng.random((self.n_part, 2)).astype(np.float32) - 0.5
        pos = np.stack([px, py], 1).astype(np.float32) + jit
        n_ = self.n_part
        self.p_pos = pos + np.float32([x0, y0])
        self.p_ts = self.T[py, px] + rng.random(n_).astype(np.float32) * 0.04
        ash = np.float32([0.045, 0.040, 0.036])
        oc = np.clip(self.old[py, px], 0, 1).astype(np.float32)
        self.p_col = oc * 0.45 + ash * 0.55
        self.p_ember = rng.random(n_) < self.ember_frac
        self.p_life = np.where(self.p_ember, rng.uniform(0.6, 1.7, n_), rng.uniform(0.6, 1.6, n_)).astype(np.float32)
        ang = rng.uniform(-np.pi, np.pi, n_)
        spd = np.where(self.p_ember, rng.uniform(30, 160, n_), rng.uniform(10, 50, n_))
        self.p_v = np.stack([np.cos(ang) * spd, np.sin(ang) * spd * 0.5 - rng.uniform(30, 100, n_)], 1).astype(np.float32)
        self.p_buoy = np.where(self.p_ember, rng.uniform(120, 300, n_), rng.uniform(70, 170, n_)).astype(np.float32)
        self.p_sw_a = rng.uniform(6, 26, n_).astype(np.float32)
        self.p_sw_w = rng.uniform(2.0, 6.0, n_).astype(np.float32)
        self.p_sw_p = rng.uniform(0, 2 * np.pi, n_).astype(np.float32)
        self.p_size = np.where(self.p_ember, rng.uniform(0.8, 2.0, n_), rng.uniform(0.7, 1.6, n_)).astype(np.float32)
        self.p_int = rng.uniform(2.0, 6.0, n_).astype(np.float32)
        self.p_tau = rng.uniform(0.25, 0.7, n_).astype(np.float32)
        self.p_fl = rng.uniform(10, 30, n_).astype(np.float32)
        self.t_part_end = float(self.p_ts.max() + self.p_life.max()) + 0.05
        # animated fine noise for flicker
        self.nf1 = value_fbm(h, w, 10, 3, self.seed + 11)
        self.nf2 = value_fbm(h, w, 10, 3, self.seed + 12)
        self._built = True

    # ------------------------------------------------------------------ queries
    def crop_active(self, t):
        return self.t_first <= t <= self.t_done

    def any_active(self, t):
        return self.t_first <= t <= self.t_part_end

    # ------------------------------------------------------------------ render
    def render_crop(self, t):
        """Returns (rgb_plate_crop, emission_crop, light_info) for the crop box."""
        self.build()
        s = (t - self.T) / self.edge_w
        reveal = sstep(-0.3, 0.6, s)
        pre = sstep(-3.5, -0.3, s)
        Oc = self.O
        char = (pre * (1.0 - reveal) * Oc)[..., None]
        oldc = self.old * (1.0 - 0.62 * char) * (1.0 + char * np.float32([0.05, -0.18, -0.40]))
        rgb = oldc * (1.0 - reveal[..., None]) + self.new * reveal[..., None]

        # emission at the burn front
        flick = self.nf1 * (0.5 + 0.5 * np.sin(t * 23.0)) + self.nf2 * (0.5 - 0.5 * np.sin(t * 23.0))
        heat = np.exp(-((s - 0.15) / 0.6) ** 2)
        tail = (s > 0) * np.exp(-np.maximum(s, 0) / 2.2) * 0.18
        E = self.Og * (heat * (0.65 + 0.7 * flick) + tail)
        # hot specks right at the front
        spk = np.clip((flick - 0.72) * 6.0, 0, 1) * np.exp(-((s - 0.0) / 0.9) ** 2) * self.Og
        em = heat_ramp(np.clip(heat * (0.75 + 0.35 * flick), 0, 1)) * (E * 3.6)[..., None]
        em += np.float32([1.0, 0.8, 0.55]) * (spk * 5.0)[..., None]

        # selection: faint cool fill + scan ripple through the object (before the burn)
        sel = sstep(self.tsel, self.tsel + self.trace_dur, t) * (1.0 - pre)
        if sel.max() > 0.001:
            ph = (t - self.tsel) / max(self.trace_dur * 1.4, 1e-3)
            band = np.exp(-((self.dscan - ph * 1.25 + 0.1) / 0.06) ** 2) * (ph > 0) * (ph < 1.3)
            fill = self.O * (0.05 * sel + 0.55 * band * (1.0 - pre))
            em += np.float32([0.45, 0.75, 1.0]) * fill[..., None]
            rgb = rgb * (1.0 + 0.25 * (self.O * sel)[..., None])

        # light spill on the surrounding wall (albedo-weighted)
        tot = float(E.sum())
        if tot > 1.0:
            small = cv2.resize(em, (max(1, self.w // 4), max(1, self.h // 4)), interpolation=cv2.INTER_AREA)
            b1 = cv2.GaussianBlur(em, (0, 0), 10.0)
            b2 = cv2.resize(cv2.GaussianBlur(small, (0, 0), 14.0), (self.w, self.h), interpolation=cv2.INTER_LINEAR)
            alb = np.clip(self.new * 1.3, 0, 1)
            em += (b1 * 0.30 + b2 * 0.55) * alb

        # light info for relighting
        info = None
        if tot > 1.0:
            Y, X = np.mgrid[0:self.h:4, 0:self.w:4].astype(np.float32)
            Es = E[::4, ::4]
            ssum = float(Es.sum()) + 1e-6
            cx = float((Es * X).sum() / ssum) + self.box[0]
            cy = float((Es * Y).sum() / ssum) + self.box[1]
            info = dict(pos=(cx, cy), power=tot)
        return rgb, em, info

    def trace_layer(self, t):
        """Additive emission of the laser trace (crop sized to the eraser box) or None."""
        self.build()
        segs0, segs1, cols, rads = [], [], [], []
        heads0, heads1, hcols = [], [], []
        for tr in self.traces:
            if t < tr['t0'] or t > self.burn_t1 + 0.1:
                continue
            tk = tr['t0'] + tr['frac'] * tr['dur']
            age = t - tk
            on = age >= 0
            if not on.any():
                continue
            hold = 0.55 + 0.20 * np.sin(2 * np.pi * 7.0 * age + tr['frac'] * 9.0)
            I = np.where(on, 6.0 * np.exp(-np.maximum(age, 0) / 0.05) + hold, 0.0)
            I *= sstep(0.0, 0.015, age)
            # consumed by the burn front
            sb = (t - tr['Tp']) / self.edge_w
            I *= 1.0 - sstep(-1.5, 0.5, sb)
            # burn-approach brightening
            I *= 1.0 + 1.5 * np.exp(-((sb + 1.0) / 1.0) ** 2)
            p = tr['pts']
            q = np.roll(p, -1, axis=0)
            keep = I > 0.02
            if keep.any():
                segs0.append(p[keep]); segs1.append(q[keep])
                c = np.zeros((keep.sum(), 4), np.float32)
                c[:, 0] = I[keep] * 0.75
                c[:, 1] = I[keep] * 0.90
                c[:, 2] = I[keep] * 1.00
                c[:, 3] = 1.0
                cols.append(c)
            # comet heads
            f = (t - tr['t0']) / tr['dur']
            if 0.0 <= f <= 1.02:
                fprev = max(0.0, (t - 1.0 / 45 - tr['t0']) / tr['dur'])
                for sgn in (0, 1):
                    def at(ff):
                        ss = ff * tr['P'] / 2
                        if sgn:
                            ss = tr['P'] - ss
                        ss = ss % tr['P']
                        x = np.interp(ss, np.append(tr['s'], tr['P']), np.append(p[:, 0], p[0, 0]))
                        y = np.interp(ss, np.append(tr['s'], tr['P']), np.append(p[:, 1], p[0, 1]))
                        return (x, y)
                    heads0.append(at(min(f, 1.0)))
                    heads1.append(at(fprev))
                    hcols.append((9.0, 10.0, 11.0, 1.0))
        if not segs0:
            return None
        x0, y0, x1, y1 = self.box
        size = (x1 - x0, y1 - y0)
        p0 = np.concatenate(segs0)
        p1 = np.concatenate(segs1)
        col = np.concatenate(cols)
        img = gl.sprites(size, p0, p1, radius=1.1, color=col, offset=(x0, y0), shape=2)[..., :3]
        if heads0:
            img += gl.sprites(size, np.float32(heads0), np.float32(heads1), radius=3.5,
                              color=np.float32(hcols), offset=(x0, y0), shape=2)[..., :3]
        halo = cv2.GaussianBlur(img, (0, 0), 5.0) * 0.9 * np.float32([0.45, 0.75, 1.3])
        halo += cv2.GaussianBlur(img, (0, 0), 16.0) * 0.5 * np.float32([0.35, 0.6, 1.2])
        return img + halo

    def particles(self, t):
        """Returns (ember_add_layer, dust_over_layer) as core-style tuples or (None, None).
        Each tuple: (rgb, a_or_None, (x0, y0))."""
        self.build()
        age = t - self.p_ts
        alive = (age > 0) & (age < self.p_life)
        if not alive.any():
            return None, None
        a = age[alive]
        dt = 1.0 / 55.0
        ap = np.maximum(a - dt, 0)

        def pos(aa, sel):
            p0 = self.p_pos[sel]
            v = self.p_v[sel]
            b = self.p_buoy[sel]
            A_ = self.p_sw_a[sel] * (aa / (aa + 0.35))
            w_ = self.p_sw_w[sel]
            ph = self.p_sw_p[sel]
            drag = (1 - np.exp(-2.2 * aa)) / 2.2
            x = p0[:, 0] + v[:, 0] * drag + A_ * (np.sin(w_ * aa + ph) - np.sin(ph))
            y = p0[:, 1] + v[:, 1] * drag - 0.5 * b * aa * aa + 0.5 * A_ * (np.cos(0.7 * w_ * aa + ph) - np.cos(ph))
            return np.stack([x, y], 1).astype(np.float32)

        P1 = pos(a, alive)
        P0 = pos(ap, alive)
        life = self.p_life[alive]
        fade_in = sstep(0.0, 0.05, a)
        fade_out = 1.0 - sstep(0.65 * life, life, a)
        emb = self.p_ember[alive]
        lo = np.minimum(P0, P1).min(0) - 12
        hi = np.maximum(P0, P1).max(0) + 12
        x0, y0 = int(max(0, lo[0])), int(max(0, lo[1]))
        x1, y1 = int(min(W, hi[0])), int(min(H, hi[1]))
        if x1 - x0 < 2 or y1 - y0 < 2:
            return None, None
        size = (x1 - x0, y1 - y0)
        add_layer, over_layer = None, None
        if emb.any():
            ae = a[emb]
            heat = np.exp(-ae / self.p_tau[alive][emb])
            fl = 0.75 + 0.25 * np.sin(self.p_fl[alive][emb] * ae + self.p_sw_p[alive][emb])
            inten = self.p_int[alive][emb] * (fade_in * fade_out)[emb] * fl * (0.25 + 0.75 * heat)
            col = heat_ramp(np.clip(heat * 1.05, 0, 1)) * inten[:, None]
            c4 = np.concatenate([col, np.ones((len(col), 1), np.float32)], 1)
            rad = self.p_size[alive][emb] * (0.6 + 0.5 * heat)
            img = gl.sprites(size, P0[emb], P1[emb], radius=rad, color=c4, offset=(x0, y0), shape=2)
            add_layer = (img[..., :3], None, (x0, y0))
        dm = ~emb
        if dm.any():
            ad = a[dm]
            op = 0.75 * (1 - ad / life[dm]) ** 1.6 * fade_in[dm]
            base = self.p_col[alive][dm] * (1.0 - 0.45 * np.clip(ad / life[dm], 0, 1))[:, None]
            hot = np.exp(-ad / 0.10)
            c4 = np.zeros((dm.sum(), 4), np.float32)
            c4[:, :3] = base * op[:, None]
            c4[:, 3] = op
            img = gl.sprites(size, P0[dm], P1[dm], radius=self.p_size[alive][dm], color=c4,
                             offset=(x0, y0), shape=1)
            # birth glow of dust bits (they leave the front hot)
            c5 = np.zeros((dm.sum(), 4), np.float32)
            c5[:, :3] = heat_ramp(hot * 0.8) * (hot * 3.0 * fade_in[dm])[:, None]
            c5[:, 3] = 0
            glo = gl.sprites(size, P0[dm], P1[dm], radius=self.p_size[alive][dm], color=c5,
                             offset=(x0, y0), shape=2)
            over_layer = (img[..., :3], np.clip(img[..., 3], 0, 1), (x0, y0))
            if add_layer is None:
                add_layer = (glo[..., :3], None, (x0, y0))
            else:
                add_layer = (add_layer[0] + glo[..., :3], None, (x0, y0))
        return add_layer, over_layer

    def light(self, t, info):
        """Interactive light(s) for the person: cool during selection, warm while burning."""
        self.build()
        out = []
        for c in self.comps:
            if c['t_sel'] <= t <= c['t_burn'] + 0.1:
                k = sstep(c['t_sel'], c['t_sel'] + 0.08, t) * (1 - sstep(c['t_burn'], c['t_burn'] + 0.1, t))
                p = c['ign'] + np.float32(self.box[:2])
                out.append(dict(pos=p, color=np.float32([0.6, 0.8, 1.0]), intensity=0.35 * k * self.light_gain,
                                radius=260.0, rim=1.2, diffuse=0.25))
        if info is not None:
            pw = min(1.0, info['power'] / 9000.0)
            if pw > 0.01:
                out.append(dict(pos=np.float32(info['pos']), color=np.float32([1.0, 0.55, 0.22]),
                                intensity=(0.25 + 1.6 * pw) * self.light_gain, radius=300.0, rim=1.6, diffuse=0.45))
        return out
