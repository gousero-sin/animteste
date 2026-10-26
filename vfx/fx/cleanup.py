"""cleanup — bathroom clean-up ("magic eraser"), bathroom -> black studio, black studio.

Owns the background from 0 s until the tunnel (EV['tunnel'][0]).
  * bathroom   : clean plates (cleanup_plates: plate 3's smeared far-left strip is rebuilt with
                 perspective-correct tile/grout) with a low-frequency shadow/exposure transfer
                 from the raw frame (keeps his real shadow / phone exposure drift), three
                 magic-eraser removals synced to EV['clean_clutter'|'clean_flush'|'clean_shower'].
  * transform  : EV['studio'] - lamp flicker, energy core behind him, an expanding electric
                 front; the room cracks with an impact pattern (cleanup_fracture) and breaks into
                 3D shards that tumble back into the void; then the studio lights strike on.
  * studio     : near-black backdrop (wall behind him ~sRGB 15-25), subtle pool, drifting haze,
                 lit dust motes, two short out-of-focus LED bars (STUDIO_BARS), crisp rim + hair
                 lights, edge despill + light wrap, person grade; the haze scatters every light.
Render-order safe: every eraser whose fixed window contains t is built before any decision.
"""
import sys
import numpy as np
import cv2
from .. import config as C
from .. import assets as A
from .. import core
from ..glfx import gl
from .cleanup_util import sstep, sstep_f
from .cleanup_eraser import Eraser
from .cleanup_shatter import Shatter, front_R, ORIGIN
from .cleanup_studio import Studio, GRADE
from . import cleanup_plates as CP

W, H = C.W, C.H
EV = C.EV

T_SHATTER0 = EV['studio'][0] - 0.05          # 13.00 anticipation (lamp flicker, core)
T_STUDIO_END = EV['studio'][1]               # 14.90


class FX(core.Effect):
    name = 'cleanup'

    def __init__(self):
        c0, c1 = EV['clean_clutter']
        f0, f1 = EV['clean_flush']
        s0, s1 = EV['clean_shower']
        # (burn windows start slightly after the EV start: the selection trace is the EV start)
        self.erasers = [
            Eraser('clutter', 0, 1, 'clutter', 'clutter', ign_t=5.55, ign_side='L',
                   fallback_ign=(770, 1150), burn_t0=c0 + 0.10, burn_t1=c1 - 0.05,
                   trace_lead=0.30, trace_dur=0.26, stagger=0.55, edge_px=10.0,
                   n_ember=420, n_flake=40, n_spark=70, seed=11, noise_amp=0.22, light_gain=1.0),
            # flush: retimed to when he points at it and his arm hides it least (occlusion
            # 51-68 % at 10.7-10.9 s vs 73-83 % at 10.95-11.15): trace 10.58 ("negócio de"),
            # burn 10.78-11.06 (covers "descarga" 10.92). Embers/sparks are thrown up through the
            # gap between his raised arm and head, plus a warm light pop on his arm.
            Eraser('flush', 1, 2, 'flush', 'flush', ign_t=10.78, ign_side='R',
                   fallback_ign=(504, 1049), burn_t0=f0 - 0.02, burn_t1=11.06,
                   trace_lead=0.20, trace_dur=0.16, stagger=0.5, edge_px=9.0,
                   n_ember=260, n_flake=10, n_spark=90, seed=23, noise_amp=0.25, light_gain=1.3,
                   ember_dir=-np.pi / 2 - 0.12, ember_spread=0.8, ember_v=(120.0, 360.0),
                   pop_gain=1.0),
            Eraser('shower', 2, 3, 'shower_region', 'shower', ign_t=11.5, ign_side='R',
                   fallback_ign=(382, 921), burn_t0=s0 + 0.22, burn_t1=s1 - 0.02,
                   trace_lead=0.24, trace_dur=0.22, stagger=0.5, edge_px=11.0,
                   n_ember=420, n_flake=36, n_spark=80, seed=37, noise_amp=0.18, light_gain=1.1,
                   dark_obj=(0.16, 0.32)),
        ]
        self._p0_small = None
        self.shatter = None
        self.studio = None

    # ------------------------------------------------------------------ helpers
    def active(self, t):
        return t < EV['tunnel'][0] + 0.2

    def _gain(self, i):
        """Low-frequency ratio raw/plate0 outside the person: shadows + exposure drift."""
        q = 4
        w, h = W // q, H // q
        if self._p0_small is None:
            self._p0_small = cv2.resize(A.plate(0), (w, h), interpolation=cv2.INTER_AREA).mean(-1)
            # removable objects (mirror reflections, etc.) never feed the ratio: no ghosting
            m = np.maximum.reduce([A.plate_mask(n) for n in ('clutter', 'flush', 'shower_region')])
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA)
            self._objfree = (cv2.dilate((m > 0.01).astype(np.uint8), np.ones((5, 5), np.uint8)) == 0).astype(np.float32)
        p = self._p0_small
        F = np.zeros((h, w), np.float32)
        V = np.zeros((h, w), np.float32)
        # 3-frame average (pure function of i): steadier shadows, real motion kept
        for j in (i - 1, i, i + 1):
            j = min(max(j, 0), C.N_FRAMES - 1)
            f = cv2.resize(A.frame(j), (w, h), interpolation=cv2.INTER_AREA).mean(-1)
            a = cv2.resize(A.alpha(j), (w, h), interpolation=cv2.INTER_AREA)
            v = (cv2.dilate((a > 0.02).astype(np.uint8), np.ones((11, 11), np.uint8)) == 0).astype(np.float32)
            v *= self._objfree
            F += f * v
            V += v
        valid = V / 3.0
        num = cv2.GaussianBlur(F / 3.0, (0, 0), 3.0)
        den = cv2.GaussianBlur(p * valid, (0, 0), 3.0)
        wv = cv2.GaussianBlur(valid, (0, 0), 3.0)
        num2 = cv2.GaussianBlur(F / 3.0, (0, 0), 18.0)
        den2 = cv2.GaussianBlur(p * valid, (0, 0), 18.0)
        wv2 = cv2.GaussianBlur(valid, (0, 0), 18.0)
        r1 = num / (den + 1e-4)
        r2 = num2 / (den2 + 1e-4)
        sel = valid > 0.99
        g0 = float(np.median((F / 3.0)[sel] / (p[sel] + 1e-4))) if sel.sum() > 500 else 1.0
        g0 = float(np.clip(g0, 0.85, 1.15))
        r2 = r2 * np.clip(wv2 / 0.3, 0, 1) + g0 * (1 - np.clip(wv2 / 0.3, 0, 1))
        k = np.clip(wv / 0.6, 0, 1)
        r = r1 * k + r2 * (1 - k)
        # low-confidence where the plate is dark (black shower panel): use the wide estimate
        conf = np.clip(den / (wv * 0.25 + 1e-4), 0, 1)
        r = r * conf + r2 * (1 - conf)
        r = np.clip(r, 0.60, 1.15)
        return cv2.resize(r, (W, H), interpolation=cv2.INTER_LINEAR)

    def bathroom(self, S, with_fx=True):
        """Bathroom background for frame S.i (plates + erasers + gain). Returns (bg, lights)."""
        t = S.t
        # Build every eraser whose FIXED window contains t before any timing query: build()
        # refines t_done/t_first, so deciding first made the result depend on render order.
        for e in self.erasers:
            if e.window(t):
                e.build()
        done = [t > e.t_done for e in self.erasers]
        k = 3 if done[2] else 2 if done[1] else 1 if done[0] else 0
        bg = CP.plate(k).copy()
        emis = []
        lights = []
        for e in self.erasers:
            if not with_fx:
                break
            if e._built and e.window(t) and e.crop_active(t):
                rgb, em, info = e.render_crop(t)
                x0, y0, x1, y1 = e.box
                bg[y0:y1, x0:x1] = rgb
                emis.append((em, (x0, y0)))
                lights += e.light(t, info)
        g = self._gain(S.i)
        bg *= g[..., None]
        for em, box in emis:
            core.blend(bg, core.Layer('add', em, box=box))
        if with_fx:
            for e in self.erasers:
                if e._built and e.window(t) and e.any_active(t):
                    tr = e.trace_layer(t)
                    if tr is not None:
                        core.blend(bg, core.Layer('add', tr, box=e.box[:2]))
                    add, over = e.particles(t)
                    if over is not None:
                        core.blend(bg, core.Layer('over', over[0], over[1], box=over[2]))
                    if add is not None:
                        core.blend(bg, core.Layer('add', add[0], box=add[2]))
        return bg, lights

    def eraser_particles_masked(self, bg, t, keep_fn):
        """Draw still-alive eraser particles over bg, multiplied by keep_fn(x0, y0, h, w)."""
        for e in self.erasers:
            if not e.window(t):
                continue
            e.build()
            if not e.any_active(t):
                continue
            add, over = e.particles(t)
            for lay, mode in ((over, 'over'), (add, 'add')):
                if lay is None:
                    continue
                rgb, a, (x0, y0) = lay
                k = keep_fn(x0, y0, rgb.shape[0], rgb.shape[1])
                if mode == 'over':
                    core.blend(bg, core.Layer('over', rgb * k[..., None], a * k, box=(x0, y0)))
                else:
                    core.blend(bg, core.Layer('add', rgb * k[..., None], box=(x0, y0)))

    # ------------------------------------------------------------------ hooks
    def background(self, S):
        t = S.t
        S.meta['cleanup_lights'] = []
        if t < T_SHATTER0:
            bg, lights = self.bathroom(S)
            S.bg = bg
            S.meta['cleanup_lights'] = lights
            return
        if self.studio is None:
            self.studio = Studio()
        st = self.studio
        lights = []
        if t < T_STUDIO_END + 0.1:
            # ---------------- bathroom -> black studio
            if self.shatter is None:
                self.shatter = Shatter()
            sh = self.shatter
            sh.build()
            bath, _ = self.bathroom(S, with_fx=False)
            rgb, a = sh.shards(t, bath)
            bg = st.backdrop(t)
            bg *= (1.0 - a)[..., None]
            bg += rgb
            R = float(front_R(t))
            O = ORIGIN

            def keep(x0, y0, h, w):
                if R <= 0:
                    return np.ones((h, w), np.float32)
                X, Y = core.grid(h, w, (x0, y0))
                r = np.sqrt((X - O[0]) ** 2 + (Y - O[1]) ** 2)
                return sstep(R * 0.9 - 40, R * 0.9 + 20, r)
            self.eraser_particles_masked(bg, t, keep)
            ring = sh.ring(t)
            if ring is not None:
                bg += ring
            prt, voidp = sh.particles(t)
            if voidp is not None:
                bg += voidp * (1.0 - a)[..., None]
            if prt is not None:
                bg += prt
            S.bg = bg
            lights += sh.lights(t, S.alpha)
            lights += st.lights(t)
            # person: lit by the room lamp (flicker / pop), then by the studio
            rl = sh.room_light(t)
            bath_exp = (0.45 + 0.55 * rl) * (1.0 - 0.22 * sstep_f(13.4, 14.2, t))
            k = sstep_f(14.30, 14.75, t)
            g = S.person_grade
            g['exposure'] = bath_exp * (1 - k) + GRADE['exposure'] * k
            g['contrast'] = 1.0 + (GRADE['contrast'] - 1.0) * k
            g['saturation'] = 1.0 + (GRADE['saturation'] - 1.0) * k
            g['tint'] = tuple(1.0 + (np.float32(GRADE['tint']) - 1.0) * k)
            S.vignette = 0.25 + 0.20 * sstep_f(13.4, 14.6, t)
        else:
            # ---------------- black studio
            S.bg = st.backdrop(t)
            lights += st.lights(t)
            g = S.person_grade
            g['exposure'] = GRADE['exposure']
            g['contrast'] = GRADE['contrast']
            g['saturation'] = GRADE['saturation']
            g['tint'] = GRADE['tint']
            S.vignette = 0.45
        S.meta['cleanup_lights'] = lights
        # quarter-res haze density (0..1) for other modules that want their light to catch it
        if t >= 14.3:
            S.meta['studio_haze_q'] = st.haze_q(t)

    def lights(self, S):
        # rims are applied by our own thin-band rim in person(); the compositor gets the
        # diffuse wash only (its rim model fogs dark hair)
        out = []
        for L in S.meta.get('cleanup_lights', []):
            d = dict(L)
            d['rim'] = 0.0
            if d.get('diffuse', 0.0) > 0.0:
                out.append(core.Light(**d))
        return out

    _G4 = None

    @staticmethod
    def _hair_mask(i, x0, y0, h, w):
        """Soft mask of the hair (inside head_box, outside the dilated face oval, above the
        cheekbones) for a crop."""
        hb = A.head_box(i)
        fo = A.face_oval(i).astype(np.float32)
        m = np.zeros((h, w), np.uint8)
        hx0, hy0, hx1, hy1 = hb
        cv2.rectangle(m, (int(hx0 - x0 - 20), int(hy0 - y0 - 30)), (int(hx1 - x0 + 20), int(hy1 - y0)), 255, -1)
        cy = float(fo[:, 1].mean())
        m[max(0, int(cy - y0)):, :] = 0
        face = np.zeros((h, w), np.uint8)
        c = fo.mean(0)
        fo2 = c + (fo - c) * 1.06
        cv2.fillPoly(face, [np.round(fo2 - np.float32([x0, y0])).astype(np.int32)], 255)
        m[face > 0] = 0
        return cv2.GaussianBlur(m.astype(np.float32) / 255.0, (0, 0), 6.0)

    @staticmethod
    def apply_rims(S, lights):
        """Rim light on the silhouette: a crisp band ~4-5 px inside the edge, only on near-opaque
        matte, facing-weighted (facing^2). Two terms: a non-albedo specular rim (light colour;
        boosted on the hair, which is where he must separate from black) + an albedo-scaled one."""
        L = [l for l in lights if l.get('rim', 0) * l.get('intensity', 0) > 1e-3]
        if not L:
            return
        a = S.alpha
        # person bbox from a coarse matte
        a8 = a[::8, ::8]
        ys, xs = np.where(a8 > 0.5)
        if len(ys) == 0:
            return
        x0, x1 = max(0, xs.min() * 8 - 16), min(W, xs.max() * 8 + 24)
        y0, y1 = max(0, ys.min() * 8 - 16), min(H, ys.max() * 8 + 24)
        ac = a[y0:y1, x0:x1]
        h, w = ac.shape
        d = cv2.distanceTransform((ac > 0.5).astype(np.uint8), cv2.DIST_L2, 3)
        band = np.exp(-np.maximum(d - 1.0, 0.0) / 1.7) * sstep(0.80, 0.96, ac)
        # (the frame border is not an edge: no zero pixels beyond it -> large d there)
        if band.max() <= 0:
            return
        w4, h4 = W // 4, H // 4
        if FX._G4 is None:
            X, Y = np.meshgrid((np.arange(w4) + 0.5) * 4, (np.arange(h4) + 0.5) * 4)
            FX._G4 = (X.astype(np.float32), Y.astype(np.float32))
        X, Y = FX._G4
        a4 = cv2.resize(a, (w4, h4), interpolation=cv2.INTER_AREA)
        an = cv2.GaussianBlur(a4, (0, 0), 2.5)
        gx = cv2.Sobel(an, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(an, cv2.CV_32F, 0, 1, ksize=3)
        gm = np.sqrt(gx * gx + gy * gy) + 1e-6
        nx, ny = -gx / gm, -gy / gm
        field = np.zeros((h4, w4, 3), np.float32)
        for l in L:
            p = np.asarray(l['pos'], np.float32)
            dx, dy = p[0] - X, p[1] - Y
            dd = np.sqrt(dx * dx + dy * dy) + 1e-3
            fall = l['intensity'] / (1.0 + (dd / l['radius']) ** 2)
            facing = np.clip((nx * dx + ny * dy) / dd, 0, 1) ** 2
            field += (facing * fall * l['rim'])[..., None] * np.asarray(l['color'], np.float32)
        field = cv2.resize(field, (W, H), interpolation=cv2.INTER_LINEAR)[y0:y1, x0:x1]
        fc = S.fg[y0:y1, x0:x1]
        lum = (fc * np.float32([0.2126, 0.7152, 0.0722])).sum(-1)
        hair = FX._hair_mask(S.i, x0, y0, h, w)
        gain = 0.24 * (1.0 + 0.9 * hair) + 1.0 * np.clip(lum, 0, 0.45)
        S.fg = S.fg.copy() if S.fg is S.raw_fg else S.fg
        S.fg[y0:y1, x0:x1] = fc + field * (band * gain)[..., None]

    @staticmethod
    def edge_fix(S, k):
        """Dark-environment edge treatment for the soft matte edge (alpha ~0.2-0.9):
          * despill: edge luminance is clamped to the inner-body luminance (the pale bathroom
            wall fringe) and its hue pulled toward the inner colour (small kernel: no 'helmet');
          * light wrap: 1-2 px of the environment light (bars, pool) bleeds over the edge."""
        q = 2
        w, h = W // q, H // q
        a2 = cv2.resize(S.alpha, (w, h), interpolation=cv2.INTER_AREA)
        f2 = cv2.resize(S.fg, (w, h), interpolation=cv2.INTER_AREA)
        wt = sstep(0.92, 1.0, a2)
        num = cv2.GaussianBlur(f2 * wt[..., None], (0, 0), 2.2)
        den = cv2.GaussianBlur(wt, (0, 0), 2.2)[..., None]
        ext = np.where(den > 0.03, num / (den + 1e-4), f2)
        ext = cv2.resize(ext, (W, H), interpolation=cv2.INTER_LINEAR)
        a = S.alpha
        m = (sstep(0.08, 0.25, a) * (1.0 - sstep(0.86, 0.96, a)) * k)[..., None]
        lw = np.float32([0.2126, 0.7152, 0.0722])
        fg = S.fg
        lf = (fg * lw).sum(-1, keepdims=True)
        le = (ext * lw).sum(-1, keepdims=True)
        sc = np.minimum(1.0, (le * 1.04 + 0.002) / (lf + 1e-5))
        fd = fg * sc
        fh = ext * ((fd * lw).sum(-1, keepdims=True) / (le + 1e-5))
        fnew = fd * 0.45 + fh * 0.55
        out = fg * (1.0 - m) + fnew * m
        # the matte's long low-alpha tail (0.02-0.25, up to ~12 px outside the body) carries
        # the pale wall colour as a grey halo against black: pull it to the environment colour
        # (a visual choke; S.alpha itself is left untouched for the other modules)
        tail = ((1.0 - sstep(0.10, 0.32, a)) * (a > 0.002) * k)[..., None]
        out = out * (1.0 - tail) + S.bg * 1.0 * tail
        # light wrap from the environment
        bgb = cv2.GaussianBlur(cv2.resize(S.bg, (w, h), interpolation=cv2.INTER_AREA), (0, 0), 2.0)
        bgb = cv2.resize(bgb, (W, H), interpolation=cv2.INTER_LINEAR)
        wb = (sstep(0.15, 0.45, a) * (1.0 - sstep(0.70, 0.95, a)) * k)[..., None]
        S.fg = (out + bgb * wb * 0.9).astype(np.float32)

    def person(self, S):
        """Edge colour fix + our rim lights (thin band) + studio key-light falloff on him."""
        t = S.t
        k = sstep_f(13.40, 14.30, t)
        if k > 0:
            self.edge_fix(S, k)
        self.apply_rims(S, S.meta.get('cleanup_lights', []))
        k = sstep_f(14.30, 14.80, t)
        if k <= 0:
            return
        fo = A.face_oval(S.i)
        cx, cy = float(fo[:, 0].mean()), float(fo[:, 1].mean())
        q = 8
        X, Y = np.meshgrid((np.arange(W // q) + 0.5) * q, (np.arange(H // q) + 0.5) * q)
        g = np.exp(-0.5 * (((X - cx) / 420.0) ** 2 + ((Y - cy - 60.0) / 520.0) ** 2))
        f = 1.0 - k * 0.42 * (1.0 - g)
        f = cv2.resize(f.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
        S.fg = S.fg * f[..., None]

    _warned = False

    def _scene_lights(self, S):
        """All lights of all modules this frame (the compositor keeps them in a local list;
        read it from the caller frame if S.lights was not filled). Read-only, optional."""
        L = list(getattr(S, 'lights', None) or [])
        if not L:
            found = False
            try:
                fr = sys._getframe(2)
                for _ in range(4):
                    if fr is None:
                        break
                    cand = fr.f_locals.get('lights')
                    if isinstance(cand, list) and fr.f_locals.get('S') is S:
                        L = list(cand)
                        found = True
                        break
                    fr = fr.f_back
            except Exception:
                L = []
            if not found and not FX._warned:
                FX._warned = True
                print('[cleanup] could not read the scene light list (S.lights empty, stack read '
                      'failed): studio haze/dust react to cleanup lights only', file=sys.stderr)
            if found:
                return L
        if not L:
            L = [core.Light(**d) for d in S.meta.get('cleanup_lights', [])]
        return L

    def behind(self, S):
        t = S.t
        if t < 14.3 or self.studio is None:
            return []
        lights = self._scene_lights(S)
        out = []
        sc = self.studio.scatter(t, lights)
        if sc is not None:
            out.append(core.Layer('add', sc))
        mo = self.studio.motes(t, lights)
        if mo is not None:
            out.append(core.Layer('add', mo))
        return out
