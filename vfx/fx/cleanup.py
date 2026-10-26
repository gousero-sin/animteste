"""cleanup — bathroom clean-up ("magic eraser"), bathroom -> black studio, black studio.

Owns the background from 0 s until the tunnel (EV['tunnel'][0]).
  * bathroom   : clean plates A.plate(0..3) with a low-frequency shadow/exposure transfer from
                 the raw frame (keeps his real shadow / phone exposure drift), three magic-eraser
                 removals synced to EV['clean_clutter'|'clean_flush'|'clean_shower'].
  * transform  : EV['studio'] - lamp flicker, energy core behind him, an expanding electric
                 front; the bathroom cracks into 3D shards that tumble back into the void;
                 then the studio lights strike on.
  * studio     : near-black seamless backdrop, soft pool behind him, drifting haze, lit dust
                 motes, rim + hair lights, person grade; haze scatters every light in the scene.
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
                   trace_lead=0.30, trace_dur=0.26, stagger=0.55, edge_w=0.05, n_part=2000,
                   seed=11, noise_amp=0.22, light_gain=1.0),
            Eraser('flush', 1, 2, 'flush', 'flush', ign_t=10.85, ign_side='R',
                   fallback_ign=(504, 1049), burn_t0=f0 + 0.18, burn_t1=f1 - 0.10,
                   trace_lead=0.20, trace_dur=0.16, stagger=0.5, edge_w=0.045, n_part=450,
                   seed=23, noise_amp=0.25, light_gain=0.8),
            Eraser('shower', 2, 3, 'shower_region', 'shower', ign_t=11.5, ign_side='R',
                   fallback_ign=(382, 921), burn_t0=s0 + 0.22, burn_t1=s1 - 0.02,
                   trace_lead=0.24, trace_dur=0.22, stagger=0.5, edge_w=0.045, n_part=2600,
                   seed=37, noise_amp=0.18, light_gain=1.1),
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
        f = cv2.resize(A.frame(i), (w, h), interpolation=cv2.INTER_AREA).mean(-1)
        a = cv2.resize(A.alpha(i), (w, h), interpolation=cv2.INTER_AREA)
        valid = (cv2.dilate((a > 0.02).astype(np.uint8), np.ones((11, 11), np.uint8)) == 0).astype(np.float32)
        valid *= self._objfree
        p = self._p0_small
        num = cv2.GaussianBlur(f * valid, (0, 0), 3.0)
        den = cv2.GaussianBlur(p * valid, (0, 0), 3.0)
        wv = cv2.GaussianBlur(valid, (0, 0), 3.0)
        num2 = cv2.GaussianBlur(f * valid, (0, 0), 18.0)
        den2 = cv2.GaussianBlur(p * valid, (0, 0), 18.0)
        r1 = num / (den + 1e-4)
        r2 = num2 / (den2 + 1e-4)
        k = np.clip(wv / 0.6, 0, 1)
        r = r1 * k + r2 * (1 - k)
        # low-confidence where the plate is dark (black shower panel): use the wide estimate
        conf = np.clip(den / (wv * 0.25 + 1e-4), 0, 1)
        r = r * conf + r2 * (1 - conf)
        r = np.clip(r, 0.55, 1.18)
        return cv2.resize(r, (W, H), interpolation=cv2.INTER_LINEAR)

    def bathroom(self, S, with_fx=True):
        """Bathroom background for frame S.i (plates + erasers + gain). Returns (bg, lights)."""
        t = S.t
        done = [t > e.t_done for e in self.erasers]
        k = 3 if done[2] else 2 if done[1] else 1 if done[0] else 0
        bg = A.plate(k).copy()
        emis = []
        lights = []
        for e in self.erasers:
            if not with_fx:
                break
            if e.t_first - 0.6 <= t <= e.t_part_end + 0.1:
                e.build()
            if e._built and e.crop_active(t):
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
                if e._built and e.any_active(t):
                    tr = e.trace_layer(t)
                    if tr is not None:
                        core.blend(bg, core.Layer('add', tr, box=e.box[:2]))
                    add, over = e.particles(t)
                    if over is not None:
                        core.blend(bg, core.Layer('over', over[0], over[1], box=over[2]))
                    if add is not None:
                        core.blend(bg, core.Layer('add', add[0], box=add[2]))
        return bg, lights

    # ------------------------------------------------------------------ hooks
    def background(self, S):
        t = S.t
        S.meta['cleanup_lights'] = []
        if t < T_SHATTER0:
            bg, lights = self.bathroom(S)
            S.bg = bg
            S.meta['cleanup_lights'] = lights
            return

    def lights(self, S):
        return [core.Light(**L) for L in S.meta.get('cleanup_lights', [])]
