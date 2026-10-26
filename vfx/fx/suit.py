"""suit — black high-tech nanotech suit that materialises on him at EV['suit'] and stays on.

Pipeline per frame (pure function of i):
  1. garment mask from tracking priors (suit_mask / suit_geom): matte - head - hands, the
     collar going up to the jaw; robust to turns, raised arms, hands over the face.
  2. maps: smoothed skin luminance (anatomy shading, small dark features removed), silhouette
     normals (rounded body profile for fresnel/spec).
  3. GLSL material (suit_shader): luminance-driven black satin/armour with micro-hex and twill
     weave anchored to a 2D skinned rest pose (no swimming), panel grooves, emissive seams,
     collar/cuff lips; nanotech materialisation front (hex cells assembling from the sternum
     along the skeleton) during EV['suit'].
  4. front(): emissive seams/front as an additive layer (+ sparks and inflowing nanite streaks
     during the materialisation); lights(): cyan interactive light during assembly/pulses.
"""
import numpy as np
import cv2

from .. import config as C
from .. import assets as A
from .. import core
from ..glfx import gl
from .suit_geom import Geometry
from . import suit_mask
from .suit_shader import FRAG

T0, T1 = C.EV['suit']
T_IGN = T0 + 0.18                      # ignition (core flash, shockwave)
EMI = np.array([0.40, 0.76, 1.00], np.float32)          # cool cyan-white seam colour
REST = dict(STERN_V=0.36, D_SH=0.62, L_UA=0.88, L_FA=0.80)


def _ss(a, b, x):
    x = min(max((x - a) / (b - a), 0.0), 1.0)
    return x * x * (3 - 2 * x)


def front_radius(ts):
    """Materialisation front radius (sw units, skeleton-geodesic from the sternum) vs seconds
    since T0.  Ease-out: bursts out of the chest, then crawls into the arms; the collar
    (delayed in the distance field) closes last, right before the settle beat."""
    x = min(max((ts - 0.17) / 1.66, 0.0), 1.0)
    e = 1.0 - (1.0 - x) ** 1.45
    return 1.86 * e + 0.6 * max(ts - 1.83, 0.0)


class FX(core.Effect):
    name = 'suit'

    def __init__(self):
        self.geo = Geometry()
        self.prog = None
        rng = np.random.default_rng(5150)
        # ---- spark / nanite particle tables (analytic trajectories) --------------------
        n_burst, n_front, n_in = 140, 520, 420
        sp = {}
        tb = np.concatenate([T0 + 0.17 + rng.random(n_burst) * 0.05,
                             T0 + 0.20 + np.sort(rng.random(n_front)) ** 1.15 * 1.75])
        sp['tb'] = tb
        sp['burst'] = np.r_[np.ones(n_burst, bool), np.zeros(n_front, bool)]
        n = len(tb)
        sp['region'] = rng.choice(5, size=n, p=[0.58, 0.12, 0.12, 0.09, 0.09])  # torso, uaR, uaL, faR, faL
        sp['theta'] = rng.random(n) * 2 * np.pi
        sp['b'] = (rng.random(n) - 0.5) * 0.24
        sp['speed'] = 0.6 + rng.random(n) ** 2 * 1.6           # sw / s
        sp['spread'] = (rng.random(n) - 0.5) * 1.6
        sp['life'] = 0.12 + rng.random(n) * 0.35
        sp['size'] = 0.7 + rng.random(n) * 0.9
        sp['bright'] = 2.0 + rng.random(n) * 5.0
        sp['vz'] = rng.random(n)
        self.sp = sp
        # inflow nanites: land on the front at t_land, fly in from outside the body
        ni = {}
        ni['tl'] = T0 + 0.25 + np.sort(rng.random(n_in)) * 1.70
        ni['region'] = rng.choice(5, size=n_in, p=[0.56, 0.13, 0.13, 0.09, 0.09])
        ni['theta'] = rng.random(n_in) * 2 * np.pi
        ni['b'] = (rng.random(n_in) - 0.5) * 0.24
        ni['dist'] = 0.25 + rng.random(n_in) * 0.55             # sw
        ni['ang'] = (rng.random(n_in) - 0.5) * 2.2
        ni['fly'] = 0.18 + rng.random(n_in) * 0.22
        ni['bright'] = 1.2 + rng.random(n_in) * 2.5
        ni['curl'] = (rng.random(n_in) - 0.5) * 2.0
        self.ni = ni
        # anticipation: energy gathering into the sternum right before ignition
        ng = 90
        ga = {}
        ga['ta'] = T_IGN - rng.random(ng) * 0.03
        ga['dur'] = 0.10 + rng.random(ng) * 0.10
        ga['ang'] = rng.random(ng) * 2 * np.pi
        ga['r0'] = 0.35 + rng.random(ng) * 0.55               # sw
        ga['curl'] = np.sign(rng.random(ng) - 0.5) * (1.2 + rng.random(ng) * 1.6)
        ga['bright'] = 0.8 + rng.random(ng) * 1.6
        self.ga = ga
        # steady-state energy pulses travelling along the seams (every ~4.7 s after assembly)
        self.pulses = [T0 + 1.80] + list(np.arange(T1 + 2.6, C.DURATION, 4.7))

    # ------------------------------------------------------------------ timing
    def active(self, t):
        return t >= T0 - 1e-6

    def _anim(self, t):
        ts = t - T0
        an = dict(ts=ts, front=front_radius(ts) if ts < 2.4 else 9.0)
        an['fx'] = 1.0 - _ss(T1 - 0.05, T1 + 0.30, t) if ts < 2.6 else 0.0
        # seams: dim while assembling, power-on flicker at the end, steady after
        if ts < 1.70:
            seam, flick = 0.30, 1.0
        else:
            k = int((ts - 1.70) * C.FPS)
            pat = [0.2, 1.4, 0.3, 0.0, 1.1, 0.6, 1.25, 1.0]
            flick = pat[k] if k < len(pat) else 1.0
            seam = 0.30 + 0.25 * _ss(1.70, 2.0, ts)
        an['seam'], an['flick'] = seam, flick
        # travelling pulse
        pulse = -1.0
        for tp in self.pulses:
            if tp - 0.05 <= t <= tp + 0.9:
                pulse = (t - tp) * 3.6 - 0.1
                break
        an['pulse'] = pulse
        an['sweep'] = (-0.6 + (ts - 1.95) / 0.55 * 2.6) if 1.95 <= ts <= 2.5 else -1.0
        # core node: charge -> flash at ignition
        an['core'] = (_ss(0.0, 0.17, ts) ** 2) * (1.0 - _ss(0.20, 0.55, ts)) * 6.0 if ts < 0.6 else 0.0
        return an

    # ------------------------------------------------------------------ geometry helpers
    def _bones(self, i):
        g = self.geo
        P4 = g.pose[i]
        out = {}
        sw = suit_mask.body_scale(g, i)
        for side, (ish, iel, iwr) in (('R', (12, 14, 16)), ('L', (11, 13, 15))):
            sh, el, wr = g.arm(i, side)
            v_el, v_wr = float(P4[iel, 3]), float(P4[iwr, 3])
            hc = float(g.hand_w[side][i])
            w_ua = 0.25 + 0.75 * float(np.clip((v_el - 0.3) / 0.4, 0, 1))
            w_fa = w_ua * max(float(np.clip((v_wr - 0.25) / 0.4, 0, 1)), hc)
            w_fa = 0.15 + 0.85 * w_fa
            out[side] = (sh, el, wr, w_ua, w_fa)
        return out, sw

    def _torso_pt(self, o, ex, ey, sw, tu, tv):
        return o + ex * (tu * sw) + ey * (tv * sw)

    def _bone_pt(self, A_, B_, sw, a, b):
        d = B_ - A_
        Ld = np.linalg.norm(d) + 1e-6
        dirv = d / Ld
        L = max(Ld, 0.30 * sw)
        pr = np.array([-dirv[1], dirv[0]])
        return A_ + dirv * (a * L) + pr * (b * sw), dirv

    def _front_points(self, region, theta, b, R, bones, sw):
        """Points on the body where the skeleton-geodesic distance == R (vectorised)."""
        (sh0, el0, wr0, _, _), (sh1, el1, wr1, _, _) = bones['R'], bones['L']
        o = 0.5 * (sh0 + sh1)
        ex = (sh1 - sh0) / (np.linalg.norm(sh1 - sh0) + 1e-6)
        ey = np.array([-ex[1], ex[0]])
        n = len(R)
        pts = np.zeros((n, 2), np.float32)
        dirs = np.zeros((n, 2), np.float32)
        ok = np.zeros(n, bool)
        St, D_SH, L_UA, L_FA = REST['STERN_V'], REST['D_SH'], REST['L_UA'], REST['L_FA']
        for k in range(n):
            r = region[k]
            if r == 0:
                tu = R[k] * np.cos(theta[k])
                tv = St + R[k] * np.sin(theta[k]) / 1.15
                if abs(tu) > 0.47 or tv < -0.30 or tv > 1.9:
                    continue
                pts[k] = self._torso_pt(o, ex, ey, sw, tu, tv)
                dv = ex * np.cos(theta[k]) + ey * np.sin(theta[k])
                dirs[k] = dv
                ok[k] = True
            else:
                if r in (1, 2):
                    A_, B_ = (sh0, el0) if r == 1 else (sh1, el1)
                    a = (R[k] - D_SH - abs(b[k]) * 0.6) / L_UA
                else:
                    A_, B_ = (el0, wr0) if r == 3 else (el1, wr1)
                    a = (R[k] - D_SH - L_UA - abs(b[k]) * 0.6) / L_FA
                if a < 0.0 or a > 1.0:
                    continue
                p, dv = self._bone_pt(A_, B_, sw, a, b[k])
                pts[k] = p
                dirs[k] = dv
                ok[k] = True
        return pts, dirs, ok

    # ------------------------------------------------------------------ main hook
    def person(self, S):
        i, t = S.i, S.t
        # Earlier person hooks may (a) recolour S.fg (cleanup: studio key falloff, rims) - we
        # build the suit from that colour so the studio lighting carries over - or (b) warp the
        # person (ceiling_portal suction publishes S.meta['person_warp']).  Only (b) changes
        # the matte object, so that is the warp test.
        maps = S.meta.get('person_warp')
        warped = (S.alpha is not S.raw_alpha) or (maps is not None)
        if S.meta.get('person_hidden') and float(S.alpha[::16, ::16].max()) < 0.01:
            return
        if warped:
            fg_src, a_src = S.raw_fg, S.raw_alpha
        else:
            fg_src, a_src = S.fg, S.alpha
        res = self._render_suit(i, t, fg_src, a_src)
        if res is None:
            return
        fg_c, emis_c, box = res
        x0, y0 = box
        h, w = fg_c.shape[:2]
        if not warped:
            S.fg = S.fg.copy() if S.fg is S.raw_fg else S.fg
            S.fg[y0:y0 + h, x0:x0 + w] = fg_c
            S.meta['suit_emis'] = (emis_c, box)
            return
        # an earlier module already warped the person (e.g. suction): transfer the suit
        full = S.raw_fg.copy()
        full[y0:y0 + h, x0:x0 + w] = fg_c
        em = np.zeros(S.alpha.shape, np.float32)
        em[y0:y0 + h, x0:x0 + w] = emis_c
        if maps is not None:
            mx, my = maps
            full = cv2.remap(full, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            em = cv2.remap(em, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        else:
            M = _moment_affine(S.raw_alpha, S.alpha)
            if M is None:
                return
            full = cv2.warpAffine(_pad_colors(full, S.raw_alpha), M, (C.W, C.H), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)
            em = cv2.warpAffine(em, M, (C.W, C.H), flags=cv2.INTER_LINEAR)
        S.fg = full.astype(np.float32)
        ys, xs = np.where(em > 1e-4)
        if len(ys):
            bx0, by0, bx1, by1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
            S.meta['suit_emis'] = (em[by0:by1, bx0:bx1].copy(), (bx0, by0))

    def _render_suit(self, i, t, fg_full, a_full):
        an = self._anim(t)
        # body crop from the matte (quarter-res search)
        q = 8
        a_s = a_full[::q, ::q]
        ys, xs = np.where(a_s > 0.03)
        if len(ys) < 4:
            return None
        pad = 14
        x0 = max(0, int(xs.min() * q) - pad)
        y0 = max(0, int(ys.min() * q) - pad)
        x1 = min(C.W, int(xs.max() * q + q) + pad)
        y1 = min(C.H, int(ys.max() * q + q) + pad)
        w, h = x1 - x0, y1 - y0
        fg = np.ascontiguousarray(fg_full[y0:y1, x0:x1])
        al = np.ascontiguousarray(a_full[y0:y1, x0:x1])
        M = suit_mask.build(self.geo, i, (x0, y0, x1, y1), fg, al)
        sw = M['sw']
        # ---- smoothed skin luminance (anatomy) with small dark features closed away
        lum = fg[..., 0] * 0.2126 + fg[..., 1] * 0.7152 + fg[..., 2] * 0.0722
        lh = cv2.resize(lum, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        k = max(5, int(0.045 * sw) | 1)
        lh = cv2.morphologyEx(lh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        lh = cv2.GaussianBlur(lh, (0, 0), 2.0)
        Ls = cv2.resize(lh, (w, h), interpolation=cv2.INTER_LINEAR)
        # ---- silhouette normals (rounded body profile)
        qq = 4
        aq = cv2.resize(al, (max(1, w // qq), max(1, h // qq)), interpolation=cv2.INTER_AREA)
        dist = cv2.distanceTransform((aq > 0.5).astype(np.uint8), cv2.DIST_L2, 5)
        R0 = max(2.0, 0.17 * sw / qq)
        x = np.clip(dist / R0, 0, 1)
        hgt = np.sqrt(np.clip(1 - (1 - x) ** 2, 0, 1)) * R0
        hgt = cv2.GaussianBlur(hgt, (0, 0), 1.2)
        gx = cv2.Sobel(hgt, cv2.CV_32F, 1, 0, ksize=3) / 8.0
        gy = cv2.Sobel(hgt, cv2.CV_32F, 0, 1, ksize=3) / 8.0
        nz = 1.0 / np.sqrt(1 + gx * gx + gy * gy)
        nrm = np.dstack([-gx * nz, -gy * nz]).astype(np.float32)
        nrm = cv2.resize(nrm, (w, h), interpolation=cv2.INTER_LINEAR)
        tex_n = np.dstack([nrm, al]).astype(np.float32)
        tex_aux = np.dstack([M['garment'], Ls, M['head_trim'], M['hands']]).astype(np.float32)
        # ---- uniforms
        bones, _ = self._bones(i)
        sh0, el0, wr0, wu0, wf0 = bones['R']
        sh1, el1, wr1, wu1, wf1 = bones['L']
        ff = M['face']
        if self.prog is None:
            self.prog = gl.program(FRAG)
        uni = dict(uSh0=tuple(map(float, sh0)), uSh1=tuple(map(float, sh1)),
                   uEl0=tuple(map(float, el0)), uEl1=tuple(map(float, el1)),
                   uWr0=tuple(map(float, wr0)), uWr1=tuple(map(float, wr1)),
                   uBoneW=(float(wu0), float(wf0), float(wu1), float(wf1)),
                   uSW=float(sw), uFaceC=tuple(map(float, ff['c'])), uFaceUp=tuple(map(float, ff['ey'])),
                   uFH=float(ff['fh']), uFront=float(an['front']), uFX=float(an['fx']),
                   uSeam=float(an['seam']), uFlick=float(an['flick']), uPulse=float(an['pulse']),
                   uSweep=float(an['sweep']), uT=float(t))
        out = gl.render(self.prog, (w, h), uniforms=uni,
                        textures={'uFg': fg, 'uAux': tex_aux, 'uNrm': tex_n}, offset=(x0, y0))
        self._last = dict(i=i, M=M, bones=bones, sw=sw, box=(x0, y0), size=(w, h), an=an)
        return out[..., :3].copy(), out[..., 3].copy(), (x0, y0)

    # ------------------------------------------------------------------ ignition shockwave
    def behind(self, S):
        t = S.t
        age = t - T_IGN
        if not (0.0 <= age <= 0.50):
            return []
        bones, sw = self._bones(S.i)
        sh0, sh1 = bones['R'][0], bones['L'][0]
        o = 0.5 * (sh0 + sh1)
        ex = (sh1 - sh0) / (np.linalg.norm(sh1 - sh0) + 1e-6)
        ey = np.array([-ex[1], ex[0]])
        c = S.tx(o + ey * (0.56 * sw))
        u = age / 0.50
        r = 1500.0 * (1 - (1 - u) ** 1.8) * 0.62
        wd = 14.0 + 26.0 * u
        amp = 9.0 * (1 - u) ** 1.6
        R = int(r + 3 * wd) + 2
        x0, y0 = int(c[0]) - R, int(c[1]) - R
        x0c, y0c = max(0, x0), max(0, y0)
        x1c, y1c = min(C.W, x0 + 2 * R), min(C.H, y0 + 2 * R)
        if x1c <= x0c or y1c <= y0c:
            return []
        X, Y = core.grid(y1c - y0c, x1c - x0c, (x0c, y0c))
        dx, dy = X - c[0], Y - c[1]
        d = np.sqrt(dx * dx + dy * dy) + 1e-3
        z = (d - r) / wd
        ring = np.exp(-z * z)
        prof = ring * z * amp                                # refractive lens profile
        S.distort.append((x0c, y0c, (dx / d * prof).astype(np.float32), (dy / d * prof).astype(np.float32)))
        glow = (ring * (0.045 * (1 - u) ** 2.5))[..., None] * EMI[None, None, :]
        return [core.Layer('add', glow.astype(np.float32), box=(x0c, y0c))]

    # ------------------------------------------------------------------ light
    def lights(self, S):
        t = S.t
        an = self._anim(t)
        L = []
        bones, sw = self._bones(S.i)
        sh0, sh1 = bones['R'][0], bones['L'][0]
        o = 0.5 * (sh0 + sh1)
        ex = (sh1 - sh0) / (np.linalg.norm(sh1 - sh0) + 1e-6)
        ey = np.array([-ex[1], ex[0]])
        stern = o + ey * (0.50 * sw)
        if an['ts'] < 2.6:
            act = an['fx'] * _ss(0.12, 0.30, an['ts']) * (1.0 - _ss(1.55, 2.25, an['ts']))
            core_i = an['core']
            if core_i > 0.01:
                L.append(core.Light(S.tx(stern), EMI, intensity=core_i * 0.45, radius=0.75 * sw,
                                    rim=0.8, diffuse=0.9, bg=0.02))
            R = an['front']
            if act > 0.01:
                for ang in (-0.35, np.pi + 0.35, -np.pi / 2):
                    tu = np.clip(R * np.cos(ang), -0.5, 0.5)
                    tv = np.clip(REST['STERN_V'] + R * np.sin(ang) / 1.15, -0.25, 1.4)
                    p = self._torso_pt(o, ex, ey, sw, tu, tv)
                    L.append(core.Light(S.tx(p), EMI, intensity=0.45 * act, radius=0.55 * sw,
                                        rim=0.6, diffuse=0.6, bg=0.015))
        if an['pulse'] > -0.5:
            k = np.exp(-((an['pulse'] - 0.6) / 0.8) ** 2)
            L.append(core.Light(S.tx(stern), EMI, intensity=0.20 * k, radius=0.7 * sw, rim=0.2,
                                diffuse=0.5, bg=0.0))
        return L

    # ------------------------------------------------------------------ emissive + particles
    def front(self, S):
        em = S.meta.get('suit_emis')
        if em is None:
            return []
        E, (x0, y0) = em
        ox, oy = float(S.person_offset[0]), float(S.person_offset[1])
        layers = []
        if E.max() > 1e-4:
            Eb = E[..., None]
            rgb = Eb * EMI[None, None, :] + np.clip(Eb - 1.0, 0, None) * 0.55
            # soft halo so the seams read at phone size (bloom adds the rest)
            halo = cv2.GaussianBlur(E, (0, 0), 3.0)[..., None] * EMI[None, None, :] * 0.35
            rgb = (rgb + halo).astype(np.float32)
            bx, by = x0 + ox, y0 + oy
            ix, iy = int(np.floor(bx)), int(np.floor(by))
            fx, fy = bx - ix, by - iy
            if fx > 1e-3 or fy > 1e-3:
                Mt = np.float32([[1, 0, fx], [0, 1, fy]])
                rgb = cv2.warpAffine(rgb, Mt, (rgb.shape[1] + 1, rgb.shape[0] + 1), flags=cv2.INTER_LINEAR)
            layers.append(core.Layer('add', rgb, box=(ix, iy)))
        t = S.t
        if T0 - 0.01 <= t <= T1 + 0.6 and getattr(self, '_last', None) is not None and self._last['i'] == S.i:
            pl = self._particles(S)
            if pl is not None:
                layers.append(pl)
        return layers

    def _particles(self, S):
        t = S.t
        last = self._last
        bones, sw = last['bones'], last['sw']
        M = last['M']
        x0, y0 = last['box']
        w, h = last['size']
        gar = M['garment']
        p0s, p1s, rads, cols = [], [], [], []
        # ---- sparks thrown off the front
        sp = self.sp
        age = t - sp['tb']
        live = (age >= 0) & (age <= sp['life'])
        idx = np.where(live)[0]
        if len(idx):
            R = np.array([front_radius(tb - T0) for tb in sp['tb'][idx]])
            R[sp['burst'][idx]] = 0.03
            pts, dirs, ok = self._front_points(sp['region'][idx], sp['theta'][idx], sp['b'][idx], R, bones, sw)
            for j, k in enumerate(idx):
                if not ok[j] and not sp['burst'][k]:
                    continue
                if sp['burst'][k]:
                    sh0, sh1 = bones['R'][0], bones['L'][0]
                    o = 0.5 * (sh0 + sh1)
                    ex = (sh1 - sh0) / (np.linalg.norm(sh1 - sh0) + 1e-6)
                    ey = np.array([-ex[1], ex[0]])
                    base = o + ey * (0.555 * sw)
                    th = sp['theta'][k]
                    d = np.array([np.cos(th), np.sin(th)])
                    spd = sp['speed'][k] * 1.9
                else:
                    base = pts[j]
                    xi, yi = int(base[0] - x0), int(base[1] - y0)
                    if not (0 <= xi < w and 0 <= yi < h) or gar[yi, xi] < 0.5:
                        continue
                    d = dirs[j]
                    c, s_ = np.cos(sp['spread'][k]), np.sin(sp['spread'][k])
                    d = np.array([d[0] * c - d[1] * s_, d[0] * s_ + d[1] * c])
                    spd = sp['speed'][k]
                a_ = age[k]
                v = d * spd * sw
                g = np.array([0.0, 2.2 * sw])
                drag = np.exp(-3.0 * a_)
                pos = base + v * (1 - drag) / 3.0 + 0.5 * g * a_ * a_
                a2 = max(a_ - 1.0 / 50, 0)
                drag2 = np.exp(-3.0 * a2)
                prev = base + v * (1 - drag2) / 3.0 + 0.5 * g * a2 * a2
                life = sp['life'][k]
                fade = (1 - a_ / life) ** 1.5 * min(1.0, a_ / 0.02 + 0.3)
                br = sp['bright'][k] * fade
                hot = 0.5 + 0.5 * sp['vz'][k]
                p0s.append(S.tx(prev)); p1s.append(S.tx(pos)); rads.append(sp['size'][k])
                cols.append((EMI[0] * br + hot * br * 0.5, EMI[1] * br + hot * br * 0.3, EMI[2] * br, 1.0))
        # ---- inflowing nanite streaks (converge onto the front where it is about to pass)
        ni = self.ni
        tl = ni['tl']
        tau = tl - t
        live = (tau > -0.05) & (tau < ni['fly'])
        idx = np.where(live)[0]
        if len(idx):
            R = np.array([front_radius(x - T0) for x in tl[idx]])
            pts, dirs, ok = self._front_points(ni['region'][idx], ni['theta'][idx], ni['b'][idx], R, bones, sw)
            for j, k in enumerate(idx):
                if not ok[j]:
                    continue
                base = pts[j]
                xi, yi = int(base[0] - x0), int(base[1] - y0)
                if not (0 <= xi < w and 0 <= yi < h) or gar[yi, xi] < 0.5:
                    continue
                d = dirs[j]
                c, s_ = np.cos(ni['ang'][k]), np.sin(ni['ang'][k])
                d = np.array([d[0] * c - d[1] * s_, d[0] * s_ + d[1] * c])
                fly = ni['fly'][k]

                def at(tt):
                    u = np.clip(tt / fly, 0, 1)
                    e = u * u                                  # accelerate into the body
                    r = ni['dist'][k] * sw * e
                    ang = ni['curl'][k] * e
                    cc, ss = np.cos(ang), np.sin(ang)
                    dd = np.array([d[0] * cc - d[1] * ss, d[0] * ss + d[1] * cc])
                    return base + dd * r
                ta = max(tau[k], 0.0)
                pos = at(ta)
                prev = at(ta + 1.0 / 40)
                u = 1 - ta / fly
                br = ni['bright'][k] * (u ** 1.5) * (1.0 if tau[k] >= 0 else np.exp(tau[k] / 0.02))
                p0s.append(S.tx(prev)); p1s.append(S.tx(pos)); rads.append(0.9)
                cols.append((EMI[0] * br, EMI[1] * br, EMI[2] * br, 1.0))
        # ---- anticipation: streaks spiralling into the sternum before ignition
        ga = self.ga
        sh0, sh1 = bones['R'][0], bones['L'][0]
        o = 0.5 * (sh0 + sh1)
        exv = (sh1 - sh0) / (np.linalg.norm(sh1 - sh0) + 1e-6)
        eyv = np.array([-exv[1], exv[0]])
        stern = o + eyv * (0.56 * sw)
        for k in range(len(ga['ta'])):
            tau = ga['ta'][k] - t
            if not (-0.01 < tau < ga['dur'][k]):
                continue

            def at(tt):
                u = np.clip(tt / ga['dur'][k], 0, 1)
                rr = ga['r0'][k] * sw * u ** 1.6
                an = ga['ang'][k] + ga['curl'][k] * u
                return stern + np.array([np.cos(an), np.sin(an)]) * rr
            ta = max(tau, 0.0)
            pos, prev = at(ta), at(ta + 1.0 / 110)
            u = 1 - ta / ga['dur'][k]
            br = ga['bright'][k] * u ** 1.2
            p0s.append(S.tx(prev)); p1s.append(S.tx(pos)); rads.append(0.9)
            cols.append((EMI[0] * br, EMI[1] * br, EMI[2] * br, 1.0))
        if not p0s:
            return None
        pad = int(0.9 * sw)
        ox, oy = float(S.person_offset[0]), float(S.person_offset[1])
        bx0, by0 = int(x0 + ox) - pad, int(y0 + oy) - pad
        bw, bh = w + 2 * pad, h + 2 * pad
        img = gl.sprites((bw, bh), np.array(p0s), np.array(p1s), np.array(rads, np.float32),
                         np.array(cols, np.float32), offset=(bx0, by0), shape=2)
        return core.Layer('add', img[..., :3].copy(), box=(bx0, by0))


# ---------------------------------------------------------------------- warp fallbacks
def _moment_affine(a_src, a_dst):
    """Affine mapping that matches the 1st/2nd moments of two mattes (used only when an
    earlier module warped the person without publishing its maps). Returns 2x3 for
    cv2.warpAffine (src -> dst)."""
    q = 4
    def mom(a):
        s = a[::q, ::q].astype(np.float64)
        m = s.sum()
        if m < 20:
            return None
        ys, xs = np.mgrid[0:s.shape[0], 0:s.shape[1]]
        xs = xs * q
        ys = ys * q
        cx, cy = (s * xs).sum() / m, (s * ys).sum() / m
        cxx = (s * (xs - cx) ** 2).sum() / m
        cyy = (s * (ys - cy) ** 2).sum() / m
        cxy = (s * (xs - cx) * (ys - cy)).sum() / m
        return np.array([cx, cy]), np.array([[cxx, cxy], [cxy, cyy]])
    a, b = mom(a_src), mom(a_dst)
    if a is None or b is None:
        return None
    def sqrtm(Cm):
        w_, v_ = np.linalg.eigh(Cm)
        return v_ @ np.diag(np.sqrt(np.maximum(w_, 1e-6))) @ v_.T
    Ms = sqrtm(b[1]) @ np.linalg.inv(sqrtm(a[1]))
    t_ = b[0] - Ms @ a[0]
    return np.hstack([Ms, t_[:, None]]).astype(np.float32)


def _pad_colors(img, alpha, iters=6):
    """Extend foreground colours outward so warped edges never pull in black."""
    out = img.copy()
    m = (alpha > 0.5).astype(np.float32)
    acc = out * m[..., None]
    wsum = m.copy()
    for _ in range(iters):
        acc = cv2.blur(acc, (9, 9))
        wsum = cv2.blur(wsum, (9, 9))
    fill = acc / np.maximum(wsum, 1e-4)[..., None]
    return np.where(m[..., None] > 0, out, fill).astype(np.float32)
