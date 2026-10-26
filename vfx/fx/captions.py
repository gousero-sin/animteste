"""captions -- kinetic typography captions living BEHIND his head (whole video).

Design
------
* Every spoken word (config.WORDS, Whisper word ends trimmed with the audio RMS envelope) is
  grouped into short phrases (1-3 words). Words appear exactly when spoken.
* The text lives in a "slot" just behind the crown of his head: the baseline sits ~40 % of the
  cap height below the top of the hair, so his head/hair occludes the lower-centre of the
  letters while the upper part stays readable. Drawn in behind(): the compositor puts him in
  front. A soft contact occlusion (AO) darkens the text next to his silhouette for depth.
* Letters RISE out of the slot (from behind his head) with a damped spring, squash/stretch and
  true multi-sample motion blur; phrases that are replaced roll up and fade, phrases followed by
  silence sink back behind his head.
* Karaoke: the word being spoken is hot white (HDR, glows, sheen sweep); spoken words fall back
  to a cool off-white.  Keyword accents: CLAUDE (warm coral), PRETO (black + white rim), FOGO
  (animated fire + embers), GELO (ice + glints), FUSAO (violet plasma), FINALIZACAO (white-hot
  punch on the boom), PORTAL (teal/violet), DIMENSAO (chromatic split), ESPACO (star white),
  BURACO NEGRO (black letters with an accretion-orange rim), PORTAL / DR. ESTRANHO (gold +
  sling-ring sparks).  Glowing accents return core.Light -> they rim-light his hair from behind.
* Readability on the white tiles AND on black/space: two-scale soft drop shadow whose density
  adapts to the background luminance under the text, plus emissive glow that adapts inversely.
* Tunnel (no person): captions centred, words fly out of the vanishing point and zoom past the
  camera.  "E ME SUGA": the phrase is sucked up into the ceiling portal.
* Pure function of the frame index: head track is precomputed for all frames (cached), smoothed
  with non-causal Gaussian windows; particles are analytic; noise is seeded.
"""
import os
import json
import math

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

from .. import config as C
from .. import assets as A
from .. import core

W, H, FPS = C.W, C.H, C.FPS
EV = C.EV
CACHE = os.path.join(C.DATA, 'cache', 'captions')
FONT_PATH = os.path.join(C.FONTS, 'Anton-Regular.ttf')

# ---------------------------------------------------------------- phrase table (word indices)
PHRASES = [
    (0, 1, 2), (3, 4), (5, 6), (7,),                       # ESSE É UM | VÍDEO PARA | TESTAR O | CLAUDE
    (8, 9, 10), (11,), (12, 13),                           # LIMPA O FUNDO | AQUI | BASICAMENTE AGORA
    (14, 15), (16, 17), (18,), (19, 20, 21),               # LIMPA ESSE | NEGÓCIO DE | DESCARGA | E ESSE CHUVEIRO
    (22, 23, 24), (25, 26), (27, 28),                      # EU COLOCO O | MEU ESTÚDIO | TODO PRETO
    (29, 30, 31), (32, 33), (34,), (35, 36, 37),           # COLOCO UMA ROUPA | EM MIM | INTERNA | TÁ TUDO BEM
    (38, 39, 40), (41, 42),                                # E VAMOS FAZER | ALGUMAS COISAS
    (43, 44, 45), (46, 47), (48,),                         # BOLA DE FOGO | NESSA MÃO | DIREITA
    (49, 50, 51), (52, 53),                                # BOLA DE GELO | NESSA MÃO
    (54, 55), (56, 57), (58, 59),                          # A FUSÃO | DAS DUAS | UMA FINALIZAÇÃO
    (60, 61, 62), (63, 64, 65), (66,), (67, 68), (69, 70),  # TODAS AS MINHAS | FALAS DEVEM SER | LEGENDADAS | ATRÁS DA | MINHA CABEÇA
    (71, 72, 73), (74, 75), (76, 77), (78, 79), (80, 81),  # DE FORMA QUE | NÃO APAREÇA | ENTÃO AQUI | VOCÊ CONSEGUE | VER TUDO
    (82,), (83, 84), (85, 86), (87, 88), (89, 90),         # BASICAMENTE | NESSE MOMENTO | UM PORTAL | SURGE ACIMA | NO TETO
    (91, 92, 93),                                          # E ME SUGA
    (94, 95, 96), (97, 98), (99,),                         # E ME LEVA | PARA OUTRA | DIMENSÃO   (tunnel)
    (100, 101), (102,), (103, 104), (105, 106),            # NESSA OUTRA | DIMENSÃO | EU ESTOU | NO ESPAÇO
    (107, 108, 109), (110, 111),                           # AQUI TEM UM | BURACO NEGRO
    (112, 113), (114,), (115, 116, 117),                   # E AGORA | ACABOU | É SÓ ISSO
    (118,), (119, 120, 121), (122, 123, 124),              # FINALIZA | COM O PORTAL | DO DR. ESTRANHO
]

STYLE = {7: 'claude', 28: 'preto', 45: 'fire', 51: 'ice', 55: 'fusion', 59: 'blast',
         86: 'portal', 99: 'dim', 102: 'dim', 106: 'space', 110: 'bh', 111: 'bh',
         121: 'gold', 123: 'gold', 124: 'gold'}
EMPH = 1.16            # size factor of accent words inside a phrase

BASE_CAP = 150.0       # cap height (px) of normal words at depth scale 1
MAX_LINE = 0.88 * W    # widest line allowed (px at depth scale 1)
OVERLAP = 0.40         # fraction of the cap height hidden behind the crown of the head
CLIP_Y = 0.42          # slot line below the baseline (cap units) - letters rise out of it
RISE = 1.45            # entry travel (cap units)
SHUTTER = 0.55 / FPS   # motion-blur shutter (s)

# light colour of each style (normalised), base intensity
LIGHT = {'fire': ((1.0, 0.42, 0.10), 0.55), 'ice': ((0.35, 0.75, 1.0), 0.45),
         'fusion': ((0.75, 0.35, 1.0), 0.50), 'blast': ((1.0, 0.85, 0.95), 0.45),
         'portal': ((0.35, 0.65, 1.0), 0.40), 'dim': ((0.75, 0.45, 1.0), 0.35),
         'space': ((0.6, 0.75, 1.0), 0.25), 'bh': ((1.0, 0.55, 0.18), 0.40),
         'gold': ((1.0, 0.60, 0.18), 0.50), 'claude': ((1.0, 0.55, 0.35), 0.30),
         'preto': ((0.9, 0.9, 1.0), 0.10), 'white': ((1.0, 0.95, 0.88), 0.10)}


def _clean_word(w):
    w = w.strip()
    if w.lower() == 'dr.':
        return 'DR.'
    return w.strip(',.;:!?').upper()


def _atomic_save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f'{path}.{os.getpid()}.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f)
    os.replace(tmp, path)


# ---------------------------------------------------------------- speech timing (audio RMS)
def _speech_timing():
    """Trim Whisper's over-long word ends (and late-voiced starts) with the speech envelope."""
    path = os.path.join(CACHE, 'timing_v2.json')
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        pass
    words = C.WORDS
    S = np.array([w['s'] for w in words], np.float64)
    E = np.array([w['e'] for w in words], np.float64)
    db = None
    try:
        import scipy.io.wavfile as wavfile
        from scipy.signal import butter, sosfilt
        wav = os.path.join(C.DATA, 'audio', 'a16k.wav')
        if not os.path.exists(wav):
            import subprocess
            os.makedirs(CACHE, exist_ok=True)
            wav = os.path.join(CACHE, 'a16k.wav')
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', C.SRC_VIDEO, '-map', '0:a:0',
                            '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', wav], check=True)
        sr, x = wavfile.read(wav)
        x = x.astype(np.float32)
        if x.ndim > 1:
            x = x.mean(1)
        x /= 32768.0
        x = sosfilt(butter(4, [100, 4000], btype='band', fs=sr, output='sos'), x)
        hop = sr // 100
        n = len(x) // hop
        db = 20 * np.log10(np.sqrt((x[:n * hop].reshape(n, hop) ** 2).mean(1) + 1e-12))
    except Exception:
        db = None
    s_new, e_new = S.copy(), E.copy()
    for k in range(len(words)):
        s, e = S[k], E[k]
        if db is None:
            e_new[k] = min(e, s + 0.6)
            continue
        a, b = int(round(s * 100)), int(round(e * 100))
        v = np.where(db[a:b] > -43.0)[0]
        if len(v) == 0:
            e_new[k] = min(e, s + 0.35)
            continue
        first = a + v[0]
        vs = np.where(db[a:b] > -39.0)[0]    # onset: clearly voiced
        if len(vs) and vs[0] > 22:          # voice starts well after Whisper's start
            s_new[k] = (a + vs[0]) / 100.0 - 0.03
        last = first
        for j in v:
            if a + j - last <= 22:
                last = a + j
            else:
                break
        e_new[k] = min(e, last / 100.0 + 0.07)
        if e_new[k] - s_new[k] < 0.12:
            e_new[k] = min(e, s_new[k] + 0.12)
    out = dict(s=s_new.tolist(), e=e_new.tolist())
    try:
        _atomic_save_json(path, out)
    except Exception:
        pass
    return out


# ---------------------------------------------------------------- head track (all frames)
def _head_track():
    """Top-of-hair point per frame from the face mesh up-axis, calibrated against the matte.
    Returns dict(top=(N,2), flen=(N,), vx=(N,)) smoothed with non-causal Gaussian windows."""
    path = os.path.join(CACHE, 'head_v1.npz')
    try:
        d = np.load(path)
        return dict(top=d['top'], flen=d['flen'], vx=d['vx'])
    except Exception:
        pass
    from scipy.ndimage import gaussian_filter1d, median_filter
    N = C.N_FRAMES
    F = A._track()['face'].astype(np.float64)
    top10, chin = F[:, 10], F[:, 152]
    flen = np.linalg.norm(chin - top10, axis=1) + 1e-6
    up = (top10 - chin) / flen[:, None]
    k = np.full(N, np.nan)
    steps = np.arange(0, 1.2, 0.01)
    for i in range(N):
        a = cv2.imread(f'{C.ALPHA_LO}/{i:04d}.png', cv2.IMREAD_GRAYSCALE)
        if a is None:
            continue
        a = a.astype(np.float32) / 255.0
        p, u, fl = top10[i] / 3.0, up[i], flen[i] / 3.0
        perp = np.array([-u[1], u[0]])
        ss = steps * fl
        vals = np.zeros(len(ss))
        for o in (-0.12, 0.0, 0.12):
            q = p[None, :] + u[None, :] * ss[:, None] + perp[None, :] * (o * fl)
            xi = np.clip(np.round(q[:, 0]).astype(int), 0, 359)
            yi = np.clip(np.round(q[:, 1]).astype(int), 0, 479)
            inside = (q[:, 0] >= 0) & (q[:, 0] < 360) & (q[:, 1] >= 0) & (q[:, 1] < 480)
            vals += np.where(inside, a[yi, xi], 0.0)
        out = np.where(vals / 3.0 < 0.5)[0]
        if len(out):
            k[i] = steps[out[0]]
    valid = (k > 0.36) & (k < 0.56)
    idx = np.arange(N)
    kk = np.interp(idx, idx[valid], k[valid]) if valid.sum() > 10 else np.full(N, 0.44)
    kk = gaussian_filter1d(median_filter(kk, 15, mode='nearest'), 6, mode='nearest')
    flen_s = gaussian_filter1d(flen, 6, mode='nearest')
    up_s = gaussian_filter1d(up, 6, axis=0, mode='nearest')
    up_s /= np.linalg.norm(up_s, axis=1, keepdims=True) + 1e-9
    t10 = gaussian_filter1d(top10, 2.5, axis=0, mode='nearest')
    top = t10 + up_s * (flen_s * kk)[:, None]
    top = gaussian_filter1d(top, 4.0, axis=0, mode='nearest')
    vx = gaussian_filter1d(np.gradient(top[:, 0]), 3.0, mode='nearest')
    res = dict(top=top.astype(np.float32), flen=flen_s.astype(np.float32), vx=vx.astype(np.float32))
    try:
        os.makedirs(CACHE, exist_ok=True)
        tmp = f'{path}.{os.getpid()}.tmp.npz'
        np.savez(tmp, **res)
        os.replace(tmp, path)
    except Exception:
        pass
    return res


# ---------------------------------------------------------------- small math helpers
def _ss(a, b, x):
    x = min(max((x - a) / (b - a + 1e-9), 0.0), 1.0)
    return x * x * (3 - 2 * x)


def _clamp(x, a, b):
    return a if x < a else (b if x > b else x)


def _spring(tau, w0=2 * math.pi * 2.4, z=0.62):
    """Under-damped step response (0 -> 1, ~8 % overshoot) and its velocity (1/s)."""
    if tau <= 0:
        return 0.0, 0.0
    wd = w0 * math.sqrt(1 - z * z)
    e = math.exp(-z * w0 * tau)
    r = 1 - e * (math.cos(wd * tau) + z * w0 / wd * math.sin(wd * tau))
    v = (w0 * w0 / wd) * e * math.sin(wd * tau)
    return r, v


def _ramp(v, stops):
    """Piecewise-linear colour ramp. v (h,w) -> (h,w,3)."""
    xs = [s[0] for s in stops]
    out = np.empty(v.shape + (3,), np.float32)
    for c in range(3):
        out[..., c] = np.interp(v, xs, [s[1][c] for s in stops])
    return out


def _hash(*a):
    x = 0.0
    for k, v in enumerate(a):
        x += v * (12.9898 + 78.233 * k)
    return math.sin(x) * 43758.5453 % 1.0


def _mat(tx, ty, rot=0.0, sx=1.0, sy=1.0):
    c, s = math.cos(rot), math.sin(rot)
    return np.array([[c * sx, -s * sy, tx], [s * sx, c * sy, ty], [0, 0, 1]], np.float64)


def _T(x, y):
    return np.array([[1, 0, x], [0, 1, y], [0, 0, 1]], np.float64)


def _splat(img, off, p0, p1, rad, col):
    """Additive gaussian streak splats into img (h,w,3) whose top-left is at frame px `off`."""
    h, w = img.shape[:2]
    for k in range(len(p0)):
        r = float(max(rad[k], 0.6))
        a = np.array(p0[k], np.float64) - off
        b = np.array(p1[k], np.float64) - off
        pad = 3.5 * r + 2
        x0, x1 = int(min(a[0], b[0]) - pad), int(max(a[0], b[0]) + pad) + 1
        y0, y1 = int(min(a[1], b[1]) - pad), int(max(a[1], b[1]) + pad) + 1
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
        if x1 <= x0 or y1 <= y0:
            continue
        ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        d = b - a
        L2 = float(d @ d)
        if L2 < 1e-6:
            dx, dy = xs - a[0], ys - a[1]
        else:
            tt = np.clip(((xs - a[0]) * d[0] + (ys - a[1]) * d[1]) / L2, 0, 1)
            dx, dy = xs - (a[0] + tt * d[0]), ys - (a[1] + tt * d[1])
        q = (dx * dx + dy * dy) / (r * r)
        g = np.exp(-0.5 * q) + 0.12 * np.exp(-0.05 * q)
        # energy conservation for streaks: longer streak -> dimmer per pixel
        g *= 1.0 / (1.0 + math.sqrt(L2) / (2.5 * r))
        img[y0:y1, x0:x1] += g[..., None] * np.asarray(col[k], np.float32)


# ---------------------------------------------------------------- the effect
class FX(core.Effect):
    name = 'captions'

    def __init__(self):
        tm = _speech_timing()
        self.ws = np.array(tm['s'])
        self.we = np.array(tm['e'])
        self.text = [_clean_word(w['w']) for w in C.WORDS]
        self.track = _head_track()
        self.plan = self._make_plan()
        self.span = (min(p['show'] for p in self.plan), max(p['vis_end'] for p in self.plan))
        self._built = {}
        self._fonts = {}
        self._memo = (None, None)
        rng = np.random.default_rng(20261026)
        noise = np.zeros((256, 256), np.float32)
        amp, tot = 1.0, 0.0
        for gsz in (6, 12, 24, 48):           # tileable fbm value noise (periodic cubic upsampling)
            g = np.tile(rng.random((gsz, gsz)).astype(np.float32), (3, 3))
            g = cv2.resize(g, (768, 768), interpolation=cv2.INTER_CUBIC)[256:512, 256:512]
            noise += g * amp
            tot += amp
            amp *= 0.55
        noise /= tot
        noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-6)
        self.noise = np.ascontiguousarray(noise)

    # ------------------------------------------------------------ timeline
    def _make_plan(self):
        plan = []
        nP = len(PHRASES)
        shows = [self.ws[p[0]] - 0.06 for p in PHRASES]
        for pi, wids in enumerate(PHRASES):
            show = shows[pi]
            end_speech = float(self.we[wids[-1]])
            nxt = shows[pi + 1] if pi + 1 < nP else 1e9
            mode = 'normal'
            if EV['tunnel'][0] - 0.05 <= show < EV['tunnel'][1]:
                mode = 'tunnel'
            gap = nxt - end_speech
            if gap < 1.1:
                exit_mode, exit_start = 'up', max(nxt - 0.12, show + 0.35)
            else:
                hold = min(1.25, 0.30 * gap + 0.35)
                exit_mode, exit_start = 'sink', min(end_speech + hold, nxt - 0.3)
            exit_dur = 0.18 if exit_mode == 'up' else 0.32
            if 93 in wids:                                 # "E ME SUGA" -> sucked up
                exit_mode, exit_start, exit_dur = 'suck', max(end_speech + 0.05, 49.95), 0.62
            if mode == 'tunnel':
                exit_mode, exit_dur = 'zoom', 0.40
                exit_start = min(exit_start, EV['tunnel'][1] - 0.42) if pi + 1 < nP and shows[pi + 1] >= EV['tunnel'][1] else exit_start
            vis_end = exit_start + exit_dur + 0.25
            plan.append(dict(pi=pi, wids=wids, show=show, end_speech=end_speech, mode=mode,
                             exit_mode=exit_mode, exit_start=exit_start, exit_dur=exit_dur,
                             vis_end=vis_end))
        return plan

    def active(self, t):
        return self.span[0] <= t <= self.span[1] and any(
            p['show'] <= t <= p['vis_end'] for p in self.plan)

    # ------------------------------------------------------------ glyph building
    def _font(self, px):
        px = int(round(px))
        if px not in self._fonts:
            self._fonts[px] = ImageFont.truetype(FONT_PATH, px)
        return self._fonts[px]

    def _cap_ratio(self):
        f = self._font(200)
        bb = f.getbbox('H', anchor='ls')
        return (bb[3] - bb[1]) / 200.0 if bb[3] - bb[1] > 0 else 0.86

    def _build(self, pi):
        if pi in self._built:
            return self._built[pi]
        wids = PHRASES[pi]
        cr = self._cap_ratio()
        styles = [STYLE.get(k, 'white') for k in wids]
        facs = [EMPH if s != 'white' else 1.0 for s in styles]
        if len(wids) == 1:
            facs = [1.18 if styles[0] != 'white' else 1.08]
        texts = [self.text[k] for k in wids]

        def widths(cap):
            ws = [self._font(cap * f / cr).getlength(tx) for tx, f in zip(texts, facs)]
            sp = 0.20 * cap
            return ws, sp, sum(ws) + sp * (len(ws) - 1)

        cap = BASE_CAP
        ws_, sp, lw = widths(cap)
        if lw > MAX_LINE:
            cap *= MAX_LINE / lw
            ws_, sp, lw = widths(cap)
        glyphs, words = [], []
        x = 0.0
        li = 0
        for wi, (tx, f, st) in enumerate(zip(texts, facs, styles)):
            chw = cap * f
            font = self._font(chw / cr)
            x0w = x
            for kc, ch in enumerate(tx):
                gx = x0w + font.getlength(tx[:kc])
                adv = font.getlength(ch)
                bb = font.getbbox(ch, anchor='ls')
                if bb[2] - bb[0] <= 0:
                    continue
                pad = int(0.06 * chw) + 3
                gw = int(bb[2] - bb[0]) + 2 * pad
                gh = int(bb[3] - bb[1]) + 2 * pad
                ox, oy = pad - bb[0], pad - bb[1]          # glyph origin (baseline-left) in image px
                im = Image.new('L', (gw, gh), 0)
                ImageDraw.Draw(im).text((ox, oy), ch, font=font, fill=255, anchor='ls')
                m = np.asarray(im, np.float32) / 255.0
                r = max(2, int(round(0.032 * chw)))
                ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
                e = np.clip(m - cv2.erode(m, ker), 0, 1)
                yy = np.arange(gh, dtype=np.float32)[:, None]
                xx = np.arange(gw, dtype=np.float32)[None, :]
                v = (oy - yy) / chw + 0 * xx                # cap units above baseline
                u = (gx + xx - ox) / cap + 0 * yy           # phrase x in cap units
                img = np.ascontiguousarray(np.stack([m, e, m * v, m * u], -1).astype(np.float32))
                glyphs.append(dict(img=img, gx=gx, adv=adv, pivot=(ox + adv / 2.0, oy), w=wi,
                                   k=kc, li=li, style=st, chw=chw))
                li += 1
            wwid = font.getlength(tx)
            words.append(dict(x0=x0w, x1=x0w + wwid, style=st, chw=chw, wid=wids[wi], text=tx,
                              nch=len(tx)))
            x += wwid + sp
        ph = dict(glyphs=glyphs, words=words, lw=lw, cap=cap)
        self._built[pi] = ph
        return ph

    # ------------------------------------------------------------ anchor
    def _presence(self, t):
        if t < 51.0:
            return 1.0 - _ss(49.75, 50.45, t)
        return _ss(53.65, 54.40, t)

    def _anchor(self, S, ph):
        i = S.i
        tr = self.track
        top = tr['top'][i]
        flen = float(tr['flen'][i])
        sd = _clamp((flen / 211.0) ** 0.5, 0.88, 1.22)
        hx, hy = S.tx(top)
        p = self._presence(S.t)
        sc = sd * p + 1.0 * (1 - p)
        cap = ph['cap'] * sc
        ax = hx * p + (W / 2) * (1 - p)
        ay = (hy + OVERLAP * cap) * p + (H / 2 + 0.5 * cap) * (1 - p)
        half = ph['lw'] * sc / 2 + 30
        ax = W / 2 if half > W / 2 else _clamp(ax, half, W - half)
        tilt = math.radians(_clamp(-float(tr['vx'][i]) * 0.16, -2.2, 2.2)) * p
        return float(ax), float(ay), float(sc), tilt

    # ------------------------------------------------------------ animation
    def _glyph_xf(self, P, ph, g, t, frame_anchor):
        """Local animation of glyph g at time t -> (3x3 matrix img->frame, opacity, clip_y)."""
        ax, ay, sc, tilt = frame_anchor
        cap = ph['cap']
        wd = ph['words'][g['w']]
        wid = wd['wid']
        cx_l = ph['lw'] / 2.0
        piv_l = np.array([g['gx'] + g['adv'] / 2.0, 0.0])
        cvis = np.array([cx_l, -0.5 * cap])
        dx = dy = 0.0
        sx = sy = 1.0
        rot = 0.0
        op = 1.0
        clip = False
        ws = float(self.ws[wid])
        # ---- entry
        if P['mode'] == 'tunnel':
            tau = t - (ws - 0.06 + g['k'] * 0.012)
            if tau < 0:
                return None
            q = min(tau / 0.45, 1.0)
            z = 0.06 + 0.94 * float(core.ease_out_back(q, 1.2))
            d = (piv_l - cvis) * (z - 1.0)
            dx += d[0]
            dy += d[1]
            sx *= z
            sy *= z
            op *= _ss(0.0, 0.15, tau)
        else:
            tau = t - (ws - 0.05 + g['k'] * 0.022)
            if tau < 0:
                return None
            r, v = _spring(tau)
            dy += (1.0 - r) * RISE * cap
            st = _clamp(v / 12.0, 0.0, 1.0)
            sy *= 1.0 + 0.16 * st
            sx *= 1.0 - 0.05 * st
            op *= _ss(0.0, 0.06, tau)
            clip = tau < 0.75
        # ---- karaoke bump + idle float
        act = self._word_act(P, g['w'], t)
        bump = 1.0 + 0.035 * act
        sx *= bump
        sy *= bump
        dy += 1.6 * math.sin(2 * math.pi * 0.55 * t + 1.7 * g['w'] + 0.9 * P['pi'])
        # ---- finale punch
        if wd['style'] == 'blast' or P['pi'] == self._blast_pi():
            tb = t - EV['finale_boom']
            if tb >= 0:
                pz = 1.0 + 0.16 * math.exp(-tb * 7.0) * math.cos(tb * 22.0)
            else:
                pz = 1.0 - 0.05 * _ss(-0.35, 0.0, tb)
            d = (piv_l - cvis) * (pz - 1.0)
            dx += d[0]
            dy += d[1]
            sx *= pz
            sy *= pz
        # ---- exit
        te = t - P['exit_start']
        if te > -1e-6:
            em = P['exit_mode']
            n_l = len(ph['glyphs'])
            if em == 'up':
                q = _clamp((te - g['li'] * 0.006) / P['exit_dur'], 0.0, 1.0)
                dy -= 0.42 * cap * q * q
                sy *= 1.0 + 0.10 * q
                sx *= 1.0 - 0.04 * q
                op *= 1.0 - _ss(0.0, 0.85, q)
            elif em == 'sink':
                q = _clamp((te - (n_l - 1 - g['li']) * 0.014) / P['exit_dur'], 0.0, 1.0)
                dy += RISE * cap * q ** 2.4
                sy *= 1.0 + 0.10 * q
                op *= 1.0 - _ss(0.75, 1.0, q)
                clip = True
            elif em == 'suck':
                dc = abs(piv_l[0] - cx_l) / (cx_l + 1e-6)
                q = _clamp((te - 0.14 * dc) / P['exit_dur'], 0.0, 1.0)
                qq = q ** 2.2
                dy -= (ay / sc + 0.25 * H) * qq
                dx += (cx_l - piv_l[0]) * 0.65 * q ** 1.5
                sy *= 1.0 + 2.6 * q ** 1.6
                sx *= 1.0 - 0.45 * q
                op *= 1.0 - _ss(0.55, 1.0, q)
            elif em == 'zoom':
                q = _clamp(te / P['exit_dur'], 0.0, 1.0)
                z = 1.0 + 2.4 * q ** 2.2
                d = (piv_l - cvis) * (z - 1.0)
                dx += d[0]
                dy += d[1]
                sx *= z
                sy *= z
                op *= 1.0 - _ss(0.25, 1.0, q)
        if op <= 1e-3:
            return None
        # ---- matrices
        px, py = g['pivot']
        Lm = _mat(piv_l[0] + dx, piv_l[1] + dy, rot, sx, sy) @ _T(-px, -py)
        Pm = _T(ax + sc * (cvis[0] - cx_l), ay + sc * cvis[1]) @ _mat(0, 0, tilt, sc, sc) @ _T(-cvis[0], -cvis[1])
        M = Pm @ Lm
        clip_y = ay + sc * CLIP_Y * cap if clip else None
        return M, op, clip_y

    def _blast_pi(self):
        if not hasattr(self, '_bpi'):
            self._bpi = next(k for k, p in enumerate(PHRASES) if 59 in p)
        return self._bpi

    def _word_act(self, P, w, t):
        wids = P['wids']
        k = wids[w]
        s = float(self.ws[k])
        if w + 1 < len(wids):
            e = min(float(self.ws[wids[w + 1]]) - 0.02, float(self.we[k]) + 0.30)
        else:
            e = max(float(self.we[k]) + 0.30, s + 0.45)
        return _ss(s - 0.06, s + 0.02, t) * (1.0 - _ss(e - 0.05, e + 0.12, t))

    # ------------------------------------------------------------ per-frame state
    def _state(self, S):
        if self._memo[0] == S.i:
            return self._memo[1]
        t = S.t
        items, words = [], []
        for P in self.plan:
            if not (P['show'] <= t <= P['vis_end']):
                continue
            ph = self._build(P['pi'])
            fa = self._anchor(S, ph)
            for g in ph['glyphs']:
                xf = self._glyph_xf(P, ph, g, t, fa)
                if xf is None:
                    continue
                # motion blur samples
                ta, tb = t - SHUTTER / 2, t + SHUTTER / 2
                xa = self._glyph_xf(P, ph, g, ta, fa)
                xb = self._glyph_xf(P, ph, g, tb, fa)
                gh, gw = g['img'].shape[:2]
                corners = np.array([[0, 0, 1], [gw, 0, 1], [0, gh, 1], [gw, gh, 1]], np.float64).T
                disp = 0.0
                if xa is not None and xb is not None:
                    disp = float(np.abs((xa[0] @ corners - xb[0] @ corners)[:2]).max())
                n = int(_clamp(math.ceil(disp / 2.5), 1, 12))
                samples = []
                if n == 1:
                    samples.append(xf)
                else:
                    for j in range(n):
                        tj = ta + (tb - ta) * (j + 0.5) / n
                        xj = self._glyph_xf(P, ph, g, tj, fa)
                        if xj is not None:
                            samples.append(xj)
                    if not samples:
                        continue
                pts = np.concatenate([(s_[0] @ corners)[:2].T for s_ in samples], 0)
                items.append(dict(g=g, P=P, ph=ph, samples=samples, n=n, fa=fa,
                                  bbox=(pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()),
                                  act=self._word_act(P, g['w'], t), op=xf[1]))
            # word-level info (lights, particles)
            ax, ay, sc, tilt = fa
            cx_l = ph['lw'] / 2
            for wi, wd in enumerate(ph['words']):
                k = wd['wid']
                gl = [it for it in items if it['ph'] is ph and it['g']['w'] == wi]
                if not gl:
                    continue
                op = float(sum(it['op'] for it in gl)) / max(1, len([g_ for g_ in ph['glyphs'] if g_['w'] == wi]))
                bx0 = min(it['bbox'][0] for it in gl)
                by0 = min(it['bbox'][1] for it in gl)
                bx1 = max(it['bbox'][2] for it in gl)
                by1 = max(it['bbox'][3] for it in gl)
                words.append(dict(P=P, ph=ph, wd=wd, wi=wi, op=op, act=self._word_act(P, wi, t),
                                  rest=(ax + sc * (wd['x0'] - cx_l), ay - sc * wd['chw'],
                                        ax + sc * (wd['x1'] - cx_l), ay),
                                  bbox=(bx0, by0, bx1, by1), sc=sc))
        st = dict(items=items, words=words)
        self._memo = (S.i, st)
        return st

    # ------------------------------------------------------------ lights
    def lights(self, S):
        st = self._state(S)
        out = []
        for wv in st['words']:
            style = wv['wd']['style']
            col, inten = LIGHT.get(style, LIGHT['white'])
            if style == 'white':
                inten *= wv['act']
            else:
                inten *= 0.6 + 0.4 * wv['act']
            if style == 'fire':
                inten *= 0.85 + 0.15 * math.sin(S.t * 31.0) * math.sin(S.t * 17.3 + 1.0)
            inten *= wv['op']
            if inten < 0.01:
                continue
            x0, y0, x1, y1 = wv['rest']
            pos = ((x0 + x1) / 2, (y0 + y1) / 2)
            out.append(core.Light(pos, col, intensity=inten, radius=380.0 * wv['sc'],
                                  rim=1.6, diffuse=0.10, bg=0.0))
        return out

    # ------------------------------------------------------------ fills
    def _noise(self, u, v, su, sv, ou, ov):
        mx = ((u * su + ou) * 64.0) % 256.0
        my = ((v * sv + ov) * 64.0) % 256.0
        return cv2.remap(self.noise, mx.astype(np.float32), my.astype(np.float32),
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

    def _fill(self, style, v, u, ef, t, act, wd, P, cap_u0, cap_u1):
        """Straight (non-premultiplied) HDR fill colour and emissive colour for a glyph region."""
        if style in ('white', 'suck'):
            base = _ramp(v, [(0.0, (0.76, 0.79, 0.88)), (1.0, (1.0, 1.0, 1.0))])
            past = np.array([0.80, 0.81, 0.86], np.float32)
            hot = np.array([1.30, 1.24, 1.08], np.float32)
            col = base * (past + (hot - past) * act)
            # entry sheen sweep across the active word
            ws = float(self.ws[wd['wid']])
            ph_ = (t - ws) / 0.42
            if -0.2 < ph_ < 1.3:
                pos = cap_u0 - 0.6 + (cap_u1 - cap_u0 + 1.2) * ph_
                band = np.exp(-((u + 0.45 * v) - pos) ** 2 / 0.05)
                col = col + band[..., None] * np.array([0.7, 0.66, 0.55], np.float32) * act
            emis = col * (0.22 * act)
        elif style == 'fire':
            n = self._noise(u, v, 0.9, 0.9, 0.0, -t * 1.9)
            n2 = self._noise(u, v, 2.1, 2.1, 3.7, -t * 3.3)
            h = v * 0.95 + 0.55 * (n - 0.5) + 0.25 * (n2 - 0.5)
            col = _ramp(h, [(-0.2, (0.30, 0.02, 0.005)), (0.12, (0.95, 0.12, 0.02)),
                            (0.45, (1.25, 0.42, 0.05)), (0.8, (1.45, 0.95, 0.30)), (1.15, (1.7, 1.55, 1.1))])
            col *= 0.85 + 0.25 * act
            emis = col * (0.28 + 0.22 * act)
        elif style == 'ice':
            n = self._noise(u, v, 1.3, 2.6, 1.3, 0.0)
            facet = np.abs(n - 0.5) * 2.0
            col = _ramp(v + 0.25 * (facet - 0.5), [(-0.1, (0.02, 0.12, 0.55)), (0.45, (0.22, 0.70, 1.15)),
                                                   (1.0, (1.05, 1.25, 1.45))])
            sweep = (t * 0.9) % 3.0 - 1.0
            band = np.exp(-(((u - cap_u0) / max(cap_u1 - cap_u0, 0.5) - 0.35 * v) - sweep) ** 2 / 0.012)
            col = col + band[..., None] * np.array([0.9, 1.1, 1.3], np.float32)
            col *= 0.85 + 0.25 * act
            emis = col * (0.25 + 0.2 * act)
        elif style == 'fusion':
            n = self._noise(u, v, 1.6, 1.6, t * 0.8, -t * 1.2)
            fl = 0.88 + 0.12 * math.sin(t * 47.0) * math.sin(t * 23.0 + 0.5)
            col = _ramp(v + 0.35 * (n - 0.5), [(-0.1, (0.80, 0.06, 0.60)), (0.5, (0.50, 0.20, 1.35)),
                                               (1.05, (1.30, 1.10, 1.70))]) * fl
            col *= 0.85 + 0.25 * act
            emis = col * (0.3 + 0.25 * act)
        elif style == 'blast':
            tb = t - EV['finale_boom']
            flash = math.exp(-max(tb, 0) * 5.0) * (tb >= 0)
            pre = _ss(-0.5, 0.0, tb) * (tb < 0)
            col = _ramp(v, [(0.0, (0.85, 0.22, 0.55)), (0.55, (1.1, 0.85, 1.25)), (1.0, (1.4, 1.35, 1.55))])
            col *= (0.85 + 0.2 * act + 0.25 * pre + 0.7 * flash)
            emis = col * (0.22 + 0.25 * act + 0.45 * flash)
        elif style == 'portal':
            n = self._noise(u, v, 1.4, 1.4, t * 0.6, t * 0.35)
            col = _ramp(v + 0.4 * (n - 0.5), [(-0.1, (0.04, 0.60, 0.80)), (0.55, (0.35, 0.30, 1.30)),
                                              (1.05, (1.05, 0.85, 1.60))])
            col *= 0.85 + 0.25 * act
            emis = col * (0.28 + 0.22 * act)
        elif style == 'dim':
            hx = np.clip((u - cap_u0) / max(cap_u1 - cap_u0, 0.5), 0, 1)
            hx = (hx + 0.15 * np.sin(t * 2.0)) % 1.0
            col = _ramp(hx, [(0.0, (1.05, 0.12, 0.80)), (0.5, (0.40, 0.32, 1.35)), (1.0, (0.10, 0.95, 1.05))])
            col *= (0.75 + 0.45 * np.clip(v, 0, 1))[..., None] * (0.85 + 0.25 * act)
            emis = col * (0.25 + 0.2 * act)
        elif style == 'space':
            col = _ramp(v, [(0.0, (0.35, 0.45, 0.95)), (1.0, (1.10, 1.15, 1.35))])
            col *= 0.85 + 0.25 * act
            emis = col * (0.2 + 0.2 * act)
        elif style == 'gold':
            n = self._noise(u, v, 1.2, 1.2, t * 0.5, -t * 0.9)
            fl = 0.92 + 0.08 * math.sin(t * 37.0) * math.sin(t * 13.0 + 0.3)
            col = _ramp(v + 0.3 * (n - 0.5), [(-0.1, (0.70, 0.18, 0.02)), (0.5, (1.20, 0.58, 0.10)),
                                              (1.05, (1.55, 1.20, 0.55))]) * fl
            col *= 0.85 + 0.25 * act
            emis = col * (0.28 + 0.22 * act)
        elif style == 'claude':
            col = _ramp(v, [(0.0, (0.62, 0.15, 0.08)), (1.0, (0.98, 0.45, 0.27))])
            col *= 0.9 + 0.25 * act
            ws = float(self.ws[wd['wid']])
            ph_ = (t - ws - 0.15) / 0.7
            if -0.2 < ph_ < 1.3:
                pos = cap_u0 - 0.6 + (cap_u1 - cap_u0 + 1.2) * ph_
                band = np.exp(-((u + 0.45 * v) - pos) ** 2 / 0.06)
                col = col + band[..., None] * np.array([0.8, 0.65, 0.5], np.float32)
            emis = col * (0.15 + 0.15 * act)
        elif style in ('bh', 'preto'):
            if style == 'bh':
                rim = np.array([1.9, 0.80, 0.16], np.float32) * (0.85 + 0.3 * act)
                fillc = np.array([0.004, 0.003, 0.003], np.float32)
            else:
                rim = np.array([1.05, 1.07, 1.12], np.float32) * (0.8 + 0.25 * act)
                fillc = np.array([0.006, 0.006, 0.007], np.float32)
            efc = np.clip(ef, 0, 1)[..., None]
            col = fillc * (1 - efc) + rim * efc
            emis = rim * efc * 0.35
        else:
            col = np.ones(v.shape + (3,), np.float32)
            emis = col * 0.0
        return col.astype(np.float32), emis.astype(np.float32)

    # ------------------------------------------------------------ particles
    def _particles(self, wv, t):
        """Analytic particles for a word -> (p0, p1, rad, col) arrays in frame px."""
        style = wv['wd']['style']
        k = wv['wd']['wid']
        x0, y0, x1, y1 = wv['bbox']
        rx0, ry0, rx1, ry1 = wv['rest']
        ww = max(rx1 - rx0, 10.0)
        hh = max(ry1 - ry0, 10.0)
        ts = float(self.ws[k])
        age0 = t - ts
        op = wv['op']
        if age0 < 0 or op < 0.02:
            return None
        rng = np.random.default_rng(1000 + k)
        if style == 'fire':
            n = 46
            px = rng.random(n)
            L = 0.55 + 0.6 * rng.random(n)
            ph = rng.random(n)
            spd = 110 + 170 * rng.random(n)
            wob = rng.random(n) * 6.28
            tt = age0 - ph * L
            ok = tt >= 0
            age = np.mod(tt, L)
            gen = np.floor(tt / L)
            jit = np.array([_hash(k, j, g_) for j, g_ in enumerate(gen)])
            xs = rx0 + (px * 0.9 + 0.05 + 0.04 * (jit - 0.5)) * ww + 18 * np.sin(wob + age * 5.0) * age
            ys = ry0 + hh * (0.15 + 0.2 * jit) - spd * age - 60 * age * age
            vx = 18 * 5.0 * np.cos(wob + age * 5.0) * age
            vy = -spd - 120 * age
            life = age / L
            b = (1 - life) ** 1.6 * np.minimum(age / 0.06, 1.0) * ok * op
            p0 = np.stack([xs, ys], 1)
            p1 = p0 - np.stack([vx, vy], 1) * (1.6 / FPS)
            rad = 1.6 + 1.6 * rng.random(n) * (1 - life)
            heat = 1 - life
            col = np.stack([2.8 * b, (0.6 + 0.9 * heat) * b, (0.08 + 0.3 * heat * heat) * b], 1)
            return p0, p1, rad, col
        if style in ('ice', 'space'):
            n = 9 if style == 'ice' else 7
            pxs = rx0 + (0.05 + 0.9 * rng.random(n)) * ww
            pys = ry1 - hh * (0.35 + 0.6 * rng.random(n))
            fr = 1.3 + 1.8 * rng.random(n)
            ph = rng.random(n) * 6.28
            b = np.maximum(np.sin(age0 * fr * 6.28 + ph), 0) ** 6 * op * np.minimum(age0 / 0.2, 1)
            ln = (14 + 16 * rng.random(n)) * (0.6 + 0.4 * b)
            colb = (np.array([1.4, 1.8, 2.4]) if style == 'ice' else np.array([1.9, 1.9, 2.2]))
            p0 = np.concatenate([np.stack([pxs - ln, pys], 1), np.stack([pxs, pys - ln], 1),
                                 np.stack([pxs, pys], 1)])
            p1 = np.concatenate([np.stack([pxs + ln, pys], 1), np.stack([pxs, pys + ln], 1),
                                 np.stack([pxs, pys], 1)])
            rad = np.concatenate([np.full(n, 1.1), np.full(n, 1.1), np.full(n, 2.2)])
            bb = np.concatenate([b * 2.2, b * 2.2, b * 1.2])
            col = bb[:, None] * colb[None, :]
            return p0, p1, rad, col
        if style == 'gold':
            n = 60
            L = 0.35 + 0.4 * rng.random(n)
            ph = rng.random(n)
            tt = age0 - ph * L
            ok = tt >= 0
            age = np.mod(tt, L)
            gen = np.floor(tt / L)
            jit = np.array([_hash(k + 7, j, g_) for j, g_ in enumerate(gen)])
            jit2 = np.array([_hash(k + 11, j, g_) for j, g_ in enumerate(gen)])
            sx0 = rx0 + (0.03 + 0.94 * jit) * ww
            top = jit2 < 0.6
            sy0 = np.where(top, ry0 + hh * 0.1, ry1 - hh * 0.05)
            ang = np.where(top, -math.pi / 2 + (jit2 - 0.3) * 3.2, math.pi / 2 + (jit2 - 0.8) * 4.0)
            spd = 260 + 380 * rng.random(n)
            vx0, vy0 = np.cos(ang) * spd, np.sin(ang) * spd
            g_ = 900.0
            xs = sx0 + vx0 * age
            ys = sy0 + vy0 * age + 0.5 * g_ * age * age
            vx, vy = vx0, vy0 + g_ * age
            life = age / L
            b = (1 - life) ** 1.3 * ok * op * 1.4
            p0 = np.stack([xs, ys], 1)
            p1 = p0 - np.stack([vx, vy], 1) * (1.2 / FPS)
            rad = 1.1 + 0.8 * rng.random(n)
            col = np.stack([3.0 * b, 1.55 * b, 0.35 * b], 1)
            return p0, p1, rad, col
        return None

    # ------------------------------------------------------------ render
    def behind(self, S):
        st = self._state(S)
        items = st['items']
        if not items:
            return []
        t = S.t
        cap_ref = max(it['ph']['cap'] * it['fa'][2] for it in items)
        x0 = min(it['bbox'][0] for it in items)
        y0 = min(it['bbox'][1] for it in items)
        x1 = max(it['bbox'][2] for it in items)
        y1 = max(it['bbox'][3] for it in items)
        mg = int(0.75 * cap_ref) + 8
        rx0, ry0 = int(max(0, math.floor(x0) - mg)), int(max(0, math.floor(y0) - mg))
        rx1, ry1 = int(min(W, math.ceil(x1) + mg)), int(min(H, math.ceil(y1) + mg))
        # particles may extend the region upward
        fx_words = [wv for wv in st['words'] if wv['wd']['style'] in ('fire', 'gold')]
        if fx_words:
            ry0 = int(max(0, ry0 - 1.6 * cap_ref))
            rx0 = int(max(0, rx0 - 0.4 * cap_ref))
            rx1 = int(min(W, rx1 + 0.4 * cap_ref))
            ry1 = int(min(H, ry1 + 0.8 * cap_ref))
        if rx1 <= rx0 or ry1 <= ry0:
            return []
        h, w = ry1 - ry0, rx1 - rx0
        Acc = np.zeros((h, w), np.float32)
        Col = np.zeros((h, w, 3), np.float32)
        Em = np.zeros((h, w, 3), np.float32)
        Dim = None
        for it in items:
            g = it['g']
            bx0, by0, bx1, by1 = it['bbox']
            gx0, gy0 = int(max(rx0, math.floor(bx0) - 1)), int(max(ry0, math.floor(by0) - 1))
            gx1, gy1 = int(min(rx1, math.ceil(bx1) + 2)), int(min(ry1, math.ceil(by1) + 2))
            if gx1 <= gx0 or gy1 <= gy0:
                continue
            gw_, gh_ = gx1 - gx0, gy1 - gy0
            acc = np.zeros((gh_, gw_, 4), np.float32)
            for (M, op, clip_y) in it['samples']:
                M2 = (_T(-gx0, -gy0) @ M)[:2].astype(np.float32)
                wimg = cv2.warpAffine(g['img'], M2, (gw_, gh_), flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                if clip_y is not None:
                    yy = gy0 + np.arange(gh_, dtype=np.float32)
                    f = np.clip((clip_y - yy) / 7.0 + 0.5, 0, 1)
                    wimg *= f[:, None, None]
                acc += wimg * op
            acc /= len(it['samples'])
            m = acc[..., 0]
            if m.max() < 1e-3:
                continue
            inv = 1.0 / np.maximum(m, 1e-4)
            v = acc[..., 2] * inv
            u = acc[..., 3] * inv
            ef = np.clip(acc[..., 1] * inv, 0, 1)
            wd = it['ph']['words'][g['w']]
            cu0, cu1 = wd['x0'] / it['ph']['cap'], wd['x1'] / it['ph']['cap']
            col, emis = self._fill(g['style'], v, u, ef, t, it['act'], wd, it['P'], cu0, cu1)
            sl = (slice(gy0 - ry0, gy1 - ry0), slice(gx0 - rx0, gx1 - rx0))
            mm = m[..., None]
            Col[sl] = Col[sl] * (1 - mm) + col * mm
            Acc[sl] = Acc[sl] + m - Acc[sl] * m
            Em[sl] += emis * mm
            if g['style'] == 'dim':
                if Dim is None:
                    Dim = np.zeros((h, w), np.float32)
                Dim[sl] = np.maximum(Dim[sl], m)
        if Acc.max() < 1e-3:
            return []
        cap = cap_ref
        # ---- environment adaptation (background luminance under the text)
        bgc = S.bg[ry0:ry1, rx0:rx1]
        lum = float((bgc[::4, ::4] * np.array([0.2126, 0.7152, 0.0722], np.float32)).sum(-1).mean())
        bright = _clamp((lum - 0.04) / 0.35, 0.0, 1.0)
        # ---- head / body contact occlusion (text sits right behind him)
        al = S.alpha[ry0:ry1, rx0:rx1]
        if al.max() > 0.01:
            ao = cv2.GaussianBlur(al, (0, 0), 0.10 * cap)
            Col *= (1.0 - 0.55 * ao)[..., None]
            Em *= (1.0 - 0.35 * ao)[..., None]
        # ---- shadows (black, under the text)
        s1, s2 = max(1.5, 0.028 * cap), max(4.0, 0.17 * cap)
        o1 = 0.45 + 0.35 * bright
        o2 = 0.30 + 0.35 * bright
        Ms = np.float32([[1, 0, 0], [0, 1, 0.03 * cap]])
        sh1 = cv2.GaussianBlur(cv2.warpAffine(Acc, Ms, (w, h)), (0, 0), s1) * o1
        Ms2 = np.float32([[1, 0, 0], [0, 1, 0.07 * cap]])
        sh2 = cv2.GaussianBlur(cv2.warpAffine(Acc, Ms2, (w, h)), (0, 0), s2) * o2
        sh = 1.0 - (1.0 - np.clip(sh1, 0, 1)) * (1.0 - np.clip(sh2, 0, 1))
        a_out = Acc + sh * (1.0 - Acc)
        over = core.Layer('over', Col, a_out, box=(rx0, ry0))
        # ---- glow
        gk = 1.0 - 0.6 * bright
        G = cv2.GaussianBlur(Em, (0, 0), max(2.0, 0.06 * cap)) * (0.45 * gk)
        sw, shh = max(1, w // 4), max(1, h // 4)
        Gs = cv2.GaussianBlur(cv2.resize(Em, (sw, shh), interpolation=cv2.INTER_AREA), (0, 0), max(1.5, 0.30 * cap / 4))
        G += cv2.resize(Gs, (w, h), interpolation=cv2.INTER_LINEAR) * (0.40 * gk)
        if Dim is not None:
            d = max(2, int(0.035 * cap))
            Md = np.float32([[1, 0, -d], [0, 1, 0]])
            Mp = np.float32([[1, 0, d], [0, 1, 0]])
            G[..., 0] += cv2.warpAffine(Dim, Md, (w, h)) * (1 - Dim) * 1.3
            G[..., 2] += cv2.warpAffine(Dim, Mp, (w, h)) * (1 - Dim) * 1.3
            G[..., 1] += cv2.warpAffine(Dim, Mp, (w, h)) * (1 - Dim) * 0.35
        # ---- particles
        for wv in st['words']:
            pr = self._particles(wv, t)
            if pr is None:
                continue
            _splat(G, np.array([rx0, ry0], np.float64), *pr)
        layers = [over, core.Layer('add', G, box=(rx0, ry0))]
        return layers

