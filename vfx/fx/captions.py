"""captions -- kinetic typography captions living BEHIND his head (whole video).

Design
------
* Every spoken word (config.WORDS, Whisper word ends trimmed with the audio RMS envelope) is
  grouped into short, grammatical Portuguese phrases (1-4 words, <= 16 chars). Each word starts
  rising ~0.1 s before it is spoken, so it is legible exactly on its onset; the karaoke highlight
  follows the spoken word.
* The text lives in a "slot" just behind the crown of his head: the baseline sits ~40 % of the
  cap height below the top of the hair, so his head/hair occludes the lower-centre of the
  letters while the upper part stays readable. Drawn in behind(): the compositor puts him in
  front. A soft contact occlusion (AO) darkens the text next to his silhouette for depth.
* The line follows the head through a critically damped spring with 0.7x parallax gain (the head
  visibly slides across the text, which reads as depth) and is centred on the words revealed so
  far: the line grows from the head and glides to the full-phrase centre as words are added.
* Letters RISE out of the crown slot with a damped spring, squash/stretch and true multi-sample
  motion blur; phrases that are replaced roll up and fade (before the next one clears the slot),
  phrases followed by silence sink back behind his head.
* Karaoke: the spoken word is hot white (HDR, glows, sheen sweep); spoken words fall back to a
  cool off-white.  Keyword accents: CLAUDE (warm coral), PRETO (black + white rim), FOGO
  (animated fire + embers), GELO (ice + glints), FUSAO (violet plasma), FINALIZACAO (white-hot
  punch on the boom), PORTAL (teal/violet), DIMENSAO (chromatic split), ESPACO (star white),
  BURACO NEGRO (black letters with an accretion-orange rim), PORTAL / DR. ESTRANHO (gold +
  sling-ring sparks).  Glowing accents return core.Light -> they rim-light his hair from behind.
* Readability on the white tiles AND on black/space: tight halo + two-scale soft drop shadow +
  a crisp dark stroke on bright backgrounds, all scaled by the background luminance under the
  text; glow fades out on bright backgrounds.
* Head: the crown is measured on the person matte (skull above the chin minus arms/hands), so it
  stays right when he turns into profile; the face-mesh estimate only fills gaps.
* Layout solver (precomputed per phrase, cached): for every frame it scores (zoom, lift, shift)
  states by per-letter visibility, separating the head (allowed - and rewarded - to cover the
  lower band) from arms/hands, the predicted fire/ice orbs and the predicted Doctor Strange
  sling ring (not allowed, hard minimum for keyword letters), plus a penalty when a fragile
  glyph (E, O, U, R...) sits on the head apex. A speed-limited Viterbi pass with look-ahead
  picks a temporally coherent path (big moves must travel through - and pay for - the states in
  between, so the line goes around a hand, never through it), then it is smoothed -> the line
  visibly dodges, starting before the hand arrives.
* Tunnel (no person): captions centred, words fly out of the vanishing point and zoom past the
  camera.  "E ME SUGA": the phrase is sucked up into the ceiling portal.
* Pure function of the frame index: head track, timings and layouts are precomputed for all
  frames (content-hash cached); particles are analytic; noise is seeded.
"""
import os
import json
import math
import hashlib
from collections import OrderedDict

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

from .. import config as C
from .. import assets as A
from .. import core

W, H, FPS = C.W, C.H, C.FPS
N = C.N_FRAMES
EV = C.EV
CACHE = os.path.join(C.DATA, 'cache', 'captions')
FONT_PATH = os.path.join(C.FONTS, 'Anton-Regular.ttf')

# ---------------------------------------------------------------- phrase table (word indices)
PHRASES = [
    (0, 1), (2, 3), (4, 5), (6, 7),                          # ESSE É | UM VÍDEO | PARA TESTAR | O CLAUDE
    (8, 9, 10), (11,), (12, 13),                             # LIMPA O FUNDO | AQUI | BASICAMENTE AGORA
    (14, 15), (16,), (17, 18), (19, 20, 21),                 # LIMPA ESSE | NEGÓCIO | DE DESCARGA | E ESSE CHUVEIRO
    (22, 23), (24, 25, 26), (27, 28),                        # EU COLOCO | O MEU ESTÚDIO | TODO PRETO
    (29, 30, 31), (32, 33), (34,), (35, 36, 37),             # COLOCO UMA ROUPA | EM MIM | INTERNA | TÁ TUDO BEM
    (38, 39, 40), (41, 42),                                  # E VAMOS FAZER | ALGUMAS COISAS
    (43, 44, 45), (46, 47), (48,),                           # BOLA DE FOGO | NESSA MÃO | DIREITA
    (49, 50, 51), (52, 53),                                  # BOLA DE GELO | NESSA MÃO
    (54, 55, 56, 57), (58, 59),                              # A FUSÃO DAS DUAS | UMA FINALIZAÇÃO
    (60, 61), (62, 63), (64, 65), (66, 67), (68, 69, 70),    # TODAS AS | MINHAS FALAS | DEVEM SER | LEGENDADAS ATRÁS | DA MINHA CABEÇA
    (71, 72, 73), (74, 75), (76, 77), (78, 79), (80, 81),    # DE FORMA QUE | NÃO APAREÇA | ENTÃO AQUI | VOCÊ CONSEGUE | VER TUDO
    (82,), (83, 84), (85, 86), (87,), (88,), (89, 90),       # BASICAMENTE | NESSE MOMENTO | UM PORTAL | SURGE | ACIMA | NO TETO
    (91, 92, 93),                                            # E ME SUGA
    (94, 95, 96, 97), (98, 99),                              # E ME LEVA PARA | OUTRA DIMENSÃO   (tunnel)
    (100,), (101, 102), (103, 104), (105, 106),              # NESSA | OUTRA DIMENSÃO | EU ESTOU | NO ESPAÇO
    (107, 108, 109), (110, 111),                             # AQUI TEM UM | BURACO NEGRO
    (112,), (113, 114), (115, 116, 117),                     # E... | AGORA ACABOU | É SÓ ISSO
    (118,), (119, 120, 121), (122, 123, 124),                # FINALIZA | COM O PORTAL | DO DR. ESTRANHO
]
# "e" (64.2) ... long pause ... "agora" (66.4): the hesitation gets its own beat with an ellipsis
TEXT_OVERRIDE = {112: 'E...'}

STYLE = {7: 'claude', 28: 'preto', 45: 'fire', 51: 'ice', 55: 'fusion', 59: 'blast',
         86: 'portal', 99: 'dim', 102: 'dim', 106: 'space', 110: 'bh', 111: 'bh',
         121: 'gold', 123: 'gold', 124: 'gold'}
EMPH = 1.16            # size factor of accent words inside a phrase

BASE_CAP = 150.0       # cap height (px) of normal words at depth scale 1
MAX_LINE = 0.88 * W    # widest line allowed (px at depth scale 1)
SHORT_LINE = 0.62 * W  # short phrases grow (up to SHORT_BOOST) until they reach this width
SHORT_BOOST = 1.35
OVERLAP = 0.40         # fraction of the cap height hidden behind the crown of the head
CLIP_Y = 0.42          # slot line below the (unlifted) baseline, cap units - letters rise out of it
RISE = 1.45            # entry travel (cap units)
SHUTTER = 0.55 / FPS   # motion-blur shutter (s)

# ---- timing
LEAD, LEAD_KW, LEAD_BLAST = 0.10, 0.05, 0.12   # rise starts this long before the spoken onset
WORD_GAP = 0.07        # min spacing of word reveals inside a phrase
REVEAL_EARLY = 0.22    # plain words after the first may rise up to this much earlier still, so a
                       # fast phrase pops in nearly at once (the karaoke highlight stays on speech)
EXIT_UP, EXIT_SINK = 0.14, 0.32
FREE_UP, FREE_SINK = 0.60, 0.50   # next phrase may rise once the old exit is this far along
MAX_LAG = 0.45         # max delay of a phrase behind speech (absorbed by the next pause)

# ---- head follow
FOLLOW_WX, FOLLOW_WY = 8.0, 16.0  # critically damped spring rates (rad/s) -> ~0.25 s / ~0.12 s lag
FOLLOW_GAIN = 0.70                # parallax: text moves 70 % of the head's motion
FOLLOW_REF_S = 1.2                # s (sigma) of the slow reference the gain is taken around
RECENTRE_W = 0.45                 # duration (s) of the visible-word re-centring glide (smootherstep)
RECENTRE_ANT = 0.36               # glide starts this long before the word rises: a fast phrase
                                  # appears already centred, a slow one makes room early

# ---- layout solver
SOLVER = 'v3-dp-15'
Q = 6.0                           # solver resolution: 1/6 of the output frame
ZOOMS = (1.0, 0.86, 0.72, 0.60)
LIFTS = (0.0, 0.12, 0.24, 0.38, 0.52, 0.68, 0.85, 1.1, 1.4, 1.75)
SHIFTS = tuple(range(-320, 321, 40))
FRAGILE = set('EÉÊLQUÚÇJOÓÔÕDBGR')   # glyphs whose identity lives in their lower part
LAM = (0.06, 0.04, 0.035)         # Viterbi transition cost per step (zoom, lift, shift)
LOOK = (-3, 8)                    # look-behind / look-ahead window (frames) of the solver cost

# ---- sling ring prediction (same R-hand tracking / timeline as the strange module)
# the small ring is a front layer until strange's depth switch at 72.27 s; from then on the ring,
# the open portal and the sparks over him are drawn behind the captions, so they never hide text
RING_T0, RING_T1 = EV['strange'][0], 72.27
RING_CF = np.array([460.0, 930.0])

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


def _atomic_save_npz(path, **arrs):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f'{path}.{os.getpid()}.tmp.npz'
        np.savez(tmp, **arrs)
        os.replace(tmp, path)
    except Exception:
        pass


# ---------------------------------------------------------------- speech timing (audio RMS)
def _speech_timing():
    """Trim Whisper's over-long word ends (and late-voiced starts) with the speech envelope."""
    hsh = hashlib.md5(json.dumps(C.WORDS, sort_keys=True).encode()).hexdigest()[:8]
    path = os.path.join(CACHE, f'timing_v2_{hsh}.json')
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
    face = A._track()['face']
    hsh = hashlib.md5(np.ascontiguousarray(face).tobytes()).hexdigest()[:8]
    path = os.path.join(CACHE, f'head_v1_{hsh}.npz')
    try:
        d = np.load(path)
        return dict(top=d['top'], flen=d['flen'], vx=d['vx'], hash=hsh)
    except Exception:
        pass
    from scipy.ndimage import gaussian_filter1d, median_filter
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
    _atomic_save_npz(path, **res)
    res['hash'] = hsh
    return res


def _crown_track(tr):
    """Crown (top of the solid hair) per frame measured on the person matte: matte above the
    chin, within +-0.85 face lengths of the head, minus arms (pose capsules) and hands (landmark
    hulls); first row whose width exceeds 30 % of the head width, x = centroid of the top band.
    Unlike the face-mesh estimate it stays on the skull when he turns into profile. Median +
    gaussian filtered over neighbouring frames; mesh estimate fills the gaps."""
    path = os.path.join(CACHE, f'crown_v1_{tr["hash"]}.npz')
    try:
        return np.load(path)['top']
    except Exception:
        pass
    from scipy.ndimage import gaussian_filter1d, median_filter
    crown = np.full((N, 2), np.nan)
    Y, X = np.mgrid[0:480, 0:360].astype(np.float32)
    k5 = np.ones((5, 5), np.uint8)
    for i in range(N):
        a = cv2.imread(f'{C.ALPHA_LO}/{i:04d}.png', cv2.IMREAD_GRAYSCALE)
        if a is None:
            continue
        a = a.astype(np.float32) / 255.0
        fo = A.face_oval(i) / 3.0
        flen = float(tr['flen'][i]) / 3.0
        top = tr['top'][i] / 3.0
        chin = float(fo[:, 1].max())
        cxh = 0.5 * (float(fo[:, 0].mean()) + float(top[0]))
        m = a * (Y < chin) * (np.abs(X - cxh) < 0.85 * flen)
        arm = np.zeros((480, 360), np.uint8)
        po = A.pose(i)
        for a_, b_ in ((11, 13), (13, 15), (12, 14), (14, 16)):
            if po[a_, 3] > 0.3 and po[b_, 3] > 0.3:
                cv2.line(arm, tuple(int(v / 3) for v in po[a_, :2]), tuple(int(v / 3) for v in po[b_, :2]),
                         1, int(0.4 * flen) + 1)
        for sd in 'RL':
            hp = A.hand(i, sd)
            if hp is not None:
                hull = cv2.convexHull((hp[:, :2] / 3.0).astype(np.float32))
                cv2.fillConvexPoly(arm, np.round(hull).astype(np.int32).reshape(-1, 2), 1)
        m = m * (1 - cv2.dilate(arm, k5))
        rows = m.sum(1)
        if rows.max() < 3:
            continue
        r = np.where(rows > 0.30 * np.percentile(rows[rows > 0], 90))[0]
        y0 = int(r[0])
        band = m[y0:int(y0 + 0.2 * flen) + 1]
        crown[i] = ((band.sum(0) * np.arange(360)).sum() / (band.sum() + 1e-6) * 3.0, y0 * 3.0 - 6.0)
    mesh = tr['top'].astype(np.float64)
    bad = np.isnan(crown[:, 0])
    crown[bad] = mesh[bad]
    crown = median_filter(crown, size=(9, 1), mode='nearest')
    crown = gaussian_filter1d(crown, 3.0, axis=0, mode='nearest').astype(np.float32)
    _atomic_save_npz(path, top=crown)
    return crown


# ---------------------------------------------------------------- small math helpers
def _ss(a, b, x):
    x = min(max((x - a) / (b - a + 1e-9), 0.0), 1.0)
    return x * x * (3 - 2 * x)


def _clamp(x, a, b):
    return a if x < a else (b if x > b else x)


def _soft_clamp(x, lo, hi, knee):
    """Clamp with tanh knees: C2-continuous at the knee (no velocity/acceleration jumps)."""
    if hi - lo < 2 * knee:
        knee = max(1.0, (hi - lo) / 2)
    a, b = lo + knee, hi - knee
    if x > b:
        return b + knee * math.tanh((x - b) / knee)
    if x < a:
        return a - knee * math.tanh((a - x) / knee)
    return x


def _spring(tau, w0=2 * math.pi * 2.4, z=0.62):
    """Under-damped step response (0 -> 1, ~8 % overshoot) and its velocity (1/s)."""
    if tau <= 0:
        return 0.0, 0.0
    wd = w0 * math.sqrt(1 - z * z)
    e = math.exp(-z * w0 * tau)
    r = 1 - e * (math.cos(wd * tau) + z * w0 / wd * math.sin(wd * tau))
    v = (w0 * w0 / wd) * e * math.sin(wd * tau)
    return r, v


def _cd_step(tau, w):
    """Critically damped step response (0 -> 1, no overshoot)."""
    if tau <= 0:
        return 0.0
    x = w * tau
    return 1.0 - (1.0 + x) * math.exp(-x)


def _smoother(tau, dur):
    """Smootherstep 0 -> 1 over dur (zero velocity and acceleration at both ends: min jerk)."""
    if tau <= 0:
        return 0.0
    x = min(tau / dur, 1.0)
    return x * x * x * (x * (6.0 * x - 15.0) + 10.0)


def _crit_follow(x, w, dt, sub=4):
    """Critically damped spring follower over a whole track (deterministic, causal lag)."""
    y = np.empty(len(x), np.float64)
    p, v = float(x[0]), 0.0
    h = dt / sub
    for k in range(len(x)):
        u = float(x[k])
        for _ in range(sub):
            v += (w * w * (u - p) - 2.0 * w * v) * h
            p += v * h
        y[k] = p
    return y


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


def _ring_R(t):
    """Sling-ring radius over time (mirrors the strange module's timeline)."""
    if t < 70.08:
        return 0.0
    if t < 70.62:
        return 44.0 * (1 - (1 - (t - 70.08) / 0.54) ** 3)
    if t < 72.45:
        u = (t - 70.62) / (72.45 - 70.62)
        return 44.0 + 36.0 * u * u * (3 - 2 * u)
    u = min(max((t - 72.36) / 0.86, 0.0), 1.0)
    return 80.0 + 290.0 * u * u * (3 - 2 * u)


# ---------------------------------------------------------------- the effect
class FX(core.Effect):
    name = 'captions'

    def __init__(self):
        tm = _speech_timing()
        self.ws = np.array(tm['s'])
        self.we = np.array(tm['e'])
        self.text = [TEXT_OVERRIDE.get(k, _clean_word(w['w'])) for k, w in enumerate(C.WORDS)]
        self.track = _head_track()
        self.track['mesh_top'] = self.track['top']
        self.track['top'] = _crown_track(self.track)
        self._make_follow()
        self._built = {}
        self._fonts = {}
        self.plan = self._make_plan()
        self._ring_track()
        self._orb_track()
        self._occ_lru = OrderedDict()
        self.lay = self._layout_all()
        self._occ_lru = None
        self.span = (min(p['show'] for p in self.plan), max(p['vis_end'] for p in self.plan))
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

    # ------------------------------------------------------------ head follow (spring + parallax)
    def _make_follow(self):
        from scipy.ndimage import gaussian_filter1d
        top = self.track['top'].astype(np.float64)
        dt = 1.0 / FPS
        ref = gaussian_filter1d(top[:, 0], FOLLOW_REF_S * FPS, mode='nearest')
        xt = ref + FOLLOW_GAIN * (top[:, 0] - ref)
        fx = _crit_follow(xt, FOLLOW_WX, dt)
        fy = _crit_follow(top[:, 1], FOLLOW_WY, dt)
        self.fx = fx.astype(np.float32)
        self.fy = fy.astype(np.float32)
        self.fvx = np.gradient(fx).astype(np.float32)          # px / frame

    # ------------------------------------------------------------ timeline
    def _lead(self, k):
        st = STYLE.get(k)
        if st == 'blast':
            return LEAD_BLAST
        return LEAD_KW if st else LEAD

    def _make_plan(self):
        """Phrase timeline. Words start rising just before they are spoken; a phrase is held long
        enough to read every word (keywords longer). If speech is too fast, the next phrase lags
        (<= MAX_LAG) and the lag is absorbed by the following pause; when a chain of fast phrases
        would exceed the lag cap, the holds of the earlier phrases in the chain are shortened a
        little (instead of cutting the last one to nothing). A phrase only starts rising once the
        previous one has mostly faded, so lines never stack."""
        hs = np.ones(len(PHRASES))
        for _ in range(40):
            plan, capped = self._plan_pass(hs)
            if not capped:
                break
            changed = False
            for c in capped:
                j = c - 1
                while j >= 0 and plan[j]['exit_mode'] == 'up' and plan[j + 1]['lag'] > 0.02:
                    if hs[j] > 0.6:
                        hs[j] = max(0.6, hs[j] * 0.93)
                        changed = True
                    j -= 1
            if not changed:
                break
        self.hold_scale = hs
        return plan

    def _plan_pass(self, hs):
        nP = len(PHRASES)
        ws, we = self.ws, self.we
        self.wt = ws.copy()                           # display (rise start) time of every word
        self.stg = np.array([min(0.022, 0.10 / max(1, len(tx) - 1)) for tx in self.text])
        r_first = [float(ws[p[0]]) - self._lead(p[0]) for p in PHRASES]
        plan, capped = [], []
        free = -1.0
        for pi, wids in enumerate(PHRASES):
            show = max(r_first[pi], free)
            for j, k in enumerate(wids):
                early = REVEAL_EARLY if (j > 0 and k not in STYLE) else 0.0
                self.wt[k] = max(float(ws[k]) - self._lead(k) - early, show + WORD_GAP * j)
                if j > 0:
                    self.wt[k] = max(self.wt[k], self.wt[wids[j - 1]] + WORD_GAP)
            last = wids[-1]
            ntx = len(self.text[last])
            nch = sum(len(self.text[k]) for k in wids)
            accent = any(k in STYLE for k in wids)
            t_ready = float(self.wt[last]) + float(self.stg[last]) * (ntx - 1) + 0.12
            hold_last = (0.15 + 0.024 * ntx) * hs[pi] + (0.25 if last in STYLE else 0.0)
            read_all = show + (0.28 + 0.032 * nch) * hs[pi] + (0.15 if accent else 0.0)
            min_exit = max(t_ready + hold_last, read_all)
            end_speech = max(float(we[last]), float(self.wt[last]) + 0.12)
            nxt = r_first[pi + 1] if pi + 1 < nP else 1e9
            mode = 'normal'
            if EV['tunnel'][0] - 0.05 <= show < EV['tunnel'][1]:
                mode = 'tunnel'
            gap = nxt - end_speech
            if gap < 1.1:
                exit_mode, exit_dur, ff = 'up', EXIT_UP, FREE_UP
                ideal = nxt - FREE_UP * EXIT_UP
                exit_start = max(ideal, min(min_exit, ideal + MAX_LAG))
                if exit_start < min_exit - 1e-6:
                    capped.append(pi)
            else:
                exit_mode, exit_dur, ff = 'sink', EXIT_SINK, FREE_SINK
                hold = min(1.25, 0.30 * gap + 0.35)
                exit_start = max(min(end_speech + hold, nxt - 0.40), min_exit)
            if 93 in wids:                                 # "E ME SUGA" -> sucked up
                exit_mode, exit_start, exit_dur, ff = 'suck', max(end_speech + 0.05, 49.95), 0.62, 0.8
            if mode == 'tunnel':
                exit_mode, exit_dur, ff = 'zoom', 0.40, 0.25
                if nxt >= EV['tunnel'][1]:
                    exit_start = min(exit_start, EV['tunnel'][1] - 0.42)
            free = exit_start + ff * exit_dur
            vis_end = exit_start + exit_dur + 0.25
            plan.append(dict(pi=pi, wids=wids, show=show, end_speech=end_speech, mode=mode,
                             exit_mode=exit_mode, exit_start=exit_start, exit_dur=exit_dur,
                             vis_end=vis_end, lag=show - r_first[pi], min_exit=min_exit))
        return plan, capped

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
        elif lw < SHORT_LINE:                       # short phrases read bigger
            cap *= min(SHORT_BOOST, SHORT_LINE / lw)
        ws_, sp, lw = widths(cap)
        glyphs, words = [], []
        x = 0.0
        li = 0
        for wi, (tx, f, st) in enumerate(zip(texts, facs, styles)):
            chw = cap * f
            font = self._font(chw / cr)
            x0w = x
            dots = tx.endswith('...') and len(tx) > 3
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
                delay = 0.30 + 0.24 * (kc - (len(tx) - 3)) if dots and kc >= len(tx) - 3 else 0.0
                glyphs.append(dict(img=img, gx=gx, adv=adv, pivot=(ox + adv / 2.0, oy), w=wi,
                                   k=kc, li=li, style=st, chw=chw, ch=ch, delay=delay))
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
        return _ss(54.05, 54.55, t)      # he has landed out of the arrival portal

    def _stages(self, P, ph, t):
        """Reveal stages of the line: [(x1_k, weight_k)], x1_k = right edge (line px) of words
        0..k, weights a partition of unity that glides (smootherstep) from one stage to the next
        RECENTRE_ANT s before word k rises. Words due when the phrase appears are there from the
        start (no glide while the line is still rising out of the slot)."""
        words = ph['words']
        if P['mode'] == 'tunnel' or len(words) == 1:
            return [(ph['lw'], 1.0)]
        prog = [1.0]
        for j in range(1, len(words)):
            t0 = float(self.wt[words[j]['wid']]) - RECENTRE_ANT
            prog.append(1.0 if t0 <= P['show'] + 0.04 else _smoother(t - t0, RECENTRE_W))
        prog.append(0.0)
        return [(words[k]['x1'], prog[k] - prog[k + 1]) for k in range(len(words))
                if prog[k] - prog[k + 1] > 1e-6]

    def _vis_x1(self, P, ph, t):
        """Right edge (line px) of the words visible now, smoothed."""
        return sum(x1 * wk for x1, wk in self._stages(P, ph, t))

    def _anchor_raw(self, i, t, P, ph, lift=0.0, shift=0.0, zoom=1.0, offset=(0.0, 0.0)):
        """Phrase anchor for frame i (pure function): (x of the line centre, baseline y, scale,
        tilt, unlifted crown-slot baseline y). The visible words are centred on the (followed)
        head and softly clamped into the frame per reveal stage; stages are cross-blended, so
        making room for a new word is always one even glide."""
        hx, hy = float(self.fx[i]) + offset[0], float(self.fy[i]) + offset[1]
        flen = float(self.track['flen'][i])
        sd = _clamp((flen / 211.0) ** 0.5, 0.88, 1.22)
        p = self._presence(t)
        sc = (sd * zoom) * p + 1.0 * (1 - p)
        sc = min(sc, (W - 2 * 44.0) / ph['lw'])          # never wider than the frame
        cap = ph['cap'] * sc
        slot = (hy + OVERLAP * cap) * p + (H / 2 + 0.5 * cap) * (1 - p)
        ay = slot - lift * cap * p
        xc = (hx + shift) * p + (W / 2) * (1 - p)       # where the visible words are centred
        left = 0.0
        for x1, wk in self._stages(P, ph, t):
            half = 0.5 * sc * x1 + 34.0
            c = W / 2 if half > W / 2 else _soft_clamp(xc, half, W - half, 70.0)
            left += wk * (c - 0.5 * sc * x1)
        ax = left + 0.5 * sc * ph['lw']
        tilt = math.radians(_clamp(-float(self.fvx[i]) * 0.16, -2.2, 2.2)) * p
        return float(ax), float(ay), float(sc), tilt, float(slot)

    def _lay_at(self, pi, i):
        L = self.lay[pi]
        k = int(min(max(i - L['i0'], 0), len(L['lift']) - 1))
        return float(L['lift'][k]), float(L['shift'][k]), float(L['zoom'][k])

    def _anchor(self, S, P, ph):
        lift, shift, zoom = self._lay_at(P['pi'], S.i)
        return self._anchor_raw(S.i, S.t, P, ph, lift, shift, zoom,
                                (float(S.person_offset[0]), float(S.person_offset[1])))

    # ------------------------------------------------------------ sling-ring prediction
    def _ring_track(self):
        """Ring centre track from the R hand, exactly like the strange module derives it:
        fingertips (index+middle) pushed out along the finger direction, median + gaussian
        smoothed, then a 55 ms one-pole lag; plus the slow 'circling' centre."""
        from scipy.ndimage import gaussian_filter1d, median_filter
        i0 = C.i_of(69.2)
        idx = np.arange(i0, N)
        pen = np.zeros((len(idx), 2))
        for k, i in enumerate(idx):
            h = A.hand(i, 'R')
            if h is not None:
                tip = (h[8, :2] + h[12, :2]) * 0.5
                base = (h[5, :2] + h[9, :2]) * 0.5
                d = tip - base
                pen[k] = tip + d / (np.linalg.norm(d) + 1e-6) * 10.0
            else:
                pen[k] = A.palm(i, 'R') + A.palm_normal_up(i, 'R') * 0.7 * A.hand_size(i, 'R')
        pen = median_filter(pen, size=(5, 1), mode='nearest')
        pen_s = gaussian_filter1d(pen, 1.6, axis=0, mode='nearest')
        lag = np.zeros_like(pen_s)
        lag[0] = pen_s[0]
        aa = 1 - np.exp(-(1 / FPS) / 0.055)
        for k in range(1, len(pen_s)):
            lag[k] = lag[k - 1] + (pen_s[k] - lag[k - 1]) * aa
        self.ring_t = idx / FPS
        self.ring_lag = lag
        self.ring_circ = gaussian_filter1d(pen, 9.0, axis=0, mode='nearest')

    def _orb_track(self):
        """Predicted fire (R palm) / ice (L palm) orb centres and radii, from the same palm anchor
        the orbs module uses (palm + 0.4 hand lengths up/out), so the text can dodge them."""
        from scipy.ndimage import gaussian_filter1d
        i0, i1 = C.i_of(22.3), C.i_of(28.3)
        idx = np.arange(i0, i1 + 1)
        self.orb_t = idx / FPS
        self.orb = {}
        for s in 'RL':
            c = np.zeros((len(idx), 2))
            hs = np.zeros(len(idx))
            for k, i in enumerate(idx):
                p = A.palm(i, s)
                d = 0.45 * A.palm_normal_up(i, s) + 0.55 * np.array([0.0, -1.0])
                d /= np.linalg.norm(d) + 1e-6
                hs[k] = float(np.clip(A.hand_size(i, s), 120, 240))
                c[k] = p + d * 0.40 * hs[k]
            self.orb[s] = (gaussian_filter1d(c, 2.0, axis=0, mode='nearest'),
                           gaussian_filter1d(hs, 4.0, mode='nearest'))

    def _orbs_at(self, t):
        """[(centre, radius)] of the orbs to keep clear of at time t (glow/flames/shards included)."""
        out = []
        if not (self.orb_t[0] <= t <= self.orb_t[-1]):
            return out
        for s, (t0, t1) in (('R', (EV['fire_on'] - 0.05, 28.0)), ('L', (26.30, 28.0))):
            if not (t0 <= t <= t1):
                continue
            c, hs = self.orb[s]
            cc = np.array([np.interp(t, self.orb_t, c[:, 0]), np.interp(t, self.orb_t, c[:, 1])])
            h = float(np.interp(t, self.orb_t, hs))
            if s == 'R':      # fireball + flames licking upward
                r = float(np.clip(0.47 * h, 70, 100))
                out.append((cc - np.array([0.0, 0.45 * r]), 1.55 * r))
            else:             # ice orb + crystal crown + frost glow
                r = float(np.clip(0.30 * h, 52, 70))
                out.append((cc, 2.1 * r))
        return out

    def _ring_at(self, t):
        R = _ring_R(t)
        ch = np.array([np.interp(t, self.ring_t, self.ring_lag[:, 0]), np.interp(t, self.ring_t, self.ring_lag[:, 1])])
        cc = np.array([np.interp(t, self.ring_t, self.ring_circ[:, 0]), np.interp(t, self.ring_t, self.ring_circ[:, 1])])
        wc = _ss(72.10, 72.60, t)
        c = ch * (1 - wc) + cc * wc
        wf = _ss(95.0, 330.0, R)
        return c * (1 - wf) + RING_CF * wf, R

    # ------------------------------------------------------------ occlusion-aware layout solver
    def _occ_maps(self, i):
        """(3, H/Q, W/Q) maps: head (may cover the lower band), other occluders (arms, hands,
        predicted sling ring - must not cover letters), total person matte."""
        if self._occ_lru is not None and i in self._occ_lru:
            return self._occ_lru[i]
        h6, w6 = int(H / Q), int(W / Q)
        a = cv2.imread(f'{C.ALPHA_LO}/{i:04d}.png', cv2.IMREAD_GRAYSCALE)
        a6 = (cv2.resize(a, (w6, h6), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
              if a is not None else np.zeros((h6, w6), np.float32))
        if not hasattr(self, '_g6'):
            self._g6 = np.mgrid[0:h6, 0:w6].astype(np.float32)[::-1]
        X, Y = self._g6
        fo = A.face_oval(i) / Q
        top = self.track['top'][i] / Q
        flen = float(self.track['flen'][i]) / Q
        chin = float(fo[:, 1].max())
        cxh = 0.5 * (float(fo[:, 0].mean()) + float(top[0]))
        # head = person matte above the neck, near the skull (robust to profile turns where the
        # face mesh shrinks), minus the arms (pose capsules) and hands (landmark hulls)
        neck = np.clip((chin + 0.12 * flen - Y) / 2.0 + 0.5, 0, 1)
        xwin = np.clip((0.85 * flen - np.abs(X - cxh)) / 2.0 + 0.5, 0, 1)
        ell = neck * xwin
        arm = np.zeros((h6, w6), np.uint8)
        po = A.pose(i)
        ra = max(1, int(round(0.20 * flen)))
        for a_, b_ in ((11, 13), (13, 15), (12, 14), (14, 16)):
            if po[a_, 3] > 0.3 and po[b_, 3] > 0.3:
                pa = tuple(int(round(v / Q)) for v in po[a_, :2])
                pb = tuple(int(round(v / Q)) for v in po[b_, :2])
                cv2.line(arm, pa, pb, 1, thickness=2 * ra + 1)
        hand = np.zeros((h6, w6), np.uint8)
        for side in 'RL':
            hp = A.hand(i, side)
            if hp is None:
                continue
            hull = cv2.convexHull((hp[:, :2] / Q).astype(np.float32))
            hm = np.zeros((h6, w6), np.uint8)
            cv2.fillConvexPoly(hm, np.round(hull).astype(np.int32).reshape(-1, 2), 1)
            r = max(1, int(round(0.22 * A.hand_size(i, side) / Q)))
            hand |= cv2.dilate(hm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
        head = a6 * ell * (1.0 - np.maximum(hand, arm).astype(np.float32))
        other = a6 - head
        t = i / FPS
        if RING_T0 <= t <= RING_T1:
            cen, R = self._ring_at(t)
            if R > 1.0:
                d = np.sqrt((X * Q - cen[0]) ** 2 + (Y * Q - cen[1]) ** 2)
                # small spinning sparkler ring + glow + spark spray, in front of everything
                other = np.maximum(other, np.clip((R + 55.0 - d) / 12.0, 0, 1))
        for cen, rr in self._orbs_at(t):
            d = np.sqrt((X * Q - cen[0]) ** 2 + (Y * Q - cen[1]) ** 2)
            other = np.maximum(other, np.clip((rr - d) / 12.0, 0, 1))
        maps = np.stack([head, other, a6]).astype(np.float32)
        if self._occ_lru is not None:
            self._occ_lru[i] = maps
            while len(self._occ_lru) > 96:
                self._occ_lru.popitem(last=False)
        return maps

    def _labels(self, ph):
        """Full-res glyph label images of a phrase at rest (row 0 = baseline - 1.34 cap)."""
        if 'lab' in ph:
            return
        cap = ph['cap']
        top = 1.34 * cap
        hh = int(math.ceil(top + 0.45 * cap)) + 2
        ww = int(math.ceil(ph['lw'])) + 2
        lab = np.zeros((hh, ww), np.uint8)
        low = np.zeros((hh, ww), np.uint8)
        for g in ph['glyphs']:
            m = g['img'][..., 0]
            v = g['img'][..., 2] / np.maximum(m, 1e-4)
            px, py = g['pivot']
            x0 = int(round(g['gx'] + g['adv'] / 2.0 - px))
            y0 = int(round(top - py))
            gh, gw = m.shape
            xa, ya, xb, yb = max(0, x0), max(0, y0), min(ww, x0 + gw), min(hh, y0 + gh)
            if xb <= xa or yb <= ya:
                continue
            sub = m[ya - y0:yb - y0, xa - x0:xb - x0] > 0.5
            lab[ya:yb, xa:xb][sub] = g['li'] + 1
            sl = sub & (v[ya - y0:yb - y0, xa - x0:xb - x0] < 0.35)
            low[ya:yb, xa:xb][sl] = g['li'] + 1
        ph['lab'], ph['low'], ph['lab_top'], ph['lab_cache'] = lab, low, top, {}

    def _lab_scaled(self, ph, sc):
        key = int(round(sc * 50))
        c = ph['lab_cache'].get(key)
        if c is not None:
            return c
        f = key / 50.0 / Q
        hh, ww = ph['lab'].shape
        h, w = max(1, int(round(hh * f))), max(1, int(round(ww * f)))
        ng = len(ph['glyphs'])
        out = []
        for src in (ph['lab'], ph['low']):
            G = np.zeros((h, w, ng), np.float32)
            for c0 in range(0, ng, 4):
                c1 = min(ng, c0 + 4)
                oh = np.stack([(src == (g + 1)).astype(np.float32) for g in range(c0, c1)], -1)
                r = cv2.resize(oh, (w, h), interpolation=cv2.INTER_AREA)
                G[..., c0:c1] = r.reshape(h, w, c1 - c0)
            G = G.reshape(h * w, ng)
            out.append((np.ascontiguousarray(G), G.sum(0) + 1e-3))
        c = (out[0][0], out[1][0], (h, w), out[0][1], out[1][1])
        ph['lab_cache'][key] = c
        return c

    def _solve_phrase(self, P):
        ph = self._build(P['pi'])
        self._labels(ph)
        i0 = max(0, int(math.ceil(P['show'] * FPS)))
        i1 = min(N - 1, int(math.floor(P['vis_end'] * FPS)))
        n = max(1, i1 - i0 + 1)
        nZ, nL, nS = len(ZOOMS), len(LIFTS), len(SHIFTS)
        cost = np.zeros((n, nZ, nL, nS), np.float32)
        glyphs = ph['glyphs']
        kw = np.array([g['style'] != 'white' for g in glyphs], np.float32)
        frag = np.array([g['ch'] in FRAGILE for g in glyphs], np.float32)
        reveal = np.array([float(self.wt[ph['words'][g['w']]['wid']]) for g in glyphs])
        shifts = np.array(SHIFTS, np.float32)
        lifts = np.array(LIFTS, np.float32)
        reg = (0.25 * lifts[None, :, None] + 0.7 * np.maximum(lifts - 0.85, 0)[None, :, None] + 0.08 * np.abs(shifts)[None, None, :] / 100.0
               + 0.8 * (1.0 - np.array(ZOOMS, np.float32))[:, None, None])
        PX, PY = 240, 200
        lw = ph['lw']
        for f in range(n):
            i = i0 + f
            t = i / FPS
            if self._presence(t) < 0.99:
                cost[f] = 40.0
                cost[f, 0, 0, nS // 2] = 0.0
                continue
            maps = self._occ_maps(i)
            mp = np.pad(maps, ((0, 0), (PY, PY), (PX, PX)))
            wg = np.where(reveal <= t + 0.03, 1.0, 0.35) * (1.0 + 0.5 * kw)
            if t > P['exit_start']:
                wg = wg * 0.3
            fo = A.face_oval(i)
            hx = float(self.track['top'][i][0])
            flen_i = float(self.track['flen'][i])
            hw_ = max(0.30 * float(fo[:, 0].max() - fo[:, 0].min()), 0.22 * flen_i) + 4.0  # crown half-width
            x1v = self._vis_x1(P, ph, t)
            # fast path: nothing but his head anywhere near the text band -> only the cheap states
            capm = ph['cap'] * 1.25
            ya = int(max(0, (self.fy[i] - 3.6 * capm) / Q))
            yb = int(min(maps.shape[1], (self.fy[i] + 1.0 * capm) / Q))
            easy = yb <= ya or float(maps[1, ya:yb].max()) < 0.05
            for zi, z in enumerate(ZOOMS):
                if easy and zi > 0:
                    break
                res = [self._anchor_raw(i, t, P, ph, 0.0, float(s), z) for s in SHIFTS]
                sc, slot = res[0][2], res[0][4]
                Gt, Glt, (h, w), cnt, cntl = self._lab_scaled(ph, sc)
                lx0 = np.array([r_[0] for r_ in res]) - 0.5 * sc * lw
                xs = [int(round(x / Q)) + PX for x in lx0]
                lx1 = lx0 + sc * x1v
                ov = np.clip(np.minimum(lx1, hx + hw_) - np.maximum(lx0, hx - hw_), 0, None) / (2 * hw_)
                capf = ph['cap'] * sc
                for li_, lf in enumerate(LIFTS):
                    if easy and li_ > 3:
                        break
                    ay = slot - lf * capf
                    y0 = int(round((ay - sc * ph['lab_top']) / Q)) + PY
                    if y0 < 0 or y0 + h > mp.shape[1] or min(xs) < 0 or max(xs) + w > mp.shape[2]:
                        cost[f, zi, li_] = 40.0
                        continue
                    rows = mp[:, y0:y0 + h, :]
                    win = np.stack([rows[:, :, x:x + w] for x in xs], 1).reshape(3, nS, h * w)
                    occ_h = win[0] @ Gt
                    occ_o = win[1] @ Gt
                    lowc = (win[2] @ Glt) / cntl
                    vis_t = 1.0 - (occ_h + occ_o) / cnt
                    vis_o = 1.0 - occ_o / cnt
                    c = (wg * (4.0 * np.maximum(0.55 - vis_t, 0) ** 2
                               + 10.0 * np.maximum(0.95 - vis_o, 0) ** 2)).sum(1)
                    c += (wg * kw * 3.0 * np.clip((0.86 - vis_o) / 0.08, 0, 1)).sum(1)
                    c += (wg * frag * 0.15 * np.clip((lowc - 0.45) / 0.10, 0, 1)).sum(1)
                    c -= 0.30 * ov * _clamp((OVERLAP - lf) / OVERLAP, 0.0, 1.0)
                    cost[f, zi, li_] = c
            if easy:
                ev = cost[f, 0, :4].copy()
                cost[f] = float(ev.min()) + 0.25
                cost[f, 0, :4] = ev
            cost[f] += reg
        # look-behind / look-ahead: a state is only good if it stays good for a little while
        cl = cost.copy()
        for d in range(LOOK[0], LOOK[1] + 1):
            if d == 0:
                continue
            src = cost[np.clip(np.arange(n) + d, 0, n - 1)]
            np.maximum(cl, src, out=cl)
        path = self._viterbi(cl.reshape(n, -1))
        zi, li_, si = np.unravel_index(path, (nZ, nL, nS))
        from scipy.ndimage import gaussian_filter1d
        sg = 4.0
        zoom = gaussian_filter1d(np.array(ZOOMS)[zi].astype(np.float64), sg, mode='nearest')
        lift = gaussian_filter1d(lifts[li_].astype(np.float64), sg, mode='nearest')
        shift = gaussian_filter1d(shifts[si].astype(np.float64), sg, mode='nearest')
        best = cost.reshape(n, -1)[np.arange(n), path]
        return dict(i0=np.int32(i0), lift=lift.astype(np.float32), shift=shift.astype(np.float32),
                    zoom=zoom.astype(np.float32), cost=best.astype(np.float32))

    def _viterbi(self, cost):
        """Min-cost state path with a speed limit: per frame the state may move at most one grid
        step along each axis (40 px shift, one lift / zoom notch), paying LAM per step. A big
        relocation therefore has to travel through - and pay for - the states in between, so the
        line goes around a hand instead of sweeping through it."""
        nZ, nL, nS = len(ZOOMS), len(LIFTS), len(SHIFTS)
        n = cost.shape[0]
        C4 = cost.reshape(n, nZ, nL, nS)
        BIG = np.float32(1e9)
        offs = [(dz, dl, ds) for dz in (-1, 0, 1) for dl in (-1, 0, 1) for ds in (-1, 0, 1)]
        pen = np.array([LAM[0] * abs(a) + LAM[1] * abs(b) + LAM[2] * abs(c) for a, b, c in offs], np.float32)
        D = C4[0].astype(np.float32).copy()
        back = np.zeros((n, nZ, nL, nS), np.int8)
        cand = np.empty((len(offs), nZ, nL, nS), np.float32)
        for f in range(1, n):
            Dp = np.pad(D, 1, constant_values=BIG)
            for k, (dz, dl, ds) in enumerate(offs):
                # predecessor of state (z, l, s) is (z - dz, l - dl, s - ds)
                cand[k] = Dp[1 - dz:1 - dz + nZ, 1 - dl:1 - dl + nL, 1 - ds:1 - ds + nS] + pen[k]
            arg = cand.argmin(0)
            D = np.take_along_axis(cand, arg[None], 0)[0] + C4[f]
            back[f] = arg
        z, l, s_ = np.unravel_index(int(D.argmin()), D.shape)
        path = np.zeros(n, np.int64)
        for f in range(n - 1, -1, -1):
            path[f] = np.ravel_multi_index((z, l, s_), (nZ, nL, nS))
            if f > 0:
                dz, dl, ds = offs[int(back[f, z, l, s_])]
                z, l, s_ = z - dz, l - dl, s_ - ds
        return path

    def _layout_all(self):
        """Per-phrase layout tracks (lift, shift, zoom per frame), cached per phrase."""
        ctx = hashlib.md5(repr((SOLVER, self.track['hash'], FOLLOW_WX, FOLLOW_WY, FOLLOW_GAIN, FOLLOW_REF_S,
                                RECENTRE_W, RECENTRE_ANT, ZOOMS, LIFTS, SHIFTS, LAM, LOOK, OVERLAP, BASE_CAP, MAX_LINE,
                                SHORT_LINE, SHORT_BOOST, EMPH, Q, RING_T0, RING_T1,
                                self._presence(54.3), self._presence(50.1))).encode()).hexdigest()[:8]
        out = {}
        for P in self.plan:
            ph = self._build(P['pi'])
            key = repr((ctx, P['wids'], [self.text[k] for k in P['wids']],
                        [round(float(self.wt[k]), 3) for k in P['wids']], round(P['show'], 3),
                        round(P['vis_end'], 3), round(P['exit_start'], 3), P['mode'], round(ph['cap'], 2),
                        [STYLE.get(k) for k in P['wids']])).encode()
            path = os.path.join(CACHE, 'layout3', f'p{P["pi"]:02d}_{hashlib.md5(key).hexdigest()[:10]}.npz')
            try:
                d = np.load(path)
                out[P['pi']] = {k: d[k] for k in ('i0', 'lift', 'shift', 'zoom', 'cost')}
                out[P['pi']]['i0'] = int(out[P['pi']]['i0'])
                continue
            except Exception:
                pass
            res = self._solve_phrase(P)
            _atomic_save_npz(path, **res)
            res['i0'] = int(res['i0'])
            out[P['pi']] = res
        return out

    # ------------------------------------------------------------ animation
    def _glyph_xf(self, P, ph, g, t, frame_anchor):
        """Local animation of glyph g at time t -> (3x3 matrix img->frame, opacity, clip_y)."""
        ax, ay, sc, tilt, slot = frame_anchor
        cap = ph['cap']
        extra = (slot - ay) / sc                  # lift above the crown slot (line px)
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
        ws = float(self.wt[wid])
        # ---- entry
        if P['mode'] == 'tunnel':
            tau = t - (ws + g['k'] * 0.012)
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
            tau = t - (ws + g['k'] * float(self.stg[wid]) + g['delay'])
            if tau < 0:
                return None
            r, v = _spring(tau)
            dy += (1.0 - r) * (RISE * cap + min(extra, 0.8 * cap))
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
        if P['pi'] == self._blast_pi():
            tb = t - EV['finale_boom']
            if tb >= 0:
                pz = 1.0 + 0.10 * math.exp(-tb * 6.0) * math.cos(tb * 15.0)
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
                q = _clamp((te - g['li'] * min(0.006, 0.05 / max(1, n_l - 1))) / P['exit_dur'], 0.0, 1.0)
                dy -= 0.42 * cap * q * q
                sy *= 1.0 + 0.10 * q
                sx *= 1.0 - 0.04 * q
                op *= 1.0 - _ss(0.0, 0.85, q)
            elif em == 'sink':
                q = _clamp((te - (n_l - 1 - g['li']) * min(0.014, 0.12 / max(1, n_l - 1))) / P['exit_dur'],
                           0.0, 1.0)
                dy += (RISE * cap + min(extra, 0.8 * cap)) * q ** 2.4
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
                op *= 1.0 - _ss(0.05, 0.75, q)
        if op <= 1e-3:
            return None
        # ---- matrices
        px, py = g['pivot']
        Lm = _mat(piv_l[0] + dx, piv_l[1] + dy, rot, sx, sy) @ _T(-px, -py)
        Pm = _T(ax + sc * (cvis[0] - cx_l), ay + sc * cvis[1]) @ _mat(0, 0, tilt, sc, sc) @ _T(-cvis[0], -cvis[1])
        M = Pm @ Lm
        if clip:
            # letters emerge from the crown slot; a line lifted far above the head (dodging a
            # hand / the sling ring) instead fades up through a soft band under its own baseline
            e = _clamp((extra / cap - 0.45) / 0.4, 0.0, 1.0)
            cy0 = slot + sc * CLIP_Y * cap
            cy1 = ay + sc * 0.30 * cap
            clip_y = (cy0 * (1 - e) + cy1 * e, 7.0 * (1 - e) + 0.35 * cap * sc * e)
        else:
            clip_y = None
        return M, op, clip_y

    def _blast_pi(self):
        if not hasattr(self, '_bpi'):
            self._bpi = next(k for k, p in enumerate(PHRASES) if 59 in p)
        return self._bpi

    def _word_act(self, P, w, t):
        """Karaoke activation of word w: on while it is being spoken."""
        wids = P['wids']
        k = wids[w]
        s = max(float(self.ws[k]), float(self.wt[k]) + 0.03)
        if w + 1 < len(wids):
            k2 = wids[w + 1]
            s2 = max(float(self.ws[k2]), float(self.wt[k2]) + 0.03)
            e = min(s2 - 0.02, max(float(self.we[k]), s + 0.15) + 0.30)
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
            fa = self._anchor(S, P, ph)
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
                elif xa is not None or xb is not None:      # entering / leaving this very frame
                    xo = xa if xa is not None else xb
                    disp = 2.0 * float(np.abs((xo[0] @ corners - xf[0] @ corners)[:2]).max())
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
            ax, ay, sc = fa[0], fa[1], fa[2]
            cx_l = ph['lw'] / 2
            for wi, wd in enumerate(ph['words']):
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
                                  rim=1.7, diffuse=0.03, bg=0.0))
        return out

    # ------------------------------------------------------------ fills
    def _noise(self, u, v, su, sv, ou, ov):
        mx = ((u * su + ou) * 64.0) % 256.0
        my = ((v * sv + ov) * 64.0) % 256.0
        return cv2.remap(self.noise, mx.astype(np.float32), my.astype(np.float32),
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

    def _fill(self, style, v, u, ef, t, act, wd, P, cap_u0, cap_u1, bright=0.0):
        """Straight (non-premultiplied) HDR fill colour and emissive colour for a glyph region."""
        if style in ('white', 'suck'):
            base = _ramp(v, [(0.0, (0.76, 0.79, 0.88)), (1.0, (1.0, 1.0, 1.0))])
            past = np.array([0.80, 0.81, 0.86], np.float32)
            hot = np.array([1.30, 1.24, 1.08], np.float32)
            col = base * (past + (hot - past) * act)
            # entry sheen sweep across the active word
            ws = float(self.wt[wd['wid']])
            ph_ = (t - ws) / 0.42
            if -0.2 < ph_ < 1.3:
                pos = cap_u0 - 0.6 + (cap_u1 - cap_u0 + 1.2) * ph_
                band = np.exp(-((u + 0.45 * v) - pos) ** 2 / 0.05)
                col = col + band[..., None] * np.array([0.7, 0.66, 0.55], np.float32) * act
            emis = col * (0.16 * act)
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
            hx = np.clip(hx * 0.85 + 0.075 + 0.075 * math.sin(t * 2.0), 0, 1)
            col = _ramp(hx, [(0.0, (1.05, 0.12, 0.80)), (0.5, (0.40, 0.32, 1.35)), (1.0, (0.10, 0.95, 1.05))])
            core_w = np.clip((v - 0.35) / 0.6, 0, 1)[..., None] * 0.55          # white-hot upper core
            col = col * (1 - core_w) + np.array([1.25, 1.2, 1.35], np.float32) * core_w
            col *= (0.80 + 0.35 * np.clip(v, 0, 1))[..., None] * (0.85 + 0.25 * act)
            emis = col * (0.25 + 0.2 * act)
        elif style == 'space':
            col = _ramp(v, [(0.0, (0.35, 0.45, 0.95)), (1.0, (1.10, 1.15, 1.35))])
            col *= 0.85 + 0.25 * act
            emis = col * (0.2 + 0.2 * act)
        elif style == 'gold':
            n = self._noise(u, v, 1.2, 1.2, t * 0.5, -t * 0.9)
            fl = 0.92 + 0.08 * math.sin(t * 37.0) * math.sin(t * 13.0 + 0.3)
            col = _ramp(v + 0.3 * (n - 0.5), [(-0.1, (0.62, 0.13, 0.015)), (0.5, (1.18, 0.50, 0.07)),
                                              (1.05, (1.50, 0.98, 0.30))]) * fl
            col *= 0.85 + 0.25 * act
            emis = col * (0.28 + 0.22 * act)
        elif style == 'claude':
            col = _ramp(v, [(0.0, (0.62, 0.15, 0.08)), (1.0, (0.98, 0.45, 0.27))])
            col *= 0.9 + 0.25 * act
            ws = float(self.wt[wd['wid']])
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
        if bright > 0.0 and style in ('white', 'suck', 'gold', 'claude'):
            # on bright backgrounds the hot tops tonemap to cream and vanish -> keep them denser
            col = col * (1.0 - 0.25 * bright * np.clip(v, 0, 1.1))[..., None]
        return col.astype(np.float32), emis.astype(np.float32)

    # ------------------------------------------------------------ particles
    def _particles(self, wv, t):
        """Analytic particles for a word -> (p0, p1, rad, col) arrays in frame px."""
        style = wv['wd']['style']
        k = wv['wd']['wid']
        rx0, ry0, rx1, ry1 = wv['rest']
        ww = max(rx1 - rx0, 10.0)
        hh = max(ry1 - ry0, 10.0)
        ts = float(self.wt[k])
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
        # ---- environment adaptation (background luminance under the text)
        bgc = S.bg[ry0:ry1, rx0:rx1]
        # busy/bright backgrounds (tiles, hyperspace streaks, portal) need more separation than
        # their mean suggests -> use a high percentile of the luminance under the text block
        lumi = (bgc[::4, ::4] * np.array([0.2126, 0.7152, 0.0722], np.float32)).sum(-1)
        lum = float(np.percentile(lumi, 85)) if lumi.size else 0.0
        bright = _clamp((lum - 0.05) / 0.35, 0.0, 1.0)
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
                    f = np.clip((clip_y[0] - yy) / clip_y[1] + 0.5, 0, 1)
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
            col, emis = self._fill(g['style'], v, u, ef, t, it['act'], wd, it['P'], cu0, cu1, bright)
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
        # ---- head / body contact occlusion (text sits right behind him)
        al = S.alpha[ry0:ry1, rx0:rx1]
        if al.max() > 0.01:
            ao = cv2.GaussianBlur(al, (0, 0), 0.10 * cap)
            Col *= (1.0 - 0.55 * ao)[..., None]
            Em *= (1.0 - 0.35 * ao)[..., None]
        # ---- shadows (black, under the text)
        s0, s1, s2 = max(1.2, 0.022 * cap), max(1.5, 0.035 * cap), max(4.0, 0.17 * cap)
        o0 = 0.10 + 0.75 * bright          # tight dark halo: crisp edge on white tiles
        o1 = 0.40 + 0.40 * bright
        o2 = 0.28 + 0.45 * bright
        tun = 1.0 - self._presence(t)       # 1 while he is gone (suction end -> tunnel -> arrival)
        if tun > 0.01:                      # hyperspace: text sits on the bright vanishing point
            o0 = o0 + (0.92 - o0) * tun
            o2 = o2 + (0.88 - o2) * tun
            s2 = s2 * (1.0 + 0.6 * tun)
        sh0 = np.clip(cv2.GaussianBlur(Acc, (0, 0), s0) * 1.6, 0, 1) * o0
        Ms = np.float32([[1, 0, 0], [0, 1, 0.035 * cap]])
        sh1 = cv2.GaussianBlur(cv2.warpAffine(Acc, Ms, (w, h)), (0, 0), s1) * o1
        Ms2 = np.float32([[1, 0, 0], [0, 1, 0.08 * cap]])
        sh2 = cv2.GaussianBlur(cv2.warpAffine(Acc, Ms2, (w, h)), (0, 0), s2) * o2
        sh = 1.0 - (1.0 - sh0) * (1.0 - np.clip(sh1, 0, 1)) * (1.0 - np.clip(sh2, 0, 1))
        kb = _clamp((bright - 0.4) / 0.4, 0.0, 1.0)
        if kb > 0.0:                        # crisp dark stroke on bright backgrounds
            r = max(1, int(round(0.025 * cap)))
            ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
            stroke = cv2.GaussianBlur(cv2.dilate(Acc, ker), (0, 0), 1.0) * (0.70 * kb)
            sh = 1.0 - (1.0 - sh) * (1.0 - stroke)
        a_out = Acc + sh * (1.0 - Acc)
        over = core.Layer('over', Col, a_out, box=(rx0, ry0))
        # ---- glow (fades out on bright backgrounds, where it would only wash the halo out)
        gk = (1.0 - 0.6 * bright) * (1.0 - _ss(0.45, 0.72, bright))
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
