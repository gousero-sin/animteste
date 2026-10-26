"""Global constants: paths, video format, and the master timeline.

All times are in seconds of the source video. Frame i <-> t = i / FPS.
Output frames are W x H = 1080 x 1440 (3x the 360x480 source).
"""
import os, json

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get(
    'VFX_DATA',
    '/tmp/claude-0/-home-user-animteste/d26ee407-c488-55a7-b50f-1389dd5fd288/scratchpad')

SRC_VIDEO = f'{DATA}/src.mov'
FRAMES_SRC = f'{DATA}/allframes'      # 0000.png .. 2264.png, 360x480 originals
FRAMES_HYB = f'{DATA}/hyb'            # 0000.jpg ..  1080x1440, person region super-resolved
ALPHA_LO = f'{DATA}/alpha_lo'         # 360x480 RVM alpha (png, 0..255)
ALPHA_HI = f'{DATA}/alpha'            # 1080x1440 RVM alpha (png)
FGR_HI = f'{DATA}/fgr'                # 1080x1440 RVM decontaminated foreground (jpg)
PLATES = f'{DATA}/plates'             # plate0..3.png (1080x1440) + mask_*.png
TRACK = f'{DATA}/track/track.npz'
FONTS = f'{DATA}/fonts'
MODELS = f'{DATA}/models'
OUT = f'{DATA}/out'

FPS = 1359000 / 45431                 # 29.9135...
N_FRAMES = 2265
DURATION = N_FRAMES / FPS             # 75.72 s
SRC_W, SRC_H = 360, 480
SCALE = 3
W, H = 1080, 1440

WORDS = json.load(open(os.path.join(ROOT, 'words.json'), encoding='utf-8'))


def t_of(i):
    return i / FPS


def i_of(t):
    return int(round(t * FPS))


# ---------------------------------------------------------------------------
# Master timeline. Every module syncs to these numbers (seconds).
# The spoken line that motivates each event is given in the comment.
# ---------------------------------------------------------------------------
EV = dict(
    # "limpa o fundo aqui" (5.24-7.14): sink/towel/bottles/mirror/holder/outlet/toilet vanish
    clean_clutter=(5.45, 6.75),
    # "limpa esse negócio de descarga" (descarga 10.92-11.30): beige flush plate vanishes
    clean_flush=(10.80, 11.55),
    # "e esse chuveiro" (11.46-12.14): black shower panel + head + hose + valve vanish
    clean_shower=(11.55, 12.55),
    # "eu coloco o meu estúdio todo preto" (12.48-15.14): bathroom -> black studio
    studio=(13.05, 14.90),
    # "coloco uma roupa em mim" (15.54-16.82): nanotech suit spreads over the body
    suit=(15.75, 17.90),
    # "e vamos fazer algumas coisas" (19.64-21.26): energy anticipation around hands
    anticipation=(19.70, 22.80),
    # "bola de fogo nessa mão direita" (fogo 22.86): fireball on the RIGHT hand (frame-left)
    fire_on=22.86,
    # "bola de gelo nessa mão" (gelo 26.56): ice orb on the LEFT hand (frame-right)
    ice_on=26.56,
    # "a fusão das duas" (28.84-30.60): hands form a ring in front of the face; orbs fuse
    fusion=(28.70, 31.00),
    # hands spread at 31.0 then drop; "uma finalização" (32.18-33.02): detonation
    release=31.00,
    finale_boom=32.62,
    finale_end=34.40,
    # "um portal surge acima no teto" (portal 45.96, teto 47.70-48.40)
    ceil_portal=(45.90, 47.80),
    # "e me suga" (49.36-50.34): pulled up into the portal
    suck=(49.30, 50.55),
    ceil_portal_close=(50.55, 51.10),
    # "e me leva para outra dimensão" (51.08-53.44): wormhole travel, no person visible
    tunnel=(51.00, 53.55),
    # "nessa outra dimensão" (53.44): arrival in space, drops out of a portal from above
    arrive=(53.55, 54.45),
    space=(53.55, 75.72),
    # "aqui tem um buraco negro" (61.98-63.12): black hole forms on the frame-right side
    blackhole=(62.00, 63.40),
    # "finaliza com o portal do Dr. Estranho" (70.10-73.22) + hand gesture 70-74.5
    strange=(70.05, 73.30),
    # portal expands and wipes the frame back to the raw bathroom footage
    strange_wipe=(73.60, 74.70),
    end=75.72,
)

# Environment that owns the background at time t
def environment(t):
    if t < EV['studio'][1]:
        return 'bathroom'            # (studio transition happens inside the cleanup module)
    if t < EV['tunnel'][0]:
        return 'studio'
    if t < EV['tunnel'][1]:
        return 'tunnel'
    return 'space'
