"""Render driver.

  # stills at given times (seconds) with a subset of modules -> PNGs + contact sheet
  python -m vfx.render stills --times 22.9,26.6,29.5 --only cleanup,suit,orbs --out /path/dir

  # every Nth frame of a time range -> PNGs + contact sheet (to judge motion)
  python -m vfx.render range --t0 22.5 --t1 24.0 --step 3 --only orbs --out /path/dir

  # tiny low-res preview mp4 of a range (with original audio)
  python -m vfx.render preview --t0 20 --t1 34 --only cleanup,suit,orbs --out /path/clip.mp4

  # full render (parallel processes) -> OUT/frames/0000.jpg ...
  python -m vfx.render full --procs 3 [--start 0 --end 2265]

Modules are discovered from vfx/fx/<name>.py, each defining `class FX(core.Effect)`.
Missing modules are skipped with a warning, so partial pipelines render fine.
"""
import os, sys, time, argparse, importlib, traceback, subprocess
import numpy as np
import cv2

from . import config as C
from .core import Compositor

ORDER = ['cleanup', 'ceiling_portal', 'space', 'suit', 'orbs', 'captions', 'strange']


def load_effects(only=None):
    names = ORDER if not only or only == 'all' else [n for n in ORDER if n in only.split(',')]
    effects = []
    for n in names:
        try:
            mod = importlib.import_module(f'vfx.fx.{n}')
            effects.append(mod.FX())
        except ModuleNotFoundError as e:
            if f'vfx.fx.{n}' in str(e):
                print(f'[render] module {n} not found, skipping', file=sys.stderr)
            else:
                raise
    return effects


def contact_sheet(paths, out, cols=5, width=270):
    ims = []
    for p in paths:
        im = cv2.imread(p)
        h = int(im.shape[0] * width / im.shape[1])
        im = cv2.resize(im, (width, h), interpolation=cv2.INTER_AREA)
        cv2.putText(im, os.path.basename(p)[:-4], (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        ims.append(im)
    rows = (len(ims) + cols - 1) // cols
    h = ims[0].shape[0]
    sheet = np.zeros((rows * h, cols * width, 3), np.uint8)
    for k, im in enumerate(ims):
        sheet[(k // cols) * h:(k // cols) * h + h, (k % cols) * width:(k % cols) * width + width] = im
    cv2.imwrite(out, sheet)


def render_frames(frames, only, outdir, scale=1.0, quality=95):
    os.makedirs(outdir, exist_ok=True)
    comp = Compositor(load_effects(only))
    paths = []
    for i in frames:
        t0 = time.time()
        img, S = comp.render(i)
        if scale != 1.0:
            img = cv2.resize(img, (int(C.W * scale), int(C.H * scale)), interpolation=cv2.INTER_AREA)
        p = os.path.join(outdir, f'{i:04d}_{C.t_of(i):06.2f}s.png')
        cv2.imwrite(p, img[:, :, ::-1])
        paths.append(p)
        print(f'frame {i} t={C.t_of(i):.2f} {time.time() - t0:.2f}s', flush=True)
    return paths


def _full_worker(args):
    """Render one chunk of frames. Each chunk runs in a fresh process (maxtasksperchild=1)
    so per-process memory growth (GL objects, module caches) stays bounded."""
    frames, only, outdir = args
    os.environ.setdefault('LP_NUM_THREADS', '2')
    todo = [i for i in frames if not os.path.exists(os.path.join(outdir, f'{i:04d}.jpg'))]
    if not todo:
        return 0
    comp = Compositor(load_effects(only))
    for i in todo:
        p = os.path.join(outdir, f'{i:04d}.jpg')
        try:
            img, S = comp.render(i)
        except Exception:
            traceback.print_exc()
            raise
        cv2.imwrite(p + '.tmp.jpg', img[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 97,
                    cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444])
        os.rename(p + '.tmp.jpg', p)
    print(f'[chunk {frames[0]}-{frames[-1]}] done', flush=True)
    return len(todo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['stills', 'range', 'preview', 'full'])
    ap.add_argument('--times', default='')
    ap.add_argument('--frames', default='')
    ap.add_argument('--t0', type=float, default=0)
    ap.add_argument('--t1', type=float, default=1)
    ap.add_argument('--step', type=int, default=1)
    ap.add_argument('--only', default='all')
    ap.add_argument('--out', default=os.path.join(C.DATA, 'preview'))
    ap.add_argument('--scale', type=float, default=1.0)
    ap.add_argument('--procs', type=int, default=3)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--end', type=int, default=C.N_FRAMES)
    a = ap.parse_args()

    if a.cmd == 'stills':
        frames = [C.i_of(float(t)) for t in a.times.split(',') if t] + [int(f) for f in a.frames.split(',') if f]
        paths = render_frames(frames, a.only, a.out, a.scale)
        contact_sheet(paths, os.path.join(a.out, 'sheet.jpg'), cols=min(5, len(paths)))
    elif a.cmd == 'range':
        frames = list(range(C.i_of(a.t0), min(C.N_FRAMES, C.i_of(a.t1) + 1), a.step))
        paths = render_frames(frames, a.only, a.out, a.scale)
        contact_sheet(paths, os.path.join(a.out, 'sheet.jpg'), cols=min(6, len(paths)), width=220)
    elif a.cmd == 'preview':
        tmp = a.out + '_frames'
        frames = list(range(C.i_of(a.t0), min(C.N_FRAMES, C.i_of(a.t1) + 1)))
        render_frames(frames, a.only, tmp, scale=a.scale if a.scale != 1.0 else 0.5)
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-framerate', str(C.FPS), '-pattern_type', 'glob',
                        '-i', f'{tmp}/*.png', '-ss', str(a.t0), '-t', str(a.t1 - a.t0), '-i', C.SRC_VIDEO,
                        '-map', '0:v', '-map', '1:a:0', '-c:v', 'libx264', '-crf', '20', '-pix_fmt', 'yuv420p',
                        '-shortest', a.out], check=True)
    elif a.cmd == 'full':
        from multiprocessing import Pool
        outdir = os.path.join(C.OUT, 'frames')
        os.makedirs(outdir, exist_ok=True)
        chunk = 48
        frames = list(range(a.start, a.end))
        jobs = [(frames[k:k + chunk], a.only, outdir) for k in range(0, len(frames), chunk)]
        with Pool(a.procs, maxtasksperchild=1) as pool:
            for _ in pool.imap_unordered(_full_worker, jobs):
                pass


if __name__ == '__main__':
    main()
