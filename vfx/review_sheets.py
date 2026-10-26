"""Contact sheets / crops from the FINAL rendered frames (OUT/frames/NNNN.jpg).

  python3 -m vfx.review_sheets --t0 20 --t1 34 --step 3 --out /path/sheet.jpg [--width 270 --cols 6]
  python3 -m vfx.review_sheets --t0 32.5 --t1 33.2 --step 1 --crop 300,500,800,1100 --out crop.jpg
"""
import os, argparse
import numpy as np
import cv2
from . import config as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--t0', type=float, required=True)
    ap.add_argument('--t1', type=float, required=True)
    ap.add_argument('--step', type=int, default=3)
    ap.add_argument('--out', required=True)
    ap.add_argument('--width', type=int, default=270)
    ap.add_argument('--cols', type=int, default=6)
    ap.add_argument('--crop', default='', help='x0,y0,x1,y1 in output px')
    ap.add_argument('--frames_dir', default=os.path.join(C.OUT, 'frames'))
    a = ap.parse_args()
    ims = []
    for i in range(C.i_of(a.t0), min(C.N_FRAMES, C.i_of(a.t1) + 1), a.step):
        p = os.path.join(a.frames_dir, f'{i:04d}.jpg')
        im = cv2.imread(p)
        if im is None:
            continue
        if a.crop:
            x0, y0, x1, y1 = map(int, a.crop.split(','))
            im = im[y0:y1, x0:x1]
        h = int(im.shape[0] * a.width / im.shape[1])
        im = cv2.resize(im, (a.width, h), interpolation=cv2.INTER_AREA)
        cv2.putText(im, f'{i} {C.t_of(i):.2f}s', (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        ims.append(im)
    if not ims:
        raise SystemExit('no frames found')
    cols = min(a.cols, len(ims))
    rows = (len(ims) + cols - 1) // cols
    h = ims[0].shape[0]
    sheet = np.zeros((rows * h, cols * a.width, 3), np.uint8)
    for k, im in enumerate(ims):
        sheet[(k // cols) * h:(k // cols) * h + h, (k % cols) * a.width:(k % cols) * a.width + a.width] = im
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    cv2.imwrite(a.out, sheet)
    print(a.out, len(ims), 'frames')


if __name__ == '__main__':
    main()
