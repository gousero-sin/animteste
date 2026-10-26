"""Encode rendered frames + soundtrack into the deliverable MP4.

  python3 -m vfx.encode [--out /home/user/animteste/output/edit_final.mp4] [--audio <wav>]

Uses OUT/frames/%04d.jpg and DATA/audio/mix.wav (falls back to the original audio).
"""
import os, argparse, subprocess
from . import config as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(os.path.dirname(C.ROOT), 'output', 'edit_final.mp4'))
    ap.add_argument('--audio', default=os.path.join(C.DATA, 'audio', 'mix.wav'))
    ap.add_argument('--crf', default='16')
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    frames = os.path.join(C.OUT, 'frames', '%04d.jpg')
    if os.path.exists(a.audio):
        audio_in = ['-i', a.audio]
        amap = ['-map', '1:a:0']
    else:
        audio_in = ['-i', C.SRC_VIDEO]
        amap = ['-map', '1:a:0']
    cmd = ['ffmpeg', '-v', 'error', '-y', '-framerate', f'{C.FPS:.6f}', '-i', frames, *audio_in,
           '-map', '0:v', *amap, '-c:v', 'libx264', '-preset', 'slow', '-crf', a.crf, '-pix_fmt', 'yuv420p',
           '-profile:v', 'high', '-movflags', '+faststart', '-c:a', 'aac', '-b:a', '256k', '-shortest', a.out]
    print(' '.join(cmd))
    subprocess.run(cmd, check=True)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
