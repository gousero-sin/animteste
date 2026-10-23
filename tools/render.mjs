// Renderiza todos os quadros em paralelo (vários Chromium) e codifica o MP4 com áudio.
// uso: node tools/render.mjs [saida.mp4] [--workers N] [--from F --to F]
import { createRequire } from 'module';
import fs from 'fs';
import path from 'path';
import { spawnSync } from 'child_process';
import { fileURLToPath } from 'url';
const require = createRequire(import.meta.url);
const { chromium } = require('/opt/node22/lib/node_modules/playwright');
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

const args = process.argv.slice(2);
const opt = (n, d) => { const i = args.indexOf(n); return i >= 0 ? args[i + 1] : d; };
const wide = args.includes('--wide');
const q = '?render=1' + (wide ? '&wide=1' : '');
const out = args[0] && !args[0].startsWith('--') ? args[0] : path.join(root, 'out', wide ? 'gelo_vs_tempestade_16x9.mp4' : 'gelo_vs_tempestade.mp4');
const workers = Number(opt('--workers', 4));
const tmp = opt('--tmp', path.join(root, 'out', wide ? 'frames_wide' : 'frames'));
fs.mkdirSync(tmp, { recursive: true });

const browser = await chromium.launch({ args: ['--disable-gpu-vsync'] });
const probe = await browser.newPage();
await probe.goto('file://' + path.join(root, 'index.html') + q);
const total = await probe.evaluate(() => Math.round(CFG.DURATION * CFG.FPS));
const fps = await probe.evaluate(() => CFG.FPS);
await probe.close();
const from = Number(opt('--from', 0)), to = Number(opt('--to', total - 1));

let next = from, done = 0;
const t0 = Date.now();
async function worker(id) {
  const page = await browser.newPage({ viewport: wide ? { width: 1920, height: 1200 } : { width: 1080, height: 1920 } });
  page.on('pageerror', e => console.error('[pageerror]', e.message));
  await page.goto('file://' + path.join(root, 'index.html') + q);
  await page.evaluate(() => document.fonts.ready);
  while (true) {
    const f = next++;
    if (f > to) break;
    const file = path.join(tmp, `f${String(f).padStart(5, '0')}.png`);
    if (fs.existsSync(file) && !args.includes('--force')) { done++; continue; }
    const data = await page.evaluate(fr => window.frameData(fr), f);
    fs.writeFileSync(file, Buffer.from(data.slice(data.indexOf(',') + 1), 'base64'));
    done++;
    if (done % 60 === 0) {
      const el = (Date.now() - t0) / 1000;
      console.log(`quadros ${done}/${to - from + 1}  ${(done / el).toFixed(1)} q/s`);
    }
  }
  await page.close();
}
await Promise.all(Array.from({ length: workers }, (_, i) => worker(i)));
await browser.close();
console.log('quadros prontos em', ((Date.now() - t0) / 1000).toFixed(0), 's');

if (args.includes('--no-encode')) process.exit(0);
const wav = path.join(root, 'out', 'audio.wav');
if (!fs.existsSync(wav) || args.includes('--audio')) {
  const r = spawnSync('node', [path.join(root, 'tools', 'audio.mjs'), wav], { stdio: 'inherit' });
  if (r.status !== 0) process.exit(1);
}
const ff = spawnSync('ffmpeg', [
  '-y', '-v', 'error', '-framerate', String(fps), '-start_number', String(from), '-i', path.join(tmp, 'f%05d.png'),
  '-i', wav,
  '-c:v', 'libx264', '-preset', 'slow', '-crf', '17', '-pix_fmt', 'yuv420p', '-profile:v', 'high', '-movflags', '+faststart',
  '-af', 'loudnorm=I=-14:TP=-1.2:LRA=11', '-c:a', 'aac', '-b:a', '192k', '-ar', '48000',
  '-shortest', out,
], { stdio: 'inherit' });
if (ff.status !== 0) process.exit(1);
console.log('vídeo pronto:', out);
