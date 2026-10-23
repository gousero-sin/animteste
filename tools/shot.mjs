// Gera folhas de contato / quadros soltos para inspeção.
// uso: node tools/shot.mjs sheet <saida.png> <ini> <fim> <passo> [colunas] [escala]
//      node tools/shot.mjs frame <saida.png> <quadro>
//      node tools/shot.mjs poses <saida.png> <A|B> nome1,nome2,...
import { createRequire } from 'module';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
const require = createRequire(import.meta.url);
const { chromium } = require('/opt/node22/lib/node_modules/playwright');
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

const [mode, out, ...rest] = process.argv.slice(2);
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1200, height: 2000 } });
page.on('console', m => console.log('[page]', m.text()));
page.on('pageerror', e => { console.error('[pageerror]', e.message); });
await page.goto('file://' + path.join(root, 'index.html') + '?render=1' + (process.env.WIDE ? '&wide=1' : ''));
await page.evaluate(() => document.fonts.ready);
let data;
if (mode === 'sheet') {
  const [a, b, step, cols = 4, scale = 0.42] = rest.map(Number);
  const frames = [];
  for (let f = a; f <= b; f += step) frames.push(f);
  data = await page.evaluate(([fr, c, s]) => contactSheet(fr, c, s), [frames, cols, scale]);
} else if (mode === 'frame') {
  data = await page.evaluate(f => window.frameData(f), Number(rest[0]));
} else if (mode === 'eval') {
  const r = await page.evaluate(code => { const f = new Function(code); return JSON.stringify(f()); }, rest.join(' '));
  console.log(r);
  await browser.close();
  process.exit(0);
} else if (mode === 'poses') {
  const who = rest[0];
  const names = rest[1].split(',');
  data = await page.evaluate(([w, n]) => poseSheet(n, w === 'B' ? { weapon: 'spear', spearLen: 210, color: '#4f4f4f' } : { weapon: 'ice' }), [who, names]);
}
fs.writeFileSync(out, Buffer.from(data.split(',')[1], 'base64'));
await browser.close();
console.log('ok', out);
