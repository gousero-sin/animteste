// Ferramentas de inspeção (folhas de contato) usadas durante o desenvolvimento.
'use strict';

function contactSheet(frames, cols, scale) {
  const main = document.getElementById('cv');
  const mctx = main.getContext('2d');
  const cw = Math.round(CFG.W * scale), ch = Math.round(CFG.BAND_H * scale);
  const rows = Math.ceil(frames.length / cols);
  const sh = document.createElement('canvas');
  sh.width = cw * cols; sh.height = ch * rows;
  const sctx = sh.getContext('2d');
  sctx.fillStyle = '#000'; sctx.fillRect(0, 0, sh.width, sh.height);
  frames.forEach((f, i) => {
    renderFrame(mctx, f);
    const x = (i % cols) * cw, y = Math.floor(i / cols) * ch;
    sctx.drawImage(main, 0, CFG.BAND_Y, CFG.W, CFG.BAND_H, x, y, cw, ch);
    sctx.fillStyle = 'rgba(0,0,0,0.6)'; sctx.fillRect(x, y, 74, 18);
    sctx.fillStyle = '#ff0'; sctx.font = '13px monospace';
    sctx.fillText(`${f} ${(f / CFG.FPS).toFixed(2)}`, x + 3, y + 13);
    sctx.strokeStyle = '#000'; sctx.strokeRect(x + 0.5, y + 0.5, cw - 1, ch - 1);
  });
  return sh.toDataURL('image/png');
}

function poseSheet(names, opts) {
  const cols = Math.min(6, names.length), cw = 260, chh = 300;
  const rows = Math.ceil(names.length / cols);
  const c = document.createElement('canvas');
  c.width = cw * cols; c.height = chh * rows;
  const ctx = c.getContext('2d');
  ctx.fillStyle = '#c4c4c4'; ctx.fillRect(0, 0, c.width, c.height);
  names.forEach((nm, i) => {
    const F = new Fighter('T', Object.assign({ color: '#111' }, opts));
    F.k(0, { p: nm, x: 0, y: 0, face: 1 });
    F.build(POSES);
    const S = F.at(0);
    const ox = (i % cols) * cw + cw / 2 - 20, oy = Math.floor(i / cols) * chh + 230;
    ctx.setTransform(1, 0, 0, 1, ox, oy);
    ctx.strokeStyle = '#999'; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(-120, 0); ctx.lineTo(140, 0); ctx.stroke();
    SCENE.fighters.T = F;
    drawFigure(ctx, S, F.color);
    if (F.weapon === 'spear') drawSpear(ctx, S, F, 0, 0);
    drawFrontArm(ctx, S, '#333');
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = '#000'; ctx.font = '16px monospace';
    ctx.fillText(nm, (i % cols) * cw + 8, Math.floor(i / cols) * chh + 20);
  });
  return c.toDataURL('image/png');
}

// imprime juntas: dbg('B', ['tip','hF'], [6.8, 6.86])
function dbg(id, joints, times) {
  const out = [];
  for (const t of times) {
    const S = SCENE.fighters[id].at(t);
    const row = { t };
    for (const j of joints) { const v = j === 'chest' ? P(id + '.chest', t) : S[j]; row[j] = v ? v.map(n => Math.round(n)) : null; }
    out.push(row);
  }
  return out;
}

function contacts(times) {
  return times.map(t => { const c = crossPoint(t); return { t, d: Math.round(c.d), p: c.p.map(Math.round), u: +c.u.toFixed(2), v: +c.v.toFixed(2) }; });
}

function pathInfo(cp) {
  const pts = catmull(cp, 16), S = arcTable(pts);
  return cp.map((c, i) => Math.round(S[Math.min(S.length - 1, i * 16)]));
}

// detecta saltos bruscos: deslocamento de juntas relativo ao quadril entre quadros consecutivos
function pops(thr) {
  const out = [];
  const J = ['head', 'hF', 'hB', 'fF', 'fB', 'eF', 'eB', 'kF', 'kB', 'neck'];
  for (const id of ['A', 'B']) {
    const F = SCENE.fighters[id];
    let prev = null;
    for (let f = 0; f < CFG.DURATION * CFG.FPS; f++) {
      const t = SCENE.warp.scene(f / CFG.FPS);
      const S = F.at(t);
      if (prev && t > prev.t) {
        let worst = 0, wj = '';
        for (const j of J) {
          const dx = (S[j][0] - S.hip[0]) - (prev[j][0] - prev.hip[0]);
          const dy = (S[j][1] - S.hip[1]) - (prev[j][1] - prev.hip[1]);
          const d = Math.hypot(dx, dy);
          if (d > worst) { worst = d; wj = j; }
        }
        if (S.tip && prev.tip) {
          const d = Math.hypot((S.tip[0] - S.hip[0]) - (prev.tip[0] - prev.hip[0]), (S.tip[1] - S.hip[1]) - (prev.tip[1] - prev.hip[1]));
          if (d > worst) { worst = d; wj = 'tip'; }
        }
        const dh = Math.hypot(S.hip[0] - prev.hip[0], S.hip[1] - prev.hip[1]);
        if (worst > thr || dh > thr * 1.6) out.push(`${id} f${f} t=${t.toFixed(3)} ${wj}=${worst.toFixed(0)} hip=${dh.toFixed(0)}`);
      }
      prev = S;
    }
  }
  return out;
}
