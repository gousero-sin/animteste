// Renderização: câmera, fundo, bonecos, rastros e pós-efeitos.
'use strict';

const SCENE = {
  fighters: {}, fx: [], env: {}, cam: {}, warp: null, sounds: [],
};

function envAt(name, t, def = 0) {
  const tr = SCENE.env[name];
  return tr ? tr.at(t) : def;
}

// ---------- câmera ----------
function camTarget(t) {
  const A = SCENE.fighters.A.at(t), B = SCENE.fighters.B.at(t);
  const xa = A.hip[0], xb = B.hip[0];
  const dist = Math.abs(xa - xb);
  let z = clamp(1080 / (dist + 640), 0.74, 1.32);
  const top = Math.min(A.head[1], B.head[1]) - RIG.hr;
  // enquadra quem está no alto
  const zv = CFG.VH / (-top + 300);
  z = Math.min(z, Math.max(zv, 0.6));
  let cy = -CFG.VH * 0.25 / z;
  const visTop = cy - CFG.VH / 2 / z;
  if (top - 75 < visTop) cy = top - 75 + CFG.VH / 2 / z;
  return { cx: (xa + xb) / 2, cy, z };
}

// janela centrada (passado e futuro): câmera suave e sem atraso
function camAt(t, r) {
  let cx = 0, cy = 0, z = 0, w = 0;
  for (let k = -6; k <= 6; k++) {
    const c = camTarget(t + k * 0.035);
    const wk = Math.exp(-(k * k) / (2 * 3 * 3));
    cx += c.cx * wk; cy += c.cy * wk; z += c.z * wk; w += wk;
  }
  cx /= w; cy /= w; z /= w;
  const C = SCENE.cam;
  const g = n => (C[n] ? C[n].at(t) : null);
  const lock = g('lock') || 0;
  if (lock > 0) {
    cx = lerp(cx, g('lx'), lock);
    cy = lerp(cy, g('ly'), lock);
    z = lerp(z, g('lz'), lock);
  }
  z *= g('zm') || 1;
  cx += g('ox') || 0;
  cy += g('oy') || 0;
  // tremor (tempo real)
  let sx = 0, sy = 0, srot = 0;
  for (const e of SCENE.fx) {
    if (e.type !== 'shake') continue;
    const age = r - e.rt;
    if (age < 0 || age > e.life) continue;
    const d = Math.pow(1 - age / e.life, 2) * e.amp;
    const f = e.freq || 38;
    sx += d * Math.sin(age * f * 6.283 + e.seed);
    sy += d * Math.cos(age * f * 5.1 * 1.13 + e.seed * 2.7);
    srot += d * 0.0009 * Math.sin(age * f * 3.7 + e.seed);
  }
  return { cx, cy, z, sx, sy, srot };
}

function worldTransform(ctx, cam) {
  const mx = CFG.W / 2 + cam.sx * CFG.SCALE, my = CFG.BAND_Y + CFG.BAND_H / 2 + cam.sy * CFG.SCALE;
  const zz = cam.z * CFG.SCALE;
  const c = Math.cos(cam.srot) * zz, s = Math.sin(cam.srot) * zz;
  ctx.setTransform(c, s, -s, c, mx - (c * cam.cx - s * cam.cy), my - (s * cam.cx + c * cam.cy));
}

// ---------- utilidades de desenho ----------
function poly(ctx, pts) {
  ctx.beginPath();
  ctx.moveTo(pts[0][0], pts[0][1]);
  for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
}
function strokeLine(ctx, pts, w) {
  ctx.lineWidth = w;
  poly(ctx, pts);
  ctx.stroke();
}
function circle(ctx, x, y, r) {
  ctx.beginPath();
  ctx.arc(x, y, Math.max(0.01, r), 0, Math.PI * 2);
}
// fita afinada ao longo de uma polilinha
function taper(ctx, pts, w0, w1) {
  const n = pts.length;
  if (n < 2) return;
  const L = [], R = [];
  for (let i = 0; i < n; i++) {
    const a = pts[Math.max(0, i - 1)], b = pts[Math.min(n - 1, i + 1)];
    let dx = b[0] - a[0], dy = b[1] - a[1];
    const d = Math.hypot(dx, dy) || 1;
    dx /= d; dy /= d;
    const w = lerp(w0, w1, i / (n - 1)) / 2;
    L.push([pts[i][0] - dy * w, pts[i][1] + dx * w]);
    R.push([pts[i][0] + dy * w, pts[i][1] - dx * w]);
  }
  ctx.beginPath();
  ctx.moveTo(L[0][0], L[0][1]);
  for (let i = 1; i < n; i++) ctx.lineTo(L[i][0], L[i][1]);
  for (let i = n - 1; i >= 0; i--) ctx.lineTo(R[i][0], R[i][1]);
  ctx.closePath();
}
function resample(pts, maxLen) {
  const out = [pts[0]];
  let acc = 0;
  for (let i = 1; i < pts.length; i++) {
    const a = pts[i - 1], b = pts[i];
    const d = Math.hypot(b[0] - a[0], b[1] - a[1]);
    if (acc + d >= maxLen) {
      const u = (maxLen - acc) / (d || 1);
      out.push([lerp(a[0], b[0], u), lerp(a[1], b[1], u)]);
      return out;
    }
    acc += d;
    out.push(b);
  }
  return out;
}
function hexA(hex, a) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}
function mixHex(h1, h2, t) {
  const a = parseInt(h1.slice(1), 16), b = parseInt(h2.slice(1), 16);
  const r = Math.round(lerp((a >> 16) & 255, (b >> 16) & 255, t));
  const g = Math.round(lerp((a >> 8) & 255, (b >> 8) & 255, t));
  const bl = Math.round(lerp(a & 255, b & 255, t));
  return `rgb(${r},${g},${bl})`;
}

// ---------- fundo ----------
function drawBackground(ctx, t, r) {
  const storm = envAt('storm', t), dark = envAt('dark', t), frost = envAt('frost', t);
  const y0 = CFG.BAND_Y, h = CFG.BAND_H;
  const base = [196, 196, 196];
  const stormC = [134, 138, 156];
  const frostC = [206, 222, 232];
  let c = base.map((v, i) => lerp(v, frostC[i], frost * 0.6));
  c = c.map((v, i) => lerp(v, stormC[i], storm));
  c = c.map(v => v * (1 - dark * 0.45));
  // relâmpagos clareiam o céu
  let fl = 0;
  for (const e of SCENE.fx) {
    if (e.type !== 'skyflash') continue;
    const age = t - e.t;
    if (age < 0 || age > e.life) continue;
    const k = 1 - age / e.life;
    fl = Math.max(fl, k * k * (e.amt || 1) * (0.6 + 0.4 * Math.sin(age * 90)));
  }
  const top = c.map(v => clamp(v * (1 - storm * 0.18) + fl * 120, 0, 255));
  const bot = c.map(v => clamp(v + fl * 60, 0, 255));
  const g = ctx.createLinearGradient(0, y0, 0, y0 + h);
  g.addColorStop(0, `rgb(${top.join(',')})`);
  g.addColorStop(1, `rgb(${bot.join(',')})`);
  ctx.fillStyle = g;
  ctx.fillRect(0, y0, CFG.W, h);
}

// nuvens de tempestade (em tela, com paralaxe leve)
function drawClouds(ctx, t, r, cam) {
  const storm = envAt('storm', t);
  if (storm < 0.02) return;
  const y0 = CFG.BAND_Y;
  ctx.save();
  for (let layer = 0; layer < 2; layer++) {
    const rr = rng(77 + layer * 13);
    const n = 9;
    for (let i = 0; i < n; i++) {
      const K = CFG.SCALE, span = 1500 * K;
      const bx = rr() * span - 210 * K - (cam.cx * (0.08 + layer * 0.06) - t * (6 + layer * 9)) * K;
      const x = ((bx % span) + span) % span - 210 * K;
      const y = y0 + (-30 + rr() * 90 + layer * 40 - (1 - storm) * 160) * K;
      const s = (90 + rr() * 110) * K;
      const g = ctx.createRadialGradient(x, y, 0, x, y, s);
      const a = storm * (0.55 - layer * 0.18);
      const col = layer ? '60,62,80' : '42,43,58';
      g.addColorStop(0, `rgba(${col},${a})`);
      g.addColorStop(0.6, `rgba(${col},${a * 0.6})`);
      g.addColorStop(1, `rgba(${col},0)`);
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.ellipse(x, y, s * 1.6, s * 0.75, 0, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  ctx.restore();
}

// ---------- boneco ----------
function drawFigure(ctx, S, col, alpha = 1) {
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = col;
  ctx.fillStyle = col;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  const lw = RIG.lw, tw = RIG.tw;
  strokeLine(ctx, [S.sh, S.eB, S.hB], lw);
  strokeLine(ctx, [S.hip, S.kB, S.fB], lw);
  strokeLine(ctx, [S.hip, S.neck], tw);
  strokeLine(ctx, [S.hip, S.kF, S.fF], lw);
  circle(ctx, S.head[0], S.head[1], RIG.hr);
  ctx.fill();
  ctx.globalAlpha = 1;
}
function drawFrontArm(ctx, S, col, alpha = 1) {
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = col;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  strokeLine(ctx, [S.sh, S.eF, S.hF], RIG.lw);
  ctx.globalAlpha = 1;
}

function hitFlashAmt(id, t) {
  let a = 0;
  for (const e of SCENE.fx) {
    if (e.type !== 'hitflash' || e.id !== id) continue;
    const age = t - e.t;
    if (age >= 0 && age < e.life) a = Math.max(a, 1 - age / e.life);
  }
  return a;
}

function ribbonPts(F, t, opt) {
  const S0 = F.at(t);
  const pts = [];
  const N = 12;
  const rot = S0.rotDeg * DEG;
  for (let k = 0; k < N; k++) {
    const tk = t - k * opt.lag;
    const S = F.at(tk);
    const anchor = opt.anchor(S);
    const back = -S.face;
    const wave = Math.sin(k * 0.9 - t * opt.wf) * k * opt.wa;
    const off = [back * k * opt.dx + wave * 0.3, k * opt.dy + wave];
    pts.push([anchor[0] + off[0], anchor[1] + off[1]]);
  }
  void rot;
  return resample(pts, opt.len);
}

function drawRibbon(ctx, F, t, opt) {
  for (const tail of opt.tails) {
    const o = Object.assign({}, opt, tail);
    const pts = ribbonPts(F, t, o);
    ctx.fillStyle = o.color;
    taper(ctx, pts, o.w0, o.w1);
    ctx.fill();
  }
}

function drawSpear(ctx, S, F, t, r) {
  if (!S.tip) return;
  const [bx, by] = S.butt, [tx, ty] = S.tip;
  const dx = tx - bx, dy = ty - by, L = Math.hypot(dx, dy);
  const ux = dx / L, uy = dy / L;
  const nx = -uy, ny = ux;
  const zap = S.P.zap;
  const bladeL = 34;
  const bs = [tx - ux * bladeL, ty - uy * bladeL];
  // brilho elétrico
  if (zap > 0.05) {
    ctx.save();
    ctx.globalCompositeOperation = 'lighter';
    const g = ctx.createRadialGradient(tx - ux * 14, ty - uy * 14, 0, tx - ux * 14, ty - uy * 14, 70 * zap);
    g.addColorStop(0, `rgba(255,235,140,${0.55 * zap})`);
    g.addColorStop(0.4, `rgba(170,120,255,${0.25 * zap})`);
    g.addColorStop(1, 'rgba(120,80,255,0)');
    ctx.fillStyle = g;
    circle(ctx, tx - ux * 14, ty - uy * 14, 70 * zap);
    ctx.fill();
    ctx.restore();
  }
  // haste
  ctx.lineCap = 'round';
  ctx.strokeStyle = '#2a2a33';
  ctx.lineWidth = 6;
  poly(ctx, [[bx, by], bs]);
  ctx.stroke();
  // anéis metálicos
  ctx.strokeStyle = '#8c8c99';
  ctx.lineWidth = 7;
  for (const u of [0.02, 0.83]) {
    const px = bx + dx * u, py = by + dy * u;
    poly(ctx, [[px - ux * 3, py - uy * 3], [px + ux * 3, py + uy * 3]]);
    ctx.stroke();
  }
  // borla (fitas) perto da lâmina
  const tas = [bs[0] - ux * 4, bs[1] - uy * 4];
  ctx.fillStyle = '#f2c230';
  const tasPts = [];
  for (let k = 0; k < 7; k++) {
    const Sk = SCENE.fighters[F.id].at(t - k * 0.016);
    if (!Sk.tip) break;
    const L2 = Math.hypot(Sk.tip[0] - Sk.butt[0], Sk.tip[1] - Sk.butt[1]);
    const u2 = [(Sk.tip[0] - Sk.butt[0]) / L2, (Sk.tip[1] - Sk.butt[1]) / L2];
    const p = [Sk.tip[0] - u2[0] * (bladeL + 4), Sk.tip[1] - u2[1] * (bladeL + 4)];
    tasPts.push([p[0], p[1] + k * 2.2]);
  }
  if (tasPts.length > 1) {
    const pts = resample(tasPts.map((p, i) => i === 0 ? tas : p), 30);
    taper(ctx, pts, 7, 2);
    ctx.fill();
  }
  // lâmina em forma de folha
  const w = 8;
  const blade = [
    bs,
    [bs[0] + ux * 9 + nx * w, bs[1] + uy * 9 + ny * w],
    [tx, ty],
    [bs[0] + ux * 9 - nx * w, bs[1] + uy * 9 - ny * w],
  ];
  ctx.fillStyle = zap > 0.3 ? '#fff6c8' : '#d9dde6';
  poly(ctx, blade); ctx.closePath(); ctx.fill();
  ctx.strokeStyle = '#3a3a48'; ctx.lineWidth = 2; ctx.stroke();
  ctx.strokeStyle = 'rgba(60,60,80,0.6)'; ctx.lineWidth = 1.5;
  poly(ctx, [bs, [tx - ux * 3, ty - uy * 3]]); ctx.stroke();
  // guarda
  ctx.strokeStyle = '#2a2a33'; ctx.lineWidth = 5;
  poly(ctx, [[bs[0] + nx * 9, bs[1] + ny * 9], [bs[0] - nx * 9, bs[1] - ny * 9]]); ctx.stroke();
  // arcos elétricos ao longo da haste
  if (zap > 0.05) {
    const seed = Math.floor(r * CFG.FPS / 2) * 31 + 7;
    const rr = rng(seed);
    const n = Math.round(1 + zap * 4);
    for (let i = 0; i < n; i++) {
      const u0 = 0.35 + rr() * 0.65, span = 0.12 + rr() * 0.25;
      const a = [bx + dx * u0, by + dy * u0];
      const u1 = Math.min(1, u0 + span);
      const b = [bx + dx * u1, by + dy * u1];
      const pts = jagged(a, b, rr, 5, 9 * zap);
      drawElectric(ctx, pts, 0.8 * zap, 1);
    }
    if (zap > 0.6) {
      for (let i = 0; i < 2; i++) {
        const ang = rr() * Math.PI * 2, len = 30 + rr() * 40 * zap;
        const b = [tx + Math.cos(ang) * len, ty + Math.sin(ang) * len];
        drawElectric(ctx, jagged([tx, ty], b, rr, 4, 10), zap * 0.9, 0.8);
      }
    }
  }
}

function jagged(a, b, rr, n, amp) {
  const pts = [a];
  const dx = b[0] - a[0], dy = b[1] - a[1];
  const L = Math.hypot(dx, dy) || 1;
  const nx = -dy / L, ny = dx / L;
  for (let i = 1; i < n; i++) {
    const u = i / n;
    const off = (rr() * 2 - 1) * amp * Math.sin(u * Math.PI) ;
    pts.push([a[0] + dx * u + nx * off, a[1] + dy * u + ny * off]);
  }
  pts.push(b);
  return pts;
}
function drawElectric(ctx, pts, alpha, width = 1) {
  ctx.save();
  ctx.lineCap = 'round'; ctx.lineJoin = 'round';
  ctx.globalCompositeOperation = 'source-over';
  ctx.strokeStyle = `rgba(150,110,255,${0.35 * alpha})`;
  ctx.lineWidth = 9 * width;
  poly(ctx, pts); ctx.stroke();
  ctx.strokeStyle = `rgba(255,225,90,${0.95 * alpha})`;
  ctx.lineWidth = 3.6 * width;
  poly(ctx, pts); ctx.stroke();
  ctx.strokeStyle = `rgba(255,255,245,${alpha})`;
  ctx.lineWidth = 1.5 * width;
  poly(ctx, pts); ctx.stroke();
  ctx.restore();
}

function drawIceSword(ctx, S) {
  if (!S.swTip) return;
  const [bx, by] = S.swBase, [tx, ty] = S.swTip;
  const dx = tx - bx, dy = ty - by, L = Math.hypot(dx, dy);
  if (L < 2) return;
  const ux = dx / L, uy = dy / L, nx = -uy, ny = ux;
  const g0 = [bx + ux * 10, by + uy * 10];
  // brilho
  ctx.save();
  ctx.shadowColor = 'rgba(120,220,255,0.9)';
  ctx.shadowBlur = 16;
  const w = 6.5 * Math.min(1, L / 60);
  const blade = [
    [g0[0] + nx * w, g0[1] + ny * w],
    [tx - ux * 14 + nx * w * 0.8, ty - uy * 14 + ny * w * 0.8],
    [tx, ty],
    [tx - ux * 14 - nx * w * 0.8, ty - uy * 14 - ny * w * 0.8],
    [g0[0] - nx * w, g0[1] - ny * w],
  ];
  const grd = ctx.createLinearGradient(bx, by, tx, ty);
  grd.addColorStop(0, '#9fe6ff');
  grd.addColorStop(1, '#f2fdff');
  ctx.fillStyle = grd;
  poly(ctx, blade); ctx.closePath(); ctx.fill();
  ctx.restore();
  ctx.strokeStyle = '#2f8fc4'; ctx.lineWidth = 2;
  poly(ctx, blade); ctx.closePath(); ctx.stroke();
  ctx.strokeStyle = 'rgba(255,255,255,0.9)'; ctx.lineWidth = 1.6;
  poly(ctx, [g0, [tx - ux * 6, ty - uy * 6]]); ctx.stroke();
  // guarda e cabo
  ctx.strokeStyle = '#3aa6d8'; ctx.lineWidth = 5; ctx.lineCap = 'round';
  poly(ctx, [[g0[0] + nx * 12, g0[1] + ny * 12], [g0[0] - nx * 12, g0[1] - ny * 12]]); ctx.stroke();
  ctx.strokeStyle = '#1b5e80'; ctx.lineWidth = 5;
  poly(ctx, [[bx - ux * 8, by - uy * 8], g0]); ctx.stroke();
}

function drawHandGlow(ctx, S, r, amt) {
  if (amt < 0.02) return;
  for (const h of [S.hF, S.hB]) {
    ctx.save();
    ctx.globalCompositeOperation = 'lighter';
    const R = 34 * amt;
    const g = ctx.createRadialGradient(h[0], h[1], 0, h[0], h[1], R);
    g.addColorStop(0, `rgba(200,245,255,${0.85 * amt})`);
    g.addColorStop(0.35, `rgba(90,200,255,${0.45 * amt})`);
    g.addColorStop(1, 'rgba(60,160,255,0)');
    ctx.fillStyle = g;
    circle(ctx, h[0], h[1], R); ctx.fill();
    ctx.restore();
    // cristais orbitando
    ctx.fillStyle = `rgba(235,252,255,${0.95 * amt})`;
    for (let i = 0; i < 4; i++) {
      const a = r * 5 + i * 1.57 + h[0] * 0.01;
      const rad = 16 + 5 * Math.sin(r * 7 + i);
      const px = h[0] + Math.cos(a) * rad, py = h[1] + Math.sin(a) * rad * 0.6;
      const s = 3.2;
      poly(ctx, [[px, py - s * 1.6], [px + s, py], [px, py + s * 1.6], [px - s, py]]);
      ctx.closePath(); ctx.fill();
    }
  }
}

function drawBodyZap(ctx, S, r, amt) {
  if (amt < 0.05) return;
  const rr = rng(Math.floor(r * CFG.FPS / 2) * 13 + 5);
  const joints = [S.head, S.hip, S.hF, S.hB, S.fF, S.fB, S.neck, S.kF];
  const n = Math.round(amt * 5);
  for (let i = 0; i < n; i++) {
    const a = joints[Math.floor(rr() * joints.length)], b = joints[Math.floor(rr() * joints.length)];
    if (a === b) continue;
    drawElectric(ctx, jagged(a, b, rr, 5, 12), Math.min(1, amt * 1.3), 0.7);
  }
}

// rastros de movimento (smear) de mãos, pés e armas
function drawTrails(ctx, F, t) {
  const N = 9, dt = 1 / 110;
  const samples = [];
  for (let k = 0; k < N; k++) samples.push(F.at(t - k * dt));
  const S0 = samples[0];
  if (S0.P.vis < 0.5) return;
  const pts = ['hF', 'hB', 'fF', 'fB'];
  for (const nm of pts) {
    const arr = samples.map(s => s[nm]);
    const sp = Math.hypot(arr[0][0] - arr[2][0], arr[0][1] - arr[2][1]) / (2 * dt);
    const a = clamp((sp - 650) / 900, 0, 0.75);
    if (a <= 0.01) continue;
    ctx.fillStyle = `rgba(255,255,255,${a})`;
    taper(ctx, arr, RIG.lw * 0.95, 1);
    ctx.fill();
  }
  // varredura da lâmina
  if (S0.tip && !S0.spearFree) {
    const tips = samples.map(s => s.tip), mids = samples.map(s => [lerp(s.butt[0], s.tip[0], 0.45), lerp(s.butt[1], s.tip[1], 0.45)]);
    const sp = Math.hypot(tips[0][0] - tips[2][0], tips[0][1] - tips[2][1]) / (2 * dt);
    const a = clamp((sp - 700) / 1400, 0, 0.8);
    if (a > 0.01) {
      for (let k = 0; k < N - 1; k++) {
        if (!tips[k + 1]) break;
        ctx.fillStyle = `rgba(255,255,255,${a * (1 - k / (N - 1)) * 0.75})`;
        poly(ctx, [tips[k], tips[k + 1], mids[k + 1], mids[k]]);
        ctx.closePath(); ctx.fill();
      }
      ctx.strokeStyle = `rgba(255,255,255,${a})`;
      ctx.lineWidth = 3;
      poly(ctx, tips); ctx.stroke();
      if (S0.P.zap > 0.3) {
        ctx.strokeStyle = `rgba(255,220,80,${a * S0.P.zap})`;
        ctx.lineWidth = 2;
        poly(ctx, tips); ctx.stroke();
      }
    }
  }
  if (S0.swTip) {
    const tips = samples.map(s => s.swTip || s.hF), bases = samples.map(s => s.swBase || s.hF);
    const sp = Math.hypot(tips[0][0] - tips[2][0], tips[0][1] - tips[2][1]) / (2 * dt);
    const a = clamp((sp - 650) / 1300, 0, 0.85);
    if (a > 0.01) {
      for (let k = 0; k < N - 1; k++) {
        ctx.fillStyle = `rgba(215,248,255,${a * (1 - k / (N - 1)) * 0.8})`;
        poly(ctx, [tips[k], tips[k + 1], bases[k + 1], bases[k]]);
        ctx.closePath(); ctx.fill();
      }
      ctx.strokeStyle = `rgba(255,255,255,${a})`;
      ctx.lineWidth = 2.5;
      poly(ctx, tips); ctx.stroke();
    }
  }
}

function drawShadow(ctx, S) {
  const low = Math.max(S.fF[1], S.fB[1], S.hip[1], S.kF[1], S.kB[1]);
  const h = Math.max(0, -low);
  const s = clamp(1 - h / 320, 0.25, 1);
  const x = (S.fF[0] + S.fB[0] + S.hip[0] * 2) / 4;
  const spread = Math.abs(S.fF[0] - S.fB[0]);
  const rx = (34 + spread * 0.45) * s, ry = 7 * s + 1;
  ctx.fillStyle = `rgba(60,60,60,${0.28 * s})`;
  ctx.beginPath();
  ctx.ellipse(x, 3, rx, ry, 0, 0, Math.PI * 2);
  ctx.fill();
}

function drawFighter(ctx, F, t, r) {
  const S = F.at(t);
  const P = S.P;
  if (P.vis <= 0.01) return;
  const hf = hitFlashAmt(F.id, t);
  let col = F.color;
  if (P.tint > 0.01) col = mixHex(F.color, F.tintColor || '#7fd3f7', P.tint);
  if (hf > 0) col = mixHex(F.color, F.flashColor || '#ffffff', hf);
  // fantasmas (afterimages)
  if (P.gh > 0.02) {
    for (let k = 4; k >= 1; k--) {
      const Sk = F.at(t - k * 0.028);
      const a = P.gh * (0.42 - k * 0.08);
      if (a <= 0) continue;
      drawFigure(ctx, Sk, F.ghostColor, a);
      drawFrontArm(ctx, Sk, F.ghostColor, a);
    }
  }
  if (F.ribbon) drawRibbon(ctx, F, t, F.ribbon);
  ctx.globalAlpha = P.vis;
  drawFigure(ctx, S, col, P.vis);
  if (F.weapon === 'spear' && !S.spearFree) {
    ctx.globalAlpha = 1;
    drawSpear(ctx, S, F, t, r);
  }
  if (F.weapon === 'ice') drawIceSword(ctx, S);
  drawFrontArm(ctx, S, col, P.vis);
  ctx.globalAlpha = 1;
  if (F.weapon === 'ice') drawHandGlow(ctx, S, r, P.glow);
  drawBodyZap(ctx, S, r, P.bz);
}

// lança solta é desenhada mesmo quando segura por ninguém
function drawFreeSpear(ctx, F, t, r) {
  const S = F.at(t);
  if (S.spearFree) drawSpear(ctx, S, F, t, r);
}

// ---------- quadro ----------
function drawCaption(ctx) {
  if (!CFG.CAPTION) return;
  ctx.save();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = '#fff';
  ctx.font = '700 70px Inter, "Liberation Sans", sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'alphabetic';
  ctx.fillText(CFG.CAPTION, CFG.W / 2, 524);
  ctx.restore();
}

function renderFrame(ctx, frame) {
  const r = frame / CFG.FPS;
  const t = SCENE.warp ? SCENE.warp.scene(r) : r;
  const env = { t, r, frame };
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.globalAlpha = 1;
  ctx.globalCompositeOperation = 'source-over';
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, CFG.W, CFG.H);
  drawCaption(ctx);
  ctx.save();
  ctx.beginPath();
  ctx.rect(0, CFG.BAND_Y, CFG.W, CFG.BAND_H);
  ctx.clip();
  const cam = camAt(t, r);
  env.cam = cam;
  drawBackground(ctx, t, r);
  drawClouds(ctx, t, r, cam);
  worldTransform(ctx, cam);
  drawFxLayer(ctx, 'sky', env);
  drawFxLayer(ctx, 'ground', env);
  const A = SCENE.fighters.A, B = SCENE.fighters.B;
  drawShadow(ctx, A.at(t));
  drawShadow(ctx, B.at(t));
  drawFxLayer(ctx, 'back', env);
  drawTrails(ctx, A, t);
  drawTrails(ctx, B, t);
  const order = SCENE.order ? SCENE.order(t) : ['B', 'A'];
  for (const id of order) {
    drawFighter(ctx, SCENE.fighters[id], t, r);
    drawFxLayer(ctx, 'mid' + id, env);
  }
  drawFreeSpear(ctx, B, t, r);
  drawFxLayer(ctx, 'front', env);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  drawFxLayer(ctx, 'screen', env);
  drawPost(ctx, env);
  ctx.restore();
}
