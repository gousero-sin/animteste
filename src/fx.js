// Biblioteca de efeitos visuais (todos sem estado: dependem só da idade do evento).
'use strict';

const FX = {};
const G = 1500; // gravidade (unidades/s²)

function P(spec, t) {
  if (!spec) return [0, 0];
  if (Array.isArray(spec)) return spec;
  if (spec === 'X') return crossPoint(t).p;
  if (typeof spec === 'string') {
    const [id, j] = spec.split('.');
    const S = SCENE.fighters[id].at(t);
    if (j === 'mid' && S.tip) return [lerp(S.butt[0], S.tip[0], 0.5), lerp(S.butt[1], S.tip[1], 0.5)];
    if (j === 'chest') return [lerp(S.hip[0], S.neck[0], 0.6), lerp(S.hip[1], S.neck[1], 0.6)];
    if (j === 'ground') return [S.hip[0], 0];
    if (j === 'base') return [S.P.x, S.P.y];
    return (S[j] || S.hip).slice();
  }
  if (spec.ref !== undefined) {
    const p = P(spec.ref, spec.at !== undefined ? spec.at : t);
    return [p[0] + (spec.dx || 0), p[1] + (spec.dy || 0)];
  }
  return [0, 0];
}

// ponto de contato entre a arma de A (espada, ou antebraço/pé) e a lança de B
function segClosest(a0, a1, b0, b1) {
  let best = null;
  for (let i = 0; i <= 20; i++) {
    const u = i / 20;
    const p = [lerp(a0[0], a1[0], u), lerp(a0[1], a1[1], u)];
    const dx = b1[0] - b0[0], dy = b1[1] - b0[1];
    const L2 = dx * dx + dy * dy || 1;
    const v = clamp(((p[0] - b0[0]) * dx + (p[1] - b0[1]) * dy) / L2, 0, 1);
    const q = [b0[0] + dx * v, b0[1] + dy * v];
    const d = Math.hypot(p[0] - q[0], p[1] - q[1]);
    if (!best || d < best.d) best = { d, p: [(p[0] + q[0]) / 2, (p[1] + q[1]) / 2], u, v };
  }
  return best;
}
function crossPoint(t) {
  const SA = SCENE.fighters.A.at(t), SB = SCENE.fighters.B.at(t);
  const a0 = SA.swBase || SA.eF, a1 = SA.swTip || SA.hF;
  return segClosest(a0, a1, SB.butt, SB.tip);
}

function prepareFx() {
  SCENE.fx.forEach((e, i) => {
    if (e.seed === undefined) e.seed = 1000 + i * 37;
    e.rt = SCENE.warp ? SCENE.warp.real(e.t) : e.t;
    if (e.life === undefined) e.life = (FX[e.type] && FX[e.type].life) || 0.5;
    if (!e.follow) {
      if (e.p !== undefined) e.p = P(e.p, e.t);
      if (e.a !== undefined) e.a = P(e.a, e.t);
      if (e.b !== undefined) e.b = P(e.b, e.t + (e.grow || 0));
    }
  });
}

function drawFxLayer(ctx, layer, env) {
  for (const e of SCENE.fx) {
    const def = FX[e.type];
    if (!def || !def.draw) continue;
    const L = e.layer || def.layer || 'front';
    if (L !== layer) continue;
    const age = env.t - e.t;
    if (age < 0 || age > e.life) continue;
    ctx.save();
    def.draw(ctx, e, age, env);
    ctx.restore();
  }
  if (layer === 'screen') { drawSnow(ctx, env); drawRain(ctx, env); }
}

const pos = (e, t) => (e.follow ? P(e.p, t) : e.p);

// ---------- poeira ----------
FX.dust = {
  layer: 'front', life: 0.9,
  draw(ctx, e, age) {
    const rr = rng(e.seed);
    const [x0, y0] = pos(e, e.t + age);
    const col = e.color || '110,110,110';
    for (let i = 0; i < (e.n || 10); i++) {
      const ang = (e.dir !== undefined ? e.dir : -90) * DEG + (rr() * 2 - 1) * (e.spread !== undefined ? e.spread : 180) * DEG / 2;
      const sp = (e.speed || 160) * (0.35 + rr() * 0.65);
      const delay = rr() * (e.stagger || 0.04);
      const sz0 = (e.size || 16) * (0.6 + rr() * 0.8);
      const life = e.life * (0.6 + rr() * 0.4);
      const a = age - delay;
      if (a < 0 || a > life) continue;
      const tr = sp * (1 - Math.exp(-a * 3.2)) / 3.2;
      const x = x0 + Math.cos(ang) * tr;
      const y = y0 + Math.sin(ang) * tr * (e.flat !== undefined ? e.flat : 0.6) - a * (e.rise || 18);
      const k = a / life;
      const s = sz0 * (1 + k * (e.grow || 1.6));
      const al = (e.alpha || 0.42) * Math.pow(1 - k, 1.4) * Math.min(1, a / 0.03);
      const g = ctx.createRadialGradient(x, y, 0, x, y, s);
      g.addColorStop(0, `rgba(${col},${al})`);
      g.addColorStop(0.55, `rgba(${col},${al * 0.7})`);
      g.addColorStop(1, `rgba(${col},0)`);
      ctx.fillStyle = g;
      circle(ctx, x, y, s); ctx.fill();
    }
  },
};

// ---------- pedras ----------
function ballistic(x0, y0, vx, vy, a, g, bounce) {
  // retorna [x, y, landed, landT]
  let tl = Infinity;
  if (y0 <= 0) {
    const disc = vy * vy - 2 * g * y0;
    tl = (-vy + Math.sqrt(Math.max(0, disc))) / g;
  }
  if (a < tl) return [x0 + vx * a, y0 + vy * a + 0.5 * g * a * a, false, tl];
  const xl = x0 + vx * tl;
  const ra = a - tl;
  const slide = vx * (bounce || 0.25) * (1 - Math.exp(-ra * 7)) / 7;
  return [xl + slide, 0, true, tl];
}

FX.debris = {
  layer: 'front', life: 1.6,
  draw(ctx, e, age) {
    const rr = rng(e.seed);
    const [x0, y0] = e.p;
    for (let i = 0; i < (e.n || 8); i++) {
      const ang = (e.dir !== undefined ? e.dir : -90) * DEG + (rr() * 2 - 1) * (e.spread || 120) * DEG / 2;
      const sp = (e.speed || 500) * (0.4 + rr() * 0.6);
      const sz = (e.size || 7) * (0.5 + rr() * 0.9);
      const spin = (rr() * 2 - 1) * 14;
      const nv = 5 + Math.floor(rr() * 2);
      const rad = []; for (let j = 0; j < nv; j++) rad.push(0.6 + rr() * 0.6);
      const shade = 55 + Math.floor(rr() * 40);
      const [x, y, landed, tl] = ballistic(x0 + (rr() - 0.5) * 20, y0 - rr() * 6, Math.cos(ang) * sp, Math.sin(ang) * sp, age, G);
      const rot = spin * Math.min(age, tl);
      const fade = Math.min(1, (e.life - age) / 0.4);
      if (fade <= 0) continue;
      ctx.globalAlpha = fade;
      ctx.fillStyle = e.color || `rgb(${shade},${shade},${shade + 4})`;
      ctx.beginPath();
      for (let j = 0; j < nv; j++) {
        const aa = rot + j / nv * Math.PI * 2;
        const px = x + Math.cos(aa) * sz * rad[j], py = (landed ? y - sz * 0.4 : y) + Math.sin(aa) * sz * rad[j] * (landed ? 0.6 : 1);
        if (j) ctx.lineTo(px, py); else ctx.moveTo(px, py);
      }
      ctx.closePath(); ctx.fill();
    }
    ctx.globalAlpha = 1;
  },
};

// ---------- faíscas ----------
FX.sparks = {
  layer: 'front', life: 0.45,
  draw(ctx, e, age) {
    const rr = rng(e.seed);
    const [x0, y0] = pos(e, e.t + age);
    const col = e.color || '255,236,140';
    ctx.lineCap = 'round';
    for (let i = 0; i < (e.n || 14); i++) {
      const ang = (e.dir !== undefined ? e.dir : -90) * DEG + (rr() * 2 - 1) * (e.spread !== undefined ? e.spread : 360) * DEG / 2;
      const sp = (e.speed || 700) * (0.3 + rr() * 0.7);
      const life = e.life * (0.5 + rr() * 0.5);
      if (age > life) continue;
      const drag = 5;
      const tr = (1 - Math.exp(-age * drag)) / drag;
      const vx = Math.cos(ang) * sp, vy = Math.sin(ang) * sp;
      const x = x0 + vx * tr, y = y0 + vy * tr + 0.5 * (e.g !== undefined ? e.g : 900) * age * age;
      const sv = Math.exp(-age * drag);
      const len = (e.len || 0.03);
      const x1 = x - vx * sv * len, y1 = y - (vy * sv + (e.g !== undefined ? e.g : 900) * age) * len;
      const k = 1 - age / life;
      ctx.strokeStyle = `rgba(${col},${k})`;
      ctx.lineWidth = (e.w || 2.6) * (0.5 + k * 0.5);
      ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x, y); ctx.stroke();
    }
  },
};

// ---------- estilhaços de gelo ----------
FX.shards = {
  layer: 'front', life: 1.2,
  draw(ctx, e, age) {
    const rr = rng(e.seed);
    const [x0, y0] = e.p;
    for (let i = 0; i < (e.n || 12); i++) {
      const ang = (e.dir !== undefined ? e.dir : -90) * DEG + (rr() * 2 - 1) * (e.spread !== undefined ? e.spread : 160) * DEG / 2;
      const sp = (e.speed || 600) * (0.3 + rr() * 0.7);
      const sz = (e.size || 9) * (0.5 + rr() * 0.9);
      const spin = (rr() * 2 - 1) * 16;
      const ox = (rr() - 0.5) * (e.w || 20), oy = -rr() * (e.h || 10);
      const [x, y, landed, tl] = ballistic(x0 + ox, y0 + oy, Math.cos(ang) * sp, Math.sin(ang) * sp, age, G * 0.9, 0.35);
      const rot = spin * Math.min(age, tl) + rr() * 6;
      const fade = Math.min(1, (e.life - age) / 0.35);
      if (fade <= 0) continue;
      const c = Math.cos(rot), s = Math.sin(rot);
      const pts = [[0, -sz * 1.5], [sz * 0.55, 0], [0, sz * 0.9], [-sz * 0.5, 0.1]].map(([px, py]) => [x + px * c - py * s, (landed ? y - 2 : y) + px * s + py * c]);
      ctx.globalAlpha = fade;
      ctx.fillStyle = rr() < 0.5 ? '#c8f1ff' : '#e9fbff';
      poly(ctx, pts); ctx.closePath(); ctx.fill();
      ctx.strokeStyle = '#3c9fd2'; ctx.lineWidth = 1.5; ctx.stroke();
    }
    ctx.globalAlpha = 1;
  },
};

// ---------- sangue ----------
FX.blood = {
  layer: 'front', life: 2.5,
  draw(ctx, e, age) {
    const rr = rng(e.seed);
    const [x0, y0] = e.p;
    const n = e.n || 16;
    for (let i = 0; i < n; i++) {
      const ang = (e.dir !== undefined ? e.dir : 0) * DEG + (rr() * 2 - 1) * (e.spread || 70) * DEG / 2;
      const sp = (e.speed || 520) * (0.25 + rr() * 0.75);
      const sz = (e.size || 4) * (0.5 + rr());
      const vx = Math.cos(ang) * sp, vy = Math.sin(ang) * sp;
      const [x, y, landed, tl] = ballistic(x0, y0, vx, vy, age, G, 0.08);
      const fade = Math.min(1, (e.life - age) / 0.6);
      ctx.fillStyle = `rgba(178,8,24,${fade})`;
      if (!landed) {
        const vyy = vy + G * age;
        const sp2 = Math.hypot(vx, vyy) || 1;
        const st = Math.min(26, sp2 * 0.022) * (sz / 4);
        ctx.strokeStyle = ctx.fillStyle; ctx.lineCap = 'round'; ctx.lineWidth = sz;
        ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x - vx / sp2 * st, y - vyy / sp2 * st); ctx.stroke();
      } else {
        const g = Math.min(1, (age - tl) / 0.08);
        ctx.beginPath();
        ctx.ellipse(x, 1, sz * (1.6 + g * 1.8), sz * 0.45, 0, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    // jato inicial
    if (age < 0.12) {
      const k = age / 0.12;
      const rr2 = rng(e.seed + 5);
      ctx.strokeStyle = `rgba(190,10,28,${1 - k})`;
      ctx.lineCap = 'round';
      for (let i = 0; i < 6; i++) {
        const ang = (e.dir || 0) * DEG + (rr2() * 2 - 1) * (e.spread || 70) * DEG / 3;
        const L = (30 + rr2() * 60) * (0.4 + k);
        ctx.lineWidth = 3 + rr2() * 3;
        ctx.beginPath(); ctx.moveTo(x0 + Math.cos(ang) * L * 0.3, y0 + Math.sin(ang) * L * 0.3);
        ctx.lineTo(x0 + Math.cos(ang) * L, y0 + Math.sin(ang) * L); ctx.stroke();
      }
    }
  },
};

// ---------- clarão de impacto ----------
FX.impact = {
  layer: 'front', life: 0.22,
  draw(ctx, e, age) {
    const k = age / e.life;
    const rr = rng(e.seed);
    const [x, y] = pos(e, e.t + age);
    const R = (e.size || 50) * (0.5 + 0.7 * Ease.o(Math.min(1, k * 2.5)));
    const col = e.color || '255,255,255';
    ctx.globalCompositeOperation = 'lighter';
    const g = ctx.createRadialGradient(x, y, 0, x, y, R * 1.4);
    g.addColorStop(0, `rgba(255,255,250,${0.95 * (1 - k)})`);
    g.addColorStop(0.3, `rgba(${col},${0.6 * (1 - k)})`);
    g.addColorStop(1, `rgba(${col},0)`);
    ctx.fillStyle = g;
    circle(ctx, x, y, R * 1.4); ctx.fill();
    ctx.globalCompositeOperation = 'source-over';
    const n = e.spikes || 9;
    const base = (e.dir !== undefined ? e.dir : 0) * DEG;
    for (let i = 0; i < n; i++) {
      const ang = e.dir !== undefined ? base + (rr() * 2 - 1) * (e.spread || 90) * DEG / 2 : rr() * Math.PI * 2;
      const len = R * (1.1 + rr() * 2.2) * (1 - k * 0.3);
      const w = R * (0.06 + rr() * 0.07) * (1 - k);
      const c = Math.cos(ang), s = Math.sin(ang);
      ctx.fillStyle = `rgba(${e.spikeColor || '255,255,255'},${1 - k})`;
      ctx.beginPath();
      ctx.moveTo(x - s * w, y + c * w);
      ctx.lineTo(x + c * len, y + s * len);
      ctx.lineTo(x + s * w, y - c * w);
      ctx.closePath(); ctx.fill();
    }
  },
};

// ---------- anel de choque ----------
FX.ring = {
  layer: 'front', life: 0.4,
  draw(ctx, e, age) {
    const k = age / e.life;
    const [x, y] = pos(e, e.t + age);
    const R = lerp(e.r0 || 10, e.r1 || 140, Ease.o(k));
    ctx.strokeStyle = `rgba(${e.color || '255,255,255'},${(e.alpha || 0.9) * (1 - k)})`;
    ctx.lineWidth = (e.w || 6) * (1 - k * 0.8);
    ctx.beginPath();
    ctx.ellipse(x, y, R, R * (e.squash || 1), (e.rot || 0) * DEG, 0, Math.PI * 2);
    ctx.stroke();
  },
};

// ---------- relâmpagos ----------
function boltPath(a, b, rr, disp, depth) {
  let pts = [a, b];
  let d = disp;
  for (let it = 0; it < depth; it++) {
    const np = [pts[0]];
    for (let i = 1; i < pts.length; i++) {
      const p = pts[i - 1], q = pts[i];
      const dx = q[0] - p[0], dy = q[1] - p[1];
      const L = Math.hypot(dx, dy) || 1;
      const off = (rr() * 2 - 1) * d;
      np.push([(p[0] + q[0]) / 2 - dy / L * off, (p[1] + q[1]) / 2 + dx / L * off]);
      np.push(q);
    }
    pts = np;
    d *= 0.55;
  }
  return pts;
}

function drawBolt(ctx, a, b, seed, alpha, width, opts = {}) {
  const rr = rng(seed);
  const L = Math.hypot(b[0] - a[0], b[1] - a[1]);
  const pts = boltPath(a, b, rr, L * (opts.jag || 0.22), opts.depth || 5);
  ctx.save();
  ctx.lineCap = 'round'; ctx.lineJoin = 'round';
  if (opts.glow !== false) {
    ctx.strokeStyle = `rgba(${opts.glowColor || '160,120,255'},${0.3 * alpha})`;
    ctx.lineWidth = 16 * width;
    poly(ctx, pts); ctx.stroke();
  }
  ctx.strokeStyle = `rgba(${opts.color || '255,226,90'},${alpha})`;
  ctx.lineWidth = 5 * width;
  poly(ctx, pts); ctx.stroke();
  ctx.strokeStyle = `rgba(255,255,250,${alpha})`;
  ctx.lineWidth = 2 * width;
  poly(ctx, pts); ctx.stroke();
  // galhos
  const nb = opts.branches !== undefined ? opts.branches : 3;
  for (let i = 0; i < nb; i++) {
    const idx = 2 + Math.floor(rr() * (pts.length - 4));
    const p = pts[idx];
    const dir = Math.atan2(b[1] - a[1], b[0] - a[0]) + (rr() < 0.5 ? -1 : 1) * (0.4 + rr() * 0.7);
    const bl = L * (0.12 + rr() * 0.22);
    const q = [p[0] + Math.cos(dir) * bl, p[1] + Math.sin(dir) * bl];
    const bp = boltPath(p, q, rr, bl * 0.3, 3);
    ctx.strokeStyle = `rgba(${opts.color || '255,226,90'},${alpha * 0.8})`;
    ctx.lineWidth = 2.5 * width;
    poly(ctx, bp); ctx.stroke();
    ctx.strokeStyle = `rgba(255,255,250,${alpha * 0.8})`;
    ctx.lineWidth = 1 * width;
    poly(ctx, bp); ctx.stroke();
  }
  ctx.restore();
}

FX.bolt = {
  layer: 'front', life: 0.3,
  draw(ctx, e, age, env) {
    const a = e.follow ? P(e.a, env.t) : e.a;
    let b = e.follow ? P(e.b, env.t) : e.b;
    if (e.grow) {
      const g = Ease.o(Math.min(1, age / e.grow));
      b = [lerp(a[0], b[0], g), lerp(a[1], b[1], g)];
    }
    const k = age / e.life;
    const flick = 0.75 + 0.25 * hash(env.frame * 3 + e.seed);
    const alpha = (e.fade === false ? 1 : 1 - Math.pow(k, 2)) * flick;
    const seed = e.seed + Math.floor(env.frame / (e.rate || 2)) * 7;
    drawBolt(ctx, a, b, seed, alpha, e.w || 1, e);
    if (e.glowEnd) {
      ctx.globalCompositeOperation = 'lighter';
      const R = 60 * (e.w || 1);
      const g = ctx.createRadialGradient(b[0], b[1], 0, b[0], b[1], R);
      g.addColorStop(0, `rgba(255,250,200,${0.8 * alpha})`);
      g.addColorStop(1, 'rgba(160,120,255,0)');
      ctx.fillStyle = g; circle(ctx, b[0], b[1], R); ctx.fill();
    }
  },
};

FX.skybolt = {
  layer: 'front', life: 0.35,
  draw(ctx, e, age, env) {
    const b = e.follow ? P(e.p, env.t) : e.p;
    const top = env.cam.cy - CFG.VH / 2 / env.cam.z - 60;
    const rr = rng(e.seed);
    const a = [b[0] + (rr() - 0.5) * 160, top];
    const k = age / e.life;
    const g = Ease.o(Math.min(1, age / 0.04));
    const bb = [lerp(a[0], b[0], g), lerp(a[1], b[1], g)];
    const alpha = (1 - k * k) * (0.7 + 0.3 * hash(env.frame + e.seed));
    drawBolt(ctx, a, bb, e.seed + Math.floor(env.frame / 2) * 3, alpha, e.w || 1.8, { jag: 0.12, depth: 6, branches: 5 });
    ctx.globalCompositeOperation = 'lighter';
    const R = 120 * (e.w || 1.8) / 1.8;
    const gr = ctx.createRadialGradient(b[0], b[1], 0, b[0], b[1], R);
    gr.addColorStop(0, `rgba(255,250,210,${alpha})`);
    gr.addColorStop(0.4, `rgba(200,170,255,${0.4 * alpha})`);
    gr.addColorStop(1, 'rgba(150,110,255,0)');
    ctx.fillStyle = gr; circle(ctx, b[0], b[1], R); ctx.fill();
  },
};

// ---------- espinhos de gelo do chão ----------
function drawCrystal(ctx, x, y, h, w, lean, alpha, opt = {}) {
  if (h < 1) return;
  const tip = [x + lean, y - h];
  const L = [x - w, y], R = [x + w, y];
  const midL = [x - w * 0.72 + lean * 0.35, y - h * 0.38];
  const midR = [x + w * 0.7 + lean * 0.35, y - h * 0.34];
  ctx.globalAlpha = alpha;
  const g = ctx.createLinearGradient(x, y, tip[0], tip[1]);
  g.addColorStop(0, opt.c0 || '#5fc0ec');
  g.addColorStop(0.55, opt.c1 || '#a9e7ff');
  g.addColorStop(1, opt.c2 || '#f4feff');
  ctx.fillStyle = g;
  poly(ctx, [L, midL, tip, midR, R]); ctx.closePath(); ctx.fill();
  ctx.strokeStyle = opt.edge || '#2b86b8'; ctx.lineWidth = 2;
  ctx.stroke();
  // faceta clara
  ctx.fillStyle = 'rgba(255,255,255,0.55)';
  poly(ctx, [[x - w * 0.15, y], midL, tip, [x + w * 0.1 + lean * 0.2, y - h * 0.3]]); ctx.closePath(); ctx.fill();
  ctx.strokeStyle = 'rgba(255,255,255,0.8)'; ctx.lineWidth = 1.2;
  poly(ctx, [[x + w * 0.1, y], tip]); ctx.stroke();
  ctx.globalAlpha = 1;
}

FX.iceSpikes = {
  layer: 'back', life: 1.6,
  draw(ctx, e, age) {
    const rr = rng(e.seed);
    const n = e.n || 7;
    const fadeT = e.fadeT || 0.35;
    for (let i = 0; i < n; i++) {
      const ti = i * (e.dt || 0.045);
      const a = age - ti;
      const hh = (e.h || 110) * (e.ramp ? lerp(0.45, 1, i / (n - 1)) : 1) * (0.75 + rr() * 0.4);
      const x = e.x0 + (e.dir || 1) * i * (e.gap || 40) + (rr() - 0.5) * 10;
      const lean = (e.dir || 1) * hh * (0.15 + rr() * 0.2);
      const w = hh * (0.2 + rr() * 0.08);
      const side = rr();
      if (a < 0) continue;
      const s = Ease.back(Math.min(1, a / (e.grow || 0.1)));
      const endAge = e.life - fadeT - (n - 1 - i) * 0.02;
      if (age > endAge) {
        // estilhaça
        const k = (age - endAge) / fadeT;
        if (k >= 1) continue;
        drawCrystal(ctx, x, 0, hh * (1 - k), w * (1 - k * 0.5), lean, 1 - k);
        continue;
      }
      drawCrystal(ctx, x - w * 0.9, 0, hh * 0.45 * s, w * 0.5, -lean * 0.8 - 8, 1);
      drawCrystal(ctx, x + w * 0.9 * (side < 0.5 ? 1 : 0.7), 0, hh * 0.55 * s, w * 0.55, lean * 1.2 + 6, 1);
      drawCrystal(ctx, x, 0, hh * s, w, lean, 1);
    }
  },
};

// parede de gelo
FX.iceWall = {
  layer: 'front', life: 1.4,
  draw(ctx, e, age) {
    const rr = rng(e.seed);
    const sh = e.shatter || 99;
    const [x0] = e.p;
    if (age < sh) {
      const s = Ease.back(Math.min(1, age / 0.14));
      const crystals = [];
      for (let i = 0; i < 8; i++) {
        const u = i / 7 - 0.5;
        crystals.push({ x: x0 + u * 90 + (rr() - 0.5) * 14, h: (e.h || 170) * (0.5 + rr() * 0.3 + (0.5 - Math.abs(u)) * 0.45), w: 20 + rr() * 14, lean: u * 50 + (rr() - 0.5) * 20 });
      }
      crystals.sort((a, b) => a.h - b.h);
      for (const c of crystals) drawCrystal(ctx, c.x, 4, c.h * s, c.w, c.lean, 0.95);
      if (e.crack && age > e.crack) {
        const k = Math.min(1, (age - e.crack) / 0.06);
        const rr2 = rng(e.seed + 3);
        const cp = e.cp || [x0, -(e.h || 170) * 0.55];
        ctx.strokeStyle = 'rgba(255,255,255,0.95)'; ctx.lineWidth = 2;
        for (let i = 0; i < 7; i++) {
          const ang = rr2() * Math.PI * 2, L = (25 + rr2() * 60) * k;
          poly(ctx, jagged(cp, [cp[0] + Math.cos(ang) * L, cp[1] + Math.sin(ang) * L], rr2, 4, 6)); ctx.stroke();
        }
      }
    } else {
      FX.shards.draw(ctx, { seed: e.seed + 9, p: [x0, -(e.h || 170) * 0.5], n: 26, speed: 700, size: 13, dir: e.dir !== undefined ? e.dir : -60, spread: 150, w: 50, h: 160, life: e.life - sh }, age - sh);
    }
  },
};

// ---------- decalques no chão ----------
FX.frost = {
  layer: 'ground', life: 6,
  draw(ctx, e, age, env) {
    const [x, y] = e.p;
    const g0 = Ease.o(Math.min(1, age / (e.grow || 0.3)));
    const fade = Math.min(1, (e.life - age) / 0.8);
    const w = (e.w || 120) * g0;
    const g = ctx.createRadialGradient(x, y, 0, x, y, w);
    g.addColorStop(0, `rgba(235,250,255,${0.85 * fade})`);
    g.addColorStop(0.6, `rgba(160,225,250,${0.5 * fade})`);
    g.addColorStop(1, 'rgba(160,225,250,0)');
    ctx.fillStyle = g;
    ctx.beginPath(); ctx.ellipse(x, y + 2, w, w * 0.12, 0, 0, Math.PI * 2); ctx.fill();
    const rr = rng(e.seed);
    ctx.fillStyle = `rgba(255,255,255,${fade})`;
    for (let i = 0; i < 10; i++) {
      const px = x + (rr() * 2 - 1) * w * 0.85, py = y + 2 + (rr() * 2 - 1) * w * 0.08;
      const tw = Math.max(0, Math.sin(env.r * 6 + i * 2.1)) * 3.5;
      if (tw < 0.4) continue;
      poly(ctx, [[px, py - tw * 1.8], [px + tw * 0.4, py], [px, py + tw * 1.8], [px - tw * 0.4, py]]); ctx.closePath(); ctx.fill();
      poly(ctx, [[px - tw * 1.8, py], [px, py + tw * 0.3], [px + tw * 1.8, py], [px, py - tw * 0.3]]); ctx.closePath(); ctx.fill();
    }
  },
};

FX.crack = {
  layer: 'ground', life: 30,
  draw(ctx, e, age) {
    const [x, y] = e.p;
    const rr = rng(e.seed);
    const k = Ease.o(Math.min(1, age / 0.12));
    const fade = Math.min(1, (e.life - age) / 1);
    ctx.strokeStyle = `rgba(70,70,74,${0.85 * fade})`;
    ctx.lineCap = 'round';
    const n = e.n || 7;
    for (let i = 0; i < n; i++) {
      const ang = (i / n) * Math.PI * 2 + rr() * 0.6;
      const L = (e.size || 70) * (0.5 + rr() * 0.7) * k;
      let px = x, py = y + 2;
      ctx.lineWidth = 3;
      ctx.beginPath(); ctx.moveTo(px, py);
      const seg = 4;
      for (let j = 1; j <= seg; j++) {
        const u = j / seg;
        const a2 = ang + (rr() - 0.5) * 0.8;
        px = x + Math.cos(a2) * L * u;
        py = y + 2 + Math.sin(a2) * L * u * 0.16;
        ctx.lineTo(px, py);
      }
      ctx.stroke();
    }
    // pedras soltas acumuladas
    ctx.fillStyle = `rgba(80,80,84,${0.9 * fade})`;
    for (let i = 0; i < (e.rocks || 0); i++) {
      const px = x + (rr() * 2 - 1) * (e.size || 70), s = (3 + rr() * 6) * k;
      poly(ctx, [[px - s, y + 3], [px - s * 0.4, y - s * 0.9], [px + s * 0.6, y - s * 0.5], [px + s, y + 3]]); ctx.closePath(); ctx.fill();
    }
  },
};

FX.scorch = {
  layer: 'ground', life: 20,
  draw(ctx, e, age) {
    const [x, y] = e.p;
    const fade = Math.min(1, age / 0.05) * Math.min(1, (e.life - age) / 2);
    const w = e.w || 70;
    const g = ctx.createRadialGradient(x, y, 0, x, y, w);
    g.addColorStop(0, `rgba(40,38,48,${0.55 * fade})`);
    g.addColorStop(1, 'rgba(40,38,48,0)');
    ctx.fillStyle = g;
    ctx.beginPath(); ctx.ellipse(x, y + 2, w, w * 0.14, 0, 0, Math.PI * 2); ctx.fill();
  },
};

// arranhões no chão (marcas de derrapagem)
FX.skid = {
  layer: 'ground', life: 20,
  draw(ctx, e, age) {
    const [x, y] = e.p;
    const k = Ease.o(Math.min(1, age / (e.grow || 0.25)));
    const fade = Math.min(1, (e.life - age) / 1.5);
    const rr = rng(e.seed);
    ctx.lineCap = 'round';
    for (let i = 0; i < 3; i++) {
      const oy = y + 1 + (rr() - 0.5) * 8;
      const L = (e.len || 120) * (0.6 + rr() * 0.4) * k;
      const x0 = x + (rr() - 0.5) * 20;
      ctx.strokeStyle = `rgba(70,70,75,${0.55 * fade})`;
      ctx.lineWidth = 1.5 + rr() * 2;
      ctx.beginPath(); ctx.moveTo(x0, oy); ctx.lineTo(x0 + (e.dir || 1) * L, oy + (rr() - 0.5) * 4); ctx.stroke();
    }
  },
};

// ---------- gelo subindo pela haste da lança ----------
FX.spearFrost = {
  layer: 'front', life: 0.5,
  draw(ctx, e, age, env) {
    const S = SCENE.fighters.B.at(env.t);
    if (!S.tip) return;
    const c = e.from ? P(e.from, env.t) : S.tip;
    const b = S.butt, tip = S.tip;
    const dx = b[0] - tip[0], dy = b[1] - tip[1], L = Math.hypot(dx, dy);
    const ux = dx / L, uy = dy / L;
    // projeção do ponto de contato na haste
    const s0 = clamp((c[0] - tip[0]) * ux + (c[1] - tip[1]) * uy, 0, L);
    const reach = Math.min(L, s0 + (e.speed || 300) * age) - s0;
    const fade = Math.min(1, (e.life - age) / 0.12);
    const rr = rng(e.seed);
    ctx.globalAlpha = fade;
    for (let d = -20; d < reach; d += 9) {
      const s = s0 + d;
      if (s < 0) continue;
      const px = tip[0] + ux * s, py = tip[1] + uy * s;
      const side = rr() < 0.5 ? -1 : 1, h = 5 + rr() * 9;
      const nx = -uy * side, ny = ux * side;
      ctx.fillStyle = rr() < 0.5 ? '#dff7ff' : '#b4ebff';
      poly(ctx, [[px - ux * 4, py - uy * 4], [px + nx * h + ux * 2, py + ny * h + uy * 2], [px + ux * 5, py + uy * 5]]);
      ctx.closePath(); ctx.fill();
      ctx.strokeStyle = '#3c9fd2'; ctx.lineWidth = 1.2; ctx.stroke();
    }
    ctx.strokeStyle = 'rgba(225,250,255,0.95)'; ctx.lineWidth = 7; ctx.lineCap = 'round';
    poly(ctx, [[tip[0] + ux * Math.max(0, s0 - 20), tip[1] + uy * Math.max(0, s0 - 20)], [tip[0] + ux * (s0 + reach), tip[1] + uy * (s0 + reach)]]);
    ctx.stroke();
    ctx.globalAlpha = 1;
  },
};

// ---------- projétil de gelo ----------
FX.iceShot = {
  layer: 'front', life: 0.3,
  draw(ctx, e, age, env) {
    const a = e.a, b = e.b;
    const k = Math.min(1, age / e.life);
    const arc = e.arc || 0;
    const x = lerp(a[0], b[0], k), y = lerp(a[1], b[1], k) - Math.sin(k * Math.PI) * arc;
    const ang = Math.atan2(b[1] - a[1] - Math.cos(k * Math.PI) * arc * Math.PI, b[0] - a[0]);
    const c = Math.cos(ang), s = Math.sin(ang);
    const sz = e.size || 14;
    // rastro
    const trail = [];
    for (let i = 0; i < 8; i++) {
      const kk = Math.max(0, k - i * 0.035);
      trail.push([lerp(a[0], b[0], kk), lerp(a[1], b[1], kk) - Math.sin(kk * Math.PI) * arc]);
    }
    ctx.fillStyle = 'rgba(210,245,255,0.7)';
    taper(ctx, trail, sz * 0.9, 0.5); ctx.fill();
    ctx.globalCompositeOperation = 'lighter';
    const g = ctx.createRadialGradient(x, y, 0, x, y, sz * 2.2);
    g.addColorStop(0, 'rgba(200,245,255,0.7)');
    g.addColorStop(1, 'rgba(100,200,255,0)');
    ctx.fillStyle = g; circle(ctx, x, y, sz * 2.2); ctx.fill();
    ctx.globalCompositeOperation = 'source-over';
    const pts = [[sz * 1.6, 0], [0, sz * 0.45], [-sz * 1.1, 0], [0, -sz * 0.45]].map(([px, py]) => [x + px * c - py * s, y + px * s + py * c]);
    ctx.fillStyle = '#e6fbff';
    poly(ctx, pts); ctx.closePath(); ctx.fill();
    ctx.strokeStyle = '#2f93c7'; ctx.lineWidth = 2; ctx.stroke();
    void env;
  },
};

// ---------- redemoinho de vento ----------
FX.wind = {
  layer: 'front', life: 0.6,
  draw(ctx, e, age, env) {
    const [x, y] = pos(e, env.t);
    const k = age / e.life;
    const rr = rng(e.seed);
    const R = (e.r || 80);
    ctx.lineCap = 'round';
    for (let i = 0; i < (e.n || 5); i++) {
      const r0 = R * (0.55 + rr() * 0.6) * (0.8 + k * 0.4);
      const a0 = rr() * Math.PI * 2 + age * (e.spin || 14) * (e.cw === false ? -1 : 1);
      const span = 0.9 + rr() * 1.3;
      const al = (1 - k) * Math.min(1, age / 0.06) * (e.alpha || 0.8);
      ctx.strokeStyle = `rgba(${e.color || '255,255,255'},${al})`;
      ctx.lineWidth = (e.w || 3) * (0.6 + rr());
      ctx.beginPath();
      ctx.ellipse(x, y, r0, r0 * (e.squash || 0.45), (e.tilt || 0) * DEG, a0, a0 + span);
      ctx.stroke();
    }
  },
};

// ---------- partículas convergindo (carregando poder) ----------
FX.charge = {
  layer: 'front', life: 1.0,
  draw(ctx, e, age, env) {
    const [x, y] = pos(e, env.t);
    const rr = rng(e.seed);
    const n = e.n || 26;
    const col = e.color || '200,245,255';
    ctx.lineCap = 'round';
    const env0 = Math.min(1, age / 0.15) * Math.min(1, (e.life - age) / 0.15);
    for (let i = 0; i < n; i++) {
      const ang = rr() * Math.PI * 2;
      const R0 = (e.r || 160) * (0.6 + rr() * 0.6);
      const ph = rr();
      const rate = 1.6 + rr() * 1.2;
      const f = (age * rate + ph) % 1;
      const rad = R0 * Math.pow(1 - f, 1.4);
      const px = x + Math.cos(ang) * rad, py = y + Math.sin(ang) * rad * (e.squash || 1);
      const L = 6 + 22 * (1 - f);
      ctx.strokeStyle = `rgba(${col},${env0 * Math.min(1, f * 4) * 0.9})`;
      ctx.lineWidth = (e.w || 2.4);
      ctx.beginPath(); ctx.moveTo(px, py);
      ctx.lineTo(px + Math.cos(ang) * L, py + Math.sin(ang) * L * (e.squash || 1)); ctx.stroke();
    }
  },
};

// ---------- linhas de velocidade (tela) ----------
FX.speedlines = {
  layer: 'screen', life: 0.4,
  draw(ctx, e, age) {
    const rr = rng(e.seed + Math.floor(age * 30));
    const k = age / e.life;
    const al = Math.sin(k * Math.PI) * (e.alpha || 0.5);
    ctx.strokeStyle = `rgba(255,255,255,${al})`;
    ctx.lineCap = 'round';
    for (let i = 0; i < (e.n || 18); i++) {
      const y = CFG.BAND_Y + rr() * CFG.BAND_H;
      const x = rr() * CFG.W;
      const L = (120 + rr() * 300) * CFG.SCALE;
      ctx.lineWidth = (1 + rr() * 2.5) * CFG.SCALE;
      ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + (e.dir || 1) * L, y); ctx.stroke();
    }
  },
};

// ---------- ideograma gigante ----------
FX.glyph = {
  layer: 'back', life: 1.2,
  draw(ctx, e, age, env) {
    const [x, y] = pos(e, env.t);
    const k = age / e.life;
    const al = Math.min(1, age / 0.12) * Math.min(1, (e.life - age) / 0.35) * (e.alpha || 0.5);
    const sc = 1.25 - 0.25 * Ease.o(Math.min(1, age / 0.25));
    ctx.translate(x, y);
    ctx.scale(sc, sc);
    ctx.font = `${e.size || 260}px "WenQuanYi Zen Hei", sans-serif`;
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillStyle = `rgba(${e.color || '120,210,245'},${al})`;
    ctx.fillText(e.ch, 0, 0);
    ctx.lineWidth = 3;
    ctx.strokeStyle = `rgba(255,255,255,${al})`;
    ctx.strokeText(e.ch, 0, 0);
    void k;
  },
};

// ---------- selo de gelo no chão (bagua) ----------
const TRIGRAMS = [[1, 1, 1], [0, 1, 1], [1, 0, 1], [0, 0, 1], [1, 1, 0], [0, 1, 0], [1, 0, 0], [0, 0, 0]];
FX.seal = {
  layer: 'ground', life: 3,
  draw(ctx, e, age, env) {
    const [x, y] = pos(e, env.t);
    const g0 = Ease.back(Math.min(1, age / 0.35));
    const fade = Math.min(1, (e.life - age) / 0.4);
    const R = (e.r || 150) * g0;
    const sq = e.squash || 0.2;
    const rot = age * (e.spin || 0.9);
    ctx.globalCompositeOperation = 'source-over';
    const g = ctx.createRadialGradient(x, y, 0, x, y, R * 1.1);
    g.addColorStop(0, `rgba(220,248,255,${0.6 * fade})`);
    g.addColorStop(1, 'rgba(120,210,250,0)');
    ctx.fillStyle = g;
    ctx.beginPath(); ctx.ellipse(x, y, R * 1.1, R * 1.1 * sq, 0, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = `rgba(40,150,210,${0.9 * fade})`;
    for (const [rf, lw] of [[1, 3], [0.86, 1.6], [0.5, 2]]) {
      ctx.lineWidth = lw;
      ctx.beginPath(); ctx.ellipse(x, y, R * rf, R * rf * sq, 0, 0, Math.PI * 2); ctx.stroke();
    }
    // trigramas
    for (let i = 0; i < 8; i++) {
      const a = rot + i / 8 * Math.PI * 2;
      const tri = TRIGRAMS[i];
      for (let j = 0; j < 3; j++) {
        const rr0 = R * (0.6 + j * 0.075);
        const half = 0.16;
        const segs = tri[j] ? [[-half, half]] : [[-half, -0.03], [0.03, half]];
        for (const [s0, s1] of segs) {
          ctx.lineWidth = 3;
          ctx.beginPath();
          const p0 = [x + Math.cos(a + s0) * rr0, y + Math.sin(a + s0) * rr0 * sq];
          const p1 = [x + Math.cos(a + s1) * rr0, y + Math.sin(a + s1) * rr0 * sq];
          ctx.moveTo(p0[0], p0[1]); ctx.lineTo(p1[0], p1[1]); ctx.stroke();
        }
      }
    }
    // taiji central simplificado
    ctx.fillStyle = `rgba(235,252,255,${0.9 * fade})`;
    ctx.beginPath(); ctx.ellipse(x, y, R * 0.3, R * 0.3 * sq, 0, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = `rgba(40,150,210,${0.9 * fade})`;
    ctx.beginPath(); ctx.ellipse(x, y, R * 0.3, R * 0.3 * sq, 0, rot * 2, rot * 2 + Math.PI); ctx.fill();
    // luz subindo (coluna suave)
    ctx.globalCompositeOperation = 'lighter';
    ctx.save();
    ctx.translate(x, y - 90 * g0);
    ctx.scale(1, 1.5);
    const lg = ctx.createRadialGradient(0, 0, 0, 0, 0, R * 0.95);
    lg.addColorStop(0, `rgba(140,220,255,${0.28 * fade})`);
    lg.addColorStop(1, 'rgba(140,220,255,0)');
    ctx.fillStyle = lg;
    circle(ctx, 0, 0, R * 0.95); ctx.fill();
    ctx.restore();
  },
};

// ---------- prisão de cristal ----------
FX.encase = {
  layer: 'midB', life: 2,
  draw(ctx, e, age, env) {
    const rr = rng(e.seed);
    const base = e.follow ? P(e.p, env.t) : e.p;
    const x0 = base[0];
    ctx.translate(0, base[1]);
    const H = e.h || 70;
    const grow = Ease.back(Math.min(1, age / (e.grow || 0.18)));
    const sh = e.shatter || 999;
    if (age >= sh) {
      FX.shards.draw(ctx, { seed: e.seed + 4, p: [x0, -H * 0.5], n: e.nShards || 22, speed: e.shSpeed || 650, size: 12, dir: -90, spread: 200, w: 60, h: H, life: e.life - sh }, age - sh);
      return;
    }
    if (e.block) {
      const w = (e.w || 70) * 0.62, Hh = H * grow;
      const blk = [[x0 - w, 4], [x0 - w - 10, -Hh * 0.5], [x0 - w * 0.45, -Hh * 0.98], [x0 + w * 0.35, -Hh * 1.04], [x0 + w + 8, -Hh * 0.62], [x0 + w, 4]];
      const g = ctx.createLinearGradient(x0 - w, -Hh, x0 + w, 0);
      g.addColorStop(0, 'rgba(215,246,255,0.62)');
      g.addColorStop(1, 'rgba(110,196,238,0.5)');
      ctx.fillStyle = g;
      poly(ctx, blk); ctx.closePath(); ctx.fill();
      ctx.strokeStyle = '#2f8fc4'; ctx.lineWidth = 2.5; ctx.stroke();
      ctx.strokeStyle = 'rgba(255,255,255,0.85)'; ctx.lineWidth = 2;
      poly(ctx, [[x0 - w * 0.6, -Hh * 0.2], [x0 - w * 0.2, -Hh * 0.85]]); ctx.stroke();
      poly(ctx, [[x0 + w * 0.5, -Hh * 0.3], [x0 + w * 0.7, -Hh * 0.55]]); ctx.stroke();
    }
    const n = e.n || 7;
    const cr = [];
    for (let i = 0; i < n; i++) {
      const u = n === 1 ? 0.5 : i / (n - 1);
      cr.push({ x: x0 + (u - 0.5) * (e.w || 70) + (rr() - 0.5) * 14, h: H * (0.6 + rr() * 0.5), w: 14 + rr() * 12, lean: (u - 0.5) * 40 + (rr() - 0.5) * 20 });
    }
    cr.sort((a, b) => a.h - b.h);
    for (const c of cr) drawCrystal(ctx, c.x, 4, c.h * grow, c.w, c.lean, e.alpha || 0.78, { c0: 'rgba(95,192,236,0.85)', c1: 'rgba(169,231,255,0.8)', c2: 'rgba(244,254,255,0.9)' });
    if (e.crack && age > e.crack) {
      const k = Math.min(1, (age - e.crack) / 0.08);
      const rr2 = rng(e.seed + 2);
      ctx.strokeStyle = 'rgba(255,255,255,0.95)'; ctx.lineWidth = 2;
      for (let i = 0; i < 8; i++) {
        const cp = [x0 + (rr2() - 0.5) * 40, -H * (0.2 + rr2() * 0.6)];
        const ang = rr2() * Math.PI * 2, L = (15 + rr2() * 40) * k;
        poly(ctx, jagged(cp, [cp[0] + Math.cos(ang) * L, cp[1] + Math.sin(ang) * L], rr2, 3, 5)); ctx.stroke();
      }
      if (e.zapCrack) {
        const rr3 = rng(e.seed + Math.floor(age * 30));
        for (let i = 0; i < 3; i++) {
          const a = [x0 + (rr3() - 0.5) * 50, -rr3() * H], b = [x0 + (rr3() - 0.5) * 70, -rr3() * H];
          drawElectric(ctx, jagged(a, b, rr3, 5, 10), 0.9, 0.8);
        }
      }
    }
  },
};

// ---------- dragão de gelo ----------
function catmull(pts, n) {
  const out = [];
  for (let i = 0; i < pts.length - 1; i++) {
    const p0 = pts[Math.max(0, i - 1)], p1 = pts[i], p2 = pts[i + 1], p3 = pts[Math.min(pts.length - 1, i + 2)];
    for (let j = 0; j < n; j++) {
      const t = j / n, t2 = t * t, t3 = t2 * t;
      out.push([0, 1].map(c => 0.5 * ((2 * p1[c]) + (-p0[c] + p2[c]) * t + (2 * p0[c] - 5 * p1[c] + 4 * p2[c] - p3[c]) * t2 + (-p0[c] + 3 * p1[c] - 3 * p2[c] + p3[c]) * t3)));
    }
  }
  out.push(pts[pts.length - 1]);
  return out;
}
function arcTable(pts) {
  const s = [0];
  for (let i = 1; i < pts.length; i++) s.push(s[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]));
  return s;
}
function atArc(pts, S, s) {
  if (s <= 0) return [pts[0][0], pts[0][1], Math.atan2(pts[1][1] - pts[0][1], pts[1][0] - pts[0][0])];
  const tot = S[S.length - 1];
  if (s >= tot) {
    const n = pts.length;
    const a = Math.atan2(pts[n - 1][1] - pts[n - 2][1], pts[n - 1][0] - pts[n - 2][0]);
    return [pts[n - 1][0] + Math.cos(a) * (s - tot), pts[n - 1][1] + Math.sin(a) * (s - tot), a];
  }
  let lo = 0, hi = S.length - 1;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (S[m] <= s) lo = m; else hi = m; }
  const u = (s - S[lo]) / (S[hi] - S[lo] || 1);
  const a = Math.atan2(pts[hi][1] - pts[lo][1], pts[hi][0] - pts[lo][0]);
  return [lerp(pts[lo][0], pts[hi][0], u), lerp(pts[lo][1], pts[hi][1], u), a];
}

FX.dragon = {
  layer: 'front', life: 3,
  draw(ctx, e, age, env) {
    if (!e._path) {
      e._path = catmull(e.path, 16);
      e._S = arcTable(e._path);
      e._sk = new Track(e.sk.map(([t, v, ee]) => ({ t, v, e: ee || 's' })));
    }
    const path = e._path, S = e._S;
    const sh = e._sk.at(age);
    const len = e.len || 560;
    const W = e.w || 24;
    const N = 70;
    const fade = Math.min(1, (e.life - age) / (e.fadeT || 0.3));
    const spine = [];
    for (let i = 0; i <= N; i++) {
      const u = i / N;
      const s = sh - u * len;
      if (s < 0) break;
      const [x, y, a] = atArc(path, S, s);
      // ondulação
      const wv = Math.sin(u * 9 - age * 9) * W * 0.35 * Math.min(1, u * 4);
      spine.push({ x: x - Math.sin(a) * wv, y: y + Math.cos(a) * wv, a, u });
    }
    if (spine.length < 3) return;
    const width = u => W * (u < 0.08 ? lerp(0.7, 1, u / 0.08) : lerp(1, 0.12, Math.pow((u - 0.08) / 0.92, 1.2)));
    const Lp = [], Rp = [];
    for (let i = 0; i < spine.length; i++) {
      const p = spine[i], q = spine[Math.min(spine.length - 1, i + 1)], o = spine[Math.max(0, i - 1)];
      const a = Math.atan2(o.y - q.y, o.x - q.x);
      const w = width(p.u);
      Lp.push([p.x - Math.sin(a) * w, p.y + Math.cos(a) * w]);
      Rp.push([p.x + Math.sin(a) * w, p.y - Math.cos(a) * w]);
      p.dir = a;
    }
    ctx.globalAlpha = fade;
    // espinhos dorsais
    ctx.fillStyle = '#d6f6ff'; ctx.strokeStyle = '#2b86b8'; ctx.lineWidth = 1.8;
    for (let i = 3; i < spine.length - 2; i += 3) {
      const p = spine[i];
      const w = width(p.u);
      const a = p.dir;
      const bx = p.x + Math.sin(a) * w * 0.9, by = p.y - Math.cos(a) * w * 0.9;
      const hs = w * 0.9 + 4;
      const tx = bx + Math.sin(a) * hs + Math.cos(a) * hs * 0.6, ty = by - Math.cos(a) * hs + Math.sin(a) * hs * 0.6;
      poly(ctx, [[bx - Math.cos(a) * w * 0.5, by - Math.sin(a) * w * 0.5], [tx, ty], [bx + Math.cos(a) * w * 0.4, by + Math.sin(a) * w * 0.4]]);
      ctx.closePath(); ctx.fill(); ctx.stroke();
    }
    // patas
    for (const lu of [0.2, 0.5]) {
      const idx = Math.round(lu * N);
      if (idx >= spine.length) continue;
      const p = spine[idx];
      const a = p.dir, w = width(p.u);
      for (const side of [-1, 1]) {
        const sw = Math.sin(age * 10 + lu * 7 + side) * 0.5;
        const bx = p.x - Math.sin(a) * w * 0.7, by = p.y + Math.cos(a) * w * 0.7;
        const la = a + Math.PI / 2 + side * 0.5 + sw;
        const kx = bx + Math.cos(la) * W * 1.1, ky = by + Math.sin(la) * W * 1.1;
        const fx = kx + Math.cos(la - 0.9) * W * 0.9, fy = ky + Math.sin(la - 0.9) * W * 0.9;
        ctx.strokeStyle = '#2b86b8'; ctx.lineWidth = 7; ctx.lineCap = 'round';
        poly(ctx, [[bx, by], [kx, ky], [fx, fy]]); ctx.stroke();
        ctx.strokeStyle = '#c9f2ff'; ctx.lineWidth = 4;
        poly(ctx, [[bx, by], [kx, ky], [fx, fy]]); ctx.stroke();
        for (let c = -1; c <= 1; c++) {
          const ca = la - 0.9 + c * 0.5;
          ctx.strokeStyle = '#2b86b8'; ctx.lineWidth = 2;
          poly(ctx, [[fx, fy], [fx + Math.cos(ca) * 8, fy + Math.sin(ca) * 8]]); ctx.stroke();
        }
      }
    }
    // corpo
    ctx.save();
    ctx.shadowColor = 'rgba(110,210,255,0.9)';
    ctx.shadowBlur = 18;
    const hx = spine[0].x, hy = spine[0].y, tx = spine[spine.length - 1].x, ty = spine[spine.length - 1].y;
    const gr = ctx.createLinearGradient(hx, hy, tx, ty);
    gr.addColorStop(0, '#f2fdff'); gr.addColorStop(0.5, '#bdeeff'); gr.addColorStop(1, '#7fd2f5');
    ctx.fillStyle = gr;
    ctx.beginPath();
    ctx.moveTo(Lp[0][0], Lp[0][1]);
    for (let i = 1; i < Lp.length; i++) ctx.lineTo(Lp[i][0], Lp[i][1]);
    for (let i = Rp.length - 1; i >= 0; i--) ctx.lineTo(Rp[i][0], Rp[i][1]);
    ctx.closePath(); ctx.fill();
    ctx.restore();
    ctx.strokeStyle = '#2b86b8'; ctx.lineWidth = 2.4;
    ctx.beginPath();
    ctx.moveTo(Lp[0][0], Lp[0][1]);
    for (let i = 1; i < Lp.length; i++) ctx.lineTo(Lp[i][0], Lp[i][1]);
    for (let i = Rp.length - 1; i >= 0; i--) ctx.lineTo(Rp[i][0], Rp[i][1]);
    ctx.closePath(); ctx.stroke();
    // escamas (chevrons)
    ctx.strokeStyle = 'rgba(43,134,184,0.75)'; ctx.lineWidth = 1.6;
    for (let i = 2; i < spine.length - 1; i += 2) {
      const p = spine[i], a = p.dir, w = width(p.u) * 0.75;
      const c = Math.cos(a), s = Math.sin(a);
      poly(ctx, [[p.x - s * w + c * w * 0.5, p.y + c * w + s * w * 0.5], [p.x - c * w * 0.3, p.y - s * w * 0.3], [p.x + s * w + c * w * 0.5, p.y - c * w + s * w * 0.5]]);
      ctx.stroke();
    }
    // linha da barriga
    ctx.strokeStyle = 'rgba(255,255,255,0.9)'; ctx.lineWidth = 2;
    poly(ctx, spine.map((p, i) => [lerp(p.x, Lp[i][0], 0.5), lerp(p.y, Lp[i][1], 0.5)])); ctx.stroke();
    // cabeça
    const h = spine[0];
    drawDragonHead(ctx, h.x, h.y, h.dir + Math.PI, W, age, e);
    // bigodes
    for (const side of [-1, 1]) {
      const wp = [];
      for (let i = 0; i < 9; i++) {
        const p = spine[Math.min(spine.length - 1, i * 2)];
        const a = p.dir;
        const off = side * (W * 0.9 + i * 2.5) + Math.sin(age * 8 + i * 0.8 + side) * i * 1.8;
        wp.push([p.x - Math.sin(a) * off + Math.cos(a) * W * 1.3, p.y + Math.cos(a) * off + Math.sin(a) * W * 1.3]);
      }
      ctx.strokeStyle = 'rgba(230,250,255,0.95)'; ctx.lineWidth = 2.5;
      poly(ctx, wp); ctx.stroke();
    }
    // brilhos de neve no corpo
    const rr = rng(e.seed + Math.floor(env.frame / 3));
    ctx.fillStyle = 'rgba(255,255,255,0.95)';
    for (let i = 0; i < 14; i++) {
      const p = spine[Math.floor(rr() * spine.length)];
      const px = p.x + (rr() - 0.5) * W * 2.4, py = p.y + (rr() - 0.5) * W * 2.4, s = 1.5 + rr() * 2.5;
      poly(ctx, [[px, py - s * 2], [px + s * 0.5, py], [px, py + s * 2], [px - s * 0.5, py]]); ctx.closePath(); ctx.fill();
    }
    ctx.globalAlpha = 1;
    if (e.electric) {
      const zr = rng(e.seed + Math.floor(env.frame / 2) * 11);
      const amt = e.electric(age);
      for (let i = 0; i < Math.round(amt * 6); i++) {
        const i0 = Math.floor(zr() * spine.length), i1 = Math.min(spine.length - 1, i0 + 2 + Math.floor(zr() * 8));
        drawElectric(ctx, jagged([spine[i0].x, spine[i0].y], [spine[i1].x, spine[i1].y], zr, 5, W * 0.8), amt, 0.9);
      }
    }
  },
};

function drawDragonHead(ctx, x, y, a, W, age, e) {
  // a = direção para onde a cabeça aponta
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(a);
  const flip = Math.cos(a) < 0 ? -1 : 1;
  ctx.scale(W / 24, (W / 24) * flip);
  const jaw = (e.jaw ? e.jaw(age) : 0.5) * 0.5;
  ctx.shadowColor = 'rgba(110,210,255,0.9)';
  ctx.shadowBlur = 14;
  ctx.lineJoin = 'round';
  // chifres
  ctx.strokeStyle = '#2b86b8'; ctx.lineWidth = 6; ctx.lineCap = 'round';
  const horn = [[-8, -16], [-34, -34], [-52, -38]];
  poly(ctx, horn); ctx.stroke();
  poly(ctx, [[-30, -31], [-36, -46]]); ctx.stroke();
  ctx.strokeStyle = '#e6fbff'; ctx.lineWidth = 3.5;
  poly(ctx, horn); ctx.stroke();
  poly(ctx, [[-30, -31], [-36, -46]]); ctx.stroke();
  // crina
  ctx.fillStyle = '#bfefff'; ctx.strokeStyle = '#2b86b8'; ctx.lineWidth = 2;
  for (let i = 0; i < 4; i++) {
    const bx = -10 - i * 9;
    poly(ctx, [[bx, -12 + i * 2], [bx - 22 - i * 4, -26 + i * 9 + Math.sin(age * 9 + i) * 3], [bx - 8, -2 + i * 2]]);
    ctx.closePath(); ctx.fill(); ctx.stroke();
  }
  // mandíbula superior
  const ja = jaw;
  ctx.fillStyle = '#f2fdff';
  ctx.save(); ctx.rotate(-ja);
  poly(ctx, [[-16, -16], [12, -16], [40, -8], [50, -2], [42, 2], [-14, 4]]); ctx.closePath(); ctx.fill();
  ctx.shadowBlur = 0; ctx.stroke();
  ctx.fillStyle = '#ffffff';
  for (let i = 0; i < 4; i++) { poly(ctx, [[14 + i * 8, 2], [18 + i * 8, 10], [22 + i * 8, 2]]); ctx.closePath(); ctx.fill(); ctx.stroke(); }
  ctx.restore();
  // mandíbula inferior
  ctx.save(); ctx.rotate(ja);
  ctx.fillStyle = '#d4f5ff';
  poly(ctx, [[-14, 2], [36, 6], [44, 10], [10, 16], [-12, 12]]); ctx.closePath(); ctx.fill(); ctx.stroke();
  ctx.fillStyle = '#ffffff';
  for (let i = 0; i < 3; i++) { poly(ctx, [[16 + i * 8, 6], [20 + i * 8, -1], [24 + i * 8, 7]]); ctx.closePath(); ctx.fill(); ctx.stroke(); }
  ctx.restore();
  // olho
  ctx.fillStyle = '#0d3d5c';
  ctx.beginPath(); ctx.ellipse(4, -9 - ja * 10, 4.5, 2.6, -0.2, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = '#ffffff';
  circle(ctx, 5.5, -10 - ja * 10, 1.2); ctx.fill();
  ctx.restore();
}

// ---------- neve e chuva (tela) ----------
function drawSnow(ctx, env) {
  const amt = envAt('snow', env.t);
  if (amt < 0.02) return;
  const n = Math.round(90 * amt);
  const rr = rng(4242);
  ctx.fillStyle = 'rgba(255,255,255,0.9)';
  for (let i = 0; i < n; i++) {
    const K = CFG.SCALE;
    const sx = rr(), sy = rr(), sp = (30 + rr() * 50) * K, s = (1.5 + rr() * 2.8) * K, ph = rr() * 6.28;
    const x = ((sx * CFG.W + (Math.sin(env.r * 1.3 + ph) * 25 - env.cam.cx * 0.2 + env.r * 12) * K) % CFG.W + CFG.W) % CFG.W;
    const y = CFG.BAND_Y + (((sy * CFG.BAND_H + env.r * sp) % CFG.BAND_H) + CFG.BAND_H) % CFG.BAND_H;
    circle(ctx, x, y, s); ctx.fill();
  }
}
function drawRain(ctx, env) {
  const amt = envAt('rain', env.t);
  if (amt < 0.02) return;
  const n = Math.round(120 * amt);
  const rr = rng(777);
  ctx.strokeStyle = 'rgba(225,230,255,0.45)';
  ctx.lineWidth = 1.5;
  for (let i = 0; i < n; i++) {
    const K = CFG.SCALE;
    const sx = rr(), sy = rr(), sp = (1400 + rr() * 600) * K;
    const x = ((sx * CFG.W - env.r * 320 * K) % CFG.W + CFG.W) % CFG.W;
    const y = CFG.BAND_Y + (((sy * CFG.BAND_H + env.r * sp) % CFG.BAND_H) + CFG.BAND_H) % CFG.BAND_H;
    ctx.lineWidth = 1.5 * K;
    ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x - 7 * K, y + 26 * K); ctx.stroke();
  }
}

// ---------- pós-processamento ----------
function drawPost(ctx, env) {
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  let flash = 0, col = '255,255,255', inv = false;
  for (const e of SCENE.fx) {
    const age = env.t - e.t;
    if (e.type === 'flash') {
      const ra = env.r - e.rt;
      if (ra < 0 || ra > e.life) continue;
      const a = (e.amt || 1) * Math.pow(1 - ra / e.life, 2);
      if (a > flash) { flash = a; col = e.color || '255,255,255'; }
    } else if (e.type === 'invert') {
      const ra = env.r - e.rt;
      if (ra >= 0 && ra < e.life) inv = true;
    }
    void age;
  }
  if (inv) {
    ctx.globalCompositeOperation = 'difference';
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, CFG.BAND_Y, CFG.W, CFG.BAND_H);
    ctx.globalCompositeOperation = 'saturation';
    ctx.fillStyle = '#000';
    ctx.fillRect(0, CFG.BAND_Y, CFG.W, CFG.BAND_H);
    ctx.globalCompositeOperation = 'source-over';
  }
  if (flash > 0.003) {
    ctx.fillStyle = `rgba(${col},${Math.min(1, flash)})`;
    ctx.fillRect(0, CFG.BAND_Y, CFG.W, CFG.BAND_H);
  }
  // vinheta leve
  const dark = envAt('vig', env.t, 0.25);
  if (dark > 0.01) {
    const g = ctx.createRadialGradient(CFG.W / 2, CFG.BAND_Y + CFG.BAND_H / 2, CFG.BAND_H * 0.45, CFG.W / 2, CFG.BAND_Y + CFG.BAND_H / 2, CFG.W * 0.75);
    g.addColorStop(0, 'rgba(0,0,0,0)');
    g.addColorStop(1, `rgba(0,0,0,${dark * 0.5})`);
    ctx.fillStyle = g;
    ctx.fillRect(0, CFG.BAND_Y, CFG.W, CFG.BAND_H);
  }
}
