// Motor de animação procedural de luta de bonecos palito.
// Tudo é função pura do tempo: renderFrame(f) desenha o quadro f sem estado.
'use strict';

const CFG = {
  W: 1080, H: 1920,
  BAND_Y: 654, BAND_H: 612,
  FPS: 60, DURATION: 30,
  CAPTION: 'gelo vs tempestade',
  SCALE: 1, VH: 612,
};
// versão horizontal 16:9 sem legenda: ?wide=1
if (typeof location !== 'undefined' && /[?&]wide=1/.test(location.search)) {
  Object.assign(CFG, { W: 1920, H: 1080, BAND_Y: 0, BAND_H: 1080, CAPTION: '', SCALE: 1920 / 1080 });
  CFG.VH = CFG.BAND_H / CFG.SCALE;
}

// ---------- matemática ----------
const DEG = Math.PI / 180;
const clamp = (x, a, b) => (x < a ? a : x > b ? b : x);
const lerp = (a, b, t) => a + (b - a) * t;
const inv = (a, b, x) => clamp((x - a) / (b - a), 0, 1);
const sstep = (a, b, x) => { const t = inv(a, b, x); return t * t * (3 - 2 * t); };

const Ease = {
  l: t => t,
  io: t => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2),
  sine: t => -(Math.cos(Math.PI * t) - 1) / 2,
  i: t => t * t * t,
  i2: t => t * t,
  o: t => 1 - Math.pow(1 - t, 3),
  o2: t => 1 - (1 - t) * (1 - t),
  snap: t => (t >= 1 ? 1 : 1 - Math.pow(2, -10 * t)),
  back: t => { const c1 = 1.70158, c3 = c1 + 1; return 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2); },
  h: t => (t < 1 ? 0 : 1),
};

function rng(seed) {
  let a = (seed * 2654435761) >>> 0;
  return () => {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const hash = (n) => rng(n * 7919 + 17)();

// ---------- trilhas de keyframes ----------
class Track {
  constructor(keys) { this.k = keys; }
  at(t) {
    const k = this.k, n = k.length;
    if (n === 1 || t <= k[0].t) return k[0].v;
    if (t >= k[n - 1].t) return k[n - 1].v;
    let lo = 0, hi = n - 1;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (k[m].t <= t) lo = m; else hi = m; }
    const a = k[lo], b = k[hi], dt = b.t - a.t;
    if (dt <= 0) return b.v;
    const u = (t - a.t) / dt;
    const e = b.e || 's';
    if (e !== 's') return a.v + (b.v - a.v) * (Ease[e] || Ease.io)(u);
    if (a.v === b.v) return a.v;
    const m0 = this.tan(lo), m1 = this.tan(hi);
    const u2 = u * u, u3 = u2 * u;
    return (2 * u3 - 3 * u2 + 1) * a.v + (u3 - 2 * u2 + u) * dt * m0 + (-2 * u3 + 3 * u2) * b.v + (u3 - u2) * dt * m1;
  }
  // tangente com limitador (Fritsch-Carlson) para evitar overshoot
  tan(i) {
    const k = this.k, n = k.length;
    if (i <= 0 || i >= n - 1) return 0;
    const p = k[i - 1], c = k[i], q = k[i + 1];
    if ((q.e || 's') !== 's' && (c.e || 's') !== 's') return 0;
    const d0 = (c.v - p.v) / Math.max(1e-6, c.t - p.t);
    const d1 = (q.v - c.v) / Math.max(1e-6, q.t - c.t);
    if (d0 * d1 <= 0) return 0;
    let m = (q.v - p.v) / Math.max(1e-6, q.t - p.t);
    const lim = 3 * Math.min(Math.abs(d0), Math.abs(d1));
    if (Math.abs(m) > lim) m = Math.sign(m) * lim;
    return m;
  }
}

// ---------- rig do boneco ----------
const RIG = { torso: 56, ua: 33, fa: 33, th: 46, sh: 46, hr: 22, lw: 16, tw: 18 };

const POSE_DEFAULT = {
  x: 0, y: 0, face: 1, rot: 0, sq: 1,
  hip: [0, -88], ln: 4, hd: 0,
  hF: [6, 60], hB: [-2, 60],
  fF: [14, 0], fB: [-14, 0],
  sF: 1, sB: 1, kF: -1, kB: -1,
  // lança (ângulo no referencial do corpo, pega da mão da frente/trás ao longo da haste)
  sa: -80, sg: 100, sb: 40, sbw: 0, spear: 0,
  // espada de gelo
  sw: 0, swa: -60,
  gh: 0, glow: 0, zap: 0, bz: 0, vis: 1, tint: 0, grab: 1,
};
const HOLD_PARAMS = new Set(['sF', 'sB', 'kF', 'kB']);
const WRAP_PARAMS = new Set(['sa', 'swa', 'rot']);

function flattenPose(p, out) {
  for (const k in p) {
    const v = p[k];
    if (Array.isArray(v)) { out[k + '.0'] = v[0]; out[k + '.1'] = v[1]; }
    else out[k] = v;
  }
  return out;
}

class Fighter {
  constructor(id, opts) {
    this.id = id;
    Object.assign(this, { color: '#111', ribbon: null, weapon: null }, opts);
    this.keys = [];
    this.free = []; // intervalos com lança solta: {t0,t1,pivot,grab,keys:[{t,x,y,a,e}|{t,auto:true}]}
  }
  // k(t, pose, ease, rootEase)
  k(t, pose, e = 's', er) {
    this.keys.push({ t, pose, e, er });
    return this;
  }
  build(POSES) {
    // semântica "pose a pose": parâmetros não citados numa chave mantêm o valor anterior.
    // valor 'auto' = esta chave não fixa o parâmetro (interpola através dela).
    this.keys.sort((a, b) => a.t - b.t);
    const defs = flattenPose(POSE_DEFAULT, {});
    const cur = Object.assign({}, defs);
    const per = {};
    for (const key of this.keys) {
      let src = key.pose;
      if (typeof src === 'string') src = { p: src };
      const p = {};
      if (src.p) {
        const names = Array.isArray(src.p) ? src.p : [src.p];
        for (const nm of names) {
          if (!POSES[nm]) throw new Error('pose desconhecida ' + nm);
          flattenPose(POSES[nm], p);
        }
      }
      const own = Object.assign({}, src); delete own.p;
      const dx = own.dx, dy = own.dy; delete own.dx; delete own.dy;
      flattenPose(own, p);
      // sb < 0 = mão de trás solta; vira peso sbw=0 para interpolar sem saltos
      if (p.sb !== undefined) {
        if (p.sb < 0) { delete p.sb; if (own.sbw === undefined) p.sbw = 0; }
        else if (own.sbw === undefined) p.sbw = 1;
      }
      if (dx !== undefined) p.x = cur.x + dx;
      if (dy !== undefined) p.y = cur.y + dy;
      const skip = new Set();
      for (const nm in p) {
        if (p[nm] === 'auto') { skip.add(nm); continue; }
        // ângulos vindos de poses nomeadas seguem o caminho mais curto
        if (WRAP_PARAMS.has(nm) && own[nm] === undefined) p[nm] += 360 * Math.round((cur[nm] - p[nm]) / 360);
        cur[nm] = p[nm];
      }
      for (const nm in cur) {
        if (skip.has(nm)) continue;
        const isRoot = nm === 'x' || nm === 'y';
        let e = (isRoot && key.er) ? key.er : key.e;
        if (HOLD_PARAMS.has(nm)) e = 'h';
        (per[nm] = per[nm] || []).push({ t: key.t, v: cur[nm], e });
      }
    }
    this.tracks = {};
    for (const nm in per) {
      const arr = [];
      for (const k of per[nm]) {
        if (arr.length && Math.abs(arr[arr.length - 1].t - k.t) < 1e-9) arr[arr.length - 1] = k;
        else arr.push(k);
      }
      this.tracks[nm] = new Track(arr);
    }
    this.cache = new Map();
    // chaves 'auto' da lança solta copiam a lança empunhada naquele instante
    this._noFree = true;
    const L = this.spearLen || 0;
    for (const f of this.free) {
      f.keys.sort((a, b) => a.t - b.t);
      for (const kk of f.keys) {
        if (!kk.auto) continue;
        const S = solve(this.pose(kk.t), this, kk.t);
        const ux = (S.tip[0] - S.butt[0]) / L, uy = (S.tip[1] - S.butt[1]) / L;
        kk.a = Math.atan2(uy, ux) / DEG;
        const piv = f.pivot === 'butt' ? S.butt : f.pivot === 'tip' ? S.tip : [(S.tip[0] + S.butt[0]) / 2, (S.tip[1] + S.butt[1]) / 2];
        kk.x = piv[0]; kk.y = piv[1];
      }
      for (let i = 1; i < f.keys.length; i++) {
        const pa = f.keys[i - 1].a, k = f.keys[i];
        if (k.auto || k.unwrap) { while (k.a - pa > 180) k.a -= 360; while (k.a - pa < -180) k.a += 360; }
      }
      f.tracks = {};
      for (const nm of ['x', 'y', 'a']) {
        f.tracks[nm] = new Track(f.keys.map(kk => ({ t: kk.t, v: kk[nm], e: kk.e || 's' })));
      }
    }
    this._noFree = false;
    this.cache.clear();
  }
  pose(t) {
    const P = {};
    for (const nm in this.tracks) P[nm] = this.tracks[nm].at(t);
    return P;
  }
  at(t) {
    const key = Math.round(t * 6000);
    let s = this.cache.get(key);
    if (s) return s;
    s = solve(this.pose(t), this, t);
    if (this.cache.size > 4000) this.cache.clear();
    this.cache.set(key, s);
    return s;
  }
}

function ik(sx, sy, tx, ty, l1, l2, sign) {
  let dx = tx - sx, dy = ty - sy;
  let d = Math.hypot(dx, dy);
  if (d < 1e-4) { dx = 0; dy = 1; d = 1e-4; }
  const ux = dx / d, uy = dy / d;
  const max = l1 + l2;
  if (d >= max * 0.999) {
    const r = l1 / max;
    return [sx + dx * r, sy + dy * r, tx, ty];
  }
  const mind = Math.abs(l1 - l2) + 2;
  const dd = Math.max(d, mind);
  const c = clamp((l1 * l1 + dd * dd - l2 * l2) / (2 * l1 * dd), -1, 1);
  const a = Math.acos(c) * (sign >= 0 ? 1 : -1);
  const ca = Math.cos(a), sa = Math.sin(a);
  const ex = sx + l1 * (ux * ca - uy * sa);
  const ey = sy + l1 * (ux * sa + uy * ca);
  return [ex, ey, sx + ux * dd, sy + uy * dd];
}

// resolve pose -> pontos no mundo
function solve(P, F, t) {
  const R = RIG;
  const hip = [P['hip.0'], P['hip.1']];
  const ln = P.ln * DEG;
  const neck = [hip[0] + Math.sin(ln) * R.torso, hip[1] - Math.cos(ln) * R.torso];
  const sh = [lerp(neck[0], hip[0], 0.07), lerp(neck[1], hip[1], 0.07)];
  const ha = (P.ln + P.hd) * DEG;
  const head = [neck[0] + Math.sin(ha) * (R.hr + 2), neck[1] - Math.cos(ha) * (R.hr + 2)];
  let hF = [sh[0] + P['hF.0'], sh[1] + P['hF.1']];
  let hB = [sh[0] + P['hB.0'], sh[1] + P['hB.1']];
  const L = F.spearLen || 0;
  const rot = P.rot * DEG, cr = Math.cos(rot), sr = Math.sin(rot);
  const face = P.face, X = P.x, Y = P.y, sq = P.sq;
  const W = (px, py) => {
    const rx = px - hip[0], ry = py - hip[1];
    const qx = hip[0] + rx * cr - ry * sr;
    const qy = hip[1] + rx * sr + ry * cr;
    return [X + face * qx, Y + qy * sq];
  };
  const Winv = (wx, wy) => {
    const f = Math.abs(face) < 0.05 ? (face < 0 ? -0.05 : 0.05) : face;
    const qx = (wx - X) / f, qy = (wy - Y) / sq;
    const rx = qx - hip[0], ry = qy - hip[1];
    return [hip[0] + rx * cr + ry * sr, hip[1] - rx * sr + ry * cr];
  };
  let fr = null;
  if (F.weapon === 'spear' && !F._noFree) {
    for (const f of F.free) if (t >= f.t0 && t < f.t1) { fr = f; break; }
  }
  let freeSpear = null, spearPts = null;
  if (fr) {
    const px = fr.tracks.x.at(t), py = fr.tracks.y.at(t), a = fr.tracks.a.at(t) * DEG;
    const ux = Math.cos(a), uy = Math.sin(a);
    let butt;
    if (fr.pivot === 'butt') butt = [px, py];
    else if (fr.pivot === 'tip') butt = [px - ux * L, py - uy * L];
    else butt = [px - ux * L / 2, py - uy * L / 2];
    freeSpear = { butt, tip: [butt[0] + ux * L, butt[1] + uy * L] };
    if (fr.grab && P.grab > 0.5) {
      hF = Winv(butt[0] + ux * P.sg, butt[1] + uy * P.sg);
      if (P.sbw > 0.001) { const q = Winv(butt[0] + ux * P.sb, butt[1] + uy * P.sb); hB = [lerp(hB[0], q[0], P.sbw), lerp(hB[1], q[1], P.sbw)]; }
    }
  } else if (F.weapon === 'spear' && P.spear > 0.5) {
    const a = P.sa * DEG, ux = Math.cos(a), uy = Math.sin(a);
    const butt = [hF[0] - ux * P.sg, hF[1] - uy * P.sg];
    spearPts = { butt, tip: [butt[0] + ux * L, butt[1] + uy * L] };
    if (P.sbw > 0.001) hB = [lerp(hB[0], butt[0] + ux * P.sb, P.sbw), lerp(hB[1], butt[1] + uy * P.sb, P.sbw)];
  }
  const aF = ik(sh[0], sh[1], hF[0], hF[1], R.ua, R.fa, P.sF);
  const aB = ik(sh[0], sh[1], hB[0], hB[1], R.ua, R.fa, P.sB);
  const lF = ik(hip[0], hip[1], P['fF.0'], P['fF.1'], R.th, R.sh, P.kF);
  const lB = ik(hip[0], hip[1], P['fB.0'], P['fB.1'], R.th, R.sh, P.kB);
  const S = {
    P, F, t,
    hip: W(hip[0], hip[1]), neck: W(neck[0], neck[1]), sh: W(sh[0], sh[1]), head: W(head[0], head[1]),
    eF: W(aF[0], aF[1]), hF: W(aF[2], aF[3]),
    eB: W(aB[0], aB[1]), hB: W(aB[2], aB[3]),
    kF: W(lF[0], lF[1]), fF: W(lF[2], lF[3]),
    kB: W(lB[0], lB[1]), fB: W(lB[2], lB[3]),
    face, rotDeg: P.rot,
  };
  if (freeSpear) {
    S.butt = freeSpear.butt; S.tip = freeSpear.tip; S.spearFree = true;
  } else if (spearPts) {
    S.butt = W(spearPts.butt[0], spearPts.butt[1]);
    S.tip = W(spearPts.tip[0], spearPts.tip[1]);
  }
  if (F.weapon === 'ice' && P.sw > 0.01) {
    const a = P.swa * DEG;
    const len = 104 * P.sw;
    const base = [aF[2], aF[3]];
    S.swBase = W(base[0], base[1]);
    S.swTip = W(base[0] + Math.cos(a) * len, base[1] + Math.sin(a) * len);
  }
  return S;
}


// ---------- distorção de tempo (câmera lenta / congelamento) ----------
class TimeWarp {
  // segs: [{s0,s1,speed}]  freezes: [{s,dur}]
  constructor(segs, freezes, sceneEnd) {
    const pts = new Set([0, sceneEnd]);
    segs.forEach(g => { pts.add(g.s0); pts.add(g.s1); });
    freezes.forEach(f => pts.add(f.s));
    const S = [...pts].filter(s => s <= sceneEnd).sort((a, b) => a - b);
    const knots = [[0, 0]];
    let r = 0;
    for (let i = 0; i < S.length; i++) {
      const s = S[i];
      for (const f of freezes) if (Math.abs(f.s - s) < 1e-9) { r += f.dur; knots.push([r, s]); }
      if (i === S.length - 1) break;
      const s1 = S[i + 1], mid = (s + s1) / 2;
      let speed = 1;
      for (const g of segs) if (mid > g.s0 && mid < g.s1) speed = g.speed;
      r += (s1 - s) / speed;
      knots.push([r, s1]);
    }
    this.knots = knots;
    this.total = r;
  }
  scene(r) {
    const k = this.knots;
    if (r <= 0) return 0;
    for (let i = 1; i < k.length; i++) {
      if (r <= k[i][0]) {
        const [r0, s0] = k[i - 1], [r1, s1] = k[i];
        return r1 > r0 ? s0 + (s1 - s0) * (r - r0) / (r1 - r0) : s1;
      }
    }
    return k[k.length - 1][1];
  }
  real(s) {
    const k = this.knots;
    for (let i = 1; i < k.length; i++) {
      if (s <= k[i][1] && k[i][1] > k[i - 1][1]) {
        const [r0, s0] = k[i - 1], [r1, s1] = k[i];
        if (s < s0) continue;
        return r0 + (r1 - r0) * (s - s0) / (s1 - s0);
      }
    }
    return this.total;
  }
}
