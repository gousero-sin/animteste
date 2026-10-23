// Coreografia: "gelo vs tempestade".
// A = lutador de gelo (preto, poderes chineses de gelo). B = lutador da lança da tempestade (cinza).
'use strict';

let A, B, FXL, SND, SEGS, FREEZES, ENVK, CAMK;
const POST = [];
const MARKS = {};
const fx = (t, type, o = {}) => { FXL.push(Object.assign({ t, type }, o)); };
const sfx = (t, name, gain = 1, o = {}) => { SND.push(Object.assign({ t, name, gain }, o)); };
const env = (name, t, v, e = 's') => { (ENVK[name] = ENVK[name] || []).push({ t, v, e }); };
const cam = (name, t, v, e = 's') => { (CAMK[name] = CAMK[name] || []).push({ t, v, e }); };
const freeze = (s, dur) => FREEZES.push({ s, dur });
const slow = (s0, s1, speed) => SEGS.push({ s0, s1, speed });
const shake = (t, amp, life = 0.35, freq = 34) => fx(t, 'shake', { amp, life, freq });

// ---------- efeitos compostos ----------
function hitIce(t, p, dir, big = 1) {
  fx(t, 'impact', { p, size: 46 * big, color: '140,225,255', spikes: 10, life: 0.2 });
  fx(t, 'shards', { p, n: Math.round(10 * big), speed: 520 * big, size: 8 * big, dir, spread: 120, w: 10, h: 10, life: 1.1 });
  fx(t, 'ring', { p, r0: 8, r1: 90 * big, w: 5, color: '200,245,255', life: 0.3 });
  fx(t, 'sparks', { p, n: 12, speed: 650 * big, dir, spread: 140, color: '225,250,255', life: 0.35 });
  shake(t, 9 * big);
  sfx(t, 'hit', big); sfx(t, 'ice', 0.8 * big);
}
function hitStorm(t, p, dir, big = 1) {
  fx(t, 'impact', { p, size: 50 * big, color: '255,215,90', spikes: 11, life: 0.2 });
  fx(t, 'sparks', { p, n: Math.round(18 * big), speed: 800 * big, dir, spread: 150, color: '255,232,120', life: 0.4 });
  fx(t, 'ring', { p, r0: 8, r1: 100 * big, w: 5, color: '255,240,170', life: 0.28 });
  for (let i = 0; i < 3; i++) {
    const a = (dir + (i - 1) * 50) * DEG;
    fx(t, 'bolt', { a: p, b: { ref: p, dx: Math.cos(a) * 90 * big, dy: Math.sin(a) * 90 * big }, life: 0.16, w: 0.6, branches: 1 });
  }
  shake(t, 10 * big);
  sfx(t, 'hit', big); sfx(t, 'zap', 0.7 * big);
}
function clash(t, p, big = 1, col = '255,240,190') {
  fx(t, 'impact', { p, size: 34 * big, color: col, spikes: 8, life: 0.16 });
  fx(t, 'sparks', { p, n: Math.round(16 * big), speed: 750 * big, color: '255,236,160', life: 0.35 });
  shake(t, 6 * big, 0.25);
  sfx(t, 'clang', big);
}
function bloodHit(t, p, dir, n = 14) {
  fx(t, 'blood', { p, dir, n, spread: 70, speed: 480, life: 3.5 });
}
function landDust(t, p, big = 1) {
  fx(t, 'dust', { p, n: 7, dir: 180, spread: 40, speed: 260 * big, size: 14 * big, flat: 0.3, rise: 10, life: 0.7, alpha: 0.35 });
  fx(t, 'dust', { p, n: 7, dir: 0, spread: 40, speed: 260 * big, size: 14 * big, flat: 0.3, rise: 10, life: 0.7, alpha: 0.35 });
  sfx(t, 'land', 0.7 * big);
}
function breathe(F, t0, t1, hip, amp = 2.2, period = 0.8) {
  let k = 0;
  for (let t = t0 + period / 2; t < t1 - 0.12; t += period / 2) {
    F.k(t, { hip: [hip[0], hip[1] + (k % 2 ? -amp : amp * 0.4)] });
    k++;
  }
}

function buildScene() {
  A = new Fighter('A', {
    color: '#0d0d10', weapon: 'ice', ghostColor: '#8fdcff', tintColor: '#8fd6f2', flashColor: '#ffffff',
    ribbon: {
      color: '#38bdf0', lag: 0.022, len: 62, dx: 4.2, dy: 1.6, wa: 0.55, wf: 11,
      anchor: S => [S.head[0] - S.face * 15, S.head[1] - 6],
      tails: [{ w0: 7, w1: 2.5 }, { w0: 6, w1: 2, len: 48, dy: 2.6, wf: 13 }],
    },
  });
  B = new Fighter('B', {
    color: '#4f4f4f', weapon: 'spear', spearLen: 210, ghostColor: '#f5d45a', tintColor: '#3f7397', flashColor: '#ffffff',
    ribbon: {
      color: '#5b3f8f', lag: 0.02, len: 70, dx: 4.5, dy: 1.2, wa: 0.7, wf: 9,
      anchor: S => [lerp(S.neck[0], S.head[0], 0.2) - S.face * 4, lerp(S.neck[1], S.head[1], 0.2) + 4],
      tails: [{ w0: 9, w1: 3 }, { w0: 7, w1: 2, len: 54, dy: 2.2, wf: 12 }],
    },
  });
  FXL = []; SND = []; SEGS = []; FREEZES = []; ENVK = {}; CAMK = {}; POST.length = 0;

  intro(0);
  beatDash(2.75);
  beatKicks(4.0);
  beatSweep(6.0);
  beatShots(7.1);
  beatSpikes(9.15);
  beatDuel(10.55);
  beatBlitz(13.75);
  beatAerial(16.05);
  beatFlurry(18.1);
  beatFinale(19.95);

  A.build(POSES); B.build(POSES);
  SCENE.fighters = { A, B };
  for (const f of POST) f();
  B.build(POSES);
  SCENE.fx = FXL;
  SCENE.sounds = SND;
  const mk = o => { const r = {}; for (const n in o) r[n] = new Track(o[n].sort((a, b) => a.t - b.t)); return r; };
  SCENE.env = mk(ENVK);
  SCENE.cam = mk(CAMK);
  SCENE.warp = new TimeWarp(SEGS, FREEZES, 40);
  prepareFx();
}

// ================= BEAT 0: apresentação =================
function intro(T) {
  // A: postura, floreio de garça e pisada congelante
  A.k(T + 0, { p: 'kfReady', x: -250, face: 1, glow: 0.35 });
  breathe(A, T + 0, T + 0.9, [-6, -76]);
  A.k(T + 0.95, { p: 'kfWind', x: -252, glow: 0.45 });
  A.k(T + 1.25, { p: 'palmUp', x: -248, glow: 0.7 });
  A.k(T + 1.52, { p: 'kfCrane', x: -250, glow: 0.85 });
  A.k(T + 1.78, { p: 'kfHorse', x: -250, glow: 1 }, 'snap');
  A.k(T + 2.15, { p: 'kfLow', x: -250, glow: 0.7 });
  breathe(A, T + 2.15, T + 2.8, [-10, -62]);
  fx(T + 1.78, 'frost', { p: [-250, 0], w: 175, life: 9 });
  fx(T + 1.78, 'ring', { p: [-250, 0], r0: 20, r1: 260, squash: 0.16, w: 6, color: '170,235,255', life: 0.5 });
  fx(T + 1.78, 'shards', { p: [-250, -4], n: 10, speed: 380, size: 6, dir: -90, spread: 120, w: 90, h: 4, life: 0.9 });
  fx(T + 1.78, 'dust', { p: [-250, 0], n: 12, dir: 0, spread: 360, speed: 300, size: 16, flat: 0.2, color: '235,248,255', alpha: 0.55, life: 0.8 });
  shake(T + 1.78, 6);
  sfx(T + 1.78, 'stomp', 1); sfx(T + 1.8, 'ice', 1);
  env('snow', 0, 0.25); env('snow', T + 1.8, 0.25); env('snow', T + 2.3, 0.55);
  env('frost', 0, 0); env('frost', T + 1.78, 0); env('frost', T + 2.2, 0.22);

  // B: giro da lança e batida no chão (raio)
  B.k(T + 0, { p: 'spTwirl', x: 252, face: -1, sa: 990, zap: 0.15 });
  B.k(T + 0.15, { sa: 950, hF: [30, -20] }, 'i2');
  B.k(T + 0.4, { hF: [34, -30] });
  B.k(T + 0.65, { hF: [26, -36] });
  B.k(T + 0.9, { hF: [34, -18] });
  B.k(T + 0.95, { sa: 50 }, 'l');
  B.k(T + 1.1, { sa: -90, hF: [26, -6], ln: 0, hip: [0, -86], zap: 0.3 }, 'o');
  B.k(T + 1.22, { hF: [26, -22], hip: [0, -90], ln: -4, hd: -6 });
  B.k(T + 1.32, { hF: [26, 36], hip: [2, -78], ln: 8, hd: 4, fF: [30, 0], fB: [-30, 0], zap: 1 }, 'snap');
  B.k(T + 1.62, { p: 'spGuard', x: 246, zap: 0.75 });
  breathe(B, T + 1.62, T + 2.7, [-4, -76]);
  fx(T + 0.2, 'wind', { p: 'B.hF', follow: true, r: 105, squash: 0.95, n: 7, spin: 13, life: 0.85, alpha: 0.7, w: 3 });
  for (const tt of [0.3, 0.52, 0.72, 0.9]) sfx(T + tt, 'whoosh', 0.6);
  const slamP = [252 - 26 - 4, 0];
  fx(T + 1.32, 'sparks', { p: [slamP[0], -4], n: 22, speed: 700, dir: -90, spread: 160, life: 0.45 });
  fx(T + 1.32, 'crack', { p: [slamP[0], 0], size: 60, n: 6, life: 40 });
  fx(T + 1.32, 'ring', { p: [slamP[0], 0], r0: 10, r1: 200, squash: 0.16, w: 5, color: '255,240,160', life: 0.4 });
  fx(T + 1.32, 'dust', { p: [slamP[0], 0], n: 10, dir: 0, spread: 360, speed: 320, size: 15, flat: 0.25, life: 0.8 });
  fx(T + 1.34, 'skybolt', { p: 'B.tip', w: 1.6, life: 0.32 });
  fx(T + 1.34, 'skyflash', { life: 0.5, amt: 1 });
  fx(T + 1.34, 'flash', { life: 0.18, amt: 0.35 });
  shake(T + 1.32, 7); shake(T + 1.34, 9, 0.5);
  sfx(T + 1.32, 'stomp', 0.8); sfx(T + 1.34, 'thunder', 1);
  env('storm', 0, 0); env('storm', T + 1.32, 0); env('storm', T + 1.9, 0.16);
}

// ================= BEAT 1: investida, aparo e palma dupla =================
function beatDash(T) {
  MARKS.fight0 = T;
  // B investe como um raio
  B.k(T + 0, { p: 'spDraw', x: 250, zap: 0.9 });
  B.k(T + 0.1, { x: 244, hip: [-14, -70] });
  B.k(T + 0.24, { p: 'spThrust', x: 40, gh: 1, zap: 1 }, 'o', 'io');
  B.k(T + 0.33, { x: 14, sa: -40, ln: 36, hd: -14, hF: [60, -10], gh: 0.4 }, 'o');
  B.k(T + 0.45, { x: 8, ln: 30, sa: -34, gh: 0 });
  fx(T + 0.12, 'speedlines', { dir: -1, life: 0.3, alpha: 0.55 });
  fx(T + 0.11, 'dust', { p: [244, 0], n: 8, dir: 0, spread: 50, speed: 260, size: 14, flat: 0.3, life: 0.6 });
  fx(T + 0.14, 'bolt', { a: [236, -96], b: [60, -104], life: 0.2, w: 0.8, branches: 2 });
  sfx(T + 0.1, 'dash', 1); sfx(T + 0.12, 'zap', 0.6);

  // A apara a lança para cima e entra com palma dupla
  A.k(T + 0, { p: 'kfLow', x: -250 });
  A.k(T + 0.16, { hF: [44, -8], ln: 4, hip: [-8, -66] });
  A.k(T + 0.23, { hF: [54, -24], hB: [24, 6], ln: -6, x: -256 }, 'o');
  A.k(T + 0.31, { hF: [40, -60], ln: -10, hd: -6 }, 'o');
  A.k(T + 0.37, { p: 'kfWind', x: -214, glow: 1 });
  A.k(T + 0.45, { p: 'palm2', x: -112, glow: 1 }, 'snap', 'o');
  A.k(T + 0.7, { p: 'palm2', x: -110, glow: 0.8 });
  A.k(T + 1.0, { p: 'kfReady', x: -128, glow: 0.5 });
  breathe(A, T + 1.0, T + 1.3, [-6, -76]);
  clash(T + 0.23, 'A.hF', 0.8, '170,235,255');
  fx(T + 0.23, 'shards', { p: 'A.hF', n: 6, speed: 360, size: 6, dir: -100, spread: 90, life: 0.8 });

  // impacto da palma dupla
  const tHit = T + 0.45;
  hitIce(tHit, 'B.chest', 160, 1.2);
  fx(tHit, 'hitflash', { id: 'B', life: 0.06 });
  fx(tHit, 'frost', { p: [8, -110], w: 1, life: 0.01 });
  freeze(tHit, 0.07);
  B.k(tHit, { p: 'spThrust', x: 8, sa: -34, ln: 30 });
  B.k(tHit + 0.07, { p: 'hit', spear: 1, sa: -70, sg: 100, sb: -1, hF: [30, -40], hB: [-20, -30], x: 64, zap: 0.4 }, 'o', 'o');
  B.k(tHit + 0.38, { p: 'spLow', x: 196, sa: -18, zap: 0.5 }, 's', 'o');
  B.k(tHit + 0.7, { p: 'spGuard', x: 204, zap: 0.6 });
  fx(tHit + 0.07, 'skid', { p: [70, 0], dir: 1, len: 120, life: 30 });
  fx(tHit + 0.1, 'dust', { p: 'B.fF', n: 6, dir: 0, spread: 40, speed: 140, size: 12, flat: 0.3, life: 0.6 });
  fx(tHit + 0.22, 'dust', { p: 'B.fB', n: 6, dir: 0, spread: 40, speed: 120, size: 12, flat: 0.3, life: 0.6 });
  sfx(tHit + 0.08, 'skid', 0.7);
  breathe(B, tHit + 0.7, T + 1.3, [-4, -76]);
}

// ================= BEAT 1b: voadora e contra-ataque com o cabo =================
function beatKicks(T) {
  // A corre e dá uma voadora lateral
  A.k(T + 0.05, { p: 'kfReady', x: -128 });
  A.k(T + 0.16, { p: 'run1', x: -96 });
  A.k(T + 0.28, { p: 'run2', x: -40 });
  A.k(T + 0.38, { p: 'crouch', x: 0, glow: 0.6 });
  A.k(T + 0.5, { p: 'kickChamber', x: 44, y: -62 }, 's', 's');
  A.k(T + 0.58, { p: 'sideKick', x: 66, y: -66 }, 'snap');
  // rebote: mortal para trás
  A.k(T + 0.66, { p: 'tuck', x: 40, y: -110, rot: -120 }, 's', 'o');
  A.k(T + 0.8, { p: 'tuck', x: -10, y: -96, rot: -280 });
  A.k(T + 0.92, { p: 'land', x: -40, y: 0, rot: -360 }, 's', 'i2');
  A.k(T + 1.1, { p: 'kfReady', x: -46, rot: -360, glow: 0.5 });
  landDust(T + 0.92, [-40, 0]);

  // B bloqueia com a haste na diagonal
  B.k(T + 0.3, { p: 'spGuard', x: 204 });
  B.k(T + 0.5, { p: 'spGuard', x: 200, sa: -62, hF: [30, -16], sg: 120, sb: 40, ln: 2 });
  const tK = T + 0.58;
  B.k(tK, { x: 196, sa: -66, hF: [24, -20], ln: -4 }, 'o');
  B.k(tK + 0.1, { x: 214, ln: -10, hd: -6, sa: -74 }, 'o', 'o');
  clash(tK, 'A.fF', 1.1);
  freeze(tK, 0.05);
  fx(tK, 'ring', { p: 'A.fF', r0: 6, r1: 80, w: 4, life: 0.25 });
  fx(tK + 0.08, 'skid', { p: [196, 0], dir: 1, len: 40, life: 30 });

  // B contra-ataca: giro do cabo de baixo para cima
  B.k(T + 0.85, { p: 'spGuard', x: 200, sa: -20, ln: 8 });
  B.k(T + 0.98, { p: 'spLow', x: 150, sa: 40, hF: [10, 30], sg: 120, sb: 50, ln: 18 }, 's', 'io');
  B.k(T + 1.1, { x: 96, sa: -60, hF: [40, 10], ln: 22, hd: -6, fF: [50, 0], fB: [-46, 0], hip: [8, -70] }, 'o', 'o');
  B.k(T + 1.18, { x: 90, sa: -150, hF: [50, -24], ln: 10 }, 'o');
  B.k(T + 1.4, { p: 'spGuard', x: 110, sa: -170 + 160, ln: 6 });
  breathe(B, T + 1.4, T + 1.9, [-4, -76]);
  sfx(T + 1.0, 'whoosh', 0.9);
  // A defende com antebraços cruzados e desliza
  A.k(T + 1.05, { p: 'kfReady', x: -46, rot: -360, hF: [36, -26], hB: [30, -12], ln: -4 });
  const tB = T + 1.12;
  A.k(tB, { x: -50, hF: [30, -34], hB: [26, -20], ln: -10, hd: -8 });
  A.k(tB + 0.22, { p: 'kfLow', x: -150, rot: -360, glow: 0.6 }, 'o', 'o');
  A.k(tB + 0.6, { p: 'kfReady', x: -170, rot: -360 });
  clash(tB, 'A.hF', 0.9, '255,250,220');
  fx(tB + 0.02, 'skid', { p: [-56, 0], dir: -1, len: 100, life: 30 });
  fx(tB + 0.05, 'dust', { p: 'A.fB', n: 6, dir: 180, spread: 40, speed: 140, size: 12, flat: 0.3, life: 0.6 });
  breathe(A, tB + 0.6, T + 1.9, [-6, -76]);
}

// ================= BEAT 2: rasteira giratória, mortal, golpe aéreo x parede de gelo =================
function beatSweep(T) {
  // B: rasteira com giro completo (face -1 -> 1 -> -1)
  B.k(T + 0.1, { p: 'spLow', x: 120, sa: -6 });
  B.k(T + 0.24, { p: 'spSweep', x: 34, face: -1, sa: 12 }, 'o', 'io');
  B.k(T + 0.38, { x: 20, face: 1, ln: 34, sa: 14 }, 'sine', 'l');
  B.k(T + 0.52, { x: 14, face: -1, ln: 30, sa: 12 }, 'sine', 'l');
  fx(T + 0.24, 'dust', { p: [-40, 0], n: 10, dir: 180, spread: 30, speed: 300, size: 13, flat: 0.25, life: 0.6 });
  fx(T + 0.3, 'dust', { p: [60, 0], n: 8, dir: 0, spread: 30, speed: 260, size: 13, flat: 0.25, life: 0.6 });
  sfx(T + 0.24, 'whoosh', 1); sfx(T + 0.42, 'whoosh', 0.8);
  // B salta e golpeia de cima
  B.k(T + 0.62, { p: 'spOver', x: 10, y: -50 }, 's', 'o');
  B.k(T + 0.76, { p: 'spOver', x: 4, y: -125, ln: -20, hd: -8 });
  const tW = T + 0.88;
  B.k(tW, { p: 'spSlam', x: 0, y: 0, sa: -8, hF: [46, -20], ln: 24 }, 'o', 'i2');
  B.k(tW + 0.1, { x: 10, sa: -30, hF: [40, -30], ln: 14, hd: -6 }, 'o');
  B.k(tW + 0.32, { p: 'spGuard', x: -30 });
  sfx(T + 0.8, 'whoosh', 1.1);

  // A: mortal para trás por cima da rasteira
  A.k(T + 0.1, { p: 'kfReady', x: -170, rot: -360 });
  A.k(T + 0.2, { p: 'crouch', x: -178 });
  A.k(T + 0.3, { p: 'tuck', x: -200, y: -80, rot: -430 }, 's', 'o');
  A.k(T + 0.42, { p: 'tuck', x: -246, y: -150, rot: -560 });
  A.k(T + 0.54, { p: 'tuck', x: -286, y: -100, rot: -680 });
  A.k(T + 0.64, { p: 'land', x: -305, y: 0, rot: -720 }, 's', 'i2');
  landDust(T + 0.64, [-305, 0]);
  // A ergue a parede de gelo
  A.k(T + 0.7, { p: 'kfWind', x: -306, glow: 1 });
  A.k(T + 0.76, { p: 'palmLow', x: -296, glow: 1 }, 'snap');
  A.k(T + 1.0, { p: 'palmLow', x: -298 });
  fx(T + 0.76, 'iceWall', { p: [-205, 0], h: 175, life: 1.5, crack: tW - T - 0.76, shatter: tW + 0.12 - T - 0.76, dir: -150, cp: [-196, -148] });
  fx(T + 0.76, 'frost', { p: [-230, 0], w: 120, life: 10 });
  fx(T + 0.76, 'dust', { p: [-205, 0], n: 10, dir: -90, spread: 180, speed: 220, size: 16, color: '230,246,255', alpha: 0.5, life: 0.7 });
  sfx(T + 0.76, 'iceRise', 1);
  // impacto na parede
  clash(tW, [-196, -148], 1.3);
  fx(tW, 'sparks', { p: [-196, -148], n: 16, speed: 600, color: '200,245,255', life: 0.4 });
  shake(tW, 12, 0.4);
  freeze(tW, 0.06);
  sfx(tW + 0.12, 'shatter', 1.2);
  shake(tW + 0.12, 8, 0.3);
}

// ================= BEAT 3: tiros de gelo, defesa giratória, raio =================
function beatShots(T) {
  // A recua e dispara três lanças de gelo
  A.k(T + 0.08, { p: 'kfWind', x: -318, glow: 1 });
  A.k(T + 0.16, { p: 'palm', x: -326 }, 'snap');
  A.k(T + 0.26, { p: 'kfWind', x: -336 });
  A.k(T + 0.33, { p: 'palmB', x: -342 }, 'snap');
  A.k(T + 0.43, { p: 'kfWind', x: -346 });
  A.k(T + 0.5, { p: 'palm', x: -350, hB: [-30, 20] }, 'snap');
  A.k(T + 0.7, { p: 'kfReady', x: -352, glow: 0.8 });
  const hitX = -168, hitY = -128;
  const shots = [[T + 0.16, 'A.hF'], [T + 0.33, 'A.hB'], [T + 0.5, 'A.hF']];
  shots.forEach(([t, src], i) => {
    fx(t, 'iceShot', { a: src, b: [hitX + i * 4, hitY + (i - 1) * 14], life: 0.15, size: 12 + i * 2 });
    fx(t, 'ring', { p: src, r0: 4, r1: 34, w: 3, color: '200,245,255', life: 0.18 });
    sfx(t, 'iceShot', 0.8);
    const ta = t + 0.15;
    fx(ta, 'impact', { p: [hitX + i * 4, hitY + (i - 1) * 14], size: 34, color: '150,230,255', spikes: 7, life: 0.16 });
    fx(ta, 'shards', { p: [hitX + i * 4, hitY + (i - 1) * 14], n: 8, speed: 420, size: 7, dir: -150, spread: 120, life: 0.9 });
    sfx(ta, 'shatter', 0.6); sfx(ta, 'clang', 0.5);
  });
  // B gira a lança como um escudo
  B.k(T + 0.12, { p: 'spTwirl', x: -56, sa: -8, hF: [40, -18] });
  B.k(T + 0.18, { sa: -60 }, 'i2');
  B.k(T + 0.66, { sa: -60 - 1260, hF: [44, -22] }, 'l');
  B.k(T + 0.74, { sa: -1444, hF: [36, -10] }, 'o');
  fx(T + 0.16, 'wind', { p: 'B.hF', follow: true, r: 100, squash: 0.98, n: 8, spin: 18, life: 0.6, alpha: 0.75, w: 3 });
  for (const tt of [0.2, 0.32, 0.44, 0.56]) sfx(T + tt, 'whoosh', 0.55);
  // B aponta e dispara o raio
  B.k(T + 0.84, { p: 'spPoint', x: -60, zap: 1 });
  B.k(T + 0.94, { p: 'spPoint', x: -64, ln: 6, zap: 1, hF: [58, -12] });
  const tF = T + 0.98;
  B.k(tF + 0.06, { p: 'spPoint', x: -50, ln: 2, hd: -6, hF: [50, -20], sa: -14, zap: 0.6 }, 'o');
  B.k(tF + 0.4, { p: 'spGuard', x: -46, zap: 0.6 });
  fx(T + 0.8, 'charge', { p: 'B.tip', follow: true, r: 90, n: 18, color: '255,230,120', life: 0.2 });
  sfx(T + 0.8, 'charge', 0.6);
  fx(tF, 'bolt', { a: 'B.tip', b: 'A.chest', grow: 0.05, life: 0.32, w: 1.5, glowEnd: true, branches: 4 });
  fx(tF, 'flash', { life: 0.12, amt: 0.3, color: '255,250,220' });
  sfx(tF, 'thunder', 0.8); sfx(tF, 'zap', 1);
  // A é atingido, voa e rola para trás
  const tH = tF + 0.05;
  hitStorm(tH, 'A.chest', 180, 1.1);
  fx(tH, 'hitflash', { id: 'A', life: 0.08 });
  freeze(tH, 0.06);
  A.k(tH, { p: 'hit', x: -354, bz: 1, glow: 0.3 });
  A.k(tH + 0.07, { p: 'flyBack', x: -392, y: -46, rot: -735, bz: 1 }, 'o', 'o');
  A.k(tH + 0.3, { p: 'flyBack', x: -462, y: -34, rot: -770, bz: 0.7 });
  A.k(tH + 0.42, { p: 'lie', x: -494, y: 0, rot: -720, bz: 0.4 }, 's', 'i2');
  A.k(tH + 0.56, { p: 'roll', x: -520, y: 44, rot: -900, bz: 0.1 });
  A.k(tH + 0.7, { p: 'roll', x: -546, y: 44, rot: -1020, bz: 0 });
  A.k(tH + 0.8, { p: 'crouch', x: -560, y: 0, rot: -1080 }, 's', 's');
  fx(tH + 0.42, 'dust', { p: [-494, 0], n: 10, dir: 180, spread: 60, speed: 240, size: 15, flat: 0.3, life: 0.7 });
  fx(tH + 0.42, 'skid', { p: [-480, 0], dir: -1, len: 80, life: 30 });
  fx(tH + 0.1, 'scorch', { p: [-360, 0], w: 60, life: 25 });
  sfx(tH + 0.42, 'land', 0.9);
}

// ================= BEAT 4: espinhos de gelo, salto com vara, mergulho com a lança =================
function beatSpikes(T) {
  // A golpeia o chão: linha de espinhos
  A.k(T + 0.04, { p: 'crouch', x: -560, rot: -1080, glow: 1 });
  A.k(T + 0.15, { p: 'slamGround', x: -560 }, 'snap');
  A.k(T + 0.5, { p: 'slamGround', x: -560, hd: -14 });
  const tS = T + 0.15;
  fx(tS, 'iceSpikes', { x0: -505, dir: 1, n: 11, gap: 44, dt: 0.04, h: 118, ramp: true, life: 1.5 });
  fx(tS, 'frost', { p: [-330, 0], w: 260, life: 9, grow: 0.5 });
  fx(tS, 'ring', { p: [-560, 0], r0: 10, r1: 140, squash: 0.16, w: 5, color: '190,240,255', life: 0.35 });
  shake(tS, 7, 0.6);
  for (let i = 0; i < 11; i += 2) sfx(tS + i * 0.04, 'iceRise', 0.5);

  // B corre para cima e usa a lança como vara
  B.k(T + 0.08, { p: 'spGuard', x: -50 });
  B.k(T + 0.18, { p: 'spRun1', x: -70 });
  B.k(T + 0.3, { p: 'spRun2', x: -122 });
  const tP = T + 0.4;
  B.k(tP, { p: 'spRun1', x: -168, sa: 40, sg: 150, sb: 90, ln: 10 });
  // haste fincada (pivô na ponta de trás, as mãos seguem a lança)
  B.free.push({
    t0: tP, t1: T + 0.64, pivot: 'butt', grab: true, keys: [
      { t: tP, auto: true, e: 's' },
      { t: T + 0.46, x: -236, y: 2, a: -62 },
      { t: T + 0.54, x: -236, y: 2, a: -92 },
      { t: T + 0.6, x: -236, y: 2, a: -112 },
      { t: T + 0.64, auto: true },
    ],
  });
  B.k(T + 0.46, { x: -190, y: -60, rot: 40, ln: 20, hip: [0, -88], fF: [-10, -20], fB: [-30, -10], sg: 160, sb: 110 });
  B.k(T + 0.54, { x: -228, y: -170, rot: 150, fF: [10, -40], fB: [-20, -20] });
  B.k(T + 0.6, { x: -280, y: -190, rot: 230, fF: [20, -40], fB: [-10, -30] });
  B.k(T + 0.64, { p: 'spDive', x: -320, y: -175, rot: 300, sa: 60, sg: 90, sb: 30 });
  B.k(T + 0.74, { p: 'spDive', x: -420, y: -110, rot: 360 + 24 });
  const tD = T + 0.84;
  B.k(tD, { p: 'spDive', x: -500, y: -14, rot: 360 + 8, sa: 66 }, 's', 'i2');
  // finca no chão e aterrissa agachado segurando a lança
  B.free.push({
    t0: tD, t1: T + 1.4, pivot: 'tip', grab: true, keys: [
      { t: tD, auto: true },
      { t: T + 1.4, x: -566, y: 12, a: 116 },
    ],
  });
  B.k(tD + 0.1, { p: 'kneel', x: -500, y: 0, rot: 360, sg: 120, sb: 60, ln: 30 }, 's', 'o');
  B.k(T + 1.3, { p: 'kneel', x: -500, rot: 360, sg: 120, sb: 60, ln: 24, hd: -8 });
  sfx(tP, 'whoosh', 0.8); sfx(T + 0.6, 'whoosh', 1);
  // impacto do mergulho
  const gp = [-566, 0];
  fx(tD, 'impact', { p: [-566, -10], size: 70, color: '255,220,110', spikes: 12, life: 0.24 });
  fx(tD, 'crack', { p: gp, size: 110, n: 9, rocks: 8, life: 40 });
  fx(tD, 'debris', { p: [-566, -4], n: 14, speed: 620, size: 8, dir: -90, spread: 140, life: 1.6 });
  fx(tD, 'dust', { p: gp, n: 16, dir: 0, spread: 360, speed: 380, size: 20, flat: 0.25, life: 1.0, alpha: 0.5 });
  fx(tD, 'ring', { p: gp, r0: 10, r1: 240, squash: 0.15, w: 6, color: '255,240,170', life: 0.4 });
  for (let i = 0; i < 4; i++) fx(tD + i * 0.02, 'bolt', { a: [-566, -2], b: [-566 + (i % 2 ? 1 : -1) * (120 + i * 30), -2 + (i - 1.5) * 4], life: 0.22, w: 0.7, jag: 0.12, branches: 2 });
  fx(tD, 'scorch', { p: gp, w: 80, life: 30 });
  shake(tD, 16, 0.5);
  freeze(tD, 0.08);
  sfx(tD, 'boom', 1); sfx(tD, 'zap', 0.8);

  // A rola para a frente, por baixo do salto, e vira para a esquerda
  A.k(T + 0.62, { p: 'crouch', x: -556, rot: -1080, face: 1 });
  A.k(T + 0.72, { p: 'roll', x: -510, y: 40, rot: -1080 + 140 }, 's', 's');
  A.k(T + 0.84, { p: 'roll', x: -450, y: 44, rot: -1080 + 280 });
  A.k(T + 0.96, { p: 'crouch', x: -372, y: 0, rot: -720, face: 1 }, 's', 's');
  A.k(T + 1.06, { p: 'kfLow', x: -356, face: -1, glow: 0.8 }, 'io');
  fx(T + 0.72, 'dust', { p: [-500, 0], n: 6, dir: 0, spread: 40, speed: 160, size: 12, flat: 0.3, life: 0.6 });
  sfx(T + 0.75, 'land', 0.5);
}

// ================= BEAT 5: espada de gelo x lança =================
function beatDuel(T) {
  // A forma a espada de gelo
  A.k(T + 0.0, { p: 'kfLow', x: -356, face: -1, glow: 1, sw: 0 });
  A.k(T + 0.1, { p: 'swordReady', x: -358, sw: 0.08 });
  A.k(T + 0.24, { p: 'swBlock', x: -362, sw: 1 });
  fx(T + 0.06, 'charge', { p: 'A.hF', follow: true, r: 70, n: 16, color: '200,245,255', life: 0.22 });
  fx(T + 0.22, 'sparks', { p: 'A.swTip', n: 10, speed: 300, color: '220,250,255', life: 0.3 });
  sfx(T + 0.08, 'iceForm', 1);

  // B arranca a lança girando e corta (clash 1)
  B.k(T + 0.0, { p: 'kneel', x: -500, face: -1, sa: 64, sg: 120, sb: 60, ln: 24, rot: 360 });
  B.k(T + 0.1, { hip: [-6, -62], ln: 14, hF: [30, 24], sa: 30, x: -506 });
  B.k(T + 0.2, { p: 'spLow', x: -536, face: 0, sa: -6, rot: 360 }, 'sine');
  const c1 = T + 0.3;
  B.k(c1, { p: 'spGuard', x: -560, face: 1, sa: -6, hF: [52, -16], sg: 90, sb: 20, ln: 18 }, 'sine');
  B.k(T + 0.4, { p: 'spGuard', x: -566, ln: 10 });
  A.k(c1, { p: 'swBlock', x: -358, ln: -4 });
  A.k(T + 0.38, { x: -352, ln: 0 });
  sfx(T + 0.14, 'whoosh', 0.9);
  clash(c1, 'X', 1);

  // A estoca; B desvia com a haste (clash 2)
  A.k(T + 0.46, { p: 'swordReady', x: -358, hF: [18, 8], swa: -10 });
  const c2 = T + 0.56;
  A.k(c2, { p: 'swThrust', x: -416 }, 'snap', 'o');
  A.k(T + 0.64, { p: 'swThrust', x: -420, swa: -34, hF: [58, -30] }, 'o');
  B.k(T + 0.5, { p: 'spBlockMid', x: -566, sa: -60 });
  B.k(c2, { p: 'spBlockMid', x: -570, sa: -104, hF: [40, -30] }, 'o');
  B.k(T + 0.68, { p: 'spGuard', x: -572 });
  clash(c2, 'X', 0.9);

  // A gira e corta na horizontal; B se abaixa
  A.k(T + 0.72, { p: 'swordReady', x: -412, face: -1 });
  A.k(T + 0.82, { p: 'swSlash', x: -428, face: 1, hF: [40, -14] }, 'sine');
  A.k(T + 0.92, { p: 'swSlash', x: -446, face: -1 }, 'sine');
  B.k(T + 0.8, { p: 'spLow', x: -574, hip: [-6, -48], ln: 34, hd: 10 }, 'o');
  B.k(T + 0.92, { p: 'spLow', x: -576, hip: [-8, -44], ln: 38 });
  sfx(T + 0.84, 'whoosh', 1);

  // B estoca para cima; A se inclina para trás
  B.k(T + 1.0, { p: 'spThrustUp', x: -534, sa: -14 }, 'snap', 'o');
  B.k(T + 1.12, { p: 'spThrustUp', x: -532, sa: -16 });
  A.k(T + 1.0, { p: 'swLean', x: -428 }, 'o');
  A.k(T + 1.12, { p: 'swLean', x: -424 });
  sfx(T + 0.98, 'whoosh', 1);

  // A corta de cima; B bloqueia acima da cabeça (clash 3)
  A.k(T + 1.22, { p: 'swHigh', x: -400 });
  const c3 = T + 1.34;
  A.k(c3, { p: 'swChop', x: -372, swa: 24, hF: [40, -70], ln: 4, hip: [6, -72] }, 'snap');
  A.k(T + 1.44, { x: -370, swa: 30, hF: [42, -62], ln: 8 });
  B.k(T + 1.22, { p: 'spBlockHi', x: -546 });
  B.k(c3, { p: 'spBlockHi', x: -548, hip: [-6, -68], ln: 6 });
  B.k(T + 1.42, { p: 'spBlockHi', x: -544, hip: [-6, -74] });
  clash(c3, 'X', 1.2);
  fx(c3, 'frost', { p: [-500, 0], w: 90, life: 8 });

  // B empurra e chuta A
  const k1 = T + 1.54;
  B.k(T + 1.48, { p: 'spKick', x: -520, fF: [40, -50] }, 'o');
  B.k(k1, { p: 'spKick', x: -476 }, 'snap', 'o');
  B.k(T + 1.7, { p: 'spGuard', x: -482 });
  A.k(k1, { x: -372, swa: 30 });
  A.k(k1 + 0.05, { p: 'hitFwd', x: -350, sw: 1, swa: 40, hF: [30, 40] }, 'o', 'o');
  A.k(T + 1.86, { p: 'kfLow', x: -244, swa: -40, hF: [30, 18] }, 's', 'o');
  hitStorm(k1, 'A.hip', 0, 0.7);
  fx(k1 + 0.05, 'skid', { p: [-350, 0], dir: 1, len: 100, life: 30 });
  fx(k1 + 0.1, 'dust', { p: 'A.fF', n: 6, dir: 0, spread: 40, speed: 140, size: 12, flat: 0.3, life: 0.6 });

  // investida dupla -> travamento das armas (câmera lenta)
  A.k(T + 1.96, { p: 'swordReady', x: -250 });
  B.k(T + 1.96, { p: 'spDraw', x: -500 });
  const lk = T + 2.1;
  A.k(lk, { p: 'swLock', x: -290 }, 'o', 'io');
  B.k(lk, { p: 'spLock', x: -480 }, 'o', 'io');
  A.k(lk + 0.2, { p: 'swLock', x: -294, ln: 22, hd: -8 });
  B.k(lk + 0.2, { p: 'spLock', x: -476, ln: 22, bz: 0.4, zap: 1 });
  A.k(lk + 0.4, { p: 'swLock', x: -288, ln: 16 });
  B.k(lk + 0.4, { p: 'spLock', x: -482, ln: 16, bz: 0.9 });
  A.k(lk + 0.56, { p: 'swLock', x: -292, ln: 20, sw: 1 });
  B.k(lk + 0.56, { p: 'spLock', x: -478, ln: 20, bz: 1, zap: 1 });
  clash(lk, 'X', 1.4);
  slow(lk + 0.02, lk + 0.5, 0.32);
  MARKS.lock0 = lk; MARKS.lock1 = lk + 0.62;
  for (let i = 0; i < 9; i++) fx(lk + 0.04 + i * 0.055, 'sparks', { p: 'X', n: 6, speed: 520, color: i % 2 ? '255,236,140' : '210,248,255', life: 0.3 });
  fx(lk, 'bolt', { a: 'X', b: 'A.swTip', follow: true, life: 0.55, w: 0.6, branches: 1 });
  fx(lk + 0.25, 'bolt', { a: 'X', b: 'A.hF', follow: true, life: 0.3, w: 0.6, branches: 1 });
  cam('lock', 0, 0); cam('lock', lk - 0.04, 0); cam('lock', lk + 0.04, 0.85); cam('lock', lk + 0.5, 0.85); cam('lock', lk + 0.68, 0);
  cam('lx', 0, -385); cam('ly', 0, -150); cam('lz', 0, 1.55);
  sfx(lk + 0.05, 'grind', 1);
  // B sobrecarrega: a espada racha e estilhaça, A é arremessado
  const sh = lk + 0.62;
  fx(sh, 'flash', { life: 0.15, amt: 0.6, color: '255,250,215' });
  fx(sh, 'invert', { life: 2 / 60 });
  fx(sh, 'shards', { p: 'A.swTip', n: 22, speed: 650, size: 10, dir: 0, spread: 300, w: 40, h: 40, life: 1.2 });
  hitStorm(sh, 'X', 0, 1.3);
  freeze(sh, 0.08);
  A.k(sh, { p: 'swLock', x: -292, sw: 1 });
  A.k(sh + 0.03, { sw: 0 }, 'l');
  A.k(sh + 0.08, { p: 'flyBack', x: -246, y: -40, bz: 1, rot: -730 }, 'o', 'o');
  A.k(sh + 0.32, { p: 'flyBack', x: -186, y: -30, bz: 0.6, rot: -750 });
  A.k(sh + 0.46, { p: 'land', x: -156, y: 0, bz: 0.2, rot: -720 }, 's', 'i2');
  A.k(sh + 0.7, { p: 'kfLow', x: -152, bz: 0, glow: 0.6 });
  landDust(sh + 0.46, [-156, 0]);
  fx(sh + 0.46, 'skid', { p: [-156, 0], dir: 1, len: 50, life: 30 });
  B.k(sh, { p: 'spLock', x: -480, bz: 1 });
  B.k(sh + 0.14, { p: 'spGuard', x: -504, bz: 0.4, zap: 0.8 }, 'o', 'o');
  B.k(sh + 0.5, { p: 'spGuard', x: -508, bz: 0, zap: 0.7 });
  sfx(sh, 'shatter', 1.3); sfx(sh, 'thunder', 0.7);
}

// ================= BEAT 6: a tempestade desce; ataque relâmpago =================
function beatBlitz(T) {
  MARKS.storm = T + 0.86;
  // B chama o raio
  B.k(T + 0.05, { p: 'spGuard', x: -508 });
  B.k(T + 0.32, { p: 'spSky', x: -506, zap: 1 });
  B.k(T + 0.8, { p: 'spSky', x: -506, hip: [0, -91], hd: -28 });
  const tL = T + 0.86;
  B.k(tL, { p: 'spSky', x: -506, hip: [0, -78], ln: 6, bz: 1, zap: 1 }, 'snap');
  B.k(T + 1.06, { p: 'spDraw', x: -510, bz: 1, zap: 1 });
  env('storm', T + 0.2, 0.3); env('storm', T + 0.95, 1);
  env('rain', 0, 0); env('rain', T + 0.4, 0); env('rain', T + 1.1, 0.85);
  env('snow', T + 0.3, 0.55); env('snow', T + 1.0, 0.15);
  env('dark', 0, 0); env('dark', T + 0.3, 0); env('dark', T + 0.95, 0.18);
  fx(T + 0.3, 'charge', { p: 'B.tip', follow: true, r: 140, n: 22, color: '255,232,130', life: 0.55 });
  fx(T + 0.45, 'skyflash', { life: 0.3, amt: 0.4 });
  sfx(T + 0.3, 'rumble', 1); sfx(T + 0.3, 'charge', 0.8);
  fx(tL, 'skybolt', { p: 'B.tip', w: 2.4, life: 0.4 });
  fx(tL + 0.08, 'skybolt', { p: 'B.tip', w: 1.4, life: 0.3, seed: 999 });
  fx(tL, 'skyflash', { life: 0.7, amt: 1.3 });
  fx(tL, 'flash', { life: 0.2, amt: 0.6, color: '255,250,225' });
  fx(tL, 'invert', { life: 2 / 60 });
  fx(tL, 'ring', { p: [-506, 0], r0: 20, r1: 300, squash: 0.15, w: 7, color: '255,240,160', life: 0.5 });
  fx(tL, 'crack', { p: [-506, 0], size: 90, n: 8, rocks: 5, life: 40 });
  fx(tL, 'scorch', { p: [-506, 0], w: 90, life: 30 });
  fx(tL, 'glyph', { ch: '雷', p: [-570, -230], size: 230, color: '255,214,80', alpha: 0.32, life: 1.2 });
  shake(tL, 16, 0.6);
  freeze(tL, 0.07);
  sfx(tL, 'thunder', 1.4); sfx(tL, 'zap', 1);

  // A se prepara (guarda cruzada)
  A.k(T + 0.05, { p: 'kfLow', x: -152, face: -1, rot: -720 });
  A.k(T + 0.45, { p: 'kfReady', x: -150, glow: 0.8 });
  A.k(T + 0.95, { p: 'guardX', x: -150, glow: 1 });
  breathe(A, T + 0.95, T + 1.12, [-4, -70], 1.5, 0.3);

  // investida 1: atravessa A da esquerda para a direita
  const d1 = T + 1.17;
  B.k(T + 1.12, { p: 'spDraw', x: -516, hip: [-14, -66], gh: 1 });
  B.k(T + 1.22, { p: 'spThrust', x: -20, gh: 1 }, 'o', 'io');
  B.k(T + 1.3, { p: 'spLow', x: 6, gh: 0.6 }, 's', 'o');
  fx(T + 1.13, 'bolt', { a: [-500, -100], b: [-30, -110], life: 0.22, w: 0.9, branches: 3 });
  fx(T + 1.13, 'speedlines', { dir: 1, life: 0.3, alpha: 0.55 });
  sfx(T + 1.12, 'dash', 1);
  A.k(d1, { p: 'guardX', x: -150, ln: -6 });
  A.k(T + 1.25, { p: 'guardX', x: -126, ln: -14, hd: -10 }, 'o', 'o');
  clash(d1, 'A.hF', 1.1, '255,236,140');
  fx(d1, 'sparks', { p: 'A.hF', n: 12, speed: 600, color: '200,245,255', dir: 0, spread: 120, life: 0.35 });
  fx(d1 + 0.02, 'skid', { p: [-140, 0], dir: 1, len: 40, life: 30 });
  // B vira, salta e mergulha sobre A
  B.k(T + 1.38, { p: 'spLow', x: 2, face: -1, hip: [0, -54], gh: 0.3 }, 'sine');
  B.k(T + 1.52, { p: 'spOver', x: -30, y: -200, gh: 1 }, 's', 'o');
  B.k(T + 1.6, { p: 'spDive', x: -86, y: -150, rot: 384 });
  const d2 = T + 1.67;
  B.k(T + 1.72, { p: 'spDive', x: -210, y: -16, rot: 368 }, 's', 'i2');
  B.k(T + 1.8, { p: 'kneel', x: -236, y: 0, rot: 360, sa: 30, sg: 100, sb: 40, gh: 0 }, 's', 'o');
  sfx(T + 1.5, 'whoosh', 1);
  A.k(T + 1.34, { p: 'guardX', x: -128, face: 1 }, 'sine');
  A.k(T + 1.58, { p: 'guardX', x: -130, hF: [30, -52], hB: [26, -40], ln: -10, hd: -14 });
  A.k(d2, { p: 'hit', x: -134, face: 1 });
  A.k(T + 1.82, { p: 'hitFwd', x: -146, face: 1 });
  hitStorm(d2, 'A.chest', 200, 1.3);
  bloodHit(d2, 'A.chest', 200, 14);
  fx(d2, 'hitflash', { id: 'A', life: 0.07 });
  freeze(d2, 0.08);
  fx(T + 1.72, 'dust', { p: [-230, 0], n: 10, dir: 0, spread: 360, speed: 300, size: 15, flat: 0.25, life: 0.7 });
  fx(T + 1.72, 'crack', { p: [-250, 0], size: 60, n: 6, life: 40 });
  // golpe lançador: A vai para o alto
  B.k(T + 1.9, { p: 'spLow', x: -236, face: 1, rot: 360 }, 'sine');
  const d3 = T + 1.98;
  B.k(d3, { p: 'spThrustUp', x: -206, sa: -56 }, 'snap', 'o');
  B.k(T + 2.24, { p: 'spThrustUp', x: -208, sa: -74, hF: [40, -62] });
  B.k(T + 2.5, { p: 'spGuard', x: -218, sa: -46, bz: 0.5 });
  A.k(d3, { p: 'hitFwd', x: -158, face: 1 });
  A.k(T + 2.1, { p: 'flyBack', x: -150, y: -160, rot: -780, bz: 1 }, 'o', 'o');
  A.k(T + 2.3, { p: 'tuck', x: -140, y: -292, rot: -920, bz: 0.4 }, 's', 'o');
  hitStorm(d3, 'A.hip', -80, 1.2);
  bloodHit(d3, 'A.hip', -70, 10);
  freeze(d3, 0.06);
  sfx(d3 + 0.05, 'whoosh', 1.2);
}

// ================= BEAT 7: contra-ataque aéreo de gelo; B congelado =================
function beatAerial(T) {
  // A se recupera no ar e desce com uma voadora de gelo
  A.k(T + 0.14, { p: 'tuck', x: -140, y: -312, rot: -1020, bz: 0 });
  A.k(T + 0.32, { p: 'kfCrane', x: -150, y: -306, rot: -1080, face: -1, glow: 1 });
  A.k(T + 0.44, { p: 'diveKick', x: -164, y: -268, rot: -1050 });
  const tI = T + 0.62;
  A.k(tI, { p: 'diveKick', x: -196, y: -192, rot: -1040 }, 's', 'i2');
  fx(T + 0.22, 'charge', { p: 'A.fF', follow: true, r: 90, n: 18, color: '200,245,255', life: 0.4 });
  sfx(T + 0.3, 'iceForm', 0.8); sfx(T + 0.46, 'whoosh', 1.1);
  // B ergue a lança para bloquear
  B.k(T + 0.2, { p: 'spGuard', x: -216, hd: -16 });
  B.k(T + 0.48, { p: 'spBlockHi', x: -222, hd: -20, hip: [-6, -78] });
  B.k(tI, { p: 'spBlockHi', x: -222, hip: [-6, -64], ln: 4 });
  B.k(T + 0.74, { p: 'spBlockHi', x: -222, hip: [-6, -70] });
  // impacto: o chão congela e prende as pernas de B
  hitIce(tI, 'A.fF', -100, 1.6);
  fx(tI, 'ring', { p: [-222, 0], r0: 20, r1: 320, squash: 0.15, w: 7, color: '180,240,255', life: 0.55 });
  fx(tI, 'frost', { p: [-222, 0], w: 260, life: 12, grow: 0.25 });
  fx(tI, 'encase', { p: [-222, 0], h: 84, w: 76, n: 7, grow: 0.16, crack: 0.95, zapCrack: true, shatter: 1.28, life: 2.3, layer: 'midB' });
  fx(tI, 'invert', { life: 2 / 60 });
  fx(tI, 'flash', { life: 0.16, amt: 0.4, color: '210,245,255' });
  freeze(tI, 0.1);
  sfx(tI, 'iceRise', 1.2);
  // A rebate num mortal para trás
  A.k(T + 0.72, { p: 'tuck', x: -150, y: -262, rot: -1130 }, 's', 'o');
  A.k(T + 0.88, { p: 'tuck', x: -60, y: -190, rot: -1290 });
  A.k(T + 1.02, { p: 'land', x: 20, y: 0, rot: -1440 }, 's', 'i2');
  A.k(T + 1.3, { p: 'kfReady', x: 30, rot: -1440, glow: 0.8 });
  landDust(T + 1.02, [20, 0]);
  // B luta contra o gelo, acumula raios e se liberta
  B.k(T + 0.9, { p: 'spGuard', x: -222, hip: [-4, -74], ln: 16, bz: 0.3 });
  B.k(T + 1.05, { ln: 6, hd: -10, bz: 0.5 });
  B.k(T + 1.18, { ln: 18, hd: 6, bz: 0.7 });
  B.k(T + 1.32, { ln: 4, hd: -12, bz: 0.9, zap: 1 });
  B.k(T + 1.46, { ln: 14, hd: 4, bz: 1 });
  const tB = T + 1.9;
  B.k(tB - 0.06, { p: 'spGuard', x: -222, hip: [-4, -70], ln: 20, hd: 10, bz: 1 });
  B.k(tB, { p: 'spRoar', x: -222, bz: 1, zap: 1 }, 'snap');
  fx(tB, 'ring', { p: 'B.chest', r0: 10, r1: 260, w: 6, color: '255,236,140', life: 0.45 });
  fx(tB, 'flash', { life: 0.14, amt: 0.35, color: '255,245,200' });
  for (let i = 0; i < 6; i++) {
    const a = (-150 + i * 24) * DEG;
    fx(tB, 'bolt', { a: 'B.chest', b: { ref: 'B.chest', dx: Math.cos(a) * 170, dy: Math.sin(a) * 140 }, life: 0.25, w: 0.8, branches: 2 });
  }
  shake(tB, 12, 0.45);
  sfx(T + 1.0, 'crackle', 1); sfx(tB, 'shatter', 1.2); sfx(tB, 'thunder', 0.9);
}

// ================= BEAT 8/9: golpes finais — dragão de gelo x lança-relâmpago =================
function beatFinale(T) {
  MARKS.ult0 = T;
  const E = T + 1.2;          // dragão emerge
  const C = T + 1.95;         // choque
  const BITE = C + 0.55;      // dragão morde
  const HIT = C + 0.8;        // dragão atinge B
  // ----- caminho do dragão -----
  const path = [[-43, -118], [-110, -92], [-176, -150], [-160, -262], [-70, -300], [-20, -226], [-96, -186], [-196, -250], [-276, -346], [-380, -420], [-520, -470]];
  const sC = 650;
  const dpts = catmull(path, 16), dS = arcTable(dpts);
  const hq = atArc(dpts, dS, sC);
  const Q = [hq[0] + Math.cos(hq[2]) * 52, hq[1] + Math.sin(hq[2]) * 52];
  const sk = [[0, 0, 'l'], [0.12, 70, 'o'], [C - E, sC], [BITE - E, sC + 22, 'l'], [HIT - E, 812, 'i2'], [HIT - E + 0.5, 1180, 'o']];
  fx(E, 'dragon', {
    path, sk, len: 560, w: 26, life: HIT - E + 0.55, fadeT: 0.4, layer: 'front',
    jaw: a => (a < C - E - 0.1 ? 0.55 + 0.25 * Math.sin(a * 14) : a < BITE - E ? 1 : Math.max(0.1, 1 - (a - (BITE - E)) * 8)),
    electric: a => (a < C - E ? 0 : a < BITE - E ? 0.4 + (a - (C - E)) * 1.2 : 1),
  });
  // ----- A: kata (mãos de nuvem), selo e glifo -----
  A.k(T + 0.0, { p: 'kfReady', x: 30, face: -1, rot: -1800, glow: 0.8 });
  A.k(T + 0.16, { p: 'kfHorse', x: 40, glow: 1 });
  A.k(T + 0.34, { hF: [20, -70], hB: [56, 18], ln: -4 });
  A.k(T + 0.52, { hF: [-28, -30], hB: [40, -52], ln: 2 });
  A.k(T + 0.7, { hF: [20, 30], hB: [-22, -58], ln: 4 });
  A.k(T + 0.88, { hF: [62, -8], hB: [-28, 12], ln: 0 });
  A.k(T + 1.02, { p: 'kfWind', x: 42, hip: [-12, -66] });
  A.k(E, { p: 'palm2', x: 34, hip: [8, -62] }, 'snap');
  A.k(E + 0.4, { p: 'palm2', x: 34, hip: [8, -64], ln: 12 });
  A.k(HIT, { p: 'palm2', x: 34, hip: [8, -64], ln: 10 });
  fx(T + 0.16, 'seal', { p: [40, 0], r: 170, squash: 0.2, life: HIT - T + 0.6 });
  fx(T + 0.4, 'glyph', { ch: '冰', p: [150, -230], size: 240, color: '110,205,245', alpha: 0.34, life: E - T + 0.2 });
  fx(T + 0.3, 'charge', { p: 'A.chest', follow: true, r: 200, n: 30, color: '220,248,255', life: 0.95 });
  env('snow', T + 0.2, 0.2); env('snow', T + 1.0, 0.9);
  env('frost', T + 0.2, 0.35); env('frost', T + 1.2, 0.55);
  sfx(T + 0.16, 'iceRise', 0.8); sfx(T + 0.3, 'charge', 1);
  // dragão sai das palmas
  fx(E, 'ring', { p: 'A.hF', r0: 10, r1: 160, w: 6, color: '200,245,255', life: 0.4 });
  fx(E, 'flash', { life: 0.15, amt: 0.35, color: '220,248,255' });
  fx(E, 'shards', { p: 'A.hF', n: 14, speed: 500, size: 8, dir: 180, spread: 120, life: 1.0 });
  shake(E, 10, 0.5);
  sfx(E, 'roar', 1.2); sfx(E, 'iceRise', 1);

  // ----- B: carrega a lança com raios, salta e arremessa -----
  B.k(T + 0.04, { p: 'spRoar', x: -222, bz: 0.6 });
  B.k(T + 0.18, { p: 'spSky', x: -222, zap: 1, bz: 0.8 });
  B.k(T + 0.55, { p: 'spSky', x: -222, hip: [0, -90], zap: 1, bz: 1 });
  B.k(T + 0.68, { p: 'spLow', x: -224, sa: -40 });
  B.k(T + 0.78, { p: 'spThrowWind', x: -240, y: -80 }, 's', 'o');
  B.k(T + 1.1, { p: 'spThrowWind', x: -296, y: -244, ln: -20, fF: [34, -36], fB: [-28, -6] }, 's', 'o');
  B.k(T + 1.5, { p: 'spThrowWind', x: -298, y: -236, ln: -24, hF: [-58, -46], fF: [30, -30], fB: [-30, -2] });
  const tT = T + 1.8;
  const relA = Math.atan2(Q[1] - (-372), Q[0] - (-238)) / DEG;
  B.k(tT - 0.08, { p: 'spThrowWind', x: -298, y: -232, ln: -28, hF: [-62, -50], sa: -20, fF: [30, -34], fB: [-30, -2] });
  B.k(tT, { p: 'spThrow', x: -292, y: -230, sa: relA, fF: [44, -18], fB: [-40, -6] }, 'o');
  B.k(C, { p: 'spThrow', x: -294, y: -228, hF: [60, 10], ln: 34 });
  B.k(HIT - 0.04, { p: 'spThrow', x: -298, y: -224, hF: [44, -30], ln: 0, hd: -20, bz: 0.6 });
  fx(T + 0.3, 'skybolt', { p: 'B.tip', w: 1.8, life: 0.3 });
  fx(T + 0.5, 'skybolt', { p: 'B.tip', w: 2.2, life: 0.34, seed: 4242 });
  fx(T + 0.3, 'skyflash', { life: 0.5, amt: 1 }); fx(T + 0.5, 'skyflash', { life: 0.5, amt: 1.2 });
  fx(T + 0.5, 'glyph', { ch: '雷', p: [-450, -200], size: 240, color: '255,214,80', alpha: 0.3, life: tT - T - 0.3 });
  fx(T + 0.78, 'bolt', { a: 'B.fF', b: [-250, 4], follow: true, life: 0.6, w: 0.7, branches: 2, fade: false });
  fx(T + 1.1, 'bolt', { a: 'B.fB', b: [-320, 4], follow: true, life: 0.7, w: 0.6, branches: 2, fade: false, seed: 77 });
  fx(T + 1.0, 'wind', { p: 'B.hip', follow: true, r: 110, squash: 0.5, n: 7, spin: 10, life: 0.8, alpha: 0.6 });
  shake(T + 0.3, 8); shake(T + 0.5, 12, 0.5);
  sfx(T + 0.3, 'thunder', 1); sfx(T + 0.5, 'thunder', 1.2); sfx(T + 0.78, 'whoosh', 1);
  sfx(tT, 'throw', 1); sfx(tT, 'zap', 1);
  // lança voando, presa nas mandíbulas e depois desviada
  const L = 210, u = [Math.cos(relA * DEG), Math.sin(relA * DEG)];
  const midQ = [Q[0] - u[0] * L / 2, Q[1] - u[1] * L / 2];
  B.free.push({
    t0: tT, t1: 99, pivot: 'mid', grab: false, keys: [
      { t: tT, auto: true, e: 'l' },
      { t: C, x: midQ[0], y: midQ[1], a: relA, e: 'l' },
      { t: BITE, x: midQ[0] - u[0] * 16, y: midQ[1] - u[1] * 16, a: relA + 4, e: 'l' },
      { t: BITE + 0.12, x: midQ[0] - 60, y: midQ[1] - 70, a: relA + 160, e: 'o' },
      { t: BITE + 0.4, x: -150, y: -300, a: relA + 420 },
      { t: BITE + 0.78, x: -86 - Math.cos(68 * DEG) * 105, y: 6 - Math.sin(68 * DEG) * 105, a: 360 + 68 + 360, e: 'i2' },
      { t: 99, x: -86 - Math.cos(68 * DEG) * 105, y: 6 - Math.sin(68 * DEG) * 105, a: 360 + 68 + 360 },
    ],
  });
  for (let i = 0; i < 3; i++) fx(tT, 'bolt', { a: 'B.tip', b: 'B.butt', follow: true, life: BITE - tT, w: 0.9, jag: 0.35, branches: 2, fade: false, seed: 300 + i * 17 });
  // ----- CHOQUE -----
  MARKS.dragon = E; MARKS.clash = C; MARKS.bite = BITE; MARKS.hit = HIT; MARKS.salute = HIT + 1.75;
  fx(C, 'invert', { life: 3 / 60 });
  fx(C, 'flash', { life: 0.3, amt: 0.7, color: '255,255,240' });
  freeze(C, 0.12);
  slow(C + 0.001, BITE, 0.25);
  fx(C, 'impact', { p: Q, size: 96, color: '255,240,190', spikes: 14, life: 0.26 });
  fx(C, 'ring', { p: Q, r0: 20, r1: 420, w: 9, color: '255,255,255', life: 0.6 });
  fx(C + 0.15, 'ring', { p: Q, r0: 20, r1: 300, w: 6, color: '180,240,255', life: 0.5 });
  fx(C + 0.3, 'ring', { p: Q, r0: 20, r1: 300, w: 6, color: '255,236,140', life: 0.5 });
  for (let i = 0; i < 10; i++) {
    const a = (i * 36 + 10) * DEG;
    fx(C + i * 0.05, 'bolt', { a: Q, b: [Q[0] + Math.cos(a) * 230, Q[1] + Math.sin(a) * 200], life: 0.2, w: 1.1, branches: 3 });
    fx(C + i * 0.055, 'sparks', { p: Q, n: 10, speed: 700, color: i % 2 ? '255,236,140' : '210,248,255', life: 0.4 });
  }
  fx(C, 'shards', { p: Q, n: 26, speed: 700, size: 10, dir: 0, spread: 360, w: 20, h: 20, life: 1.6 });
  shake(C, 22, 1.4, 26);
  cam('lock', C - 0.12, 0); cam('lock', C, 0.8); cam('lock', BITE, 0.8); cam('lock', HIT + 0.2, 0);
  cam('lx', C - 0.2, -470); cam('lx', C, Q[0] - 20); cam('ly', C, Q[1] + 40); cam('lz', C, 1.06);
  cam('lx', BITE, Q[0] - 30); cam('ly', BITE, Q[1] + 30); cam('lz', BITE, 1.12);
  sfx(C, 'boom', 1.4); sfx(C, 'thunder', 1.3); sfx(C + 0.02, 'shatter', 1); sfx(C + 0.05, 'grind', 1.2);
  // mordida
  fx(BITE, 'flash', { life: 0.12, amt: 0.4, color: '220,248,255' });
  fx(BITE, 'impact', { p: Q, size: 80, color: '170,235,255', spikes: 12, life: 0.3 });
  sfx(BITE, 'roar', 1.3); sfx(BITE, 'clang', 1);
  // dragão atinge B: explosão de gelo e prisão de cristal
  fx(HIT, 'impact', { p: 'B.chest', size: 140, color: '190,240,255', spikes: 16, life: 0.4 });
  fx(HIT, 'ring', { p: 'B.chest', r0: 20, r1: 360, w: 8, color: '220,248,255', life: 0.5 });
  fx(HIT, 'shards', { p: 'B.chest', n: 34, speed: 800, size: 12, dir: 0, spread: 360, w: 40, h: 60, life: 1.8 });
  fx(HIT, 'dust', { p: 'B.chest', n: 16, dir: 0, spread: 360, speed: 420, size: 28, flat: 0.8, color: '235,248,255', alpha: 0.6, life: 1.2, rise: 5 });
  fx(HIT, 'encase', { p: 'B.base', follow: true, h: 214, w: 120, n: 8, grow: 0.12, alpha: 0.6, block: true, life: 99, layer: 'midB' });
  fx(HIT, 'invert', { life: 2 / 60 });
  fx(HIT, 'flash', { life: 0.25, amt: 0.55, color: '225,248,255' });
  freeze(HIT, 0.1);
  shake(HIT, 18, 0.6);
  sfx(HIT, 'boom', 1.2); sfx(HIT, 'ice', 1.4); sfx(HIT, 'shatter', 1);
  // B congelado cai
  B.k(HIT, { p: 'spThrow', x: -298, y: -224, hF: [44, -30], ln: 0, hd: -20, tint: 0, bz: 0.3, zap: 0.4 });
  B.k(HIT + 0.08, { tint: 1, bz: 0, zap: 0 }, 'l');
  const tG = HIT + 0.55;
  B.k(HIT + 0.18, { x: -298, y: -222 });
  B.k(tG, { x: -300, y: 0 }, 's', 'i2');
  fx(tG, 'debris', { p: [-300, -4], n: 14, speed: 560, size: 8, dir: -90, spread: 150, life: 1.8 });
  fx(tG, 'shards', { p: [-300, -10], n: 16, speed: 520, size: 9, dir: -90, spread: 160, w: 90, h: 10, life: 1.4 });
  fx(tG, 'dust', { p: [-300, 0], n: 18, dir: 0, spread: 360, speed: 420, size: 22, flat: 0.25, alpha: 0.5, life: 1.3 });
  fx(tG, 'crack', { p: [-300, 0], size: 140, n: 10, rocks: 10, life: 40 });
  fx(tG, 'frost', { p: [-300, 0], w: 220, life: 40 });
  fx(tG, 'ring', { p: [-300, 0], r0: 20, r1: 300, squash: 0.15, w: 6, color: '220,248,255', life: 0.5 });
  shake(tG, 16, 0.6);
  sfx(tG, 'boom', 1); sfx(tG, 'land', 1);
  // lança finca no chão
  const tSp = BITE + 0.78;
  fx(tSp, 'dust', { p: [-86, 0], n: 8, dir: -90, spread: 160, speed: 160, size: 12, flat: 0.4, life: 0.6 });
  fx(tSp, 'crack', { p: [-86, 0], size: 40, n: 5, life: 40 });
  sfx(tSp, 'stab', 1);
  // ----- desfecho -----
  env('storm', HIT, 1); env('storm', HIT + 2.2, 0.15);
  env('rain', HIT, 0.85); env('rain', HIT + 1.2, 0);
  env('dark', HIT, 0.18); env('dark', HIT + 1.6, 0);
  env('snow', HIT, 0.9); env('snow', HIT + 2.5, 0.5);
  A.k(HIT + 0.45, { p: 'palm2', x: 34, hip: [6, -66], ln: 8, glow: 0.8 });
  A.k(HIT + 0.95, { p: 'kfReady', x: 36, glow: 0.5 });
  A.k(HIT + 1.35, { p: 'stand', x: 40, glow: 0.3, hF: [10, 40], hB: [6, 44] });
  A.k(HIT + 1.75, { p: 'salute', x: 40, glow: 0.2 });
  breathe(A, HIT + 1.75, HIT + 6, [0, -88], 1.6, 1.0);
  cam('zm', 0, 1); cam('zm', HIT + 1.2, 1); cam('zm', HIT + 5.5, 1.12);
  sfx(HIT + 1.75, 'salute', 0.8);
}

// ================= BEAT 7b: troca rápida — aparo, giro, rasteira, lança presa entre as palmas =================
function beatFlurry(T) {
  // B investe; A desvia com a palma
  B.k(T + 0.0, { p: 'spRoar', x: -222, bz: 0.6, zap: 1 });
  B.k(T + 0.1, { p: 'spDraw', x: -270, bz: 0.3 });
  const p1 = T + 0.24;
  B.k(p1, { p: 'spThrust', x: -262, sg: 100, sb: 30, gh: 0.6, bz: 0 }, 'o', 'io');
  B.k(T + 0.32, { x: -258, sa: -30, ln: 34, gh: 0.1 }, 'o');
  A.k(T + 0.1, { p: 'kfReady', x: 30, face: -1, rot: -1440, glow: 0.8 });
  A.k(T + 0.2, { hF: [44, -10], ln: 2 });
  A.k(p1, { hF: [56, -30], hB: [24, 0], ln: -6, x: 34 }, 'o');
  clash(p1, 'A.hF', 0.9, '170,235,255');
  sfx(T + 0.12, 'whoosh', 0.9);
  // A gira e chuta; B bloqueia com a haste
  A.k(T + 0.32, { p: 'kfWind', x: 14 });
  A.k(T + 0.4, { p: 'kickChamber', x: -50, face: 1 }, 'sine');
  const k1 = T + 0.5;
  A.k(k1, { p: 'sideKick', x: -128, face: -1 }, 'sine');
  A.k(T + 0.6, { p: 'kfReady', x: -118 });
  B.k(T + 0.42, { p: 'spBlockMid', x: -262 });
  B.k(k1, { p: 'spBlockMid', x: -266, ln: -6 });
  B.k(T + 0.58, { p: 'spBlockMid', x: -262 });
  clash(k1, 'A.fF', 1.1);
  freeze(k1, 0.04);
  sfx(T + 0.42, 'whoosh', 1);
  // B varre baixo; A salta para trás
  B.k(T + 0.64, { p: 'spLow', x: -262, sa: -6 });
  B.k(T + 0.74, { p: 'spSweep', x: -252, sa: 12 }, 'o');
  A.k(T + 0.64, { p: 'crouch', x: -112 });
  A.k(T + 0.74, { p: 'tuck', x: -92, y: -74 }, 's', 'o');
  A.k(T + 0.88, { p: 'land', x: -62, y: 0 }, 's', 'i2');
  landDust(T + 0.88, [-62, 0], 0.7);
  sfx(T + 0.72, 'whoosh', 0.9);
  // B estoca; A prende a lança entre as palmas e congela a haste
  B.k(T + 0.9, { p: 'spDraw', x: -270 });
  const tc = T + 1.02;
  B.k(tc, { p: 'spThrust', x: -268, hF: [40, 8], sg: 100, sb: 30 }, 'snap', 'o');
  B.k(T + 1.14, { p: 'spThrust', x: -272, hF: [34, 8], ln: 20, hd: -6 });
  B.k(T + 1.26, { p: 'spThrust', x: -266, hF: [40, 6], ln: 24 });
  B.k(T + 1.36, { p: 'spThrust', x: -276, hF: [30, 8], ln: 14, hd: -10 });
  A.k(T + 0.96, { p: 'kfHorse', x: -62 });
  A.k(tc, { p: 'clap', x: -62, glow: 1 }, 'snap');
  A.k(T + 1.14, { p: 'clap', x: -60, ln: 2 });
  A.k(T + 1.26, { p: 'clap', x: -64, ln: 8 });
  A.k(T + 1.36, { p: 'clap', x: -58, ln: 0 });
  // ângulo da lança calculado para passar entre as palmas de A
  POST.push(() => {
    for (const t of [tc, T + 1.14, T + 1.26, T + 1.36]) {
      const SA = A.at(t), SB = B.at(t);
      const hm = [(SA.hF[0] + SA.hB[0]) / 2, (SA.hF[1] + SA.hB[1]) / 2];
      const a = Math.atan2(hm[1] - SB.hF[1], hm[0] - SB.hF[0]) / DEG;
      B.k(t, { sa: a, sg: 100 + Math.max(0, Math.hypot(hm[0] - SB.hF[0], hm[1] - SB.hF[1]) - 78) });
    }
  });
  clash(tc, 'A.hF', 1.0, '200,245,255');
  freeze(tc, 0.06);
  fx(tc, 'spearFrost', { from: 'A.hF', speed: 260, life: 0.44 });
  fx(tc, 'ring', { p: 'A.hF', r0: 4, r1: 60, w: 4, color: '200,245,255', life: 0.25 });
  sfx(tc, 'clap', 1); sfx(tc + 0.05, 'iceForm', 1);
  // A puxa e chuta B para longe; mortal para trás
  const kk = T + 1.5;
  A.k(T + 1.44, { p: 'clap', x: -50, ln: -10, hF: [40, -16], hB: [38, -12] }, 'o');
  B.k(T + 1.44, { p: 'spThrust', x: -176, hF: [44, 8], ln: 30, hd: 8 }, 'o', 'o');
  A.k(kk, { p: 'kick', x: -74, fF: [88, -76], hF: [30, -10], hB: [-30, 10] }, 'snap');
  B.k(kk, { p: 'hit', x: -150, spear: 1, sa: -70, sg: 100, sb: -1, hF: [30, -40] });
  B.k(kk + 0.08, { p: 'hit', x: -196, y: -16, spear: 1, sa: -60, sg: 100, sb: -1, hF: [30, -30] }, 'o', 'o');
  B.k(T + 1.84, { p: 'spLow', x: -236, y: 0, sa: -20 }, 's', 'o');
  POST.push(() => B.k(T + 1.44, { sa: B.at(T + 1.36).P.sa - 6 }));
  hitIce(kk, 'B.chest', 180, 1.1);
  fx(kk, 'shards', { p: 'B.mid', n: 16, speed: 420, size: 8, dir: -90, spread: 200, w: 60, h: 10, life: 1.0 });
  fx(kk, 'hitflash', { id: 'B', life: 0.06 });
  freeze(kk, 0.06);
  fx(kk + 0.16, 'skid', { p: [-200, 0], dir: -1, len: 50, life: 30 });
  sfx(kk, 'shatter', 0.9);
  A.k(T + 1.6, { p: 'tuck', x: -20, y: -90, rot: -1530 }, 's', 'o');
  A.k(T + 1.74, { p: 'tuck', x: 14, y: -60, rot: -1680 });
  A.k(T + 1.85, { p: 'land', x: 30, y: 0, rot: -1800 }, 's', 'i2');
  landDust(T + 1.85, [30, 0], 0.8);
}
