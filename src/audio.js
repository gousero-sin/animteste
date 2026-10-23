// Trilha sonora: efeitos sintetizados + música (taiko + "guzheng" pentatônico) via OfflineAudioContext.
'use strict';

const NOTE = {
  D2: 73.42, A2: 110, D3: 146.83, E3: 164.81, Fs3: 185, A3: 220, B3: 246.94,
  D4: 293.66, E4: 329.63, Fs4: 369.99, A4: 440, B4: 493.88,
  D5: 587.33, E5: 659.26, Fs5: 739.99, A5: 880, B5: 987.77, D6: 1174.66,
};

function aEnv(g, t, a, peak, d, hold = 0) {
  g.gain.setValueAtTime(0.0001, t);
  g.gain.linearRampToValueAtTime(Math.max(0.0002, peak), t + a);
  if (hold) g.gain.setValueAtTime(Math.max(0.0002, peak), t + a + hold);
  g.gain.exponentialRampToValueAtTime(0.0001, t + a + hold + d);
}
function aOut(X, node, pan, dest) {
  if (pan) {
    const p = X.ac.createStereoPanner();
    p.pan.value = clamp(pan, -1, 1);
    node.connect(p); p.connect(dest);
  } else node.connect(dest);
}
function aNoise(X, t, dur, o, dest) {
  const src = X.ac.createBufferSource();
  src.buffer = o.brown ? X.bb : X.nb;
  let node = src;
  if (o.f) {
    const f = X.ac.createBiquadFilter();
    f.type = o.type || 'bandpass';
    f.frequency.setValueAtTime(o.f, t);
    if (o.f1) f.frequency.exponentialRampToValueAtTime(o.f1, t + (o.ft || dur));
    f.Q.value = o.q || 1;
    node.connect(f); node = f;
  }
  if (o.hp) {
    const f = X.ac.createBiquadFilter();
    f.type = 'highpass'; f.frequency.value = o.hp; f.Q.value = 0.7;
    node.connect(f); node = f;
  }
  const g = X.ac.createGain();
  node.connect(g);
  aEnv(g, t, o.a !== undefined ? o.a : 0.004, o.peak !== undefined ? o.peak : 0.5, o.d !== undefined ? o.d : dur, o.hold || 0);
  aOut(X, g, o.pan, dest);
  src.start(t, X.R() * 2, dur + (o.hold || 0) + 0.6);
}
function aTone(X, t, o, dest) {
  const osc = X.ac.createOscillator();
  osc.type = o.type || 'sine';
  osc.frequency.setValueAtTime(o.f, t);
  if (o.f1) osc.frequency.exponentialRampToValueAtTime(o.f1, t + (o.ft || o.d || 0.3));
  if (o.detune) osc.detune.value = o.detune;
  let node = osc;
  if (o.lp) {
    const f = X.ac.createBiquadFilter();
    f.type = 'lowpass'; f.frequency.value = o.lp; f.Q.value = o.lq || 0.7;
    node.connect(f); node = f;
  }
  const g = X.ac.createGain();
  node.connect(g);
  const a = o.a !== undefined ? o.a : 0.003, d = o.d || 0.3;
  aEnv(g, t, a, o.peak !== undefined ? o.peak : 0.3, d, o.hold || 0);
  aOut(X, g, o.pan, dest);
  osc.start(t);
  osc.stop(t + a + (o.hold || 0) + d + 0.05);
  return osc;
}
function makeIR(ac, secs, R) {
  const len = Math.round(ac.sampleRate * secs);
  const ir = ac.createBuffer(2, len, ac.sampleRate);
  for (let c = 0; c < 2; c++) {
    const d = ir.getChannelData(c);
    for (let i = 0; i < len; i++) {
      const k = i / len;
      d[i] = (R() * 2 - 1) * Math.pow(1 - k, 3.2) * (i < 200 ? i / 200 : 1);
    }
  }
  return ir;
}
function shaper(ac, amt) {
  const ws = ac.createWaveShaper();
  const n = 1024, c = new Float32Array(n);
  for (let i = 0; i < n; i++) { const x = i / (n - 1) * 2 - 1; c[i] = Math.tanh(x * amt) / Math.tanh(amt); }
  ws.curve = c;
  return ws;
}

// ---------- efeitos ----------
const SFX = {
  whoosh(X, t, g) {
    aNoise(X, t, 0.3, { type: 'bandpass', f: 420, f1: 2400, ft: 0.13, q: 1.3, a: 0.07, peak: 0.42 * g, d: 0.2, pan: (X.R() - 0.5) * 0.7 }, X.sfx);
    aNoise(X, t + 0.02, 0.2, { type: 'bandpass', f: 900, f1: 4200, ft: 0.12, q: 2, a: 0.05, peak: 0.12 * g, d: 0.14 }, X.sfx);
  },
  hit(X, t, g) {
    aTone(X, t, { f: 170, f1: 40, ft: 0.15, d: 0.24, peak: 0.85 * g, a: 0.002 }, X.sfx);
    aNoise(X, t, 0.14, { type: 'lowpass', f: 3200, f1: 260, ft: 0.1, q: 0.8, a: 0.001, peak: 0.75 * g, d: 0.11 }, X.sfx);
    aNoise(X, t, 0.03, { type: 'highpass', f: 2600, q: 0.7, a: 0.0005, peak: 0.45 * g, d: 0.028 }, X.sfx);
  },
  ice(X, t, g) {
    for (let i = 0; i < 10; i++) {
      aTone(X, t + X.R() * 0.1, { f: 1700 + X.R() * 5200, d: 0.12 + X.R() * 0.42, peak: (0.035 + X.R() * 0.05) * g, a: 0.001, pan: (X.R() - 0.5) * 1.2 }, X.sfx);
    }
    aNoise(X, t, 0.22, { type: 'highpass', f: 4200, q: 0.5, a: 0.001, peak: 0.22 * g, d: 0.16 }, X.sfx);
  },
  clang(X, t, g) {
    const base = 470 + X.R() * 160;
    [1, 2.76, 4.07, 5.43, 6.81, 8.93].forEach((m, i) => aTone(X, t, { f: base * m, d: 0.7 - i * 0.07, peak: 0.12 * g / (1 + i * 0.35), a: 0.001, pan: (X.R() - 0.5) * 0.5 }, X.sfx));
    aNoise(X, t, 0.06, { type: 'bandpass', f: 3800, q: 0.8, a: 0.0005, peak: 0.55 * g, d: 0.05 }, X.sfx);
    aTone(X, t, { f: 220, f1: 80, ft: 0.1, d: 0.12, peak: 0.35 * g }, X.sfx);
  },
  land(X, t, g) {
    aTone(X, t, { f: 120, f1: 45, ft: 0.12, d: 0.18, peak: 0.6 * g, a: 0.002 }, X.sfx);
    aNoise(X, t, 0.16, { type: 'lowpass', f: 1200, f1: 300, q: 0.7, a: 0.002, peak: 0.35 * g, d: 0.14 }, X.sfx);
  },
  stomp(X, t, g) {
    aTone(X, t, { f: 90, f1: 30, ft: 0.3, d: 0.45, peak: 0.9 * g, a: 0.002 }, X.sfx);
    aNoise(X, t, 0.3, { type: 'lowpass', f: 1500, f1: 200, q: 0.7, a: 0.001, peak: 0.6 * g, d: 0.28 }, X.sfx);
  },
  thunder(X, t, g) {
    aNoise(X, t, 0.09, { type: 'highpass', f: 1400, q: 0.5, a: 0.0008, peak: 0.85 * g, d: 0.09 }, X.sfx);
    aNoise(X, t + 0.01, 0.45, { type: 'bandpass', f: 1500, f1: 280, ft: 0.4, q: 0.6, a: 0.004, peak: 0.55 * g, d: 0.4 }, X.sfx);
    aNoise(X, t + 0.03, 2.8, { brown: true, type: 'lowpass', f: 380, f1: 110, ft: 2.4, q: 0.7, a: 0.06, peak: 1.0 * g, d: 2.6 }, X.sfx);
    for (let i = 0; i < 3; i++) aNoise(X, t + 0.3 + X.R() * 1.1, 1.0, { brown: true, type: 'lowpass', f: 220, q: 0.7, a: 0.12, peak: 0.45 * g, d: 0.9, pan: (X.R() - 0.5) }, X.sfx);
  },
  zap(X, t, g) {
    const dur = 0.34;
    const osc = X.ac.createOscillator();
    osc.type = 'sawtooth';
    for (let k = 0; k < dur / 0.012; k++) osc.frequency.setValueAtTime(60 + X.R() * 220, t + k * 0.012);
    const f = X.ac.createBiquadFilter(); f.type = 'bandpass'; f.frequency.value = 1900; f.Q.value = 1.6;
    const gg = X.ac.createGain();
    osc.connect(f); f.connect(gg);
    aEnv(gg, t, 0.004, 0.5 * g, dur);
    aOut(X, gg, (X.R() - 0.5) * 0.5, X.sfx);
    osc.start(t); osc.stop(t + dur + 0.05);
    for (let i = 0; i < 14; i++) aNoise(X, t + X.R() * dur, 0.006, { type: 'highpass', f: 2500, q: 0.7, a: 0.0003, peak: (0.2 + X.R() * 0.3) * g, d: 0.006, pan: (X.R() - 0.5) }, X.sfx);
  },
  dash(X, t, g) {
    aNoise(X, t, 0.25, { type: 'bandpass', f: 600, f1: 3500, ft: 0.1, q: 1.2, a: 0.02, peak: 0.55 * g, d: 0.2 }, X.sfx);
    SFX.zap(X, t + 0.02, 0.6 * g);
  },
  skid(X, t, g) {
    aNoise(X, t, 0.45, { type: 'bandpass', f: 1100, f1: 500, q: 1.3, a: 0.01, peak: 0.3 * g, d: 0.4 }, X.sfx);
  },
  iceRise(X, t, g) {
    for (let i = 0; i < 6; i++) aNoise(X, t + i * 0.025, 0.06, { type: 'bandpass', f: 700 + i * 600, q: 3, a: 0.001, peak: 0.3 * g, d: 0.06 }, X.sfx);
    aTone(X, t, { f: 120, f1: 60, ft: 0.15, d: 0.2, peak: 0.4 * g }, X.sfx);
    SFX.ice(X, t + 0.05, 0.6 * g);
  },
  shatter(X, t, g) {
    for (let i = 0; i < 34; i++) aNoise(X, t + Math.pow(X.R(), 1.6) * 0.4, 0.008, { type: 'bandpass', f: 2500 + X.R() * 6000, q: 4, a: 0.0004, peak: (0.15 + X.R() * 0.3) * g, d: 0.01 + X.R() * 0.03, pan: (X.R() - 0.5) * 1.4 }, X.sfx);
    aNoise(X, t, 0.2, { type: 'lowpass', f: 1500, f1: 300, q: 0.7, a: 0.001, peak: 0.35 * g, d: 0.16 }, X.sfx);
    SFX.ice(X, t, 0.7 * g);
  },
  iceShot(X, t, g) {
    aNoise(X, t, 0.16, { type: 'bandpass', f: 1500, f1: 5200, ft: 0.1, q: 2, a: 0.01, peak: 0.35 * g, d: 0.12 }, X.sfx);
    aTone(X, t, { f: 3100, f1: 2400, ft: 0.15, d: 0.18, peak: 0.07 * g }, X.sfx);
  },
  charge(X, t, g) {
    const osc = X.ac.createOscillator(); osc.type = 'sawtooth';
    osc.frequency.setValueAtTime(110, t); osc.frequency.exponentialRampToValueAtTime(620, t + 0.75);
    const f = X.ac.createBiquadFilter(); f.type = 'lowpass'; f.Q.value = 7;
    f.frequency.setValueAtTime(400, t); f.frequency.exponentialRampToValueAtTime(3200, t + 0.75);
    const gg = X.ac.createGain();
    gg.gain.setValueAtTime(0.0001, t); gg.gain.exponentialRampToValueAtTime(0.16 * g, t + 0.65); gg.gain.exponentialRampToValueAtTime(0.0001, t + 0.85);
    osc.connect(f); f.connect(gg); gg.connect(X.sfx);
    osc.start(t); osc.stop(t + 0.9);
    aNoise(X, t, 0.8, { type: 'highpass', f: 3000, q: 0.7, a: 0.6, peak: 0.12 * g, d: 0.2 }, X.sfx);
  },
  boom(X, t, g) {
    const osc = X.ac.createOscillator();
    osc.frequency.setValueAtTime(62, t); osc.frequency.exponentialRampToValueAtTime(24, t + 0.9);
    const ws = shaper(X.ac, 2.2);
    const gg = X.ac.createGain();
    osc.connect(ws); ws.connect(gg);
    aEnv(gg, t, 0.003, 0.95 * g, 1.0);
    gg.connect(X.sfx);
    osc.start(t); osc.stop(t + 1.1);
    aNoise(X, t, 1.4, { type: 'lowpass', f: 1600, f1: 90, ft: 1.2, q: 0.7, a: 0.002, peak: 0.85 * g, d: 1.3 }, X.sfx);
    aNoise(X, t, 0.06, { type: 'highpass', f: 2000, q: 0.5, a: 0.0005, peak: 0.6 * g, d: 0.06 }, X.sfx);
  },
  iceForm(X, t, g) {
    [1760, 2093, 2349, 2637, 3136, 3520, 4186].forEach((f, i) => aTone(X, t + i * 0.035, { f, d: 0.5, peak: 0.07 * g, a: 0.002, pan: (i / 6 - 0.5) }, X.sfx));
    aNoise(X, t, 0.4, { type: 'highpass', f: 5000, q: 0.5, a: 0.15, peak: 0.1 * g, d: 0.25 }, X.sfx);
  },
  grind(X, t, g) {
    const dur = 1.5;
    const src = X.ac.createBufferSource(); src.buffer = X.nb;
    const f = X.ac.createBiquadFilter(); f.type = 'bandpass'; f.Q.value = 9;
    for (let k = 0; k < dur / 0.05; k++) f.frequency.setValueAtTime(1800 + X.R() * 1600, t + k * 0.05);
    const gg = X.ac.createGain();
    src.connect(f); f.connect(gg);
    aEnv(gg, t, 0.05, 0.55 * g, 0.6, dur - 0.6);
    gg.connect(X.sfx);
    src.start(t, 0, dur + 0.2);
    for (const fq of [1310, 1870, 2730]) {
      const o = aTone(X, t, { f: fq, d: 0.6, hold: dur - 0.6, peak: 0.035 * g, a: 0.05 }, X.sfx);
      const lfo = X.ac.createOscillator(); lfo.frequency.value = 7 + X.R() * 5;
      const lg = X.ac.createGain(); lg.gain.value = 18;
      lfo.connect(lg); lg.connect(o.frequency);
      lfo.start(t); lfo.stop(t + dur + 0.1);
    }
  },
  rumble(X, t, g) {
    aNoise(X, t, 2.0, { brown: true, type: 'lowpass', f: 170, q: 0.7, a: 0.9, peak: 0.8 * g, d: 1.1 }, X.sfx);
  },
  crackle(X, t, g) {
    for (let i = 0; i < 40; i++) aNoise(X, t + X.R() * 0.9, 0.005, { type: 'highpass', f: 2200, q: 0.7, a: 0.0003, peak: (0.1 + X.R() * 0.3) * g, d: 0.006, pan: (X.R() - 0.5) }, X.sfx);
    SFX.zap(X, t + 0.3, 0.5 * g);
  },
  roar(X, t, g) {
    const dur = 1.4;
    const bus = X.ac.createGain();
    const ws = shaper(X.ac, 1.6);
    bus.connect(ws);
    const out = X.ac.createGain();
    aEnv(out, t, 0.14, 0.75 * g, 0.65, dur - 0.8);
    for (const [fq, q, gn] of [[480, 5, 1], [1150, 6, 0.7], [2600, 8, 0.35]]) {
      const bp = X.ac.createBiquadFilter(); bp.type = 'bandpass'; bp.frequency.value = fq; bp.Q.value = q;
      const bg = X.ac.createGain(); bg.gain.value = gn;
      ws.connect(bp); bp.connect(bg); bg.connect(out);
    }
    out.connect(X.sfx);
    const lfo = X.ac.createOscillator(); lfo.frequency.value = 6.5;
    const lg = X.ac.createGain(); lg.gain.value = 7;
    lfo.connect(lg);
    for (const [mult, type, det] of [[1, 'sawtooth', 0], [1.01, 'sawtooth', 8], [0.5, 'square', 0]]) {
      const o = X.ac.createOscillator(); o.type = type; o.detune.value = det;
      o.frequency.setValueAtTime(80 * mult, t);
      o.frequency.linearRampToValueAtTime(118 * mult, t + 0.35);
      o.frequency.linearRampToValueAtTime(68 * mult, t + dur);
      lg.connect(o.frequency);
      o.connect(bus);
      o.start(t); o.stop(t + dur + 0.1);
    }
    lfo.start(t); lfo.stop(t + dur + 0.1);
    aNoise(X, t, dur, { type: 'bandpass', f: 900, q: 0.8, a: 0.1, peak: 0.25 * g, d: 0.7, hold: 0.5 }, X.sfx);
    SFX.ice(X, t + 0.05, 0.8 * g);
  },
  throw(X, t, g) {
    aNoise(X, t, 0.35, { type: 'bandpass', f: 300, f1: 1800, ft: 0.15, q: 1, a: 0.03, peak: 0.6 * g, d: 0.28 }, X.sfx);
  },
  stab(X, t, g) {
    aTone(X, t, { f: 240, f1: 70, ft: 0.12, d: 0.16, peak: 0.6 * g }, X.sfx);
    aNoise(X, t, 0.12, { type: 'lowpass', f: 2000, f1: 400, q: 0.7, a: 0.001, peak: 0.4 * g, d: 0.1 }, X.sfx);
    [1, 2.7, 5.1].forEach((m, i) => aTone(X, t, { f: 900 * m, d: 0.35, peak: 0.05 * g / (1 + i) }, X.sfx));
  },
  clap(X, t, g) {
    aNoise(X, t, 0.04, { type: 'bandpass', f: 1600, q: 1.1, a: 0.0005, peak: 0.7 * g, d: 0.04 }, X.sfx);
    aNoise(X, t + 0.008, 0.06, { type: 'bandpass', f: 1300, q: 1.1, a: 0.0005, peak: 0.5 * g, d: 0.06 }, X.sfx);
    aTone(X, t, { f: 160, f1: 70, ft: 0.08, d: 0.1, peak: 0.4 * g }, X.sfx);
  },
  salute(X, t, g) { gong(X, t, 0.5 * g, X.sfx, 98); },
};

function gong(X, t, g, dest, base = 92) {
  [1, 2.02, 2.94, 4.1, 5.33, 6.9].forEach((m, i) => {
    const o = aTone(X, t, { f: base * m * (1 + 0.004 * i), f1: base * m * 0.992, ft: 3, d: 4.5 - i * 0.5, peak: 0.22 * g / (1 + i * 0.4), a: 0.01 + i * 0.01 }, dest);
    void o;
  });
  aNoise(X, t, 0.3, { type: 'lowpass', f: 900, q: 0.7, a: 0.005, peak: 0.25 * g, d: 0.28 }, dest);
}

// ---------- instrumentos da música ----------
function ksBuffer(X, f, dur, bright) {
  const key = f.toFixed(2) + '|' + dur + '|' + bright;
  if (X.ks[key]) return X.ks[key];
  const SR = X.SR, N = Math.max(2, Math.floor(SR / f - 0.5)), len = Math.round(SR * dur);
  const buf = X.ac.createBuffer(1, len, SR), d = buf.getChannelData(0);
  const ring = new Float32Array(N);
  let lp = 0;
  for (let i = 0; i < N; i++) { const w = X.R() * 2 - 1; lp = lp + (w - lp) * bright; ring[i] = lp; }
  let idx = 0;
  const decay = 0.9975;
  for (let i = 0; i < len; i++) {
    const cur = ring[idx], nxt = ring[(idx + 1) % N];
    d[i] = cur;
    ring[idx] = decay * 0.5 * (cur + nxt);
    idx = (idx + 1) % N;
  }
  const res = { buf, rate: f * (N + 0.5) / SR };
  X.ks[key] = res;
  return res;
}
function pluck(X, t, f, g, opt = {}) {
  const ks = ksBuffer(X, f, opt.dur || 2.2, opt.bright || 0.55);
  const src = X.ac.createBufferSource();
  src.buffer = ks.buf;
  src.playbackRate.setValueAtTime(ks.rate, t);
  if (opt.bend) {
    src.playbackRate.setValueAtTime(ks.rate, t + 0.08);
    src.playbackRate.linearRampToValueAtTime(ks.rate * Math.pow(2, opt.bend / 12), t + 0.22);
  }
  if (opt.vib) {
    const lfo = X.ac.createOscillator(); lfo.frequency.value = 5.5;
    const lg = X.ac.createGain(); lg.gain.value = ks.rate * 0.006 * opt.vib;
    lfo.connect(lg); lg.connect(src.playbackRate);
    lfo.start(t + 0.15); lfo.stop(t + (opt.dur || 2.2));
  }
  const body = X.ac.createBiquadFilter(); body.type = 'peaking'; body.frequency.value = 520; body.gain.value = 5; body.Q.value = 1.2;
  const gg = X.ac.createGain(); gg.gain.value = g;
  src.connect(body); body.connect(gg);
  aOut(X, gg, opt.pan || 0, X.mus);
  src.start(t);
}
function don(X, t, g, big) {
  aTone(X, t, { f: big ? 74 : 96, f1: big ? 40 : 54, ft: big ? 0.35 : 0.2, d: big ? 1.0 : 0.42, peak: 0.85 * g, a: 0.003 }, X.mus);
  aNoise(X, t, 0.22, { type: 'lowpass', f: 800, f1: 180, ft: 0.15, q: 0.8, a: 0.001, peak: 0.45 * g, d: 0.2 }, X.mus);
  aNoise(X, t, 0.03, { type: 'bandpass', f: 1700, q: 1, a: 0.0005, peak: 0.18 * g, d: 0.03 }, X.mus);
}
function ka(X, t, g) {
  aNoise(X, t, 0.04, { type: 'bandpass', f: 2700, q: 2.2, a: 0.0004, peak: 0.32 * g, d: 0.04, pan: 0.25 }, X.mus);
  aTone(X, t, { f: 1250, d: 0.035, peak: 0.07 * g }, X.mus);
}
function shaker(X, t, g) {
  aNoise(X, t, 0.05, { type: 'highpass', f: 6500, q: 0.7, a: 0.006, peak: 0.07 * g, d: 0.04, pan: -0.3 }, X.mus);
}
function drone(X, t0, t1, g, notes = [NOTE.D2, NOTE.A2]) {
  for (const f of notes) {
    for (const det of [-6, 6]) {
      const o = X.ac.createOscillator(); o.type = 'sawtooth'; o.frequency.value = f; o.detune.value = det;
      const lp = X.ac.createBiquadFilter(); lp.type = 'lowpass'; lp.frequency.value = 420; lp.Q.value = 0.8;
      const gg = X.ac.createGain();
      gg.gain.setValueAtTime(0.0001, t0);
      gg.gain.linearRampToValueAtTime(g * 0.5, t0 + Math.min(1.2, (t1 - t0) * 0.3));
      gg.gain.setValueAtTime(g * 0.5, Math.max(t0 + 0.01, t1 - Math.min(1.0, (t1 - t0) * 0.3)));
      gg.gain.linearRampToValueAtTime(0.0001, t1);
      o.connect(lp); lp.connect(gg); gg.connect(X.mus);
      o.start(t0); o.stop(t1 + 0.05);
    }
  }
}
function swell(X, t0, t1, g) {
  aNoise(X, t0, t1 - t0, { type: 'bandpass', f: 600, f1: 7000, ft: t1 - t0, q: 0.8, a: (t1 - t0) * 0.95, peak: g, d: 0.06 }, X.mus);
}

// padrões de 16 passos
const OSTINATO = [
  [NOTE.D5, NOTE.A4, NOTE.Fs4, NOTE.A4, NOTE.D5, NOTE.E5, NOTE.Fs5, NOTE.E5],
  [NOTE.D5, NOTE.B4, NOTE.A4, NOTE.B4, NOTE.D5, NOTE.E5, NOTE.A5, NOTE.Fs5],
  [NOTE.E5, NOTE.B4, NOTE.A4, NOTE.B4, NOTE.E5, NOTE.Fs5, NOTE.A5, NOTE.B5],
  [NOTE.A5, NOTE.Fs5, NOTE.E5, NOTE.D5, NOTE.E5, NOTE.D5, NOTE.B4, NOTE.A4],
];
function groove(X, t0, t1, I = 1) {
  const st = 60 / 140 / 4;
  const DON = [0, 3, 6, 8, 11, 14], DONB = [0, 8];
  const KA = [4, 12], KAG = [2, 7, 10, 15];
  const BASS = [[0, NOTE.D3], [3, NOTE.D3], [8, NOTE.A3], [11, NOTE.A3]];
  let bar = 0;
  for (let tb = t0; tb < t1 - 0.05; tb += st * 16, bar++) {
    for (let i = 0; i < 16; i++) {
      const t = tb + i * st;
      if (t >= t1) break;
      if (DON.includes(i)) don(X, t, (DONB.includes(i) ? 0.9 : 0.55) * I, DONB.includes(i) && bar % 2 === 0 && i === 0);
      if (KA.includes(i)) ka(X, t, 0.8 * I);
      if (I > 1 && KAG.includes(i)) ka(X, t, 0.3 * I);
      shaker(X, t, (i % 2 ? 0.6 : 1) * I);
      for (const [s, f] of BASS) if (s === i) pluck(X, t, f * (bar % 4 === 3 && s >= 8 ? 1.122 : 1), 0.22 * I, { dur: 1.4, bright: 0.35 });
      if (i % 2 === 0) {
        const ph = OSTINATO[bar % 4];
        pluck(X, t, ph[i / 2], 0.1 * I, { dur: 1.2, bright: 0.6, pan: (i % 4 ? 0.35 : -0.35) });
      }
    }
  }
}
function roll(X, t0, t1) {
  let t = t0, k = 0;
  while (t < t1) {
    const u = (t - t0) / (t1 - t0);
    don(X, t, 0.25 + 0.75 * u, false);
    const step = lerp(0.2, 0.045, u);
    if (k % 2 === 0) pluck(X, t, [NOTE.D5, NOTE.E5, NOTE.Fs5, NOTE.A5, NOTE.B5, NOTE.D6][Math.min(5, Math.floor(u * 6))], 0.08 + 0.08 * u, { dur: 0.8, bright: 0.65, pan: (k % 4 ? 0.3 : -0.3) });
    t += step; k++;
  }
}

function music(X) {
  const W = SCENE.warp, M = MARKS, r = s => W.real(s);
  const D = CFG.DURATION;
  const tFight = r(M.fight0), tL0 = r(M.lock0), tL1 = r(M.lock1), tUlt = r(M.ult0);
  const tClash = r(M.clash), tHit = r(M.hit), tSal = r(M.salute);
  // introdução
  drone(X, 0, tFight + 0.6, 0.11);
  [NOTE.D4, NOTE.E4, NOTE.Fs4, NOTE.A4, NOTE.B4, NOTE.D5, NOTE.E5, NOTE.Fs5, NOTE.A5].forEach((f, i) => pluck(X, 0.05 + i * 0.05, f, 0.12, { dur: 2, bright: 0.6, pan: -0.5 + i / 8 }));
  pluck(X, r(0.95), NOTE.A4, 0.16, { dur: 2, bend: 2 });
  pluck(X, r(1.25), NOTE.B4, 0.14, { dur: 2 });
  pluck(X, r(1.52), NOTE.D5, 0.16, { dur: 2.4, vib: 1 });
  don(X, r(1.32), 1, true);
  don(X, r(1.78), 1, true);
  swell(X, r(2.0), tFight, 0.12);
  // luta (parte 1)
  groove(X, tFight, tL0, 1);
  // travamento em câmera lenta: só batimentos e drone
  drone(X, tL0, tL1 + 0.3, 0.12, [NOTE.D2, NOTE.A2, NOTE.D3]);
  for (let t = tL0 + 0.1; t < tL1 - 0.1; t += 0.62) { don(X, t, 0.7, true); don(X, t + 0.16, 0.4, false); }
  don(X, tL1, 1, true);
  // luta (parte 2, mais intensa)
  groove(X, tL1 + 0.25, tUlt, 1.2);
  // preparação dos golpes finais
  roll(X, tUlt, tClash - 0.05);
  drone(X, tUlt, tClash, 0.1, [NOTE.D2, NOTE.A2]);
  swell(X, tClash - 0.9, tClash, 0.18);
  // choque: quase silêncio musical
  drone(X, tClash, tHit + 0.4, 0.13, [NOTE.D2, NOTE.A2, NOTE.E3]);
  don(X, tClash, 1.1, true);
  for (let t = tClash + 0.5; t < tHit - 0.2; t += 0.75) don(X, t, 0.55, true);
  // desfecho
  don(X, tHit, 1.2, true);
  gong(X, tHit + 0.02, 0.9, X.mus, 82);
  const mel = [[0.8, NOTE.A4, 0], [1.25, NOTE.B4, 0], [1.7, NOTE.D5, 2], [2.4, NOTE.E5, 0], [2.85, NOTE.D5, 0], [3.3, NOTE.B4, 0], [3.9, NOTE.A4, 0], [4.6, NOTE.D5, 0]];
  for (const [dt, f, bend] of mel) {
    const t = tHit + dt;
    if (t < D - 0.3) pluck(X, t, f, 0.17, { dur: 2.6, bend, vib: 1, pan: 0.15 });
  }
  pluck(X, tHit + 0.8, NOTE.D3, 0.2, { dur: 3, bright: 0.3 });
  pluck(X, tHit + 3.3, NOTE.A2 * 2, 0.18, { dur: 3, bright: 0.3 });
  drone(X, tHit + 0.3, D, 0.08, [NOTE.D2, NOTE.A2]);
  gong(X, tSal, 0.45, X.mus, 98);
  // saída suave da música no fim
  X.musOut.gain.setValueAtTime(1, D - 1.2);
  X.musOut.gain.linearRampToValueAtTime(0.0001, D - 0.05);
}

async function renderAudio() {
  const SR = 48000, D = CFG.DURATION;
  const ac = new OfflineAudioContext(2, Math.round(SR * D), SR);
  const R = rng(9001);
  const nb = ac.createBuffer(1, SR * 3, SR);
  { const d = nb.getChannelData(0); for (let i = 0; i < d.length; i++) d[i] = R() * 2 - 1; }
  const bb = ac.createBuffer(1, SR * 3, SR);
  { const d = bb.getChannelData(0); let l = 0; for (let i = 0; i < d.length; i++) { l = (l + 0.02 * (R() * 2 - 1)) / 1.02; d[i] = l * 3.5; } }
  const master = ac.createGain(); master.gain.value = 0.8;
  const comp = ac.createDynamicsCompressor();
  comp.threshold.value = -14; comp.knee.value = 8; comp.ratio.value = 5; comp.attack.value = 0.002; comp.release.value = 0.22;
  master.connect(comp); comp.connect(ac.destination);
  const rev = ac.createConvolver(); rev.buffer = makeIR(ac, 2.4, R);
  const revOut = ac.createGain(); revOut.gain.value = 0.3;
  rev.connect(revOut); revOut.connect(master);
  const sfx = ac.createGain(); sfx.gain.value = 0.9; sfx.connect(master);
  const sSend = ac.createGain(); sSend.gain.value = 0.22; sfx.connect(sSend); sSend.connect(rev);
  const mus = ac.createGain(); mus.gain.value = 0.5;
  const musOut = ac.createGain(); musOut.gain.value = 1;
  mus.connect(musOut); musOut.connect(master);
  const mSend = ac.createGain(); mSend.gain.value = 0.38; musOut.connect(mSend); mSend.connect(rev);
  const X = { ac, SR, R, nb, bb, sfx, mus, musOut, ks: {} };
  for (const s of SCENE.sounds) {
    const fn = SFX[s.name];
    if (!fn) continue;
    const t = SCENE.warp.real(s.t);
    if (t >= D - 0.05) continue;
    fn(X, Math.max(0.001, t), s.gain || 1, s);
  }
  music(X);
  const buf = await ac.startRendering();
  const L = buf.getChannelData(0), Rr = buf.getChannelData(1);
  let peak = 0;
  for (let i = 0; i < L.length; i++) peak = Math.max(peak, Math.abs(L[i]), Math.abs(Rr[i]));
  const norm = peak > 0.98 ? 0.98 / peak : 1;
  const pcm = new Int16Array(L.length * 2);
  for (let i = 0; i < L.length; i++) {
    pcm[i * 2] = Math.round(clamp(L[i] * norm, -1, 1) * 32767);
    pcm[i * 2 + 1] = Math.round(clamp(Rr[i] * norm, -1, 1) * 32767);
  }
  const bytes = new Uint8Array(pcm.buffer);
  let bin = '';
  for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  return { sr: SR, peak, b64: btoa(bin) };
}
