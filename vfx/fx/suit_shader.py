"""GLSL source for the suit material + nanotech materialisation (used by vfx/fx/suit.py).

Coordinate systems (all evaluated per pixel from bone uniforms, nothing swims):
  torso   (tu, tv): origin = shoulder midpoint, tu along R->L shoulder, tv down, unit = sw
  bones   (a, b)  : a = 0..1 along shoulder->elbow / elbow->wrist, b = across in sw units
  rest q          : 2D linear-blend-skinning of the bone frames into a canonical rest pose,
                    used for the micro texture / assembly cells (moves rigidly with each limb)

Output: rgb = final person colour (skin where not covered), a = emissive intensity (scalar).
"""

FRAG = r'''
uniform sampler2D uFg;      // linear rgb crop
uniform sampler2D uAux;     // r garment coverage, g Ls (smoothed lum), b head soft, a hands soft
uniform sampler2D uNrm;     // r nx, g ny, b alpha
uniform vec2 uSh0, uSh1, uEl0, uEl1, uWr0, uWr1;  // 0 = R (frame-left) 1 = L (frame-right)
uniform vec4 uBoneW;        // confidences: upperR, foreR, upperL, foreL
uniform float uSW;
uniform vec2 uFaceC, uFaceUp;
uniform float uFH;
uniform float uFront;       // materialisation front radius (sw units)
uniform float uFX;          // materialisation fx strength
uniform float uSeam;        // steady seam emission level
uniform float uPulse;       // travelling pulse position (sw units along the skeleton), <0 = off
uniform float uSweep;       // sheen sweep position, < -0.5 = off
uniform float uFlick;       // seam power-on flicker multiplier
uniform float uT;

const vec2 SH0R = vec2(-0.50, 0.00), EL0R = vec2(-0.60, 0.88), WR0R = vec2(-0.64, 1.66);
const vec2 SH1R = vec2( 0.50, 0.00), EL1R = vec2( 0.60, 0.88), WR1R = vec2( 0.64, 1.66);
const float STERN_V = 0.36;
const float D_SH = 0.62, L_UA = 0.88, L_FA = 0.80;

vec2 perp(vec2 v){ return vec2(-v.y, v.x); }
float segd(vec2 p, vec2 a, vec2 b, out float h){
  vec2 pa = p - a, ba = b - a; h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-3), 0.0, 1.0);
  return length(pa - ba * h); }
vec2 boneRest(vec2 p, vec2 A, vec2 B, vec2 Ar, vec2 Br, out float a, out float b){
  vec2 d = B - A; float Ld = length(d); vec2 dir = d / max(Ld, 1e-3);
  float L = max(Ld, 0.30 * uSW);
  a = dot(p - A, dir) / L; b = dot(p - A, perp(dir)) / uSW;
  vec2 dr = Br - Ar; return Ar + a * dr + b * perp(normalize(dr)); }

// hex grid, cell radius (centre->edge) = 0.5 in grid units
vec4 hexCell(vec2 p){
  const vec2 s = vec2(1.0, 1.7320508);
  vec4 hC = floor(vec4(p, p - vec2(0.5, 1.0)) / s.xyxy) + 0.5;
  vec4 h = vec4(p - hC.xy * s, p - (hC.zw + 0.5) * s);
  return dot(h.xy, h.xy) < dot(h.zw, h.zw) ? vec4(h.xy, hC.xy) : vec4(h.zw, hC.zw + 0.5); }
float hexD(vec2 p){ p = abs(p); return max(dot(p, vec2(0.5, 0.8660254)), p.x); }   // pointy-top, edge = 0.5

float lineAA(float dpx, float w){ return 1.0 - smoothstep(w, w + 1.25, dpx); }
float glowL(float dpx, float w){ return exp(-dpx * dpx / (w * w)) + 0.20 * exp(-dpx / (3.0 * w)); }
// distance (px) to curve tv = f(tu) given f and slope
float curveD(float tv, float f, float fp){ return abs(tv - f) / sqrt(1.0 + fp * fp) * uSW; }

void main(){
  vec2 uv = vUV;
  vec2 p = fpx();
  vec3 fg = texture(uFg, uv).rgb;
  vec4 aux = texture(uAux, uv);
  vec3 nt = texture(uNrm, uv).rgb;
  float cov = aux.r, Ls = aux.g, headS = aux.b, handS = aux.a, alpha = nt.b;
  if (alpha < 0.003 || cov < 0.003) { fragColor = vec4(fg, 0.0); return; }
  vec3 N = vec3(nt.rg, sqrt(max(1.0 - dot(nt.rg, nt.rg), 0.02)));

  // ---------------- torso frame
  vec2 o = 0.5 * (uSh0 + uSh1);
  vec2 ex = normalize(uSh1 - uSh0), ey = perp(ex);
  float tu = dot(p - o, ex) / uSW, tv = dot(p - o, ey) / uSW;
  float hh;
  float dsp = segd(p, o - ey * 0.12 * uSW, o + ey * 1.7 * uSW, hh) / uSW;
  // ---------------- bones
  float a0, b0, a1, b1, a2, b2, a3, b3;
  vec2 q0 = boneRest(p, uSh0, uEl0, SH0R, EL0R, a0, b0);
  vec2 q1 = boneRest(p, uEl0, uWr0, EL0R, WR0R, a1, b1);
  vec2 q2 = boneRest(p, uSh1, uEl1, SH1R, EL1R, a2, b2);
  vec2 q3 = boneRest(p, uEl1, uWr1, EL1R, WR1R, a3, b3);
  float h0, h1, h2, h3;
  float d0 = segd(p, uSh0, uEl0, h0) / uSW, d1 = segd(p, uEl0, uWr0, h1) / uSW;
  float d2 = segd(p, uSh1, uEl1, h2) / uSW, d3 = segd(p, uEl1, uWr1, h3) / uSW;
  float wt = exp(-0.5 * pow(max(dsp - 0.36, 0.0) / 0.09, 2.0)) + 1e-3;
  float w0 = uBoneW.x * exp(-0.5 * pow(max(d0 - 0.04, 0.0) / 0.075, 2.0));
  float w1 = uBoneW.y * exp(-0.5 * pow(max(d1 - 0.03, 0.0) / 0.07, 2.0));
  float w2 = uBoneW.z * exp(-0.5 * pow(max(d2 - 0.04, 0.0) / 0.075, 2.0));
  float w3 = uBoneW.w * exp(-0.5 * pow(max(d3 - 0.03, 0.0) / 0.07, 2.0));
  float ws = wt + w0 + w1 + w2 + w3;
  vec2 q = (wt * vec2(tu, tv) + w0 * q0 + w1 * q1 + w2 * q2 + w3 * q3) / ws;
  float fT = wt / ws, s0 = w0 / ws, s1 = w1 / ws, s2 = w2 / ws, s3 = w3 / ws;
  // pattern visibility: fades where several bone frames blend (no lattice tearing)
  float pv = smoothstep(0.72, 0.94, max(fT, max(max(s0, s1), max(s2, s3))));

  // ---------------- skeleton-geodesic distance from the sternum (assembly + pulses)
  float dT = length(vec2(tu, (tv - STERN_V) * 1.15)) + max(0.0, 0.02 - tv) * 3.0;
  float dU0 = D_SH + abs(a0) * L_UA + abs(b0) * 0.6, dU1 = D_SH + abs(a2) * L_UA + abs(b2) * 0.6;
  float dF0 = D_SH + L_UA + max(a1, 0.0) * L_FA + abs(b1) * 0.6, dF1 = D_SH + L_UA + max(a3, 0.0) * L_FA + abs(b3) * 0.6;
  float dist = (wt * dT + w0 * dU0 + w1 * dF0 + w2 * dU1 + w3 * dF1) / ws;
  float distN = dist + 0.11 * snoise(vec3(q * 2.2, 0.7)) + 0.045 * snoise(vec3(q * 6.5, 2.1));

  // ---------------- panel layout: grooves, emissive accents, armour regions, bevels
  float px = uSW;
  float groove = 0.0, emi = 0.0, armour = 0.0, bevel = 0.0;
  float atu = abs(tu);
  {
    // breastplate: top edge sweeps from the collar out/down to the armpit, bottom = chevron
    float xt = clamp((atu - 0.09) / 0.34, 0.0, 1.0);
    float ft = -0.04 + 0.42 * pow(xt, 1.35);
    float ftp = (atu > 0.09 && atu < 0.43) ? 0.42 * 1.35 * pow(max(xt, 1e-3), 0.35) / 0.34 : 0.0;
    float xc = atu / 0.43;
    float fc = 0.62 - 0.24 * pow(min(xc, 1.1), 1.5);
    float fcp = -0.24 * 1.5 * pow(max(min(xc, 1.1), 1e-3), 0.5) / 0.43;
    float onP = 1.0 - smoothstep(0.42, 0.47, atu);
    float dTop = curveD(tv, ft, ftp), dChev = curveD(tv, fc, fcp);
    groove = max(groove, lineAA(dTop, 1.1) * onP * step(0.06, atu));
    groove = max(groove, lineAA(dChev, 1.1) * onP);
    float plate = step(ft, tv) * step(tv, fc) * onP;
    armour = max(armour, plate);
    // raised lip under the top edge (catches the top light) and above the chevron
    float inTop = (tv - ft) * px, inChev = (fc - tv) * px;
    bevel += exp(-pow((inTop - 3.0) / 1.6, 2.0)) * onP * step(0.06, atu) * 1.0;
    bevel -= exp(-pow((inTop + 2.5) / 2.0, 2.0)) * onP * step(0.06, atu) * 0.6;
    bevel += exp(-pow((inChev - 3.0) / 1.6, 2.0)) * onP * 0.5;
    // chevron accent light
    emi += glowL(dChev, 1.25) * onP * smoothstep(0.035, 0.11, atu) * (1.0 - smoothstep(0.36, 0.43, atu) * 0.7);
    // centre split
    groove = max(groove, lineAA(atu * px, 0.8) * step(ft - 0.01, tv) * step(0.0, tv));
    // abdomen plates
    float abz = step(fc, tv) * (1.0 - smoothstep(0.29, 0.31, atu));
    armour = max(armour, abz);
    for (int k = 0; k < 4; k++){
      float y = 0.84 + 0.22 * float(k) + 0.05 * atu;
      groove = max(groove, lineAA(abs(tv - y) * px, 0.9) * abz);
    }
    float dS = abs(atu - 0.30 - 0.02 * max(tv - 0.6, 0.0)) * px;
    groove = max(groove, lineAA(dS, 1.0) * step(fc, tv));
    // collar base ring (neck opening) + collar ribs
    // sternum node
    vec2 sn = vec2(tu, tv - 0.56) / 0.042;
    float shd = hexD(sn);
    emi += glowL(abs(shd - 0.5) * 0.042 * px, 0.9) * 1.1 + exp(-shd * shd * 10.0) * 0.8;
    groove = max(groove, lineAA(abs(shd - 0.66) * 0.042 * px, 0.7));
  }
  groove *= fT; emi *= fT; armour *= fT; bevel *= fT;
  // arms (R: outer = +b, L: outer = -b)
  float eA = 0.0, gA = 0.0, aA = 0.0, bA = 0.0;
  {
    // details of low-confidence (hidden / hallucinated) limbs fade out
    float c0 = smoothstep(0.35, 0.80, uBoneW.x), c1 = smoothstep(0.35, 0.80, uBoneW.y);
    float c2 = smoothstep(0.35, 0.80, uBoneW.z), c3 = smoothstep(0.35, 0.80, uBoneW.w);
    s0 *= c0; s1 *= c1; s2 *= c2; s3 *= c3;
    float dCap0 = (a0 - 0.30) * L_UA * px, dCap2 = (a2 - 0.30) * L_UA * px;
    gA = max(gA, lineAA(abs(dCap0), 1.1) * s0); gA = max(gA, lineAA(abs(dCap2), 1.1) * s2);
    aA = max(aA, step(a0, 0.30) * s0); aA = max(aA, step(a2, 0.30) * s2);
    bA += exp(-pow((-dCap0 - 3.0) / 1.6, 2.0)) * s0 + exp(-pow((-dCap2 - 3.0) / 1.6, 2.0)) * s2;
    float dO0 = abs(b0 - 0.07) * px, dO2 = abs(b2 + 0.07) * px;
    float on0 = smoothstep(0.33, 0.42, a0) * (1.0 - smoothstep(0.82, 0.92, a0));
    float on2 = smoothstep(0.33, 0.42, a2) * (1.0 - smoothstep(0.82, 0.92, a2));
    eA += (glowL(dO0, 1.0) * on0 * s0 + glowL(dO2, 1.0) * on2 * s2) * 0.6;
    gA = max(gA, lineAA(dO0, 0.8) * on0 * s0); gA = max(gA, lineAA(dO2, 0.8) * on2 * s2);
    // forearm bracers (a > 0.5) and cuff ring at the wrist
    aA = max(aA, step(0.50, a1) * s1); aA = max(aA, step(0.50, a3) * s3);
    gA = max(gA, lineAA(abs(a1 - 0.50) * L_FA * px, 1.1) * s1); gA = max(gA, lineAA(abs(a3 - 0.50) * L_FA * px, 1.1) * s3);
    gA = max(gA, lineAA(abs(a1 - 0.02) * L_FA * px, 1.0) * s1); gA = max(gA, lineAA(abs(a3 - 0.02) * L_FA * px, 1.0) * s3);
    float cu1 = abs(a1 - 0.90) * L_FA * px, cu3 = abs(a3 - 0.90) * L_FA * px;
    eA += (glowL(cu1, 1.0) * s1 + glowL(cu3, 1.0) * s3) * 0.75;
    gA = max(gA, lineAA(cu1, 0.9) * s1); gA = max(gA, lineAA(cu3, 0.9) * s3);
  }
  groove = max(groove, gA); emi += eA; armour = max(armour, aA); bevel += bA;

  // ---------------- collar lip (garment edge under the jaw) + edge occlusion
  vec2 fr = p - uFaceC;
  float below = dot(fr, -uFaceUp) / uFH;
  float nearNeck = smoothstep(0.05, 0.30, below) * (1.0 - smoothstep(1.0, 1.5, below));
  float lip = smoothstep(0.03, 0.16, headS) * (1.0 - smoothstep(0.32, 0.55, headS)) * nearNeck;
  float collarLine = exp(-pow((headS - 0.075) / 0.03, 2.0)) * nearNeck;
  float dw0 = length(p - uWr0) / uSW, dw1 = length(p - uWr1) / uSW;
  float nearW = 1.0 - smoothstep(0.10, 0.20, min(dw0, dw1));
  float cuffLip = smoothstep(0.04, 0.18, handS) * (1.0 - smoothstep(0.32, 0.55, handS)) * nearW;
  float aoEdge = smoothstep(0.08, 0.6, headS) * nearNeck * 0.6 + smoothstep(0.08, 0.6, handS) * 0.35;

  // ---------------- micro texture (rest-space anchored)
  vec4 hc = hexCell(q * 58.0);
  float hd = hexD(hc.xy);
  float hgroove = smoothstep(0.39, 0.47, hd) * pv;
  float cellR = hash21(hc.zw);
  vec2 wv = q * 95.0;
  float weave = 0.5 + 0.5 * sin((wv.x + wv.y) * 3.14159) * sin((wv.x - wv.y) * 1.5708 + floor(wv.x + wv.y) * 1.5708);
  weave = mix(0.5, weave, pv);

  // ---------------- shading (original skin luminance drives the form)
  float shade = clamp(Ls / 0.085, 0.0, 7.0);
  float lit = clamp(sqrt(shade) * 0.65, 0.0, 1.7);
  vec3 albA = vec3(0.0080, 0.0088, 0.0105), albF = vec3(0.0098, 0.0102, 0.0114);
  vec3 alb = mix(albF * (0.90 + 0.20 * weave), albA * (1.0 - 0.22 * hgroove), armour);
  vec3 col = alb * shade;
  vec3 Nm = N;
  Nm.xy += (vec2(hash21(hc.zw + 3.1), hash21(hc.zw + 7.7)) - 0.5) * 0.35 * armour * pv;
  Nm = normalize(Nm);
  vec3 Lk = normalize(vec3(-0.30, -0.80, 0.55));
  vec3 H = normalize(Lk + vec3(0.0, 0.0, 1.0));
  float ndh = max(dot(Nm, H), 0.0);
  float sat = pow(clamp(shade / 3.0, 0.0, 2.4), 1.9);
  float cellv = mix(1.0, 0.70 + 0.6 * cellR, pv);
  float sheenA = sat * 0.052 * cellv * (1.0 - 0.6 * hgroove) + pow(ndh, 40.0) * 0.10 * lit * cellv;
  float sheenF = sat * 0.027 * (0.85 + 0.3 * weave) + pow(ndh, 12.0) * 0.020 * lit;
  float fres = pow(1.0 - N.z, 3.0) * 0.020 * (0.6 + 0.4 * min(shade, 2.0));
  col += vec3(0.88, 0.94, 1.04) * mix(sheenF, sheenA, armour) + vec3(0.55, 0.65, 0.82) * fres;
  col *= 1.0 - 0.80 * groove;
  col += vec3(0.80, 0.90, 1.0) * max(bevel, 0.0) * 0.022 * lit;
  col *= 1.0 + min(bevel, 0.0) * 0.5;
  col *= 1.0 - 0.55 * aoEdge * (1.0 - lip);
  col += vec3(0.85, 0.92, 1.0) * (lip + cuffLip) * (0.010 + 0.035 * lit);
  if (uSweep > -0.5){
    float sw = (tu * 0.6 + tv * 0.8) - uSweep;
    col += vec3(0.75, 0.88, 1.0) * exp(-sw * sw / 0.012) * (0.06 + 0.20 * armour * (1.0 - hgroove)) * lit;
  }

  // ---------------- emissive seams (+ travelling energy pulse)
  float pulse = 0.0;
  if (uPulse > -0.5){
    float pd = dist - uPulse;
    pulse = exp(-pd * pd / 0.010) * 2.6;
  }
  float seamE = emi * (uSeam * uFlick + pulse) + collarLine * (0.40 * uSeam * uFlick + 0.35 * pulse);

  // ---------------- materialisation: hex tiles assemble behind a travelling front
  float reveal = 1.0, fxE = 0.0;
  if (uFX > 0.001){
    vec4 bc = hexCell(q * 15.0);
    float bj = hash21(bc.zw);
    float x = uFront - distN - (bj - 0.5) * 0.12;          // >0 : front has passed this cell
    float c = clamp(x / 0.10, 0.0, 1.0);
    float cm = c - 1.0;
    float sc = 1.0 + 2.2 * cm * cm * cm + 1.2 * cm * cm;      // ease-out-back (overshoot ~1.07)
    sc = (x <= 0.0) ? 0.0 : sc;
    float bhd = hexD(bc.xy);
    float tileR = 0.5 * sc;
    reveal = (x > 0.10) ? 1.0 : smoothstep(tileR + 0.012, tileR - 0.012, bhd);
    // hex border grid lights up as the front arrives, then cools down behind it
    float env = smoothstep(-0.035, 0.0, x) * exp(-max(x, 0.0) / 0.085);
    float border = exp(-pow((bhd - 0.485) / 0.016, 2.0));
    float tw = 0.6 + 0.8 * step(0.3, hash21(bc.zw + floor(uT * 30.0) * 0.37));
    fxE += border * env * (1.0 + 1.4 * bj) * tw;
    fxE += border * 0.18 * smoothstep(0.0, 0.05, x) * exp(-max(x, 0.0) / 0.45);   // residual cool glow
    // expanding tile edge (hot, thin) while the tile scales in
    fxE += exp(-pow((bhd - tileR) / 0.018, 2.0)) * (1.0 - c) * step(0.0, x) * 2.6;
    // short lock-in shimmer inside the tile
    fxE += exp(-max(x - 0.10, 0.0) / 0.035) * step(0.10, x) * 0.35 * (0.5 + bj);
    // thin hot leading line
    fxE += exp(-x * x / 0.0005) * 1.1;
    // nanite scaffold crawling ahead: flickering micro-hex wireframe on the skin
    float ahead = smoothstep(-0.20, 0.0, x) * step(x, 0.0);
    float fl = step(0.45, hash21(hc.zw + floor(uT * 24.0)));
    fxE += ahead * ahead * smoothstep(0.40, 0.47, hd) * (0.25 + 0.9 * fl) * 0.8;
    fxE *= uFX;
    seamE *= smoothstep(0.10, 0.35, x);
  }

  float k = cov * reveal;
  vec3 outc = mix(fg, col, k);
  float E = (seamE * k + fxE * cov) * alpha;
  fragColor = vec4(outc, E);
}
'''
