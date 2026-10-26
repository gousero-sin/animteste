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
uniform sampler2D uAux;     // r garment coverage, g shade (normalised skin luminance), b head soft, a hands
uniform sampler2D uNrm;     // r nx, g ny, b alpha
uniform sampler2D uAux2;    // r limbFront (forearm / hand in front of the torso), g hands wide, b blur
uniform vec2 uSh0, uSh1, uEl0, uEl1, uWr0, uWr1;  // 0 = R (frame-left) 1 = L (frame-right)
uniform vec4 uBoneW;        // frame confidences: upperR, foreR, upperL, foreL
uniform vec4 uArmDet;       // detail (seam / groove) visibility per bone
uniform vec4 uCuff0, uCuff1;    // cuff line centre xy + forearm dir (toward the hand) per hand
uniform vec4 uCuffWR;       // weights (x, y) and half widths (z, w) of the two cuffs
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
// person grade of the current environment (applied by the compositor after us): exposure,
// contrast, saturation, tint, ambient.  Used to cancel the additive ambient lift on black.
uniform vec4 uGrade;        // exposure, contrast, saturation, ambient compensation amount
uniform vec3 uTint, uAmb;

const vec2 SH0R = vec2(-0.50, 0.00), EL0R = vec2(-0.60, 0.88), WR0R = vec2(-0.64, 1.66);
const vec2 SH1R = vec2( 0.50, 0.00), EL1R = vec2( 0.60, 0.88), WR1R = vec2( 0.64, 1.66);
const float STERN_V = 0.56;                     // assembly seed = sternum node
const float D_SH = 0.80, L_UA = 0.88, L_FA = 0.80;

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

vec3 gradeNoAmb(vec3 x){
  x = max(x * uGrade.x, 0.0);
  x = 0.18 * pow(x / 0.18, vec3(uGrade.y));
  float l = dot(x, vec3(0.2126, 0.7152, 0.0722));
  x = l + (x - l) * uGrade.z;
  return x * uTint; }
vec3 gradeInvNoAmb(vec3 y){
  y = max(y / uTint, 0.0);
  float l = dot(y, vec3(0.2126, 0.7152, 0.0722));
  y = max(l + (y - l) / max(uGrade.z, 0.05), 0.0);
  y = 0.18 * pow(y / 0.18, vec3(1.0 / uGrade.y));
  return y / max(uGrade.x, 1e-3); }

// cuff at the real sleeve edge: s = signed distance along the forearm from the cuff line
// (negative = sleeve side), lat = distance across
void cuffTerms(vec2 p, vec4 cf, float wgt, float hw, inout float ring, inout float lip, inout float ao){
  if (wgt < 0.01) return;
  vec2 d = cf.zw; vec2 n = perp(d);
  float s = dot(p - cf.xy, d), lat = abs(dot(p - cf.xy, n));
  float onW = 1.0 - smoothstep(hw * 0.95, hw * 1.25, lat);
  ring += exp(-pow((s + 4.0) / 1.15, 2.0)) * onW * wgt;
  lip += exp(-pow((s + 6.5) / 2.2, 2.0)) * onW * wgt;
  ao += exp(-pow((s + 11.0) / 2.6, 2.0)) * onW * wgt;
}

void main(){
  vec2 uv = vUV;
  vec2 p = fpx();
  vec3 fg = texture(uFg, uv).rgb;
  vec4 aux = texture(uAux, uv);
  vec3 nt = texture(uNrm, uv).rgb;
  vec4 aux2 = texture(uAux2, uv);
  float cov = aux.r, shadeIn = aux.g, headS = aux.b, handS = aux.a, alpha = nt.b;
  float limbF = aux2.r, handW = aux2.g, blurW = aux2.b;
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
  // a forearm / hand in front of the torso owns its pixels (no torso frame there)
  float wt = exp(-0.5 * pow(max(dsp - 0.36, 0.0) / 0.09, 2.0)) * (1.0 - 0.97 * limbF) + 1e-3;
  float w0 = uBoneW.x * exp(-0.5 * pow(max(d0 - 0.04, 0.0) / 0.075, 2.0));
  float w1 = uBoneW.y * exp(-0.5 * pow(max(d1 - 0.03, 0.0) / 0.07, 2.0));
  float w2 = uBoneW.z * exp(-0.5 * pow(max(d2 - 0.04, 0.0) / 0.075, 2.0));
  float w3 = uBoneW.w * exp(-0.5 * pow(max(d3 - 0.03, 0.0) / 0.07, 2.0));
  float ws = wt + w0 + w1 + w2 + w3;
  vec2 q = (wt * vec2(tu, tv) + w0 * q0 + w1 * q1 + w2 * q2 + w3 * q3) / ws;
  float fT = wt / ws, s0 = w0 / ws, s1 = w1 / ws, s2 = w2 / ws, s3 = w3 / ws;
  // pattern visibility: fades where several bone frames blend (no lattice tearing)
  float pv = smoothstep(0.72, 0.94, max(fT, max(max(s0, s1), max(s2, s3))));
  float torsoOn = fT * (1.0 - limbF);

  // ---------------- skeleton-geodesic distance from the sternum node (assembly + pulses)
  // curved, noisy top barrier: the front climbs the trapezius before the clavicle line
  float bar = max(0.0, 0.03 + 0.10 * tu * tu - tv + 0.035 * snoise(vec3(tu * 5.0, 1.3, 0.0)));
  float dT = length(vec2(tu, (tv - STERN_V) * 1.15)) + bar * 2.2;
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
    float plate = smoothstep(ft - 0.004, ft + 0.004, tv) * smoothstep(fc + 0.004, fc - 0.004, tv) * onP;
    armour = max(armour, plate);
    float inTop = (tv - ft) * px, inChev = (fc - tv) * px;
    bevel += exp(-pow((inTop - 3.0) / 1.6, 2.0)) * onP * step(0.06, atu) * 1.0;
    bevel -= exp(-pow((inTop + 2.5) / 2.0, 2.0)) * onP * step(0.06, atu) * 0.6;
    bevel += exp(-pow((inChev - 3.0) / 1.6, 2.0)) * onP * 0.5;
    emi += glowL(dChev, 1.25) * onP * smoothstep(0.035, 0.11, atu) * (1.0 - smoothstep(0.36, 0.43, atu) * 0.7);
    groove = max(groove, lineAA(atu * px, 0.8) * step(ft - 0.01, tv) * step(0.0, tv));
    float abz = step(fc, tv) * (1.0 - smoothstep(0.29, 0.31, atu));
    armour = max(armour, abz);
    for (int k = 0; k < 4; k++){
      float y = 0.84 + 0.22 * float(k) + 0.05 * atu;
      groove = max(groove, lineAA(abs(tv - y) * px, 0.9) * abz);
    }
    float dS = abs(atu - 0.30 - 0.02 * max(tv - 0.6, 0.0)) * px;
    groove = max(groove, lineAA(dS, 1.0) * step(fc, tv));
    // sternum node
    vec2 sn = vec2(tu, tv - 0.56) / 0.042;
    float shd = hexD(sn);
    emi += glowL(abs(shd - 0.5) * 0.042 * px, 0.9) * 1.1 + exp(-shd * shd * 10.0) * 0.8;
    groove = max(groove, lineAA(abs(shd - 0.66) * 0.042 * px, 0.7));
  }
  groove *= torsoOn; emi *= torsoOn; armour *= torsoOn; bevel *= torsoOn;
  // arms (R: outer = +b, L: outer = -b); details of low-confidence / foreshortened limbs fade
  float eA = 0.0, gA = 0.0, aA = 0.0, bA = 0.0;
  {
    float c0 = s0 * uArmDet.x, c1 = s1 * uArmDet.y, c2 = s2 * uArmDet.z, c3 = s3 * uArmDet.w;
    float dCap0 = (a0 - 0.30) * L_UA * px, dCap2 = (a2 - 0.30) * L_UA * px;
    gA = max(gA, lineAA(abs(dCap0), 1.1) * c0); gA = max(gA, lineAA(abs(dCap2), 1.1) * c2);
    aA = max(aA, smoothstep(0.31, 0.29, a0) * c0); aA = max(aA, smoothstep(0.31, 0.29, a2) * c2);
    bA += exp(-pow((-dCap0 - 3.0) / 1.6, 2.0)) * c0 + exp(-pow((-dCap2 - 3.0) / 1.6, 2.0)) * c2;
    float dO0 = abs(b0 - 0.07) * px, dO2 = abs(b2 + 0.07) * px;
    float on0 = smoothstep(0.33, 0.42, a0) * (1.0 - smoothstep(0.82, 0.92, a0));
    float on2 = smoothstep(0.33, 0.42, a2) * (1.0 - smoothstep(0.82, 0.92, a2));
    eA += (glowL(dO0, 1.0) * on0 * c0 + glowL(dO2, 1.0) * on2 * c2) * 0.6;
    gA = max(gA, lineAA(dO0, 0.8) * on0 * c0); gA = max(gA, lineAA(dO2, 0.8) * on2 * c2);
    // forearm bracer (a > 0.5) + elbow joint line
    aA = max(aA, smoothstep(0.49, 0.51, a1) * c1 * step(a1, 1.05)); aA = max(aA, smoothstep(0.49, 0.51, a3) * c3 * step(a3, 1.05));
    gA = max(gA, lineAA(abs(a1 - 0.50) * L_FA * px, 1.1) * c1); gA = max(gA, lineAA(abs(a3 - 0.50) * L_FA * px, 1.1) * c3);
    gA = max(gA, lineAA(abs(a1 - 0.02) * L_FA * px, 1.0) * c1); gA = max(gA, lineAA(abs(a3 - 0.02) * L_FA * px, 1.0) * c3);
  }
  groove = max(groove, gA); emi += eA; armour = max(armour, aA); bevel += bA;

  // ---------------- collar lip (garment edge under the jaw) + edge occlusion
  vec2 fr = p - uFaceC;
  float below = dot(fr, -uFaceUp) / uFH;
  float nearNeck = smoothstep(0.05, 0.30, below) * (1.0 - smoothstep(1.0, 1.5, below));
  float lip = smoothstep(0.03, 0.16, headS) * (1.0 - smoothstep(0.32, 0.55, headS)) * nearNeck;
  float collarLine = exp(-pow((headS - 0.075) / 0.03, 2.0)) * nearNeck;
  float aoEdge = smoothstep(0.08, 0.6, headS) * nearNeck * 0.6;
  // cuffs: geometric, at the real sleeve edge (the flat wrist cut of the hand mask)
  float cRing = 0.0, cLip = 0.0, cAo = 0.0;
  cuffTerms(p, uCuff0, uCuffWR.x, uCuffWR.z, cRing, cLip, cAo);
  cuffTerms(p, uCuff1, uCuffWR.y, uCuffWR.w, cRing, cLip, cAo);
  cRing = min(cRing, 1.0); cLip = min(cLip, 1.0); cAo = min(cAo, 1.0);

  // ---------------- shading (original skin luminance drives the form)
  float shade = shadeIn;                          // ~1 = median body luminance
  float shS = min(shade, 2.0);                    // sheen driver (no glitter from blown skin)
  float lit = clamp(sqrt(shS), 0.0, 1.4);
  float hiL = smoothstep(0.55, 1.9, shS);         // where light hits: texture shows there

  // ---------------- micro texture (rest-space anchored, analytically antialiased)
  vec2 qm = q * 36.0;
  vec4 hc = hexCell(qm);
  float hd = hexD(hc.xy);
  float cpp = max(fwidth(qm.x), fwidth(qm.y));    // cells per pixel
  float fadeM = (1.0 - smoothstep(0.10, 0.22, cpp)) * pv;
  float fw = max(fwidth(hd), 1e-4);
  float hgroove = smoothstep(0.43 - fw, 0.46 + fw, hd) * fadeM;
  float cellR = hash21(hc.zw);
  vec2 wv = q * 62.0;
  float wcpp = max(fwidth(wv.x), fwidth(wv.y));
  float weave = sin((wv.x + wv.y) * 3.14159) * sin((wv.x - wv.y) * 1.5708 + floor(wv.x + wv.y) * 1.5708);
  weave *= (1.0 - smoothstep(0.12, 0.28, wcpp)) * pv;   // [-1, 1] -> fades to 0 when aliased

  vec3 albA = vec3(0.0105, 0.0112, 0.0128), albF = vec3(0.0125, 0.0128, 0.0138);
  vec3 alb = mix(albF, albA * (1.0 - 0.18 * hgroove), armour);
  vec3 col = alb * (0.25 + 0.95 * shade);
  float sat = smoothstep(0.75, 2.0, shS);
  vec3 Nm = N;
  Nm.xy += (vec2(hash21(hc.zw + 3.1), hash21(hc.zw + 7.7)) - 0.5) * 0.08 * (1.0 - sat) * armour * fadeM;
  Nm = normalize(Nm);
  vec3 Lk = normalize(vec3(-0.30, -0.80, 0.55));
  vec3 H = normalize(Lk + vec3(0.0, 0.0, 1.0));
  float ndh = max(dot(Nm, H), 0.0);
  float cellv = 1.0 + (cellR - 0.5) * 0.12 * fadeM;
  float texF = 1.0 + 0.22 * weave * hiL;          // fabric weave only visible in the light
  float sheenA = (sat * 0.060 + pow(ndh, 36.0) * 0.07 * lit) * cellv * (1.0 - 0.55 * hgroove * hiL);
  float sheenF = (sat * 0.040 + pow(ndh, 12.0) * 0.016 * lit) * texF;
  float fres = pow(1.0 - N.z, 3.0) * 0.018 * (0.6 + 0.4 * shS);
  col += vec3(0.86, 0.93, 1.05) * mix(sheenF, sheenA, armour) + vec3(0.55, 0.65, 0.85) * fres;
  col *= 1.0 - 0.80 * groove;
  col += vec3(0.80, 0.90, 1.0) * max(bevel, 0.0) * 0.020 * lit;
  col *= 1.0 + min(bevel, 0.0) * 0.5;
  col *= 1.0 - 0.55 * aoEdge * (1.0 - lip);
  col += vec3(0.85, 0.92, 1.0) * lip * (0.010 + 0.030 * lit);
  // cuff: raised lip catching light, occlusion groove behind it
  col *= 1.0 - 0.55 * cAo;
  col += vec3(0.85, 0.92, 1.0) * cLip * (0.012 + 0.030 * lit);

  // ---------------- sheen sweep: a narrow specular glint travelling over the curved body
  float sweepE = 0.0;
  if (uSweep > -0.5){
    float sq = (tu * 0.6 + tv * 0.8) - uSweep + 0.10 * (N.x * 0.6 + N.y * 0.8);   // bends with the form
    float g = exp(-sq * sq / 0.0030);
    float curv = 0.45 + 0.55 * pow(1.0 - N.z, 0.6) + 0.35 * armour;
    col += vec3(0.75, 0.88, 1.0) * g * curv * 0.10 * (0.4 + lit);
    sweepE = g * curv * (0.25 + 0.5 * armour) * (0.5 + 0.5 * lit);
  }

  // ---------------- cancel the environment's additive ambient on the black material
  if (uGrade.w > 0.0){
    vec3 gcol = gradeNoAmb(col);
    gcol = max(gcol - uAmb * uGrade.w, gcol * 0.25);
    col = gradeInvNoAmb(gcol);
  }

  // ---------------- emissive seams (+ travelling energy pulse); never over a hand
  float pulse = 0.0;
  if (uPulse > -0.5){
    float pd = dist - uPulse;
    pulse = exp(-pd * pd / 0.010) * 2.6;
  }
  float handGate = 1.0 - smoothstep(0.15, 0.35, handS);
  float seamE = (emi * (uSeam * uFlick + pulse) + collarLine * (0.40 * uSeam * uFlick + 0.35 * pulse)
                 + cRing * (0.55 * uSeam * uFlick + 0.4 * pulse)) * handGate;

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
    float bfw = max(fwidth(bhd), 1e-4);
    reveal = (x > 0.10) ? 1.0 : smoothstep(tileR + bfw, tileR - bfw, bhd);
    float env = smoothstep(-0.035, 0.0, x) * exp(-max(x, 0.0) / 0.085);
    float border = exp(-pow((bhd - 0.485) / 0.016, 2.0));
    float tw = 0.6 + 0.8 * step(0.3, hash21(bc.zw + floor(uT * 30.0) * 0.37));
    fxE += border * env * (1.0 + 1.4 * bj) * tw;
    fxE += border * 0.18 * smoothstep(0.0, 0.05, x) * exp(-max(x, 0.0) / 0.45);   // residual cool glow
    fxE += exp(-pow((bhd - tileR) / 0.018, 2.0)) * (1.0 - c) * step(0.0, x) * 2.6;
    fxE += exp(-max(x - 0.10, 0.0) / 0.035) * step(0.10, x) * 0.35 * (0.5 + bj);
    fxE += exp(-x * x / 0.0005) * 1.1;
    // nanite scaffold crawling ahead: flickering micro-hex wireframe on the skin
    float ahead = smoothstep(-0.20, 0.0, x) * step(x, 0.0);
    vec4 hc2 = hexCell(q * 40.0);
    float hd2 = hexD(hc2.xy);
    float fl = step(0.45, hash21(hc2.zw + floor(uT * 24.0)));
    fxE += ahead * ahead * smoothstep(0.40, 0.47, hd2) * (0.25 + 0.9 * fl) * 0.8;
    fxE *= uFX * handGate;
    seamE *= smoothstep(0.10, 0.35, x);
  }

  float k = cov * reveal;
  vec3 outc = mix(fg, col, k);
  float E = (seamE * k + fxE * cov + sweepE * k) * alpha;
  fragColor = vec4(outc, E);
}
'''
