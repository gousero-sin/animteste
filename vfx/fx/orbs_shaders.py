"""GLSL sources for the orbs module (fire, ice, plasma, explosion, mist).

Conventions (see glfx.py): fpx() = full-frame pixel coords (x right, y DOWN).
All local object spaces below use y UP (we flip), units of the object radius.
"""

COMMON = r'''
// blackbody-ish emission ramp for flames, x ~ temperature 0..1.6 -> HDR linear colour
vec3 bbody(float x){
    x = max(x, 0.0);
    vec3 c = vec3(0.50, 0.045, 0.008) * smoothstep(0.02, 0.24, x);    // deep red
    c += vec3(1.00, 0.26, 0.025) * smoothstep(0.16, 0.50, x);          // orange
    c += vec3(0.75, 0.55, 0.12) * smoothstep(0.38, 0.78, x);           // yellow
    c += vec3(1.10, 1.05, 0.85) * smoothstep(0.72, 1.15, x);           // white hot
    c += vec3(1.6, 1.7, 1.9) * smoothstep(1.1, 1.6, x);                // blue-white core
    return c;
}
'''

# ------------------------------------------------------------------ volumetric fire
FIRE = COMMON + r'''
uniform vec2 uC;        // orb centre (frame px)
uniform float uR;       // orb radius (px)
uniform float uT;       // time (s)
uniform float uAmp;     // emission gain
uniform float uGrow;    // 0..1+ size / development
uniform vec2 uLean;     // flame lean (local units), opposite to motion
uniform float uSeed;
uniform float uCore;    // white-hot core strength
uniform float uPlume;   // height of the flame plume (local units, ~2.4)
uniform float uSmoke;   // smoke amount

float fireField(vec3 p, out float temp, out float smoke){
    float h = max(p.y, 0.0);
    // lean the plume (velocity drag + a slow sway)
    p.xz -= (uLean + vec2(0.20 * sin(uT * 1.7 + uSeed), 0.0)) * h * h * 0.22;
    // rising, vertically stretched noise coordinates
    vec3 q = p * vec3(1.55, 0.95, 1.55);
    q.y -= uT * 2.5;
    q += vec3(uSeed, uSeed * 0.37, 0.0);
    vec2 w = vec2(snoise(q * 0.6 + vec3(0.0, 0.0, uT * 0.7)),
                  snoise(q * 0.6 + vec3(5.2, 1.3, uT * 0.7)));
    q.xz += w * (0.45 + 0.35 * h);
    float n = fbm3(q * 1.35, 4);                // ~[-0.8, 0.8]
    // ball + upward stretched plume
    vec3 s = vec3(p.x * (1.0 + 0.12 * h), p.y > 0.0 ? p.y / uPlume * 1.25 : p.y * 1.02, p.z * (1.0 + 0.12 * h));
    float d = length(s) - 1.0;
    float amp = 0.32 + 0.95 * smoothstep(0.0, 1.6, h);
    float f = -d + n * amp;
    float dens = smoothstep(0.0, 0.22, f);
    // temperature: deeper = hotter, turbulent cells, cooling with height
    temp = 0.12 + 0.72 * clamp(f * 1.4, 0.0, 1.0) + 0.55 * n - 0.24 * h;
    // smoke above the flame tips
    float fs = -d + n * (amp + 0.4) + 0.30;
    smoke = smoothstep(0.0, 0.5, fs) * smoothstep(0.9, 2.8, h) * (1.0 - dens);
    return dens;
}

void main(){
    vec2 P = fpx() - uC;
    vec2 xy = vec2(P.x, -P.y) / max(uR, 1.0);     // y up
    float gscale = max(uGrow, 0.02);
    xy /= gscale;
    float zext = 1.45;
    float bx = abs(xy.x) - (1.6 + 0.35 * max(xy.y, 0.0));
    if (bx > 0.0 || xy.y < -1.6 || xy.y > uPlume * 1.35 + 0.9) { fragColor = vec4(0.0); return; }
    const int N = 24;
    float dz = 2.0 * zext / float(N);
    float jitter = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715)))) * dz;
    vec3 col = vec3(0.0);
    float Tr = 1.0;
    float smk = 0.0;
    for (int k = 0; k < N; k++){
        float z = zext - dz * float(k) - jitter;
        vec3 p = vec3(xy, z);
        float temp, smoke;
        float dens = fireField(p, temp, smoke);
        // white-hot churning core, seen through gaps in the outer flame
        float r2 = dot(p, p);
        float core = uCore * exp(-r2 * 3.2);
        temp += core * (0.85 + 0.45 * snoise(p * 3.3 + vec3(0.0, -uT * 1.8, uT * 0.9)));
        vec3 em = bbody(temp) * dens;
        col += Tr * em * dz;
        smk += Tr * smoke * dz;
        Tr *= exp(-dens * dz * 2.3);
        if (Tr < 0.02) break;
    }
    col *= uAmp;
    float a = clamp(smk * 0.6 * uSmoke, 0.0, 0.85);
    fragColor = vec4(col, a);
}
'''

# ------------------------------------------------------------------ hot glow / flash disc
GLOW = r'''
uniform vec2 uC; uniform float uR; uniform vec3 uCol; uniform float uPow;
void main(){
    vec2 P = fpx() - uC;
    float d = length(P) / max(uR, 1.0);
    float g = 1.0 / (1.0 + pow(d, uPow) * 6.0) - 1.0 / 7.0;
    fragColor = vec4(uCol * max(g, 0.0), 0.0);
}
'''

# ------------------------------------------------------------------ faceted refractive ice
ICE = r'''
#define NP 26
#define NS 10
uniform vec2 uC;          // centre (frame px)
uniform float uR;         // unit radius (px)
uniform float uSS;        // supersampling factor of the render target
uniform float uT;
uniform float uCore;      // core scale (0..1+)
uniform vec4 uPl[NP];     // core planes: xyz normal (object space), w offset
uniform vec4 uSh[NS];     // shard axis xyz, w = length
uniform vec4 uShB[NS];    // shard perpendicular xyz, w = radius
uniform vec3 uRx; uniform vec3 uRy; uniform vec3 uRz;   // rows of object rotation R
uniform sampler2D uScene; // linear scene behind (texture box)
uniform vec2 uTexOff; uniform vec2 uTexSize;
uniform vec3 uFireDir;    // direction toward the fire (view space), w/ intensity in length
uniform float uGlow;      // inner glow gain
uniform float uFlash;     // lock flash
uniform float uOpacity;

vec3 toObj(vec3 v){ return vec3(dot(uRx, v), dot(uRy, v), dot(uRz, v)); }
vec3 toView(vec3 v){ return uRx * v.x + uRy * v.y + uRz * v.z; }

// plane k of convex c: c = 0 core, c = 1..NS shards
vec4 plane(int c, int k){
    if (c == 0){ vec4 p = uPl[k]; return vec4(p.xyz, p.w * uCore); }
    vec4 A = uSh[c - 1]; vec4 B = uShB[c - 1];
    vec3 a = A.xyz; vec3 b1 = B.xyz; vec3 b2 = cross(a, b1);
    float rad = B.w, L = A.w;
    if (k < 6){ float th = float(k) * 1.0471976; vec3 n = cos(th) * b1 + sin(th) * b2; return vec4(n, rad); }
    if (k < 12){ float th = float(k - 6) * 1.0471976; vec3 n = cos(th) * b1 + sin(th) * b2;
        float phi = 0.62; vec3 m = normalize(n * cos(phi) + a * sin(phi)); return vec4(m, dot(m, a * L)); }
    return vec4(-a, -0.15);
}
int nplanes(int c){ return c == 0 ? NP : 13; }

bool hitConvex(int c, vec3 ro, vec3 rd, out float tN, out float tF, out vec3 nN){
    tN = -1e9; tF = 1e9; nN = vec3(0.0, 0.0, 1.0);
    int np = nplanes(c);
    for (int k = 0; k < NP; k++){
        if (k >= np) break;
        vec4 P = plane(c, k);
        float den = dot(P.xyz, rd);
        float dist = P.w - dot(P.xyz, ro);
        if (abs(den) < 1e-6){ if (dist < 0.0) return false; continue; }
        float t = dist / den;
        if (den < 0.0){ if (t > tN){ tN = t; nN = P.xyz; } }
        else { if (t < tF) tF = t; }
    }
    return tN < tF && tF > 0.0;
}
// exit of a ray starting inside convex c
float exitConvex(int c, vec3 ro, vec3 rd, out vec3 nF){
    float tF = 1e9; nF = rd;
    int np = nplanes(c);
    for (int k = 0; k < NP; k++){
        if (k >= np) break;
        vec4 P = plane(c, k);
        float den = dot(P.xyz, rd);
        if (den > 1e-6){ float t = (P.w - dot(P.xyz, ro)) / den; if (t < tF){ tF = t; nF = P.xyz; } }
    }
    return max(tF, 0.0);
}
float edgeDist(int c, vec3 p, vec3 nHit){
    float m = 1e9;
    int np = nplanes(c);
    for (int k = 0; k < NP; k++){
        if (k >= np) break;
        vec4 P = plane(c, k);
        float cs = dot(P.xyz, nHit);
        if (cs > 0.995) continue;
        float s = (P.w - dot(P.xyz, p)) / max(sqrt(1.0 - cs * cs), 0.15);
        m = min(m, s);
    }
    return m;
}
vec3 env(vec3 d){
    // dark studio with an overhead softbox, a cool side strip and the fire's warm glow
    vec3 c = vec3(0.006, 0.008, 0.012);
    c += vec3(1.6, 1.75, 2.0) * smoothstep(0.80, 0.96, d.y) * smoothstep(-0.2, 0.3, d.z);
    c += vec3(0.25, 0.45, 0.8) * smoothstep(0.85, 0.98, d.x) * 0.8;
    c += vec3(0.25, 0.45, 0.8) * smoothstep(0.88, 0.99, -d.x) * 0.4;
    float fi = length(uFireDir);
    if (fi > 1e-3){ vec3 fd = uFireDir / fi; c += vec3(2.6, 0.9, 0.22) * fi * pow(max(dot(d, fd), 0.0), 6.0); }
    return c;
}
vec3 sceneAt(vec3 pe, vec3 r2){
    // project the exit ray onto a backdrop plane behind the orb
    float bz = 1.4;
    vec2 s = pe.xy + r2.xy * ((pe.z + bz) / max(-r2.z, 0.25));
    vec2 pix = uC + vec2(s.x, -s.y) * uR;
    vec2 uv = (pix - uTexOff) / uTexSize;
    return texture(uScene, uv).rgb;
}

void main(){
    vec2 P = px() / uSS + uOffset - uC;
    vec2 xy = vec2(P.x, -P.y) / uR;
    if (dot(xy, xy) > 6.5){ fragColor = vec4(0.0); return; }
    vec3 roV = vec3(xy, 4.0), rdV = vec3(0.0, 0.0, -1.0);
    vec3 ro = toObj(roV), rd = toObj(rdV);
    float best = 1e9; int hc = -1; vec3 hn = vec3(0.0);
    for (int c = 0; c <= NS; c++){
        if (c > 0 && uSh[c - 1].w < 0.02) continue;
        if (c == 0 && uCore < 0.02) continue;
        float tN, tF; vec3 nN;
        if (hitConvex(c, ro, rd, tN, tF, nN) && tN < best){ best = tN; hc = c; hn = nN; }
    }
    if (hc < 0){ fragColor = vec4(0.0); return; }
    vec3 p = ro + rd * best;                         // object-space entry point
    vec3 N = hn;
    float cosi = clamp(-dot(rd, N), 0.0, 1.0);
    float F = 0.03 + 0.97 * pow(1.0 - cosi, 5.0);
    vec3 refl = reflect(rd, N);
    vec3 col = vec3(0.0);
    // --- refraction, per channel with dispersion
    vec3 iors = vec3(1.285, 1.31, 1.345);
    float pathL = 0.0;
    vec3 peG = p;
    for (int ch = 0; ch < 3; ch++){
        float ior = iors[ch];
        vec3 r1 = refract(rd, N, 1.0 / ior);
        vec3 nF;
        float tx = exitConvex(hc, p + r1 * 1e-3, r1, nF);
        vec3 pe = p + r1 * tx;
        vec3 r2 = refract(r1, -nF, ior);
        float tir = 0.0;
        if (dot(r2, r2) < 1e-4){ r2 = reflect(r1, -nF); tir = 1.0; }
        vec3 bg = sceneAt(toView(pe), toView(r2));
        bg = mix(bg, env(toView(r2)) * 0.6, tir * 0.7);
        col[ch] = bg[ch];
        if (ch == 1){ pathL = tx; peG = pe; }
    }
    vec3 sigma = vec3(0.55, 0.20, 0.06);
    vec3 absorb = exp(-sigma * pathL * 1.6);
    col *= absorb * vec3(0.80, 0.95, 1.05);
    // --- interior: glow, milky frost cloud, fracture planes
    vec3 glow = vec3(0.0); float milky = 0.0; float crack = 0.0;
    const int M = 6;
    for (int k = 0; k < M; k++){
        float f = (float(k) + 0.5) / float(M);
        vec3 q = mix(p, peG, f);
        float r2 = dot(q, q);
        glow += vec3(0.25, 0.60, 1.0) * exp(-r2 * 4.0) + vec3(0.8, 0.95, 1.0) * exp(-r2 * 30.0) * 0.9;
        float n = fbm3(q * 2.6 + vec3(0.0, 0.0, uT * 0.05), 3);
        milky += smoothstep(0.05, 0.55, n) * exp(-r2 * 1.2);
        float cr = abs(snoise(q * 1.05 + vec3(3.1, 0.0, 0.0)));
        crack += smoothstep(0.022, 0.0, cr) * exp(-r2 * 1.5);
    }
    float seg = pathL / float(M);
    glow *= seg * uGlow;
    milky *= seg;
    crack *= seg;
    vec3 inner = vec3(0.45, 0.75, 1.0) * uGlow;      // light inside the ice
    col = col * (1.0 - clamp(milky * 0.8, 0.0, 0.55)) + glow * 0.75 + inner * milky * 0.40 + vec3(0.7, 0.9, 1.2) * crack * 0.9 * uGlow;
    // --- surface: frost, reflections, edges, glints
    vec3 Nv = toView(N);
    vec3 rv = toView(refl);
    float frost = smoothstep(0.15, 0.75, fbm3(p * 5.5, 3) + 0.25) * 0.55;
    vec3 lightIn = vec3(0.35, 0.6, 0.9) * (0.35 + 0.65 * max(Nv.y, 0.0)) * (0.4 + 0.5 * uGlow);
    col = mix(col, lightIn * 0.9 + vec3(0.02), frost * (1.0 - F) * 0.6);
    col += env(rv) * F * 1.1;
    vec3 Lk = normalize(vec3(-0.35, 0.8, 0.5));
    col += vec3(3.5, 3.8, 4.2) * pow(max(dot(rv, Lk), 0.0), 220.0);
    vec3 Lk2 = normalize(vec3(0.6, 0.3, 0.75));
    col += vec3(1.4, 2.0, 2.6) * pow(max(dot(rv, Lk2), 0.0), 120.0);
    float ed = edgeDist(hc, p, N);
    float edge = exp(-ed / (0.007 + 0.004 * float(hc > 0)));
    float spark = 0.35 + 0.65 * pow(0.5 + 0.5 * sin(dot(N, vec3(17.0, 31.0, 23.0)) + uT * 2.5), 3.0);
    col += vec3(0.7, 0.9, 1.25) * edge * (0.30 + 0.25 * uGlow) * spark * (0.5 + F * 2.0);
    // silhouette rim (grazing facets catch the light)
    col += vec3(0.5, 0.75, 1.1) * pow(1.0 - cosi, 3.0) * 0.6;
    col += vec3(0.9, 1.0, 1.2) * uFlash;
    fragColor = vec4(col * uOpacity, uOpacity);
}
'''

# ------------------------------------------------------------------ cold mist (2D fbm, flows down)
MIST = r'''
uniform vec2 uC; uniform float uR; uniform float uT; uniform float uAmt; uniform vec3 uCol;
void main(){
    vec2 P = (fpx() - uC) / uR;       // y down, units of radius
    // envelope: around the orb, spilling down and outward
    float yd = max(P.y, 0.0);
    float spread = 1.0 + 0.55 * yd;
    float env = exp(-pow(length(vec2(P.x / spread, (P.y - 0.6) / 2.0)), 2.0) * 1.2);
    env *= smoothstep(-2.2, -0.4, P.y);
    vec2 q = P * 1.3 + vec2(0.0, -uT * 0.55);
    vec2 w = vec2(fbm2(q * 1.1 + vec2(0.0, uT * 0.2), 3), fbm2(q * 1.1 + vec2(4.3, -uT * 0.25), 3));
    float n = fbm2(q * 1.6 + w * 1.4 + vec2(uT * 0.1, 0.0), 5);
    float d = pow(smoothstep(0.30, 0.95, n), 1.6) * env * uAmt * 0.8;
    // keep the very centre clear (the orb is there)
    d *= smoothstep(0.6, 1.25, length(P));
    fragColor = vec4(uCol * d, d * 0.35);
}
'''

# ------------------------------------------------------------------ plasma charge orb (fire + ice swirl)
PLASMA = r'''
uniform vec2 uC; uniform float uR; uniform float uT; uniform float uAmp; uniform float uCharge; uniform float uCorona;
vec3 rotY(vec3 p, float a){ float c = cos(a), s = sin(a); return vec3(c * p.x + s * p.z, p.y, -s * p.x + c * p.z); }
void main(){
    vec2 P = (fpx() - uC) / uR; P.y = -P.y;
    float r = length(P);
    vec3 vio = vec3(1.25, 0.75, 2.4);
    vec3 col = vec3(0.0);
    // corona: flame-like rays streaming outward (seam-free polar noise)
    float ca = P.x / max(r, 1e-4), sa = P.y / max(r, 1e-4);
    float cn = fbm3(vec3(ca * 2.2, sa * 2.2, r * 2.4 - uT * 3.5), 4);
    float cn2 = snoise(vec3(ca * 5.0, sa * 5.0, r * 4.0 - uT * 6.0));
    float out_r = max(r - 1.0, 0.0);
    float corona = exp(-out_r * (3.2 - 1.6 * cn)) * smoothstep(0.85, 1.02, r) * (0.55 + 0.9 * max(cn, 0.0) + 0.25 * cn2);
    float halo = 0.30 / (1.0 + out_r * out_r * 9.0) * step(1.0, r);
    col += (mix(vio, vec3(2.4, 2.2, 2.6), 0.35) * corona * 1.2 + vio * halo) * uCorona;
    if (r < 1.0){
        float z = sqrt(1.0 - r * r);
        vec3 n = vec3(P, z);
        vec3 q = rotY(n, uT * 1.6);
        vec3 w = vec3(snoise(q * 1.4 + vec3(0.0, uT * 0.5, 0.0)), snoise(q * 1.4 + vec3(7.1, -uT * 0.4, 1.0)), snoise(q * 1.4 + vec3(3.3, 2.0, uT * 0.3)));
        float band = snoise(q * 1.25 + w * 0.9 + vec3(0.0, uT * 0.7, 0.0));
        float det = fbm3(q * 4.2 + w * 1.2 + vec3(0.0, 0.0, uT * 1.3), 3);
        vec3 fireC = vec3(2.3, 0.75, 0.16), iceC = vec3(0.35, 0.95, 2.4);
        vec3 base = mix(fireC, iceC, smoothstep(-0.22, 0.22, band));
        float seam = exp(-abs(band) * 10.0);
        vec3 surf = base * (0.55 + 0.6 * det) + vio * seam * 2.2;
        surf = surf * (0.45 + 0.9 * z * z) + vec3(2.6, 2.5, 3.0) * pow(z, 5.0) * (0.8 + 1.6 * uCharge);
        surf += vio * pow(1.0 - z, 2.5) * 1.6;
        float arc = 1.0 - abs(snoise(q * 3.2 + vec3(0.0, 0.0, floor(uT * 15.0) * 0.37)));
        surf += vec3(2.0, 1.8, 3.0) * smoothstep(0.955, 0.995, arc) * (0.8 + uCharge);
        float aa = smoothstep(1.0, 0.97, r);
        col = mix(col, surf, aa);
    }
    fragColor = vec4(col * uAmp, 0.0);
}
'''

# ------------------------------------------------------------------ detonation cloud (fire + ice + smoke)
BOOM = COMMON + r'''
uniform vec2 uC; uniform float uR; uniform float uT; uniform float uAge; uniform float uHeat; uniform float uSmoke; uniform float uAmp;
vec3 iceRamp(float x){ x = max(x, 0.0);
    return vec3(0.10, 0.35, 0.9) * smoothstep(0.05, 0.4, x) + vec3(0.5, 0.8, 1.2) * smoothstep(0.35, 0.9, x) + vec3(1.4, 1.5, 1.7) * smoothstep(0.85, 1.4, x); }
float field(vec3 p, out float temp, out float iceF, out float smoke){
    float r = length(p);
    vec3 q = p * 1.7 + vec3(0.0, -uAge * 0.9, uAge * 0.7);
    vec2 w = vec2(snoise(q * 0.7), snoise(q * 0.7 + vec3(4.1, 2.2, 0.0)));
    q.xy += w * 0.5;
    float n = fbm3(q, 5);
    float d = r - 1.0 - n * 0.42;
    float body = smoothstep(0.0, 0.22, -d);
    float hollow = 0.30 + 0.70 * smoothstep(0.15, 0.80, r + n * 0.25);
    float dens = body * hollow;
    temp = uHeat * (0.55 + 0.75 * n + 0.45 * (1.0 - r));
    iceF = smoothstep(0.05, 0.45, snoise(p * 1.15 + vec3(9.0, 1.0, uAge * 0.3)));
    smoke = body * smoothstep(0.75, 0.25, temp);
    return dens;
}
void main(){
    vec2 P = fpx() - uC;
    vec2 xy = vec2(P.x, -P.y) / max(uR, 1.0);
    if (dot(xy, xy) > 2.6){ fragColor = vec4(0.0); return; }
    const int N = 16;
    float zext = 1.35;
    float dz = 2.0 * zext / float(N);
    float jitter = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715)))) * dz;
    vec3 col = vec3(0.0); float Tr = 1.0; float sm = 0.0;
    for (int k = 0; k < N; k++){
        vec3 p = vec3(xy, zext - dz * float(k) - jitter);
        float temp, iceF, smoke;
        float dens = field(p, temp, iceF, smoke);
        if (dens < 1e-3 && smoke < 1e-3) continue;
        vec3 em = mix(bbody(temp), iceRamp(temp) * 1.2, iceF * 0.55) * dens;
        col += Tr * em * dz;
        sm += Tr * smoke * dens * dz;
        Tr *= exp(-dens * dz * (1.6 + 2.0 * uSmoke));
        if (Tr < 0.02) break;
    }
    float a = clamp(sm * 1.4 * uSmoke, 0.0, 0.92);
    fragColor = vec4(col * uAmp, a);
}
'''
