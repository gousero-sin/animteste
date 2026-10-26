"""GLSL sources for the ceiling_portal module (portal vortex, god rays, wormhole tunnel)."""

# --------------------------------------------------------------------------------------------
# Portal vortex seen in perspective on a horizontal plane (ceiling).  A screen point s maps to
# disk coordinates (u, v) of the plane via   1 + k v = A / (yh - s.y),  u = (s.x - px) (1+kv) / rx0
# (v > 0 = farther from the camera = lower on screen).  rho = |uv| / open.
# Output: premultiplied rgb (interior + emission) and a = opacity of the hole interior.
# --------------------------------------------------------------------------------------------
PORTAL = r'''
float sst(float a, float b, float x){ float t = clamp((x - a) / (b - a), 0.0, 1.0); return t * t * (3.0 - 2.0 * t); }
float ppow(float x, float e){ return pow(max(x, 0.0), e); }

uniform vec2 uP;        // screen position of disk centre (px, yc)
uniform float uRx;      // rx0
uniform float uA;       // yh - yc
uniform float uYh;      // horizon y
uniform float uK;       // perspective strength
uniform float uOpen;    // radius scale (disk units)
uniform float uSpin;    // integrated rotation (rad) at rho = 1
uniform float uPow;     // energy 0..2
uniform float uInner;   // interior reveal 0..1
uniform float uShockR;  // shock ring radius (unscaled disk units)
uniform float uShockI;
uniform float uCore;    // pinpoint brightness (ignition / collapse)
uniform float uHaze;    // ceiling haze swirl
uniform float uSeed;
uniform float uScale;   // render scale (1 = full res)
uniform float uDir;     // rotation direction (+1 / -1)

vec2 diskUV(vec2 s){
    float dy = max(uYh - s.y, 1.0);
    float q = uA / dy;                // = 1 + k v
    float v = (q - 1.0) / uK;
    float u = (s.x - uP.x) * q / uRx;
    return vec2(u, v);
}

vec3 vortexPal(float x){            // x: 0 inner .. 1 outer
    vec3 vio = vec3(0.55, 0.16, 1.00);
    vec3 blu = vec3(0.16, 0.36, 1.25);
    vec3 tea = vec3(0.10, 0.95, 0.95);
    return x < 0.5 ? mix(vio, blu, x * 2.0) : mix(blu, tea, (x - 0.5) * 2.0);
}

void main(){
    vec2 s = px() * uScale + uOffset;
    vec2 uv = diskUV(s);
    float R = length(uv);
    float op = max(uOpen, 1e-3);
    float rho = R / op;
    float th = atan(uv.y, uv.x);
    float fw = max(fwidth(rho), 1e-4);
    float t = uTime;
    vec3 col = vec3(0.0);
    float a = 0.0;
    float pw = uPow;

    // ---------------------------------------------------------------- rim turbulence
    float rimN = snoise(vec3(cos(th) * 2.2, sin(th) * 2.2, t * 1.7 + uSeed)) * 0.6
               + snoise(vec3(cos(th) * 6.0, sin(th) * 6.0, t * 3.1 + uSeed * 2.0)) * 0.4;
    float rimR = 1.0 + rimN * 0.022 * (0.6 + 0.6 * pw);
    float dr = rho - rimR;

    // ---------------------------------------------------------------- interior (hole)
    // A throat seen in perspective: rings of the throat at depth z are circles of radius
    // r = 1/(1+z) whose centre slides toward the NEAR lip (top of the screen, v<0) as they recede,
    // so the far wall (lower part of the ellipse) faces the camera and the near lip hides the deep end.
    if (rho < 1.15 && uOpen > 0.002){
        float cover = sst(rimR + fw * 1.5, rimR - fw * 1.5, rho);
        vec2 p = uv / op;
        vec2 cc = vec2(0.0, -0.40);
        vec2 w = p - cc;
        float wc = dot(w, cc);
        float k1 = 1.0 - dot(cc, cc);
        float r = (wc + sqrt(max(wc * wc + k1 * dot(w, w), 0.0))) / k1;
        r = clamp(r, 2e-3, 1.3);
        float z = 1.0 / r - 1.0;                    // 0 at the lip -> inf at the deep end
        vec2 qv = (p - cc * (1.0 - r)) / r;         // unit vector on the ring
        float face = 0.5 + 0.5 * qv.y;              // 1 = far wall facing the camera
        float zz = log(1.0 + z);
        float ang = atan(qv.y, qv.x) * uDir + uSpin * 0.55 + zz * 2.4;
        vec2 cs = vec2(cos(ang), sin(ang));
        // gas walls flowing down the throat
        float g1 = fbm3(vec3(cs * 1.5, zz * 1.8 - t * 1.1 + uSeed), 4);
        float g2 = fbm3(vec3(cs * 3.0 + g1, zz * 3.6 - t * 1.7), 3);
        float gas = sst(-0.10, 0.75, g1 * 0.6 + g2 * 0.6);
        float fil = ppow(1.0 - abs(snoise(vec3(cs * 7.0, zz * 6.0 - t * 2.6 + uSeed))), 10.0);
        vec3 gasCol = mix(vec3(0.42, 0.08, 0.78), vec3(0.05, 0.42, 0.75), sst(-0.3, 0.6, g2));
        float wallL = exp(-z * 1.1) * (0.18 + 0.82 * face);
        vec3 inner = vec3(0.002, 0.0015, 0.008);
        inner += gasCol * gas * wallL * 0.55 * (0.6 + 0.4 * pw);
        inner += vec3(0.55, 0.65, 1.3) * fil * gas * wallL * 0.55;
        // wall occlusion: the deep end is darker between the gas lanes
        inner *= 0.55 + 0.45 * sst(0.0, 0.5, gas);
        // near-lip sheen (fresnel-like) and the lip's own shadow on the far wall just below it
        inner += vec3(0.40, 0.50, 1.25) * exp(-z * 16.0) * (1.0 - face) * 0.55 * (0.5 + 0.5 * pw);
        // deep field of the other dimension: parallax star layers around the deep end
        float vd = cc.y * op;
        vec2 dpt = vec2(uP.x, uYh - uA / (1.0 + uK * vd));
        vec2 sp0 = (s - dpt) / (uRx * op);
        float deep = sst(0.35, 2.2, z);
        vec3 far = vec3(0.0);
        for (int L = 0; L < 3; L++){
            float fl = float(L);
            float sc = 14.0 + fl * 9.0;
            vec2 sp = rot2(uSpin * (0.10 - 0.035 * fl) * uDir) * sp0 * sc * (1.0 + 0.04 * t * (1.0 - 0.3 * fl)) + fl * 17.31;
            vec2 cell = floor(sp);
            vec2 f = fract(sp) - 0.5;
            float h = hash21(cell + 3.7 * fl + uSeed);
            vec2 off = (hash22(cell + 11.0 * fl) - 0.5) * 0.6;
            float dd = length(f - off);
            float tw = 0.65 + 0.35 * sin(t * (3.0 + 5.0 * h) + h * 40.0);
            float st = step(0.80 - 0.04 * fl, h) * exp(-dd * dd * (260.0 + 160.0 * fl)) * (0.35 + 2.6 * ppow(fract(h * 13.7), 3.0)) * tw;
            vec3 scol = mix(vec3(0.7, 0.8, 1.3), vec3(1.3, 0.95, 0.75), fract(h * 7.3));
            far += scol * st * (1.0 - 0.25 * fl);
        }
        float neb = fbm3(vec3(sp0 * 1.8 + 3.0, t * 0.04 + 4.0), 5);
        float neb2 = fbm3(vec3(sp0 * 3.5 - 1.0, t * 0.05), 3);
        far += mix(vec3(0.30, 0.04, 0.34), vec3(0.04, 0.16, 0.36), sst(-0.2, 0.4, neb2)) * sst(0.0, 0.7, neb) * 0.34;
        float cd = length(sp0 * vec2(1.0, 2.2));
        far += vec3(0.6, 0.75, 1.4) * (exp(-cd * cd * 120.0) * 1.5 + exp(-cd * 8.0) * 0.14) * (0.5 + 0.5 * pw);
        inner += far * deep;
        inner *= uInner;
        col += inner * cover;
        a = cover;
    }

    // ---------------------------------------------------------------- spiral arms (no winding:
    // rigid rotation + log spiral; noise flows inward along log(rho))
    {
        float lr = log(max(rho, 0.05));
        float tha = th * uDir - uSpin + 2.3 * lr;
        float w1 = fbm3(vec3(cos(tha) * 1.5, sin(tha) * 1.5, lr * 2.2 + t * 0.45 + uSeed), 4);
        float arms = 0.5 + 0.5 * cos(3.0 * tha + w1 * 2.8);
        arms = ppow(arms, 2.4);
        float det = fbm3(vec3(cos(tha) * 4.0, sin(tha) * 4.0, lr * 6.0 + t * 1.1), 3);
        float fil = ppow(1.0 - abs(snoise(vec3(cos(tha) * 9.0, sin(tha) * 9.0, lr * 9.0 + t * 1.6))), 6.0);
        float dens = arms * (0.45 + 1.0 * sst(-0.25, 0.6, det + w1 * 0.4)) + fil * 0.35 * arms;
        float throat = sst(0.55, 0.93, rho) * sst(1.02, 0.97, rho);          // inside the lip
        float outer = sst(0.98, 1.04, rho) * exp(-(rho - 1.0) * 3.2);               // inflow outside
        vec3 cin = vortexPal(0.15 + 0.5 * sst(0.55, 1.0, rho));
        vec3 cout = vortexPal(0.65 + 0.35 * sst(1.0, 1.6, rho));
        float on = sst(0.0, 0.25, uOpen);
        col += cin * dens * throat * (0.30 + 0.45 * pw) * on;
        col += cout * dens * outer * (0.22 + 0.30 * pw) * on;
        col += vec3(0.9, 0.95, 1.1) * fil * arms * (throat + outer * 0.6) * 0.25 * pw * on;
    }

    // ---------------------------------------------------------------- rim
    {
        float flick = 0.85 + 0.15 * snoise(vec3(th * 3.0, t * 9.0, 1.0));
        float tk = th * uDir - uSpin * 1.3;
        float kn = 0.5 + 0.5 * snoise(vec3(cos(tk) * 2.2, sin(tk) * 2.2, t * 0.8 + uSeed));
        kn = 0.45 + 1.1 * kn * kn;                                  // bright knots riding the ring
        float thick = 0.005 + 0.009 * (0.5 + 0.5 * snoise(vec3(cos(tk) * 4.0, sin(tk) * 4.0, t * 2.1)));
        float coreW = thick + fw * 0.8;
        float rimCore = exp(-dr * dr / (coreW * coreW));
        float rimGlow = exp(-abs(dr) / 0.026);
        float rimWide = exp(-max(dr, 0.0) / 0.14) * sst(-0.12, 0.0, dr);
        float flame = fbm3(vec3(cos(tk) * 6.0, sin(tk) * 6.0, abs(dr) * 22.0 - t * 3.5), 3);
        float flames = exp(-abs(dr) / 0.05) * sst(0.0, 0.55, flame);
        float on = sst(0.0, 0.06, uOpen);
        col += vec3(0.80, 0.92, 1.25) * rimCore * (1.2 + 1.2 * pw) * kn * flick * on;
        col += vec3(0.28, 0.36, 1.00) * rimGlow * (0.30 + 0.35 * pw) * (0.6 + 0.4 * kn) * on;
        col += vec3(0.35, 0.40, 1.20) * flames * (0.35 + 0.45 * pw) * on;
        col += vec3(0.22, 0.10, 0.60) * rimWide * 0.22 * (0.6 + 0.5 * pw) * on;
        float dr2 = rho - (rimR + 0.04 + 0.01 * sin(tk * 3.0 + t * 2.0));
        float w2 = 0.004 + fw * 0.7;
        col += vec3(0.55, 0.28, 1.25) * exp(-dr2 * dr2 / (w2 * w2)) * 0.45 * (0.5 + 0.5 * pw) * on;
    }

    // ---------------------------------------------------------------- ceiling haze pulled in
    if (uHaze > 0.0){
        float lr = log(max(R, 0.05));
        float omega = uSpin * 0.6 * ppow(max(R, 0.4), -1.2);
        float tha = th * uDir - omega + 1.8 * lr;
        float h = fbm3(vec3(cos(tha) * 1.4, sin(tha) * 1.4, lr * 1.6 + t * 0.25 * uDir + 7.0), 4);
        float hz = sst(-0.1, 0.8, h) * exp(-max(R - max(uOpen, 0.15), 0.0) * 1.1) * sst(0.0, 0.3, R);
        col += vec3(0.20, 0.18, 0.50) * hz * uHaze * 0.16;
    }

    // ---------------------------------------------------------------- shock ring: thin hot leading
    // edge with a short trailing tail, broken up along theta (the refraction carries the rest)
    if (uShockI > 0.0){
        float fwR = max(fwidth(R), 1e-4);
        float wR = max(0.010 + 0.006 * uShockR, fwR * 1.2);
        float d = R - uShockR;
        float lead = d > 0.0 ? exp(-d * d / (wR * wR * 0.30)) : exp(-d * d / (wR * wR * 5.0)) * 0.55;
        float nb = snoise(vec3(cos(th) * 3.5, sin(th) * 3.5, uShockR * 3.0 + uSeed * 1.7));
        float nb2 = snoise(vec3(cos(th) * 11.0, sin(th) * 11.0, uShockR * 6.0 - uSeed));
        float br = sst(-0.45, 0.6, nb) * (0.65 + 0.35 * nb2);
        col += vec3(0.62, 0.72, 1.45) * lead * br * uShockI;
    }

    // ---------------------------------------------------------------- pinpoint core
    if (uCore > 0.0){
        vec2 dv = s - uP;
        float dd = length(dv);
        col += vec3(0.9, 0.95, 1.3) * uCore * (exp(-dd * dd / 30.0) * 6.0 + exp(-dd / 28.0) * 0.6);
        // anamorphic streak
        col += vec3(0.45, 0.55, 1.4) * uCore * exp(-abs(dv.y) / 2.5) * exp(-abs(dv.x) / 140.0) * 0.7;
    }
    fragColor = vec4(col, a);
}
'''

# --------------------------------------------------------------------------------------------
# Volumetric god rays spilling down from the portal into the hazy studio.
# --------------------------------------------------------------------------------------------
RAYS = r'''
float sst(float a, float b, float x){ float t = clamp((x - a) / (b - a), 0.0, 1.0); return t * t * (3.0 - 2.0 * t); }
float ppow(float x, float e){ return pow(max(x, 0.0), e); }

uniform vec2 uP;        // portal centre
uniform float uRx, uRy; // screen ellipse radii
uniform float uI;       // intensity
uniform float uScale;
uniform float uSeed;
void main(){
    vec2 s = px() * uScale + uOffset;
    // virtual source high above the ceiling: the shafts stay roughly portal-wide
    float h = uRy * 7.0;
    vec2 src = uP - vec2(0.0, h);
    vec2 d = s - src;
    float xAt = d.x * h / max(d.y, 1.0);             // where this ray crosses the portal plane
    float rc = xAt / uRx;
    float lat = exp(-2.0 * rc * rc);
    float cone = sst(1.0, 0.55, abs(rc));
    float r1 = fbm2(vec2(rc * 5.5 + uSeed, uTime * 0.22), 4);
    float r2 = fbm2(vec2(rc * 15.0 - uSeed, uTime * 0.35 + 3.0), 3);
    float rays = ppow(clamp(r1 * 1.35, 0.0, 1.0), 5.5) * 2.0 + ppow(clamp(r2 * 1.3, 0.0, 1.0), 6.0) * 0.9;
    float yy = s.y - uP.y;
    float fall = exp(-max(yy, 0.0) / 560.0) * sst(uRy * 0.2, uRy * 1.4, yy);
    float haze = 0.45 + 0.55 * fbm3(vec3(s * 0.0035, uTime * 0.12 + uSeed), 3);
    float centre = exp(-rc * rc * 3.0);
    vec3 c = mix(vec3(0.22, 0.35, 1.0), vec3(0.62, 0.55, 1.0), centre);
    vec3 col = c * (rays * 0.85 + 0.08 * centre) * lat * cone * fall * haze * uI;
    fragColor = vec4(col, 1.0);
}
'''

# --------------------------------------------------------------------------------------------
# Wormhole / hyperspace tunnel (full frame).  Curving tube, swirling plasma walls, light-speed
# filaments, energy hoops, bright exit.
# --------------------------------------------------------------------------------------------
TUNNEL = r'''
float sst(float a, float b, float x){ float t = clamp((x - a) / (b - a), 0.0, 1.0); return t * t * (3.0 - 2.0 * t); }
float ppow(float x, float e){ return pow(max(x, 0.0), e); }

uniform vec2 uC;        // tunnel axis vanishing point (frame px)
uniform float uRoll;
uniform float uDist;    // travelled distance (tube units)
uniform float uSpeed;   // current speed
uniform float uExit;    // exit disc radius (screen units, 0 = none)
uniform float uScale;
uniform float uFoc;     // focal (px per unit)
uniform float uBend;    // tube bend strength
uniform float uBright;
uniform float uSeed;

vec2 bendAt(float Z){    // axis offset of the tube at absolute distance Z
    return vec2(sin(Z * 0.21) + 0.5 * sin(Z * 0.47 + 1.3), cos(Z * 0.17 + 0.4) + 0.4 * sin(Z * 0.39)) ;
}

vec3 pal(float x){
    // cyclic: violet -> electric blue -> teal -> blue -> magenta -> violet
    x = fract(x) * 5.0;
    vec3 c0 = vec3(0.50, 0.12, 1.00), c1 = vec3(0.10, 0.32, 1.20), c2 = vec3(0.04, 0.85, 0.95);
    vec3 c3 = vec3(0.12, 0.28, 1.10), c4 = vec3(0.95, 0.12, 0.75);
    if (x < 1.0) return mix(c0, c1, sst(0.0, 1.0, x));
    if (x < 2.0) return mix(c1, c2, sst(1.0, 2.0, x));
    if (x < 3.0) return mix(c2, c3, sst(2.0, 3.0, x));
    if (x < 4.0) return mix(c3, c4, sst(3.0, 4.0, x));
    return mix(c4, c0, sst(4.0, 5.0, x));
}

vec3 wall(vec2 d, float Rl, vec2 b0, float layer, out float cov){
    // depth of the tube of radius Rl seen through screen point d (curving axis)
    float z = Rl / max(length(d), 1e-3);
    vec2 q = d;
    for (int it = 0; it < 3; it++){
        vec2 off = (bendAt(uDist + z) - b0) * uBend;
        q = d * z - off;
        z = z * Rl / max(length(q), 1e-3);
    }
    z = min(z, 70.0);
    float ang = atan(q.y, q.x);
    float Z = z + uDist + layer * 13.7;
    float sp = 1.0 + uSpeed * 0.07 / max(z, 0.35);         // near walls smear more
    float a2 = ang + Z * (0.30 + 0.08 * layer) + 0.35 * sin(Z * 0.17);   // strong spiral twist
    vec2 cs = vec2(cos(a2), sin(a2));
    float n1 = fbm3(vec3(cs * 0.9, Z * 0.07 / sp + uSeed + layer * 5.0), 4);
    float n2 = fbm3(vec3(cs * 1.9 + n1 * 0.8, Z * 0.16 / sp - uTime * 0.2 + layer * 9.0), 4);
    float dens = sst(-0.15, 0.55, n2 + 0.6 * n1);
    dens *= dens;
    float wisps = ppow(sst(0.05, 0.6, n2 + 0.2 * n1), 3.0);
    // fine flow lines along the spiral (filaments) - two scales
    float f1 = 1.0 - abs(snoise(vec3(cs * 5.0, Z * 0.025 / sp + 3.0 + layer)));
    float f2 = 1.0 - abs(snoise(vec3(cs * 11.0 + 1.0, Z * 0.04 / sp + 7.0 + layer)));
    float fil = ppow(f1, 30.0) * 1.4 + ppow(f2, 40.0) * 0.8;
    float hue = Z * 0.010 + n1 * 0.10 + layer * 0.3;
    vec3 c = pal(hue);
    vec3 c2 = pal(hue + 0.18);
    float near = sst(0.25, 1.1, z);
    float lum = (0.12 + 0.9 * exp(-z * 0.035)) * near;      // walls glow, brighter deeper
    vec3 col = mix(c, c2, sst(-0.3, 0.5, n2)) * (dens * 0.55 + wisps * 0.9) * lum;
    col += mix(c2, vec3(1.0), 0.45) * fil * (0.2 + 1.2 * exp(-z * 0.05)) * near * sst(-0.2, 0.3, n1);
    // energy hoops rushing past
    float hz = fract(Z * 0.065);
    float hh_ = (hz - 0.5) * 30.0; float hoop = exp(-hh_ * hh_) * sst(-0.1, 0.5, n1 + n2);
    col += mix(vec3(0.45, 0.6, 1.4), c2, 0.4) * hoop * (0.2 + 0.9 * exp(-z * 0.04)) * near;
    cov = clamp(dens * 0.8 + wisps * 0.5, 0.0, 1.0) * near;
    return col;
}

void main(){
    vec2 s = px() * uScale + uOffset;
    vec2 d = rot2(uRoll) * (s - uC) / uFoc;
    vec2 b0 = bendAt(uDist);
    float covI, covO;
    vec3 outer = wall(d, 1.0, b0, 0.0, covO);
    vec3 inner = wall(d, 0.58, b0, 1.0, covI);
    vec3 col = outer * (1.0 - 0.55 * covI) + inner * 0.75;
    // depth haze toward the far end
    float r = length(d - (bendAt(uDist + 60.0) - b0) * uBend / 60.0);
    col += vec3(0.16, 0.12, 0.45) * exp(-r * 3.0) * 0.5;
    // bright far light with soft radial shafts
    float ra = atan(d.y, d.x);
    float shafts = 0.6 + 0.4 * snoise(vec3(cos(ra) * 3.0, sin(ra) * 3.0, uTime * 0.5));
    col += vec3(0.75, 0.85, 1.35) * (exp(-r * 18.0) * 2.4 + exp(-r * 5.5) * 0.35 * shafts);
    if (uExit > 0.0){
        float e = sst(uExit, uExit * 0.55, r);
        col += vec3(1.0, 1.05, 1.25) * e * 5.0;
        col += vec3(0.6, 0.7, 1.3) * exp(-max(r - uExit, 0.0) * 5.0) * sst(0.0, 0.2, uExit) * 0.9;
    }
    fragColor = vec4(col * uBright, 1.0);
}
'''
