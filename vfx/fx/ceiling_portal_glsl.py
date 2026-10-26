"""GLSL sources for the ceiling_portal module (portal vortex, god rays, wormhole tunnel)."""

# --------------------------------------------------------------------------------------------
# Portal vortex seen in perspective on a horizontal plane (ceiling).  A screen point s maps to
# disk coordinates (u, v) of the plane via   1 + k v = A / (yh - s.y),  u = (s.x - px) (1+kv) / rx0
# (v > 0 = farther from the camera = lower on screen).  rho = |uv| / open.
# Output: premultiplied rgb (interior + emission) and a = opacity of the hole interior.
# --------------------------------------------------------------------------------------------
PORTAL = r'''
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
    if (rho < 1.15 && uOpen > 0.002){
        float cover = smoothstep(rimR + fw * 1.5, rimR - fw * 1.5, rho);
        // deep vortex throat: depth grows toward the centre
        float zi = 1.0 / (rho + 0.12);
        float ang = th * uDir + uSpin * 0.45 + zi * 0.6;
        float g1 = fbm3(vec3(cos(ang) * 1.3, sin(ang) * 1.3, zi * 0.5 - t * 0.9 + uSeed), 4);
        float g2 = fbm3(vec3(cos(ang + g1) * 2.6, sin(ang + g1) * 2.6, zi * 1.0 - t * 1.4), 3);
        float gas = smoothstep(0.0, 0.8, g1 * 0.6 + g2 * 0.6);
        float depthFade = exp(-zi * 0.30);
        vec3 inner = vec3(0.004, 0.002, 0.014);
        vec3 gasCol = mix(vec3(0.42, 0.08, 0.75), vec3(0.05, 0.40, 0.70), smoothstep(-0.3, 0.6, g2));
        inner += gasCol * gas * (0.05 + 0.45 * depthFade) * (0.6 + 0.4 * pw) * smoothstep(0.25, 0.85, rho);
        float ring = pow(0.5 + 0.5 * sin(zi * 5.0 - t * 6.0 + g1 * 3.0), 8.0);
        inner += vec3(0.20, 0.26, 0.9) * ring * 0.10 * depthFade * smoothstep(0.3, 0.9, rho);
        // stars at infinity (screen space, slowly rotating around the centre)
        vec2 sp0 = (s - uP) / (uRx * op);
        for (int L = 0; L < 3; L++){
            float fl = float(L);
            vec2 sp = rot2(uSpin * (0.06 + 0.05 * fl) * uDir) * sp0 * (16.0 + fl * 10.0) + fl * 17.31;
            vec2 cell = floor(sp);
            vec2 f = fract(sp) - 0.5;
            float h = hash21(cell + 3.7 * fl + uSeed);
            vec2 off = (hash22(cell + 11.0 * fl) - 0.5) * 0.6;
            float dd = length(f - off);
            float tw = 0.65 + 0.35 * sin(t * (3.0 + 5.0 * h) + h * 40.0);
            float st = step(0.78 - 0.04 * fl, h) * exp(-dd * dd * (300.0 + 150.0 * fl)) * (0.4 + 2.4 * pow(fract(h * 13.7), 3.0)) * tw;
            vec3 sc = mix(vec3(0.7, 0.8, 1.3), vec3(1.3, 0.95, 0.75), fract(h * 7.3));
            inner += sc * st;
        }
        // nebula of the other dimension
        float neb = fbm3(vec3(sp0 * 1.8 + 3.0, t * 0.04 + 4.0), 5);
        float neb2 = fbm3(vec3(sp0 * 3.5 - 1.0, t * 0.05), 3);
        inner += mix(vec3(0.30, 0.04, 0.32), vec3(0.04, 0.16, 0.34), smoothstep(-0.2, 0.4, neb2)) * smoothstep(0.0, 0.7, neb) * 0.30;
        // far light at the end of the throat
        float cdist = length((s - uP) / vec2(uRx, uRx * 0.5) / op);
        inner += vec3(0.6, 0.75, 1.4) * (exp(-cdist * cdist * 90.0) * 1.3 + exp(-cdist * 7.0) * 0.12) * (0.5 + 0.5 * pw);
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
        arms = pow(arms, 2.4);
        float det = fbm3(vec3(cos(tha) * 4.0, sin(tha) * 4.0, lr * 6.0 + t * 1.1), 3);
        float fil = pow(1.0 - abs(snoise(vec3(cos(tha) * 9.0, sin(tha) * 9.0, lr * 9.0 + t * 1.6))), 6.0);
        float dens = arms * (0.45 + 1.0 * smoothstep(-0.25, 0.6, det + w1 * 0.4)) + fil * 0.35 * arms;
        float throat = smoothstep(0.55, 0.93, rho) * smoothstep(1.02, 0.97, rho);          // inside the lip
        float outer = smoothstep(0.98, 1.04, rho) * exp(-(rho - 1.0) * 3.2);               // inflow outside
        vec3 cin = vortexPal(0.15 + 0.5 * smoothstep(0.55, 1.0, rho));
        vec3 cout = vortexPal(0.65 + 0.35 * smoothstep(1.0, 1.6, rho));
        float on = smoothstep(0.0, 0.25, uOpen);
        col += cin * dens * throat * (0.30 + 0.45 * pw) * on;
        col += cout * dens * outer * (0.22 + 0.30 * pw) * on;
        col += vec3(0.9, 0.95, 1.1) * fil * arms * (throat + outer * 0.6) * 0.25 * pw * on;
    }

    // ---------------------------------------------------------------- rim
    {
        float flick = 0.85 + 0.15 * snoise(vec3(th * 3.0, t * 9.0, 1.0));
        float coreW = 0.007 + fw * 0.8;
        float rimCore = exp(-dr * dr / (coreW * coreW));
        float rimGlow = exp(-abs(dr) / 0.028);
        float rimWide = exp(-max(dr, 0.0) / 0.14) * smoothstep(-0.12, 0.0, dr);
        float on = smoothstep(0.0, 0.06, uOpen);
        col += vec3(0.80, 0.92, 1.25) * rimCore * (1.5 + 1.3 * pw) * flick * on;
        col += vec3(0.28, 0.36, 1.00) * rimGlow * (0.40 + 0.40 * pw) * on;
        col += vec3(0.22, 0.10, 0.60) * rimWide * 0.22 * (0.6 + 0.5 * pw) * on;
    }

    // ---------------------------------------------------------------- crackling arcs
    if (uOpen > 0.02){
        for (int L = 0; L < 2; L++){
            float fl = float(L);
            float tq = floor(t * 15.0 + fl * 0.5);
            float reach = 0.08 + 0.16 * pw;
            float sector = floor((th + PI) / TAU * 11.0 + fl * 0.5);
            float gate = step(0.80 - 0.12 * pw, hash21(vec2(sector, tq + 31.0 * fl + uSeed)));
            float x = rho - 1.0;
            float n = snoise(vec3(cos(th) * 7.0 + fl * 5.0, sin(th) * 7.0, x * 7.0 + tq * 0.61))
                    + 0.5 * snoise(vec3(cos(th) * 19.0, sin(th) * 19.0 + fl, x * 15.0 + tq * 1.37));
            float line = smoothstep(0.05, 0.0, abs(n));
            float m = smoothstep(-0.02, 0.02, x) * (1.0 - smoothstep(reach * 0.3, reach, x));
            col += vec3(0.55, 0.75, 1.6) * line * m * gate * (1.4 + 1.4 * pw) * smoothstep(0.02, 0.3, uOpen);
        }
    }

    // ---------------------------------------------------------------- ceiling haze pulled in
    if (uHaze > 0.0){
        float lr = log(max(R, 0.05));
        float omega = uSpin * 0.6 * pow(max(R, 0.4), -1.2);
        float tha = th * uDir - omega + 1.8 * lr;
        float h = fbm3(vec3(cos(tha) * 1.4, sin(tha) * 1.4, lr * 1.6 + t * 0.25 * uDir + 7.0), 4);
        float hz = smoothstep(-0.1, 0.8, h) * exp(-max(R - max(uOpen, 0.15), 0.0) * 1.1) * smoothstep(0.0, 0.3, R);
        col += vec3(0.20, 0.18, 0.50) * hz * uHaze * 0.16;
    }

    // ---------------------------------------------------------------- shock ring
    if (uShockI > 0.0){
        float d = (R - uShockR) / (0.035 + 0.05 * uShockR);
        col += vec3(0.6, 0.7, 1.4) * exp(-d * d) * uShockI;
        col += vec3(0.3, 0.2, 0.9) * exp(-abs(d) * 0.35) * uShockI * 0.12;
    }

    // ---------------------------------------------------------------- pinpoint core
    if (uCore > 0.0){
        vec2 dv = s - uP;
        float dd = length(dv);
        col += vec3(0.9, 0.95, 1.3) * uCore * (exp(-dd * dd / 30.0) * 6.0 + exp(-dd / 28.0) * 0.6);
        // anamorphic streak
        col += vec3(0.45, 0.55, 1.4) * uCore * exp(-abs(dv.y) / 3.0) * exp(-abs(dv.x) / 260.0) * 0.9;
    }
    fragColor = vec4(col, a);
}
'''

# --------------------------------------------------------------------------------------------
# Volumetric god rays spilling down from the portal into the hazy studio.
# --------------------------------------------------------------------------------------------
RAYS = r'''
uniform vec2 uP;        // portal centre
uniform float uRx, uRy; // screen ellipse radii
uniform float uI;       // intensity
uniform float uScale;
uniform float uSeed;
void main(){
    vec2 s = px() * uScale + uOffset;
    float h = uRy * 2.4;
    vec2 src = uP - vec2(0.0, h);
    vec2 d = s - src;
    float ang = atan(d.x, max(d.y, 1.0));
    float xAt = d.x * h / max(d.y, 1.0);             // where this ray crosses the portal plane
    float cone = smoothstep(uRx * 1.02, uRx * 0.35, abs(xAt));
    float r1 = fbm2(vec2(ang * 9.0 + uSeed, uTime * 0.22), 4);
    float r2 = fbm2(vec2(ang * 27.0 - uSeed, uTime * 0.35 + 3.0), 3);
    float rays = pow(clamp(r1 * 1.25, 0.0, 1.0), 3.0) * 1.7 + pow(clamp(r2 * 1.2, 0.0, 1.0), 4.0) * 0.9;
    float yy = s.y - uP.y;
    float fall = exp(-max(yy, 0.0) / 620.0) * smoothstep(uRy * 0.2, uRy * 1.6, yy);
    float haze = 0.55 + 0.45 * fbm3(vec3(s * 0.0035, uTime * 0.12 + uSeed), 3);
    float centre = exp(-pow(xAt / (uRx * 0.75), 2.0));
    vec3 c = mix(vec3(0.22, 0.35, 1.0), vec3(0.65, 0.55, 1.0), centre);
    vec3 col = c * (rays * 0.75 + 0.35 * centre) * cone * fall * haze * uI;
    fragColor = vec4(col, 1.0);
}
'''

# --------------------------------------------------------------------------------------------
# Wormhole / hyperspace tunnel (full frame).  Curving tube, swirling plasma walls, light-speed
# filaments, energy hoops, bright exit.
# --------------------------------------------------------------------------------------------
TUNNEL = r'''
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
    if (x < 1.0) return mix(c0, c1, smoothstep(0.0, 1.0, x));
    if (x < 2.0) return mix(c1, c2, smoothstep(1.0, 2.0, x));
    if (x < 3.0) return mix(c2, c3, smoothstep(2.0, 3.0, x));
    if (x < 4.0) return mix(c3, c4, smoothstep(3.0, 4.0, x));
    return mix(c4, c0, smoothstep(4.0, 5.0, x));
}

void main(){
    vec2 s = px() * uScale + uOffset;
    vec2 d = rot2(uRoll) * (s - uC) / uFoc;
    // solve for depth with a curving axis: |d * z - (bend(Z) - bend(D))| = 1
    vec2 b0 = bendAt(uDist);
    float z = 1.0 / max(length(d), 1e-3);
    vec2 q = d;
    for (int it = 0; it < 3; it++){
        vec2 off = (bendAt(uDist + z) - b0) * uBend;
        q = d * z - off;
        z = z / max(length(q), 1e-3);
    }
    z = min(z, 60.0);
    float ang = atan(q.y, q.x);
    float Z = z + uDist;
    float sp = 1.0 + uSpeed * 0.10;                       // streak stretch with speed
    float a2 = ang + Z * 0.11 + 0.3 * sin(Z * 0.21);      // spiral twist of the throat
    vec2 cs = vec2(cos(a2), sin(a2));
    float n1 = fbm3(vec3(cs * 1.2, Z * 0.20 / sp + uSeed), 4);
    float n2 = fbm3(vec3(cs * 2.4 + n1 * 1.1, Z * 0.42 / sp - uTime * 0.3), 4);
    float dens = smoothstep(0.0, 0.85, n2 + 0.45 * n1);
    dens = dens * dens * (3.0 - 2.0 * dens);
    float lanes = smoothstep(-0.15, 0.35, n1);            // big dark gaps between plasma lanes
    // light-speed filaments: high angular frequency, low depth frequency (sparse)
    float f1 = 1.0 - abs(snoise(vec3(cs * 6.0, Z * 0.04 / sp + 3.0)));
    float f2 = 1.0 - abs(snoise(vec3(cs * 14.0 + 2.0, Z * 0.07 / sp + 9.0)));
    float fil = pow(f1, 40.0) * 1.8 + pow(f2, 50.0) * 1.2;
    float hue = Z * 0.016 + n1 * 0.18 + 0.08 * sin(a2 * 2.0);
    vec3 c = pal(hue);
    vec3 c2 = pal(hue + 0.22);
    float near = smoothstep(0.30, 1.4, z);                // walls very close to camera dim (and blur)
    float far = exp(-z * 0.05);
    float mid = smoothstep(0.6, 3.0, z) * exp(-z * 0.03);
    vec3 col = mix(c, c2, n2 * 0.5 + 0.5) * dens * lanes * (0.10 + 0.85 * mid) * near;
    col += mix(c2, vec3(1.0), 0.35) * fil * (0.15 + 1.3 * far) * near * (0.4 + 0.6 * lanes);
    // energy hoops rushing past
    float hz = fract(Z * 0.07);
    float hoop = exp(-pow((hz - 0.5) * 26.0, 2.0)) * smoothstep(-0.2, 0.5, n1);
    col += vec3(0.40, 0.55, 1.3) * hoop * (0.15 + 0.8 * far) * near;
    // depth fog toward the bright far end
    col += vec3(0.10, 0.12, 0.35) * smoothstep(4.0, 30.0, z) * 0.6;
    // bright exit / far light
    float r = length(d - (bendAt(uDist + 60.0) - b0) * uBend / 60.0);
    col += vec3(0.75, 0.85, 1.35) * (exp(-r * 16.0) * 2.0 + exp(-r * 5.0) * 0.30);
    if (uExit > 0.0){
        float e = smoothstep(uExit, uExit * 0.55, r);
        col += vec3(1.0, 1.05, 1.25) * e * 5.0;
        col += vec3(0.6, 0.7, 1.3) * exp(-max(r - uExit, 0.0) * 5.0) * smoothstep(0.0, 0.2, uExit) * 0.9;
    }
    fragColor = vec4(col * uBright, 1.0);
}
'''
