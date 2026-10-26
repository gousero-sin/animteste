"""Bathroom -> black studio transformation ("eu coloco o meu estúdio todo preto").

  13.00-13.26  anticipation: lamp flickers, the room light is drawn toward an energy core that
               forms behind him (pinch distortion, halo peeking around his head)
  13.26-14.20  an electric front expands from behind him to the frame corners. Ahead of it the
               wall cracks (impact pattern, cracks propagate + refraction ripple); at the front the
               walls break into 3D shards that tumble back into the void (perspective, rotation,
               glazed-tile glints, eroding hot edges, fog to black), sparks fly off the front,
               the ceiling lamp pops when the front reaches it.
  14.30-14.90  (studio lights strike on - handled by the studio code)

Shards are real geometry rendered with moderngl (perspective-correct, depth tested). The
fracture is an impact pattern (cleanup_fracture.py): radial + concentric cracks from behind his
neck, growing with distance and room depth, following the real tile grout and ceiling slats.
Cracks PROPAGATE ahead of the detachment front (radial cracks run first, tips glow hottest).
Motion blur: 10 sub-frame samples (5 while receding) centred on t (180 deg shutter), each with
a Halton sub-pixel jitter (accumulation-buffer AA), accumulated on the GPU. Ring/core lights
stay outside his silhouette (rim only). Everything is analytic in t.
"""
import numpy as np
import cv2
import moderngl
from .. import config as C
from ..glfx import gl, PRELUDE
from .cleanup_util import sstep, sstep_f, pulse
from .cleanup_fracture import fracture, T_BORDER, T_RADIAL, T_CONC, T_GROUT, T_SLAT, T_SPLINTER

W, H = C.W, C.H

T_ANT = 13.00
TW0, TW1 = 13.26, 14.20
ORIGIN = np.float32([580.0, 1100.0])     # behind his neck / upper back
PCENTER = np.float32([580.0, 1000.0])    # perspective centre (vanishing point of the void)
FOCAL = 1100.0
LAMP = np.float32([1015.0, 488.0])
R_MAX = 1560.0


def front_R(t):
    u = np.clip((np.asarray(t, np.float64) - TW0) / (TW1 - TW0), 0.0, 1.0)
    r = 40.0 + (R_MAX - 40.0) * (0.22 * u + 0.78 * u * u)
    return np.where(np.asarray(t) < TW0, 0.0, r)


def front_mod(th):
    return 1.0 + 0.06 * np.sin(3 * th + 1.3) + 0.04 * np.sin(5 * th + 4.1) + 0.03 * np.sin(9 * th + 0.7)


GLSL_FRONT = '''
float frontMod(float th){ return 1.0 + 0.06*sin(3.0*th+1.3) + 0.04*sin(5.0*th+4.1) + 0.03*sin(9.0*th+0.7); }
float frontR(float t){ float u = clamp((t - %.4f) / (%.4f - %.4f), 0.0, 1.0);
  return t < %.4f ? 0.0 : 40.0 + (%.1f - 40.0) * (0.22 * u + 0.78 * u * u); }
''' % (TW0, TW1, TW0, TW0, R_MAX)

SHARD_VS = '''
#version 330
in vec2 aPos; in vec2 aCen; in float aE; in vec4 aP0; in vec4 aP1; in vec4 aP2; in vec3 aK;
uniform float uT;
uniform vec2 uJit;
uniform vec2 uPC;
uniform float uF;
uniform vec2 uFrame;
out vec2 vUV; out float vE; out float vAge; out vec3 vN; out float vZ; out float vSeed; out vec2 vRest;
flat out vec3 vK;
mat3 rotAxis(vec3 a, float ang){
  float c = cos(ang), s = sin(ang), k = 1.0 - c;
  return mat3(c + a.x*a.x*k,     a.y*a.x*k + a.z*s, a.z*a.x*k - a.y*s,
              a.x*a.y*k - a.z*s, c + a.y*a.y*k,     a.z*a.y*k + a.x*s,
              a.x*a.z*k + a.y*s, a.y*a.z*k - a.x*s, c + a.z*a.z*k);
}
void main(){
  float age = uT - aP0.x;
  vec3 lp = vec3(aPos - aCen, 0.0);
  vec3 n = vec3(0.0, 0.0, -1.0);
  vec3 c = vec3(aCen - uPC, 0.0);
  if (age > 0.0){
    float ang = aP0.y * age * (1.0 + 0.35 * age);
    mat3 R = rotAxis(normalize(aP1.xyz), ang);
    lp = R * lp; n = R * n;
    c.xy += aP2.xy * age + vec2(0.0, 0.5 * aP2.z * age * age);
    c.z += aP0.w * age + 0.5 * aP1.w * age * age;
  } else if (age > -0.12) {
    // the wall trembles just before it breaks
    float k = (age + 0.12) / 0.12;
    c.xy += k * 1.6 * vec2(sin(uT * 97.0 + aP0.z * 40.0), cos(uT * 83.0 + aP0.z * 23.0));
  }
  vec3 w = c + lp;
  float Wc = max((uF + w.z) / uF, 0.05);
  vec2 sp = uPC + w.xy / Wc + uJit;
  vec2 ndc = vec2(sp.x / uFrame.x * 2.0 - 1.0, 1.0 - sp.y / uFrame.y * 2.0);
  float zn = clamp(w.z / 9000.0, -0.99, 0.99);
  gl_Position = vec4(ndc * Wc, zn * Wc, Wc);
  vUV = aPos / uFrame; vE = aE; vAge = age; vN = n; vZ = w.z; vSeed = aP0.z; vRest = aPos; vK = aK;
}
'''

SHARD_FS = GLSL_FRONT + '''
in float vE; in float vAge; in vec3 vN; in float vZ; in float vSeed; in vec2 vRest;
flat in vec3 vK;
uniform sampler2D uTex;
uniform float uT;
uniform vec2 uFrame;
uniform vec2 uOrigin;
uniform float uR;          // front radius (px), 0 before the wave
uniform float uPinch;      // anticipation pinch (px)
uniform float uExpo;       // room light level
void main(){
  vec2 rel = vRest - uOrigin; float r = length(rel); vec2 dir = rel / max(r, 1.0);
  float th = atan(rel.y, rel.x);
  float d = r - uR * frontMod(th);              // >0: ahead of the front
  vec2 off = vec2(0.0);
  if (vAge < 0.0){
    off += dir * uPinch * exp(-r / 650.0);       // room drawn toward the core
    if (uR > 0.0) off += dir * 10.0 * exp(-pow((d - 40.0) / 34.0, 2.0)) - dir * 5.0 * exp(-pow((d - 110.0) / 40.0, 2.0));
  }
  vec3 tex = texture(uTex, (vRest + off) / uFrame).rgb;
  float face = -vN.z;
  float fw = max(fwidth(vE), 1e-5);
  float edgePx = (1.0 - vE) / fw;
  vec3 col;
  vec3 glowC = vec3(0.55, 0.78, 1.0);
  float fog = exp(-max(vZ, 0.0) / 650.0);
  if (vAge < 0.0){
    col = tex * uExpo;
    // light from the approaching front washes the wall
    if (uR > 0.0) col *= 1.0 + 0.30 * exp(-max(d, 0.0) / 55.0);
    // cracks propagate ahead of the front: radial cracks lead, then grout, slats, rings.
    // Visible where this fragment is inside the crack front R(t + lead); the running tip is
    // white-hot, older crack lines cool to a thin blue glow.
    float Rl = frontR(uT + vK.x) * frontMod(th);
    float behind = Rl - r;                      // px behind the crack tip (>0: cracked)
    float on = smoothstep(-6.0, 10.0, behind) * step(0.5, vK.y);
    if (vK.x > 0.0 && on > 0.0){
      float tip = exp(-max(behind, 0.0) / 38.0);
      float line = 0.9 * exp(-edgePx * edgePx / 0.55) + 0.10 * exp(-edgePx / 3.0);
      col *= 1.0 - 0.55 * on * exp(-edgePx * edgePx / 1.6);          // dark split line
      col += mix(glowC, vec3(1.25, 1.2, 1.15), tip) * line * on * vK.y * (0.22 + 1.5 * tip);
    }
  } else {
    // relit, tumbling shard: room light fades as it leaves the wall
    vec3 L = normalize(vec3(0.35, -0.55, -0.75));
    float l0 = 0.75;
    float lam = max(dot(vN, L), 0.0) / l0;
    float dark = mix(1.0, 0.30, smoothstep(0.0, 0.35, vAge));
    if (face > 0.0){
      col = tex * uExpo * mix(1.0, lam, 0.85) * dark;
      vec3 Hh = normalize(L + vec3(0.0, 0.0, -1.0));
      float sp = pow(max(dot(vN, Hh), 0.0), 70.0);
      col += vec3(0.8, 0.9, 1.0) * sp * 1.3 * dark;
    } else {
      col = vec3(0.030, 0.029, 0.028) * (0.4 + 0.6 * max(dot(-vN, L), 0.0)) * dark;   // cement back
    }
    // erosion from the edges inward, with a hot rim
    float er = smoothstep(0.06, 0.62, vAge);
    float nv = 0.6 * vnoise(vRest * 0.05 + vSeed * 17.0) + 0.4 * vnoise(vRest * 0.17 + vSeed * 5.0);
    float inner = 1.0 - vE;
    float thr = er * 1.2 - 0.12 + (nv - 0.5) * 0.45;
    if (inner < thr) discard;
    float eg = exp(-(inner - thr) / 0.035) * step(0.0, thr);
    float rim = exp(-vAge / 0.20) * (0.9 * exp(-edgePx / 1.3) + 0.15 * exp(-edgePx / 6.0));
    col += glowC * (rim + eg * 1.5 * (1.0 - 0.5 * er));
  }
  col *= fog;
  fragColor = vec4(col, 1.0);
}
'''

RING_FS = GLSL_FRONT + '''
uniform vec2 uOrigin; uniform float uR; uniform float uAmp; uniform float uCore; uniform float uT2;
void main(){
  vec2 p = fpx(); vec2 rel = p - uOrigin; float r = length(rel); float th = atan(rel.y, rel.x);
  vec3 col = vec3(0.0);
  // anticipation core: hot halo behind him
  if (uCore > 0.0){
    float g = exp(-r * r / (2.0 * 120.0 * 120.0)) * 0.6 + exp(-r / 200.0) * 0.12;
    float sw = 0.75 + 0.25 * snoise(vec3(cos(th) * 2.0, sin(th) * 2.0, uT2 * 3.0));
    col += vec3(0.45, 0.65, 1.6) * g * uCore * sw;
  }
  if (uR > 0.0 && uAmp > 0.0){
    float n1 = snoise(vec3(cos(th) * 3.0, sin(th) * 3.0, uT2 * 1.7));
    float d = r - uR * frontMod(th) - 10.0 * n1;
    if (abs(d) < 320.0){
      float core = exp(-d * d / (2.0 * 2.0 * 2.0));
      float inner = d < 0.0 ? exp(d / 45.0) : 0.0;
      float outer = d > 0.0 ? exp(-d / 20.0) : 0.0;
      float wide = exp(-abs(d) / 140.0);
      float fil = 0.0;
      if (abs(d) < 70.0){
        float rn = ridged3(vec3(cos(th) * 16.0, sin(th) * 16.0, d * 0.045 - uT2 * 5.0), 3);
        fil = pow(clamp(rn, 0.0, 1.0), 5.0) * exp(-abs(d) / 26.0);
      }
      col += vec3(1.0, 1.0, 1.05) * core * 4.0
           + vec3(0.30, 0.55, 1.50) * inner * 0.55
           + vec3(0.45, 0.70, 1.40) * outer * 0.45
           + vec3(0.20, 0.35, 0.90) * wide * 0.05
           + vec3(0.55, 0.65, 1.80) * fil * 3.0;
      col *= uAmp;
    }
  }
  fragColor = vec4(col, 1.0);
}
'''


ACC_FS = '''
#version 330
uniform sampler2D uSrc;
uniform float uW;
out vec4 fragColor;
void main(){ fragColor = texelFetch(uSrc, ivec2(gl_FragCoord.xy), 0) * uW; }
'''

# crack type -> (lead seconds ahead of the detachment front, glow weight)
CRACK = {T_BORDER: (0.0, 0.0), T_RADIAL: (0.17, 1.0), T_GROUT: (0.12, 0.85), T_SLAT: (0.09, 0.65),
         T_CONC: (0.07, 0.8), T_SPLINTER: (0.04, 0.6)}
# Halton(2,3) sub-pixel offsets for the accumulation samples
def _halton(i, b):
    f, r = 1.0, 0.0
    while i > 0:
        f /= b
        r += f * (i % b)
        i //= b
    return r


_HALTON = np.array([[_halton(i + 1, 2), _halton(i + 1, 3)] for i in range(16)], np.float32) - 0.5


class Shatter:
    def __init__(self, seed=5):
        self.seed = seed
        self._built = False

    # ------------------------------------------------------------------ build
    def build(self):
        if self._built:
            return
        rng = np.random.default_rng(self.seed)
        cells = fracture(ORIGIN, self.seed)
        # inverse of the front radius on a dense table
        tt = np.linspace(TW0, TW1, 4000)
        rr = front_R(tt)
        verts = []
        cen_list, tdet_list, vel_list, z_list, r_list = [], [], [], [], []
        for poly, etype, splinter in cells:
            poly = poly.astype(np.float64)
            # area-weighted centroid (fan centre)
            x, y = poly[:, 0], poly[:, 1]
            cr = x * np.roll(y, -1) - np.roll(x, -1) * y
            A2 = cr.sum()
            c = (np.array([((x + np.roll(x, -1)) * cr).sum(), ((y + np.roll(y, -1)) * cr).sum()]) / (3 * A2)
                 if abs(A2) > 1e-6 else poly.mean(0))
            area = abs(A2) / 2
            rs = float(np.sqrt(np.mean(np.sum((poly - c) ** 2, 1))))
            rel = c - ORIGIN
            dd = float(np.hypot(*rel))
            th = float(np.arctan2(rel[1], rel[0]))
            need = (dd - 0.35 * rs) / front_mod(th)
            tdet = float(np.interp(need, rr, tt)) + rng.uniform(-0.015, 0.025)
            big = area > 9000
            kspin = 1.5 if splinter else (0.6 if big else 1.0)
            kvel = 1.45 if splinter else (0.8 if big else 1.0)
            spin = rng.uniform(2.0, 5.0) * rng.choice([-1, 1]) * kspin
            phi = rng.uniform(0, 2 * np.pi)
            axis = np.array([np.cos(phi), np.sin(phi), 0.3 * rng.standard_normal()], np.float32)
            axis /= np.linalg.norm(axis)
            rdir = rel / max(dd, 1.0)
            vxy = (rdir * rng.uniform(60, 220) + rng.uniform(-40, 40, 2)) * kvel
            grav = rng.uniform(150, 450)
            zv = rng.uniform(250, 700) * (1.2 if splinter else 1.0)
            za = rng.uniform(1800, 3500) * (0.85 if big else 1.0)
            sd = rng.random()
            P0 = (tdet, spin, sd, zv)
            P1 = (axis[0], axis[1], axis[2], za)
            P2 = (vxy[0], vxy[1], grav, rs)
            n = len(poly)
            for k in range(n):
                a_, b_ = poly[k], poly[(k + 1) % n]
                lead, glow = CRACK[int(etype[k])]
                K = (lead, glow, float(etype[k]))
                for (p, e) in ((c, 0.0), (a_, 1.0), (b_, 1.0)):
                    verts.append([p[0], p[1], c[0], c[1], e, *P0, *P1, *P2, *K])
            cen_list.append(c); tdet_list.append(tdet); vel_list.append((vxy[0], vxy[1], grav)); z_list.append((zv, za)); r_list.append(rs)
        self.verts = np.ascontiguousarray(np.array(verts, np.float32))
        self.cen = np.array(cen_list, np.float32)
        self.tdet = np.array(tdet_list, np.float32)
        self.vel = np.array(vel_list, np.float32)
        self.zp = np.array(z_list, np.float32)
        self.rsz = np.array(r_list, np.float32)
        ctx = gl.ctx
        self.prog = ctx.program(vertex_shader=SHARD_VS, fragment_shader=PRELUDE + '\n' + SHARD_FS)
        self.vbo = ctx.buffer(self.verts.tobytes())
        self.vao = ctx.vertex_array(self.prog, [(self.vbo, '2f 2f 1f 4f 4f 4f 3f', 'aPos', 'aCen', 'aE', 'aP0', 'aP1', 'aP2', 'aK')])
        # scene target (colour texture + depth) and the accumulation target
        self.stex = ctx.texture((W, H), 4, dtype='f4')
        self.sdb = ctx.depth_renderbuffer((W, H))
        self.fbo = ctx.framebuffer([self.stex], self.sdb)
        self.atex = ctx.texture((W, H), 4, dtype='f4')
        self.afbo = ctx.framebuffer([self.atex])
        from ..glfx import VERT
        self.acc_prog = ctx.program(vertex_shader=VERT, fragment_shader=ACC_FS)
        self.acc_vao = ctx.vertex_array(self.acc_prog, [(gl.vbo, '2f', 'in_vert')])
        self.ring_prog = gl.program(RING_FS)
        # front sparks: short-lived, fast, long-tailed speed distribution (they ride the front
        # instead of hanging behind it)
        n = 800
        self.sp_tb = rng.uniform(TW0 + 0.06, TW1 - 0.02, n).astype(np.float32)
        self.sp_th = rng.uniform(-np.pi, np.pi, n).astype(np.float32)
        self.sp_v = (np.exp(rng.normal(np.log(520), 0.45, n))).astype(np.float32)
        self.sp_tan = rng.uniform(-160, 160, n).astype(np.float32)
        self.sp_life = rng.uniform(0.08, 0.26, n).astype(np.float32)
        self.sp_int = rng.uniform(1.5, 5.0, n).astype(np.float32)
        self.sp_rad = np.clip(np.exp(rng.normal(np.log(0.9), 0.35, n)), 0.6, 2.0).astype(np.float32)
        # shard debris (erosion particles)
        ns = len(self.cen)
        k = 4
        self.db_idx = np.repeat(np.arange(ns), k)
        self.db_off = (rng.standard_normal((ns * k, 2)) * 0.45 * np.repeat(self.rsz, k)[:, None]).astype(np.float32)
        self.db_t = (np.repeat(self.tdet, k) + rng.uniform(0.05, 0.55, ns * k)).astype(np.float32)
        self.db_v = rng.standard_normal((ns * k, 2)).astype(np.float32) * 60
        self.db_life = rng.uniform(0.25, 0.7, ns * k).astype(np.float32)
        self.db_int = rng.uniform(0.5, 1.6, ns * k).astype(np.float32)
        self.t_lamp = float(np.interp(np.hypot(*(LAMP - ORIGIN)) / front_mod(np.arctan2(*(LAMP - ORIGIN)[::-1])), rr, tt))
        self._built = True

    # ------------------------------------------------------------------ timing helpers
    def room_light(self, t):
        """Lamp/room light level: flicker, surge, pop when the front reaches the lamp."""
        self.build()
        l = 1.0
        l -= 0.55 * pulse(t, 13.06, 0.018) + 0.40 * pulse(t, 13.12, 0.012) + 0.65 * pulse(t, 13.17, 0.02)
        l -= 0.18 * sstep_f(13.18, 13.26, t)                 # light drawn into the core
        l += 0.35 * pulse(t, self.t_lamp - 0.03, 0.03)        # surge
        l *= 1.0 - 0.70 * sstep_f(self.t_lamp, self.t_lamp + 0.05, t)
        return max(l, 0.0)

    def pinch(self, t):
        return 22.0 * sstep_f(13.08, 13.26, t) * (1.0 - sstep_f(13.26, 13.5, t))

    def core_amp(self, t):
        return 0.9 * sstep_f(13.08, 13.27, t) * (1.0 - sstep_f(13.30, 13.65, t)) + 0.6 * pulse(t, 13.27, 0.04)

    def ring_amp(self, t):
        return sstep_f(TW0, TW0 + 0.08, t) * (1.0 - sstep_f(TW1 - 0.06, TW1 + 0.04, t))

    # ------------------------------------------------------------------ render
    def shards(self, t, tex_img):
        """Render the shard layer. Returns (premultiplied rgb, alpha) full frame."""
        self.build()
        ctx = gl.ctx
        tex = ctx.texture((W, H), 3, np.ascontiguousarray(tex_img, np.float32).tobytes(), dtype='f4')
        tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        tex.use(0)
        pr = self.prog
        pr['uTex'].value = 0
        pr['uFrame'].value = (float(W), float(H))
        pr['uPC'].value = (float(PCENTER[0]), float(PCENTER[1]))
        pr['uF'].value = FOCAL
        pr['uOrigin'].value = (float(ORIGIN[0]), float(ORIGIN[1]))
        pr['uPinch'].value = float(self.pinch(t))
        pr['uExpo'].value = float(0.30 + 0.70 * self.room_light(t))
        # motion blur: samples centred on t over a 1/60 s (180 deg) shutter, each with a
        # Halton sub-pixel jitter -> temporal + spatial antialiasing in one accumulation
        if t < TW0 - 0.13:
            nsub = 1
        elif t < TW1 + 0.06:
            nsub = 10
        else:
            nsub = 5
        shutter = 1.0 / 60.0
        self.afbo.use()
        self.afbo.clear(0, 0, 0, 0)
        for k in range(nsub):
            ts = t + ((k + 0.5) / nsub - 0.5) * shutter if nsub > 1 else t
            jit = _HALTON[k] if nsub > 1 else (0.0, 0.0)
            pr['uT'].value = float(ts)
            pr['uR'].value = float(front_R(ts))
            pr['uJit'].value = (float(jit[0]), float(-jit[1]))
            self.fbo.use()
            self.fbo.clear(0, 0, 0, 0, depth=1.0)
            ctx.enable(moderngl.DEPTH_TEST)
            tex.use(0)
            self.vao.render(moderngl.TRIANGLES)
            ctx.disable(moderngl.DEPTH_TEST)
            # add into the accumulation buffer
            self.afbo.use()
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = moderngl.ONE, moderngl.ONE
            self.stex.use(1)
            self.acc_prog['uSrc'].value = 1
            self.acc_prog['uW'].value = 1.0 / nsub
            self.acc_vao.render(moderngl.TRIANGLES)
            ctx.disable(moderngl.BLEND)
        acc = np.frombuffer(self.afbo.read(components=4, dtype='f4'), np.float32).reshape(H, W, 4)
        tex.release()
        acc = acc[::-1]
        return acc[..., :3].copy(), np.clip(acc[..., 3], 0, 1)

    def ring(self, t):
        self.build()
        R = float(front_R(t))
        amp = self.ring_amp(t)
        core = self.core_amp(t)
        if amp <= 0.001 and core <= 0.001:
            return None
        img = gl.render(self.ring_prog, (W, H), uniforms={
            'uOrigin': (float(ORIGIN[0]), float(ORIGIN[1])), 'uR': R, 'uAmp': float(amp),
            'uCore': float(core), 'uT2': float(t)})
        return img[..., :3]

    def particles(self, t):
        """Returns (front, void): additive full-frame emissions (or None). 'front' = sparks
        off the front + lamp pop (at the wall plane); 'void' = debris shed by receding shards
        (to be occluded by the shard coverage)."""
        self.build()
        P0s, P1s, cols, rads = [], [], [], []
        front_img = None
        # front sparks
        a = t - self.sp_tb
        al = (a > 0) & (a < self.sp_life)
        if al.any():
            aa = a[al]
            tb = self.sp_tb[al]
            th = self.sp_th[al]
            Rb = front_R(tb) * front_mod(th)
            dirv = np.stack([np.cos(th), np.sin(th)], 1)
            tanv = np.stack([-np.sin(th), np.cos(th)], 1)
            base = ORIGIN + dirv * Rb[:, None]

            def pos(x):
                dr = (1 - np.exp(-3.0 * x)) / 3.0
                return base + dirv * (self.sp_v[al] * dr)[:, None] + tanv * (self.sp_tan[al] * dr)[:, None]
            p1 = pos(aa)
            p0 = pos(np.maximum(aa - 1.0 / 45, 0))
            heat = 1.0 - aa / self.sp_life[al]
            inten = self.sp_int[al] * heat ** 1.5 * sstep(0, 0.015, aa)
            c = np.stack([(0.6 + 0.5 * heat) * inten, (0.8 + 0.3 * heat) * inten, 1.3 * inten, np.ones_like(inten)], 1)
            front_img = gl.sprites((W, H), p0, p1, radius=self.sp_rad[al], color=c, shape=2)[..., :3]
        # debris shed by eroding shards (follow the shard into the void)
        a = t - self.db_t
        al = (a > 0) & (a < self.db_life)
        if al.any():
            idx = self.db_idx[al]
            aa = a[al]

            def proj(age_s, extra):
                cen = self.cen[idx]
                ag = np.maximum(age_s, 0)
                v = self.vel[idx]
                zp = self.zp[idx]
                cx = cen[:, 0] - PCENTER[0] + v[:, 0] * ag + extra[:, 0]
                cy = cen[:, 1] - PCENTER[1] + v[:, 1] * ag + 0.5 * v[:, 2] * ag * ag + extra[:, 1]
                z = zp[:, 0] * ag + 0.5 * zp[:, 1] * ag * ag
                s = FOCAL / (FOCAL + z)
                return np.stack([PCENTER[0] + cx * s, PCENTER[1] + cy * s], 1), z
            sage = t - self.tdet[idx]
            ex = self.db_off[al] + self.db_v[al] * aa[:, None]
            p1, z = proj(sage, ex)
            ex0 = self.db_off[al] + self.db_v[al] * np.maximum(aa - 1.0 / 50, 0)[:, None]
            p0, _ = proj(sage - 1.0 / 50, ex0)
            fog = np.exp(-z / 650.0)
            inten = self.db_int[al] * (1 - aa / self.db_life[al]) * fog * sstep(0, 0.03, aa)
            c = np.stack([0.55 * inten, 0.75 * inten, 1.1 * inten, np.ones_like(inten)], 1)
            P0s.append(p0); P1s.append(p1); cols.append(c); rads.append(np.full(len(p0), 1.0, np.float32))
        img = front_img
        void_img = None
        if P0s:
            void_img = gl.sprites((W, H), np.concatenate(P0s), np.concatenate(P1s), radius=np.concatenate(rads),
                                  color=np.concatenate(cols), shape=2)[..., :3]
        # lamp pop flash
        fl = pulse(t, self.t_lamp + 0.01, 0.03)
        if fl > 0.01:
            x0, y0 = int(LAMP[0] - 200), int(LAMP[1] - 200)
            X, Y = np.meshgrid(np.arange(400) + x0, np.arange(400) + y0)
            r2 = (X - LAMP[0]) ** 2 + (Y - LAMP[1]) ** 2
            g = (np.exp(-r2 / (2 * 14.0 ** 2)) * 25.0 + np.exp(-r2 / (2 * 70.0 ** 2)) * 1.5) * fl
            lay = g[..., None].astype(np.float32) * np.float32([1.0, 0.95, 0.85])
            if img is None:
                img = np.zeros((H, W, 3), np.float32)
            from ..core import Layer, blend
            blend(img, Layer('add', lay, box=(x0, y0)))
        return img, void_img

    @staticmethod
    def _edge_dist(alpha, th_list):
        """Distance from ORIGIN to the person's silhouette edge along each direction (quarter
        res ray march; frame edge if the ray never leaves the body)."""
        q = 4
        a4 = cv2.resize(alpha, (W // q, H // q), interpolation=cv2.INTER_AREA)
        rs = np.arange(0.0, 1500.0, 6.0)
        out = []
        for th in th_list:
            x = (ORIGIN[0] + rs * np.cos(th)) / q
            y = (ORIGIN[1] + rs * np.sin(th)) / q
            ok = (x >= 0) & (x < W // q) & (y >= 0) & (y < H // q)
            if not ok.any():
                out.append(0.0)
                continue
            v = np.zeros_like(rs)
            v[ok] = a4[y[ok].astype(int), x[ok].astype(int)]
            inside = v > 0.3
            # last sample that is inside the body, walking outward from the origin
            idx = np.where(inside & ok)[0]
            out.append(float(rs[idx.max()]) if len(idx) else 0.0)
        return np.array(out)

    def lights(self, t, alpha=None):
        """Ring of rim lights on the expanding front + core glow + lamp pop. Lights are kept
        OUTSIDE his silhouette (>= 60 px beyond the alpha edge along their direction), rim only:
        light from behind must not wash his face frontally."""
        self.build()
        out = []
        amp = self.ring_amp(t)
        core = self.core_amp(t)
        R = float(front_R(t))
        need = amp > 0.001 or core > 0.001
        ths8 = -np.pi + (np.arange(8) + 0.5) * 2 * np.pi / 8
        ths6 = -np.pi + (np.arange(6) + 0.5) * 2 * np.pi / 6
        if need and alpha is not None:
            ed = self._edge_dist(alpha, np.r_[ths8, ths6])
        else:
            ed = np.zeros(14)
        if amp > 0.001:
            for j, th in enumerate(ths8):
                rr = max(R * front_mod(th), ed[j] + 60.0, 140.0)
                p = ORIGIN + rr * np.float32([np.cos(th), np.sin(th)])
                # only once the front is actually outside the body does it light him
                vis = sstep_f(ed[j] - 20.0, ed[j] + 80.0, R * front_mod(th))
                near = 1.0 / (1.0 + (max(R - 250.0, 0.0) / 450.0) ** 2)
                k = 0.55 * amp * near * vis
                if k > 1e-3:
                    out.append(dict(pos=p, color=np.float32([0.55, 0.75, 1.0]), intensity=k,
                                    radius=260.0, rim=1.6, diffuse=0.0))
        if core > 0.001:
            for j, th in enumerate(ths6):
                rr = max(230.0, ed[8 + j] + 60.0)
                p = ORIGIN + rr * np.float32([np.cos(th), np.sin(th)])
                out.append(dict(pos=p, color=np.float32([0.5, 0.7, 1.0]), intensity=0.35 * core,
                                radius=220.0, rim=1.4, diffuse=0.0))
        fl = pulse(t, self.t_lamp + 0.01, 0.035)
        if fl > 0.01:
            # visible flash stays in the background; on him: a rim on the frame-right top edge
            out.append(dict(pos=LAMP, color=np.float32([1.0, 0.95, 0.85]), intensity=1.5 * fl,
                            radius=700.0, rim=2.0, diffuse=0.35))
        return out
