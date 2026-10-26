"""Tiny headless GLSL renderer (moderngl on Xvfb + Mesa llvmpipe).

Usage:
    from vfx.glfx import gl
    prog = gl.program('''
        void main(){ fragColor = vec4(vUV, 0., 1.); }
    ''')
    img = gl.render(prog, (w, h), uniforms={'uTime': 1.0}, textures={'uTex': np_array})

Conventions (important):
  * Returned arrays are float32 (h, w, 4) with row 0 = TOP of the image.
  * In shaders `vUV` is in [0,1] with (0,0) = top-left, and `px()` returns the pixel
    coordinate with y pointing DOWN, i.e. exactly numpy (x=col, y=row) coordinates.
    If you render a crop of the frame, pass uniform `uOffset` (top-left of the crop in
    frame pixels) and use `fpx()` to get full-frame pixel coordinates.
  * Textures uploaded from numpy (row 0 = top) are sampled with texture(t, vUV)
    directly - no flipping needed.
  * Values are not clamped: you can output HDR (>1) linear light.
Every fragment shader is automatically prefixed with PRELUDE (uniforms uRes, uOffset,
uTime, varyings, and a noise library: hash, value noise, simplex 3D, fbm, curl, etc).
"""
import os
import numpy as np

if 'DISPLAY' not in os.environ:
    os.environ['DISPLAY'] = ':99'

import moderngl

VERT = '''
#version 330
in vec2 in_vert;
out vec2 vUV;
void main(){
    vUV = vec2((in_vert.x + 1.0) * 0.5, 1.0 - (in_vert.y + 1.0) * 0.5);
    gl_Position = vec4(in_vert, 0.0, 1.0);
}
'''

PRELUDE = r'''
#version 330
in vec2 vUV;
out vec4 fragColor;
uniform vec2 uRes;      // render target size in px
uniform vec2 uOffset;   // top-left of this render target inside the full 1080x1440 frame
uniform float uTime;
#define PI 3.14159265359
#define TAU 6.28318530718
vec2 px(){ return vec2(gl_FragCoord.x, uRes.y - gl_FragCoord.y); }
vec2 fpx(){ return px() + uOffset; }
float saturate(float x){ return clamp(x, 0.0, 1.0); }
vec3 saturate(vec3 x){ return clamp(x, 0.0, 1.0); }
mat2 rot2(float a){ float c = cos(a), s = sin(a); return mat2(c, -s, s, c); }
float hash11(float p){ p = fract(p * .1031); p *= p + 33.33; p *= p + p; return fract(p); }
float hash21(vec2 p){ vec3 p3 = fract(vec3(p.xyx) * .1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }
vec2 hash22(vec2 p){ vec3 p3 = fract(vec3(p.xyx) * vec3(.1031, .1030, .0973)); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.xx + p3.yz) * p3.zy); }
vec3 hash33(vec3 p3){ p3 = fract(p3 * vec3(.1031, .1030, .0973)); p3 += dot(p3, p3.yxz + 33.33); return fract((p3.xxy + p3.yxx) * p3.zyx); }
float vnoise(vec2 p){ vec2 i = floor(p), f = fract(p); vec2 u = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash21(i), hash21(i + vec2(1, 0)), u.x), mix(hash21(i + vec2(0, 1)), hash21(i + vec2(1, 1)), u.x), u.y); }
// simplex 3D (Ashima / Stefan Gustavson), returns [-1,1]
vec3 mod289(vec3 x){ return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec4 mod289(vec4 x){ return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec4 permute(vec4 x){ return mod289(((x * 34.0) + 1.0) * x); }
vec4 taylorInvSqrt(vec4 r){ return 1.79284291400159 - 0.85373472095314 * r; }
float snoise(vec3 v){
  const vec2 C = vec2(1.0 / 6.0, 1.0 / 3.0); const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);
  vec3 i = floor(v + dot(v, C.yyy)); vec3 x0 = v - i + dot(i, C.xxx);
  vec3 g = step(x0.yzx, x0.xyz); vec3 l = 1.0 - g; vec3 i1 = min(g.xyz, l.zxy); vec3 i2 = max(g.xyz, l.zxy);
  vec3 x1 = x0 - i1 + C.xxx; vec3 x2 = x0 - i2 + C.yyy; vec3 x3 = x0 - D.yyy;
  i = mod289(i);
  vec4 p = permute(permute(permute(i.z + vec4(0.0, i1.z, i2.z, 1.0)) + i.y + vec4(0.0, i1.y, i2.y, 1.0)) + i.x + vec4(0.0, i1.x, i2.x, 1.0));
  float n_ = 0.142857142857; vec3 ns = n_ * D.wyz - D.xzx;
  vec4 j = p - 49.0 * floor(p * ns.z * ns.z); vec4 x_ = floor(j * ns.z); vec4 y_ = floor(j - 7.0 * x_);
  vec4 x = x_ * ns.x + ns.yyyy; vec4 y = y_ * ns.x + ns.yyyy; vec4 h = 1.0 - abs(x) - abs(y);
  vec4 b0 = vec4(x.xy, y.xy); vec4 b1 = vec4(x.zw, y.zw);
  vec4 s0 = floor(b0) * 2.0 + 1.0; vec4 s1 = floor(b1) * 2.0 + 1.0; vec4 sh = -step(h, vec4(0.0));
  vec4 a0 = b0.xzyw + s0.xzyw * sh.xxyy; vec4 a1 = b1.xzyw + s1.xzyw * sh.zzww;
  vec3 p0 = vec3(a0.xy, h.x); vec3 p1 = vec3(a0.zw, h.y); vec3 p2 = vec3(a1.xy, h.z); vec3 p3 = vec3(a1.zw, h.w);
  vec4 norm = taylorInvSqrt(vec4(dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3)));
  p0 *= norm.x; p1 *= norm.y; p2 *= norm.z; p3 *= norm.w;
  vec4 m = max(0.6 - vec4(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), 0.0); m = m * m;
  return 42.0 * dot(m * m, vec4(dot(p0, x0), dot(p1, x1), dot(p2, x2), dot(p3, x3)));
}
float fbm3(vec3 p, int oct){ float a = 0.5, s = 0.0; for (int k = 0; k < 8; k++){ if (k >= oct) break; s += a * snoise(p); p = p * 2.03 + vec3(1.7, 9.2, 3.1); a *= 0.5; } return s; }
float fbm2(vec2 p, int oct){ float a = 0.5, s = 0.0; for (int k = 0; k < 8; k++){ if (k >= oct) break; s += a * vnoise(p); p = rot2(0.5) * p * 2.02 + 7.3; a *= 0.5; } return s; }
float ridged3(vec3 p, int oct){ float a = 0.5, s = 0.0; for (int k = 0; k < 8; k++){ if (k >= oct) break; s += a * (1.0 - abs(snoise(p))); p *= 2.07; a *= 0.5; } return s; }
vec2 curl2(vec2 p, float t){ float e = 0.01;
  float n1 = snoise(vec3(p.x, p.y + e, t)), n2 = snoise(vec3(p.x, p.y - e, t));
  float n3 = snoise(vec3(p.x + e, p.y, t)), n4 = snoise(vec3(p.x - e, p.y, t));
  return vec2(n1 - n2, -(n3 - n4)) / (2.0 * e); }
// physically-flavoured blackbody-ish ramp, x in [0,1+] -> HDR linear colour
vec3 fireRamp(float x){ x = max(x, 0.0);
  return vec3(1.6, 0.55, 0.10) * smoothstep(0.0, 0.35, x) + vec3(0.9, 0.8, 0.35) * smoothstep(0.35, 0.8, x) + vec3(1.5, 1.5, 1.4) * smoothstep(0.8, 1.3, x); }
'''


class GLFX:
    def __init__(self):
        self.ctx = moderngl.create_standalone_context(require=330)
        self.vbo = self.ctx.buffer(np.array([-1, -1, 3, -1, -1, 3], dtype='f4').tobytes())
        self._progs = {}
        self._fbos = {}

    def program(self, frag_src):
        key = hash(frag_src)
        if key not in self._progs:
            src = PRELUDE + '\n' + frag_src
            prog = self.ctx.program(vertex_shader=VERT, fragment_shader=src)
            vao = self.ctx.vertex_array(prog, [(self.vbo, '2f', 'in_vert')])
            self._progs[key] = (prog, vao)
        return self._progs[key]

    def texture(self, arr, linear=True, repeat=False):
        """numpy (h,w), (h,w,c) float32/uint8 -> moderngl texture (caller should .release())."""
        arr = np.ascontiguousarray(arr)
        if arr.ndim == 2:
            arr = arr[..., None]
        h, w, c = arr.shape
        dtype = 'f4' if arr.dtype != np.uint8 else 'f1'
        if dtype == 'f4':
            arr = arr.astype(np.float32)
        tex = self.ctx.texture((w, h), c, arr.tobytes(), dtype=dtype)
        f = moderngl.LINEAR if linear else moderngl.NEAREST
        tex.filter = (f, f)
        tex.repeat_x = tex.repeat_y = repeat
        return tex

    def _fbo(self, w, h):
        k = (w, h)
        if k not in self._fbos:
            tex = self.ctx.texture((w, h), 4, dtype='f4')
            self._fbos[k] = (self.ctx.framebuffer(color_attachments=[tex]), tex)
        return self._fbos[k][0]

    def render(self, prog, size, uniforms=None, textures=None, offset=(0, 0), time=None):
        """Render a full-target quad. size=(w,h). Returns float32 (h, w, 4), row 0 = top.

        textures: dict name -> numpy array or moderngl.Texture. numpy arrays are uploaded
        and released automatically.
        """
        prog, vao = prog
        w, h = int(size[0]), int(size[1])
        fbo = self._fbo(w, h)
        fbo.use()
        fbo.clear(0, 0, 0, 0)
        if 'uRes' in prog:
            prog['uRes'].value = (float(w), float(h))
        if 'uOffset' in prog:
            prog['uOffset'].value = (float(offset[0]), float(offset[1]))
        if time is not None and 'uTime' in prog:
            prog['uTime'].value = float(time)
        for k, v in (uniforms or {}).items():
            if k in prog:
                if isinstance(v, np.ndarray):
                    prog[k].write(v.astype('f4').tobytes())
                else:
                    prog[k].value = v
        temp = []
        for unit, (k, v) in enumerate((textures or {}).items()):
            if not isinstance(v, moderngl.Texture):
                v = self.texture(v)
                temp.append(v)
            v.use(unit)
            if k in prog:
                prog[k].value = unit
        vao.render(moderngl.TRIANGLES)
        data = np.frombuffer(fbo.read(components=4, dtype='f4'), dtype=np.float32)
        for t in temp:
            t.release()
        return data.reshape(h, w, 4)[::-1].copy()


SPRITE_VERT = '''
#version 330
in vec2 in_corner;            // per-vertex quad corner in [-1,1]
in vec2 p0;                   // per-instance start (frame px)
in vec2 p1;                   // per-instance end (frame px); == p0 for round sprites
in float radius;              // per-instance half-width in px (gaussian sigma ~ radius/2)
in vec4 color;                // per-instance rgb (linear HDR) * intensity, a = opacity weight
uniform vec2 uRes;
uniform vec2 uOffset;
out vec2 vLocal;              // (along, across) in units of radius
out float vLen;
out vec4 vColor;
void main(){
    vec2 d = p1 - p0; float L = length(d);
    vec2 ax = L > 1e-3 ? d / L : vec2(1.0, 0.0);
    vec2 ay = vec2(-ax.y, ax.x);
    float r = max(radius, 0.5);
    float along = in_corner.x * (L * 0.5 + 2.0 * r);
    float across = in_corner.y * 2.0 * r;
    vec2 c = (p0 + p1) * 0.5 + ax * along + ay * across - uOffset;
    vLocal = vec2(along, across) / r; vLen = L * 0.5 / r; vColor = color;
    vec2 ndc = vec2(c.x / uRes.x * 2.0 - 1.0, 1.0 - c.y / uRes.y * 2.0);
    gl_Position = vec4(ndc, 0.0, 1.0);
}
'''
SPRITE_FRAG = '''
#version 330
in vec2 vLocal; in float vLen; in vec4 vColor;
uniform int uShape;   // 0 gaussian, 1 soft disc, 2 sharp core + halo (spark)
out vec4 fragColor;
void main(){
    float ax = max(abs(vLocal.x) - vLen, 0.0);
    float d2 = ax * ax + vLocal.y * vLocal.y;
    float w;
    if (uShape == 0) w = exp(-2.0 * d2);
    else if (uShape == 1) w = smoothstep(1.0, 0.7, sqrt(d2));
    else w = exp(-6.0 * d2) + 0.25 * exp(-0.8 * d2);
    fragColor = vec4(vColor.rgb * w, vColor.a * w);
}
'''


def _sprites(self, size, p0, p1=None, radius=2.0, color=(1, 1, 1, 1), offset=(0, 0), shape=0):
    """Additively splat N gaussian sprites / motion streaks into a float RGBA image.

    p0, p1: (N,2) frame-pixel positions (p1 = streak end, None for round sprites)
    radius: scalar or (N,) half-width in px; color: (4,) or (N,4) linear HDR rgb + weight.
    Returns (h, w, 4) float32, row 0 = top. rgb is additive emission; a = coverage-ish weight.
    """
    p0 = np.asarray(p0, np.float32).reshape(-1, 2)
    n = len(p0)
    w, h = int(size[0]), int(size[1])
    if n == 0:
        return np.zeros((h, w, 4), np.float32)
    p1 = p0 if p1 is None else np.asarray(p1, np.float32).reshape(-1, 2)
    radius = np.broadcast_to(np.asarray(radius, np.float32), (n,)).reshape(n, 1)
    color = np.broadcast_to(np.asarray(color, np.float32), (n, 4))
    inst = np.ascontiguousarray(np.concatenate([p0, p1, radius, color], 1).astype('f4'))
    key = '__sprite__'
    if key not in self._progs:
        prog = self.ctx.program(vertex_shader=SPRITE_VERT, fragment_shader=SPRITE_FRAG)
        quad = self.ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1], 'f4').tobytes())
        self._progs[key] = (prog, quad)
    prog, quad = self._progs[key]
    ibuf = self.ctx.buffer(inst.tobytes())
    vao = self.ctx.vertex_array(prog, [(quad, '2f', 'in_corner'),
                                       (ibuf, '2f 2f 1f 4f/i', 'p0', 'p1', 'radius', 'color')])
    fbo = self._fbo(w, h)
    fbo.use()
    fbo.clear(0, 0, 0, 0)
    prog['uRes'].value = (float(w), float(h))
    prog['uOffset'].value = (float(offset[0]), float(offset[1]))
    prog['uShape'].value = int(shape)
    self.ctx.enable(moderngl.BLEND)
    self.ctx.blend_func = moderngl.ONE, moderngl.ONE
    vao.render(moderngl.TRIANGLES, instances=n)
    self.ctx.disable(moderngl.BLEND)
    data = np.frombuffer(fbo.read(components=4, dtype='f4'), dtype=np.float32)
    vao.release()
    ibuf.release()
    return data.reshape(h, w, 4)[::-1].copy()


GLFX.sprites = _sprites


class _Lazy:
    _inst = None

    def __getattr__(self, name):
        if _Lazy._inst is None:
            _Lazy._inst = GLFX()
        return getattr(_Lazy._inst, name)


gl = _Lazy()
