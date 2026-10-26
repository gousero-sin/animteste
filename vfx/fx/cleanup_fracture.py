"""Impact fracture pattern for the bathroom -> studio shatter (built once, cached on disk).

  * polar 'spider-web' around the impact ORIGIN: radial cracks that continue ring to ring and
    branch as the circumference grows, wobbly concentric rings. Cell size grows with distance
    from the impact AND with a rough room-depth factor (small near the far room corner, large
    near the frame edges where the walls are closest to the camera): ~38 px near his head,
    ~55 px at the room corner, 120-165 px in the top corners.
  * the walls break along their real grout joints: every cell is split by the tile grout lines
    found with an LSD line detector on the clean plate (merged, deduplicated).
  * the PVC-slat ceiling breaks into long strips: ceiling cells are split along real slat seams.
  * some neighbouring cells stay joined (large slabs), some get a corner chipped off (splinters).
  * every polygon edge carries a crack type (radial / grout / slat / concentric / splinter)
    which the shader uses to light radial cracks first (they run ahead of the front).
"""
import os
import numpy as np
import cv2
from .. import config as C
from . import cleanup_plates as CP

W, H = C.W, C.H
CACHE = os.path.join(C.DATA, 'cache', 'cleanup')
VERSION = 3

# crack types
T_BORDER, T_RADIAL, T_CONC, T_GROUT, T_SLAT, T_SPLINTER = 0, 1, 2, 3, 4, 5
CORNER = np.float32([770.0, 600.0])          # far room corner (walls + ceiling meet)


def ceil_y(x):
    """y of the wall/ceiling junction (crown moulding) at column x."""
    x = np.asarray(x, np.float64)
    return np.where(x < 770.0, 230.0 + (590.0 - 230.0) * x / 770.0, 590.0 - 50.0 * (x - 770.0) / 310.0)


def depth_scale(p):
    p = np.asarray(p, np.float64)
    x, y = p[..., 0], p[..., 1]
    return np.clip(0.75 + 0.0009 * np.abs(x - CORNER[0]) + 0.0006 * np.maximum(0.0, 650.0 - y), 0.70, 1.35)


def cell_size(p, origin):
    r = np.hypot(p[..., 0] - origin[0], p[..., 1] - origin[1])
    return (40.0 + 0.10 * r) * depth_scale(p)


# ------------------------------------------------------------------ polygon helpers
def clip_half(poly, types, n, c, ctype):
    """Keep the part of polygon with n.p <= c (Sutherland-Hodgman, with edge-type tracking)."""
    out_p, out_t = [], []
    N = len(poly)
    for i in range(N):
        P, Q = poly[i], poly[(i + 1) % N]
        tP = types[i]
        dP = n[0] * P[0] + n[1] * P[1] - c
        dQ = n[0] * Q[0] + n[1] * Q[1] - c
        if dP <= 0:
            out_p.append(P); out_t.append(tP)
            if dQ > 0:
                I = P + (Q - P) * (dP / (dP - dQ))
                out_p.append(I); out_t.append(ctype)
        elif dQ <= 0:
            I = P + (Q - P) * (dP / (dP - dQ))
            out_p.append(I); out_t.append(tP)
    return out_p, out_t


def area(poly):
    p = np.asarray(poly)
    return 0.5 * float(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]))


def split(poly, types, n, c, ctype, min_area=140.0):
    a, ta = clip_half(poly, types, n, c, ctype)
    b, tb = clip_half(poly, types, (-n[0], -n[1]), -c, ctype)
    if len(a) < 3 or len(b) < 3 or abs(area(a)) < min_area or abs(area(b)) < min_area:
        return None
    return (a, ta), (b, tb)


def chord(poly, n, c):
    """Intersection points of line n.p=c with the polygon boundary."""
    pts = []
    N = len(poly)
    for i in range(N):
        P, Q = poly[i], poly[(i + 1) % N]
        dP = n[0] * P[0] + n[1] * P[1] - c
        dQ = n[0] * Q[0] + n[1] * Q[1] - c
        if (dP <= 0) != (dQ <= 0):
            pts.append(P + (Q - P) * (dP / (dP - dQ)))
    return pts


# ------------------------------------------------------------------ line structure of the room
def _room_lines():
    s = (np.clip(CP.A.lin_to_srgb(CP.plate(3)), 0, 1) * 255).astype(np.uint8)
    g = cv2.cvtColor(s, cv2.COLOR_RGB2GRAY)
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    L = lsd.detect(g)[0].reshape(-1, 4).astype(np.float64)
    ln = np.hypot(L[:, 2] - L[:, 0], L[:, 3] - L[:, 1])
    L, ln = L[ln > 45], ln[ln > 45]
    mid = (L[:, :2] + L[:, 2:]) / 2
    ceiling = mid[:, 1] < ceil_y(mid[:, 0]) - 22
    segs = []
    for (x0, y0, x1, y1), isc in zip(L, ceiling):
        d = np.array([x1 - x0, y1 - y0]); d /= np.linalg.norm(d)
        if d[0] < 0 or (abs(d[0]) < 1e-9 and d[1] < 0):
            d = -d
        nrm = np.array([-d[1], d[0]])
        rho = nrm @ np.array([x0, y0])
        u0, u1 = sorted([d @ np.array([x0, y0]), d @ np.array([x1, y1])])
        segs.append([d, nrm, rho, u0, u1, bool(isc)])
    # merge near-collinear segments (LSD returns both edges of a grout line, and broken pieces)
    merged = []
    used = [False] * len(segs)
    order = np.argsort([-(s_[4] - s_[3]) for s_ in segs])
    for a in order:
        if used[a]:
            continue
        d, nrm, rho, u0, u1, isc = segs[a]
        used[a] = True
        changed = True
        while changed:
            changed = False
            for b in order:
                if used[b] or segs[b][5] != isc:
                    continue
                db, nb, rb, v0, v1, _ = segs[b]
                if abs(d @ db) < np.cos(np.radians(2.5)):
                    continue
                pb0 = rb * nb + v0 * db
                pb1 = rb * nb + v1 * db
                if abs(nrm @ pb0 - rho) > 6 or abs(nrm @ pb1 - rho) > 6:
                    continue
                w0, w1 = sorted([d @ pb0, d @ pb1])
                if w0 > u1 + 40 or w1 < u0 - 40:
                    continue
                L0, L1 = u1 - u0, w1 - w0
                rho = (rho * L0 + (nrm @ (pb0 + pb1) / 2) * L1) / (L0 + L1)
                u0, u1 = min(u0, w0), max(u1, w1)
                used[b] = True
                changed = True
        merged.append(dict(d=d, n=nrm, rho=rho, u0=u0, u1=u1, ceiling=isc))
    return merged


# ------------------------------------------------------------------ the pattern
def _build(origin, seed):
    rng = np.random.default_rng(seed)
    O = np.asarray(origin, np.float64)
    R_END = 1420.0
    # ---- polar web
    n0 = 6
    A = [np.sort((np.arange(n0) + rng.uniform(-0.25, 0.25, n0)) * 2 * np.pi / n0)]
    Rr = [np.full(n0, 30.0) * rng.uniform(0.85, 1.15, n0)]
    maps = []          # maps[k][j] = index in ring k+1 of crack j of ring k
    while Rr[-1].min() < R_END:
        a, r = A[-1], Rr[-1]
        n = len(a)
        # inherited cracks wobble a little
        gaps = np.diff(np.r_[a, a[0] + 2 * np.pi])
        na = a + rng.normal(0, 0.10, n) * np.minimum(gaps, np.roll(gaps, 1)) * 0.5
        # outer radius per crack: ring spacing follows the local cell size, with a SMOOTH
        # angular wobble (no per-crack random walk -> rings stay clean arcs, not zig-zags)
        p = O + r[:, None] * np.stack([np.cos(a), np.sin(a)], 1)
        ph = rng.uniform(0, 2 * np.pi, 3)
        wob = (rng.uniform(0.8, 1.2) + 0.20 * np.sin(2 * na + ph[0]) + 0.10 * np.sin(5 * na + ph[1])
               + 0.05 * np.sin(9 * na + ph[2]))
        nr = r + 1.25 * cell_size(p, O) * wob
        # branching: insert new cracks where the outer arc is too long for the local cell size
        new_a, new_r, m = [], [], []
        for j in range(n):
            a0, a1 = na[j], na[(j + 1) % n] + (2 * np.pi if j == n - 1 else 0)
            r0, r1 = nr[j], nr[(j + 1) % n]
            m.append(len(new_a))
            new_a.append(na[j]); new_r.append(r0)
            pm = O + 0.5 * (r0 + r1) * np.array([np.cos(0.5 * (a0 + a1)), np.sin(0.5 * (a0 + a1))])
            arc = 0.5 * (r0 + r1) * (a1 - a0)
            k = int(np.floor(arc / (1.55 * float(cell_size(pm, O)))))
            for q in range(1, k + 1):
                f = q / (k + 1) + rng.uniform(-0.15, 0.15) / (k + 1)
                new_a.append(a0 + f * (a1 - a0))
                new_r.append(r0 + f * (r1 - r0))
        maps.append(np.array(m))
        A.append(np.array(new_a))
        Rr.append(np.array(new_r))
    rings = [O + r[:, None] * np.stack([np.cos(a), np.sin(a)], 1) for a, r in zip(A, Rr)]
    polys = []
    # centre cell
    polys.append(([q for q in rings[0]], [T_CONC] * len(rings[0]), 'cell'))
    nb = len(rings) - 1
    # 1) angular grouping per band: mostly single cells, some 2-wide slabs (outer bands)
    groups = []
    for k in range(nb):
        n = len(rings[k])
        g, j = [], 0
        while j < n:
            w = 2 if (k >= 3 and rng.random() < 0.12 and j < n - 1) else 1
            g.append((j, w))
            j += w
        groups.append(g)
    # 2) radial merges: a single cell joins the single cell just outside it when no crack
    #    branches in between -> long radial slabs, concentric rings become broken/irregular
    starts = [dict((j, w) for j, w in g) for g in groups]
    merged = set()                      # (k, j): cell of band k at index j absorbs band k+1 cell
    absorbed = set()
    for k in range(1, nb - 1):
        n = len(rings[k])
        m = maps[k]
        for j, w in groups[k]:
            if w != 1 or (k, j) in absorbed:
                continue
            j2 = (j + 1) % n
            mo0 = m[j]
            mo1 = m[j2] if j2 != 0 else m[0] + len(rings[k + 1])
            if mo1 - mo0 != 1 or starts[k + 1].get(mo0) != 1:
                continue
            if rng.random() < (0.22 if k >= 2 else 0.0):
                merged.add((k, j))
                absorbed.add((k + 1, mo0))

    def ring_span(kk, i0, i1):
        nn = len(rings[kk])
        return [rings[kk][q % nn] for q in range(i0, i1 + 1)]

    def outer_idx(kk, j, j1):
        n = len(rings[kk]); nn = len(rings[kk + 1]); m = maps[kk]
        return m[j], (m[j1 % n] if j1 < n else m[0] + nn)

    for k in range(nb):
        for j, w in groups[k]:
            if (k, j) in absorbed:
                continue
            j1 = j + w
            inner = ring_span(k, j, j1)
            o0, o1 = outer_idx(k, j, j1)
            if (k, j) in merged:
                # spans bands k and k+1: sides pass through ring k+1 at o0 / o1
                mid = rings[k + 1]
                nn = len(mid)
                p0, p1 = outer_idx(k + 1, o0 % nn, o0 % nn + 1)
                if o0 % nn + 1 >= nn:
                    p0, p1 = outer_idx(k + 1, nn - 1, nn)
                outer = ring_span(k + 2, p0, p1)
                pts = inner + [mid[o1 % nn]] + outer[::-1] + [mid[o0 % nn]]
                ty = [T_CONC] * w + [T_RADIAL, T_RADIAL] + [T_CONC] * (len(outer) - 1) + [T_RADIAL, T_RADIAL]
            else:
                outer = ring_span(k + 1, o0, o1)
                pts = inner + outer[::-1]
                ty = [T_CONC] * w + [T_RADIAL] + [T_CONC] * (len(outer) - 1) + [T_RADIAL]
            polys.append((pts, ty, 'cell'))
    # ---- clip to the frame (+ margin) and drop outside cells
    M = 2.0
    out = []
    for pts, ty, kind in polys:
        p, t = [np.asarray(q, np.float64) for q in pts], list(ty)
        for (n_, c_) in (((-1, 0), M), ((1, 0), W + M), ((0, -1), M), ((0, 1), H + M)):
            p, t = clip_half(p, t, n_, c_, T_BORDER)
            if len(p) < 3:
                break
        if len(p) >= 3 and abs(area(p)) > 30:
            if area(p) < 0:
                p, t = p[::-1], t[::-1]
                t = t[1:] + t[:1]          # edge i now runs p[i] -> p[i+1] after reversal
            out.append((p, t, kind))
    polys = out
    # ---- walls break along grout joints, ceiling into slat strips
    lines = _room_lines()
    walls = [l for l in lines if not l['ceiling'] and l['u1'] - l['u0'] > 70]
    slats = [l for l in lines if l['ceiling'] and l['u1'] - l['u0'] > 60]
    res = []
    for pts, ty, kind in polys:
        work = [(pts, ty)]
        for l in walls:
            nxt = []
            for p, t in work:
                ch = chord(p, l['n'], l['rho'])
                ok = False
                if len(ch) == 2:
                    u = sorted([l['d'] @ ch[0], l['d'] @ ch[1]])
                    inside = max(0.0, min(u[1], l['u1'] + 20) - max(u[0], l['u0'] - 20))
                    ok = (u[1] - u[0]) > 10 and inside > 0.85 * (u[1] - u[0])
                if ok:
                    sp = split(p, t, l['n'], l['rho'], T_GROUT, min_area=160.0)
                    if sp is not None:
                        nxt += [sp[0], sp[1]]
                        continue
                nxt.append((p, t))
            work = nxt
        for p, t in work:
            res.append((p, t, kind))
    polys = res
    res = []
    if slats:
        smid = np.array([l['rho'] * l['n'] + 0.5 * (l['u0'] + l['u1']) * l['d'] for l in slats])
    for pts, ty, kind in polys:
        c = np.mean(pts, 0)
        if not slats or c[1] > ceil_y(c[0]) - 10:
            res.append((pts, ty, kind))
            continue
        # strips ~38-60 px wide along real slat seams
        k = int(np.argmin(np.hypot(*(smid - c).T)))
        dn = slats[k]['n']
        proj = [dn @ q for q in pts]
        lo, hi = min(proj), max(proj)
        wdt = rng.uniform(50, 80) * float(depth_scale(c))
        work = [(pts, ty)]
        off = lo + wdt * rng.uniform(0.6, 1.0)
        while off < hi - 0.5 * wdt:
            # snap to the nearest real slat seam with this direction
            cand = [l for l in slats if abs(l['n'] @ dn) > np.cos(np.radians(4))]
            if cand:
                best = min(cand, key=lambda l: abs(np.sign(l['n'] @ dn) * l['rho'] - off)
                           + 0.02 * np.hypot(*(l['rho'] * l['n'] - c)))
                cn = best['n'] * np.sign(best['n'] @ dn)
                cc = best['rho'] * np.sign(best['n'] @ dn)
            else:
                cn, cc = dn, off
            nxt = []
            for p, t in work:
                sp = split(p, t, cn, cc, T_SLAT, min_area=200.0)
                nxt += [sp[0], sp[1]] if sp is not None else [(p, t)]
            work = nxt
            off += wdt
        for p, t in work:
            res.append((p, t, kind))
    polys = res
    # ---- splinters: chip a corner off some cells (more near the impact)
    res = []
    for pts, ty, kind in polys:
        c = np.mean(pts, 0)
        r = float(np.hypot(*(c - O)))
        pr = 0.30 if r < 500 else 0.16
        if len(pts) >= 4 and abs(area(pts)) > 900 and rng.random() < pr:
            i = int(rng.integers(len(pts)))
            P = np.asarray(pts[i])
            Pa, Pb = np.asarray(pts[i - 1]), np.asarray(pts[(i + 1) % len(pts)])
            fa, fb = rng.uniform(0.25, 0.5, 2)
            qa, qb = P + (Pa - P) * fa, P + (Pb - P) * fb
            d = qb - qa
            nrm = np.array([-d[1], d[0]]); nrm /= (np.linalg.norm(nrm) + 1e-9)
            cc = nrm @ qa
            if nrm @ P > cc:
                nrm, cc = -nrm, -cc
            sp = split(pts, ty, nrm, cc, T_SPLINTER, min_area=60.0)
            if sp is not None:
                (p1, t1), (p2, t2) = sp
                # the piece that contains P is the splinter
                if abs(area(p1)) < abs(area(p2)):
                    res += [(p1, t1, 'splinter'), (p2, t2, kind)]
                else:
                    res += [(p2, t2, 'splinter'), (p1, t1, kind)]
                continue
        res.append((pts, ty, kind))
    return res


def fracture(origin, seed=5):
    """List of (poly (n,2) float32, edge_types (n,) int8, is_splinter bool), cached."""
    path = os.path.join(CACHE, f'fracture_v{VERSION}_{int(origin[0])}_{int(origin[1])}_{seed}.npz')
    if os.path.exists(path):
        try:
            z = np.load(path)
            P, T, S, off = z['P'], z['T'], z['S'], z['off']
            return [(P[off[i]:off[i + 1]], T[off[i]:off[i + 1]], bool(S[i])) for i in range(len(S))]
        except Exception:
            pass
    polys = _build(origin, seed)
    P = np.concatenate([np.asarray(p, np.float32) for p, t, k in polys])
    T = np.concatenate([np.asarray(t, np.int8) for p, t, k in polys])
    S = np.array([k == 'splinter' for p, t, k in polys])
    off = np.r_[0, np.cumsum([len(p) for p, t, k in polys])]
    os.makedirs(CACHE, exist_ok=True)
    tmp = path + f'.{os.getpid()}.tmp.npz'
    np.savez(tmp, P=P, T=T, S=S, off=off)
    os.replace(tmp, path)
    z = np.load(path)       # every process uses the identical (cached) float32 data
    P, T, S, off = z['P'], z['T'], z['S'], z['off']
    return [(P[off[i]:off[i + 1]], T[off[i]:off[i + 1]], bool(S[i])) for i in range(len(S))]
