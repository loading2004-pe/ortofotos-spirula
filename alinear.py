"""Alinear un .ply limpiado (p. ej. girado en SuperSplat) con su modelo original de Spirula, para
reutilizar las cámaras que calculó Spirula (plano, lado y vertical reales de la puerta)."""
import itertools
import numpy as np

try:
    from numba import njit
except Exception:          # pragma: no cover
    def njit(*a, **k):
        return (lambda f: f) if not (a and callable(a[0])) else a[0]


def _rotations24():
    R = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            M = np.zeros((3, 3))
            for i, (p, s) in enumerate(zip(perm, signs)):
                M[i, p] = s
            if np.linalg.det(M) > 0:
                R.append(M)
    return R


@njit(cache=True)
def _nn_grid(Q, keys_sorted, order, starts, counts, P, lo, h, dims):
    out = np.empty(len(Q))
    idx = np.empty(len(Q), np.int64)
    for i in range(len(Q)):
        cx = int((Q[i, 0] - lo[0]) / h)
        cy = int((Q[i, 1] - lo[1]) / h)
        cz = int((Q[i, 2] - lo[2]) / h)
        best = 1e30
        bi = -1
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                for dz in range(-1, 2):
                    x, y, z = cx + dx, cy + dy, cz + dz
                    if x < 0 or y < 0 or z < 0 or x >= dims[0] or y >= dims[1] or z >= dims[2]:
                        continue
                    key = (x * dims[1] + y) * dims[2] + z
                    # búsqueda binaria de la celda
                    a, b = 0, len(keys_sorted)
                    while a < b:
                        m = (a + b) // 2
                        if keys_sorted[m] < key:
                            a = m + 1
                        else:
                            b = m
                    if a < len(keys_sorted) and keys_sorted[a] == key:
                        s = starts[a]
                        for j in range(s, s + counts[a]):
                            k = order[j]
                            d = (P[k, 0] - Q[i, 0]) ** 2 + (P[k, 1] - Q[i, 1]) ** 2 + (P[k, 2] - Q[i, 2]) ** 2
                            if d < best:
                                best = d
                                bi = k
        out[i] = np.sqrt(best) if bi >= 0 else 1e15
        idx[i] = bi
    return out, idx


class Grid:
    def __init__(self, P, h):
        self.P = np.ascontiguousarray(P, np.float64)
        self.lo = P.min(0) - h
        self.h = float(h)
        self.dims = (np.ceil((P.max(0) + h - self.lo) / h).astype(np.int64) + 1)
        c = ((P - self.lo) / h).astype(np.int64)
        keys = (c[:, 0] * self.dims[1] + c[:, 1]) * self.dims[2] + c[:, 2]
        self.order = np.argsort(keys, kind="stable")
        ks = keys[self.order]
        self.keys, self.starts, self.counts = np.unique(ks, return_index=True, return_counts=True)

    def query(self, Q):
        return _nn_grid(np.ascontiguousarray(Q, np.float64), self.keys, self.order, self.starts,
                        self.counts, self.P, self.lo, self.h, self.dims)


def _kabsch(A, B):
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, _S, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cb - R @ ca


def _icp(A, B, g_coarse, g_fine, R, t, iters, fine_from):
    for it in range(iters):
        Q = A @ R.T + t
        d, k = (g_coarse if it < fine_from else g_fine).query(Q)
        ok = d < np.percentile(d, 70)
        if ok.sum() < 30:
            break
        dR, dt = _kabsch(Q[ok], B[k[ok]])
        R, t = dR @ R, dR @ t + dt
    return R, t


def register(clean_pts, orig_pts, log=None):
    """Transformación rígida (R, t) con orig ≈ R·clean + t. Devuelve (R, t, error) o None.
    error = mediana de distancia / tamaño del modelo (bien alineado: < 0.006).
    Prueba los 24 giros de 90° (lo habitual al enderezar en SuperSplat) y afina con ICP."""
    rng = np.random.default_rng(0)
    A = clean_pts[rng.choice(len(clean_pts), min(12000, len(clean_pts)), replace=False)]
    As = A[rng.choice(len(A), min(1500, len(A)), replace=False)]
    B = orig_pts[rng.choice(len(orig_pts), min(200000, len(orig_pts)), replace=False)]
    lo, hi = np.percentile(A, 2, 0), np.percentile(A, 98, 0)
    size = float(np.linalg.norm(hi - lo))
    blo, bhi = np.percentile(B, 1, 0), np.percentile(B, 99, 0)
    bsize = float(np.linalg.norm(bhi - blo))
    if size <= 0 or bsize <= 0 or size > 3 * bsize:
        return None
    gB = Grid(B, 0.02 * size)
    gBc = Grid(B, 0.08 * size)
    cands = []
    for R0 in _rotations24():                       # etapa rápida: pocos puntos, pocas vueltas
        for t0 in _translation_seeds(As @ R0.T, B):
            R, t = _icp(As, B, gBc, gBc, R0.copy(), t0.copy(), 5, 99)
            d, _k = gBc.query(As @ R.T + t)
            cands.append((float(np.median(np.minimum(d, 1e3))), R, t))
    cands.sort(key=lambda c: c[0])
    best = None
    for _e, R, t in cands[:4]:                      # afinado de los mejores
        R, t = _icp(A, B, gBc, gB, R, t, 25, 6)
        d, _k = gB.query(A @ R.T + t)
        err = float(np.median(np.minimum(d, 1e3))) / size
        inl = float((d < 0.01 * size).mean())
        if best is None or err < best[2]:
            best = (R, t, err, inl)
    if log and best:
        log(f"[alinear] error {best[2]:.4f}, coinciden {best[3] * 100:.0f}%")
    return best


def is_good(res):
    """¿Alineado de verdad? (el modelo correcto da ~0.002–0.003 y >85% de puntos que coinciden)."""
    return bool(res) and res[2] < 0.005 and res[3] > 0.75


def _translation_seeds(A1, B, n=6):
    ca = np.median(A1, 0)
    seeds = [np.median(B, 0) - ca]
    rng = np.random.default_rng(1)
    for p in B[rng.choice(len(B), n - 1, replace=False)]:
        seeds.append(p - ca)
    return seeds


def cams_into_clean(cams, ups, R, t):
    """Cámaras del modelo original llevadas al sistema del .ply limpio."""
    Ri = R.T
    return (cams - t) @ Ri.T, ups @ Ri.T
