"""Render ortográfico de un modelo Gaussian Splatting (PLY de Spirula Studio).

- Detecta automáticamente el plano de la fachada (puerta / ventana) con RANSAC.
- Coloca una cámara ortográfica perpendicular a ese plano (equivale a FOV ~0°,
  mejor que el truco de FOV 1° + distancia focal 2000).
- Orienta la imagen con la vertical de la escena y luego la endereza
  midiendo los bordes rectos de la propia imagen (marcos, jambas, dinteles).
- Rasteriza los splats en CPU con numba (multi-hilo).
"""
from __future__ import annotations

import json
import math
import os
import struct
from dataclasses import dataclass

import numpy as np
from PIL import Image

try:
    from numba import njit, prange
except ImportError:  # pragma: no cover - fallback lento sin numba
    def njit(*a, **k):
        def deco(f):
            return f
        return deco if not (a and callable(a[0])) else a[0]
    prange = range

import sys as _sys
_CACHE = not getattr(_sys, "frozen", False)   # en el .exe no hay archivos fuente que cachear

SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
         -1.0925484305920792, 0.5462742152960396)
SH_C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658,
         0.3731763325901154, -0.4570457994644658, 1.445305721320277,
         -0.5900435899266435)

# Spirula: PLY exportado "Y-up" (SuperSplat) -> coordenadas nativas.
_YUP_TO_NATIVE = np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]], dtype=np.float64)


# ---------------------------------------------------------------------------
# Lectura de archivos
# ---------------------------------------------------------------------------
_PLY_TYPES = {"float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
              "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
              "short": "<i2", "int16": "<i2", "ushort": "<u2", "uint16": "<u2",
              "int": "<i4", "int32": "<i4", "uint": "<u4", "uint32": "<u4"}


def _read_ply_elements(path):
    """Lee todos los elementos de un PLY binario (vertex, chunk, sh...)."""
    with open(path, "rb") as f:
        header = []
        while True:
            raw = f.readline()
            if not raw:
                raise ValueError("Cabecera PLY inválida: " + path)
            line = raw.decode("ascii", "replace").strip()
            header.append(line)
            if line == "end_header":
                break
            if len(header) > 2000:
                raise ValueError("Cabecera PLY inválida: " + path)
        elements, comments = [], []
        for line in header:
            p = line.split()
            if not p:
                continue
            if p[0] == "format" and p[1] != "binary_little_endian":
                raise ValueError("Solo PLY binario little-endian (exporta .ply desde SuperSplat)")
            if p[0] == "comment":
                comments.append(line)
            elif p[0] == "element":
                elements.append([p[1], int(p[2]), []])
            elif p[0] == "property" and elements:
                if p[1] == "list":
                    raise ValueError("PLY con listas no soportado")
                elements[-1][2].append((p[2], _PLY_TYPES[p[1]]))
        data = {}
        for name, n, props in elements:
            data[name] = np.fromfile(f, dtype=np.dtype(props), count=n)
    return data, comments


def _decode_compressed(data):
    """PLY comprimido de SuperSplat / PlayCanvas (element chunk + packed_*)."""
    ch, v = data["chunk"], data["vertex"]
    n = len(v)
    ci = np.arange(n) // 256

    def lerp(lo, hi, t):
        return ch[lo][ci] + (ch[hi][ci] - ch[lo][ci]) * t

    def u111011(u):
        return (((u >> 21) & 2047) / 2047.0, ((u >> 11) & 1023) / 1023.0, (u & 2047) / 2047.0)

    pp = v["packed_position"].astype(np.uint32)
    px, py, pz = u111011(pp)
    xyz = np.stack([lerp("min_x", "max_x", px), lerp("min_y", "max_y", py),
                    lerp("min_z", "max_z", pz)], 1).astype(np.float64)
    ps = v["packed_scale"].astype(np.uint32)
    sx, sy, sz = u111011(ps)
    scale = np.stack([lerp("min_scale_x", "max_scale_x", sx), lerp("min_scale_y", "max_scale_y", sy),
                      lerp("min_scale_z", "max_scale_z", sz)], 1).astype(np.float32)
    pc = v["packed_color"].astype(np.uint32)
    r, g, b = ((pc >> 24) & 255) / 255.0, ((pc >> 16) & 255) / 255.0, ((pc >> 8) & 255) / 255.0
    a = (pc & 255) / 255.0
    if "min_r" in ch.dtype.names:
        r, g, b = lerp("min_r", "max_r", r), lerp("min_g", "max_g", g), lerp("min_b", "max_b", b)
    dc = ((np.stack([r, g, b], 1) - 0.5) / SH_C0).astype(np.float32)
    a = np.clip(a, 1e-6, 1 - 1e-6)
    opacity = (-np.log(1 / a - 1)).astype(np.float32)
    pr = v["packed_rotation"].astype(np.uint32)
    norm = 1.0 / (np.sqrt(2) * 0.5)
    qa = (((pr >> 20) & 1023) / 1023.0 - 0.5) * norm
    qb = (((pr >> 10) & 1023) / 1023.0 - 0.5) * norm
    qc = ((pr & 1023) / 1023.0 - 0.5) * norm
    qd = np.sqrt(np.clip(1 - (qa * qa + qb * qb + qc * qc), 0, 1))
    largest = pr >> 30
    rot = np.empty((n, 4))
    for k in range(4):
        m = largest == k
        comps = [qa[m], qb[m], qc[m]]
        comps.insert(k, qd[m])
        rot[m] = np.stack(comps, 1)
    rest = None
    if "sh" in data:
        sh = data["sh"]
        names = sorted(sh.dtype.names, key=lambda s: int(s.split("_")[-1]))
        arr = np.stack([sh[k] for k in names], 1).astype(np.float32)
        rest = ((arr + 0.5) / 256.0 - 0.5) * 8.0
    return xyz, dc, rest, opacity, scale, rot


def read_splat_ply(path: str) -> dict:
    """PLY de splats: el estándar (Spirula, SuperSplat 'PLY') o el comprimido de SuperSplat."""
    data, comments = _read_ply_elements(path)
    mip = any("SplatRenderMode: mip" in c for c in comments)
    yup = any("supersplat_y_up" in c for c in comments)
    if "chunk" in data and "packed_position" in (data["vertex"].dtype.names or ()):
        xyz, dc, rest, opacity, scale, rot = _decode_compressed(data)
    else:
        vtx = data["vertex"]
        names = vtx.dtype.names

        def col(k):
            return np.asarray(vtx[k], dtype=np.float32)

        xyz = np.stack([col("x"), col("y"), col("z")], 1).astype(np.float64)
        dc = np.stack([col("f_dc_0"), col("f_dc_1"), col("f_dc_2")], 1)
        rest_names = sorted([k for k in names if k.startswith("f_rest_")],
                            key=lambda s: int(s.split("_")[-1]))
        rest = np.stack([col(k) for k in rest_names], 1) if rest_names else None
        opacity = col("opacity")
        scale = np.stack([col("scale_0"), col("scale_1"), col("scale_2")], 1)
        rot = np.stack([col("rot_0"), col("rot_1"), col("rot_2"), col("rot_3")], 1)
    Rm = quat_to_mat(rot)
    if yup:
        xyz = xyz @ _YUP_TO_NATIVE.T
        Rm = np.einsum("ij,njk->nik", _YUP_TO_NATIVE, Rm)
        rest = None
    return dict(xyz=xyz, dc=dc, rest=rest, opacity=opacity, scale=scale,
                R=Rm.astype(np.float32), mip=mip)


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((len(q), 3, 3), dtype=np.float64)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def read_colmap_cameras(sparse_dir: str):
    """Centros y ejes 'arriba' de las cámaras (marco COLMAP / mundo)."""
    out_c, out_up = [], []
    bin_path = os.path.join(sparse_dir, "images.bin")
    txt_path = os.path.join(sparse_dir, "images.txt")
    poses = []
    if os.path.exists(bin_path):
        with open(bin_path, "rb") as f:
            (num,) = struct.unpack("<Q", f.read(8))
            for _ in range(num):
                f.read(4)
                q = struct.unpack("<4d", f.read(32))
                t = struct.unpack("<3d", f.read(24))
                f.read(4)
                while f.read(1) not in (b"\x00", b""):
                    pass
                (npts,) = struct.unpack("<Q", f.read(8))
                f.seek(24 * npts, 1)
                poses.append((q, t))
    elif os.path.exists(txt_path):
        with open(txt_path, encoding="utf-8", errors="replace") as f:
            lines = [l for l in f if l.strip() and not l.startswith("#")]
        for l in lines[::2]:
            p = l.split()
            poses.append((tuple(map(float, p[1:5])), tuple(map(float, p[5:8]))))
    for q, t in poses:
        R = quat_to_mat(np.array([q]))[0]          # mundo -> cámara
        c = -R.T @ np.array(t)
        out_c.append(c)
        out_up.append(-R[1])                       # eje -y de la cámara, en mundo
    if not out_c:
        return None, None
    return np.array(out_c), np.array(out_up)


def find_sparse_dir(workspace: str) -> str | None:
    best, best_size = None, -1
    for root, _dirs, files in os.walk(workspace):
        if "images.bin" in files or "images.txt" in files:
            fn = "images.bin" if "images.bin" in files else "images.txt"
            size = os.path.getsize(os.path.join(root, fn))
            if size > best_size:
                best, best_size = root, size
    return best


def read_scene_transform(run_dir: str):
    p = None
    d = run_dir
    for _ in range(4):          # la carpeta del .ply o sus padres (step-*.ckpt/..)
        cand = os.path.join(d, "scene_transform.json")
        if os.path.exists(cand):
            p = cand
            break
        d = os.path.dirname(d)
    if p is None:
        return None
    with open(p, encoding="utf-8") as f:
        j = json.load(f)
    if isinstance(j.get("train_from_world"), dict) and "matrix_4x4" in j["train_from_world"]:
        return np.array(j["train_from_world"]["matrix_4x4"], dtype=np.float64)
    if "matrix_4x4" in j:
        return np.array(j["matrix_4x4"], dtype=np.float64)
    for v in j.values():  # por si viene anidado
        if isinstance(v, dict) and "matrix_4x4" in v:
            return np.array(v["matrix_4x4"], dtype=np.float64)
    return None


# ---------------------------------------------------------------------------
# Geometría: plano de fachada y cámara
# ---------------------------------------------------------------------------
@dataclass
class OrthoCam:
    origin: np.ndarray   # punto sobre el plano
    right: np.ndarray
    up: np.ndarray
    normal: np.ndarray   # apunta hacia el observador
    u_range: tuple       # (min, max) en unidades de escena, eje right
    v_range: tuple       # eje up


def _unit(v):
    return v / max(np.linalg.norm(v), 1e-12)


def fit_facade_plane(xyz, alpha, scale, up_ref, cam_centers, rng, vertical=True,
                     iters=600):
    # modelos cortos (5000 pasos) tienen opacidades bajas: umbral adaptativo
    thr = min(0.4, float(np.percentile(alpha, 90)))
    keep = alpha >= thr
    pts = xyz[keep]
    if len(pts) < 100:
        pts = xyz
    lo, hi = np.percentile(pts, 1, axis=0), np.percentile(pts, 99, axis=0)
    pts = pts[np.all((pts >= lo) & (pts <= hi), axis=1)]
    if len(pts) > 150_000:
        pts = pts[rng.choice(len(pts), 150_000, replace=False)]
    extent = float(np.linalg.norm(hi - lo))
    tau = 0.008 * extent
    look_from = cam_centers.mean(0) if cam_centers is not None else None

    best_n, best_p, best_score = None, None, -1
    for _ in range(iters):
        a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        n /= nn
        if vertical and up_ref is not None and abs(float(n @ up_ref)) > 0.45:
            continue
        d = np.abs((pts - a) @ n)
        score = int(np.count_nonzero(d < tau))
        if score > best_score:
            best_n, best_p, best_score = n, a, score
    if best_n is None:  # sin plano vertical: PCA simple
        ctr = pts.mean(0)
        _, _, vt = np.linalg.svd(pts - ctr, full_matrices=False)
        best_n, best_p = vt[2], ctr
    # refinamiento por mínimos cuadrados sobre inliers
    inl = pts[np.abs((pts - best_p) @ best_n) < tau]
    if len(inl) > 50:
        ctr = inl.mean(0)
        _, _, vt = np.linalg.svd(inl - ctr, full_matrices=False)
        n = vt[2]
        if n @ best_n < 0:
            n = -n
        best_n, best_p = n, ctr
    # la normal mira hacia donde estuvieron las cámaras
    if look_from is not None and (look_from - best_p) @ best_n < 0:
        best_n = -best_n
    elif look_from is None and up_ref is not None:
        pass
    return _unit(best_n), best_p, inl if len(inl) > 50 else pts, extent


def guess_up_without_cams(model, alpha=None, rng=None):
    """Vertical de un modelo SIN cámaras (p. ej. un .ply limpiado en SuperSplat).
    Spirula/SuperSplat suelen dejar la vertical en -Y; algunos modelos la tienen en -Z.
    Se mira hacia dónde mira el plano dominante (la puerta/ventana): si mira a lo largo de Y,
    Y no puede ser la vertical y se usa Z."""
    rng = rng or np.random.default_rng(0)
    if alpha is None:
        alpha = 1 / (1 + np.exp(-model["opacity"]))
    P = model["xyz"][alpha.ravel() > 0.5]
    if len(P) > 150_000:
        P = P[rng.choice(len(P), 150_000, replace=False)]
    if len(P) < 100:
        return np.array([0.0, -1.0, 0.0])
    lo, hi = np.percentile(P, 2, 0), np.percentile(P, 98, 0)
    tau = 0.01 * float(np.linalg.norm(hi - lo))
    best, bn = -1, np.array([0.0, 0.0, 1.0])
    for _ in range(800):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        ln = np.linalg.norm(n)
        if ln < 1e-12:
            continue
        n = n / ln
        k = int((np.abs((P - a) @ n) < tau).sum())
        if k > best:
            best, bn = k, n
    if abs(bn[1]) < 0.6:
        return np.array([0.0, -1.0, 0.0])
    if abs(bn[2]) < 0.6:
        return np.array([0.0, 0.0, -1.0])
    return np.array([0.0, -1.0, 0.0])


def build_camera(model, cam_centers=None, cam_ups=None, vertical=True,
                 margin=0.03, seed=0) -> OrthoCam:
    rng = np.random.default_rng(seed)
    alpha = 1 / (1 + np.exp(-model["opacity"]))
    if cam_ups is not None and len(cam_ups):
        up_ref = _unit(cam_ups.mean(0))
    else:
        up_ref = guess_up_without_cams(model, alpha, rng)
    n, p0, inl, _ = fit_facade_plane(model["xyz"], alpha, model["scale"], up_ref,
                                     cam_centers, rng, vertical=vertical)
    up = up_ref - (up_ref @ n) * n
    if np.linalg.norm(up) < 1e-3:  # plano horizontal (planta): usar eje X
        up = np.array([1.0, 0, 0]) - n[0] * n
    up = _unit(up)
    right = _unit(np.cross(up, n))
    uv = np.stack([(inl - p0) @ right, (inl - p0) @ up], 1)
    lo = np.percentile(uv, 0.2, axis=0)
    hi = np.percentile(uv, 99.8, axis=0)
    pad = (hi - lo) * margin
    return OrthoCam(p0, right, up, n, (lo[0] - pad[0], hi[0] + pad[0]),
                    (lo[1] - pad[1], hi[1] + pad[1]))


def rotate_camera(cam: OrthoCam, deg: float) -> OrthoCam:
    """Gira la imagen 'deg' grados (positivo = antihorario) sobre la normal."""
    t = math.radians(deg)
    r = cam.right * math.cos(t) + cam.up * math.sin(t)
    u = -cam.right * math.sin(t) + cam.up * math.cos(t)
    # recalcula el encuadre para que el rectángulo original entre completo
    corners = [(a, b) for a in cam.u_range for b in cam.v_range]
    nu = [a * math.cos(t) + b * math.sin(t) for a, b in corners]
    nv = [-a * math.sin(t) + b * math.cos(t) for a, b in corners]
    return OrthoCam(cam.origin, r, u, cam.normal, (min(nu), max(nu)), (min(nv), max(nv)))


# ---------------------------------------------------------------------------
# Rasterizador
# ---------------------------------------------------------------------------
@njit(cache=_CACHE, nogil=True)
def _bin_strips(y0, y1, n_strips, strip_h):
    n = len(y0)
    counts = np.zeros(n_strips + 1, np.int64)
    for i in range(n):
        a = max(0, y0[i] // strip_h)
        b = min(n_strips - 1, y1[i] // strip_h)
        for s in range(a, b + 1):
            counts[s + 1] += 1
    for s in range(n_strips):
        counts[s + 1] += counts[s]
    idx = np.empty(counts[n_strips], np.int64)
    fill = counts[:-1].copy()
    for i in range(n):  # en orden de profundidad: se conserva por franja
        a = max(0, y0[i] // strip_h)
        b = min(n_strips - 1, y1[i] // strip_h)
        for s in range(a, b + 1):
            idx[fill[s]] = i
            fill[s] += 1
    return counts, idx


@njit(parallel=True, cache=_CACHE, nogil=True)
def _raster(W, H, strip_h, counts, idx, mx, my, ca, cb, cc, op, col,
            x0, x1, y0, y1):
    n_strips = len(counts) - 1
    img = np.zeros((H, W, 3), np.float32)
    T = np.ones((H, W), np.float32)
    for s in prange(n_strips):
        ys, ye = s * strip_h, min(H, (s + 1) * strip_h)
        remaining = (ye - ys) * W          # píxeles aún no opacos en la franja
        for k in range(counts[s], counts[s + 1]):
            if remaining <= 0:
                break
            i = idx[k]
            ya, yb = max(ys, y0[i]), min(ye - 1, y1[i])
            xa, xb = max(0, x0[i]), min(W - 1, x1[i])
            a_ = ca[i]
            b_ = cb[i]
            c_ = cc[i]
            o_ = op[i]
            for py in range(ya, yb + 1):
                dy = py + 0.5 - my[i]
                qy = -0.5 * c_ * dy * dy
                for px in range(xa, xb + 1):
                    t = T[py, px]
                    if t < 1e-4:
                        continue
                    dx = px + 0.5 - mx[i]
                    pw = qy - 0.5 * a_ * dx * dx - b_ * dx * dy
                    if pw < -5.55:          # alpha < 1/255 aunque la opacidad sea 1
                        continue
                    a = o_ * math.exp(pw)
                    if a > 0.99:
                        a = 0.99
                    if a < 1.0 / 255.0:
                        continue
                    w = a * t
                    img[py, px, 0] += w * col[i, 0]
                    img[py, px, 1] += w * col[i, 1]
                    img[py, px, 2] += w * col[i, 2]
                    nt = t * (1.0 - a)
                    T[py, px] = nt
                    if nt < 1e-4:
                        remaining -= 1
    return img, T


def _sh_color(model, view_dir):
    """Color con armónicos esféricos para una dirección de vista fija."""
    dc = model["dc"].astype(np.float64)
    rgb = SH_C0 * dc
    rest = model["rest"]
    if rest is not None and rest.shape[1] in (9, 24, 45):
        K = rest.shape[1] // 3
        sh = rest.reshape(-1, 3, K).astype(np.float64)  # canal-mayor
        x, y, z = view_dir
        basis = [-SH_C1 * y, SH_C1 * z, -SH_C1 * x]
        if K >= 8:
            xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
            basis += [SH_C2[0] * xy, SH_C2[1] * yz, SH_C2[2] * (2 * zz - xx - yy),
                      SH_C2[3] * xz, SH_C2[4] * (xx - yy)]
            if K >= 15:
                basis += [SH_C3[0] * y * (3 * xx - yy), SH_C3[1] * xy * z,
                          SH_C3[2] * y * (4 * zz - xx - yy),
                          SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy),
                          SH_C3[4] * x * (4 * zz - xx - yy),
                          SH_C3[5] * z * (xx - yy), SH_C3[6] * x * (xx - 3 * yy)]
        basis = np.array(basis[:K])
        rgb = rgb + sh @ basis
    return np.clip(rgb + 0.5, 0.0, 1.0).astype(np.float32)


def render(model, cam: OrthoCam, long_side=4000, front_clip=None,
           background=(255, 255, 255), transparent=False, color_override=None,
           return_raw=False):
    du = cam.u_range[1] - cam.u_range[0]
    dv = cam.v_range[1] - cam.v_range[0]
    pix = max(du, dv) / long_side
    W, H = max(8, int(round(du / pix))), max(8, int(round(dv / pix)))

    rel = model["xyz"] - cam.origin
    depth = rel @ cam.normal                       # + = hacia el observador
    sel = np.ones(len(rel), bool)
    if front_clip is not None:
        sel &= depth < front_clip * max(du, dv)
    u = rel @ cam.right
    v = rel @ cam.up
    mx = (u - cam.u_range[0]) / pix
    my = (cam.v_range[1] - v) / pix

    s2 = np.exp(2 * model["scale"].astype(np.float64))         # (N,3)
    Rm = model["R"].astype(np.float64)                          # columnas = ejes
    pr = np.einsum("nik,i->nk", Rm, cam.right) / pix           # (N,3)
    pu = np.einsum("nik,i->nk", Rm, -cam.up) / pix
    sxx = (s2 * pr * pr).sum(1)
    syy = (s2 * pu * pu).sum(1)
    sxy = (s2 * pr * pu).sum(1)
    det0 = sxx * syy - sxy * sxy
    sxx += 0.3
    syy += 0.3
    det = sxx * syy - sxy * sxy
    alpha = 1 / (1 + np.exp(-model["opacity"].astype(np.float64)))
    if model.get("mip"):
        alpha = alpha * np.sqrt(np.clip(det0 / np.maximum(det, 1e-12), 0, 1))
    mid = 0.5 * (sxx + syy)
    lam = mid + np.sqrt(np.maximum(mid * mid - det, 0.1))
    rad = np.ceil(3.0 * np.sqrt(lam))

    sel &= (det > 1e-12) & (alpha > 1 / 255)
    sel &= (mx + rad >= 0) & (mx - rad < W) & (my + rad >= 0) & (my - rad < H)
    sel &= rad < 0.6 * max(W, H)      # descarta 'nubes' gigantes
    ids = np.nonzero(sel)[0]
    ids = ids[np.argsort(-depth[ids], kind="stable")]  # de adelante hacia atrás

    if color_override is not None:
        col = np.ascontiguousarray(color_override[ids], dtype=np.float32)
    else:
        col = _sh_color({"dc": model["dc"][ids],
                         "rest": None if model["rest"] is None else model["rest"][ids]},
                        -cam.normal)
    inv = 1.0 / det[ids]
    ca = (syy[ids] * inv).astype(np.float32)
    cb = (-sxy[ids] * inv).astype(np.float32)
    cc = (sxx[ids] * inv).astype(np.float32)
    mxs, mys, r = mx[ids], my[ids], rad[ids]
    x0 = np.floor(mxs - r).astype(np.int64)
    x1 = np.ceil(mxs + r).astype(np.int64)
    y0 = np.clip(np.floor(mys - r), 0, H - 1).astype(np.int64)
    y1 = np.clip(np.ceil(mys + r), 0, H - 1).astype(np.int64)

    strip_h = 16
    n_strips = (H + strip_h - 1) // strip_h
    counts, idx = _bin_strips(y0, y1, n_strips, strip_h)
    img, T = _raster(W, H, strip_h, counts, idx, mxs.astype(np.float32),
                     mys.astype(np.float32), ca, cb, cc,
                     alpha[ids].astype(np.float32), col, x0, x1, y0, y1)
    if return_raw:
        return img, T
    if transparent:
        a = 1.0 - T
        rgb = img / np.maximum(a[..., None], 1e-6)
        out = np.dstack([np.clip(rgb, 0, 1), a])
        return Image.fromarray((out * 255 + 0.5).astype(np.uint8), "RGBA")
    bg = np.array(background, np.float32) / 255.0
    out = img + T[..., None] * bg
    return Image.fromarray((np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8), "RGB")


# ---------------------------------------------------------------------------
# Enderezado automático por bordes rectos
# ---------------------------------------------------------------------------
def straighten_angle(img: Image.Image, max_deg=8.0) -> float:
    """Ángulo (grados) que hay que girar la cámara para que los bordes
    dominantes queden horizontales / verticales. 0 si no hay señal clara."""
    from PIL import ImageFilter
    gray = img.convert("L")
    scale = max(gray.size) / 1600
    if scale > 1:
        gray = gray.resize((int(gray.size[0] / scale), int(gray.size[1] / scale)),
                           Image.BILINEAR)
    g = np.asarray(gray.filter(ImageFilter.GaussianBlur(2)), dtype=np.float32)
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[:, 1:-1] = g[:, 2:] - g[:, :-2]
    gy[1:-1, :] = g[2:, :] - g[:-2, :]
    mag = np.hypot(gx, gy)
    thr = np.percentile(mag, 92)
    m = mag > max(thr, 6)
    if np.count_nonzero(m) < 300:
        return 0.0
    ys, xs = np.nonzero(m)
    w = mag[m]
    vert = np.abs(gx[m]) > np.abs(gy[m])      # puntos de bordes verticales
    if len(w) > 80_000:
        sel = np.random.default_rng(0).choice(len(w), 80_000, replace=False)
        ys, xs, w, vert = ys[sel], xs[sel], w[sel], vert[sel]
    xs = xs.astype(np.float64); ys = ys.astype(np.float64)
    nb = int(max(gray.size)) + 1

    def sharp(deg):
        t = math.radians(deg)
        c, s_ = math.cos(t), math.sin(t)
        # al girar la cámara +deg, un punto de la imagen se mueve así:
        xr = xs * c - ys * s_
        yr = xs * s_ + ys * c
        hv = np.bincount((xr[vert] - xr.min()).astype(np.int64), w[vert], nb * 2)
        hh = np.bincount((yr[~vert] - yr.min()).astype(np.int64), w[~vert], nb * 2)
        return float((hv ** 2).sum() + (hh ** 2).sum())

    cands = np.arange(-max_deg, max_deg + 1e-6, 0.25)
    sc = np.array([sharp(d) for d in cands])
    best = cands[np.argmax(sc)]
    fine = np.arange(best - 0.25, best + 0.25 + 1e-6, 0.05)
    best = fine[np.argmax([sharp(d) for d in fine])]
    base = sharp(0.0)
    if sc.max() < base * 1.03:                # sin mejora clara: no tocar
        return 0.0
    return float(best) if abs(best) >= 0.1 else 0.0


# ---------------------------------------------------------------------------
# Entrada de alto nivel
# ---------------------------------------------------------------------------
def cams_from_workspace(workspace, run_dir, log=print):
    """Centros y 'arriba' de las cámaras COLMAP en el marco del .ply entrenado."""
    sp = find_sparse_dir(workspace) if workspace else None
    if not sp:
        return None, None
    cams, ups = read_colmap_cameras(sp)
    M = read_scene_transform(run_dir) if run_dir else None
    if cams is None:
        return None, None
    if M is None:
        log("  aviso: sin scene_transform.json, uso solo la nube de splats")
        return None, None
    A, t = M[:3, :3], M[:3, 3]
    cams = cams @ A.T + t
    ups = ups @ A.T
    ups /= np.linalg.norm(ups, axis=1, keepdims=True)
    return cams, ups


def save_cams_json(path, cams, ups):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"centros": np.round(cams, 6).tolist(), "arriba": np.round(ups, 6).tolist()}, f)


def load_cams_json(path):
    with open(path, encoding="utf-8") as f:
        j = json.load(f)
    return np.array(j["centros"], float), np.array(j["arriba"], float)


def flip_camera(cam: OrthoCam) -> OrthoCam:
    """Mira la fachada desde el otro lado (corrige una vista espejada)."""
    return OrthoCam(cam.origin, -cam.right, cam.up, -cam.normal,
                    (-cam.u_range[1], -cam.u_range[0]), cam.v_range)


def _front_haze(model, cam):
    """Cuánto 'material' ve la cámara delante de la fachada (en fracción del ancho).
    Desde el lado filmado lo primero que se ve es la superficie (valor ~0); desde atrás
    se ve primero la niebla de splats que quedó detrás de la pared (valor alto)."""
    du = cam.u_range[1] - cam.u_range[0]
    t = ((model["xyz"] - cam.origin) @ cam.normal) / max(du, 1e-9)
    col = np.repeat(t[:, None], 3, 1).astype(np.float32)
    img, T = render(model, cam, 500, None, color_override=col, return_raw=True)
    cov = 1.0 - T
    mask = cov > 0.5
    if not mask.any():
        return 0.0
    return float(np.mean(np.clip(img[..., 0][mask] / cov[mask], 0, None)))


def pick_visible_side(model, cam, front_clip=None, has_cams=False, log=print):
    """Elige el lado de la fachada que se filmó (validado con modelos con cámaras).
    Con cámaras solo se cambia si el otro lado es muchísimo más limpio."""
    other = flip_camera(cam)
    if not has_cams:
        # sin cámaras (.ply limpiado): visto desde atrás la puerta sale "a través" (borrosa,
        # fantasma). Se dibujan los dos lados y gana el más nítido si la diferencia es clara.
        sa, sb = _side_sharpness(model, cam, front_clip), _side_sharpness(model, other, front_clip)
        r = max(sa, sb) / max(min(sa, sb), 1e-9)
        if r >= 1.3:
            return (cam, r) if sa >= sb else (other, r)
        a = _front_haze(model, cam)
        b = _front_haze(model, other)
        return (other, r) if b < a else (cam, r)     # r < 1.3: lado dudoso (se avisa)
    a = _front_haze(model, cam)
    b = _front_haze(model, other)
    if b * 4.0 < a:
        return other, a / max(b, 1e-6)
    return cam, b / max(a, 1e-6)


def _side_sharpness(model, cam, front_clip=None, side=600):
    """Nitidez media (bordes) de lo que se ve desde ese lado, solo donde hay superficie."""
    try:
        img, T = render(model, cam, side, front_clip, return_raw=True)
    except Exception:
        return 0.0
    g = np.asarray(img, np.float32)
    g = g.mean(2) if g.ndim == 3 else g
    solid = (1.0 - np.asarray(T, np.float32)) > 0.6
    gx = np.abs(np.diff(g, axis=1))[:-1, :]
    gy = np.abs(np.diff(g, axis=0))[:, :-1]
    m = solid[:-1, :-1]
    return float((gx + gy)[m].mean()) if m.sum() > 100 else 0.0


def frame_to_content(model, cam, front_clip=None, margin=0.05, side=900):
    """Encuadre centrado en lo que realmente hay (la puerta/ventana), con el mismo margen a los
    cuatro lados. Se ignoran los bordes ralos (motas sueltas, restos del recorte)."""
    try:
        # solo lo que está EN el plano de la puerta (ni lo de delante ni lo de detrás)
        d = (model["xyz"] - cam.origin) @ cam.normal
        ext = max(cam.u_range[1] - cam.u_range[0], cam.v_range[1] - cam.v_range[0])
        near = np.abs(d) < 0.06 * ext
        sub = {k: (v[near] if isinstance(v, np.ndarray) and len(v) == len(near) else v)
               for k, v in model.items()}
        if near.sum() < 1000:
            sub = model
        _img, T = render(sub, cam, side, front_clip, return_raw=True)
    except Exception:
        return cam
    a = 1.0 - np.asarray(T, np.float32)
    H, W = a.shape
    solid = a > 0.6
    if solid.sum() < 0.01 * H * W:
        return cam
    oc, orr = solid.mean(0), solid.mean(1)          # ocupación por columna / fila

    def span(occ):
        ref = float(np.percentile(occ[occ > 0], 90)) if (occ > 0).any() else 0.0
        keep = np.nonzero(occ >= 0.25 * ref)[0]
        return (int(keep[0]), int(keep[-1]) + 1) if len(keep) else (0, len(occ))
    x0, x1 = span(oc)
    y0, y1 = span(orr)
    (u0, u1), (v0, v1) = cam.u_range, cam.v_range
    pu, pv = (u1 - u0) / W, (v1 - v0) / H
    nu0, nu1 = u0 + x0 * pu, u0 + x1 * pu
    nv1, nv0 = v1 - y0 * pv, v1 - y1 * pv           # la fila 0 es arriba
    pad = margin * max(nu1 - nu0, nv1 - nv0)
    return OrthoCam(cam.origin, cam.right, cam.up, cam.normal,
                    (nu0 - pad, nu1 + pad), (nv0 - pad, nv1 + pad))


def door_camera(model, cams=None, ups=None, front_clip=0.12, vertical=True, margin=0.05,
                straighten=True, log=print):
    """Cámara de puerta/ventana: plano de la fachada, lado filmado, enderezada y centrada."""
    cam = build_camera(model, cams, ups, vertical=vertical, margin=margin)
    cam, _r = pick_visible_side(model, cam, front_clip, cams is not None, log)
    if straighten:
        for _ in range(3):
            a = straighten_angle(render(model, cam, 1200, front_clip))
            if a == 0.0:
                break
            cam = rotate_camera(cam, a)
    return frame_to_content(model, cam, front_clip, margin)


def render_run(ply_path, out_path, workspace=None, run_dir=None, long_side=4000,
               front_clip=0.35, vertical=True, auto_straighten=True,
               transparent=False, margin=0.03, background=(0, 0, 0), log=print,
               preview_side=1600, cams=None, ups=None, flip=False, rot90=0,
               save_cams=None, auto_side=True, model=None) -> dict:
    model = model if model is not None else read_splat_ply(ply_path)
    if cams is None and workspace:
        cams, ups = cams_from_workspace(workspace, run_dir or os.path.dirname(ply_path), log)
    if save_cams and cams is not None:
        try:
            save_cams_json(save_cams, cams, ups)
        except OSError:
            pass
    cam = build_camera(model, cams, ups, vertical=vertical, margin=margin)
    side_ratio = None
    if auto_side:
        cam, side_ratio = pick_visible_side(model, cam, front_clip, cams is not None, log)
    if flip:
        cam = flip_camera(cam)
    if rot90 % 4:
        cam = rotate_camera(cam, 90.0 * (rot90 % 4))
    total = 0.0
    if auto_straighten:   # se mide sobre una vista reducida (rápido)
        for _ in range(3):
            prev = render(model, cam, min(preview_side, long_side), front_clip,
                          background=background)
            a = straighten_angle(prev)
            if a == 0.0:
                break
            total += a
            cam = rotate_camera(cam, a)
    cam = frame_to_content(model, cam, front_clip, margin)     # centrado, margen parejo
    img = render(model, cam, long_side, front_clip, background=background,
                 transparent=transparent)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    img.save(out_path, compress_level=3)
    return {"splats": len(model["xyz"]), "size": img.size, "roll_fix_deg": total,
            "con_camaras": cams is not None, "nitidez_lado": side_ratio,
            "lado_dudoso": bool(auto_side and cams is None and side_ratio is not None
                                and side_ratio < 1.3 and not flip),
            "transparencia": transparency(model, cam, front_clip)}


def transparency(model, cam, front_clip=None, side=500):
    """Parte de la puerta/ventana que quedó semitransparente (zonas poco grabadas,
    contraluz, nubes): 0 = todo sólido. Se mide en el centro (sin los bordes del recorte)."""
    try:
        _img, T = render(model, cam, side, front_clip, return_raw=True)
    except Exception:
        return None
    a = 1.0 - np.asarray(T, np.float32)
    H, W = a.shape
    c = a[int(H * 0.12):int(H * 0.88), int(W * 0.12):int(W * 0.88)]
    if c.size == 0:
        return None
    return float(((c > 0.03) & (c < 0.7)).mean())
