"""Conversión de splats (.ply de Spirula / SuperSplat) a nube de puntos y vuelta.

IDA   splat .ply  ->  nube con color (lo que se ve en SuperSplat) para CloudCompare / ReCap
        * .ply  (CloudCompare)        x y z + RGB + splat_id + opacidad
        * .las  (CloudCompare y ReCap) x y z + RGB; splat_id en "GPS time", opacidad en intensidad
        * .pts  (ReCap)               x y z intensidad R G B
      Junto a cada nube se guarda <nombre>.splatref.json con la ruta del splat original.

VUELTA nube editada (borrada/recortada en CloudCompare o ReCap)  ->  splat .ply
      Cada punto de la nube ES un splat. Se recuperan los splats originales que siguen en la
      nube editada, con TODOS sus datos (escala, rotación, opacidad, armónicos esféricos),
      así que el .ply resultante se renderiza igual que antes en Spirula / Ortofotos.
        * Si la nube conserva "splat_id" (CloudCompare .ply / .las): coincidencia exacta.
        * Si no (ReCap .pts / .e57 ...): coincidencia por posición (detecta cambio de
          unidades m/mm/cm/pies).
      No muevas ni gires la nube al editarla: solo borra o recorta.
"""
from __future__ import annotations

import json
import os
import re
import struct
import time

import numpy as np

SH_C0 = 0.28209479177387814
REF_SUFFIX = ".splatref.json"
CLOUD_EXT = (".ply", ".las", ".pts", ".xyz", ".txt", ".asc", ".csv", ".e57")

_PLY_TYPES = {"float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
              "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
              "short": "<i2", "int16": "<i2", "ushort": "<u2", "uint16": "<u2",
              "int": "<i4", "int32": "<i4", "uint": "<u4", "uint32": "<u4"}
_DT_TO_PLY = {"f4": "float", "f8": "double", "u1": "uchar", "i1": "char", "i2": "short",
              "u2": "ushort", "i4": "int", "u4": "uint"}


# ---------------------------------------------------------------------------
# Lectura de PLY genérico (binario LE/BE o ASCII; ignora elementos con listas tras el vértice)
# ---------------------------------------------------------------------------
def read_ply(path):
    """Devuelve (elementos {nombre: array estructurado}, comentarios, formato)."""
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
            if len(header) > 5000:
                raise ValueError("Cabecera PLY inválida: " + path)
        fmt, elements, comments = "binary_little_endian", [], []
        for line in header:
            p = line.split()
            if not p:
                continue
            if p[0] == "format":
                fmt = p[1]
            elif p[0] in ("comment", "obj_info"):
                comments.append(line)
            elif p[0] == "element":
                elements.append([p[1], int(p[2]), [], False])
            elif p[0] == "property" and elements:
                if p[1] == "list":
                    elements[-1][3] = True
                    elements[-1][2].append((p[-1], None))
                else:
                    elements[-1][2].append((p[2], _PLY_TYPES[p[1]]))
        data = {}
        if fmt == "ascii":
            for name, n, props, has_list in elements:
                if has_list:
                    break
                dt = np.dtype([(k, t.replace("<", "")) for k, t in props])
                rows = [f.readline().decode("ascii", "replace") for _ in range(n)]
                arr = np.loadtxt(rows, dtype=np.float64, ndmin=2) if n else np.zeros((0, len(props)))
                out = np.zeros(n, dtype=dt)
                for i, (k, _t) in enumerate(props):
                    out[k] = arr[:, i]
                data[name] = out
        else:
            be = fmt == "binary_big_endian"
            for name, n, props, has_list in elements:
                if has_list:
                    break
                dt = np.dtype([(k, (t.replace("<", ">") if be else t)) for k, t in props])
                data[name] = np.fromfile(f, dtype=dt, count=n)
    return data, comments, fmt


def write_ply_struct(path, arr, comments=()):
    """Escribe un array estructurado como PLY binario little-endian (un elemento 'vertex')."""
    arr = np.ascontiguousarray(arr.astype(arr.dtype.newbyteorder("<")))
    lines = ["ply", "format binary_little_endian 1.0"]
    lines += [c for c in comments if c.startswith(("comment", "obj_info"))]
    lines.append(f"element vertex {len(arr)}")
    for k in arr.dtype.names:
        code = arr.dtype[k].str[1:]
        lines.append(f"property {_DT_TO_PLY[code]} {k}")
    lines.append("end_header")
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(("\n".join(lines) + "\n").encode("ascii"))
        arr.tofile(f)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Splat original
# ---------------------------------------------------------------------------
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def load_splat(path):
    """Lee un splat .ply (estándar o comprimido de SuperSplat).
    Devuelve dict con: vtx (array estructurado sin pérdidas), xyz, rgb (uint8), alpha, comments,
    compressed (bool)."""
    data, comments, fmt = read_ply(path)
    if "vertex" not in data:
        raise ValueError("El .ply no tiene vértices: " + path)
    v = data["vertex"]
    names = v.dtype.names or ()
    if "chunk" in data and "packed_position" in names:
        from render_ortho import _decode_compressed
        xyz, dc, rest, opacity, scale, rot = _decode_compressed(data)
        props = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                 ("f_dc_0", "<f4"), ("f_dc_1", "<f4"), ("f_dc_2", "<f4")]
        nrest = 0 if rest is None else rest.shape[1]
        props += [(f"f_rest_{i}", "<f4") for i in range(nrest)]
        props += [("opacity", "<f4"), ("scale_0", "<f4"), ("scale_1", "<f4"), ("scale_2", "<f4"),
                  ("rot_0", "<f4"), ("rot_1", "<f4"), ("rot_2", "<f4"), ("rot_3", "<f4")]
        vtx = np.zeros(len(xyz), dtype=np.dtype(props))
        for i, k in enumerate("xyz"):
            vtx[k] = xyz[:, i]
        for i in range(3):
            vtx[f"f_dc_{i}"] = dc[:, i]
        if nrest:
            # el comprimido guarda los SH canal por canal (como el estándar)
            for i in range(nrest):
                vtx[f"f_rest_{i}"] = rest[:, i]
        vtx["opacity"] = opacity
        for i in range(3):
            vtx[f"scale_{i}"] = scale[:, i]
        for i in range(4):
            vtx[f"rot_{i}"] = rot[:, i]
        compressed = True
    else:
        if not all(k in names for k in ("x", "y", "z")):
            raise ValueError("El .ply no tiene coordenadas x y z: " + path)
        vtx, compressed = v, False
    names = vtx.dtype.names
    xyz = np.stack([vtx["x"], vtx["y"], vtx["z"]], 1).astype(np.float64)
    if "f_dc_0" in names:
        dc = np.stack([vtx["f_dc_0"], vtx["f_dc_1"], vtx["f_dc_2"]], 1).astype(np.float64)
        rgb = np.clip((0.5 + SH_C0 * dc) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    elif "red" in names:
        rgb = np.stack([vtx["red"], vtx["green"], vtx["blue"]], 1).astype(np.uint8)
    else:
        rgb = np.full((len(xyz), 3), 200, np.uint8)
    alpha = _sigmoid(vtx["opacity"].astype(np.float64)) if "opacity" in names else np.ones(len(xyz))
    is_splat = "f_dc_0" in names and "scale_0" in names
    return dict(vtx=vtx, xyz=xyz, rgb=rgb, alpha=alpha, comments=comments,
                compressed=compressed, is_splat=is_splat)


# ---------------------------------------------------------------------------
# Orientación (poner "de pie" para ReCap / CloudCompare)
# ---------------------------------------------------------------------------
_YUP_TO_NATIVE = np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]], dtype=np.float64)


def _rot_to_z(u):
    """Rotación que lleva el vector u a +Z."""
    u = np.asarray(u, float)
    u = u / max(np.linalg.norm(u), 1e-12)
    z = np.array([0.0, 0.0, 1.0])
    c = float(u @ z)
    if c > 1 - 1e-9:
        return np.eye(3)
    if c < -1 + 1e-9:
        return np.diag([1.0, -1.0, -1.0])
    ax = np.cross(u, z)
    s = np.linalg.norm(ax)
    ax /= s
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + s * K + (1 - c) * (K @ K)


def up_rotation(splat, ups_native=None):
    """Rotación para que el modelo quede con Z hacia arriba.
    Usa las cámaras si las hay; si no, la convención de Spirula (arriba = -Z nativo)."""
    yup = any("supersplat_y_up" in c for c in splat["comments"])
    if ups_native is not None and len(ups_native):
        u = np.mean(np.asarray(ups_native, float), 0)
        if np.linalg.norm(u) > 1e-6:
            u_file = _YUP_TO_NATIVE.T @ u if yup else u
            return _rot_to_z(u_file)
    return _rot_to_z([0, 1, 0] if yup else [0, 0, -1])


# ---------------------------------------------------------------------------
# Escritores de nube
# ---------------------------------------------------------------------------
SP = "sp_"     # prefijo de los campos del splat que viajan dentro de la nube (campos escalares)


def splat_fields(vtx):
    """Campos del splat que se guardan dentro de la nube para poder rearmarlo sin el original:
    posición original (sp_x, sp_y, sp_z) y todas las propiedades del splat (sp_opacity, sp_scale_0...)."""
    out = []
    for k in vtx.dtype.names:
        if k in ("nx", "ny", "nz"):
            continue
        out.append((SP + k, np.asarray(vtx[k], np.float32)))
    return out


def write_cloud_ply(path, xyz, rgb, ids, alpha, fields=()):
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"),
                   ("blue", "u1"), ("splat_id", "<f4"), ("opacidad", "<f4")]
                  + [(k, "<f4") for k, _v in fields])
    a = np.zeros(len(xyz), dtype=dt)
    for k, v in fields:
        a[k] = v
    a["x"], a["y"], a["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    a["red"], a["green"], a["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    a["splat_id"] = ids
    a["opacidad"] = alpha
    write_ply_struct(path, a, ["comment Ortofotos Spirula: nube desde splat (splat_id = indice)"])


def write_cloud_las(path, xyz, rgb, ids, alpha, fields=()):
    """LAS 1.2, formato de punto 3 (XYZ + intensidad + GPS time + RGB 16 bits)
    + campos del splat como 'extra bytes' (float32; CloudCompare los ve como campos escalares)."""
    n = len(xyz)
    mn, mx = xyz.min(0), xyz.max(0)
    ext = float(np.max(np.abs(mx - mn))) or 1.0
    scale = 10.0 ** np.ceil(np.log10(ext / 2.0e9))       # cabe en int32
    scale = float(max(scale, 1e-9))
    off = np.floor(mn / scale) * scale
    rec = np.dtype([("X", "<i4"), ("Y", "<i4"), ("Z", "<i4"), ("intensity", "<u2"),
                    ("flags", "u1"), ("cls", "u1"), ("angle", "i1"), ("user", "u1"),
                    ("src", "<u2"), ("gps", "<f8"), ("R", "<u2"), ("G", "<u2"), ("B", "<u2")]
                   + [(k, "<f4") for k, _v in fields])
    pts = np.zeros(n, dtype=rec)
    for k, v in fields:
        pts[k] = v
    q = np.round((xyz - off) / scale).astype(np.int64)
    pts["X"], pts["Y"], pts["Z"] = q[:, 0], q[:, 1], q[:, 2]
    pts["intensity"] = np.clip(alpha * 65535, 0, 65535).astype(np.uint16)
    pts["flags"] = 0b00010001     # retorno 1 de 1
    pts["cls"] = 1
    pts["gps"] = ids
    c16 = rgb.astype(np.uint16) * 257
    pts["R"], pts["G"], pts["B"] = c16[:, 0], c16[:, 1], c16[:, 2]
    vlr = b""
    if fields:
        desc = bytearray()
        for k, _v in fields:
            d = bytearray(192)
            d[2] = 9                                   # float
            d[4:36] = k.encode("ascii")[:32].ljust(32, b"\0")
            d[160:192] = b"splat (Ortofotos Spirula)".ljust(32, b"\0")
            desc += d
        vh = bytearray(54)
        vh[2:18] = b"LASF_Spec".ljust(16, b"\0")
        struct.pack_into("<HH", vh, 18, 4, len(desc))
        vh[22:54] = b"campos del splat".ljust(32, b"\0")
        vlr = bytes(vh) + bytes(desc)
    hdr = bytearray(227)
    hdr[0:4] = b"LASF"
    hdr[24], hdr[25] = 1, 2
    hdr[26:58] = b"OrtofotosSpirula".ljust(32, b"\0")
    hdr[58:90] = b"splat a nube".ljust(32, b"\0")
    lt = time.localtime()
    struct.pack_into("<HHHIIBHI", hdr, 90, lt.tm_yday, lt.tm_year, 227, 227 + len(vlr),
                     1 if vlr else 0, 3, 34 + 4 * len(fields), n)
    rets = [n, 0, 0, 0, 0]
    struct.pack_into("<5I", hdr, 111, *rets)
    struct.pack_into("<3d3d", hdr, 131, scale, scale, scale, *off)
    struct.pack_into("<6d", hdr, 179, mx[0], mn[0], mx[1], mn[1], mx[2], mn[2])
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(bytes(hdr))
        f.write(vlr)
        pts.tofile(f)
    os.replace(tmp, path)


def write_cloud_pts(path, xyz, rgb, alpha, decimals=6):
    n = len(xyz)
    inten = np.round(alpha * 4095 - 2048).astype(np.int32)
    tmp = path + ".tmp"
    fmt = " ".join([f"%.{decimals}f"] * 3) + " %d %d %d %d"
    with open(tmp, "w", encoding="ascii", newline="\n") as f:
        f.write(f"{n}\n")
        step = 200000
        for i in range(0, n, step):
            block = np.column_stack([xyz[i:i + step], inten[i:i + step], rgb[i:i + step]])
            np.savetxt(f, block, fmt=fmt)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Lectores de nube editada
# ---------------------------------------------------------------------------
def _decimals_of(text_lines):
    best = 0
    for ln in text_lines:
        for tok in ln.replace(",", " ").split()[:3]:
            if "." in tok and "e" not in tok.lower():
                best = max(best, len(tok.split(".")[1]))
    return best


def read_cloud(path):
    """Devuelve dict(xyz, ids o None, step = precisión de las coordenadas)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".ply":
        data, _c, fmt = read_ply(path)
        v = data.get("vertex")
        if v is None or not len(v):
            raise ValueError("La nube no tiene puntos.")
        xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
        ids = None
        campos = _campos_sp({k: v[k] for k in v.dtype.names})
        for k in v.dtype.names:
            if k.lower() in ("splat_id", "scalar_splat_id", "scalar_splat id", "splat id") or \
                    re.sub(r"[^a-z]", "", k.lower()) in ("splatid", "scalarsplatid"):
                ids = np.asarray(v[k], np.float64)
                break
        step = 0.0
        if fmt == "ascii":
            with open(path, "rb") as f:
                lines = f.read(400000).decode("ascii", "replace").split("end_header", 1)[-1]
            step = 10.0 ** -max(_decimals_of(lines.splitlines()[1:200]), 1)
        elif v.dtype["x"].itemsize == 4:
            step = float(np.max(np.abs(xyz))) * 1.2e-7
        return dict(xyz=xyz, ids=ids, step=step, campos=campos, color=_color_de(v), alfa=_alfa_de(v))
    if ext == ".las":
        return _read_las(path)
    if ext == ".laz":
        raise ValueError("LAZ comprimido no soportado: guárdalo como .las sin comprimir.")
    if ext == ".e57":
        return _read_e57(path)
    return _read_ascii(path)


def _norm_campo(k):
    k = k.strip().lower().replace(" ", "_")
    while k.startswith("scalar_"):
        k = k[7:]
    return k


def _campos_sp(d):
    """Campos del splat (sp_...) de una nube, con los nombres que les deja CloudCompare."""
    out = {}
    for k, v in d.items():
        nk = _norm_campo(k)
        if nk.startswith(SP):
            out[nk] = np.asarray(v, np.float32)
    return out


def _color16(c):
    """Color a 8 bits. LAS guarda 16 bits: CloudCompare multiplica por 256, otros por 257."""
    c = np.asarray(c)
    if c.size == 0 or float(c.max()) <= 255:
        return np.clip(np.round(c.astype(np.float64)), 0, 255).astype(np.uint8)
    ci = c.astype(np.int64)
    if np.all(ci % 256 == 0):
        return np.clip(ci // 256, 0, 255).astype(np.uint8)
    if np.all(ci % 257 == 0):
        return np.clip(ci // 257, 0, 255).astype(np.uint8)
    return np.clip(np.round(ci / 257.0), 0, 255).astype(np.uint8)


def _color_de(v):
    names = {_norm_campo(k): k for k in v.dtype.names}
    for r, g, b in (("red", "green", "blue"), ("r", "g", "b"), ("diffuse_red", "diffuse_green", "diffuse_blue")):
        if r in names and g in names and b in names:
            c = np.stack([v[names[r]], v[names[g]], v[names[b]]], 1)
            if c.dtype.kind == "f" and float(c.max()) <= 1.0:
                return np.clip(np.round(c.astype(np.float64) * 255.0), 0, 255).astype(np.uint8)
            return _color16(c)
    return None


def _alfa_de(v):
    names = {_norm_campo(k): k for k in v.dtype.names}
    for k in ("opacidad", "alpha"):
        if k in names:
            return np.clip(np.asarray(v[names[k]], np.float64), 0, 1)
    return None


_EB_TYPES = {1: "u1", 2: "i1", 3: "<u2", 4: "<i2", 5: "<u4", 6: "<i4", 7: "<u8", 8: "<i8", 9: "<f4", 10: "<f8"}
_PT_BASE = {0: 20, 1: 28, 2: 26, 3: 34, 4: 57, 5: 63, 6: 30, 7: 36, 8: 38, 9: 59, 10: 67}


def _read_las(path):
    with open(path, "rb") as f:
        h = f.read(375)
        if h[:4] != b"LASF":
            raise ValueError("No es un archivo LAS válido.")
        vmin = h[25]
        off_pts = struct.unpack_from("<I", h, 96)[0]
        pfmt = h[104] & 0x3F
        rlen = struct.unpack_from("<H", h, 105)[0]
        n = struct.unpack_from("<I", h, 107)[0]
        if vmin >= 4 and n == 0:
            n = struct.unpack_from("<Q", h, 247)[0]
        sc = struct.unpack_from("<3d", h, 131)
        of = struct.unpack_from("<3d", h, 155)
        hsize = struct.unpack_from("<H", h, 94)[0]
        nvlr = struct.unpack_from("<I", h, 100)[0]
        extras = []                                   # (nombre, tipo, escala, desplazamiento)
        f.seek(hsize)
        for _ in range(nvlr):
            vh = f.read(54)
            if len(vh) < 54:
                break
            uid = vh[2:18].rstrip(b"\0")
            rid, ln = struct.unpack_from("<HH", vh, 18)
            body = f.read(ln)
            if uid == b"LASF_Spec" and rid == 4:
                for i in range(0, len(body) - 191, 192):
                    d = body[i:i + 192]
                    t = d[2]
                    nm = d[4:36].split(b"\0")[0].decode("ascii", "replace")
                    opt = d[3]
                    escl = struct.unpack_from("<d", d, 112)[0] if opt & 8 else 1.0
                    desp = struct.unpack_from("<d", d, 136)[0] if opt & 16 else 0.0
                    extras.append((nm, t, d[160:192], escl, desp, struct.unpack_from("<H", d, 0)[0]))
        f.seek(off_pts)
        raw = np.fromfile(f, dtype=np.dtype((np.void, rlen)), count=n)
    buf = raw.view(np.uint8).reshape(-1, rlen)
    q = buf[:, :12].copy().view("<i4").reshape(-1, 3).astype(np.float64)
    xyz = q * np.array(sc) + np.array(of)
    ids = None
    gps_at = 20 if pfmt in (1, 3, 4, 5) else 22 if pfmt >= 6 else None
    if gps_at is not None and rlen >= gps_at + 8:
        g = buf[:, gps_at:gps_at + 8].copy().view("<f8").ravel()
        if len(g) and np.all(np.isfinite(g)) and np.all(g >= 0) and np.allclose(g, np.round(g)) \
                and (len(g) < 3 or np.ptp(g) > 0):
            ids = g
    # color e intensidad
    color = alfa = None
    rgb_at = {2: 20, 3: 28, 5: 28, 7: 30, 8: 30, 10: 30}.get(pfmt)
    if rgb_at is not None and rlen >= rgb_at + 6:
        color = _color16(buf[:, rgb_at:rgb_at + 6].copy().view("<u2").reshape(-1, 3))
    inten = buf[:, 12:14].copy().view("<u2").ravel()
    if inten.max() > 0:
        alfa = inten.astype(np.float64) / 65535.0
    # campos extra (extra bytes): aquí viajan los datos del splat
    campos = {}
    pos = _PT_BASE.get(pfmt, rlen)
    for nm, t, _desc, escl, desp, _res in extras:
        dt = _EB_TYPES.get(t)
        if dt is None:          # tipo no estándar (p. ej. 0 = bytes sin tipo): se salta
            break
        size = np.dtype(dt).itemsize
        if pos + size > rlen:
            break
        val = buf[:, pos:pos + size].copy().view(dt).ravel().astype(np.float64) * escl + desp
        pos += size
        nk = _norm_campo(nm)
        if nk.startswith(SP):
            campos[nk] = val.astype(np.float32)
    return dict(xyz=xyz, ids=ids, step=float(min(sc)), campos=campos, color=color, alfa=alfa,
                cuant=(np.array(sc, float), np.array(of, float)))


def _read_ascii(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        head = [f.readline() for _ in range(60)]
    head = [h for h in head if h.strip()]
    skip = 0
    for ln in head[:5]:
        toks = ln.replace(",", " ").replace(";", " ").split()
        try:
            [float(t) for t in toks]
            if len(toks) >= 3:
                break
        except ValueError:
            pass
        skip += 1
    # .pts: primera línea = número de puntos
    first = head[skip].replace(",", " ").split() if len(head) > skip else []
    if len(first) == 1:
        skip += 1
    delim = "," if "," in head[min(skip, len(head) - 1)] else None
    xyz = np.loadtxt(path, skiprows=skip, usecols=(0, 1, 2), delimiter=delim,
                     dtype=np.float64, ndmin=2, comments=("#", "//"))
    step = 10.0 ** -max(_decimals_of(head[skip:]), 1)
    return dict(xyz=xyz, ids=None, step=step)


def _read_e57(path):
    try:
        import pye57  # opcional
    except ImportError:
        raise ValueError("Para leer .e57 exporta desde ReCap como .pts (o abre el .e57 en "
                         "CloudCompare y guárdalo como .ply o .las).")
    e = pye57.E57(path)
    parts = []
    for i in range(e.scan_count):
        d = e.read_scan(i, ignore_missing_fields=True, transform=True)
        parts.append(np.column_stack([d["cartesianX"], d["cartesianY"], d["cartesianZ"]]))
    xyz = np.vstack(parts).astype(np.float64)
    return dict(xyz=xyz, ids=None, step=float(np.max(np.abs(xyz))) * 1.2e-7)


# ---------------------------------------------------------------------------
# Coincidencia por posición (solo numpy)
# ---------------------------------------------------------------------------
_OFFS = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)],
                 dtype=np.int64)


def near_mask(ref, query, tol):
    """True para cada punto de query que tiene algún punto de ref a ~tol (rejilla de celdas)."""
    if not len(ref) or not len(query):
        return np.zeros(len(query), bool)
    lo = np.minimum(ref.min(0), query.min(0))
    hi = np.maximum(ref.max(0), query.max(0))
    cell = max(float(tol), float(np.max(hi - lo)) / 2.0e6, 1e-12)

    def keys(cells):
        return (cells[:, 0] << 42) | (cells[:, 1] << 21) | cells[:, 2]
    rc = np.floor((ref - lo) / cell).astype(np.int64) + 1
    rk = np.unique(keys(rc))
    qc = np.floor((query - lo) / cell).astype(np.int64) + 1
    out = np.zeros(len(query), bool)
    for o in _OFFS:
        k = keys(qc + o)
        pos = np.searchsorted(rk, k)
        pos[pos >= len(rk)] = len(rk) - 1
        out |= rk[pos] == k
    return out


UNIT_FACTORS = [(1.0, ""), (0.001, "mm→m"), (1000.0, "m→mm"), (0.01, "cm→m"), (100.0, "m→cm"),
                (0.3048, "pies→m"), (1 / 0.3048, "m→pies"), (0.0254, "pulg→m"), (1 / 0.0254, "m→pulg")]


def _best_factor(cloud_xyz, orig_xyz, tol, log):
    rng = np.random.default_rng(0)
    smp = cloud_xyz[rng.choice(len(cloud_xyz), min(len(cloud_xyz), 20000), replace=False)]
    best = (0.0, 1.0, "")
    for fct, lab in UNIT_FACTORS:
        frac = float(near_mask(orig_xyz, smp * fct, tol).mean())
        if frac > best[0]:
            best = (frac, fct, lab)
        if frac > 0.9 and fct == 1.0:
            break
    return best


def _match_cuantizado(cl, sxyz, srgb):
    """Empareja cada punto de una nube LAS con SU splat. Se redondean las posiciones de los splats igual
    que el LAS (escala y desplazamiento de su cabecera); entre los candidatos de la celda y sus vecinas
    decide el color (CloudCompare redondea a 1 mm y varios splats caen en la misma celda) y cada splat se
    usa una sola vez. Devuelve (índice de splat por punto o -1, fracción emparejada)."""
    cu = cl.get("cuant")
    if cu is None:
        return None, 0.0
    sc, of = cu
    qc = np.round((cl["xyz"] - of) / sc).astype(np.int64)
    qs = np.round((sxyz.astype(np.float32).astype(np.float64) - of) / sc).astype(np.int64)
    lo = np.minimum(qc.min(0), qs.min(0)) - 2
    qc -= lo
    qs -= lo
    span = np.maximum(qc.max(0), qs.max(0)) + 3
    if float(np.prod(span.astype(np.float64))) > 9.0e18:
        return None, 0.0

    def key(q):
        return (q[:, 0] * span[1] + q[:, 1]) * span[2] + q[:, 2]
    ks = key(qs)
    order = np.argsort(ks, kind="stable")
    kss = ks[order]
    ccol = cl.get("color")
    # todos los pares (punto, splat candidato) con su costo: color + distancia en celdas
    P, S, C = [], [], []
    for o in _OFFS:
        kc = key(qc + o)
        a = np.searchsorted(kss, kc, "left")
        b = np.searchsorted(kss, kc, "right")
        cnt = b - a
        m = int(cnt.max()) if len(cnt) else 0
        dist = float(np.abs(o).sum())
        for j in range(min(m, 16)):
            has = np.nonzero(cnt > j)[0]
            if not len(has):
                break
            cand = order[a[has] + j]
            if ccol is not None:
                d = np.abs(srgb[cand].astype(np.int32) - ccol[has].astype(np.int32)).sum(1).astype(np.float64)
            else:
                d = np.zeros(len(has))
            # el color manda; la celda exacta solo desempata
            P.append(has); S.append(cand); C.append(d * 4.0 + dist)
    out = np.full(len(qc), -1, np.int64)
    if not P:
        return out, 0.0
    P = np.concatenate(P); S = np.concatenate(S); C = np.concatenate(C)
    # asignación ávida por costo: cada punto y cada splat una sola vez
    o = np.lexsort((S, P, C))
    P, S = P[o], S[o]
    used_p = np.zeros(len(qc), bool)
    used_s = np.zeros(len(sxyz), bool)
    for _ in range(4):
        # pasadas vectorizadas: en cada una, el primer par libre de cada punto y de cada splat
        libre = ~used_p[P] & ~used_s[S]
        if not libre.any():
            break
        pp, ss = P[libre], S[libre]
        _u, fp = np.unique(pp, return_index=True)
        pp, ss = pp[fp], ss[fp]
        _u, fs = np.unique(ss, return_index=True)
        pp, ss = pp[fs], ss[fs]
        out[pp] = ss
        used_p[pp] = True
        used_s[ss] = True
    return out, float((out >= 0).mean())


def _rotaciones_posibles(sp):
    """Giros con que la app pudo exportar la nube (sin ficha no se sabe cuál fue)."""
    rs = [np.eye(3), np.diag([1.0, -1.0, -1.0])]
    try:
        rs.append(up_rotation(sp, None))
    except Exception:
        pass
    rs.append(_YUP_TO_NATIVE.T)
    rs.append(_YUP_TO_NATIVE)
    return rs


# ---------------------------------------------------------------------------
# Operaciones principales
# ---------------------------------------------------------------------------
def splat_to_clouds(src, out_dir, formats=("ply", "las", "pts"), min_opacity=0.05,
                    upright=True, ups_native=None, name=None, log=print):
    """Convierte un splat .ply en nubes. Devuelve la lista de archivos creados."""
    t0 = time.time()
    sp = load_splat(src)
    n = len(sp["xyz"])
    keep = sp["alpha"] >= float(min_opacity) if min_opacity else np.ones(n, bool)
    ids = np.nonzero(keep)[0]
    xyz, rgb, alpha = sp["xyz"][keep], sp["rgb"][keep], sp["alpha"][keep]
    R = up_rotation(sp, ups_native) if upright else np.eye(3)
    xyz_out = xyz @ R.T
    os.makedirs(out_dir, exist_ok=True)
    stem = name or os.path.splitext(os.path.basename(src))[0]
    stem = re.sub(r"\.compressed$", "", stem, flags=re.I)
    made = []
    campos = splat_fields(sp["vtx"][keep])     # el splat viaja dentro de la nube
    if "ply" in formats:
        p = os.path.join(out_dir, stem + ".ply")
        if os.path.abspath(p) == os.path.abspath(src):
            p = os.path.join(out_dir, stem + "_nube.ply")
        write_cloud_ply(p, xyz_out, rgb, ids.astype(np.float32), alpha, campos)
        made.append(p)
    if "las" in formats:
        p = os.path.join(out_dir, stem + ".las")
        write_cloud_las(p, xyz_out, rgb, ids.astype(np.float64), alpha, campos)
        made.append(p)
    if "pts" in formats:
        p = os.path.join(out_dir, stem + ".pts")
        write_cloud_pts(p, xyz_out, rgb, alpha)
        made.append(p)
    ref = {"original": os.path.abspath(src), "nombre": stem, "splats": int(n),
           "exportados": int(len(ids)), "opacidad_min": float(min_opacity or 0),
           "rotacion": R.round(12).tolist(), "comprimido": bool(sp["compressed"]),
           "fecha": time.strftime("%Y-%m-%d %H:%M:%S"),
           "tam_original": os.path.getsize(src), "mtime_original": os.path.getmtime(src),
           "caja": [xyz_out.min(0).round(6).tolist(), xyz_out.max(0).round(6).tolist()]}
    ref_path = os.path.join(out_dir, stem + REF_SUFFIX)
    with open(ref_path, "w", encoding="utf-8") as f:
        json.dump(ref, f, indent=1, ensure_ascii=False)
    _registrar(ref_path, ref)
    log(f"{stem}: {len(ids):,} de {n:,} splats → {', '.join(os.path.basename(m) for m in made)} "
        f"({time.time() - t0:.0f} s)")
    return made


REGISTRY = os.environ.get("ORTOFOTOS_REGISTRO")   # <datos de la app>/nubes_exportadas.json


def _registrar(ref_path, ref):
    """Anota cada exportación en un registro central: así se encuentra el original aunque la nube
    editada se guarde en otra carpeta o con otro nombre."""
    if not REGISTRY:
        return
    try:
        try:
            with open(REGISTRY, encoding="utf-8") as f:
                reg = json.load(f)
        except (OSError, ValueError):
            reg = []
        reg = [r for r in reg if r.get("ref_path") != os.path.abspath(ref_path)]
        reg.append(dict(ref, ref_path=os.path.abspath(ref_path)))
        tmp = REGISTRY + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(reg[-300:], f, ensure_ascii=False)
        os.replace(tmp, REGISTRY)
    except OSError:
        pass


def _caja_nube(path):
    """Caja (min, max) de una nube: rápido con la cabecera LAS; si no, leyendo los puntos."""
    try:
        if path.lower().endswith(".las"):
            with open(path, "rb") as f:
                h = f.read(227)
            v = struct.unpack_from("<6d", h, 179)
            mn, mx = np.array([v[1], v[3], v[5]]), np.array([v[0], v[2], v[4]])
            if np.all(mx >= mn) and np.any(mx > mn):
                return mn, mx
        xyz = read_cloud(path)["xyz"]
        return xyz.min(0), xyz.max(0)
    except Exception:
        return None


def _caja_de_ficha(ref_path, ref):
    c = ref.get("caja")
    if c:
        return np.array(c[0], float), np.array(c[1], float)
    stem = os.path.basename(ref_path)[:-len(REF_SUFFIX)]
    for ext in (".las", ".ply", ".pts"):          # fichas antiguas: se mide la nube exportada
        f = os.path.join(os.path.dirname(ref_path), stem + ext)
        if os.path.isfile(f):
            return _caja_nube(f)
    return None


def guess_ref(cloud_path, log=print):
    """Si la nube editada cambió de nombre o carpeta: busca de qué exportación salió comparando su caja
    con la de cada nube exportada (registro central + fichas en las carpetas de alrededor).
    Devuelve (ruta_ficha, ficha) o (None, None)."""
    caja = _caja_nube(cloud_path)
    if caja is None:
        return None, None
    mn, mx = caja
    fichas = {}
    if REGISTRY and os.path.isfile(REGISTRY):
        try:
            with open(REGISTRY, encoding="utf-8") as f:
                for r in json.load(f):
                    if r.get("ref_path"):
                        fichas[r["ref_path"]] = r
        except (OSError, ValueError):
            pass
    d = os.path.dirname(os.path.abspath(cloud_path))
    tops = [d]
    for _ in range(2):
        d2 = os.path.dirname(tops[-1])
        if d2 and d2 != tops[-1]:
            tops.append(d2)
    vistos = 0
    for top in tops[-1:]:
        for root, dirs, files in os.walk(top):
            if root.count(os.sep) - top.count(os.sep) >= 4:
                dirs[:] = []
            for f in files:
                if f.lower().endswith(REF_SUFFIX):
                    rp = os.path.abspath(os.path.join(root, f))
                    if rp not in fichas:
                        try:
                            with open(rp, encoding="utf-8") as fh:
                                fichas[rp] = json.load(fh)
                        except (OSError, ValueError):
                            pass
                    vistos += 1
            if vistos > 400:
                break
    best, bscore = None, None
    for rp, ref in fichas.items():
        cj = _caja_de_ficha(rp, ref)
        if cj is None:
            continue
        a, b = cj
        diag = float(np.linalg.norm(b - a)) or 1.0
        tol = diag * 0.02
        if np.all(mn >= a - tol) and np.all(mx <= b + tol):      # la nube editada cabe en la exportada
            score = float(np.prod(np.maximum(b - a, 1e-9)))       # la más ajustada gana
            if best is None or score < bscore:
                best, bscore = (rp, ref), score
    if best:
        log(f"  {os.path.basename(cloud_path)}: sale de «{os.path.basename(best[0])[:-len(REF_SUFFIX)]}» "
            "(reconocida por su posición)")
        return best
    return None, None


def ref_for_original(orig):
    """Ficha de exportación de un splat original elegido a mano (para saber con qué giro se exportó).
    Busca en el registro y en las carpetas junto al original. Devuelve (ruta, ficha) o (None, {})."""
    o = os.path.normcase(os.path.abspath(orig))
    cands = []
    if REGISTRY and os.path.isfile(REGISTRY):
        try:
            with open(REGISTRY, encoding="utf-8") as f:
                cands += [r.get("ref_path") for r in json.load(f)]
        except (OSError, ValueError):
            pass
    d = os.path.dirname(o)
    for root, dirs, files in os.walk(d):
        if root.count(os.sep) - d.count(os.sep) >= 2:
            dirs[:] = []
        cands += [os.path.join(root, f) for f in files if f.lower().endswith(REF_SUFFIX)]
    for rp in reversed(cands):
        if not rp or not os.path.isfile(rp):
            continue
        try:
            with open(rp, encoding="utf-8") as f:
                ref = json.load(f)
        except (OSError, ValueError):
            continue
        ro = os.path.normcase(os.path.abspath(ref.get("original", "")))
        if ro == o or re.split(r"[\\/]", ro)[-1] == re.split(r"[\\/]", o)[-1]:
            return rp, ref
    return None, {}


def find_ref(cloud_path):
    """Busca el .splatref.json que corresponde a una nube editada (por nombre)."""
    from pipeline import _norm_name
    d = os.path.dirname(os.path.abspath(cloud_path))
    target = _norm_name(cloud_path)
    stem = os.path.splitext(os.path.basename(cloud_path))[0]
    cands = []
    for folder in (d, os.path.dirname(d)):
        try:
            for f in os.listdir(folder):
                if f.lower().endswith(REF_SUFFIX):
                    cands.append(os.path.join(folder, f))
        except OSError:
            pass
    best, blen = None, -1
    for c in cands:
        base = os.path.basename(c)[:-len(REF_SUFFIX)]
        nb = _norm_name(base + ".x")
        if base.lower() == stem.lower() or nb == target:
            return c
        if nb and (target.startswith(nb) or nb in target) and len(nb) > blen:
            best, blen = c, len(nb)
    return best


def es_splat(path):
    """True si el .ply es un splat (tiene escalas/colores SH o es el comprimido de SuperSplat), no una nube."""
    try:
        with open(path, "rb") as f:
            head = f.read(20000).split(b"end_header", 1)[0]
        return (b" scale_0" in head and b" f_dc_0" in head) or b"packed_position" in head
    except OSError:
        return False


def resolve_original(ref_path, ref):
    """Ruta del splat original: la guardada o uno con el mismo nombre cerca (que sea un splat: la nube
    exportada suele llamarse igual)."""
    p = ref.get("original", "")
    if p and os.path.isfile(p) and es_splat(p):
        return p
    nm = re.split(r"[\\/]", p)[-1] if p else ref.get("nombre", "") + ".ply"
    d = os.path.dirname(ref_path)
    for folder in (d, os.path.dirname(d)):
        for root, dirs, files in os.walk(folder):
            if nm in files and es_splat(os.path.join(root, nm)):
                return os.path.join(root, nm)
            if root.count(os.sep) - folder.count(os.sep) >= 2:
                dirs[:] = []
    return None


_SPLAT_NEED = ("sp_f_dc_0", "sp_f_dc_1", "sp_f_dc_2", "sp_opacity",
               "sp_scale_0", "sp_scale_1", "sp_scale_2", "sp_rot_0", "sp_rot_1", "sp_rot_2", "sp_rot_3")


def has_splat_fields(cl):
    c = cl.get("campos") or {}
    return all(k in c for k in _SPLAT_NEED)


def splat_from_fields(cl):
    """Rearma el splat EXACTO con los campos que viajan en la nube (no hace falta el original)."""
    c = cl["campos"]
    n = len(cl["xyz"])
    if all(k in c for k in ("sp_x", "sp_y", "sp_z")):
        xyz = np.stack([c["sp_x"], c["sp_y"], c["sp_z"]], 1)
    else:
        xyz = cl["xyz"]
    rest = sorted((k for k in c if re.fullmatch(r"sp_f_rest_\d+", k)), key=lambda k: int(k.rsplit("_", 1)[1]))
    props = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
             ("f_dc_0", "<f4"), ("f_dc_1", "<f4"), ("f_dc_2", "<f4")]
    props += [(k[3:], "<f4") for k in rest]
    props += [("opacity", "<f4"), ("scale_0", "<f4"), ("scale_1", "<f4"), ("scale_2", "<f4"),
              ("rot_0", "<f4"), ("rot_1", "<f4"), ("rot_2", "<f4"), ("rot_3", "<f4")]
    v = np.zeros(n, dtype=np.dtype(props))
    v["x"], v["y"], v["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    for k, _t in props[6:]:
        v[k] = c[SP + k]
    return v


def _espaciado(xyz):
    """Distancia típica entre puntos vecinos, por punto (rejilla; superficies escaneadas)."""
    n = len(xyz)
    lo, hi = xyz.min(0), xyz.max(0)
    diag = float(np.linalg.norm(hi - lo)) or 1.0
    # celda tal que caigan unos 30 puntos por celda ocupada (se ajusta en 2 pasadas)
    cell = diag / 200.0
    for _ in range(3):
        k = np.floor((xyz - lo) / cell).astype(np.int64)
        key = (k[:, 0] << 42) | (k[:, 1] << 21) | k[:, 2]
        _u, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
        media = n / len(cnt)
        if 15 <= media <= 60:
            break
        cell *= float(np.sqrt(30.0 / max(media, 1e-6)))
    per = cnt[inv].astype(np.float64)
    return cell / np.sqrt(np.maximum(per, 1.0))          # puntos sobre una superficie


def splat_from_points(cl):
    """Splat APROXIMADO desde una nube sin datos del splat: un punto = un splat redondo y plano en
    color, del tamaño de la separación entre puntos."""
    xyz = cl["xyz"].astype(np.float64)
    n = len(xyz)
    rgb = cl.get("color")
    if rgb is None:
        rgb = np.full((n, 3), 200, np.uint8)
    alfa = cl.get("alfa")
    if alfa is None or float(np.mean(alfa)) < 0.05:
        alfa = np.full(n, 0.95)
    alfa = np.clip(alfa, 0.02, 0.995)
    esp = _espaciado(xyz)
    med = float(np.median(esp)) or 1e-6
    sig = np.maximum(np.clip(esp, 0.6 * med, 1.5 * med) * 0.7, 1e-6)   # sin manchas grandes en los bordes
    props = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
             ("f_dc_0", "<f4"), ("f_dc_1", "<f4"), ("f_dc_2", "<f4"), ("opacity", "<f4"),
             ("scale_0", "<f4"), ("scale_1", "<f4"), ("scale_2", "<f4"),
             ("rot_0", "<f4"), ("rot_1", "<f4"), ("rot_2", "<f4"), ("rot_3", "<f4")]
    v = np.zeros(n, dtype=np.dtype(props))
    v["x"], v["y"], v["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    dc = (rgb.astype(np.float64) / 255.0 - 0.5) / SH_C0
    v["f_dc_0"], v["f_dc_1"], v["f_dc_2"] = dc[:, 0], dc[:, 1], dc[:, 2]
    v["opacity"] = np.log(alfa / (1.0 - alfa))
    ls = np.log(sig)
    v["scale_0"], v["scale_1"], v["scale_2"] = ls, ls, ls
    v["rot_0"] = 1.0
    return v


def cloud_to_splat(cloud_path, out_path, original=None, ref_path=None, log=print):
    """Recupera el splat .ply a partir de una nube editada. Devuelve dict con el resumen."""
    t0 = time.time()
    ref = {}
    ref_path = ref_path or find_ref(cloud_path)
    if ref_path:
        with open(ref_path, encoding="utf-8") as f:
            ref = json.load(f)
    cl = read_cloud(cloud_path)
    if not len(cl["xyz"]):
        raise ValueError("La nube editada está vacía.")
    sin_original = None
    if has_splat_fields(cl):
        sin_original = (splat_from_fields(cl), "exacto, con los datos del splat que lleva la nube")
    else:
        if not original:
            original = resolve_original(ref_path, ref) if ref_path else None
        if not original or not os.path.isfile(original):
            sin_original = (splat_from_points(cl), "aproximado desde los puntos (la nube no trae los "
                            "datos del splat; exporta con esta versión para que salga exacto)")
    if sin_original:
        out, method = sin_original
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        write_ply_struct(out_path, out, [f"comment Ortofotos Spirula: rearmado desde {os.path.basename(cloud_path)}"])
        res = dict(salida=out_path, original=None, total=len(out), quedan=len(out),
                   nube=int(len(cl["xyz"])), metodo=method, transparentes=0, segundos=time.time() - t0)
        log(f"{os.path.basename(out_path)}: {len(out):,} splats ({method}, {res['segundos']:.0f} s)")
        return res
    if ref and ref.get("tam_original") and os.path.getsize(original) != ref["tam_original"]:
        log("  aviso: el splat original cambió desde que se exportó la nube; se intenta igual.")
    sp = load_splat(original)
    n = len(sp["xyz"])
    thr = float(ref.get("opacidad_min", 0.05 if not ref else 0.0))
    if ref.get("rotacion"):
        R = np.array(ref["rotacion"], float)
    else:
        # sin ficha: se prueba con qué giro se exportó (la nube se compara "de pie" o no)
        exp0 = sp["alpha"] >= thr
        rng = np.random.default_rng(0)
        smp = rng.choice(len(cl["xyz"]), min(len(cl["xyz"]), 20000), replace=False)
        sub = dict(cl, xyz=cl["xyz"][smp], color=None if cl.get("color") is None else cl["color"][smp])
        best = (-1.0, np.eye(3))
        diag0 = float(np.linalg.norm(sp["xyz"].max(0) - sp["xyz"].min(0))) or 1.0
        for Rc in _rotaciones_posibles(sp):
            sx = sp["xyz"][exp0] @ Rc.T
            _o, frac = _match_cuantizado(sub, sx, sp["rgb"][exp0])
            if frac < 0.5:
                tol = max(float(cl.get("step") or 0) * 0.75, diag0 * 2e-6)
                frac = float(near_mask(sx, sub["xyz"], tol).mean())
            if frac > best[0]:
                best = (frac, Rc)
            if frac > 0.95:
                break
        R = best[1]
        log(f"  giro de la exportación deducido ({best[0]:.0%} coincide)")
    xyz_file = cl["xyz"] @ R           # deshace la rotación "de pie"
    exported = sp["alpha"] >= thr if thr else np.ones(n, bool)
    diag = float(np.linalg.norm(sp["xyz"].max(0) - sp["xyz"].min(0))) or 1.0
    method = ""
    keep = np.zeros(n, bool)
    ids = cl.get("ids")
    if ids is not None:
        ii = np.round(ids).astype(np.int64)
        ok = (ii >= 0) & (ii < n)
        if ok.mean() > 0.99:
            ii = ii[ok]
            chk = np.linalg.norm(sp["xyz"][ii] - xyz_file[ok], axis=1)
            if np.median(chk) < diag * 1e-3:
                keep[ii] = True
                method = "por splat_id (exacto)"
    if not method and cl.get("cuant") is not None:
        idx_exp = np.nonzero(exported)[0]
        mi, frac = _match_cuantizado(cl, sp["xyz"][idx_exp] @ R.T, sp["rgb"][idx_exp])
        if mi is not None and frac > 0.9:
            keep[idx_exp[mi[mi >= 0]]] = True
            method = f"punto a punto ({frac:.1%} de la nube)"
    if not method:
        step = max(float(cl.get("step") or 0), 0.0)
        tol = max(step * 0.75, diag * 2e-6)
        frac, fct, lab = _best_factor(xyz_file, sp["xyz"][exported], tol, log)
        if frac < 0.5:
            raise ValueError(f"La nube no coincide con el splat original (solo {frac:.0%} de "
                             "puntos). ¿Se movió, giró o escaló la nube al editarla?")
        xyz_file = xyz_file * fct
        keep = near_mask(xyz_file, sp["xyz"], tol) & exported
        method = f"por posición ({frac:.0%} coincide" + (f", unidades {lab}" if lab else "") + ")"
    hidden = ~exported
    nh = 0
    if hidden.any() and keep.any():
        # splats casi transparentes que no se exportaron: se conservan si están dentro de lo que quedó
        r = diag * 0.004
        hk = near_mask(sp["xyz"][keep], sp["xyz"][hidden], r)
        idx = np.nonzero(hidden)[0][hk]
        keep[idx] = True
        nh = int(len(idx))
    if not keep.any():
        raise ValueError("No quedó ningún splat tras la coincidencia.")
    out = sp["vtx"][keep]
    comments = [c for c in sp["comments"] if "Ortofotos" not in c]
    comments.append(f"comment Ortofotos Spirula: recuperado desde {os.path.basename(cloud_path)}")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    write_ply_struct(out_path, out, comments)
    res = dict(salida=out_path, original=original, total=n, quedan=int(keep.sum()),
               nube=int(len(cl["xyz"])), metodo=method, transparentes=nh,
               segundos=time.time() - t0)
    log(f"{os.path.basename(out_path)}: {res['quedan']:,} de {n:,} splats ({method}"
        + (f"; +{nh:,} transparentes" if nh else "") + f", {res['segundos']:.0f} s)")
    return res
