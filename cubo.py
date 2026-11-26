"""Ambientes de 6 caras (interiores y exteriores tipo caja).

Detecta si un modelo de splats es una habitación (interior: 4 paredes + piso + techo,
cámaras dentro) o un volumen visto por fuera (exterior: 4 elevaciones + cubierta), y
renderiza cada cara en ortográfica, alineadas entre sí, más una lámina con el cubo
desplegado. Si no es una caja (una puerta, una ventana, una fachada) se usa el render
normal de siempre (plano dominante + enderezado).

Idea: casi todos los splats de una superficie son "planos"; su eje más corto es la
normal de la superficie. Con esas normales se obtiene la vertical, las dos direcciones
de las paredes (mundo "Manhattan") y la posición de cada pared.
"""
from __future__ import annotations

import json
import math
import os
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from render_ortho import OrthoCam, render, rotate_camera, straighten_angle

CARAS = ("Norte", "Este", "Sur", "Oeste", "Techo", "Suelo")
TIPOS = ("fachada", "interior", "exterior")
CONFIG_NAME = "cubo.json"


def _unit(v):
    v = np.asarray(v, float)
    return v / max(float(np.linalg.norm(v)), 1e-12)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(x, np.float64), -60, 60)))


# ---------------------------------------------------------------------------
# Análisis
# ---------------------------------------------------------------------------
def _peaks_extremes(s, w, lo, hi, nb=400):
    """Posiciones del primer y último pico fuerte de un histograma ponderado."""
    if len(s) < 20 or hi <= lo:
        return None, None
    h, edges = np.histogram(s, bins=nb, range=(lo, hi), weights=w)
    k = np.array([1, 2, 3, 2, 1], float)
    h = np.convolve(h, k / k.sum(), mode="same")
    if h.max() <= 0:
        return None, None
    thr = 0.18 * h.max()
    pk = [i for i in range(1, nb - 1) if h[i] >= thr and h[i] >= h[i - 1] and h[i] >= h[i + 1]]
    if not pk:
        return None, None
    c = 0.5 * (edges[:-1] + edges[1:])
    return float(c[pk[0]]), float(c[pk[-1]])


def _up_without_cams(model, P):
    """Sin cámaras: la vertical es el eje más delgado de la nube (un edificio o una calle son
    más anchos que altos; una habitación también) y el sentido es el lado desde el que se
    filmó (el más limpio, sin la neblina que queda detrás de las superficies)."""
    from render_ortho import _front_haze
    lo, hi = np.percentile(P, 2, axis=0), np.percentile(P, 98, axis=0)
    Q = P[np.all((P >= lo) & (P <= hi), axis=1)]
    if len(Q) < 100:
        return np.array([0.0, 0.0, -1.0])
    ctr = Q.mean(0)
    _, _, vt = np.linalg.svd(Q - ctr, full_matrices=False)
    up = _unit(vt[2])
    r = _unit(vt[0])
    u2 = _unit(np.cross(up, r))
    a, b = (Q - ctr) @ r, (Q - ctr) @ u2
    rng_u = (float(np.percentile(a, 1)), float(np.percentile(a, 99)))
    rng_v = (float(np.percentile(b, 1)), float(np.percentile(b, 99)))
    try:
        top = OrthoCam(ctr, r, u2, up, rng_u, rng_v)
        bot = OrthoCam(ctr, -r, u2, -up, (-rng_u[1], -rng_u[0]), rng_v)
        if _front_haze(model, bot) < _front_haze(model, top):
            up = -up
    except Exception:
        pass
    return up


def analyze(model, cams=None, ups=None, max_pts=400_000, seed=0, up=None, d0=None):
    """Analiza la geometría. Devuelve un dict serializable (listas) con la detección.
    up / d0 fijan a mano la vertical y la dirección de una pared (alineación manual)."""
    fixed_up = None if up is None else _unit(up)
    fixed_d0 = None if d0 is None else np.asarray(d0, float)
    rng = np.random.default_rng(seed)
    xyz = model["xyz"]
    alpha = _sigmoid(model["opacity"])
    thr = min(0.3, float(np.percentile(alpha, 90)))
    idx = np.nonzero(alpha >= thr)[0]
    if len(idx) > max_pts:
        idx = rng.choice(idx, max_pts, replace=False)
    P = xyz[idx]
    a = alpha[idx]
    S = np.exp(model["scale"][idx].astype(np.float64))
    order = np.argsort(S, 1)
    ar = np.arange(len(idx))
    smin, smid, smax = S[ar, order[:, 0]], S[ar, order[:, 1]], S[ar, order[:, 2]]
    flat = smin < 0.35 * smid
    R = model["R"][idx].astype(np.float64)
    N = R[ar, :, order[:, 0]]                       # normal = eje más corto
    W = a * smid * smax
    W = np.minimum(W, np.percentile(W, 99)) if len(W) else W

    # vertical
    if fixed_up is not None:
        up = fixed_up
    elif ups is not None and len(ups):
        up = _unit(np.mean(np.asarray(ups, float), 0))
    else:
        up = _up_without_cams(model, P)
    c = N @ up
    hz = flat & (np.abs(c) > 0.94)
    if fixed_up is None and W[hz].sum() > 0.03 * W[flat].sum():
        v = (np.sign(c[hz])[:, None] * N[hz] * W[hz, None]).sum(0)
        if np.linalg.norm(v) > 0 and v @ up > 0:
            up = _unit(v)

    # direcciones de las paredes (mundo Manhattan)
    e1 = _unit(np.cross(up, [1.0, 0, 0]) if abs(up[0]) < 0.9 else np.cross(up, [0, 1.0, 0]))
    e2 = _unit(np.cross(up, e1))
    vert = flat & (np.abs(N @ up) < 0.25)
    wv = W[vert]
    phi = np.arctan2(N[vert] @ e2, N[vert] @ e1)
    z = (wv * np.exp(4j * phi)).sum()
    theta = float(np.angle(z) / 4) if wv.sum() > 0 else 0.0
    for _ in range(2):   # refinamiento con los que están a menos de 12°
        dphi = (phi - theta + math.pi / 4) % (math.pi / 2) - math.pi / 4
        m = np.abs(dphi) < math.radians(12)
        if wv[m].sum() > 0:
            theta += float(np.average(dphi[m], weights=wv[m]))
    manhattan = float(abs(z) / max(wv.sum(), 1e-12))
    vfrac = float(wv.sum() / max(W[flat].sum(), 1e-12))     # peso de superficies verticales
    d0 = _unit(math.cos(theta) * e1 + math.sin(theta) * e2)
    if fixed_d0 is not None:
        v = fixed_d0 - (fixed_d0 @ up) * up
        if np.linalg.norm(v) > 1e-6:
            d0 = _unit(v)
    d1 = _unit(np.cross(up, d0))
    axes = [d0, d1, up]

    # posición de los planos extremos en cada eje
    sall = [P @ ax for ax in axes]
    tot = W[flat].sum() + 1e-12
    box, sup = [], []
    for k, ax in enumerate(axes):
        s = sall[k]
        lo, hi = np.percentile(s, 0.3), np.percentile(s, 99.7)
        m = flat & (np.abs(N @ ax) > 0.85)
        plo, phi_ = _peaks_extremes(s[m], W[m], lo, hi)
        if plo is None:
            plo, phi_ = float(lo), float(hi)
        box.append([plo, phi_])
        tau = 0.012 * (hi - lo + 1e-9)
        sup.append([float(W[m & (np.abs(s - plo) < tau)].sum() / tot),
                    float(W[m & (np.abs(s - phi_) < tau)].sum() / tot)])
    L0, L1 = box[0][1] - box[0][0], box[1][1] - box[1][0]
    H = box[2][1] - box[2][0]
    walls = [sup[0][1], sup[1][1], sup[0][0], sup[1][0]]      # +d0, +d1, -d0, -d1

    def coverage(k, pos, L_other_axis):
        """Fracción del rectángulo de la pared ocupada por splats pegados a ella."""
        s = sall[k]
        Lk = box[k][1] - box[k][0]
        near = np.abs(s - pos) < max(0.025 * Lk, 1e-6)
        o = 1 - k if k < 2 else 0
        t, h = sall[o][near], sall[2][near]
        (t0, t1), (h0, h1) = box[o], box[2]
        g = np.zeros((16, 32), bool)
        ok = (t >= t0) & (t <= t1) & (h >= h0) & (h <= h1)
        if not ok.any() or t1 <= t0 or h1 <= h0:
            return 0.0
        ix = np.minimum(((t[ok] - t0) / (t1 - t0) * 32).astype(int), 31)
        iy = np.minimum(((h[ok] - h0) / (h1 - h0) * 16).astype(int), 15)
        g[iy, ix] = True
        return float(g.mean())
    cov = [coverage(0, box[0][1], 0), coverage(1, box[1][1], 0),
           coverage(0, box[0][0], 0), coverage(1, box[1][0], 0)]
    n_walls = int(sum(x >= 0.45 for x in cov))
    ratio = min(L0, L1) / max(L0, L1, 1e-9)

    inside = None
    if cams is not None and len(cams):
        C = np.asarray(cams, float)
        c0, c1 = C @ d0, C @ d1
        mg0, mg1 = 0.05 * L0, 0.05 * L1
        ins = (c0 > box[0][0] + mg0) & (c0 < box[0][1] - mg0) & \
              (c1 > box[1][0] + mg1) & (c1 < box[1][1] - mg1)
        inside = float(ins.mean())
    # automático solo con las 4 paredes (una puerta en un pasillo tiene 3: sigue siendo fachada);
    # con 3 caras o menos se puede elegir a mano en el visor
    aerial = (ratio >= 0.3 and 0.12 < H / max(min(L0, L1), 1e-9) < 0.6 and manhattan > 0.2
              and vfrac >= 0.15
              and (inside is None or inside < 0.3) and n_walls >= 1)
    if ratio >= 0.3 and n_walls == 4 and manhattan > 0.25:
        if inside is not None:
            tipo = "interior" if inside >= 0.6 else "exterior"
        else:
            tipo = "interior" if min(sup[2]) >= 0.05 else "exterior"
    elif aerial:
        # manzana / edificio capturado desde arriba (dron): más ancho que alto y sin cámaras
        # dentro -> exterior (elevaciones + cubierta) aunque las paredes no estén completas
        tipo = "exterior"
    else:
        tipo = "fachada"
    why = (f"{n_walls} paredes completas (cobertura " + "/".join(f"{x:.0%}" for x in cov)
           + f"), proporción {ratio:.2f}, altura/ancho {H / max(min(L0, L1), 1e-9):.2f}"
           + f", ortogonalidad {manhattan:.2f}, superficies verticales {vfrac:.0%}"
           + (f", cámaras dentro {inside:.0%}" if inside is not None else ""))
    return {"tipo_auto": tipo, "motivo": why, "up": up.tolist(), "d0": d0.tolist(),
            "d1": d1.tolist(), "caja": box, "soporte": sup, "paredes": walls, "cobertura": cov,
            "camaras_dentro": inside, "verticales": vfrac}


# ---------------------------------------------------------------------------
# Cámaras de cada cara
# ---------------------------------------------------------------------------
def _dirs(an):
    d0, d1 = np.array(an["d0"]), np.array(an["d1"])
    return [d0, d1, -d0, -d1]          # antihorario visto desde arriba


def cardinal(an, norte=0):
    """{'Norte': dirección, ...} (dirección = hacia afuera de esa pared)."""
    D = _dirs(an)
    k = int(norte) % 4
    return {"Norte": D[k], "Este": D[(k - 1) % 4], "Sur": D[(k - 2) % 4], "Oeste": D[(k - 3) % 4]}


def _extent(an, d):
    """(posición del plano, índice de eje, signo) de la pared en la dirección d."""
    for k in (0, 1):
        ax = np.array(an["d%d" % k])
        dot = float(d @ ax)
        if abs(dot) > 0.9:
            lo, hi = an["caja"][k]
            return (hi if dot > 0 else lo), k, (1 if dot > 0 else -1)
    raise ValueError("dirección no alineada")


def _rot(v, axis, deg):
    t = math.radians(deg)
    axis = _unit(axis)
    return v * math.cos(t) + np.cross(axis, v) * math.sin(t) + axis * (axis @ v) * (1 - math.cos(t))


def adjust_cam(cam, adj):
    """Ajuste propio de UNA cara: giro (alrededor de su vertical), inclinación (alrededor de su
    horizontal), rotación de la imagen, mover el plano (hacia el observador +) y ampliar el
    encuadre (fracción)."""
    if not adj:
        return cam
    g, i, r = float(adj.get("giro", 0)), float(adj.get("incl", 0)), float(adj.get("rot", 0))
    mv, amp = float(adj.get("mover", 0)), float(adj.get("ampliar", 0))
    right, up, normal = cam.right.copy(), cam.up.copy(), cam.normal.copy()
    if g:
        right, normal = _rot(right, up, g), _rot(normal, up, g)
    if i:
        up, normal = _rot(up, right, i), _rot(normal, right, i)
    k = 1.0 + amp
    c = OrthoCam(cam.origin + normal * mv, _unit(right), _unit(up), _unit(normal),
                 (cam.u_range[0] * k, cam.u_range[1] * k), (cam.v_range[0] * k, cam.v_range[1] * k))
    c = rotate_camera(c, r) if r else c
    dx, dy = float(adj.get("dx", 0)), float(adj.get("dy", 0))   # desplazar el encuadre
    if dx or dy:
        c = OrthoCam(c.origin, c.right, c.up, c.normal,
                     (c.u_range[0] + dx, c.u_range[1] + dx), (c.v_range[0] + dy, c.v_range[1] + dy))
    return c


def face_cams(an, tipo, norte=0, margin=0.03, ajustes=None):
    """{cara: (OrthoCam, recorte_frontal_absoluto)} para 'interior' o 'exterior'.
    ajustes = {cara: {giro, incl, rot, mover, ampliar}} para corregir cada cara por separado."""
    up = np.array(an["up"])
    d0, d1 = np.array(an["d0"]), np.array(an["d1"])
    (a0, b0), (a1, b1), (au, bu) = an["caja"]
    L = [b0 - a0, b1 - a1]
    H = bu - au
    ctr = 0.5 * (a0 + b0) * d0 + 0.5 * (a1 + b1) * d1 + 0.5 * (au + bu) * up
    card = cardinal(an, norte)
    inner = tipo == "interior"
    out = {}
    for name, D in card.items():
        pos, k, sg = _extent(an, D)
        ax = np.array(an["d%d" % k])
        origin = ctr + (pos - ctr @ ax) * ax
        normal = -D if inner else D                 # hacia el observador
        right = _unit(np.cross(up, normal))
        half = 0.5 * L[1 - k]
        mu, mv = half * 2 * margin, H * margin
        cam = OrthoCam(origin, right, up, normal, (-half - mu, half + mu),
                       (-0.5 * H - mv, 0.5 * H + mv))
        out[name] = (cam, (0.02 if inner else 0.04) * L[k])
    N = card["Norte"]
    for name in ("Techo", "Suelo"):
        top = name == "Techo"
        if not inner and not top:
            continue                                  # el "suelo" de un exterior no se ve
        origin = ctr + ((bu if top else au) - ctr @ up) * up
        if inner:
            normal = -up if top else up
            img_up = -N if top else N                 # para que encaje en el despiece
        else:
            normal, img_up = up, N
        right = _unit(np.cross(img_up, normal))
        hu = 0.5 * abs(L[0] * (right @ d0)) + 0.5 * abs(L[1] * (right @ d1))
        hv = 0.5 * abs(L[0] * (img_up @ d0)) + 0.5 * abs(L[1] * (img_up @ d1))
        cam = OrthoCam(origin, right, img_up, normal, (-hu * (1 + 2 * margin), hu * (1 + 2 * margin)),
                       (-hv * (1 + 2 * margin), hv * (1 + 2 * margin)))
        out[name] = (cam, (0.02 if inner else 0.04) * H)
    if ajustes:
        out = {k: (adjust_cam(c, ajustes.get(k)), clip) for k, (c, clip) in out.items()}
    return out


def rotate_frame(an, yaw_deg=0.0, tilt_a=0.0, tilt_b=0.0):
    """Copia de 'an' con los ejes girados: yaw alrededor de la vertical, inclinaciones
    alrededor de d0 y d1 (grados). La caja conserva sus medidas alrededor del centro."""
    up, d0, d1 = (np.array(an[k], float) for k in ("up", "d0", "d1"))
    (a0, b0), (a1, b1), (au, bu) = an["caja"]
    ctr = 0.5 * (a0 + b0) * d0 + 0.5 * (a1 + b1) * d1 + 0.5 * (au + bu) * up

    def rot(v, axis, deg):
        t = math.radians(deg)
        axis = _unit(axis)
        return v * math.cos(t) + np.cross(axis, v) * math.sin(t) + axis * (axis @ v) * (1 - math.cos(t))
    for axis, deg in ((d0, tilt_a), (d1, tilt_b), (up, yaw_deg)):
        if deg:
            ax0 = axis.copy()
            up, d0, d1 = rot(up, ax0, deg), rot(d0, ax0, deg), rot(d1, ax0, deg)
    out = dict(an)
    out.update(up=_unit(up).tolist(), d0=_unit(d0).tolist(), d1=_unit(d1).tolist())
    half = [0.5 * (b0 - a0), 0.5 * (b1 - a1), 0.5 * (bu - au)]
    c = [float(ctr @ d0), float(ctr @ d1), float(ctr @ up)]
    out["caja"] = [[c[i] - half[i], c[i] + half[i]] for i in range(3)]
    return out


def plan_cam(an, norte=0, extra=0.18):
    """Planta (corte a media altura, vista desde arriba) para el visor."""
    up = np.array(an["up"])
    d0, d1 = np.array(an["d0"]), np.array(an["d1"])
    (a0, b0), (a1, b1), (au, bu) = an["caja"]
    ctr = 0.5 * (a0 + b0) * d0 + 0.5 * (a1 + b1) * d1 + 0.5 * (au + bu) * up
    N = cardinal(an, norte)["Norte"]
    right = _unit(np.cross(N, up))
    L0, L1 = b0 - a0, b1 - a1
    h = 0.5 * max(L0, L1) * (1 + extra)
    return OrthoCam(ctr, right, N, up, (-h, h), (-h, h))


def plan_model(model, an, grow=0.25):
    """Splats útiles para la planta: opacos, no gigantes y dentro de la caja (sin la neblina)."""
    up = np.array(an["up"])
    d0, d1 = np.array(an["d0"]), np.array(an["d1"])
    xyz = model["xyz"]
    keep = _sigmoid(model["opacity"]) > 0.5
    size = max(b - a for a, b in an["caja"])
    keep &= np.exp(model["scale"].max(1)) < 0.03 * size
    for ax, (a, b) in zip((d0, d1, up), an["caja"]):
        s = xyz @ ax
        g = (b - a) * grow
        keep &= (s > a - g) & (s < b + g)
    return {k: (v[keep] if isinstance(v, np.ndarray) and len(v) == len(xyz) else v)
            for k, v in model.items()}


def project(cam, pts, W, H):
    """Puntos 3D -> píxeles de una imagen renderizada con 'cam' de tamaño W x H."""
    pts = np.atleast_2d(np.asarray(pts, float))
    rel = pts - cam.origin
    u, v = rel @ cam.right, rel @ cam.up
    du = cam.u_range[1] - cam.u_range[0]
    dv = cam.v_range[1] - cam.v_range[0]
    return np.stack([(u - cam.u_range[0]) / du * W, (cam.v_range[1] - v) / dv * H], 1)


def wall_segments(an, norte=0):
    """{cara: (p_inicio, p_fin)} a media altura, para dibujar la planta."""
    up = np.array(an["up"])
    d0, d1 = np.array(an["d0"]), np.array(an["d1"])
    (a0, b0), (a1, b1), (au, bu) = an["caja"]
    mid_u = 0.5 * (au + bu)
    corners = {(i, j): i * d0 + j * d1 + mid_u * up for i in (a0, b0) for j in (a1, b1)}
    segs = {}
    for name, D in cardinal(an, norte).items():
        pos, k, sg = _extent(an, D)
        if k == 0:
            segs[name] = (corners[(pos, a1)], corners[(pos, b1)])
        else:
            segs[name] = (corners[(a0, pos)], corners[(b0, pos)])
    return segs


# ---------------------------------------------------------------------------
# Render de las caras y lámina del cubo desplegado
# ---------------------------------------------------------------------------
def preview_model(model, k=350_000):
    """Modelo reducido para la vista en vivo: los splats que más aportan (opacos y grandes)."""
    n = len(model["xyz"])
    if n <= k:
        return model
    w = _sigmoid(model["opacity"]) * np.exp(model["scale"].astype(np.float64).sum(1))
    idx = np.argpartition(-w, k)[:k]
    idx.sort()
    return {key: (v[idx] if isinstance(v, np.ndarray) and len(v) == n else v) for key, v in model.items()}


def render_faces(model, an, tipo, norte=0, caras=None, long_side=4000, background=(0, 0, 0),
                 margin=0.03, log=print, ajustes=None, progress=None):
    """Devuelve {cara: PIL.Image} con todas las caras a la MISMA escala (px por metro).
    progress(i, n, cara) se llama antes de cada cara."""
    fc = face_cams(an, tipo, norte, margin, ajustes)
    if caras:
        fc = {k: v for k, v in fc.items() if k in caras}
    if not fc:
        return {}
    ext = {k: max(c.u_range[1] - c.u_range[0], c.v_range[1] - c.v_range[0]) for k, (c, _) in fc.items()}
    pix = max(ext.values()) / float(long_side)
    imgs = {}
    for i, (k, (cam, clip)) in enumerate(fc.items()):
        if progress:
            progress(i, len(fc), k)
        ls = max(64, int(round(ext[k] / pix)))
        fclip = clip / max(ext[k], 1e-9)
        imgs[k] = render(model, cam, ls, fclip, background=background)
        log(f"  cara {k}: {imgs[k].size[0]}x{imgs[k].size[1]} px")
    return imgs


def _font(size):
    for f in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def sheet(imgs, tipo, max_width=8000, background=(0, 0, 0), titulo=""):
    """Cubo desplegado: interior O-N-E-S con techo arriba y suelo abajo del Norte;
    exterior S-E-N-O con la cubierta sobre la elevación Sur."""
    row = ["Oeste", "Norte", "Este", "Sur"] if tipo == "interior" else ["Sur", "Este", "Norte", "Oeste"]
    anchor = "Norte" if tipo == "interior" else "Sur"
    walls = [imgs.get(k) for k in row]
    wh = max([im.size[1] for im in walls if im is not None] or [1])
    ww = [im.size[0] if im is not None else wh for im in walls]
    top, bot = imgs.get("Techo"), imgs.get("Suelo")
    th = top.size[1] if top is not None else 0
    bh = bot.size[1] if bot is not None else 0
    gap = max(8, wh // 60)
    lab = max(28, wh // 25, sum(ww) // 140)
    W = sum(ww) + gap * (len(ww) + 1)
    H = th + wh + bh + gap * 4 + lab * 3
    canvas = Image.new("RGB", (W, H), tuple(background))
    dr = ImageDraw.Draw(canvas)
    font = _font(lab)
    col = (230, 230, 230) if sum(background) < 384 else (30, 30, 30)
    x = gap
    y_w = gap + (th + lab + gap if top is not None else 0) + lab
    ax = None
    for name, im, w in zip(row, walls, ww):
        if name == anchor:
            ax = (x, w)
        if im is not None:
            canvas.paste(im, (x, y_w + (wh - im.size[1]) // 2))
            dr.text((x + 4, y_w - lab - 2), name, fill=col, font=font)
        x += w + gap
    if ax is not None:
        x0, w0 = ax
        if top is not None:
            canvas.paste(top, (x0 + (w0 - top.size[0]) // 2, gap + lab))
            dr.text((x0 + (w0 - top.size[0]) // 2 + 4, gap), "Techo" if tipo == "interior" else
                    "Cubierta", fill=col, font=font)
        if bot is not None:
            yb = y_w + wh + gap + lab
            canvas.paste(bot, (x0 + (w0 - bot.size[0]) // 2, yb))
            dr.text((x0 + (w0 - bot.size[0]) // 2 + 4, yb - lab - 2), "Suelo", fill=col, font=font)
    if titulo:
        dr.text((W - gap - dr.textlength(titulo, font=font), gap), titulo, fill=col, font=font)
    if W > max_width:
        s = max_width / W
        canvas = canvas.resize((max_width, max(1, int(H * s))), Image.LANCZOS)
    return canvas


def load_config(work_dir):
    try:
        with open(os.path.join(work_dir, CONFIG_NAME), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_config(work_dir, cfg):
    os.makedirs(work_dir, exist_ok=True)
    with open(os.path.join(work_dir, CONFIG_NAME), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=1, ensure_ascii=False)


def faces_dir(results_dir, name):
    return os.path.join(results_dir, f"{name} (6 caras)")


def render_cube_job(model, an, tipo, norte, out_png, name, caras=None, long_side=4000,
                    background=(0, 0, 0), margin=0.03, log=print, ajustes=None, progress=None):
    """Guarda cada cara en '<resultados>/<nombre> (6 caras)/<nombre> - <Cara>.png' y la
    lámina del cubo desplegado en out_png. Devuelve info."""
    imgs = render_faces(model, an, tipo, norte, caras, long_side, background, margin, log, ajustes,
                        progress)
    d = faces_dir(os.path.dirname(os.path.abspath(out_png)), name)
    os.makedirs(d, exist_ok=True)
    for f in os.listdir(d):
        if f.lower().endswith(".png"):
            try:
                os.remove(os.path.join(d, f))
            except OSError:
                pass
    for k, im in imgs.items():
        nm = ("Cubierta" if (k == "Techo" and tipo == "exterior") else k)
        if tipo == "exterior" and k in ("Norte", "Este", "Sur", "Oeste"):
            nm = "Elevación " + k
        elif tipo == "interior" and k in ("Norte", "Este", "Sur", "Oeste"):
            nm = "Pared " + k
        im.save(os.path.join(d, f"{name} - {nm}.png"), compress_level=3)
    sh = sheet(imgs, tipo, background=background,
               titulo=f"{name} · {'interior' if tipo == 'interior' else 'exterior'}")
    sh.save(out_png, compress_level=3)
    return {"caras": sorted(imgs), "size": sh.size, "carpeta": d}


# ---------------------------------------------------------------------------
# Ambiente por partes: una habitación recortada en varios .ply (CARA A, CARA B, …, PLANTA,
# TECHO) del MISMO modelo. Se usan los recortes como guía: el TECHO y la PLANTA dan la
# vertical, cada pared se orienta hacia el centro del ambiente y se renderiza con su propio
# recorte (limpio), todas a la misma escala.
# ---------------------------------------------------------------------------
import re as _re

_ROLE_WORDS = {      # el orden importa: «planta de techos» es techo
    "techo": ("techos?", "cielos?", "cielorrasos?", "cielo raso", "ceiling", "cubiertas?"),
    "suelo": ("plantas?", "pisos?", "suelos?", "floor"),
}
_WALL = r"(caras?|lados?|pared(es)?|muros?|elev(aci[oó]n(es)?)?|fachadas?|vistas?|alzados?)"


def part_role(name):
    n = name.lower()
    for role, words in _ROLE_WORDS.items():
        if any(_re.search(r"\b" + w + r"\b", n) for w in words):
            return role
    return "pared"


def group_key(name):
    n = name.lower().replace("_", " ").replace("-", " ")
    n = _re.sub(r"\b" + _WALL + r"\.?\s*[a-z0-9]{0,3}\b\.?", " ", n)
    for words in _ROLE_WORDS.values():
        for w in words:
            n = _re.sub(r"\b" + w + r"\b", " ", n)
    n = _re.sub(r"\b(ambientes?|interior|amb|de|del|la|las|los|el)\b", " ", n)
    n = _re.sub(r"[^a-z0-9áéíóúñ ]+", " ", n)
    return " ".join(n.split())


def _geo_part(P):
    lo, hi = np.percentile(P, 2, axis=0), np.percentile(P, 98, axis=0)
    Q = P[np.all((P >= lo) & (P <= hi), axis=1)]
    if len(Q) < 50:
        return None
    return {"lo": lo, "hi": hi, "c": Q.mean(0)}


def geo_roles(allp, name_roles=None, strict=True):
    """Rol de cada recorte por su FORMA, sin depender del nombre del archivo.
    En los modelos de Spirula/SuperSplat la vertical es -Y: el techo y la planta son los recortes
    bajos en altura que están en los extremos de la altura de las paredes (techo arriba, planta
    abajo). Los roles leídos del nombre (techo/suelo) mandan; «pared» es solo «no reconocido».
    Devuelve {'roles', 'up'} o None si los recortes no forman un ambiente."""
    name_roles = name_roles or {}
    G = {n: _geo_part(P) for n, P in allp.items()}
    G = {n: g for n, g in G.items() if g is not None}
    if len(G) < 2:
        return None
    a = 1                                   # eje Y
    up = np.array([0.0, -1.0, 0.0])
    ext = {n: float(g["hi"][a] - g["lo"][a]) for n, g in G.items()}
    H = max(ext.values())
    if H <= 0:
        return None
    roles = {}
    for n, g in G.items():
        nr = name_roles.get(n, "pared")
        roles[n] = nr if nr in ("techo", "suelo") else ("h" if ext[n] < 0.75 * H else "pared")
    walls = [n for n in G if roles[n] == "pared"]
    if not walls:
        return None
    wlo = min(G[n]["lo"][a] for n in walls)
    whi = max(G[n]["hi"][a] for n in walls)
    wh = max(whi - wlo, 1e-9)
    ok = True
    for n in G:
        if roles[n] != "h":
            continue
        t = (G[n]["c"][a] - wlo) / wh        # 0 = arriba (Y menor), 1 = abajo
        if t < 0.35:
            roles[n] = "techo"
        elif t > 0.65:
            roles[n] = "suelo"
        else:                               # a media altura: no es techo ni planta
            roles[n] = "pared"
            ok = False
    if strict:
        hz = [n for n in G if roles[n] in ("techo", "suelo")]
        if len(walls) < 2 or not hz or not ok:
            return None
        # las paredes ocupan casi toda la altura del ambiente y empiezan/terminan a la misma
        # altura (son cortes del mismo modelo); modelos sueltos de puertas no cumplen esto
        if any(ext[n] < 0.55 * wh for n in walls):
            return None
        if any(G[n]["lo"][a] - wlo > 0.2 * wh or whi - G[n]["hi"][a] > 0.2 * wh for n in walls):
            return None
        # todos los recortes juntos (mismo modelo, no escenas distintas)
        diag = {n: float(np.linalg.norm(g["hi"] - g["lo"])) for n, g in G.items()}
        ulo = np.min([g["lo"] for g in G.values()], 0)
        uhi = np.max([g["hi"] for g in G.values()], 0)
        if float(np.linalg.norm(uhi - ulo)) > 2.5 * max(diag.values()):
            return None
    return {"roles": roles, "up": up}


_ELEM_RE = re.compile(r"(?:^|[^a-z0-9])([pv])\s*-?\s*(\d{1,3})(?=$|[^0-9])", re.I)


def element_code(name):
    """Código de puerta/ventana en el nombre ('b12-p19-a' -> 'p19', 'B12-V09-B' -> 'v9'), o None.
    Archivos con códigos distintos son elementos distintos: nunca forman un ambiente."""
    m = _ELEM_RE.search(os.path.basename(name).lower())
    return f"{m.group(1)}{int(m.group(2))}" if m else None


def find_ambientes(items, geo=None):
    """items [(ply, nombre)] -> ({clave: [(ply, nombre, rol)]}, resto). Un ambiente es un grupo
    de 3 o más recortes del mismo modelo con al menos un TECHO o PLANTA y paredes.
    Primero por nombre; si los nombres no alcanzan y se da geo(partes) -> roles|None, se
    reconoce por la forma de los recortes de la carpeta (sirve con cualquier nombre)."""
    by_dir = {}
    for ply, name in items:
        by_dir.setdefault(os.path.dirname(os.path.abspath(ply)), []).append((ply, name))
    amb, rest = {}, []
    for d, its in by_dir.items():
        folder = os.path.basename(d)
        fkey = group_key(folder) or folder.lower()
        groups = {}
        for ply, name in its:
            k = group_key(name) or fkey
            groups.setdefault(k, []).append((ply, name, part_role(name)))
        valid, left = {}, []
        for k, parts in groups.items():
            roles = [r for _p, _n, r in parts]
            if len(parts) >= 3 and ("techo" in roles or "suelo" in roles) and roles.count("pared") >= 1:
                valid[k] = parts
            else:
                left += parts
        if len(valid) <= 1 and geo is not None:
            cand = (next(iter(valid.values())) if valid else []) + left
            if len(cand) >= 3 and left and len({c for c in (element_code(n) for _p, n, _r in cand)
                                                if c}) < 2:
                try:
                    rr = geo(cand)
                except Exception:
                    rr = None
                if rr:
                    key = next(iter(valid)) if valid else fkey
                    valid = {key: [(p, n, rr.get(n, r)) for p, n, r in cand if n in rr] or cand}
                    left = [x for x in cand if x[1] not in rr]
        for k, parts in valid.items():
            kk = k if k not in amb else f"{k} ({folder})"
            amb[kk] = parts
        rest += [(p, n) for p, n, _r in left]
    return amb, rest


def _opaque_pts(model, k=200_000, seed=0):
    a = _sigmoid(model["opacity"])
    idx = np.nonzero(a > 0.5)[0]
    if len(idx) > k:
        idx = np.random.default_rng(seed).choice(idx, k, replace=False)
    return model["xyz"][idx]


def _subset(model, mask):
    n = len(model["xyz"])
    return {k: (v[mask] if isinstance(v, np.ndarray) and len(v) == n else v) for k, v in model.items()}


def _ransac_plane(P, rng, ref, want_parallel, iters=500):
    """Normal del plano con más puntos: paralela a 'ref' (techo/planta) o perpendicular (pared)."""
    if len(P) < 100:
        return None
    lo, hi = np.percentile(P, 1, axis=0), np.percentile(P, 99, axis=0)
    Q = P[np.all((P >= lo) & (P <= hi), axis=1)]
    if len(Q) > 60_000:
        Q = Q[rng.choice(len(Q), 60_000, replace=False)]
    tau = 0.006 * float(np.linalg.norm(hi - lo))
    best, bn, bp = -1, None, None
    for _ in range(iters):
        p1, p2, p3 = Q[rng.choice(len(Q), 3, replace=False)]
        n = np.cross(p2 - p1, p3 - p1)
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        n = n / nn
        c = abs(float(n @ ref))
        if (want_parallel and c < 0.75) or (not want_parallel and c > 0.6):
            continue
        sc = int(np.count_nonzero(np.abs((Q - p1) @ n) < tau))
        if sc > best:
            best, bn, bp = sc, n, p1
    if bn is None:
        return None
    inl = Q[np.abs((Q - bp) @ bn) < tau]
    if len(inl) > 50:
        _, _, vt = np.linalg.svd(inl - inl.mean(0), full_matrices=False)
        bn = vt[2]
    return _unit(bn)


def wide_tilt(img):
    """Giro de cámara (grados, ±45) que deja las líneas dominantes de la imagen horizontales y
    verticales. 0 si no hay una orientación clara."""
    from PIL import ImageFilter
    g = img.convert("L")
    if max(g.size) > 1000:
        k = 1000 / max(g.size)
        g = g.resize((max(1, int(g.size[0] * k)), max(1, int(g.size[1] * k))))
    a = np.asarray(g.filter(ImageFilter.GaussianBlur(2)), np.float32)
    gx = np.zeros_like(a)
    gy = np.zeros_like(a)
    gx[:, 1:-1] = a[:, 2:] - a[:, :-2]
    gy[1:-1, :] = a[2:, :] - a[:-2, :]
    mag = np.hypot(gx, gy)
    m = mag > max(np.percentile(mag, 92), 6)
    if m.sum() < 300:
        return 0.0
    th = (np.degrees(np.arctan2(gy[m], gx[m])) + 45) % 90 - 45
    h, e = np.histogram(th, bins=180, range=(-45, 45), weights=mag[m])
    h = np.convolve(np.r_[h[-3:], h, h[:3]], np.ones(5) / 5, "same")[3:-3]
    j = int(np.argmax(h))
    if h[j] < 2.5 * h.mean():
        return 0.0
    return -float(0.5 * (e[j] + e[j + 1]))


def ambiente_cams(parts, margin=0.04, progress=None, read=None):
    """Prepara un ambiente por partes: {'models', 'cams': {nombre: (OrthoCam, recorte)}, 'roles'}."""
    import render_ortho as ro
    read = read or ro.read_splat_ply
    models = {}
    bad = {}
    for i, (ply, name, role) in enumerate(parts):
        if progress:
            progress(i, len(parts) * 2, f"leyendo {name}")
        try:
            models[name] = read(ply)
        except Exception as e:     # una parte dañada no detiene el resto del ambiente
            bad[name] = str(e) or e.__class__.__name__
    parts = [p for p in parts if p[1] in models]
    if len(parts) < 2:
        raise ValueError("no hay suficientes partes legibles en el ambiente" +
                         (f" ({', '.join(bad)} dañado)" if bad else ""))
    # modelo conjunto (muestra) para la orientación del ambiente
    allp = {n: _opaque_pts(m) for n, m in models.items()}
    keys = [k for k in next(iter(models.values())) if isinstance(next(iter(models.values()))[k], np.ndarray)]
    merged = {}
    for k in keys:
        arrs = []
        for m in models.values():
            v = m[k]
            if isinstance(v, np.ndarray) and len(v) == len(m["xyz"]):
                step = max(1, len(v) // 250_000)
                arrs.append(v[::step])
        if len(arrs) == len(models):
            merged[k] = np.concatenate(arrs)
    merged["rest"] = None
    merged["mip"] = False
    roles = {n: r for _p, n, r in parts}
    g = geo_roles(allp, roles, strict=False)    # completa los roles que el nombre no dice
    if g:
        roles.update({n: r for n, r in g["roles"].items() if roles.get(n) == "pared"})
    tops = [np.median(allp[n], 0) for n in allp if roles[n] == "techo"]
    bots = [np.median(allp[n], 0) for n in allp if roles[n] == "suelo"]
    up0 = None
    if tops and bots:
        up0 = _unit(np.mean(tops, 0) - np.mean(bots, 0))
    elif g:
        up0 = g["up"]
    # vertical precisa con los planos de cada recorte: paredes ⟂ arriba, techo/planta ∥ arriba
    if up0 is not None:
        rng0 = np.random.default_rng(1)
        M = np.zeros((3, 3))
        wall_ns = []
        for n_, P_ in allp.items():
            horiz = roles[n_] != "pared"
            pn = _ransac_plane(P_, rng0, ref=up0, want_parallel=horiz)
            if pn is None:
                continue
            if horiz:
                M += 2.0 * np.outer(pn, pn)
            else:
                M -= np.outer(pn, pn)
                wall_ns.append(pn)
        if wall_ns:
            ev, V = np.linalg.eigh(M)
            upr = _unit(V[:, -1])
            if upr @ up0 < 0:
                upr = -upr
            if upr @ up0 > math.cos(math.radians(60)):
                up0 = upr
        d0g = None
        if wall_ns:
            w = wall_ns[0] - (wall_ns[0] @ up0) * up0
            if np.linalg.norm(w) > 1e-6:
                d0g = w
        an = analyze(merged, None, None, up=up0, d0=d0g)
    else:
        an = analyze(merged, None, None)
    up, d0, d1 = (np.array(an[k], float) for k in ("up", "d0", "d1"))
    allpts = np.vstack(list(allp.values()))
    ctr = np.median(allpts, 0)
    size = float(np.percentile(np.linalg.norm(allpts - ctr, axis=1), 95)) * 2
    dirs = [d0, d1, -d0, -d1]
    walls = [n for n in allp if roles[n] == "pared"]
    # referencia para la orientación de techo y planta: la primera pared (orden alfabético)
    ref_wall = sorted(walls)[0] if walls else None
    cams = {}
    for name in allp:
        P = allp[name]
        c = np.median(P, 0) - ctr
        if roles[name] == "techo":
            D = up
        elif roles[name] == "suelo":
            D = -up
        else:
            h = c - (c @ up) * up
            D = max(dirs, key=lambda d: float(h @ d))
        cams[name] = D
    ref_dir = cams[ref_wall] if ref_wall else d0
    built = {}
    rng = np.random.default_rng(0)
    cos_lim = math.cos(math.radians(10))     # paredes a escuadra con el ambiente (±10°)
    cos_h = math.cos(math.radians(12))
    for name, D in cams.items():
        P = allp[name]
        # plano de ESTA cara: RANSAC solo con normales parecidas a la dirección esperada
        lo, hi = np.percentile(P, 1, axis=0), np.percentile(P, 99, axis=0)
        Pp = P[np.all((P >= lo) & (P <= hi), axis=1)]
        tau = 0.006 * float(np.linalg.norm(hi - lo))
        best, bn, bp = -1, None, None
        for _ in range(900):
            i3 = rng.choice(len(Pp), 3, replace=False)
            p1, p2, p3 = Pp[i3]
            n = np.cross(p2 - p1, p3 - p1)
            nn = np.linalg.norm(n)
            if nn < 1e-12:
                continue
            n = n / nn
            if abs(float(n @ D)) < (cos_lim if roles[name] == "pared" else cos_h):
                continue
            sc = int(np.count_nonzero(np.abs((Pp - p1) @ n) < tau))
            if sc > best:
                best, bn, bp = sc, n, p1
        if bn is None:
            bn, bp = D, np.median(P, 0)
        inl = Pp[np.abs((Pp - bp) @ bn) < tau]
        if len(inl) > 50:
            c0 = inl.mean(0)
            _, _, vt = np.linalg.svd(inl - c0, full_matrices=False)
            bn, bp = (vt[2] if vt[2] @ bn > 0 else -vt[2]), c0
        normal = _unit(bn if (ctr - bp) @ bn > 0 else -bn)      # hacia dentro del ambiente
        ref = up if roles[name] == "pared" else (ref_dir if roles[name] == "suelo" else -ref_dir)
        img_up = ref - (ref @ normal) * normal
        if np.linalg.norm(img_up) < 1e-6:
            img_up = d0 - (d0 @ normal) * normal
        img_up = _unit(img_up)
        right = _unit(np.cross(img_up, normal))
        Q = (inl if len(inl) > 200 else P) - bp
        u, v = Q @ right, Q @ img_up
        u0, u1 = np.percentile(u, 0.3), np.percentile(u, 99.7)
        v0, v1 = np.percentile(v, 0.3), np.percentile(v, 99.7)
        mu, mv = (u1 - u0) * margin, (v1 - v0) * margin
        cam = OrthoCam(bp, right, img_up, normal, (u0 - mu, u1 + mu), (v0 - mv, v1 + mv))
        # enderezar la cara con sus propias líneas (puertas, ventanas, bordes)
        try:
            ext0 = max(u1 - u0, v1 - v0)
            pmod = preview_model(models[name], 300_000)
            fcl = (0.04 * size) / max(ext0, 1e-9)     # enderezar SOLO con la pared (sin muebles)
            prev = render(pmod, cam, 1000, fcl)
            ang = wide_tilt(prev)            # hasta ±45°: líneas dominantes a horizontal/vertical
            if abs(ang) > 0.3:
                cam = rotate_camera(cam, ang)
                prev = render(pmod, cam, 1000, fcl)
            ang = straighten_angle(prev)       # ajuste fino
            if abs(ang) > 0.05:
                cam = rotate_camera(cam, ang)
        except Exception:
            pass
        # profundidad: TODO lo que el recorte trae delante de su plano (muebles, columnas…);
        # el recorte ya lo eligió el usuario, no se corta nada de él
        dep = (P - cam.origin) @ cam.normal
        clip_d = max(0.04 * size, float(np.percentile(dep, 99.7)) * 1.02 + 0.01 * size)
        built[name] = (cam, clip_d)
    return {"models": models, "cams": built, "roles": roles, "malos": bad}


def ui_to_adj(ui, cam):
    """Ajuste de la interfaz (grados y %) -> ajuste en unidades del modelo para adjust_cam."""
    if not ui:
        return None
    ext = max(cam.u_range[1] - cam.u_range[0], cam.v_range[1] - cam.v_range[0])
    d = dict(ui)
    d["mover"] = float(d.get("mover", 0)) / 100.0 * ext
    d["ampliar"] = float(d.get("ampliar", 0)) / 100.0
    d["dx"] = float(d.get("dx", 0)) / 100.0 * ext
    d["dy"] = float(d.get("dy", 0)) / 100.0 * ext
    return d


def door_cams(parts, front_clip=0.12, margin=0.05, cams=None, ups=None, read=None):
    """Puerta/ventana (una sola fachada) con la misma forma que ambiente_cams, para usar el
    mismo visor y los mismos ajustes (girar, inclinar, enderezar, mover, encuadre)."""
    import render_ortho as ro
    read = read or ro.read_splat_ply
    p, n, _r = parts[0]
    m = read(p)
    fc = front_clip or 0.12
    cam = ro.door_camera(m, cams, ups, front_clip=fc, margin=margin, log=lambda *_: None)
    ext = max(cam.u_range[1] - cam.u_range[0], cam.v_range[1] - cam.v_range[0])
    return {"models": {n: m}, "cams": {n: (cam, fc * ext)}, "roles": {n: "fachada"}, "malos": []}


def render_ambiente(parts, out_dir, long_side=4000, background=(0, 0, 0), margin=0.04,
                    log=print, progress=None, rots=None, read=None, ajustes=None, setup=None, only=None):
    """parts [(ply, nombre, rol)] -> guarda <out_dir>/<nombre>.png de cada parte.
    ajustes = {nombre: ajuste de la interfaz}; only = nombres a guardar. Devuelve {nombre: info}."""
    rots = rots or {}
    ajustes = ajustes or {}
    st = setup or ambiente_cams(parts, margin, progress, read)
    models, roles = st["models"], st["roles"]
    built = {n: (adjust_cam(c, ui_to_adj(ajustes.get(n), c)), cl) for n, (c, cl) in st["cams"].items()}
    infos = {}
    ext = {n: max(c.u_range[1] - c.u_range[0], c.v_range[1] - c.v_range[0]) for n, (c, _cl) in built.items()}
    pix = max(ext.values()) / float(long_side)
    os.makedirs(out_dir, exist_ok=True)
    for i, (name, (cam, clip)) in enumerate(built.items()):
        if only and name not in only:
            continue
        if progress:
            progress(len(parts) + i, len(parts) * 2, name)
        ls = max(64, int(round(ext[name] / pix)))
        im = render(models[name], cam, ls, clip / max(ext[name], 1e-9), background=background)
        k = int(rots.get(name, 0)) % 4
        for _ in range(k):
            im = im.transpose(Image.ROTATE_270)
        out = os.path.join(out_dir, name + ".png")
        im.save(out, compress_level=3)
        infos[name] = {"size": im.size, "rol": roles[name], "salida": out}
        log(f"  {name}: {roles[name]} → {im.size[0]}x{im.size[1]} px")
    return infos
