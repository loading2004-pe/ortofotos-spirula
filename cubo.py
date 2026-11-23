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

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from render_ortho import OrthoCam, render, rotate_camera

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
    return rotate_camera(c, r) if r else c


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
