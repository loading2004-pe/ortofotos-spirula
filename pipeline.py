"""Detección de carpetas y procesamiento por lotes con la línea de comandos de Spirula.

Cada carpeta con >= 2 videos, o con suficientes fotos, es un "trabajo". Las etapas de cada trabajo:
  1. extraer   -> spirula sam extract (fotogramas de cada video)
  2. sfm       -> spirula sfm auto     (poses de cámara = "Crear conjunto de datos")
  3. entrenar  -> spirula train        (5000 pasos, resolución 1/4)
  4. render    -> render ortográfico propio (render_ortho.py), alineado y enderezado

Las etapas se solapan entre trabajos (mientras uno entrena en la GPU, otros extraen
fotogramas o calculan poses en la CPU). Cada etapa tiene su propio límite de
trabajos simultáneos. Si se interrumpe, al volver a ejecutar se retoma donde quedó.
"""
from __future__ import annotations

import glob
import json
import os
import re
import shutil
import subprocess
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

VIDEO_EXT = (".mov", ".mp4", ".m4v", ".mkv", ".avi", ".wmv", ".mts", ".m2ts", ".3gp", ".webm")
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")

DEFAULT_CONFIG = {
    "spirula_exe": r"C:\Spirula\spirula.exe",
    "carpeta_trabajo": "",          # vacío = <raíz>/_orto_trabajo
    "carpeta_resultados": "",       # vacío = <raíz>/resultados
    "pasos_entrenamiento": 5000,
    "divisor_resolucion": 4,        # 4 = 1/4
    "calidad_sfm": "high",          # igual que la ventana de Spirula
    "fotogramas_por_segundo": 2,    # igual que la ventana de Spirula (video_fps = 2)
    "ventana_nitidez": 3,           # elige el más nítido de cada 3
    "extraer_adaptativo": False,
    "args_extra_extraer": "",
    "args_extra_sfm": "",
    "args_extra_train": "",
    "paralelo_extraer": 2,
    "paralelo_sfm": 2,
    "paralelo_entrenar": 1,         # medido: 1 a la vez es ~40% más rápido que 3 (la GPU se satura)
    "paralelo_render": 2,
    "render_lado_largo_px": 7680,
    "render_recorte_frontal": 0.12, # elimina lo que esté delante de la fachada (fracción del ancho)
    "render_plano_vertical": True,
    "render_enderezar": True,
    "render_fondo_transparente": False,
    "render_fondo_color": [0, 0, 0],
    "render_margen": 0.05,
    "nombre_minusculas": True,
    "min_videos": 2,
    "min_fotos": 8,                 # fotos mínimas para procesar una carpeta solo con fotos
    "entrenar_cache_ram": True,     # --cache-images cpu si caben en RAM (no cambia el resultado)
    "url_actualizaciones": "",      # vacío = servidor por defecto
    "actualizar_auto": True,
}


def load_config(path: str) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            cfg.update(json.load(f))
    return cfg


def save_config(path: str, cfg: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Escaneo
# ---------------------------------------------------------------------------
@dataclass
class Job:
    name: str
    folder: str
    videos: list
    images: list = field(default_factory=list)
    status: str = "pendiente"
    detail: str = ""
    result: str = ""
    times: dict = field(default_factory=dict)


@dataclass
class ScanReport:
    jobs: list
    empty: list          # carpetas sin videos ni fotos
    insufficient: list   # (carpeta, n_videos, n_fotos) con menos del mínimo


def job_name(folder: str, lower: bool) -> str:
    leaf = os.path.basename(folder.rstrip("\\/"))
    parent = os.path.basename(os.path.dirname(folder.rstrip("\\/")))
    m = re.match(r"(?i)^\s*cara[\s\-_.]*([a-z0-9]+)\s*$", leaf)
    if m and parent:                      # "CARA -A", "CARA-A", "cara A" -> "cara a"
        name = f"{parent} cara {m.group(1)}"
    else:
        name = leaf
    name = re.sub(r'[<>:"/\\|?*]', "_", name).strip()
    return name.lower() if lower else name


def scan(roots, cfg) -> ScanReport:
    skip_dirs = {"_orto_trabajo", "resultados", "outputs", "renders", "sparse",
                 "features", "logs", "masks", "ortofotos_limpias", "ply_para_supersplat",
                 "ply_limpios"}

    def skip(x):
        xl = x.lower()
        return (xl in skip_dirs or x.startswith(".") or xl.endswith("_dataset")
                or "_dataset_" in xl or xl.startswith("images"))
    jobs, empty, insufficient = [], [], []
    for root in roots:
        for d, dirs, files in os.walk(root):
            dirs[:] = sorted(x for x in dirs if not skip(x))
            vids = sorted(os.path.join(d, f) for f in files
                          if f.lower().endswith(VIDEO_EXT))
            imgs = sorted(os.path.join(d, f) for f in files
                          if f.lower().endswith(IMAGE_EXT) and not f.lower().startswith("x_"))
            mv, mf = int(cfg["min_videos"]), int(cfg.get("min_fotos", 8))
            ok = (len(vids) >= mv or len(imgs) >= mf
                  or (len(vids) >= 1 and len(imgs) >= max(1, mf // 2)))
            if ok:
                jobs.append(Job(job_name(d, cfg["nombre_minusculas"]), d, vids, imgs))
            elif vids or len(imgs) >= 3:
                insufficient.append((d, len(vids), len(imgs)))
            elif not dirs and not imgs:
                empty.append(d)
    # nombres repetidos -> sufijo
    seen = {}
    for j in jobs:
        k = j.name
        if k in seen:
            seen[k] += 1
            j.name = f"{k} ({seen[k]})"
        else:
            seen[k] = 1
    return ScanReport(jobs, empty, insufficient)


# ---------------------------------------------------------------------------
# Ejecución
# ---------------------------------------------------------------------------
class Cancelled(Exception):
    pass


class Runner:
    def __init__(self, cfg, roots, on_update=None, log=None):
        self.cfg = cfg
        self.root = roots[0]
        self.work = cfg["carpeta_trabajo"] or os.path.join(self.root, "_orto_trabajo")
        self.results = cfg["carpeta_resultados"] or os.path.join(self.root, "resultados")
        self.export_dir, self.clean_dir = side_dirs(cfg, self.root)
        self.on_update = on_update or (lambda job: None)
        self._log = log or print
        self.cancel = threading.Event()
        self.soft = threading.Event()      # terminar lo que está corriendo y no empezar más
        self.procs = set()
        self.lock = threading.Lock()
        self.sem = {k: threading.Semaphore(max(1, int(cfg[f"paralelo_{k}"])))
                    for k in ("extraer", "sfm", "entrenar", "render")}

    def log(self, msg):
        self._log(time.strftime("%H:%M:%S ") + msg)

    # -- utilidades ---------------------------------------------------------
    def _run(self, job, args, logfile):
        if self.cancel.is_set():
            raise Cancelled()
        with open(logfile, "a", encoding="utf-8", errors="replace") as lf:
            lf.write("\n$ " + subprocess.list2cmdline(args) + "\n")
            lf.flush()
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 creationflags=flags, text=True, encoding="utf-8",
                                 errors="replace", bufsize=1)
            with self.lock:
                self.procs.add(p)
            try:
                last = ""
                for line in p.stdout:
                    lf.write(line)
                    line = line.strip()
                    if line and line != last:
                        last = line
                        job.detail = line[:150]
                        self.on_update(job)
                    if self.cancel.is_set():
                        p.kill()
                rc = p.wait()
            finally:
                with self.lock:
                    self.procs.discard(p)
        if self.cancel.is_set():
            raise Cancelled()
        if rc != 0:
            raise RuntimeError(f"{os.path.basename(args[0])} {args[1]} terminó con código {rc} "
                               f"(ver {logfile})")

    def stop_after_current(self):
        """Deja terminar las etapas en marcha (p. ej. un entrenamiento) y no empieza otras."""
        self.soft.set()

    def _gate(self, job):
        if self.soft.is_set() or self.cancel.is_set():
            raise Cancelled()

    def stop(self):
        self.cancel.set()
        with self.lock:
            for p in list(self.procs):
                try:
                    p.kill()
                except Exception:
                    pass

    def _set(self, job, status, detail=""):
        job.status, job.detail = status, detail
        self.on_update(job)

    # -- etapas ---------------------------------------------------------------
    def process(self, job: Job):
        cfg, exe = self.cfg, self.cfg["spirula_exe"]
        jdir = os.path.join(self.work, job.name)
        ds = os.path.join(jdir, "dataset")
        img = os.path.join(ds, "images")
        runs = os.path.join(jdir, "runs")
        logf = os.path.join(jdir, "log.txt")
        os.makedirs(jdir, exist_ok=True)
        done = lambda stage: os.path.exists(os.path.join(jdir, f".ok_{stage}"))
        mark = lambda stage: open(os.path.join(jdir, f".ok_{stage}"), "w").close()
        out_png = os.path.join(self.results, job.name + ".png")
        solo_render = bool(cfg.get("_solo_render"))
        try:
            # 1) fotogramas
            if not solo_render and not done("extraer"):
                self._set(job, "en cola: extraer")
                with self.sem["extraer"]:
                    self._gate(job)
                    self._set(job, "extrayendo fotogramas")
                    t = time.time()
                    shutil.rmtree(img, ignore_errors=True)
                    os.makedirs(img)
                    used = set()
                    if job.images:
                        self._set(job, f"copiando {len(job.images)} fotos")
                        copy_photos(job.images, os.path.join(img, "fotos"))
                    for v in job.videos:
                        stem = re.sub(r"[^\w\-]", "_", os.path.splitext(os.path.basename(v))[0])
                        while stem in used:
                            stem += "_b"
                        used.add(stem)
                        fps = video_fps(v) or 30.0
                        skip = max(1, round(fps / float(cfg["fotogramas_por_segundo"])))
                        args = [exe, "sam", "extract", v, "-o", os.path.join(img, stem),
                                "--skip", str(skip), "--keep", str(cfg["ventana_nitidez"])]
                        if cfg["extraer_adaptativo"]:
                            args.append("--adaptive")
                        args += cfg["args_extra_extraer"].split()
                        self._run(job, args, logf)
                    n = sum(len(fs) for _r, _d, fs in os.walk(img))
                    if n < 6:
                        raise RuntimeError(f"solo {n} imágenes para reconstruir (mínimo 6)")
                    job.times["extraer"] = time.time() - t
                    mark("extraer")
            # 2) poses (crear conjunto de datos)
            if not solo_render and not done("sfm"):
                self._set(job, "en cola: sfm")
                with self.sem["sfm"]:
                    self._gate(job)
                    self._set(job, "calculando poses (SfM)")
                    t = time.time()
                    for d in ("sparse", "features"):
                        shutil.rmtree(os.path.join(ds, d), ignore_errors=True)
                    args = [exe, "sfm", "auto", ds, "-o", ds,
                            "--quality", cfg["calidad_sfm"]] + cfg["args_extra_sfm"].split()
                    self._run(job, args, logf)
                    if not glob.glob(os.path.join(ds, "sparse", "*", "images.*")):
                        raise RuntimeError("SfM no generó cámaras (sparse/0)")
                    job.times["sfm"] = time.time() - t
                    mark("sfm")
            # 3) entrenamiento
            run_dir = os.path.join(runs, "modelo")
            if cfg.get("_forzar_entrenar") and done("entrenar"):
                os.remove(os.path.join(jdir, ".ok_entrenar"))
            if not solo_render and not done("entrenar"):
                self._set(job, "en cola: entrenar (GPU)")
                with self.sem["entrenar"]:
                    self._gate(job)
                    self._set(job, "entrenando")
                    t = time.time()
                    shutil.rmtree(runs, ignore_errors=True)
                    args = [exe, "train", "--data", ds,
                            "--num-iterations", str(cfg["pasos_entrenamiento"]),
                            "--train-resolution-divisor", str(cfg["divisor_resolucion"]),
                            "--disable-viewer", "1",
                            "--output-dir-prefix", runs, "--output-dir-name", "modelo"]
                    extra = cfg["args_extra_train"].split()
                    perf = []
                    if cfg.get("entrenar_cache_ram", True) and "--cache-images" not in extra \
                            and ram_fits_images(img, int(cfg["divisor_resolucion"])):
                        perf = ["--cache-images", "cpu"]
                    gpu_note(logf)
                    try:
                        self._run(job, args + perf + extra, logf)
                    except RuntimeError:
                        if not perf or time.time() - t > 120 or self.cancel.is_set():
                            raise
                        # esta versión de Spirula no acepta la opción: se repite sin ella
                        shutil.rmtree(runs, ignore_errors=True)
                        self._run(job, args + extra, logf)
                    job.times["entrenar"] = time.time() - t
                    mark("entrenar")
                    old = os.path.join(self.clean_dir, job.name + ".ply")
                    if os.path.isfile(old):   # modelo nuevo: el .ply limpio anterior ya no vale
                        os.replace(old, old + ".antiguo")
            trained = newest_ply(runs)
            cleaned = os.path.join(self.clean_dir, job.name + ".ply")
            ply = cleaned if os.path.isfile(cleaned) else trained
            if not ply:
                raise RuntimeError("no se encontró el .ply entrenado en " + runs)
            origen = " · desde .ply limpio" if ply == cleaned else ""
            # 4) render ortográfico
            self._set(job, "en cola: render")
            with self.sem["render"]:
                if self.cancel.is_set() or (self.soft.is_set() and not os.path.exists(
                        os.path.join(jdir, ".ok_entrenar"))):
                    raise Cancelled()
                self._set(job, "renderizando ortofoto")
                t = time.time()
                from render_ortho import render_run
                for old in (out_png, os.path.join(self.results, "x_" + job.name + ".png")):
                    if os.path.exists(old):
                        os.remove(old)
                info = render_run(ply, out_png, workspace=ds if os.path.isdir(ds) else None,
                                  run_dir=os.path.join(runs, "modelo"),
                                  long_side=int(cfg["render_lado_largo_px"]),
                                  front_clip=cfg["render_recorte_frontal"] or None,
                                  vertical=cfg["render_plano_vertical"],
                                  auto_straighten=cfg["render_enderezar"],
                                  transparent=cfg["render_fondo_transparente"],
                                  background=tuple(cfg.get("render_fondo_color", (0, 0, 0))),
                                  save_cams=(os.path.join(cfg["_cam_store"], job.name + ".json")
                                             if cfg.get("_cam_store") else None),
                                  margin=cfg["render_margen"])
                job.times["render"] = time.time() - t
            job.result = out_png
            tt = sum(job.times.values())
            try:
                with open(logf, "a", encoding="utf-8") as lf:
                    lf.write("\n[tiempos] " + "  ".join(f"{k}={v/60:.1f} min" for k, v in job.times.items())
                             + "\n")
            except OSError:
                pass
            self._set(job, "listo", f"{info['size'][0]}x{info['size'][1]} px, "
                                    f"giro {info['roll_fix_deg']:+.2f}°, {tt/60:.1f} min{origen}")
            self.log(f"OK  {job.name}")
        except Cancelled:
            self._set(job, "cancelado")
        except Exception as e:
            self._set(job, "error", str(e))
            with open(logf, "a", encoding="utf-8") as lf:
                lf.write(traceback.format_exc())
            self.log(f"ERROR {job.name}: {e}")

    def run_all(self, jobs):
        os.makedirs(self.work, exist_ok=True)
        os.makedirs(self.results, exist_ok=True)
        n = len(jobs)
        width = max(int(self.cfg[f"paralelo_{k}"]) for k in self.sem) * 4
        with ThreadPoolExecutor(max_workers=min(n, max(4, width))) as ex:
            list(ex.map(self.process, jobs))
        ok = sum(j.status == "listo" for j in jobs)
        self.log(f"Terminado: {ok}/{n} correctos. Resultados en {self.results}")


def copy_photos(paths, dest):
    """Copia las fotos al dataset (los originales no se tocan). Las JPG con giro EXIF
    se enderezan; el resto se copia tal cual."""
    os.makedirs(dest, exist_ok=True)
    used = set()
    for i, src in enumerate(paths):
        base, ext = os.path.splitext(os.path.basename(src))
        base = re.sub(r"[^\w\-]", "_", base) or f"foto_{i:04d}"
        name = base
        while name.lower() in used:
            name += "_b"
        used.add(name.lower())
        ext_l = ext.lower()
        dst = os.path.join(dest, name + (".jpg" if ext_l in (".jpeg",) else ext_l))
        rotated = False
        if ext_l in (".jpg", ".jpeg", ".tif", ".tiff"):
            try:
                from PIL import Image, ImageOps
                with Image.open(src) as im:
                    if im.getexif().get(0x0112, 1) not in (1, None):
                        out = ImageOps.exif_transpose(im)
                        out.save(dst, quality=96, subsampling=0) if ext_l in (".jpg", ".jpeg") \
                            else out.save(dst)
                        rotated = True
            except Exception:
                rotated = False
        if ext_l in (".bmp", ".webp"):   # formatos que no todo SfM lee: a PNG
            from PIL import Image
            with Image.open(src) as im:
                im.convert("RGB").save(os.path.splitext(dst)[0] + ".png")
            continue
        if not rotated:
            shutil.copy2(src, dst)


def video_fps(path):
    """FPS de un .mov/.mp4 leyendo sus cajas MP4 (sin ffmpeg)."""
    import struct
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            end = f.tell()

            def boxes(start, stop):
                pos = start
                while pos + 8 <= stop:
                    f.seek(pos)
                    size, typ = struct.unpack(">I4s", f.read(8))
                    hdr = 8
                    if size == 1:
                        size = struct.unpack(">Q", f.read(8))[0]
                        hdr = 16
                    elif size == 0:
                        size = stop - pos
                    if size < hdr:
                        return
                    yield typ, pos + hdr, pos + size
                    pos += size

            def find(start, stop, typ):
                for t, a, b in boxes(start, stop):
                    if t == typ:
                        return a, b
                return None

            moov = find(0, end, b"moov")
            if not moov:
                return None
            for t, a, b in boxes(*moov):
                if t != b"trak":
                    continue
                mdia = find(a, b, b"mdia")
                if not mdia:
                    continue
                hdlr = find(*mdia, b"hdlr")
                f.seek(hdlr[0] + 8)
                if f.read(4) != b"vide":
                    continue
                mdhd = find(*mdia, b"mdhd")
                f.seek(mdhd[0])
                ver = f.read(1)[0]
                if ver == 1:
                    f.seek(mdhd[0] + 20)
                    ts, dur = struct.unpack(">IQ", f.read(12))
                else:
                    f.seek(mdhd[0] + 12)
                    ts, dur = struct.unpack(">II", f.read(8))
                stbl = find(*find(*mdia, b"minf"), b"stbl")
                stsz = find(*stbl, b"stsz")
                f.seek(stsz[0] + 8)
                count = struct.unpack(">I", f.read(4))[0]
                if dur and ts and count:
                    return count / (dur / ts)
    except Exception:
        return None
    return None


def total_ram_bytes():
    try:
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                        ("avail", ctypes.c_ulonglong), ("tp", ctypes.c_ulonglong), ("ap", ctypes.c_ulonglong),
                        ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong), ("ae", ctypes.c_ulonglong)]
        m = MS()
        m.l = ctypes.sizeof(MS)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
            return int(m.total)
    except Exception:
        pass
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except Exception:
        return 0


def ram_fits_images(img_dir, divisor=1, fraction=0.35):
    """¿Caben las imágenes decodificadas en RAM? (4 bytes/píxel, con margen)."""
    try:
        from PIL import Image
        files = [os.path.join(d, f) for d, _s, fs in os.walk(img_dir) for f in fs
                 if f.lower().endswith(IMAGE_EXT)]
        if not files:
            return False
        px = 0
        step = max(1, len(files) // 20)
        sample = files[::step]
        for f in sample:
            with Image.open(f) as im:
                px += im.size[0] * im.size[1]
        need = px / len(sample) * len(files) * 4 / max(1, divisor) ** 2
        ram = total_ram_bytes()
        return bool(ram) and need < ram * fraction
    except Exception:
        return False


def gpu_note(logf):
    """Anota en el registro cuánta memoria de GPU usan otros programas."""
    try:
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True,
                             timeout=10, creationflags=flags).stdout.strip().splitlines()
        if out:
            used, total = [int(x) for x in out[0].split(",")]
            msg = f"[gpu] memoria en uso antes de entrenar: {used} de {total} MB"
            if used > 1500:
                msg += " (cierra navegadores u otros programas para dejar más memoria al entrenamiento)"
            with open(logf, "a", encoding="utf-8") as lf:
                lf.write("\n" + msg + "\n")
    except Exception:
        pass


def side_dirs(cfg, root):
    """Carpetas para SuperSplat: exportados y limpios (junto a 'resultados')."""
    res = cfg.get("carpeta_resultados") or os.path.join(root, "resultados")
    base = os.path.dirname(os.path.abspath(res))
    return os.path.join(base, "ply_para_supersplat"), os.path.join(base, "ply_limpios")


def work_dir(cfg, root, name):
    return os.path.join(cfg.get("carpeta_trabajo") or os.path.join(root, "_orto_trabajo"), name)


def export_plys(cfg, root, names):
    """Copia el .ply entrenado de cada trabajo a ply_para_supersplat/<nombre>.ply."""
    exp, _clean = side_dirs(cfg, root)
    os.makedirs(exp, exist_ok=True)
    out = []
    for n in names:
        src = newest_ply(os.path.join(work_dir(cfg, root, n), "runs"))
        if not src:
            out.append((n, None, "no tiene .ply entrenado"))
            continue
        dst = os.path.join(exp, n + ".ply")
        shutil.copy2(src, dst)
        try:   # cámaras al lado, para poder renderizar el limpio en cualquier PC
            from render_ortho import cams_from_workspace, save_cams_json
            wd = work_dir(cfg, root, n)
            c, u = cams_from_workspace(os.path.join(wd, "dataset"), os.path.join(wd, "runs", "modelo"),
                                       log=lambda *_: None)
            if c is not None:
                save_cams_json(os.path.join(exp, n + ".cam.json"), c, u)
        except Exception:
            pass
        out.append((n, dst, ""))
    return out


def _norm_name(s):
    s = os.path.splitext(os.path.basename(s))[0].lower()
    s = re.sub(r"\.compressed$", "", s)
    s = re.sub(r"^x_", "", s)
    s = re.sub(r"[\s_\-]*(cleaned|clean|limpio|limpia|editado|edit|final|v\d+)$", "", s)
    s = re.sub(r"\s*\(\d+\)$", "", s)
    return re.sub(r"[\s_]+", " ", s).strip()


def match_job_name(filename, names):
    """Nombre de trabajo que corresponde a un archivo (p-34 cara a-cleaned.ply -> p-34 cara a)."""
    f = _norm_name(filename)
    norm = {re.sub(r"[\s_]+", " ", n.lower()).strip(): n for n in names}
    if f in norm:
        return norm[f]
    hits = [n for k, n in norm.items() if k and k in f]
    return max(hits, key=len) if hits else None


def link_clean_ply(cfg, root, src, name):
    """Copia un .ply limpio a ply_limpios/<nombre>.ply."""
    _exp, clean = side_dirs(cfg, root)
    os.makedirs(clean, exist_ok=True)
    dst = os.path.join(clean, name + ".ply")
    if os.path.abspath(src) != os.path.abspath(dst):
        shutil.copy2(src, dst)
    return dst


def clean_names(cfg, root, names):
    """Trabajos que ya tienen un .ply limpio vinculado."""
    _exp, clean = side_dirs(cfg, root)
    return [n for n in names if os.path.isfile(os.path.join(clean, n + ".ply"))]


# ---------------------------------------------------------------------------
# Revisar independiente: carpeta de .ply limpios -> ortofotos
# ---------------------------------------------------------------------------
MANIFEST = ".ortofotos_manifest.json"


def list_plys(folder, recursive=True):
    """[(ruta, nombre)] de los .ply de una carpeta; nombre sacado del archivo."""
    found = []
    for d, dirs, files in os.walk(folder):
        dirs[:] = [x for x in dirs if not x.startswith(".") and x.lower() != "ortofotos_limpias"]
        for f in sorted(files):
            if f.lower().endswith(".ply"):
                found.append(os.path.join(d, f))
        if not recursive:
            break
    out, seen = [], {}
    for pth in found:
        n = _norm_name(pth) or "modelo"
        if n in seen:
            seen[n] += 1
            n = f"{n} ({seen[n]})"
        else:
            seen[n] = 1
        out.append((pth, n))
    return out


def find_cams(name, ply_path, cfg, root=None, store=None):
    """Busca las cámaras de un modelo: junto al .ply, en el almacén del programa o en
    la carpeta de trabajo de la raíz. Devuelve (centros, arriba, origen) o (None, None, '')."""
    from render_ortho import load_cams_json, cams_from_workspace
    d = os.path.dirname(ply_path)
    stem = os.path.splitext(os.path.basename(ply_path))[0]
    cands = [os.path.join(d, stem + ".cam.json"), os.path.join(d, name + ".cam.json")]
    if store:
        cands.append(os.path.join(store, name + ".json"))
    for c in cands:
        if os.path.isfile(c):
            try:
                cc, uu = load_cams_json(c)
                return cc, uu, os.path.basename(c)
            except Exception:
                pass
    roots = [r for r in (root, os.path.dirname(d), os.path.dirname(os.path.dirname(d))) if r]
    for r in roots:
        wd = work_dir(cfg, r, name)
        if os.path.isdir(os.path.join(wd, "dataset")):
            cc, uu = cams_from_workspace(os.path.join(wd, "dataset"), os.path.join(wd, "runs", "modelo"),
                                         log=lambda *_: None)
            if cc is not None:
                return cc, uu, "_orto_trabajo"
    return None, None, ""


def read_manifest(out_dir):
    p = os.path.join(out_dir, MANIFEST)
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def write_manifest(out_dir, man):
    with open(os.path.join(out_dir, MANIFEST), "w", encoding="utf-8") as f:
        json.dump(man, f, indent=1, ensure_ascii=False)


def render_linked(items, out_dir, cfg, root=None, store=None, on_item=None, cancel=None):
    """Renderiza [(ruta_ply, nombre)] en out_dir sin Spirula ni carpetas de 'Procesar'.
    on_item(nombre, estado, detalle). Guarda ajustes por imagen (voltear/girar) en un manifiesto."""
    from render_ortho import render_run
    os.makedirs(out_dir, exist_ok=True)
    man = read_manifest(out_dir)
    lock = threading.Lock()
    on_item = on_item or (lambda *a: None)

    def one(item):
        ply, name = item
        if cancel is not None and cancel.is_set():
            return
        prev = man.get(name, {})
        flip, rot = bool(prev.get("flip")), int(prev.get("rot", 0))
        on_item(name, "renderizando", "")
        try:
            cams, ups, origen = find_cams(name, ply, cfg, root, store)
            out_png = os.path.join(out_dir, name + ".png")
            for old in (out_png, os.path.join(out_dir, "x_" + name + ".png")):
                if os.path.exists(old):
                    os.remove(old)
            t = time.time()
            info = render_run(ply, out_png, cams=cams, ups=ups, flip=flip, rot90=rot,
                              long_side=int(cfg["render_lado_largo_px"]),
                              front_clip=cfg["render_recorte_frontal"] or None,
                              vertical=cfg["render_plano_vertical"],
                              auto_straighten=cfg["render_enderezar"],
                              transparent=cfg["render_fondo_transparente"],
                              margin=cfg["render_margen"],
                              background=tuple(cfg.get("render_fondo_color", (0, 0, 0))),
                              log=lambda *_: None)
            with lock:
                man[name] = {"ply": ply, "flip": flip, "rot": rot}
                write_manifest(out_dir, man)
            nota = "con cámaras" if cams is not None else "lado detectado automáticamente"
            on_item(name, "listo", f"{info['size'][0]}x{info['size'][1]} px, "
                                   f"{time.time() - t:.0f} s, {nota}")
        except Exception as e:
            on_item(name, "error", str(e))

    with ThreadPoolExecutor(max_workers=max(1, int(cfg.get("paralelo_render", 2)))) as ex:
        list(ex.map(one, items))


def newest_ply(runs_dir):
    plys = glob.glob(os.path.join(runs_dir, "**", "*.ply"), recursive=True)
    plys = [p for p in plys if "sparse" not in os.path.basename(p).lower()]
    return max(plys, key=os.path.getmtime) if plys else None


# ---------------------------------------------------------------------------
# Revisión: marcar mal (x_) / bien
# ---------------------------------------------------------------------------
def set_mark(path: str, bad: bool) -> str:
    d, f = os.path.split(path)
    clean = f[2:] if f.lower().startswith("x_") else f
    new = os.path.join(d, ("x_" if bad else "") + clean)
    if new != path:
        os.replace(path, new)
    return new
