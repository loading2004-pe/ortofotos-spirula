"""Ortofotos Spirula — procesamiento por lotes + revisión.

Uso: abre el programa (o arrastra la carpeta raíz sobre el .exe), pulsa
Escanear, luego Procesar. Al terminar, pestaña Revisar: B = bien, M = mal (x_).
"""
from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

FROZEN = getattr(sys, "frozen", False)
APP_DIR = os.environ.get("ORTO_APP_DIR") or os.path.dirname(
    sys.executable if FROZEN else os.path.abspath(__file__))
PENDING_PATH = None   # se define tras DATA_DIR


def _writable(d):
    try:
        t = os.path.join(d, ".orto_test")
        open(t, "w").close()
        os.remove(t)
        return True
    except OSError:
        return False


# Junto al .exe (portátil, p. ej. en un USB); si no se puede escribir ahí, en AppData.
DATA_DIR = APP_DIR if _writable(APP_DIR) else os.path.join(
    os.environ.get("APPDATA", os.path.expanduser("~")), "OrtofotosSpirula")
os.makedirs(DATA_DIR, exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(
    os.environ.get("LOCALAPPDATA", DATA_DIR), "OrtofotosSpirula", "cache"))
CONFIG_PATH = os.path.join(DATA_DIR, "ortofotos_config.json" if FROZEN else "config.json")
PENDING_PATH = os.path.join(DATA_DIR, "proceso_pendiente.json")


def code_version():
    v = os.environ.get("ORTO_CODE_VERSION")
    if v:
        return v
    try:
        import app_version
        return app_version.VERSION
    except Exception:
        return "?"


def find_spirula(current=""):
    """Busca spirula.exe: el configurado, junto al programa, Escritorio, Descargas, PATH."""
    import shutil
    home = os.path.expanduser("~")
    cands = [current, os.path.join(APP_DIR, "spirula.exe"),
             os.path.join(os.path.dirname(APP_DIR), "spirula.exe"),
             os.path.join(home, "Desktop", "spirula.exe"),
             os.path.join(home, "OneDrive", "Desktop", "spirula.exe"),
             os.path.join(home, "OneDrive", "Escritorio", "spirula.exe"),
             os.path.join(home, "Downloads", "spirula.exe"),
             shutil.which("spirula") or ""]
    for c in cands:
        if c and os.path.isfile(c):
            return c
    for base in (os.path.join(home, "Desktop"), os.path.join(home, "Downloads"),
                 os.environ.get("ProgramFiles", r"C:\Program Files")):
        try:
            for d in os.listdir(base):
                c = os.path.join(base, d, "spirula.exe")
                if os.path.isfile(c):
                    return c
        except OSError:
            pass
    return ""

import pipeline as pl  # noqa: E402

if sys.platform.startswith("win"):
    try:  # texto nítido con el escalado de pantalla de Windows
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
from PIL import Image, ImageTk  # noqa: E402

PERFILES = {  # perfiles de calidad rápidos
    "Rápido": {"pasos_entrenamiento": 3000, "divisor_resolucion": 4, "calidad_sfm": "medium",
               "render_lado_largo_px": 4000},
    "Normal (como Spirula a mano)": {"pasos_entrenamiento": 5000, "divisor_resolucion": 4,
                                    "calidad_sfm": "high", "render_lado_largo_px": 7680},
    "Alta": {"pasos_entrenamiento": 10000, "divisor_resolucion": 2, "calidad_sfm": "high",
             "render_lado_largo_px": 7680},
    "Máxima (lento)": {"pasos_entrenamiento": 30000, "divisor_resolucion": 1,
                       "calidad_sfm": "extreme", "render_lado_largo_px": 7680},
}
CALIDADES = [("Rápida (2000 px)", 2000), ("Normal (4000 px)", 4000),
             ("Alta (7680 px)", 7680), ("Máxima (10000 px)", 10000)]

CHOICES = {
    "calidad_sfm": ["low", "medium", "high", "extreme"],
    "divisor_resolucion": ["1", "2", "4", "8"],
    "render_lado_largo_px": ["2000", "4000", "6000", "7680", "10000"],
    "pasos_entrenamiento": ["3000", "5000", "7000", "10000", "15000", "30000"],
}

FIELDS = [  # (clave, etiqueta, tipo)
    ("spirula_exe", "Ejecutable de Spirula (spirula.exe)", "file"),
    ("carpeta_resultados", "Carpeta de resultados (vacío = <raíz>/resultados)", "dir"),
    ("carpeta_trabajo", "Carpeta de trabajo temporal (vacío = <raíz>/_orto_trabajo)", "dir"),
    ("pasos_entrenamiento", "Pasos de entrenamiento", int),
    ("divisor_resolucion", "Resolución de imagen para entrenar 1/N (N =)", int),
    ("calidad_sfm", "Calidad SfM (low / medium / high / extreme)", str),
    ("fotogramas_por_segundo", "Fotogramas por segundo de video a extraer", float),
    ("ventana_nitidez", "Ventana de nitidez (elige el más nítido de N)", int),
    ("extraer_adaptativo", "Extracción adaptativa", bool),
    ("paralelo_extraer", "Simultáneos: extracción", int),
    ("paralelo_sfm", "Simultáneos: SfM (poses)", int),
    ("paralelo_entrenar", "Simultáneos: entrenamiento (GPU; 1 es lo más rápido)", int),
    ("paralelo_render", "Simultáneos: render", int),
    ("render_lado_largo_px", "Render: lado largo (px)", int),
    ("render_recorte_frontal", "Render: recorte delante de la fachada (0 = sin recorte)", float),
    ("render_enderezar", "Render: enderezar automáticamente", bool),
    ("render_plano_vertical", "Render: fachada vertical (puertas/ventanas)", bool),
    ("render_fondo_transparente", "Render: fondo transparente", bool),
    ("render_margen", "Render: margen alrededor", float),
    ("nombre_minusculas", "Nombres en minúsculas (p-34 cara a)", bool),
    ("min_videos", "Mínimo de videos por carpeta", int),
    ("min_fotos", "Mínimo de fotos por carpeta (si no hay videos)", int),
    ("args_extra_extraer", "Args extra: sam extract", str),
    ("args_extra_sfm", "Args extra: sfm auto", str),
    ("args_extra_train", "Args extra: train", str),
    ("entrenar_cache_ram", "Entrenar con imágenes en RAM si caben (más rápido, mismo resultado)", bool),
    ("actualizar_auto", "Buscar actualizaciones al abrir", bool),
    ("url_actualizaciones", "Servidor de actualizaciones (vacío = el de siempre)", str),
]


class App(tk.Tk):
    def __init__(self, start_root=None):
        super().__init__()
        self.title(f"Ortofotos Spirula — automatización  ·  versión {code_version()}")
        self._restart_when_idle = False
        self.report_callback_exception = self._on_error
        # Con escalado de pantalla, Tk puede recibir el "soltar" del clic con coordenadas
        # desplazadas y no activar el botón: se activa si el botón quedó presionado.
        for cls in ("TButton", "TCheckbutton", "TRadiobutton"):
            self.bind_class(cls, "<ButtonPress-1>", self._btn_press)
            self.bind_class(cls, "<ButtonRelease-1>", lambda e: "break")
        dpi = self.winfo_fpixels("1i")
        self.tk.call("tk", "scaling", dpi / 72.0)
        k = dpi / 96.0
        import tkinter.font as tkfont
        for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont"):
            try:
                tkfont.nametofont(name).configure(size=10)
            except tk.TclError:
                pass
        st = ttk.Style(self)
        st.configure("Treeview", rowheight=int(22 * k))
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{int(sw * 0.8)}x{int(sh * 0.82)}+{int(sw * 0.1)}+{int(sh * 0.05)}")
        self.cfg = pl.load_config(CONFIG_PATH)
        found = find_spirula(self.cfg.get("spirula_exe", ""))
        if found and found != self.cfg.get("spirula_exe"):
            self.cfg["spirula_exe"] = found
            pl.save_config(CONFIG_PATH, self.cfg)
        self.q = queue.Queue()
        self.jobs, self.runner, self.report = [], None, None
        self.root_var = tk.StringVar(value=start_root or self.cfg.get("ultima_raiz", ""))

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)
        self.nb = nb
        self._build_process(nb)
        self._build_review(nb)
        self._build_convert(nb)
        self._build_config(nb)
        self.after(150, self._poll)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        nb.bind("<<NotebookTabChanged>>", self._on_tab)
        upd = os.environ.get("ORTO_UPDATE_MSG", "")
        if upd:
            self.q.put(("log", f"Actualizaciones: {upd}"))
        if os.path.isfile(PENDING_PATH):
            self.after(600, self._resume_pending)
        elif start_root or os.path.isdir(self.root_var.get()):
            self.after(300, self.do_scan)
        self.after(2 * 3600 * 1000, self._periodic_update)

    def _btn_press(self, e):
        """Activa el botón al presionar (robusto ante el escalado de pantalla)."""
        w = e.widget
        try:
            if w.instate(["disabled"]):
                return "break"
            w.focus_set()
            w.state(["pressed"])
            self.after(90, lambda: (w.state(["!pressed"]) if w.winfo_exists() else None))
            self.after(10, w.invoke)
        except tk.TclError:
            pass
        return "break"

    def _on_error(self, exc, val, tb):
        """Errores inesperados: se guardan en errores.log y se muestran."""
        import traceback
        txt = "".join(traceback.format_exception(exc, val, tb))
        try:
            with open(os.path.join(DATA_DIR, "errores.log"), "a", encoding="utf-8") as f:
                f.write(time.strftime("%Y-%m-%d %H:%M:%S\n") + txt + "\n")
        except OSError:
            pass
        messagebox.showerror("Error", f"{val}\n\n(detalle en errores.log)")

    # ------------------------------------------------------------- Procesar
    def _build_process(self, nb):
        f = ttk.Frame(nb, padding=8)
        nb.add(f, text="1. Procesar")
        top = ttk.Frame(f)
        top.pack(fill="x")
        ttk.Label(top, text="Carpeta raíz:").pack(side="left")
        ttk.Entry(top, textvariable=self.root_var).pack(side="left", fill="x", expand=True, padx=4)
        ttk.Button(top, text="Elegir…", command=self._pick_root).pack(side="left")
        ttk.Button(top, text="Escanear", command=self.do_scan).pack(side="left", padx=4)
        self.btn_run = ttk.Button(top, text="▶ Procesar", command=self.do_run)
        self.btn_run.pack(side="left")
        self.btn_soft = ttk.Button(top, text="⏸ Detener al terminar lo actual",
                                   command=self.do_soft_stop, state="disabled")
        self.btn_soft.pack(side="left", padx=(4, 0))
        self.btn_stop = ttk.Button(top, text="■ Detener ya", command=self.do_stop, state="disabled")
        self.btn_stop.pack(side="left", padx=4)
        ttk.Button(top, text="Limpiar lista", command=self.clear_process).pack(side="left")

        self.summary = ttk.Label(f, text="Elige la carpeta raíz y pulsa Escanear.")
        self.summary.pack(fill="x", pady=6)

        pan = ttk.PanedWindow(f, orient="vertical")
        pan.pack(fill="both", expand=True)
        cols = ("videos", "estado", "detalle")
        tv = ttk.Treeview(pan, columns=cols, show="tree headings", selectmode="extended")
        tv.heading("#0", text="Nombre")
        tv.heading("videos", text="Videos / fotos")
        tv.heading("estado", text="Estado")
        tv.heading("detalle", text="Detalle")
        tv.column("#0", width=220)
        tv.column("videos", width=110, anchor="center")
        tv.column("estado", width=170)
        tv.column("detalle", width=640)
        tv.tag_configure("listo", foreground="#1a7f37")
        tv.tag_configure("error", foreground="#c62828")
        tv.tag_configure("aviso", foreground="#9a6700")
        tv.bind("<Double-1>", self._open_job_folder)
        self.tv = tv
        pan.add(tv, weight=3)
        self.logbox = tk.Text(pan, height=8, font=("Consolas", 10))
        pan.add(self.logbox, weight=1)
        ttk.Label(f, text="Doble clic en una fila abre su carpeta. Los trabajos se retoman "
                          "donde quedaron si los vuelves a procesar.").pack(anchor="w")

    def clear_process(self):
        """Vacía la tabla y el registro (no borra ningún archivo)."""
        if self.runner and self.btn_stop["state"] == "normal":
            messagebox.showinfo("Limpiar lista", "Espera a que termine o pulsa Detener.")
            return
        self.tv.delete(*self.tv.get_children())
        self.logbox.delete("1.0", "end")
        self.jobs, self.report = [], None
        self.summary.config(text="Lista vacía. Elige la carpeta raíz y pulsa Escanear.")

    def _pick_root(self):
        d = filedialog.askdirectory(title="Carpeta raíz con las carpetas de puertas/ventanas")
        if d:
            self.root_var.set(d)
            self.do_scan()

    def do_scan(self):
        root = self.root_var.get().strip().strip('"')
        if not os.path.isdir(root):
            messagebox.showerror("Carpeta", "La carpeta raíz no existe.")
            return
        self.cfg["ultima_raiz"] = root
        pl.save_config(CONFIG_PATH, self.cfg)
        rep = pl.scan([root], self.cfg)
        self.report, self.jobs = rep, rep.jobs
        self.tv.delete(*self.tv.get_children())
        res = self.cfg["carpeta_resultados"] or os.path.join(root, "resultados")
        for j in rep.jobs:
            done = os.path.exists(os.path.join(res, j.name + ".png")) or \
                os.path.exists(os.path.join(res, "x_" + j.name + ".png"))
            j.status = "ya procesado" if done else "pendiente"
            self.tv.insert("", "end", iid=j.name, text=j.name,
                           values=(media_txt(j), j.status, os.path.relpath(j.folder, root)))
        for d, nv, nf in rep.insufficient:
            self.tv.insert("", "end", text=os.path.relpath(d, root), tags=("aviso",),
                           values=(f"{nv} vid · {nf} fotos", "omitido",
                                   f"pocos archivos: mínimo {self.cfg['min_videos']} videos "
                                   f"o {self.cfg.get('min_fotos', 8)} fotos"))
        for d in rep.empty:
            self.tv.insert("", "end", text=os.path.relpath(d, root), tags=("aviso",),
                           values=(0, "vacía", ""))
        pend = sum(j.status == "pendiente" for j in rep.jobs)
        self.summary.config(text=f"{len(rep.jobs)} carpetas con videos/fotos ({pend} pendientes)  ·  "
                                 f"{len(rep.empty)} vacías  ·  {len(rep.insufficient)} con "
                                 f"pocos archivos")

    def do_run(self, only=None, cfg_override=None):
        if not self.jobs:
            self.do_scan()
        if not self.jobs:
            return
        solo = bool((cfg_override or {}).get("_solo_render"))
        if not solo and not os.path.isfile(self.cfg["spirula_exe"]):
            found = find_spirula(self.cfg.get("spirula_exe", ""))
            if not found:
                messagebox.showinfo("Spirula", "No encontré spirula.exe. Indícame dónde está.")
                found = filedialog.askopenfilename(title="Ubica spirula.exe",
                                                   filetypes=[("spirula.exe", "*.exe")])
            if not found:
                return
            self.cfg["spirula_exe"] = found
            pl.save_config(CONFIG_PATH, self.cfg)
            v, _t = self.vars["spirula_exe"]
            v.set(found)
        sel = only if only is not None else [self.tv.item(i, "text") for i in self.tv.selection()]
        todo = [j for j in self.jobs if j.name in sel] if sel else \
            [j for j in self.jobs if j.status != "ya procesado"]
        if not todo and only is not None:
            return
        if not todo:
            if messagebox.askyesno("Procesar", "Todo está procesado. ¿Volver a renderizar todo?"):
                todo = list(self.jobs)
            else:
                return
        run_cfg = dict(self.cfg)
        run_cfg["_cam_store"] = os.path.join(DATA_DIR, "camaras")
        run_cfg.update(cfg_override or {})
        self.runner = pl.Runner(run_cfg, [self.root_var.get()],
                                on_update=lambda j: self.q.put(("job", j)),
                                log=lambda m: self.q.put(("log", m)))
        self.btn_run.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.btn_soft.config(state="normal")
        if not cfg_override:
            self._write_pending([j.name for j in todo])
        self.q.put(("log", f"Procesando {len(todo)} trabajos…"))

        def work():
            self.runner.run_all(todo)
            self.q.put(("fin", [j.name for j in todo if j.status not in ("listo", "error")]))
        threading.Thread(target=work, daemon=True).start()

    def do_stop(self):
        if self.runner:
            if not messagebox.askyesno("Detener ya", "Se corta lo que está corriendo (un entrenamiento "
                                       "a medias se pierde).\n¿Detener ya?\n\n(Para no perder nada usa "
                                       "'Detener al terminar lo actual'.)"):
                return
            self.runner.stop()
            self.q.put(("log", "Deteniendo…"))

    def do_soft_stop(self):
        if self.runner:
            self.runner.stop_after_current()
            self.btn_soft.config(state="disabled")
            self.q.put(("log", "Se detendrá al terminar las etapas en marcha (no se pierde nada). "
                               "Lo pendiente queda guardado para continuar."))

    # ---------------------------------------------- continuar lo pendiente
    def _write_pending(self, names, auto=False):
        import json
        try:
            with open(PENDING_PATH, "w", encoding="utf-8") as f:
                json.dump({"raiz": self.root_var.get(), "trabajos": names, "auto": auto,
                           "fecha": time.strftime("%Y-%m-%d %H:%M:%S")}, f, ensure_ascii=False)
        except OSError:
            pass

    def _reset_job(self, root, name):
        """Rehace una carpeta desde las poses con la configuración actual (guarda la imagen
        anterior en resultados\\anteriores)."""
        wd = pl.work_dir(self.cfg, root, name)
        for st in ("sfm", "entrenar"):
            try:
                os.remove(os.path.join(wd, f".ok_{st}"))
            except OSError:
                pass
        res = self.cfg["carpeta_resultados"] or os.path.join(root, "resultados")
        for pre in ("", "x_"):
            png = os.path.join(res, pre + name + ".png")
            if os.path.isfile(png):
                try:
                    old_dir = os.path.join(res, "anteriores")
                    os.makedirs(old_dir, exist_ok=True)
                    os.replace(png, os.path.join(old_dir, pre + name + ".png"))
                except OSError:
                    pass
        self.q.put(("log", f"{name}: se rehará desde las poses con la configuración actual"))

    def _clear_pending(self):
        try:
            os.remove(PENDING_PATH)
        except OSError:
            pass

    def _resume_pending(self):
        import json
        try:
            with open(PENDING_PATH, encoding="utf-8") as f:
                p = json.load(f)
        except (OSError, ValueError):
            self._clear_pending()
            return
        root = p.get("raiz", "")
        if not os.path.isdir(root):
            self._clear_pending()
            return
        self.root_var.set(root)
        for n in p.get("reprocesar", []):
            self._reset_job(root, n)
        self.do_scan()
        names = set(p.get("trabajos", [])) | set(p.get("reprocesar", []))
        rest = [j.name for j in self.jobs if j.name in names and j.status != "ya procesado"]
        if not rest:
            self._clear_pending()
            return
        if p.get("auto") or messagebox.askyesno(
                "Continuar", f"Quedaron {len(rest)} carpeta(s) por procesar en:\n{root}\n\n"
                             "¿Continuar ahora? (se retoma desde la etapa donde quedó cada una)"):
            self.q.put(("log", f"Continuando {len(rest)} trabajos pendientes…"))
            self.do_run(only=rest)
        else:
            self._clear_pending()

    def _open_job_folder(self, _e):
        it = self.tv.focus()
        j = next((j for j in self.jobs if j.name == it), None)
        if j:
            open_path(j.folder)

    def _poll(self):
        try:
            while True:
                kind, data = self.q.get_nowait()
                if kind == "job":
                    tag = ("listo",) if data.status == "listo" else \
                        ("error",) if data.status == "error" else ()
                    if self.tv.exists(data.name):
                        self.tv.item(data.name, values=(media_txt(data), data.status,
                                                        data.detail), tags=tag)
                elif kind == "log":
                    self.logbox.insert("end", data + "\n")
                    self.logbox.see("end")
                elif kind == "fin":
                    self.btn_run.config(state="normal")
                    self.btn_stop.config(state="disabled")
                    self.btn_soft.config(state="disabled")
                    if data:
                        self._write_pending(data, auto=self._restart_when_idle)
                    else:
                        self._clear_pending()
                    if self._restart_when_idle:
                        self._restart()
                        return
                    self.review_dir = None
                    self.load_review()
                elif kind == "conv":
                    self.conv_log.insert("end", data + "\n")
                    self.conv_log.see("end")
                elif kind == "convfin":
                    self._conv_busy = False
                    if data:
                        self._conv_items = data
                        self.btn_conv_render.state(["!disabled"])
                    self.bell()
                elif kind == "link":
                    self._on_link_item(*data)
                elif kind == "linkfin":
                    self._linked_busy = False
                    self.review_dir = data
                    self.load_review()
                    self.link_lbl.config(text=f"Listo: {self._link_done} ortofoto(s) en {data}")
                    self.bell()
                    self.bell()
        except queue.Empty:
            pass
        self.after(150, self._poll)

    # -------------------------------------------------------------- Revisar
    def _build_review(self, nb):
        f = ttk.Frame(nb, padding=8)
        nb.add(f, text="2. Revisar")
        self.review_tab = f
        left = ttk.Frame(f)
        left.pack(side="left", fill="y")
        ttk.Button(left, text="Recargar", command=self.load_review).pack(fill="x")
        ttk.Button(left, text="Limpiar lista", command=self.clear_review).pack(fill="x", pady=(4, 0))
        self.lst = tk.Listbox(left, width=34, font=("Consolas", 10), activestyle="none")
        self.lst.pack(fill="y", expand=True, pady=4)
        self.lst.bind("<<ListboxSelect>>", lambda e: self._show_current())
        self.rev_count = ttk.Label(left, text="")
        self.rev_count.pack(fill="x")

        right = ttk.Frame(f)
        right.pack(side="left", fill="both", expand=True, padx=(8, 0))
        top = ttk.Frame(right)
        top.pack(fill="x", pady=(0, 4))
        ttk.Label(top, text="Revisando:").pack(side="left")
        self.rev_dir_lbl = ttk.Label(top, text="", foreground="#1f5fa8")
        self.rev_dir_lbl.pack(side="left", padx=4)
        ttk.Button(top, text="Cambiar carpeta…", command=self._pick_review_dir).pack(side="left")
        ttk.Button(top, text="Usar resultados de 1. Procesar",
                   command=self._review_default).pack(side="left", padx=4)
        ttk.Button(top, text="Abrir carpeta", command=lambda: open_path(self._results_dir())
                   if os.path.isdir(self._results_dir()) else None).pack(side="left")
        ttk.Label(top, text="   Calidad del render:").pack(side="left")
        self.calidad_var = tk.StringVar(value=self._calidad_label(self.cfg["render_lado_largo_px"]))
        cbq = ttk.Combobox(top, textvariable=self.calidad_var, state="readonly", width=24,
                           values=[c[0] for c in CALIDADES])
        cbq.pack(side="left", padx=4)
        cbq.bind("<<ComboboxSelected>>", self._set_calidad)
        bar = ttk.Frame(right)
        bar.pack(fill="x")
        ttk.Button(bar, text="← Anterior", command=lambda: self._step(-1)).pack(side="left")
        ttk.Button(bar, text="✓ Bien  [B]", command=lambda: self._mark(False)).pack(side="left", padx=4)
        ttk.Button(bar, text="✗ Mal  [M]", command=lambda: self._mark(True)).pack(side="left")
        ttk.Button(bar, text="Siguiente →", command=lambda: self._step(1)).pack(side="left", padx=4)
        ttk.Button(bar, text="Abrir en visor", command=self._open_current).pack(side="left")
        ttk.Button(bar, text="Reprocesar marcados ✗", command=self._reprocess_bad).pack(side="right")
        bar2 = ttk.Frame(right)
        bar2.pack(fill="x", pady=(4, 0))
        ttk.Label(bar2, text="SuperSplat:").pack(side="left")
        ttk.Button(bar2, text="1) Exportar .ply de las ✗",
                   command=self._export_bad).pack(side="left", padx=4)
        ttk.Button(bar2, text="2) Vincular carpeta de .ply limpios…",
                   command=self._link_folder).pack(side="left")
        ttk.Button(bar2, text="3) Renderizar .ply limpios",
                   command=self._render_linked).pack(side="left", padx=4)
        ttk.Button(bar2, text="↔ Voltear vista", command=lambda: self._adjust(flip=True)).pack(side="left")
        ttk.Button(bar2, text="⟳ Girar 90°", command=lambda: self._adjust(rot=1)).pack(side="left", padx=4)
        self.link_lbl = ttk.Label(right, text="", foreground="#555")
        self.link_lbl.pack(anchor="w", pady=(4, 0))
        self.rev_name = ttk.Label(right, text="", font=("Segoe UI", 12, "bold"))
        self.rev_name.pack(anchor="w", pady=4)
        self.canvas = tk.Canvas(right, bg="#2b2b2b", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self._show_current())
        for k, fn in (("b", lambda: self._mark(False)), ("m", lambda: self._mark(True)),
                      ("x", lambda: self._mark(True)), ("<Right>", lambda: self._step(1)),
                      ("<Left>", lambda: self._step(-1))):
            seq = k if k.startswith("<") else f"<KeyPress-{k}>"
            self.bind(seq, lambda e, fn=fn: fn() if self.nb.select() == str(self.review_tab) else None)
        self.files, self._img_cache = [], None
        self.review_dir = None          # None = resultados de "1. Procesar"
        self._linked_busy = False
        self._review_cleared = False
        self._thumbs = {}          # ruta -> (mtime, imagen reducida)
        self._thumb_lock = threading.Lock()

    def _on_tab(self, _e=None):
        if self.nb.select() == str(self.review_tab) and not self._review_cleared:
            self.load_review()

    def clear_review(self):
        """Vacía la lista de revisión (los archivos siguen en 'resultados')."""
        self.files = []
        self._review_cleared = True
        self.lst.delete(0, "end")
        self.canvas.delete("all")
        self.rev_name.config(text="")
        self.rev_count.config(text="Lista vacía (pulsa Recargar para verlas de nuevo)")

    def _process_results_dir(self):
        return self.cfg["carpeta_resultados"] or os.path.join(self.root_var.get(), "resultados")

    def _results_dir(self):
        """Carpeta que muestra 'Revisar' (independiente de '1. Procesar' si se eligió otra)."""
        return self.review_dir or self._process_results_dir()

    @staticmethod
    def _calidad_label(px):
        px = int(px)
        for lab, v in CALIDADES:
            if v == px:
                return lab
        return f"Personalizada ({px} px)"

    def _set_calidad(self, _e=None):
        px = dict(CALIDADES).get(self.calidad_var.get())
        if px:
            self.cfg["render_lado_largo_px"] = px
            pl.save_config(CONFIG_PATH, self.cfg)
            if "render_lado_largo_px" in self.vars:
                self.vars["render_lado_largo_px"][0].set(str(px))
            self.link_lbl.config(text=f"Calidad del render: {px} px en el lado largo "
                                      "(se aplica a los próximos renders).")

    def _pick_review_dir(self):
        d = filedialog.askdirectory(title="Carpeta con ortofotos para revisar")
        if d:
            self.review_dir = d
            self.load_review()

    def _review_default(self):
        self.review_dir = None
        self.load_review()

    def load_review(self):
        self._review_cleared = False
        d = self._results_dir()
        self.rev_dir_lbl.config(text=d)
        lk = self.cfg.get("carpeta_ply_limpios", "")
        if lk and not self._linked_busy:
            self.link_lbl.config(text=f"Carpeta de .ply limpios vinculada: {lk}")
        self.files = sorted(os.path.join(d, f) for f in os.listdir(d)
                            if f.lower().endswith((".png", ".jpg", ".tif"))) if os.path.isdir(d) else []
        self._refresh_list()
        if self.files:
            self.lst.selection_set(0)
            self._show_current()
            files = list(self.files)

            def warm():   # precarga todas en segundo plano
                for p in files:
                    try:
                        self._thumb(p)
                    except Exception:
                        pass
            threading.Thread(target=warm, daemon=True).start()

    def _refresh_list(self, keep=None):
        self.lst.delete(0, "end")
        bad = 0
        limpios = set()
        if self.files:
            _e, cdir = pl.side_dirs(self.cfg, self.root_var.get())
            if os.path.isdir(cdir):
                limpios = {os.path.splitext(f)[0].lower() for f in os.listdir(cdir)}
        for p in self.files:
            n = os.path.basename(p)
            is_bad = n.lower().startswith("x_")
            bad += is_bad
            base = os.path.splitext(n[2:] if is_bad else n)[0].lower()
            tag = "   ◆ limpio" if base in limpios else ""
            self.lst.insert("end", ("✗ " if is_bad else "  ") + n + tag)
            self.lst.itemconfig("end", fg="#c62828" if is_bad else "black")
        if keep is not None and self.files:
            self.lst.selection_set(keep)
            self.lst.see(keep)
        self.rev_count.config(text=f"{len(self.files)} imágenes · {bad} marcadas mal")

    def _cur(self):
        s = self.lst.curselection()
        return s[0] if s else None

    def _show_current(self):
        i = self._cur()
        self.canvas.delete("all")
        if i is None or i >= len(self.files):
            return
        p = self.files[i]
        self.rev_name.config(text=os.path.basename(p))
        try:
            im = self._thumb(p).copy()
        except Exception as e:
            self.canvas.create_text(20, 20, anchor="nw", fill="white", text=str(e))
            return
        cw, ch = max(50, self.canvas.winfo_width()), max(50, self.canvas.winfo_height())
        im.thumbnail((cw, ch), Image.LANCZOS)
        self._img_cache = ImageTk.PhotoImage(im)
        self.canvas.create_image(cw // 2, ch // 2, image=self._img_cache)
        self._prefetch(i)

    def _thumb(self, p):
        """Versión reducida (1600 px) en memoria; las PNG de 7680 px tardan en abrir."""
        key = os.path.basename(p)[2:] if os.path.basename(p).lower().startswith("x_") \
            else os.path.basename(p)
        mt = os.path.getmtime(p)
        with self._thumb_lock:
            hit = self._thumbs.get(key)
        if hit and hit[0] == mt:
            return hit[1]
        im = Image.open(p)
        im.draft("RGB", (1600, 1600))
        im = im.convert("RGB")
        im.thumbnail((1600, 1600), Image.BILINEAR)
        with self._thumb_lock:
            self._thumbs[key] = (mt, im)
        return im

    def _prefetch(self, i):
        nxt = [self.files[j] for j in (i + 1, i + 2, i - 1) if 0 <= j < len(self.files)]

        def work():
            for p in nxt:
                try:
                    self._thumb(p)
                except Exception:
                    pass
        threading.Thread(target=work, daemon=True).start()

    def _step(self, d):
        i = self._cur()
        if i is None or not self.files:
            return
        j = min(max(0, i + d), len(self.files) - 1)
        self.lst.selection_clear(0, "end")
        self.lst.selection_set(j)
        self.lst.see(j)
        self._show_current()

    def _mark(self, bad):
        i = self._cur()
        if i is None:
            return
        try:
            self.files[i] = pl.set_mark(self.files[i], bad)
        except OSError as e:
            messagebox.showerror("Renombrar", str(e))
            return
        self._refresh_list(keep=i)
        self._step(1)

    def _open_current(self):
        i = self._cur()
        if i is not None:
            open_path(self.files[i])

    # ------------------------------ Revisar independiente: carpeta de .ply limpios
    def _cam_store(self):
        return os.path.join(DATA_DIR, "camaras")

    def _link_folder(self):
        d = filedialog.askdirectory(title="Carpeta con los .ply limpios (exportados de SuperSplat)",
                                    initialdir=self.cfg.get("carpeta_ply_limpios") or None)
        if not d:
            return
        items = pl.list_plys(d)
        if not items:
            messagebox.showinfo("Vincular", f"No encontré archivos .ply en:\n{d}")
            return
        self.cfg["carpeta_ply_limpios"] = d
        pl.save_config(CONFIG_PATH, self.cfg)
        lista = "\n".join(f"  {n}  ←  {os.path.basename(p_)}" for p_, n in items[:25])
        mas = f"\n  … y {len(items) - 25} más" if len(items) > 25 else ""
        self.link_lbl.config(text=f"Carpeta de .ply limpios vinculada: {d}  ({len(items)} .ply)")
        if messagebox.askyesno("Vincular carpeta",
                               f"Encontré {len(items)} .ply:\n{lista}{mas}\n\n"
                               f"Las ortofotos se guardarán en:\n{os.path.join(d, 'ortofotos_limpias')}"
                               "\n\n¿Renderizar ahora? (no usa Spirula)"):
            self._start_linked(items, d)

    def _render_linked(self):
        d = self.cfg.get("carpeta_ply_limpios", "")
        if not d or not os.path.isdir(d):
            messagebox.showinfo("Renderizar", "Primero usa '2) Vincular carpeta de .ply limpios…'.")
            return
        items = pl.list_plys(d)
        if not items:
            messagebox.showinfo("Renderizar", f"No hay .ply en:\n{d}")
            return
        out = os.path.join(d, "ortofotos_limpias")
        bad = set(self._bad_names()) if os.path.abspath(self._results_dir()) == os.path.abspath(out) else set()
        sel = [it for it in items if it[1] in bad]
        if sel and messagebox.askyesno("Renderizar",
                                       f"Hay {len(sel)} marcadas con ✗ en esta carpeta.\n"
                                       "¿Renderizar solo esas? (No = todas)"):
            items = sel
        self._start_linked(items, d)

    def _start_linked(self, items, folder):
        if self._linked_busy:
            messagebox.showinfo("Renderizar", "Ya hay un render en marcha.")
            return
        self._linked_busy = True
        out = os.path.join(folder, "ortofotos_limpias")
        cfg = dict(self.cfg)
        self._link_total, self._link_done = len(items), 0
        self.link_lbl.config(text=f"Renderizando 0/{len(items)}…")
        self.q.put(("log", f"Revisar: renderizando {len(items)} .ply limpios de {folder}"))

        def work():
            pl.render_linked(items, out, cfg, root=self.root_var.get() or None,
                             store=self._cam_store(),
                             on_item=lambda n, st, det: self.q.put(("link", (n, st, det))))
            self.q.put(("linkfin", out))
        threading.Thread(target=work, daemon=True).start()

    def _on_link_item(self, name, st, det):
        if st in ("listo", "error"):
            self._link_done += 1
        self.link_lbl.config(text=f"Renderizando {self._link_done}/{self._link_total} · {name}: {st} {det}"
                             if self._link_done < self._link_total
                             else f"Listo: {self._link_total} ortofoto(s) · última: {name} {det}")
        self.q.put(("log", f"{time.strftime('%H:%M:%S')} {name}: {st} {det}"))

    def _adjust(self, flip=False, rot=0):
        """Voltea o gira 90° la ortofoto seleccionada y la vuelve a renderizar."""
        i = self._cur()
        if i is None:
            return
        d = self._results_dir()
        fname = os.path.basename(self.files[i])
        name = os.path.splitext(fname[2:] if fname.lower().startswith("x_") else fname)[0]
        man = pl.read_manifest(d)
        entry = man.get(name)
        ply = entry.get("ply") if entry else None
        if not ply:   # ortofoto de '1. Procesar': su .ply está en _orto_trabajo
            src = pl.newest_ply(os.path.join(pl.work_dir(self.cfg, self.root_var.get(), name), "runs"))
            limpio = os.path.join(pl.side_dirs(self.cfg, self.root_var.get())[1], name + ".ply")
            ply = limpio if os.path.isfile(limpio) else src
            entry = {"ply": ply, "flip": False, "rot": 0}
        if not ply or not os.path.isfile(ply):
            messagebox.showinfo("Ajustar", "No encuentro el .ply de esta imagen.")
            return
        entry["flip"] = (not entry.get("flip")) if flip else bool(entry.get("flip"))
        entry["rot"] = (int(entry.get("rot", 0)) + rot) % 4
        man[name] = entry
        pl.write_manifest(d, man)
        if self._linked_busy:
            messagebox.showinfo("Ajustar", "Espera a que termine el render en marcha.")
            return
        self._linked_busy = True
        self.link_lbl.config(text=f"Re-renderizando {name}…")
        cfg = dict(self.cfg)
        root = self.root_var.get() or None

        def work():
            pl.render_linked([(ply, name)], d, cfg, root=root, store=self._cam_store(),
                             on_item=lambda n, st, det: self.q.put(("link", (n, st, det))))
            self.q.put(("linkfin", d))
        self._link_total, self._link_done = 1, 0
        threading.Thread(target=work, daemon=True).start()

    # ---------------------------------------------- SuperSplat (.ply limpios)
    def _job_names(self):
        if not self.jobs:
            self.do_scan()
        return [j.name for j in self.jobs]

    def _bad_names(self):
        return [os.path.basename(p)[2:-4] for p in self.files
                if os.path.basename(p).lower().startswith("x_")]

    def _export_bad(self):
        bad = self._bad_names()
        if not bad:
            messagebox.showinfo("Exportar .ply", "No hay imágenes marcadas con ✗ en la lista.")
            return
        names = [n for n in self._job_names() if n in bad]
        res = pl.export_plys(self.cfg, self.root_var.get(), names)
        ok = [d for _n, d, e in res if d]
        err = [f"{n}: {e}" for n, d, e in res if not d]
        exp, _c = pl.side_dirs(self.cfg, self.root_var.get())
        messagebox.showinfo(
            "Exportar .ply",
            f"{len(ok)} archivo(s) en:\n{exp}\n\n"
            "Ábrelos en SuperSplat, borra lo que sobra (sin mover ni girar el modelo) y "
            "expórtalos como .ply. Guárdalos en la carpeta 'ply_limpios' con el mismo "
            "nombre, o usa '2) Vincular .ply limpios…'." + ("\n\nSin .ply: " + ", ".join(err)
                                                          if err else ""))
        if ok:
            open_path(exp)

    def _link_clean(self):
        files = filedialog.askopenfilenames(title="Elige los .ply limpios (exportados de SuperSplat)",
                                            filetypes=[("PLY", "*.ply")])
        if not files:
            return
        names = self._job_names()
        linked = []
        for f in files:
            n = pl.match_job_name(f, names) or self._ask_job(f, names)
            if n:
                pl.link_clean_ply(self.cfg, self.root_var.get(), f, n)
                linked.append(n)
        if not linked:
            return
        self._refresh_list(keep=self._cur())
        if messagebox.askyesno("Vincular .ply limpios",
                               f"Vinculados: {', '.join(linked)}\n\n¿Renderizar ahora sus ortofotos "
                               "con los .ply limpios? (no usa Spirula, segundos por imagen)"):
            self.nb.select(0)
            self.do_run(only=linked, cfg_override={"_solo_render": True})

    def _render_clean(self):
        names = pl.clean_names(self.cfg, self.root_var.get(), self._job_names())
        if not names:
            _e, c = pl.side_dirs(self.cfg, self.root_var.get())
            messagebox.showinfo("Renderizar limpios", f"No hay .ply limpios en:\n{c}")
            return
        bad = set(self._bad_names())
        sel = [n for n in names if n in bad] or names
        if not messagebox.askyesno("Renderizar limpios",
                                   f"Se renderizarán {len(sel)} ortofoto(s) desde sus .ply limpios:\n"
                                   + ", ".join(sel)):
            return
        self.nb.select(0)
        self.do_run(only=sel, cfg_override={"_solo_render": True})

    def _open_ply_dirs(self):
        exp, clean = pl.side_dirs(self.cfg, self.root_var.get())
        for d in (exp, clean):
            os.makedirs(d, exist_ok=True)
        open_path(os.path.dirname(clean))

    def _ask_job(self, filename, names):
        """Si el nombre del archivo no coincide, pregunta a qué puerta/ventana pertenece."""
        win = tk.Toplevel(self)
        win.title("¿De qué puerta/ventana es?")
        win.transient(self)
        win.grab_set()
        ttk.Label(win, text=f"No reconozco el nombre:\n{os.path.basename(filename)}\n\n"
                            "Elige a cuál corresponde:").pack(padx=10, pady=6)
        lb = tk.Listbox(win, height=min(15, len(names)), width=40)
        for n in names:
            lb.insert("end", n)
        lb.pack(padx=10)
        res = {"v": None}

        def ok():
            s_ = lb.curselection()
            res["v"] = names[s_[0]] if s_ else None
            win.destroy()
        fr = ttk.Frame(win)
        fr.pack(pady=8)
        ttk.Button(fr, text="Aceptar", command=ok).pack(side="left", padx=4)
        ttk.Button(fr, text="Omitir", command=win.destroy).pack(side="left")
        lb.bind("<Double-1>", lambda e: ok())
        self.wait_window(win)
        return res["v"]

    def _reprocess_bad(self):
        bad = {os.path.basename(p)[2:-4] for p in self.files
               if os.path.basename(p).lower().startswith("x_")}
        if not bad:
            messagebox.showinfo("Reprocesar", "No hay imágenes marcadas con ✗.")
            return
        if not self.jobs:
            self.do_scan()
        names = [j.name for j in self.jobs if j.name in bad]
        if not names:
            messagebox.showinfo("Reprocesar", "No encontré esas carpetas en la raíz escaneada.")
            return
        pasos = int(self.cfg["pasos_entrenamiento"]) * 2
        r = messagebox.askyesnocancel(
            "Reprocesar marcados",
            f"{len(names)} imágenes marcadas como mal.\n\n"
            f"Sí  = reentrenar con más calidad ({pasos} pasos) y renderizar de nuevo\n"
            f"No  = solo renderizar de nuevo (segundos)\n")
        if r is None:
            return
        self.nb.select(0)
        override = {"_forzar_entrenar": True, "pasos_entrenamiento": pasos} if r else {}
        self.do_run(only=names, cfg_override=override)

    # ------------------------------------------- Convertir nube (CloudCompare / ReCap)
    def _build_convert(self, nb):
        f = ttk.Frame(nb, padding=8)
        nb.add(f, text="3. Convertir nube")
        self._conv_src, self._conv_clouds, self._conv_items = [], [], []
        self._conv_busy = False

        a = ttk.LabelFrame(f, text="  A) Splat .ply (Spirula / SuperSplat)  →  nube con color para "
                                   "CloudCompare / ReCap  ", padding=8)
        a.pack(fill="x")
        r1 = ttk.Frame(a)
        r1.pack(fill="x")
        ttk.Button(r1, text="Elegir .ply…", command=self._conv_pick_splats).pack(side="left")
        ttk.Button(r1, text="Elegir carpeta…", command=lambda: self._conv_pick_splats(folder=True)
                   ).pack(side="left", padx=4)
        ttk.Button(r1, text="Usar los .ply de las ✗ (ply_para_supersplat)",
                   command=self._conv_use_exported).pack(side="left")
        self.conv_src_lbl = ttk.Label(r1, text="  Ningún .ply elegido", foreground="#555")
        self.conv_src_lbl.pack(side="left", padx=8)
        r2 = ttk.Frame(a)
        r2.pack(fill="x", pady=(6, 0))
        ttk.Label(r2, text="Formatos:").pack(side="left")
        self.conv_fmt = {k: tk.BooleanVar(value=self.cfg.get("conv_fmt_" + k, d))
                         for k, d in (("ply", True), ("las", True), ("pts", False))}
        for k, lab in (("ply", ".ply (CloudCompare)"), ("las", ".las (CloudCompare y ReCap)"),
                       ("pts", ".pts (ReCap)")):
            ttk.Checkbutton(r2, text=lab, variable=self.conv_fmt[k]).pack(side="left", padx=6)
        r3 = ttk.Frame(a)
        r3.pack(fill="x", pady=(6, 0))
        self.conv_hide = tk.BooleanVar(value=self.cfg.get("conv_ocultar", True))
        ttk.Checkbutton(r3, text="No exportar splats casi transparentes, opacidad menor a",
                        variable=self.conv_hide).pack(side="left")
        self.conv_thr = tk.StringVar(value=str(self.cfg.get("conv_opacidad_min", 0.05)))
        ttk.Spinbox(r3, from_=0.0, to=0.9, increment=0.01, width=6,
                    textvariable=self.conv_thr).pack(side="left", padx=4)
        self.conv_up = tk.BooleanVar(value=self.cfg.get("conv_de_pie", True))
        ttk.Checkbutton(r3, text="Poner de pie (Z arriba)", variable=self.conv_up).pack(side="left", padx=12)
        r4 = ttk.Frame(a)
        r4.pack(fill="x", pady=(6, 0))
        ttk.Label(r4, text="Guardar en (vacío = <carpeta del .ply>\\nube_para_editar):").pack(side="left")
        self.conv_out_a = tk.StringVar(value="")
        ttk.Entry(r4, textvariable=self.conv_out_a).pack(side="left", fill="x", expand=True, padx=4)
        ttk.Button(r4, text="…", width=3, command=lambda: self.conv_out_a.set(
            filedialog.askdirectory() or self.conv_out_a.get())).pack(side="left")
        ttk.Button(r4, text="▶ Convertir a nube", command=self._conv_to_cloud).pack(side="left", padx=6)

        b = ttk.LabelFrame(f, text="  B) Nube editada en CloudCompare / ReCap  →  splat .ply para "
                                   "Spirula y Ortofotos (con todos sus datos)  ", padding=8)
        b.pack(fill="x", pady=(10, 0))
        s1 = ttk.Frame(b)
        s1.pack(fill="x")
        ttk.Button(s1, text="Elegir nubes editadas…", command=self._conv_pick_clouds).pack(side="left")
        ttk.Button(s1, text="Elegir carpeta…", command=lambda: self._conv_pick_clouds(folder=True)
                   ).pack(side="left", padx=4)
        self.conv_cl_lbl = ttk.Label(s1, text="  Ninguna nube elegida (.ply, .las, .pts, .xyz, .txt, .e57)",
                                     foreground="#555")
        self.conv_cl_lbl.pack(side="left", padx=8)
        s2 = ttk.Frame(b)
        s2.pack(fill="x", pady=(6, 0))
        ttk.Label(s2, text="Guardar en (vacío = <carpeta de la nube>\\ply_desde_nube):").pack(side="left")
        self.conv_out_b = tk.StringVar(value="")
        ttk.Entry(s2, textvariable=self.conv_out_b).pack(side="left", fill="x", expand=True, padx=4)
        ttk.Button(s2, text="…", width=3, command=lambda: self.conv_out_b.set(
            filedialog.askdirectory() or self.conv_out_b.get())).pack(side="left")
        s3 = ttk.Frame(b)
        s3.pack(fill="x", pady=(6, 0))
        self.conv_to_clean = tk.BooleanVar(value=self.cfg.get("conv_a_limpios", False))
        ttk.Checkbutton(s3, text="Copiar también a ply_limpios (los usa 1. Procesar al renderizar)",
                        variable=self.conv_to_clean).pack(side="left")
        ttk.Button(s3, text="▶ Convertir a splat .ply", command=self._conv_to_splat).pack(side="left", padx=10)
        self.btn_conv_render = ttk.Button(s3, text="Renderizar ortofotos de estos",
                                          command=self._conv_render, state="disabled")
        self.btn_conv_render.pack(side="left")
        ttk.Button(s3, text="Abrir carpeta de salida", command=self._conv_open_out).pack(side="left", padx=6)

        ttk.Label(f, foreground="#555", wraplength=1100, justify="left", text=(
            "Cómo editar: en CloudCompare abre el .ply o .las, borra/recorta con las tijeras y guarda "
            "como .ply o .las (no muevas, gires ni escales la nube). En ReCap importa el .las o .pts, "
            "borra lo que sobra y exporta como .pts. Al volver (B) cada punto que quedó recupera su "
            "splat original con escala, rotación, opacidad y color completos, así que el render sale "
            "igual que en SuperSplat. El archivo .splatref.json junto a la nube indica de qué splat "
            "viene: no lo borres.")).pack(anchor="w", pady=(8, 4))
        self.conv_log = tk.Text(f, height=12, font=("Consolas", 10))
        self.conv_log.pack(fill="both", expand=True)

    def _conv_say(self, msg):
        self.q.put(("conv", f"{time.strftime('%H:%M:%S')} {msg}"))

    def _conv_pick_splats(self, folder=False):
        if folder:
            d = filedialog.askdirectory(title="Carpeta con splats .ply")
            files = [p for p, _n in pl.list_plys(d, recursive=False)] if d else []
        else:
            files = list(filedialog.askopenfilenames(title="Splats .ply (Spirula / SuperSplat)",
                                                     filetypes=[("PLY", "*.ply")]))
        if files:
            self._conv_src = files
            self.conv_src_lbl.config(text=f"  {len(files)} .ply: " + ", ".join(
                os.path.basename(x) for x in files[:4]) + (" …" if len(files) > 4 else ""))

    def _conv_use_exported(self):
        exp, _c = pl.side_dirs(self.cfg, self.root_var.get())
        files = [p for p, _n in pl.list_plys(exp, recursive=False)] if os.path.isdir(exp) else []
        if not files:
            messagebox.showinfo("Convertir", f"No hay .ply en:\n{exp}\n\nUsa primero "
                                             "'1) Exportar .ply de las ✗' en 2. Revisar.")
            return
        self._conv_src = files
        self.conv_src_lbl.config(text=f"  {len(files)} .ply de ply_para_supersplat")

    def _conv_pick_clouds(self, folder=False):
        import ply_convert as pc
        if folder:
            d = filedialog.askdirectory(title="Carpeta con las nubes editadas")
            files = sorted(os.path.join(d, x) for x in os.listdir(d)
                           if x.lower().endswith(pc.CLOUD_EXT)) if d else []
        else:
            files = list(filedialog.askopenfilenames(
                title="Nubes editadas (CloudCompare / ReCap)",
                filetypes=[("Nubes de puntos", "*.ply *.las *.pts *.xyz *.txt *.asc *.csv *.e57"),
                           ("Todos", "*.*")]))
        if files:
            self._conv_clouds = files
            self.conv_cl_lbl.config(text=f"  {len(files)} nube(s): " + ", ".join(
                os.path.basename(x) for x in files[:4]) + (" …" if len(files) > 4 else ""))

    def _conv_save_opts(self):
        for k, v in self.conv_fmt.items():
            self.cfg["conv_fmt_" + k] = bool(v.get())
        self.cfg["conv_ocultar"] = bool(self.conv_hide.get())
        self.cfg["conv_de_pie"] = bool(self.conv_up.get())
        self.cfg["conv_a_limpios"] = bool(self.conv_to_clean.get())
        try:
            self.cfg["conv_opacidad_min"] = float(self.conv_thr.get())
        except ValueError:
            pass
        pl.save_config(CONFIG_PATH, self.cfg)

    def _conv_to_cloud(self):
        import ply_convert as pc
        if self._conv_busy:
            messagebox.showinfo("Convertir", "Ya hay una conversión en marcha.")
            return
        if not self._conv_src:
            messagebox.showinfo("Convertir", "Primero elige los splats .ply.")
            return
        fmts = tuple(k for k, v in self.conv_fmt.items() if v.get())
        if not fmts:
            messagebox.showinfo("Convertir", "Marca al menos un formato.")
            return
        self._conv_save_opts()
        thr = float(self.cfg.get("conv_opacidad_min", 0.05)) if self.conv_hide.get() else 0.0
        upright, out_user = bool(self.conv_up.get()), self.conv_out_a.get().strip()
        files, cfg, root = list(self._conv_src), dict(self.cfg), self.root_var.get() or None
        self._conv_busy = True
        self._conv_say(f"Splat → nube: {len(files)} archivo(s) ({', '.join(fmts)})…")

        def work():
            outs = set()
            for src in files:
                try:
                    out = out_user or os.path.join(os.path.dirname(src), "nube_para_editar")
                    ups = None
                    if upright:
                        try:
                            _c, ups, _o = pl.find_cams(pl._norm_name(src), src, cfg, root, self._cam_store())
                        except Exception:
                            ups = None
                    pc.splat_to_clouds(src, out, fmts, thr, upright, ups, log=self._conv_say)
                    outs.add(out)
                except Exception as e:
                    self._conv_say(f"ERROR {os.path.basename(src)}: {e}")
            for o in outs:
                self._conv_say(f"Listo → {o}")
            self.q.put(("convfin", None))
        threading.Thread(target=work, daemon=True).start()

    def _conv_to_splat(self):
        import json
        import ply_convert as pc
        if self._conv_busy:
            messagebox.showinfo("Convertir", "Ya hay una conversión en marcha.")
            return
        if not self._conv_clouds:
            messagebox.showinfo("Convertir", "Primero elige las nubes editadas.")
            return
        self._conv_save_opts()
        tasks = []          # (nube, splatref, original, nombre)
        for c in self._conv_clouds:
            ref_p = pc.find_ref(c)
            ref = {}
            if ref_p:
                try:
                    with open(ref_p, encoding="utf-8") as fh:
                        ref = json.load(fh)
                except (OSError, ValueError):
                    ref_p = None
            orig = pc.resolve_original(ref_p, ref) if ref_p else None
            if not orig:
                orig = filedialog.askopenfilename(
                    title=f"Splat .ply original de «{os.path.basename(c)}»",
                    filetypes=[("PLY", "*.ply")])
                if not orig:
                    self._conv_say(f"Omitido {os.path.basename(c)}: sin splat original.")
                    continue
            name = ref.get("nombre") or pl._norm_name(c) or "modelo"
            tasks.append((c, ref_p, orig, name))
        if not tasks:
            return
        latest = {}         # varias nubes con el mismo nombre: la modificada más recientemente
        for t in tasks:
            if t[3] not in latest or os.path.getmtime(t[0]) > os.path.getmtime(latest[t[3]][0]):
                latest[t[3]] = t
        if len(latest) < len(tasks):
            self._conv_say("Varias nubes del mismo modelo: se usa la guardada más recientemente.")
        tasks = list(latest.values())
        out_user, to_clean = self.conv_out_b.get().strip(), bool(self.conv_to_clean.get())
        cfg, root = dict(self.cfg), self.root_var.get()
        self._conv_busy = True
        self.btn_conv_render.state(["disabled"])
        self._conv_say(f"Nube → splat: {len(tasks)} archivo(s)…")

        def work():
            import shutil
            items = []
            for c, ref_p, orig, name in tasks:
                try:
                    out_dir = out_user or os.path.join(os.path.dirname(c), "ply_desde_nube")
                    out = os.path.join(out_dir, name + ".ply")
                    if os.path.abspath(out) in (os.path.abspath(c), os.path.abspath(orig)):
                        out = os.path.join(out_dir, name + "_splat.ply")
                    pc.cloud_to_splat(c, out, original=orig, ref_path=ref_p, log=self._conv_say)
                    stem = os.path.splitext(os.path.basename(orig))[0]
                    for cam in (os.path.join(os.path.dirname(orig), stem + ".cam.json"),
                                os.path.join(os.path.dirname(orig), name + ".cam.json")):
                        if os.path.isfile(cam):
                            shutil.copy2(cam, os.path.join(out_dir, name + ".cam.json"))
                            break
                    if to_clean and root:
                        dst = pl.link_clean_ply(cfg, root, out, name)
                        self._conv_say(f"  copiado a {dst}")
                    items.append((out, name))
                except Exception as e:
                    self._conv_say(f"ERROR {os.path.basename(c)}: {e}")
            if items:
                self._conv_say(f"Listo: {len(items)} splat(s) en {os.path.dirname(items[0][0])}. "
                               "Puedes abrirlos en Spirula/SuperSplat o pulsar 'Renderizar ortofotos'.")
            self.q.put(("convfin", items))
        threading.Thread(target=work, daemon=True).start()

    def _conv_render(self):
        if not self._conv_items:
            return
        folder = os.path.dirname(self._conv_items[0][0])
        self.nb.select(self.review_tab)
        self._start_linked(list(self._conv_items), folder)

    def _conv_open_out(self):
        if self._conv_items:
            open_path(os.path.dirname(self._conv_items[0][0]))
        elif self._conv_clouds:
            d = self.conv_out_b.get().strip() or os.path.join(os.path.dirname(self._conv_clouds[0]),
                                                              "ply_desde_nube")
            open_path(d if os.path.isdir(d) else os.path.dirname(self._conv_clouds[0]))
        elif self._conv_src:
            d = self.conv_out_a.get().strip() or os.path.join(os.path.dirname(self._conv_src[0]),
                                                              "nube_para_editar")
            open_path(d if os.path.isdir(d) else os.path.dirname(self._conv_src[0]))

    # ------------------------------------------------------ actualizaciones
    def _periodic_update(self):
        if self.cfg.get("actualizar_auto", True):
            self._check_update(silent=True)
        self.after(2 * 3600 * 1000, self._periodic_update)

    def _check_update(self, silent=False):
        try:
            import updater
        except Exception as e:
            if not silent:
                messagebox.showerror("Actualizaciones", f"No disponible: {e}")
            return
        if not silent:
            self.upd_lbl.config(text="  Buscando…")
        cur = code_version()

        def work():
            rv, msg = updater.check(DATA_DIR, cur, self.cfg.get("url_actualizaciones") or None)
            self.after(0, lambda: self._update_done(rv, msg, silent))
        threading.Thread(target=work, daemon=True).start()

    def _update_done(self, rv, msg, silent):
        self.upd_lbl.config(text=f"  Versión {code_version()} · {msg}")
        self.q.put(("log", f"Actualizaciones: {msg}"))
        if not rv:
            if not silent:
                messagebox.showinfo("Actualizaciones", msg)
            return
        busy = self.runner is not None and self.btn_stop["state"] == "normal"
        if busy:
            r = messagebox.askyesno("Actualización lista",
                                    f"Versión {rv} descargada.\n\nHay un proceso en marcha. ¿Reiniciar "
                                    "automáticamente al terminar las etapas en curso? (no se pierde nada "
                                    "y luego continúa con lo pendiente)")
            if r:
                self._restart_when_idle = True
                self.runner.stop_after_current()
                self.btn_soft.config(state="disabled")
                self.q.put(("log", "Se reiniciará con la versión nueva al terminar lo actual."))
        elif messagebox.askyesno("Actualización lista", f"Versión {rv} descargada. ¿Reiniciar ahora?"):
            self._restart()

    def _restart(self):
        launcher = os.environ.get("ORTO_LAUNCHER") or os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "launcher.py")
        if FROZEN:
            cmd = [sys.executable]
        else:
            py = sys.executable
            if py.lower().endswith("python.exe") and os.path.isfile(py[:-10] + "pythonw.exe"):
                py = py[:-10] + "pythonw.exe"
            cmd = [py, launcher]
        try:
            subprocess.Popen(cmd, cwd=APP_DIR, close_fds=True)
        except OSError as e:
            messagebox.showerror("Reiniciar", f"No pude reiniciar: {e}\nÁbrelo de nuevo a mano.")
            return
        self.destroy()

    # --------------------------------------------------------- Configuración
    def _build_config(self, nb):
        outer = ttk.Frame(nb, padding=8)
        nb.add(outer, text="Configuración")
        self.vars = {}
        ttk.Label(outer, text="Perfil de calidad", font=("Segoe UI", 10, "bold")).grid(
            row=0, column=0, sticky="w", pady=(0, 8))
        self.perfil = tk.StringVar(value="")
        cb = ttk.Combobox(outer, textvariable=self.perfil, values=list(PERFILES), state="readonly",
                          width=40)
        cb.grid(row=0, column=1, sticky="w", pady=(0, 8))
        cb.bind("<<ComboboxSelected>>", self._apply_profile)
        for r0, (k, label, typ) in enumerate(FIELDS):
            r = r0 + 1
            ttk.Label(outer, text=label).grid(row=r, column=0, sticky="w", pady=2)
            if k in CHOICES:
                v = tk.StringVar(value=str(self.cfg[k]))
                ttk.Combobox(outer, textvariable=v, values=CHOICES[k], width=20).grid(
                    row=r, column=1, sticky="w")
            elif typ is bool:
                v = tk.BooleanVar(value=bool(self.cfg[k]))
                ttk.Checkbutton(outer, variable=v).grid(row=r, column=1, sticky="w")
            else:
                v = tk.StringVar(value=str(self.cfg[k]))
                ttk.Entry(outer, textvariable=v, width=70).grid(row=r, column=1, sticky="we")
                if typ in ("file", "dir"):
                    ttk.Button(outer, text="…", width=3,
                               command=lambda v=v, t=typ: v.set(
                                   (filedialog.askopenfilename(filetypes=[("spirula", "*.exe"), ("todo", "*")])
                                    if t == "file" else filedialog.askdirectory()) or v.get())
                               ).grid(row=r, column=2)
            self.vars[k] = (v, typ)
        outer.columnconfigure(1, weight=1)
        bar = ttk.Frame(outer)
        bar.grid(row=len(FIELDS) + 1, column=1, sticky="w", pady=10)
        ttk.Button(bar, text="Guardar configuración", command=self._save_cfg).pack(side="left")
        ttk.Button(bar, text="Restaurar valores por defecto",
                   command=self._reset_cfg).pack(side="left", padx=8)
        ttk.Button(bar, text="Buscar actualizaciones", command=self._check_update).pack(side="left")
        self.upd_lbl = ttk.Label(bar, text=f"  Versión {code_version()}", foreground="#1f5fa8")
        self.upd_lbl.pack(side="left", padx=8)
        ttk.Label(outer, text="Los cambios se aplican a lo que proceses después de guardar. "
                              "Para que un modelo ya entrenado use más pasos, márcalo con ✗ "
                              "en Revisar y usa 'Reprocesar marcados'.",
                  foreground="#555", wraplength=900).grid(row=len(FIELDS) + 2, column=0, columnspan=3, sticky="w")

    def _apply_profile(self, _e=None):
        for k, val in PERFILES[self.perfil.get()].items():
            v, _t = self.vars[k]
            v.set(str(val))

    def _reset_cfg(self):
        keep = {k: self.cfg.get(k) for k in ("spirula_exe", "ultima_raiz")}
        for k, (v, typ) in self.vars.items():
            val = keep.get(k) if k in keep and keep.get(k) else pl.DEFAULT_CONFIG[k]
            v.set(val if typ is bool else str(val))

    def _save_cfg(self):
        try:
            for k, (v, typ) in self.vars.items():
                val = v.get()
                self.cfg[k] = val if typ in (str, "file", "dir", bool) else typ(val)
        except ValueError as e:
            messagebox.showerror("Configuración", f"Valor inválido: {e}")
            return
        pl.save_config(CONFIG_PATH, self.cfg)
        running = self.runner and self.btn_stop["state"] == "normal"
        messagebox.showinfo("Configuración", "Guardado." + (
            "\nEl proceso en marcha sigue con los valores anteriores; "
            "lo próximo que proceses usará los nuevos." if running else ""))

    def _on_close(self):
        if self.runner and self.btn_stop["state"] == "normal":
            if not messagebox.askyesno("Salir", "Hay un proceso en marcha. ¿Detener y salir?\n"
                                       "(Lo pendiente se ofrecerá continuar al abrir de nuevo; un "
                                       "entrenamiento a medias se repite.)"):
                return
            self.runner.stop()
        self.destroy()


def media_txt(j):
    parts = []
    if j.videos:
        parts.append(f"{len(j.videos)} vid")
    if getattr(j, "images", None):
        parts.append(f"{len(j.images)} fotos")
    return " · ".join(parts) or "0"


def open_path(p):
    if sys.platform.startswith("win"):
        os.startfile(p)  # noqa
    elif sys.platform == "darwin":
        subprocess.Popen(["open", p])
    else:
        subprocess.Popen(["xdg-open", p])


def autotest(out_dir):
    """Abre la ventana, recorre las pestañas y guarda capturas (para verificar)."""
    from PIL import ImageGrab
    a = App(None)
    shots = []

    def grab(name):
        a.update()
        x, y = a.winfo_rootx(), a.winfo_rooty()
        w, h = a.winfo_width(), a.winfo_height()
        ImageGrab.grab(bbox=(x, y, x + w, y + h)).save(os.path.join(out_dir, name))
        shots.append(name)

    def s1():
        a.do_scan(); a.after(800, lambda: (grab("t1_procesar.png"), a.nb.select(1), a.load_review(),
                                           a.after(1200, s2)))

    def s2():
        grab("t2_revisar.png"); a.nb.select(2); a.after(800, s25)

    def s25():
        grab("t3_convertir.png"); a.nb.select(3); a.after(800, s3)

    def s3():
        grab("t4_config.png"); a.destroy()
    a.attributes("-topmost", True)
    a.after(1500, s1)
    a.mainloop()


def diag(root):
    """Diagnóstico sin ventana: escribe diag.txt junto al programa."""
    import traceback
    out = os.path.join(DATA_DIR, "diag.txt")
    with open(out, "w", encoding="utf-8") as f:
        try:
            f.write(f"frozen={FROZEN} app_dir={APP_DIR} data_dir={DATA_DIR}\n")
            f.write(f"root={root!r} isdir={os.path.isdir(root)}\n")
            cfg = pl.load_config(CONFIG_PATH)
            f.write(f"spirula={find_spirula(cfg.get('spirula_exe',''))}\n")
            rep = pl.scan([root], cfg)
            f.write(f"jobs={len(rep.jobs)} vacias={len(rep.empty)}\n")
            for j in rep.jobs:
                f.write(f"  {j.name}: {media_txt(j)}\n")
            import render_ortho  # noqa: F401
            f.write("render_ortho OK\n")
        except Exception:
            f.write(traceback.format_exc())


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--diag":
        diag(sys.argv[2])
        return
    if len(sys.argv) > 2 and sys.argv[1] == "--autotest":
        autotest(sys.argv[2])
        return
    start = sys.argv[1] if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]) else None
    App(start).mainloop()


if __name__ == "__main__":
    main()
