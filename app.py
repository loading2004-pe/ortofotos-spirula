"""Ortofotos Spirula — procesamiento por lotes + revisión.

Uso: abre el programa (o arrastra la carpeta raíz sobre el .exe), pulsa
Escanear, luego Procesar. Al terminar, pestaña Revisar: B = bien, M = mal (x_).
"""
from __future__ import annotations

import math
import numpy as np
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
        self.after(60 * 1000, self._periodic_update)

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
        self._modo_widget(top)
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
        root = self.root_var.get()
        if only is None and os.path.isdir(root) and \
                not any(j.status not in ("vacía", "vacia") for j in self.jobs):
            plys = pl.list_plys(root, recursive=False)
            if plys:   # carpeta de modelos .ply (no de videos): se renderizan directamente
                if messagebox.askyesno("Procesar", f"Esta carpeta no tiene videos pero sí {len(plys)} modelos "
                                       ".ply.\n\n¿Renderizarlos ahora con el tipo «"
                                       f"{self.modo_var.get()}»?\n(Si son recortes de un mismo ambiente "
                                       "— CARA…, LADO…, PLANTA, TECHO — se orientan juntos.)"):
                    self.nb.select(self.review_tab)
                    self._start_linked(plys, root)
                return
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
        if p.get("todos"):
            names |= {j.name for j in self.jobs}
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
                    self.bell()
                    if isinstance(data, dict):
                        if data.get("abrir"):
                            self._conv_last_dir = data["abrir"][0]
                            open_path(data["abrir"][0])
                    elif data:
                        self._conv_items = data
                        self.btn_conv_render.state(["!disabled"])
                        if messagebox.askyesno("Listo", f"{len(data)} modelo(s) de vuelta como .ply.\n\n"
                                                        "¿Renderizar sus ortofotos ahora?"):
                            self._conv_render()
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
        self._modo_widget(top)
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
        ttk.Button(bar2, text="🧊 Visor 6 caras", command=self._open_cube_viewer).pack(side="left")
        self.link_lbl = ttk.Label(right, text="", foreground="#555")
        self.link_lbl.pack(anchor="w", pady=(4, 0))
        nm_row = ttk.Frame(right)
        nm_row.pack(fill="x", pady=4)
        self.rev_name = ttk.Label(nm_row, text="", font=("Segoe UI", 12, "bold"))
        self.rev_name.pack(side="left")
        self.faces_btn = ttk.Button(nm_row, text="📂 Abrir sus 6 caras", command=self._open_faces)
        self.faces_lbl = ttk.Label(nm_row, text="", foreground="#1f5fa8")
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
        if self.review_dir:
            if self.cfg.get("ultima_revision") != self.review_dir:
                self.cfg["ultima_revision"] = self.review_dir
                try:
                    pl.save_config(CONFIG_PATH, self.cfg)
                except OSError:
                    pass
            return self.review_dir
        d = self._process_results_dir()
        has = os.path.isdir(d) and any(f.lower().endswith(".png") for f in os.listdir(d))
        if not has:   # sin resultados de '1. Procesar': la última carpeta revisada o la de .ply limpios
            for alt in (self.cfg.get("ultima_revision", ""),
                        os.path.join(self.cfg.get("carpeta_ply_limpios", "") or "", "ortofotos_limpias")):
                if alt and os.path.isdir(alt) and any(f.lower().endswith(".png") for f in os.listdir(alt)):
                    return alt
        return d

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
        self._update_faces_link(p)
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

    MODOS = (("normal", "Normal (puertas y ventanas)"), ("exterior", "Exterior · 6 caras"),
             ("interior", "Interior · 6 caras"), ("auto", "Detectar solo"))

    def _modo_widget(self, parent):
        """Selector del tipo de render (el mismo en 1. Procesar y 2. Revisar)."""
        if not hasattr(self, "modo_var"):
            lab = dict(self.MODOS).get(self.cfg.get("modo_render", "normal"), self.MODOS[0][1])
            self.modo_var = tk.StringVar(value=lab)
        ttk.Label(parent, text="  Tipo:").pack(side="left")
        cb = ttk.Combobox(parent, textvariable=self.modo_var, state="readonly", width=26,
                          values=[m[1] for m in self.MODOS])
        cb.pack(side="left", padx=(2, 6))
        cb.bind("<<ComboboxSelected>>", self._set_modo)

    def _set_modo(self, _e=None):
        inv = {v: k for k, v in self.MODOS}
        self.cfg["modo_render"] = inv.get(self.modo_var.get(), "normal")
        pl.save_config(CONFIG_PATH, self.cfg)
        self.q.put(("log", f"Tipo de render: {self.modo_var.get()} (se aplica a los próximos renders; "
                           "las carpetas ajustadas en el visor 6 caras conservan su elección)"))

    def _faces_dir_for(self, p):
        import cubo
        fname = os.path.basename(p)
        name = os.path.splitext(fname[2:] if fname.lower().startswith("x_") else fname)[0]
        return cubo.faces_dir(os.path.dirname(p), name)

    def _update_faces_link(self, p):
        d = self._faces_dir_for(p)
        if os.path.isdir(d):
            n = len([f for f in os.listdir(d) if f.lower().endswith(".png")])
            self.faces_lbl.config(text=f"  ambiente de {n} caras · guardadas en: {d}")
            self.faces_btn.pack(side="left", padx=(12, 0))
            self.faces_lbl.pack(side="left")
        else:
            self.faces_btn.pack_forget()
            self.faces_lbl.pack_forget()

    def _open_faces(self):
        i = self._cur()
        if i is not None:
            d = self._faces_dir_for(self.files[i])
            if os.path.isdir(d):
                open_path(d)

    def _ply_for(self, name, d):
        """(.ply, entrada del manifiesto) de una imagen de la carpeta de revisión."""
        man = pl.read_manifest(d)
        entry = man.get(name)
        ply = entry.get("ply") if entry else None
        if not ply:   # ortofoto de '1. Procesar': su .ply está en _orto_trabajo
            src = pl.newest_ply(os.path.join(pl.work_dir(self.cfg, self.root_var.get(), name), "runs"))
            limpio = os.path.join(pl.side_dirs(self.cfg, self.root_var.get())[1], name + ".ply")
            ply = limpio if os.path.isfile(limpio) else src
            entry = {"ply": ply, "flip": False, "rot": 0}
        return ply, entry

    def _open_cube_viewer(self):
        i = self._cur()
        if i is None:
            messagebox.showinfo("Visor 6 caras", "Elige primero una imagen de la lista.")
            return
        d = self._results_dir()
        fname = os.path.basename(self.files[i])
        name = os.path.splitext(fname[2:] if fname.lower().startswith("x_") else fname)[0]
        ply, entry = self._ply_for(name, d)
        if not ply or not os.path.isfile(ply):
            messagebox.showinfo("Visor 6 caras", "No encuentro el .ply de esta imagen.")
            return
        if entry.get("ambiente"):     # recorte de un ambiente por partes: se ajusta el ambiente entero
            man = pl.read_manifest(d)
            parts = [(e["ply"], n, e.get("rol", "pared")) for n, e in sorted(man.items())
                     if e.get("ambiente") == entry["ambiente"] and e.get("ply") and os.path.isfile(e["ply"])]
            if len(parts) >= 2:
                AmbienteViewer(self, entry["ambiente"], parts, d)
                return
        root = self.root_var.get()
        wd = pl.work_dir(self.cfg, root, name) if root else None
        CubeViewer(self, name, ply, wd if (wd and os.path.isdir(wd)) else None, d, entry)

    def _adjust(self, flip=False, rot=0):
        """Gira 90° (al instante, sobre la imagen) o voltea (re-render) la ortofoto seleccionada."""
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
        if rot and not flip and not os.path.isdir(self._faces_dir_for(self.files[i])):
            # girar 90° es instantáneo: se gira la imagen ya hecha (sin volver a renderizar)
            path = self.files[i]
            self.link_lbl.config(text=f"Girando {name}…")

            def turn():
                from PIL import Image
                Image.MAX_IMAGE_PIXELS = None
                try:
                    with Image.open(path) as im:
                        im.load()
                        out = im.transpose(Image.ROTATE_270)
                    out.save(path, compress_level=1)
                    msg = f"{name}: girado 90°"
                except Exception as e:
                    msg = f"{name}: no se pudo girar ({e})"
                self.after(0, lambda: (self._thumbs.clear(), self.link_lbl.config(text=msg),
                                       self._show_current()))
            threading.Thread(target=turn, daemon=True).start()
            return
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
        f = ttk.Frame(nb, padding=16)
        nb.add(f, text="3. Convertir nube")
        self._conv_src, self._conv_clouds, self._conv_items = [], [], []
        self._conv_busy = False
        big = ttk.Style(self)
        big.configure("Big.TButton", font=("Segoe UI", 11, "bold"), padding=(14, 10))

        ttk.Label(f, text="Limpiar un modelo en CloudCompare o ReCap", font=("Segoe UI", 13, "bold")
                  ).pack(anchor="w")
        ttk.Label(f, foreground="#444", justify="left", text=(
            "1.  Elige a qué programa lo pasas y selecciona el .ply.\n"
            "2.  CloudCompare: abre el .ply, borra lo que sobra y guárdalo (.ply o .las).\n"
            "     ReCap: importa el .pts, borra lo que sobra y expórtalo como .pts.\n"
            "     No muevas ni gires la nube; solo borra.\n"
            "3.  Pulsa  «Traer de vuelta a .ply»  y elige lo que guardaste. La app reconoce sola de qué "
            "programa viene y busca el modelo original: queda un .ply listo para Spirula, con todos sus datos.")
                  ).pack(anchor="w", pady=(6, 14))

        row = ttk.Frame(f)
        row.pack(anchor="w")
        ttk.Button(row, text="①  A CloudCompare  (.ply)", style="Big.TButton",
                   command=lambda: self._conv_quick_a(("ply",))).pack(side="left")
        ttk.Button(row, text="①  A ReCap  (.pts)", style="Big.TButton",
                   command=lambda: self._conv_quick_a(("pts",))).pack(side="left", padx=(8, 0))
        ttk.Button(row, text="②  Traer de vuelta a .ply", style="Big.TButton",
                   command=self._conv_quick_b).pack(side="left", padx=(24, 0))
        self.btn_conv_render = ttk.Button(row, text="Renderizar ortofotos", command=self._conv_render,
                                          state="disabled")
        self.btn_conv_render.pack(side="left", padx=(12, 0))
        ttk.Button(row, text="Abrir carpeta", command=self._conv_open_out).pack(side="left", padx=6)

        # ---- opciones avanzadas (ocultas)
        self.conv_fmt = {"las": tk.BooleanVar(value=False)}
        self.conv_hide = tk.BooleanVar(value=self.cfg.get("conv_ocultar", True))
        self.conv_thr = tk.StringVar(value=str(self.cfg.get("conv_opacidad_min", 0.05)))
        self.conv_up = tk.BooleanVar(value=self.cfg.get("conv_de_pie", True))
        self.conv_out_a = tk.StringVar(value="")
        self.conv_out_b = tk.StringVar(value="")
        self.conv_to_clean = tk.BooleanVar(value=self.cfg.get("conv_a_limpios", False))
        self._adv_shown = tk.BooleanVar(value=False)
        tog = ttk.Checkbutton(f, text="Opciones avanzadas", variable=self._adv_shown,
                              command=self._conv_toggle_adv)
        tog.pack(anchor="w", pady=(16, 0))
        self._conv_tog = tog
        adv = ttk.Frame(f, padding=(18, 6, 0, 0))
        self._conv_adv = adv
        r = ttk.Frame(adv)
        r.pack(anchor="w")
        ttk.Checkbutton(r, text="Crear también .las (lo abren CloudCompare y ReCap)",
                        variable=self.conv_fmt["las"]).pack(side="left")
        r = ttk.Frame(adv)
        r.pack(anchor="w", pady=(4, 0))
        ttk.Checkbutton(r, text="Quitar puntos casi transparentes (opacidad menor a",
                        variable=self.conv_hide).pack(side="left")
        ttk.Spinbox(r, from_=0.0, to=0.9, increment=0.01, width=5, textvariable=self.conv_thr
                    ).pack(side="left", padx=2)
        ttk.Label(r, text=")").pack(side="left")
        ttk.Checkbutton(r, text="Poner de pie (Z arriba)", variable=self.conv_up).pack(side="left", padx=12)
        ttk.Checkbutton(adv, text="Al traer de vuelta, copiar también a ply_limpios (lo usa 1. Procesar)",
                        variable=self.conv_to_clean).pack(anchor="w", pady=(4, 0))
        for lab, var in (("Carpeta para las nubes (vacío = junto al .ply, en nube_para_editar):", self.conv_out_a),
                         ("Carpeta para los .ply de vuelta (vacío = junto a la nube, en ply_desde_nube):",
                          self.conv_out_b)):
            r = ttk.Frame(adv)
            r.pack(fill="x", pady=(4, 0))
            ttk.Label(r, text=lab).pack(side="left")
            ttk.Entry(r, textvariable=var, width=50).pack(side="left", fill="x", expand=True, padx=4)
            ttk.Button(r, text="…", width=3, command=lambda v=var: v.set(
                filedialog.askdirectory() or v.get())).pack(side="left")

        self.conv_log = tk.Text(f, height=10, font=("Consolas", 10))
        self.conv_log.pack(fill="both", expand=True, pady=(12, 0))

    def _conv_toggle_adv(self):
        if self._adv_shown.get():
            self._conv_adv.pack(anchor="w", fill="x", after=self._conv_tog)
        else:
            self._conv_adv.pack_forget()

    def _conv_quick_a(self, fmts):
        prog = "CloudCompare" if "ply" in fmts else "ReCap"
        files = list(filedialog.askopenfilenames(title=f"Elige el .ply para pasar a {prog}",
                                                 filetypes=[("PLY", "*.ply")]))
        if files:
            self._conv_src = files
            if self.conv_fmt["las"].get():
                fmts = tuple(fmts) + ("las",)
            self._conv_to_cloud(fmts)

    def _conv_quick_b(self):
        import ply_convert as pc
        files = list(filedialog.askopenfilenames(
            title="Elige la nube que guardaste en CloudCompare / ReCap",
            filetypes=[("Nubes de puntos", "*.ply *.las *.pts *.xyz *.txt *.asc *.csv *.e57"),
                       ("Todos", "*.*")]))
        files = [x for x in files if x.lower().endswith(pc.CLOUD_EXT)]
        if files:
            self._conv_clouds = files
            self._conv_to_splat()

    def _conv_say(self, msg):
        self.q.put(("conv", f"{time.strftime('%H:%M:%S')} {msg}"))

    def _conv_save_opts(self):
        self.cfg["conv_ocultar"] = bool(self.conv_hide.get())
        self.cfg["conv_de_pie"] = bool(self.conv_up.get())
        self.cfg["conv_a_limpios"] = bool(self.conv_to_clean.get())
        try:
            self.cfg["conv_opacidad_min"] = float(self.conv_thr.get())
        except ValueError:
            pass
        pl.save_config(CONFIG_PATH, self.cfg)

    def _conv_to_cloud(self, fmts=("ply",)):
        import ply_convert as pc
        if self._conv_busy:
            messagebox.showinfo("Convertir", "Ya hay una conversión en marcha.")
            return
        if not self._conv_src:
            messagebox.showinfo("Convertir", "Primero elige los splats .ply.")
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
            self.q.put(("convfin", {"abrir": sorted(outs)}))
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
        if getattr(self, "_conv_last_dir", None) and not self._conv_items:
            open_path(self._conv_last_dir)
        elif self._conv_items:
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
        self.after(30 * 60 * 1000, self._periodic_update)

    def _check_update(self, silent=False):
        try:
            import updater
        except Exception as e:
            if not silent:
                messagebox.showerror("Actualizaciones", f"No disponible: {e}")
            return
        if getattr(self, "_upd_win", None) is not None:
            return
        if not silent:
            self.upd_lbl.config(text="  Buscando…")
        cur = code_version()

        def work():
            man, msg = updater.available(DATA_DIR, cur, self.cfg.get("url_actualizaciones") or None)
            self.after(0, lambda: self._update_found(man, msg, silent))
        threading.Thread(target=work, daemon=True).start()

    def _update_found(self, man, msg, silent):
        self.upd_lbl.config(text=f"  Versión {code_version()} · {msg}")
        if not man:
            if not silent:
                messagebox.showinfo("Actualizaciones", msg)
            return
        self._show_update_window(man)

    def _show_update_window(self, man):
        """Ventana encima de todo con el botón 'Actualizar'."""
        w = tk.Toplevel(self)
        self._upd_win = w
        w.title("Actualización disponible")
        w.attributes("-topmost", True)
        w.resizable(False, False)
        f = ttk.Frame(w, padding=18)
        f.pack()
        ttk.Label(f, text="Hay una actualización disponible", font=("Segoe UI", 13, "bold")).pack(anchor="w")
        ttk.Label(f, text=f"Versión {man.get('version')}  (tienes la {code_version()})",
                  foreground="#1f5fa8").pack(anchor="w", pady=(4, 0))
        if man.get("notas"):
            ttk.Label(f, text=man["notas"], wraplength=400, justify="left").pack(anchor="w", pady=(6, 0))
        busy = self.runner is not None and self.btn_stop["state"] == "normal"
        if busy:
            ttk.Label(f, text="Hay un proceso en marcha: se actualizará al terminar la etapa en curso "
                              "(no se pierde nada) y luego continúa con lo pendiente.",
                      wraplength=400, foreground="#555").pack(anchor="w", pady=(6, 0))
        st = ttk.Label(f, text="")
        st.pack(anchor="w", pady=(6, 0))
        bar = ttk.Frame(f)
        bar.pack(fill="x", pady=(12, 0))

        def later():
            self._upd_win = None
            w.destroy()

        def go():
            for c in bar.winfo_children():
                c.state(["disabled"])
            st.config(text="Descargando…")
            import updater

            def work():
                ok, msg = updater.apply(DATA_DIR, man, self.cfg.get("url_actualizaciones") or None)
                self.after(0, lambda: done(ok, msg))
            threading.Thread(target=work, daemon=True).start()

        def done(ok, msg):
            self.q.put(("log", f"Actualizaciones: {msg}"))
            if not ok:
                st.config(text=msg)
                for c in bar.winfo_children():
                    c.state(["!disabled"])
                return
            self._upd_win = None
            w.destroy()
            if self.runner is not None and self.btn_stop["state"] == "normal":
                self._restart_when_idle = True
                self.runner.stop_after_current()
                self.btn_soft.config(state="disabled")
                self.q.put(("log", "Se reiniciará con la versión nueva al terminar la etapa en curso."))
            else:
                self._restart()
        ttk.Button(bar, text="Actualizar", command=go).pack(side="right")
        ttk.Button(bar, text="Más tarde", command=later).pack(side="right", padx=8)
        w.protocol("WM_DELETE_WINDOW", later)
        w.update_idletasks()
        x = self.winfo_rootx() + (self.winfo_width() - w.winfo_width()) // 2
        y = self.winfo_rooty() + (self.winfo_height() - w.winfo_height()) // 3
        w.geometry(f"+{max(0, x)}+{max(0, y)}")
        w.lift()

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


class CubeViewer(tk.Toplevel):
    """Alinear ambientes de 6 caras: ventana grande, todo en vivo.
    Izquierda: planta (clic en una pared para elegirla, arrastrarla la mueve, el punto blanco
    la gira). Centro: la cara elegida en grande con cuadrícula. Derecha: controles de esa cara."""
    COLORS = {"Norte": "#ff5252", "Este": "#ffd740", "Sur": "#69f0ae", "Oeste": "#40c4ff",
              "Techo": "#e0e0e0", "Suelo": "#bcaaa4"}
    TIPOS = (("auto", "Detectar"), ("fachada", "Normal"), ("interior", "Interior"), ("exterior", "Exterior"))
    SLIDERS = (("giro", "Girar", -45, 45, "°"), ("incl", "Inclinar", -30, 30, "°"),
               ("rot", "Enderezar", -20, 20, "°"), ("mover", "Mover plano", -15, 15, "%"),
               ("ampliar", "Encuadre", -40, 60, "%"))

    def __init__(self, app, name, ply, wd, results_dir, entry):
        super().__init__(app)
        import cubo
        self.app, self.name, self.ply, self.wd, self.results = app, name, ply, wd, results_dir
        self.entry = entry or {}
        conf = dict(self.entry.get("cubo") or (cubo.load_config(wd) if wd else {}))
        self.title(f"Alinear 6 caras — {name}")
        try:
            self.state("zoomed")
        except tk.TclError:
            self.geometry(f"{self.winfo_screenwidth()}x{self.winfo_screenheight() - 60}+0+0")
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.PS = int(max(380, min(620, sh * 0.62)))                 # planta
        self.FW = int(max(500, sw - self.PS - 420))                   # cara grande
        self.FH = int(max(320, sh * 0.62))
        self.tipo = tk.StringVar(value=conf.get("tipo", "auto"))
        self.norte = int(conf.get("norte", 0))
        sel = conf.get("caras")
        self.incl = {k: tk.BooleanVar(value=(k in sel) if sel else True) for k in cubo.CARAS}
        self.ajustes = {k: dict(v) for k, v in (conf.get("ajustes_ui") or {}).items()}
        self.manual = conf.get("manual")
        self.cur = None
        self.sel = "Norte"
        self.model = self.pmodel = self.an = self.cams = self.ups = None
        self._pc = self._plan_photo = self._face_photo = None
        self._plan_key = None
        self._busy = False
        self._dirty = False
        self._gen = 0
        self._drag = None
        self._sliding = False
        self._slide_after = None
        self._thumb_after = None
        self._thumbs = {}
        self._build()
        threading.Thread(target=self._load, daemon=True).start()

    # ------------------------------------------------------------ interfaz
    def _build(self):
        import cubo
        top = ttk.Frame(self, padding=(10, 8))
        top.pack(fill="x")
        ttk.Label(top, text=self.name, font=("Segoe UI", 13, "bold")).pack(side="left")
        ttk.Label(top, text="   Tipo:").pack(side="left")
        for v, lab in self.TIPOS:
            ttk.Radiobutton(top, text=lab, value=v, variable=self.tipo,
                            command=self._tipo_changed).pack(side="left")
        ttk.Button(top, text="⟳ Norte", command=self._rot_north).pack(side="left", padx=(14, 0))
        self.btn_go = ttk.Button(top, text="✓ Guardar y renderizar", command=self._render_final,
                                 state="disabled")
        self.btn_go.pack(side="right")
        ttk.Button(top, text="📂 Abrir carpeta", command=self._open_dir).pack(side="right", padx=6)
        bar = ttk.Frame(self, padding=(10, 0))
        bar.pack(fill="x")
        self.prog = ttk.Progressbar(bar, mode="determinate", maximum=100)
        self.prog.pack(side="left", fill="x", expand=True)
        self.status = ttk.Label(bar, text="Cargando modelo…", width=60)
        self.status.pack(side="left", padx=8)

        body = ttk.Frame(self, padding=10)
        body.pack(fill="both", expand=True)
        left = ttk.Frame(body)
        left.pack(side="left", fill="y")
        self.plan = tk.Canvas(left, width=self.PS, height=self.PS, bg="#222", highlightthickness=0)
        self.plan.pack()
        self.info = ttk.Label(left, text="", wraplength=self.PS, foreground="#666")
        self.info.pack(anchor="w", pady=(4, 0))
        self.plan.bind("<ButtonPress-1>", self._press)
        self.plan.bind("<B1-Motion>", self._motion)
        self.plan.bind("<ButtonRelease-1>", self._release)

        mid = ttk.Frame(body)
        mid.pack(side="left", fill="both", expand=True, padx=10)
        self.face_title = ttk.Label(mid, text="", font=("Segoe UI", 12, "bold"))
        self.face_title.pack(anchor="w")
        self.face_cv = tk.Canvas(mid, width=self.FW, height=self.FH, bg="#222", highlightthickness=0)
        self.face_cv.pack(fill="both", expand=True)
        self.strip = ttk.Frame(mid)
        self.strip.pack(fill="x", pady=(6, 0))
        self._strip_cells = {}
        for k in cubo.CARAS:
            cell = tk.Frame(self.strip, bd=3, relief="flat")
            cell.pack(side="left", padx=3)
            tk.Frame(cell, bg=self.COLORS[k], height=5).pack(fill="x")
            lab = ttk.Label(cell, text=k, font=("Segoe UI", 9, "bold"), cursor="hand2")
            lab.pack()
            img = tk.Label(cell, bg="#333", width=18, height=6, cursor="hand2")
            img.pack()
            chk = ttk.Checkbutton(cell, text="incluir", variable=self.incl[k], command=self._changed)
            chk.pack()
            for w in (lab, img):
                w.bind("<Button-1>", lambda e, k=k: self._select(k))
            self._strip_cells[k] = (cell, img)

        right = ttk.Frame(body, width=380)
        right.pack(side="left", fill="y")
        self.sel_bar = tk.Frame(right, bg=self.COLORS["Norte"], height=6)
        self.sel_bar.pack(fill="x")
        self.sel_lbl = ttk.Label(right, text="Norte", font=("Segoe UI", 16, "bold"))
        self.sel_lbl.pack(anchor="w")
        ttk.Label(right, text="Ajusta solo esta cara. Clic en una pared de la planta o en una "
                              "miniatura para cambiar de cara.", wraplength=360, foreground="#666").pack(anchor="w")
        self.sv = {}
        for key, lab, lo, hi, unit in self.SLIDERS:
            row = ttk.Frame(right)
            row.pack(fill="x", pady=(10, 0))
            head = ttk.Frame(row)
            head.pack(fill="x")
            ttk.Label(head, text=lab, font=("Segoe UI", 10, "bold")).pack(side="left")
            val = ttk.Label(head, text="0", width=8, anchor="e")
            val.pack(side="right")
            v = tk.DoubleVar(value=0.0)
            sc = ttk.Scale(row, from_=lo, to=hi, variable=v, length=360,
                           command=lambda _x, k=key: self._slid(k))
            sc.pack(fill="x")
            sc.bind("<Double-Button-1>", lambda e, k=key: self._set_val(k, 0.0))
            self.sv[key] = (v, val, unit)
        ttk.Button(right, text="↺ Reiniciar esta cara", command=self._reset_face).pack(anchor="w", pady=(12, 0))
        self.adv_on = tk.BooleanVar(value=bool(self.manual))
        ttk.Checkbutton(right, text="Avanzado (toda la caja)", variable=self.adv_on,
                        command=self._toggle_adv).pack(anchor="w", pady=(18, 0))
        self.adv = ttk.Frame(right, padding=(16, 4, 0, 0))
        r = ttk.Frame(self.adv)
        r.pack(fill="x")
        ttk.Label(r, text="Arriba:").pack(side="left")
        self.up_choice = tk.StringVar(value="Auto")
        cb = ttk.Combobox(r, textvariable=self.up_choice, state="readonly", width=6,
                          values=["Auto", "+X", "-X", "+Y", "-Y", "+Z", "-Z"])
        cb.pack(side="left", padx=4)
        cb.bind("<<ComboboxSelected>>", lambda e: self._set_up())
        ttk.Button(r, text="Re-detectar paredes", command=self._refit).pack(side="left", padx=4)
        self.gv = {}
        for key, lab in (("yaw", "Girar caja"), ("ta", "Inclinar A"), ("tb", "Inclinar B")):
            rr = ttk.Frame(self.adv)
            rr.pack(fill="x", pady=(4, 0))
            ttk.Label(rr, text=lab, width=11).pack(side="left")
            v = tk.DoubleVar(value=0.0)
            ttk.Scale(rr, from_=-30, to=30, variable=v, length=200,
                      command=lambda _x, k=key: self._gslid(k)).pack(side="left")
            self.gv[key] = [v, 0.0]
        rr = ttk.Frame(self.adv)
        rr.pack(fill="x", pady=(4, 0))
        for lab, idx in (("Suelo", 0), ("Techo", 1)):
            ttk.Label(rr, text=lab).pack(side="left")
            ttk.Button(rr, text="▼", width=2, command=lambda i=idx: self._height(i, -1)).pack(side="left")
            ttk.Button(rr, text="▲", width=2, command=lambda i=idx: self._height(i, 1)).pack(side="left", padx=(0, 8))
        ttk.Button(self.adv, text="↺ Todo automático", command=self._reset_all).pack(anchor="w", pady=(6, 0))
        if self.adv_on.get():
            self.adv.pack(fill="x")
        self.res_lbl = ttk.Label(right, text="", wraplength=360, foreground="#1f5fa8")
        self.res_lbl.pack(anchor="w", pady=(16, 0))
        self._refresh_res()

    # ------------------------------------------------------------ carga
    def _load(self):
        import render_ortho as ro
        import cubo
        try:
            self.model = ro.read_splat_ply(self.ply)
            self.pmodel = cubo.preview_model(self.model)
            cams, ups, _o = pl.find_cams(self.name, self.ply, self.app.cfg,
                                         self.app.root_var.get() or None, self.app._cam_store())
            self.cams, self.ups = cams, ups
            self.an = cubo.analyze(self.model, cams, ups)
            ro.render(self.pmodel, cubo.plan_cam(self.an), 32, 0.0)       # compila numba (1ª vez)
        except Exception as e:
            err = f"Error al cargar: {e}"
            self.after(0, lambda: self.status.config(text=err))
            return
        self.after(0, self._loaded)

    def _loaded(self):
        nombres = {"fachada": "fachada / puerta / ventana", "interior": "interior de 6 caras",
                   "exterior": "exterior de 6 caras"}
        self.cur = dict(self.manual) if self.manual else dict(self.an)
        self._ref_d0 = np.array(self.cur["d0"], float)
        self.info.config(text=f"Detectado: {nombres[self.an['tipo_auto']]} · {self.an['motivo']}")
        self.status.config(text="Listo. Mueve los controles: la vista se actualiza en vivo.")
        self.btn_go.state(["!disabled"])
        self._select(self.sel)
        self._changed(plan=True)

    # ------------------------------------------------------------ estado
    def _eff_tipo(self):
        t = self.tipo.get()
        if t == "auto":
            t = self.an["tipo_auto"] if self.an else "fachada"
        if t == "fachada" and (self.adv_on.get() or self.ajustes):
            t = "interior" if self.an and self.an["tipo_auto"] == "interior" else "exterior"
        return t

    def _cube_tipo(self):
        t = self._eff_tipo()
        return t if t in ("interior", "exterior") else "exterior"

    def _use(self):
        return self.cur if self.adv_on.get() else self.an

    def _span(self):
        an = self._use() if self.an is not None else None
        return max(b - a for a, b in an["caja"][:2]) if an else 1.0

    def _adj_for_render(self):
        """Ajustes con 'mover' en unidades del modelo (en pantalla se muestra en %)."""
        out = {}
        span = self._span()
        for k, v in self.ajustes.items():
            d = dict(v)
            if "mover" in d:
                d["mover"] = float(d["mover"]) / 100.0 * span
            if "ampliar" in d:
                d["ampliar"] = float(d["ampliar"]) / 100.0
            out[k] = d
        return out

    # ------------------------------------------------------------ bucle en vivo
    def _changed(self, plan=False):
        if plan:
            self._plan_key = None
        self._draw_overlay()
        self._dirty = True
        if not self._busy:
            self._kick()
        if self._thumb_after:
            self.after_cancel(self._thumb_after)
        self._thumb_after = self.after(500, self._thumbs_update)

    def _kick(self):
        if self.pmodel is None or self.an is None or self._use() is None or not self._dirty:
            return
        self._dirty = False
        self._busy = True
        self._gen += 1
        gen = self._gen
        tipo, an, sel = self._eff_tipo(), dict(self._use()), self.sel
        adj = self._adj_for_render()
        need_plan = self._plan_key != self._plan_sig(an)
        fw, fh, ps = self.FW, self.FH, self.PS
        cap = 700 if self._drag or self._sliding else 1600      # mientras arrastras: más rápido

        def work():
            import render_ortho as ro
            import cubo
            plan = pc = None
            if need_plan:
                pc = self._plan_frame(an)
                pm = cubo.plan_model(self.pmodel, an, grow=0.6)
                plan = ro.render(pm, pc, ps, 0.0, background=(34, 34, 34))
            face = None
            if tipo in ("interior", "exterior") and sel in cubo.face_cams(an, tipo, self.norte):
                fc = cubo.face_cams(an, tipo, self.norte, ajustes=adj)
                cam, clip = fc[sel]
                ext = max(cam.u_range[1] - cam.u_range[0], cam.v_range[1] - cam.v_range[0])
                du, dv = cam.u_range[1] - cam.u_range[0], cam.v_range[1] - cam.v_range[0]
                ls = int(max(fw * ext / du if du / dv > fw / fh else fh * ext / dv, 64))
                face = ro.render(self.pmodel, cam, min(ls, cap), clip / max(ext, 1e-9),
                                 background=(34, 34, 34))
            self.after(0, lambda: self._live_done(gen, plan, pc, face, an, tipo))
        threading.Thread(target=work, daemon=True).start()

    def _live_done(self, gen, plan, pc, face, an, tipo):
        from PIL import ImageTk
        self._busy = False
        if plan is not None:
            self._pc = pc
            self._plan_key = self._plan_sig(an)
            self._plan_photo = ImageTk.PhotoImage(plan)
            self.plan.delete("img")
            self.plan.create_image(0, 0, image=self._plan_photo, anchor="nw", tags="img")
            self.plan.tag_lower("img")
            if self.cams is not None and len(self.cams):
                import cubo
                for x, y in cubo.project(pc, self.cams, self.PS, self.PS):
                    self.plan.create_oval(x - 2, y - 2, x + 2, y + 2, fill="white", outline="", tags="img")
            self._draw_overlay()
        if gen == self._gen:
            self._face_draw(face, tipo)
        if self._dirty:
            self._kick()

    def _face_draw(self, im, tipo):
        from PIL import ImageTk
        cv = self.face_cv
        cv.delete("all")
        W, H = int(cv.winfo_width() or self.FW), int(cv.winfo_height() or self.FH)
        if tipo not in ("interior", "exterior"):
            cv.create_text(W // 2, H // 2, fill="#bbb", font=("Segoe UI", 13), text=(
                "Tipo «Fachada»: se renderiza como siempre (un plano, enderezado).\n"
                "Elige Interior o Exterior arriba para trabajar por caras."))
            return
        if im is None:
            cv.create_text(W // 2, H // 2, fill="#bbb", font=("Segoe UI", 13),
                           text=f"«{self.sel}» no aplica a un {tipo}.")
            return
        im = im.copy()
        im.thumbnail((W, H))
        self._face_photo = ImageTk.PhotoImage(im)
        x0, y0 = (W - im.size[0]) // 2, (H - im.size[1]) // 2
        cv.create_image(x0, y0, image=self._face_photo, anchor="nw")
        for i in range(1, 12):
            x = x0 + im.size[0] * i / 12
            cv.create_line(x, y0, x, y0 + im.size[1], fill="#9e9e9e", stipple="gray25")
        for i in range(1, 8):
            y = y0 + im.size[1] * i / 8
            cv.create_line(x0, y, x0 + im.size[0], y, fill="#9e9e9e", stipple="gray25")
        cv.create_rectangle(x0, y0, x0 + im.size[0], y0 + im.size[1], outline=self.COLORS.get(self.sel, "#fff"))

    def _thumbs_update(self):
        if self.pmodel is None or self.an is None or self._use() is None:
            return
        tipo, an, adj = self._eff_tipo(), dict(self._use()), self._adj_for_render()
        if tipo not in ("interior", "exterior"):
            return

        def work():
            import cubo
            imgs = cubo.render_faces(self.pmodel, an, tipo, self.norte, None, long_side=150,
                                     background=(34, 34, 34), log=lambda *_: None, ajustes=adj)
            self.after(0, lambda: self._thumbs_show(imgs))
        threading.Thread(target=work, daemon=True).start()

    def _thumbs_show(self, imgs):
        from PIL import ImageTk
        self._thumbs = {}
        for k, (cell, lab) in self._strip_cells.items():
            im = imgs.get(k)
            cell.configure(relief="solid" if k == self.sel else "flat")
            if im is None:
                lab.configure(image="", text="—", width=18, height=6)
                continue
            im = im.copy()
            im.thumbnail((150, 100))
            t = ImageTk.PhotoImage(im)
            self._thumbs[k] = t
            lab.configure(image=t, width=150, height=100)

    # ------------------------------------------------------------ planta
    def _plan_sig(self, an):
        return (tuple(np.round(an["up"], 3)), self.norte, self.adv_on.get())

    def _plan_frame(self, an):
        import cubo
        ref = dict(an)
        up = np.array(an["up"], float)
        d = self._ref_d0 - (self._ref_d0 @ up) * up
        if np.linalg.norm(d) < 1e-6:
            d = np.array(an["d0"], float)
        d = d / np.linalg.norm(d)
        ref["d0"], ref["d1"] = d.tolist(), np.cross(up, d).tolist()
        return cubo.plan_cam(ref, self.norte, extra=0.45)

    @staticmethod
    def _center(an):
        up, d0, d1 = (np.array(an[k], float) for k in ("up", "d0", "d1"))
        (a0, b0), (a1, b1), (au, bu) = an["caja"]
        return 0.5 * (a0 + b0) * d0 + 0.5 * (a1 + b1) * d1 + 0.5 * (au + bu) * up

    def _draw_overlay(self):
        import cubo
        cv, pc, an = self.plan, self._pc, self._use()
        if pc is None or an is None:
            return
        cv.delete("ov")
        S = self.PS
        self._segs = {}
        self._handle = None
        cube = self._eff_tipo() in ("interior", "exterior")
        fc = cubo.face_cams(an, self._cube_tipo(), self.norte, ajustes=self._adj_for_render())
        ctr = cubo.project(pc, [self._center(an)], S, S)[0]
        for nm in ("Norte", "Este", "Sur", "Oeste"):
            cam = fc[nm][0]
            p0 = cam.origin + cam.right * cam.u_range[0]
            p1 = cam.origin + cam.right * cam.u_range[1]
            (x0, y0), (x1, y1) = cubo.project(pc, [p0, p1], S, S)
            self._segs[nm] = (x0, y0, x1, y1)
            on = nm == self.sel
            cv.create_line(x0, y0, x1, y1, fill=self.COLORS[nm], width=(7 if on else 3) if cube else 1,
                           dash=() if cube else (4, 4), tags="ov")
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            cv.create_text(mx + (mx - ctr[0]) * 0.1, my + (my - ctr[1]) * 0.1,
                           text=nm + (" ●" if self.ajustes.get(nm) else ""), fill=self.COLORS[nm],
                           font=("Segoe UI", 11, "bold"), tags="ov")
            if on and cube:
                cv.create_oval(x1 - 8, y1 - 8, x1 + 8, y1 + 8, fill="white", outline="#222", tags="ov")
                self._handle = (x1, y1, mx, my)
        cv.create_text(S - 22, 18, text="N ↑", fill="#ff5252", font=("Segoe UI", 12, "bold"), tags="ov")

    def _world(self, x, y):
        pc, S = self._pc, float(self.PS)
        u = pc.u_range[0] + x / S * (pc.u_range[1] - pc.u_range[0])
        v = pc.v_range[1] - y / S * (pc.v_range[1] - pc.v_range[0])
        return pc.origin + u * pc.right + v * pc.up

    def _press(self, e):
        if self._pc is None:
            return
        h = self._handle
        if h and (e.x - h[0]) ** 2 + (e.y - h[1]) ** 2 < 196:
            self._drag = ("rot", math.atan2(e.y - h[3], e.x - h[2]))
            return
        best, bd = None, 14.0
        for nm, (x0, y0, x1, y1) in getattr(self, "_segs", {}).items():
            dx, dy = x1 - x0, y1 - y0
            t = max(0.0, min(1.0, ((e.x - x0) * dx + (e.y - y0) * dy) / max(dx * dx + dy * dy, 1e-9)))
            d = math.hypot(e.x - (x0 + t * dx), e.y - (y0 + t * dy))
            if d < bd:
                best, bd = nm, d
        if best:
            self._select(best)
            self._drag = ("move", best, self._world(e.x, e.y),
                          float(self.ajustes.get(best, {}).get("mover", 0)))

    def _motion(self, e):
        import cubo
        if not self._drag:
            return
        if self._drag[0] == "rot":
            h = self._handle
            if not h:
                return
            ang = math.atan2(e.y - h[3], e.x - h[2])
            d = math.degrees(ang - self._drag[1])
            self._drag = ("rot", ang)
            cur = float(self.ajustes.get(self.sel, {}).get("giro", 0))
            self._set_val("giro", max(-45.0, min(45.0, cur - d)))
            return
        _k, nm, p0, mv0 = self._drag
        cam = cubo.face_cams(self._use(), self._cube_tipo(), self.norte)[nm][0]
        delta = float((self._world(e.x, e.y) - p0) @ cam.normal) / max(self._span(), 1e-9) * 100
        self._set_val("mover", max(-15.0, min(15.0, mv0 + delta)))

    def _release(self, _e):
        if self._drag:
            self._drag = None
            self._changed()

    # ------------------------------------------------------------ controles de la cara
    def _select(self, k):
        self.sel = k
        self.sel_lbl.config(text=k)
        self.sel_bar.config(bg=self.COLORS[k])
        adj = self.ajustes.get(k, {})
        for key, (v, lab, unit) in self.sv.items():
            val = float(adj.get(key, 0))
            v.set(val)
            lab.config(text=f"{val:+.1f}{unit}")
        tip = self._eff_tipo()
        nm = k
        if tip == "exterior":
            nm = "Cubierta" if k == "Techo" else ("Elevación " + k if k in ("Norte", "Este", "Sur", "Oeste") else k)
        elif tip == "interior" and k in ("Norte", "Este", "Sur", "Oeste"):
            nm = "Pared " + k
        self.face_title.config(text=nm)
        for kk, (cell, _l) in self._strip_cells.items():
            cell.configure(relief="solid" if kk == k else "flat")
        self._changed()

    def _set_val(self, key, val):
        v, lab, unit = self.sv[key]
        val = round(float(val), 1)
        v.set(val)
        lab.config(text=f"{val:+.1f}{unit}")
        adj = self.ajustes.setdefault(self.sel, {})
        adj[key] = val
        if not any(abs(float(x)) > 1e-9 for x in adj.values()):
            self.ajustes.pop(self.sel, None)
        self._changed()

    def _slid(self, key):
        self._sliding = True
        if self._slide_after:
            self.after_cancel(self._slide_after)
        self._slide_after = self.after(350, self._slide_end)
        self._set_val(key, float(self.sv[key][0].get()))

    def _slide_end(self):
        self._sliding = False
        self._changed()          # una pasada final a resolución completa

    def _reset_face(self):
        self.ajustes.pop(self.sel, None)
        self._select(self.sel)

    def _rot_north(self):
        self.norte = (self.norte + 1) % 4
        self._select(self.sel)
        self._changed(plan=True)

    def _tipo_changed(self):
        self._select(self.sel)
        self._changed()

    # ------------------------------------------------------------ avanzado (toda la caja)
    def _toggle_adv(self):
        if self.adv_on.get():
            self.adv.pack(fill="x", after=self.adv.master.winfo_children()[-3])
            if self.an is not None and not self.manual:
                self.cur = dict(self.an)
        else:
            self.adv.pack_forget()
        self._changed(plan=True)

    def _gslid(self, key):
        import cubo
        if self.an is None:
            return
        v = self.gv[key]
        val = round(float(v[0].get()) * 2) / 2
        d = val - v[1]
        if not d:
            return
        v[1] = val
        self.cur = cubo.rotate_frame(self.cur, yaw_deg=d if key == "yaw" else 0,
                                     tilt_a=d if key == "ta" else 0, tilt_b=d if key == "tb" else 0)
        self._changed(plan=key != "yaw")

    def _set_up(self):
        import cubo
        axes = {"+X": [1, 0, 0], "-X": [-1, 0, 0], "+Y": [0, 1, 0], "-Y": [0, -1, 0],
                "+Z": [0, 0, 1], "-Z": [0, 0, -1]}
        up = axes.get(self.up_choice.get())
        self.status.config(text="Recalculando…")

        def work():
            an = cubo.analyze(self.model, self.cams, self.ups, up=up)
            self.after(0, lambda: self._apply_frame(an))
        threading.Thread(target=work, daemon=True).start()

    def _refit(self):
        import cubo
        up, d0 = list(self.cur["up"]), list(self.cur["d0"])
        self.status.config(text="Re-detectando paredes…")

        def work():
            an = cubo.analyze(self.model, self.cams, self.ups, up=up, d0=d0)
            self.after(0, lambda: self._apply_frame(an))
        threading.Thread(target=work, daemon=True).start()

    def _apply_frame(self, an):
        self.cur = an
        self._ref_d0 = np.array(an["d0"], float)
        for v in self.gv.values():
            v[0].set(0.0)
            v[1] = 0.0
        self.status.config(text="Listo.")
        self._changed(plan=True)

    def _height(self, idx, sign):
        box = [list(x) for x in self.cur["caja"]]
        step = 0.02 * (box[2][1] - box[2][0] or 1.0)
        box[2][idx] += sign * step
        if box[2][1] - box[2][0] > step:
            self.cur = dict(self.cur, caja=box)
            self._changed()

    def _reset_all(self):
        self.ajustes.clear()
        self.cur = dict(self.an)
        self._ref_d0 = np.array(self.an["d0"], float)
        self.up_choice.set("Auto")
        for v in self.gv.values():
            v[0].set(0.0)
            v[1] = 0.0
        self.adv_on.set(False)
        self.adv.pack_forget()
        self._select(self.sel)
        self._changed(plan=True)

    # ------------------------------------------------------------ guardar y renderizar
    def _faces_dir(self):
        import cubo
        return cubo.faces_dir(self.results, self.name)

    def _refresh_res(self):
        d = self._faces_dir()
        n = len([f for f in os.listdir(d) if f.lower().endswith(".png")]) if os.path.isdir(d) else 0
        self.res_lbl.config(text=(f"Guardado: {n} caras en\n{d}" if n else
                                  f"Se guardará en:\n{d}"))

    def _open_dir(self):
        d = self._faces_dir()
        open_path(d if os.path.isdir(d) else self.results)

    def _conf(self):
        caras = [k for k, v in self.incl.items() if v.get()]
        span = self._span()
        c = {"tipo": self.tipo.get(), "norte": self.norte}
        if len(caras) < 6:
            c["caras"] = caras
        if self.ajustes:
            c["ajustes"] = self._adj_for_render()
            c["ajustes_ui"] = {k: dict(v) for k, v in self.ajustes.items()}
            c["span"] = span
        if self.adv_on.get() and self.cur:
            c["manual"] = {k: self.cur[k] for k in ("up", "d0", "d1", "caja")}
        if c["tipo"] == "auto" and self.an is not None:
            c["tipo"] = self.an["tipo_auto"]          # lo detectado en el visor queda fijo
        if c["tipo"] in ("auto", "fachada") and (self.ajustes or self.adv_on.get()):
            c["tipo"] = self._cube_tipo()
        return c

    def _render_final(self):
        import cubo
        conf = self._conf()
        if self.wd:
            cubo.save_config(self.wd, conf)
        man = pl.read_manifest(self.results)
        e = man.get(self.name) or {"ply": self.ply, "flip": False, "rot": 0}
        e["cubo"] = conf
        man[self.name] = e
        try:
            pl.write_manifest(self.results, man)
        except OSError:
            pass
        self.btn_go.state(["disabled"])
        cube = conf["tipo"] in ("interior", "exterior") or (conf["tipo"] == "auto" and
                                                          self.an["tipo_auto"] != "fachada")
        self.prog.configure(mode="determinate" if cube else "indeterminate", value=0)
        if not cube:
            self.prog.start(12)
        self.status.config(text="Renderizando en alta…")
        out_png = os.path.join(self.results, self.name + ".png")
        cfg = dict(self.app.cfg)
        cfg["_cam_store"] = self.app._cam_store()

        def prog(i, n, k):
            self.after(0, lambda: (self.prog.configure(value=100.0 * i / max(n, 1)),
                                   self.status.config(text=f"Renderizando {k} ({i + 1}/{n})…")))

        def work():
            try:
                for old in (out_png, os.path.join(self.results, "x_" + self.name + ".png")):
                    if os.path.exists(old):
                        os.remove(old)
                info = pl.render_auto(self.ply, out_png, self.name, cfg, wd=self.wd, cams=self.cams,
                                      ups=self.ups, conf=conf, flip=bool(e.get("flip")),
                                      rot90=int(e.get("rot", 0)), progress=prog, model=self.model)
                msg = f"✓ Listo ({info.get('modo')})"
            except Exception as ex:
                msg = f"Error: {ex}"
            self.after(0, lambda: self._done(msg))
        threading.Thread(target=work, daemon=True).start()

    def _done(self, msg):
        self.prog.stop()
        self.prog.configure(mode="determinate", value=100 if msg.startswith("✓") else 0)
        self.status.config(text=msg)
        self.btn_go.state(["!disabled"])
        self._refresh_res()
        self.app.q.put(("log", f"{self.name}: {msg}"))
        self.app._thumbs.clear()
        self.app.load_review()


class AmbienteViewer(tk.Toplevel):
    """Ajustar un ambiente recortado por partes (CARA…, LADO…, PLANTA, TECHO): cada parte es una
    cara; se ve en grande y en vivo, y se corrige sola sin tocar las demás."""
    SLIDERS = CubeViewer.SLIDERS

    def __init__(self, app, key, parts, results_dir):
        super().__init__(app)
        self.app, self.key, self.parts, self.results = app, key, parts, results_dir
        self.title(f"Ambiente por partes — {key}")
        try:
            self.state("zoomed")
        except tk.TclError:
            self.geometry(f"{self.winfo_screenwidth()}x{self.winfo_screenheight() - 60}+0+0")
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.FW, self.FH = int(max(500, sw - 460)), int(max(320, sh * 0.62))
        man = pl.read_manifest(results_dir)
        self.ajustes = {n: dict(man.get(n, {}).get("ajuste") or {}) for _p, n, _r in parts}
        self.ajustes = {k: v for k, v in self.ajustes.items() if v}
        self.names = [n for _p, n, _r in parts]
        self.roles = {n: r for _p, n, r in parts}
        self.sel = self.names[0]
        self.st = self.pm = None
        self._busy = self._dirty = self._sliding = False
        self._slide_after = self._thumb_after = None
        self._gen = 0
        self._build()
        threading.Thread(target=self._load, daemon=True).start()

    def _build(self):
        top = ttk.Frame(self, padding=(10, 8))
        top.pack(fill="x")
        ttk.Label(top, text=f"Ambiente «{self.key}» · {len(self.parts)} partes",
                  font=("Segoe UI", 13, "bold")).pack(side="left")
        self.btn_go = ttk.Button(top, text="✓ Guardar y renderizar", command=self._render_final,
                                 state="disabled")
        self.btn_go.pack(side="right")
        ttk.Button(top, text="📂 Abrir carpeta", command=lambda: open_path(self.results)).pack(side="right", padx=6)
        bar = ttk.Frame(self, padding=(10, 0))
        bar.pack(fill="x")
        self.prog = ttk.Progressbar(bar, mode="indeterminate", maximum=100)
        self.prog.pack(side="left", fill="x", expand=True)
        self.prog.start(12)
        self.status = ttk.Label(bar, text="Preparando el ambiente (orientando las partes)…", width=60)
        self.status.pack(side="left", padx=8)
        body = ttk.Frame(self, padding=10)
        body.pack(fill="both", expand=True)
        mid = ttk.Frame(body)
        mid.pack(side="left", fill="both", expand=True)
        self.face_title = ttk.Label(mid, text="", font=("Segoe UI", 12, "bold"))
        self.face_title.pack(anchor="w")
        self.face_cv = tk.Canvas(mid, width=self.FW, height=self.FH, bg="#222", highlightthickness=0)
        self.face_cv.pack(fill="both", expand=True)
        strip = ttk.Frame(mid)
        strip.pack(fill="x", pady=(6, 0))
        self.cells = {}
        for n in self.names:
            cell = tk.Frame(strip, bd=3, relief="flat")
            cell.pack(side="left", padx=3)
            lab = ttk.Label(cell, text=n[-22:], font=("Segoe UI", 8, "bold"), cursor="hand2")
            lab.pack()
            img = tk.Label(cell, bg="#333", width=18, height=6, cursor="hand2")
            img.pack()
            for w in (lab, img):
                w.bind("<Button-1>", lambda e, n=n: self._select(n))
            self.cells[n] = (cell, img)
        right = ttk.Frame(body, width=380)
        right.pack(side="left", fill="y", padx=(10, 0))
        self.sel_lbl = ttk.Label(right, text="", font=("Segoe UI", 14, "bold"), wraplength=360)
        self.sel_lbl.pack(anchor="w")
        ttk.Label(right, text="Ajusta solo esta parte. Clic en una miniatura para cambiar.",
                  wraplength=360, foreground="#666").pack(anchor="w")
        self.sv = {}
        for key, lab, lo, hi, unit in self.SLIDERS:
            row = ttk.Frame(right)
            row.pack(fill="x", pady=(10, 0))
            head = ttk.Frame(row)
            head.pack(fill="x")
            ttk.Label(head, text=lab, font=("Segoe UI", 10, "bold")).pack(side="left")
            val = ttk.Label(head, text="0", width=8, anchor="e")
            val.pack(side="right")
            v = tk.DoubleVar(value=0.0)
            sc = ttk.Scale(row, from_=lo, to=hi, variable=v, length=360, command=lambda _x, k=key: self._slid(k))
            sc.pack(fill="x")
            sc.bind("<Double-Button-1>", lambda e, k=key: self._set_val(k, 0.0))
            self.sv[key] = (v, val, unit)
        ttk.Button(right, text="↺ Reiniciar esta parte", command=self._reset).pack(anchor="w", pady=(12, 0))
        ttk.Label(right, text=f"\nSe guarda en:\n{self.results}", wraplength=360,
                  foreground="#1f5fa8").pack(anchor="w")

    def _load(self):
        import cubo
        try:
            self.st = cubo.ambiente_cams(self.parts)
            self.pm = {n: cubo.preview_model(m, 300_000) for n, m in self.st["models"].items()}
        except Exception as e:
            err = f"Error: {e}"
            self.after(0, lambda: self.status.config(text=err))
            return
        self.after(0, self._loaded)

    def _loaded(self):
        self.prog.stop()
        self.prog.configure(mode="determinate", value=0)
        self.status.config(text="Listo. Mueve los controles: la vista se actualiza en vivo.")
        self.btn_go.state(["!disabled"])
        self._select(self.sel)
        self._thumbs()

    def _cam(self, n, adj):
        import cubo
        cam, clip = self.st["cams"][n]
        return cubo.adjust_cam(cam, cubo.ui_to_adj(adj, cam)), clip

    def _changed(self):
        self._dirty = True
        if not self._busy:
            self._kick()
        if self._thumb_after:
            self.after_cancel(self._thumb_after)
        self._thumb_after = self.after(600, self._thumbs)

    def _kick(self):
        if self.st is None or not self._dirty:
            return
        self._dirty, self._busy = False, True
        self._gen += 1
        gen, n = self._gen, self.sel
        cam, clip = self._cam(n, self.ajustes.get(n))
        cap = 700 if self._sliding else 1600
        fw, fh = self.FW, self.FH

        def work():
            import render_ortho as ro
            du, dv = cam.u_range[1] - cam.u_range[0], cam.v_range[1] - cam.v_range[0]
            ext = max(du, dv)
            ls = int(max(fw * ext / du if du / dv > fw / fh else fh * ext / dv, 64))
            im = ro.render(self.pm[n], cam, min(ls, cap), clip / max(ext, 1e-9), background=(34, 34, 34))
            self.after(0, lambda: self._done_live(gen, im))
        threading.Thread(target=work, daemon=True).start()

    def _done_live(self, gen, im):
        from PIL import ImageTk
        self._busy = False
        if gen == self._gen:
            cv = self.face_cv
            cv.delete("all")
            W, H = int(cv.winfo_width() or self.FW), int(cv.winfo_height() or self.FH)
            im = im.copy()
            im.thumbnail((W, H))
            self._photo = ImageTk.PhotoImage(im)
            x0, y0 = (W - im.size[0]) // 2, (H - im.size[1]) // 2
            cv.create_image(x0, y0, image=self._photo, anchor="nw")
            for i in range(1, 12):
                x = x0 + im.size[0] * i / 12
                cv.create_line(x, y0, x, y0 + im.size[1], fill="#9e9e9e", stipple="gray25")
            for i in range(1, 8):
                y = y0 + im.size[1] * i / 8
                cv.create_line(x0, y, x0 + im.size[0], y, fill="#9e9e9e", stipple="gray25")
        if self._dirty:
            self._kick()

    def _thumbs(self):
        if self.st is None:
            return
        cams = {n: self._cam(n, self.ajustes.get(n)) for n in self.names}

        def work():
            import render_ortho as ro
            out = {}
            for n, (cam, clip) in cams.items():
                ext = max(cam.u_range[1] - cam.u_range[0], cam.v_range[1] - cam.v_range[0])
                out[n] = ro.render(self.pm[n], cam, 150, clip / max(ext, 1e-9), background=(34, 34, 34))
            self.after(0, lambda: self._thumbs_show(out))
        threading.Thread(target=work, daemon=True).start()

    def _thumbs_show(self, imgs):
        from PIL import ImageTk
        self._tph = {}
        for n, (cell, lab) in self.cells.items():
            cell.configure(relief="solid" if n == self.sel else "flat")
            im = imgs.get(n)
            if im is None:
                continue
            im = im.copy()
            im.thumbnail((150, 100))
            self._tph[n] = ImageTk.PhotoImage(im)
            lab.configure(image=self._tph[n], width=150, height=100)

    def _select(self, n):
        self.sel = n
        rol = {"pared": "pared", "suelo": "planta / piso", "techo": "techo"}[self.roles[n]]
        self.sel_lbl.config(text=n)
        self.face_title.config(text=f"{n}  ({rol}, vista desde dentro del ambiente)")
        adj = self.ajustes.get(n, {})
        for key, (v, lab, unit) in self.sv.items():
            val = float(adj.get(key, 0))
            v.set(val)
            lab.config(text=f"{val:+.1f}{unit}")
        for k, (cell, _l) in self.cells.items():
            cell.configure(relief="solid" if k == n else "flat")
        self._changed()

    def _set_val(self, key, val):
        v, lab, unit = self.sv[key]
        val = round(float(val), 1)
        v.set(val)
        lab.config(text=f"{val:+.1f}{unit}")
        adj = self.ajustes.setdefault(self.sel, {})
        adj[key] = val
        if not any(abs(float(x)) > 1e-9 for x in adj.values()):
            self.ajustes.pop(self.sel, None)
        self._changed()

    def _slid(self, key):
        self._sliding = True
        if self._slide_after:
            self.after_cancel(self._slide_after)
        self._slide_after = self.after(350, self._slide_end)
        self._set_val(key, float(self.sv[key][0].get()))

    def _slide_end(self):
        self._sliding = False
        self._changed()

    def _reset(self):
        self.ajustes.pop(self.sel, None)
        self._select(self.sel)

    def _render_final(self):
        import cubo
        man = pl.read_manifest(self.results)
        rots = {}
        for p_, n, r in self.parts:
            e = man.get(n) or {"ply": p_, "flip": False, "rot": 0}
            e.update(ambiente=self.key, rol=r)
            if self.ajustes.get(n):
                e["ajuste"] = self.ajustes[n]
            else:
                e.pop("ajuste", None)
            man[n] = e
            rots[n] = int(e.get("rot", 0))
        pl.write_manifest(self.results, man)
        self.btn_go.state(["disabled"])
        cfg = self.app.cfg

        def prog(i, n, k):
            self.after(0, lambda: (self.prog.configure(value=100.0 * i / max(n, 1)),
                                   self.status.config(text=f"Renderizando {k}…")))

        def work():
            try:
                cubo.render_ambiente(self.parts, self.results, long_side=int(cfg["render_lado_largo_px"]),
                                     background=tuple(cfg.get("render_fondo_color", (0, 0, 0))),
                                     rots=rots, ajustes=self.ajustes, setup=self.st,
                                     progress=prog, log=lambda *_: None)
                msg = f"✓ Listo: {len(self.parts)} caras guardadas"
            except Exception as ex:
                msg = f"Error: {ex}"
            self.after(0, lambda: self._final_done(msg))
        threading.Thread(target=work, daemon=True).start()

    def _final_done(self, msg):
        self.prog.configure(value=100 if msg.startswith("✓") else 0)
        self.status.config(text=msg)
        self.btn_go.state(["!disabled"])
        self.app.q.put(("log", f"Ambiente {self.key}: {msg}"))
        self.app._thumbs.clear()
        self.app._show_current()


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
    if not FROZEN and not os.environ.get("ORTO_LAUNCHER") and os.path.isfile(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "launcher.py")):
        # abierto directamente (acceso directo antiguo): pasar por el lanzador para actualizarse
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import launcher
        launcher.main()
    else:
        main()
