"""Arranque de Ortofotos Spirula con actualización automática.

Es lo que ejecuta el .exe (y OrtofotosSpirula.bat). Busca una versión nueva del
código, la descarga si existe y abre la app con el código más reciente. Si el código
nuevo falla al abrir, vuelve solo a la versión anterior.
"""
from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import json
import os
import sys
import traceback

FROZEN = getattr(sys, "frozen", False)
HERE = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(sys.executable) if FROZEN else HERE
if not FROZEN and HERE not in sys.path:
    sys.path.insert(0, HERE)

MODULES = ("app", "pipeline", "render_ortho", "ply_convert", "cli", "updater", "app_version", "cubo")


def _writable(d):
    try:
        t = os.path.join(d, ".orto_test")
        open(t, "w").close()
        os.remove(t)
        return True
    except OSError:
        return False


DATA_DIR = APP_DIR if _writable(APP_DIR) else os.path.join(
    os.environ.get("APPDATA", os.path.expanduser("~")), "OrtofotosSpirula")
os.makedirs(DATA_DIR, exist_ok=True)
CONFIG_PATH = os.path.join(DATA_DIR, "ortofotos_config.json" if FROZEN else "config.json")


class _CodeFinder(importlib.abc.MetaPathFinder):
    """Carga nuestros módulos desde la carpeta del código descargado."""

    def __init__(self, folder):
        self.folder = folder

    def find_spec(self, name, path=None, target=None):
        if "." in name:
            return None
        f = os.path.join(self.folder, name + ".py")
        if os.path.isfile(f):
            return importlib.util.spec_from_file_location(name, f)
        return None


def _log(msg):
    try:
        with open(os.path.join(DATA_DIR, "actualizaciones.log"), "a", encoding="utf-8") as f:
            import time
            f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")
    except OSError:
        pass


def _ask_update(man):
    """Ventana encima de todo: 'Hay una actualización' [Actualizar] [Más tarde]."""
    try:
        import tkinter as tk
        from tkinter import ttk
    except Exception:
        return True
    res = {"v": False}
    w = tk.Tk()
    w.title("Ortofotos Spirula — actualización")
    w.attributes("-topmost", True)
    w.resizable(False, False)
    f = ttk.Frame(w, padding=18)
    f.pack()
    ttk.Label(f, text="Hay una actualización disponible", font=("Segoe UI", 13, "bold")).pack(anchor="w")
    ttk.Label(f, text=f"Versión {man.get('version')}", foreground="#1f5fa8").pack(anchor="w", pady=(4, 0))
    if man.get("notas"):
        ttk.Label(f, text=man["notas"], wraplength=380, justify="left").pack(anchor="w", pady=(6, 0))
    b = ttk.Frame(f)
    b.pack(fill="x", pady=(14, 0))

    def yes():
        res["v"] = True
        w.destroy()
    ttk.Button(b, text="Actualizar", command=yes).pack(side="right")
    ttk.Button(b, text="Más tarde", command=w.destroy).pack(side="right", padx=8)
    w.update_idletasks()
    x = (w.winfo_screenwidth() - w.winfo_width()) // 2
    y = (w.winfo_screenheight() - w.winfo_height()) // 3
    w.geometry(f"+{x}+{y}")
    w.after(200, lambda: (w.lift(), w.focus_force()))
    w.mainloop()
    return res["v"]


def _bundle_hint():  # pragma: no cover - nunca se ejecuta; solo para que PyInstaller lo incluya
    import app, pipeline, render_ortho, ply_convert, cli, updater, app_version, cubo  # noqa
    import zipfile, hashlib, ssl, urllib.request, urllib.parse, http.client, csv, uuid, glob  # noqa
    import struct, shutil, tempfile, gzip, base64, platform, webbrowser, socket, datetime  # noqa
    import statistics, random, itertools, functools, textwrap, decimal, fractions, math  # noqa
    import concurrent.futures, multiprocessing, ctypes, xml.etree.ElementTree, html, queue  # noqa
    import tkinter.simpledialog, tkinter.colorchooser, tkinter.scrolledtext, tkinter.font  # noqa
    import PIL.ImageDraw, PIL.ImageFilter, PIL.ImageOps, PIL.ImageFont, PIL.ImageGrab  # noqa
    import PIL.ImageEnhance, PIL.ImageChops, PIL.ImageStat, PIL.ImageTk  # noqa
    import numpy.linalg, numpy.fft, numpy.random  # noqa


def main():
    os.environ["ORTO_APP_DIR"] = APP_DIR
    os.environ["ORTO_LAUNCHER"] = sys.executable if FROZEN else os.path.abspath(__file__)
    import updater
    import app_version
    bundled = app_version.VERSION
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            cfg = json.load(f)
    except (OSError, ValueError):
        cfg = {}
    msg = ""
    if "--sin-actualizar" not in sys.argv and cfg.get("actualizar_auto", True):
        src = cfg.get("url_actualizaciones") or None
        man, msg = updater.available(DATA_DIR, bundled, src, timeout=4)
        _log(msg)
        if man and _ask_update(man):
            ok, msg = updater.apply(DATA_DIR, man, src)
            _log(msg)
        elif man:
            msg = f"versión {man.get('version')} disponible (pospuesta)"
    sys.argv = [a for a in sys.argv if a != "--sin-actualizar"]
    code_dir = os.path.join(DATA_DIR, updater.CODE_DIRNAME)
    loc = updater.local_version(code_dir)
    use_dl = bool(loc) and updater.vkey(loc.get("version")) > updater.vkey(bundled) \
        and loc.get("version") not in updater._bad_versions(DATA_DIR)
    finder = None
    if use_dl:
        for m in MODULES:
            sys.modules.pop(m, None)
        finder = _CodeFinder(code_dir)
        sys.meta_path.insert(0, finder)
        os.environ["ORTO_CODE_VERSION"] = loc.get("version")
    else:
        os.environ["ORTO_CODE_VERSION"] = bundled
    os.environ["ORTO_UPDATE_MSG"] = msg or ""
    try:
        app = importlib.import_module("app")
        app.main()
    except SystemExit:
        raise
    except Exception:
        err = traceback.format_exc()
        _log("ERROR al abrir:\n" + err)
        if not finder:
            raise
        # el código descargado falló: se marca y se abre la versión incluida en el .exe
        updater.mark_bad(DATA_DIR, loc.get("version"))
        sys.meta_path.remove(finder)
        for m in MODULES:
            sys.modules.pop(m, None)
        os.environ["ORTO_CODE_VERSION"] = bundled
        os.environ["ORTO_UPDATE_MSG"] = (f"La versión {loc.get('version')} falló al abrir; "
                                         f"se usa la {bundled}.")
        app = importlib.import_module("app")
        app.main()


if __name__ == "__main__":
    main()
