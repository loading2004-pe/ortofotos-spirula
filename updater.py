"""Actualizaciones automáticas de Ortofotos Spirula (como una tienda de apps).

El .exe trae una copia del código. Al abrirse, mira en internet si hay una versión
más nueva del código (archivos .py + version.json) y, si la hay, la descarga a
<datos>/codigo y la usa. Así las mejoras llegan a todas las PC sin repartir otro .exe.

Fuente de actualizaciones (config "url_actualizaciones"): una URL http(s) que termine
en / (p. ej. GitHub raw) o una carpeta (p. ej. OneDrive / red compartida).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import ssl
import time
import urllib.request

LAUNCHER_VERSION = 2          # sube solo si cambia el .exe (lanzador)
DEFAULT_URL = "https://raw.githubusercontent.com/loading2004-pe/ortofotos-spirula/main/"
CODE_DIRNAME = "codigo"


def vkey(v):
    """'2026.11.21.3' -> (2026, 11, 21, 3) para comparar versiones."""
    out = []
    for p in str(v or "0").replace("-", ".").split("."):
        try:
            out.append(int(p))
        except ValueError:
            out.append(0)
    return tuple(out)


def _read(src, name, timeout=6):
    if src.lower().startswith(("http://", "https://")):
        url = src if src.endswith("/") else src + "/"
        req = urllib.request.Request(url + urllib.request.quote(name) + f"?t={int(time.time())}",
                                     headers={"User-Agent": "OrtofotosSpirula", "Cache-Control": "no-cache"})
        ctx = ssl.create_default_context()
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
            return r.read()
    with open(os.path.join(src, name), "rb") as f:
        return f.read()


def local_version(code_dir):
    try:
        with open(os.path.join(code_dir, "version.json"), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def remote_manifest(src, timeout=6):
    return json.loads(_read(src, "version.json", timeout).decode("utf-8"))


def download(src, man, data_dir, log=lambda *_: None):
    """Descarga y verifica los archivos del manifiesto en <datos>/codigo (atómico)."""
    final = os.path.join(data_dir, CODE_DIRNAME)
    tmp = final + "_nuevo"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    for name, sha in man.get("archivos", {}).items():
        if "/" in name or "\\" in name or name.startswith("."):
            raise ValueError("nombre de archivo no permitido: " + name)
        data = _read(src, name, timeout=30)
        if hashlib.sha256(data).hexdigest() != sha:
            raise ValueError(f"{name}: la descarga no coincide (sha256)")
        with open(os.path.join(tmp, name), "wb") as f:
            f.write(data)
        log(f"  descargado {name}")
    with open(os.path.join(tmp, "version.json"), "w", encoding="utf-8") as f:
        json.dump(man, f, indent=1, ensure_ascii=False)
    prev = final + "_anterior"
    shutil.rmtree(prev, ignore_errors=True)
    if os.path.isdir(final):
        os.replace(final, prev)
    os.replace(tmp, final)
    return final


def available(data_dir, current_version, src=None, timeout=6):
    """¿Hay una versión más nueva? Devuelve (manifiesto | None, mensaje). No descarga nada."""
    src = src or DEFAULT_URL
    try:
        man = remote_manifest(src, timeout)
    except Exception as e:
        return None, f"sin conexión con el servidor de actualizaciones ({e.__class__.__name__})"
    rv = man.get("version", "0")
    loc = local_version(os.path.join(data_dir, CODE_DIRNAME)) or {}
    current = max(vkey(current_version), vkey(loc.get("version")) if loc else (0,))
    if vkey(rv) <= current:
        return None, f"ya tienes la última versión ({rv})"
    if rv in _bad_versions(data_dir):
        return None, f"la versión {rv} falló antes en esta PC; se omite"
    if int(man.get("min_lanzador", 1)) > LAUNCHER_VERSION:
        return None, f"la versión {rv} necesita un .exe nuevo"
    return man, f"versión {rv} disponible"


def apply(data_dir, man, src=None, log=lambda *_: None):
    """Descarga la versión del manifiesto. Devuelve (ok, mensaje)."""
    try:
        download(src or DEFAULT_URL, man, data_dir, log)
        return True, f"descargada la versión {man.get('version')}"
    except Exception as e:
        return False, f"no se pudo descargar: {e}"


def check(data_dir, bundled_version, src=None, timeout=6, log=lambda *_: None):
    """Busca y descarga una versión nueva. Devuelve (versión_descargada | None, mensaje)."""
    src = src or DEFAULT_URL
    try:
        man = remote_manifest(src, timeout)
    except Exception as e:
        return None, f"sin conexión con el servidor de actualizaciones ({e.__class__.__name__})"
    rv = man.get("version", "0")
    code_dir = os.path.join(data_dir, CODE_DIRNAME)
    loc = local_version(code_dir) or {}
    bad = _bad_versions(data_dir)
    current = max(vkey(bundled_version), vkey(loc.get("version")) if loc else (0,))
    if vkey(rv) <= current:
        return None, f"ya tienes la última versión ({rv})"
    if rv in bad:
        return None, f"la versión {rv} falló antes en esta PC; se omite"
    if int(man.get("min_lanzador", 1)) > LAUNCHER_VERSION:
        return None, (f"la versión {rv} necesita un .exe nuevo (descárgalo una vez); "
                      "sigues con la actual")
    try:
        download(src, man, data_dir, log)
    except Exception as e:
        return None, f"no se pudo descargar la versión {rv}: {e}"
    return rv, f"descargada la versión {rv}" + (f": {man['notas']}" if man.get("notas") else "")


def _bad_versions(data_dir):
    try:
        with open(os.path.join(data_dir, "versiones_malas.json"), encoding="utf-8") as f:
            return set(json.load(f))
    except (OSError, ValueError):
        return set()


def mark_bad(data_dir, version):
    bad = _bad_versions(data_dir)
    bad.add(version)
    with open(os.path.join(data_dir, "versiones_malas.json"), "w", encoding="utf-8") as f:
        json.dump(sorted(bad), f)
