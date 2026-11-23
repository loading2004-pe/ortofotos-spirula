"""Publica una actualización del código (la reciben todas las PC al abrir la app).

    python publicar.py <carpeta_del_repositorio> [--notas "qué cambió"]

Sube la versión (fecha.N), escribe app_version.py y version.json (con sha256 de cada
archivo), copia todo a la carpeta del repositorio y, si es un repositorio git, hace
commit y push. La carpeta también puede ser una carpeta compartida (OneDrive, red).
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = ["app.py", "pipeline.py", "render_ortho.py", "ply_convert.py", "cli.py", "updater.py",
         "app_version.py", "cubo.py"]


def _k(v):
    return tuple(int(p) if p.isdigit() else 0 for p in v.split("."))


def next_version(cur):
    """fecha.N, siempre mayor que la actual."""
    cand = time.strftime("%Y.%m.%d") + ".1"
    if _k(cand) > _k(cur):
        return cand
    head, n = cur.rsplit(".", 1)
    return f"{head}.{int(n) + 1}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("destino")
    ap.add_argument("--notas", default="")
    ap.add_argument("--min-lanzador", type=int, default=1)
    a = ap.parse_args()
    ns = {}
    exec(open(os.path.join(HERE, "app_version.py"), encoding="utf-8").read(), ns)
    ver = next_version(ns.get("VERSION", "0"))
    with open(os.path.join(HERE, "app_version.py"), "w", encoding="utf-8") as f:
        f.write(f'VERSION = "{ver}"\n')
    man = {"version": ver, "fecha": time.strftime("%Y-%m-%d %H:%M"), "notas": a.notas,
           "min_lanzador": a.min_lanzador, "archivos": {}}
    os.makedirs(a.destino, exist_ok=True)
    for fn in FILES:
        data = open(os.path.join(HERE, fn), "rb").read()
        man["archivos"][fn] = hashlib.sha256(data).hexdigest()
        if os.path.abspath(a.destino) != HERE:
            shutil.copy2(os.path.join(HERE, fn), os.path.join(a.destino, fn))
    with open(os.path.join(a.destino, "version.json"), "w", encoding="utf-8") as f:
        json.dump(man, f, indent=1, ensure_ascii=False)
    print("Versión", ver)
    if os.path.isdir(os.path.join(a.destino, ".git")):
        for cmd in (["git", "add", "-A"], ["git", "commit", "-m", f"Versión {ver}: {a.notas}"],
                    ["git", "push"]):
            subprocess.run(cmd, cwd=a.destino, check=False)


if __name__ == "__main__":
    main()
