"""Procesamiento sin ventana (para pruebas o tareas programadas).

    python cli.py "C:\\ruta\\raiz" [--solo "p-34 cara a" ...] [--render-solo]
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(HERE, "_cache_numba"))

import pipeline as pl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raiz")
    ap.add_argument("--solo", nargs="*", default=None, help="nombres de trabajo a procesar")
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    ap.add_argument("--reentrenar", action="store_true", help="repite el entrenamiento")
    ap.add_argument("--limpios", action="store_true",
                    help="'raiz' es una carpeta de .ply limpios: solo renderiza (sin Spirula)")
    ap.add_argument("--set", nargs="*", default=[], help="clave=valor para la config")
    args = ap.parse_args()
    cfg = pl.load_config(args.config)
    for kv in args.set:
        k, v = kv.split("=", 1)
        cfg[k] = type(cfg[k])(v) if k in cfg and not isinstance(cfg[k], (list, bool)) else v
    if args.reentrenar:
        cfg["_forzar_entrenar"] = True
    if args.limpios:
        items = pl.list_plys(args.raiz)
        out = os.path.join(args.raiz, "ortofotos_limpias")
        print(f"{len(items)} .ply en {args.raiz} -> {out}", flush=True)
        t = time.time()
        pl.render_linked(items, out, cfg, store=os.path.join(HERE, "camaras"),
                         on_item=lambda n, st, d: print(f"[{time.strftime('%H:%M:%S')}] {n}: {st} {d}",
                                                        flush=True))
        print(f"TIEMPO TOTAL {(time.time() - t) / 60:.1f} min", flush=True)
        return
    rep = pl.scan([args.raiz], cfg)
    print(f"Detectados {len(rep.jobs)} trabajos, {len(rep.empty)} vacías, "
          f"{len(rep.insufficient)} con pocos archivos", flush=True)
    for j in rep.jobs:
        print(f"  {j.name:<20} {len(j.videos)} videos {len(j.images)} fotos  {j.folder}", flush=True)
    for d, nv, nf in rep.insufficient:
        print(f"  [omitido] {d} ({nv} videos, {nf} fotos)", flush=True)
    for d in rep.empty:
        print(f"  [vacía]   {d}", flush=True)
    jobs = rep.jobs
    if args.solo is not None:
        want = {s.lower() for s in args.solo}
        jobs = [j for j in jobs if j.name.lower() in want]
    last = {}

    def upd(j):
        key = (j.status, j.detail[:40])
        if last.get(j.name) != key:
            last[j.name] = key
            print(f"[{time.strftime('%H:%M:%S')}] {j.name}: {j.status}  {j.detail}", flush=True)

    t = time.time()
    pl.Runner(cfg, [args.raiz], on_update=upd, log=lambda m: print(m, flush=True)).run_all(jobs)
    print(f"TIEMPO TOTAL {(time.time() - t) / 60:.1f} min", flush=True)
    for j in jobs:
        print(f"RESULTADO {j.name}: {j.status} {j.detail} {j.times}", flush=True)


if __name__ == "__main__":
    main()
