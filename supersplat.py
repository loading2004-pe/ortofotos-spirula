"""SuperSplat (editor de Gaussian splats de PlayCanvas, licencia MIT) dentro de la app.

Se descarga una sola vez (≈4 MB) a <datos>/supersplat/<versión>, se sirve desde esta PC
(http://127.0.0.1:<puerto>, sin internet ni la página web) y se abre en una ventana propia de
Edge/Chrome en modo aplicación (sin barra de direcciones). Los .ply se le pasan por la misma
dirección local, así que se abren directamente en el editor.
"""
import hashlib
import http.server
import os
import secrets
import shutil
import socketserver
import ssl
import subprocess
import sys
import threading
import urllib.parse
import urllib.request
import zipfile

SS_VERSION = "3.9.2"
SS_ZIP = f"supersplat-{SS_VERSION}.zip"
SS_SHA256 = "31696d423669f768be5e3efb8a4582b06ef9004610aaf2bebb6070538c094984"
SS_URLS = [
    f"https://raw.githubusercontent.com/loading2004-pe/ortofotos-spirula/main/extras/{SS_ZIP}",
    f"https://github.com/loading2004-pe/ortofotos-spirula/raw/main/extras/{SS_ZIP}",
]

MIME = {".html": "text/html; charset=utf-8", ".js": "text/javascript", ".mjs": "text/javascript",
        ".css": "text/css", ".json": "application/json", ".wasm": "application/wasm",
        ".png": "image/png", ".jpg": "image/jpeg", ".svg": "image/svg+xml",
        ".ply": "application/octet-stream", ".splat": "application/octet-stream",
        ".sog": "application/octet-stream", ".spz": "application/octet-stream"}


def install_dir(data_dir):
    return os.path.join(data_dir, "supersplat", SS_VERSION)


def is_installed(data_dir):
    return os.path.isfile(os.path.join(install_dir(data_dir), "index.html"))


def install(data_dir, log=lambda *_: None, zip_path=None):
    """Descarga y descomprime SuperSplat (solo la primera vez). Devuelve la carpeta."""
    dest = install_dir(data_dir)
    if is_installed(data_dir):
        return dest
    data, err = None, None
    if zip_path:
        with open(zip_path, "rb") as f:
            data = f.read()
    else:
        for url in SS_URLS:
            try:
                log(f"Descargando SuperSplat {SS_VERSION} (≈4 MB, solo esta vez)…")
                req = urllib.request.Request(url, headers={"User-Agent": "OrtofotosSpirula"})
                with urllib.request.urlopen(req, timeout=60, context=ssl.create_default_context()) as r:
                    data = r.read()
                if hashlib.sha256(data).hexdigest() == SS_SHA256:
                    break
                err, data = ValueError("la descarga de SuperSplat no coincide (sha256)"), None
            except Exception as e:      # siguiente dirección
                err, data = e, None
    if data is None:
        raise RuntimeError(f"No se pudo descargar SuperSplat: {err}")
    if hashlib.sha256(data).hexdigest() != SS_SHA256:
        raise ValueError("el archivo de SuperSplat no coincide (sha256)")
    tmp = dest + "_nuevo"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    zpath = os.path.join(tmp, "_ss.zip")
    with open(zpath, "wb") as f:
        f.write(data)
    with zipfile.ZipFile(zpath) as z:
        for m in z.namelist():
            p = os.path.normpath(m)
            if p.startswith("..") or os.path.isabs(p):
                raise ValueError("zip de SuperSplat no válido")
        z.extractall(tmp)
    os.remove(zpath)
    shutil.rmtree(dest, ignore_errors=True)
    os.replace(tmp, dest)
    log("SuperSplat instalado.")
    return dest


class _Server:
    def __init__(self, root):
        self.root = root
        self.files = {}         # token -> ruta del .ply
        srv = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send_file(self, path, head=False):
                try:
                    size = os.path.getsize(path)
                except OSError:
                    self.send_error(404)
                    return
                ext = os.path.splitext(path)[1].lower()
                self.send_response(200)
                self.send_header("Content-Type", MIME.get(ext, "application/octet-stream"))
                self.send_header("Content-Length", str(size))
                self.send_header("Cache-Control", "no-store")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                if head:
                    return
                with open(path, "rb") as f:
                    while True:
                        b = f.read(1 << 20)
                        if not b:
                            break
                        try:
                            self.wfile.write(b)
                        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                            return

            def _route(self, head=False):
                u = urllib.parse.urlparse(self.path)
                parts = [urllib.parse.unquote(x) for x in u.path.split("/") if x]
                if len(parts) >= 2 and parts[0] == "archivo":
                    p = srv.files.get(parts[1])
                    if not p:
                        self.send_error(404)
                        return
                    self._send_file(p, head)
                    return
                rel = "/".join(parts) or "index.html"
                full = os.path.normpath(os.path.join(srv.root, rel))
                if not full.startswith(os.path.normpath(srv.root)) or not os.path.isfile(full):
                    self.send_error(404)
                    return
                self._send_file(full, head)

            def do_GET(self):
                self._route()

            def do_HEAD(self):
                self._route(head=True)

        class TS(socketserver.ThreadingMixIn, http.server.HTTPServer):
            daemon_threads = True
            allow_reuse_address = True

        self.httpd = TS(("127.0.0.1", 0), H)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def url_for(self, ply):
        tok = hashlib.sha1(os.path.abspath(ply).lower().encode("utf-8")).hexdigest()[:12] + \
            secrets.token_hex(2)
        self.files[tok] = os.path.abspath(ply)
        return f"http://127.0.0.1:{self.port}/archivo/{tok}/{urllib.parse.quote(os.path.basename(ply))}"


_server = None
_lock = threading.Lock()


def server(data_dir):
    global _server
    with _lock:
        if _server is None:
            _server = _Server(install_dir(data_dir))
        return _server


def find_browser():
    """Edge (viene con Windows) o Chrome, para abrir SuperSplat como ventana propia."""
    cands = []
    if os.name == "nt":
        for base in (os.environ.get("PROGRAMFILES(X86)"), os.environ.get("PROGRAMFILES"),
                     os.environ.get("LOCALAPPDATA")):
            if base:
                cands += [os.path.join(base, "Microsoft", "Edge", "Application", "msedge.exe"),
                          os.path.join(base, "Google", "Chrome", "Application", "chrome.exe")]
    else:
        cands += [shutil.which(n) or "" for n in ("microsoft-edge", "google-chrome", "chromium",
                                                  "chromium-browser")]
    for c in cands:
        if c and os.path.isfile(c):
            return c
    return None


def editor_url(data_dir, plys=()):
    s = server(data_dir)
    q = []
    for p in plys:
        q.append(("load", s.url_for(p)))
        q.append(("filename", os.path.basename(p)))
    return f"http://127.0.0.1:{s.port}/index.html" + ("?" + urllib.parse.urlencode(q) if q else "")


def open_editor(data_dir, plys=(), log=lambda *_: None):
    """Abre SuperSplat (instalándolo si hace falta) con los .ply indicados."""
    install(data_dir, log)
    url = editor_url(data_dir, plys)
    br = find_browser()
    if br:
        prof = os.path.join(data_dir, "supersplat", "perfil_ventana")
        os.makedirs(prof, exist_ok=True)
        cmd = [br, f"--app={url}", f"--user-data-dir={prof}", "--no-first-run",
               "--no-default-browser-check", "--lang=es", "--window-size=1500,950",
               "--disable-features=Translate"]
        flags = 0x08000000 if os.name == "nt" else 0       # sin consola
        env = {k: v for k, v in os.environ.items() if not k.startswith(("_PYI", "_MEI"))}
        subprocess.Popen(cmd, close_fds=True, creationflags=flags, env=env)
    else:
        import webbrowser
        webbrowser.open(url)
    return url


if __name__ == "__main__":          # prueba: python supersplat.py <datos> [archivo.ply …]
    print(open_editor(sys.argv[1], sys.argv[2:], log=print))
