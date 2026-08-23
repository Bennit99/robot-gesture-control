import http.server
import os
import platform
import socketserver
import threading

from dotenv import load_dotenv

load_dotenv()


PORT = int(os.getenv("FRONTEND_PORT", 8000))
FRONTEND_DIR = os.path.dirname(os.path.abspath(__file__))
frontend_started = threading.Event()


def _content_type_for(path: str) -> str:
    if path.endswith(".json"):
        return "application/manifest+json"
    if path.endswith(".css"):
        return "text/css"
    if path.endswith(".svg"):
        return "image/svg+xml"
    if path.endswith(".js"):
        return "application/javascript"
    if path.endswith(".png"):
        return "image/png"
    if path.endswith(".ico"):
        return "image/x-icon"
    if path.endswith(".html"):
        return "text/html"
    return "application/octet-stream"


class CustomHTTPRequestHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split("?", 1)[0]

        if path == "/logs":
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            from web_ui.streaming_web_server.backend import logs_page

            self.wfile.write(logs_page().encode("utf-8"))
            return

        if path == "/logs/stream":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            from web_ui.streaming_web_server.backend import log_interceptor

            q = log_interceptor.subscribe()
            try:
                while True:
                    chunk = q.get()
                    self.wfile.write(
                        chunk.encode("utf-8") if isinstance(chunk, str) else chunk
                    )
                    self.wfile.flush()
            except Exception:
                pass
            finally:
                log_interceptor.unsubscribe(q)
            return

        if path in ("/", "/gesture", "/gesture/"):
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            from web_ui.streaming_web_server.backend import active_host, active_port

            with open(
                os.path.join(FRONTEND_DIR, "index.html"), "r", encoding="utf-8"
            ) as file:
                html = file.read()
                html = html.replace("{{BACKEND_PORT}}", str(active_port))
                # If BACKEND_HOST env var is set, use it; otherwise use the active_host from Flask
                backend_host = os.getenv(
                    "BACKEND_HOST",
                    active_host if active_host != "0.0.0.0" else "localhost",
                )
                html = html.replace("{{BACKEND_HOST}}", backend_host)
            self.wfile.write(html.encode("utf-8"))
            return

        # Attempt to serve static files automatically
        safe_path = path.lstrip("/")
        # Prevent directory traversal
        if ".." in safe_path:
            self.send_response(403)
            self.end_headers()
            self.wfile.write(b"403 Forbidden")
            return

        file_path = os.path.join(FRONTEND_DIR, safe_path)
        if os.path.isfile(file_path):
            self.send_response(200)
            self.send_header("Content-type", _content_type_for(safe_path))
            self.end_headers()
            mode = "rb" if safe_path.endswith((".png", ".ico")) else "r"
            if mode == "rb":
                with open(file_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                with open(file_path, "r", encoding="utf-8") as f:
                    self.wfile.write(f.read().encode("utf-8"))
            return

        self.send_response(404)
        self.end_headers()
        self.wfile.write(b"404 Not Found")
        return


def start():
    with socketserver.TCPServer(("0.0.0.0", PORT), CustomHTTPRequestHandler) as httpd:
        import socket

        # On Jetson/Spot, the hostname-based IP can resolve to the wrong interface.
        # Use the known Spot-facing frontend URL there; otherwise keep the detected IP.
        machine = platform.machine().lower()
        frontend_host = os.getenv("FRONTEND_HOST")
        if not frontend_host:
            if machine in {"aarch64", "arm64"}:
                frontend_host = "192.168.10.3"
            else:
                hostname = socket.gethostname()
                frontend_host = socket.gethostbyname(hostname)

        print("\033[92m" + "=" * 60 + "\033[0m")
        print("\033[92;1m  [Frontend] SERVER STARTED SUCCESSFULLY\033[0m")
        if machine in {"aarch64", "arm64"} and not os.getenv("FRONTEND_HOST"):
            print(
                f"\033[92m  Access at (spotNet): http://192.168.10.3:{PORT}/gesture\033[0m"
            )
            print(
                f"\033[92m  Access at (BDNet):   http://192.168.80.3:{PORT}/gesture\033[0m"
            )
        else:
            print(f"\033[92m  Access at: http://{frontend_host}:{PORT}/gesture\033[0m")
        print("\033[92m" + "=" * 60 + "\033[0m")
        httpd.serve_forever()


def start_frontend():
    """Starts the frontend server in a separate thread."""
    if frontend_started.is_set():
        return
    thread = threading.Thread(target=start, daemon=True)
    thread.start()
    frontend_started.set()
    print("Frontend server started in a separate thread.")
