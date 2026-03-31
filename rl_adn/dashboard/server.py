from __future__ import annotations

import json
import threading
import webbrowser
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources

from rl_adn.dashboard.state import DashboardStateStore

STATIC_PACKAGE = "rl_adn.dashboard.static"


class _DashboardRequestHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, store: DashboardStateStore, **kwargs):
        self._store = store
        super().__init__(*args, **kwargs)

    def do_GET(self):  # noqa: N802
        if self.path in {"/", "/index.html"}:
            self._serve_static("index.html", "text/html; charset=utf-8")
            return
        if self.path == "/styles.css":
            self._serve_static("styles.css", "text/css; charset=utf-8")
            return
        if self.path == "/app.js":
            self._serve_static("app.js", "application/javascript; charset=utf-8")
            return
        if self.path == "/api/state":
            payload = json.dumps(self._store.snapshot()).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if self.path == "/favicon.ico":
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
            return
        self.send_response(HTTPStatus.NOT_FOUND)
        self.end_headers()

    def log_message(self, format, *args):  # noqa: A003
        return

    def _serve_static(self, filename: str, content_type: str) -> None:
        content = (resources.files(STATIC_PACKAGE) / filename).read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)


class DashboardServer:
    def __init__(self, *, host: str = "127.0.0.1", port: int = 8787, history_limit: int = 500) -> None:
        self.store = DashboardStateStore(history_limit=history_limit)
        handler = partial(_DashboardRequestHandler, store=self.store)
        self._httpd = ThreadingHTTPServer((host, port), handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    @property
    def host(self) -> str:
        return str(self._httpd.server_address[0])

    @property
    def port(self) -> int:
        return int(self._httpd.server_address[1])

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    def start(self, *, open_browser: bool = False) -> "DashboardServer":
        self._thread.start()
        if open_browser:
            webbrowser.open(self.url)
        return self

    def close(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)


def launch_dashboard(*, port: int = 8787, open_browser: bool = True, history_limit: int = 500) -> DashboardServer:
    return DashboardServer(port=port, history_limit=history_limit).start(open_browser=open_browser)
