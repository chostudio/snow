"""The dashboard's HTTP surface: the bundle routes plus the live coach.

    snow serve                      # http://127.0.0.1:8000
    VITE_API_BASE=http://127.0.0.1:8000 npm run dev

Serves exactly what ``web/src/api.ts`` asks for, out of the bundles that
``web/tools/export_session.py`` wrote:

    GET  /index.json
    GET  /<name>/session.json
    GET  /<name>/clip.mp4          (byte ranges, so the video can seek)
    POST /<name>/coach             {"event_id": "air-1" | null} -> CoachReport

Standard library only. It binds to localhost by default: it holds no auth of
its own and every coach request spends the NVIDIA key's quota.
"""

from __future__ import annotations

import json
import mimetypes
import re
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import coach
from . import config as cfg

SESSIONS_DIR = cfg.REPO_ROOT / "web" / "public" / "sessions"
NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
FILES = {"session.json", "clip.mp4"}
CHUNK = 1 << 20


def _route(path: str) -> tuple[str, str] | None:
    """``/<name>/<file>`` with a safe name, else None."""
    parts = path.split("?", 1)[0].strip("/").split("/")
    if len(parts) != 2 or not NAME.match(parts[0]) or parts[0] in (".", ".."):
        return None
    return parts[0], parts[1]


def make_handler(sessions_dir: Path, coach_fn=coach.coach_session):
    class Handler(BaseHTTPRequestHandler):
        server_version = "snow-serve"

        def _cors(self) -> None:
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Range")

        def _json(self, status: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self._cors()
            self.end_headers()
            self.wfile.write(data)

        def do_OPTIONS(self) -> None:  # noqa: N802  (http.server naming)
            self.send_response(HTTPStatus.NO_CONTENT)
            self._cors()
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            if self.path.split("?", 1)[0] == "/index.json":
                return self._file(sessions_dir / "index.json")
            route = _route(self.path)
            if not route or route[1] not in FILES:
                return self._json(404, {"error": f"no route {self.path}"})
            self._file(sessions_dir / route[0] / route[1])

        def do_POST(self) -> None:  # noqa: N802
            route = _route(self.path)
            if not route or route[1] != "coach":
                return self._json(404, {"error": f"no route {self.path}"})
            name = route[0]
            path = sessions_dir / name / "session.json"
            if not path.exists():
                return self._json(404, {"error": f"no session {name!r}"})
            try:
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                event_id = body.get("event_id")
            except (ValueError, AttributeError):
                return self._json(400, {"error": "body must be JSON {\"event_id\": ...}"})
            try:
                report = coach_fn(json.loads(path.read_text()), event_id)
            except ValueError as exc:
                return self._json(400, {"error": str(exc)})
            except coach.CoachError as exc:
                # Loud on purpose: the UI shows this instead of a template.
                self.log_message("coach failed: %s", exc)
                return self._json(502, {"error": str(exc)})
            self._json(200, report)

        def _file(self, path: Path) -> None:
            if not path.is_file():
                return self._json(404, {"error": f"missing {path.name}"})
            size = path.stat().st_size
            start, end = 0, size - 1
            ranged = re.match(r"bytes=(\d*)-(\d*)$", self.headers.get("Range") or "")
            if ranged and (ranged.group(1) or ranged.group(2)):
                if ranged.group(1):
                    start = int(ranged.group(1))
                    end = min(int(ranged.group(2)), size - 1) if ranged.group(2) else size - 1
                else:  # suffix range: the last N bytes
                    start = max(0, size - int(ranged.group(2)))
                if start > end:
                    self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self._cors()
                    self.end_headers()
                    return
                self.send_response(HTTPStatus.PARTIAL_CONTENT)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            else:
                self.send_response(HTTPStatus.OK)
            kind = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Accept-Ranges", "bytes")
            self._cors()
            self.end_headers()
            with path.open("rb") as fh:
                fh.seek(start)
                left = end - start + 1
                while left > 0:
                    chunk = fh.read(min(CHUNK, left))
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                    except (BrokenPipeError, ConnectionResetError):
                        return  # the browser cancelled a range it no longer wants
                    left -= len(chunk)

    return Handler


def serve(host: str = "127.0.0.1", port: int = 8000, sessions_dir: Path = SESSIONS_DIR) -> None:
    if not (sessions_dir / "index.json").exists():
        sys.exit(f"no bundles in {sessions_dir}; run web/tools/export_session.py first")
    httpd = ThreadingHTTPServer((host, port), make_handler(sessions_dir))
    print(f"serving {sessions_dir} on http://{host}:{port}  (coach model {coach.model_id()})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
