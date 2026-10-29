"""``snow label``: a local page for hand-labelling real airs and their tricks.

    snow label                      # http://127.0.0.1:8765, opens a browser
    snow label --fetch              # first copy the model weights from S3

Routes (standard library only, bound to localhost, no CORS: the page is
served from here and nothing else should be able to write the labels):

    GET  /                          the page (static/index.html)
    GET  /static/<file>             its script and style
    GET  /api/clips                 every labellable clip, with fps, frames, slowmo
    GET  /api/clip/<name>           one clip: info, saved rows, machine guesses
    GET  /api/poses/<name>          2D skeletons for the overlay
    GET  /api/guide                 docs/labelling.md as HTML
    GET  /video/<name>              the clip, with byte ranges so it can seek
    POST /api/save/<name>           {"rows": [...]} -> autosave, plus problems
    POST /api/reading/<name>        {"takeoff", "landing"} -> both trick readings

Labels go to one CSV in the format ``snow.learn.labels`` validates, so
``snow segmenter build --gold`` reads it directly.
"""

from __future__ import annotations

import fnmatch
import json
import mimetypes
import queue
import re
import subprocess
import sys
import threading
import webbrowser
from http import HTTPStatus
from http.server import ThreadingHTTPServer
from pathlib import Path

from .. import config as cfg
from .. import serve
from ..learn import labels as L
from . import guide, prefill

STATIC = Path(__file__).parent / "static"
GUIDE = cfg.REPO_ROOT / "docs" / "labelling.md"
DEFAULT_CSV = cfg.DATA_DIR / "labels" / "real_airs.csv"
NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
# Real broadcast clips only: the synthetic renders have exact labels already,
# and awstest-* are copies of run1 made to test the AWS box.
DEFAULT_SKIP = ("synth*", "awstest*")
# Files in data/clips that are views of a clip, not clips.
DERIVED = (".track", ".zoom", ".overlay", ".numbered", ".src", ".scout")
EDITABLE = ["air", "takeoff_frame", "landing_frame", "event", "sure", "labeller", "notes",
            *L.TRICK_COLUMNS]


class State:
    """Everything the handler reads: directories, the CSV, guesses being computed."""

    def __init__(self, clips_dir: Path, csv_path: Path, cache_dir: Path,
                 skip: tuple[str, ...] = DEFAULT_SKIP, labeller: str = "",
                 seg_weights: Path | None = None, clf_weights: Path | None = None,
                 guide_path: Path = GUIDE, compute: bool = True):
        self.clips_dir, self.csv_path, self.cache_dir = clips_dir, csv_path, cache_dir
        self.skip, self.labeller, self.guide_path = skip, labeller, guide_path
        self.seg_weights, self.clf_weights = seg_weights, clf_weights
        self.compute = compute
        self.write_lock = threading.Lock()
        self.prefills: dict[str, dict] = {}
        self.status: dict[str, str] = {}
        self._probe: dict[str, dict] = {}
        self._poses: dict[str, dict] = {}
        self._queue: queue.Queue[tuple[str, bool]] = queue.Queue()
        if compute:
            threading.Thread(target=self._worker, daemon=True).start()

    # -- clips ---------------------------------------------------------------

    def clip_names(self) -> list[str]:
        names = []
        for p in sorted(self.clips_dir.glob("*.mp4")):
            stem = p.stem
            if any(stem.endswith(d) or f"{d}." in stem for d in DERIVED):
                continue
            if any(fnmatch.fnmatch(stem, pat) for pat in self.skip):
                continue
            names.append(stem)
        return names

    def info(self, name: str) -> dict:
        """fps, frames, size and slow motion, from meta.json or else ffprobe."""
        meta_path = self.clips_dir / f"{name}.meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        probed = self.probe(name) if not all(meta.get(k) for k in ("fps", "frames")) else {}
        fps = float(meta.get("fps") or probed.get("fps") or 30.0)
        poses = cfg.POSES_DIR / f"{name}.json"
        return {
            "name": name,
            "fps": fps,
            "frames": int(meta.get("frames") or probed.get("frames") or 0),
            "width": int(meta.get("width") or probed.get("width") or 0),
            "height": int(meta.get("height") or probed.get("height") or 0),
            "slowmo_factor": meta.get("slowmo_factor"),
            "has_meta": bool(meta),
            "has_poses": poses.exists(),
            "pose_input": (meta.get("extra") or {}).get("pose_input", "full"),
            "source": meta.get("source", ""),
        }

    def probe(self, name: str) -> dict:
        if name not in self._probe:
            out = {}
            try:
                done = subprocess.run(
                    ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                     "-show_entries", "stream=width,height,r_frame_rate,nb_read_packets",
                     "-of", "json", str(self.clips_dir / f"{name}.mp4")],
                    capture_output=True, text=True, timeout=60)
                stream = json.loads(done.stdout)["streams"][0]
                num, den = stream["r_frame_rate"].split("/")
                out = {"fps": float(num) / float(den), "width": stream["width"],
                       "height": stream["height"], "frames": int(stream["nb_read_packets"])}
            except (OSError, ValueError, KeyError, IndexError, subprocess.SubprocessError):
                pass
            self._probe[name] = out
        return self._probe[name]

    # -- labels --------------------------------------------------------------

    def all_rows(self) -> list[dict]:
        return L.read(self.csv_path) if self.csv_path.exists() else []

    def rows(self, name: str) -> list[dict]:
        return [r for r in self.all_rows() if r["clip"].strip() == name]

    def save(self, name: str, rows: list[dict]) -> dict:
        info = self.info(name)
        clean = []
        for row in rows:
            out = {c: str(row.get(c) if row.get(c) is not None else "").strip() for c in EDITABLE}
            out["notes"] = out["notes"].replace("\n", " ")
            out.update(clip=name, frames=str(info["frames"] or ""), fps=str(info["fps"]),
                       slowmo_factor=str(info["slowmo_factor"] or ""))
            if not out["labeller"]:
                out["labeller"] = self.labeller
            clean.append(out)
        with self.write_lock:
            L.replace_clip(self.csv_path, name, clean)
            # Validate what is on disk, exactly as `snow segmenter build --gold` will read it.
            problems = L.validate(L.read(self.csv_path))
        return {"rows": self.rows(name), "problems": problems, "csv": str(self.csv_path)}

    def events(self) -> list[str]:
        return sorted({(r.get("event") or "").strip() for r in self.all_rows()} - {""})

    # -- machine guesses -------------------------------------------------------

    def request_prefill(self, name: str, refresh: bool = False) -> None:
        if not self.compute:
            return
        if refresh or (name not in self.prefills and self.status.get(name) != "queued"):
            self.status[name] = "queued"
            self._queue.put((name, refresh))

    def _worker(self) -> None:
        while True:
            name, refresh = self._queue.get()
            self.status[name] = "computing"
            try:
                self.prefills[name] = prefill.cached(name, self.cache_dir, self.seg_weights,
                                                     self.clf_weights, refresh=refresh)
                self.status[name] = "ready"
            except Exception as exc:  # noqa: BLE001  (shown on the page, labelling goes on)
                self.prefills[name] = {"clip": name, "airs": {},
                                       "sources": {"all": f"failed: {exc}"}}
                self.status[name] = "failed"

    # -- poses -------------------------------------------------------------------

    def poses(self, name: str) -> dict:
        """The rider's 2D keypoints per frame, in the clip's pixels, rounded."""
        if name not in self._poses:
            path = cfg.POSES_DIR / f"{name}.json"
            frames: dict[int, list] = {}
            if path.exists():
                for line in path.read_text().splitlines():
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    dets = row.get("detections") or []
                    det = next((d for d in dets if d.get("tracking_id") == 1),
                               dets[0] if dets else None)
                    if det is None:
                        continue
                    conf = det.get("keypoints_confidence") or [1.0] * len(det["keypoints_2d"])
                    frames[int(row["frame_id"])] = [
                        [round(x, 1), round(y, 1), round(c, 2)]
                        for (x, y), c in zip(det["keypoints_2d"], conf)]
            from ..skeleton import LINKS

            self._poses[name] = {"frames": frames, "links": LINKS}
        return self._poses[name]


def make_handler(state: State):
    base = serve.make_handler(Path("/nonexistent"))

    class Handler(base):
        server_version = "snow-label"

        def _cors(self) -> None:
            # Deliberately none: only this page may read or write the labels.
            pass

        def log_message(self, fmt: str, *args) -> None:  # quiet: one line per request is noise
            if args and str(args[1] if len(args) > 1 else "").startswith(("4", "5")):
                super().log_message(fmt, *args)

        def _name(self, raw: str) -> str | None:
            name = raw.split("?", 1)[0]
            if not NAME.match(name) or name in (".", "..") or name not in state.clip_names():
                return None
            return name

        def _html(self, body: str) -> None:
            data = body.encode()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802  (http.server naming)
            path, _, query = self.path.partition("?")
            if path in ("/", "/index.html"):
                return self._file(STATIC / "index.html")
            if path.startswith("/static/"):
                rel = path[len("/static/"):]
                if not NAME.match(rel):
                    return self._json(404, {"error": "no such file"})
                return self._file(STATIC / rel)
            if path == "/api/clips":
                rows = state.all_rows()
                clips = []
                for n in state.clip_names():
                    info = state.info(n)
                    mine = [r for r in rows if r["clip"].strip() == n]
                    info["labelled_airs"] = sum(1 for r in mine if r.get("air", "").strip() != "0")
                    info["no_air"] = any(r.get("air", "").strip() == "0" for r in mine)
                    clips.append(info)
                    state.request_prefill(n)
                return self._json(200, {"clips": clips, "csv": str(state.csv_path),
                                        "labeller": state.labeller})
            if path == "/api/guide":
                if not state.guide_path.exists():
                    return self._html("<p>docs/labelling.md is missing.</p>")
                return self._html(guide.to_html(state.guide_path.read_text()))
            for prefix in ("/api/clip/", "/api/poses/", "/video/"):
                if path.startswith(prefix):
                    name = self._name(path[len(prefix):])
                    if name is None:
                        return self._json(404, {"error": f"no clip {path[len(prefix):]!r}"})
                    if prefix == "/video/":
                        return self._file(state.clips_dir / f"{name}.mp4")
                    if prefix == "/api/poses/":
                        return self._json(200, state.poses(name))
                    state.request_prefill(name, refresh="refresh=1" in query)
                    rows = state.all_rows()
                    return self._json(200, {
                        "info": state.info(name),
                        "rows": [r for r in rows if r["clip"].strip() == name],
                        "events": state.events(),
                        "problems": L.validate(rows),
                        "prefill": state.prefills.get(name),
                        "prefill_status": state.status.get(name, "off"),
                        "labeller": state.labeller,
                    })
            return self._json(404, {"error": f"no route {path}"})

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            try:
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
            except ValueError:
                return self._json(400, {"error": "body must be JSON"})
            for prefix in ("/api/save/", "/api/reading/"):
                if not path.startswith(prefix):
                    continue
                name = self._name(path[len(prefix):])
                if name is None:
                    return self._json(404, {"error": "no such clip"})
                if prefix == "/api/save/":
                    rows = body.get("rows")
                    if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
                        return self._json(400, {"error": "rows must be a list of objects"})
                    return self._json(200, state.save(name, rows))
                try:
                    takeoff, landing = int(body["takeoff"]), int(body["landing"])
                except (KeyError, TypeError, ValueError):
                    return self._json(400, {"error": "need integer takeoff and landing"})
                if landing <= takeoff:
                    return self._json(400, {"error": "landing must be after takeoff"})
                if not state.info(name)["has_poses"]:
                    return self._json(200, {"geometry": None, "classifier": None})
                return self._json(200, prefill.reading(name, takeoff, landing,
                                                       state.clf_weights))
            return self._json(404, {"error": f"no route {path}"})

    return Handler


mimetypes.add_type("text/javascript", ".js")


def run(host: str = "127.0.0.1", port: int = 8765, csv_path: Path = DEFAULT_CSV,
        open_browser: bool = True, labeller: str = "", segmenter: str | None = None,
        classifier: str | None = None, refresh: bool = False) -> None:
    seg = prefill.segmenter_weights(segmenter)
    clf = prefill.classifier_weights(classifier)
    csv_path = csv_path.resolve()
    state = State(cfg.CLIPS_DIR, csv_path,csv_path.parent / "prefill", labeller=labeller,
                  seg_weights=seg, clf_weights=clf)
    names = state.clip_names()
    if not names:
        sys.exit(f"no clips in {cfg.CLIPS_DIR}; `snow ingest` one first")
    for n in names:
        state.request_prefill(n, refresh=refresh)
    httpd = ThreadingHTTPServer((host, port), make_handler(state))
    url = f"http://{host}:{httpd.server_address[1]}/"
    print(f"labelling {len(names)} clips ({', '.join(names)})")
    print(f"  labels   -> {csv_path}")
    print(f"  segmenter : {seg or 'none (snow label --fetch)'}")
    print(f"  classifier: {clf or 'none (snow label --fetch)'}")
    print(f"open {url}   (Ctrl-C to stop; every edit is saved as you go)")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
