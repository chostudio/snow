"""Machine guesses shown beside the labeller's own answers, never instead of them.

For each clip with pipeline output (``data/poses/<clip>.json`` and its
features table) this computes:

- the airs the dashboard's **rate detector** finds (``export_session.py``,
  the shipped detector, run with ``detector="rate"``);
- the airs the **segmenter** (model A) finds, when weights are on disk;
- for every one of those airs, and on request for any window the labeller
  sets, the trick as the **geometry** reads it (``dataset.geometric``: the
  dashboard's spin count, its sign and the inversion count) and as the
  **classifier** (model B) reads it, when its weights are on disk.

None of it is validated on real footage (docs/validation.md,
docs/classifier.md), and the page says so beside every number. Results are
cached in ``data/labels/prefill/<clip>.json``, keyed by the inputs' sizes
and modification times, so a re-run of the pipeline invalidates them.

Weights are looked up, never installed: ``snow label --fetch`` copies them
from S3 into ``data/models/``, and nothing here writes to ``data/weights``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from pathlib import Path

import numpy as np

from .. import config as cfg

S3_SEGMENTER = "s3://snow-artifacts-009160029345/models/segmenter/weights/segmenter_scale154.pt"
S3_CLASSIFIER = "s3://snow-artifacts-009160029345/models/classifier/classifier.pt"
FETCHED_SEGMENTER = cfg.DATA_DIR / "models" / "segmenter" / "weights" / "segmenter_scale154.pt"
FETCHED_CLASSIFIER = cfg.DATA_DIR / "models" / "classifier" / "classifier.pt"
CACHE_VERSION = 1

_lock = threading.Lock()
_models: dict[str, object] = {}


# --- which weights --------------------------------------------------------------

def segmenter_weights(choice: str | None = None) -> Path | None:
    """The segmenter checkpoint to use, or None.

    An explicit path wins; then ``$SNOW_SEGMENTER``, ``data/weights/segmenter.pt``
    (what ``snow features`` would use), then the copies ``snow label --fetch``
    and ``snow segmenter train`` leave under ``data/models/segmenter``.
    """
    if choice:
        return Path(choice) if Path(choice).exists() else None
    env = os.environ.get("SNOW_SEGMENTER")
    candidates = [Path(env)] if env else []
    candidates += [cfg.WEIGHTS_DIR / "segmenter.pt", FETCHED_SEGMENTER,
                   cfg.DATA_DIR / "models" / "segmenter" / "segmenter_scale154.pt"]
    return next((p for p in candidates if p.exists()), None)


def classifier_weights(choice: str | None = None) -> Path | None:
    if choice:
        return Path(choice) if Path(choice).exists() else None
    return FETCHED_CLASSIFIER if FETCHED_CLASSIFIER.exists() else None


def fetch_weights() -> list[str]:
    """Copy the shipped weights from S3 with the aws CLI; report what happened."""
    if shutil.which("aws") is None:
        return ["aws CLI not found; install it or copy the weights by hand (docs/labelling.md)"]
    out = []
    for src, dest in ((S3_SEGMENTER, FETCHED_SEGMENTER), (S3_CLASSIFIER, FETCHED_CLASSIFIER)):
        if dest.exists():
            out.append(f"have {dest}")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        done = subprocess.run(["aws", "s3", "cp", "--only-show-errors", src, str(dest)],
                              capture_output=True, text=True)
        out.append(f"fetched {dest}" if done.returncode == 0
                   else f"could not fetch {src}: {done.stderr.strip() or done.returncode}")
    return out


# --- small pure helpers (tested) ------------------------------------------------

def spans_from_mask(mask: np.ndarray, frame_ids: np.ndarray,
                    segment: np.ndarray | None = None) -> list[tuple[int, int]]:
    """``[takeoff, landing)`` in clip frames for every airborne run of rows.

    Rows index the features table, whose ``frame_id`` can skip frames; the
    landing is the frame after the last airborne row.
    """
    from ..learn.events import runs

    return [(int(frame_ids[a]), int(frame_ids[b - 1]) + 1)
            for a, b in runs(np.asarray(mask, dtype=bool), segment)]


def rounding_margin(path_deg: float) -> float:
    """How clearly a measured spin names one multiple of 180: 1 on it, 0 halfway.

    This is not a probability. It says how far the measurement sits from the
    rounding boundary, which is all the geometry has to offer as confidence.
    """
    off = abs(abs(path_deg) - round(abs(path_deg) / 180.0) * 180.0)
    return round(1.0 - off / 90.0, 2)


def fingerprint(paths: list[Path]) -> list:
    return [[p.name, p.stat().st_size, int(p.stat().st_mtime)] if p.exists() else [p.name, None]
            for p in paths]


# --- the readings -----------------------------------------------------------------

def _classifier(path: Path):
    with _lock:
        key = f"clf:{path}"
        if key not in _models:
            from ..learn import classifier as clf

            _models[key] = clf.Classifier.load(path)
        return _models[key]


def _track(name: str):
    from .. import features as feat
    from ..camera import CameraTrack
    from ..learn import airs

    camera = cfg.POSES_DIR / f"{name}.camera.npz"
    return airs.track_from_poses(feat.load_poses(cfg.pose_path(name)), 0,
                                 CameraTrack.load(camera) if camera.exists() else None)


def geometry_reading(name: str, takeoff: int, landing: int) -> dict | None:
    """The geometric trick reading over ``[takeoff, landing)``, or None."""
    from ..learn import dataset as ds

    table = cfg.POSES_DIR / f"{name}.features.csv"
    if not table.exists() or landing - takeoff < 2:
        return None
    got = ds.geometric(table, takeoff, landing - 1)
    if "spin_path_deg" in got:
        got["spin_margin"] = rounding_margin(got["spin_path_deg"])
    import pandas as pd

    df = pd.read_csv(table, usecols=["frame_id", "confidence"])
    inside = df[(df["frame_id"] >= takeoff) & (df["frame_id"] < landing)]
    got["posed_fraction"] = round(float(inside["confidence"].notna().sum())
                                  / max(1, landing - takeoff), 2)
    return got


def classifier_reading(name: str, takeoff: int, landing: int, weights: Path | None,
                       track=None) -> dict | None:
    if weights is None or landing - takeoff < 2:
        return None
    model = _classifier(weights)
    return model.classify(track if track is not None else _track(name), takeoff, landing)


def reading(name: str, takeoff: int, landing: int, clf_weights: Path | None,
            track=None) -> dict:
    """Both trick readings for one window, each failing on its own."""
    out: dict = {}
    for key, fn in (("geometry", lambda: geometry_reading(name, takeoff, landing)),
                    ("classifier", lambda: classifier_reading(name, takeoff, landing,
                                                              clf_weights, track))):
        try:
            out[key] = fn()
        except Exception as exc:  # noqa: BLE001  (a guess failing must not stop labelling)
            out[key] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


def _rate_airs(name: str) -> list[tuple[int, int]]:
    from ..synth import compare

    return [(int(e["start_f"]), int(e["end_f"]) + 1)
            for e in compare.air_events(name, detector="rate")]


def _segmenter_airs(name: str, weights: Path) -> tuple[list[tuple[int, int]], list]:
    import pandas as pd

    from .. import features as feat
    from ..learn.inputs import frame_channels
    from ..learn.segmenter import Segmenter

    with _lock:
        key = f"seg:{weights}"
        if key not in _models:
            _models[key] = Segmenter.load(weights)
        model = _models[key]
    df = pd.read_csv(cfg.POSES_DIR / f"{name}.features.csv")
    camera = cfg.POSES_DIR / f"{name}.camera.npz"
    rotations = np.load(camera)["rotations"] if camera.exists() else None
    x = frame_channels(feat.load_poses(cfg.pose_path(name)), df, rotations)
    segment = df["segment"].to_numpy(dtype=int)
    prob = model.probabilities(x, segment)
    mask = model.mask(prob, segment)
    frame_ids = df["frame_id"].to_numpy(dtype=int)
    per_frame = [None if not np.isfinite(p) else round(float(p), 3) for p in prob]
    return spans_from_mask(mask, frame_ids, segment), [
        [int(f), p] for f, p in zip(frame_ids, per_frame)]


def compute(name: str, seg_weights: Path | None, clf_weights: Path | None) -> dict:
    """Every machine guess for one clip. Never raises for a missing input."""
    poses = cfg.pose_path(name)
    table = cfg.POSES_DIR / f"{name}.features.csv"
    out: dict = {"clip": name, "version": CACHE_VERSION, "airs": {}, "sources": {},
                 "airborne_prob": None}
    if not (poses.exists() and table.exists()):
        out["sources"]["all"] = ("no pose output for this clip, so no machine guesses: "
                                 f"run `snow run` and `snow features {name}` first")
        return out
    track = None
    try:
        track = _track(name)
    except Exception as exc:  # noqa: BLE001
        out["sources"]["track"] = f"could not build the pose track: {exc}"

    detectors = {"rate": ("the dashboard's rate detector", lambda: _rate_airs(name))}
    if seg_weights is not None:
        def seg() -> list[tuple[int, int]]:
            spans, prob = _segmenter_airs(name, seg_weights)
            out["airborne_prob"] = prob
            return spans
        detectors["segmenter"] = (f"segmenter ({seg_weights.name})", seg)
    else:
        out["sources"]["segmenter"] = ("no segmenter weights on disk; `snow label --fetch` "
                                       "copies them from S3")
    for key, (label, fn) in detectors.items():
        try:
            spans = fn()
        except Exception as exc:  # noqa: BLE001
            out["sources"][key] = f"{label} failed: {type(exc).__name__}: {exc}"
            continue
        out["sources"][key] = label
        out["airs"][key] = [{"takeoff": a, "landing": b,
                             **reading(name, a, b, clf_weights, track)} for a, b in spans]
    out["sources"]["classifier"] = (f"classifier ({clf_weights.name})" if clf_weights else
                                    "no classifier weights on disk; `snow label --fetch` "
                                    "copies them from S3")
    return out


def cached(name: str, cache_dir: Path, seg_weights: Path | None, clf_weights: Path | None,
           refresh: bool = False) -> dict:
    """``compute`` through the on-disk cache."""
    inputs = [cfg.pose_path(name), cfg.POSES_DIR / f"{name}.features.csv",
              cfg.POSES_DIR / f"{name}.camera.npz"]
    inputs += [p for p in (seg_weights, clf_weights) if p is not None]
    key = {"version": CACHE_VERSION, "inputs": fingerprint(inputs),
           "segmenter": str(seg_weights), "classifier": str(clf_weights)}
    path = cache_dir / f"{name}.json"
    if path.exists() and not refresh:
        try:
            got = json.loads(path.read_text())
            if got.get("key") == key:
                return got
        except ValueError:
            pass
    got = compute(name, seg_weights, clf_weights)
    got["key"] = key
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(got))
    return got
