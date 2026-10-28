"""The classifier's dataset: one record per synthetic air, labelled exactly.

Labels come from each clip's ``labels.json``, which is measured from the
generated motion (docs/synthetic.md), never from ``axis_spin_deg`` or
``spin_path_deg``: those are the geometric measurement, which the model is
compared *against*, not trained on (docs/training.md).

Inputs come from the **pipeline** variant of each clip: the render through the
tracker, the NIM and ``camera.py``. The ``-oracle`` variant (the rig's exact
skeleton) is never read, because a model trained on clean skeletons never sees
the NIM's failure modes.

Each record carries two windows: the true air (``takeoff_frame`` to
``landing_frame``), for training and for the upper-bound evaluation, and the
air the dashboard's heuristic detector finds, for the evaluation a user would
actually see. The segmenter (model A) will replace that detector; nothing
here depends on it.

Split by group, never by clip. ``MANIFEST.scale.jsonl`` names each clip's
split group; the original twenty are one event each.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import config as cfg
from .. import features as feat
from ..camera import CameraTrack
from . import airs

ROTATIONS = tuple(range(0, 1980, 180))           # 0 .. 1800, one class per 180
SENSES = ("clockwise", "counter-clockwise")      # about world up, from above
STANCES = ("regular", "goofy")
AXES = ("flat", "cork", "double_cork")
GRABS = ("none", "indy", "melon", "mute", "stalefish", "tail", "nose")
HEADS = {"rotation": ROTATIONS, "sense": SENSES, "stance": STANCES,
         "axis": AXES, "grab": GRABS}

MANIFEST = "MANIFEST.scale.jsonl"
# Fractions of groups held out when the manifest names a group but no split.
VAL_FRACTION, TEST_FRACTION = 0.15, 0.15


@dataclass
class Air:
    name: str
    group: str
    split: str
    track: airs.Track
    gt: tuple[int, int]
    heuristic: tuple[int, int] | None
    labels: dict[str, int]
    baseline: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)


def labels_of(raw: dict) -> dict[str, int]:
    """Class indices per head from a synthetic ``labels.json``."""
    spin = int(raw["spin_nominal_deg"])
    return {
        "rotation": ROTATIONS.index(spin),
        "sense": SENSES.index(raw["rotation_sense"]),
        "stance": STANCES.index(raw["stance"]),
        "axis": AXES.index(raw["kind"]),
        "grab": GRABS.index(raw.get("grab", "none")),
    }


def mirrored_labels(labels: dict[str, int]) -> dict[str, int]:
    """Labels of the mirrored air: the turn reverses and so does the stance.

    Frontside/backside is the product of the two, so it survives a mirror
    (``direction``); what a mirror flips is the sense the rider turned in,
    which is what the ``sense`` head predicts.
    """
    out = dict(labels)
    out["sense"] = 1 - labels["sense"]
    out["stance"] = 1 - labels["stance"]
    return out


def direction(sense: int, stance: int) -> str:
    """Frontside or backside from the turn's sense and the stance.

    ``tricks.Trick.spin_sign``: counter-clockwise is frontside for a regular
    rider and backside for a goofy one.
    """
    ccw = SENSES[sense] == "counter-clockwise"
    regular = STANCES[stance] == "regular"
    return "frontside" if ccw == regular else "backside"


def split_of(group: str) -> str:
    """Deterministic train/val/test from a group name, for manifests without one."""
    h = int(hashlib.sha1(group.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if h < TEST_FRACTION:
        return "test"
    if h < TEST_FRACTION + VAL_FRACTION:
        return "val"
    return "train"


def manifest(data: Path) -> dict[str, dict]:
    """Clip name -> manifest row, from the synthetic scale-up's manifest."""
    path = data / "synth" / MANIFEST
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # a line still being written
        name = row.get("name") or row.get("clip")
        if name:
            out[name] = row
    return out


def _group_and_split(name: str, row: dict | None) -> tuple[str, str]:
    if not row:
        # The original twenty: one generated trick each, no replays.
        return name, "original"
    group = str(row.get("split_group") or row.get("group") or row.get("event") or name)
    split = row.get("split")
    if split not in ("train", "val", "test"):
        split = split_of(group)
    return group, split


def _heuristic_window(name: str) -> tuple[tuple[int, int] | None, dict]:
    """The dashboard's main air for ``name``, exactly as it would show it."""
    from ..synth import compare

    events = compare.air_events(name)
    main = max(events, key=lambda e: abs(e["metrics"]["rotation_deg"]), default=None)
    if main is None:
        return None, {}
    m = main["metrics"]
    return (int(main["start_f"]), int(main["end_f"])), {
        "rotation_deg": m["rotation_deg"], "spin_deg": m["nearest_spin_deg"],
        "sense": m["direction"], "max_tilt_deg": m["max_axis_tilt_deg"],
    }


def _inversions(tilt: np.ndarray, threshold: float = 90.0, min_frames: int = 3,
                merge: int = 10) -> int:
    """Separate stretches with the rider past ``threshold`` degrees of tilt."""
    over = np.flatnonzero(np.nan_to_num(tilt) > threshold)
    if not len(over):
        return 0
    runs = np.split(over, np.flatnonzero(np.diff(over) > merge) + 1)
    return sum(1 for r in runs if r[-1] - r[0] + 1 >= min_frames)


def geometric(features_csv: Path, start: int, end: int) -> dict:
    """What the current measurement says over one window: the baseline.

    Spin is ``spin_path_deg`` over the window, rounded to the nearest 180
    (the dashboard's count); the sense is its sign; the axis is the number of
    separate inversions in ``tilt_deg`` (none flat, one cork, two or more a
    double cork). Read from the longest single-shot stretch, never across a
    cut, as ``synth.compare`` does.
    """
    import pandas as pd

    df = pd.read_csv(features_csv)
    inside = df[(df["frame_id"] >= start) & (df["frame_id"] <= end)]
    out: dict = {}
    path = inside.dropna(subset=["spin_path_deg"])
    if len(path) >= 2:
        seg = path.groupby("segment").size().idxmax()
        path = path[path["segment"] == seg]
        value = float(path["spin_path_deg"].iloc[-1] - path["spin_path_deg"].iloc[0])
        out["spin_path_deg"] = round(value, 1)
        out["spin_deg"] = int(round(abs(value) / 180.0)) * 180
        out["sense"] = "counter-clockwise" if value > 0 else "clockwise"
    if "spin_deg" in inside and inside["spin_deg"].notna().sum() >= 2:
        yaw = inside["spin_deg"].dropna()
        out["yaw_deg"] = round(float(yaw.iloc[-1] - yaw.iloc[0]), 1)
    tilt = inside["tilt_deg"].to_numpy(dtype=float)
    out["max_tilt_deg"] = round(float(np.nanmax(tilt)), 1) if np.isfinite(tilt).any() else None
    count = _inversions(tilt)
    out["axis"] = AXES[min(count, 2)]
    return out


def load_air(data: Path, name: str, row: dict | None = None,
             heuristic: bool = True) -> Air | None:
    """One synthetic clip's pipeline output as an ``Air``, or None if incomplete."""
    clip_dir = data / "synth" / name
    poses = data / "poses" / f"{name}.json"
    camera = data / "poses" / f"{name}.camera.npz"
    csv = data / "poses" / f"{name}.features.csv"
    if not ((clip_dir / "labels.json").exists() and poses.exists()):
        return None
    raw = json.loads((clip_dir / "labels.json").read_text())
    cam = CameraTrack.load(camera) if camera.exists() else None
    detections = feat.load_poses(poses)
    frames = int(raw.get("frames") or (max(detections) + 1 if detections else 0))
    track = airs.track_from_poses(detections, frames, cam)
    group, split = _group_and_split(name, row)
    gt = (int(raw["takeoff_frame"]), int(raw["landing_frame"]))
    air = Air(name=name, group=group, split=split, track=track, gt=gt,
              heuristic=None, labels=labels_of(raw),
              info={k: raw.get(k) for k in ("spin_nominal_deg", "kind", "direction",
                                            "stance", "grab", "camera_angle",
                                            "slowmo_factor", "rotation_sense")})
    if csv.exists():
        air.baseline["gt"] = geometric(csv, *gt)
    if heuristic and csv.exists():
        window, metrics = _heuristic_window(name)
        air.heuristic = window
        if window is not None:
            air.baseline["heuristic"] = {**geometric(csv, *window), **{
                "spin_deg": metrics["spin_deg"], "sense": metrics["sense"],
                "rotation_deg": metrics["rotation_deg"]}}
    return air


def clip_names(data: Path) -> list[str]:
    """Synthetic clips with pipeline output: the original twenty plus the manifest."""
    names = {p.parent.name for p in (data / "synth").glob("*/labels.json")}
    names |= set(manifest(data))
    return sorted(n for n in names if (data / "poses" / f"{n}.json").exists()
                  and not n.endswith("-oracle"))


def build(data: Path | None = None, heuristic: bool = True,
          progress: bool = True) -> list[Air]:
    data = data or cfg.DATA_DIR
    rows = manifest(data)
    out = []
    for name in clip_names(data):
        air = load_air(data, name, rows.get(name), heuristic=heuristic)
        if air is None:
            continue
        out.append(air)
        if progress:
            print(f"  {name}: {air.split:8s} group {air.group}  gt {air.gt}  "
                  f"heuristic {air.heuristic}")
    return out


def save(records: list[Air], dest: Path) -> Path:
    """One ``.npz`` of arrays plus a JSON index beside it; no pickles."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    arrays, index = {}, []
    for i, a in enumerate(records):
        arrays[f"{i}_k3"] = a.track.k3
        arrays[f"{i}_conf"] = a.track.confidence
        arrays[f"{i}_camera"] = a.track.camera
        arrays[f"{i}_segment"] = a.track.segment
        index.append({"name": a.name, "group": a.group, "split": a.split, "gt": list(a.gt),
                      "heuristic": list(a.heuristic) if a.heuristic else None,
                      "labels": a.labels, "baseline": a.baseline, "info": a.info})
    np.savez_compressed(dest, **arrays)
    dest.with_suffix(".json").write_text(json.dumps(index, indent=1))
    return dest


def load(src: Path) -> list[Air]:
    index = json.loads(src.with_suffix(".json").read_text())
    z = np.load(src)
    out = []
    for i, row in enumerate(index):
        track = airs.Track(k3=z[f"{i}_k3"], confidence=z[f"{i}_conf"],
                           camera=z[f"{i}_camera"], segment=z[f"{i}_segment"])
        out.append(Air(name=row["name"], group=row["group"], split=row["split"],
                       track=track, gt=tuple(row["gt"]),
                       heuristic=tuple(row["heuristic"]) if row["heuristic"] else None,
                       labels=row["labels"], baseline=row["baseline"], info=row["info"]))
    return out
