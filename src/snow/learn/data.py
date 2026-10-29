"""The segmenter's dataset: labelled clips, their groups, and their inputs.

Each clip contributes the pipeline's own output for its rendered video
(``data/poses/<name>.json``, ``.features.csv``, ``.camera.npz``), never the
rig's clean skeleton: the ``-oracle`` variants that ``snow synth prepare``
writes are refused by name (docs/training.md, "Train on NIM output").

Labels come from ``labels.json`` (exact, synthetic) or a hand-labelled gold
file (``labeltool.py``). The baselines the model is compared against, the
features table's old ``airborne_hint`` and the dashboard's air detector, are
computed here too, on the same frames, so every method is scored on exactly
the same rows.

Groups: clips that show the same event (one trick seen from several cameras,
a live shot and its replay) share a group, and a split never separates a
group. Without a manifest each synthetic clip is its own group, which is
right for the 20 hand-planned tricks: each is a separately seeded trick.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as cfg
from ..features import load_poses
from . import inputs as I


@dataclass
class ClipSource:
    name: str
    group: str
    takeoff: list[int]
    landing: list[int]
    slowmo: float = 1.0          # for reporting errors in real time only; never an input
    split: str | None = None     # "train" / "test" when the manifest fixes it
    kind: str = ""
    camera_angle: str = ""
    # "long-ground" (standing start, deck riding, turns, long ride-out) or
    # "big-air" (short straight in-run and ride-out); "real" for gold labels.
    variant: str = ""


def heuristic_hint(df: pd.DataFrame) -> np.ndarray:
    """The placeholder this model replaces: under the clip's median grab distance."""
    grab = df["grab_distance"].to_numpy(dtype=float)
    return grab < np.nanmedian(grab)


def web_detector_mask(name: str, frame_ids: np.ndarray) -> np.ndarray:
    """The dashboard's rate-hysteresis air detector, as a per-row mask.

    Runs the shipped detector (``synth.compare.air_events`` mirrors
    ``export_session.main``); its ``end_f`` is inclusive.
    """
    from ..synth.compare import air_events

    index = {int(f): k for k, f in enumerate(frame_ids)}
    mask = np.zeros(len(frame_ids), dtype=bool)
    for event in air_events(name, detector="rate"):
        for f in range(int(event["start_f"]), int(event["end_f"]) + 1):
            if f in index:
                mask[index[f]] = True
    return mask


def from_synth_labels(clip_dir: Path, group: str | None = None,
                      split: str | None = None) -> ClipSource:
    labels = json.loads((clip_dir / "labels.json").read_text())
    trick_path = clip_dir / "trick.json"
    trick = json.loads(trick_path.read_text()) if trick_path.exists() else {}
    return ClipSource(
        variant="long-ground" if trick.get("terrain") else "big-air",
        name=labels.get("name", clip_dir.name), group=group or labels.get("name", clip_dir.name),
        takeoff=[int(labels["takeoff_frame"])], landing=[int(labels["landing_frame"])],
        slowmo=float(labels.get("slowmo_factor", 1.0)), split=split,
        kind=str(labels.get("kind", "")), camera_angle=str(labels.get("camera_angle", "")),
    )


def _first(record: dict, *keys: str):
    return next((record[k] for k in keys if record.get(k) not in (None, "")), None)


def discover(synth_dir: Path | None = None, manifest: Path | None = None) -> list[ClipSource]:
    """Every labelled synthetic clip, with its group from the manifest when given.

    The manifest is JSON Lines with at least the clip name (``snow synth``'s
    ``MANIFEST.scale.jsonl``); the group is read from ``group``,
    ``split_group`` or ``event``, a fixed ``split`` is honoured when present,
    and a clip whose ``status`` is not ``done`` is skipped. Clips in ``synth_dir`` that the manifest does not
    mention keep their own name as group.
    """
    synth_dir = synth_dir or cfg.DATA_DIR / "synth"
    meta: dict[str, dict] = {}
    if manifest and manifest.exists():
        for line in manifest.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                name = _first(record, "name", "clip")
                if name:
                    meta[str(name)] = record
    out = []
    for labels in sorted(synth_dir.glob("*/labels.json")):
        clip_dir = labels.parent
        record = meta.get(clip_dir.name, {})
        if record.get("status", "done") != "done":
            continue  # the scaling run says this clip's pipeline output is incomplete
        group = _first(record, "group", "split_group", "event")
        split = _first(record, "split")
        out.append(from_synth_labels(clip_dir, str(group) if group else None,
                                     str(split) if split else None))
    return out


def load_clip(src: ClipSource, poses_dir: Path | None = None, baselines: bool = True) -> dict | None:
    """Inputs (and their mirror), labels and baseline masks for one clip."""
    if src.name.endswith("-oracle"):
        raise ValueError(f"{src.name}: oracle skeletons are never training data")
    poses_dir = poses_dir or cfg.POSES_DIR
    poses = poses_dir / f"{src.name}.json"
    table = poses_dir / f"{src.name}.features.csv"
    if not poses.exists() or not table.exists():
        return None
    df = pd.read_csv(table)
    detections = load_poses(poses)
    camera_path = poses_dir / f"{src.name}.camera.npz"
    rotations = np.load(camera_path)["rotations"] if camera_path.exists() else None
    frame_ids = df["frame_id"].to_numpy(dtype=int)
    y = np.zeros(len(df), dtype=bool)
    for t0, t1 in zip(src.takeoff, src.landing):
        y |= (frame_ids >= t0) & (frame_ids < t1)
    clip = {
        "name": src.name, "group": src.group, "split": src.split, "slowmo": src.slowmo,
        "kind": src.kind, "camera_angle": src.camera_angle, "variant": src.variant,
        "frame_id": frame_ids,
        "segment": df["segment"].to_numpy(dtype=int),
        "x": I.frame_channels(detections, df, rotations),
        "x_mirror": I.frame_channels(detections, df, rotations, mirror=True),
        "y": y,
    }
    if baselines:
        clip["hint"] = heuristic_hint(df)
        clip["web"] = web_detector_mask(src.name, frame_ids)
    return clip


ARRAYS = ("frame_id", "segment", "x", "x_mirror", "y", "hint", "web")
FIELDS = ("name", "group", "split", "slowmo", "kind", "camera_angle", "variant")


def save_bundle(clips: list[dict], dest: Path) -> None:
    """One ``.npz`` (arrays, no pickles) plus a JSON index beside it."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f"{c['name']}::{k}": c[k] for c in clips for k in ARRAYS if k in c}
    np.savez_compressed(dest, **arrays)
    index = [{k: c.get(k) for k in FIELDS} for c in clips]
    dest.with_suffix(".json").write_text(json.dumps(
        {"channels": list(I.CHANNELS), "clips": index}, indent=1))


def load_bundle(src: Path) -> list[dict]:
    index = json.loads(src.with_suffix(".json").read_text())
    if index["channels"] != list(I.CHANNELS):
        raise ValueError(f"{src}: built with different input channels; rebuild it")
    z = np.load(src)
    clips = []
    for entry in index["clips"]:
        clip = dict(entry)
        for k in ARRAYS:
            key = f"{entry['name']}::{k}"
            if key in z:
                clip[k] = z[key]
        clips.append(clip)
    return clips


def group_folds(clips: list[dict], folds: int, seed: int = 0) -> list[int]:
    """Fold index per clip; every clip of a group lands in the same fold."""
    groups = sorted({c["group"] for c in clips})
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    fold_of = {g: k % folds for k, g in enumerate(groups)}
    return [fold_of[c["group"]] for c in clips]


def check_no_leak(train: list[dict], test: list[dict]) -> None:
    shared = {c["group"] for c in train} & {c["group"] for c in test}
    if shared:
        raise AssertionError(f"groups in both train and test: {sorted(shared)[:5]}")
