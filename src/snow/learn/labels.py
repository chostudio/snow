"""Hand labels for real airs: the gold set the segmenter is finally judged on.

One CSV row per air (spec in docs/segmenter.md, "Labelling real airs"):

``clip``
    Name under ``data/clips``; frame numbers index that clip, which is also
    ``frame_id`` in ``data/poses/<clip>.json``.
``air``
    1, 2, ... within the clip. ``0`` with both frames blank means the clip
    was watched and holds **no** air, which is a label too.
``takeoff_frame``
    First frame the board is off the snow.
``landing_frame``
    First frame the board is back on the snow (so the air is
    ``[takeoff, landing)``, matching the synthetic labels).
``event``
    The real-world jump this shows: contest, round, rider, run, hit. A live
    shot and its replays share it, and a split never separates it.
``sure``
    ``y``, or ``n`` when the boundary is hidden (cut, occlusion, blur).
``labeller``, ``notes``
    Free text.

The template also carries ``frames``, ``fps`` and ``slowmo_factor`` from the
clip's metadata, read-only, to save opening each file.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .. import config as cfg
from .data import ClipSource

COLUMNS = ["clip", "air", "takeoff_frame", "landing_frame", "event", "sure",
           "labeller", "notes", "frames", "fps", "slowmo_factor"]


def write_template(dest: Path, clips: list[str] | None = None) -> int:
    """One blank row per clip, with its frame count and rates filled in."""
    if clips is None:
        clips = sorted(p.name[: -len(".meta.json")] for p in cfg.CLIPS_DIR.glob("*.meta.json")
                       if not p.name.startswith("synth"))
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        for name in clips:
            meta_path = cfg.CLIPS_DIR / f"{name}.meta.json"
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            writer.writerow({"clip": name, "air": 1, "frames": meta.get("frames", ""),
                             "fps": meta.get("fps", ""),
                             "slowmo_factor": meta.get("slowmo_factor", "")})
    return len(clips)


def read(src: Path) -> list[dict]:
    with src.open(newline="") as fh:
        return [row for row in csv.DictReader(fh) if (row.get("clip") or "").strip()]


def validate(rows: list[dict]) -> list[str]:
    """Problems that would silently corrupt a label; empty when clean."""
    problems = []
    by_clip: dict[str, list[tuple[int, int]]] = {}
    for k, row in enumerate(rows, start=2):  # line 1 is the header
        clip, air = row["clip"].strip(), (row.get("air") or "").strip()
        t, l = (row.get("takeoff_frame") or "").strip(), (row.get("landing_frame") or "").strip()
        if air == "0":
            if t or l:
                problems.append(f"line {k}: {clip} air 0 (no air) must leave the frames blank")
            continue
        if not t or not l:
            problems.append(f"line {k}: {clip} air {air} is missing a frame")
            continue
        if int(l) <= int(t):
            problems.append(f"line {k}: {clip} landing {l} is not after takeoff {t}")
        if not (row.get("event") or "").strip():
            problems.append(f"line {k}: {clip} has no event; the split needs it")
        frames = (row.get("frames") or "").strip()
        if frames and int(l) > int(frames):
            problems.append(f"line {k}: {clip} landing {l} is past the clip's {frames} frames")
        by_clip.setdefault(clip, []).append((int(t), int(l)))
    for clip, spans in by_clip.items():
        spans.sort()
        for (a0, a1), (b0, _) in zip(spans, spans[1:]):
            if b0 < a1:
                problems.append(f"{clip}: airs overlap ({a0}-{a1} and {b0})")
    return problems


def gold_sources(src: Path) -> list[ClipSource]:
    rows = read(src)
    problems = validate(rows)
    if problems:
        raise ValueError("labels CSV has problems:\n  " + "\n  ".join(problems))
    out: dict[str, ClipSource] = {}
    for row in rows:
        clip = row["clip"].strip()
        slowmo = float(row.get("slowmo_factor") or 1.0)
        entry = out.setdefault(clip, ClipSource(
            name=clip, group=(row.get("event") or clip).strip(), takeoff=[], landing=[],
            slowmo=slowmo, kind="real"))
        if (row.get("air") or "").strip() != "0":
            entry.takeoff.append(int(row["takeoff_frame"]))
            entry.landing.append(int(row["landing_frame"]))
    return list(out.values())
