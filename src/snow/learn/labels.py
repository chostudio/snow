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

The trick, factored the way the classifier predicts it (docs/classifier.md),
follows in optional columns. A file without them is still valid. A blank cell
means "not labelled", which is different from ``unsure``:

``spin``
    ``0`` (a straight air), ``180`` .. ``1800`` in steps of 180, or ``unsure``.
``direction``
    ``frontside``, ``backside`` or ``unsure``.
``axis``
    ``flat``, ``cork``, ``double_cork``, ``triple_cork`` or ``unsure``.
``grab``
    ``none``, ``indy``, ``melon``, ``mute``, ``stalefish``, ``tail``, ``nose``,
    ``method``, ``japan``, ``other`` (named in ``notes``) or ``unsure``.
``stance``
    ``regular``, ``goofy`` or ``unsure``: the rider's natural stance.
``switch``
    ``y`` when the rider takes off riding switch, ``n``, or ``unsure``.

``snow label`` (``snow.label``) writes this file; docs/labelling.md is the
guide to filling it in.
"""

from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path

from .. import config as cfg
from .data import ClipSource

BASE_COLUMNS = ["clip", "air", "takeoff_frame", "landing_frame", "event", "sure",
                "labeller", "notes", "frames", "fps", "slowmo_factor"]
UNSURE = "unsure"
# Allowed values per trick column. Every one admits "unsure" (and blank).
TRICK_VALUES = {
    "spin": tuple(str(d) for d in range(0, 1980, 180)) + (UNSURE,),
    "direction": ("frontside", "backside", UNSURE),
    "axis": ("flat", "cork", "double_cork", "triple_cork", UNSURE),
    "grab": ("none", "indy", "melon", "mute", "stalefish", "tail", "nose", "method",
             "japan", "other", UNSURE),
    "stance": ("regular", "goofy", UNSURE),
    "switch": ("y", "n", UNSURE),
}
TRICK_COLUMNS = list(TRICK_VALUES)
COLUMNS = BASE_COLUMNS + TRICK_COLUMNS


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
        tricks = [c for c in TRICK_COLUMNS if (row.get(c) or "").strip()]
        if air == "0":
            if t or l:
                problems.append(f"line {k}: {clip} air 0 (no air) must leave the frames blank")
            if tricks:
                problems.append(f"line {k}: {clip} air 0 (no air) has a trick ({tricks[0]})")
            continue
        for column in tricks:
            value = row[column].strip()
            if value not in TRICK_VALUES[column]:
                problems.append(f"line {k}: {clip} {column} {value!r} is not one of "
                                f"{', '.join(TRICK_VALUES[column])}")
        sure = (row.get("sure") or "").strip()
        if sure not in ("", "y", "n"):
            problems.append(f"line {k}: {clip} sure {sure!r} must be y or n")
        if not air.isdigit():
            problems.append(f"line {k}: {clip} air {air!r} is not a number")
            continue
        if not t or not l:
            problems.append(f"line {k}: {clip} air {air} is missing a frame")
            continue
        if not (t.isdigit() and l.isdigit()):
            problems.append(f"line {k}: {clip} frames {t!r}, {l!r} must be whole numbers")
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


def write(dest: Path, rows: list[dict]) -> None:
    """All rows in ``COLUMNS`` order, replacing ``dest`` atomically.

    Columns this module does not know are kept after the known ones, so a
    column added by hand survives a save from ``snow label``.
    """
    extra = [c for row in rows for c in row if c and c not in COLUMNS]
    fields = COLUMNS + list(dict.fromkeys(extra))
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=f".{dest.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({c: "" if row.get(c) is None else row.get(c) for c in fields})
        os.replace(tmp, dest)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def replace_clip(dest: Path, clip: str, rows: list[dict]) -> list[dict]:
    """Swap ``clip``'s rows in ``dest`` for ``rows``, keeping every other clip's.

    The edited clip's rows go where its first row was (or at the end),
    ordered by air. Returns the whole file's rows as written.
    """
    existing = read(dest) if dest.exists() else []

    def air_of(row: dict) -> int:
        value = str(row.get("air") or "0").strip()
        return int(value) if value.isdigit() else 0

    rows = sorted(rows, key=air_of)
    out, placed = [], False
    for row in existing:
        if row["clip"].strip() == clip:
            if not placed:
                out.extend(rows)
                placed = True
            continue
        out.append(row)
    if not placed:
        out.extend(rows)
    write(dest, out)
    return out


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
            slowmo=slowmo, kind="real", variant="real"))
        if (row.get("air") or "").strip() != "0":
            entry.takeoff.append(int(row["takeoff_frame"]))
            entry.landing.append(int(row["landing_frame"]))
    return list(out.values())
