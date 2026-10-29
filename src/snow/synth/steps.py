"""The synthetic loop, one step at a time: plan, render, prepare, pipeline, compare.

Each step reads and writes files under ``data/synth/<clip>/`` (and, for the
pipeline steps, the usual ``data/clips`` and ``data/poses``), so any step can
be re-run alone, like every other stage.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .. import config as cfg
from .. import meta as mt
from . import compare, groundtruth, tricks

SYNTH_DIR = cfg.DATA_DIR / "synth"
BLENDER = os.environ.get("SNOW_BLENDER", "/Applications/Blender.app/Contents/MacOS/Blender")
RUNNER = Path(__file__).resolve().parent / "blender" / "run.py"


def clips(only: list[str] | None = None) -> list[Path]:
    found = sorted(p for p in SYNTH_DIR.glob("synth*") if (p / "trick.json").exists())
    if only:
        found = [p for p in found if any(p.name.startswith(o) for o in only)]
    return found


def plan(seed: int = 0, scale: int = 0, start: int = 1, long_ground: bool = False) -> list[Path]:
    """Write ``trick.json`` + ``motion.npz`` + ``labels.json`` per trick.

    ``scale=0`` is the 20-trick validation set; ``scale=N`` is ``N`` clips of
    the balanced training set (``tricks.scale_plan``) numbered from ``start``.
    A clip that is already planned is left alone, so its render stays valid.
    """
    out = []
    planned = (tricks.scale_plan(scale, seed=seed, start=start, long_ground=long_ground)
               if scale else tricks.plan(seed))
    for trick in planned:
        clip = SYNTH_DIR / trick.name
        if scale and (clip / "trick.json").exists():
            out.append(clip)
            continue
        clip.mkdir(parents=True, exist_ok=True)
        (clip / "trick.json").write_text(trick.to_json())
        motion = tricks.motion(trick)
        motion.save(clip / "motion.npz")
        labels = tricks.labels(trick, motion)
        (clip / "labels.json").write_text(json.dumps(labels, indent=2))
        out.append(clip)
    return out


def render(clip: Path, width: int, height: int, samples: int, engine: str = "eevee") -> None:
    """Build, record and render in a headless Blender (needs MPFB installed).

    ``engine="cycles"`` renders on the GPU with Cycles, for headless Linux
    boxes where EEVEE has no GPU context. The engine is recorded in
    ``renderer.txt`` by ``encode``.
    """
    frames = clip / "frames"
    if frames.exists():
        shutil.rmtree(frames)
    cmd = [BLENDER, "-b", "--python", str(RUNNER), "--", str(clip),
           "--width", str(width), "--height", str(height), "--samples", str(samples),
           "--engine", engine]
    log = (clip / "blender.log").open("w")
    result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0 or not (clip / "gt_raw.npz").exists():
        raise SystemExit(f"blender failed on {clip.name}; see {clip / 'blender.log'}")


def encode(clip: Path, renderer: str = "blender", engine: str = "eevee") -> Path:
    """Rendered frames -> an H.264 file at 30 fps, as a broadcast would deliver it.

    ``renderer="replicator"`` encodes ``replicator/rgb_*.png`` from
    ``replicator_render.py`` instead of Blender's frames; the labels are the
    same either way, because both render the same USD stage.
    """
    dest = clip / "render.mp4"
    if renderer == "replicator":
        source = ["-pattern_type", "glob", "-i", str(clip / "replicator" / "rgb_*.png")]
    else:
        source = ["-i", str(clip / "frames" / "frame_%04d.png")]
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-framerate", str(tricks.OUTPUT_FPS), *source,
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(dest)],
        check=True,
    )
    if renderer == "replicator":
        label = "isaac-replicator"
    else:
        label = "blender" if engine == "eevee" else f"blender-{engine}"
    (clip / "renderer.txt").write_text(label + "\n")
    return dest


def prepare(clip: Path) -> dict:
    """Ground-truth files, final labels, and the oracle variant's features."""
    facts = groundtruth.convert(clip)
    labels = json.loads((clip / "labels.json").read_text())
    labels.update(facts)
    (clip / "labels.json").write_text(json.dumps(labels, indent=2))

    name = f"{clip.name}-oracle"
    shutil.copy(clip / "gt.poses.json", cfg.pose_path(name))
    shutil.copy(clip / "gt.camera.npz", cfg.POSES_DIR / f"{name}.camera.npz")
    raw = groundtruth.load_raw(clip)
    mt.ClipMeta(
        name=name, source=f"synthetic ground truth for {clip.name}",
        fps=float(tricks.OUTPUT_FPS), frames=len(raw["positions"]),
        width=int(raw["width"]), height=int(raw["height"]),
    ).save()
    _snow(["features", name])
    return labels


def _snow(args: list[str]) -> None:
    subprocess.run([sys.executable, "-m", "snow.cli", *args], check=True)


def pipeline(clip: Path) -> None:
    """The unchanged pipeline on the render: ingest, track, NIM, features.

    No slow-motion factor is passed, exactly as with real broadcast footage
    whose factor is unknown; totals are unaffected (invariant 8).
    """
    _snow(["run", str(clip / "render.mp4"), clip.name, "--features"])


def default_spacing() -> float:
    """Seconds between NIM calls: 120 on the hosted endpoint, none self-hosted.

    The hosted endpoint is one account and one key, and spacing the calls
    keeps a batch from tripping its rate limit. A self-hosted NIM
    (``SNOW_NIM_PREVIEW=0``) has no such limit.
    """
    return 120.0 if cfg.settings().preview else 0.0


def pipeline_all(selected: list[Path], spacing_s: float | None = None) -> None:
    """Every clip through the NIM, at least ``spacing_s`` between calls."""
    if spacing_s is None:
        spacing_s = default_spacing()
    last = 0.0
    for clip in selected:
        if cfg.pose_path(clip.name).exists():
            print(f"{clip.name}: poses exist, skipping the NIM")
            continue
        wait = spacing_s - (time.monotonic() - last)
        if last and wait > 0:
            print(f"waiting {wait:.0f}s before the next NIM call")
            time.sleep(wait)
        last = time.monotonic()
        try:
            pipeline(clip)
        except subprocess.CalledProcessError as exc:
            print(f"{clip.name}: pipeline failed ({exc}); continuing", file=sys.stderr)


def clip_set(clip: Path) -> str:
    """Which family a clip belongs to: the 20-trick validation set, or the
    training set with short (original) or long (terrain) ground time."""
    trick = json.loads((clip / "trick.json").read_text())
    if not trick.get("group"):
        return "validation"
    return "long_ground" if trick.get("terrain") else "short_ground"


def _plain(value):
    """numpy scalars as Python ones, so a cached ``False`` reads back false."""
    return value.item() if hasattr(value, "item") else str(value)


def results(selected: list[Path]) -> list[dict]:
    """Both variants' rows for every clip; also written to ``results.json``.

    Each clip's rows are cached in its ``compare.json``. A clip whose pose
    files have since been pruned to save disk (a batch keeps the NIM's poses
    but drops the ground-truth JSON) is read back from that cache, so the
    table always covers the whole set.
    """
    rows = []
    for clip in selected:
        if not (clip / "labels.json").exists():
            continue
        cache = clip / "compare.json"
        oracle = f"{clip.name}-oracle"
        if cfg.pose_path(oracle).exists() and cfg.pose_path(clip.name).exists():
            pair = [compare.row(clip, "oracle", oracle), compare.row(clip, "pipeline", clip.name)]
            # Only batch clips (they have a split group) are cached; the
            # validation set's directories are left exactly as they are.
            if json.loads((clip / "trick.json").read_text()).get("group"):
                cache.write_text(json.dumps(pair, indent=2, default=_plain))
        elif cache.exists():
            pair = json.loads(cache.read_text())
        else:
            pair = [compare.row(clip, "oracle", oracle), compare.row(clip, "pipeline", clip.name)]
        for r in pair:
            r["true_set"] = clip_set(clip)
        rows.extend(pair)
    (SYNTH_DIR / "results.json").write_text(json.dumps(rows, indent=2, default=_plain))
    return rows


MANIFEST = SYNTH_DIR / "MANIFEST.scale.jsonl"


def manifest_entry(clip: Path, status: str = "done", **extra) -> dict:
    """One manifest line: the clip, its labels, its split group, where it lives.

    ``status`` is ``done`` when the pipeline's poses and features exist,
    else whatever the caller says went wrong. Training reads only the
    pipeline's output (``data/poses/<name>.*``), never the oracle's.
    """
    labels = json.loads((clip / "labels.json").read_text())
    trick = json.loads((clip / "trick.json").read_text())
    name = clip.name
    outputs = {
        "poses": f"data/poses/{name}.json",
        "features": f"data/poses/{name}.features.parquet",
        "features_summary": f"data/poses/{name}.features.json",
        "camera": f"data/poses/{name}.camera.npz",
        "bbox": f"data/clips/{name}.bbox.txt",
        "meta": f"data/clips/{name}.meta.json",
        "labels": f"data/synth/{name}/labels.json",
        "motion": f"data/synth/{name}/motion.npz",
    }
    renderer = clip / "renderer.txt"
    return {
        "name": name,
        "status": status,
        "group": trick.get("group") or name,
        "renderer": renderer.read_text().strip() if renderer.exists() else None,
        "labels": labels,
        "files": outputs,
        **extra,
    }


def manifest_names(path: Path = MANIFEST) -> set[str]:
    """Clips already in the manifest, so appending never duplicates one."""
    if not path.exists():
        return set()
    return {json.loads(line)["name"] for line in path.read_text().splitlines() if line.strip()}


def manifest_append(entry: dict, path: Path = MANIFEST) -> None:
    """Append one line atomically enough for readers polling the file."""
    line = json.dumps(entry, default=str) + "\n"
    with path.open("a") as fh:
        fh.write(line)
        fh.flush()
        os.fsync(fh.fileno())
