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


def plan(seed: int = 0) -> list[Path]:
    """Write ``trick.json`` + ``motion.npz`` for the 20-trick validation set."""
    out = []
    for trick in tricks.plan(seed):
        clip = SYNTH_DIR / trick.name
        clip.mkdir(parents=True, exist_ok=True)
        (clip / "trick.json").write_text(trick.to_json())
        motion = tricks.motion(trick)
        motion.save(clip / "motion.npz")
        labels = tricks.labels(trick, motion)
        (clip / "labels.json").write_text(json.dumps(labels, indent=2))
        out.append(clip)
    return out


def render(clip: Path, width: int, height: int, samples: int) -> None:
    """Build, record and render in a headless Blender (needs MPFB installed)."""
    frames = clip / "frames"
    if frames.exists():
        shutil.rmtree(frames)
    cmd = [BLENDER, "-b", "--python", str(RUNNER), "--", str(clip),
           "--width", str(width), "--height", str(height), "--samples", str(samples)]
    log = (clip / "blender.log").open("w")
    result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0 or not (clip / "gt_raw.npz").exists():
        raise SystemExit(f"blender failed on {clip.name}; see {clip / 'blender.log'}")


def encode(clip: Path, renderer: str = "blender") -> Path:
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
    (clip / "renderer.txt").write_text(renderer + "\n")
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


def pipeline_all(selected: list[Path], spacing_s: float) -> None:
    """Every clip through the NIM, at least ``spacing_s`` between calls.

    One account, one key: spacing the calls out keeps a 20-clip run from
    tripping the hosted endpoint's rate limit.
    """
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


def results(selected: list[Path]) -> list[dict]:
    rows = []
    for clip in selected:
        if not (clip / "labels.json").exists():
            continue
        rows.append(compare.row(clip, "oracle", f"{clip.name}-oracle"))
        rows.append(compare.row(clip, "pipeline", clip.name))
    (SYNTH_DIR / "results.json").write_text(json.dumps(rows, indent=2, default=str))
    return rows
