"""Send a clip and its tracked boxes to the 3D Body Pose NIM."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from .config import Settings, pose_path


class PoseError(RuntimeError):
    pass


def build_argv(
    settings: Settings,
    video: Path,
    bbox: Path,
    out: Path,
    overlay: Path | None = None,
    focal_length: float = 0.0,
    timeout: float = 3600.0,
    session_id: str | None = None,
) -> list[str]:
    argv = [
        "--target", settings.target,
        "--video-input", str(video),
        "--bbox-input", str(bbox),
        "--output", str(out),
        "--focal-length", str(focal_length),
        "--timeout", str(timeout),
    ]
    if settings.preview:
        argv += ["--preview-mode", "--function-id", settings.function_id]
    if overlay:
        argv += ["--overlay-output", str(overlay)]
    if session_id:
        argv += ["--client-session-id", session_id]
    return argv


def estimate(
    settings: Settings,
    video: Path,
    bbox: Path,
    out: Path,
    overlay: Path | None = None,
    focal_length: float = 0.0,
    timeout: float = 3600.0,
    session_id: str | None = None,
) -> Path:
    """Run inference. Returns the pose file on success."""
    for path in (video, bbox):
        if not path.exists():
            raise PoseError(f"missing input: {path}")
    if out.suffix.lower() != ".json":
        raise PoseError("the upstream client requires the output path to end in .json")
    out.parent.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    if settings.preview:
        env["SNOW_NIM_API_KEY"] = settings.require_key()

    argv = build_argv(
        settings, video, bbox, out, overlay, focal_length, timeout, session_id
    )
    cmd = [sys.executable, "-m", "snow._nim_launcher", *argv]
    proc = subprocess.run(cmd, env=env)

    partial = out.with_suffix(out.suffix + ".partial")
    if proc.returncode != 0:
        hint = ""
        if partial.exists():
            hint = f"\nA partial result was kept at {partial}"
        raise PoseError(f"the NIM client exited with status {proc.returncode}{hint}")
    if not out.exists():
        raise PoseError(f"the client reported success but {out} was not written")
    return out


def summarize(path: Path) -> dict:
    """Read back the JSON Lines result and report what actually came through."""
    frames = 0
    detections = 0
    per_track: dict[int, int] = {}
    joint_counts: set[int] = set()
    first_frame = last_frame = None
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            frames += 1
            frame_id = record.get("frame_id")
            if first_frame is None:
                first_frame = frame_id
            last_frame = frame_id
            for det in record.get("detections", []):
                detections += 1
                tid = det.get("tracking_id")
                per_track[tid] = per_track.get(tid, 0) + 1
                if det.get("keypoints_3d"):
                    joint_counts.add(len(det["keypoints_3d"]))
    return {
        "frames_with_pose": frames,
        "detections": detections,
        "per_track": per_track,
        "joints": sorted(joint_counts),
        "frame_range": (first_frame, last_frame),
    }
