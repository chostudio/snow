"""Find the segments of a broadcast worth running pose estimation on.

Televised runs are cut between many cameras, and the good material — the
super-slow-motion replay from a high-speed camera, where the rider is large,
sharp and moving slowly — is a handful of seconds buried in minutes of wide
shots, crowd cutaways and interviews.

This walks a video once at low resolution and scores every shot on:

* **motion** — mean optical flow. A slow-motion replay has a fraction of the
  frame-to-frame motion of a real-time shot of the same action.
* **duplicate frames** — slow motion produced by duplicating frames rather
  than by a high-speed camera shows up here, and the duplicate ratio gives
  the conform factor exactly.
* **subject size** — how tall the largest person is as a fraction of frame
  height.
* **whether that person is a whole body** — measured with a pose model rather
  than a plain detector, because "a person fills the frame" is equally true of
  a face close-up, and an interview portrait is useless for body pose. A shot
  only counts if ankles and hips are actually visible.

It prints ready-to-paste ``--start``/``--duration`` arguments. It does not
decide for you; slow motion and a genuinely slow real-time shot look alike to
any one of these signals, which is why all three are reported.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .config import WEIGHTS_DIR


@dataclass
class Shot:
    index: int
    start_frame: int
    end_frame: int
    fps: float
    flow: float = 0.0
    duplicate_ratio: float = 0.0
    max_person_height: float = 0.0
    median_person_height: float = 0.0
    person_frames: int = 0
    fullbody_frames: int = 0
    sampled: int = 0
    _heights: list[float] = field(default_factory=list, repr=False)

    @property
    def frames(self) -> int:
        return self.end_frame - self.start_frame + 1

    @property
    def seconds(self) -> float:
        return self.frames / self.fps if self.fps else 0.0

    @property
    def start_time(self) -> float:
        return self.start_frame / self.fps if self.fps else 0.0

    def timecode(self) -> str:
        t = self.start_time
        return f"{int(t // 3600):02d}:{int(t % 3600 // 60):02d}:{t % 60:06.3f}"

    @property
    def person_ratio(self) -> float:
        return self.person_frames / self.sampled if self.sampled else 0.0

    @property
    def fullbody_ratio(self) -> float:
        return self.fullbody_frames / self.sampled if self.sampled else 0.0

    def score(self, flow_reference: float) -> float:
        """Rank shots by how useful they are for pose estimation.

        Wants a big *whole* subject, consistently present, moving slowly
        relative to the rest of this video. Very short shots are penalised: the
        NIM buffers roughly 120 frames before it emits anything, so a two
        second cut is not worth sending on its own.
        """
        if self.median_person_height <= 0 or self.frames < 60:
            return 0.0
        slowness = flow_reference / (self.flow + 1e-6)
        return (
            self.median_person_height
            * self.fullbody_ratio
            * min(slowness, 6.0)
            * min(self.frames / 150.0, 1.0)
        )


def _detect_cuts(diffs: np.ndarray, threshold: float) -> list[int]:
    """Frame indices where a hard cut begins, from the frame-difference signal."""
    return [int(i) + 1 for i in np.nonzero(diffs > threshold)[0]]


def analyze(
    video: Path,
    sample_every: int = 15,
    cut_threshold: float = 22.0,
    duplicate_threshold: float = 0.35,
    min_shot_frames: int = 45,
    weights: str = "yolo11s-pose.pt",
    conf: float = 0.2,
    imgsz: int = 640,
    device: str | None = None,
    progress: bool = True,
) -> tuple[list[Shot], dict]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    small: list[np.ndarray] = []
    diffs: list[float] = []
    flows: list[float] = []
    keep: dict[int, np.ndarray] = {}

    prev_small = None
    frame_id = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(
            cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2GRAY,
        )
        if prev_small is not None:
            diffs.append(float(np.mean(cv2.absdiff(gray, prev_small))))
            flow = cv2.calcOpticalFlowFarneback(
                prev_small, gray, None, 0.5, 2, 15, 2, 5, 1.2, 0
            )
            flows.append(float(np.mean(np.linalg.norm(flow, axis=2))))
        if frame_id % sample_every == 0:
            keep[frame_id] = frame
        prev_small = gray
        frame_id += 1
        if progress and frame_id % 600 == 0:
            print(f"    scanned {frame_id} frames")
    cap.release()
    total = frame_id
    if total < 2:
        raise RuntimeError(f"{video}: too few frames")

    diff_arr = np.array(diffs)
    flow_arr = np.array(flows)

    # Build shots from detected cuts, dropping fragments.
    bounds = [0, *_detect_cuts(diff_arr, cut_threshold), total]
    bounds = sorted(set(b for b in bounds if 0 <= b <= total))
    shots: list[Shot] = []
    for start, end in zip(bounds, bounds[1:]):
        if end - start < min_shot_frames:
            continue
        shots.append(Shot(len(shots), start, end - 1, fps))
    if not shots:
        shots = [Shot(0, 0, total - 1, fps)]

    for shot in shots:
        # diffs[i] compares frame i+1 with i.
        lo, hi = max(shot.start_frame, 1), shot.end_frame
        if hi > lo:
            seg_diff = diff_arr[lo - 1 : hi - 1]
            seg_flow = flow_arr[lo - 1 : hi - 1]
            if seg_flow.size:
                shot.flow = float(np.median(seg_flow))
            if seg_diff.size:
                shot.duplicate_ratio = float(np.mean(seg_diff < duplicate_threshold))

    _measure_subjects(shots, keep, weights, conf, imgsz, device)

    moving = [s.flow for s in shots if s.flow > 0]
    reference = float(np.median(moving)) if moving else 1.0
    shots.sort(key=lambda s: s.score(reference), reverse=True)
    return shots, {
        "frames": total,
        "fps": fps,
        "seconds": total / fps if fps else 0.0,
        "flow_reference": reference,
        "shots_found": len(shots),
    }


def _measure_subjects(
    shots: list[Shot],
    frames: dict[int, np.ndarray],
    weights: str,
    conf: float,
    imgsz: int,
    device: str | None,
) -> None:
    """Measure subject size and whether the subject is a whole body.

    Uses a pose model: COCO keypoints 11/12 are the hips and 15/16 the ankles,
    so requiring a hip and an ankle separates a rider in frame from a talking
    head that happens to be large.
    """
    from ultralytics import YOLO

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    local = WEIGHTS_DIR / weights
    model = YOLO(str(local) if local.exists() else weights)
    if not local.exists() and Path(weights).exists():
        Path(weights).replace(local)

    by_shot: dict[int, list[int]] = {}
    for frame_id in frames:
        for shot in shots:
            if shot.start_frame <= frame_id <= shot.end_frame:
                by_shot.setdefault(shot.index, []).append(frame_id)
                break

    if getattr(model, "task", None) != "pose":
        raise ValueError(
            f"{weights} is a '{getattr(model, 'task', 'unknown')}' model; "
            "scouting needs a pose model (e.g. yolo11s-pose.pt) to tell a "
            "full-body shot from a face close-up"
        )

    index = {s.index: s for s in shots}
    hips, ankles = (11, 12), (15, 16)
    for shot_index, frame_ids in by_shot.items():
        shot = index[shot_index]
        batch = [frames[f] for f in sorted(frame_ids)]
        shot.sampled = len(batch)
        for result in model.predict(
            batch, classes=[0], conf=conf, imgsz=imgsz,
            device=device, verbose=False,
        ):
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            frame_height = result.orig_shape[0]
            heights = (boxes.xyxy[:, 3] - boxes.xyxy[:, 1]).cpu().numpy() / frame_height
            tallest = int(np.argmax(heights))
            shot.person_frames += 1
            shot._heights.append(float(heights[tallest]))

            keypoints = getattr(result, "keypoints", None)
            if keypoints is None or keypoints.conf is None:
                continue
            visible = keypoints.conf[tallest].cpu().numpy() > 0.5
            if any(visible[i] for i in hips) and any(visible[i] for i in ankles):
                shot.fullbody_frames += 1
    for shot in shots:
        if shot._heights:
            shot.max_person_height = max(shot._heights)
            shot.median_person_height = float(np.median(shot._heights))


def report(shots: list[Shot], stats: dict, top: int) -> str:
    lines = [
        f"{stats['frames']} frames, {stats['seconds']:.1f}s, "
        f"{stats['shots_found']} shots, median flow {stats['flow_reference']:.2f}px/frame",
        "",
        f"{'#':>3} {'start':>12} {'len':>7} {'flow':>7} {'rel':>5} "
        f"{'dup':>5} {'subj':>5} {'body':>5} {'seen':>5}  suggested command",
    ]
    reference = stats["flow_reference"]
    for shot in shots[:top]:
        relative = reference / (shot.flow + 1e-6)
        marks = []
        if relative > 2.0:
            marks.append("SLOW")
        if shot.duplicate_ratio > 0.25:
            marks.append(f"DUP{shot.duplicate_ratio:.0%}")
        if shot.frames < 120:
            marks.append("SHORT")
        lines.append(
            f"{shot.index:>3} {shot.timecode():>12} {shot.seconds:>6.1f}s "
            f"{shot.flow:>7.2f} {relative:>5.1f} {shot.duplicate_ratio:>5.0%} "
            f"{shot.median_person_height:>5.2f} {shot.fullbody_ratio:>5.0%} "
            f"{shot.person_ratio:>5.0%}  "
            f"--start {shot.timecode()} --duration {shot.seconds:.1f}"
            + ("   " + " ".join(marks) if marks else "")
        )
    lines += [
        "",
        "flow = median optical flow px/frame; rel = how much slower than this "
        "video's median shot",
        "dup  = fraction of near-identical frames (frame-doubled slow motion)",
        "subj = median height of the largest person, as a fraction of the frame",
        "body = fraction of sampled frames where that person shows hips AND "
        "ankles",
        "       (a low body with a high subj is a face close-up, not a rider)",
        "seen = fraction of sampled frames containing a person",
        "",
        "A shot with rel > 2, a large subj and a high body is a slow-motion "
        "replay worth using.",
        "SHORT marks shots under 120 frames; the NIM buffers about that many "
        "before",
        "it emits anything, so widen the window around one.",
        "Pass --slowmo-factor to `snow ingest` so rates stay in real time.",
    ]
    return "\n".join(lines)


def save(shots: list[Shot], stats: dict, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stats": stats,
        "shots": [
            {
                "index": s.index,
                "start_frame": s.start_frame,
                "end_frame": s.end_frame,
                "start_time": s.start_time,
                "seconds": s.seconds,
                "flow": s.flow,
                "duplicate_ratio": s.duplicate_ratio,
                "median_person_height": s.median_person_height,
                "fullbody_ratio": s.fullbody_ratio,
                "person_ratio": s.person_ratio,
                "score": s.score(stats["flow_reference"]),
            }
            for s in shots
        ],
    }
    dest.write_text(json.dumps(payload, indent=2))
