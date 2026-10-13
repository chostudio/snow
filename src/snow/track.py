"""Detect and track the rider, and write the NIM's bounding-box annotation.

The 3D Body Pose NIM does not detect or track people — it needs a stable
tracked box per person per frame, which this module produces with
YOLO + ByteTrack.

Two things matter more than detection quality:

1. ``frame_id`` must be the NIM's *decoded* frame index. If the annotation
   mentions a frame the server did not decode, the request fails ``INTERNAL``.
   So frames are counted from the same file the NIM will receive, and clamped
   to the frame count ffprobe gets by decoding it.
2. A box with non-positive width or height is rejected unless it is exactly the
   absent sentinel ``-1 -1 -1 -1``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .config import MAX_BODIES, WEIGHTS_DIR
from .ingest import probe

ABSENT = "-1.000000 -1.000000 -1.000000 -1.000000"


class TrackError(RuntimeError):
    pass


@dataclass
class Track:
    raw_id: int
    boxes: dict[int, tuple[float, float, float, float]] = field(default_factory=dict)

    @property
    def frames_present(self) -> int:
        return len(self.boxes)

    @property
    def median_area(self) -> float:
        if not self.boxes:
            return 0.0
        return float(np.median([w * h for _, _, w, h in self.boxes.values()]))

    @property
    def score(self) -> float:
        # The rider is both persistently tracked and large in frame; a
        # spectator is usually one or the other, not both.
        return self.frames_present * np.sqrt(self.median_area)

    def span(self) -> tuple[int, int]:
        keys = self.boxes.keys()
        return min(keys), max(keys)


def _clip_box(
    x: float, y: float, w: float, h: float, width: int, height: int
) -> tuple[float, float, float, float] | None:
    """Clamp to the image and drop anything degenerate."""
    x0, y0 = max(0.0, x), max(0.0, y)
    x1, y1 = min(float(width), x + w), min(float(height), y + h)
    if x1 - x0 < 2.0 or y1 - y0 < 2.0:
        return None
    return x0, y0, x1 - x0, y1 - y0


def run_tracker(
    video: Path,
    weights: str = "yolo11s.pt",
    conf: float = 0.25,
    iou: float = 0.5,
    imgsz: int = 960,
    device: str | None = None,
    tracker: str = "bytetrack.yaml",
) -> tuple[dict[int, Track], int, int, int]:
    """Run detection+tracking over every frame.

    Returns (tracks by raw id, frames seen, width, height).
    """
    from ultralytics import YOLO

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    weights_path = WEIGHTS_DIR / weights
    model = YOLO(str(weights_path) if weights_path.exists() else weights)
    if not weights_path.exists():
        # Ultralytics downloads to CWD; keep the repo clean.
        downloaded = Path(weights)
        if downloaded.exists():
            downloaded.replace(weights_path)

    tracks: dict[int, Track] = {}
    frames = 0
    width = height = 0

    stream = model.track(
        source=str(video),
        stream=True,
        persist=True,
        tracker=tracker,
        classes=[0],  # person
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )
    for frame_id, result in enumerate(stream):
        frames = frame_id + 1
        if result.orig_shape:
            height, width = result.orig_shape
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        ids = boxes.id.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), raw_id in zip(xyxy, ids):
            clipped = _clip_box(
                float(x1), float(y1), float(x2 - x1), float(y2 - y1), width, height
            )
            if clipped is None:
                continue
            tracks.setdefault(int(raw_id), Track(int(raw_id))).boxes[frame_id] = clipped

    return tracks, frames, width, height


def save_tracks(
    tracks: dict[int, Track], frames: int, width: int, height: int, dest: Path
) -> None:
    """Cache every raw track so subject selection can be retried cheaply.

    Detection over a clip costs minutes; choosing which of its tracks is the
    rider is guesswork that wants several attempts. Keeping the raw result
    separates the two.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        json.dumps(
            {
                "frames": frames,
                "width": width,
                "height": height,
                "tracks": {
                    str(raw_id): {str(f): list(b) for f, b in t.boxes.items()}
                    for raw_id, t in tracks.items()
                },
            }
        )
    )


def load_tracks(src: Path) -> tuple[dict[int, Track], int, int, int]:
    data = json.loads(src.read_text())
    tracks: dict[int, Track] = {}
    for raw_id, boxes in data["tracks"].items():
        track = Track(int(raw_id))
        track.boxes = {int(f): tuple(b) for f, b in boxes.items()}
        tracks[int(raw_id)] = track
    return tracks, data["frames"], data["width"], data["height"]


def select(tracks: dict[int, Track], max_bodies: int, keep: list[int] | None) -> list[Track]:
    if keep:
        missing = [i for i in keep if i not in tracks]
        if missing:
            raise TrackError(
                f"requested track id(s) {missing} were never tracked; "
                f"available: {sorted(tracks)}"
            )
        return [tracks[i] for i in keep]
    ranked = sorted(tracks.values(), key=lambda t: t.score, reverse=True)
    return ranked[:max_bodies]


def merge(tracks: dict[int, Track], groups: list[list[int]]) -> list[Track]:
    """Stitch raw track ids into one body.

    A broadcast cut or a long occlusion makes ByteTrack issue the rider a fresh
    id, and the NIM wants one stable id per person. Which ids are the same
    rider is a judgement call best made by watching the preview, so it is an
    explicit argument rather than a heuristic.
    """
    merged: list[Track] = []
    for group in groups:
        missing = [i for i in group if i not in tracks]
        if missing:
            raise TrackError(
                f"track id(s) {missing} were never tracked; available: {sorted(tracks)}"
            )
        combined = Track(group[0])
        for raw_id in group:
            for frame_id, box in tracks[raw_id].boxes.items():
                # Earlier ids in the group win a contested frame.
                combined.boxes.setdefault(frame_id, box)
        merged.append(combined)
    return merged


def interpolate_gaps(track: Track, max_gap: int) -> int:
    """Fill short detection dropouts by linear interpolation.

    ByteTrack loses a rider for a frame or two mid-rotation; a hole there costs
    pose continuity through exactly the part of the trick we care about. Long
    gaps are left absent on purpose — the rider really is out of frame.
    """
    if max_gap <= 0 or not track.boxes:
        return 0
    present = sorted(track.boxes)
    filled = 0
    for a, b in zip(present, present[1:]):
        gap = b - a - 1
        if 0 < gap <= max_gap:
            box_a = np.array(track.boxes[a])
            box_b = np.array(track.boxes[b])
            for step in range(1, gap + 1):
                t = step / (gap + 1)
                track.boxes[a + step] = tuple(box_a * (1 - t) + box_b * t)
                filled += 1
    return filled


def write_bbox_file(
    tracks: list[Track], frame_count: int, dest: Path
) -> None:
    """Write the annotation: a header count, then a dense frame x body grid."""
    if not tracks:
        raise TrackError("no tracks to write — nobody was detected in the clip")
    if not 0 < len(tracks) <= MAX_BODIES:
        raise TrackError(f"number of bodies must be between 1 and {MAX_BODIES}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    lines = [str(len(tracks))]
    for frame_id in range(frame_count):
        for tracking_id, track in enumerate(tracks, start=1):
            box = track.boxes.get(frame_id)
            if box is None:
                lines.append(f"{frame_id} {tracking_id} {ABSENT}")
            else:
                x, y, w, h = box
                lines.append(
                    f"{frame_id} {tracking_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
                )
    dest.write_text("\n".join(lines) + "\n")


def render_preview(video: Path, tracks: list[Track], dest: Path) -> None:
    """Draw the selected boxes so the subject choice can be eyeballed."""
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dest.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(dest), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    palette = [(0, 255, 0), (0, 180, 255), (255, 128, 0), (255, 0, 255)]
    frame_id = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for tracking_id, track in enumerate(tracks, start=1):
            box = track.boxes.get(frame_id)
            if box is None:
                continue
            x, y, bw, bh = (int(round(v)) for v in box)
            color = palette[(tracking_id - 1) % len(palette)]
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
            cv2.putText(
                frame, f"id {tracking_id}", (x, max(18, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA,
            )
        writer.write(frame)
        frame_id += 1
    cap.release()
    writer.release()


def build(
    video: Path,
    dest: Path,
    max_bodies: int = 1,
    keep: list[int] | None = None,
    max_gap: int = 8,
    preview: Path | None = None,
    cache: Path | None = None,
    retrack: bool = False,
    merge_groups: list[list[int]] | None = None,
    **tracker_kwargs,
) -> dict:
    info = probe(video)
    if cache and cache.exists() and not retrack:
        tracks, decoded, width, height = load_tracks(cache)
        print(f"  reusing cached tracks from {cache.name} (--retrack to redo)")
    else:
        tracks, decoded, width, height = run_tracker(video, **tracker_kwargs)
        if cache:
            save_tracks(tracks, decoded, width, height, cache)
    if not tracks:
        raise TrackError(
            f"no people detected in {video.name}. Try --conf 0.15 or a larger "
            "--weights (yolo11m.pt)."
        )

    # The NIM must not be told about a frame it will not decode.
    frame_count = min(decoded, info.frames) if info.frames else decoded
    if info.frames and decoded != info.frames:
        print(
            f"  note: decoder saw {decoded} frames, ffprobe counts {info.frames}; "
            f"annotating {frame_count}"
        )

    if merge_groups:
        selected = merge(tracks, merge_groups)
    else:
        selected = select(tracks, max_bodies, keep)
    filled = 0
    for track in selected:
        # Drop anything past the clamp before interpolating across it.
        for frame_id in [f for f in track.boxes if f >= frame_count]:
            del track.boxes[frame_id]
        filled += interpolate_gaps(track, max_gap)

    selected = [t for t in selected if t.boxes]
    if not selected:
        raise TrackError("every selected track was empty after clamping to frame count")

    write_bbox_file(selected, frame_count, dest)
    if preview:
        render_preview(video, selected, preview)

    return {
        "frames": frame_count,
        "width": width,
        "height": height,
        "candidates": len(tracks),
        "selected": [
            {
                "tracking_id": i,
                "raw_id": t.raw_id,
                "frames_present": t.frames_present,
                "coverage": t.frames_present / frame_count if frame_count else 0.0,
                "span": t.span(),
                "median_area_px": round(t.median_area),
            }
            for i, t in enumerate(selected, start=1)
        ],
        "interpolated_boxes": filled,
        "top_candidates": [
            {
                "raw_id": t.raw_id,
                "frames_present": t.frames_present,
                "span": t.span(),
                "median_area_px": round(t.median_area),
            }
            for t in sorted(tracks.values(), key=lambda t: t.score, reverse=True)[:12]
        ],
    }
