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

from .camera import _colour_histogram
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


def _model(weights: str):
    from ultralytics import YOLO

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    weights_path = WEIGHTS_DIR / weights
    model = YOLO(str(weights_path) if weights_path.exists() else weights)
    if not weights_path.exists():
        # Ultralytics downloads to CWD; keep the repo clean.
        downloaded = Path(weights)
        if downloaded.exists():
            downloaded.replace(weights_path)
    return model


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
    model = _model(weights)
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


def view_changes(video: Path, threshold: float = 0.35) -> set[int]:
    """Frames whose view differs from the previous frame's: likely cuts.

    The colour-histogram half of ``camera.py``'s cut test. Alone it is enough
    to stop the two steps below from joining boxes across a cut, which is the
    one mistake they must never make.
    """
    cap = cv2.VideoCapture(str(video))
    cuts: set[int] = set()
    prev = None
    frame_id = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        hist = _colour_histogram(frame)
        if prev is not None and cv2.compareHist(
            prev, hist, cv2.HISTCMP_BHATTACHARYYA
        ) > threshold:
            cuts.add(frame_id)
        prev = hist
        frame_id += 1
    cap.release()
    return cuts


def _crosses(cuts: set[int], a: int, b: int) -> bool:
    """Whether a cut falls in frames ``a+1 .. b``."""
    return any(a < c <= b for c in cuts)


def _continues(
    box_a: tuple, box_b: tuple, max_jump: float
) -> bool:
    """Could ``box_b`` be the same body as ``box_a`` shortly after?

    Uses the diagonal, not width or height: a rider turning upside down swaps
    the box's aspect but keeps its diagonal.
    """
    (xa, ya, wa, ha), (xb, yb, wb, hb) = box_a, box_b
    da, db = np.hypot(wa, ha), np.hypot(wb, hb)
    if not 0.5 <= db / da <= 2.0:
        return False
    jump = np.hypot((xa + wa / 2) - (xb + wb / 2), (ya + ha / 2) - (yb + hb / 2))
    return jump <= max_jump * max(da, db)


def stitch(
    tracks: dict[int, Track],
    rider: Track,
    cuts: set[int],
    max_gap: int = 60,
    max_jump: float = 1.0,
    max_overlap: float = 0.1,
) -> list[int]:
    """Absorb the fragments ByteTrack split the rider into within one shot.

    An inverted rider is detected intermittently, and ByteTrack often issues a
    new id when detection resumes, so one trick arrives as three or four ids
    and selection keeps only the largest. A fragment is absorbed only if it
    (a) barely overlaps the rider's frames in time, so it is not another
    person on screen at the same moment, (b) starts or ends within
    ``max_gap`` frames of the rider's track, at a box that continues the
    rider's, and (c) is not separated from it by a cut. The gap is generous
    because slow motion stretches a dropout: 40-50 frames between fragments
    at 4x. The broadcast camera keeps the rider in the same part of the
    picture, so the box test still discriminates. Joining across cuts
    stays manual (``--merge``).
    """
    absorbed: list[int] = []
    changed = True
    while changed:
        changed = False
        for raw_id, other in sorted(tracks.items()):
            if raw_id == rider.raw_id or raw_id in absorbed or not other.boxes:
                continue
            frames = set(other.boxes)
            if len(frames & rider.boxes.keys()) > max_overlap * len(frames):
                continue
            start, end = min(frames), max(frames)
            mine = sorted(set(rider.boxes) - frames)
            before = [f for f in mine if start - max_gap <= f < start]
            after = [f for f in mine if end < f <= end + max_gap]
            linked = (
                before and not _crosses(cuts, before[-1], start)
                and _continues(rider.boxes[before[-1]], other.boxes[start], max_jump)
            ) or (
                after and not _crosses(cuts, end, after[0])
                and _continues(other.boxes[end], rider.boxes[after[0]], max_jump)
            )
            if linked:
                for frame_id, box in other.boxes.items():
                    rider.boxes.setdefault(frame_id, box)
                absorbed.append(raw_id)
                changed = True
    return absorbed


# Rotations to try a crop at, and how to map a point in the rotated crop
# (u, v) back to the crop (x, y), for a crop of size (w, h).
_ROTATIONS = [
    (None, lambda u, v, w, h: (u, v)),
    (cv2.ROTATE_90_CLOCKWISE, lambda u, v, w, h: (v, h - u)),
    (cv2.ROTATE_180, lambda u, v, w, h: (w - u, h - v)),
    (cv2.ROTATE_90_COUNTERCLOCKWISE, lambda u, v, w, h: (w - v, u)),
]


def _detect_rotated(model, crop: np.ndarray, conf: float, imgsz: int) -> list[tuple]:
    """People in ``crop`` at four rotations, as (conf, x, y, w, h) in the crop."""
    h, w = crop.shape[:2]
    images = [crop if rot is None else cv2.rotate(crop, rot) for rot, _ in _ROTATIONS]
    results = model.predict(images, classes=[0], conf=conf, imgsz=imgsz, verbose=False)
    found = []
    for (_, back), result in zip(_ROTATIONS, results):
        if result.boxes is None:
            continue
        for (u0, v0, u1, v1), score in zip(
            result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()
        ):
            corners = np.array([back(u, v, w, h) for u in (u0, u1) for v in (v0, v1)])
            (x0, y0), (x1, y1) = corners.min(0), corners.max(0)
            found.append((float(score), x0, y0, x1 - x0, y1 - y0))
    return found


def recover_gaps(
    video: Path,
    track: Track,
    cuts: set[int],
    model,
    conf: float = 0.15,
    imgsz: int = 640,
    search: float = 3.0,
    min_side: int = 320,
    reach: int = 150,
    patience: int = 30,
    others: dict[int, Track] | None = None,
) -> int:
    """Look again for the rider where the track has a hole, rotated.

    COCO-trained detectors see people upright. On synthetic big air with the
    true rider box known, YOLO found the rider in 87% of upright airborne
    frames, 35% of sideways ones and 28% of inverted ones, so the hole is
    exactly where a cork is. Frames without a box, up to ``reach`` frames
    from one, are searched in a window around where the rider should be, at
    0, 90, 180 and 270 degrees; the window is also upscaled to ``imgsz``,
    which helps a small rider.

    The search walks a chain out from each detection: every box found
    anchors the next frame, both where it was and carried on at the rider's
    recent velocity (``_ahead``). The velocity is what follows a halfpipe
    rider across the picture; on big air the camera holds the rider still
    and the velocity is mostly the jitter of a tumbling body's box, which
    alone would steer the anchor off them. A candidate must be where the
    rider could have got to from either anchor, sized like the rider's own
    detections (``_reachable``), and not on a raw track that has been
    somewhere else at the same moment as the rider (``_someone_else``); of
    those, the nearest wins. A chain gives up
    ``patience`` frames after its last box, because by then its anchor is a
    guess and its allowance wide, and resuming is how a chain picks up a
    bystander; it never crosses a cut.

    That is not enough on a halfpipe: in a fast pan the crowd at the lip
    sweeps past at the rider's speed, and a chain can step onto a spectator
    and follow them. So in a hole between two detections, a chain walks in
    from each side and is kept only if it arrives where the detection on the
    far side is (``joins``); one that wandered off never does. A chain
    with nothing to arrive at (the track's ends, or a cut) is kept as is.
    """
    if not track.boxes:
        return 0
    original = sorted(track.boxes)
    size = {f: float(np.hypot(*track.boxes[f][2:])) for f in original}
    frames = _frames_near(video, original, reach)
    if not frames:
        return 0
    others = [o for o in (others or {}).values() if o is not track]

    def rider_size(frame_id: int) -> float | None:
        """The rider's box diagonal here, from detections in the same shot."""
        near = [f for f in original if abs(f - frame_id) <= reach
                and not _crosses(cuts, min(f, frame_id), max(f, frame_id))]
        if not near:
            return None
        near.sort(key=lambda f: abs(f - frame_id))
        return float(np.median([size[f] for f in near[:15]]))

    def chain(start: int, stop: int, toward: int | None = None) -> dict[int, tuple]:
        """Walk from detection ``start`` toward ``stop`` (exclusive)."""
        step = 1 if stop > start else -1
        boxes = dict(track.boxes)
        found: dict[int, tuple] = {}
        last = start
        for frame_id in range(start + step, stop, step):
            if frame_id not in frames:
                continue
            if abs(frame_id - last) > patience or _crosses(
                cuts, min(last, frame_id), max(last, frame_id)
            ):
                break
            reference = rider_size(frame_id)
            if reference is None:
                break
            age = abs(frame_id - last)
            anchors = [_ahead(boxes, last, frame_id, cuts), boxes[last]]
            if toward is not None:
                # Lean toward the detection on the far side of the hole.
                t = age / abs(toward - last)
                anchors = [tuple(np.array(a) * (1 - t) + np.array(boxes[toward]) * t)
                           for a in anchors]
                age = min(age, abs(toward - frame_id))
            taken = [o.boxes[frame_id] for o in others
                     if frame_id in o.boxes and _someone_else(o.boxes, boxes)]
            box = _search(frames[frame_id], anchors, age, reference, model,
                          conf, imgsz, search, min_side, taken)
            if box is not None:
                boxes[frame_id] = found[frame_id] = box
                last = frame_id
        return found

    def joins(found: dict[int, tuple], end: int) -> bool:
        """Does the chain arrive where the detection at ``end`` is?"""
        if not found:
            return False
        last = max(found) if end > max(found) else min(found)
        anchors = [_ahead({**track.boxes, **found}, last, end, cuts), found[last]]
        return any(_reachable(a, track.boxes[end], abs(end - last), size[end])
                   for a in anchors)

    accepted: dict[int, tuple] = {}
    for prev, nxt in zip(original, original[1:]):
        if nxt - prev < 2:
            continue
        if _crosses(cuts, prev, nxt):
            accepted.update(chain(prev, nxt))
            accepted.update(chain(nxt, prev))
            continue
        for start, end in ((prev, nxt), (nxt, prev)):
            found = chain(start, end, toward=end)
            if joins(found, end):
                for frame_id, box in found.items():
                    accepted.setdefault(frame_id, box)
    accepted.update(chain(original[0], min(frames) - 1))
    accepted.update(chain(original[-1], max(frames) + 1))
    track.boxes.update(accepted)
    return len(accepted)


def _ahead(boxes: dict[int, tuple], source: int, frame_id: int, cuts: set[int],
           window: int = 6) -> tuple:
    """``source``'s box moved on to ``frame_id`` at the track's recent velocity.

    The velocity is the median per-frame step of the box centre over the
    boxes up to ``window`` frames behind ``source`` (behind in the direction
    of travel), within the shot. A rider leaving the top of the frame keeps
    going up; the spectator below the lip does not.
    """
    step = 1 if frame_id > source else -1
    trail = [f for f in range(source - step * window, source + step, step)
             if f in boxes and not _crosses(cuts, min(f, source), max(f, source))]
    x, y, w, h = boxes[source]
    if len(trail) < 2:
        return x, y, w, h
    centres = np.array([[b[0] + b[2] / 2, b[1] + b[3] / 2]
                        for b in (boxes[f] for f in trail)])
    gaps = np.abs(np.diff(trail))[:, None]
    velocity = np.median(np.diff(centres, axis=0) / gaps, axis=0)
    dx, dy = velocity * abs(frame_id - source)
    return x + dx, y + dy, w, h


def _reachable(anchor: tuple, box: tuple, age: int, reference: float) -> bool:
    """Could the rider have moved from ``anchor`` to ``box`` in ``age`` frames?

    On the synthetic set the rider's box centre moves 0.03 diagonals a frame
    (median), 0.16 at the 99th percentile, and a halfpipe rider 0.09-0.14.
    The allowance grows with the frames since the anchor's box and is
    capped at 0.3 diagonals, twice the rider's 99th percentile step: past a
    few frames the anchor is a guess, and a wide allowance is how a chain
    that resumes after a miss picks up a bystander. Size is
    judged against ``reference``, the rider's detected size nearby: a tucked
    rider's box is as small as 0.54 of a standing one.
    """
    (xa, ya, wa, ha), (xb, yb, wb, hb) = anchor, box
    if not 0.5 <= np.hypot(wb, hb) / reference <= 1.6:
        return False
    jump = np.hypot((xa + wa / 2) - (xb + wb / 2), (ya + ha / 2) - (yb + hb / 2))
    return jump <= min(0.1 + 0.1 * age, 0.3) * reference


class _Encoded:
    """A frame held as JPEG, so a few hundred of them fit in memory."""

    def __init__(self, frame: np.ndarray):
        self.data = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])[1]

    def decode(self) -> np.ndarray:
        return cv2.imdecode(self.data, cv2.IMREAD_COLOR)


def _frames_near(video: Path, present: list[int], reach: int) -> dict[int, _Encoded]:
    """Every frame without a box within ``reach`` frames of one."""
    have = set(present)
    marks = np.array(present)
    cap = cv2.VideoCapture(str(video))
    frames: dict[int, _Encoded] = {}
    frame_id = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_id not in have and np.min(np.abs(marks - frame_id)) <= reach:
            frames[frame_id] = _Encoded(frame)
        frame_id += 1
    cap.release()
    return frames


def _someone_else(other: dict[int, tuple], rider: dict[int, tuple], frames: int = 2) -> bool:
    """Has ``other`` been in a different place from ``rider`` at the same time?

    Then it is another person. An unstitched fragment of the rider passes:
    wherever both have a box, the boxes coincide.
    """
    apart = 0
    for frame_id in other.keys() & rider.keys():
        if _iou(other[frame_id], rider[frame_id]) < 0.3:
            apart += 1
            if apart >= frames:
                return True
    return False


def _iou(a: tuple, b: tuple) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / (aw * ah + bw * bh - inter)


def _search(encoded, anchors, age, reference, model, conf, imgsz, search, min_side,
            taken):
    """The person box nearest the anchors that the rider could be, at any rotation."""
    frame = encoded.decode()
    height, width = frame.shape[:2]
    centres = np.array([(x + w / 2, y + h / 2) for x, y, w, h in anchors])
    cx, cy = centres.mean(0)
    spread = np.ptp(centres, axis=0).max()
    side = max(search * max(max(w, h) for _, _, w, h in anchors) + spread, min_side)
    x0, y0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    x1, y1 = int(min(width, cx + side / 2)), int(min(height, cy + side / 2))
    if x1 - x0 < 16 or y1 - y0 < 16:
        return None
    best = None
    for _, bx, by, bw, bh in _detect_rotated(model, frame[y0:y1, x0:x1], conf, imgsz):
        box = (bx + x0, by + y0, bw, bh)
        if not any(_reachable(a, box, age, reference) for a in anchors):
            continue
        if any(_iou(box, other) > 0.3 for other in taken):
            continue
        distance = np.min(np.hypot(*(centres - (box[0] + bw / 2, box[1] + bh / 2)).T))
        if best is None or distance < best[0]:
            best = (distance, box)
    if best is None:
        return None
    return _clip_box(*best[1], width, height)

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
    auto_stitch: bool = True,
    recover: bool = True,
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
    # Ids chosen by hand are the user's call; only automatic selection is
    # stitched.
    stitching = auto_stitch and not (merge_groups or keep)
    cuts = view_changes(video) if (recover or stitching) else set()
    absorbed: list[int] = []

    def stitch_into(track: Track) -> list[int]:
        if not stitching:
            return []
        # Never hand one body's fragment, or another body, to a second.
        taken = {t.raw_id for t in selected if t is not track} | set(absorbed)
        more = stitch({i: t for i, t in tracks.items() if i not in taken}, track, cuts)
        absorbed.extend(more)
        return more

    recovered = 0
    model = _model(tracker_kwargs.get("weights", "yolo11s.pt")) if recover else None
    for track in selected:
        stitch_into(track)
        if recover:
            recovered += recover_gaps(video, track, cuts, model, others=tracks)
            # Recovery can bring the track within reach of a fragment that
            # was too far away before.
            if stitch_into(track):
                recovered += recover_gaps(video, track, cuts, model, others=tracks)
        # Drop anything past the clamp before interpolating across it.
        for frame_id in [f for f in track.boxes if f >= frame_count]:
            del track.boxes[frame_id]
    filled = 0
    for track in selected:
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
        "stitched_ids": absorbed,
        "recovered_boxes": recovered,
        "cuts": sorted(cuts),
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
