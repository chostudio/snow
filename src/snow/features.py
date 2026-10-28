"""Turn raw NIM poses into a per-frame table for the DS/ML work.

The central problem is that everything the NIM returns is in **camera**
coordinates, and the camera is itself rotating to follow the rider. This
module builds a body frame from the 3D keypoints, removes the camera's
rotation (see ``camera.py``), and integrates what is left into rotation
measures that mean something.

Four rotation measures are produced, because no single one covers every
trick:

``spin_deg``
    Cumulative yaw about the world vertical — the number people mean by "540"
    or "1080". Read from the hip axis, which degenerates when the rider is
    inverted and the hip axis points at the vertical; ``spin_quality`` says
    how well conditioned each frame was, and frames below ``--min-quality``
    are interpolated across rather than trusted.

``rotation_deg``
    Cumulative *total* angular path of the body frame, regardless of axis.
    Robust where ``spin_deg`` degenerates, and it counts flips and corks,
    but it cannot distinguish a flip from a spin.

``axis_spin_deg``
    Cumulative rotation about the trick's **own dominant axis**, found by
    summing the per-frame rotation vectors over a segment and normalising.
    ``spin_axis_tilt_deg`` reports how far that axis sits from vertical,
    which is itself a measure of how corked the trick was.

    Worth being precise about why this exists, because the obvious reason is
    wrong. For a clean rigid rotation about *any* fixed axis, the horizontal
    projection winds once per revolution, so ``spin_deg`` and
    ``axis_spin_deg`` agree — tilting the axis does not make vertical yaw
    undercount. What breaks ``spin_deg`` is the hip axis passing *through*
    the vertical, where its horizontal projection collapses and its direction
    is undefined; that is common once a rider is inverted, and it is what
    ``spin_quality`` measures. ``axis_spin_deg`` never projects onto a fixed
    world axis, so it stays well conditioned there. On clean synthetic motion
    the two agree to within a degree; on real inverted footage they diverge,
    and the divergence is a noise signal, not a geometric correction.

    It undercounts corks even on exact input, though: projecting onto one
    axis drops every rotation that was not about it, and a cork's axis moves.
    On 20 synthetic tricks with exact skeletons it named 1 of 13 corks
    correctly (docs/validation.md). The axis is oriented upward, so the sign
    is the turn's sense: positive is counter-clockwise from above.

``spin_path_deg``
    ``rotation_deg`` with a sign: every step's angle, signed by the sense of
    the turn about world up over the surrounding half second or so (a few
    noisy frames, or a pose that flips while inverted, cannot reverse it). It counts the whole rotation a cork performs, however
    its axis moves, and still says which way the rider turned. Over the air
    it named 20 of 20 synthetic tricks on exact skeletons. Like
    ``rotation_deg`` it also accumulates pose jitter, so measure it over the
    air, not over a whole clip. Its meaning is unchanged, but it is no longer
    the dashboard's count: see ``spin_count_deg``.

``body_qx``, ``body_qy``, ``body_qz``, ``body_qw``
    The rider's camera-compensated body orientation as a quaternion, so the
    spin can be re-counted over any window. ``spin_count_deg`` does that: the
    angular path between orientations about 90 degrees of turn apart rather
    than one frame apart. A frame-to-frame path adds every frame's pose
    jitter, and slow motion multiplies the frames for the same turn, so
    through the NIM ``spin_path_deg`` over the true air read a median 96
    degrees high on 189 synthetic clips and named 74 of them; the count named
    149 (docs/validation.md). The chord length is set from the turn itself,
    so no slow-motion factor is needed. ``robust_speed`` is the matching
    per-frame rate, which the dashboard's air detector reads.

``tilt_deg``
    Angle between the rider's own up axis and the world vertical: 0 upright,
    180 fully inverted. This is what separates a cork from a flat spin.

``airborne_hint``, ``airborne_prob``
    Per frame, is the rider in the air. With trained segmenter weights (see
    ``segmenter_weights`` and docs/segmenter.md) this is the learned model's
    mask and ``airborne_prob`` its probability; without them it falls back
    to the old placeholder, under the clip's median ``grab_distance``, which
    calls about half of every clip airborne by construction, and
    ``airborne_prob`` is empty. ``df.attrs["airborne_source"]`` says which.

Rates use real time, so a slow-motion clip must carry its ``slowmo_factor``
(see ``meta.py``) or every deg/s will be wrong by that factor.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .camera import CameraTrack, rotation_angle, rotation_vector
from .meta import ClipMeta
from .usd import quat_from_matrix

ROOT = 0
NECK = 6
HIP_L = 67
HIP_R = 72
WRISTS = (14, 42)
FEET = (71, 76)

# Y is down in the NIM's camera frame (verified against the data: the head
# sits at negative Y relative to the pelvis), so world up is -Y.
WORLD_UP = np.array([0.0, -1.0, 0.0])

# The spin's sense comes from rotation summed over this many steps either
# side, and only when its axis is more than ~12 degrees off horizontal (see
# ``spin_path_deg``).
SENSE_WINDOW = 15
SENSE_MIN_VERTICAL = 0.2

# The rider's camera-compensated orientation, one quaternion (x, y, z, w) per
# frame, so a spin can be re-counted over any window (see ``spin_count_deg``).
ORIENTATION_COLUMNS = ("body_qx", "body_qy", "body_qz", "body_qw")

# ``spin_count_deg`` steps between orientations about this many degrees
# apart; ``robust_speed`` reads its rate over chords whose fastest are about
# this many. Both are angles, not frame counts or seconds, which is what makes
# them indifferent to an unknown slow-motion factor (docs/rotation.md).
COUNT_CHORD_DEG = 90.0
SPEED_CHORD_DEG = 60.0
SPEED_PERCENTILE = 95
MAX_SPAN = 40


def correct_principal_point(
    keypoints_3d: np.ndarray, offset: tuple[float, float], focal: float
) -> np.ndarray:
    """Undo the NIM's assumption that the principal point is the image centre.

    A panning crop moves the true principal point away from the centre of the
    image the NIM was handed, so it back-projects along rays that are off by
    ``offset`` pixels. Depth is unaffected; the lateral components are not::

        X_nim = (u - w/2) * Z / f        X_true = (u - (w/2 + dx)) * Z / f

    so subtracting ``dx * Z / f`` recovers the true X, and likewise for Y.
    Because the correction scales with Z it differs between joints, which is
    precisely the part that would otherwise read as rotation.
    """
    dx, dy = offset
    if not dx and not dy:
        return keypoints_3d
    corrected = keypoints_3d.copy()
    depth = keypoints_3d[:, 2]
    corrected[:, 0] -= dx * depth / focal
    corrected[:, 1] -= dy * depth / focal
    return corrected


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 1e-9 else vector


def body_frame(keypoints: np.ndarray) -> np.ndarray | None:
    """Orthonormal body frame [right, up, forward] as columns, in camera coords."""
    up = _unit(keypoints[NECK] - keypoints[ROOT])
    hip = keypoints[HIP_R] - keypoints[HIP_L]
    if np.linalg.norm(up) < 1e-6 or np.linalg.norm(hip) < 1e-6:
        return None
    right = hip - np.dot(hip, up) * up
    if np.linalg.norm(right) < 1e-6:
        return None
    right = _unit(right)
    forward = np.cross(right, up)
    return np.column_stack([right, up, forward])


def load_poses(path: Path) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                record = json.loads(line)
                out[record["frame_id"]] = record.get("detections", [])
    return out


def _interpolate_angles(angles: np.ndarray, usable: np.ndarray) -> np.ndarray:
    """Fill unusable frames by interpolating on the unit circle, then unwrap."""
    index = np.arange(len(angles))
    if usable.sum() < 2:
        return np.full(len(angles), np.nan)
    sin = np.interp(index, index[usable], np.sin(angles[usable]))
    cos = np.interp(index, index[usable], np.cos(angles[usable]))
    return np.unwrap(np.arctan2(sin, cos))


def _signed_path(path: np.ndarray, steps: list[tuple[int, np.ndarray]],
                 length: int) -> np.ndarray:
    """``rotation_deg`` signed by the sense of the turn it belongs to.

    The sense is the vertical component of the rotation summed over
    ``SENSE_WINDOW`` steps either side, not of each step: through an
    inversion the NIM's per-frame steps can flip sense for a stretch, which
    would subtract real rotation, while a trick turns one way throughout.
    Where the window says nothing (rotation about a near-horizontal axis),
    the previous sense is kept.
    """
    out = np.full(length, np.nan)
    if not steps:
        first = np.flatnonzero(np.isfinite(path))
        out[first] = 0.0
        return out
    vectors = np.array([v for _, v in steps])
    cumulative = np.vstack([np.zeros(3), np.cumsum(vectors, axis=0)])
    signed, sense = 0.0, 1.0
    start = steps[0][0] - 1
    if start >= 0 and np.isfinite(path[start]):
        out[start] = 0.0
    for k, (position, vector) in enumerate(steps):
        lo, hi = max(0, k - SENSE_WINDOW), min(len(steps), k + SENSE_WINDOW + 1)
        around = cumulative[hi] - cumulative[lo]
        vertical = float(np.dot(around, WORLD_UP))
        if abs(vertical) >= SENSE_MIN_VERTICAL * np.linalg.norm(around) > 0:
            sense = np.sign(vertical)
        signed += sense * np.rad2deg(np.linalg.norm(vector))
        out[position] = signed
    return out


def chord_angles(q: np.ndarray, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """Degrees between orientations ``q[i]`` and ``q[j]`` (unit quaternions)."""
    dot = np.abs(np.sum(q[i] * q[j], axis=-1))
    return np.rad2deg(2.0 * np.arccos(np.clip(dot, -1.0, 1.0)))


def _mean_span(span: int, rows: int) -> int:
    return int(max(1, min(span, rows - 1)))


def spin_count_deg(q: np.ndarray, chord_deg: float = COUNT_CHORD_DEG,
                   iterations: int = 4) -> float:
    """Total rotation over a run of orientations, without counting pose jitter.

    ``q`` is one quaternion per posed frame, in order, inside one shot (a
    hole in the pose track is crossed by the chord over it). The angular
    path frame to frame (``rotation_deg``) adds every frame's jitter, and a
    slowed clip has k times the frames for the same turn, so its path grows
    with the slow-motion factor while the turn does not. Here the path is
    taken between orientations ``k`` frames apart, ``k`` chosen so a chord
    spans about ``chord_deg`` of the turn itself: on a 1x 1080 that is about
    every frame, on the same trick at 4x every few. A chord well under 180
    degrees cannot alias, and jitter adds to a chord roughly once rather
    than once per frame. The count is averaged over the ``k`` possible
    starting frames, so it does not depend on which frames the chords hit.

    ``k`` comes from the count itself (fixed point from ``k = 1``), so no
    frame rate or slow-motion factor is needed.
    """
    rows = len(q)
    if rows < 2:
        return 0.0
    total = _stride_path(q, 1)
    span = 1
    for _ in range(iterations):
        if total <= 0:
            break
        new = _mean_span(round(chord_deg * rows / total), rows)
        if new == span:
            break
        span = new
        total = _stride_path(q, span)
    return float(total)


def _stride_path(q: np.ndarray, span: int) -> float:
    """Chord path at stride ``span``, averaged over the ``span`` phases."""
    rows = len(q)
    span = _mean_span(span, rows)
    totals = []
    for phase in range(span):
        picks = list(range(phase, rows, span))
        if picks[0] != 0:
            picks.insert(0, 0)
        if picks[-1] != rows - 1:
            picks.append(rows - 1)
        picks = np.asarray(picks)
        totals.append(float(chord_angles(q, picks[:-1], picks[1:]).sum()))
    return float(np.mean(totals))


def chord_speed(q: np.ndarray, frame_ids: np.ndarray, segments: np.ndarray,
                span: int) -> np.ndarray:
    """Turn rate per frame, in degrees per frame, read over a chord.

    For each posed frame, the angle between the posed frames about ``span``
    either side (inside its shot), over the frames between them. Per-frame
    jitter enters once per chord instead of once per frame.
    """
    rows = len(q)
    out = np.full(rows, np.nan)
    for n in range(rows):
        lo = int(np.searchsorted(frame_ids, frame_ids[n] - span))
        hi = int(np.searchsorted(frame_ids, frame_ids[n] + span, side="right")) - 1
        while segments[lo] != segments[n]:
            lo += 1
        while segments[hi] != segments[n]:
            hi -= 1
        if frame_ids[hi] > frame_ids[lo]:
            out[n] = chord_angles(q, np.array([lo]), np.array([hi]))[0] / (
                frame_ids[hi] - frame_ids[lo])
    return out


def robust_speed(q: np.ndarray, frame_ids: np.ndarray, segments: np.ndarray,
                 chord_deg: float = SPEED_CHORD_DEG) -> tuple[np.ndarray, int]:
    """``chord_speed`` with its span chosen so the fastest chords are about
    ``chord_deg``: (speed, span).

    The span is set from the clip's own 95th-percentile rate, iterated from
    one frame, so it is long where the turn is slow per frame (slow motion)
    and one frame where it is fast. Returns the span too: it is the clip's
    own unit of time, and the air detector scales its hole tolerance by it.
    """
    frame_ids = np.asarray(frame_ids)
    segments = np.asarray(segments)
    span = 1
    speed = chord_speed(q, frame_ids, segments, span)
    for _ in range(4):
        finite = speed[np.isfinite(speed)]
        if not len(finite):
            break
        peak = float(np.percentile(finite, SPEED_PERCENTILE))
        new = int(np.clip(round(chord_deg / (2 * max(peak, 1e-3))), 1, MAX_SPAN))
        if new == span:
            break
        span = new
        speed = chord_speed(q, frame_ids, segments, span)
    return speed, span


def segmenter_weights(choice: str | Path | None = "auto") -> Path | None:
    """Which segmenter weights to use, or None for the heuristic.

    ``"auto"`` takes ``$SNOW_SEGMENTER`` or ``data/weights/segmenter.pt`` when
    the file exists; ``"off"`` (or None) never uses a model; anything else is
    a path that must exist. Weights are not committed (``*.pt`` is ignored);
    they live in S3, see docs/segmenter.md.
    """
    if choice is None or str(choice) in ("off", "none", "heuristic"):
        return None
    if str(choice) == "auto":
        from .config import WEIGHTS_DIR

        env = os.environ.get("SNOW_SEGMENTER")
        path = Path(env) if env else WEIGHTS_DIR / "segmenter.pt"
        return path if path.exists() else None
    path = Path(choice)
    if not path.exists():
        raise FileNotFoundError(f"segmenter weights not found: {path}")
    return path


def _heuristic_airborne(df: pd.DataFrame) -> None:
    df["airborne_hint"] = df["grab_distance"] < df["grab_distance"].median()
    df["airborne_prob"] = np.nan
    df.attrs["airborne_source"] = "heuristic"


def _segmenter_airborne(df: pd.DataFrame, detections: dict[int, list[dict]],
                        camera: CameraTrack | None, tracking_id: int, weights: Path) -> None:
    # Imported here: torch is heavy, and a clip without weights never needs it.
    from .learn.inputs import frame_channels
    from .learn.segmenter import Segmenter

    model = Segmenter.load(weights)
    x = frame_channels(detections, df, camera.rotations if camera is not None else None,
                       tracking_id=tracking_id)
    segment = df["segment"].to_numpy(dtype=int)
    prob = model.probabilities(x, segment)
    df["airborne_hint"] = model.mask(prob, segment)
    df["airborne_prob"] = prob
    df.attrs["airborne_source"] = f"segmenter:{weights.name}"


def build(
    poses: Path,
    meta: ClipMeta,
    camera: CameraTrack | None = None,
    tracking_id: int = 1,
    min_quality: float = 0.35,
    crop_plan: dict | None = None,
    segmenter: str | Path | None = "off",
) -> pd.DataFrame:
    detections = load_poses(poses)
    offsets = (crop_plan or {}).get("principal_offsets")
    crop_focal = (crop_plan or {}).get("focal_length")
    frames = max(detections) + 1 if detections else 0

    records = []
    for frame_id in range(frames):
        row: dict = {
            "frame_id": frame_id,
            "segment": int(camera.segment[frame_id])
            if camera is not None and frame_id < camera.frames
            else 0,
        }
        found = next(
            (d for d in detections.get(frame_id, []) if d["tracking_id"] == tracking_id),
            None,
        )
        if found is None:
            records.append(row)
            continue

        k3 = np.asarray(found["keypoints_3d"], dtype=float)
        if offsets and crop_focal and frame_id < len(offsets):
            k3 = correct_principal_point(k3, tuple(offsets[frame_id]), crop_focal)
        conf = np.asarray(found["keypoints_confidence"], dtype=float)
        frame = body_frame(k3)
        if frame is None:
            records.append(row)
            continue

        # Remove the camera's own rotation. R_cam maps world -> camera, so its
        # transpose takes a camera-frame orientation back into the world.
        if camera is not None and frame_id < camera.frames:
            world_frame = camera.rotations[frame_id].T @ frame
            delta_cam = camera.deltas[frame_id]
        else:
            world_frame = frame
            delta_cam = np.eye(3)

        right, up, _ = world_frame[:, 0], world_frame[:, 1], world_frame[:, 2]
        horizontal = right - np.dot(right, WORLD_UP) * WORLD_UP
        quality = float(np.linalg.norm(horizontal))

        scale = float(np.linalg.norm(k3[NECK] - k3[ROOT])) or np.nan
        grab = min(
            np.linalg.norm(k3[w] - k3[f]) for w in WRISTS for f in FEET
        )

        row.update(
            {
                "yaw_rad": float(np.arctan2(horizontal[2], horizontal[0]))
                if quality > 1e-6
                else np.nan,
                "spin_quality": quality,
                "tilt_deg": float(
                    np.rad2deg(np.arccos(np.clip(np.dot(up, WORLD_UP), -1.0, 1.0)))
                ),
                "camera_rotation_deg": float(np.rad2deg(rotation_angle(delta_cam))),
                "root_x": float(k3[ROOT, 0]),
                "root_y": float(k3[ROOT, 1]),
                "root_z": float(k3[ROOT, 2]),
                "torso_scale_m": scale,
                "grab_distance": float(grab / scale) if scale else np.nan,
                "confidence": float(np.mean(conf)),
                "_frame": world_frame,
                **dict(zip(ORIENTATION_COLUMNS, quat_from_matrix(world_frame).tolist())),
            }
        )
        records.append(row)

    df = pd.DataFrame(records)
    for column in ("yaw_rad", "spin_quality", "tilt_deg", *ORIENTATION_COLUMNS):
        if column not in df:
            df[column] = np.nan

    df["spin_deg"] = np.nan
    df["rotation_deg"] = np.nan
    df["axis_spin_deg"] = np.nan
    df["spin_path_deg"] = np.nan
    axis_by_segment: dict[int, np.ndarray] = {}

    # Integrate within each camera segment: a cut gives a new world frame, so
    # rotation cannot carry across it.
    for segment, group in df.groupby("segment"):
        idx = group.index.to_numpy()
        yaw = group["yaw_rad"].to_numpy(dtype=float)
        usable = np.isfinite(yaw) & (group["spin_quality"].to_numpy() > min_quality)
        if usable.sum() >= 2:
            unwrapped = _interpolate_angles(np.nan_to_num(yaw), usable)
            first = np.nonzero(usable)[0][0]
            df.loc[idx, "spin_deg"] = np.rad2deg(unwrapped - unwrapped[first])

        frames_in = group["_frame"].tolist() if "_frame" in group else []
        total = 0.0
        path = np.full(len(idx), np.nan)
        steps: list[tuple[int, np.ndarray]] = []
        previous = None
        for position, current in enumerate(frames_in):
            if not isinstance(current, np.ndarray):
                previous = None
                continue
            if previous is not None:
                delta = current @ previous.T
                total += np.rad2deg(rotation_angle(delta))
                steps.append((position, rotation_vector(delta)))
            path[position] = total
            previous = current
        spin_path = _signed_path(path, steps, len(idx))
        df.loc[idx, "rotation_deg"] = path
        df.loc[idx, "spin_path_deg"] = spin_path

        # The dominant axis is the direction the segment's rotation actually
        # went, so projecting each step onto it integrates the spin the way a
        # judge counts it, whatever the cork angle.
        if len(steps) >= 5:
            vectors = np.array([v for _, v in steps])
            net = vectors.sum(axis=0)
            norm = np.linalg.norm(net)
            if norm > 1e-9:
                axis = net / norm
                # Point the axis up, so the projection carries the turn's
                # sense: counter-clockwise from above is positive. Left as
                # the net direction, it only ever grew positive and every
                # spin read counter-clockwise.
                if float(np.dot(axis, WORLD_UP)) < 0:
                    axis = -axis
                axis_by_segment[int(segment)] = axis
                along = np.cumsum(np.rad2deg(vectors @ axis))
            else:
                # No net rotation, so no dominant axis exists — but "did not
                # rotate" is a real answer and more useful downstream than a
                # column of NaN.
                along = np.zeros(len(steps))
            series = np.full(len(idx), np.nan)
            for (position, _), value in zip(steps, along):
                series[position] = value
            df.loc[idx, "axis_spin_deg"] = series

    dt = meta.dt
    df["time_s"] = df["frame_id"] * dt
    for column in ("spin_deg", "rotation_deg"):
        df[f"{column[:-4]}_rate_dps"] = df[column].diff() / dt if dt else np.nan
    if "grab_distance" not in df:
        df["grab_distance"] = np.nan
    weights = segmenter_weights(segmenter)
    if weights is None:
        _heuristic_airborne(df)
    else:
        try:
            _segmenter_airborne(df, detections, camera, tracking_id, weights)
        except ImportError as exc:
            # The model needs torch; without it the old heuristic still runs.
            print(f"  segmenter unavailable ({exc}); using the heuristic airborne_hint")
            _heuristic_airborne(df)
    df.attrs["spin_axes"] = {k: v.tolist() for k, v in axis_by_segment.items()}

    return df.drop(columns=[c for c in ("_frame",) if c in df])


def summarize(df: pd.DataFrame, meta: ClipMeta) -> dict:
    segments = []
    for segment, group in df.groupby("segment"):
        valid = group.dropna(subset=["spin_deg"])
        if valid.empty:
            continue
        spin = valid["spin_deg"].to_numpy()
        rotation = group["rotation_deg"].dropna()
        axis_spin = group["axis_spin_deg"].dropna()
        spin_path = group["spin_path_deg"].dropna()
        axis = df.attrs.get("spin_axes", {}).get(int(segment))
        axis_tilt = (
            float(
                np.rad2deg(
                    np.arccos(np.clip(abs(float(np.dot(axis, WORLD_UP))), -1.0, 1.0))
                )
            )
            if axis is not None
            else None
        )
        segments.append(
            {
                "segment": int(segment),
                "frames": int(len(group)),
                "tracked_frames": int(len(valid)),
                "net_spin_deg": round(float(spin[-1] - spin[0]), 1),
                "peak_spin_deg": round(float(np.nanmax(np.abs(spin - spin[0]))), 1),
                "total_rotation_deg": round(
                    float(rotation.iloc[-1] - rotation.iloc[0]) if len(rotation) else 0.0, 1
                ),
                "axis_spin_deg": round(
                    float(axis_spin.iloc[-1] - axis_spin.iloc[0]) if len(axis_spin) else 0.0,
                    1,
                ),
                "spin_path_deg": round(
                    float(spin_path.iloc[-1] - spin_path.iloc[0]) if len(spin_path) else 0.0,
                    1,
                ),
                "spin_axis_tilt_deg": round(axis_tilt, 1) if axis_tilt is not None else None,
                "max_tilt_deg": round(float(group["tilt_deg"].max(skipna=True)), 1),
                "median_quality": round(float(group["spin_quality"].median(skipna=True)), 3),
                "camera_rotation_deg": round(
                    float(group["camera_rotation_deg"].sum(skipna=True)), 1
                ),
            }
        )
    from .learn.events import runs

    airborne = (df["airborne_hint"].fillna(False).to_numpy(dtype=bool)
                if "airborne_hint" in df else np.zeros(len(df), dtype=bool))
    segment_ids = df["segment"].to_numpy(dtype=int) if "segment" in df else None
    return {
        "clip": meta.name,
        "real_fps": meta.real_fps,
        "slowmo_factor": meta.slowmo_factor,
        "frames": int(len(df)),
        "tracked_frames": int(df["spin_deg"].notna().sum()),
        "segments": segments,
        "airborne": {
            "source": df.attrs.get("airborne_source", "heuristic"),
            "frames": int(airborne.sum()),
            # [first airborne frame, first frame back down), as frame ids.
            "airs": [[int(df["frame_id"].iloc[a]), int(df["frame_id"].iloc[b - 1]) + 1]
                     for a, b in runs(airborne, segment_ids)],
        },
    }
