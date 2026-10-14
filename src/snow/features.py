"""Turn raw NIM poses into a per-frame table for the DS/ML work.

The central problem is that everything the NIM returns is in **camera**
coordinates, and the camera is itself rotating to follow the rider. This
module builds a body frame from the 3D keypoints, removes the camera's
rotation (see ``camera.py``), and integrates what is left into rotation
measures that mean something.

Three rotation measures are produced, because no single one covers every
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

``tilt_deg``
    Angle between the rider's own up axis and the world vertical: 0 upright,
    180 fully inverted. This is what separates a cork from a flat spin.

Rates use real time, so a slow-motion clip must carry its ``slowmo_factor``
(see ``meta.py``) or every deg/s will be wrong by that factor.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .camera import CameraTrack, rotation_angle, rotation_vector
from .meta import ClipMeta

ROOT = 0
NECK = 6
HIP_L = 67
HIP_R = 72
WRISTS = (14, 42)
FEET = (71, 76)

# Y is down in the NIM's camera frame (verified against the data: the head
# sits at negative Y relative to the pelvis), so world up is -Y.
WORLD_UP = np.array([0.0, -1.0, 0.0])


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


def build(
    poses: Path,
    meta: ClipMeta,
    camera: CameraTrack | None = None,
    tracking_id: int = 1,
    min_quality: float = 0.35,
    crop_plan: dict | None = None,
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
            }
        )
        records.append(row)

    df = pd.DataFrame(records)
    for column in ("yaw_rad", "spin_quality", "tilt_deg"):
        if column not in df:
            df[column] = np.nan

    df["spin_deg"] = np.nan
    df["rotation_deg"] = np.nan
    df["axis_spin_deg"] = np.nan
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
        df.loc[idx, "rotation_deg"] = path

        # The dominant axis is the direction the segment's rotation actually
        # went, so projecting each step onto it integrates the spin the way a
        # judge counts it, whatever the cork angle.
        if len(steps) >= 5:
            vectors = np.array([v for _, v in steps])
            net = vectors.sum(axis=0)
            norm = np.linalg.norm(net)
            if norm > 1e-9:
                axis = net / norm
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
    df["airborne_hint"] = df["grab_distance"] < df["grab_distance"].median()
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
                "spin_axis_tilt_deg": round(axis_tilt, 1) if axis_tilt is not None else None,
                "max_tilt_deg": round(float(group["tilt_deg"].max(skipna=True)), 1),
                "median_quality": round(float(group["spin_quality"].median(skipna=True)), 3),
                "camera_rotation_deg": round(
                    float(group["camera_rotation_deg"].sum(skipna=True)), 1
                ),
            }
        )
    return {
        "clip": meta.name,
        "real_fps": meta.real_fps,
        "slowmo_factor": meta.slowmo_factor,
        "frames": int(len(df)),
        "tracked_frames": int(df["spin_deg"].notna().sum()),
        "segments": segments,
    }
