"""Turn a Blender recording into the NIM's own file formats.

``gt_raw.npz`` (from ``blender/run.py``) holds every joint's world position
and rest-relative world orientation, and the camera's exact pose and focal.
This module writes them as the pipeline's own artifacts, so the unchanged
stages can read ground truth exactly as they read the NIM:

* ``<clip>/gt.poses.json``: JSON Lines in the NIM's schema, 77 joints, camera
  frame (X right, Y down, Z forward), ``joint_rotations`` local and
  ``rest_pose`` in skeleton space, obeying the same UsdSkel forward
  kinematics the NIM's output does (``tests/test_synth.py`` pins both)
* ``<clip>/gt.camera.npz``: a ``CameraTrack`` of the true camera, world =
  the first frame's camera, as ``camera.py`` defines it

The rest pose is the rider's own (see ``nova.py``): positions and the
hierarchy compare with the NIM's directly, local rotations do not.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..camera import CameraTrack
from ..skeleton import PARENTS
from ..usd import quat_from_matrix

BOX_MARGIN = 0.08


def load_raw(clip: Path) -> dict[str, np.ndarray]:
    z = np.load(clip / "gt_raw.npz")
    return {k: z[k] for k in z.files}


def camera_frame(raw: dict) -> tuple[np.ndarray, np.ndarray]:
    """Joint positions and orientations in each frame's camera coordinates."""
    rotation = raw["world_to_camera"]                      # (n, 3, 3)
    offset = raw["positions"] - raw["camera_centre"][:, None, :]
    points = np.einsum("nij,nkj->nki", rotation, offset)
    orientations = np.einsum("nij,nkjl->nkil", rotation, raw["orientations"])
    return points, orientations


def project(points: np.ndarray, focal: np.ndarray, width: int, height: int) -> np.ndarray:
    """Pinhole projection with the principal point at the image centre."""
    z = points[..., 2]
    u = focal[:, None] * points[..., 0] / z + width / 2.0
    v = focal[:, None] * points[..., 1] / z + height / 2.0
    return np.stack([u, v], axis=-1)


def local_rotations(orientations: np.ndarray) -> np.ndarray:
    """Global (camera-frame) orientations -> parent-relative, root global."""
    local = np.empty_like(orientations)
    for index, parent in enumerate(PARENTS):
        if parent < 0:
            local[:, index] = orientations[:, index]
        else:
            local[:, index] = np.einsum(
                "nji,njk->nik", orientations[:, parent], orientations[:, index]
            )
    return local


def write_poses(clip: Path, raw: dict) -> Path:
    width, height = int(raw["width"]), int(raw["height"])
    points, orientations = camera_frame(raw)
    pixels = project(points, raw["focal_px"], width, height)
    local = local_rotations(orientations)
    rest = raw["rest"] - raw["rest"][0]
    rest_list = rest.tolist()
    dest = clip / "gt.poses.json"
    with dest.open("w") as fh:
        for frame in range(len(points)):
            lo, hi = pixels[frame].min(axis=0), pixels[frame].max(axis=0)
            pad = (hi - lo) * BOX_MARGIN
            lo, hi = lo - pad, hi + pad
            quats = [quat_from_matrix(m).tolist() for m in local[frame]]
            detection = {
                "tracking_id": 1,
                "bbox": [float(lo[0]), float(lo[1]), float(hi[0] - lo[0]), float(hi[1] - lo[1])],
                "keypoints_2d": pixels[frame].tolist(),
                "keypoints_confidence": [1.0] * len(PARENTS),
                "keypoints_3d": points[frame].tolist(),
                "rest_pose": rest_list,
                "joint_rotations": quats,
                "root_pose": {"translation": points[frame, 0].tolist(), "rotation": quats[0]},
            }
            fh.write(json.dumps({"frame_id": frame, "detections": [detection]}) + "\n")
    return dest


def camera_track(raw: dict) -> CameraTrack:
    """The true camera in ``camera.py``'s convention: one segment, world = frame 0."""
    absolute = raw["world_to_camera"]
    rotations = np.einsum("nij,kj->nik", absolute, absolute[0])
    deltas = np.empty_like(rotations)
    deltas[0] = np.eye(3)
    deltas[1:] = np.einsum("nij,nkj->nik", absolute[1:], absolute[:-1])
    frames = len(absolute)
    return CameraTrack(
        rotations=rotations, deltas=deltas, segment=np.zeros(frames, dtype=int),
        inliers=np.zeros(frames, dtype=int), reliable=np.ones(frames, dtype=bool),
        focal=float(np.median(raw["focal_px"])), frames=frames,
    )


def convert(clip: Path) -> dict:
    """Write ``gt.poses.json`` and ``gt.camera.npz``; return camera facts."""
    raw = load_raw(clip)
    write_poses(clip, raw)
    track = camera_track(raw)
    track.save(clip / "gt.camera.npz")
    from ..camera import rotation_angle
    path = float(np.rad2deg(sum(rotation_angle(d) for d in track.deltas)))
    width, height = int(raw["width"]), int(raw["height"])
    return {
        "camera_path_deg": round(path, 2),
        "camera_net_deg": round(float(np.rad2deg(rotation_angle(track.rotations[-1]))), 2),
        "focal_px_min": round(float(raw["focal_px"].min()), 1),
        "focal_px_max": round(float(raw["focal_px"].max()), 1),
        "focal_assumed_px": round(float(np.hypot(width, height)), 1),
        "rider_visible_fraction": round(float(raw["visible"].mean()), 3)
        if "visible" in raw else None,
    }
