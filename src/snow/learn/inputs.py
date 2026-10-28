"""Per-frame model inputs, computed the same way for training and for serving.

The segmenter reads exactly what ``snow features`` has at inference time: the
NIM's per-frame detections, the features table ``features.build`` returns,
and the camera track. Training reads the same three things back from
``data/poses/`` and calls the same function, so there is no second feature
implementation to drift (docs/training.md, "no separate training-time
feature code").

What goes in, and why each is allowed:

* **pose**: a handful of joints relative to the root, in the rider's own body
  frame, divided by the torso length. Scale-free (invariant 7) and blind to
  where the camera is pointing.
* **tilt**: ``cos(tilt_deg)``, the body's up axis against world up.
* **rates**: rotation of the body frame, and the angular motion of the ray to
  the rider in the camera-compensated world frame, per *frame*. Per frame,
  not per second, because slow motion makes per-second rates unknowable
  (invariant 8); training resamples time at random so that the model cannot
  lean on their absolute size (``RATE_CHANNELS``).
* **grab**, **confidence**, and a **present** flag for frames the NIM had no
  body for.

Missing frames are interpolated from their neighbours inside the same camera
segment and never across a cut (invariant 6).

The principal point correction that ``features.build`` applies to a panning
crop is not applied here: it moves joints by a few millimetres relative to
one another, well below what this model reads.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..features import WORLD_UP, body_frame

ROOT = 0
# spine3, head, head_top, then each side's shoulder, elbow, wrist, knee,
# ankle, toe (indices from synth/nova.py, sides read off real NIM output).
POSE_JOINTS = (3, 6, 7, 12, 13, 14, 40, 41, 42, 68, 69, 71, 73, 74, 76)
# Anatomical left/right pairs, swapped when the image is mirrored.
MIRROR_PAIRS = ((11, 39), (12, 40), (13, 41), (14, 42), (67, 72), (68, 73),
                (69, 74), (70, 75), (71, 76))

CHANNELS = (
    [f"pose{j}_{axis}" for j in POSE_JOINTS for axis in "xyz"]
    + ["cos_tilt", "rot_step", "elev_step", "azim_step", "grab", "confidence", "present"]
)
NUM_CHANNELS = len(CHANNELS)
# Channels measured per frame, which scale by 1/r when time is stretched by r.
RATE_CHANNELS = tuple(CHANNELS.index(c) for c in ("rot_step", "elev_step", "azim_step"))
PRESENT = CHANNELS.index("present")

# Rough unit scaling so every channel starts near unit size before the
# trained per-channel standardisation; degrees and radians per frame are small.
_ROT_STEP_SCALE = 1.0 / 10.0     # degrees per frame
_RAY_STEP_SCALE = 100.0          # radians per frame
_GRAB_CLIP = 5.0                 # torso lengths

_MIRROR = np.diag([-1.0, 1.0, 1.0])
_SWAP = np.arange(77)
for _a, _b in MIRROR_PAIRS:
    _SWAP[_a], _SWAP[_b] = _b, _a


def mirror_keypoints(k3: np.ndarray) -> np.ndarray:
    """The skeleton the NIM would return for the horizontally flipped video.

    X is negated, and anatomical left and right trade places: in a mirrored
    image the rider's left hand is where the right one was, and the NIM labels
    it by anatomy. Airborne is unchanged by a mirror; a direction label would
    have to flip (docs/training.md, "Mirroring flips the label").
    """
    return (k3 @ _MIRROR)[_SWAP]


def _pick(detections: list[dict], tracking_id: int) -> dict | None:
    return next((d for d in detections if d.get("tracking_id") == tracking_id), None)


def observed(present: np.ndarray, segment: np.ndarray) -> np.ndarray:
    """Rows between a segment's first and last frame with a body, inclusive.

    A hole inside that span is a dropout mid-shot (the tracker loses an
    inverted rider for 10 to 20 frames) and is bridged. Before the first body
    or after the last there is no evidence at all: the rider is hidden by the
    kicker, or out of shot, so nothing is claimed there.
    """
    present = np.asarray(present) > 0.5
    out = np.zeros(len(present), dtype=bool)
    for seg in np.unique(segment):
        rows = np.flatnonzero((segment == seg) & present)
        if len(rows):
            out[rows[0]:rows[-1] + 1] = True
    return out


def _fill_within(values: np.ndarray, valid: np.ndarray, segment: np.ndarray) -> np.ndarray:
    """Linear interpolation over holes between valid rows of one segment;
    zero outside the segment's observed span (see ``observed``)."""
    out = np.zeros_like(values)
    index = np.arange(len(values))
    for seg in np.unique(segment):
        rows = index[segment == seg]
        good = rows[valid[rows]]
        if len(good) == 0:
            continue
        inside = rows[(rows >= good[0]) & (rows <= good[-1])]
        for c in range(values.shape[1]):
            out[inside, c] = np.interp(inside, good, values[good, c])
    return out


def _segment_diff(values: np.ndarray, segment: np.ndarray) -> np.ndarray:
    """First difference, zero at the first frame of every segment."""
    step = np.zeros_like(values)
    step[1:] = values[1:] - values[:-1]
    first = np.ones(len(values), dtype=bool)
    first[1:] = segment[1:] != segment[:-1]
    step[first] = 0.0
    return step


def frame_channels(
    detections: dict[int, list[dict]],
    df: pd.DataFrame,
    camera_rotations: np.ndarray | None = None,
    tracking_id: int = 1,
    mirror: bool = False,
) -> np.ndarray:
    """``(len(df), NUM_CHANNELS)`` float32, one row per row of ``df``.

    ``df`` is ``features.build``'s table (or its CSV), which supplies
    ``frame_id``, ``segment``, ``tilt_deg``, ``rotation_deg``,
    ``grab_distance`` and ``confidence``. ``camera_rotations`` is
    ``CameraTrack.rotations`` (world to camera), or None for no compensation.
    """
    frame_ids = df["frame_id"].to_numpy(dtype=int)
    segment = df["segment"].to_numpy(dtype=int) if "segment" in df else np.zeros(len(df), int)
    n = len(df)
    pose = np.zeros((n, len(POSE_JOINTS) * 3))
    ray = np.zeros((n, 2))
    valid = np.zeros(n, dtype=bool)

    for row, frame_id in enumerate(frame_ids):
        found = _pick(detections.get(int(frame_id), []), tracking_id)
        if found is None:
            continue
        k3 = np.asarray(found["keypoints_3d"], dtype=float)
        rotation = (
            camera_rotations[frame_id]
            if camera_rotations is not None and frame_id < len(camera_rotations)
            else np.eye(3)
        )
        if mirror:
            k3 = mirror_keypoints(k3)
            rotation = _MIRROR @ rotation @ _MIRROR
        frame = body_frame(k3)
        scale = float(np.linalg.norm(k3[6] - k3[ROOT]))
        if frame is None or not np.isfinite(scale) or scale < 1e-6:
            continue
        local = (k3[list(POSE_JOINTS)] - k3[ROOT]) @ frame / scale
        pose[row] = local.reshape(-1)
        root = k3[ROOT]
        norm = np.linalg.norm(root)
        if norm > 1e-9:
            world = rotation.T @ (root / norm)
            up = float(np.dot(world, WORLD_UP))
            # Horizontal plane is X/Z (world up is -Y).
            ray[row] = (np.arcsin(np.clip(up, -1.0, 1.0)), np.arctan2(world[2], world[0]))
        valid[row] = True

    ray[:, 1] = _unwrap_within(ray[:, 1], valid, segment)

    def column(name: str) -> np.ndarray:
        return df[name].to_numpy(dtype=float) if name in df else np.full(n, np.nan)

    tilt, rotation, grab = column("tilt_deg"), column("rotation_deg"), column("grab_distance")
    usable = valid & np.isfinite(tilt) & np.isfinite(rotation) & np.isfinite(grab)
    levels = _fill_within(np.column_stack([
        pose, ray, np.cos(np.deg2rad(np.nan_to_num(tilt))), np.nan_to_num(rotation),
        np.clip(np.nan_to_num(grab), 0.0, _GRAB_CLIP),
    ]), usable, segment)
    k = pose.shape[1]
    pose, ray, cos_tilt, total, grab = (levels[:, :k], levels[:, k:k + 2], levels[:, k + 2],
                                        levels[:, k + 3], levels[:, k + 4])

    # A step is only measured between two observed frames of one segment, so
    # nothing jumps where the rider first appears or at a cut.
    seen = observed(usable, segment)
    paired = seen.copy()
    paired[1:] &= seen[:-1]
    paired[0] = False
    ray_step = _segment_diff(ray, segment) * _RAY_STEP_SCALE * paired[:, None]
    rot_step = np.abs(_segment_diff(total[:, None], segment))[:, 0] * _ROT_STEP_SCALE * paired
    confidence = np.nan_to_num(column("confidence")) * usable

    out = np.column_stack([
        pose, cos_tilt, rot_step, ray_step[:, 0], ray_step[:, 1],
        grab, confidence, usable.astype(float),
    ])
    out[~seen] = 0.0
    return out.astype(np.float32)


def _unwrap_within(angles: np.ndarray, valid: np.ndarray, segment: np.ndarray) -> np.ndarray:
    out = angles.copy()
    for seg in np.unique(segment):
        rows = np.flatnonzero((segment == seg) & valid)
        if len(rows):
            out[rows] = np.unwrap(angles[rows])
    return out


def resample(x: np.ndarray, y: np.ndarray | None, factor: float) -> tuple[np.ndarray, np.ndarray | None]:
    """Stretch one segment's time axis by ``factor`` (2.0 plays twice as slow).

    Level channels are interpolated, per-frame rate channels are interpolated
    and divided by the factor, and the present flag and labels take the
    nearest source frame. This is the timebase augmentation: the model sees
    the same air at many playback rates and so cannot learn one.
    """
    n = len(x)
    m = max(2, int(round(n * factor)))
    source = np.clip(np.arange(m) * (n - 1) / max(m - 1, 1), 0, n - 1)
    index = np.arange(n)
    out = np.empty((m, x.shape[1]), dtype=np.float32)
    for c in range(x.shape[1]):
        out[:, c] = np.interp(source, index, x[:, c])
    stretch = (m - 1) / max(n - 1, 1)
    for c in RATE_CHANNELS:
        out[:, c] /= stretch
    nearest = np.rint(source).astype(int)
    out[:, PRESENT] = x[nearest, PRESENT]
    return out, (y[nearest] if y is not None else None)
