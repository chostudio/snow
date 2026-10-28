"""One air as a fixed-shape tensor, for the trick classifier.

The classifier never sees a clip, only an air: the frames from takeoff to
landing. Everything here is plain numpy, so the dataset can be built and
tested without torch, and so inference computes exactly what training did.

What goes in is what ``features.py`` itself works from: the NIM's
``keypoints_3d`` per frame and the camera rotation ``camera.py`` estimated,
both from the *rendered video through the pipeline*, never the rig's clean
skeleton (docs/training.md). The rules from that document, and where each is
kept:

* **Torso-normalised.** Joint positions are expressed in the rider's own body
  frame and divided by the neck-to-root length, so the uncalibrated metric
  scale (invariant 7) carries nothing.
* **Timebase-invariant.** The air is resampled to ``SEQ_LEN`` steps whatever
  its frame count, and rotation is fed as totals and per-step fractions of
  the air, never per second. ``augment`` also thins frames at random, which
  is what a lower slow-motion factor looks like.
* **Never across a cut.** ``window`` keeps the one camera segment holding most
  of the air.
* **Mirroring flips the turn.** ``mirror`` reflects the raw keypoints and the
  camera, then everything is recomputed, so the sense of every derived
  rotation flips by construction; the caller flips the label with it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import features as feat
from ..camera import rotation_vector

# The Nova-77 joints kept: enough for the body frame, the limbs and a grab.
JOINTS = (0, 3, 6, 7, 12, 13, 14, 40, 41, 42, 67, 68, 69, 71, 72, 73, 74, 76)
_POS = {j: k for k, j in enumerate(JOINTS)}
# Left/right pairs (nova.py: 11-38 left arm, 39-66 right, 67-71 left leg).
_PAIRS = ((12, 40), (13, 41), (14, 42), (67, 72), (68, 73), (69, 74), (71, 76))
MIRROR = np.arange(len(JOINTS))
for _a, _b in _PAIRS:
    MIRROR[_POS[_a]], MIRROR[_POS[_b]] = _POS[_b], _POS[_a]

ROOT, NECK = _POS[feat.ROOT], _POS[feat.NECK]
HIP_L, HIP_R = _POS[feat.HIP_L], _POS[feat.HIP_R]
WRISTS = tuple(_POS[j] for j in feat.WRISTS)
FEET = tuple(_POS[j] for j in feat.FEET)
WORLD_UP = feat.WORLD_UP

SEQ_LEN = 64
# Cumulative rotation channels, in turns: unsigned path, the part about world
# up (signed), the part about horizontal axes, and the vertical part read
# without removing the camera (camera.py over-reads on a long lens, so the
# model is given both and left to weigh them).
CUMULATIVE = ("path", "vertical", "horizontal", "vertical_cam")
CHANNELS = (
    [f"cum_{c}" for c in CUMULATIVE]
    + [f"step_{c}" for c in CUMULATIVE]
    + ["up_x", "up_y", "up_z"]
    + [f"j{j}_{a}" for j in JOINTS if j != feat.ROOT for a in "xyz"]
    + [f"grab_{w}_{f}" for w in feat.WRISTS for f in feat.FEET]
    + ["root_dx", "root_dy", "root_dz", "posed", "confidence"]
)
N_CHANNELS = len(CHANNELS)
MIN_POSED = 4


@dataclass
class Track:
    """One clip's pose track, reduced to what an air needs."""

    k3: np.ndarray        # (T, len(JOINTS), 3), NaN where the NIM gave no pose
    confidence: np.ndarray  # (T,), mean keypoint confidence, NaN where none
    camera: np.ndarray    # (T, 3, 3), world -> camera within the segment
    segment: np.ndarray   # (T,) int

    @property
    def frames(self) -> int:
        return len(self.segment)


def track_from_poses(detections: dict[int, list[dict]], frames: int,
                     camera=None, tracking_id: int = 1) -> Track:
    """A ``Track`` from ``features.load_poses`` output and a ``CameraTrack``."""
    if camera is not None:
        frames = max(frames, camera.frames)
    k3 = np.full((frames, len(JOINTS), 3), np.nan, dtype=np.float32)
    conf = np.full(frames, np.nan, dtype=np.float32)
    for frame_id, dets in detections.items():
        found = next((d for d in dets if d.get("tracking_id") == tracking_id), None)
        if found is None or frame_id >= frames:
            continue
        k3[frame_id] = np.asarray(found["keypoints_3d"], dtype=np.float32)[list(JOINTS)]
        conf[frame_id] = float(np.mean(found["keypoints_confidence"]))
    rotations = np.tile(np.eye(3, dtype=np.float32), (frames, 1, 1))
    segment = np.zeros(frames, dtype=np.int32)
    if camera is not None:
        n = min(frames, camera.frames)
        rotations[:n] = camera.rotations[:n]
        segment[:n] = camera.segment[:n]
        if n < frames:
            segment[n:] = segment[n - 1] if n else 0
    return Track(k3=k3, confidence=conf, camera=rotations, segment=segment)


def mirror(track: Track) -> Track:
    """Reflect the clip left to right: X -> -X, left and right joints swapped.

    A reflected camera is ``M R M`` with ``M = diag(-1, 1, 1)``. Every rotation
    about world up then turns the other way, which is the point: a mirrored
    counter-clockwise spin is a clockwise one, and its label must say so.
    """
    k3 = track.k3[:, MIRROR].copy()
    k3[..., 0] *= -1
    m = np.diag([-1.0, 1.0, 1.0]).astype(np.float32)
    return Track(k3=k3, confidence=track.confidence, camera=m @ track.camera @ m,
                 segment=track.segment)


def window(track: Track, start: int, end: int) -> tuple[int, int]:
    """Clamp an air to the clip and to the one shot holding most of its poses."""
    start, end = max(0, int(start)), min(track.frames - 1, int(end))
    if end <= start:
        return start, start
    seg = track.segment[start:end + 1]
    posed = np.isfinite(track.k3[start:end + 1, 0, 0])
    values, counts = np.unique(seg[posed] if posed.any() else seg, return_counts=True)
    keep = values[np.argmax(counts)]
    inside = np.flatnonzero(seg == keep)
    # The longest contiguous run of that segment; a shot does not resume.
    runs = np.split(inside, np.flatnonzero(np.diff(inside) > 1) + 1)
    run = max(runs, key=len)
    return start + int(run[0]), start + int(run[-1])


def _full(sub: np.ndarray) -> np.ndarray:
    full = np.zeros((77, 3))
    full[list(JOINTS)] = sub
    return full


def _resample(values: np.ndarray, at: np.ndarray, length: int) -> np.ndarray:
    """Values known at frame positions ``at``, read at ``length`` even steps."""
    grid = np.linspace(at[0], at[-1], length) if len(at) > 1 else np.full(length, at[0])
    if values.ndim == 1:
        return np.interp(grid, at, values)
    return np.stack([np.interp(grid, at, values[:, c]) for c in range(values.shape[1])],
                    axis=1)


def encode(track: Track, start: int, end: int, length: int = SEQ_LEN,
           keep: np.ndarray | None = None) -> np.ndarray | None:
    """The air ``start``..``end`` as a ``(N_CHANNELS, length)`` float32 array.

    ``keep`` optionally drops frames (augmentation). Returns None when fewer
    than ``MIN_POSED`` frames inside the air have a pose.
    """
    start, end = window(track, start, end)
    frames = np.arange(start, end + 1)
    posed = np.isfinite(track.k3[frames, 0, 0])
    if keep is not None:
        posed &= keep[: len(frames)]
    at = frames[posed]
    if len(at) < MIN_POSED:
        return None

    cumulative = np.zeros((len(at), len(CUMULATIVE)))
    up = np.zeros((len(at), 3))
    joints = np.zeros((len(at), len(JOINTS) - 1, 3))
    grab = np.zeros((len(at), len(WRISTS) * len(FEET)))
    root = np.zeros((len(at), 3))
    usable = np.ones(len(at), dtype=bool)
    previous = previous_cam = None
    totals = np.zeros(len(CUMULATIVE))
    scales = []
    others = [k for k in range(len(JOINTS)) if k != ROOT]
    for i, f in enumerate(at):
        k = track.k3[f].astype(float)
        frame = feat.body_frame(_full(k))
        if frame is None:
            usable[i] = False
            continue
        world = track.camera[f].T @ frame
        if previous is not None:
            w = rotation_vector(world @ previous.T)
            c = rotation_vector(frame @ previous_cam.T)
            vertical = float(np.dot(w, WORLD_UP))
            totals += (np.linalg.norm(w), vertical,
                       np.linalg.norm(w - vertical * WORLD_UP), float(np.dot(c, WORLD_UP)))
        previous, previous_cam = world, frame
        cumulative[i] = totals
        up[i] = world[:, 1]
        scale = float(np.linalg.norm(k[NECK] - k[ROOT])) or 1.0
        scales.append(scale)
        joints[i] = (frame.T @ (k[others] - k[ROOT]).T).T / scale
        grab[i] = [np.linalg.norm(k[w] - k[ft]) / scale for w in WRISTS for ft in FEET]
        root[i] = track.camera[f].T @ k[ROOT]
    if usable.sum() < MIN_POSED:
        return None
    at, cumulative, up = at[usable], cumulative[usable], up[usable]
    joints, grab, root = joints[usable], grab[usable], root[usable]
    conf = np.nan_to_num(track.confidence[at].astype(float))
    root = (root - root[0]) / float(np.median(scales))

    cum = _resample(np.rad2deg(cumulative) / 360.0, at, length)
    step = np.diff(cum, axis=0, prepend=cum[:1]) * (length / 8.0)
    upr = _resample(up, at, length)
    upr /= np.maximum(np.linalg.norm(upr, axis=1, keepdims=True), 1e-6)
    jr = _resample(joints.reshape(len(at), -1), at, length)
    gr = _resample(grab, at, length)
    rr = _resample(root, at, length)
    rr = np.diff(rr, axis=0, prepend=rr[:1]) * (length / 8.0)
    present = np.isin(np.arange(at[0], at[-1] + 1), at).astype(float)
    pr = _resample(present, np.arange(at[0], at[-1] + 1), length)
    cr = _resample(conf, at, length)
    out = np.concatenate([cum, step, upr, jr, gr, rr, pr[:, None], cr[:, None]], axis=1)
    return np.nan_to_num(out.T.astype(np.float32))


def augment(track: Track, start: int, end: int, rng: np.random.Generator,
            jitter: tuple[float, float] = (-0.1, 0.3), camera_gain: tuple[float, float] = (0.5, 2.0)
            ) -> tuple[np.ndarray | None, bool]:
    """A randomised encoding of one air. Returns (array, mirrored).

    * Boundaries move by a fraction of the air: ``jitter`` is how far each end
      may move *inward* (negative is outward), since an air detector usually
      cuts an air short (docs/validation.md: takeoff 12 frames late).
    * Frames are thinned by a random stride, which is the same air at a lower
      slow-motion factor, and a random hole is punched in the pose track, as
      the tracker leaves while a rider is inverted.
    * The camera's rotation is scaled: ``camera.py`` over-reads it by a median
      1.6x on a long lens, so the model must not trust it exactly.
    * Half the time the clip is mirrored, and the caller must flip the turn's
      label (``mirrored`` is returned for that).
    """
    n = max(1, end - start)
    lo, hi = jitter
    start = int(round(start + n * rng.uniform(lo, hi)))
    end = int(round(end - n * rng.uniform(lo, hi)))
    if end - start < 8:
        mid = (start + end) // 2
        start, end = mid - 4, mid + 4
    start, end = max(0, start), min(track.frames - 1, end)

    mirrored = bool(rng.uniform() < 0.5)
    t = mirror(track) if mirrored else track
    gain = float(np.exp(rng.uniform(np.log(camera_gain[0]), np.log(camera_gain[1]))))
    if abs(gain - 1.0) > 1e-3:
        t = Track(k3=t.k3, confidence=t.confidence,
                  camera=scale_camera(t.camera, t.segment, gain), segment=t.segment)

    length = end - start + 1
    keep = np.zeros(length, dtype=bool)
    stride = int(rng.integers(1, 5))
    while stride > 1 and length // stride < 16:
        stride -= 1
    keep[int(rng.integers(0, stride))::stride] = True
    if rng.uniform() < 0.3 and length > 30:
        hole = int(rng.integers(4, min(20, length // 4) + 1))
        at = int(rng.integers(0, length - hole))
        keep[at:at + hole] = False
    return encode(t, start, end, keep=keep), mirrored


def scale_camera(rotations: np.ndarray, segment: np.ndarray, gain: float) -> np.ndarray:
    """The camera track with every frame-to-frame rotation scaled by ``gain``."""
    out = rotations.astype(float).copy()
    for i in range(1, len(rotations)):
        if segment[i] != segment[i - 1]:
            continue
        delta = rotations[i] @ rotations[i - 1].T
        out[i] = _exp(gain * rotation_vector(delta)) @ out[i - 1]
    return out.astype(np.float32)


def _exp(vector: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return np.eye(3)
    k = vector / angle
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * kx + (1 - np.cos(angle)) * kx @ kx
