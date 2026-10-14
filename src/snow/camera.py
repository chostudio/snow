"""Estimate the broadcast camera's own rotation, so the rider's isn't confused with it.

``root_pose.rotation`` from the NIM is expressed in **camera** coordinates. A
halfpipe camera pans hard to follow the rider, so a 1080 does not integrate to
1080 degrees of root yaw — the camera's pan is mixed in, with the wrong sign.

For a camera that pans, tilts and zooms about a fixed centre, consecutive
frames are related by ``H = K R K^-1``. Tracking background features (with the
rider masked out) gives ``H``; then ``R = K^-1 H K``, orthonormalised to the
nearest true rotation, which also absorbs the scale that zoom introduces.

This model has limits worth stating. A camera that *translates* (a dolly, or a
handheld operator walking) breaks the fixed-centre assumption, and parallax
from the near pipe wall then leaks into the estimate. And at a hard cut the
orientation jumps arbitrarily — there is no continuous rotation between two
different cameras — so accumulation restarts and each shot gets its own world
frame. A trick that spans a cut cannot be integrated across it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class CameraTrack:
    """Per-frame camera orientation, in segments separated by cuts."""

    rotations: np.ndarray  # (frames, 3, 3), world->camera within the segment
    deltas: np.ndarray     # (frames, 3, 3), rotation from frame i-1 to i
    segment: np.ndarray    # (frames,) int, which shot each frame belongs to
    inliers: np.ndarray    # (frames,) int, feature matches behind each estimate
    reliable: np.ndarray   # (frames,) bool, False where the estimate was a guess
    focal: float
    frames: int

    def save(self, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            dest,
            rotations=self.rotations, deltas=self.deltas,
            segment=self.segment, inliers=self.inliers,
            reliable=self.reliable, focal=np.array([self.focal]),
        )

    @classmethod
    def load(cls, src: Path) -> "CameraTrack":
        z = np.load(src)
        return cls(
            rotations=z["rotations"], deltas=z["deltas"], segment=z["segment"],
            inliers=z["inliers"],
            reliable=z["reliable"] if "reliable" in z
            else np.ones(len(z["segment"]), dtype=bool),
            focal=float(z["focal"][0]), frames=len(z["segment"]),
        )


def _nearest_rotation(matrix: np.ndarray) -> np.ndarray:
    """Closest true rotation to an arbitrary 3x3, by SVD."""
    u, _, vt = np.linalg.svd(matrix)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return rotation


def rotation_angle(rotation: np.ndarray) -> float:
    """Magnitude of a rotation, in radians."""
    cos = (np.trace(rotation) - 1.0) / 2.0
    return float(np.arccos(np.clip(cos, -1.0, 1.0)))


def rotation_vector(rotation: np.ndarray) -> np.ndarray:
    """Log map: a rotation as axis * angle, in radians."""
    angle = rotation_angle(rotation)
    if angle < 1e-9:
        return np.zeros(3)
    axis = np.array([
        rotation[2, 1] - rotation[1, 2],
        rotation[0, 2] - rotation[2, 0],
        rotation[1, 0] - rotation[0, 1],
    ])
    norm = np.linalg.norm(axis)
    if norm < 1e-9:  # angle near pi; axis from the symmetric part
        values, vectors = np.linalg.eigh(rotation + np.eye(3))
        axis = vectors[:, int(np.argmax(values))]
        return axis / np.linalg.norm(axis) * angle
    return axis / norm * angle


def estimate(
    video: Path,
    boxes: dict[int, tuple[float, float, float, float]] | None = None,
    focal: float | None = None,
    max_features: int = 1200,
    min_inliers: int = 25,
    cut_difference: float = 22.0,
    max_rotation_deg: float = 15.0,
    mask_margin: float = 0.6,
    progress: bool = True,
) -> CameraTrack:
    """Track camera orientation frame by frame.

    Args:
        boxes: rider box per frame, masked out so the subject's own motion
            never drives the estimate.
        focal: pixels. Defaults to the image diagonal, which is what the NIM
            itself assumes.
    """
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if focal is None:
        focal = float(np.hypot(width, height))
    K = np.array([[focal, 0, width / 2.0], [0, focal, height / 2.0], [0, 0, 1.0]])
    K_inv = np.linalg.inv(K)

    rotations: list[np.ndarray] = []
    deltas: list[np.ndarray] = []
    segments: list[int] = []
    inliers: list[int] = []
    reliable: list[bool] = []

    prev_gray = None
    prev_small = None
    current = np.eye(3)
    segment = 0
    frame_id = 0
    max_rotation = np.deg2rad(max_rotation_deg)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (320, 180), interpolation=cv2.INTER_AREA)

        # A hard cut is a whole-frame change, not merely a hard estimate.
        # Conflating the two shatters a long shot into dozens of fragments
        # every time the tracker runs short of background texture.
        is_cut = prev_small is not None and (
            float(np.mean(cv2.absdiff(small, prev_small))) > cut_difference
        )

        delta = np.eye(3)
        count = 0
        ok_estimate = False
        if prev_gray is not None:
            mask = np.full(gray.shape, 255, dtype=np.uint8)
            box = (boxes or {}).get(frame_id) or (boxes or {}).get(frame_id - 1)
            if box:
                x, y, w, h = box
                pad_x, pad_y = w * mask_margin, h * mask_margin
                x0 = max(0, int(x - pad_x)); y0 = max(0, int(y - pad_y))
                x1 = min(gray.shape[1], int(x + w + pad_x))
                y1 = min(gray.shape[0], int(y + h + pad_y))
                mask[y0:y1, x0:x1] = 0

            prev_points = cv2.goodFeaturesToTrack(
                prev_gray, maxCorners=max_features, qualityLevel=0.01,
                minDistance=12, mask=mask, blockSize=7,
            )
            if prev_points is not None and len(prev_points) >= min_inliers:
                next_points, status, _ = cv2.calcOpticalFlowPyrLK(
                    prev_gray, gray, prev_points, None,
                    winSize=(21, 21), maxLevel=3,
                )
                good_prev = prev_points[status.ravel() == 1].reshape(-1, 2)
                good_next = next_points[status.ravel() == 1].reshape(-1, 2)
                if len(good_prev) >= min_inliers:
                    homography, inlier_mask = cv2.findHomography(
                        good_prev, good_next, cv2.RANSAC, 3.0, maxIters=2000
                    )
                    if homography is not None:
                        count = int(inlier_mask.sum()) if inlier_mask is not None else 0
                        if count >= min_inliers:
                            candidate = _nearest_rotation(K_inv @ homography @ K)
                            if rotation_angle(candidate) <= max_rotation:
                                delta = candidate
                                ok_estimate = True

        if is_cut:
            # No continuous rotation exists across a cut, so the next shot
            # gets its own world frame.
            segment += 1
            current = np.eye(3)
            delta = np.eye(3)
        else:
            # A frame we could not solve is carried as "no rotation" and
            # flagged, rather than ending the segment.
            current = delta @ current

        rotations.append(current.copy())
        deltas.append(delta)
        segments.append(segment)
        inliers.append(count)
        reliable.append(bool(ok_estimate or prev_gray is None))

        prev_gray = gray
        prev_small = small
        frame_id += 1
        if progress and frame_id % 300 == 0:
            print(f"    camera: {frame_id} frames, {segment + 1} segment(s)")
    cap.release()

    return CameraTrack(
        rotations=np.array(rotations), deltas=np.array(deltas),
        segment=np.array(segments), inliers=np.array(inliers),
        reliable=np.array(reliable, dtype=bool), focal=focal, frames=frame_id,
    )


def summarize(track: CameraTrack) -> dict:
    per_segment = []
    for seg in sorted(set(track.segment.tolist())):
        idx = np.nonzero(track.segment == seg)[0]
        total = sum(rotation_angle(track.deltas[i]) for i in idx)
        per_segment.append(
            {
                "segment": int(seg),
                "frames": int(len(idx)),
                "start_frame": int(idx[0]),
                "end_frame": int(idx[-1]),
                "camera_rotation_deg": float(np.rad2deg(total)),
                "median_inliers": int(np.median(track.inliers[idx])),
                "reliable_fraction": float(np.mean(track.reliable[idx])),
            }
        )
    return {
        "segments": len(per_segment),
        "focal": track.focal,
        "detail": per_segment,
    }
