"""How the NIM reads a render, frame by frame, against the exact skeleton.

``compare.py`` scores the *measurements* (spin, tilt, air) a render ends up
with. This scores the step before: the NIM's per-frame reading of the
pixels, so two renderers of the same stage can be compared directly, and a
render can be compared with real broadcast footage where no answer exists.

Per clip, over the frames where the rider is in view:

``detected``
    fraction of those frames with a NIM detection (tracker plus NIM)
``confidence``
    mean ``keypoints_confidence``; the one number that exists for real
    footage too, so "looks like broadcast to the NIM" is measurable
``kp2d_rel``
    median 2D keypoint error over the ground truth's box diagonal
``pa_mpjpe_mm``
    median per-frame error of the root-relative 3D joints after a
    similarity (Procrustes) alignment, in the rig's millimetres. The
    alignment removes the NIM's uncalibrated scale and the rest-pose offset
    of the whole body (invariant 7), and leaves articulation error.

The rig's joints sit on MPFB landmarks and the NIM's on its own, so neither
error reaches zero on a perfect reading; compare renderers on the same
clips, not against zero.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def read_poses(path: Path) -> dict[int, dict]:
    """First detection per frame of a JSON Lines pose file, keyed by frame."""
    out = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("detections"):
            out[int(row["frame_id"])] = row["detections"][0]
    return out


def procrustes_error(got: np.ndarray, true: np.ndarray) -> float:
    """Mean joint distance after the best similarity transform of ``got``.

    Both are (joints, 3). Rotation, uniform scale and translation are
    solved in closed form (Umeyama), reflections excluded.
    """
    a = got - got.mean(axis=0)
    b = true - true.mean(axis=0)
    norm_a = np.sqrt((a ** 2).sum())
    if norm_a == 0:
        return float(np.linalg.norm(b, axis=1).mean())
    u, s, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(u @ vt))
    fix = np.diag([1.0, 1.0, d])
    rotation = u @ fix @ vt
    scale = (s * np.diag(fix)).sum() / norm_a ** 2
    return float(np.linalg.norm(scale * a @ rotation - b, axis=1).mean())


def score(got: dict[int, dict], truth: dict[int, dict],
          visible: np.ndarray | None = None) -> dict:
    """Per-clip fidelity of ``got`` (NIM) against ``truth`` (``gt.poses.json``)."""
    frames = sorted(truth)
    if visible is not None:
        frames = [f for f in frames if f < len(visible) and visible[f]]
    hit = [f for f in frames if f in got]
    out = {"frames": len(frames), "detected": round(len(hit) / max(len(frames), 1), 3)}
    if not hit:
        return out
    conf, rel2d, pa = [], [], []
    for f in hit:
        g, t = got[f], truth[f]
        conf.append(float(np.mean(g["keypoints_confidence"])))
        k2g, k2t = np.asarray(g["keypoints_2d"]), np.asarray(t["keypoints_2d"])
        lo, hi = k2t.min(axis=0), k2t.max(axis=0)
        diag = float(np.hypot(*(hi - lo))) or 1.0
        rel2d.append(float(np.linalg.norm(k2g - k2t, axis=1).mean()) / diag)
        pa.append(procrustes_error(np.asarray(g["keypoints_3d"]), np.asarray(t["keypoints_3d"])))
    out.update({
        "confidence": round(float(np.mean(conf)), 3),
        "kp2d_rel": round(float(np.median(rel2d)), 4),
        "pa_mpjpe_mm": round(1000.0 * float(np.median(pa)), 1),
    })
    return out


def confidence(path: Path) -> dict:
    """Keypoint confidence alone, for footage with no ground truth."""
    conf = [float(np.mean(d["keypoints_confidence"])) for d in read_poses(path).values()]
    if not conf:
        return {"frames": 0}
    return {"frames": len(conf), "confidence": round(float(np.mean(conf)), 3),
            "confidence_p10": round(float(np.percentile(conf, 10)), 3)}


def truth_from_raw(clip: Path) -> tuple[dict[int, dict], np.ndarray | None]:
    """Exact 2D and 3D joints per frame from ``gt_raw.npz``, plus visibility.

    The same numbers ``groundtruth.write_poses`` puts in ``gt.poses.json``,
    computed directly, because a batch box prunes that file after scoring.
    """
    from . import groundtruth

    raw = groundtruth.load_raw(clip)
    points, _ = groundtruth.camera_frame(raw)
    pixels = groundtruth.project(points, raw["focal_px"], int(raw["width"]), int(raw["height"]))
    truth = {f: {"keypoints_2d": pixels[f], "keypoints_3d": points[f]}
             for f in range(len(points))}
    return truth, raw.get("visible")


def clip_score(clip: Path, poses: Path) -> dict:
    """``score`` for one synthetic clip's NIM output."""
    truth, visible = truth_from_raw(clip)
    if not poses.exists():
        return {"frames": len(truth), "detected": 0.0}
    return score(read_poses(poses), truth, visible)
