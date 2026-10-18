"""Camera compensation in the USD export, against motion with a known answer.

A rider holds a fixed orientation and position in the world while the camera
pans across them, then the broadcast cuts to a second camera that pans too.
The NIM would report the root turning with each pan. The export must undo
that per shot, so the root is constant within each shot, and it must never
carry the first shot's pan across the cut into the second.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pytest

from snow import usd as U
from snow.camera import CameraTrack
from snow.meta import ClipMeta

from test_rotation import axis_angle

CUT = 30
FRAMES = 60
# Camera frame is Y down, so -Y is up: a pan is a turn about Y.
PAN_1 = [axis_angle(np.array([0.0, 1.0, 0.0]), np.deg2rad(60.0) * i / (CUT - 1))
         for i in range(CUT)]
PAN_2 = [axis_angle(np.array([0.3, 1.0, 0.2]), np.deg2rad(-45.0) * i / (FRAMES - CUT - 1))
         for i in range(FRAMES - CUT)]
# The rider as each shot's world frame sees them; shot 2 is a new camera, so
# a new world frame with an unrelated orientation.
BODY = [axis_angle(np.array([0.2, 1.0, -0.4]), 0.7),
        axis_angle(np.array([1.0, 0.1, 0.3]), -1.1)]
WHERE = [np.array([0.4, 0.2, 6.0]), np.array([-1.0, 0.5, 9.0])]
LOCAL = U.quat_from_matrix(axis_angle(np.array([1.0, 0.0, 0.0]), 0.3)).tolist()


def _shot(frame_id: int) -> tuple[int, np.ndarray]:
    if frame_id < CUT:
        return 0, PAN_1[frame_id]
    return 1, PAN_2[frame_id - CUT]


def _track(accumulate_across_cut: bool = False) -> CameraTrack:
    """The camera track `camera.estimate` would produce for this motion.

    With ``accumulate_across_cut`` it instead keeps composing through the
    cut, as a buggy estimator would; the export must still not leak shot 1's
    pan into shot 2.
    """
    rotations, deltas, segment = [], [], []
    for f in range(FRAMES):
        seg, rotation = _shot(f)
        if accumulate_across_cut and seg == 1:
            rotation = rotation @ PAN_1[-1]
        previous = rotations[-1] if f and _shot(f - 1)[0] == seg else rotation
        rotations.append(rotation)
        deltas.append(rotation @ previous.T)
        segment.append(seg)
    return CameraTrack(
        rotations=np.array(rotations), deltas=np.array(deltas),
        segment=np.array(segment), inliers=np.full(FRAMES, 500),
        reliable=np.ones(FRAMES, dtype=bool), focal=2203.0, frames=FRAMES,
    )


def _write_poses(path) -> None:
    """What the NIM would report: the world pose seen through the panning camera."""
    rest = np.zeros((77, 3))
    for i in range(1, 77):
        rest[i] = [0.0, -0.05 * i, 0.0]
    with path.open("w") as fh:
        for f in range(FRAMES):
            seg, camera = _shot(f)
            root = U.quat_from_matrix(camera @ BODY[seg]).tolist()
            det = {
                "tracking_id": 1,
                "bbox": [0, 0, 10, 10],
                "keypoints_2d": [[0.0, 0.0]] * 77,
                "keypoints_3d": rest.tolist(),
                "keypoints_confidence": [1.0] * 77,
                "rest_pose": rest.tolist(),
                "joint_rotations": [root] + [LOCAL] * 76,
                "root_pose": {
                    "translation": (camera @ WHERE[seg]).tolist(),
                    "rotation": root,
                },
            }
            fh.write(json.dumps({"frame_id": f, "detections": [det]}) + "\n")


def _parse(text: str) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Root translation, root quaternion and joint-1 quaternion per frame, in USD."""
    number = r"-?\d+\.\d+"
    translate = text.split("xformOp:translate.timeSamples")[1].split("}")[0]
    translations = {
        int(m[0]): np.array([float(v) for v in m[1:]])
        for m in re.findall(rf"(\d+): \(({number}), ({number}), ({number})\)", translate)
    }
    roots, joint1 = {}, {}
    rotations = text.split("rotations.timeSamples")[1]
    for line in rotations.splitlines():
        match = re.match(r"\s+(\d+): \[(.*)\],", line)
        if not match:
            continue
        quats = re.findall(rf"\(({number}), ({number}), ({number}), ({number})\)", match[2])
        roots[int(match[1])] = np.array([float(v) for v in quats[0]])
        joint1[int(match[1])] = np.array([float(v) for v in quats[1]])
    return translations, roots, joint1


def _matrix(wxyz: np.ndarray) -> np.ndarray:
    """A USD (w, x, y, z) quaternion as a matrix, so q and -q compare equal."""
    w, x, y, z = wxyz / np.linalg.norm(wxyz)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _export(tmp_path, camera: CameraTrack | None):
    poses = tmp_path / "p.json"
    _write_poses(poses)
    dest = tmp_path / "out.usda"
    report = U.export(poses, dest, ClipMeta(name="t", fps=30.0), camera=camera)
    return report, _parse(dest.read_text())


def test_quaternion_helpers_round_trip():
    for axis, angle in [((0, 1, 0), 0.4), ((1, 0.2, -0.3), 3.1), ((0, 0, 1), -2.9)]:
        rotation = axis_angle(np.array(axis, dtype=float), angle)
        x, y, z, w = U.quat_from_matrix(rotation)
        assert _matrix(np.array([w, x, y, z])) == pytest.approx(rotation, abs=1e-9)
    a, b = BODY
    qa, qb = U.quat_from_matrix(a), U.quat_from_matrix(b)
    x, y, z, w = U.quat_multiply(qa, qb)
    assert _matrix(np.array([w, x, y, z])) == pytest.approx(a @ b, abs=1e-9)


@pytest.mark.parametrize("accumulate", [False, True])
def test_root_is_still_in_each_shots_world_frame(tmp_path, accumulate):
    report, (translations, roots, joint1) = _export(tmp_path, _track(accumulate))
    assert report["camera_compensated"]
    assert report["segments"] == 2

    flip = np.diag([1.0, -1.0, -1.0])  # camera frame -> USD, a half turn about X
    for f in range(FRAMES):
        seg = 0 if f < CUT else 1
        assert _matrix(roots[f]) == pytest.approx(flip @ BODY[seg] @ flip, abs=1e-4), f
        assert translations[f] == pytest.approx(flip @ WHERE[seg], abs=1e-4), f


def test_non_root_joints_are_untouched(tmp_path):
    """Parent-relative rotations do not depend on where the camera points."""
    _, (_, _, compensated) = _export(tmp_path, _track())
    _, (_, _, raw) = _export(tmp_path, None)
    for f in range(FRAMES):
        assert compensated[f] == pytest.approx(raw[f], abs=1e-6)


def test_without_a_camera_the_root_turns_with_the_pan(tmp_path):
    """The defect being fixed, so the compensated test cannot pass vacuously."""
    report, (translations, roots, _) = _export(tmp_path, None)
    assert not report["camera_compensated"]
    first, last = _matrix(roots[0]), _matrix(roots[CUT - 1])
    turned = np.rad2deg(np.arccos(np.clip((np.trace(first.T @ last) - 1) / 2, -1, 1)))
    assert turned == pytest.approx(60.0, abs=0.1)
    assert np.linalg.norm(translations[0] - translations[CUT - 1]) > 1.0


def test_a_short_camera_track_is_refused(tmp_path):
    """Half-compensated animation would splice two reference frames together."""
    track = _track()
    short = CameraTrack(
        rotations=track.rotations[:40], deltas=track.deltas[:40],
        segment=track.segment[:40], inliers=track.inliers[:40],
        reliable=track.reliable[:40], focal=track.focal, frames=40,
    )
    with pytest.raises(ValueError, match="camera track covers 40 frames"):
        _export(tmp_path, short)


def test_compensated_usd_skins_with_pxr(tmp_path):
    """USD's own resolver agrees the root's world transform holds still per shot."""
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom, UsdSkel

    poses = tmp_path / "p.json"
    _write_poses(poses)
    dest = tmp_path / "out.usda"
    U.export(poses, dest, ClipMeta(name="t", fps=30.0), camera=_track())

    stage = Usd.Stage.Open(str(dest))
    cache = UsdSkel.Cache()
    cache.Populate(UsdSkel.Root(stage.GetPrimAtPath("/Rider")), Usd.TraverseInstanceProxies())
    query = cache.GetSkelQuery(UsdSkel.Skeleton(stage.GetPrimAtPath("/Rider/Skel")))
    xform = UsdGeom.Xformable(stage.GetPrimAtPath("/Rider"))

    def head(frame: int) -> np.ndarray:
        """World position of the last joint, which every root error moves."""
        skel = query.ComputeJointSkelTransforms(frame)[76]
        world = skel * xform.ComputeLocalToWorldTransform(frame)
        return np.array(world.ExtractTranslation())

    for start, end in [(0, CUT - 1), (CUT, FRAMES - 1)]:
        assert head(start) == pytest.approx(head(end), abs=1e-3)
