"""Validate the rotation maths against synthetic motion with a known answer.

Real footage cannot settle whether ``axis_spin_deg`` is right: the trick's
true rotation is only known from a commentator's label, and the shot may not
even be the trick that label names. A synthetic rider rotating by an exact
amount about an exact axis has no such ambiguity.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from snow import features as F
from snow.camera import CameraTrack, rotation_vector
from snow.meta import ClipMeta

# Camera frame: X right, Y DOWN, Z forward. World up is therefore -Y.
REST = {
    F.ROOT: np.array([0.0, 0.0, 0.0]),
    F.NECK: np.array([0.0, -0.5, 0.0]),
    F.HIP_L: np.array([-0.11, 0.0, 0.0]),
    F.HIP_R: np.array([0.11, 0.0, 0.0]),
}


def axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


_SEQUENCE = [0]


def synthetic_poses(tmp_path, total_deg: float, axis: np.ndarray, frames: int = 361):
    """A rider rotating by exactly `total_deg` about `axis`, written as JSONL.

    Each call gets its own file; sharing one filename within a test silently
    overwrites the earlier clip and makes both variables point at the same
    data.
    """
    rotations = [axis_angle(axis, np.deg2rad(total_deg) * i / (frames - 1))
                 for i in range(frames)]
    return poses_from(tmp_path, rotations)


def poses_from(tmp_path, rotations: list[np.ndarray]):
    """A rider taking each of ``rotations`` in turn, one per frame, as JSONL."""
    _SEQUENCE[0] += 1
    path = tmp_path / f"poses_{_SEQUENCE[0]}.json"
    depth = np.array([0.0, 0.0, 6.0])
    with path.open("w") as fh:
        for i, rotation in enumerate(rotations):
            k3 = np.zeros((77, 3))
            for index, point in REST.items():
                k3[index] = rotation @ point + depth
            fh.write(
                json.dumps(
                    {
                        "frame_id": i,
                        "detections": [
                            {
                                "tracking_id": 1,
                                "bbox": [0.0, 0.0, 10.0, 10.0],
                                "keypoints_2d": [[0.0, 0.0]] * 77,
                                "keypoints_3d": k3.tolist(),
                                "keypoints_confidence": [1.0] * 77,
                                "rest_pose": k3.tolist(),
                                "joint_rotations": [[0.0, 0.0, 0.0, 1.0]] * 77,
                                "root_pose": {
                                    "translation": list(k3[F.ROOT]),
                                    "rotation": [0.0, 0.0, 0.0, 1.0],
                                },
                            }
                        ],
                    }
                )
                + "\n"
            )
    return path


def meta() -> ClipMeta:
    return ClipMeta(name="synthetic", fps=30.0, frames=361)


@pytest.mark.parametrize("total", [360.0, 720.0, 1080.0])
def test_vertical_spin_is_recovered(tmp_path, total):
    """A flat spin about the vertical: every measure should agree."""
    poses = synthetic_poses(tmp_path, total, np.array([0.0, -1.0, 0.0]))
    df = F.build(poses, meta())

    assert df["axis_spin_deg"].dropna().iloc[-1] == pytest.approx(total, abs=2.0)
    assert abs(df["spin_deg"].dropna().iloc[-1]) == pytest.approx(total, abs=2.0)
    assert df["rotation_deg"].dropna().iloc[-1] == pytest.approx(total, abs=2.0)
    assert df["tilt_deg"].max() == pytest.approx(0.0, abs=1.0)


def test_tilted_axis_spin_is_recovered_and_its_tilt_reported(tmp_path):
    """A 1080 about an axis tilted 35 degrees is still a 1080.

    Note what this also pins down: vertical yaw agrees. Tilting the rotation
    axis does **not** make ``spin_deg`` undercount, because the horizontal
    projection still winds once per revolution. The next test covers the case
    that genuinely breaks it.
    """
    tilt = 35.0
    axis = np.array([np.sin(np.deg2rad(tilt)), -np.cos(np.deg2rad(tilt)), 0.0])
    poses = synthetic_poses(tmp_path, 1080.0, axis)
    df = F.build(poses, meta())
    segment = F.summarize(df, meta())["segments"][0]

    assert segment["axis_spin_deg"] == pytest.approx(1080.0, abs=5.0)
    assert segment["spin_axis_tilt_deg"] == pytest.approx(tilt, abs=2.0)
    assert abs(segment["net_spin_deg"]) == pytest.approx(1080.0, abs=5.0)


def test_axis_spin_survives_where_vertical_yaw_degenerates(tmp_path):
    """The case that actually motivates axis_spin_deg.

    Rolling about the view axis sweeps the hip axis through the vertical
    twice per revolution. There the horizontal projection collapses and the
    yaw reading is meaningless, while the dominant-axis measure is unaffected.
    """
    poses = synthetic_poses(tmp_path, 720.0, np.array([0.0, 0.0, 1.0]))
    df = F.build(poses, meta())
    segment = F.summarize(df, meta())["segments"][0]

    assert segment["axis_spin_deg"] == pytest.approx(720.0, abs=5.0)
    # The hip axis really does pass through vertical, so quality bottoms out.
    assert df["spin_quality"].min() < 0.05
    # ...and vertical yaw does not recover the rotation.
    assert abs(segment["net_spin_deg"]) < 360.0


@pytest.mark.parametrize("sense", [1.0, -1.0])
def test_spin_measures_carry_the_turns_sense(tmp_path, sense):
    """Counter-clockwise from above is positive, clockwise negative.

    ``axis_spin_deg`` used to project onto its own net axis, which made it
    grow positive whatever the rider did, and the dashboard called every
    spin counter-clockwise.
    """
    up = np.array([0.0, -1.0, 0.0])
    poses = synthetic_poses(tmp_path, sense * 720.0, up)
    df = F.build(poses, meta())
    segment = F.summarize(df, meta())["segments"][0]
    assert segment["axis_spin_deg"] == pytest.approx(sense * 720.0, abs=5.0)
    assert segment["spin_path_deg"] == pytest.approx(sense * 720.0, abs=5.0)
    assert np.sign(segment["net_spin_deg"]) == sense


def test_spin_path_counts_a_cork_whose_axis_moves(tmp_path):
    """A turn whose axis swings from vertical to horizontal and back.

    Projecting onto one dominant axis drops what was not about it, so
    ``axis_spin_deg`` undercounts; the signed path counts all of it, and
    keeps the sense through the stretch where the axis is horizontal.
    """
    up = np.array([0.0, -1.0, 0.0])
    side = np.array([1.0, 0.0, 0.0])
    frames, total = 361, 1080.0
    rotations, current = [np.eye(3)], np.eye(3)
    for i in range(1, frames):
        lean = np.pi / 2 * np.sin(np.pi * i / (frames - 1)) ** 2   # 0 -> 90 -> 0 deg
        axis = np.cos(lean) * up + np.sin(lean) * side
        current = axis_angle(axis, np.deg2rad(total / (frames - 1))) @ current
        rotations.append(current)
    df = F.build(poses_from(tmp_path, rotations), meta())
    segment = F.summarize(df, meta())["segments"][0]
    assert segment["spin_path_deg"] == pytest.approx(total, abs=10.0)
    assert segment["axis_spin_deg"] < total - 100.0


def test_camera_rotation_is_removed(tmp_path):
    """Compensation must move rotation between the camera and the rider.

    A rider who holds a fixed orientation *relative to a panning camera* is
    really turning in the world, and a rider who turns exactly with the
    camera is really not. Both readings are wrong before compensation and
    right after it, which is the whole point of the stage.
    """
    poses = synthetic_poses(tmp_path, 0.0, np.array([0.0, -1.0, 0.0]), frames=120)

    # The camera yaws 1 degree per frame; the rider does nothing.
    step = axis_angle(np.array([0.0, -1.0, 0.0]), np.deg2rad(1.0))
    rotations, deltas = [], []
    current = np.eye(3)
    for _ in range(120):
        rotations.append(current.copy())
        deltas.append(step)
        current = step @ current
    track = CameraTrack(
        rotations=np.array(rotations), deltas=np.array(deltas),
        segment=np.zeros(120, dtype=int), inliers=np.full(120, 500),
        reliable=np.ones(120, dtype=bool), focal=2203.0, frames=120,
    )

    # Fixed in the camera's view while the camera pans 119 degrees: the
    # rider turned 119 degrees in the world, and only compensation sees it.
    assert F.build(poses, meta(), camera=None)["rotation_deg"].dropna().iloc[
        -1
    ] == pytest.approx(0.0, abs=2.0)
    assert F.build(poses, meta(), camera=track)["rotation_deg"].dropna().iloc[
        -1
    ] == pytest.approx(119.0, abs=3.0)

    # Turning exactly with the camera: the apparent rotation is entirely the
    # camera's, and compensation must cancel it to nothing.
    spinning = synthetic_poses(tmp_path, 119.0, np.array([0.0, -1.0, 0.0]), frames=120)
    assert F.build(spinning, meta(), camera=None)["axis_spin_deg"].dropna().iloc[
        -1
    ] == pytest.approx(119.0, abs=3.0)
    cancelled = F.build(spinning, meta(), camera=track)
    assert cancelled["rotation_deg"].dropna().iloc[-1] == pytest.approx(0.0, abs=3.0)
    assert cancelled["axis_spin_deg"].dropna().iloc[-1] == pytest.approx(0.0, abs=4.0)


def test_rotation_vector_round_trips():
    for angle in (0.0, 0.3, np.pi / 2, 3.0):
        axis = np.array([0.3, -0.8, 0.5])
        axis = axis / np.linalg.norm(axis)
        vector = rotation_vector(axis_angle(axis, angle))
        assert np.linalg.norm(vector) == pytest.approx(angle, abs=1e-6)
        if angle > 1e-6:
            assert np.allclose(vector / angle, axis, atol=1e-6)


def test_principal_point_correction_is_depth_dependent():
    """The depth dependence is the part that would otherwise fake rotation."""
    keypoints = np.array([[0.0, 0.0, 4.0], [0.0, 0.0, 8.0]])
    corrected = F.correct_principal_point(keypoints, (100.0, 0.0), 2000.0)

    assert corrected[0, 0] == pytest.approx(-0.2)
    assert corrected[1, 0] == pytest.approx(-0.4)
    assert np.array_equal(corrected[:, 2], keypoints[:, 2])  # depth untouched
    assert np.array_equal(
        F.correct_principal_point(keypoints, (0.0, 0.0), 2000.0), keypoints
    )


def _jittered(rotations: list[np.ndarray], jitter_deg: float, seed: int = 0) -> list[np.ndarray]:
    """Each orientation perturbed by an independent random rotation, as pose
    noise is: per frame, not accumulating."""
    rng = np.random.default_rng(seed)
    out = []
    for rotation in rotations:
        axis = rng.normal(size=3)
        out.append(axis_angle(axis, np.deg2rad(rng.normal(0.0, jitter_deg))) @ rotation)
    return out


def _quaternions(df) -> np.ndarray:
    return df[list(F.ORIENTATION_COLUMNS)].dropna().to_numpy()


def _cork_rotations(total: float, frames: int) -> list[np.ndarray]:
    up, side = np.array([0.0, -1.0, 0.0]), np.array([1.0, 0.0, 0.0])
    rotations, current = [np.eye(3)], np.eye(3)
    for i in range(1, frames):
        lean = np.pi / 3 * np.sin(np.pi * i / (frames - 1)) ** 2   # 0 -> 60 -> 0 deg
        axis = np.cos(lean) * up + np.sin(lean) * side
        current = axis_angle(axis, np.deg2rad(total / (frames - 1))) @ current
        rotations.append(current)
    return rotations


@pytest.mark.parametrize("slowmo", [1, 4])
def test_spin_count_ignores_pose_jitter_at_any_slow_motion(tmp_path, slowmo):
    """The same corked 1080 at real time and at 4x slow motion, with the same
    per-frame pose noise: the angular path frame to frame grows with the
    number of frames, the count does not, and needs no slow-motion factor.

    This is the synthetic training set's slow-motion failure
    (docs/validation.md): k times the frames for the same turn, so k times
    the jitter summed into ``rotation_deg``.
    """
    frames = 90 * slowmo
    rotations = _jittered(_cork_rotations(1080.0, frames), jitter_deg=4.0)
    df = F.build(poses_from(tmp_path, rotations), ClipMeta(name="synthetic", fps=30.0, frames=frames))
    path = df["rotation_deg"].dropna().iloc[-1]
    count = F.spin_count_deg(_quaternions(df))
    assert count == pytest.approx(1080.0, abs=60.0)
    if slowmo == 4:
        assert path > 1080.0 + 180.0   # the old count would name a 1260 or more


def test_spin_count_is_exact_on_clean_motion(tmp_path):
    """No jitter, any axis motion: the count agrees with the path."""
    rotations = _cork_rotations(720.0, 181)
    df = F.build(poses_from(tmp_path, rotations), meta())
    assert F.spin_count_deg(_quaternions(df)) == pytest.approx(
        df["rotation_deg"].dropna().iloc[-1], abs=15.0)


def test_orientation_columns_are_the_camera_compensated_frame(tmp_path):
    """The stored quaternion turns with the rider in the world, not in the
    camera: a rider fixed in a panning camera's view turned in the world."""
    poses = synthetic_poses(tmp_path, 0.0, np.array([0.0, -1.0, 0.0]), frames=120)
    step = axis_angle(np.array([0.0, -1.0, 0.0]), np.deg2rad(1.0))
    rotations, current = [], np.eye(3)
    for _ in range(120):
        rotations.append(current.copy())
        current = step @ current
    track = CameraTrack(
        rotations=np.array(rotations), deltas=np.array([step] * 120),
        segment=np.zeros(120, dtype=int), inliers=np.full(120, 500),
        reliable=np.ones(120, dtype=bool), focal=2203.0, frames=120,
    )
    q = _quaternions(F.build(poses, meta(), camera=track))
    assert F.chord_angles(q, np.array([0]), np.array([len(q) - 1]))[0] == pytest.approx(119.0, abs=2.0)


def test_robust_speed_span_follows_the_turn_not_the_frame_rate():
    """The chord span is read from the clip itself: a slowed turn gets a
    longer span, in proportion, without being told the factor."""
    up = np.array([0.0, -1.0, 0.0])

    def span_for(slowmo: int) -> int:
        frames = 60 * slowmo
        rotations = [axis_angle(up, np.deg2rad(720.0) * i / (frames - 1)) for i in range(frames)]
        from snow.usd import quat_from_matrix
        q = np.array([quat_from_matrix(r) for r in _jittered(rotations, 2.0)])
        _, span = F.robust_speed(q, np.arange(frames), np.zeros(frames, dtype=int))
        return span

    assert span_for(1) <= 3
    assert span_for(4) >= 3 * span_for(1)
