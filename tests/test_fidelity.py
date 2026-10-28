"""The NIM-reading scores that compare two renderers of the same stage.

Hermetic. Pinned: a reading that differs from the truth only by the NIM's
uncalibrated scale, a rotation and an offset scores zero articulation
error; a mirrored body does not (no reflections); frames out of view are
not counted against detection; 2D error is relative to the rider's size.
"""

import numpy as np

from snow.synth import fidelity


def _rotation(rng):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


def _truth(rng, frames=6):
    out = {}
    for f in range(frames):
        joints = rng.normal(size=(77, 3)) * 0.4 + [0.0, 0.0, 30.0]
        pixels = joints[:, :2] / joints[:, 2:] * 2000.0 + [640.0, 360.0]
        out[f] = {"keypoints_3d": joints, "keypoints_2d": pixels}
    return out


def _as_nim(truth, transform=None, conf=0.7):
    out = {}
    for f, t in truth.items():
        joints = t["keypoints_3d"] if transform is None else transform(t["keypoints_3d"])
        out[f] = {"keypoints_3d": joints.tolist(), "keypoints_2d": t["keypoints_2d"].tolist(),
                  "keypoints_confidence": [conf] * 77}
    return out


def test_scale_rotation_and_offset_are_not_articulation_error():
    rng = np.random.default_rng(0)
    truth = _truth(rng)
    rot = _rotation(rng)
    got = _as_nim(truth, lambda j: 0.6 * j @ rot.T + [1.0, -2.0, 5.0])
    s = fidelity.score(got, truth)
    assert s["pa_mpjpe_mm"] < 1e-6
    assert s["kp2d_rel"] == 0.0
    assert s["detected"] == 1.0 and s["confidence"] == 0.7


def test_a_mirrored_body_is_an_error():
    rng = np.random.default_rng(1)
    truth = _truth(rng)
    got = _as_nim(truth, lambda j: j * [-1.0, 1.0, 1.0])
    assert fidelity.score(got, truth)["pa_mpjpe_mm"] > 50.0


def test_frames_out_of_view_do_not_count_as_misses():
    rng = np.random.default_rng(2)
    truth = _truth(rng, frames=10)
    got = {f: v for f, v in _as_nim(truth).items() if f < 5}
    visible = np.array([True] * 5 + [False] * 5)
    assert fidelity.score(got, truth, visible)["detected"] == 1.0
    assert fidelity.score(got, truth)["detected"] == 0.5


def test_2d_error_is_relative_to_the_riders_size():
    rng = np.random.default_rng(3)
    truth = _truth(rng)
    near = {f: {**t, "keypoints_2d": t["keypoints_2d"] * 2.0} for f, t in truth.items()}
    shift = np.array([4.0, 0.0])
    far_got = {f: {**v, "keypoints_2d": (np.asarray(v["keypoints_2d"]) + shift).tolist()}
               for f, v in _as_nim(truth).items()}
    near_got = {f: {**v, "keypoints_2d": (np.asarray(v["keypoints_2d"]) + shift).tolist()}
                for f, v in _as_nim(near).items()}
    far, close = fidelity.score(far_got, truth), fidelity.score(near_got, near)
    assert np.isclose(far["kp2d_rel"], 2.0 * close["kp2d_rel"], rtol=0.02)
