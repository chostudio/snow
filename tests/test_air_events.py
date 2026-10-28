"""The dashboard's air detector: one trick is one air, whatever the pipeline
drops or misreads inside it, and a wind-up before it is not part of it.

On synthetic tricks through the pipeline, three things split or misread an
air: a hole in the pose track while the rider was inverted, a few frames of
pose flip that set the clip's peak rate, and a stretch where the signed
path read the turn backwards (docs/validation.md).
"""

import importlib.util

import numpy as np
import pytest

from snow import config as cfg

FPS = 30.0


def _exporter():
    path = cfg.REPO_ROOT / "web" / "tools" / "export_session.py"
    spec = importlib.util.spec_from_file_location("export_session", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


es = _exporter()


def _recs(steps, missing=()):
    """Per-frame records from per-frame (signed path, vertical yaw) steps.

    The unsigned path grows by each signed step's size, as `snow features`
    writes it.
    """
    recs, rotation, path, yaw = [], 0.0, 0.0, 0.0
    for f, (turn, yaw_turn) in enumerate(steps):
        rotation += turn
        path += abs(turn)
        yaw += yaw_turn
        if f in missing:
            continue
        recs.append({"f": f, "segment": 0, "rotation": rotation, "path": path, "yaw": yaw,
                     "ankle_height": 0.0, "axis_tilt": 0.0, "tuck_radius": 0.3,
                     "knee_flex": 120.0, "conf": 0.9})
    return recs


def _events(recs):
    es.compute_rates(recs, FPS)
    return es.find_events(recs, FPS, FPS)


def _clockwise_air(n=200, rate=5.4):
    """Ground, a clockwise 1080 over ``n`` frames, ground."""
    return [(0.0, 0.0)] * 50 + [(-rate, -rate)] * n + [(0.0, 0.0)] * 50


def test_a_hole_inside_the_air_does_not_split_it():
    events = _events(_recs(_clockwise_air(), missing=set(range(140, 155))))
    assert len(events) == 1
    assert events[0]["metrics"]["nearest_spin_deg"] == 1080


def test_a_few_flip_frames_do_not_gate_out_the_spin():
    steps = _clockwise_air()
    for f in (120, 121, 122, 180, 181):  # the pose flips as the rider goes over
        steps[f] = (-60.0, -60.0)
    events = _events(_recs(steps))
    assert len(events) == 1
    assert events[0]["start_f"] <= 55 and events[0]["end_f"] >= 245


def test_a_stretch_read_backwards_keeps_the_air_and_its_sense():
    # Signed path reverses for 40 frames mid-air while the yaw keeps turning
    # the trick's way: a measurement error, not a second trick.
    steps = _clockwise_air()
    for f in range(110, 150):
        steps[f] = (5.4, -2.0)
    events = _events(_recs(steps))
    assert len(events) == 1
    m = events[0]["metrics"]
    assert m["direction"] == "clockwise"
    assert m["rotation_deg"] == pytest.approx(-1080, abs=30)


def test_a_wind_up_before_takeoff_is_not_counted():
    steps = _clockwise_air()
    for f in range(20, 50):  # counter-clockwise wind-up, read by both measures
        steps[f] = (5.0, 5.0)
    events = _events(_recs(steps))
    assert len(events) == 1
    assert events[0]["start_f"] >= 48
    assert events[0]["metrics"]["rotation_deg"] == pytest.approx(-1080, abs=30)


# With the body orientation in the features table (``body_q*``), the detector
# reads a chord speed and the air is counted by ``features.spin_count_deg``.

UP = np.array([0.0, -1.0, 0.0])


def _quat(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    return np.concatenate([axis * np.sin(angle / 2), [np.cos(angle / 2)]])


def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


def _oriented_recs(turns_deg, jitter_deg=0.0, seed=0):
    """Per-frame records from per-frame turns about world up, each frame's
    orientation also jittered independently (pose noise), with ``path`` and
    ``rotation`` derived frame to frame as ``snow features`` derives them,
    so the path sums the jitter."""
    rng = np.random.default_rng(seed)
    true, recs = 0.0, []
    previous, path, rotation = None, 0.0, 0.0
    for f, turn in enumerate(turns_deg):
        true += turn
        q = _quat(UP, np.deg2rad(true))
        if jitter_deg:
            q = _qmul(_quat(rng.normal(size=3), np.deg2rad(rng.normal(0.0, jitter_deg))), q)
        if previous is not None:
            step = float(np.rad2deg(2 * np.arccos(min(1.0, abs(float(np.dot(q, previous)))))))
            path += step
            rotation += (np.sign(turn) if turn else 1.0) * step
        previous = q
        recs.append({"f": f, "segment": 0, "rotation": rotation, "path": path, "yaw": true,
                     "q": q.tolist(), "ankle_height": 0.0, "axis_tilt": 0.0,
                     "tuck_radius": 0.3, "knee_flex": 120.0, "conf": 0.9})
    return recs


@pytest.mark.parametrize("slowmo", [1, 4])
def test_a_slowed_noisy_spin_is_one_air_counted_right(slowmo):
    # A clockwise 1080 over 2 s of real time, conformed to 30 fps with no
    # factor given, under the same per-frame pose noise at every speed.
    n = 60 * slowmo
    turns = [0.0] * (25 * slowmo) + [-1080.0 / n] * n + [0.0] * (25 * slowmo)
    events = _events(_oriented_recs(turns, jitter_deg=3.0))
    assert len(events) == 1
    m = events[0]["metrics"]
    assert m["nearest_spin_deg"] == 1080
    assert m["direction"] == "clockwise"
    # The frame-to-frame path over the same air, kept for comparison, is
    # what the dashboard used to show: at 4x it reads a step or more high.
    if slowmo == 4:
        assert abs(m["path_rotation_deg"]) > 1080 + 90


def test_a_wind_up_is_trimmed_with_orientations_too():
    # 120 degrees the other way before a clockwise 1080. Read over a chord,
    # the turn's rate dips where it reverses, so growth stops there; the
    # trim (with orientations, only outside the air's fast core) is the
    # second line.
    turns = [0.0] * 20 + [2.0] * 60 + [-1080.0 / 200] * 200 + [0.0] * 50
    events = _events(_oriented_recs(turns))
    assert len(events) == 1
    main = events[0]
    assert main["start_f"] >= 75
    assert main["metrics"]["rotation_deg"] == pytest.approx(-1080, abs=40)
