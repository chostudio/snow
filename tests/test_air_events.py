"""The dashboard's air detector: one trick is one air, whatever the pipeline
drops or misreads inside it, and a wind-up before it is not part of it.

On synthetic tricks through the pipeline, three things split or misread an
air: a hole in the pose track while the rider was inverted, a few frames of
pose flip that set the clip's peak rate, and a stretch where the signed
path read the turn backwards (docs/validation.md).
"""

import importlib.util

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
