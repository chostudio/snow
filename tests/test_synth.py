"""The synthetic ground truth: its labels, its skeleton, its file formats.

Hermetic: no Blender, no NIM. What is pinned is what the validation table
depends on: tricks really have the inversions their names claim, labels are
measured from motion rather than copied from the request, and ground truth
written in the NIM's schema obeys the same forward kinematics as the NIM.
"""

import numpy as np
import pytest

from snow.camera import rotation_angle
from snow.skeleton import PARENTS
from snow.synth import groundtruth, nova, tricks


def _trick(**kw):
    base = dict(name="t", spin_deg=1080, kind="double_cork", direction="frontside",
                stance="regular", grab="none", camera_angle="side", slowmo=1.0, seed=3)
    base.update(kw)
    trick = tricks.Trick(**base)
    tricks.solve_tilt(trick)
    return trick


@pytest.mark.parametrize("spin,kind,wanted", [
    (720, "flat", 0), (540, "cork", 1), (1080, "cork", 1),
    (1080, "double_cork", 2), (1440, "double_cork", 2),
])
def test_named_inversions_are_real(spin, kind, wanted):
    trick = _trick(spin_deg=spin, kind=kind)
    labels = tricks.labels(trick, tricks.motion(trick))
    assert labels["inversions"] == wanted
    if wanted:
        assert labels["max_tilt_deg"] >= 110.0
    else:
        assert labels["max_tilt_deg"] < 60.0


def test_double_cork_needs_room_for_two_inversions():
    with pytest.raises(ValueError):
        tricks.inversion_targets(360, "double_cork")


@pytest.mark.parametrize("stance,direction,sign", [
    ("regular", "frontside", 1), ("regular", "backside", -1),
    ("goofy", "frontside", -1), ("goofy", "backside", 1),
])
def test_spin_sense_follows_stance_and_direction(stance, direction, sign):
    trick = _trick(spin_deg=720, kind="flat", stance=stance, direction=direction)
    motion = tricks.motion(trick)
    assert trick.spin_sign == sign
    # Facing direction (rig -Y) turns the right way about world up.
    facing = -motion.rotation[:, :, 1]
    yaw = np.unwrap(np.arctan2(facing[:, 1], facing[:, 0]))
    assert np.sign(yaw[-1] - yaw[0]) == sign
    assert abs(np.rad2deg(yaw[-1] - yaw[0])) == pytest.approx(720.0, abs=0.5)


def test_landing_is_clean():
    """After the air, the rider faces where a clean landing needs them to."""
    trick = _trick(spin_deg=1260, kind="double_cork", direction="backside")
    motion = tricks.motion(trick)
    target = (tricks.rot_x(-trick.landing_deg)
              @ tricks.rot_z(trick.spin_sign * trick.spin_deg)
              @ tricks.stance_rotation(trick.stance))
    error = rotation_angle(motion.rotation[motion.landing_frame] @ target.T)
    assert np.rad2deg(error) < 0.01


def test_slow_motion_stretches_frames_not_the_trick():
    fast = _trick(slowmo=1.0)
    slow = _trick(slowmo=4.0)
    a, b = tricks.labels(fast, tricks.motion(fast)), tricks.labels(slow, tricks.motion(slow))
    assert b["air_frames"] == pytest.approx(4 * a["air_frames"], abs=2)
    # Same trick, integrated in 4x finer steps: equal to within a few degrees.
    assert b["axis_spin_deg"] == pytest.approx(a["axis_spin_deg"], abs=5.0)


def test_axis_spin_reads_below_the_named_spin_on_exact_motion():
    """The dominant-axis projection drops off-axis rotation: a definition
    property the validation table must not blame on the NIM."""
    trick = _trick()
    labels = tricks.labels(trick, tricks.motion(trick))
    assert labels["axis_spin_deg"] < trick.spin_deg - 20.0


def test_plan_is_twenty_tricks_with_exact_labels():
    plan = tricks.plan(seed=0)
    assert len(plan) == 20
    assert {t.kind for t in plan} == set(tricks.KINDS)
    assert {t.camera_angle for t in plan} == set(tricks.CAMERA_ANGLES)
    assert min(t.spin_deg for t in plan) == 360 and max(t.spin_deg for t in plan) == 1440


# --- the skeleton -----------------------------------------------------------

def test_nova_names_and_landmarks_cover_all_77():
    assert len(nova.JOINT_NAMES) == len(PARENTS) == 77
    assert sorted(nova.landmarks()) == list(range(77))
    # The indices features.py and the dashboard name.
    assert nova.JOINT_NAMES[0] == "pelvis"
    assert nova.JOINT_NAMES[67] == "hip_l" and nova.JOINT_NAMES[72] == "hip_r"
    assert nova.JOINT_NAMES[14] == "wrist_l" and nova.JOINT_NAMES[42] == "wrist_r"
    assert nova.JOINT_NAMES[71] == "toe_l" and nova.JOINT_NAMES[76] == "toe_r"


@pytest.mark.parametrize("mpfb,index", [
    ("upperleg02.L", 67), ("lowerleg01.R", 73), ("foot.L", 69), ("toe4-2.R", 75),
    ("upperarm01.L", 12), ("lowerarm02.R", 41), ("wrist.R", 42),
    ("finger1-2.L", 16), ("metacarpal1.R", 47), ("finger3-3.L", 27),
    ("spine01", 3), ("neck03", 5), ("head", 6), ("eye.L", 9), ("jaw", 8),
    ("orbicularis03.R", 6),
])
def test_mpfb_weights_land_on_the_right_nova_bone(mpfb, index):
    assert nova.weight_target(mpfb) == index


def test_names_match_usd_joint_tokens():
    from snow.usd import joint_paths
    assert [p.rsplit("/", 1)[-1] for p in joint_paths()] == [
        nova.bone_name(i) for i in range(77)]


# --- ground truth in the NIM's schema ----------------------------------------

def _random_rotation(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _quat_matrix(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def test_ground_truth_obeys_the_nims_forward_kinematics(tmp_path):
    """Posed rig -> gt.poses.json -> UsdSkel FK gives back keypoints_3d.

    The NIM's own output satisfies exactly this (checked on a real clip to
    1e-4 m), which is what makes the two files compare joint for joint.
    """
    import json

    rng = np.random.default_rng(0)
    frames = 3
    rest = rng.normal(scale=0.2, size=(77, 3))
    rest[0] = 0
    orientations = np.array([[_random_rotation(rng) for _ in range(77)]
                             for _ in range(frames)])
    positions = np.zeros((frames, 77, 3))
    for f in range(frames):
        positions[f, 0] = rng.normal(size=3) + [0, 0, 5]
        for i, p in enumerate(PARENTS):
            if p >= 0:
                positions[f, i] = positions[f, p] + orientations[f, p] @ (rest[i] - rest[p])
    world_to_camera = np.array([_random_rotation(rng) for _ in range(frames)])
    raw = {
        "positions": positions, "orientations": orientations, "rest": rest,
        "world_to_camera": world_to_camera, "camera_centre": np.zeros((frames, 3)) - [0, 0, 20],
        "focal_px": np.full(frames, 3000.0), "width": 1280, "height": 720,
    }
    groundtruth.write_poses(tmp_path, raw)
    rows = [json.loads(line) for line in (tmp_path / "gt.poses.json").read_text().splitlines()]
    for row in rows:
        det = row["detections"][0]
        q = np.array(det["joint_rotations"])
        r = np.array(det["rest_pose"])
        kp = np.array(det["keypoints_3d"])
        global_ = [None] * 77
        fk = np.zeros((77, 3))
        for i, p in enumerate(PARENTS):
            m = _quat_matrix(q[i])
            if p < 0:
                global_[i] = m
                fk[i] = det["root_pose"]["translation"]
                continue
            global_[i] = global_[p] @ m
            fk[i] = fk[p] + global_[p] @ (r[i] - r[p])
        assert np.abs(fk - kp).max() < 1e-6
        assert det["joint_rotations"][0] == det["root_pose"]["rotation"]
        assert np.allclose(r[0], 0.0)


def test_true_camera_track_uses_camera_py_conventions():
    rng = np.random.default_rng(1)
    absolute = np.array([_random_rotation(rng) for _ in range(5)])
    track = groundtruth.camera_track({"world_to_camera": absolute,
                                      "focal_px": np.full(5, 2000.0)})
    assert np.allclose(track.rotations[0], np.eye(3))
    for i in range(1, 5):
        # camera.estimate accumulates exactly so: current = delta @ current.
        assert np.allclose(track.rotations[i], track.deltas[i] @ track.rotations[i - 1])


def test_scale_plan_is_balanced_grouped_and_real():
    planned = tricks.scale_plan(12, seed=5, group_size=4)
    kinds = [t.kind for t in planned]
    assert {k: kinds.count(k) for k in tricks.KINDS} == {k: 4 for k in tricks.KINDS}
    assert sorted(t.camera_angle for t in planned) == sorted(tricks.CAMERA_ANGLES * 3)
    assert [t.direction for t in planned].count("frontside") == 6
    # A split group is one rider at one venue, and nothing is shared across groups.
    groups = {}
    for t in planned:
        groups.setdefault(t.group, set()).add((t.rider_seed, t.venue_seed, t.body["height"]))
    assert len(groups) == 3 and all(len(v) == 1 for v in groups.values())
    assert len({next(iter(v)) for v in groups.values()}) == 3
    for t in planned:
        assert t.spin_deg in tricks.SPINS[t.kind]
        m = tricks.motion(t)
        labels = tricks.labels(t, m)
        assert labels["inversions"] == {"flat": 0, "cork": 1, "double_cork": 2}[t.kind]
        assert labels["group"] == t.group
        # Ground either side of the air, so the segmenter sees both classes.
        assert labels["takeoff_frame"] > 0 and labels["frames"] - 1 > labels["landing_frame"]
        if t.grab == "none":
            assert labels["grab_start_frame"] == -1
        else:
            assert labels["takeoff_frame"] < labels["grab_start_frame"] < labels["landing_frame"]
    # Extending the set later numbers on; it never renames what exists.
    more = tricks.scale_plan(2, seed=5, start=13, group_size=4)
    assert [t.name[:10] for t in more] == ["synthb0013", "synthb0014"]
    assert more[0].group == "synthbg003"


def test_long_ground_rides_the_terrain_and_leaves_the_air_alone():
    import copy
    import re
    from pathlib import Path

    # tricks.py mirrors animate.py's pelvis drops, which need bpy to import.
    source = (Path(tricks.__file__).parent / "blender" / "animate.py").read_text()
    for name in ("DROP_STANDING", "DROP_TUCKED"):
        assert float(re.search(rf"^{name} = ([0-9.]+)", source, re.M).group(1)) \
            == getattr(tricks, name)

    planned = tricks.scale_plan(6, seed=3, start=101, long_ground=True)
    air = frames = 0
    for t in planned:
        m = tricks.motion(t)
        flat = copy.deepcopy(t)
        flat.terrain = False
        reference = tricks.motion(flat)
        T, L = m.takeoff_frame, m.landing_frame
        # The air is exactly the trick the labels describe.
        assert np.allclose(m.rotation[T:L + 1], reference.rotation[T:L + 1])
        assert np.allclose(m.position[T:L + 1], reference.position[T:L + 1])
        labels = tricks.labels(t, m)
        assert labels["inversions"] == {"flat": 0, "cork": 1, "double_cork": 2}[t.kind]
        air += labels["air_frames"]
        frames += labels["frames"]

        board = m.position - m.rotation[:, :, 2] * tricks._drop(m.tuck)[:, None]
        lip, k = board[T], np.deg2rad(t.kicker_deg)
        deck_z = lip[2] - tricks.DECK_BELOW_LIP
        ramp_start = lip[1] - tricks.DECK_BELOW_LIP / np.tan(k)
        before = board[:T]
        on_deck = before[:, 1] <= ramp_start
        # On the snow before the lip: on the deck, or on the kicker's slope.
        assert np.allclose(before[on_deck, 2], deck_z)
        assert np.allclose(before[~on_deck, 2], deck_z + (before[~on_deck, 1] - ramp_start) * np.tan(k))
        # Down the landing, never below it, and slowing down.
        touch, land = board[L], np.deg2rad(t.landing_deg)
        after = board[L + 1:]
        on_landing = after[:, 1] <= touch[1] + tricks.LANDING_RUN_M * np.cos(land)
        assert np.allclose(after[on_landing, 2], touch[2] - np.tan(land) * (after[on_landing, 1] - touch[1]))
        speed = np.linalg.norm(np.diff(after, axis=0), axis=1)
        assert speed[-1] < speed[0]
        # No ground frame turns the body faster than a rider can.
        fps30 = [rotation_angle(m.rotation[i + 1] @ m.rotation[i].T) * t.slowmo
                 for i in range(len(m.t) - 1) if i < T - 1 or i > L]
        assert np.rad2deg(max(fps30)) < 12.0
    assert air / frames < 0.4


def test_trick_json_round_trips_with_group_seeds():
    t = tricks.scale_plan(1, seed=2)[0]
    assert tricks.Trick.from_json(t.to_json()) == t
    old = ('{"name": "x", "spin_deg": 360, "kind": "flat", "direction": "frontside", '
           '"stance": "regular", "grab": "none", "camera_angle": "side", "slowmo": 1.0, '
           '"seed": 0, "grab_window": [0.3, 0.7]}')
    assert tricks.Trick.from_json(old).rider_seed is None
