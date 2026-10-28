"""The trick classifier's input encoding and its invariances (docs/training.md).

The encoding is where the rules that silently ruin a classifier live: a
mirror that does not reverse the turn, an air read across a camera cut, a
slow-motion clip that looks like a different trick. Each is pinned here on a
synthetic rider with a known answer. No GPU, no network, no data/.
"""

import numpy as np
import pytest

from snow.camera import rotation_angle
from snow.learn import airs
from snow.learn import dataset as ds

POSE = np.random.default_rng(3).normal(scale=0.3, size=(77, 3))
POSE[0] = 0.0
POSE[6] = (0.0, -0.55, 0.0)    # neck above the root; Y is down
POSE[67] = (-0.12, 0.0, 0.0)   # hips along X
POSE[72] = (0.12, 0.0, 0.0)


def _yaw(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    # Positive is counter-clockwise seen from above, about world up (0, -1, 0).
    return np.array([[np.cos(a), 0, -np.sin(a)], [0, 1, 0], [np.sin(a), 0, np.cos(a)]])


def _spin(total_deg: float, frames: int, lead: int = 10,
          segment: np.ndarray | None = None) -> airs.Track:
    """A rider turning ``total_deg`` about the vertical between frames
    ``lead`` and ``lead + frames``, standing still either side."""
    count = frames + 2 * lead
    k3 = np.zeros((count, len(airs.JOINTS), 3), dtype=np.float32)
    for f in range(count):
        s = np.clip((f - lead) / frames, 0.0, 1.0)
        rot = _yaw(total_deg * s)
        k3[f] = (rot @ POSE[list(airs.JOINTS)].T).T + (0.0, 0.0, 8.0)
    return airs.Track(k3=k3, confidence=np.ones(count, dtype=np.float32),
                      camera=np.tile(np.eye(3, dtype=np.float32), (count, 1, 1)),
                      segment=np.zeros(count, dtype=np.int32) if segment is None
                      else segment)


def _channel(x: np.ndarray, name: str) -> np.ndarray:
    return x[airs.CHANNELS.index(name)]


def test_encoding_shape_and_spin_total():
    track = _spin(720, 60)
    x = airs.encode(track, 10, 70)
    assert x.shape == (airs.N_CHANNELS, airs.SEQ_LEN)
    assert _channel(x, "cum_path")[-1] == pytest.approx(2.0, abs=0.02)
    assert _channel(x, "cum_vertical")[-1] == pytest.approx(2.0, abs=0.02)
    assert abs(_channel(x, "cum_horizontal")[-1]) < 0.02


def test_mirror_reverses_the_turn_and_keeps_its_size():
    track = _spin(540, 60)
    x = airs.encode(track, 10, 70)
    xm = airs.encode(airs.mirror(track), 10, 70)
    for name in ("cum_vertical", "cum_vertical_cam"):
        assert _channel(xm, name)[-1] == pytest.approx(-_channel(x, name)[-1], abs=1e-3)
    assert _channel(xm, "cum_path")[-1] == pytest.approx(_channel(x, "cum_path")[-1],
                                                          abs=1e-3)


def test_mirror_is_an_involution_that_swaps_sides():
    assert np.array_equal(airs.MIRROR[airs.MIRROR], np.arange(len(airs.JOINTS)))
    pos = {j: k for k, j in enumerate(airs.JOINTS)}
    assert airs.MIRROR[pos[67]] == pos[72]
    assert airs.MIRROR[pos[14]] == pos[42]
    assert airs.MIRROR[pos[0]] == pos[0]


def test_mirrored_labels_flip_sense_and_stance_but_not_frontside():
    labels = {"rotation": 4, "sense": 1, "stance": 0, "axis": 1, "grab": 2}
    flipped = ds.mirrored_labels(labels)
    assert flipped["sense"] == 0 and flipped["stance"] == 1
    assert {k: flipped[k] for k in ("rotation", "axis", "grab")} == \
        {k: labels[k] for k in ("rotation", "axis", "grab")}
    assert ds.direction(labels["sense"], labels["stance"]) == \
        ds.direction(flipped["sense"], flipped["stance"])


def test_direction_matches_the_trick_generator():
    """``tricks.Trick.spin_sign``: frontside regular turns counter-clockwise."""
    ccw, cw = ds.SENSES.index("counter-clockwise"), ds.SENSES.index("clockwise")
    regular, goofy = ds.STANCES.index("regular"), ds.STANCES.index("goofy")
    assert ds.direction(ccw, regular) == "frontside"
    assert ds.direction(cw, goofy) == "frontside"
    assert ds.direction(cw, regular) == "backside"


def test_slow_motion_does_not_change_the_encoding():
    """The same 1080 at 1x and at 4x the frames encodes (nearly) the same."""
    real = airs.encode(_spin(1080, 40), 10, 50)
    slow = airs.encode(_spin(1080, 160), 10, 170)
    for name in ("cum_path", "cum_vertical", "up_y"):
        assert np.abs(_channel(real, name) - _channel(slow, name)).max() < 0.03


def test_an_air_is_never_read_across_a_cut():
    segment = np.zeros(80, dtype=np.int32)
    segment[50:] = 1
    track = _spin(720, 60, segment=segment)
    assert airs.window(track, 10, 70) == (10, 49)
    x = airs.encode(track, 10, 70)
    # Two thirds of the turn happened before the cut, and only that is read.
    assert _channel(x, "cum_path")[-1] == pytest.approx(2.0 * 39 / 60, abs=0.05)


def test_too_few_poses_gives_no_encoding():
    track = _spin(360, 30)
    track.k3[12:40] = np.nan
    assert airs.encode(track, 10, 40) is None


def test_camera_gain_scales_the_camera_rotation():
    count = 20
    rotations = np.stack([_yaw(3.0 * f) for f in range(count)]).astype(np.float32)
    segment = np.zeros(count, dtype=np.int32)
    same = airs.scale_camera(rotations, segment, 1.0)
    assert np.allclose(same, rotations, atol=1e-5)
    doubled = airs.scale_camera(rotations, segment, 2.0)
    assert np.rad2deg(rotation_angle(doubled[-1] @ doubled[0].T)) == pytest.approx(114, abs=0.5)


def test_augment_returns_whether_it_mirrored():
    rng = np.random.default_rng(0)
    track = _spin(720, 60)
    seen = set()
    for _ in range(12):
        x, mirrored = airs.augment(track, 10, 70, rng)
        assert x is not None and x.shape == (airs.N_CHANNELS, airs.SEQ_LEN)
        seen.add(mirrored)
        # Whatever else changed, a mirrored copy turns the other way.
        assert (_channel(x, "cum_vertical")[-1] < 0) == mirrored
    assert seen == {True, False}


def test_inversion_count_is_the_axis_baseline():
    tilt = np.zeros(100)
    assert ds._inversions(tilt) == 0
    tilt[20:30] = 120
    assert ds._inversions(tilt) == 1
    tilt[60:70] = 150
    assert ds._inversions(tilt) == 2
    tilt[31:33] = 100  # a blip close to the first inversion is the same one
    assert ds._inversions(tilt) == 2


def test_split_is_by_group_and_deterministic():
    assert ds.split_of("event-7") == ds.split_of("event-7")
    splits = {ds.split_of(f"g{i}") for i in range(200)}
    assert splits == {"train", "val", "test"}
    assert ds._group_and_split("synth01", None) == ("synth01", "original")
    group, split = ds._group_and_split("x", {"split_group": "g1", "split": "test"})
    assert (group, split) == ("g1", "test")


def test_no_weights_means_fallback(tmp_path):
    from snow.learn import classifier as clf

    assert not clf.available(tmp_path / "missing.pt")


def test_prediction_is_mirror_equivariant():
    torch = pytest.importorskip("torch")
    from snow.learn import classifier as clf

    torch.manual_seed(0)
    model = clf.build_model()
    checkpoint = {
        "state_dict": model.state_dict(), "config": {"width": 64, "dropout": 0.2},
        "mean": np.zeros((airs.N_CHANNELS, 1), dtype=np.float32),
        "std": np.ones((airs.N_CHANNELS, 1), dtype=np.float32),
        "channels": list(airs.CHANNELS),
    }
    c = clf.Classifier(checkpoint)
    track = _spin(900, 60)
    got = c.classify(track, 10, 70)
    flipped = c.classify(airs.mirror(track), 10, 70)
    assert set(ds.HEADS) <= set(got) and got["name"]
    assert got["sense"]["label"] != flipped["sense"]["label"]
    assert got["sense"]["p"] == pytest.approx(flipped["sense"]["p"], abs=1e-5)
    assert got["rotation"] == flipped["rotation"]
    assert got["direction"] == flipped["direction"]
