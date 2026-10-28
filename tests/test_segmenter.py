"""The airborne segmenter: its inputs, its training rules, and its fallback.

What is pinned here is the set of properties docs/training.md says a number
silently depends on: inputs are scale-free and the same for training and
serving, a mirror changes only what it should, time can be stretched without
changing what the model is asked, no window crosses a camera cut, a split
never separates a group, and with no weights ``snow features`` still runs the
old heuristic. No GPU, no network; torch-dependent tests skip without torch.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from snow import features as F
from snow.learn import data as D
from snow.learn import events as E
from snow.learn import inputs as I
from snow.learn import labels as L
from snow.meta import ClipMeta

# Camera frame: X right, Y down, Z forward.
STAND = {
    0: (0.0, 0.0, 0.0), 3: (0.0, -0.35, 0.0), 6: (0.0, -0.55, 0.0), 7: (0.0, -0.7, 0.0),
    12: (-0.18, -0.45, 0.0), 13: (-0.3, -0.25, 0.0), 14: (-0.35, -0.05, 0.0),
    40: (0.18, -0.45, 0.0), 41: (0.3, -0.25, 0.1), 42: (0.35, -0.05, 0.25),
    67: (-0.1, 0.0, 0.0), 68: (-0.12, 0.4, 0.05), 69: (-0.12, 0.8, 0.0), 71: (-0.12, 0.85, 0.15),
    72: (0.1, 0.0, 0.0), 73: (0.12, 0.4, 0.05), 74: (0.12, 0.8, 0.0), 76: (0.12, 0.85, 0.15),
}


def _rot_y(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def _clip(n=120, takeoff=40, landing=90, missing=(), scale=1.0):
    """A rider who rides, spins 360 while tucked in [takeoff, landing), rides on."""
    frames = {}
    for f in range(n):
        if f in missing:
            frames[f] = []
            continue
        air = takeoff <= f < landing
        spin = 360.0 * (f - takeoff) / (landing - takeoff) if air else (0.0 if f < takeoff else 360.0)
        rotation = _rot_y(spin)
        k3 = np.zeros((77, 3))
        for j, p in STAND.items():
            p = np.array(p)
            if air and j in (68, 73, 69, 74, 71, 76):
                p = p * np.array([1.0, 0.6, 1.0])  # knees and feet drawn up
            k3[j] = scale * (rotation @ p) + np.array([0.0, -0.02 * f, 12.0])
        frames[f] = [{"tracking_id": 1, "keypoints_3d": k3.tolist(),
                      "keypoints_confidence": [0.9] * 77}]
    return frames


def _write(tmp_path, frames, name="clip"):
    path = tmp_path / f"{name}.json"
    with path.open("w") as fh:
        for f in sorted(frames):
            dets = [dict(d, bbox=[0, 0, 1, 1], keypoints_2d=[[0, 0]] * 77,
                         rest_pose=d["keypoints_3d"], joint_rotations=[[0, 0, 0, 1]] * 77,
                         root_pose={"translation": [0, 0, 0], "rotation": [0, 0, 0, 1]})
                    for d in frames[f]]
            fh.write(json.dumps({"frame_id": f, "detections": dets}) + "\n")
    return path


def _meta(n=120):
    return ClipMeta(name="clip", fps=30.0, frames=n)


# --- inputs -----------------------------------------------------------------


def test_inputs_are_scale_free(tmp_path):
    """The same motion at twice the size (an assumed focal length that is off
    by two) gives the same channels: torso-normalised, never metres."""
    a, b = _clip(), _clip(scale=2.0)
    df_a = F.build(_write(tmp_path, a, "a"), _meta())
    df_b = F.build(_write(tmp_path, b, "b"), _meta())
    x_a, x_b = I.frame_channels(a, df_a), I.frame_channels(b, df_b)
    assert x_a.shape == (120, I.NUM_CHANNELS)
    pose = slice(0, 3 * len(I.POSE_JOINTS))
    np.testing.assert_allclose(x_a[:, pose], x_b[:, pose], atol=1e-5)


def test_mirror_keeps_what_a_mirror_keeps():
    k3 = np.random.default_rng(0).normal(size=(77, 3))
    np.testing.assert_allclose(I.mirror_keypoints(I.mirror_keypoints(k3)), k3)


def test_mirrored_channels(tmp_path):
    frames = _clip()
    df = F.build(_write(tmp_path, frames), _meta())
    x, xm = I.frame_channels(frames, df), I.frame_channels(frames, df, mirror=True)
    for name in ("cos_tilt", "rot_step", "elev_step", "grab", "present"):
        c = I.CHANNELS.index(name)
        np.testing.assert_allclose(x[:, c], xm[:, c], atol=1e-5, err_msg=name)
    azim = I.CHANNELS.index("azim_step")
    np.testing.assert_allclose(x[:, azim], -xm[:, azim], atol=1e-5)
    # A mirror is not the identity on the pose: left and right trade places.
    assert not np.allclose(x[:, :45], xm[:, :45])


def test_hidden_frames_before_the_rider_appears_are_not_invented(tmp_path):
    """The kicker hides the in-run: nothing is extrapolated into those
    frames, while a hole mid-shot is bridged."""
    frames = _clip(missing=set(range(0, 10)) | set(range(60, 66)))
    df = F.build(_write(tmp_path, frames), _meta())
    x = I.frame_channels(frames, df)
    seen = I.observed(x[:, I.PRESENT], df["segment"].to_numpy())
    assert not seen[:10].any() and seen[10:].all()
    assert np.all(x[:10] == 0)
    assert np.all(x[60:66, I.PRESENT] == 0) and np.abs(x[60:66, :45]).sum() > 0


def test_inputs_never_interpolate_across_a_cut():
    frames = _clip()
    frames[50] = []
    df = pd.DataFrame({"frame_id": np.arange(120), "segment": (np.arange(120) >= 51).astype(int),
                       "tilt_deg": 0.0, "rotation_deg": 0.0, "grab_distance": 1.0,
                       "confidence": 0.9})
    x = I.frame_channels(frames, df)
    # Frame 50 ends segment 0 and has no body: it is outside segment 0's
    # observed span, so it stays empty rather than borrowing segment 1's pose.
    assert np.all(x[50] == 0)


def test_resample_stretches_time_and_rates_together():
    x = np.zeros((100, I.NUM_CHANNELS), np.float32)
    x[:, I.RATE_CHANNELS[0]] = 2.0
    x[:, I.PRESENT] = 1.0
    y = np.arange(100) >= 50
    xs, ys = I.resample(x, y, 2.0)
    assert len(xs) == 200
    assert xs[:, I.RATE_CHANNELS[0]] == pytest.approx(2.0 * 99 / 199, rel=1e-4)
    assert ys.mean() == pytest.approx(0.5, abs=0.01)
    assert set(np.unique(xs[:, I.PRESENT])) == {1.0}


# --- events and metrics -------------------------------------------------------


def test_runs_stop_at_a_cut():
    mask = np.ones(10, bool)
    segment = np.array([0] * 5 + [1] * 5)
    assert E.runs(mask, segment) == [(0, 5), (5, 10)]


def test_clean_mask_bridges_gaps_and_drops_slivers():
    mask = np.zeros(100, bool)
    mask[10:30] = mask[35:60] = True   # one air with a 5-frame hole
    mask[80:83] = True                  # a sliver
    out = E.clean_mask(mask, min_frames=8, merge_gap=10)
    assert E.runs(out) == [(10, 60)]


def test_boundary_errors_and_false_airs():
    pred = np.zeros(100, bool)
    pred[22:58] = True
    pred[80:95] = True
    errs = E.boundary_errors(pred, [(20, 60)])
    assert errs[0]["takeoff_err"] == 2 and errs[0]["landing_err"] == -2
    assert errs[0]["false_airs"] == 1
    assert E.boundary_errors(np.zeros(100, bool), [(20, 60)])[0]["missed"]


# --- splits and labels --------------------------------------------------------


def test_folds_never_split_a_group():
    clips = [{"name": f"c{i}", "group": f"g{i // 3}"} for i in range(30)]
    fold = D.group_folds(clips, 4)
    by_group: dict[str, set] = {}
    for c, f in zip(clips, fold):
        by_group.setdefault(c["group"], set()).add(f)
    assert all(len(v) == 1 for v in by_group.values())
    with pytest.raises(AssertionError):
        D.check_no_leak([clips[0]], [clips[1]])


def test_oracle_skeletons_are_refused():
    with pytest.raises(ValueError):
        D.load_clip(D.ClipSource("synth01-oracle", "g", [1], [2]))


def test_gold_labels_are_validated(tmp_path):
    path = tmp_path / "airs.csv"
    path.write_text(
        "clip,air,takeoff_frame,landing_frame,event,sure,labeller,notes,frames,fps,slowmo_factor\n"
        "run1,1,40,90,xg26/bigair/rider/run1,y,cc,,200,30,1\n"
        "run1,2,180,170,xg26/bigair/rider/run1,y,cc,,200,30,1\n"
        "run2,0,,,,y,cc,no air,100,30,1\n"
    )
    problems = L.validate(L.read(path))
    assert any("not after takeoff" in p for p in problems)
    path.write_text(path.read_text().replace("180,170", "120,170"))
    sources = {s.name: s for s in L.gold_sources(path)}
    assert sources["run1"].takeoff == [40, 120] and sources["run1"].group.startswith("xg26")
    assert sources["run2"].takeoff == []


# --- the model ------------------------------------------------------------------


def test_models_are_small():
    torch = pytest.importorskip("torch")  # noqa: F841
    from snow.learn.segmenter import build_model, parameter_count

    for arch in ("tcn", "bigru"):
        for window in (64, 128):
            assert parameter_count(build_model(arch, window=window)) < 1_000_000


def test_no_window_crosses_a_cut():
    torch = pytest.importorskip("torch")
    from snow.learn.segmenter import Segmenter, build_model

    torch.manual_seed(0)
    seg = Segmenter(model=build_model("tcn"), arch="tcn",
                    mean=np.zeros(I.NUM_CHANNELS, np.float32), std=np.ones(I.NUM_CHANNELS, np.float32))
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, I.NUM_CHANNELS)).astype(np.float32)
    x[:, I.PRESENT] = 1.0
    segment = (np.arange(200) >= 120).astype(int)
    before = seg.logits(x, segment)
    x[120:] = rng.normal(size=(80, I.NUM_CHANNELS))
    after = seg.logits(x, segment)
    np.testing.assert_array_equal(before[:120], after[:120])


def _clip_dict(frames, df, name, group, takeoff, landing):
    y = (df["frame_id"].to_numpy() >= takeoff) & (df["frame_id"].to_numpy() < landing)
    return {"name": name, "group": group, "split": None, "slowmo": 1.0,
            "segment": df["segment"].to_numpy(), "x": I.frame_channels(frames, df),
            "x_mirror": I.frame_channels(frames, df, mirror=True), "y": y,
            "hint": D.heuristic_hint(df), "web": np.zeros(len(df), bool)}


def test_training_learns_an_obvious_air_and_features_uses_it(tmp_path, monkeypatch):
    """End to end on CPU: fit, save, and `features.build` picks the weights up.

    The toy air is unmistakable (tucked and spinning), so this checks the
    plumbing, not the model's accuracy.
    """
    pytest.importorskip("torch")
    from snow.learn import train as T

    clips = []
    for k, (t0, t1) in enumerate([(30, 80), (40, 100), (20, 60), (50, 90)]):
        frames = _clip(n=120, takeoff=t0, landing=t1)
        df = F.build(_write(tmp_path, frames, f"c{k}"), _meta())
        clips.append(_clip_dict(frames, df, f"c{k}", f"g{k}", t0, t1))
    monkeypatch.setattr(T, "JITTER_MAX", {})  # plumbing, not robustness
    seg = T.fit(clips[:3], epochs=25, seed=0, device="cpu", log=lambda *_: None)
    prob, mask = T.predict(seg, clips[3])
    counts = E.frame_counts(mask, clips[3]["y"])
    assert E.f1(counts) > 0.8

    weights = tmp_path / "segmenter.pt"
    seg.save(weights)
    poses = _write(tmp_path, _clip(n=120, takeoff=50, landing=90), "served")
    served = F.build(poses, _meta(), segmenter=weights)
    assert served.attrs["airborne_source"] == "segmenter:segmenter.pt"
    assert served["airborne_prob"].notna().all()
    summary = F.summarize(served, _meta())
    assert summary["airborne"]["source"].startswith("segmenter")


def test_without_weights_features_falls_back_to_the_heuristic(tmp_path, monkeypatch):
    monkeypatch.setenv("SNOW_SEGMENTER", str(tmp_path / "absent.pt"))
    poses = _write(tmp_path, _clip())
    for choice in ("off", "auto"):
        df = F.build(poses, _meta(), segmenter=choice)
        assert df.attrs["airborne_source"] == "heuristic"
        assert df["airborne_prob"].isna().all()
        expected = df["grab_distance"] < df["grab_distance"].median()
        assert (df["airborne_hint"] == expected).all()
    with pytest.raises(FileNotFoundError):
        F.build(poses, _meta(), segmenter=tmp_path / "missing.pt")


# --- the dashboard -------------------------------------------------------------


def test_dashboard_uses_the_segmenters_airs_when_present():
    """With the segmenter's frames in the features table, the dashboard's air
    is exactly those frames; the rate detector stays available by name."""
    from test_air_events import _clockwise_air, _recs, es

    recs = _recs(_clockwise_air())       # spin over frames 50..249
    for r in recs:
        r["airborne"] = 60 <= r["f"] < 240
    es.compute_rates(recs, 30.0)
    learned = es.find_events(recs, 30.0, 30.0)
    assert [(e["start_f"], e["end_f"]) for e in learned] == [(60, 239)]
    rate = es.find_events(recs, 30.0, 30.0, detector="rate")
    assert len(rate) == 1 and rate[0]["start_f"] <= 52
    for r in recs:
        del r["airborne"]
    with pytest.raises(ValueError):
        es.find_events(recs, 30.0, 30.0, detector="segmenter")


def test_jitter_keeps_steps_unsigned_and_hidden_frames_empty():
    pytest.importorskip("torch")
    from snow.learn import train as T

    x = np.zeros((50, I.NUM_CHANNELS), np.float32)
    seen = np.arange(50) >= 10
    x[seen, I.PRESENT] = 1.0
    T.jitter(x, np.random.default_rng(0), seen)
    assert np.all(x[:10] == 0)
    assert np.all(x[:, I.CHANNELS.index("rot_step")] >= 0)
    assert x[10:, I.CHANNELS.index("elev_step")].std() > 0
