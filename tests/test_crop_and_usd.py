"""Crop coordinate round-tripping and USD export structure."""

from __future__ import annotations

import json

import numpy as np
import pytest

from snow import crop as C
from snow import usd as U
from snow.meta import ClipMeta
from snow.skeleton import PARENTS
from snow.track import Track, write_bbox_file


def test_bbox_file_round_trips_through_the_reader(tmp_path):
    path = tmp_path / "b.txt"
    track = Track(1)
    track.boxes = {0: (10.0, 20.0, 30.0, 40.0), 2: (12.0, 22.0, 31.0, 41.0)}
    write_bbox_file([track], frame_count=3, dest=path)

    bodies, tracks = C.read_bbox_file(path)
    assert bodies == 1
    assert sorted(tracks[1]) == [0, 2]
    assert tracks[1][0] == pytest.approx((10.0, 20.0, 30.0, 40.0))


def _plan(scale: float = 1.0, panned: bool = False) -> C.CropPlan:
    crop_w, crop_h = 800, 450
    out_w, out_h = int(crop_w * scale), int(crop_h * scale)
    centers = np.tile([960.0, 540.0], (4, 1))
    if panned:
        centers[:, 0] = [600.0, 700.0, 800.0, 900.0]
    return C.CropPlan(
        crop_w=crop_w, crop_h=crop_h, out_w=out_w, out_h=out_h,
        centers=centers, source_focal=2203.0, frames=4,
        source_w=1920, source_h=1080, panned=panned,
    )


def test_focal_length_scales_with_the_crop():
    """Cropping without rescaling must not change the focal length."""
    assert _plan(scale=1.0).focal_length == pytest.approx(2203.0)
    # Doubling the output size doubles the pixel scale, so f doubles too.
    assert _plan(scale=2.0).focal_length == pytest.approx(4406.0)


def test_static_crop_has_a_constant_principal_offset():
    plan = _plan(panned=False)
    offsets = [plan.principal_offset(f) for f in range(plan.frames)]
    assert all(o == pytest.approx(offsets[0]) for o in offsets)


def test_centred_static_crop_has_no_principal_offset():
    """A window centred on the frame centre needs no correction at all."""
    plan = _plan(panned=False)  # centres are (960, 540) = frame centre
    assert plan.principal_offset(0) == pytest.approx((0.0, 0.0))


def test_panning_crop_moves_the_principal_point():
    """The motivation for keeping the window static by default."""
    plan = _plan(panned=True)
    offsets = [plan.principal_offset(f)[0] for f in range(plan.frames)]
    assert max(offsets) - min(offsets) == pytest.approx(300.0)


def test_remap_puts_boxes_in_crop_coordinates(tmp_path):
    source = tmp_path / "in.txt"
    track = Track(1)
    # A box at the frame centre, where the crop window also sits.
    track.boxes = {f: (900.0, 500.0, 120.0, 80.0) for f in range(4)}
    write_bbox_file([track], frame_count=4, dest=source)

    plan = _plan()
    dest = tmp_path / "out.txt"
    kept = C.remap_bbox(source, dest, plan)
    assert kept == 4

    _, tracks = C.read_bbox_file(dest)
    x, y, w, h = tracks[1][0]
    # origin is (960-400, 540-225) = (560, 315); scale is 1.
    assert (x, y) == pytest.approx((340.0, 185.0))
    assert (w, h) == pytest.approx((120.0, 80.0))


def test_boxes_outside_the_window_become_absent(tmp_path):
    source = tmp_path / "in.txt"
    track = Track(1)
    track.boxes = {0: (10.0, 10.0, 40.0, 40.0)}  # far outside the crop
    write_bbox_file([track], frame_count=1, dest=source)

    dest = tmp_path / "out.txt"
    assert C.remap_bbox(source, dest, _plan()) == 0
    _, tracks = C.read_bbox_file(dest)
    assert tracks.get(1, {}) == {}


# --- USD -------------------------------------------------------------------

def test_joint_paths_follow_the_skeleton_hierarchy():
    paths = U.joint_paths()
    assert len(paths) == len(PARENTS)
    for index, parent in enumerate(PARENTS):
        if parent >= 0:
            assert paths[index] == f"{paths[parent]}/joint{index}"
        else:
            assert "/" not in paths[index]


def test_camera_to_usd_conversion_is_a_half_turn_about_x():
    """Y is down and Z forward in the NIM's frame; USD wants Y up."""
    assert U.to_usd_point((1.0, 2.0, 3.0)) == (1.0, -2.0, -3.0)
    # (x, y, z, w) -> (w, x, -y, -z)
    assert U.to_usd_quat((0.1, 0.2, 0.3, 0.9)) == (0.9, 0.1, -0.2, -0.3)


def _write_poses(path, frames=3):
    rest = np.zeros((77, 3))
    for i in range(1, 77):
        rest[i] = [0.0, -0.05 * i, 0.0]
    with path.open("w") as fh:
        for f in range(frames):
            fh.write(
                json.dumps(
                    {
                        "frame_id": f,
                        "detections": [
                            {
                                "tracking_id": 1,
                                "bbox": [0, 0, 10, 10],
                                "keypoints_2d": [[0.0, 0.0]] * 77,
                                "keypoints_3d": rest.tolist(),
                                "keypoints_confidence": [1.0] * 77,
                                "rest_pose": rest.tolist(),
                                "joint_rotations": [[0.0, 0.0, 0.0, 1.0]] * 77,
                                "root_pose": {
                                    "translation": [0.0, 1.0, 5.0],
                                    "rotation": [0.0, 0.0, 0.0, 1.0],
                                },
                            }
                        ],
                    }
                )
                + "\n"
            )


def test_usd_export_is_well_formed(tmp_path):
    poses = tmp_path / "p.json"
    _write_poses(poses)
    dest = tmp_path / "out.usda"
    report = U.export(poses, dest, ClipMeta(name="t", fps=30.0))

    assert report["joints"] == 77
    assert report["frames"] == 3
    text = dest.read_text()
    assert text.startswith("#usda 1.0")
    assert 'upAxis = "Y"' in text
    assert "rotations.timeSamples" in text
    # Depth +5 in camera coords must become -5 in USD.
    assert "-5.000000" in text


def test_usd_export_rejects_a_missing_track(tmp_path):
    poses = tmp_path / "p.json"
    _write_poses(poses)
    with pytest.raises(ValueError):
        U.export(poses, tmp_path / "o.usda", ClipMeta(name="t"), tracking_id=99)


def test_usd_parses_and_skins_with_pxr(tmp_path):
    """The real check: USD's own resolver, when pxr is available."""
    pxr = pytest.importorskip("pxr")
    from pxr import Usd, UsdSkel

    poses = tmp_path / "p.json"
    _write_poses(poses)
    dest = tmp_path / "out.usda"
    U.export(poses, dest, ClipMeta(name="t", fps=30.0))

    stage = Usd.Stage.Open(str(dest))
    assert stage
    cache = UsdSkel.Cache()
    root = UsdSkel.Root(stage.GetPrimAtPath("/Rider"))
    cache.Populate(root, Usd.TraverseInstanceProxies())
    query = cache.GetSkelQuery(UsdSkel.Skeleton(stage.GetPrimAtPath("/Rider/Skel")))
    assert query
    assert len(query.ComputeJointSkelTransforms(0)) == 77
