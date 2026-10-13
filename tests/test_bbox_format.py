"""The bbox annotation is the part the NIM is pickiest about.

These assert against the format rules in the upstream client's
``read_tracked_bboxes``, then actually parse the output with that very
function so the test fails if upstream changes its mind.
"""

from __future__ import annotations

import importlib.util
import sys

import pytest

from snow.config import VENDOR_CLIENT
from snow.track import ABSENT, Track, TrackError, interpolate_gaps, merge, write_bbox_file


def _upstream_reader():
    """Import the vendored client's parser without running its main()."""
    scripts = VENDOR_CLIENT.parent
    for path in (str(scripts), str(scripts.parent / "interfaces"), str(scripts.parents[2])):
        if path not in sys.path:
            sys.path.insert(0, path)
    spec = importlib.util.spec_from_file_location("upstream_body_pose", VENDOR_CLIENT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.read_tracked_bboxes


def _track(raw_id: int, boxes: dict[int, tuple]) -> Track:
    t = Track(raw_id)
    t.boxes = dict(boxes)
    return t


def test_header_is_body_count_and_grid_is_dense(tmp_path):
    dest = tmp_path / "b.txt"
    tracks = [
        _track(1, {0: (10.0, 20.0, 30.0, 40.0), 2: (11.0, 21.0, 30.0, 40.0)}),
        _track(2, {1: (50.0, 60.0, 20.0, 25.0)}),
    ]
    write_bbox_file(tracks, frame_count=3, dest=dest)
    lines = dest.read_text().strip().split("\n")

    assert lines[0] == "2"
    # one row per frame per body
    assert len(lines) == 1 + 3 * 2
    assert all(len(line.split()) == 6 for line in lines[1:])


def test_absent_rows_use_the_exact_sentinel(tmp_path):
    dest = tmp_path / "b.txt"
    write_bbox_file([_track(1, {1: (1.0, 2.0, 3.0, 4.0)})], frame_count=3, dest=dest)
    rows = dest.read_text().strip().split("\n")[1:]

    assert rows[0] == f"0 1 {ABSENT}"
    assert rows[2] == f"2 1 {ABSENT}"
    # A box the NIM would reject must never be emitted as a real box.
    for row in rows:
        _, _, x, y, w, h = row.split()
        if (float(x), float(y), float(w), float(h)) != (-1.0, -1.0, -1.0, -1.0):
            assert float(w) > 0 and float(h) > 0


def test_tracking_ids_are_renumbered_from_one(tmp_path):
    dest = tmp_path / "b.txt"
    write_bbox_file(
        [_track(97, {0: (1.0, 1.0, 5.0, 5.0)}), _track(412, {0: (9.0, 9.0, 5.0, 5.0)})],
        frame_count=1,
        dest=dest,
    )
    ids = {line.split()[1] for line in dest.read_text().strip().split("\n")[1:]}
    assert ids == {"1", "2"}


def test_upstream_client_accepts_what_we_write(tmp_path):
    """The real check: parse our file with NVIDIA's own reader."""
    dest = tmp_path / "b.txt"
    write_bbox_file(
        [_track(1, {0: (10.0, 20.0, 30.0, 40.0), 2: (12.0, 22.0, 30.0, 40.0)})],
        frame_count=3,
        dest=dest,
    )
    frames = _upstream_reader()(dest)

    # Absent rows are dropped by the reader, so only frames 0 and 2 survive.
    assert [f.frame_id for f in frames] == [0, 2]
    assert frames[0].boxes[0].tracking_id == 1
    assert frames[0].boxes[0].width == pytest.approx(30.0)


def test_empty_selection_is_an_error(tmp_path):
    with pytest.raises(TrackError):
        write_bbox_file([], frame_count=5, dest=tmp_path / "b.txt")


def test_short_gaps_are_interpolated_long_ones_are_not():
    track = _track(1, {0: (0.0, 0.0, 10.0, 10.0), 3: (30.0, 0.0, 10.0, 10.0)})
    assert interpolate_gaps(track, max_gap=5) == 2
    assert track.boxes[1][0] == pytest.approx(10.0)
    assert track.boxes[2][0] == pytest.approx(20.0)

    wide = _track(1, {0: (0.0, 0.0, 10.0, 10.0), 50: (30.0, 0.0, 10.0, 10.0)})
    assert interpolate_gaps(wide, max_gap=5) == 0


def test_merge_stitches_ids_and_prefers_the_earlier_one():
    tracks = {
        3: _track(3, {0: (1.0, 1.0, 5.0, 5.0)}),
        9: _track(9, {0: (99.0, 99.0, 5.0, 5.0), 1: (2.0, 2.0, 5.0, 5.0)}),
    }
    merged = merge(tracks, [[3, 9]])
    assert len(merged) == 1
    assert merged[0].boxes[0][0] == pytest.approx(1.0)  # id 3 wins frame 0
    assert merged[0].boxes[1][0] == pytest.approx(2.0)

    with pytest.raises(TrackError):
        merge(tracks, [[3, 404]])
