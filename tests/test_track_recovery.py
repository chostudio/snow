"""Finding the rider where YOLO did not: stitching fragments, rotated search.

On synthetic big air YOLO found the rider in 28% of inverted airborne
frames, so the track had a hole exactly where a cork is. These pin the
pieces that fill it, and the rules that stop them filling it with someone
else.
"""

import cv2
import numpy as np

from snow import track as trk
from snow.track import Track


def _track(raw_id, boxes):
    t = Track(raw_id)
    t.boxes = dict(boxes)
    return t


def test_rotated_boxes_map_back_to_the_crop():
    h, w = 120, 200
    x0, y0, x1, y1 = 30, 20, 70, 100
    mask = np.zeros((h, w), np.uint8)
    mask[y0:y1, x0:x1] = 255
    for rotation, back in trk._ROTATIONS:
        rotated = mask if rotation is None else cv2.rotate(mask, rotation)
        ys, xs = np.nonzero(rotated)
        u0, v0, u1, v1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        corners = np.array([back(u, v, w, h) for u in (u0, u1) for v in (v0, v1)])
        assert np.allclose(corners.min(0), (x0, y0)), rotation
        assert np.allclose(corners.max(0), (x1, y1)), rotation


def _walk(start, stop, at=(100.0, 100.0), size=(60.0, 90.0), step=(0.0, 0.0)):
    return {f: (at[0] + step[0] * (f - start), at[1] + step[1] * (f - start), *size)
            for f in range(start, stop)}


def test_stitch_joins_fragments_but_not_bystanders_or_across_cuts():
    rider = _track(1, _walk(0, 50))
    tracks = {
        1: rider,
        # The rider again after a 40-frame dropout (slow motion stretches it).
        2: _track(2, _walk(90, 140, at=(110.0, 95.0))),
        # Someone standing beside the rider the whole time.
        3: _track(3, _walk(0, 140, at=(180.0, 100.0))),
        # Same place, but after a cut.
        4: _track(4, _walk(150, 200, at=(105.0, 100.0))),
    }
    absorbed = trk.stitch(tracks, rider, cuts={145})
    assert absorbed == [2]
    assert set(rider.boxes) == set(range(0, 50)) | set(range(90, 140))


def test_someone_else_needs_two_places_at_once():
    rider = _walk(0, 10)
    fragment = _walk(5, 20)  # the rider, unstitched: coincides where both exist
    bystander = _walk(5, 20, at=(300.0, 100.0))
    assert not trk._someone_else(fragment, rider)
    assert trk._someone_else(bystander, rider)


def _recover(monkeypatch, track, detections, frames):
    """Run recover_gaps with a scripted detector: ``detections[f]`` is every
    person box a rotated search would find in frame ``f``."""
    monkeypatch.setattr(trk, "_frames_near", lambda video, present, reach:
                        {f: f for f in frames if f not in present})

    def search(frame_id, anchors, age, reference, *args):
        taken = args[-1]
        best = None
        for box in detections.get(frame_id, []):
            if not any(trk._reachable(a, box, age, reference) for a in anchors):
                continue
            if any(trk._iou(box, other) > 0.3 for other in taken):
                continue
            d = min(np.hypot(box[0] + box[2] / 2 - ax - aw / 2, box[1] + box[3] / 2 - ay - ah / 2)
                    for ax, ay, aw, ah in anchors)
            if best is None or d < best[0]:
                best = (d, box)
        return best and best[1]

    monkeypatch.setattr(trk, "_search", search)
    return trk.recover_gaps(None, track, cuts=set(), model=None)


def test_a_hole_is_filled_when_the_rider_can_be_followed_across(monkeypatch):
    # Rider rises 6 px a frame; YOLO lost them for frames 20-39 (inverted),
    # the rotated search finds them.
    path = _walk(0, 60, at=(400.0, 500.0), step=(0.0, -6.0))
    track = _track(1, {f: b for f, b in path.items() if not 20 <= f < 40})
    found = _recover(monkeypatch, track, {f: [path[f]] for f in range(20, 40)}, range(60))
    assert found == 20
    assert all(np.allclose(track.boxes[f], path[f]) for f in range(20, 40))


def test_a_chain_that_wanders_onto_a_spectator_is_dropped(monkeypatch):
    # The rider flies up out of reach of detection at frame 25; a spectator
    # at the lip beside them, carried down the picture by the pan, is the
    # only person near the chain (run1, frames 644-672).
    path = _walk(0, 60, at=(400.0, 500.0), step=(0.0, -6.0))
    spectator = _walk(25, 40, at=(445.0, 360.0), step=(0.0, 3.0))
    track = _track(1, {f: b for f, b in path.items() if not 20 <= f < 40})
    detections = {f: [path[f]] for f in range(20, 25)}
    for f, box in spectator.items():
        detections.setdefault(f, []).append(box)
    _recover(monkeypatch, track, detections, range(60))
    # Nothing from the forward chain survives: it never reaches frame 40.
    assert not any(f in track.boxes for f in range(20, 40) if f >= 25)
    assert not any(np.allclose(track.boxes.get(f, (0, 0, 0, 0)), spectator[f])
                   for f in spectator)
