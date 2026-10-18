"""Cut detection: a fast pan is one shot, a cut is two.

A whip pan over high-contrast texture changes as many pixels per frame as a
cut does, and the old difference-only test split synthetic single-shot
tricks into 10-17 "shots", across which no rotation is ever integrated.
"""

import cv2
import numpy as np

from snow import camera


def _panorama(rng, colours, width=4000, height=360):
    """Blocky high-contrast texture in the given palette."""
    small = rng.integers(0, len(colours), size=(height // 20, width // 20))
    image = np.array(colours, dtype=np.uint8)[small]
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)


def _write(path, frames, size=(640, 360)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, size)
    for frame in frames:
        writer.write(frame)
    writer.release()


def test_fast_pan_is_one_shot_and_a_cut_is_two(tmp_path):
    rng = np.random.default_rng(0)
    venue = _panorama(rng, [(30, 30, 200), (240, 240, 240), (20, 160, 20), (10, 10, 10)])
    other = _panorama(rng, [(200, 120, 40), (90, 60, 30), (230, 200, 160)])
    step, pan_frames = 24, 40
    frames = [venue[:, i * step:i * step + 640] for i in range(pan_frames)]
    frames += [other[:, i * 4:i * 4 + 640] for i in range(20)]
    video = tmp_path / "pan_then_cut.mp4"
    _write(video, frames)

    # The pan really is as violent as a cut by the old test's measure.
    a = cv2.cvtColor(frames[10], cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(frames[11], cv2.COLOR_BGR2GRAY)
    small = [cv2.resize(x, (320, 180), interpolation=cv2.INTER_AREA) for x in (a, b)]
    assert float(np.mean(cv2.absdiff(*small))) > 22.0

    track = camera.estimate(video, progress=False)
    assert track.segment.max() == 1
    assert int(np.argmax(track.segment == 1)) == pan_frames
    # Every pan frame was solved, not skipped as unreliable.
    assert track.reliable[:pan_frames].all()
