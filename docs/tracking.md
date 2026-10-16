# Tracking and framing

Covers `src/snow/track.py` and `src/snow/crop.py` — producing the boxes the
NIM requires, and optionally reframing the clip without corrupting its
intrinsics.

## The bbox annotation

The NIM does not detect people. This file is how you tell it where they are.

```
<number of bodies, 1..50>
<frame_id> <tracking_id> <x> <y> <w> <h>
<frame_id> <tracking_id> <x> <y> <w> <h>
...
```

- Line 1 is the body count, 1–50.
- The grid is **dense**: every body gets a row for every frame.
- A body absent from a frame is written as exactly `-1 -1 -1 -1`.
- Any other non-positive width or height is `INVALID_ARGUMENT`.
- Boxes may only reference frames that actually decode, or `INTERNAL`.
- Tracking ids are renumbered from 1.

`tests/test_bbox_format.py` pins all of this, and the key test parses our
output with **NVIDIA's own reader** rather than a reimplementation. If you
touch the writer, that is the test that matters.

## Tracking

```bash
snow track run1 --preview --show-candidates
```

YOLO + ByteTrack. Detections are cached in `data/clips/<name>.tracks.json`, so
re-selecting tracks is instant; `--retrack` forces a re-detect.

Short gaps in a track are interpolated (`--max-gap`, default 8 frames); longer
ones are left absent, because inventing a position across a long gap invents a
pose.

### Broadcast cuts shatter tracking

Every camera cut makes ByteTrack issue the rider a **new track id**. On a test
clip, automatic selection covered only 35% of frames.

`--show-candidates` lists the raw tracks; watch the preview, then stitch them
by hand:

```bash
snow track run1 --merge "1,8,945,1097,1738,1842" --preview   # 35% -> 73%
snow track trick --merge "1,9"                               # -> 97%
```

This is manual on purpose. Automatic re-identification across a cut would be
guessing, and a wrong stitch silently fuses two different people into one
"rider" whose rotation is nonsense.

For a rider far up the pipe, raise `--imgsz 1920` and lower `--conf 0.12`.

## Cropping

```bash
snow crop run1
```

The gain is **bitrate and source resolution, not magnification**. An encoder
spends its bits on the pipe wall and the crowd; the 50 MB hosted cap forces a
wide 4K clip down to 1080p. Cropping first keeps the rider's own pixels.

`--out-height 0` (the default) keeps native pixel scale. The stage refuses
rather than silently downscaling, and also refuses when a static window would
keep more than 92% of the frame width — at that point it is not buying
anything.

An earlier version of that guard checked the *crop* ratio (`width / crop_w`)
instead of the final *resampling* scale (`out_w / crop_w`), and happily
produced 0.90× output — throwing away pixels in a stage whose entire purpose
is to keep them. Check the quantity you actually care about, not a proxy that
usually moves with it.

### Why the window is static

The NIM assumes a pinhole camera whose principal point is the centre of the
image it is handed. That makes window motion dangerous:

- A window that **zooms** changes the effective focal length every frame.
- A window that **pans** moves the true principal point every frame. Because
  that error *changes over time*, it reads as **rotation the rider never
  performed** — fatal for the one measurement this pipeline exists to produce.

So the window is a fixed size and, by default, static.

`--pan` re-enables panning for shots where the rider will not otherwise fit.
The per-frame offset is recorded in the crop plan as `principal_offsets`, and
`features.correct_principal_point()` subtracts `dx*Z/f` and `dy*Z/f`. **The
depth dependence is the part that fakes rotation** — a constant offset would
merely shift the pose, while one that scales with Z tilts it differently at
every distance. `tests/test_crop_and_usd.py` pins that property.

### Intrinsics

With `focal_length = 0` the NIM assumes `f = hypot(width, height)`, the image
diagonal (measured — see [nim.md](nim.md)). `crop` therefore passes the focal
length explicitly, scaled by the resampling factor and not by the crop:
cropping without rescaling must not change `f`, because it does not change the
optics. That is a test of its own.

### Do not apply a crop correction to full-frame poses

`cmd_pose` records `clip.extra["pose_input"] = "zoom" | "full"`. `features`
only applies the principal-point correction when the poses actually came from
the zoomed clip, and otherwise prints

> crop plan ignored: these poses came from the full frame

A stale crop plan silently "correcting" full-frame poses would be a
hard-to-spot error, so the provenance is recorded rather than inferred.
