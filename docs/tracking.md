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

After the rider's track is chosen, two steps fill its holes (below), then
gaps of up to `--max-gap` frames (default 8) are interpolated. Longer gaps are
left absent, because inventing a position across a long gap invents a pose.

### The detector loses the rider exactly when they are upside down

YOLO is trained on COCO, where people stand up. On the 20 synthetic big-air
clips, with the true rider box known every frame, the chosen track covered:

| rider pose in the air | frames | before | after stitch + recover | wrong box after |
|---|---|---|---|---|
| upright | 3,750 | 76% | **98%** | 0.3% |
| sideways (tilt 60-100) | 721 | 28% | **93%** | 0.7% |
| inverted (tilt > 100) | 512 | 34% | **98%** | 0.2% |

The holes were the pipeline's biggest loss on corks, and they were not the
NIM's doing: given a box, the NIM's body orientation was within about 10
degrees of the truth even upside down, and it called 96-100% of truly
inverted boxed frames inverted. It was simply never shown most of them
([validation.md](validation.md)). Two steps, both on by default:

**Stitch** (`--no-stitch` to skip). ByteTrack issues the rider a new id when
detection resumes after a dropout, so one cork arrives as three or four ids
and selection kept the largest. Another id is absorbed into the rider when
it (a) barely overlaps the rider's frames, so it is not someone on screen at
the same time, (b) starts or ends within 60 frames of the rider's track at a
box of the same size in the same place, and (c) no cut lies between them.
The 60 frames are generous because slow motion stretches a dropout (40-50
frames at 4x); the broadcast camera keeps the rider in the same part of the
picture, so the box test still discriminates. On `trick` this picked the
same id (9) a person had chosen by hand. Ids chosen with `--merge` or
`--track-ids` are never stitched further.

**Recover** (`--no-recover` to skip). Each frame without a box near the
track is searched again in a window around where the rider should be, at 0,
90, 180 and 270 degrees, with the window upscaled to the detector's input
size. The search walks chains out from the rider's detections, each box
found anchoring the next frame. A chain on a halfpipe can step onto a
spectator at the lip: in a fast pan the crowd sweeps past at the rider's
speed (run1, frames 644-672, did exactly this). The rules that stop it:

- a candidate must be sized like the rider's own detections in the same
  shot (0.5-1.6 of their diagonal; a tucked rider is 0.54 of standing), and
  within 0.1 + 0.1 per frame, at most 0.3, diagonals of an anchor: the
  last box, or the last box carried on at the rider's recent velocity;
- not on a raw track that has been somewhere else at the same moment as
  the rider, counting boxes recovered so far;
- a chain gives up 30 frames after its last box, and never crosses a cut;
- in a hole between two detections a chain walks in from each side and is
  kept only if it arrives where the detection on the far side is. One that
  wandered onto a spectator never does, and is dropped whole.

Checked by eye on run1 and on run1 with its manual merge: every sampled
recovered box was the rider, most of them inverted mid-air. Remaining risk:
a bystander standing on the rider's own path, so that a chain resuming
after a miss can step onto them and still arrive where the rider is.
Recovery costs 10-70 s a clip on a laptop, mostly YOLO on the crops; the
cached raw tracks are reused, so `snow track` without `--retrack` repeats
only this part.

### Broadcast cuts shatter tracking

Every camera cut makes ByteTrack issue the rider a **new track id**. On a test
clip, automatic selection covered only 31% of frames (59% now that fragments
within a shot are stitched and gaps recovered; still not across cuts).

`--show-candidates` lists the raw tracks; watch the preview, then stitch them
by hand:

```bash
snow track run1 --merge "1,8,945,1097,1738,1842" --preview   # 31% -> 93%
snow track trick --merge "1,9"                               # -> 97%
```

This is manual on purpose. Automatic re-identification across a cut would be
guessing, and a wrong stitch silently fuses two different people into one
"rider" whose rotation is nonsense. Within a shot the box continuity is
evidence, so that is automatic (above); cuts are found with the colour test
from `camera.py`, and nothing is ever joined across one.

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
