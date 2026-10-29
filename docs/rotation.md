# Measuring rotation

Covers `src/snow/camera.py` and `src/snow/features.py` — the measurement this
pipeline exists to produce, and the easiest place to be confidently wrong.

Read [validation.md](validation.md) alongside this.

## The problem

Everything the NIM returns is in **camera** coordinates, and a halfpipe camera
pans hard to follow the rider. So `root_pose.rotation` mixes the rider's turn
with the camera's own, and reading it directly reports rotation nobody
performed.

`snow features` estimates the camera's rotation, removes it, and integrates
what is left.

On a real trick this moved the measured spin axis from **67° off vertical to
33°** — much closer to what a snowboard rotation physically is.

## Coordinate frame

The NIM's camera frame is **X right, Y down, Z forward** (OpenCV convention).
Verified from the data, not assumed: the head sits at negative Y relative to
the pelvis, feet at positive Y.

So **world up is `(0, -1, 0)`**. This is `WORLD_UP` in `features.py` and it is
easy to get backwards; a sign error here inverts every tilt measurement while
still producing plausible-looking numbers.

Joint indices used: `0` root, `6` neck, `67`/`72` hips, `14`/`42` wrists,
`71`/`76` feet.

## Camera estimation

`camera.py` fits a homography to **background** features, with the rider
masked out, and decomposes it as `H = K R K⁻¹`: rotation about a fixed
centre, with `K` built from an assumed focal (see below).

Two limits follow directly from that model, and neither is a bug to fix:

- A camera that **translates** breaks the assumption. Pure rotation about a
  fixed centre is what a homography can express; a dolly or a drone cannot be
  recovered this way.
- **At a hard cut there is no continuous rotation at all.** Each shot gets its
  own world frame, and a trick spanning a cut cannot be integrated across it.

### What counts as a cut

A frame is a cut only when three things hold together: the frame changed a
lot (mean difference over 22 on a 320x180 copy), **no homography explains
it** (fewer than a quarter of the tracked background points agree with the
best one; a pan keeps hundreds that do), and **what is in view changed**
(HSV histogram distance over 0.35; a pan over sparse texture can fail the
second test, never this one).

The difference test alone split synthetic single-shot tricks into 10-17
"shots" during fast telephoto pans, and run1 into 17. Real cuts in run1 and
vito score 0.51-0.95 on the histogram; the false ones scored at most 0.15.
The trade: a cut between two shots of near-identical colour (two cameras on
the same white slope) needs the histogram to move, and could be missed.
`tests/test_camera_cuts.py` pins the pan-versus-cut case.

### The focal length is assumed, and on a long lens that matters

`K` uses `f = hypot(w, h)`, the NIM's own assumption. A broadcast camera on
a long lens is often 2-11x that (measured on the synthetic set), and
`H = K R K⁻¹` read with too short a focal over-reads every pan and tilt by
about that ratio. So **`camera_rotation_deg` is uncalibrated in magnitude**,
like metric scale.

Two things were tried and are recorded so they are not re-tried blind:

- **Self-calibrating `f` from the background does not work reliably here.**
  A pan-tilt head never rolls, so the right `f` is the one that keeps the
  image's horizontal axis level; but the roll that makes `f` observable
  scales with the camera's tilt, and a broadcast camera is usually close to
  level with the rider. On 20 synthetic clips it landed within 1.5x of the
  true focal on 7.
- **Decomposing with the true `f` is worse, not better**, through a full
  homography: its perspective terms are tiny and noisy on a long lens, and
  `K⁻¹ H K` multiplies them by `f`. A similarity model (shift, roll, zoom)
  fixes that, should a known focal ever be supplied.

And it barely matters for the spin itself on big air: with the **true**
camera, spin over the air named 7 of 20 synthetic tricks through the NIM,
against 6 with the estimated camera and 7 with no compensation at all. The
camera turns 26-90 degrees over a whole clip, slowly beside a spin. It would
matter more on halfpipe, where the pans are larger.

The camera track is cached at `data/poses/<name>.camera.npz`;
`--recompute-camera` forces a refit, `--no-camera` skips compensation
entirely.

## The four measures

No single measure covers every trick. Columns in
`data/poses/<name>.features.csv`:

| column | meaning |
|---|---|
| `spin_path_deg` | angular path frame to frame, signed by the turn's sense about world up; the curve on screen, no longer the count (see below) |
| `body_qx`..`body_qw` | camera-compensated body orientation as a quaternion; `spin_count_deg` over an air is **the headline number** |
| `axis_spin_deg` | rotation about the trick's **own** dominant axis; undercounts corks |
| `spin_deg` | cumulative yaw about the world vertical, with `spin_quality` |
| `rotation_deg` | total angular path, axis-agnostic; counts flips too |
| `tilt_deg` | body axis vs world vertical: 0 upright, 180 inverted |
| `segment` | camera shot id — **rotation never accumulates across a change** |
| `camera_rotation_deg` | per-frame camera rotation that was subtracted |
| `spin_rate_dps`, `rotation_rate_dps` | rates, in **real** time (see slow motion below) |
| `airborne_hint`, `grab_distance`, `torso_scale_m`, `confidence` | supporting signals |

Plus `frame_id`, `time_s`, `yaw_rad` and `root_x/y/z` (root position in camera
metres — uncalibrated, see [validation.md](validation.md)).

### Why `axis_spin_deg` exists — the obvious reason is wrong

The intuitive story is "a corked spin tilts the axis, so vertical yaw
undercounts it." **That is false**, and the code and tests say so explicitly.

For a clean rigid rotation about *any* fixed axis, the horizontal projection
still winds once per revolution. On synthetic motion `spin_deg` and
`axis_spin_deg` both return 1080 for a 1080, whatever the axis tilt.

What actually breaks `spin_deg` is the hip axis passing **through** the
vertical, where its horizontal projection collapses and its direction is
undefined. That happens routinely once a rider is inverted. `spin_quality`
measures exactly that conditioning, and frames below `--min-quality` (default
0.35) are interpolated across rather than trusted.

`axis_spin_deg` never projects onto a fixed world axis, so it stays well
conditioned there. On clean synthetic motion the two agree to within a degree;
on real inverted footage they diverge, and **that divergence is a noise
signal, not a geometric correction**.

This matters for anyone extending the maths: do not "fix" `spin_deg` by
tilting its reference axis. The problem is conditioning, not geometry.

### Why `spin_path_deg` replaced it as the headline

The argument above holds for a *fixed* axis. A cork's axis is not fixed: it
leans away from vertical through the inversions and back for the landing,
and the rider also pitches from the kicker's angle to the landing's.
Projecting every step onto one dominant axis drops whatever was not about
that axis, so `axis_spin_deg` undercounts corks **even on an exact
skeleton**. On 20 synthetic tricks it named 1 of 13 corks correctly
([validation.md](validation.md)).

`spin_path_deg` is `rotation_deg`, the total angular path, with a sign. On
the same exact skeletons it named 20 of 20, measured over the air. The sign
is the sense of the turn about world up (positive is counter-clockwise from
above), taken from the rotation summed over about 15 frames either side
rather than from each step: through an inversion the NIM's per-frame steps
can flip sense for a stretch, which would subtract real rotation.

What it gives up: like `rotation_deg` it accumulates pose jitter, which
reads as a few degrees of rotation per frame even when the rider is still.
Measure it over an air, as the dashboard does, not over a whole clip. It
also cannot tell a spin from a flip; it counts both.

`axis_spin_deg` is kept, now signed the same way (its axis is pointed up,
so it no longer reads every spin as counter-clockwise), and its axis still
gives `spin_axis_tilt_deg`.

### Counting a spin through pose jitter

`spin_path_deg` is still a column and still means what it says, but the
dashboard no longer counts an air with it. The frame-to-frame path adds
every frame's pose jitter, a few degrees each, whether or not the rider
turned. Slow motion is where that bites: a clip slowed k times has k times
the frames for the same turn, so k times the jitter against an unchanged
total. On the 285-clip synthetic set, through the NIM, the path over the
**true** air read a median 96 degrees high and named 112 of 285 tricks;
the dashboard's own, shorter air window happened to cancel some of that
(116 named), so neither the measure nor the window was right
([validation.md](validation.md#fixed-after-this-table-slow-motion-and-pose-jitter)).

`features.spin_count_deg` counts the same path between orientations about
**90 degrees of turn apart** rather than one frame apart:

- A chord well under 180 degrees cannot alias, and jitter adds to a chord
  roughly once instead of once per frame.
- The chord's length in frames is found from the count itself (a fixed
  point from one frame), so it is about one frame on a real-time 1080 and
  several at 4x. It is an angle, not a time, which is why **no slow-motion
  factor is needed**. Knowing the factor did not help: chords of a fixed
  number of real-time frames named 127 to 139 of the development split's
  189 where the chord set by the turn named 142 to 143.
- The result is averaged over every starting phase of the chords, so it
  does not depend on which frames they land on.
- It cuts corners where the axis moves within a chord, so on exact poses it
  reads double corks a median 21 degrees low (95 of 96 still named).

The air detector reads the matching rate, `features.robust_speed`: the
chord angle over a span set so the clip's fastest chords are about 60
degrees. On slow motion the five-frame rate the detector used was mostly
jitter, which set the peak its gates are fractions of and misplaced airs
(takeoff off by 67 frames on average at 4x). The span is also the clip's own unit of
time, so a pose hole is followed for up to four spans.

Both need the body orientation, which `snow features` now writes as
`body_qx`..`body_qw`: the camera-compensated body frame as a quaternion.

What was tried and not kept, on the development split only:

| idea | result |
|---|---|
| Gaussian smoothing of the orientation, fixed width in frames | helps slow motion, hurts real time past 3 frames (a 1440 at 1x turns 40 degrees a frame) |
| smoothing scaled by the known slow-motion factor | about as good as the chord count with the true air (140 against 149 of 189), and needs the factor |
| local quadratic fit of the rotation, integrated speed | 155 of 189 over the true air against 149, but inside the detected air 143 to 147 against 143, mixed by kind, and costlier |
| parity: heading at landing against takeoff, so the count must be an odd or even multiple of 180 | with the true parity 163 of 189 against 143; measured through the pipeline the parity was right on 84%, worth 3 clips when gated on confidence. Not shipped |
| correcting the NIM's front/back flips by a depth reflection (with left and right swapped) chosen per frame by smoothness | the reflection does recover the true pose in a flip (error 155 to 15 degrees), but a smoothness criterion did not pick it; flips are 1.4% of airborne frames |
| merging airs split by a brief dip in the rate | fixed one split, over-read three flat spins |
| the wind-up trim anywhere in the air | cut real air on slow motion: 130 against 141 without it. It now cuts only outside the air's fast core (140) |

### Tilt

`tilt_deg` is the angle between the rider's up axis and world vertical. It is
what separates a cork from a flat spin, and it is among the most trustworthy
outputs here — it needs no integration, so it cannot drift, and it does not
depend on metric scale.

## Slow motion

Rates use real time. A slow-motion clip **must** carry its `slowmo_factor`
(see [footage.md](footage.md)) or every deg/s is wrong by that factor. Totals
are unaffected in principle, but only if the measure does not add something
per frame: the frame-to-frame path did (pose jitter), which is why the count
is now taken over chords (above).

## Principal-point correction

If the poses came from a `--pan` crop, the moving window has shifted the
principal point every frame. `correct_principal_point()` subtracts `dx*Z/f`
and `dy*Z/f`.

The **depth dependence is the whole point**: a constant offset would merely
shift the pose, but one that scales with Z tilts it differently at each
distance, which is what reads as rotation. The correction is applied only when
`clip.extra["pose_input"] == "zoom"`. See [tracking.md](tracking.md).

## What is proven

- **Synthetic: exact.** `tests/test_rotation.py` builds a rider turning a
  known amount about a known axis. `axis_spin_deg` recovers 360/720/1080 to
  within 2°, axis tilt to within 2°, and camera compensation moves rotation
  between camera and rider in both directions.
- **Real footage: open.** See [validation.md](validation.md) before quoting
  any absolute spin count.
