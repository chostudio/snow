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
masked out, and decomposes it as `H = K R K⁻¹` — rotation about a fixed
centre.

Two limits follow directly from that model, and neither is a bug to fix:

- A camera that **translates** breaks the assumption. Pure rotation about a
  fixed centre is what a homography can express; a dolly or a drone cannot be
  recovered this way.
- **At a hard cut there is no continuous rotation at all.** Each shot gets its
  own world frame, and a trick spanning a cut cannot be integrated across it.

The camera track is cached at `data/poses/<name>.camera.npz`;
`--recompute-camera` forces a refit, `--no-camera` skips compensation
entirely.

## The three measures

No single measure covers every trick. Columns in
`data/poses/<name>.features.csv`:

| column | meaning |
|---|---|
| `axis_spin_deg` | rotation about the trick's **own** dominant axis — the headline number |
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

### Tilt

`tilt_deg` is the angle between the rider's up axis and world vertical. It is
what separates a cork from a flat spin, and it is among the most trustworthy
outputs here — it needs no integration, so it cannot drift, and it does not
depend on metric scale.

## Slow motion

Rates use real time. A slow-motion clip **must** carry its `slowmo_factor`
(see [footage.md](footage.md)) or every deg/s is wrong by that factor. Totals
are unaffected.

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
