# snow

Extract 3D body pose from snowboarding video using the
[NVIDIA 3D Body Pose NIM](https://build.nvidia.com/nvidia/body-pose), and turn
it into rotation measurements for trick analysis.

No local GPU required — inference runs on NVIDIA's hosted NVCF endpoint.
Everything else runs locally on CPU/MPS.

```
source video
    │
    ├─ scout ──────── find the slow-motion / full-body shots worth using
    │
    └─ ingest ─────── cut + re-encode to what the NIM accepts
         └─ track ─── YOLO+ByteTrack → the NIM's bbox annotation
              ├─ crop ──── optional: zoom onto the rider
              └─ pose ──── the NIM → 77-joint 3D skeleton per frame
                   ├─ features ── camera-compensated rotation → CSV/Parquet
                   └─ usd ─────── UsdSkelAnimation for Omniverse
```

## Setup

```bash
scripts/setup.sh          # venv (python 3.12), deps, vendored NVIDIA client
cp .env.example .env      # then paste your key from build.nvidia.com
source .venv/bin/activate
snow doctor
```

The NVIDIA client supports Python 3.10–3.12 only, so the venv is pinned to
3.12 regardless of the system Python.

## Finding footage worth analysing

Televised runs are mostly useless for pose: wide shots where the rider is 60 px
tall, crowd cutaways, interviews. The good material is the **super-slow-motion
replay** — a high-speed camera, rider large and sharp, motion slow enough that
nothing smears. `snow scout` finds it:

```bash
snow scout "https://www.youtube.com/watch?v=..." --name olympics
```

```
  #        start     len    flow   rel   dup  subj  body  seen  suggested command
 13 00:00:34.935    2.8s    0.09   3.0   28%  0.58   33%  100%  --start ... --duration 2.8   SLOW DUP28% SHORT
 12 00:00:28.795    6.1s    0.18   1.4    2%  0.21  100%  100%  --start ... --duration 6.1
```

It splits the video into shots and scores each on:

| column | meaning |
|---|---|
| `flow` / `rel` | median optical flow, and how much slower than this video's median shot. `rel > 2` means slow motion |
| `dup` | fraction of near-identical frames — slow motion made by *duplicating* frames rather than by a high-speed camera |
| `subj` | median height of the largest person, as a fraction of the frame |
| `body` | fraction of frames where that person shows **hips and ankles** |
| `seen` | fraction of sampled frames containing a person |

`body` exists because `subj` alone is a trap: a talking-head interview also
fills the frame. Scouting therefore runs a *pose* model, not a detector, and a
shot only scores if the subject is a whole body. On a real clip this was the
difference between the top-ranked shot being a face close-up and being the
actual rider.

`SHORT` marks shots under 120 frames — the NIM buffers about that many before
it emits anything, so widen the window around one rather than sending it alone.

### Slow motion corrupts rates, not totals

Broadcast slow motion is shot fast and *conformed* to 30 fps, so the file
claims 30 fps and the timestamps lie about how much real time passed. Total
rotation is unaffected — it is the same turn, just sampled more often — but
every deg/s is wrong by the slow-motion factor. Tell `ingest` about it:

```bash
snow ingest <src> trick --start 00:00:34 --duration 8 --slowmo-factor 4
```

This is stored in the clip's metadata and used by `features`. The factor cannot
be recovered from the file; a high `dup` reading in `scout` implies frame
doubling, but a genuine high-speed camera leaves no trace.

## Running the pipeline

```bash
snow ingest <url|file> run1 --start 00:00:34 --duration 24
snow track  run1 --preview --show-candidates
snow crop   run1                 # optional, see below
snow pose   run1 --overlay
snow features run1
snow usd    run1
```

### Broadcast footage needs its tracks stitched

Every camera cut makes ByteTrack issue the rider a new track id, so
auto-selection alone covered ~35% of a test clip. `--show-candidates` lists the
raw tracks; watch the preview, then stitch:

```bash
snow track run1 --merge "1,8,945,1097,1738,1842" --preview    # 35% -> 73%
```

Detections are cached in `data/clips/<name>.tracks.json`, so re-selecting is
instant. `--retrack` forces a re-detect. For a rider far up the pipe, raise
`--imgsz 1920` and lower `--conf 0.12`.

### Cropping

`snow crop` re-encodes a window around the rider. The gain is **bitrate and
source resolution**, not magnification: an encoder spends its bits on the pipe
wall and the crowd, and the 50 MB cap forces a wide 4K clip down to 1080p.
Cropping first keeps the rider's own pixels. `--out-height 0` (the default)
keeps native pixel scale; the stage refuses rather than silently downscaling.

The window is a **fixed size and, by default, static**, because the NIM assumes
a pinhole camera whose principal point is the centre of the image it is handed:

- a window that **zoomed** would change the effective focal length every frame;
- a window that **pans** moves the true principal point every frame, and
  because that error *changes over time* it reads as rotation the rider never
  performed — fatal for the one measurement this pipeline exists to produce.

`--pan` re-enables panning for shots where the rider will not otherwise fit;
the per-frame offset is recorded and `features` corrects for it. If a static
crop would keep almost the whole frame, the stage tells you to skip it.

Measured from NIM output: with `focal_length = 0` the server assumes
**`f = hypot(width, height)`**, the image diagonal — fitted 2206 px against
2203 predicted on a 1920×1080 clip, over 423 frames. `crop` passes the
correctly scaled focal length explicitly.

## Rotation features

`snow features` estimates the camera's own rotation, removes it, and
integrates what is left.

Why it is needed: `root_pose.rotation` is in **camera** coordinates, and a
halfpipe camera pans hard to follow the rider. On a real trick, compensation
moved the measured spin axis from 67° off vertical to **33°** — much closer to
what a snowboard rotation physically is.

Three measures, because no single one covers every trick:

| column | meaning |
|---|---|
| `axis_spin_deg` | rotation about the trick's own dominant axis — the headline number |
| `spin_deg` | cumulative yaw about the world vertical, with `spin_quality` |
| `rotation_deg` | total angular path, axis-agnostic; counts flips too |
| `tilt_deg` | body axis vs world vertical: 0 upright, 180 inverted |

A nuance worth stating because the intuitive version is wrong: for a clean
rigid rotation, tilting the axis does **not** make vertical yaw undercount —
the horizontal projection still winds once per revolution, and on synthetic
motion both measures return 1080 for a 1080. What breaks `spin_deg` is the hip
axis passing *through* the vertical, where its projection collapses; that is
common once a rider is inverted, and `spin_quality` measures it.
`axis_spin_deg` never projects onto a fixed world axis, so it survives there.

Camera motion is modelled as rotation about a fixed centre (`H = K R K⁻¹`). A
camera that *translates* breaks that assumption, and at a hard cut there is no
continuous rotation at all, so each shot gets its own world frame and a trick
spanning a cut cannot be integrated across it.

Output is `data/poses/<name>.features.{csv,parquet}` plus a JSON summary.

### How far this is validated

- **Synthetic motion: exact.** `tests/test_rotation.py` builds a rider turning
  by a known amount about a known axis; `axis_spin_deg` recovers 360/720/1080
  to within 2°, the axis tilt to within 2°, and camera compensation moves
  rotation between camera and rider in both directions.
- **Real footage: not yet.** On a clip from a video titled *"Frontside Double
  Cork 1080"* the pipeline reads **668°** about an axis 33° off vertical, with
  a 1300° total path and 136° max tilt (correctly inverted). That is well short
  of 1080°. It has not been established whether that shot is the trick the
  title names — it was chosen by scouting heuristics, from a 22-shot edit —
  so this is an open discrepancy, not a measured error rate. Settling it needs
  a clip whose trick is known frame by frame.

Treat absolute spin counts as unvalidated. Relative comparisons, axis tilt and
the inversion signal are on firmer ground.

## Omniverse / USD

```bash
snow usd run1          # -> data/poses/<name>.usda
```

The mapping is close because the NIM already returns what `UsdSkelAnimation`
wants — local joint rotations against a rest pose:

- `rest_pose` → the skeleton's bind and rest transforms
- `joint_rotations` → per-frame `rotations` (joint 0 is the root, and
  `joint_rotations[0]` is exactly `root_pose.rotation`, so nothing is
  transformed twice)
- `root_pose.translation` → the SkelRoot's own xform, i.e. root motion

Coordinates convert from the NIM's camera frame (X right, **Y down**, Z
forward — verified against the data) to USD's Y-up frame by `(x, −y, −z)`, a
half turn about X; the same turn takes a quaternion `(x, y, z, w)` to
`(x, −y, −z, w)`.

Written as text `.usda`, so it needs no `pxr` install and opens directly in
Omniverse, usdview or Blender. The test suite validates it with USD's own
skinning resolver when `usd-core` is present.

Note the skeleton comes out around 1.2 m tall for an adult rider: metric scale
inherits the uncalibrated focal length. Angles and proportions are sound;
absolute metres are not.

## Output format

`data/poses/<name>.json` is JSON Lines, one object per frame (the `.json`
extension is required by the upstream client even though it writes JSONL):

```json
{"frame_id": 100, "detections": [{"tracking_id": 1, "bbox": [x, y, w, h],
 "keypoints_2d": [[x, y], ...], "keypoints_confidence": [c, ...],
 "keypoints_3d": [[x, y, z], ...], "rest_pose": [[x, y, z], ...],
 "joint_rotations": [[x, y, z, w], ...],
 "root_pose": {"translation": [x, y, z], "rotation": [x, y, z, w]}}]}
```

Every per-joint array is length 77, on the Nova-77 skeleton whose parent
hierarchy is in `src/snow/skeleton.py`.

## What the NIM requires

It does not detect or track people, and it is strict about input:

| Requirement | Enforced how |
|---|---|
| H.264, 4:2:0 8-bit, MP4 | `INVALID_ARGUMENT` |
| **Constant frame rate** | rejected — even one dropped frame |
| ≤ 50 MB (hosted endpoint) | rejected |
| boxes reference only decoded frames | `INTERNAL` |
| positive box dimensions, or exactly `-1 -1 -1 -1` | rejected |

`ingest` re-probes its own output and checks each property rather than
trusting the encoder — CFR in particular is verified from real per-frame
timestamps, since a CFR-*labelled* stream with a dropped frame is exactly what
the NIM rejects.

## Self-hosting

The hosted endpoint caps input at 50 MB. To run the container yourself you need
Linux/amd64 and an NVIDIA GPU of compute capability 8.0+ with 24 GB
(AWS `g5.xlarge`/A10G or `g6.xlarge`/L4; **not** `g4dn`/T4, unsupported):

```bash
SNOW_NIM_PREVIEW=0 SNOW_NIM_TARGET=127.0.0.1:8001 snow pose run1
```

## Layout

```
src/snow/scout.py     find usable shots in a broadcast
src/snow/ingest.py    download + re-encode + verify NIM input rules
src/snow/track.py     YOLO/ByteTrack -> the NIM's bbox annotation
src/snow/crop.py      crop/zoom onto the rider, with correct intrinsics
src/snow/pose.py      drives the NIM
src/snow/camera.py    camera rotation from background homography
src/snow/features.py  camera-compensated rotation features
src/snow/usd.py       UsdSkelAnimation export
src/snow/skeleton.py  Nova-77 joint hierarchy
vendor/nim-clients/   upstream NVIDIA gRPC client (cloned by setup.sh)
data/                 clips, tracks, poses — all gitignored
```

## Next

Trick segmentation (airborne detection, splitting a run into individual hits)
and the ML work on top. The open validation question above is the thing most
worth closing first — every downstream label depends on the spin count being
right.
