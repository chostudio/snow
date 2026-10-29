# Synthetic ground truth

Covers `src/snow/synth/`. The design and the reasons for it are in
[training.md](training.md#synthetic-ground-truth-in-omniverse); this is what
was built and how to run it. Results are in [validation.md](validation.md).

```bash
snow synth plan        # 20 tricks -> data/synth/<clip>/{trick.json,motion.npz,labels.json}
snow synth render      # headless Blender: rig, animate, venue, camera, GT, USD, frames, mp4
snow synth prepare     # GT in the NIM's schema + the "oracle" run through snow features
snow synth pipeline    # the unchanged pipeline on each render (hosted NIM: calls spaced 120 s)
snow synth compare     # error tables, per kind and per camera angle
snow synth fidelity    # how the NIM reads each render: detection, confidence, 2D and 3D joint error

snow synth plan --scale 400 --seed 1           # the training set: synthb0001.. (see below)
snow synth render --engine cycles --only synthb0001   # GPU render on a headless Linux box
```

Every step reads and writes files under `data/synth/<clip>/`, so any step can
be re-run alone, for all clips or `--only synth14`.

## What is needed

- **Blender 4.3** at `/Applications/Blender.app` (or `SNOW_BLENDER`), with the
  **MPFB 2.0** extension installed. Headless runs enable it for their own
  session; the user's saved preferences are not touched.
- **ffmpeg**, and the NIM key in `.env`, as for real clips.
- For batches on a headless Linux GPU box (how the training set was made):
  the Blender 4.3 Linux tarball with MPFB copied into
  `~/.config/blender/4.3/extensions/`, `--engine cycles` (EEVEE has no GPU
  context there and falls back to software GL), an ffmpeg new enough for
  `-fps_mode` (5.1+; Ubuntu 22.04 ships 4.4), and `LD_LIBRARY_PATH` unset
  on AWS Deep Learning AMIs, whose system cuDNN breaks torch's.
- For the Omniverse render: a Linux machine with an **RTX** GPU and Docker
  with the NVIDIA container toolkit, for the Isaac Sim container. Run on
  an L4; not an H100/A100 (no RT cores), and an A10G crashed (see below).

## The rider is rigged as the NIM's skeleton

The armature has exactly the Nova-77 joints: 77 bones named `joint0` ..
`joint76`, with `skeleton.PARENTS` as the hierarchy. Those are the names
`usd.joint_paths` builds tokens from, so the exported rider's `UsdSkel`
joint list is **identical** to `snow usd`'s export of the NIM's reading of
the same render (checked on the exported `scene.usdc`).

Shared one to one:

- joint indices, hierarchy and USD joint tokens
- joint *positions*, and everything `features.py` and the dashboard derive
  from them
- the forward-kinematics convention: `keypoints_3d[i] = keypoints_3d[p] +
  G_p (rest_i - rest_p)` with `G_i = G_p q_i`. The NIM's own output satisfies
  this to 1e-4 m on a real clip; `gt.poses.json` satisfies it exactly
  (`tests/test_synth.py`)

Not shared: the **rest pose**. The NIM's `rest_pose` is not an anatomical
stance (one arm and one leg point along +X, the other pair along -X) and it
shifts a few centimetres between clips because it carries a body-shape
estimate. The rider rests in its own MPFB A-pose. So local `joint_rotations`
only compare between files with the same rest; positions compare directly.

Joint names are inferred, because NVIDIA publishes only the parent list.
Left and right were read off real NIM output (`keypoints_3d[42]` is on the
`67 -> 72` side in every frame), roles from bone lengths. Table in
`synth/nova.py`. Each Nova joint sits on an MPFB landmark (pelvis at
`spine05`, shoulder at `upperarm01`, ball of the foot at `toe3-1`, ...), and
MPFB's skin weights are summed onto the Nova bone whose segment each MPFB
bone lies in. MPFB's own rig is then deleted.

Feet are held to the board by IK, hands follow IK targets, and the spine
bends and twists for tuck, grab and wind-up. The ground truth is read back
from the *evaluated* skeleton every frame, not from the parameters.

## Tricks from parameters

`synth/tricks.py`. The root's angular velocity is a spin rate along an axis
leaning `alpha(t)` off vertical. About a fixed axis leaning `alpha`, the body
is furthest from upright (`2 alpha`) each time the spin passes 180 + 360k,
so a cork leans the axis past 45 degrees around one of those moments and a
double cork around two consecutive ones.

How much lean is **solved**, not chosen: every lean on a 1-degree grid is
integrated and its inversions counted (head below the board, tilt over 90,
peak at least 110 so it is a clear inversion), and a seeded draw is taken
from the middle of the working range. Some lean directions cannot give a
clean single cork (the landing pitch adds to leftover lean and grazes a
second inversion), so other directions are tried. A "double cork" is
therefore a property of the generated motion.

A correction rotation is blended in over the air so the board lands
straight, and the rider's pitch goes from the kicker's angle to the
landing's. Every label is then **measured from the final motion**.

Randomised per trick: stance, grab and its timing, pre-wind, tuck depth, air
time, lean direction, MPFB body shape (gender, age, muscle, weight, height,
proportions), outfit colours, sun, venue dressing, camera position, framing,
lag and wobble, and slow-motion factor.

### A definition finding before any rendering

`axis_spin_deg` and `axis_tilt_deg` in `labels.json` use `features.py`'s
own definitions on the exact motion, and **they read below the named spin**:
a double cork 1080 reads 790-1045, a flat 360 reads 336. Projecting every
frame's rotation onto one dominant axis drops whatever was not about that
axis: a cork's axis moves, and the rider pitches about 60 degrees from the
kicker to the landing. That is error in the *definition*, which no better
pose model can remove. `test_synth.py` pins it so nobody blames the NIM.

## Scene and camera

`synth/blender/venue.py`. A big-air venue built around each trick's board
path, so the board meets the snow at takeoff and landing whatever the
trick's speed: straight kicker, deck, knuckle, landing at the trick's
landing angle, outrun. Dressed with banners, towers, trees, a crowd and
mountains, because `camera.py` measures rotation from background features
and an empty white slope would starve it as no real venue does.

The camera is a **fixed** broadcast position (side, front quarter, back
quarter, low at the landing) that pans, tilts and zooms after the rider:
the operator leads the rider by a reaction time, then lags, plus a fluid
head's wobble. Fixed position is the model `camera.py` assumes. Framing runs
from wide to tight; across the 20 clips the true focal was 3,200 to 16,000
px, 2.2 to 10.9 times the `hypot(w, h)` = 1,469 px the pipeline assumes at
1280x720.

Slow motion is rendered for real: a clip at factor `k` renders `30 k`
frames per real second and plays them at 30 fps, with a half-frame shutter
(1/(60 k) s real exposure). The pipeline is **not** told `k`, as with real
broadcast footage.

Dressing that stands between the camera and the rider is removed. If the
hill itself hides the rider (the kicker hides the in-run from a down-hill
camera), that is real and is recorded per frame as `visible`.

## Files

| Path | Holds |
|---|---|
| `trick.json` | the parameters, including the solved lean |
| `motion.npz` | root position/rotation, tuck, grab, wind per rendered frame |
| `labels.json` | exact answers (see `tricks.labels`), plus camera facts from `prepare` |
| `gt_raw.npz` | evaluated joints, rest pose, true camera and focal, visibility |
| `gt.poses.json` | ground truth in the NIM's JSONL schema, camera frame |
| `gt.camera.npz` | the true camera as a `CameraTrack` |
| `scene.blend`, `scene.usdc` | the whole scene; USD is the Replicator input |
| `frames/`, `render.mp4` | the render, and what `snow run` ingests |
| `renderer.txt` | `blender` (EEVEE) or `blender-cycles` |
| `compare.json` | training-set clips only: both variants' scores, kept when the heavy files are pruned |

## Two variants per clip

- **oracle**: `gt.poses.json` and the true camera through `snow features`
  and the dashboard's air detector. Measures the definitions.
- **pipeline**: `render.mp4` through `snow run --features`: tracker, NIM,
  `camera.py`, `features.py`, air detector. Measures what a user sees.

The gap between them is what perception (NIM plus camera estimate) adds.

## Omniverse Replicator

Blender authors everything, and `scene.usdc` carries it: the Nova-77
skeleton and skin, its animation, the venue and the animated camera (focal
in USD's tenths of a scene unit). `synth/replicator_render.py` renders that
stage with Omniverse Replicator inside the Isaac Sim container. The labels
do not change, because it is the same stage: a clip re-exported from the
same `trick.json` reproduces the Blender clip's `gt_raw.npz` to 1e-5 m.

**It ran** (October 2026) on Isaac Sim 6.2.0 on an AWS `g6.xlarge` (L4):
30 clips, `synthi0001` to `synthi0030`, each the same scene as a Cycles
clip of the training set, so the two renderers are compared pair by pair.

```bash
# on the Mac (Blender + MPFB): rig, venue, camera, ground truth, scene.usdc; no render
blender -b --python src/snow/synth/blender/run.py -- data/synth/synthi0001_... --no-render
snow synth prepare --only synthi                 # ground truth + oracle, as for Blender clips

# on the GPU box: Isaac Sim's own Python in the NGC container, headless
docker login nvcr.io -u '$oauthtoken'            # key from SSM /app/snow/NVIDIA_API_KEY
docker run --rm --gpus all --network host -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  -v ~/snow:/snow -u 1234:1234 --entrypoint /isaac-sim/python.sh \
  nvcr.io/nvidia/isaac-sim:6.2.0 /snow/src/snow/synth/replicator_render.py \
  /snow/data/synth/synthi0001_... [more clips] --blur 4 --subframes 2
snow synth encode --renderer replicator --only synthi0001   # renderer.txt: isaac-replicator

# then the unchanged pipeline, against a self-hosted NIM
SNOW_NIM_PREVIEW=0 SNOW_NIM_TARGET=127.0.0.1:18021 snow synth pipeline --only synthi
```

Mount Isaac's cache directories too (`/isaac-sim/.cache`,
`/isaac-sim/.nv/ComputeCache`, ...), owned by uid 1234, or it starts cold.
One Isaac process renders a list of clips: RTX compiles its shaders for
about 7 minutes on the box's 4 vCPUs at the first frame, then never again.

What the render does, all on the session layer so `scene.usdc` is never
rewritten:

- **Daylight for a broadcast.** Blender's lights arrive in Blender's units
  (a sun of about 1.2, a grey world of 0.35), which RTX reads as near
  darkness. The sun is set to 3000, the dome to a 800 sky blue; nothing
  that moves changes.
- **Motion blur accumulated, not post-processed**: each frame averages 4
  full RTX renders across the same half-frame shutter the Blender render
  uses (frame `f` exposes `f - 0.25` to `f + 0.25` of scene time, which is
  the slow-motion rate), in linear light. The rider, the board and the pan
  smear as they moved; with 4 samples a fast limb shows as steps, not a
  streak.
- RTX real-time (`RaytracedLighting`), 2 subframes per sample, fixed
  exposure, ACES tone map.

Two things in Blender's USD export had to be fixed in `blender/run.py`
before the stage rendered like the Blender scene (neither changes the
Blender render):

- **The rider wore a skirt.** MPFB's base mesh carries helper geometry
  (skirt, tights, joint markers) that a Mask modifier hides; Blender exports
  a skinned mesh without its modifiers, so in USD the helpers come back.
  The masked vertices (5,778) are deleted before the export.
- **The venue was white.** The venue's checker, brick, wave and noise
  textures are procedural nodes, which the exporter drops, leaving a plain
  0.8 grey. They are flattened to the mean of their two colours for the
  export: the palette survives, the pattern does not. Banners, scaffold and
  snow are therefore plainer than in the Cycles render, which leaves
  `camera.py` fewer features to track (see the comparison).

Checked: Replicator's `camera_params` focal agrees with `gt_raw.npz` on every
frame, and frames side by side show the same framing and pose as the Cycles
render of the same scene. Its `cameraViewTransform` is **not** a check: in
Isaac Sim 6.2 it stays at the stage's last frame while the image follows
the animated camera, so it is not written.

**Hardware.** An A10G (`g5.xlarge`) could not run it: every start, even an
empty stage, died with `VkResult: ERROR_DEVICE_LOST` and Xid 109 (CTX
SWITCH TIMEOUT), on Isaac Sim 6.2 and 5.1, drivers 595 and 580, open and
proprietary kernel modules, GSP on and off, clocks unlocked. The L4 ran
first time on the AMI's own driver 595.

**Cost.** 6,795 frames in 1.41 hours of rendering: 0.69 s a frame
(median; 4 shutter samples plus 1 for the focal check, 2 subframes each),
plus about 7 minutes of shader compile per Isaac process. A median clip
(227 frames) renders in 166 s, about **$0.04** on a `g6.xlarge` at $0.80 an
hour, close to Cycles' 0.6 s a frame on the same GPU. The pipeline ran on
the Mac against the box's NIM through an SSH tunnel (the box's ffmpeg is
4.4; ingest needs 5.1+), 24 to 145 s a clip.

**Where it lives.** `data/synth/synthi00NN_<kind>_<spin>/` in the usual
layout (`trick.json` is the Cycles clip's, renamed, with `source_clip`; the
same split `group`), plus `replicator/render.json` (timings) and
`replicator/camera.npz` (the rendered focal). `data/synth/MANIFEST.isaac.jsonl`
has one line per clip in `MANIFEST.scale.jsonl`'s format with `renderer:
isaac-replicator`, `source_clip`, render and pipeline seconds. Everything
(without the PNG frames) is under `s3://snow-artifacts-009160029345/synth-isaac/`,
with the comparison in `isaac_vs_blender.{txt,json}` and side-by-side
frames in `isaac_samples/`.

### Isaac Replicator against Blender Cycles, same 30 scenes

Each Isaac clip is a Cycles clip of the training set re-rendered: same
rider, venue, camera path, motion and labels. 10 flat, 10 cork, 10 double
cork; 9 side, 9 front quarter, 6 back quarter, 6 low landing; slow motion
1-3x. Both sets were re-measured with today's `features.py` (including
`spin_count_deg`) from their stored poses and camera tracks, so only the
renderer, and where the tracker ran (below), differ. Mean absolute errors
in degrees unless stated:

| variant | n | spin count over air vs named | spin count names it | dashboard spin right | dashboard direction right | spin path names it | axis tilt | max tilt | camera rotation | takeoff (frames) | landing (frames) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| oracle (exact poses) | 30 | 23 | 30/30 | 30/30 | 30/30 | 28/30 | 8 | 15 | 0 | 4 | 4 |
| Blender Cycles | 30 | 93 | **23/30** | 22/30 | 27/30 | 7/30 | 16 | 26 | 83 | 8 | 15 |
| Isaac Replicator | 30 | 203 | **15/30** | 15/30 | 25/30 | 8/30 | 18 | 33 | 120 | 31 | 15 |

By kind, spin count names it: flat 7/10 Cycles, 8/10 Isaac; cork 9/10,
6/10; double cork 7/10, **1/10**. The loss is almost all in the corks.

How the NIM reads the pixels (`snow synth fidelity`; detection over the
frames where the rider is in view, the rest over detected frames):

| renderer | detected | keypoint confidence | 2D error / box | PA-MPJPE (mm) |
|---|---|---|---|---|
| Blender Cycles | 0.979 | 0.739 | 0.041 | 70 |
| Isaac Replicator | 0.865 | 0.696 | 0.070 | 90 |
| real broadcast (`run1`, `trick`) | – | 0.62, 0.54 | – | – |

Paired, Isaac's confidence is lower on 23 of 30 scenes, its 2D error higher on
29 and its 3D joint error higher on 27.

**What it says.** The NIM finds the Replicator renders harder, and in the
direction of real footage: its confidence moves from 0.74 towards the 0.54
to 0.62 it gives real broadcast clips, and its 2D and 3D errors rise by
about 70% and 30%. So the Blender numbers were flattering, as
[training.md](training.md#what-it-does-not-replace) warned they might be,
and corks suffer most.

Much of the loss is the **tracker**, not the NIM: YOLO keeps the rider in
86.5% of in-view frames of the Isaac renders against 97.9% of the Cycles
renders, and on 9 of 30 Isaac clips it holds the rider for only 47-64% of
them (it picks the rider up late, from the air, or drops the inverted
body). That is not the machine: the Cycles renders re-tracked on the Mac,
where the Isaac clips were tracked, keep 98.1%. The NIM's own reading is
worse too, on the frames it is given (the 2D and 3D errors above count
detected frames only).

**Why it is harder, as far as is known.** Not one cause, and not yet
separated:

- Light. A hard sun gives the body real shading and cast shadows, where
  the Cycles render is flat and washed out; the Isaac venue is plainer
  (patterns flattened) but more saturated against a blue sky.
- Blur. Four discrete shutter samples leave fast limbs as steps, not a
  smooth streak; a real camera's blur is continuous. More samples would
  cost proportionally more.
- Camera: `camera.py` over-reads the pan more (120 against 83 degrees of
  error), plausibly because the flattened banners and snow carry less
  texture to track.

**Unverified.** Which of these matters is not measured; neither is whether
"harder for the NIM" means "more like broadcast" beyond the one number real
footage has (confidence). The comparison is 30 scenes. The pipeline for
the Isaac clips ran on the Mac (tracker on CPU) and for the Cycles clips on
the box (tracker on the GPU), both against the same NIM image, each on an
L4; the tracker control above rules out the machine for tracking, not for
the NIM call.

## The training set

October 2026: **285 clips**, `synthb0001` to `synthb0285`, rendered on one
AWS L4 (`g6.xlarge`) and run through the unchanged pipeline against a
self-hosted NIM. For the airborne segmenter and the factored trick
classifier, which train on the pipeline's output, never on the rig
([training.md](training.md#train-on-nim-output-never-on-the-rigs-clean-skeleton)).

```bash
snow synth plan --scale 400 --seed 1                              # rendered 0001-0146
snow synth plan --scale 254 --start 147 --seed 1 --long-ground    # replaced 0147-0400; rendered 0147-0285
SNOW_NIM_PREVIEW=0 SNOW_NIM_TARGET=127.0.0.1:8001 SNOW_SYNTH_PRUNE=1 \
  scripts/synth_batch.sh render synthb &  scripts/synth_batch.sh pipeline synthb
```

The draws depend on the plan's size, so these exact commands reproduce the
clips; a plan of 146 would not. Every prefix of a plan is balanced, so
rendering in order and stopping at the budget leaves a balanced set. The
unrendered plans 0286-0400 are kept on the box in `data/synth/_unrendered/`.

**Balanced** in shuffled blocks across what the classifier learns, so any
prefix is balanced: kind in thirds (94 flat, 95 cork, 96 double cork), spin
evenly within each kind (flat 180-1440, cork 540-1440, double cork
900-1440), direction (142 / 143), stance, grab (40-41 of each of the seven,
`none` included), camera position (70-72 each) and slow motion 1-4x (71-72
each). Each factor is shuffled on its own. Inversions match the kind on
every clip (0 / 1 / 2).

**Split groups.** Clips come in groups of four: one rider (body shape,
outfit, board) at one venue (dressing, sun), like one athlete's runs at one
event; 72 groups. `labels.json` and the manifest carry `group`; keep a group
on one side of any train/test split.

**Two families of ground time**, because a segmenter trained on the first
one called a real rider airborne while standing still:

| | clips | frames | airborne | approach | ride-out |
|---|---|---|---|---|---|
| `short_ground` (0001-0146) | 146 | 50,356 | 60.8% | 0.5-1.3 s straight up the kicker | 0.5-1.3 s straight down the landing |
| `long_ground` (0147-0285) | 139 | 96,891 | 30.1% | 1.6-4.0 s, half stood still for 0.5-1.5 s first | 2.0-4.5 s, carving to a stop |

`long_ground` (`Trick.terrain`, `tricks._ground`) rides the venue's own
surface: stood still at the top of the deck, gathering speed along it, a
transition onto the kicker of about 15 m radius, then down the landing and
onto the outrun. 70% carve S-turns sized by a 17-39 degree lean, and every
rider has a slow balance wobble of 0.5-3 degrees. Air frames are exactly
those of the same trick without it, so the labels are unchanged
(`test_synth.py`). No ground frame turns the body faster than 12 degrees
per 30 fps frame.

**A flaw in `short_ground`.** Its approach is the kicker's line extended
backwards, which is below the deck for approaches longer than the ramp: 75
of the 146 clips start with the board under the snow, 1,267 frames (2.5% of
the family's). The rider is hidden there, and `gt_raw.npz`'s per-frame
`visible` says so; mask those frames when training. `long_ground` has none.

**Cost.** Median per clip, short / long: 332 / 657 frames, render 277 / 541
s (Cycles, 16 samples, OptiX denoiser, about 0.6 s a frame plus about 30 s
to build the scene), pipeline 100 / 161 s, on two render loops and one
pipeline loop. 285 clips took about 18.6 hours of the L4, about $15, or
about $0.05 a clip.

**Where it lives.** Locally, lean: `data/synth/<clip>/` (trick, labels,
motion, `gt_raw.npz`, true camera, render, `compare.json`), the pipeline's
`data/poses/<clip>.*` and `data/clips/<clip>.{bbox.txt,meta.json,tracks.json}`.
Everything, including `scene.blend`, `scene.usdc` and the ground truth in
the NIM's schema, is under `s3://snow-artifacts-009160029345/synth/data/` in
the same layout. `data/synth/MANIFEST.scale.jsonl` (also at
`s3://.../synth/MANIFEST.scale.jsonl`) has one line per clip: `name`,
`status`, `group`, `renderer`, `labels`, file paths, render and pipeline
seconds. All 285 are `done`. Errors: [validation.md](validation.md#synthetic-training-set-285-clips).

## Limits

- **Blender renders for the published tables**: EEVEE for the first 20,
  Cycles for the training set; 30 Replicator renders exist and read worse
  (above) (the same scene and materials; Cycles was checked
  against EEVEE on one frame and matches in exposure). Body
  and board are plain materials: no cloth, hair, snow spray or crowd motion.
  Renders may be easier or harder for the NIM than broadcast footage, so
  synthetic accuracy is not a claim about real clips (see
  [validation.md](validation.md)).
- Big air only. Halfpipe needs a moving camera position and a wall, which
  break `camera.py`'s fixed-centre model differently.
- The cork model is one family of motions. Real riders' axis paths differ,
  which matters most for `axis_spin_deg`, the definition most sensitive to
  axis movement.
