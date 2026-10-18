# Synthetic ground truth

Covers `src/snow/synth/`. The design and the reasons for it are in
[training.md](training.md#synthetic-ground-truth-in-omniverse); this is what
was built and how to run it. Results are in [validation.md](validation.md).

```bash
snow synth plan        # 20 tricks -> data/synth/<clip>/{trick.json,motion.npz,labels.json}
snow synth render      # headless Blender: rig, animate, venue, camera, GT, USD, frames, mp4
snow synth prepare     # GT in the NIM's schema + the "oracle" run through snow features
snow synth pipeline    # the unchanged pipeline on each render (NIM calls spaced 120 s)
snow synth compare     # error tables, per kind and per camera angle
```

Every step reads and writes files under `data/synth/<clip>/`, so any step can
be re-run alone, for all clips or `--only synth14`.

## What is needed

- **Blender 4.3** at `/Applications/Blender.app` (or `SNOW_BLENDER`), with the
  **MPFB 2.0** extension installed. Headless runs enable it for their own
  session; the user's saved preferences are not touched.
- **ffmpeg**, and the NIM key in `.env`, as for real clips.
- For the Omniverse render: a Linux machine with an **RTX** GPU (L40S, RTX
  6000 Ada, A6000, 4090; not H100/A100, which have no RT cores) and Isaac Sim.

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
stage on an RTX machine with Replicator, real-time ray traced or path traced,
with motion blur and Replicator's `camera_params` written per frame as an
independent check on the camera ground truth. Then
`snow synth encode --renderer replicator` and the remaining steps run
unchanged; the labels do not change, because it is the same stage.

**Unverified**: no RTX machine was available. The first Replicator render
must go through `prepare` and agree with the Blender ground truth before
its numbers are used.

## Limits

- **Blender EEVEE renders, not Replicator**, for the published table. Body
  and board are plain materials: no cloth, hair, snow spray or crowd motion.
  Renders may be easier or harder for the NIM than broadcast footage, so
  synthetic accuracy is not a claim about real clips (see
  [validation.md](validation.md)).
- Big air only. Halfpipe needs a moving camera position and a wall, which
  break `camera.py`'s fixed-centre model differently.
- The cork model is one family of motions. Real riders' axis paths differ,
  which matters most for `axis_spin_deg`, the definition most sensitive to
  axis movement.
