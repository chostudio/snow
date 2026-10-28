# The airborne segmenter (model A)

Covers `src/snow/learn/`. The design, and the traps it has to avoid, are in
[training.md](training.md); this is what was built, how to run it, and what it
measured. Read [validation.md](validation.md) before quoting any number here.

> **Status: built and measured on synthetic clips only.** Its accuracy on
> real broadcast footage is **unmeasured**: there is no hand-labelled real
> set yet (see [labelling real airs](#labelling-real-airs)). On the real
> clips available it over-calls air badly, including a rider standing still
> (below), so it is **opt-in** and must stay off for real footage until the
> gold set says otherwise.

## What it replaces

`features.py` shipped `airborne_hint = grab_distance < median(grab_distance)`:
by construction about half of every clip "airborne". The dashboard has its
own air detector, hysteresis on the rotation rate
(`web/tools/export_session.py`). The segmenter is a learned per-frame
replacement for the first and an alternative source of air boundaries for the
second.

## Running it

```bash
uv pip install -e '.[learn]'                       # torch
snow segmenter build                               # labelled clips -> data/models/segmenter/dataset.npz
snow segmenter evaluate                            # held-out scores vs both heuristics
snow segmenter train --out data/weights/segmenter.pt   # or fetch segmenter_ground240.pt from S3
snow features run1 --segmenter data/weights/segmenter.pt
web/tools/export_session.py run1                   # takes the segmenter's airs (--air rate to refuse)
```

`snow segmenter` is also `python -m snow.learn`, which needs only torch,
numpy, pandas and opencv: that is how it ran on the GPU box. `build` needs
the repo's `data/` (poses, features, camera tracks, labels); `evaluate` and
`train` need only the bundle it writes.

`snow features --segmenter` takes `auto` (the default:
`$SNOW_SEGMENTER`, else `data/weights/segmenter.pt`, if either exists),
`off`, or a path. With no weights, or no torch, it falls back to the old
heuristic and says so: the features summary carries
`airborne.source` (`heuristic` or `segmenter:<file>`) and the air spans.
`features.build()` itself defaults to `off`, so library callers and the test
suite never pick up weights by accident. Weights are not in git (`*.pt`); they
are in `s3://snow-artifacts-009160029345/models/segmenter/`.

New columns: `airborne_prob` (NaN without a model, and NaN on frames before
the rider first appears in a shot or after it last does), and `airborne_hint`
is the model's cleaned mask. The dashboard's `--air auto` uses the segmenter
only when the features table carries `airborne_prob`, and records which
detector it used in `source.air_detector`.

## Model

A residual dilated temporal CNN (kernel 3, dilations 1 to 16, 64 channels,
GroupNorm, GELU), 128,257 parameters, over **64-frame windows** (about 2 s of
playback), one logit per frame. At inference every camera segment is covered
by windows at stride 8 and the logits averaged. A BiGRU of the same width is
available (`--arch bigru`) and scored lower (frame F1 0.927 against 0.952 in
the same protocol before the jitter change). Then clean-up identical to the
dashboard's frame counts: gaps of up to 10 frames bridged, runs under 8
frames dropped, never across a cut. Threshold 0.5, not tuned.

## Inputs (`learn/inputs.py`)

One function, `frame_channels`, used for both training and serving, from
exactly what `snow features` has: the NIM detections, the features table and
the camera track. There is no second feature implementation to drift.

| channels | what | used by default |
|---|---|---|
| `pose*` (45) | 15 joints relative to the root, in the rider's body frame, divided by torso length | no (see below) |
| `cos_tilt` | body up against world up | yes |
| `rot_step` | body-frame rotation per frame, from `rotation_deg` | yes |
| `elev_step`, `azim_step` | angular motion of the ray to the rider in the camera-compensated world frame, per frame | yes |
| `grab` | `grab_distance`, torso lengths | yes |
| `confidence` | mean NIM keypoint confidence | no (see below) |
| `present` | the NIM had a body this frame | yes |

Everything is scale-free (no metres, invariant 7). Rates are per **frame**,
never per second, and training stretches time at random by 0.5x to 2x (on
top of the renders' own 1x to 4x slow motion), so the model cannot use a
playback rate it could never know on real footage (invariant 8). Holes in the
pose track are interpolated inside a segment; frames before the rider first
appears in a shot or after it last does are left empty, carry no loss in
training, and get no prediction: the in-run hidden behind the kicker is not
evidence of anything.

Ablations on the 20-clip cross-validation below (frame F1): all inputs 0.952;
without pose 0.975; without the three rate channels 0.929; without grab and
confidence 0.955. With 16 training clips the 45 pose channels mostly let
the model memorise riders. Confidence is a property of the renderer
(median 0.91 synthetic, 0.59 and 0.64 on the two real clips). Both are left
out by default (`--ignore` in the CLI).

**Jitter augmentation.** Rendered riders come out of the NIM much steadier
than broadcast riders. On the ground, synthetic `rot_step` has a median of
0.05 (0.5 degrees per frame); the two real clips read 0.26 and 0.35, the same
as synthetic *air*. A model trained without noise called every real frame
airborne. Training now adds per-frame noise to the rates and pose at a
random level per segment.

## How the rules in training.md are met

| rule | how |
|---|---|
| NIM output from rendered video, never the rig | inputs are `data/poses/<clip>.json` from `snow synth pipeline`; `-oracle` clips are refused by name |
| torso-normalised, not metres | every channel is an angle, a ratio or torso lengths |
| invariant to timebase | per-frame rates plus random time stretching; `slowmo_factor` is never an input, only used to report errors in milliseconds |
| never window across a cut | windows, interpolation, steps and clean-up are all per segment; `test_no_window_crosses_a_cut` |
| split by group, never by clip | folds and the hold-out assign whole groups; `check_no_leak` asserts it every fold |
| mirroring flips direction labels | mirror augmentation swaps left and right joints and negates X; airborne does not change under a mirror, and there is no direction label in this model |
| evaluate against ground truth, never the geometric measurement | labels are `takeoff_frame`/`landing_frame` from the trick's own motion |
| synthetic accuracy is synthetic only | every number on this page is on Blender renders |

## Results

**Synthetic only.** Blender renders of big-air tricks through the unchanged
pipeline (tracker, NIM, camera estimate, features). Every method is scored on
the same frames against the exact labels. An air is `[takeoff, landing)`;
boundary errors compare the predicted air that overlaps the true one most.
"Every seen frame" calls each frame with a body airborne: synthetic clips are
about 64% air, so frame F1 alone flatters everything.

### 20 hand-planned clips, 5-fold cross-validation by group

20 clips, 20 groups (each a separately seeded trick), 7,815 frames, 64%
airborne. Each clip is scored once, by a model that never saw it. The
settings above were chosen on these folds, so this table is optimistic.

| method | frame F1 (95% CI) | precision | recall | airs found | split | false airs | takeoff error, median (mean) frames | landing error, median (mean) frames |
|---|---|---|---|---|---|---|---|---|
| segmenter (tcn) | 0.970 (0.956 to 0.981) | 0.983 | 0.958 | 20/20 | 3 | 2 | 3.0 (8.2) | 2.5 (7.7) |
| `airborne_hint` (old) | 0.708 (0.655 to 0.758) | 0.866 | 0.599 | 20/20 | 17 | 21 | 69.0 (79.9) | 47.0 (65.0) |
| dashboard detector | 0.946 (0.920 to 0.966) | 0.992 | 0.903 | 20/20 | 0 | 0 | 4.5 (12.2) | 8.5 (13.6) |
| every seen frame (floor) | 0.825 (0.804 to 0.846) | 0.708 | 0.989 | 20/20 | 0 | 0 | 24.0 (40.0) | 73.0 (64.2) |

In real time (using the renders' known slow-motion factor, for reporting
only) the median boundary error is 33 ms at takeoff and 33 ms at landing for
the segmenter, against 53 ms and 96 ms for the dashboard detector. 9 of 20
airs have both boundaries within 3 frames, against 3 of 20.

Its weak spot against the dashboard detector is fragmentation: 3 airs split
in two and 2 false airs, where the rate detector produced none. The interval
is 20 clips wide; the two detectors' F1 intervals overlap.

### 154 clips: the scaled set, held out by group

The 20 clips above plus 134 from the scaled run (`MANIFEST.scale.jsonl`,
Cycles renders, groups of four clips sharing a rider and venue): 154 clips,
54 groups, 53,585 frames, 61% airborne. Taken when the manifest had 134
clips and was still growing.

**Headline: one held-out set never used for tuning.** 30 clips (8 groups)
drawn only from the scaled set's groups; the 20 hand-planned clips the
settings were chosen on always train. Trained on the other 124 clips.

| method | frame F1 (95% CI) | precision | recall | airs found | split | false airs | takeoff error, median (mean) frames | landing error, median (mean) frames |
|---|---|---|---|---|---|---|---|---|
| segmenter (tcn) | 0.953 (0.916 to 0.983) | 0.980 | 0.927 | 30/30 | 2 | 2 | 2.0 (14.7) | 1.0 (4.3) |
| `airborne_hint` (old) | 0.707 (0.658 to 0.746) | 0.876 | 0.592 | 30/30 | 26 | 39 | 50.0 (71.7) | 38.0 (50.4) |
| dashboard detector | 0.852 (0.777 to 0.917) | 0.983 | 0.751 | 30/30 | 5 | 0 | 5.5 (33.7) | 10.0 (31.5) |
| every seen frame (floor) | 0.801 (0.773 to 0.823) | 0.688 | 0.958 | 30/30 | 0 | 0 | 22.5 (41.3) | 51.0 (58.3) |

Median boundary error in real time: segmenter 33 ms at takeoff and 17 ms at
landing, dashboard detector 108 ms and 96 ms. Both boundaries within 3
frames: 19 of 30 airs against 7 of 30.

**Every clip once: 5-fold cross-validation by group** over all 154 clips
(the settings were tuned on 20 of them, so slightly optimistic):

| method | frame F1 (95% CI) | precision | recall | airs found | split | false airs | takeoff error, median (mean) frames | landing error, median (mean) frames |
|---|---|---|---|---|---|---|---|---|
| segmenter (tcn) | 0.968 (0.958 to 0.977) | 0.981 | 0.955 | 154/154 | 6 | 9 | 1.0 (6.7) | 1.0 (5.9) |
| `airborne_hint` (old) | 0.705 (0.686 to 0.723) | 0.867 | 0.594 | 154/154 | 136 | 201 | 54.5 (73.5) | 39.5 (53.6) |
| dashboard detector | 0.884 (0.861 to 0.907) | 0.979 | 0.806 | 153/154 | 26 | 5 | 6.0 (27.4) | 9.0 (27.6) |
| every seen frame (floor) | 0.812 (0.802 to 0.822) | 0.695 | 0.977 | 154/154 | 0 | 0 | 21.0 (37.8) | 55.5 (58.3) |

On synthetic big air the segmenter beats both heuristics clearly: on
boundaries (median 1 to 2 frames against 5.5 to 10 for the dashboard
detector) and on frame F1, where its interval clears the dashboard's. The
dashboard detector is more conservative (it misses the slow ends of airs,
recall 0.75 to 0.81) and splits more airs; the segmenter's mean errors are
pulled up by a few airs where it misses a hidden takeoff or fragments.

The 154-clip weights are `segmenter_scale154.pt`, superseded by the 240-clip
weights below.

### 240 clips: long ground added

The scaled run then added clips with a **long ground model** (`terrain` in
`trick.json`): standing still at the top, riding the flat deck, S-turns, the
kicker transition, and a long ride-out carving to a stop, about 29%
airborne against 61% for the big-air-only clips. Data at the time of
training (the manifest was still growing; it had 228 lines when this was
written, fewer than 50 more, so no second retrain): 240 clips, 75 groups,
110,265 frames, 46% airborne; 166 big-air-only clips (57 groups) and 74
long-ground clips (19 groups). Tables are split by kind of clip, because a
mix can hide a weak one.

**One held-out set never used for tuning:** 56 clips (14 groups) drawn from
the scaled run's groups, trained on the other 184. The draw put only 6
long-ground clips in it, so the long-ground row here is thin; the
cross-validation below scores all 74.

| clips | method | frame F1 (95% CI) | precision | recall | airs found | split | false airs | takeoff error, median (mean) frames | landing error, median (mean) frames |
|---|---|---|---|---|---|---|---|---|---|
| all 56 | segmenter | 0.953 (0.930 to 0.974) | 0.974 | 0.933 | 56/56 | 2 | 4 | 2.0 (11.2) | 1.0 (7.1) |
| | `airborne_hint` (old) | 0.715 (0.682 to 0.747) | 0.848 | 0.619 | 56/56 | 49 | 84 | 50.5 (66.0) | 39.5 (56.7) |
| | dashboard detector | 0.895 (0.859 to 0.928) | 0.973 | 0.829 | 56/56 | 9 | 1 | 4.5 (20.3) | 11.0 (31.2) |
| | every seen frame (floor) | 0.770 (0.743 to 0.793) | 0.643 | 0.960 | 56/56 | 0 | 0 | 32.0 (47.6) | 56.0 (73.2) |
| big air, 50 | segmenter | 0.954 (0.926 to 0.976) | 0.978 | 0.930 | 50/50 | 2 | 1 | 2.0 (11.5) | 1.0 (7.6) |
| | `airborne_hint` (old) | 0.702 (0.666 to 0.733) | 0.860 | 0.592 | 50/50 | 44 | 67 | 54.5 (70.3) | 40.5 (59.9) |
| | dashboard detector | 0.891 (0.846 to 0.926) | 0.972 | 0.823 | 50/50 | 9 | 1 | 4.5 (21.3) | 11.5 (33.0) |
| long ground, 6 | segmenter | 0.948 (0.905 to 0.978) | 0.942 | 0.955 | 6/6 | 0 | 3 | 3.0 (8.7) | 1.5 (3.3) |
| | `airborne_hint` (old) | 0.812 (0.718 to 0.877) | 0.780 | 0.846 | 6/6 | 5 | 17 | 27.0 (30.0) | 17.0 (30.7) |
| | dashboard detector | 0.928 (0.863 to 0.976) | 0.982 | 0.879 | 6/6 | 0 | 0 | 8.0 (11.3) | 7.0 (17.0) |

**Every clip once: 5-fold cross-validation by group**, all 240:

| clips | method | frame F1 (95% CI) | precision | recall | airs found | split | false airs | takeoff error, median (mean) frames | landing error, median (mean) frames |
|---|---|---|---|---|---|---|---|---|---|
| all 240 | segmenter | 0.946 (0.932 to 0.957) | 0.960 | 0.932 | 238/240 | 17 | 25 | 2.0 (10.2) | 1.0 (8.4) |
| | `airborne_hint` (old) | 0.684 (0.664 to 0.703) | 0.742 | 0.634 | 238/240 | 211 | 614 | 45.0 (66.4) | 34.0 (51.4) |
| | dashboard detector | 0.873 (0.849 to 0.893) | 0.958 | 0.802 | 235/240 | 44 | 29 | 6.0 (25.5) | 8.0 (27.2) |
| | every seen frame (floor) | 0.702 (0.678 to 0.725) | 0.552 | 0.965 | 238/240 | 0 | 2 | 30.5 (71.4) | 69.5 (100.1) |
| big air, 166 | segmenter | 0.967 (0.958 to 0.974) | 0.986 | 0.948 | 166/166 | 8 | 3 | 2.0 (8.5) | 1.0 (6.3) |
| | `airborne_hint` (old) | 0.702 (0.682 to 0.721) | 0.863 | 0.591 | 166/166 | 147 | 214 | 53.5 (74.0) | 40.5 (55.6) |
| | dashboard detector | 0.886 (0.863 to 0.907) | 0.980 | 0.808 | 165/166 | 29 | 5 | 5.0 (27.4) | 9.0 (28.4) |
| long ground, 74 | segmenter | 0.900 (0.864 to 0.929) | 0.903 | 0.896 | 72/74 | 9 | 22 | 3.0 (14.0) | 1.0 (13.1) |
| | `airborne_hint` (old) | 0.653 (0.613 to 0.695) | 0.590 | 0.732 | 72/74 | 64 | 400 | 28.0 (48.8) | 24.5 (41.9) |
| | dashboard detector | 0.845 (0.798 to 0.888) | 0.909 | 0.788 | 70/74 | 15 | 24 | 6.0 (21.0) | 5.0 (24.2) |
| | every seen frame (floor) | 0.534 (0.506 to 0.563) | 0.373 | 0.942 | 72/74 | 0 | 2 | 107.5 (146.0) | 176.0 (195.3) |

Median boundary error in real time (cross-validation): segmenter 33 ms at
takeoff and 19 ms at landing, dashboard detector 100 ms and 111 ms. On
long-ground clips alone: 44 and 17 ms against 100 and 83 ms. Both
boundaries within 3 frames: 118 of 240 airs against 36.

Long ground is harder for every method, and that is the point of it: the
floor drops from 0.81 to 0.53, so frame F1 now rewards finding the air
rather than calling everything air. The segmenter still leads, but on long
ground its false airs (22 on 74 clips, about one clip in three) are as many
as the dashboard detector's: 15 of the 22 fall on the ride-out after
landing and 7 on the approach, median 0.5 s of real time. Leaving the tilt channel out as well (`--ignore confidence pose
cos_tilt`) scored 0.951 overall and 0.912 on long ground: within the
intervals, so the default is unchanged.

The shipped weights are **`segmenter_ground240.pt`**, trained on all 240,
in `s3://snow-artifacts-009160029345/models/segmenter/weights/`, with the
dataset bundle (`dataset_ground.npz`) and every evaluation run in `runs/`.
They load with `Segmenter.load` and `snow features --segmenter PATH`, the
same interface as before.

### Real footage: unmeasured, and still a warning

No real clip has labels, so there is no real accuracy number. As a sanity
check only, the 240-clip model on the three real clips, against the
dashboard's rate detector (frame ranges `[first, end)`):

| clip | segmenter airs | dashboard airs |
|---|---|---|
| `run1` (halfpipe, 720 frames, 4 shots, 644 with a body) | 0-31, 31-66, 111-381, 501-588, 588-606, 692-720 (469 frames) | 357-381, 492-580 |
| `trick` (one trick, 312 frames, 303 with a body) | 0-303 | 40-260 |
| `vito` (240 frames, 5 shots, 84 with a body) | none | none |

`run1` opens with the rider standing at the top (shot 0, frames 0-30, and
the start of shot 1): **the segmenter still calls that standing start
airborne**, so the long-ground clips did not fix it (the 154-clip model
called 499 frames of `run1` airborne; this one 469). `vito`'s poses were made
for this check by the self-hosted NIM and are in
`data/models/segmenter/real/`; only 84 of its 240 frames have a tracked
rider, and neither detector finds an air there.

Why, measured on `run1` frames 0-66 by replacing one input at a time with
its median on synthetic standing frames: the mean probability goes from 0.88
to 0.59 with `rot_step`, to 0.53 with `grab`, and barely moves with tilt or
the ray rates. Real NIM output for a standing rider turns 0.2 (2 degrees)
per frame, against 0.03 for a rendered one, and reads the hands closer to
the feet (1.31 torso lengths against 1.58). The renders are still far
cleaner than broadcast pose, and the jitter augmentation does not close
that. **Do not enable it on real footage** until the gold set says
otherwise; the most useful next data is real labelled ground, or renders
whose pose noise matches broadcast.

## Labelling real airs

The gold set is about 50 to 100 real airs with takeoff and landing frames,
held out by event. Boundaries take roughly 100 airs an hour (training.md).

```bash
snow segmenter template                 # data/labels/airs.csv, one row per clip in data/clips
snow segmenter build --gold data/labels/airs.csv
```

Columns (`learn/labels.py`; `build` refuses a file that fails `validate`):

| column | meaning |
|---|---|
| `clip` | name under `data/clips`; frame numbers index that clip (= `frame_id` in its poses) |
| `air` | 1, 2, ... within the clip, one row per air. `0` with blank frames means the clip was watched and has **no** air, which is a label too: keep those rows |
| `takeoff_frame` | first frame the board is off the snow |
| `landing_frame` | first frame the board is back on the snow, so the air is `[takeoff, landing)` |
| `event` | the real jump: contest, round, rider, run, hit (`xgames26/sbp-m-final/rider/run2/hit3`). A live shot and its replays share it; splits never separate it. Required |
| `sure` | `y`, or `n` when a boundary is hidden (cut, occlusion, blur) |
| `labeller`, `notes` | free text |
| `frames`, `fps`, `slowmo_factor` | filled in by the template from the clip's metadata; leave them |

Conventions that matter:

- Board contact decides, not the body: a hand drag is still air, a butter is
  not.
- An air cut by a camera change: label the part on this clip and set
  `sure = n`; the model never sees across the cut anyway.
- A replay of an air already labelled: label it again on its own clip with
  the same `event`.
- Label every clip you open, including the boring ones; airs are rare in
  real broadcast and the negatives are what the model lacks most.

To find the frames, burn frame numbers into a copy and step through it in
any player that steps frame by frame:

```bash
ffmpeg -i data/clips/run1.mp4 -vf "drawtext=text='%{n}':x=10:y=10:fontsize=32:fontcolor=yellow:box=1" \
       -c:a copy /tmp/run1.numbered.mp4
```

Then `snow segmenter evaluate --bundle <bundle with gold clips>`. To report
real accuracy, hold the real clips out (`--holdout`, by event), and keep the
synthetic and real rows separate in anything published.

## Cost

Trained and evaluated on the shared `g6.xlarge` (L4) in us-west-2, which
other work kept running. On 154 clips a 5-fold cross-validation takes a few
minutes and a full training run about a minute; everything on this page
used well under an hour of GPU time.
