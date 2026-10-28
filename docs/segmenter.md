# The airborne segmenter (model A)

Covers `src/snow/learn/`. The design, and the traps it has to avoid, are in
[training.md](training.md); this is what was built, how to run it, and what it
measured. Read [validation.md](validation.md) before quoting any number here.

> **Status: built and measured on synthetic clips only.** Its accuracy on
> real broadcast footage is **unmeasured**: there is no hand-labelled real
> set yet (see [labelling real airs](#labelling-real-airs)). On the two real
> clips available it over-calls air badly (below), so it is **opt-in** and
> must stay off for real footage until the gold set says otherwise.

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
snow segmenter train --out data/weights/segmenter.pt
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

<!-- SCALED-RESULTS -->

### Real footage: unmeasured, and a warning

No real clip has labels, so there is no real accuracy number. As a sanity
check only, the 20-clip model was run on the two real clips with poses:

- `run1` (halfpipe, 720 frames, 4 shots): the segmenter marks most of the
  clip airborne, including the rider standing at the top before dropping in;
  the dashboard detector finds 2 airs.
- `trick` (one slow-motion trick, 312 frames): the segmenter marks frames 3
  to 303 airborne; the dashboard detector 40 to 260.

That is a domain gap, not a tuning problem. The synthetic clips are big air
only, with 0.8 s of straight in-run and 0.8 s of ride-out: the model has never
seen standing, walking, carving a halfpipe wall, crowds or a far camera. The
jitter augmentation reduced the damage (on `run1`, from about 640 to 490 of
the 644 frames with a body called airborne); it did not fix it. **Do not enable it
on real footage** until a gold set shows otherwise; the next synthetic batch
should include long riding, standing and non-trick stretches.

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

Trained and evaluated on the shared `g6.xlarge` (L4) in us-west-2. A
5-fold cross-validation run on 20 clips takes about a minute of GPU time; a
full training run about 15 s.
