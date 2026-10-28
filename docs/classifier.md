# The trick classifier (model B)

Covers `src/snow/learn/`. The design and its rules are in
[training.md](training.md#why-the-classifier-is-factored); this is what was
built, how to run it, and what it measured. **Every number here is on
synthetic renders.** Renders may be easier or harder for the NIM than
broadcast footage, so none of it is a claim about real clips
([validation.md](validation.md)).

```bash
snow classify build            # data/synth + pipeline poses -> data/models/classifier/dataset.npz
snow classify cv --include-scale --folds 10   # grouped cross-validation, report beside it
snow classify train            # fit on the train groups, score val, test and the original 20
snow classify predict synth14_double_cork_1080   # the dashboard's airs, named
python web/tools/export_session.py synth14_double_cork_1080 --classify   # into the bundle
```

Weights are not committed (`*.pt` is gitignored). The trained checkpoint is
`s3://snow-artifacts-009160029345/models/classifier/classifier.pt`; copy it to
`data/models/classifier/classifier.pt`. Without it, `predict` says so and
`export_session.py --classify` writes the bundle unchanged, with
`source.classifier` saying why.

## What it predicts

One air in, five heads out, over one shared encoder:

| head | classes | composed into |
|---|---|---|
| rotation | 0 to 1800 in steps of 180 | the number |
| sense | clockwise / counter-clockwise, about world up | frontside/backside, with stance |
| stance | regular / goofy | frontside/backside, with sense |
| axis | flat / cork / double cork | the cork |
| grab | none, indy, melon, mute, stalefish, tail, nose | the grab |

A composed name ("frontside double cork 1080 melon") is assembled from the
heads; a head can be ignored without touching the others.

### Mirroring flips sense and stance, not frontside/backside

training.md says mirroring a frontside rotation makes it backside. It does
not, and the code follows the geometry, not that sentence. A mirror reverses
the turn's **sense** (counter-clockwise becomes clockwise) *and* the
**stance** (a regular rider becomes goofy). Frontside is counter-clockwise
for a regular rider and clockwise for a goofy one (`tricks.Trick.spin_sign`),
so it survives the mirror. That is why the model predicts sense and stance
rather than frontside/backside directly: sense is what a mirror flips, and
what the label must flip with it (`dataset.mirrored_labels`, pinned in
`tests/test_classifier.py`). The trap the doc warns about is real; it is
just on the sense head.

## Input: one air as a fixed-shape array

`airs.encode` turns frames `start..end` of a clip's pose track into 71
channels by 64 steps:

- **rotation totals**, in turns, from the body frame with the camera removed
  exactly as `features.py` does it: the unsigned path, its part about world
  up (signed), its part about horizontal axes, and the vertical part read
  *without* removing the camera (`camera.py` over-reads on a long lens, so the
  model gets both); plus their per-step increments
- the rider's up axis in the world
- 17 joints in the rider's own body frame, divided by the neck-to-root length
- four wrist-to-foot distances over the same length (the grab)
- root motion over torso lengths, the fraction of frames posed, confidence

What the rules in training.md became:

- **NIM output only.** Built from the render through tracker, NIM and
  `camera.py`; the `-oracle` variant (the rig's exact skeleton) is never read.
- **Torso-normalised.** No length in metres reaches the model.
- **Timebase-invariant.** Every air is resampled to 64 steps and rotation is
  in turns and fractions of the air, never per second. Augmentation thins
  frames by a random stride, which is the same air at a lower slow-motion
  factor. `tests/test_classifier.py` checks a 1080 at 1x and 4x encode alike.
- **Never across a cut.** `airs.window` keeps the shot holding most of the
  air's poses.
- **Labels from the scene.** `labels.json`, measured from the generated
  motion; `spin_path_deg` and `axis_spin_deg` are the baseline, never a label.
- **Split by group.** The scale-up's `group` (several tricks per group) is
  split whole, by the rank of its hash, 70/15/15 with at least one test and
  one val group. The original 20 are one event each.

Augmentation, per training sample: mirror (labels flipped), boundaries moved
by -5% to +10% of the air at each end, frame thinning, a random pose hole of
up to 20 frames, and the camera's rotation scaled by 0.7 to 1.4x. Prediction
averages an air and its mirror, with the mirrored sense and stance swapped
back, so the answer cannot depend on which way the camera faced.

### Augmentation must not change what the label means

The first run used boundary jitter of up to 30% inward per end and a 0.5 to
2x camera gain. With the label unchanged, the encoded path read **0.92x**
the true spin in training against **1.11x** on clean airs at evaluation, and
the rotation head fell to **16%** held out (geometry 41%) on 81 airs: it had
learned that the total turn means nothing. At the current settings training
reads 1.06x against 1.11x. Any new augmentation that can change the true
answer (trimming rotation off an air, scaling it) needs this check.

## Model

A dilated temporal CNN (5 residual blocks, width 64, dilations 1-8) with
attention pooling and mean pooling; the heads also see the air's rotation
totals (the last step of each cumulative channel) directly. A sixth output
regresses the spin in turns, as an auxiliary loss and as a second rotation
answer. 111k parameters. Cross-entropy per head plus 0.5 x smooth-L1 on
turns, AdamW, one-cycle, 40 epochs over 128 augmented variants per air;
one to two minutes per fit on the shared L4. Training loss ends near 0.002:
it memorises the train groups, so held-out scores are the only ones that
mean anything, and more groups are the next lever, not a bigger model.

On the AWS box, cuDNN only loads with the venv's `nvidia/cu13/lib` and
`nvidia/cudnn/lib` on `LD_LIBRARY_PATH` (torch 2.14 cu130, driver 595);
without them a step takes 100 ms instead of 5.5.

## Results

October 2026. 154 airs: the 20 original synthetic clips (Blender EEVEE) and
134 from the scale-up (Blender Cycles, 34 groups), flat / cork / double cork
in equal thirds, four camera positions, slow motion 1-4x. Both windows are
scored:

- **true air**: the label's takeoff to landing frames, which is what the
  classifier can do once the air is right;
- **detector's air**: the dashboard's heuristic event (`export_session.py`),
  which is what a user sees until the segmenter (model A) replaces it.

The geometric baseline on the same frames: `spin_path_deg` rounded to 180
(the dashboard's count), the sign of net vertical yaw for the sense, and the
number of separate inversions in `tilt_deg` for the axis. Grab and stance
have no geometric reading, so their baseline is the training majority (chance
is 1 in 7 and 1 in 2).

### Grouped 10-fold cross-validation, all 154 airs

Every air is scored once, by a model that never saw its group.

| head | model, true air | geometry, true air | model, detector's air | geometry, detector's air |
|---|---|---|---|---|
| rotation | **64/154 (42%)** | 60/154 (39%) | 55/154 (36%) | **68/154 (44%)** |
| rotation within 180 | 119/154 (77%) | 116/154 (75%) | 105/154 (68%) | 117/154 (76%) |
| rotation, mean abs error | **152** | 198 | 194 | 174 |
| rotation, regressed and rounded | **73/154 (47%)** | | 60/154 (39%) | |
| sense | **154/154 (100%)** | 148/154 (96%) | 148/154 (96%) | 147/154 (95%) |
| stance | **111/154 (72%)** | 75/154 (majority) | **102/154 (66%)** | 75/154 |
| frontside/backside | **111/154 (72%)** | none | 100/154 (65%) | none |
| axis | 114/154 (74%) | **121/154 (79%)** | 109/154 (71%) | 111/154 (72%) |
| grab | **65/154 (42%)** | 23/154 (majority) | **59/154 (38%)** | 23/154 |
| spin, sense and axis all right | 49/154 (32%) | 50/154 (32%) | 41/154 (27%) | **56/154 (36%)** |

Confusion, true air (rows true, columns predicted):

**rotation, model**

| | 180 | 360 | 540 | 720 | 900 | 1080 | 1260 | 1440 |
|---|---|---|---|---|---|---|---|---|
| 180 | 2 | 1 | 2 | 1 | 0 | 0 | 0 | 0 |
| 360 | 1 | 2 | 2 | 1 | 0 | 0 | 0 | 0 |
| 540 | 3 | 0 | 0 | 7 | 3 | 3 | 0 | 0 |
| 720 | 0 | 0 | 4 | 5 | 3 | 5 | 0 | 0 |
| 900 | 0 | 0 | 1 | 3 | 11 | 5 | 5 | 1 |
| 1080 | 0 | 0 | 0 | 3 | 4 | 11 | 6 | 6 |
| 1260 | 0 | 0 | 0 | 0 | 1 | 3 | 13 | 9 |
| 1440 | 0 | 0 | 0 | 0 | 0 | 0 | 7 | 20 |

**rotation, geometry** (`spin_path_deg` rounded; it also reads 0, 1620, 1800)

| | 0 | 180 | 360 | 540 | 720 | 900 | 1080 | 1260 | 1440 | 1620 | 1800 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 180 | 0 | 2 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 360 | 0 | 1 | 3 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 540 | 0 | 2 | 0 | 5 | 7 | 1 | 1 | 0 | 0 | 0 | 0 |
| 720 | 0 | 1 | 3 | 0 | 7 | 4 | 0 | 2 | 0 | 0 | 0 |
| 900 | 0 | 1 | 0 | 3 | 5 | 10 | 5 | 2 | 0 | 0 | 0 |
| 1080 | 1 | 1 | 1 | 3 | 3 | 2 | 10 | 7 | 2 | 0 | 0 |
| 1260 | 0 | 1 | 2 | 1 | 1 | 0 | 4 | 8 | 6 | 2 | 1 |
| 1440 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 1 | 15 | 8 | 1 |

**axis**, model then geometry

| | flat | cork | double cork | | flat | cork | double cork |
|---|---|---|---|---|---|---|---|
| flat | 49 | 2 | 1 | | 45 | 5 | 2 |
| cork | 3 | 27 | 21 | | 3 | 36 | 12 |
| double cork | 2 | 11 | 38 | | 0 | 11 | 40 |

**grab, model**

| | none | indy | melon | mute | stalefish | tail | nose |
|---|---|---|---|---|---|---|---|
| none | 22 | 0 | 2 | 0 | 0 | 1 | 1 |
| indy | 0 | 3 | 4 | 6 | 8 | 0 | 0 |
| melon | 1 | 8 | 10 | 2 | 1 | 0 | 0 |
| mute | 0 | 3 | 4 | 11 | 2 | 0 | 0 |
| stalefish | 2 | 8 | 2 | 4 | 3 | 2 | 0 |
| tail | 1 | 0 | 3 | 0 | 0 | 11 | 9 |
| nose | 3 | 0 | 2 | 0 | 0 | 10 | 5 |

Sense on the detector's air was wrong 6 times for the model and 7 for the
geometry; full matrices for both windows are in `cv_report.md` beside the
dataset.

### The shipped checkpoint, on its held-out groups

Trained on the 94 train airs; val (20) and test (20) are whole groups it never
saw. The original 20 were not trained on either, and they are a different
renderer (EEVEE, not Cycles).

| | test, model | test, geometry | val, model | val, geometry | original 20, model | original 20, geometry |
|---|---|---|---|---|---|---|
| rotation, true air | **8/20** | 4/20 | **11/20** | 8/20 | 4/20 | **11/20** |
| rotation, detector's air | 9/20 | **10/20** | **10/20** | 7/20 | 4/20 | **13/20** |
| sense, true air | **19/20** | 18/20 | 20/20 | 20/20 | 20/20 | 20/20 |
| stance, true air | 14/20 | 10/20 maj. | 14/20 | 10/20 maj. | 12/20 | 9/20 maj. |
| axis, true air | 13/20 | **14/20** | 14/20 | **18/20** | 13/20 | **16/20** |
| grab, true air | 7/20 | 2/20 maj. | 8/20 | 2/20 maj. | 9/20 | 1/20 maj. |

## What this says

- **Sense works, and is the one head to trust.** 100% on the true air,
  96% on the detector's air, at or above the geometry (96% and 95%).
- **Rotation is a coin toss between the model and the geometry.** The model
  is ahead on the true air (42% against 39%, mean error 152 against 198
  degrees; rounded regression 47%), behind on the detector's air (36%
  against 44%). Neither names more than about half of synthetic spins
  through the pipeline. The geometry's errors are mostly over-reads by a
  step (pose jitter accumulating in the path; validation.md); the model's
  are spread one step either way. The model does **not** transfer to the
  EEVEE renders it was not trained on (4/20 against 11/20 on the original
  20), which is exactly the failure synthetic-to-real transfer would show,
  only smaller. Keep the geometric count as the dashboard's number.
- **Stance, and so frontside/backside, is learnable from pose: 72%**, where
  geometry has no reading at all. The dashboard says only
  clockwise/counter-clockwise today because "frontside/backside needs the
  rider's stance, which pose alone does not give"; this says it partly does.
- **Axis: the inversion count is better** (79% against 74%). Both confuse
  cork with double cork, the model more so. Keep the measurement, as
  training.md anticipated.
- **Grab is above chance but not usable**: 42% against 1 in 7, with the
  hand-to-same-edge pairs (indy/stalefish, tail/nose) mostly confused.

The detector's air costs every head a few points; that gap is what the
segmenter has to close.

## Limits

- Synthetic only, two Blender renderers, one family of cork motions. No real
  clip has a label, so nothing here is evidence about broadcast footage.
- 154 airs is still small: one group moves a held-out score by several
  points. The 20-air test split has wide error bars; the CV table is the one
  to quote.
- Grab labels assume the grab is held; the scale-up records the grab window,
  which is not yet used.
