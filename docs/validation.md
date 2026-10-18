# What is actually validated

Read this before writing anything user-facing about accuracy, and before
trusting a number out of this pipeline.

The failure mode this project is most exposed to is **a confident wrong
number**. Every stage produces plausible-looking output when it is wrong: a
mis-stitched track yields a smooth skeleton of two different people, an
uncompensated camera yields rotation nobody performed, an unset slow-motion
factor yields rates that are simply a few times too low. None of it looks
broken.

## Proven

**Synthetic motion: exact.** `tests/test_rotation.py` constructs a rider
turning by a known amount about a known axis and checks the recovered answer:

- `axis_spin_deg` recovers 360 / 720 / 1080 to within **2°**
- axis tilt to within **2°**
- camera compensation moves rotation between camera and rider **in both
  directions** and cancels

**The bbox format.** `tests/test_bbox_format.py` parses our annotation with
**NVIDIA's own reader**, so the format contract is tested against the real
consumer rather than our understanding of it.

**The USD export.** Validated with **USD's own skinning resolver** when
`usd-core` is installed: 77 joints resolve, and the skeleton is correctly
inverted at the point in the clip where the rider is corked.

**The NIM's focal-length assumption.** Measured, not guessed: `f = hypot(w,h)`
fitted 2206.1 px against 2202.9 predicted, over 423 frames, IQR 2182–2229.

**Camera compensation helps on real footage.** It moved a measured spin axis
from 67° off vertical to 33°. That is a sanity improvement — a snowboard spin
really is close to vertical — but it is not a ground-truth comparison.

**Coordinate conventions.** X right, Y down, Z forward; read off the data (the
head sits at negative Y relative to the pelvis), not assumed from docs.

## Not proven

**Absolute spin counts on real footage.** This is the open one.

On a clip taken from a video titled *"Frontside Double Cork 1080"* the
pipeline reads:

| | |
|---|---|
| `axis_spin_deg` | **668°** about an axis 33° off vertical |
| total angular path | 1300° |
| max tilt | 136° (correctly inverted) |
| vertical yaw | −577° |

That is well short of 1080°.

**It has not been established that the scouted shot is the trick the title
names.** It was chosen by scouting heuristics out of a 22-shot edit. So this
is an **open discrepancy, not a measured error rate** — it could be a
measurement error, or it could be the right answer for a different shot.

Settling it needs a clip whose trick is known frame by frame. This is the
single highest-value thing left to do, because every downstream ML label
depends on the spin count being right.

**Update:** the synthetic loop below reproduces this undercount on known
double cork 1080s, and finds its causes. See *Synthetic ground truth:
measured error*.

**Until then: treat absolute spin counts as unvalidated.** Relative
comparisons, axis tilt and the inversion signal are on firmer ground.

**Metric scale.** The focal length is assumed, so every length is off by an
unknown factor. The visible tell is that exported skeletons stand about
1.1–1.4 m tall. Angles and ratios hold; absolute metres do not. Never print a
length as though it were a real-world measurement — the web report labels its
one camera-relative length explicitly rather than calling it a pop height.

**Real-time rates on slow-motion footage.** The slow-motion factor cannot be
recovered from the file, and is 1.0 unless someone supplied it. Any per-second
number on conformed slow motion is low by that factor, and any duration is
high by it.

## Synthetic ground truth: measured error

October 2026. Twenty parametric big-air tricks (flat 360-1440, cork
540-1080, double cork 1080-1440; four camera positions; slow motion 1-4x)
rendered with exact labels and run through the **unchanged** pipeline. How
they are made: [synthetic.md](synthetic.md). Reproduce with `snow synth all`.

Two variants of every clip:

- **oracle**: the rig's exact skeleton and the true camera through `snow
  features` and the dashboard's air detector. Error here is the measurement
  *definition*.
- **pipeline**: the rendered video through `snow run --features` (tracker,
  NIM, `camera.py`, `features.py`), then the dashboard's air detector. Error
  here is what a user sees.

**What this is not.** Renders are Blender EEVEE with plain materials, not
Omniverse Replicator (written, not yet run on an RTX machine), and the corks
are one generated family of motions. Synthetic accuracy is not a claim about
broadcast footage; synthetic *failure* is still a failure, and every finding
below has a mechanism that does not depend on rendering style.

### Findings

1. **`axis_spin_deg` undercounts corks even on perfect input.** On the exact
   skeleton it reads, as a fraction of the named spin (median): flat 0.94,
   cork 0.81, double cork 0.85. A double cork 1080 reads 786-1050. It agrees
   with its own definition to 9 degrees mean, so the maths is right; the
   definition is wrong for corks: projecting onto one dominant axis drops
   the rotation that was not about it (the cork's moving axis, and the
   rider pitching from the kicker's angle to the landing's). The dashboard
   rounds it to the nearest 180 and names the trick correctly on **7 of 20
   exact clips**: 6 of 7 flat spins, 1 of 13 corks.
2. **Vertical yaw (`spin_deg`) is exact on flat spins and loses whole
   revolutions on corks**, also on perfect input: median 1.00 x named on
   flat, but a double cork 1080 reads 728 and a cork 540 reads 190. That is
   the documented hip-axis degeneracy, now measured. Neither existing
   measure counts a corked spin correctly; a cork-aware count is needed.
3. **The dashboard's direction is always "counter-clockwise".** 0 of 9
   clockwise tricks were reported clockwise, in both variants. The event's
   rotation is the `axis_spin_deg` series, which is projected onto its own
   net axis and so only ever grows. The sign of vertical yaw is right on
   20 of 20, in both variants.
4. **The NIM misses inversions.** Of 13 corked tricks, the pipeline's max
   tilt passed 90 degrees on 7 (the oracle: 13 of 13), and it reported an
   inversion on 2 of 7 flat spins. Where the inversion is missed, the spin
   collapses: five of those six read under 230 degrees. Pipeline `axis_spin_deg` as a
   fraction of named (median): flat 0.84, cork 0.17, double cork 0.35.
5. **`camera.py` overestimates the camera's rotation**: median 1.63x the
   true amount (range 0.49-4.44, over 1.3x on 12 of 20). The broadcast
   camera's true focal was 2.2-10.9x the assumed `hypot(w, h)`, and a
   homography read with too short a focal over-reads rotation by roughly
   that ratio. The excess is subtracted from the rider.
6. **False cuts split continuous shots.** On 2 of 20 single-shot clips the
   cut detector found 10 and 17 cuts during fast telephoto pans. Rotation is
   never integrated across a cut (invariant 6), so a double cork 1080 read
   214 degrees. The real 668-degree clip has one segment, so this does not
   explain it.
7. **Air detection is sound on clean poses**: on the oracle it finds 92% of
   the air (median), takeoff about 8 frames late and landing about 10 early
   at slow-motion frame rates. Through the pipeline, 46%.

**What this settles.** Synthetic frontside double cork 1080s read 214-810
degrees through the pipeline, and 786-1050 even on exact data. The open
668-degree reading on a clip titled as a double cork 1080 no longer needs
the wrong-shot explanation: the pipeline undercounts that trick
reproducibly, for reasons 1, 4 and 5 above.

### Fixed after this table: direction, and a cork-aware count

Findings 1-3 are measurement definitions, so they were fixable without a
better pose model. `features.py` now produces `spin_path_deg`, the angular
path signed by the turn's sense about world up, and orients
`axis_spin_deg`'s axis upward so it carries the sense too. The dashboard
counts airs with `spin_path_deg` ([rotation.md](rotation.md)).

On the same 20 clips, re-run through `snow features` and the dashboard's
air detector (no new NIM calls):

| | named spin right, before | after | direction right, before | after |
|---|---|---|---|---|
| exact skeleton, true camera | 7/20 | **20/20** | 11/20 | **20/20** |
| pipeline (NIM, `camera.py`) | 3/18 | **6/18** | 9/18 | **17/18** |

Through the pipeline the remaining misses are findings 4-6: missed
inversions, `camera.py`'s over-read rotation and false cuts. No choice of
measure recovers rotation the pose never showed. The path also accumulates
pose jitter: through the NIM it read flat spins 5-10% high (a 360 as 395).

On the real clip titled *"Frontside Double Cork 1080"*, the main air now
reads **-998 degrees, clockwise, named 1080** (it read 668 as
`axis_spin_deg`). That agrees with the title. It is one reading of one clip
by a measure that over-reads flat spins by up to 10% through the NIM, so it
is consistent with the title, not proof of it.

`snow features` also crashed printing its summary when a segment had no
dominant axis (finding 6 produces such segments); fixed in `cli.py`, no
effect on any number.

### Pipeline, by trick

| kind | n | axis spin vs named | axis spin vs exact | vertical yaw vs named | dashboard spin right | yaw sense right | dashboard direction right | axis tilt | max tilt | camera rotation | takeoff (frames) | landing (frames) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cork | 6 | 550 | 399 | 625 | 0/4 | 6/6 | 1/4 | 12 | 71 | 50 | 103 | 60 |
| double_cork | 7 | 646 | 461 | 755 | 0/7 | 7/7 | 4/7 | 16 | 35 | 66 | 76 | 81 |
| flat | 7 | 120 | 70 | 75 | 3/7 | 7/7 | 4/7 | 27 | 36 | 51 | 42 | 21 |
| **all** | 20 | 433 | 306 | 478 | 3/18 | 20/20 | 9/18 | 18 | 46 | 56 | 69 | 53 |

### Pipeline, by camera position

| camera_angle | n | axis spin vs named | axis spin vs exact | vertical yaw vs named | dashboard spin right | yaw sense right | dashboard direction right | axis tilt | max tilt | camera rotation | takeoff (frames) | landing (frames) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| back_quarter | 4 | 539 | 368 | 583 | 0/4 | 4/4 | 2/4 | 25 | 50 | 52 | 4 | 99 |
| front_quarter | 5 | 278 | 148 | 444 | 1/5 | 5/5 | 3/5 | 13 | 37 | 110 | 82 | 29 |
| low_landing | 5 | 429 | 297 | 428 | 1/5 | 5/5 | 2/5 | 24 | 53 | 27 | 118 | 34 |
| side | 6 | 496 | 404 | 478 | 1/4 | 6/6 | 2/4 | 14 | 44 | 38 | 55 | 62 |
| **all** | 20 | 433 | 306 | 478 | 3/18 | 20/20 | 9/18 | 18 | 46 | 56 | 69 | 53 |

### Oracle (exact skeleton, true camera), by trick

| kind | n | axis spin vs named | axis spin vs exact | vertical yaw vs named | dashboard spin right | yaw sense right | dashboard direction right | axis tilt | max tilt | camera rotation | takeoff (frames) | landing (frames) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cork | 6 | 165 | 10 | 241 | 0/6 | 6/6 | 3/6 | 8 | 11 | 0 | 12 | 19 |
| double_cork | 7 | 187 | 8 | 207 | 1/7 | 7/7 | 4/7 | 7 | 19 | 0 | 8 | 10 |
| flat | 7 | 57 | 9 | 7 | 6/7 | 7/7 | 4/7 | 10 | 15 | 0 | 11 | 10 |
| **all** | 20 | 135 | 9 | 147 | 7/20 | 20/20 | 11/20 | 8 | 15 | 0 | 10 | 13 |

### Oracle, by camera position

| camera_angle | n | axis spin vs named | axis spin vs exact | vertical yaw vs named | dashboard spin right | yaw sense right | dashboard direction right | axis tilt | max tilt | camera rotation | takeoff (frames) | landing (frames) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| back_quarter | 4 | 175 | 11 | 281 | 1/4 | 4/4 | 2/4 | 16 | 26 | 0 | 11 | 7 |
| front_quarter | 5 | 149 | 12 | 74 | 1/5 | 5/5 | 3/5 | 5 | 20 | 0 | 11 | 11 |
| low_landing | 5 | 145 | 7 | 74 | 2/5 | 5/5 | 2/5 | 5 | 12 | 0 | 10 | 11 |
| side | 6 | 87 | 6 | 181 | 3/6 | 6/6 | 4/6 | 8 | 6 | 0 | 10 | 20 |
| **all** | 20 | 135 | 9 | 147 | 7/20 | 20/20 | 11/20 | 8 | 15 | 0 | 10 | 13 |

Mean absolute errors in degrees unless marked; "right" columns are counts.
"axis spin vs named" is against the trick's name (1080); "vs exact" is
against `axis_spin_deg` computed on the exact motion. Two pipeline clips
produced no air event, so dashboard columns count 18.

### Per clip

| clip | variant | named | exact axis spin | got axis spin | got vertical yaw | dashboard | direction (true / dashboard) | axis tilt (true / got) | camera ° (true / got) | air frames (true / got) |
|---|---|---|---|---|---|---|---|---|---|---|
| synth01_flat_360 | oracle | 360 | 336 | 355 | +370 | 360 | cou / cou | 4 / 13 | 72 / 72 | 92 / 85 |
| synth01_flat_360 | pipeline | 360 | 336 | 304 | +294 | 360 | cou / cou | 4 / 12 | 72 / 133 | 92 / 78 |
| synth02_flat_540 | oracle | 540 | 508 | 512 | +543 | 540 | cou / cou | 7 / 14 | 83 / 83 | 326 / 278 |
| synth02_flat_540 | pipeline | 540 | 508 | 502 | +531 | 540 | cou / cou | 7 / 18 | 83 / 111 | 326 / 266 |
| synth03_flat_720 | oracle | 720 | 668 | 657 | +742 | 720 | cou / cou | 4 / 29 | 65 / 65 | 175 / 165 |
| synth03_flat_720 | pipeline | 720 | 668 | 602 | +612 | 540 | cou / cou | 4 / 61 | 65 / 159 | 175 / 158 |
| synth04_flat_900 | oracle | 900 | 831 | 821 | -902 | 900 | clo / cou | 5 / 12 | 26 / 26 | 308 / 284 |
| synth04_flat_900 | pipeline | 900 | 831 | 641 | -654 | 540 | clo / cou | 5 / 48 | 26 / 52 | 308 / 208 |
| synth05_flat_1080 | oracle | 1080 | 1012 | 1012 | -1069 | 1080 | clo / cou | 3 / 14 | 68 / 68 | 307 / 290 |
| synth05_flat_1080 | pipeline | 1080 | 1012 | 1011 | -1082 | 900 | clo / cou | 3 / 23 | 68 / 133 | 307 / 275 |
| synth06_flat_1440 | oracle | 1440 | 1352 | 1352 | -1442 | 1260 | clo / cou | 5 / 10 | 39 / 39 | 243 / 228 |
| synth06_flat_1440 | pipeline | 1440 | 1352 | 1369 | -1444 | 1440 | clo / cou | 5 / 23 | 39 / 109 | 243 / 226 |
| synth07_flat_720 | oracle | 720 | 668 | 649 | +718 | 720 | cou / cou | 5 / 12 | 65 / 65 | 287 / 262 |
| synth07_flat_720 | pipeline | 720 | 668 | 489 | +634 | 180 | cou / cou | 5 / 38 | 65 / 51 | 287 / 84 |
| synth08_cork_540 | oracle | 540 | 439 | 434 | +190 | 360 | cou / cou | 52 / 54 | 63 / 63 | 325 / 235 |
| synth08_cork_540 | pipeline | 540 | 439 | 50 | +30 | – | cou / – | 52 / 27 | 63 / 70 | – |
| synth09_cork_720 | oracle | 720 | 571 | 565 | -717 | 540 | clo / cou | 49 / 53 | 90 / 90 | 280 / 255 |
| synth09_cork_720 | pipeline | 720 | 571 | 99 | -70 | 180 | clo / cou | 49 / 44 | 90 / 171 | 280 / 62 |
| synth10_cork_900 | oracle | 900 | 695 | 697 | -514 | 720 | clo / cou | 35 / 49 | 83 / 83 | 274 / 245 |
| synth10_cork_900 | pipeline | 900 | 695 | 418 | -433 | 360 | clo / cou | 35 / 32 | 83 / 69 | 274 / 127 |
| synth11_cork_1080 | oracle | 1080 | 882 | 884 | -1082 | 900 | clo / cou | 37 / 41 | 33 / 33 | 312 / 296 |
| synth11_cork_1080 | pipeline | 1080 | 882 | 226 | -14 | 180 | clo / cou | 37 / 55 | 33 / 33 | 312 / 38 |
| synth12_cork_720 | oracle | 720 | 592 | 598 | +369 | 540 | cou / cou | 44 / 57 | 74 / 74 | 234 / 215 |
| synth12_cork_720 | pipeline | 720 | 592 | 49 | +37 | – | cou / – | 44 / 60 | 74 / 131 | – |
| synth13_cork_1080 | oracle | 1080 | 837 | 873 | +724 | 900 | cou / cou | 30 / 36 | 66 / 66 | 82 / 74 |
| synth13_cork_1080 | pipeline | 1080 | 837 | 894 | +704 | 900 | cou / cou | 30 / 27 | 66 / 208 | 82 / 72 |
| synth14_double_cork_1080 | oracle | 1080 | 1044 | 1050 | +728 | 1080 | cou / cou | 48 / 43 | 71 / 71 | 322 / 305 |
| synth14_double_cork_1080 | pipeline | 1080 | 1044 | 214 | +227 | 180 | cou / cou | 48 / 44 | 71 / 69 | 322 / 59 |
| synth15_double_cork_1080 | oracle | 1080 | 916 | 918 | +1083 | 900 | cou / cou | 56 / 52 | 86 / 86 | 279 / 261 |
| synth15_double_cork_1080 | pipeline | 1080 | 916 | 810 | +671 | 540 | cou / cou | 56 / 40 | 86 / 382 | 279 / 106 |
| synth16_double_cork_1080 | oracle | 1080 | 979 | 990 | -1058 | 900 | clo / cou | 50 / 68 | 77 / 78 | 87 / 80 |
| synth16_double_cork_1080 | pipeline | 1080 | 979 | 535 | -525 | 540 | clo / cou | 50 / 86 | 77 / 91 | 87 / 40 |
| synth17_double_cork_1080 | oracle | 1080 | 792 | 786 | +718 | 720 | cou / cou | 55 / 58 | 51 / 51 | 265 / 241 |
| synth17_double_cork_1080 | pipeline | 1080 | 792 | 352 | +342 | 360 | cou / cou | 55 / 48 | 51 / 76 | 265 / 99 |
| synth18_double_cork_1260 | oracle | 1260 | 1067 | 1067 | -1249 | 1080 | clo / cou | 43 / 54 | 71 / 71 | 248 / 220 |
| synth18_double_cork_1260 | pipeline | 1260 | 1067 | 437 | -506 | 360 | clo / cou | 43 / 31 | 71 / 34 | 248 / 91 |
| synth19_double_cork_1440 | oracle | 1440 | 1258 | 1246 | -1436 | 1260 | clo / cou | 37 / 39 | 74 / 74 | 191 / 180 |
| synth19_double_cork_1440 | pipeline | 1440 | 1258 | 1163 | -665 | 720 | clo / cou | 37 / 69 | 74 / 76 | 191 / 97 |
| synth20_double_cork_1440 | oracle | 1440 | 1114 | 1097 | +745 | 1080 | cou / cou | 35 / 43 | 72 / 72 | 326 / 299 |
| synth20_double_cork_1440 | pipeline | 1440 | 1114 | 430 | +237 | 360 | cou / cou | 35 / 31 | 72 / 160 | 326 / 124 |

## Claims you may not make

- that a spin count is correct in absolute terms
- that any length in metres is a real-world measurement
- that a rate is real-time unless `slowmo_factor` was actually set
- that rotation across a camera cut was measured — it was not, by design
- that a corked trick's spin count is right through the pipeline: the
  measure names corks correctly on exact poses, but the NIM misses half the
  inversions and the pipeline named 1 of 11 corks it measured
- that `axis_spin_deg` is a spin count: it undercounts corks even on exact
  poses; `spin_path_deg` is the count

## Two corrections worth keeping

Both of these were wrong in an earlier version of this project's own
reasoning, and both are now pinned by tests. They are recorded because the
wrong version is the intuitive one and will be re-derived otherwise.

1. **Tilting a rotation axis does not make vertical yaw undercount.** For a
   clean rigid rotation about any fixed axis the horizontal projection winds
   once per revolution, so a synthetic 1080 reads as 1080 on both measures.
   What breaks `spin_deg` is the hip axis passing *through* vertical, where
   its projection collapses — a conditioning problem, not a geometric one.
   See [rotation.md](rotation.md).

2. **An air detector gated on absolute RPM is wrong on slow motion.** A fixed
   18 RPM threshold clipped a continuous ~600° slow-motion air down to the
   142° sliver that happened to exceed it, and reported a double cork as a
   180. Gates are now fractions of the clip's own peak rate, which is
   invariant to the unknown conformance factor. See
   [../web/README.md](../web/README.md).

The shared lesson: a threshold in real-world units is only meaningful on
footage whose timebase you actually know.
