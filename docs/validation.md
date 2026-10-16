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

## Claims you may not make

- that a spin count is correct in absolute terms
- that any length in metres is a real-world measurement
- that a rate is real-time unless `slowmo_factor` was actually set
- that rotation across a camera cut was measured — it was not, by design

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
