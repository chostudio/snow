# USD / Omniverse export

Covers `src/snow/usd.py`.

```bash
snow usd run1          # -> data/poses/run1.usda
```

## Why the mapping is short

The NIM already returns what `UsdSkelAnimation` wants — **local joint
rotations against a rest pose** — so this is a translation, not a solve.

| NIM | USD |
|---|---|
| `rest_pose` | the skeleton's bind and rest transforms |
| `joint_rotations` | per-frame `rotations` |
| `root_pose.translation` | the `SkelRoot`'s own xform, i.e. root motion |

### Do not transform the root twice

`joint_rotations[0]` **is exactly** `root_pose.rotation`, and `rest_pose[0]`
is the origin. Both confirmed from real output.

So the root's rotation is already present in the joint array. Applying
`root_pose.rotation` *and* `joint_rotations[0]` double-rotates the whole
skeleton — which looks like a plausible animation, just wrong. Only the
translation goes on the `SkelRoot`.

## Coordinates

The NIM's camera frame is X right, **Y down**, Z forward. USD is Y-up. The
conversion is a half turn about X:

```
point       (x, y, z)     -> (x, -y, -z)
quaternion  (x, y, z, w)  -> (x, -y, -z, w)
```

`tests/test_crop_and_usd.py` pins that these are the *same* rotation expressed
two ways, which is the part that is easy to get subtly wrong.

### Camera compensation, one world frame per shot

The NIM reports the root in the broadcast camera's frame, so a raw export
turns the rider with every pan. `snow usd` loads the camera track from
`data/poses/<name>.camera.npz` (estimating and caching it if absent, exactly
as `snow features` does) and applies its inverse to the root: rotation and
translation both, because a camera turning about its own centre moves where
the rider appears as well as which way they face.

Only the root changes. Every other joint rotation is parent-relative and does
not depend on where the camera points. `joint_rotations[0]` and
`root_pose.rotation` are replaced together, so the root is still rotated once.

**Never across a cut** (invariant 6). Each shot's world frame is its own first
frame, and the compensation is rebased at every segment start rather than
trusting the track to have been reset, so one shot's pan cannot leak into the
next. The rider's orientation jumps at a cut in the export; that jump is
honest, because the two shots share no reference frame.

A camera track shorter than the poses is refused rather than half-applied.
`--no-camera` keeps the old camera-frame export.

`tests/test_usd_camera.py` pins it: a rider fixed in the world, a camera that
pans 60° then cuts to a second camera panning on a tilted axis. The exported
root is constant within each shot, including when the input track
(wrongly) accumulates across the cut, and USD's own skinning resolver agrees
that the skeleton holds still in world space.

## Written as text

Output is a text `.usda` on purpose: it needs no `pxr` installed to produce,
and it opens directly in Omniverse, usdview or Blender.

The test suite does validate it with **USD's own skinning resolver** when
`usd-core` is present (a dev dependency), rather than only checking that the
file parses. On a real clip that resolver confirmed: stage opens, 77 joints,
303 rotation samples, all joint transforms computable, and the pose at
t=0/100/200/300 goes upright → tucked → **inverted, head below feet** →
upright, matching the cork in the source video.

## The height tell

The exported skeleton comes out around **1.1–1.4 m** tall for an adult rider.

That is not a bug in the export — it is metric scale inheriting the NIM's
assumed focal length (see [nim.md](nim.md)). It is a useful, visible reminder
that angles and proportions are sound while absolute metres are not. See
[validation.md](validation.md).
