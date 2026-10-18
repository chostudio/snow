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

### Not yet camera-compensated

The export converts axes only. It does **not** subtract the camera rotation
`features.py` estimates, so the animation is in the broadcast camera's frame:
when the camera pans, the rider appears to turn with it. Fine for eyeballing
pose; wrong as a world-frame reference. Fixing it (apply the inverse camera
rotation from `<name>.camera.npz` to the root, per segment) is the first step
of the synthetic-validation plan in [training.md](training.md#synthetic-ground-truth-in-omniverse).

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
