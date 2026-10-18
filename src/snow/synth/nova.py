"""Nova-77 joint names, and where each joint sits on an MPFB human.

The synthetic rider is rigged with exactly the NIM's skeleton: 77 bones named
``joint0`` .. ``joint76`` with ``PARENTS`` as their hierarchy. Those are the
names ``usd.joint_paths`` builds its tokens from, so the rider's exported
``UsdSkel`` and ``snow usd``'s export of the NIM's reading of the same render
share one joint list and compare index for index, with no retargeting.

NVIDIA publishes the parent list but not joint names. The names below are
inferred, and checked against real NIM output rather than assumed:

* Left/right: ``keypoints_3d[42]`` lies on the ``67 -> 72`` side of the body in
  every frame of a real clip, as does ``[10]`` relative to ``[9]``. With
  ``features.HIP_L = 67``, joints 11-38 are the left arm and 39-66 the right.
* Roles: from bone lengths in the NIM's ``rest_pose`` (thigh 0.395 m, shin
  0.39, upper arm 0.257, forearm 0.236, four thumb joints and five per finger
  with metacarpals shortening index -> pinky).

What is *not* shared is the rest pose. The NIM's ``rest_pose`` is not an
anatomical stance (one arm and one leg point along +X, the other pair along
-X) and it varies by a few centimetres between clips, because it carries a
body-shape estimate. The rider's rest is its own MPFB A-pose. So local
``joint_rotations`` are only comparable between files that share a rest;
joint positions, the hierarchy and every feature derived from positions are
comparable directly. ``tests/test_synth.py`` pins that the NIM's own output
obeys plain UsdSkel forward kinematics, which is what makes that true.
"""

from __future__ import annotations

from ..skeleton import PARENTS

_FINGERS = ("index", "middle", "ring", "pinky")


def _arm(side: str) -> list[str]:
    names = [f"clavicle_{side}", f"shoulder_{side}", f"elbow_{side}", f"wrist_{side}"]
    names += [f"thumb{k}_{side}" for k in range(1, 5)]
    for finger in _FINGERS:
        names += [f"{finger}{k}_{side}" for k in range(5)]
    return names


def _leg(side: str) -> list[str]:
    return [f"hip_{side}", f"knee_{side}", f"ankle_{side}", f"ball_{side}", f"toe_{side}"]


JOINT_NAMES = [
    "pelvis", "spine1", "spine2", "spine3", "neck1", "neck2", "head", "head_top",
    "jaw", "eye_l", "eye_r",
    *_arm("l"), *_arm("r"), *_leg("l"), *_leg("r"),
]
assert len(JOINT_NAMES) == len(PARENTS) == 77


def bone_name(index: int) -> str:
    """The rig's bone name for a Nova-77 joint, matching ``usd.joint_paths``."""
    return f"joint{index}"


# Where each Nova joint sits on MPFB's "default" rig: (MPFB bone, "head" or
# "tail"). Sides use MPFB's ``.L``/``.R``; filled in for both by ``landmarks()``.
_CENTRE = {
    0: ("spine05", "head"),
    1: ("spine04", "head"),
    2: ("spine03", "head"),
    3: ("spine02", "head"),
    4: ("neck01", "head"),
    5: ("neck03", "head"),
    6: ("head", "head"),
    7: ("head", "tail"),
    8: ("jaw", "mid"),
}
_ARM = [
    ("clavicle", "head"), ("upperarm01", "head"), ("lowerarm01", "head"),
    ("wrist", "head"),
    ("finger1-1", "head"), ("finger1-2", "head"), ("finger1-3", "head"),
    ("finger1-3", "tail"),
]
for _n in range(2, 6):
    _ARM += [
        (f"metacarpal{_n - 1}", "head"), (f"finger{_n}-1", "head"),
        (f"finger{_n}-2", "head"), (f"finger{_n}-3", "head"), (f"finger{_n}-3", "tail"),
    ]
_LEG = [
    ("upperleg01", "head"), ("lowerleg01", "head"), ("foot", "head"),
    ("toe3-1", "head"), ("toe3-3", "tail"),
]
ARM_START = {"L": 11, "R": 39}
LEG_START = {"L": 67, "R": 72}


def landmarks() -> dict[int, tuple[str, str]]:
    """Nova index -> (MPFB bone, "head" | "tail" | "mid") for all 77 joints."""
    out = dict(_CENTRE)
    out[9] = ("eye.L", "head")
    out[10] = ("eye.R", "head")
    for side in ("L", "R"):
        for k, (bone, end) in enumerate(_ARM):
            out[ARM_START[side] + k] = (f"{bone}.{side}", end)
        for k, (bone, end) in enumerate(_LEG):
            out[LEG_START[side] + k] = (f"{bone}.{side}", end)
    assert sorted(out) == list(range(77))
    return out


# MPFB deform bone -> the Nova bone whose segment it lies in. A Nova bone runs
# from its joint to its first child, so e.g. both MPFB upper-arm twist bones
# skin to the bone that starts at the shoulder joint.
_WEIGHT_CENTRE = {
    "root": 0, "spine05": 0, "spine04": 1, "spine03": 2,
    "spine02": 3, "spine01": 3, "breast": 3,
    "neck01": 4, "neck02": 4, "neck03": 5, "head": 6, "jaw": 8,
}
_WEIGHT_ARM = {  # offset from the arm's first Nova index
    "clavicle": 0, "shoulder01": 0, "upperarm01": 1, "upperarm02": 1,
    "lowerarm01": 2, "lowerarm02": 2, "wrist": 3,
    "finger1-1": 4, "finger1-2": 5, "finger1-3": 6,
}
for _n in range(2, 6):
    _base = 8 + (_n - 2) * 5
    _WEIGHT_ARM[f"metacarpal{_n - 1}"] = _base
    for _k in range(1, 4):
        _WEIGHT_ARM[f"finger{_n}-{_k}"] = _base + _k
_WEIGHT_LEG = {
    "pelvis": None, "upperleg01": 0, "upperleg02": 0, "lowerleg01": 1,
    "lowerleg02": 1, "foot": 2, "toe": 3,
}
# Face bones that are not the jaw, the eyes or the tongue move with the head.
_JAW_FAMILY = ("jaw", "tongue", "oris", "special04", "levator0")


def weight_target(mpfb_bone: str) -> int | None:
    """Which Nova bone an MPFB vertex group's weights move to, or None to drop."""
    stem, _, side = mpfb_bone.partition(".")
    side = side.upper()
    if side in ("L", "R"):
        if stem == "eye":
            return 9 if side == "L" else 10
        if stem == "pelvis":
            return 0
        if stem in _WEIGHT_ARM:
            return ARM_START[side] + _WEIGHT_ARM[stem]
        if stem in _WEIGHT_LEG and _WEIGHT_LEG[stem] is not None:
            return LEG_START[side] + _WEIGHT_LEG[stem]
        if stem.startswith("toe"):
            return LEG_START[side] + _WEIGHT_LEG["toe"]
        if stem == "breast":
            return 3
    if stem in _WEIGHT_CENTRE:
        return _WEIGHT_CENTRE[stem]
    if any(stem.startswith(p) for p in _JAW_FAMILY):
        return 8
    # Everything else MPFB rigs is facial (orbicularis, temporalis, ...).
    return 6 if stem else None
