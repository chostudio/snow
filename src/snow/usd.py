"""Export poses as a USD skeletal animation for Omniverse.

Written as text ``.usda`` on purpose: it needs no ``pxr`` install, and the
result opens directly in Omniverse, usdview or Blender's USD importer.

The mapping is a close fit, because the NIM already returns what
``UsdSkelAnimation`` wants — local joint rotations relative to a rest pose:

* ``rest_pose`` (root-relative joint positions) becomes the skeleton's bind
  and rest transforms; ``rest_pose[0]`` is the origin, as verified in the data.
* ``joint_rotations`` becomes the animation's per-frame ``rotations``. Joint 0
  is the root, and the data confirms ``joint_rotations[0]`` is exactly
  ``root_pose.rotation``, so the root is not transformed twice.
* ``root_pose.translation`` drives the SkelRoot's own xform — standard root
  motion, and far smaller than authoring 77 translations per frame.

Coordinates are converted from the NIM's camera frame (X right, **Y down**,
Z forward — verified against the data) to USD's Y-up right-handed frame by
``(x, -y, -z)``, which is a 180 degree turn about X. The same turn applied to
a quaternion ``(x, y, z, w)`` gives ``(x, -y, -z, w)``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .meta import ClipMeta
from .skeleton import PARENTS


def joint_paths() -> list[str]:
    """USD joint tokens: each joint's full path through the hierarchy."""
    paths: list[str] = []
    for index, parent in enumerate(PARENTS):
        name = f"joint{index}"
        paths.append(name if parent < 0 else f"{paths[parent]}/{name}")
    return paths


def to_usd_point(point) -> tuple[float, float, float]:
    x, y, z = point
    return (float(x), float(-y), float(-z))


def to_usd_quat(quat) -> tuple[float, float, float, float]:
    """(x, y, z, w) in camera frame -> (w, x, y, z) in USD frame."""
    x, y, z, w = quat
    return (float(w), float(x), float(-y), float(-z))


def _matrix(translation) -> str:
    """A translation-only matrix4d. USD is row-vector, so translation is row 3."""
    x, y, z = translation
    return (
        "( (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), "
        f"({x:.6f}, {y:.6f}, {z:.6f}, 1) )"
    )


def _fmt3(values) -> str:
    return f"({values[0]:.6f}, {values[1]:.6f}, {values[2]:.6f})"


def _fmt4(values) -> str:
    return (
        f"({values[0]:.6f}, {values[1]:.6f}, {values[2]:.6f}, {values[3]:.6f})"
    )


def export(
    poses: Path,
    dest: Path,
    meta: ClipMeta,
    tracking_id: int = 1,
    real_time: bool = False,
) -> dict:
    """Write a .usda for one tracked body. Returns a short report."""
    frames: dict[int, dict] = {}
    rest: np.ndarray | None = None
    with poses.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            for det in record.get("detections", []):
                if det["tracking_id"] != tracking_id:
                    continue
                frames[record["frame_id"]] = det
                if rest is None:
                    rest = np.asarray(det["rest_pose"], dtype=float)
    if not frames or rest is None:
        raise ValueError(f"{poses.name}: no detections for tracking id {tracking_id}")

    paths = joint_paths()
    count = len(paths)
    if rest.shape[0] != count:
        raise ValueError(f"expected {count} joints in rest_pose, got {rest.shape[0]}")

    # Bind transforms are skeleton-space; rest transforms are parent-relative.
    rest_usd = np.array([to_usd_point(p) for p in rest])
    bind = [_matrix(p) for p in rest_usd]
    local = []
    for index, parent in enumerate(PARENTS):
        offset = rest_usd[index] - (rest_usd[parent] if parent >= 0 else np.zeros(3))
        local.append(_matrix(offset))

    ordered = sorted(frames)
    # The timeline matches the video by default. A slow-motion clip can be
    # played at its true speed instead, which is what physical analysis wants.
    fps = meta.real_fps if real_time else (meta.fps or 30.0)

    joints_token = ", ".join(f'"{p}"' for p in paths)
    lines = [
        "#usda 1.0",
        "(",
        '    defaultPrim = "Rider"',
        "    metersPerUnit = 1",
        '    upAxis = "Y"',
        f"    timeCodesPerSecond = {fps:g}",
        f"    startTimeCode = {ordered[0]}",
        f"    endTimeCode = {ordered[-1]}",
        f'    doc = "snow: NVIDIA 3D Body Pose, clip {meta.name}, track {tracking_id}"',
        ")",
        "",
        'def SkelRoot "Rider" (',
        '    prepend apiSchemas = ["SkelBindingAPI"]',
        ")",
        "{",
        "    uniform token[] xformOpOrder = [\"xformOp:translate\"]",
        "    double3 xformOp:translate.timeSamples = {",
    ]
    for frame_id in ordered:
        point = to_usd_point(frames[frame_id]["root_pose"]["translation"])
        lines.append(f"        {frame_id}: {_fmt3(point)},")
    lines += [
        "    }",
        "",
        '    def Skeleton "Skel" (',
        '        prepend apiSchemas = ["SkelBindingAPI"]',
        "    )",
        "    {",
        f"        uniform token[] joints = [{joints_token}]",
        f"        uniform matrix4d[] bindTransforms = [{', '.join(bind)}]",
        f"        uniform matrix4d[] restTransforms = [{', '.join(local)}]",
        "        rel skel:animationSource = </Rider/Skel/Anim>",
        "",
        '        def SkelAnimation "Anim"',
        "        {",
        f"            uniform token[] joints = [{joints_token}]",
    ]

    # Translations and scales are constant; only rotations animate.
    offsets = ", ".join(
        _fmt3(rest_usd[i] - (rest_usd[p] if p >= 0 else np.zeros(3)))
        for i, p in enumerate(PARENTS)
    )
    lines.append(f"            float3[] translations = [{offsets}]")
    lines.append(
        "            half3[] scales = ["
        + ", ".join(["(1, 1, 1)"] * count)
        + "]"
    )
    lines.append("            quatf[] rotations.timeSamples = {")
    for frame_id in ordered:
        quats = frames[frame_id]["joint_rotations"]
        rendered = ", ".join(_fmt4(to_usd_quat(q)) for q in quats)
        lines.append(f"                {frame_id}: [{rendered}],")
    lines += ["            }", "        }", "    }", "}", ""]

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines))
    return {
        "path": str(dest),
        "joints": count,
        "frames": len(ordered),
        "frame_range": (ordered[0], ordered[-1]),
        "fps": fps,
        "size_mb": round(dest.stat().st_size / 1024 / 1024, 2),
    }
