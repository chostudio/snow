"""Key a ``tricks.Motion`` onto the rider, and read the exact skeleton back.

Runs inside Blender. The root pose goes on the armature object; tuck moves
the board toward the hips (the leg IK folds the knees); grabs move a hand's
IK target onto the board; wind-up and reaching are spine rotations. After
the constraints are evaluated, every joint's world position and orientation
is read back per frame: that, not the parameters, is the ground truth.
"""

from __future__ import annotations

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

from .. import nova
from .rider import BOARD_LENGTH, BOARD_WIDTH

# Pelvis to the board's base: legs nearly straight at tuck 0, knees to the
# chest at tuck 1.
DROP_STANDING = 0.88
DROP_TUCKED = 0.34
SPINE = (1, 2, 3)

# Board-frame grab points. The board runs along the rig's X with the rider's
# left at +X; toes point -Y, so -Y is the toe edge and +Y the heel edge.
_EDGE = {
    "indy": ("rear", (0.0, -BOARD_WIDTH / 2, 0.02)),
    "mute": ("front", (0.0, -BOARD_WIDTH / 2, 0.02)),
    "melon": ("front", (0.0, BOARD_WIDTH / 2, 0.02)),
    "stalefish": ("rear", (0.0, BOARD_WIDTH / 2, 0.02)),
    "nose": ("front", (0.34 * BOARD_LENGTH, 0.0, 0.05)),
    "tail": ("rear", (-0.34 * BOARD_LENGTH, 0.0, 0.05)),
}


def _hands(stance: str) -> dict[str, str]:
    """Which hand is 'front': the one on the nose side, left for regular."""
    front = "L" if stance == "regular" else "R"
    return {"front": front, "rear": "R" if front == "L" else "L"}


def _grab_point(grab: str, stance: str) -> tuple[str, Vector]:
    which, (x, y, z) = _EDGE[grab]
    # The nose is on the front foot's side: +X for regular, -X for goofy.
    nose_sign = 1.0 if stance == "regular" else -1.0
    return _hands(stance)[which], Vector((x * nose_sign, y, z))


def _local_rotation(rig, index: int, armature_rotation: Matrix) -> Quaternion:
    """An armature-space rotation as bone ``index``'s local pose rotation."""
    rest = rig.data.bones[nova.bone_name(index)].matrix_local.to_3x3()
    return (rest.inverted() @ armature_rotation @ rest).to_quaternion()


def key(rig, board, controls: dict, trick, motion) -> None:
    scene = bpy.context.scene
    scene.frame_start = 0
    scene.frame_end = len(motion.t) - 1
    hand_rest = {"L": Vector((0.45, -0.12, 0.05)), "R": Vector((-0.45, -0.12, 0.05))}
    grab_hand, grab_point = (None, None)
    if trick.grab != "none":
        grab_hand, grab_point = _grab_point(trick.grab, trick.stance)
    reach_side = 0.0
    if grab_point is not None:
        reach_side = float(np.sign(grab_point.x)) if abs(grab_point.x) > 0.1 else 0.0

    for index in SPINE:
        rig.pose.bones[nova.bone_name(index)].rotation_mode = "QUATERNION"

    for frame in range(len(motion.t)):
        rotation = Matrix(motion.rotation[frame].tolist())
        rig.matrix_world = Matrix.Translation(Vector(motion.position[frame])) @ rotation.to_4x4()
        rig.keyframe_insert("location", frame=frame)
        rig.rotation_mode = "QUATERNION"
        rig.rotation_quaternion = rotation.to_quaternion()
        rig.keyframe_insert("rotation_quaternion", frame=frame)

        grab = float(motion.grab[frame])
        # A grab brings the board up to the hand: knees to the chest.
        tuck = max(float(motion.tuck[frame]), grab)
        drop = DROP_STANDING + (DROP_TUCKED - DROP_STANDING) * tuck
        board.location = (0.0, 0.0, -drop)
        board.keyframe_insert("location", frame=frame)

        for side in ("L", "R"):
            target = hand_rest[side].copy()
            target.z -= 0.25 * tuck
            if side == grab_hand:
                on_board = board.location + grab_point
                target = target.lerp(on_board, grab)
            controls[f"hand_{side}"].location = target
            controls[f"hand_{side}"].keyframe_insert("location", frame=frame)

        # Fold forward with the tuck and further to reach a grab, lean toward
        # a nose or tail grab, and twist the shoulders for the wind-up.
        flex = np.deg2rad(12.0 * tuck + 40.0 * grab) / len(SPINE)
        lean = np.deg2rad(18.0 * grab * reach_side) / len(SPINE)
        twist = np.deg2rad(float(motion.wind[frame])) / len(SPINE)
        bend = (Matrix.Rotation(twist, 3, "Z") @ Matrix.Rotation(lean, 3, "Y")
                @ Matrix.Rotation(flex, 3, "X"))
        for index in SPINE:
            bone = rig.pose.bones[nova.bone_name(index)]
            bone.rotation_quaternion = _local_rotation(rig, index, bend)
            bone.keyframe_insert("rotation_quaternion", frame=frame)


def sample(rig, frames: int) -> dict[str, np.ndarray]:
    """Evaluated joints per frame: world positions and world orientations.

    Orientations are each bone's world rotation *relative to its rest*, so
    identity everywhere means the rig is at rest; that is the convention
    UsdSkel uses, and ``groundtruth.py`` turns it into the NIM's schema.
    """
    scene = bpy.context.scene
    positions = np.zeros((frames, 77, 3))
    orientations = np.zeros((frames, 77, 3, 3))
    rest = [rig.data.bones[nova.bone_name(i)].matrix_local.to_3x3() for i in range(77)]
    for frame in range(frames):
        scene.frame_set(frame)
        world = rig.matrix_world
        world_rot = world.to_3x3().normalized()
        for i in range(77):
            pose_bone = rig.pose.bones[nova.bone_name(i)]
            positions[frame, i] = tuple(world @ pose_bone.head)
            current = world_rot @ pose_bone.matrix.to_3x3().normalized()
            orientations[frame, i] = np.array(current @ rest[i].inverted())
    return {"positions": positions, "orientations": orientations}
