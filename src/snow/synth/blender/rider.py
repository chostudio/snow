"""Build the synthetic rider in Blender: an MPFB body on a Nova-77 skeleton.

Runs inside Blender (needs bpy and the MPFB extension). The armature has
exactly the NIM's 77 joints, named ``joint<i>`` with ``skeleton.PARENTS`` as
the hierarchy (see ``nova.py``), and MPFB's own rig is used only to find
where those joints sit and to inherit its skin weights; then it is deleted.

The armature object's origin is the pelvis (joint 0), so its world transform
*is* the root pose a trick animates. Everything the trick controls besides
the root is driven through IK targets parented to the armature object, which
keeps the code independent of bone roll.
"""

from __future__ import annotations

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector

from ...skeleton import PARENTS
from .. import nova

RIDER = "Rider"
BOARD = "Board"
BODY = "RiderBody"

# Riding stance in the armature's frame: the rider faces -Y, left side +X,
# so the board runs along X. Metres.
STANCE_HALF = 0.26
BOARD_LENGTH = 1.56
BOARD_WIDTH = 0.27
ANKLE_ABOVE_BOARD = 0.09


def _mpfb():
    # Headless Blender only loads extensions enabled in saved preferences.
    # MPFB needs its preferences entry to exist, so it is enabled with
    # default_set; background Blender never saves preferences, so the user's
    # saved prefs are untouched.
    import addon_utils
    module = "bl_ext.user_default.mpfb"
    if not addon_utils.check(module)[1]:
        addon_utils.enable(module, default_set=True)
    from bl_ext.user_default.mpfb.services.humanservice import HumanService
    from bl_ext.user_default.mpfb.services.targetservice import TargetService
    return HumanService, TargetService


def _clear_scene() -> None:
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj)
    for block in (bpy.data.meshes, bpy.data.armatures, bpy.data.materials,
                  bpy.data.actions, bpy.data.cameras, bpy.data.lights):
        for item in list(block):
            if item.users == 0:
                block.remove(item)


def _landmarks(rig) -> np.ndarray:
    """World positions of all 77 Nova joints on the MPFB rig, rest pose."""
    points = np.zeros((77, 3))
    for index, (bone_name, end) in nova.landmarks().items():
        bone = rig.data.bones[bone_name]
        head = rig.matrix_world @ bone.head_local
        tail = rig.matrix_world @ bone.tail_local
        point = {"head": head, "tail": tail, "mid": (head + tail) / 2}[end]
        points[index] = tuple(point)
    return points


def _children(index: int) -> list[int]:
    return [i for i, p in enumerate(PARENTS) if p == index]


# Branching joints point their bone at the child that continues the body's
# main line; the rest point at their only child.
_MAIN_CHILD = {0: 1, 3: 4, 6: 7, 14: 24, 42: 52}


def _build_armature(points: np.ndarray):
    pelvis = points[0].copy()
    local = points - pelvis
    data = bpy.data.armatures.new(RIDER)
    rig = bpy.data.objects.new(RIDER, data)
    bpy.context.scene.collection.objects.link(rig)
    rig.location = Vector(pelvis)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")
    bones = []
    for index, parent in enumerate(PARENTS):
        bone = data.edit_bones.new(nova.bone_name(index))
        head = Vector(local[index])
        kids = _children(index)
        if kids:
            target = local[_MAIN_CHILD.get(index, kids[0])]
            tail = Vector(target)
        else:
            direction = Vector(local[index] - local[parent])
            tail = head + direction.normalized() * max(0.02, 0.5 * direction.length)
        if (tail - head).length < 1e-3:
            tail = head + Vector((0, 0, 0.02))
        bone.head, bone.tail = head, tail
        if parent >= 0:
            bone.parent = bones[parent]
            bone.use_connect = False
        bones.append(bone)
    bpy.ops.object.mode_set(mode="OBJECT")
    for index, name in enumerate(nova.JOINT_NAMES):
        data.bones[nova.bone_name(index)]["nova_name"] = name
    return rig


def _rebind(body, mpfb_rig, rig) -> None:
    """Move MPFB's skin weights onto the Nova bones, then drop MPFB's rig."""
    weights: dict[int, dict[int, float]] = {}
    groups = {g.index: g.name for g in body.vertex_groups}
    mpfb_bones = set(mpfb_rig.data.bones.keys())
    for vertex in body.data.vertices:
        for g in vertex.groups:
            name = groups[g.group]
            if name not in mpfb_bones or g.weight <= 0:
                continue
            target = nova.weight_target(name)
            if target is None:
                continue
            slot = weights.setdefault(target, {})
            slot[vertex.index] = slot.get(vertex.index, 0.0) + g.weight
    for name in list(mpfb_bones):
        if name in body.vertex_groups:
            body.vertex_groups.remove(body.vertex_groups[name])
    for target, members in weights.items():
        group = body.vertex_groups.new(name=nova.bone_name(target))
        for vertex_index, weight in members.items():
            group.add([vertex_index], min(1.0, weight), "REPLACE")

    for modifier in list(body.modifiers):
        if modifier.type == "ARMATURE":
            body.modifiers.remove(modifier)
    world = body.matrix_world.copy()
    body.parent = rig
    body.matrix_world = world
    modifier = body.modifiers.new("Nova77", "ARMATURE")
    modifier.object = rig
    # Deform first, then MPFB's mask hides the helper geometry.
    body.modifiers.move(body.modifiers.find("Nova77"), 0)
    bpy.data.objects.remove(mpfb_rig)


def _material(name: str, color, roughness=0.6, metallic=0.0, sheen=0.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    if "Sheen Weight" in bsdf.inputs:
        bsdf.inputs["Sheen Weight"].default_value = sheen
    return mat


# Outerwear by Nova bone: a snowboarder is a jacket, pants, gloves, a helmet
# and goggles, which is also what the NIM sees on broadcast footage.
_JACKET = {1, 2, 3, 4, 5, 11, 12, 13, 39, 40, 41}
_PANTS = {0, 67, 68, 72, 73}
_BOOTS = {69, 70, 71, 74, 75, 76}
_HEAD = {6, 7, 8, 9, 10}


def _dress(body, rig, rng: np.random.Generator) -> None:
    # Outerwear colours, never skin-like: a jacket that matches the skin
    # reads as a bare torso, which no broadcast rider is.
    outer = [(0.8, 0.05, 0.05), (0.02, 0.15, 0.6), (0.95, 0.7, 0.02), (0.05, 0.45, 0.2),
             (0.03, 0.03, 0.03), (0.9, 0.9, 0.9), (0.45, 0.05, 0.5), (0.95, 0.35, 0.02),
             (0.1, 0.55, 0.75), (0.3, 0.32, 0.25)]
    picks = rng.choice(len(outer), 3, replace=False)
    jacket = _material("Jacket", outer[picks[0]], roughness=0.75, sheen=0.3)
    pants = _material("Pants", outer[picks[1]], roughness=0.8, sheen=0.3)
    gloves = _material("Gloves", rng.uniform(0.0, 0.2, 3), roughness=0.7)
    boots = _material("Boots", rng.uniform(0.0, 0.15, 3), roughness=0.6)
    helmet = _material("Helmet", outer[picks[2]], roughness=0.3)
    goggles = _material("Goggles", (0.9, 0.5, 0.1), roughness=0.05, metallic=0.9)
    skin = _material("Skin", rng.choice([(0.8, 0.6, 0.5), (0.55, 0.38, 0.28),
                                         (0.33, 0.22, 0.15)]), roughness=0.5)
    mats = [jacket, pants, gloves, boots, helmet, goggles, skin]
    body.data.materials.clear()
    for mat in mats:
        body.data.materials.append(mat)

    # MPFB has its own "joint-..." helper groups; only ours are digits.
    index_of = {g.index: int(g.name[5:]) for g in body.vertex_groups
                if g.name.startswith("joint") and g.name[5:].isdigit()}
    dominant = np.full(len(body.data.vertices), -1)
    for vertex in body.data.vertices:
        best, best_w = -1, 0.0
        for g in vertex.groups:
            if g.group in index_of and g.weight > best_w:
                best, best_w = index_of[g.group], g.weight
        dominant[vertex.index] = best

    bones = rig.data.bones
    eye_z = (bones["joint9"].head_local.z + bones["joint10"].head_local.z) / 2
    to_world = rig.matrix_world
    eye_z_world = (to_world @ Vector((0, 0, eye_z))).z
    for poly in body.data.polygons:
        owners = [dominant[v] for v in poly.vertices]
        owner = max(set(owners), key=owners.count)
        centre = body.matrix_world @ poly.center
        if owner in _JACKET:
            slot = 0
        elif owner in _PANTS:
            slot = 1
        elif owner in _BOOTS:
            slot = 3
        elif owner in _HEAD:
            front = centre.y < (to_world @ bones["joint6"].head_local).y - 0.05
            if centre.z > eye_z_world + 0.035:
                slot = 4
            elif centre.z > eye_z_world - 0.03 and front:
                slot = 5
            else:
                slot = 6
        elif owner >= 0:
            slot = 2   # hands and fingers
        else:
            slot = 0
        poly.material_index = slot


def _board(rig, rng: np.random.Generator):
    """A snowboard: rounded, with upturned nose and tail, along the rig's X."""
    mesh = bpy.data.meshes.new(BOARD)
    bm = bmesh.new()
    half_l, half_w, thick = BOARD_LENGTH / 2, BOARD_WIDTH / 2, 0.014
    rings = []
    steps = 24
    for i in range(steps + 1):
        x = -half_l + BOARD_LENGTH * i / steps
        # Sidecut, rounded tips, and a kick over the last 13 cm at each end.
        edge = abs(x) / half_l
        width = half_w * (1.0 - 0.06 * np.sin(np.pi * (x / BOARD_LENGTH + 0.5)))
        if edge > 0.92:
            width *= np.sqrt(max(0.0, 1 - ((edge - 0.92) / 0.08) ** 2)) * 0.7 + 0.3
        lift = max(0.0, (abs(x) - (half_l - 0.13)) / 0.13) ** 2 * 0.06
        ring = [bm.verts.new((x, y * width, lift + z))
                for y, z in ((-1, 0), (1, 0), (1, thick), (-1, thick))]
        rings.append(ring)
    for a, b in zip(rings, rings[1:]):
        for k in range(4):
            bm.faces.new((a[k], a[(k + 1) % 4], b[(k + 1) % 4], b[k]))
    bm.faces.new(rings[0][::-1])
    bm.faces.new(rings[-1])
    bm.to_mesh(mesh)
    bm.free()
    board = bpy.data.objects.new(BOARD, mesh)
    bpy.context.scene.collection.objects.link(board)
    board.data.materials.append(_material(
        "BoardTop", rng.uniform(0.0, 1.0, 3), roughness=0.25))
    for poly in board.data.polygons:
        poly.use_smooth = True
    board.parent = rig
    # Bindings: two dark blocks the boots sit in.
    for side in (-1, 1):
        bpy.ops.mesh.primitive_cube_add(size=1.0)
        binding = bpy.context.active_object
        binding.name = f"Binding{'L' if side > 0 else 'R'}"
        binding.scale = (0.12, 0.26, 0.05)
        binding.location = (side * STANCE_HALF, 0.0, 0.035)
        binding.parent = board
        binding.data.materials.append(_material("Binding", (0.02, 0.02, 0.02)))
    return board


def _empty(name: str, parent, location=(0, 0, 0)):
    obj = bpy.data.objects.new(name, None)
    bpy.context.scene.collection.objects.link(obj)
    obj.empty_display_size = 0.05
    obj.parent = parent
    obj.location = location
    return obj


def _ik(rig, bone: str, target, pole=None, chain=2, pole_angle=0.0) -> None:
    constraint = rig.pose.bones[bone].constraints.new("IK")
    constraint.target = target
    constraint.chain_count = chain
    if pole is not None:
        constraint.pole_target = pole
        constraint.pole_angle = pole_angle


def _controls(rig, board) -> dict:
    """IK targets. Feet are pinned to the board; hands follow animated empties."""
    controls = {}
    ankle = {s: rig.data.bones[f"joint{nova.LEG_START[s] + 2}"].head_local
             for s in ("L", "R")}
    for side, sign in (("L", 1), ("R", -1)):
        start = nova.LEG_START[side]
        foot = _empty(f"Foot{side}", board, (sign * STANCE_HALF, 0.02, ANKLE_ABOVE_BOARD))
        toe = _empty(f"Toe{side}", board, (sign * (STANCE_HALF + 0.03), -0.16, 0.02))
        knee = _empty(f"Knee{side}", rig, (sign * 0.25, -1.2, -0.45))
        _ik(rig, f"joint{start + 1}", foot, knee, pole_angle=-np.pi / 2)
        track = rig.pose.bones[f"joint{start + 2}"].constraints.new("DAMPED_TRACK")
        track.target = toe
        controls[f"foot_{side}"] = foot

        arm = nova.ARM_START[side]
        hand = _empty(f"Hand{side}", rig)
        elbow = _empty(f"Elbow{side}", rig, (sign * 0.7, 0.4, 0.1))
        _ik(rig, f"joint{arm + 2}", hand, elbow, pole_angle=-np.pi / 2)
        controls[f"hand_{side}"] = hand
        controls[f"ankle_rest_{side}"] = ankle[side]
    return controls


def build(body: dict, seed: int):
    """A fresh rider: MPFB body with ``body`` macros, Nova-77 rig, board."""
    HumanService, TargetService = _mpfb()
    rng = np.random.default_rng(seed)
    _clear_scene()
    macros = TargetService.get_default_macro_info_dict()
    macros.update({k: v for k, v in body.items() if k in macros})
    mesh = HumanService.create_human(macro_detail_dict=macros, feet_on_ground=True)
    mpfb_rig = HumanService.add_builtin_rig(mesh, "default", import_weights=True)
    bpy.context.view_layer.update()
    points = _landmarks(mpfb_rig)
    rig = _build_armature(points)
    mesh.name = BODY
    _rebind(mesh, mpfb_rig, rig)
    _dress(mesh, rig, rng)
    for poly in mesh.data.polygons:
        poly.use_smooth = True
    board = _board(rig, rng)
    controls = _controls(rig, board)
    return rig, mesh, board, controls


def rest_heads(rig) -> np.ndarray:
    """Joint positions in the armature's own space at rest, (77, 3)."""
    return np.array([tuple(rig.data.bones[nova.bone_name(i)].head_local)
                     for i in range(77)])
