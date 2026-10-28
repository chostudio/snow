"""A big-air venue and a broadcast camera, built around one trick's path.

Runs inside Blender. The kicker, knuckle and landing are placed from the
trick's own board path, so the board meets the snow exactly at takeoff and
landing whatever the trick's speed and air time.

The background is deliberately busy (banners, towers, a crowd, groomed snow,
mountains): ``camera.py`` measures the camera's rotation from background
features, and an empty white slope would starve it the way no real venue
does.

The camera stands still and pans, tilts and zooms to follow the rider with
an operator's lag, the model ``camera.py`` assumes. Its exact orientation and
focal length are recorded every frame as ground truth.
"""

from __future__ import annotations

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector

from .animate import DROP_STANDING, DROP_TUCKED

SENSOR_WIDTH_MM = 36.0
RIDER_HEIGHT_M = 1.75


def _mat(name, color, roughness=0.6, emission=0.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    if emission:
        bsdf.inputs["Emission Color"].default_value = (*color, 1.0)
        bsdf.inputs["Emission Strength"].default_value = emission
    return mat


def _textured(name, colors, kind: str, scale: float):
    """Two-colour procedural material: enough texture to track features on."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = next(n for n in nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Roughness"].default_value = 0.55
    coords = nodes.new("ShaderNodeTexCoord")
    tex = nodes.new({"checker": "ShaderNodeTexChecker", "brick": "ShaderNodeTexBrick",
                     "wave": "ShaderNodeTexWave", "noise": "ShaderNodeTexNoise"}[kind])
    if kind in ("checker", "brick"):
        tex.inputs["Color1"].default_value = (*colors[0], 1)
        tex.inputs["Color2"].default_value = (*colors[1], 1)
        tex.inputs["Scale"].default_value = scale
        links.new(coords.outputs["Object"], tex.inputs["Vector"])
        links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    else:
        tex.inputs["Scale"].default_value = scale
        ramp = nodes.new("ShaderNodeValToRGB")
        ramp.color_ramp.elements[0].color = (*colors[0], 1)
        ramp.color_ramp.elements[1].color = (*colors[1], 1)
        links.new(coords.outputs["Object"], tex.inputs["Vector"])
        links.new(tex.outputs["Fac"], ramp.inputs["Fac"])
        links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    return mat


def _snow_material():
    """Groomed snow: corduroy lines and blotches, low contrast like the real thing."""
    mat = bpy.data.materials.new("Snow")
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = next(n for n in nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Roughness"].default_value = 0.45
    coords = nodes.new("ShaderNodeTexCoord")
    wave = nodes.new("ShaderNodeTexWave")
    wave.inputs["Scale"].default_value = 0.9
    wave.inputs["Distortion"].default_value = 2.0
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 0.35
    noise.inputs["Detail"].default_value = 10.0
    mix = nodes.new("ShaderNodeMath")
    mix.operation = "MULTIPLY_ADD"
    links.new(coords.outputs["Object"], wave.inputs["Vector"])
    links.new(coords.outputs["Object"], noise.inputs["Vector"])
    links.new(wave.outputs["Fac"], mix.inputs[0])
    mix.inputs[1].default_value = 0.12
    links.new(noise.outputs["Fac"], mix.inputs[2])
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.5, 0.56, 0.68, 1)
    ramp.color_ramp.elements[1].color = (0.97, 0.98, 1.0, 1)
    links.new(mix.outputs[0], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.25
    links.new(mix.outputs[0], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def _strip(name, rows: list[list[tuple]], material):
    """A mesh from rows of points (each row a cross-section), quads between."""
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    grid = [[bm.verts.new(p) for p in row] for row in rows]
    for a, b in zip(grid, grid[1:]):
        for k in range(len(a) - 1):
            bm.faces.new((a[k], a[k + 1], b[k + 1], b[k]))
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.data.materials.append(material)
    return obj


def _box(name, location, size, material, rotation_z=0.0):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = size
    obj.rotation_euler = (0.0, 0.0, rotation_z)
    obj.data.materials.append(material)
    return obj


def board_path(trick, motion) -> np.ndarray:
    """World position of the board's base each frame (mirrors animate.key)."""
    tuck = np.maximum(motion.tuck, motion.grab)
    drop = DROP_STANDING + (DROP_TUCKED - DROP_STANDING) * tuck
    down = -motion.rotation[:, :, 2]
    return motion.position + down * drop[:, None]


def build(trick, motion, rng: np.random.Generator) -> dict:
    """Kicker, deck, landing, outrun, dressing, sky. Returns key heights."""
    path = board_path(trick, motion)
    lip = path[int(motion.takeoff_frame)]
    touch = path[int(motion.landing_frame)]
    snow = _snow_material()
    kick = np.deg2rad(trick.kicker_deg)
    land = np.deg2rad(trick.landing_deg)

    deck_z = lip[2] - 6.5
    ramp_len = (lip[2] - deck_z) / np.sin(kick)
    ramp_start = lip[1] - ramp_len * np.cos(kick)
    # The landing is the plane through the touchdown point at the landing's
    # angle; the knuckle is where it meets the deck.
    knuckle_y = touch[1] - (deck_z - touch[2]) / np.tan(land)

    def landing_z(y):
        return touch[2] - np.tan(land) * (y - touch[1])

    bottom_z = touch[2] - 40.0 * np.sin(land)
    bottom_y = touch[1] + 40.0 * np.cos(land)
    half = 3.5

    # Kicker: a straight ramp ending at the lip, with vertical side walls.
    _strip("Kicker", [
        [(-half, ramp_start, deck_z), (half, ramp_start, deck_z)],
        [(-half, lip[1], lip[2]), (half, lip[1], lip[2])],
    ], snow)
    for side in (-1, 1):
        _strip(f"KickerWall{side}", [
            [(side * half, ramp_start, deck_z), (side * half, lip[1], deck_z)],
            [(side * half, ramp_start, deck_z), (side * half, lip[1], lip[2])],
        ], snow)
    _strip("KickerFace", [
        [(-half, lip[1], deck_z), (half, lip[1], deck_z)],
        [(-half, lip[1], lip[2]), (half, lip[1], lip[2])],
    ], snow)

    wide = 40.0
    # The in-run deck, the knuckle, the landing and the outrun, one surface.
    profile = [
        (ramp_start - 60.0, deck_z), (knuckle_y - 1.5, deck_z),
        (knuckle_y + 1.5, landing_z(knuckle_y + 1.5)),
        (bottom_y, bottom_z), (bottom_y + 15.0, bottom_z - 3.0),
        (bottom_y + 120.0, bottom_z - 3.0),
    ]
    _strip("Hill", [[(x, y, z) for x in np.linspace(-wide, wide, 9)]
                    for y, z in profile], snow)
    _strip("Valley", [[(-600, -400, bottom_z - 6), (600, -400, bottom_z - 6)],
                      [(-600, 700, bottom_z - 6), (600, 700, bottom_z - 6)]], snow)

    # Banners along the landing and the deck, every few metres both sides.
    palette = [(0.9, 0.1, 0.1), (0.05, 0.2, 0.8), (0.95, 0.75, 0.05),
               (0.05, 0.6, 0.25), (0.1, 0.1, 0.1), (0.95, 0.95, 0.95)]
    kinds = ("checker", "brick", "wave", "noise")
    for i, y in enumerate(np.arange(ramp_start - 25.0, bottom_y + 30.0, 4.5)):
        z = np.interp(y, [p[0] for p in profile], [p[1] for p in profile])
        for side in (-1, 1):
            c = rng.choice(len(palette), 2, replace=False)
            mat = _textured(f"Banner{i}{side}", [palette[c[0]], palette[c[1]]],
                            kinds[int(rng.integers(len(kinds)))], float(rng.uniform(2, 8)))
            _box(f"Banner{i}_{side}", (side * (half + 4.5 + rng.uniform(0, 1.5)), y, z + 0.6),
                 (0.08, 3.6, 1.1), mat)

    # Scaffold towers, flags, a crowd at the bottom: depth and texture.
    grey = _textured("Scaffold", [(0.25, 0.25, 0.28), (0.7, 0.7, 0.72)], "brick", 3.0)
    for k in range(10):
        side = 1 if k % 2 else -1
        y = float(rng.uniform(ramp_start - 30.0, bottom_y + 40.0))
        z = np.interp(y, [p[0] for p in profile], [p[1] for p in profile])
        height = float(rng.uniform(6, 16))
        _box(f"Tower{k}", (side * float(rng.uniform(14, 34)), y, z + height / 2),
             (float(rng.uniform(2, 4)), float(rng.uniform(2, 4)), height), grey)
    crowd = [_mat(f"Crowd{k}", tuple(rng.uniform(0.05, 0.95, 3)), 0.8) for k in range(8)]
    for k in range(350):
        x = float(rng.uniform(-30, 30))
        y = float(bottom_y + rng.uniform(18, 40))
        _box(f"Fan{k}", (x, y, bottom_z - 3.0 + 0.8), (0.45, 0.35, 1.6),
             crowd[k % len(crowd)], float(rng.uniform(0, 3.14)))

    # Conifers off both sides of the course and down the valley.
    needles = _textured("Tree", [(0.03, 0.12, 0.05), (0.1, 0.25, 0.12)], "noise", 2.0)
    for k in range(220):
        x = float(rng.choice([-1, 1]) * rng.uniform(46, 260))
        y = float(rng.uniform(ramp_start - 120.0, bottom_y + 250.0))
        h = float(rng.uniform(6, 16))
        z = np.interp(y, [p[0] for p in profile], [p[1] for p in profile]) if abs(x) < wide \
            else bottom_z - 6
        bpy.ops.mesh.primitive_cone_add(vertices=6, radius1=h * 0.3, depth=h,
                                        location=(x, y, z + h / 2))
        tree = bpy.context.active_object
        tree.name = f"Tree{k}"
        tree.data.materials.append(needles)

    rock = _textured("Mountain", [(0.22, 0.21, 0.22), (0.95, 0.96, 0.98)], "noise", 0.06)
    for k in range(9):
        angle = float(rng.uniform(0, 2 * np.pi))
        dist = float(rng.uniform(350, 650))
        bpy.ops.mesh.primitive_cone_add(
            vertices=7, radius1=float(rng.uniform(120, 260)), depth=float(rng.uniform(120, 260)),
            location=(dist * np.cos(angle), dist * np.sin(angle), bottom_z + 40.0))
        cone = bpy.context.active_object
        cone.name = f"Mountain{k}"
        cone.rotation_euler = (0, 0, float(rng.uniform(0, 3.14)))
        cone.data.materials.append(rock)

    _sky(rng)
    return {"deck_z": deck_z, "bottom_z": bottom_z, "lip": lip.tolist(),
            "touch": touch.tolist(), "knuckle_y": knuckle_y}


def _sky(rng: np.random.Generator) -> None:
    world = bpy.context.scene.world or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    nodes, links = world.node_tree.nodes, world.node_tree.links
    background = next(n for n in nodes if n.type == "BACKGROUND")
    sky = nodes.new("ShaderNodeTexSky")
    try:
        sky.sky_type = "NISHITA"
        sky.sun_elevation = float(np.deg2rad(rng.uniform(15, 50)))
        sky.sun_rotation = float(rng.uniform(0, 2 * np.pi))
        sky.altitude = 2000.0
    except (TypeError, AttributeError):
        pass
    links.new(sky.outputs["Color"], background.inputs["Color"])
    background.inputs["Strength"].default_value = 0.35
    sun = bpy.data.lights.new("Sun", "SUN")
    sun.energy = float(rng.uniform(3.0, 5.0))
    sun.angle = np.deg2rad(1.0)
    sun_obj = bpy.data.objects.new("Sun", sun)
    bpy.context.scene.collection.objects.link(sun_obj)
    sun_obj.rotation_euler = (float(np.deg2rad(rng.uniform(35, 70))), 0.0,
                              float(rng.uniform(0, 2 * np.pi)))


# --- the broadcast camera ---------------------------------------------------

def _look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Blender camera rotation (looks down -Z, +Y up) aimed at ``target``, no roll."""
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return np.column_stack([right, up, -forward])


def camera(trick, motion, rng: np.random.Generator, width: int, height: int):
    """A fixed broadcast camera that pans, tilts and zooms after the rider."""
    params = trick.camera
    takeoff, landing = int(motion.takeoff_frame), int(motion.landing_frame)
    focus = motion.position[(takeoff + landing) // 2]
    azimuth = np.deg2rad(params["azimuth_deg"])
    eye = focus + np.array([
        params["distance_m"] * np.cos(azimuth),
        params["distance_m"] * np.sin(azimuth),
        0.0,
    ])
    eye[2] = motion.position[takeoff][2] + params["height_m"]

    data = bpy.data.cameras.new("Broadcast")
    data.sensor_fit = "HORIZONTAL"
    data.sensor_width = SENSOR_WIDTH_MM
    data.clip_end = 3000.0
    cam = bpy.data.objects.new("Broadcast", data)
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    cam.location = Vector(eye)
    cam.rotation_mode = "QUATERNION"

    dt = 1.0 / trick.render_fps
    follow = 1.0 - np.exp(-dt / params["lag_s"])
    zoom_follow = 1.0 - np.exp(-dt / params["zoom_lag_s"])
    sensor_height = SENSOR_WIDTH_MM * height / width
    aim = motion.position[0].copy()
    lens = None
    wobble = np.zeros(2)
    lenses = np.zeros(len(motion.t))
    velocity = np.gradient(motion.position, dt, axis=0)
    for frame in range(len(motion.t)):
        # Operators lead a rider they can see coming: aim where the rider
        # will be one reaction time from now, then lag behind that. What is
        # left is error from the rider's acceleration and the operator's lag.
        lead = motion.position[frame] + velocity[frame] * params["lag_s"]
        aim += follow * (lead - aim)
        distance = float(np.linalg.norm(motion.position[frame] - eye))
        wanted = params["frame_fill"] * sensor_height * distance / RIDER_HEIGHT_M
        lens = wanted if lens is None else lens + zoom_follow * (wanted - lens)
        # A fluid head's small wobble: a slow random walk in pan and tilt.
        wobble = 0.97 * wobble + rng.normal(0.0, params["jitter_deg"], 2)
        rotation = _look_at(eye, aim)
        rotation = (Matrix.Rotation(np.deg2rad(wobble[0]), 3, "Z").to_3x3()
                    @ Matrix(rotation.tolist())
                    @ Matrix.Rotation(np.deg2rad(wobble[1]), 3, "X").to_3x3())
        cam.rotation_quaternion = rotation.to_quaternion()
        cam.keyframe_insert("rotation_quaternion", frame=frame)
        data.lens = float(np.clip(lens, 12.0, 1200.0))
        data.keyframe_insert("lens", frame=frame)
        lenses[frame] = data.lens
    return cam


_DRESSING = ("Tower", "Banner", "Fan", "Tree")


def clear_sightlines(cam, motion, step: int = 5) -> int:
    """Remove dressing standing between the camera and the rider.

    A broadcast position is chosen so nothing blocks the shot; random
    placement does not know that. Only dressing is removed: if the hill
    itself hides the rider, that is recorded by ``visibility``, not hidden.
    """
    eye = Vector(cam.location)
    targets = [Vector(p) for p in motion.position[::step]]
    removed = 0
    for obj in list(bpy.context.scene.objects):
        if not obj.name.startswith(_DRESSING):
            continue
        corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
        lo = Vector([min(c[i] for c in corners) - 0.5 for i in range(3)])
        hi = Vector([max(c[i] for c in corners) + 0.5 for i in range(3)])
        for target in targets:
            if _segment_hits_box(eye, target, lo, hi):
                bpy.data.objects.remove(obj)
                removed += 1
                break
    return removed


def _segment_hits_box(a: Vector, b: Vector, lo: Vector, hi: Vector) -> bool:
    """Slab test: does the segment a->b pass through the axis-aligned box?"""
    t0, t1 = 0.0, 1.0
    for i in range(3):
        d = b[i] - a[i]
        if abs(d) < 1e-9:
            if a[i] < lo[i] or a[i] > hi[i]:
                return False
            continue
        u0, u1 = (lo[i] - a[i]) / d, (hi[i] - a[i]) / d
        t0, t1 = max(t0, min(u0, u1)), min(t1, max(u0, u1))
        if t0 > t1:
            return False
    return True


def visibility(cam, rig, frames: int) -> np.ndarray:
    """Per frame, is the pelvis in line of sight from the camera?"""
    scene = bpy.context.scene
    rider_parts = {rig.name} | {c.name for c in rig.children_recursive}
    seen = np.zeros(frames, dtype=bool)
    for frame in range(frames):
        scene.frame_set(frame)
        depsgraph = bpy.context.evaluated_depsgraph_get()
        eye = cam.matrix_world.translation
        target = rig.matrix_world.translation
        direction = target - eye
        hit, _, _, _, obj, _ = scene.ray_cast(depsgraph, eye, direction.normalized(),
                                               distance=direction.length + 0.5)
        seen[frame] = (not hit) or obj.name in rider_parts
    return seen


def camera_truth(cam, frames: int, width: int) -> dict[str, np.ndarray]:
    """World->camera rotation in the NIM's OpenCV frame, and focal in pixels."""
    scene = bpy.context.scene
    flip = np.diag([1.0, -1.0, -1.0])   # Blender camera axes -> X right, Y down, Z forward
    rotations = np.zeros((frames, 3, 3))
    centres = np.zeros((frames, 3))
    focal = np.zeros(frames)
    for frame in range(frames):
        scene.frame_set(frame)
        world = np.array(cam.matrix_world)
        rotations[frame] = flip @ world[:3, :3].T
        centres[frame] = world[:3, 3]
        focal[frame] = cam.data.lens / cam.data.sensor_width * width
    return {"world_to_camera": rotations, "centre": centres, "focal_px": focal}


def _cycles_gpu(samples: int) -> None:
    """Cycles on the GPU (OptiX, else CUDA), denoised, few bounces.

    For headless Linux boxes, where EEVEE has no GPU context and falls back
    to software GL. Few samples plus the denoiser look close to EEVEE's
    plain materials; what matters to the NIM is the silhouette and blur.
    """
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for kind in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = kind
        except TypeError:
            continue
        prefs.get_devices()
        devices = [d for d in prefs.devices if d.type == kind]
        if devices:
            for d in prefs.devices:
                d.use = d.type == kind
            scene.cycles.device = "GPU"
            break
    scene.cycles.samples = samples
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.use_denoising = True
    # The default denoiser (OpenImageDenoise) runs on the CPU here: 10 s a
    # frame on a 4-core box, against 0.3 s for OptiX's on the GPU.
    if scene.cycles.device == "GPU" and prefs.compute_device_type == "OPTIX":
        scene.cycles.denoiser = "OPTIX"
    scene.cycles.max_bounces = 4
    scene.cycles.diffuse_bounces = 2
    scene.cycles.glossy_bounces = 2
    scene.cycles.transmission_bounces = 2
    scene.cycles.transparent_max_bounces = 4
    scene.render.use_persistent_data = True


def render_settings(width: int, height: int, samples: int, out_dir: str,
                    engine: str = "eevee") -> None:
    scene = bpy.context.scene
    if engine == "cycles":
        _cycles_gpu(samples)
    else:
        try:
            scene.render.engine = "BLENDER_EEVEE_NEXT"
        except TypeError:
            scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.fps = 30
    scene.render.use_motion_blur = True
    # Shutter is in frames of *scene* time; the scene runs at the slow-motion
    # rate, so a half-frame shutter is 1/(60 k) s of real exposure, which is
    # what a high-speed broadcast camera gives.
    scene.render.motion_blur_shutter = 0.5
    if hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = samples
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = out_dir.rstrip("/") + "/frame_"
    for transform in ("AgX", "Filmic"):
        try:
            scene.view_settings.view_transform = transform
            break
        except TypeError:
            continue
