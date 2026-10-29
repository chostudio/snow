"""Build, key, record and render one synthetic trick. Runs inside Blender.

    blender -b --python src/snow/synth/blender/run.py -- data/synth/<name> [--no-render]

Not ``--factory-startup``: MPFB is an extension enabled in the user's prefs.

Reads ``trick.json`` and ``motion.npz`` from the clip directory (written by
``snow synth plan``) and writes next to them:

* ``gt_raw.npz``: every joint's world position and orientation, the rest
  pose, and the camera's exact world->camera rotation and focal per frame
* ``scene.blend``: the whole scene, to inspect
* ``scene.usdc``: the same scene as USD, the input to the Replicator render
* ``frames/frame_####.png``: the render, unless ``--no-render``
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import bpy
import numpy as np

SRC = Path(__file__).resolve().parents[3]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from snow.synth.blender import animate, rider, venue  # noqa: E402

OUTPUT_FPS = 30


def load(clip: Path):
    trick = types.SimpleNamespace(**json.loads((clip / "trick.json").read_text()))
    trick.render_fps = OUTPUT_FPS * trick.slowmo
    z = np.load(clip / "motion.npz")
    motion = types.SimpleNamespace(**{k: z[k] for k in z.files})
    return trick, motion


def _flat_colours() -> list:
    """Unlink procedural base colours, leaving each material's mean colour.

    Blender's USD exporter writes ``UsdPreviewSurface`` and drops procedural
    nodes (the venue's checker, brick, wave and noise textures), leaving a
    linked base colour as plain 0.8 grey: a Replicator render of the stage
    then shows a white venue. The mean of the two colours keeps the venue's
    palette, though not its pattern. Returns what to re-link afterwards, so
    the Blender render itself is untouched.
    """
    undo = []
    for mat in bpy.data.materials:
        if not mat.use_nodes:
            continue
        bsdf = next((n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
        if bsdf is None or not bsdf.inputs["Base Color"].is_linked:
            continue
        link = bsdf.inputs["Base Color"].links[0]
        source = link.from_node
        if source.type == "VALTORGB":
            colours = [np.array(e.color[:3]) for e in source.color_ramp.elements]
        elif source.type in ("TEX_CHECKER", "TEX_BRICK"):
            colours = [np.array(source.inputs[k].default_value[:3]) for k in ("Color1", "Color2")]
        else:
            continue
        old = tuple(bsdf.inputs["Base Color"].default_value)
        undo.append((mat, link.from_socket, old))
        mat.node_tree.links.remove(link)
        bsdf.inputs["Base Color"].default_value = (*np.mean(colours, axis=0), 1.0)
    return undo


def _relink(undo: list) -> None:
    for mat, socket, old in undo:
        bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
        bsdf.inputs["Base Color"].default_value = old
        mat.node_tree.links.new(socket, bsdf.inputs["Base Color"])


def _drop_masked_helpers() -> int:
    """Delete the vertices MPFB's Mask modifier hides, before a skinned export.

    The MakeHuman base mesh carries helper geometry (a skirt, tights, joint
    markers) that a Mask modifier hides. Blender exports a skinned mesh
    without its modifiers, so in USD the helpers come back and a Replicator
    render shows the rider in a skirt. Deleting exactly the masked vertices
    changes nothing Blender renders: they were never visible. Returns how
    many were removed.
    """
    import bmesh

    removed = 0
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        for mod in [m for m in obj.modifiers if m.type == "MASK" and m.mode == "VERTEX_GROUP"]:
            group = obj.vertex_groups.get(mod.vertex_group)
            if group is None:
                continue
            keep = set()
            for v in obj.data.vertices:
                if any(g.group == group.index and g.weight > mod.threshold for g in v.groups):
                    keep.add(v.index)
            bm = bmesh.new()
            bm.from_mesh(obj.data)
            bm.verts.ensure_lookup_table()
            drop = [v for v in bm.verts if (v.index in keep) == mod.invert_vertex_group]
            bmesh.ops.delete(bm, geom=drop, context="VERTS")
            bm.to_mesh(obj.data)
            bm.free()
            removed += len(drop)
    return removed


def export_usd(path: Path) -> None:
    """The whole animated scene as USD, skeleton and skin included.

    Procedural colours are flattened for the export only (``_flat_colours``),
    and the body's masked helper vertices are deleted first
    (``_drop_masked_helpers``; the scene is already saved and the render is
    unchanged).
    """
    print(f"snow-synth: dropped {_drop_masked_helpers()} masked helper vertices")
    undo = _flat_colours()
    try:
        _export(path)
    finally:
        _relink(undo)


def _export(path: Path) -> None:
    options = {
        "filepath": str(path),
        "export_animation": True,
        "export_armatures": True,
        "export_materials": True,
        "export_cameras": True,
        "export_lights": True,
    }
    known = set(bpy.ops.wm.usd_export.get_rna_type().properties.keys())
    bpy.ops.wm.usd_export(**{k: v for k, v in options.items() if k in known})


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("clip", type=Path)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--engine", choices=["eevee", "cycles"], default="eevee")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--no-usd", action="store_true")
    args = parser.parse_args(argv)
    clip = args.clip.resolve()

    trick, motion = load(clip)
    frames = len(motion.t)
    # Clips in one split group share a venue and a rider; older tricks have
    # no group seeds and use their own.
    venue_seed = getattr(trick, "venue_seed", None)
    rider_seed = getattr(trick, "rider_seed", None)
    rng = np.random.default_rng((trick.seed if venue_seed is None else venue_seed) + 101)
    rig, body, board, controls = rider.build(
        trick.body, seed=trick.seed if rider_seed is None else rider_seed)
    animate.key(rig, board, controls, trick, motion)
    layout = venue.build(trick, motion, rng)
    cam = venue.camera(trick, motion, rng, args.width, args.height)
    cleared = venue.clear_sightlines(cam, motion)
    venue.render_settings(args.width, args.height, args.samples, str(clip / "frames"),
                          engine=args.engine)

    joints = animate.sample(rig, frames)
    truth = venue.camera_truth(cam, frames, args.width)
    seen = venue.visibility(cam, rig, frames)
    np.savez_compressed(
        clip / "gt_raw.npz",
        positions=joints["positions"], orientations=joints["orientations"],
        rest=rider.rest_heads(rig),
        world_to_camera=truth["world_to_camera"], camera_centre=truth["centre"],
        focal_px=truth["focal_px"], width=args.width, height=args.height,
        visible=seen,
    )
    (clip / "layout.json").write_text(json.dumps(layout, indent=2))
    bpy.ops.wm.save_as_mainfile(filepath=str(clip / "scene.blend"))
    if not args.no_usd:
        export_usd(clip / "scene.usdc")
    print(f"snow-synth: {clip.name} recorded {frames} frames, "
          f"focal {truth['focal_px'].min():.0f}-{truth['focal_px'].max():.0f}px, "
          f"rider visible {seen.mean():.0%}, {cleared} sightline obstruction(s) cleared")
    if not args.no_render:
        (clip / "frames").mkdir(exist_ok=True)
        bpy.ops.render.render(animation=True)
        print(f"snow-synth: {clip.name} rendered")


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
