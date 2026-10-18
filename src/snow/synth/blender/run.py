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


def export_usd(path: Path) -> None:
    """The whole animated scene as USD, skeleton and skin included."""
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
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--no-usd", action="store_true")
    args = parser.parse_args(argv)
    clip = args.clip.resolve()

    trick, motion = load(clip)
    frames = len(motion.t)
    rng = np.random.default_rng(trick.seed + 101)
    rig, body, board, controls = rider.build(trick.body, seed=trick.seed)
    animate.key(rig, board, controls, trick, motion)
    layout = venue.build(trick, motion, rng)
    cam = venue.camera(trick, motion, rng, args.width, args.height)
    cleared = venue.clear_sightlines(cam, motion)
    venue.render_settings(args.width, args.height, args.samples, str(clip / "frames"))

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
