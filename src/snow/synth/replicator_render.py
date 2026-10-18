"""Render one synthetic clip with Omniverse Replicator, from its exported USD.

    # on a Linux machine with an RTX GPU (L40S, RTX 6000 Ada, A6000, 4090; not
    # an H100 or A100, which have no RT cores), from the Isaac Sim install:
    ./python.sh /path/to/snow/src/snow/synth/replicator_render.py \\
        data/synth/<clip> [--path-traced] [--subframes 8]

Then copy ``replicator/rgb_*.png`` back and encode with ``snow synth``'s
encoder in place of the Blender frames (see docs/synthetic.md).

Blender authors everything: the Nova-77 rider, its animation, the venue and
the broadcast camera, all in ``<clip>/scene.usdc`` (``blender/run.py``). This
script only renders it. The skeleton and the camera are the same prims the
ground truth was read from, so labels need no change between renderers, and
the camera's own parameters are also written per frame from inside
Omniverse (``camera_params``) as a cross-check on ``gt_raw.npz``.

Not run in this repository's CI or on the author's machine: it needs an RTX
GPU and an Isaac Sim install. Treat it as unverified until a render from it
has gone through ``snow synth prepare`` and matched the Blender ground truth.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("clip", type=Path)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--path-traced", action="store_true",
                        help="path tracing instead of real-time ray tracing")
    parser.add_argument("--subframes", type=int, default=8,
                        help="render subframes per frame; carries the motion blur")
    args = parser.parse_args()
    clip = args.clip.resolve()
    stage_path = clip / "scene.usdc"
    out_dir = clip / "replicator"
    if not stage_path.exists():
        sys.exit(f"missing {stage_path}; run `snow synth render` first")

    from isaacsim import SimulationApp  # noqa: E402  (must start before omni imports)

    app = SimulationApp({"headless": True, "width": args.width, "height": args.height})

    import carb.settings
    import omni.replicator.core as rep
    import omni.timeline
    import omni.usd
    from pxr import UsdGeom

    context = omni.usd.get_context()
    context.open_stage(str(stage_path))
    stage = context.get_stage()
    camera = next(p for p in stage.Traverse() if p.IsA(UsdGeom.Camera))
    start, end = int(stage.GetStartTimeCode()), int(stage.GetEndTimeCode())
    per_second = stage.GetTimeCodesPerSecond()

    settings = carb.settings.get_settings()
    if args.path_traced:
        rep.settings.set_render_pathtraced(samples_per_pixel=64)
    else:
        rep.settings.set_render_rtx_realtime()
    # Motion blur over half a frame of *scene* time, as in the Blender render:
    # the scene runs at the slow-motion rate, so this is a high-speed shutter.
    settings.set("/rtx/post/motionblur/enabled", True)
    settings.set("/rtx/post/motionblur/exposureFraction", 0.5)
    settings.set("/omni/replicator/captureMotionBlur", True)
    settings.set("/omni/replicator/pathTracingMotionBlurSubSamples", args.subframes)

    product = rep.create.render_product(str(camera.GetPath()), (args.width, args.height))
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(output_dir=str(out_dir), rgb=True, camera_params=True)
    writer.attach([product])

    timeline = omni.timeline.get_timeline_interface()
    timeline.set_time_codes_per_second(per_second)
    for frame in range(start, end + 1):
        timeline.set_current_time(frame / per_second)
        rep.orchestrator.step(rt_subframes=args.subframes, pause_timeline=True)
    rep.orchestrator.wait_until_complete()

    (out_dir / "render.json").write_text(json.dumps({
        "renderer": "omniverse-replicator",
        "path_traced": args.path_traced,
        "frames": end - start + 1,
        "camera": str(camera.GetPath()),
    }, indent=2))
    app.close()


if __name__ == "__main__":
    main()
