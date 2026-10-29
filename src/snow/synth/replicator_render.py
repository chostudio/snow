"""Render synthetic clips with Omniverse Replicator, from their exported USD.

    # inside the Isaac Sim container on a Linux machine with an RTX GPU (L40S,
    # L4, RTX 6000 Ada, A6000, 4090; not an H100 or A100, which have no RT
    # cores); see docs/synthetic.md for the docker command:
    ./python.sh /snow/src/snow/synth/replicator_render.py \\
        /snow/data/synth/<clip> [<clip> ...] [--blur 4] [--subframes 2]

Then ``snow synth encode --renderer replicator --only <clip>`` encodes
``replicator/rgb_*.png`` in place of the Blender frames.

Blender authors everything: the Nova-77 rider, its animation, the venue and
the broadcast camera, all in ``<clip>/scene.usdc`` (``blender/run.py``). This
script only renders it. The skeleton and the camera are the same prims the
ground truth was read from, so labels need no change between renderers.
The focal Omniverse actually rendered with is written per frame
(``replicator/camera.npz``, from Replicator's ``camera_params``) and matches
``gt_raw.npz`` exactly. That annotator's ``cameraViewTransform`` is not
written: in Isaac Sim 6.2 it stays at the stage's last frame while the image
itself follows the animated camera, so it is no check on the pose.

Run on Isaac Sim 6.2.0 (``nvcr.io/nvidia/isaac-sim:6.2.0``) on an L4
(``g6.xlarge``). On an A10G (``g5.xlarge``) every start, even an empty
stage, died with ``VkResult: ERROR_DEVICE_LOST`` and Xid 109 (CTX SWITCH
TIMEOUT) on drivers 595 and 580, open and proprietary kernel modules alike.

Motion blur is accumulated, not post-processed: each output frame averages
``--blur`` renders spread over the half-frame shutter the Blender render
uses (scene frame ``f`` exposes ``f - 0.25`` to ``f + 0.25``), averaged in
linear light. Every sample is a full RTX render of the posed stage at that
instant, so the rider, the board and the panning camera all smear exactly
as they moved.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


SKY_BLUE = (0.55, 0.70, 0.95)


def _broadcast_look(stage, sun: float, sky: float) -> None:
    """Daylight the way a broadcast sees a venue: hard sun, blue sky, snow.

    Blender's lights arrive in Blender's units (a sun of ~1.2, a grey world
    of 0.35), which RTX reads as near darkness. Only the lights
    change here: nothing that moves, and nothing the labels read.
    Edits go to the session layer, so ``scene.usdc`` is never rewritten.
    """
    from pxr import Gf, Sdf, Usd, UsdLux

    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for prim in stage.Traverse():
            if prim.IsA(UsdLux.DistantLight):
                light = UsdLux.DistantLight(prim)
                light.GetIntensityAttr().Set(sun)
                light.GetAngleAttr().Set(0.53)
            elif prim.IsA(UsdLux.DomeLight):
                dome = UsdLux.DomeLight(prim)
                dome.GetTextureFileAttr().Set(Sdf.AssetPath(""))
                dome.GetColorAttr().Set(Gf.Vec3f(*SKY_BLUE))
                dome.GetIntensityAttr().Set(sky)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("clips", type=Path, nargs="+")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--path-traced", action="store_true",
                        help="path tracing instead of real-time ray tracing")
    parser.add_argument("--spp", type=int, default=32, help="path-traced samples per pixel")
    parser.add_argument("--subframes", type=int, default=2,
                        help="RTX subframes per sample, so lighting settles after a move")
    parser.add_argument("--blur", type=int, default=4,
                        help="renders averaged across the shutter per output frame")
    parser.add_argument("--shutter", type=float, default=0.5,
                        help="open shutter, in frames of scene time (Blender's 0.5)")
    parser.add_argument("--frames", type=str, default=None,
                        help="first:last (inclusive) to render a slice, for a test")
    parser.add_argument("--force", action="store_true", help="re-render finished clips")
    parser.add_argument("--sun", type=float, default=3000.0, help="sun intensity (RTX units)")
    parser.add_argument("--sky", type=float, default=800.0, help="sky dome intensity")
    args = parser.parse_args()

    from isaacsim import SimulationApp  # noqa: E402  (must start before omni imports)

    app = SimulationApp({"headless": True, "width": args.width, "height": args.height,
                         "renderer": "PathTracing" if args.path_traced else "RaytracedLighting"})

    import carb.settings
    import numpy as np
    import omni.replicator.core as rep
    import omni.timeline
    import omni.usd
    from PIL import Image
    from pxr import Usd, UsdGeom

    settings = carb.settings.get_settings()
    if args.path_traced:
        settings.set("/rtx/rendermode", "PathTracing")
        settings.set("/rtx/pathtracing/spp", args.spp)
        settings.set("/rtx/pathtracing/totalSpp", args.spp)
    else:
        settings.set("/rtx/rendermode", "RaytracedLighting")
    # Accumulated blur replaces the post-process one, and nothing else may
    # carry one frame's image into the next (a slow pan would ghost).
    settings.set("/rtx/post/motionblur/enabled", False)
    settings.set("/rtx/ambientOcclusion/enabled", True)
    settings.set("/rtx/directLighting/sampledLighting/enabled", True)
    # Fixed exposure: auto exposure would adapt between the blur samples.
    settings.set("/rtx/post/histogram/enabled", False)
    settings.set("/rtx/post/tonemap/op", 6)   # ACES, close to Blender's AgX

    timeline = omni.timeline.get_timeline_interface()

    def srgb_to_linear(x):
        return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

    def linear_to_srgb(x):
        x = np.clip(x, 0.0, 1.0)
        return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)

    for clip in [c.resolve() for c in args.clips]:
        stage_path = clip / "scene.usdc"
        out_dir = clip / "replicator"
        if not stage_path.exists():
            print(f"skip {clip.name}: no scene.usdc", flush=True)
            continue
        if (out_dir / "render.json").exists() and not args.force and args.frames is None:
            print(f"skip {clip.name}: rendered", flush=True)
            continue
        out_dir.mkdir(exist_ok=True)
        t_open = time.monotonic()
        omni.usd.get_context().open_stage(str(stage_path))
        for _ in range(20):
            app.update()
        stage = omni.usd.get_context().get_stage()
        _broadcast_look(stage, args.sun, args.sky)
        camera = next(p for p in stage.Traverse() if p.IsA(UsdGeom.Camera))
        start, end = int(stage.GetStartTimeCode()), int(stage.GetEndTimeCode())
        if args.frames:
            a, b = (int(v) for v in args.frames.split(":"))
            start, end = max(start, a), min(end, b)
        per_second = stage.GetTimeCodesPerSecond()
        timeline.set_time_codes_per_second(per_second)
        timeline.set_start_time(stage.GetStartTimeCode() / per_second)
        timeline.set_end_time(stage.GetEndTimeCode() / per_second)
        timeline.stop()

        product = rep.create.render_product(str(camera.GetPath()), (args.width, args.height))
        rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb.attach([product])
        cam_params = rep.AnnotatorRegistry.get_annotator("camera_params")
        cam_params.attach([product])

        def shoot(code: float):
            timeline.set_current_time(code / per_second)
            # The first steps after attaching return an empty buffer while
            # the render product warms up; step again until pixels arrive.
            for _ in range(30):
                rep.orchestrator.step(rt_subframes=args.subframes, delta_time=0.0,
                                      pause_timeline=True)
                data = np.asarray(rgb.get_data())
                if data.size:
                    break
            else:
                raise RuntimeError(f"{clip.name}: no pixels from the rgb annotator")
            image = data[..., :3].astype(np.float64) / 255.0
            return image, cam_params.get_data()

        offsets = [0.0] if args.blur <= 1 else list(
            np.linspace(-args.shutter / 2, args.shutter / 2, args.blur + 1)[:-1]
            + args.shutter / (2 * args.blur))
        t_render = time.monotonic()
        focals = []
        for frame in range(start, end + 1):
            acc = None
            for k, off in enumerate(offsets):
                image, params = shoot(frame + off)
                lin = srgb_to_linear(image)
                acc = lin if acc is None else acc + lin
                if k == len(offsets) // 2 or len(offsets) == 1:
                    centre_params = params
            # The focal at the frame's own instant, for the cross-check.
            if len(offsets) > 1:
                _, centre_params = shoot(float(frame))
            out = (linear_to_srgb(acc / len(offsets)) * 255.0 + 0.5).astype(np.uint8)
            Image.fromarray(out).save(out_dir / f"rgb_{frame:04d}.png")
            focals.append(float(centre_params["cameraFocalLength"])
                          / float(centre_params["cameraAperture"][0]) * args.width)
            if frame % 25 == 0:
                print(f"{clip.name} frame {frame}/{end} "
                      f"{(time.monotonic() - t_render) / (frame - start + 1):.2f}s/frame",
                      flush=True)
        rep.orchestrator.wait_until_complete()
        np.savez_compressed(out_dir / "camera.npz", focal_px=np.array(focals), start=start)
        took = time.monotonic() - t_render
        info = {
            "renderer": "isaac-replicator",
            "mode": "path-traced" if args.path_traced else "rtx-realtime",
            "frames": end - start + 1,
            "blur_samples": len(offsets),
            "rt_subframes": args.subframes,
            "camera": str(camera.GetPath()),
            "open_s": round(t_render - t_open, 1),
            "render_s": round(took, 1),
            "s_per_frame": round(took / (end - start + 1), 3),
        }
        if args.frames is None:
            (out_dir / "render.json").write_text(json.dumps(info, indent=2))
        print(f"done {clip.name}: {json.dumps(info)}", flush=True)
        rgb.detach()
        cam_params.detach()
        product.destroy()

    app.close()


if __name__ == "__main__":
    sys.exit(main())
