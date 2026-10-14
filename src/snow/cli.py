"""Command line entry point: snow ingest | track | pose | run | doctor."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from . import camera as cam
from . import config as cfg
from . import crop as cp
from . import features as feat
from . import ingest as ing
from . import meta as mt
from . import pose as ps
from . import scout as sct
from . import track as trk
from . import usd as usdx


def _print_info(info: ing.VideoInfo) -> None:
    print(
        f"  {info.path.name}: {info.width}x{info.height} {info.codec}/{info.pix_fmt} "
        f"{info.fps:.3f}fps {info.frames} frames {info.duration:.1f}s "
        f"{info.size_mb:.1f}MB"
    )


def cmd_doctor(args: argparse.Namespace) -> int:
    settings = cfg.settings()
    ok = True
    print("tools")
    for tool in ("ffmpeg", "ffprobe", "yt-dlp"):
        path = shutil.which(tool)
        print(f"  {tool:9s} {path or 'MISSING'}")
        ok &= bool(path)

    print("python packages")
    for mod in ("cv2", "grpc", "numpy", "ultralytics"):
        try:
            __import__(mod)
            print(f"  {mod:11s} ok")
        except Exception as exc:  # noqa: BLE001
            print(f"  {mod:11s} MISSING ({exc})")
            ok = False

    print("vendored client")
    exists = cfg.VENDOR_CLIENT.exists()
    print(f"  {cfg.VENDOR_CLIENT} {'ok' if exists else 'MISSING — run scripts/setup.sh'}")
    ok &= exists

    print("endpoint")
    print(f"  target      {settings.target}")
    print(f"  function-id {settings.function_id}")
    print(f"  mode        {'hosted (NVCF preview)' if settings.preview else 'self-hosted'}")
    key = settings.api_key
    print(f"  api key     {'set (' + key[:9] + '…)' if key else 'MISSING — see .env.example'}")
    ok &= bool(key) or not settings.preview

    print("\n" + ("all good" if ok else "problems found"))
    return 0 if ok else 1


def cmd_ingest(args: argparse.Namespace) -> int:
    cfg.ensure_dirs()
    dest = cfg.clip_path(args.name)
    source = args.source

    if source.startswith(("http://", "https://", "www.")):
        print(f"downloading {source}")
        src = ing.download(source, cfg.CLIPS_DIR / args.name)
    else:
        src = Path(source).expanduser().resolve()
        if not src.exists():
            print(f"no such file: {src}", file=sys.stderr)
            return 1
    print(f"source: {src.name}")
    _print_info(ing.probe(src))

    print(f"transcoding -> {dest.name} (H.264 / yuv420p / {args.fps}fps CFR)")
    info = ing.transcode(
        src, dest,
        fps=args.fps, start=args.start, duration=args.duration,
        height=args.height, crf=args.crf,
    )
    _print_info(info)
    cfr_ok, detail = ing.frame_intervals_are_constant(dest)
    print(f"  cfr check: {detail}")

    clip = mt.ClipMeta(
        name=args.name, source=source, start=args.start, duration=args.duration,
        fps=info.fps, frames=info.frames, width=info.width, height=info.height,
        slowmo_factor=args.slowmo_factor,
    )
    clip.save()
    if clip.slowmo_factor != 1.0:
        print(
            f"  slow motion {clip.slowmo_factor}x -> real {clip.real_fps:.1f} fps "
            "(rates are corrected, totals are unaffected)"
        )

    if args.keep_source:
        print(f"  source kept at {src}")
    elif src.parent == cfg.CLIPS_DIR and ".src." in src.name:
        src.unlink()
    return 0


def cmd_track(args: argparse.Namespace) -> int:
    cfg.ensure_dirs()
    video = cfg.clip_path(args.name)
    if not video.exists():
        print(f"no clip at {video} — run `snow ingest` first", file=sys.stderr)
        return 1
    dest = cfg.bbox_path(args.name)
    preview = cfg.CLIPS_DIR / f"{args.name}.track.mp4" if args.preview else None
    cache = cfg.CLIPS_DIR / f"{args.name}.tracks.json"
    merge_groups = None
    if args.merge:
        merge_groups = [[int(i) for i in g.split(",")] for g in args.merge]

    print(f"tracking people in {video.name} ({args.weights})")
    report = trk.build(
        video, dest,
        max_bodies=args.max_bodies,
        keep=args.track_ids,
        max_gap=args.max_gap,
        preview=preview,
        cache=cache,
        retrack=args.retrack,
        merge_groups=merge_groups,
        weights=args.weights,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
    )
    print(f"  {report['candidates']} candidate track(s), kept {len(report['selected'])}")
    for sel in report["selected"]:
        span = sel["span"]
        print(
            f"    id {sel['tracking_id']} (raw {sel['raw_id']}): "
            f"{sel['frames_present']}/{report['frames']} frames "
            f"({sel['coverage']:.0%}), frames {span[0]}-{span[1]}, "
            f"median box {sel['median_area_px']}px^2"
        )
    if report["interpolated_boxes"]:
        print(f"  interpolated {report['interpolated_boxes']} box(es) across short gaps")
    if args.show_candidates:
        print("  top candidates (raw id: frames, span, median box):")
        for c in report["top_candidates"]:
            print(
                f"    {c['raw_id']:>4}: {c['frames_present']:>4} frames, "
                f"{c['span'][0]}-{c['span'][1]}, {c['median_area_px']}px^2"
            )
    print(f"  wrote {dest}")
    if preview:
        print(f"  preview {preview}")
    return 0


def cmd_pose(args: argparse.Namespace) -> int:
    cfg.ensure_dirs()
    settings = cfg.settings()
    video = cfg.clip_path(args.name)
    bbox = cfg.bbox_path(args.name)
    out = cfg.pose_path(args.name)
    focal = args.focal_length

    zoom = cfg.CLIPS_DIR / f"{args.name}.zoom.mp4"
    zoom_bbox = cfg.CLIPS_DIR / f"{args.name}.zoom.bbox.txt"
    used_zoom = (
        zoom.exists() and zoom_bbox.exists() and not getattr(args, "no_zoom", False)
    )
    if used_zoom:
        video, bbox = zoom, zoom_bbox
        if not focal:
            plan = cp.load_plan(cfg.CLIPS_DIR / f"{args.name}.crop.json")
            focal = plan["focal_length"]
        print(
            f"using the cropped clip ({zoom.name}), focal length {focal:.0f}px"
        )

    for path, step in ((video, "ingest"), (bbox, "track")):
        if not path.exists():
            print(f"missing {path} — run `snow {step}` first", file=sys.stderr)
            return 1

    info = ing.probe(video)
    if settings.preview and info.size_bytes > cfg.HOSTED_MAX_BYTES:
        print(
            f"{video.name} is {info.size_mb:.1f}MB, over the hosted 50MB limit",
            file=sys.stderr,
        )
        return 1

    overlay = cfg.POSES_DIR / f"{args.name}.overlay.mp4" if args.overlay else None
    print(f"sending {video.name} to {settings.target}")
    try:
        result = ps.estimate(
            settings, video, bbox, out,
            overlay=overlay,
            focal_length=focal,
            timeout=args.timeout,
            session_id=args.name,
        )
    except ps.PoseError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1

    # Record which input produced these poses. Applying the crop's
    # principal-point correction to poses taken from the full frame would be
    # worse than not correcting at all.
    try:
        clip = mt.load(args.name)
        clip.extra["pose_input"] = "zoom" if used_zoom else "full"
        clip.extra["pose_focal_length"] = focal
        clip.save()
    except FileNotFoundError:
        pass

    summary = ps.summarize(result)
    print(f"\nwrote {result}")
    print(
        f"  {summary['frames_with_pose']} frames with pose, "
        f"{summary['detections']} detections, joints per body: {summary['joints']}"
    )
    print(f"  frame range {summary['frame_range'][0]}-{summary['frame_range'][1]}")
    for tid, count in sorted(summary["per_track"].items()):
        print(f"    track {tid}: {count} frames")
    if overlay:
        print(f"  overlay {overlay}")
    return 0


def cmd_scout(args: argparse.Namespace) -> int:
    cfg.ensure_dirs()
    source = args.source
    if source.startswith(("http://", "https://", "www.")):
        name = args.name or "scout"
        video = cfg.CLIPS_DIR / f"{name}.scout.mkv"
        if video.exists() and not args.refetch:
            print(f"reusing {video.name} (--refetch to download again)")
        else:
            print(f"downloading {source}")
            fetched = ing.download(source, cfg.CLIPS_DIR / f"{name}.scout")
            fetched.replace(video)
    else:
        video = Path(source).expanduser().resolve()
        if not video.exists():
            print(f"no such file: {video}", file=sys.stderr)
            return 1

    print(f"scanning {video.name}")
    shots, stats = sct.analyze(
        video,
        sample_every=args.sample_every,
        cut_threshold=args.cut_threshold,
        weights=args.weights,
        device=args.device,
    )
    print()
    print(sct.report(shots, stats, args.top))
    dest = cfg.CLIPS_DIR / f"{video.stem}.shots.json"
    sct.save(shots, stats, dest)
    print(f"\nfull results: {dest}")
    return 0


def cmd_crop(args: argparse.Namespace) -> int:
    cfg.ensure_dirs()
    video = cfg.clip_path(args.name)
    bbox = cfg.bbox_path(args.name)
    for path, step in ((video, "ingest"), (bbox, "track")):
        if not path.exists():
            print(f"missing {path} — run `snow {step}` first", file=sys.stderr)
            return 1
    clip = mt.load(args.name)

    plan = cp.plan(
        video, bbox,
        fill=args.fill, out_height=args.out_height,
        smooth_window=args.smooth, min_zoom=args.min_zoom, pan=args.pan,
    )
    zoom = cfg.CLIPS_DIR / f"{args.name}.zoom.mp4"
    zoom_bbox = cfg.CLIPS_DIR / f"{args.name}.zoom.bbox.txt"
    plan_path = cfg.CLIPS_DIR / f"{args.name}.crop.json"

    offset = plan.principal_offset(0)
    print(
        f"cropping {plan.crop_w}x{plan.crop_h} -> {plan.out_w}x{plan.out_h} "
        f"({plan.scale:.2f}x scale, {'panning' if plan.panned else 'static'})"
    )
    print(
        f"  principal point offset {offset[0]:+.0f},{offset[1]:+.0f}px "
        + (
            "— varies per frame, which biases rotation; features corrects it"
            if plan.panned
            else "— constant, so frame-to-frame rotation is unaffected"
        )
    )
    print(
        f"  focal length {plan.source_focal:.0f}px -> {plan.focal_length:.0f}px "
        "(passed explicitly; the server would otherwise assume the crop is a "
        "whole camera view)"
    )
    cp.render(video, zoom, plan, fps=clip.fps)
    kept = cp.remap_bbox(bbox, zoom_bbox, plan)
    cp.save_plan(plan, plan_path)

    info = ing.probe(zoom)
    _print_info(info)
    ing.verify(info)
    print(f"  {kept} boxes remapped into crop coordinates")

    clip.crop = {
        "out_w": plan.out_w, "out_h": plan.out_h,
        "crop_w": plan.crop_w, "crop_h": plan.crop_h,
        "scale": plan.scale, "focal_length": plan.focal_length,
        "plan": str(plan_path),
    }
    clip.save()
    print(f"  wrote {zoom.name}, {zoom_bbox.name}")
    print("  `snow pose` will use these automatically (--no-zoom to skip)")
    return 0


def cmd_features(args: argparse.Namespace) -> int:
    cfg.ensure_dirs()
    poses = cfg.pose_path(args.name)
    if not poses.exists():
        print(f"missing {poses} — run `snow pose` first", file=sys.stderr)
        return 1
    clip = mt.load(args.name)
    video = cfg.clip_path(args.name)

    track = None
    if not args.no_camera:
        cache = cfg.POSES_DIR / f"{args.name}.camera.npz"
        if cache.exists() and not args.recompute_camera:
            track = cam.CameraTrack.load(cache)
            print(f"reusing camera track from {cache.name}")
        else:
            print(f"estimating camera motion from {video.name}")
            _, boxes = cp.read_bbox_file(cfg.bbox_path(args.name))
            merged = {}
            for per_track in boxes.values():
                merged.update(per_track)
            track = cam.estimate(video, boxes=merged)
            track.save(cache)
        info = cam.summarize(track)
        print(f"  {info['segments']} camera segment(s), focal {info['focal']:.0f}px")
        for detail in info["detail"][:6]:
            print(
                f"    seg {detail['segment']}: frames "
                f"{detail['start_frame']}-{detail['end_frame']}, camera turned "
                f"{detail['camera_rotation_deg']:.0f}deg, "
                f"{detail['median_inliers']} inliers"
            )

    crop_plan = None
    plan_path = cfg.CLIPS_DIR / f"{args.name}.crop.json"
    pose_input = clip.extra.get("pose_input")
    if plan_path.exists() and (clip.crop or {}).get("plan") and pose_input == "zoom":
        crop_plan = cp.load_plan(plan_path)
        if crop_plan.get("panned"):
            print("  correcting the panning crop's principal point offset")
        else:
            crop_plan = None
    elif plan_path.exists() and pose_input == "full":
        print("  crop plan ignored: these poses came from the full frame")

    df = feat.build(poses, clip, camera=track, tracking_id=args.track_id,
                    min_quality=args.min_quality, crop_plan=crop_plan)
    csv = cfg.POSES_DIR / f"{args.name}.features.csv"
    parquet = cfg.POSES_DIR / f"{args.name}.features.parquet"
    df.to_csv(csv, index=False)
    try:
        df.to_parquet(parquet, index=False)
    except Exception as exc:  # noqa: BLE001
        parquet = None
        print(f"  (parquet skipped: {exc})")

    summary = feat.summarize(df, clip)
    print(f"\n{summary['tracked_frames']}/{summary['frames']} frames with a body frame")
    print(f"  real fps {summary['real_fps']:.1f} (slow motion {summary['slowmo_factor']}x)")
    for seg in summary["segments"]:
        print(
            f"  segment {seg['segment']}: {seg['tracked_frames']} frames | "
            f"axis spin {seg['axis_spin_deg']:+.0f}deg "
            f"(axis {seg['spin_axis_tilt_deg']:.0f}deg off vertical) | "
            f"vertical yaw {seg['net_spin_deg']:+.0f}deg | "
            f"total path {seg['total_rotation_deg']:.0f}deg | "
            f"max tilt {seg['max_tilt_deg']:.0f}deg | "
            f"camera {seg['camera_rotation_deg']:.0f}deg"
        )
    (cfg.POSES_DIR / f"{args.name}.features.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(f"\nwrote {csv}" + (f" and {parquet.name}" if parquet else ""))
    return 0


def cmd_usd(args: argparse.Namespace) -> int:
    cfg.ensure_dirs()
    poses = cfg.pose_path(args.name)
    if not poses.exists():
        print(f"missing {poses} — run `snow pose` first", file=sys.stderr)
        return 1
    clip = mt.load(args.name)
    dest = cfg.POSES_DIR / f"{args.name}.usda"
    report = usdx.export(poses, dest, clip, tracking_id=args.track_id,
                         real_time=args.real_time)
    print(f"wrote {report['path']}")
    print(
        f"  {report['joints']} joints, {report['frames']} frames "
        f"({report['frame_range'][0]}-{report['frame_range'][1]}) at "
        f"{report['fps']:g} fps, {report['size_mb']}MB"
    )
    print("  open in Omniverse / usdview, or import into Blender")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    steps = [cmd_ingest, cmd_track]
    if getattr(args, "zoom", False):
        steps.append(cmd_crop)
    steps.append(cmd_pose)
    if getattr(args, "features", False):
        steps.append(cmd_features)
    if getattr(args, "usd", False):
        steps.append(cmd_usd)
    for step in steps:
        code = step(args)
        if code != 0:
            return code
        print()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="snow",
        description="3D body-pose extraction from snowboarding video "
        "via the NVIDIA 3D Body Pose NIM.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="check tools, packages and credentials").set_defaults(
        func=cmd_doctor
    )

    def add_ingest_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("source", help="YouTube URL or a local video file")
        p.add_argument("name", help="short clip name, used for every output file")
        p.add_argument("--start", help="seek before cutting, e.g. 00:01:23.5")
        p.add_argument("--duration", help="clip length, e.g. 25 or 00:00:25")
        p.add_argument("--fps", type=float, default=30.0, help="output frame rate (CFR)")
        p.add_argument("--height", type=int, default=1080, help="cap height, 0 to keep")
        p.add_argument("--crf", type=int, default=23, help="x264 quality, lower is better")
        p.add_argument("--keep-source", action="store_true", help="keep the download")
        p.add_argument("--slowmo-factor", type=float, default=1.0,
                       help="playback is this many times slower than reality "
                            "(4.0 = a 120fps camera conformed to 30fps)")

    def add_track_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--max-bodies", type=int, default=1,
                       help="how many people to keep (the rider alone by default)")
        p.add_argument("--track-ids", type=int, nargs="+",
                       help="keep these raw track ids instead of auto-selecting")
        p.add_argument("--max-gap", type=int, default=8,
                       help="interpolate detection dropouts up to this many frames")
        p.add_argument("--weights", default="yolo11s.pt", help="YOLO weights")
        p.add_argument("--conf", type=float, default=0.25, help="detection confidence")
        p.add_argument("--imgsz", type=int, default=960, help="inference image size")
        p.add_argument("--device", help="cpu, mps or a cuda index")
        p.add_argument("--preview", action="store_true",
                       help="render a video with the kept boxes drawn")
        p.add_argument("--merge", action="append", metavar="ID,ID,...",
                       help="stitch these raw track ids into one body; repeat "
                            "the flag for a second body")
        p.add_argument("--retrack", action="store_true",
                       help="re-run detection instead of reusing the cache")
        p.add_argument("--show-candidates", action="store_true",
                       help="list the strongest raw tracks to pick from")

    def add_pose_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--overlay", action="store_true",
                       help="also render the skeleton overlay video")
        p.add_argument("--focal-length", type=float, default=0.0,
                       help="pinhole focal length in pixels, 0 derives it from frame size")
        p.add_argument("--timeout", type=float, default=3600.0, help="RPC deadline")
        p.add_argument("--no-zoom", action="store_true",
                       help="ignore a cropped clip even if `snow crop` made one")

    p_ingest = sub.add_parser("ingest", help="download and re-encode for the NIM")
    add_ingest_args(p_ingest)
    p_ingest.set_defaults(func=cmd_ingest)

    p_track = sub.add_parser("track", help="detect and track the rider")
    p_track.add_argument("name")
    add_track_args(p_track)
    p_track.set_defaults(func=cmd_track)

    p_pose = sub.add_parser("pose", help="run 3D pose estimation on the NIM")
    p_pose.add_argument("name")
    add_pose_args(p_pose)
    p_pose.set_defaults(func=cmd_pose)

    p_scout = sub.add_parser(
        "scout", help="find slow-motion / close-up shots worth analysing"
    )
    p_scout.add_argument("source", help="YouTube URL or a local video file")
    p_scout.add_argument("--name", help="name for the cached download")
    p_scout.add_argument("--top", type=int, default=12, help="how many shots to list")
    p_scout.add_argument("--sample-every", type=int, default=15,
                         help="run the detector on every Nth frame")
    p_scout.add_argument("--cut-threshold", type=float, default=22.0,
                         help="frame difference that counts as a hard cut")
    p_scout.add_argument("--weights", default="yolo11s-pose.pt",
                         help="must be a POSE model: scouting needs keypoints "
                              "to tell a rider from a face close-up")
    p_scout.add_argument("--device")
    p_scout.add_argument("--refetch", action="store_true")
    p_scout.set_defaults(func=cmd_scout)

    p_crop = sub.add_parser(
        "crop", help="crop and zoom onto the rider before pose estimation"
    )
    p_crop.add_argument("name")
    p_crop.add_argument("--fill", type=float, default=0.5,
                        help="fraction of the output height the rider should span")
    p_crop.add_argument("--out-height", type=int, default=0,
                        help="output height; 0 keeps the crop at native pixel "
                             "scale (no resampling, no detail lost)")
    p_crop.add_argument("--smooth", type=int, default=15,
                        help="frames of moving average on the pan")
    p_crop.add_argument("--pan", action="store_true",
                        help="let the window follow the rider. Zooms tighter "
                             "on travelling shots, but moves the principal "
                             "point every frame")
    p_crop.add_argument("--min-zoom", type=float, default=1.0,
                        help="refuse to crop if the rider would end up scaled "
                             "below this (1.0 = never lose pixels)")
    p_crop.set_defaults(func=cmd_crop)

    p_feat = sub.add_parser(
        "features", help="camera-compensated rotation features for DS/ML"
    )
    p_feat.add_argument("name")
    p_feat.add_argument("--track-id", type=int, default=1)
    p_feat.add_argument("--min-quality", type=float, default=0.35,
                        help="below this the hip axis is too close to vertical "
                             "to read yaw from")
    p_feat.add_argument("--no-camera", action="store_true",
                        help="skip camera compensation (rotations stay in "
                             "camera coordinates and will be wrong)")
    p_feat.add_argument("--recompute-camera", action="store_true")
    p_feat.set_defaults(func=cmd_features)

    p_usd = sub.add_parser("usd", help="export a USD skeletal animation for Omniverse")
    p_usd.add_argument("name")
    p_usd.add_argument("--track-id", type=int, default=1)
    p_usd.add_argument("--real-time", action="store_true",
                       help="author the timeline at real speed rather than "
                            "the clip's playback speed")
    p_usd.set_defaults(func=cmd_usd)

    p_run = sub.add_parser("run", help="ingest + track + pose in one go")
    add_ingest_args(p_run)
    add_track_args(p_run)
    add_pose_args(p_run)
    p_run.add_argument("--zoom", action="store_true",
                       help="also run the crop stage before pose estimation")
    p_run.add_argument("--features", action="store_true",
                       help="also extract rotation features afterwards")
    p_run.add_argument("--usd", action="store_true",
                       help="also export a USD animation afterwards")
    p_run.add_argument("--fill", type=float, default=0.5)
    p_run.add_argument("--out-height", type=int, default=0)
    p_run.add_argument("--smooth", type=int, default=15)
    p_run.add_argument("--min-zoom", type=float, default=1.0)
    p_run.add_argument("--pan", action="store_true")
    p_run.add_argument("--track-id", type=int, default=1)
    p_run.add_argument("--min-quality", type=float, default=0.35)
    p_run.add_argument("--no-camera", action="store_true")
    p_run.add_argument("--recompute-camera", action="store_true")
    p_run.add_argument("--real-time", action="store_true")
    p_run.set_defaults(func=cmd_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ing.IngestError, trk.TrackError, ps.PoseError, cp.CropError) as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
