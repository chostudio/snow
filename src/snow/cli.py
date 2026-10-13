"""Command line entry point: snow ingest | track | pose | run | doctor."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from . import config as cfg
from . import ingest as ing
from . import pose as ps
from . import track as trk


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
            focal_length=args.focal_length,
            timeout=args.timeout,
            session_id=args.name,
        )
    except ps.PoseError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1

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


def cmd_run(args: argparse.Namespace) -> int:
    for step in (cmd_ingest, cmd_track, cmd_pose):
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

    p_run = sub.add_parser("run", help="ingest + track + pose in one go")
    add_ingest_args(p_run)
    add_track_args(p_run)
    add_pose_args(p_run)
    p_run.set_defaults(func=cmd_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ing.IngestError, trk.TrackError, ps.PoseError) as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
