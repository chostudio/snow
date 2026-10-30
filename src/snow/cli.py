"""Command line entry point: snow ingest | track | pose | run | classify | label | doctor | coach | serve."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import camera as cam
from . import config as cfg
from . import crop as cp
from . import features as feat
from . import harvest as harv
from . import ingest as ing
from . import meta as mt
from . import pose as ps
from . import scout as sct
from . import track as trk
from . import usd as usdx
from .coach import CoachError


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
        # Resolve exactly as ingest does, or doctor reports a binary that is
        # not the one the pipeline will run — the venv's pinned yt-dlp beats
        # a stale Homebrew copy on PATH, and that difference is the whole
        # point of asking.
        try:
            path = ing._require(tool)
        except ing.IngestError:
            path = None
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
        auto_stitch=not args.no_stitch,
        recover=not args.no_recover,
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
    if report["stitched_ids"]:
        print(f"  stitched raw track(s) {report['stitched_ids']} into the rider")
    if report["recovered_boxes"]:
        print(f"  recovered {report['recovered_boxes']} box(es) by rotated search")
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


def _camera_track(args: argparse.Namespace) -> "cam.CameraTrack | None":
    """The clip's camera track: cached in <name>.camera.npz, estimated if absent."""
    if getattr(args, "no_camera", False):
        return None
    video = cfg.clip_path(args.name)
    cache = cfg.POSES_DIR / f"{args.name}.camera.npz"
    if cache.exists() and not getattr(args, "recompute_camera", False):
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
    return track


def cmd_features(args: argparse.Namespace) -> int:
    cfg.ensure_dirs()
    poses = cfg.pose_path(args.name)
    if not poses.exists():
        print(f"missing {poses} — run `snow pose` first", file=sys.stderr)
        return 1
    clip = mt.load(args.name)
    track = _camera_track(args)

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
                    min_quality=args.min_quality, crop_plan=crop_plan,
                    segmenter=getattr(args, "segmenter", "auto"))
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
    air = summary["airborne"]
    print(f"  airborne: {air['frames']} frames in {len(air['airs'])} air(s), from {air['source']}")
    print(f"  real fps {summary['real_fps']:.1f} (slow motion {summary['slowmo_factor']}x)")
    for seg in summary["segments"]:
        # A segment too short to have a dominant axis reports its tilt as None.
        tilt = seg["spin_axis_tilt_deg"]
        axis = f"{tilt:.0f}deg off vertical" if tilt is not None else "no dominant axis"
        print(
            f"  segment {seg['segment']}: {seg['tracked_frames']} frames | "
            f"axis spin {seg['axis_spin_deg']:+.0f}deg "
            f"({axis}) | "
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
    track = _camera_track(args)
    report = usdx.export(poses, dest, clip, tracking_id=args.track_id,
                         real_time=args.real_time, camera=track)
    print(f"wrote {report['path']}")
    print(
        f"  {report['joints']} joints, {report['frames']} frames "
        f"({report['frame_range'][0]}-{report['frame_range'][1]}) at "
        f"{report['fps']:g} fps, {report['size_mb']}MB"
    )
    if report["camera_compensated"]:
        print(f"  camera removed from the root, {report['segments']} shot(s), "
              "each its own world frame")
    else:
        print("  root left in the camera's frame: it turns when the camera pans")
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


def cmd_synth(args: argparse.Namespace) -> int:
    """The synthetic ground-truth loop; see docs/synthetic.md."""
    from .synth import compare, steps

    cfg.ensure_dirs()
    if args.step == "plan":
        for clip in steps.plan(args.seed, scale=args.scale, start=args.start,
                               long_ground=args.long_ground):
            labels = json.loads((clip / "labels.json").read_text())
            print(f"{clip.name:28s} {labels['inversions']} inversion(s), "
                  f"axis spin {labels['axis_spin_deg']:.0f}deg, "
                  f"{labels['camera_angle']}, {labels['slowmo_factor']:g}x")
        return 0
    selected = steps.clips(args.only)
    if not selected:
        print("no synthetic clips; run `snow synth plan` first", file=sys.stderr)
        return 1
    if args.step in ("render", "all"):
        for clip in selected:
            print(f"rendering {clip.name}")
            steps.render(clip, args.width, args.height, args.samples, args.engine)
            steps.encode(clip, engine=args.engine)
    if args.step == "encode":
        for clip in selected:
            steps.encode(clip, args.renderer, args.engine)
            print(f"encoded {clip.name} from {args.renderer} frames")
    if args.step in ("prepare", "all"):
        for clip in selected:
            print(f"ground truth + oracle for {clip.name}")
            steps.prepare(clip)
    if args.step in ("pipeline", "all"):
        steps.pipeline_all(selected, args.spacing)
    if args.step in ("compare", "all"):
        rows = steps.results(selected)
        for by in ("true_set", "true_kind", "true_camera_angle"):
            for variant in ("oracle", "pipeline"):
                print(f"\n{variant}, by {by.replace('true_', '')}")
                print(compare.table([r for r in rows if r["variant"] == variant], by))
        print("\n" + compare.per_clip(rows))
    if args.step == "fidelity":
        from .synth import fidelity

        print("| clip | renderer | frames | detected | confidence | 2D error / box "
              "| PA-MPJPE (mm) |")
        print("|---|---|---|---|---|---|---|")
        for clip in selected:
            if not (clip / "gt_raw.npz").exists():
                continue
            s = fidelity.clip_score(clip, cfg.pose_path(clip.name))
            renderer = clip / "renderer.txt"
            label = renderer.read_text().strip() if renderer.exists() else "?"
            print(f"| {clip.name} | {label} | {s['frames']} | {s['detected']:.2f} "
                  f"| {s.get('confidence', '–')} | {s.get('kp2d_rel', '–')} "
                  f"| {s.get('pa_mpjpe_mm', '–')} |")
    return 0


def cmd_coach(args: argparse.Namespace) -> int:
    """One Nemotron coaching report over an exported bundle; see docs/coach.md."""
    from . import coach, serve

    if not (serve.SESSIONS_DIR / args.name / "session.json").exists():
        print(f"no bundle for {args.name!r}; run web/tools/export_session.py first",
              file=sys.stderr)
        return 1
    session = coach.load_session(serve.SESSIONS_DIR, args.name)
    try:
        report = coach.coach_session(session, args.event)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from . import serve

    serve.serve(args.host, args.port)
    return 0


def cmd_label(args: argparse.Namespace) -> int:
    """The hand-labelling page; see docs/labelling.md."""
    from .label import prefill, server

    if args.fetch:
        for line in prefill.fetch_weights():
            print(line)
    server.run(args.host, args.port, Path(args.csv), open_browser=not args.no_browser,
               labeller=args.labeller, segmenter=args.segmenter, classifier=args.classifier,
               refresh=args.refresh)
    return 0


def cmd_classify(args: argparse.Namespace) -> int:
    """Model B, the trick classifier; see docs/training.md and docs/classifier.md."""
    from .learn import classifier as clf
    from .learn import dataset as ds
    from .learn import experiment as exp

    data = Path(args.data) if args.data else cfg.DATA_DIR
    out = Path(args.out) if args.out else data / "models" / "classifier"
    dataset = Path(args.dataset) if args.dataset else out / "dataset.npz"
    if args.step == "build":
        records = ds.build(data, heuristic=not args.no_heuristic)
        ds.save(records, dataset)
        splits = {s: sum(a.split == s for a in records) for s in ("train", "val", "test",
                                                                   "original")}
        print(f"{len(records)} airs -> {dataset}  {splits}")
        return 0
    if args.step == "predict":
        if not clf.available(Path(args.weights) if args.weights else clf.WEIGHTS):
            print("no classifier weights; run `snow classify train` or fetch them from S3",
                  file=sys.stderr)
            return 1
        from .synth import compare

        model = clf.Classifier.load(Path(args.weights) if args.weights else clf.WEIGHTS)
        name = args.name
        camera = cfg.POSES_DIR / f"{name}.camera.npz"
        track = clf.airs.track_from_poses(
            feat.load_poses(cfg.pose_path(name)), 0,
            cam.CameraTrack.load(camera) if camera.exists() else None)
        events = compare.air_events(name)
        if not events:
            print(f"{name}: no air found by the detector")
        for e in events:
            got = model.classify(track, e["start_f"], e["end_f"])
            measured = e["metrics"]["nearest_spin_deg"]
            print(f"  {e['id']} frames {e['start_f']}-{e['end_f']}: "
                  f"{got['name'] if got else 'no poses'}  (geometry says {measured})")
            if got:
                print("    " + "  ".join(f"{h} {got[h]['label']} p={got[h]['p']:.2f}"
                                         for h in ds.HEADS))
        return 0

    records = ds.load(dataset)
    config = clf.TrainConfig(epochs=args.epochs, seed=args.seed, width=args.width)
    print(f"{len(records)} airs; building {args.per_air} variants each")
    variants = clf.bank(records, per_air=args.per_air, seed=args.seed, workers=args.workers)
    if args.step == "cv":
        pool = [a for a in records if args.include_scale or a.split == "original"] or records
        report = exp.cross_validate(pool, variants, folds=args.folds, config=config,
                                    device=args.device)
        exp.write(report, out / "cv_report.json")
        print(exp.markdown(report))
        return 0
    checkpoint, report = exp.fit_split(records, variants, config=config, device=args.device,
                                       extra_train=args.extra_train)
    clf.save(checkpoint, Path(args.weights) if args.weights else out / "classifier.pt")
    exp.write(report, out / "split_report.json")
    print(exp.markdown(report))
    return 0


def cmd_harvest_search(args: argparse.Namespace) -> int:
    cfg.ensure_dirs()
    cache = harv.HarvestCache()
    limiter = harv.PoliteRateLimiter(min_delay=args.min_delay, max_delay=args.max_delay)

    print(f"Searching broadcasts: '{args.query}' (polite delay: {args.min_delay}-{args.max_delay}s)")
    results = harv.search_youtube(
        query=args.query,
        limit=args.limit,
        limiter=limiter,
        cache=cache,
        cookies_from_browser=args.cookies_from_browser,
        cookies=args.cookies,
        dry_run=args.dry_run,
        no_cache=args.no_cache,
    )

    if not results:
        if args.dry_run:
            print("[dry-run] Dry-run search complete.")
            return 0
        print("No results found.")
        return 0

    print(f"\nFound {len(results)} broadcast replays:")
    for i, r in enumerate(results, 1):
        print(f"[{i}] {r.title} ({r.duration_str})")
        print(f"    ID:      {r.id}")
        print(f"    URL:     {r.url}")
        print(f"    Channel: {r.uploader}")

        if args.extract_tricks:
            print(f"    Extracting transcript for {r.id}...")
            cues = harv.fetch_transcript(
                r.url,
                limiter=limiter,
                cache=cache,
                cookies_from_browser=args.cookies_from_browser,
                cookies=args.cookies,
                dry_run=args.dry_run,
                no_cache=args.no_cache,
            )
            mentions = harv.scan_transcript_for_tricks(cues)
            candidates = harv.cluster_trick_windows(mentions)
            if candidates:
                print(f"    Detected {len(candidates)} trick windows:")
                for c in candidates:
                    print(f"      * {c.start_ts} - {c.end_ts} ({c.duration}s): {c.trick_name} [score: {c.score:.1f}]")
                    print(f"        \"{c.transcript_excerpt[:100]}...\"")
            else:
                print("    No commentator trick callouts detected in transcript.")
        print()
    return 0


def cmd_harvest_get(args: argparse.Namespace) -> int:
    cfg.ensure_dirs()
    cache = harv.HarvestCache()
    limiter = harv.PoliteRateLimiter(min_delay=args.min_delay, max_delay=args.max_delay)

    # Derive base name if not supplied
    video_id = args.url
    if "v=" in args.url:
        video_id = args.url.split("v=")[1].split("&")[0]
    elif "/" in args.url:
        video_id = args.url.rstrip("/").split("/")[-1]
    base_name = args.name or f"harvest_{video_id[:12]}"

    print(f"Harvesting from: {args.url}")
    print("Fetching transcript first to identify trick windows...")
    cues = harv.fetch_transcript(
        args.url,
        limiter=limiter,
        cache=cache,
        cookies_from_browser=args.cookies_from_browser,
        cookies=args.cookies,
        dry_run=args.dry_run,
        no_cache=args.no_cache,
    )
    mentions = harv.scan_transcript_for_tricks(cues)
    candidates = harv.cluster_trick_windows(mentions)

    if candidates:
        print(f"Detected {len(candidates)} trick moments in commentator transcript:")
        for idx, c in enumerate(candidates, 1):
            print(f"  [{idx}] {c.start_ts} - {c.end_ts} ({c.duration}s): {c.trick_name} (score: {c.score:.1f})")
            print(f"      Transcript: \"{c.transcript_excerpt[:120]}\"")
    else:
        print("No automated trick callouts detected in transcript.")

    # If --auto-cut: cut all detected trick windows
    if args.auto_cut:
        if not candidates:
            print("Error: --auto-cut requested but no tricks were found in transcript.", file=sys.stderr)
            return 1
        print(f"\nAuto-cutting {len(candidates)} clips into {cfg.CLIPS_DIR}...")
        for idx, c in enumerate(candidates, 1):
            safe_trick = re.sub(r"[^a-zA-Z0-9_-]", "_", c.trick_name.lower())[:30]
            clip_name = f"{base_name}_{idx}_{safe_trick}"
            print(f"\n[{idx}/{len(candidates)}] Harvesting '{clip_name}' ({c.start_ts}, {c.duration}s)...")
            harv.harvest_clip(
                url=args.url,
                name=clip_name,
                start=c.start_ts,
                duration=str(c.duration),
                fps=args.fps,
                slowmo_factor=args.slowmo_factor,
                limit_rate=args.limit_rate,
                limiter=limiter,
                cookies_from_browser=args.cookies_from_browser,
                cookies=args.cookies,
                dry_run=args.dry_run,
                extra_meta={"trick": c.trick_name, "score": c.score, "transcript": c.transcript_excerpt},
            )
        print(f"\nAll {len(candidates)} clips harvested successfully.")
        return 0

    # Single clip harvest: either use explicit --start/--duration or select top candidate
    start = args.start
    duration = args.duration
    extra_meta = {}

    if not start or not duration:
        if candidates:
            top = max(candidates, key=lambda c: c.score)
            print(f"\nUsing top detected trick: {top.trick_name} at {top.start_ts} ({top.duration}s)")
            start = start or top.start_ts
            duration = duration or str(top.duration)
            extra_meta = {"trick": top.trick_name, "score": top.score, "transcript": top.transcript_excerpt}
        else:
            print("Error: No trick detected and --start / --duration not provided.", file=sys.stderr)
            print("Please provide --start <time> and --duration <sec> manually.", file=sys.stderr)
            return 1

    print(f"\nHarvesting clip '{base_name}' ({start}, {duration}s)...")
    dest = harv.harvest_clip(
        url=args.url,
        name=base_name,
        start=start,
        duration=duration,
        fps=args.fps,
        slowmo_factor=args.slowmo_factor,
        limit_rate=args.limit_rate,
        limiter=limiter,
        cookies_from_browser=args.cookies_from_browser,
        cookies=args.cookies,
        dry_run=args.dry_run,
        extra_meta=extra_meta,
    )
    if dest:
        print(f"Successfully harvested {base_name} -> {dest}")
    return 0


def cmd_harvest_fis(args: argparse.Namespace) -> int:
    cfg.ensure_dirs()
    print(f"Cross-referencing FIS Big Air events (query='{args.event}', season={args.season or 'any'})...")
    events = harv.search_fis_events(
        query=args.event,
        season=args.season,
        gender=args.gender,
        rider=args.rider,
    )

    if not events:
        print("No matching FIS events found in scoring/data/jumps.csv.")
        return 0

    print(f"\nFound {len(events)} matching FIS Big Air events:")
    for i, ev in enumerate(events[:10], 1):
        gender_label = "Men" if ev["gender"] == "M" else "Women"
        print(f"[{i}] {ev['season']} {ev['place']} - {ev['category']} ({gender_label}) - {ev['date']}")
        print(f"    Riders recorded: {len(ev['riders'])} | Jumps: {ev['jumps_count']}")
        if ev["top_tricks"]:
            print("    Top scored tricks:")
            for t in ev["top_tricks"][:4]:
                dec = t["decoded"]
                dec_str = f" ({dec.get('direction', '')} {dec.get('spin_deg', '')} {dec.get('grab', '')})" if dec else ""
                print(f"      * {t['rider']}: {t['code']}{dec_str} - Score: {t['score']:.1f}")

        queries = harv.build_fis_queries(ev)
        print("    Suggested YouTube queries:")
        for q in queries[:2]:
            print(f"      - \"{q}\"")

        if args.search and i == 1:
            print(f"\nSearching YouTube for top event query: \"{queries[0]}\"...")
            cache = harv.HarvestCache()
            limiter = harv.PoliteRateLimiter(min_delay=args.min_delay, max_delay=args.max_delay)
            vids = harv.search_youtube(
                queries[0],
                limit=args.limit,
                limiter=limiter,
                cache=cache,
                cookies_from_browser=args.cookies_from_browser,
                cookies=args.cookies,
                dry_run=args.dry_run,
                no_cache=args.no_cache,
            )
            for v in vids:
                print(f"      Found: {v.title} ({v.duration_str}) - {v.url}")
                if args.extract_tricks:
                    cues = harv.fetch_transcript(
                        v.url, limiter=limiter, cache=cache,
                        cookies_from_browser=args.cookies_from_browser, cookies=args.cookies,
                        dry_run=args.dry_run, no_cache=args.no_cache,
                    )
                    mentions = harv.scan_transcript_for_tricks(cues)
                    candidates = harv.cluster_trick_windows(mentions)
                    for c in candidates:
                        print(f"        * {c.start_ts} - {c.end_ts}: {c.trick_name}")
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
        p.add_argument("--no-stitch", action="store_true",
                       help="do not absorb the rider's other track ids within a shot")
        p.add_argument("--no-recover", action="store_true",
                       help="do not search the rider's gaps again at 0/90/180/270 deg")
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
    p_feat.add_argument("--segmenter", default="auto",
                        help="airborne model weights: auto (data/weights/segmenter.pt "
                             "or $SNOW_SEGMENTER if present), off (the old heuristic), "
                             "or a path")
    p_feat.set_defaults(func=cmd_features)

    p_usd = sub.add_parser("usd", help="export a USD skeletal animation for Omniverse")
    p_usd.add_argument("name")
    p_usd.add_argument("--track-id", type=int, default=1)
    p_usd.add_argument("--real-time", action="store_true",
                       help="author the timeline at real speed rather than "
                            "the clip's playback speed")
    p_usd.add_argument("--no-camera", action="store_true",
                       help="leave the root in the broadcast camera's frame")
    p_usd.add_argument("--recompute-camera", action="store_true")
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

    p_synth = sub.add_parser(
        "synth", help="synthetic tricks with exact labels: plan, render, compare"
    )
    p_synth.add_argument("step", choices=["plan", "render", "encode", "prepare",
                                          "pipeline", "compare", "fidelity", "all"])
    p_synth.add_argument("--renderer", choices=["blender", "replicator"], default="blender",
                         help="whose frames `encode` uses")
    p_synth.add_argument("--only", nargs="+", help="clip name prefixes, e.g. synth14")
    p_synth.add_argument("--seed", type=int, default=0)
    p_synth.add_argument("--width", type=int, default=1280)
    p_synth.add_argument("--height", type=int, default=720)
    p_synth.add_argument("--samples", type=int, default=16, help="render samples")
    p_synth.add_argument("--engine", choices=["eevee", "cycles"], default="eevee",
                         help="Blender engine; cycles for headless GPU boxes")
    p_synth.add_argument("--scale", type=int, default=0,
                         help="plan N clips of the balanced training set instead of the 20")
    p_synth.add_argument("--start", type=int, default=1,
                         help="first clip number for --scale")
    p_synth.add_argument("--long-ground", action="store_true",
                         help="--scale clips with long approaches and ride-outs on the snow")
    p_synth.add_argument("--spacing", type=float, default=None,
                         help="minimum seconds between NIM calls "
                              "(default 120 hosted, 0 self-hosted)")
    p_synth.set_defaults(func=cmd_synth)

    p_coach = sub.add_parser(
        "coach", help="ask Nemotron (NVIDIA API) to coach an exported session bundle"
    )
    p_coach.add_argument("name")
    p_coach.add_argument("--event", help="event id to focus on, e.g. air-1")
    p_coach.set_defaults(func=cmd_coach)

    p_serve = sub.add_parser(
        "serve", help="serve the dashboard's bundles plus POST /<name>/coach"
    )
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.set_defaults(func=cmd_serve)

    p_label = sub.add_parser(
        "label", help="hand-label real airs and tricks in the browser (docs/labelling.md)"
    )
    p_label.add_argument("--host", default="127.0.0.1")
    p_label.add_argument("--port", type=int, default=8765)
    p_label.add_argument("--csv", default=str(cfg.DATA_DIR / "labels" / "real_airs.csv"),
                         help="labels file, autosaved (labels.py format)")
    p_label.add_argument("--labeller", default=os.environ.get("USER", ""),
                         help="written into each row's labeller column")
    p_label.add_argument("--fetch", action="store_true",
                         help="first copy the segmenter and classifier weights from S3 into "
                              "data/models/ (never data/weights)")
    p_label.add_argument("--segmenter", help="segmenter weights for the machine guesses")
    p_label.add_argument("--classifier", help="classifier weights for the machine guesses")
    p_label.add_argument("--refresh", action="store_true",
                         help="recompute the machine guesses instead of using the cache")
    p_label.add_argument("--no-browser", action="store_true", help="do not open a browser")
    p_label.set_defaults(func=cmd_label)

    from .learn import cli as learn_cli

    p_seg = sub.add_parser(
        "segmenter", help="airborne segmenter: build dataset, evaluate, train, label template"
    )
    learn_cli.build_parser(p_seg)

    p_cls = sub.add_parser(
        "classify", help="trick classifier (model B): build, cv, train, predict"
    )
    p_cls.add_argument("step", choices=["build", "cv", "train", "predict"])
    p_cls.add_argument("name", nargs="?", help="clip to classify (predict)")
    p_cls.add_argument("--data", help="data directory (default: the repo's data/)")
    p_cls.add_argument("--out", help="artifact directory (default: data/models/classifier)")
    p_cls.add_argument("--dataset", help="dataset .npz (default: out/dataset.npz)")
    p_cls.add_argument("--weights", help="checkpoint path (default: out/classifier.pt)")
    p_cls.add_argument("--no-heuristic", action="store_true",
                       help="build: skip the dashboard detector's windows")
    p_cls.add_argument("--per-air", type=int, default=128,
                       help="augmented variants per training air")
    p_cls.add_argument("--epochs", type=int, default=40)
    p_cls.add_argument("--width", type=int, default=64)
    p_cls.add_argument("--seed", type=int, default=0)
    p_cls.add_argument("--folds", type=int, help="cv: folds (default: one per group)")
    p_cls.add_argument("--include-scale", action="store_true",
                       help="cv: pool the scale-up clips too, not just the original 20")
    p_cls.add_argument("--extra-train", action="store_true",
                       help="train: also train on the original 20 (then unscored)")
    p_cls.add_argument("--workers", type=int, default=4)
    p_cls.add_argument("--device", help="cuda or cpu (default: cuda if available)")
    p_cls.set_defaults(func=cmd_classify)

    # snow harvest
    p_harvest = sub.add_parser("harvest", help="safe anti-ban video harvester and FIS event matcher")
    sub_harvest = p_harvest.add_subparsers(dest="harvest_cmd", required=True)

    # snow harvest search <query>
    p_hsearch = sub_harvest.add_parser("search", help="search YouTube for broadcast replays with rate limiting")
    p_hsearch.add_argument("query", help="search terms, e.g. 'X Games 2024 snowboard big air'")
    p_hsearch.add_argument("--limit", type=int, default=5, help="number of results")
    p_hsearch.add_argument("--extract-tricks", "-t", action="store_true", help="scan transcripts for commentator trick callouts")
    p_hsearch.add_argument("--min-delay", type=float, default=harv.DEFAULT_MIN_DELAY, help="min seconds between requests")
    p_hsearch.add_argument("--max-delay", type=float, default=harv.DEFAULT_MAX_DELAY, help="max seconds between requests")
    p_hsearch.add_argument("--cookies-from-browser", help="browser to extract cookies from (e.g. chrome, firefox)")
    p_hsearch.add_argument("--cookies", help="path to custom cookies.txt file")
    p_hsearch.add_argument("--no-cache", action="store_true", help="bypass local harvest cache")
    p_hsearch.add_argument("--dry-run", action="store_true", help="preview search without network downloads")
    p_hsearch.set_defaults(func=cmd_harvest_search)

    # snow harvest get <url|id>
    p_hget = sub_harvest.add_parser("get", help="transcript-first downloader: finds tricks and cuts NIM-ready clips")
    p_hget.add_argument("url", help="YouTube URL or video ID")
    p_hget.add_argument("--name", help="output clip name stem (in data/clips/<name>.mp4)")
    p_hget.add_argument("--start", help="start timestamp (e.g. 00:01:23 or 83.0)")
    p_hget.add_argument("--duration", help="clip length in seconds (e.g. 24 or 00:00:24)")
    p_hget.add_argument("--auto-cut", action="store_true", help="automatically cut clips for all detected tricks")
    p_hget.add_argument("--limit-rate", default=harv.DEFAULT_LIMIT_RATE, help="yt-dlp rate limit (default 3M)")
    p_hget.add_argument("--fps", type=float, default=30.0, help="output frame rate CFR (default 30.0)")
    p_hget.add_argument("--slowmo-factor", type=float, default=1.0, help="slow-motion factor (default 1.0)")
    p_hget.add_argument("--min-delay", type=float, default=harv.DEFAULT_MIN_DELAY)
    p_hget.add_argument("--max-delay", type=float, default=harv.DEFAULT_MAX_DELAY)
    p_hget.add_argument("--cookies-from-browser", help="browser for cookies")
    p_hget.add_argument("--cookies", help="cookies file")
    p_hget.add_argument("--no-cache", action="store_true")
    p_hget.add_argument("--dry-run", action="store_true", help="preview trick timestamps and commands without downloading")
    p_hget.set_defaults(func=cmd_harvest_get)

    # snow harvest fis <event>
    p_hfis = sub_harvest.add_parser("fis", help="cross-reference FIS big air database and generate broadcast queries")
    p_hfis.add_argument("event", nargs="?", default="", help="event keyword, place, or codex (e.g. 'Kreischberg', 'Chur')")
    p_hfis.add_argument("--season", type=int, help="season year (e.g. 2021, 2024)")
    p_hfis.add_argument("--gender", choices=["M", "W"], help="gender category")
    p_hfis.add_argument("--rider", help="rider name filter")
    p_hfis.add_argument("--search", action="store_true", help="search YouTube for matching broadcasts")
    p_hfis.add_argument("--extract-tricks", "-t", action="store_true", help="scan transcripts for trick callouts")
    p_hfis.add_argument("--limit", type=int, default=3, help="number of search results")
    p_hfis.add_argument("--min-delay", type=float, default=harv.DEFAULT_MIN_DELAY)
    p_hfis.add_argument("--max-delay", type=float, default=harv.DEFAULT_MAX_DELAY)
    p_hfis.add_argument("--cookies-from-browser", help="browser for cookies")
    p_hfis.add_argument("--cookies", help="cookies file")
    p_hfis.add_argument("--no-cache", action="store_true")
    p_hfis.add_argument("--dry-run", action="store_true")
    p_hfis.set_defaults(func=cmd_harvest_fis)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ing.IngestError, trk.TrackError, ps.PoseError, cp.CropError, CoachError, harv.HarvestError) as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
