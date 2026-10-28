"""``snow segmenter``: build the dataset, evaluate, train, label.

Also runnable as ``python -m snow.learn`` on a training box that has only
torch, numpy, pandas and opencv installed (the GPU box does not need the
tracker, the NIM client or the dashboard).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .. import config as cfg

MODEL_DIR = cfg.DATA_DIR / "models" / "segmenter"
DEFAULT_BUNDLE = MODEL_DIR / "dataset.npz"
DEFAULT_WEIGHTS = cfg.WEIGHTS_DIR / "segmenter.pt"


def cmd_build(args: argparse.Namespace) -> int:
    from . import data as D
    from . import labels as L

    sources = D.discover(Path(args.synth_dir) if args.synth_dir else None,
                         Path(args.manifest) if args.manifest else None)
    if args.gold:
        sources += L.gold_sources(Path(args.gold))
    clips, skipped = [], []
    for src in sources:
        clip = D.load_clip(src)
        (clips.append(clip) if clip else skipped.append(src.name))
    if not clips:
        print("no labelled clips with pipeline output", file=sys.stderr)
        return 1
    D.save_bundle(clips, Path(args.out))
    frames = sum(len(c["y"]) for c in clips)
    air = sum(int(c["y"].sum()) for c in clips)
    print(f"{len(clips)} clips, {len({c['group'] for c in clips})} groups, {frames} frames "
          f"({air / frames:.0%} airborne) -> {args.out}")
    if skipped:
        print(f"  skipped {len(skipped)} without pipeline output: {', '.join(skipped[:8])}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from . import data as D
    from . import train as T

    clips = D.load_bundle(Path(args.bundle))
    keep: set[str] = set()
    if args.keep_in_train:
        keep = {c["name"] for c in json.loads(Path(args.keep_in_train).read_text())["clips"]}
    result = T.evaluate(clips, folds=args.folds, arch=args.arch, epochs=args.epochs,
                        seed=args.seed, window=args.window, holdout=args.holdout,
                        keep_in_train=keep, ignore=tuple(args.ignore),
                        noise=_noise(args.noise))
    T.save_result(result, Path(args.out))
    print(f"\n{result['protocol']}: {result['clips']} clips, {result['groups']} groups, "
          f"{result['arch']} with {result['params']:,} parameters\n")
    print(T.report_table(result))
    for variant, summary in result.get("by_variant", {}).items():
        if len(result.get("by_variant", {})) > 1:
            clips = next(iter(summary.values()))["clips"]
            print(f"\n{variant or 'unlabelled'} clips only ({clips}):\n")
            print(T.report_table(result, summary))
    print(f"\nwrote {args.out}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from . import data as D
    from . import train as T

    clips = D.load_bundle(Path(args.bundle))
    seg = T.fit(clips, arch=args.arch, epochs=args.epochs, seed=args.seed, window=args.window,
                ignore=tuple(args.ignore), noise=_noise(args.noise))
    seg.info["bundle"] = str(args.bundle)
    seg.save(Path(args.out))
    print(f"{seg.arch}, {seg.info['params']:,} parameters, trained on {len(clips)} clips "
          f"-> {args.out}")
    return 0


def _noise(choice: str):
    from . import noise as N

    if choice == "off":
        return None
    if choice == "real":
        return N.REAL_NOISE
    return N.NoiseModel.load(Path(choice))


def cmd_noise(args: argparse.Namespace) -> int:
    """Measure real NIM noise on ground stretches and fit the augmentation."""
    from . import data as D
    from . import noise as N

    real = N.summarise(N.real_stretches(args.spans))
    names = sorted({N.parse_span(t)[0] for t in args.spans})
    drops = N.clip_dropouts(names)
    real.update(holes_per_frame=drops["holes_per_frame"], hole_mean_frames=drops["hole_mean_frames"])
    print(f"dropouts over whole clips {names}: {drops['holes']} in {drops['frames']} frames")
    clips = D.load_bundle(Path(args.bundle))
    synth_runs = N.synthetic_ground(clips)
    synth = N.summarise(synth_runs)
    model = N.fit(real, synth_runs)
    model.source = "fitted on " + ", ".join(args.spans)
    print(json.dumps({"real": real, "synthetic_ground": synth}, indent=1))
    print(json.dumps(model.__dict__, indent=1))
    model.save(Path(args.out))
    print(f"wrote {args.out}")
    return 0


def cmd_template(args: argparse.Namespace) -> int:
    from . import labels as L

    rows = L.write_template(Path(args.out), args.clips or None)
    print(f"{rows} clip rows -> {args.out}; see docs/segmenter.md#labelling-real-airs")
    return 0


def build_parser(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    parser = parser or argparse.ArgumentParser(prog="python -m snow.learn")
    sub = parser.add_subparsers(dest="step", required=True)

    p = sub.add_parser("build", help="labelled clips -> one dataset bundle")
    p.add_argument("--synth-dir", help="default data/synth")
    p.add_argument("--manifest", help="JSON Lines with name and group (and optional split)")
    p.add_argument("--gold", help="hand-labelled airs CSV (labels template format)")
    p.add_argument("--out", default=str(DEFAULT_BUNDLE))
    p.set_defaults(func=cmd_build)

    for name, func, helptext in (
        ("evaluate", cmd_evaluate, "held-out scores, segmenter vs both heuristics"),
        ("train", cmd_train, "train on the whole bundle and save weights"),
    ):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--bundle", default=str(DEFAULT_BUNDLE))
        p.add_argument("--arch", choices=["tcn", "bigru"], default="tcn")
        p.add_argument("--epochs", type=int, default=60)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--window", type=int, default=64, help="frames per training window")
        p.add_argument("--noise", default="real",
                       help="real-footage noise augmentation: real (the measured model in "
                            "noise.py), off, or a noise_model.json from `snow segmenter noise`")
        p.add_argument("--ignore", nargs="*", default=["confidence", "pose"],
                       help="input channel name prefixes to blank (default: confidence "
                            "pose; pass --ignore with nothing to use every input)")
        if name == "evaluate":
            p.add_argument("--folds", type=int, default=5)
            p.add_argument("--holdout", type=float,
                           help="instead of k-fold, hold out this fraction of groups once")
            p.add_argument("--keep-in-train",
                           help="a bundle index (.json) whose clips always train: the "
                                "ones the settings were tuned on")
            p.add_argument("--out", default=str(MODEL_DIR / "eval.json"))
        else:
            p.add_argument("--out", default=str(DEFAULT_WEIGHTS))
        p.set_defaults(func=func)

    p = sub.add_parser("noise", help="measure real NIM noise on ground stretches, fit the "
                                     "training augmentation")
    p.add_argument("spans", nargs="+", help="clip:first-end frames that are plainly on the "
                                            "snow, e.g. run1:460-495")
    p.add_argument("--bundle", default=str(DEFAULT_BUNDLE),
                   help="dataset bundle whose long-ground clips are the synthetic reference")
    p.add_argument("--out", default=str(MODEL_DIR / "noise_model.json"))
    p.set_defaults(func=cmd_noise)

    p = sub.add_parser("template", help="CSV template for hand-labelling real airs")
    p.add_argument("clips", nargs="*", help="clip names (default: every clip in data/clips)")
    p.add_argument("--out", default=str(cfg.DATA_DIR / "labels" / "airs.csv"))
    p.set_defaults(func=cmd_template)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
