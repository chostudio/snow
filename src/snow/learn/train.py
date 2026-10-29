"""Train and evaluate the segmenter on a dataset bundle (``data.save_bundle``).

Evaluation is always on clips whose *group* the model never saw, scored
against the ground-truth labels, side by side with the two heuristics on the
very same frames. With a fixed ``split`` in the manifest that split is used;
otherwise grouped k-fold cross-validation scores every clip exactly once as
held out. The shipped weights are then trained on everything.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import events as E
from . import inputs as I
from . import noise as N
from .data import check_no_leak, group_folds
from .segmenter import WINDOW, Segmenter, build_model, parameter_count

TIMEBASE_RANGE = (0.5, 2.0)   # random playback-rate stretch, log-uniform
NOISE = 0.05                  # gaussian noise, in standardised units
# Per-frame pose jitter, drawn per segment up to these sizes in the channels'
# own units. Rendered riders come out of the NIM far steadier than broadcast
# ones: on the ground, synthetic rot_step has a median of 0.05 (0.5 deg per
# frame) while the two real clips read 0.26 and 0.35, as much as synthetic
# air. Without this the model learns "any rotation is air".
JITTER_MAX = {"rot_step": 0.5, "elev_step": 0.5, "azim_step": 0.5, "pose": 0.05}
# Inputs left out by default, measured on synthetic data and checked against
# the real clips (docs/segmenter.md): confidence is a renderer property
# (median 0.91 synthetic, 0.59 to 0.64 real), and the body-frame pose let
# 16 training clips be memorised.
DEFAULT_IGNORE = ("confidence", "pose")


def standardiser(clips: list[dict], ignore: tuple[str, ...] = ()) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean and std over the training clips (and their mirrors).

    A channel whose name starts with one of ``ignore`` gets a huge std, so it
    reads as zero after standardising: an ablation that travels with the
    weights rather than needing a different input layout.
    """
    stacked = np.concatenate([np.concatenate([c["x"], c["x_mirror"]]) for c in clips])
    mean = stacked.mean(axis=0)
    std = np.maximum(stacked.std(axis=0), 1e-3)
    # The present flag stays 0/1 so a padded frame reads as absent.
    mean[I.PRESENT], std[I.PRESENT] = 0.0, 1.0
    for k, name in enumerate(I.CHANNELS):
        if name.startswith(tuple(ignore)) and ignore:
            mean[k], std[k] = 0.0, 1e9
    return mean.astype(np.float32), std.astype(np.float32)


def _segments(clip: dict):
    seg = clip["segment"]
    for s in np.unique(seg):
        yield np.flatnonzero(seg == s)


def jitter(x: np.ndarray, rng: np.random.Generator, seen: np.ndarray) -> None:
    """Add NIM-like per-frame noise in place, at a random level per segment."""
    for name, top in JITTER_MAX.items():
        cols = [k for k, c in enumerate(I.CHANNELS) if c.startswith(name)]
        sigma = rng.uniform(0.0, top)
        noise = rng.normal(0.0, sigma, size=(len(x), len(cols)))
        if name == "rot_step":
            x[:, cols] = np.abs(x[:, cols] + noise)  # an angle step is unsigned
        else:
            x[:, cols] += noise
        x[np.ix_(~seen, cols)] = 0.0


def sample_windows(clips: list[dict], mean: np.ndarray, std: np.ndarray,
                   rng: np.random.Generator, window: int = WINDOW,
                   per_window: float = 8.0, noise: "N.NoiseModel | None" = N.REAL_NOISE,
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One epoch of training windows, freshly augmented.

    Per clip and segment: mirror with probability 0.5, stretch time by a
    random factor, then take random windows plus windows centred near each
    takeoff and landing (with jitter) so boundaries are seen often. Every
    window lies inside one camera segment.
    """
    xs, ys, ms = [], [], []
    pad_row = ((np.zeros(I.NUM_CHANNELS) - mean) / std).astype(np.float32)
    for clip in clips:
        for rows in _segments(clip):
            source = clip["x_mirror"] if rng.random() < 0.5 else clip["x"]
            factor = float(np.exp(rng.uniform(*np.log(TIMEBASE_RANGE))))
            x, y = I.resample(source[rows], clip["y"][rows].astype(np.float32), factor)
            # No loss where there is no evidence (see inputs.observed).
            seen = I.observed(x[:, I.PRESENT], np.zeros(len(x), int)).astype(np.float32)
            # Real-footage NIM noise (measured, see noise.py), then the
            # generic jitter on top.
            if noise is not None and rng.random() < N.APPLY_PROBABILITY:
                N.apply(x, rng, seen > 0, noise)
            jitter(x, rng, seen > 0)
            x = (x - mean) / std
            n = len(x)
            if n < window:
                px = np.tile(pad_row, (window, 1))
                px[:n] = x
                py = np.zeros(window, np.float32)
                py[:n] = y
                pm = np.zeros(window, np.float32)
                pm[:n] = seen
                xs.append(px), ys.append(py), ms.append(pm)
                continue
            starts = list(rng.integers(0, n - window + 1, size=max(1, int(per_window * n / window))))
            edges = np.flatnonzero(np.diff(y) != 0) + 1
            for e in edges:
                centre = int(e + rng.integers(-window // 3, window // 3 + 1))
                starts.append(int(np.clip(centre - window // 2, 0, n - window)))
            for s in starts:
                xs.append(x[s:s + window]), ys.append(y[s:s + window])
                ms.append(seen[s:s + window])
    return (np.stack(xs).astype(np.float32), np.stack(ys).astype(np.float32),
            np.stack(ms).astype(np.float32))


def fit(train: list[dict], arch: str = "tcn", epochs: int = 60, batch: int = 64,
        lr: float = 2e-3, seed: int = 0, window: int = WINDOW, device: str | None = None,
        ignore: tuple[str, ...] = DEFAULT_IGNORE, noise: "N.NoiseModel | None" = N.REAL_NOISE,
        log=print) -> Segmenter:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    mean, std = standardiser(train, ignore)
    model = build_model(arch, window=window).to(device)
    labels = np.concatenate([c["y"][I.observed(c["x"][:, I.PRESENT], c["segment"])]
                             for c in train])
    pos = max(labels.mean(), 1e-3)
    # Balance the classes: airborne may be the minority (real broadcast) or
    # the majority (synthetic clips are mostly air), and either way each
    # class should carry half the loss.
    pos_weight = torch.tensor((1 - pos) / pos, device=device)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs, pct_start=0.15)
    # Ignored channels stay exactly zero, as they are at inference.
    live = torch.tensor(std < 1e8, dtype=torch.float32, device=device)
    for epoch in range(epochs):
        model.train()
        x, y, m = sample_windows(train, mean, std, rng, window=window, noise=noise)
        order = rng.permutation(len(x))
        total, count = 0.0, 0.0
        for k in range(0, len(order), batch):
            idx = order[k:k + batch]
            xb = torch.from_numpy(x[idx]).to(device)
            xb = xb + NOISE * torch.randn_like(xb) * live
            yb = torch.from_numpy(y[idx]).to(device)
            mb = torch.from_numpy(m[idx]).to(device)
            loss = (loss_fn(model(xb), yb) * mb).sum() / mb.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(idx)
            count += len(idx)
        sched.step()
        if epoch % 10 == 0 or epoch == epochs - 1:
            log(f"    epoch {epoch:3d}  loss {total / max(count, 1):.4f}  windows {len(x)}")
    model = model.cpu().eval()
    return Segmenter(model=model, arch=arch, mean=mean, std=std, window=window,
                     info={"params": parameter_count(model), "epochs": epochs,
                           "train_clips": [c["name"] for c in train],
                           "pos_fraction": float(pos), "ignore": list(ignore),
                           "noise": noise.__dict__ if noise is not None else None})


# --- scoring -----------------------------------------------------------------

# "always" is a floor, not a method: every frame with a body called airborne.
# Synthetic clips are about 63% air, so frame F1 alone flatters everything.
METHODS = ("segmenter", "hint", "web", "always")


def predict(seg: Segmenter, clip: dict) -> tuple[np.ndarray, np.ndarray]:
    prob = seg.probabilities(clip["x"], clip["segment"])
    return prob, seg.mask(prob, clip["segment"])


def score_clip(clip: dict, masks: dict[str, np.ndarray]) -> dict:
    truth = E.runs(clip["y"], clip["segment"])
    out = {"name": clip["name"], "group": clip["group"], "slowmo": clip["slowmo"],
           "kind": clip.get("kind"), "camera_angle": clip.get("camera_angle"),
           "variant": clip.get("variant") or "",
           "frames": int(len(clip["y"])), "air_frames": int(clip["y"].sum())}
    for method, mask in masks.items():
        counts = E.frame_counts(mask, clip["y"])
        errs = E.boundary_errors(mask, truth, clip["segment"])
        out[method] = {"counts": counts, "f1": E.f1(counts), "airs": errs}
    return out


def summarise(rows: list[dict], methods=METHODS) -> dict:
    out = {}
    for method in methods:
        have = [r for r in rows if method in r]
        if not have:
            continue
        counts = {k: sum(r[method]["counts"][k] for r in have) for k in ("tp", "fp", "fn", "tn")}
        # 95% interval for pooled frame F1, resampling whole clips.
        rng = np.random.default_rng(0)
        boot = []
        for _ in range(1000):
            pick = rng.integers(0, len(have), len(have))
            c = {k: sum(have[i][method]["counts"][k] for i in pick) for k in ("tp", "fp", "fn")}
            boot.append(E.f1(c))
        airs = [a for r in have for a in r[method]["airs"]]
        slow = [r["slowmo"] for r in have for _ in r[method]["airs"]]
        hit = [(a, s) for a, s in zip(airs, slow) if not a["missed"]]
        take = np.array([abs(a["takeoff_err"]) for a, _ in hit], dtype=float)
        land = np.array([abs(a["landing_err"]) for a, _ in hit], dtype=float)
        # Real-time milliseconds, using the render's known slow-motion factor:
        # for reporting only, the model never sees it.
        take_ms = np.array([abs(a["takeoff_err"]) / (30.0 * s) * 1000 for a, s in hit])
        land_ms = np.array([abs(a["landing_err"]) / (30.0 * s) * 1000 for a, s in hit])
        precision = counts["tp"] / max(counts["tp"] + counts["fp"], 1)
        recall = counts["tp"] / max(counts["tp"] + counts["fn"], 1)

        def stat(values, fn):
            return round(float(fn(values)), 1) if len(values) else None

        out[method] = {
            "clips": len(have),
            "frame_f1": round(E.f1(counts), 4),
            "frame_f1_ci95": [round(float(np.percentile(boot, 2.5)), 4),
                              round(float(np.percentile(boot, 97.5)), 4)],
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "mean_clip_f1": round(float(np.mean([r[method]["f1"] for r in have])), 4),
            "airs": len(airs),
            "missed": sum(a["missed"] for a in airs),
            "split_airs": sum(1 for a, _ in hit if a.get("pieces", 1) > 1),
            "false_airs": sum(a.get("false_airs", 0) for a in airs),
            "takeoff_median_abs_frames": stat(take, np.median),
            "takeoff_mean_abs_frames": stat(take, np.mean),
            "landing_median_abs_frames": stat(land, np.median),
            "landing_mean_abs_frames": stat(land, np.mean),
            "takeoff_median_abs_ms": stat(take_ms, np.median),
            "landing_median_abs_ms": stat(land_ms, np.median),
            "within_3_frames": sum(1 for a, _ in hit
                                   if abs(a["takeoff_err"]) <= 3 and abs(a["landing_err"]) <= 3),
        }
    return out


def holdout_split(clips: list[dict], fraction: float, seed: int = 0,
                  keep_in_train: set[str] = frozenset()) -> tuple[list[dict], list[dict]]:
    """Hold out about ``fraction`` of the eligible groups, whole.

    Groups containing any clip in ``keep_in_train`` (the clips the settings
    were tuned on) always train, so the test set was never looked at.
    """
    pinned = {c["group"] for c in clips if c["name"] in keep_in_train}
    groups = sorted({c["group"] for c in clips} - pinned)
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    test_groups = set(groups[:max(1, int(round(fraction * len(groups))))])
    return ([c for c in clips if c["group"] not in test_groups],
            [c for c in clips if c["group"] in test_groups])


def evaluate(clips: list[dict], folds: int = 5, arch: str = "tcn", epochs: int = 60,
             seed: int = 0, window: int = WINDOW, holdout: float | None = None,
             keep_in_train: set[str] = frozenset(), ignore: tuple[str, ...] = DEFAULT_IGNORE,
             noise: "N.NoiseModel | None" = N.REAL_NOISE,
             log=print) -> dict:
    """Held-out scores for every method on the same clips."""
    fixed = [c for c in clips if c.get("split") in ("train", "test")]
    if fixed and len(fixed) == len(clips):
        plan = [([c for c in clips if c["split"] == "train"],
                 [c for c in clips if c["split"] == "test"])]
        protocol = "fixed split from the manifest"
    elif holdout:
        plan = [holdout_split(clips, holdout, seed, keep_in_train)]
        protocol = (f"one held-out set of {len(plan[0][1])} clips "
                    f"({len({c['group'] for c in plan[0][1]})} groups), never used for tuning")
    else:
        fold = group_folds(clips, folds, seed)
        plan = [([c for c, f in zip(clips, fold) if f != k],
                 [c for c, f in zip(clips, fold) if f == k]) for k in range(folds)]
        protocol = f"{folds}-fold cross-validation grouped by event"
    rows = []
    for k, (train, test) in enumerate(plan):
        check_no_leak(train, test)
        log(f"  fold {k}: {len(train)} train clips, {len(test)} held out "
            f"({len({c['group'] for c in test})} groups)")
        start = time.time()
        seg = fit(train, arch=arch, epochs=epochs, seed=seed + k, window=window,
                  ignore=ignore, noise=noise, log=log)
        log(f"    trained in {time.time() - start:.0f}s")
        for clip in test:
            prob, mask = predict(seg, clip)
            always = I.observed(clip["x"][:, I.PRESENT], clip["segment"])
            row = score_clip(clip, {"segmenter": mask, "hint": clip["hint"], "web": clip["web"],
                                    "always": always})
            row["fold"] = k
            row["prob"] = [round(float(p), 3) if np.isfinite(p) else None for p in prob]
            rows.append(row)
    return {"protocol": protocol, "arch": arch, "epochs": epochs, "window": window,
            "ignore": list(ignore),
            "noise": noise.__dict__ if noise is not None else None,
            "params": parameter_count(build_model(arch, window=window)),
            "clips": len(clips), "groups": len({c["group"] for c in clips}),
            "summary": summarise(rows),
            # The same scores per kind of clip, so a mix cannot hide a weak one.
            "by_variant": {v: summarise([r for r in rows if r["variant"] == v])
                           for v in sorted({r["variant"] for r in rows})},
            "per_clip": rows}


def report_table(result: dict, summary: dict | None = None) -> str:
    names = {"segmenter": f"segmenter ({result['arch']})", "hint": "airborne_hint (old)",
             "web": "dashboard detector", "always": "every seen frame (floor)"}
    lines = ["| method | frame F1 (95% CI) | precision | recall | airs found | split | false airs "
             "| takeoff err, median (mean) frames | landing err, median (mean) frames |",
             "|---|---|---|---|---|---|---|---|---|"]
    for method, s in (summary or result["summary"]).items():
        lo, hi = s["frame_f1_ci95"]
        lines.append(
            f"| {names[method]} | {s['frame_f1']:.3f} ({lo:.3f} to {hi:.3f}) "
            f"| {s['precision']:.3f} | {s['recall']:.3f} "
            f"| {s['airs'] - s['missed']}/{s['airs']} | {s['split_airs']} | {s['false_airs']} "
            f"| {s['takeoff_median_abs_frames']} ({s['takeoff_mean_abs_frames']}) "
            f"| {s['landing_median_abs_frames']} ({s['landing_mean_abs_frames']}) |")
    return "\n".join(lines)


def save_result(result: dict, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=1))
