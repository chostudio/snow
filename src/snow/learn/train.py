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
from .data import check_no_leak, group_folds
from .segmenter import WINDOW, Segmenter, build_model, parameter_count

TIMEBASE_RANGE = (0.5, 2.0)   # random playback-rate stretch, log-uniform
NOISE = 0.05                  # gaussian noise, in standardised units


def standardiser(clips: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    stacked = np.concatenate([np.concatenate([c["x"], c["x_mirror"]]) for c in clips])
    mean = stacked.mean(axis=0)
    std = np.maximum(stacked.std(axis=0), 1e-3)
    # The present flag stays 0/1 so a padded frame reads as absent.
    mean[I.PRESENT], std[I.PRESENT] = 0.0, 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def _segments(clip: dict):
    seg = clip["segment"]
    for s in np.unique(seg):
        yield np.flatnonzero(seg == s)


def sample_windows(clips: list[dict], mean: np.ndarray, std: np.ndarray,
                   rng: np.random.Generator, window: int = WINDOW,
                   per_window: float = 8.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        log=print) -> Segmenter:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    mean, std = standardiser(train)
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
    for epoch in range(epochs):
        model.train()
        x, y, m = sample_windows(train, mean, std, rng, window=window)
        order = rng.permutation(len(x))
        total, count = 0.0, 0.0
        for k in range(0, len(order), batch):
            idx = order[k:k + batch]
            xb = torch.from_numpy(x[idx]).to(device)
            xb = xb + NOISE * torch.randn_like(xb)
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
                           "pos_fraction": float(pos)})


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


def evaluate(clips: list[dict], folds: int = 5, arch: str = "tcn", epochs: int = 60,
             seed: int = 0, window: int = WINDOW, log=print) -> dict:
    """Held-out scores for every method on the same clips."""
    fixed = [c for c in clips if c.get("split") in ("train", "test")]
    if fixed and len(fixed) == len(clips):
        plan = [([c for c in clips if c["split"] == "train"],
                 [c for c in clips if c["split"] == "test"])]
        protocol = "fixed split from the manifest"
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
        seg = fit(train, arch=arch, epochs=epochs, seed=seed + k, window=window, log=log)
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
            "params": parameter_count(build_model(arch, window=window)),
            "clips": len(clips), "groups": len({c["group"] for c in clips}),
            "summary": summarise(rows), "per_clip": rows}


def report_table(result: dict) -> str:
    names = {"segmenter": f"segmenter ({result['arch']})", "hint": "airborne_hint (old)",
             "web": "dashboard detector", "always": "every seen frame (floor)"}
    lines = ["| method | frame F1 | precision | recall | airs found | split | false airs "
             "| takeoff err, median (mean) frames | landing err, median (mean) frames |",
             "|---|---|---|---|---|---|---|---|---|"]
    for method, s in result["summary"].items():
        lines.append(
            f"| {names[method]} | {s['frame_f1']:.3f} | {s['precision']:.3f} | {s['recall']:.3f} "
            f"| {s['airs'] - s['missed']}/{s['airs']} | {s['split_airs']} | {s['false_airs']} "
            f"| {s['takeoff_median_abs_frames']} ({s['takeoff_mean_abs_frames']}) "
            f"| {s['landing_median_abs_frames']} ({s['landing_mean_abs_frames']}) |")
    return "\n".join(lines)


def save_result(result: dict, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=1))
