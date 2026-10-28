"""Train and score the classifier: grouped cross-validation, or a fixed split.

``cross_validate`` holds out whole groups (a group is one generated event, so
a clip and its replays never sit on both sides) and pools the held-out
predictions of every fold into one score. ``fit_split`` trains on the
manifest's train groups, picks nothing on test, and scores the test groups
and, separately, the original twenty clips, which no scale-up clip shares an
event with.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from . import classifier as clf
from . import dataset as ds
from . import evaluate as ev


def _fold_of(group: str, folds: int) -> int:
    return int(hashlib.sha1(group.encode()).hexdigest()[:8], 16) % folds


def _score_both(records, model, majority) -> tuple[dict, dict]:
    results, rows = {}, {}
    for which in ev.WINDOWS:
        preds = ev.predict(model, records, which)
        bases = [ev.baseline(a, which, majority) for a in records]
        results[which] = ev.score(records, preds, bases)
        rows[which] = ev.per_air(records, preds, bases)
    return results, rows


def cross_validate(records: list[ds.Air], variants: dict, folds: int | None = None,
                   config: clf.TrainConfig | None = None, device: str | None = None,
                   log=print) -> dict:
    groups = sorted({a.group for a in records})
    folds = min(folds or len(groups), len(groups))
    if folds == len(groups):
        fold_of = {g: i for i, g in enumerate(groups)}
    else:
        fold_of = {g: _fold_of(g, folds) for g in groups}
    preds = {w: [None] * len(records) for w in ev.WINDOWS}
    bases = {w: [None] * len(records) for w in ev.WINDOWS}
    for k in range(folds):
        test = [i for i, a in enumerate(records) if fold_of[a.group] == k]
        train = [a for a in records if fold_of[a.group] != k]
        if not test:
            continue
        log(f"  fold {k + 1}/{folds}: train {len(train)} airs, "
            f"hold out {', '.join(records[i].name for i in test[:4])}"
            + (" ..." if len(test) > 4 else ""))
        checkpoint = clf.train([a.name for a in train], variants, config, device,
                               log=log)
        model = clf.Classifier(checkpoint, device=device or "cpu")
        majority = ev.majority_labels(train)
        held = [records[i] for i in test]
        for w in ev.WINDOWS:
            got = ev.predict(model, held, w)
            for i, p in zip(test, got):
                preds[w][i] = p
                bases[w][i] = ev.baseline(records[i], w, majority)
    results = {w: ev.score(records, preds[w], bases[w]) for w in ev.WINDOWS}
    rows = {w: ev.per_air(records, preds[w], bases[w]) for w in ev.WINDOWS}
    return {"protocol": f"grouped {folds}-fold cross-validation", "n": len(records),
            "groups": len(groups), "results": results, "per_air": rows}


def fit_split(records: list[ds.Air], variants: dict, config: clf.TrainConfig | None = None,
              device: str | None = None, extra_train: bool = False, log=print) -> tuple[dict, dict]:
    """Train on the manifest's train split; score test, val and the originals.

    ``extra_train`` adds the original twenty to training (then they are not
    scored).
    """
    by = {s: [a for a in records if a.split == s] for s in ("train", "val", "test", "original")}
    train = by["train"] + (by["original"] if extra_train else [])
    val_x = val_y = None
    if by["val"]:
        enc = [(clf._encode_variants((a.track, a.gt, a.labels, 2, 0))) for a in by["val"]]
        val_x = np.concatenate([e[0][:1] for e in enc])
        val_y = np.concatenate([e[1][:1] for e in enc])
    log(f"  train {len(train)} airs ({len({a.group for a in train})} groups), "
        f"val {len(by['val'])}, test {len(by['test'])}, original {len(by['original'])}")
    checkpoint = clf.train([a.name for a in train], variants, config, device,
                           val=(val_x, val_y) if val_x is not None else None, log=log)
    model = clf.Classifier(checkpoint, device=device or "cpu")
    majority = ev.majority_labels(train)
    report = {"protocol": "manifest split", "n_train": len(train),
              "train_groups": len({a.group for a in train}), "results": {}, "per_air": {}}
    for split in ("test", "val") + (() if extra_train else ("original",)):
        if not by[split]:
            continue
        results, rows = _score_both(by[split], model, majority)
        report["results"][split] = results
        report["per_air"][split] = rows
    return checkpoint, report


def markdown(report: dict) -> str:
    parts = []
    blocks = report["results"]
    nested = all(isinstance(v, dict) and "gt" in v for v in blocks.values())
    sections = blocks.items() if nested else [("held out", blocks)]
    for name, results in sections:
        n = next(iter(results.values()))["n"]
        parts.append(f"### {name} ({n} airs)\n")
        parts.append(ev.table(results))
        for w in ev.WINDOWS:
            for head in ("rotation", "sense", "axis", "grab"):
                parts.append("")
                parts.append(ev.confusion_md(results[w][head]["confusion"], ds.HEADS[head],
                                             f"{head}, model, {w} window"))
        parts.append("")
    return "\n".join(parts)


def write(report: dict, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=1))
    dest.with_suffix(".md").write_text(markdown(report))
