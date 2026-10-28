"""Score the classifier, head by head, against exact labels and the geometry.

Two windows per air, so the cost of finding the air is visible on its own:

``gt``
    the true takeoff to landing frames. What the classifier can do when the
    air is right; an upper bound until the segmenter exists.
``heuristic``
    the main air the dashboard's detector finds (``web/tools/
    export_session.py``). What a user would see today. An air the detector
    misses is scored wrong for the model and the baseline alike.

The baseline on each window is the current geometric measurement over the
same frames (``dataset.geometric``): ``spin_path_deg`` rounded to 180 for the
spin, the sign of net vertical yaw for the sense (the dashboard's choice; the
signed path's sign is reported too), and the number of inversions in
``tilt_deg`` for the axis. Grab and stance have no geometric reading, so their
baseline is the majority class of the training labels.

Every number here is on synthetic renders. It is not a claim about broadcast
footage (docs/validation.md).
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from . import airs
from . import dataset as ds

WINDOWS = ("gt", "heuristic")


def window_of(air: ds.Air, which: str) -> tuple[int, int] | None:
    return air.gt if which == "gt" else air.heuristic


def predict(clf, records: list[ds.Air], which: str) -> list[dict | None]:
    """Per air, the predicted class index per head, or None if no air/poses."""
    xs, xms, slots = [], [], []
    out: list[dict | None] = [None] * len(records)
    for i, a in enumerate(records):
        w = window_of(a, which)
        if w is None:
            continue
        x = airs.encode(a.track, *w)
        xm = airs.encode(airs.mirror(a.track), *w)
        if x is None or xm is None:
            continue
        xs.append(x)
        xms.append(xm)
        slots.append(i)
    if xs:
        probs = clf.probabilities(xs, xms)
        for k, i in enumerate(slots):
            out[i] = {h: int(np.argmax(probs[h][k])) for h in ds.HEADS}
            out[i]["rotation_reg"] = nearest_rotation(float(probs["turns"][k]) * 360.0)
            out[i]["_p"] = {h: probs[h][k].tolist() for h in ds.HEADS}
    return out


def nearest_rotation(deg: float) -> int:
    """Class index of the multiple of 180 nearest ``deg`` (sign ignored)."""
    return int(np.clip(round(abs(deg) / 180.0), 0, len(ds.ROTATIONS) - 1))


def baseline(air: ds.Air, which: str, majority: dict[str, int]) -> dict:
    """The geometric reading of one window, as class indices per head."""
    b = air.baseline.get(which) or {}
    out: dict = {}
    spin = b.get("spin_deg")
    if spin is not None:
        out["rotation"] = ds.ROTATIONS.index(min(int(spin), ds.ROTATIONS[-1]))
    yaw = b.get("yaw_deg")
    sense = b.get("sense")
    if which == "gt" and yaw:
        out["sense"] = ds.SENSES.index("counter-clockwise" if yaw > 0 else "clockwise")
    elif sense:
        out["sense"] = ds.SENSES.index(sense)
    if sense:
        out["sense_path"] = ds.SENSES.index(sense)
    if b.get("axis"):
        out["axis"] = ds.AXES.index(b["axis"])
    out["grab"] = majority["grab"]
    out["stance"] = majority["stance"]
    return out


def majority_labels(records: list[ds.Air]) -> dict[str, int]:
    return {h: Counter(a.labels[h] for a in records).most_common(1)[0][0] for h in ds.HEADS}


def confusion(truth: list[int], got: list[int | None], classes) -> np.ndarray:
    """Rows true, columns predicted; the extra last column counts no answer."""
    m = np.zeros((len(classes), len(classes) + 1), dtype=int)
    for t, g in zip(truth, got):
        m[t, len(classes) if g is None else g] += 1
    return m


def score(records: list[ds.Air], predictions: list[dict | None],
          baselines: list[dict]) -> dict:
    """Per head: model and baseline accuracy, and the model's confusion matrix."""
    out: dict = {"n": len(records)}
    for head, classes in ds.HEADS.items():
        truth = [a.labels[head] for a in records]
        got = [p[head] if p else None for p in predictions]
        base = [b.get(head) for b in baselines]
        out[head] = {
            "model": _acc(truth, got), "baseline": _acc(truth, base),
            "confusion": confusion(truth, got, classes).tolist(),
            "baseline_confusion": confusion(truth, base, classes).tolist(),
        }
    truth_rot = [a.labels["rotation"] for a in records]
    reg = [p["rotation_reg"] if p else None for p in predictions]
    out["rotation"]["model_regressed"] = _acc(truth_rot, reg)
    out["rotation"]["regressed_confusion"] = confusion(truth_rot, reg, ds.ROTATIONS).tolist()
    truth = [a.labels["sense"] for a in records]
    out["sense"]["baseline_path"] = _acc(truth, [b.get("sense_path") for b in baselines])
    # Rotation: also how far off, in degrees, and within one step.
    for key, src in (("model", [p["rotation"] if p else None for p in predictions]),
                     ("baseline", [b.get("rotation") for b in baselines])):
        err = [abs(ds.ROTATIONS[g] - ds.ROTATIONS[a.labels["rotation"]])
               for a, g in zip(records, src) if g is not None]
        out["rotation"][f"{key}_mae_deg"] = round(float(np.mean(err)), 1) if err else None
        out["rotation"][f"{key}_within_180"] = _acc(
            [0] * len(records),
            [None if g is None else
             (0 if abs(g - a.labels["rotation"]) <= 1 else 1)
             for a, g in zip(records, src)])
    # Frontside/backside, composed from sense and stance.
    truth_dir = [ds.direction(a.labels["sense"], a.labels["stance"]) for a in records]
    got_dir = [ds.direction(p["sense"], p["stance"]) if p else None for p in predictions]
    out["direction"] = {"model": _acc(truth_dir, got_dir)}
    # The whole trick: spin, sense and axis together.
    whole = [p is not None and all(p[h] == a.labels[h] for h in ("rotation", "sense", "axis"))
             for a, p in zip(records, predictions)]
    whole_b = [all(b.get(h) == a.labels[h] for h in ("rotation", "sense", "axis"))
               for a, b in zip(records, baselines)]
    out["spin_sense_axis"] = {"model": [int(sum(whole)), len(whole)],
                              "baseline": [int(sum(whole_b)), len(whole_b)]}
    return out


def _acc(truth, got) -> list[int]:
    """[right, total]: an answer of None is wrong."""
    return [int(sum(g is not None and g == t for t, g in zip(truth, got))), len(truth)]


def per_air(records: list[ds.Air], predictions: list[dict | None],
            baselines: list[dict]) -> list[dict]:
    rows = []
    for a, p, b in zip(records, predictions, baselines):
        row = {"name": a.name, "split": a.split}
        for head, classes in ds.HEADS.items():
            row[f"true_{head}"] = classes[a.labels[head]]
            row[f"model_{head}"] = classes[p[head]] if p else None
            row[f"baseline_{head}"] = classes[b[head]] if b.get(head) is not None else None
        rows.append(row)
    return rows


# --- markdown --------------------------------------------------------------------

def _frac(pair) -> str:
    right, total = pair
    return f"{right}/{total} ({100 * right / total:.0f}%)" if total else "–"


def table(results: dict[str, dict]) -> str:
    """Model against baseline per head, one column pair per window."""
    windows = list(results)
    head = "| head | " + " | ".join(f"model, {w} | baseline, {w}" for w in windows) + " |"
    lines = [head, "|---" * (1 + 2 * len(windows)) + "|"]
    for name in ("rotation", "sense", "stance", "direction", "axis", "grab",
                 "spin_sense_axis"):
        cells = []
        for w in windows:
            r = results[w][name]
            cells += [_frac(r["model"]), _frac(r["baseline"]) if "baseline" in r else "–"]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    for extra, label in (("within_180", "rotation within 180"),
                         ("mae_deg", "rotation mean abs error (deg)")):
        cells = []
        for w in windows:
            r = results[w]["rotation"]
            for who in ("model", "baseline"):
                v = r[f"{who}_{extra}"]
                cells.append(_frac(v) if isinstance(v, list) else str(v))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    cells = []
    for w in windows:
        cells += [_frac(results[w]["rotation"]["model_regressed"]), ""]
    lines.append("| rotation, regressed turns rounded | " + " | ".join(cells) + " |")
    cells = []
    for w in windows:
        cells += ["", _frac(results[w]["sense"]["baseline_path"])]
    lines.append("| sense, signed-path sign | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def confusion_md(matrix, classes, title: str) -> str:
    m = np.asarray(matrix)
    used = [i for i in range(len(classes)) if m[i].sum() or m[:, i].sum()]
    cols = used + ([len(classes)] if m[:, -1].sum() else [])
    names = [str(classes[i]) if i < len(classes) else "none" for i in cols]
    lines = [f"**{title}** (rows true, columns predicted)", "",
             "| | " + " | ".join(names) + " |", "|---" * (len(cols) + 1) + "|"]
    for i in used:
        lines.append(f"| {classes[i]} | " + " | ".join(str(m[i, j]) for j in cols) + " |")
    return "\n".join(lines)
