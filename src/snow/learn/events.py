"""Airborne masks, the airs they contain, and how they are scored.

Numpy only, so the metrics and the clean-up run without torch.

Frames are indexed by row of the features table. An air is a half-open run
``[takeoff, landing)``: ``takeoff`` is the first airborne frame and
``landing`` the first frame back on the snow, which is how the synthetic
``labels.json`` counts ``air_frames = landing - takeoff``.
"""

from __future__ import annotations

import numpy as np

# The same frame counts as the dashboard's detector (web/tools/export_session.py),
# so the two are compared on equal clean-up.
MIN_AIR_FRAMES = 8
MERGE_GAP_FRAMES = 10


def runs(mask: np.ndarray, segment: np.ndarray | None = None) -> list[tuple[int, int]]:
    """``[start, end)`` of every True run; a run never continues across a cut."""
    mask = np.asarray(mask, dtype=bool)
    if segment is None:
        segment = np.zeros(len(mask), dtype=int)
    out: list[tuple[int, int]] = []
    start = None
    for i, on in enumerate(mask):
        if start is not None and (not on or segment[i] != segment[start]):
            out.append((start, i))
            start = None
        if on and start is None:
            start = i
    if start is not None:
        out.append((start, len(mask)))
    return out


def clean_mask(mask: np.ndarray, segment: np.ndarray | None = None,
               min_frames: int = MIN_AIR_FRAMES, merge_gap: int = MERGE_GAP_FRAMES) -> np.ndarray:
    """Bridge short gaps inside one segment, then drop runs too short to be an air."""
    mask = np.asarray(mask, dtype=bool).copy()
    if segment is None:
        segment = np.zeros(len(mask), dtype=int)
    spans = runs(mask, segment)
    for (a0, a1), (b0, _) in zip(spans, spans[1:]):
        if b0 - a1 <= merge_gap and segment[a1 - 1] == segment[b0]:
            mask[a1:b0] = True
    out = np.zeros_like(mask)
    for a, b in runs(mask, segment):
        if b - a >= min_frames:
            out[a:b] = True
    return out


def mask_from_spans(n: int, spans: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    for a, b in spans:
        mask[max(0, a):min(n, b)] = True
    return mask


def frame_counts(pred: np.ndarray, truth: np.ndarray) -> dict:
    pred, truth = np.asarray(pred, bool), np.asarray(truth, bool)
    return {"tp": int((pred & truth).sum()), "fp": int((pred & ~truth).sum()),
            "fn": int((~pred & truth).sum()), "tn": int((~pred & ~truth).sum())}


def f1(counts: dict) -> float:
    denom = 2 * counts["tp"] + counts["fp"] + counts["fn"]
    return 2 * counts["tp"] / denom if denom else 1.0


def boundary_errors(pred: np.ndarray, truth_spans: list[tuple[int, int]],
                    segment: np.ndarray | None = None) -> list[dict]:
    """For each true air, the predicted air overlapping it most.

    Signed errors in frames, predicted minus true: ``takeoff_err`` and
    ``landing_err``. A true air no predicted run touches is ``missed``.
    Predicted runs that touch no true air are counted as ``false_airs`` on
    the first entry (so a clip's total is easy to sum).
    """
    spans = runs(pred, segment)
    used: set[int] = set()
    out = []
    for t0, t1 in truth_spans:
        overlaps = [(min(b, t1) - max(a, t0), k) for k, (a, b) in enumerate(spans)]
        best = max(overlaps, default=(0, -1))
        if best[0] <= 0:
            out.append({"missed": True, "takeoff_err": None, "landing_err": None})
            continue
        a, b = spans[best[1]]
        for k, (c, d) in enumerate(spans):
            if min(d, t1) - max(c, t0) > 0:
                used.add(k)
        out.append({"missed": False, "takeoff_err": a - t0, "landing_err": b - t1,
                    "pieces": sum(1 for c, d in spans if min(d, t1) - max(c, t0) > 0)})
    false_airs = sum(1 for k in range(len(spans)) if k not in used)
    if out:
        out[0]["false_airs"] = false_airs
    return out
