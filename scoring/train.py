"""Train the big air score model on data/jumps.csv and report how good it is.

Two models, because a results sheet mixes two different things:

- **landed**: P(the jump is landed) from what was attempted. Falls score in
  the teens regardless of difficulty, so they are not a "low score" for the
  trick and must not be regressed together with clean jumps.
- **score**: the panel's score for a landed jump, from what was attempted.

Following docs/training.md ("The judging score"): folds are split by contest,
never by jump; ranking quality within a contest is reported next to absolute
error because panels score relative to the field; and the ceiling is judge
disagreement, i.e. how far one judge sits from the panel on the same jump.

    uv run python train.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from tricks import decode

HERE = Path(__file__).parent
# Below this a jump is treated as a fall / hand down rather than a scored trick.
# Picked from the valley in the score histogram; `report.json` records the split.
LANDED_MIN = 40.0

NUMERIC = ["spin_deg", "inverts", "switch", "corked", "n_grabs", "final"]
CATEGORICAL = ["direction", "axis", "grab", "gender", "level"]
FEATURES = NUMERIC + CATEGORICAL


# FIS category code (the race label's "| WC") -> competition tier.
LEVELS = {
    "OWG": "olympics", "WSC": "worlds", "WC": "world_cup",
    "WJC": "junior", "NJC": "junior", "YOG": "junior", "EYOF": "junior",
    "EC": "continental", "ECP": "continental", "NAC": "continental",
    "ANC": "continental", "SAC": "continental", "AC": "continental",
}


def level(cat_code: str) -> str:
    return LEVELS.get(cat_code, "fis")


MIN_CATEGORY = 10  # rarer grabs / axes are pooled as "other"


def features(df: pd.DataFrame, categories: dict | None = None) -> pd.DataFrame:
    """Model inputs. `categories` (from a fitted model) fixes each categorical's levels;
    without it they are learned here, pooling anything seen under MIN_CATEGORY times."""
    decoded = pd.DataFrame([decode(t) for t in df["trick"]], index=df.index)
    out = decoded[["spin_deg", "inverts", "n_grabs", "direction", "grab", "axis"]].copy()
    out["switch"] = decoded["switch"].astype(int)
    out["corked"] = decoded["corked"].astype(int)
    out["gender"] = df["gender"].fillna("")
    out["level"] = df["cat_code"].fillna("").map(level)
    out["final"] = df["round"].fillna("").str.contains("Final").astype(int)
    for col in CATEGORICAL:
        values = out[col].astype(str)
        if categories is None:
            counts = values.value_counts()
            keep = sorted(counts[counts >= MIN_CATEGORY].index) + ["other"]
        else:
            keep = categories[col]
        out[col] = pd.Categorical(values.where(values.isin(keep), "other"), categories=keep)
    return out[FEATURES]


def _regressor():
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                         min_samples_leaf=20, categorical_features="from_dtype",
                                         random_state=0)


def _classifier():
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                          min_samples_leaf=20, categorical_features="from_dtype",
                                          random_state=0)


def judge_spread(df: pd.DataFrame) -> float:
    """Mean |one judge - the other judges' mean|: the error of a single extra judge.

    Leave-one-out, so a judge is never compared against a score it helped make.
    """
    errs = []
    for marks in df["judges"]:
        js = np.array([int(j) for j in str(marks).split()], dtype=float)
        if len(js) < 3:
            continue
        others = (js.sum() - js) / (len(js) - 1)
        errs.extend(np.abs(js - others))
    return float(np.mean(errs))


def within_contest_rank(df: pd.DataFrame, pred: np.ndarray) -> float:
    """Mean Spearman rho between predicted and real score inside each contest."""
    rhos = []
    for _, g in df.assign(pred=pred).groupby(["event", "gender"]):
        if len(g) >= 5 and g["score"].nunique() > 1 and g["pred"].nunique() > 1:
            rhos.append(g["score"].rank().corr(g["pred"].rank()))
    return float(np.mean(rhos)) if rhos else float("nan")


def main() -> None:
    df = pd.read_csv(HERE / "data" / "jumps.csv")
    df = df[df["trick"].notna() & df["score"].notna()].reset_index(drop=True)
    X = features(df)
    groups = df["event"].astype(str) + df["gender"].fillna("")
    landed = (df["score"] >= LANDED_MIN).to_numpy()
    folds = GroupKFold(n_splits=min(5, groups.nunique()))

    # Landing model over every jump.
    p_land = np.zeros(len(df))
    for tr, te in folds.split(X, landed, groups):
        p_land[te] = _classifier().fit(X.iloc[tr], landed[tr]).predict_proba(X.iloc[te])[:, 1]

    # Score model over landed jumps only.
    L = df[landed].reset_index(drop=True)
    XL, yL, gL = X[landed].reset_index(drop=True), L["score"].to_numpy(), groups[landed].to_numpy()
    pred = np.zeros(len(L))
    spin_only = np.zeros(len(L))
    for tr, te in folds.split(XL, yL, gL):
        pred[te] = _regressor().fit(XL.iloc[tr], yL[tr]).predict(XL.iloc[te])
        lin = LinearRegression().fit(XL.iloc[tr][["spin_deg"]], yL[tr])
        spin_only[te] = lin.predict(XL.iloc[te][["spin_deg"]])

    report = {
        "jumps": int(len(df)),
        "contests": int(groups.nunique()),
        "landed_min": LANDED_MIN,
        "landed_jumps": int(landed.sum()),
        "landing_auc": float(roc_auc_score(landed, p_land)) if 0 < landed.mean() < 1 else None,
        "score": {
            "mae": float(mean_absolute_error(yL, pred)),
            "r2": float(r2_score(yL, pred)),
            "within_contest_spearman": within_contest_rank(L, pred),
        },
        "baselines": {
            "mean_mae": float(mean_absolute_error(yL, np.full(len(yL), yL.mean()))),
            "spin_only_linear_mae": float(mean_absolute_error(yL, spin_only)),
            "spin_only_within_contest_spearman": within_contest_rank(L, spin_only),
        },
        "ceiling_single_judge_mae": judge_spread(L),
        # The one-number summary: what another half turn is worth to a panel.
        "points_per_180_spin": float(
            LinearRegression().fit(XL[["spin_deg"]], yL).coef_[0] * 180),
    }
    print(json.dumps(report, indent=2))
    (HERE / "data" / "report.json").write_text(json.dumps(report, indent=2))

    model = {
        "landed": _classifier().fit(X, landed),
        "score": _regressor().fit(XL, yL),
        "categories": {c: list(X[c].cat.categories) for c in CATEGORICAL},
        "landed_min": LANDED_MIN,
        # What a fall typically scores, for predict.py's expected value.
        "fall_score": float(df.loc[~landed, "score"].median()),
    }
    joblib.dump(model, HERE / "data" / "model.joblib")
    print("model -> data/model.joblib")


if __name__ == "__main__":
    main()
