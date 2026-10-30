"""Predict a big air score for one trick.

    uv run python predict.py b-DC-16-Wed --gender M
    uv run python predict.py x-b-12-Jp --gender W --level olympics --final

`expected` folds the landing risk in: P(land) * score-if-landed plus
(1 - P(land)) * the typical fall score.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from train import LEVELS, features

HERE = Path(__file__).parent
CODE_FOR_LEVEL = {name: code for code, name in reversed(list(LEVELS.items()))} | {"fis": "FIS"}


def predict(trick: str, gender: str = "M", level: str = "world_cup", final: bool = False,
            model: dict | None = None) -> dict:
    model = model or joblib.load(HERE / "data" / "model.joblib")
    row = pd.DataFrame({"trick": [trick], "gender": [gender], "cat_code": [CODE_FOR_LEVEL[level]],
                        "round": ["Final" if final else "Qualification"]})
    X = features(row, model["categories"])
    p = float(model["landed"].predict_proba(X)[0, 1])
    score = float(model["score"].predict(X)[0])
    return {"trick": trick, "score_if_landed": round(score, 2), "p_land": round(p, 3),
            "expected": round(p * score + (1 - p) * model["fall_score"], 2)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("trick", nargs="+", help="FIS Trick ID(s), e.g. b-DC-16-Wed")
    ap.add_argument("--gender", default="M", choices=["M", "W"])
    ap.add_argument("--level", default="world_cup",
                    choices=["olympics", "worlds", "world_cup", "continental", "junior", "fis"])
    ap.add_argument("--final", action="store_true")
    args = ap.parse_args()
    model = joblib.load(HERE / "data" / "model.joblib")
    for t in args.trick:
        print(predict(t, args.gender, args.level, args.final, model))
