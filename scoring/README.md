# scoring: big air score prediction

Given a trick (spin, direction, switch, inverts, grab), predict what an FIS
judging panel scores it. Trained on every scored jump in the official FIS big
air result sheets. This is Model C from
[docs/training.md](../docs/training.md#the-judging-score), started from the
labels end: the features come from the published Trick ID today, and from the
pipeline's measurements once the trick classifier exists.

Self-contained: its own `pyproject.toml` and its own `data/` (gitignored by
the repo's `data/` rule). Nothing in `src/snow/` or `web/` imports it.

```
uv run python fis.py --seasons 2019-2026   # crawl + cache result PDFs  -> data/raw/, data/index.json
uv run python parse.py                     # PDFs -> one row per jump   -> data/jumps.csv
uv run python train.py                     # fit + cross-validate       -> data/model.joblib, data/report.json
uv run python predict.py b-DC-16-Wed --gender M --final
uv run --extra dev pytest
```

## The data

FIS publishes a results PDF for every big air contest (World Cup, Worlds,
Olympics, continental cups, FIS races). Each run row carries the rider, the
**Trick ID** (`b-DC-10-Wed` = backside double cork 1080 weddle), every judge's
integer mark and the panel score:

```
1 32 9325080 YU Seung Eun KOR R 2008 Run 2 b-DC-12-Me 96 99 99 99 99 96 98.25
```

`fis.py` reaches them the way a browser would (season calendar, event, race,
"Results - ..." attachment) from www.fis-ski.com. The `medias*.fis-ski.com`
mirror that search engines index returned 502 throughout 2026-10-28, so it is
not used. Requests are cached and spaced one second apart.

| file | holds |
|---|---|
| `tricks.py` | Trick ID -> `spin_deg`, `direction`, `switch`, `inverts`, `corked`, `grab` |
| `parse.py` | PDF text -> jumps; recomputes DNI scores from the marks (ICR 3403) |
| `train.py` | landing classifier + score regressor, contest-grouped CV, report |
| `predict.py` | one trick -> score if landed, P(land), expected score |

## Current numbers (crawl of 2026-10-28)

543 big air result sheets, seasons 2019 to 2026. Trick IDs only appear from
the 2021 season on, and mostly at World Cup level and above, so 407 sheets
drop out. That leaves **4,995 unique jumps from 57 contests** (World Cup,
Worlds, Olympics 2022 and 2026, junior worlds); 3,048 of them landed.

Contest-grouped 5-fold CV, landed jumps (`data/report.json`):

| | MAE | within-contest Spearman |
|---|---|---|
| predict the mean | 11.2 | n/a |
| spin only (linear) | 10.2 | 0.65 |
| **trick model** | **7.8** | **0.68** |
| one judge vs the rest of the panel (ceiling) | 1.95 | n/a |

Each extra 180 of spin is worth about 2.2 points (pooled). Landing is barely
predictable from the trick (AUC 0.59): whether a jump is landed is execution,
which a results sheet does not describe.

Read that gap honestly: the trick explains about 40% of the variance in a
landed score. The rest is execution and amplitude, which only the video
can supply, so that is where the pipeline's measurements come in.

## What the model is, and what it is not

- **Two models.** A fall scores in the teens whatever was attempted, so
  falls are not "low scores for that trick". `landed` predicts P(land);
  `score` predicts the panel score given a landing. A jump under
  `LANDED_MIN` (40) counts as a fall.
- **Inputs are what was attempted, not how well.** Execution and amplitude
  are not on the sheet, so they are the bulk of the remaining error. The
  pipeline can add them later (air time, grab hold, landing stability) as
  extra columns.
- **Evaluated the way docs/training.md asks:** folds split by contest, never
  by jump; ranking inside each contest (Spearman) reported next to MAE,
  because panels score relative to the field; and the ceiling is a single
  judge's distance from the panel on the same jumps. `report.json` has all
  three against a spin-only baseline.

## Known biases

- **DNI rows.** "Does not improve" qualification runs usually print no marks,
  so the worse of a rider's two qualification runs is often missing. Scores
  skew upward in qualification; the `final` flag partly absorbs this.
- **Trick IDs are the judges' call.** They are entered by the scoring crew,
  so they are a human label (which is what docs/training.md requires), but
  they are hand-typed: `bT14Wed`, `xbs12Tg`, `d` for `D`. `tricks.tokenize`
  repairs those; grabs and axes seen fewer than 10 times are pooled as `other`.
- **Re-listed jumps.** Summary sheets repeat qualification rows, sometimes
  without a round header (Olympics) or under the wrong round. Jumps are
  de-duplicated on rider, run, trick and every judge's mark; 142 jumps keep
  no round and count as non-final.
- **Women's triples are thin.** Few women's triple corks are on record, so
  the model leans on men's data for them.
- **Contest anchoring.** Absolute scores drift between contests; `level`
  and `final` carry some of that, the rest is in the error.
