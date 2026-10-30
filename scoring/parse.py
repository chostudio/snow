"""Turn cached FIS big air result PDFs into one row per judged jump.

A results sheet lists each rider once per run:

    1 32 9325080 YU Seung Eun KOR R 2008 Run 1 b-DC-10-Wed 90 90 92 92 93 92 91.50 98.25
    1 32 9325080 KOR 2008 Run 2 b-DC-12-Me 96 99 99 99 99 96 98.25

i.e. rider, run number, Trick ID, one integer per judge, the run score (in a
final, with the Jump ID letter glued on: `87.50A`), then best score / totals
which are ignored. Runs marked DNI ("does not improve") often still carry the
judges' marks but no score; the score is recomputed from them with the ICR
3403 rule rather than dropping the jump.

An "Overall Results" sheet repeats the qualification rows of riders who did not
reach the final, so jumps are de-duplicated on (event, gender, rider, run, trick, judges).

    uv run python parse.py          # data/index.json -> data/jumps.csv
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pandas as pd
import pdfplumber

HERE = Path(__file__).parent

_RUN = re.compile(r"\bRun (\d)\b\s*(.*)$")
_FIS_CODE = re.compile(r"\b(\d{7})\b")
_SCORE = re.compile(r"^(\d{1,3}\.\d{2})([A-Z])?")
_FOOTER = re.compile(r"^(\d{1,2} [A-Z]{3} \d{4}) / (.+?) / (\d+)\b")
_ROUND = re.compile(r"^Results from (.+)$|^Results - (.+)$")
_HEAT = re.compile(r"^(Heat \d+)\b")
_STATUS = {"DNI", "DNS", "DNF", "DSQ", "DQ"}


def official_score(judges: list[int]) -> float:
    """ICR 3403: with six or more judges drop the high and low, average, truncate to 2 dp."""
    marks = sorted(judges)
    if len(marks) >= 6:
        marks = marks[1:-1]
    return math.floor(sum(marks) / len(marks) * 100) / 100


def parse_run_tail(tail: str) -> dict | None:
    """`b-DC-12-Me 96 99 99 99 99 96 98.25` -> trick, judges, score, jump id."""
    tokens = tail.split()
    status = next((t for t in tokens if t in _STATUS), "")
    tokens = [t for t in tokens if t not in _STATUS]
    if not tokens:
        return None
    trick = ""
    if not tokens[0].isdigit() and not _SCORE.match(tokens[0]):
        trick = tokens.pop(0)
        # Older sheets put a "*" marker where newer ones print the Trick ID.
        if not re.search(r"[A-Za-z]", trick):
            trick = ""
    judges = []
    while tokens and tokens[0].isdigit():
        judges.append(int(tokens.pop(0)))
    score, jump_id = None, ""
    if tokens and (m := _SCORE.match(tokens[0])):
        score, jump_id = float(m.group(1)), m.group(2) or ""
    if score is None and len(judges) >= 3:
        score = official_score(judges)
    if not trick or score is None:
        return None
    return {"trick": trick, "judges": judges, "score": score,
            "jump_id": jump_id, "status": status}


def parse_text(text: str) -> list[dict]:
    """All jumps on one sheet. Header fields are attached to every jump."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    header = {"category": lines[0] if lines else "", "gender": "", "date": "",
              "place": "", "codex": None}
    for ln in lines:
        if ("Women's" in ln or "Ladies'" in ln) and "Big Air" in ln:
            header["gender"] = "W"
        elif "Men's" in ln and "Big Air" in ln:
            header["gender"] = "M"
        if m := _FOOTER.match(ln):
            header.update(date=m.group(1), place=m.group(2), codex=int(m.group(3)))
    jumps, rnd, heat, rider = [], "", "", {}
    for ln in lines:
        if m := _ROUND.match(ln):
            # "Qualification Heat 1" on its own sheet is "Qualification" on the overall one.
            label = (m.group(1) or m.group(2)).strip()
            rnd = next((r for r in ("Semifinal", "Final", "Qualification") if r.lower() in label.lower()), label)
            heat = h.group(1) if (h := re.search(r"(Heat \d+)", label)) else ""
            continue
        if m := _HEAT.match(ln):
            heat = m.group(1)
            continue
        run = _RUN.search(ln)
        if not run:
            continue
        head = ln[:run.start()].split()
        code = _FIS_CODE.search(" ".join(head))
        if not code:
            continue
        if code.group(1) != rider.get("fis_code"):
            rider = {"fis_code": code.group(1), "name": "", "nation": "", "stance": ""}
        after = head[head.index(code.group(1)) + 1:]
        # NAME... NSA [R|G] YB on a rider's first line; NSA YB on later ones.
        nsa = [i for i, t in enumerate(after) if re.fullmatch(r"[A-Z]{3}", t)]
        if nsa:
            i = nsa[-1]
            rider["nation"] = after[i]
            if after[:i]:
                rider["name"] = " ".join(after[:i])
            if i + 1 < len(after) and after[i + 1] in ("R", "G"):
                rider["stance"] = after[i + 1]
        parsed = parse_run_tail(run.group(2))
        if not parsed:
            continue
        jumps.append({**header, "round": rnd, "heat": heat, **rider,
                      "run": int(run.group(1)), **parsed})
    return jumps


def parse_pdf(path: Path) -> list[dict]:
    with pdfplumber.open(path) as pdf:
        text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    if "Big Air" not in text:
        return []
    return parse_text(text)


def build(index_path: Path = HERE / "data" / "index.json") -> pd.DataFrame:
    index = json.loads(index_path.read_text())
    # A race label ends in its category code ("... | WC"). Qualification is often its
    # own race coded QUA, so a race takes the code of its event's main race.
    event_code = {}
    for entry in index:
        code = entry["label"].rsplit("|", 1)[-1].strip()
        if code != "QUA":
            event_code.setdefault(entry["event"], code)
    rows, without = [], 0
    for entry in index:
        cat_code = event_code.get(entry["event"], "QUA")
        jumps = parse_pdf(HERE / entry["pdf"])
        # Sheets before the 2021 season print no Trick ID column at all.
        without += not jumps
        rows += [{"season": entry["season"], "event": entry["event"], "race": entry["race"],
                  "cat_code": cat_code, **j}
                 for j in jumps]
    print(f"{without} result sheets had no Trick IDs (older layout) and were skipped")
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["n_judges"] = df["judges"].str.len()
    df["judges"] = df["judges"].map(lambda j: " ".join(map(str, j)))
    # The same jump is re-listed by summary sheets, which may drop the heat, the
    # round header (Olympic summaries) or even mislabel the round, so the key is the
    # jump's fingerprint: rider, run, trick and every judge's mark. The copy from the
    # most specific sheet (heat named, then round named) is kept.
    df = (df.assign(_spec=df["heat"].ne("").astype(int) * 2 + df["round"].ne("").astype(int))
            .sort_values("_spec", ascending=False, kind="stable")
            .drop_duplicates(["event", "gender", "fis_code", "run", "trick", "judges"])
            .drop(columns="_spec").sort_index())
    return df.reset_index(drop=True)


if __name__ == "__main__":
    df = build()
    out = HERE / "data" / "jumps.csv"
    df.to_csv(out, index=False)
    contests = df.groupby(["event", "gender"]).ngroups if len(df) else 0
    print(f"{len(df)} jumps from {contests} contests -> {out.relative_to(HERE)}")
