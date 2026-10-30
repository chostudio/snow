"""Decode an FIS Trick ID into features a model, or the pipeline, can supply.

A Trick ID is dash-separated: takeoff, then any inverts / axis, then the spin
in hundreds of degrees, then grabs (chained with `to`):

    b-DC-10-Wed        backside double cork 1080, weddle
    x-b-12-Jp          switch backside 1260, japan
    Cab-D-9-Wed-to-Tg  cab (switch frontside) double 900, weddle to tail
    f-Uf-7             frontside underflip 720, no grab
    Str-A              straight air

Codes come from the FIS Park & Pipe trick list
(fis-ski.com/DB/snowboard/park-and-pipe/tricks-list.html); tokens not listed
here still reach the model through `tokens`, they just get no hand-written
meaning.

The derived fields deliberately mirror what the video pipeline measures
(`axis_spin_deg` -> spin_deg, sense of rotation -> direction, inverts), so the
same model can score a trick read off a clip instead of off a results sheet.
"""

from __future__ import annotations

import re

# Takeoff tokens. Cab is a switch frontside spin, so it sets both fields.
SWITCH = {"x", "Cab", "Fa"}
FRONTSIDE = {"f", "Cab", "fs"}
BACKSIDE = {"b", "bs"}

# Inverts / axis tokens -> number of inverted rotations they imply.
INVERTS = {
    "C": 1, "Rd": 1, "Uf": 1, "Mc": 1, "Rl": 1, "Bf": 1, "Ff": 1, "Wf": 1,
    "D": 2, "DC": 2, "DR": 2, "DRd": 2,
    "T": 3, "TC": 3, "TR": 3,
    "Q": 4, "QC": 4,
}
NOT_GRABS = SWITCH | FRONTSIDE | BACKSIDE | set(INVERTS) | {"to", "A", "Str", "No", "Rg"}

# Scoring crews type codes by hand: "d" for D, "X" for x, "UF" for Uf.
_CANON = {t.lower(): t for t in SWITCH | FRONTSIDE | BACKSIDE | set(INVERTS)}

_SPIN = re.compile(r"^\d{1,2}$")
# Pieces of a code typed without dashes: "bT14Wed" -> b, T, 14, Wed.
_PIECE = re.compile(r"\d+|[A-Z][a-z]*|[a-z]+")


def tokenize(trick: str) -> list[str]:
    """Dash-split, un-glue codes typed without dashes, canonicalise takeoff/axis case."""
    tokens = []
    for raw in trick.split("-"):
        if not raw:
            continue
        # Only a token mixing letters and a spin is glued; "Ddr" or "Tdr" are grabs.
        pieces = _PIECE.findall(raw) if re.search(r"\d", raw) and re.search(r"[A-Za-z]", raw) else [raw]
        for p in pieces:
            # A leading lowercase run like "xbs" is switch + direction.
            if p.startswith("x") and len(p) > 1 and p[1:] in FRONTSIDE | BACKSIDE:
                tokens += ["x", p[1:]]
            else:
                tokens.append(p if p in NOT_GRABS else _CANON.get(p.lower(), p))
    return tokens


def spin_degrees(hundreds: int) -> int:
    """`10` -> 1080, `16` -> 1620, `23` -> 2340: the nearest multiple of 180."""
    return round(hundreds * 100 / 180) * 180


def decode(trick: str) -> dict:
    tokens = tokenize(trick)
    spin_at = next((i for i, t in enumerate(tokens) if _SPIN.match(t)), None)
    spin_deg = spin_degrees(int(tokens[spin_at])) if spin_at is not None else 0
    before = tokens[:spin_at] if spin_at is not None else tokens
    after = tokens[spin_at + 1:] if spin_at is not None else []

    direction = ("frontside" if FRONTSIDE & set(before)
                 else "backside" if BACKSIDE & set(before) else "none")
    # "T-C" (triple cork) is written as two tokens; T already carries the count.
    inverts = max((INVERTS.get(t, 0) for t in tokens), default=0)
    grabs = [t for t in after if t not in NOT_GRABS]
    return {
        "spin_deg": spin_deg,
        "direction": direction,
        "switch": bool(SWITCH & set(before)),
        "inverts": inverts,
        # The invert part of the code, e.g. "DC", "T-C", "D-Rd"; "flat" for a plain spin.
        "axis": "-".join(t for t in before if t in INVERTS) or "flat",
        "corked": "C" in tokens or any(t.endswith("C") and t in INVERTS for t in tokens),
        "grab": grabs[0] if grabs else "none",
        "n_grabs": len(grabs),
        "tokens": tokens,
    }
