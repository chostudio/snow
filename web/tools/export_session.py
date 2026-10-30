#!/usr/bin/env python3
"""Build a browser-ready session bundle from a clip's pose output.

    ../.venv/bin/python tools/export_session.py run1
    ../.venv/bin/python tools/export_session.py run1 --classify   # + model B, if trained

Reads ``data/poses/<name>.json`` (JSONL, one frame per line) plus the clip
itself and writes ``web/public/sessions/<name>/`` — a quantised pose track, a
telemetry series, detected air events, and a copy of the video. The frontend is
a pure view layer over this file; nothing in it recomputes kinematics.

Rotation comes from ``snow features`` when that has been run — the CSV it
writes beside the poses carries camera-compensated spin, which is the number
that actually means "720". Without it this falls back to a DISPLAY SHIM:
rotation about the rider's own body axis, far less sensitive to a panning
broadcast camera than root yaw but not a substitute for subtracting the camera.
``source.metrics`` says which one is on screen, and the UI badges it.

The coach report in the bundle is a rule-based template unless you ask for
the real one, which calls Nemotron on NVIDIA's API (``src/snow/coach.py``) so
the static build has it without ``snow serve``:

    ../.venv/bin/python tools/export_session.py run1 --coach nemotron
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
WEB = REPO / "web"
sys.path.insert(0, str(REPO / "src"))
from snow import features as snow_features  # noqa: E402  (after sys.path edit)
from snow.skeleton import LINKS, PARENTS  # noqa: E402

# Nova-77 indices this UI names. Handedness follows the skeleton's own index
# order; every displayed metric is built to be symmetric so a swapped L/R
# cannot change a number.
J = {
    "pelvis": 0, "spine": 3, "neck": 6, "head": 7,
    "shoulder_l": 12, "elbow_l": 13, "wrist_l": 14,
    "shoulder_r": 40, "elbow_r": 41, "wrist_r": 42,
    "hip_l": 67, "knee_l": 68, "ankle_l": 69, "toe_l": 71,
    "hip_r": 72, "knee_r": 73, "ankle_r": 74, "toe_r": 76,
}
CAM_UP = np.array([0.0, -1.0, 0.0])  # NIM camera frame is X right, Y down, Z forward

# Air detection: find a core of confidently fast frames, then grow it outward
# while the rider is still turning the same way.
#
# Both gates are fractions of the clip's OWN peak rate rather than absolute
# RPM. An absolute gate is only meaningful on footage shot at real time:
# broadcast slow motion is conformed to 30 fps and carries no trace of its own
# factor, so `real_fps` is a guess and every real-world rate scales with it. A
# fixed 18 RPM gate silently rejected most of a slow-motion spin — it clipped a
# continuous 600° air down to the 142° sliver that happened to exceed it. A
# within-clip fraction is invariant to that unknown factor.
ENTER_FRACTION = 0.45
EXIT_FRACTION = 0.18
# Not max(), and not the 99th: where the NIM's pose flips as the rider goes
# over, a handful of frames read 5-15x the real rate, and on synthetic double
# corks those few frames set the 99th percentile and gated out the spin.
PEAK_PERCENTILE = 95
# Floors stop a clip holding no trick from scaling its gate down onto its own
# noise. MIN_NET_ROTATION_DEG is what actually rejects those.
RPM_ENTER_FLOOR = 8.0
RPM_EXIT_FLOOR = 4.0
MIN_AIR_FRAMES = 8
MERGE_GAP_FRAMES = 10
# An air is followed across a hole in the pose track up to this long, inside
# one shot. The tracker still loses an inverted rider for 10-20 frames at a
# time, and ending the air there split one trick into two.
HOLE_FRAMES = 20
# With the body orientation available, a hole may also be this many of the
# clip's own chord spans (``features.robust_speed``): a slowed clip loses
# the inverted rider for k times the frames, and 20 frames at 4x is five
# real ones.
HOLE_SPANS = 4
# Rotation against the air's sense at either end, read by both the signed
# path and the vertical yaw, is a wind-up or a landing check, not the trick;
# past this much it is trimmed off.
COUNTER_TURN_DEG = 30.0
# A short burst can hit a high instantaneous rate while netting no rotation —
# that is tracking noise oscillating, not a spin, so require real net turn.
MIN_NET_ROTATION_DEG = 120.0


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(np.dot(unit(a), unit(b)), -1.0, 1.0))))


def probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams",
         "-select_streams", "v:0", str(path)],
        capture_output=True, text=True, check=True,
    )
    s = json.loads(out.stdout)["streams"][0]
    num, den = (int(x) for x in s["r_frame_rate"].split("/"))
    return {
        "fps": num / den,
        "frames": int(s.get("nb_frames") or 0),
        "width": int(s["width"]),
        "height": int(s["height"]),
        "duration": float(s.get("duration") or 0.0),
    }


def body_frame(k3: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The rider's up axis and hip axis, the hip axis made perpendicular to up."""
    hip_mid = (k3[J["hip_l"]] + k3[J["hip_r"]]) / 2
    sho_mid = (k3[J["shoulder_l"]] + k3[J["shoulder_r"]]) / 2
    up = unit(sho_mid - hip_mid)
    hip_ax = k3[J["hip_r"]] - k3[J["hip_l"]]
    return up, unit(hip_ax - np.dot(hip_ax, up) * up)


def features_path(name: str) -> Path:
    return REPO / "data" / "poses" / f"{name}.features.csv"


# Which rotation column to put on screen, best first. `spin_deg` (yaw about
# world vertical) is deliberately not in this list as the headline: it
# degenerates exactly when a trick is corked, which is when it matters.
# `axis_spin_deg` was the headline until synthetic tricks with exact labels
# showed it undercounts corks even on perfect poses (docs/validation.md);
# the signed angular path counts them, and its sign is the turn's sense.
ROTATION_PREFERENCE = [
    ("spin_path_deg", "signed angular path, camera removed"),
    ("axis_spin_deg", "rotation about the trick's own axis"),
    ("rotation_deg", "total angular path, all axes"),
]


def load_features(name: str) -> tuple[dict[int, dict], str, str] | None:
    """Per-frame rows from ``snow features``, camera compensation included.

    Read defensively: that module is under active development, so a column
    that is not there yet degrades to the next choice rather than failing.
    """
    path = features_path(name)
    if not path.exists():
        return None
    import pandas as pd

    df = pd.read_csv(path)
    if "frame_id" not in df:
        return None
    pick = next(((c, label) for c, label in ROTATION_PREFERENCE if c in df), None)
    if pick is None:
        return None
    column, label = pick

    df = df[df[column].notna()]
    out: dict[int, dict] = {}
    camera_cum = 0.0
    yaw = 0.0
    for row in df.to_dict("records"):
        value = row.get("camera_rotation_deg")
        camera_cum += float(value) if value == value else 0.0  # NaN-safe
        tilt = row.get("tilt_deg")
        spin = row.get("spin_deg")
        yaw = float(spin) if spin is not None and spin == spin else yaw
        path = row.get("rotation_deg")
        quat = [row.get(c) for c in snow_features.ORIENTATION_COLUMNS]
        prob = row.get("airborne_prob")
        learned = prob is not None and prob == prob  # NaN-safe
        out[int(row["frame_id"])] = {
            # Cumulative WITHIN a camera segment: it restarts at every cut.
            "segment_rotation": float(row[column]),
            # Unsigned path and vertical yaw, for finding the air and its
            # sense (see find_events); absent, the signed column stands in.
            "segment_path": float(path) if path is not None and path == path else None,
            "segment_yaw": yaw if spin is not None else None,
            "segment": int(row.get("segment", 0) or 0),
            "axis_tilt": float(tilt) if tilt == tilt else 0.0,
            "camera_cum": camera_cum,
            # Camera-compensated body orientation, for counting the spin
            # (``features.spin_count_deg``); absent in an older table.
            **({"q": [float(v) for v in quat]}
               if all(v is not None and v == v for v in quat) else {}),
            # The learned segmenter's call, when `snow features` ran with
            # weights (docs/segmenter.md); absent, the rate detector decides.
            **({"airborne": bool(row.get("airborne_hint"))} if learned else {}),
        }
    return out, column, label


def per_frame(frames: list[tuple[int, dict]]) -> list[dict]:
    recs = []
    for fid, det in frames:
        k3 = np.array(det["keypoints_3d"], dtype=float)
        up, hip_ax = body_frame(k3)
        ankle_y = min(k3[J["ankle_l"]][1], k3[J["ankle_r"]][1])  # Y is down
        recs.append({
            "f": fid,
            "_up": up,
            "_hip_ax": hip_ax,
            # Height above camera centre line, so up is positive for a viewer.
            "ankle_height": -float(ankle_y),
            "axis_tilt": angle_between(up, CAM_UP),
            "tuck_radius": float(
                np.linalg.norm(k3[J["wrist_l"]] - k3[J["spine"]])
                + np.linalg.norm(k3[J["wrist_r"]] - k3[J["spine"]])
            ) / 2,
            "knee_flex": (
                angle_between(k3[J["hip_l"]] - k3[J["knee_l"]], k3[J["ankle_l"]] - k3[J["knee_l"]])
                + angle_between(k3[J["hip_r"]] - k3[J["knee_r"]], k3[J["ankle_r"]] - k3[J["knee_r"]])
            ) / 2,
            "conf": float(np.mean(det["keypoints_confidence"])),
        })
    return recs


def integrate_rotation(recs: list[dict], real_fps: float) -> None:
    """Signed rotation about the rider's own up axis, accumulated frame to frame.

    Only consecutive frames contribute — a camera cut leaves a hole in the
    track, and inventing the rotation across it would invent a trick.
    """
    cum = 0.0
    prev: dict | None = None
    for r in recs:
        if prev is not None and r["f"] - prev["f"] == 1:
            up = unit((r["_up"] + prev["_up"]) / 2)
            a = unit(prev["_hip_ax"] - np.dot(prev["_hip_ax"], up) * up)
            b = unit(r["_hip_ax"] - np.dot(r["_hip_ax"], up) * up)
            cum += float(np.degrees(np.arctan2(float(np.dot(np.cross(a, b), up)),
                                               np.clip(float(np.dot(a, b)), -1.0, 1.0))))
        r["rotation"] = cum
        prev = r


def rebase_segments(recs: list[dict]) -> None:
    """Chain the per-segment feature series into one continuous curve.

    ``axis_spin_deg`` restarts at every camera cut, so reading it directly puts
    a several-hundred-degree cliff at each one. Accumulating only the deltas
    *inside* a segment keeps the same rule the shim follows: rotation across a
    cut was never measured, so it is never counted.
    """
    for source, dest in (("segment_rotation", "rotation"), ("segment_path", "path"),
                         ("segment_yaw", "yaw")):
        if any(r.get(source) is None for r in recs):
            continue
        cum = 0.0
        prev: dict | None = None
        for r in recs:
            if prev is not None and prev["segment"] == r["segment"]:
                cum += r[source] - prev[source]
            r[dest] = cum
            prev = r


def compute_rates(recs: list[dict], real_fps: float) -> None:
    """Centred 5-frame window, never spanning a camera cut. Rates use REAL fps,
    so conformed slow motion cannot inflate them.

    ``rpm`` is the signed rate of the displayed rotation; ``speed`` is the
    unsigned turn rate, which is what finds an air. With the body
    orientation in the table, ``speed`` is read over a chord set by the
    clip's own turn rate (``features.robust_speed``) rather than five
    frames: on slow motion the five-frame rate is mostly pose jitter, which
    set the peak the gates scale from and broke airs apart."""
    oriented = bool(recs) and all("q" in r for r in recs)
    if oriented:
        speed, chord = snow_features.robust_speed(
            np.array([r["q"] for r in recs]), np.array([r["f"] for r in recs]),
            np.array([r.get("segment", 0) for r in recs]))
        for r, value in zip(recs, speed):
            r["chord_speed"] = float(value) * real_fps / 6.0 if value == value else 0.0
            r["hole"] = max(HOLE_FRAMES, HOLE_SPANS * chord)
    unsigned = all("path" in r for r in recs)
    for i, r in enumerate(recs):
        lo, hi = i, i
        while lo > 0 and i - lo < 2 and recs[lo - 1].get("segment") == r.get("segment"):
            lo -= 1
        while hi < len(recs) - 1 and hi - i < 2 and recs[hi + 1].get("segment") == r.get("segment"):
            hi += 1
        span = recs[hi]["f"] - recs[lo]["f"]
        dps = (recs[hi]["rotation"] - recs[lo]["rotation"]) / (span / real_fps) if span else 0.0
        r["rpm"] = dps / 360 * 60
        if oriented:
            r["speed"] = r["chord_speed"]
        elif unsigned:
            path_dps = (recs[hi]["path"] - recs[lo]["path"]) / (span / real_fps) if span else 0.0
            r["speed"] = path_dps / 360 * 60
        else:
            r["speed"] = abs(r["rpm"])


def adjacent(recs: list[dict], i: int, j: int) -> bool:
    """True when j follows i across at most a short hole, inside the same
    camera shot."""
    return (0 < recs[j]["f"] - recs[i]["f"] <= recs[i].get("hole", HOLE_FRAMES)
            and recs[i].get("segment") == recs[j].get("segment"))


def _sense(inside: list[dict]) -> float:
    """Which way one air turned: the sign of its net vertical yaw.

    Read once per air, because a trick turns one way throughout. The signed
    path's sense is read over half a second, and on synthetic corks it went
    wrong for 50-60 frames at a time while the rider was inverted (the
    camera estimate's error outweighs the small vertical part of the turn
    there, and a pose mirrored in depth turns the other way); net yaw over
    the air had the right sign on 20 of 20.
    """
    for key in ("yaw", "rotation"):
        if all(key in r for r in inside):
            net = inside[-1][key] - inside[0][key]
            if net:
                return float(np.sign(net))
    return 1.0


def _trim_counter_turn(recs: list[dict], lo: int, hi: int, sense: float,
                       core: tuple[int, int] | None = None) -> tuple[int, int]:
    """Drop a wind-up before the air and a check after it.

    Growth reads the unsigned path, so it runs on into a rider winding up
    the other way before takeoff (the real "Frontside Double Cork 1080" clip
    opens with 170 degrees of it). A stretch at either end is cut only when
    the signed path and the vertical yaw both turn against the air: through
    an inversion either one alone can read backwards while the trick goes on.

    With the body orientation in the table (the chord count), the yaw must
    turn back by ``COUNTER_TURN_DEG`` too, not merely some amount: the
    signed path sums every frame's jitter, so on slow motion it passes the
    threshold by noise alone, and a yaw a few degrees back trimmed real air.
    The yaw is an angle, not a sum, so its noise does not grow with frames.
    For the same reason a cut is then looked for only outside ``core``, the
    frames whose speed passed the enter gate: a wind-up is what growth adds
    around the spin, and the spin's own frames are not trimmed however the
    signed path reads them (inverted, it can read backwards for a stretch).
    """
    def against(i: int, j: int) -> bool:
        turned = sense * (recs[j]["rotation"] - recs[i]["rotation"])
        if turned > -COUNTER_TURN_DEG:
            return False
        if "yaw" not in recs[i]:
            return True
        yaw_back = sense * (recs[j]["yaw"] - recs[i]["yaw"])
        return yaw_back <= -COUNTER_TURN_DEG if "q" in recs[i] else yaw_back < 0

    first, last = core if core is not None else (hi, lo)
    signed = [sense * recs[k]["rotation"] for k in range(lo, hi + 1)]
    reach = max(first, lo) - lo + 1 if core is not None else len(signed)
    low = lo + int(np.argmin(signed[:reach]))
    if low > lo and against(lo, low):
        lo = low
    signed = signed[lo - (hi + 1 - len(signed)):]
    start = min(max(last, lo), hi) - lo if core is not None else 0
    high = lo + start + int(np.argmax(signed[start:]))
    if high < hi and against(high, hi):
        hi = high
    return lo, hi


def segmenter_spans(recs: list[dict]) -> list[list[int]]:
    """Airs as the learned segmenter marked them: runs of airborne frames.

    A run continues across a hole in the pose track inside one shot, exactly
    as the rate detector's growth does (``adjacent``), and never across a cut.
    """
    spans: list[list[int]] = []
    for i, r in enumerate(recs):
        if not r.get("airborne"):
            continue
        if spans and spans[-1][1] == i - 1 and adjacent(recs, i - 1, i):
            spans[-1][1] = i
        else:
            spans.append([i, i])
    return spans


def find_events(recs: list[dict], real_fps: float, fps: float,
                detector: str = "auto") -> list[dict]:
    """Air events. ``detector``: "rate" (hysteresis on rotation rate),
    "segmenter" (the learned model's frames, see docs/segmenter.md), or
    "auto", the segmenter when `snow features` wrote its output."""
    if not recs:
        return []
    learned = all("airborne" in r for r in recs)
    if detector == "segmenter" and not learned:
        raise ValueError("no segmenter output in the features table; run `snow features` "
                         "with weights, or use the rate detector")
    use_segmenter = detector == "segmenter" or (detector == "auto" and learned)
    by_f = {r["f"]: r for r in recs}
    speeds = [r["speed"] for r in recs]
    peak = float(np.percentile(speeds, PEAK_PERCENTILE))
    # The floors are in real-time RPM, which slow motion silently divides
    # (the lesson of the 18 RPM gate, docs/validation.md): on synthetic 4x
    # clips the exit floor cut real air off. With the chord speed, jitter no
    # longer lifts a still clip's rate, so the floors are not applied and
    # MIN_NET_ROTATION_DEG alone rejects a clip holding no trick.
    oriented = all("q" in r for r in recs)
    enter = max(0.0 if oriented else RPM_ENTER_FLOOR, ENTER_FRACTION * peak)
    leave = max(0.0 if oriented else RPM_EXIT_FLOOR, EXIT_FRACTION * peak)

    # Cores: runs over the enter gate, bridged across brief tracking dropouts.
    cores: list[list[int]] = []
    for i, r in enumerate(recs):
        if r["speed"] < enter:
            continue
        last = cores[-1][1] if cores else None
        if (last is not None and recs[last].get("segment") == r.get("segment")
                and r["f"] - recs[last]["f"] <= MERGE_GAP_FRAMES):
            cores[-1][1] = i
        else:
            cores.append([i, i])

    # A core is only the fastest part of a spin: its own entry and exit sit
    # below the gate and belong to the same trick. Grow outward while the
    # rider is still rotating, never past a cut or a long hole. Growth reads
    # the unsigned path, not the turn's sense: through the pipeline the sense
    # of a corked spin can read backwards for a stretch, and stopping there
    # split one air into two.
    spans: list[list[int]] = [] if not use_segmenter else segmenter_spans(recs)
    core_of: list[tuple[int, int]] = []
    for a, b in ([] if use_segmenter else cores):
        core = (a, b)
        while a > 0 and adjacent(recs, a - 1, a) and recs[a - 1]["speed"] >= leave:
            a -= 1
        while b < len(recs) - 1 and adjacent(recs, b, b + 1) and recs[b + 1]["speed"] >= leave:
            b += 1
        if spans and a <= spans[-1][1]:  # two cores grew into one another
            spans[-1][1] = max(spans[-1][1], b)
            core_of[-1] = (core_of[-1][0], core[1])
        else:
            spans.append([a, b])
            core_of.append(core)

    events = []
    for n, (lo, hi) in enumerate(spans):
        sense = _sense(recs[lo:hi + 1])
        if not use_segmenter:
            # The segmenter was trained on true takeoffs, so a wind-up is
            # already outside its air; only the rate detector's growth needs
            # trimming back.
            lo, hi = _trim_counter_turn(recs, lo, hi, sense,
                                        core_of[n] if oriented else None)
        inside = recs[lo:hi + 1]
        start, end = inside[0]["f"], inside[-1]["f"]
        if len(inside) < MIN_AIR_FRAMES:
            continue
        sense = _sense(inside)
        path_rot = None
        if all("path" in r for r in inside):
            path_rot = sense * (inside[-1]["path"] - inside[0]["path"])
        if all("q" in r for r in inside):
            # The count: chords about 90 degrees of turn apart, so pose
            # jitter is not summed once per frame (``features.spin_count_deg``).
            rot = sense * snow_features.spin_count_deg(np.array([r["q"] for r in inside]))
        elif path_rot is not None:
            rot = path_rot
        else:
            rot = inside[-1]["rotation"] - inside[0]["rotation"]
        if abs(rot) < MIN_NET_ROTATION_DEG:
            continue
        rpms = sense * np.array([r["speed"] for r in inside])
        peak_i = int(np.argmax(np.abs(rpms)))
        duration = (end - start) / real_fps
        ground = np.median([by_f[f]["ankle_height"]
                            for f in range(max(0, start - 12), start) if f in by_f] or [0.0])

        # Rotational decay: how the spin rate falls off after its peak.
        decay = 0.0
        if len(rpms) - peak_i >= 4:
            tail = np.abs(rpms[peak_i:])
            t = np.arange(len(tail)) / real_fps
            decay = float(np.polyfit(t, tail, 1)[0]) * 6.0  # RPM/s -> deg/s^2

        events.append({
            "id": f"air-{len(events) + 1}",
            "kind": "air",
            "start_f": start,
            "end_f": end,
            "peak_f": inside[peak_i]["f"],
            "t_start": start / fps,
            "t_end": end / fps,
            "metrics": {
                "rotation_deg": round(rot, 1),
                "nearest_spin_deg": int(round(abs(rot) / 180.0) * 180),
                # The frame-to-frame angular path over the same air, the
                # count before chords replaced it; kept for comparison.
                **({"path_rotation_deg": round(path_rot, 1)} if path_rot is not None else {}),
                # Sign is the right-hand rule about world up when the rotation
                # comes from `snow features` (counter-clockwise from above is
                # positive), about the rider's own up axis in the display
                # shim. Frontside/backside needs the rider's stance, which pose alone
                # does not give, so this stays a geometric statement.
                "direction": "counter-clockwise" if rot > 0 else "clockwise",
                "peak_rpm": round(float(rpms[peak_i]), 1),
                "air_time_s": round(duration, 2),
                "pop_height_m": round(max(r["ankle_height"] for r in inside) - float(ground), 2),
                "max_axis_tilt_deg": round(max(r["axis_tilt"] for r in inside), 1),
                "min_tuck_radius_m": round(min(r["tuck_radius"] for r in inside), 2),
                "landing_knee_flex_deg": round(min(r["knee_flex"] for r in inside[-6:]), 1),
                "rotation_decay_dps2": round(decay, 1),
                "mean_confidence": round(float(np.mean([r["conf"] for r in inside])), 3),
            },
        })
    return events


def rule_based_coach(events: list[dict], clip: dict, measure: str) -> dict:
    """The offline fallback for the Nemotron coach (``src/snow/coach.py``).

    Every sentence here is a template over a measured number — it is NOT model
    output, and the UI badges it as such. ``--coach nemotron`` replaces it in
    the bundle, and ``snow serve`` answers ``POST /<name>/coach`` live; the
    shape is the contract.
    """
    if not events:
        return {"source": "rule-based", "model": None, "verdict": "No air detected in this clip.",
                "headline_metric": None, "findings": [], "cues": []}

    hero = max(events, key=lambda e: abs(e["metrics"]["rotation_deg"]))
    m = hero["metrics"]
    short_by = m["nearest_spin_deg"] - abs(m["rotation_deg"])
    findings, cues = [], []

    if m["max_axis_tilt_deg"] > 60:
        findings.append({
            "severity": "good",
            "title": f"Corked rotation — axis tilted {m['max_axis_tilt_deg']:.0f}° off vertical",
            "detail": "The body axis leaves vertical well before the peak rate, which is what "
                      "separates a cork from a flat spin.",
        })
    if abs(short_by) >= 15:
        findings.append({
            # Both directions are a miss off the called spin, and over-rotating
            # lands you sideways — it is not the "good" outcome it reads as.
            "severity": "warning",
            "title": (f"Under-rotated {short_by:.0f}° short of {m['nearest_spin_deg']}°"
                      if short_by > 0 else
                      f"Over-rotated {-short_by:.0f}° past {m['nearest_spin_deg']}°"),
            "detail": f"Measured {abs(m['rotation_deg']):.0f}° of {measure} across "
                      f"{m['air_time_s']:.2f}s of air.",
        })
    if m["rotation_decay_dps2"] < -40:
        findings.append({
            "severity": "serious",
            "title": f"Spin rate decaying {abs(m['rotation_decay_dps2']):.0f}°/s² after peak",
            "detail": "Angular momentum is being shed before the landing window — usually an "
                      "early arm release.",
        })
    if m["landing_knee_flex_deg"] > 150:
        findings.append({
            "severity": "critical",
            "title": f"Stiff landing — knees only {m['landing_knee_flex_deg']:.0f}° at touchdown",
            "detail": "Extended legs on impact put the load through the joint instead of the "
                      "muscle. This is the single biggest rotational-landing injury factor.",
        })

    if short_by > 15:
        cues.append(f"Get the arms inside {m['min_tuck_radius_m']:.2f} m sooner — reaching that tuck "
                    f"before the peak, instead of at it, is where the missing {short_by:.0f}° comes from.")
    elif short_by < -15:
        cues.append(f"Open out of the tuck {-short_by:.0f}° earlier — the spin is still carrying past "
                    f"{m['nearest_spin_deg']}°, which is what puts the board across the landing.")
    if m["landing_knee_flex_deg"] > 150:
        cues.append("Spot the landing and start absorbing a quarter-rotation out, not on contact.")
    cues.append(f"Peak rate was {abs(m['peak_rpm']):.0f} RPM at t={hero['peak_f'] / clip['fps']:.2f}s; "
                "hold that shape through the second half of the air.")

    return {
        "source": "rule-based",
        "model": None,
        "verdict": f"{m['nearest_spin_deg']}° spin, {m['direction']} — "
                   f"{abs(m['rotation_deg']):.0f}° measured, {abs(m['peak_rpm']):.0f} RPM peak, "
                   f"{m['air_time_s']:.2f}s air.",
        "headline_metric": {"label": measure.capitalize(), "value": abs(m["rotation_deg"]),
                            "unit": "°", "event_id": hero["id"]},
        "findings": findings,
        "cues": cues,
    }


def score_events(name: str, events: list[dict]) -> str:
    """Add Model C's FIS judging score prediction to each air, when available."""
    model_path = REPO / "data" / "models" / "scoring" / "model.joblib"
    if not model_path.exists():
        return "unavailable (no scoring model)"
    try:
        import joblib
        import pandas as pd
        sys.path.insert(0, str(REPO / "scoring"))
        from train import features as scoring_features
    except Exception as exc:
        return f"unavailable ({exc})"

    try:
        scoring_bundle = joblib.load(model_path)
        categories = scoring_bundle["categories"]
        score_model = scoring_bundle["score"]
        landed_model = scoring_bundle["landed"]

        for e in events:
            m = e.get("metrics", {})
            clf = m.get("classifier", {})
            # Construct a canonical Trick ID or use components
            spin = clf.get("rotation", {}).get("label") or m.get("nearest_spin_deg", 720)
            axis = clf.get("axis", {}).get("label") or "flat"
            grab = clf.get("grab", {}).get("label") or "none"
            direction = clf.get("direction") or m.get("direction", "frontside")
            
            # Map components to shorthand trick token for scoring/tricks.py
            dir_code = "b" if "back" in str(direction).lower() else "f"
            axis_code = "T" if "triple" in axis else ("D" if "double" in axis else ("C" if "cork" in axis else ""))
            spin_code = str(spin // 100) if spin >= 100 else str(spin)
            grab_code = grab[:2].capitalize() if grab and grab != "none" else ""
            
            parts = [dir_code]
            if axis_code:
                parts.append(axis_code)
            parts.append(spin_code)
            if grab_code:
                parts.append(grab_code)
            trick_id = "-".join(parts)

            df = pd.DataFrame([{
                "trick": trick_id,
                "cat_code": "WC",
                "round": "Final",
                "gender": "M"
            }])
            feat_df = scoring_features(df, categories=categories)
            score_pred = float(score_model.predict(feat_df)[0])
            p_land = float(landed_model.predict_proba(feat_df)[0, 1])
            m["scoring"] = {
                "trick_id": trick_id,
                "score_if_landed": round(score_pred, 1),
                "p_land": round(p_land, 3),
                "expected_score": round(score_pred * p_land, 1),
            }
        return "model-c (FIS big air judge model)"
    except Exception as exc:
        return f"error ({exc})"


def classify_events(name: str, events: list[dict]) -> str:
    """Add model B's trick reading to each air, when weights exist.

    Opt-in (``--classify``) and never required: without weights, or without
    torch, the bundle is written exactly as before and says why. The model
    was trained on synthetic renders only, so its answer sits beside the
    measured rotation rather than replacing it (docs/classifier.md).
    """
    try:
        from snow import camera as cam
        from snow import features as feat
        from snow.learn import airs
        from snow.learn import classifier as clf
    except ImportError as exc:
        return f"unavailable ({exc})"
    if not clf.available():
        return f"unavailable (no weights at {clf.WEIGHTS.relative_to(REPO)})"
    model = clf.Classifier.load()
    camera_path = REPO / "data" / "poses" / f"{name}.camera.npz"
    track = airs.track_from_poses(
        feat.load_poses(REPO / "data" / "poses" / f"{name}.json"), 0,
        cam.CameraTrack.load(camera_path) if camera_path.exists() else None)
    for e in events:
        got = model.classify(track, e["start_f"], e["end_f"])
        if got is not None:
            e["metrics"]["classifier"] = got
    return "model-b (trained on synthetic renders only)"


def main(name: str, coach: str = "rule-based", detector: str = "auto",
         classify: bool = False) -> None:
    poses = REPO / "data" / "poses" / f"{name}.json"
    clip_src = REPO / "data" / "clips" / f"{name}.mp4"
    for p in (poses, clip_src):
        if not p.exists():
            sys.exit(f"missing {p} — run `snow run` for '{name}' first")

    rows = [json.loads(line) for line in poses.read_text().splitlines() if line.strip()]
    detected = [(r["frame_id"], r["detections"][0]) for r in rows if r.get("detections")]
    if not detected:
        sys.exit(f"{poses} has no detections")

    clip = probe(clip_src)
    meta_path = REPO / "data" / "clips" / f"{name}.meta.json"
    slowmo = 1.0
    if meta_path.exists():
        slowmo = float(json.loads(meta_path.read_text()).get("slowmo_factor", 1.0) or 1.0)
    clip["slowmo_factor"] = slowmo
    clip["real_fps"] = clip["fps"] * slowmo
    clip["src"] = "clip.mp4"
    clip["tracked_frames"] = len(detected)
    clip["total_frames"] = len(rows)

    recs = per_frame(detected)
    integrate_rotation(recs, clip["real_fps"])
    metrics_source = "display-shim"
    rotation_measure = "body-axis rotation, camera NOT compensated"

    # `snow features` removes the camera's own rotation; prefer it wherever it
    # has a row. Posture metrics (tuck, knee flex) are not in that table, so
    # those stay as computed above and are joined on frame id.
    loaded = load_features(name)
    if loaded:
        feats, column, rotation_measure = loaded
        recs = [r for r in recs if r["f"] in feats]
        for r in recs:
            r.update(feats[r["f"]])
        rebase_segments(recs)
        metrics_source = "snow-features"
        print(f"  rotation from {column} ({rotation_measure})")

    compute_rates(recs, clip["real_fps"])
    events = find_events(recs, clip["real_fps"], clip["fps"], detector)
    air_source = ("segmenter" if detector != "rate" and recs and all("airborne" in r for r in recs)
                  else "rate")
    classifier_source = classify_events(name, events) if classify else "off"
    scoring_source = score_events(name, events)
    if classify:
        print(f"  classifier: {classifier_source}")
    print(f"  scoring: {scoring_source}")

    ids = [r["f"] for r in recs]
    gaps = [[a + 1, b - 1] for a, b in zip(ids, ids[1:]) if b - a > 1]

    out_dir = WEB / "public" / "sessions" / name
    bundle = {
        "schema": 1,
        "name": name,
        "clip": clip,
        "source": {
            "metrics": metrics_source,
            "rotation_measure": rotation_measure,
            "coach": "rule-based",
            "pose": "nvidia-body-pose-nim",
            "air_detector": air_source,
            "classifier": classifier_source,
            "scoring": scoring_source,
        },
        "skeleton": {"parents": PARENTS, "links": LINKS, "joints": J, "num_joints": len(PARENTS)},
        "gaps": gaps,
        "telemetry": {
            "f": ids,
            "t": [round(f / clip["fps"], 4) for f in ids],
            "rotation_deg": [round(r["rotation"], 2) for r in recs],
            "rpm": [round(r["rpm"], 2) for r in recs],
            "axis_tilt_deg": [round(r["axis_tilt"], 1) for r in recs],
            "ankle_height_m": [round(r["ankle_height"], 3) for r in recs],
            "tuck_radius_m": [round(r["tuck_radius"], 3) for r in recs],
            "knee_flex_deg": [round(r["knee_flex"], 1) for r in recs],
            "confidence": [round(r["conf"], 3) for r in recs],
            **(
                {"camera_rotation_cum_deg": [round(r["camera_cum"], 2) for r in recs]}
                if metrics_source == "snow-features"
                else {}
            ),
        },
        "events": events,
        "coach": rule_based_coach(events, clip, rotation_measure),
        # Quantised pose: 2D in source pixels, 3D in millimetres, conf 0-100.
        "frames": [
            {
                "f": fid,
                "bbox": [round(v, 1) for v in det["bbox"]],
                "k2": [[round(x, 1), round(y, 1)] for x, y in det["keypoints_2d"]],
                "k3": [[int(round(x * 1000)), int(round(y * 1000)), int(round(z * 1000))]
                       for x, y, z in det["keypoints_3d"]],
                "conf": [int(round(c * 100)) for c in det["keypoints_confidence"]],
            }
            for fid, det in detected
        ],
    }
    if coach == "nemotron":
        # Fails the export rather than shipping the template under a live badge.
        from snow import coach as nemotron

        try:
            bundle["coach"] = nemotron.coach_session(bundle)
        except nemotron.CoachError as exc:
            sys.exit(f"coach failed, bundle not written: {exc}")
        bundle["source"]["coach"] = "nemotron"
        print(f"  coach: {bundle['coach']['model']}")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "session.json").write_text(json.dumps(bundle, separators=(",", ":")))
    shutil.copy2(clip_src, out_dir / "clip.mp4")

    index_path = WEB / "public" / "sessions" / "index.json"
    known = {}
    if index_path.exists():
        known = {s["name"]: s for s in json.loads(index_path.read_text()).get("sessions", [])}
    known[name] = {
        "name": name,
        "duration": clip["duration"],
        "events": len(events),
        "tracked_frames": len(detected),
    }
    index_path.write_text(json.dumps({"sessions": sorted(known.values(), key=lambda s: s["name"])}, indent=2))

    size = (out_dir / "session.json").stat().st_size / 1e6
    print(f"{name}: {len(detected)}/{len(rows)} frames, {len(events)} events, "
          f"{len(gaps)} track gaps, metrics={metrics_source} "
          f"-> {out_dir.relative_to(REPO)}/session.json ({size:.1f} MB)")
    for e in events:
        m = e["metrics"]
        print(f"  {e['id']}: {e['t_start']:5.2f}-{e['t_end']:5.2f}s  "
              f"{m['rotation_deg']:+8.1f}deg (~{m['nearest_spin_deg']})  "
              f"peak {m['peak_rpm']:+6.1f} RPM  tilt {m['max_axis_tilt_deg']:5.1f}deg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a dashboard bundle for one clip.")
    parser.add_argument("name", nargs="?", default="run1")
    parser.add_argument("--coach", choices=["rule-based", "nemotron"], default="rule-based",
                        help="nemotron precomputes a real coach report (needs NVIDIA_API_KEY)")
    parser.add_argument("--air", choices=["auto", "rate", "segmenter"], default="auto",
                        help="air detector: the learned segmenter when `snow features` "
                             "wrote its output (auto), or the rate hysteresis")
    parser.add_argument("--classify", action="store_true",
                        help="add model B's trick reading to each air, when weights exist")
    args = parser.parse_args()
    main(args.name, args.coach, args.air, args.classify)
