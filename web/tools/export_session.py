#!/usr/bin/env python3
"""Build a browser-ready session bundle from a clip's pose output.

    ../.venv/bin/python tools/export_session.py run1

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
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
WEB = REPO / "web"
sys.path.insert(0, str(REPO / "src"))
from snow.skeleton import LINKS, PARENTS  # noqa: E402  (after sys.path edit)

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
PEAK_PERCENTILE = 99  # not max(): one bad frame must not set the scale
# Floors stop a clip holding no trick from scaling its gate down onto its own
# noise. MIN_NET_ROTATION_DEG is what actually rejects those.
RPM_ENTER_FLOOR = 8.0
RPM_EXIT_FLOOR = 4.0
MIN_AIR_FRAMES = 8
MERGE_GAP_FRAMES = 10
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
ROTATION_PREFERENCE = [
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
    for row in df.to_dict("records"):
        value = row.get("camera_rotation_deg")
        camera_cum += float(value) if value == value else 0.0  # NaN-safe
        tilt = row.get("tilt_deg")
        out[int(row["frame_id"])] = {
            # Cumulative WITHIN a camera segment: it restarts at every cut.
            "segment_rotation": float(row[column]),
            "segment": int(row.get("segment", 0) or 0),
            "axis_tilt": float(tilt) if tilt == tilt else 0.0,
            "camera_cum": camera_cum,
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
    cum = 0.0
    prev: dict | None = None
    for r in recs:
        if prev is not None and prev["segment"] == r["segment"]:
            cum += r["segment_rotation"] - prev["segment_rotation"]
        r["rotation"] = cum
        prev = r


def compute_rates(recs: list[dict], real_fps: float) -> None:
    """Centred 5-frame window, never spanning a camera cut. Rates use REAL fps,
    so conformed slow motion cannot inflate them."""
    for i, r in enumerate(recs):
        lo, hi = i, i
        while lo > 0 and i - lo < 2 and recs[lo - 1].get("segment") == r.get("segment"):
            lo -= 1
        while hi < len(recs) - 1 and hi - i < 2 and recs[hi + 1].get("segment") == r.get("segment"):
            hi += 1
        span = recs[hi]["f"] - recs[lo]["f"]
        dps = (recs[hi]["rotation"] - recs[lo]["rotation"]) / (span / real_fps) if span else 0.0
        r["rpm"] = dps / 360 * 60


def adjacent(recs: list[dict], i: int, j: int) -> bool:
    """True when j is the very next frame after i, inside the same camera shot."""
    return recs[j]["f"] - recs[i]["f"] == 1 and recs[i].get("segment") == recs[j].get("segment")


def find_events(recs: list[dict], real_fps: float, fps: float) -> list[dict]:
    if not recs:
        return []
    by_f = {r["f"]: r for r in recs}
    peak = float(np.percentile(np.abs([r["rpm"] for r in recs]), PEAK_PERCENTILE))
    enter = max(RPM_ENTER_FLOOR, ENTER_FRACTION * peak)
    leave = max(RPM_EXIT_FLOOR, EXIT_FRACTION * peak)

    # Cores: runs over the enter gate, bridged across brief tracking dropouts.
    cores: list[list[int]] = []
    for i, r in enumerate(recs):
        if abs(r["rpm"]) < enter:
            continue
        last = cores[-1][1] if cores else None
        if (last is not None and recs[last].get("segment") == r.get("segment")
                and r["f"] - recs[last]["f"] <= MERGE_GAP_FRAMES):
            cores[-1][1] = i
        else:
            cores.append([i, i])

    # A core is only the fastest part of a spin: its own entry and exit sit
    # below the gate and belong to the same trick. Grow outward while the
    # rider keeps turning the same way, never past a cut or a dropped frame.
    spans: list[list[int]] = []
    for a, b in cores:
        turn = np.sign(recs[b]["rotation"] - recs[a]["rotation"]) or np.sign(recs[a]["rpm"])
        while (a > 0 and adjacent(recs, a - 1, a)
               and abs(recs[a - 1]["rpm"]) >= leave and np.sign(recs[a - 1]["rpm"]) == turn):
            a -= 1
        while (b < len(recs) - 1 and adjacent(recs, b, b + 1)
               and abs(recs[b + 1]["rpm"]) >= leave and np.sign(recs[b + 1]["rpm"]) == turn):
            b += 1
        if spans and a <= spans[-1][1]:  # two cores grew into one another
            spans[-1][1] = max(spans[-1][1], b)
        else:
            spans.append([a, b])

    events = []
    for lo, hi in spans:
        inside = recs[lo:hi + 1]
        start, end = inside[0]["f"], inside[-1]["f"]
        if len(inside) < MIN_AIR_FRAMES:
            continue
        rot = inside[-1]["rotation"] - inside[0]["rotation"]
        if abs(rot) < MIN_NET_ROTATION_DEG:
            continue
        rpms = np.array([r["rpm"] for r in inside])
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
                # Sign is the right-hand rule about the rider's own up axis.
                # Frontside/backside needs the rider's stance, which pose alone
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
    """A deterministic stand-in for the Nemotron reasoning layer.

    Every sentence here is a template over a measured number — it is NOT model
    output, and the UI badges it as such. Replacing it means pointing
    ``api.fetchCoachReport`` at the real endpoint; the shape is the contract.
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


def main(name: str) -> None:
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
    events = find_events(recs, clip["real_fps"], clip["fps"])

    ids = [r["f"] for r in recs]
    gaps = [[a + 1, b - 1] for a, b in zip(ids, ids[1:]) if b - a > 1]

    out_dir = WEB / "public" / "sessions" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "schema": 1,
        "name": name,
        "clip": clip,
        "source": {
            "metrics": metrics_source,
            "rotation_measure": rotation_measure,
            "coach": "rule-based",
            "pose": "nvidia-body-pose-nim",
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
    main(sys.argv[1] if len(sys.argv) > 1 else "run1")
