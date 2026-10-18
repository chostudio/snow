"""Measured against exact: one row per clip and pipeline variant.

Two variants of every clip go through the *unchanged* measurement code:

``oracle``
    the rig's exact skeleton and the true camera, through ``snow features``
    and the dashboard's air detector. Error here is the measurement's own
    definition, not perception.
``pipeline``
    the render through ``snow run --features``: tracker, NIM, ``camera.py``,
    ``features.py``, air detector. Error here is what a user would see.

The difference between the two is what the NIM and the camera estimate add.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

from .. import config as cfg
from .. import meta as mt

REPO = cfg.REPO_ROOT


def _exporter():
    """The dashboard's own session exporter, loaded from web/tools."""
    path = REPO / "web" / "tools" / "export_session.py"
    spec = importlib.util.spec_from_file_location("export_session", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def air_events(name: str) -> list[dict]:
    """Exactly what the dashboard would show as air events for ``name``.

    Mirrors ``export_session.main`` up to ``find_events`` without writing a
    bundle, so the detector under test is the shipped one.
    """
    es = _exporter()
    rows = [json.loads(line) for line in cfg.pose_path(name).read_text().splitlines()
            if line.strip()]
    detected = [(r["frame_id"], r["detections"][0]) for r in rows if r.get("detections")]
    if not detected:
        return []
    clip = mt.load(name)
    real_fps = clip.real_fps
    recs = es.per_frame(detected)
    es.integrate_rotation(recs, real_fps)
    loaded = es.load_features(name)
    if loaded:
        feats, _, _ = loaded
        recs = [r for r in recs if r["f"] in feats]
        for r in recs:
            r.update(feats[r["f"]])
        es.rebase_segments(recs)
    es.compute_rates(recs, real_fps)
    return es.find_events(recs, real_fps, clip.fps)


def _over(name: str, column: str, start: int, end: int) -> float | None:
    """Change in a per-frame feature column across frames ``start``..``end``."""
    import pandas as pd

    df = pd.read_csv(cfg.POSES_DIR / f"{name}.features.csv")
    if column not in df:
        return None
    inside = df[(df["frame_id"] >= start) & (df["frame_id"] <= end)].dropna(subset=[column])
    if len(inside) < 2:
        return None
    # Never across a cut: take the longest stretch within one segment.
    seg = inside.groupby("segment").size().idxmax()
    inside = inside[inside["segment"] == seg]
    return float(inside[column].iloc[-1] - inside[column].iloc[0])


def measured(name: str, takeoff: int | None = None, landing: int | None = None) -> dict | None:
    summary_path = cfg.POSES_DIR / f"{name}.features.json"
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text())
    segments = summary.get("segments", [])
    if not segments:
        return {"segments": 0}
    # The trick lives in the longest shot; a synthetic clip has one, unless
    # camera.py wrongly finds a cut, which is itself worth reporting.
    seg = max(segments, key=lambda s: s["frames"])
    events = air_events(name)
    main = max(events, key=lambda e: abs(e["metrics"]["rotation_deg"]), default=None)
    out = {
        "segments": len(segments),
        "tracked_frames": summary.get("tracked_frames"),
        "axis_spin_deg": seg["axis_spin_deg"],
        "vertical_yaw_deg": seg["net_spin_deg"],
        "total_rotation_deg": seg["total_rotation_deg"],
        "axis_tilt_deg": seg["spin_axis_tilt_deg"],
        "max_tilt_deg": seg["max_tilt_deg"],
        "camera_rotation_deg": seg["camera_rotation_deg"],
        "events": len(events),
    }
    if takeoff is not None:
        # Over the true air: what the measure says when the window is right.
        for column in ("spin_path_deg", "axis_spin_deg"):
            value = _over(name, column, takeoff, landing)
            if value is not None:
                out[f"air_{column}"] = round(value, 1)
    if main is not None:
        m = main["metrics"]
        out.update({
            "event_rotation_deg": m["rotation_deg"],
            "event_spin_deg": m["nearest_spin_deg"],
            "event_direction": m["direction"],
            "air_start": main["start_f"],
            "air_end": main["end_f"],
        })
    return out


def row(clip: Path, variant: str, name: str) -> dict:
    labels = json.loads((clip / "labels.json").read_text())
    got = measured(name, labels["takeoff_frame"], labels["landing_frame"]) or {}
    r = {"clip": clip.name, "variant": variant, **{f"true_{k}": v for k, v in labels.items()
                                                    if not isinstance(v, (dict, list))}}
    r.update({f"got_{k}": v for k, v in got.items()})
    if not got or got.get("segments", 0) == 0:
        r["status"] = "no measurement"
        return r
    r["status"] = "ok"
    nominal = labels["spin_nominal_deg"]
    r["err_axis_spin_vs_nominal"] = round(abs(got["axis_spin_deg"]) - nominal, 1)
    r["err_axis_spin_vs_exact"] = round(abs(got["axis_spin_deg"]) - labels["axis_spin_deg"], 1)
    r["err_vertical_yaw"] = round(abs(got["vertical_yaw_deg"]) - nominal, 1)
    if got.get("air_spin_path_deg") is not None:
        path = got["air_spin_path_deg"]
        r["err_air_spin_path"] = round(abs(path) - nominal, 1)
        r["air_spin_path_named_ok"] = int(round(abs(path) / 180.0)) * 180 == nominal
        r["air_spin_path_sense_ok"] = bool(np.sign(path) == labels["spin_sign"])
    r["vertical_yaw_sense_ok"] = (np.sign(got["vertical_yaw_deg"]) == labels["spin_sign"])
    if got.get("axis_tilt_deg") is not None:
        r["err_axis_tilt"] = round(got["axis_tilt_deg"] - labels["axis_tilt_deg"], 1)
    r["err_max_tilt"] = round(got["max_tilt_deg"] - labels["max_tilt_deg"], 1)
    r["err_camera_rotation"] = round(got["camera_rotation_deg"] - labels["camera_path_deg"], 1)
    if "event_spin_deg" in got:
        r["event_spin_ok"] = bool(got["event_spin_deg"] == nominal)
        r["err_event_spin"] = got["event_spin_deg"] - nominal
        r["event_direction_ok"] = bool(got["event_direction"] == labels["rotation_sense"])
        r["err_takeoff_frames"] = got["air_start"] - labels["takeoff_frame"]
        r["err_landing_frames"] = got["air_end"] - labels["landing_frame"]
        r["err_air_frames"] = (got["air_end"] - got["air_start"]) - labels["air_frames"]
    return r


def _mean_abs(rows, key):
    values = [abs(r[key]) for r in rows if r.get(key) is not None]
    return f"{np.mean(values):.0f}" if values else "–"


def _rate(rows, key):
    values = [bool(r[key]) for r in rows if r.get(key) is not None]
    return f"{sum(values)}/{len(values)}" if values else "–"


COLUMNS = [
    ("n", None),
    ("axis spin vs named", "err_axis_spin_vs_nominal"),
    ("vertical yaw vs named", "err_vertical_yaw"),
    ("spin path over air vs named", "err_air_spin_path"),
    ("spin path names it", "air_spin_path_named_ok"),
    ("spin path sense right", "air_spin_path_sense_ok"),
    ("dashboard spin right", "event_spin_ok"),
    ("dashboard direction right", "event_direction_ok"),
    ("axis tilt", "err_axis_tilt"),
    ("max tilt", "err_max_tilt"),
    ("camera rotation", "err_camera_rotation"),
    ("takeoff (frames)", "err_takeoff_frames"),
    ("landing (frames)", "err_landing_frames"),
]


def table(rows: list[dict], by: str) -> str:
    """Markdown: mean absolute errors (degrees unless stated), grouped by ``by``."""
    groups: dict[str, list[dict]] = {}
    for r in rows:
        if r.get("status") == "ok":
            groups.setdefault(str(r.get(by)), []).append(r)
    header = "| " + " | ".join([by.replace("true_", "")] + [c for c, _ in COLUMNS]) + " |"
    lines = [header, "|" + "---|" * (len(COLUMNS) + 1)]
    for key in sorted(groups) + ["all"]:
        members = groups[key] if key != "all" else [r for g in groups.values() for r in g]
        cells = [f"**{key}**" if key == "all" else key]
        for _, field in COLUMNS:
            if field is None:
                cells.append(str(len(members)))
            elif field.endswith("_ok"):
                cells.append(_rate(members, field))
            else:
                cells.append(_mean_abs(members, field))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def per_clip(rows: list[dict]) -> str:
    lines = ["| clip | variant | named | exact axis spin | got axis spin | got vertical yaw "
             "| dashboard | direction (true / dashboard) | axis tilt (true / got) "
             "| camera ° (true / got) | air frames (true / got) |",
             "|" + "---|" * 11]
    for r in rows:
        if r.get("status") != "ok":
            lines.append(f"| {r['clip']} | {r['variant']} | {r['true_spin_nominal_deg']} "
                         f"| {r['true_axis_spin_deg']:.0f} | {r['status']} |||||||")
            continue
        air = "–"
        if r.get("got_air_start") is not None:
            air = f"{r['true_air_frames']} / {r['got_air_end'] - r['got_air_start']}"
        lines.append(
            f"| {r['clip']} | {r['variant']} | {r['true_spin_nominal_deg']} "
            f"| {r['true_axis_spin_deg']:.0f} | {r['got_axis_spin_deg']:.0f} "
            f"| {r['got_vertical_yaw_deg']:+.0f} | {r.get('got_event_spin_deg', '–')} "
            f"| {r['true_rotation_sense'][:3]} / {str(r.get('got_event_direction', '–'))[:3]} "
            f"| {r['true_axis_tilt_deg']:.0f} / {r['got_axis_tilt_deg'] or 0:.0f} "
            f"| {r['true_camera_path_deg']:.0f} / {r['got_camera_rotation_deg']:.0f} "
            f"| {air} |"
        )
    return "\n".join(lines)
