"""The coach: a Nemotron model reasoning over one clip's measurements.

The model is served by NVIDIA's API (``integrate.api.nvidia.com``, OpenAI
compatible chat completions) with the same ``NVIDIA_API_KEY`` the NIM uses.
It sees **only** numbers this pipeline measured, renamed so that each name
carries its own caveat (``pop_height_uncalibrated``, ``peak_rate_rpm_clip_timebase``),
plus the claims from docs/validation.md it may not make. It never sees the
video or the pose.

What comes back must match the dashboard's ``CoachReport`` (web/src/types.ts).
A reply that does not is sent back once with the reason; a second miss raises
``CoachSchemaError`` rather than quietly showing a template, because a
plausible report that was not checked is exactly the failure this project
exists to avoid.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

from . import config as cfg

# A reasoning model, verified on GET /v1/models and with a real completion.
# Override with SNOW_COACH_MODEL. A model listed there can still 404 for a
# given account (nemotron-3-super-120b-a12b did), so check with a real call.
DEFAULT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"

SEVERITIES = ("good", "warning", "serious", "critical")
REPORT_KEYS = {"verdict", "headline_metric", "findings", "cues"}
HEADLINE_KEYS = {"label", "value", "unit", "event_id"}
FINDING_KEYS = {"severity", "title", "detail"}
MAX_FINDINGS = 5
MAX_CUES = 4
# Units that would present an uncalibrated length as a real one.
LENGTH_UNITS = {"m", "metre", "metres", "meter", "meters", "cm", "ft", "feet"}

# features.json segment fields passed through. `axis_spin_deg` is deliberately
# absent: synthetic tricks with exact labels showed it undercounts corks even
# on perfect poses, and handing the model two spin counts invites it to pick
# the wrong one. `net_spin_deg` / `peak_spin_deg` (vertical yaw) degenerate
# when the rider is inverted, so they stay out too. `spin_path_deg` is out for
# the same reason: summed frame to frame over the whole shot, it accrues pose
# and camera jitter and over-reads against the per-air chord count
# (`spin_count_deg`, docs/rotation.md), most of all on slow motion.
SEGMENT_FIELDS = {
    "segment": "segment",
    "frames": "frames",
    "total_rotation_deg": "total_angular_path_deg",
    "max_tilt_deg": "max_body_tilt_deg",
    "camera_rotation_deg": "camera_rotation_removed_deg",
    "median_quality": "median_spin_quality",
}


class CoachError(RuntimeError):
    """The coach could not produce a report (network, auth, model)."""


class CoachSchemaError(CoachError):
    """The model's reply did not match ``CoachReport``."""


def model_id() -> str:
    return os.environ.get("SNOW_COACH_MODEL") or DEFAULT_MODEL


def base_url() -> str:
    return (os.environ.get("SNOW_COACH_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")


def load_segments(name: str) -> list[dict] | None:
    """Per-shot summaries from ``snow features``, if it has been run."""
    path = cfg.POSES_DIR / f"{name}.features.json"
    if not path.exists():
        return None
    return json.loads(path.read_text()).get("segments")


def _timed(rate_known: bool) -> str:
    return "real_time" if rate_known else "clip_timebase"


def measurements(session: dict, segments: list[dict] | None = None,
                 event_id: str | None = None) -> dict:
    """The model's whole view of the clip: measured numbers, caveat-named.

    Built from the session bundle (web/tools/export_session.py), so the coach
    reads exactly what the dashboard shows.
    """
    clip = session["clip"]
    rate_known = float(clip.get("slowmo_factor") or 1.0) != 1.0
    tb = _timed(rate_known)

    events = []
    for e in session.get("events", []):
        m = e["metrics"]
        events.append({
            "event_id": e["id"],
            "t_start_clip_s": round(e["t_start"], 2),
            "t_end_clip_s": round(e["t_end"], 2),
            "spin_count_deg": m["rotation_deg"],
            "nearest_180_deg": m["nearest_spin_deg"],
            "direction": m["direction"],
            f"peak_rate_rpm_{tb}": m["peak_rpm"],
            f"air_time_s_{tb}": m["air_time_s"],
            f"rotation_decay_deg_per_s2_{tb}": m["rotation_decay_dps2"],
            "max_body_tilt_deg": m["max_axis_tilt_deg"],
            "landing_knee_flex_deg": m["landing_knee_flex_deg"],
            "pop_height_uncalibrated": m["pop_height_m"],
            "min_tuck_radius_uncalibrated": m["min_tuck_radius_m"],
            "mean_pose_confidence": m["mean_confidence"],
        })

    out = {
        "clip": {
            "name": session["name"],
            "duration_clip_s": round(clip["duration"], 2),
            "fps": clip["fps"],
            "slowmo_factor": clip.get("slowmo_factor", 1.0),
            "slowmo_factor_set": rate_known,
            "tracked_frames": clip["tracked_frames"],
            "total_frames": clip["total_frames"],
            "camera_cuts": len(session.get("gaps", [])),
        },
        "spin_count_measure": session["source"]["rotation_measure"],
        "camera_compensated": session["source"]["metrics"] == "snow-features",
        "focus_event_id": event_id,
        "events": events,
    }
    if segments:
        out["shots"] = [
            {new: s[old] for old, new in SEGMENT_FIELDS.items() if s.get(old) is not None}
            for s in segments
        ]
    return out


def caveats(data: dict) -> list[str]:
    """docs/validation.md's "Claims you may not make", applied to this clip."""
    out = [
        "Spin counts are UNVALIDATED in absolute terms on real footage. spin_count_deg "
        "(per air) is the only measured count; never state that the rider "
        "landed a named spin (\"a 1080\"). Say what was measured and treat nearest_180_deg "
        "as a rounding, not a call.",
        "Lengths (*_uncalibrated) are camera-relative in unknown units because the focal "
        "length is assumed. Never express them in metres, centimetres or feet, and never "
        "call them a real height or reach; compare them only to each other.",
        "Rotation across a camera cut was not measured, by design; never claim it was.",
        "direction is geometric (sense of rotation about the up axis). Frontside/backside "
        "needs the rider's stance, which the pose does not give; do not name it.",
    ]
    if data["clip"]["slowmo_factor_set"]:
        out.append(
            f"Rates and durations were divided by a user-supplied slow-motion factor "
            f"(x{data['clip']['slowmo_factor']}) and are real-time only if that factor is right."
        )
    else:
        out.append(
            "slowmo_factor was NOT set. Every rate and duration (*_clip_timebase) is in the "
            "clip's own timebase; on slow-motion footage rates are low and durations high by "
            "an unknown factor. Never call them real-time; angles and totals are unaffected."
        )
    if not data["camera_compensated"]:
        out.append(
            "The broadcast camera's own rotation was NOT subtracted; absolute spin totals "
            "are indicative only."
        )
    return out


SCHEMA_TEXT = """{
  "verdict": string,                  // one or two sentences on the focus air
  "headline_metric": {                // or null when there are no events
    "label": string,
    "value": number,                  // copied exactly from the measurements
    "unit": string,                   // e.g. "°", "RPM (clip timebase)"; never a length unit
    "event_id": string                // one of the event_id values given
  } | null,
  "findings": [                       // 1 to 5
    {"severity": "good" | "warning" | "serious" | "critical", "title": string, "detail": string}
  ],
  "cues": [string]                    // 1 to 4 short, actionable coaching cues
}"""


def build_messages(data: dict) -> list[dict]:
    """System + user messages for one coaching request. Pure, so it is testable."""
    rules = "\n".join(f"- {c}" for c in caveats(data))
    system = (
        "You are a freestyle snowboard coach reviewing ONE clip from measurements a pose "
        "pipeline produced. You do not see the video. Use only the numbers provided; do not "
        "invent measurements, and when you cite a number, cite it as given.\n\n"
        f"Rules you must follow:\n{rules}\n\n"
        "Reply with a single JSON object and nothing else (no prose, no code fences), "
        f"exactly this shape with no extra keys:\n{SCHEMA_TEXT}"
    )
    focus = data.get("focus_event_id")
    ask = (f"Coach the air {focus}; use the other events only for comparison."
           if focus else "Coach the air with the largest spin_count_deg; mention the others briefly if useful.")
    user = f"{ask}\n\nMeasurements:\n{json.dumps(data, indent=2)}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _numbers(value) -> list[float]:
    """Every measured number in ``value``. The nearest-180 rounding is not a
    measurement, so a headline may not quote it as one."""
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, dict):
        return [n for k, v in value.items() if k != "nearest_180_deg" for n in _numbers(v)]
    if isinstance(value, list):
        return [n for v in value for n in _numbers(v)]
    return []


def parse_reply(text: str) -> dict:
    """Pull the JSON object out of a reply, tolerating fences and think tags."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S)
    if fenced:
        text = fenced.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise CoachSchemaError("reply contains no JSON object")
    try:
        obj = json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        raise CoachSchemaError(f"reply is not valid JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise CoachSchemaError("reply JSON is not an object")
    return obj


def _text(obj: dict, key: str, where: str) -> str:
    value = obj.get(key)
    if not isinstance(value, str) or not value.strip():
        raise CoachSchemaError(f"{where}.{key} must be a non-empty string")
    return value.strip()


def validate_report(obj: dict, data: dict, model: str) -> dict:
    """Check a parsed reply against ``CoachReport`` and the measurements.

    Beyond shape: the headline must point at a real event and quote a number
    that was actually measured, and must not put a length unit on it.
    """
    keys = set(obj)
    if keys != REPORT_KEYS:
        missing, extra = REPORT_KEYS - keys, keys - REPORT_KEYS
        raise CoachSchemaError(
            "top-level keys must be exactly " + ", ".join(sorted(REPORT_KEYS))
            + (f"; missing {sorted(missing)}" if missing else "")
            + (f"; unexpected {sorted(extra)}" if extra else "")
        )
    verdict = _text(obj, "verdict", "report")

    event_ids = {e["event_id"] for e in data["events"]}
    head = obj["headline_metric"]
    if head is not None:
        if not isinstance(head, dict) or set(head) != HEADLINE_KEYS:
            raise CoachSchemaError(
                "headline_metric must be null or have exactly " + ", ".join(sorted(HEADLINE_KEYS))
            )
        value = head["value"]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise CoachSchemaError("headline_metric.value must be a number")
        if head["event_id"] not in event_ids:
            raise CoachSchemaError(
                f"headline_metric.event_id {head['event_id']!r} is not one of {sorted(event_ids)}"
            )
        unit = _text(head, "unit", "headline_metric")
        if unit.lower().rstrip(".") in LENGTH_UNITS:
            raise CoachSchemaError(
                f"headline_metric.unit {unit!r} presents an uncalibrated length as real"
            )
        measured = _numbers(data)
        if not any(abs(abs(value) - abs(n)) <= 0.51 for n in measured):
            raise CoachSchemaError(
                f"headline_metric.value {value} is not one of the measured numbers"
            )
        head = {"label": _text(head, "label", "headline_metric"), "value": float(value),
                "unit": unit, "event_id": head["event_id"]}
    elif event_ids:
        raise CoachSchemaError("headline_metric may only be null when there are no events")

    findings = obj["findings"]
    if not isinstance(findings, list) or not 1 <= len(findings) <= MAX_FINDINGS:
        raise CoachSchemaError(f"findings must be a list of 1 to {MAX_FINDINGS} items")
    clean_findings = []
    for i, f in enumerate(findings):
        if not isinstance(f, dict) or set(f) != FINDING_KEYS:
            raise CoachSchemaError(f"findings[{i}] must have exactly severity, title, detail")
        if f["severity"] not in SEVERITIES:
            raise CoachSchemaError(f"findings[{i}].severity must be one of {list(SEVERITIES)}")
        clean_findings.append({"severity": f["severity"],
                               "title": _text(f, "title", f"findings[{i}]"),
                               "detail": _text(f, "detail", f"findings[{i}]")})

    cues = obj["cues"]
    if (not isinstance(cues, list) or not 1 <= len(cues) <= MAX_CUES
            or not all(isinstance(c, str) and c.strip() for c in cues)):
        raise CoachSchemaError(f"cues must be a list of 1 to {MAX_CUES} non-empty strings")

    return {
        "source": "nemotron",
        "model": model,
        "verdict": verdict,
        "headline_metric": head,
        "findings": clean_findings,
        "cues": [c.strip() for c in cues],
    }


Chat = Callable[[list[dict]], str]


def nvidia_chat(model: str, api_key: str, *, timeout: float = 180.0) -> Chat:
    """A chat function bound to NVIDIA's OpenAI-compatible endpoint."""

    def chat(messages: list[dict]) -> str:
        body = {
            "model": model,
            "messages": messages,
            # Reasoning tokens count against max_tokens; leave them room.
            "max_tokens": 8192,
            "temperature": 0.6,
            "top_p": 0.95,
        }
        request = urllib.request.Request(
            f"{base_url()}/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
                     "Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                reply = json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read()[:400].decode(errors="replace")
            raise CoachError(f"NVIDIA API {exc.code} for {model}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise CoachError(f"NVIDIA API unreachable: {exc}") from exc
        try:
            return reply["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise CoachError(f"unexpected completion shape from {model}") from exc

    return chat


def report(data: dict, chat: Chat, model: str) -> dict:
    """Ask once; on a schema miss, show the model its error and ask once more."""
    messages = build_messages(data)
    reply = chat(messages)
    try:
        return validate_report(parse_reply(reply), data, model)
    except CoachSchemaError as first:
        messages = messages + [
            {"role": "assistant", "content": reply},
            {"role": "user", "content": f"That reply was rejected: {first}. "
                                        "Return only the corrected JSON object."},
        ]
        retry = chat(messages)
        try:
            return validate_report(parse_reply(retry), data, model)
        except CoachSchemaError as second:
            raise CoachSchemaError(
                f"{model} returned an invalid CoachReport twice: {first}; then: {second}"
            ) from second


def coach_session(session: dict, event_id: str | None = None,
                  segments: list[dict] | None = None, chat: Chat | None = None) -> dict:
    """A validated Nemotron ``CoachReport`` for one session bundle."""
    ids = [e["id"] for e in session.get("events", [])]
    if event_id is not None and event_id not in ids:
        raise ValueError(f"unknown event {event_id!r}; this clip has {ids}")
    model = model_id()
    if chat is None:
        key = os.environ.get("NVIDIA_API_KEY")
        if not key:
            raise CoachError("NVIDIA_API_KEY is not set (see .env.example)")
        chat = nvidia_chat(model, key)
    if segments is None:
        segments = load_segments(session["name"])
    return report(measurements(session, segments, event_id), chat, model)


def load_session(sessions_dir: Path, name: str) -> dict:
    return json.loads((sessions_dir / name / "session.json").read_text())
