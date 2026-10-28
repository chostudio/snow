"""The Nemotron coach, without the network: what the model is shown, and
what it is allowed to send back.

The prompt is the only place docs/validation.md's "claims you may not make"
reach the model, so it is pinned here; and a reply that does not match
``CoachReport`` must fail loudly after one retry, never fall through to the UI.
"""

from __future__ import annotations

import io
import json
import threading
import urllib.error
import urllib.request

import pytest

from snow import coach as C
from snow import serve


def _session(slowmo: float = 1.0, metrics: str = "snow-features") -> dict:
    return {
        "name": "run1",
        "clip": {"fps": 30.0, "slowmo_factor": slowmo, "real_fps": 30.0 * slowmo,
                 "duration": 24.0, "tracked_frames": 690, "total_frames": 720},
        "source": {"metrics": metrics, "rotation_measure": "signed angular path, camera removed",
                   "coach": "rule-based", "pose": "nvidia-body-pose-nim"},
        "gaps": [[300, 309]],
        "events": [{
            "id": "air-1", "kind": "air", "start_f": 400, "end_f": 520, "peak_f": 450,
            "t_start": 13.33, "t_end": 17.33,
            "metrics": {
                "rotation_deg": 1111.7, "nearest_spin_deg": 1080, "direction": "clockwise",
                "peak_rpm": -42.0, "air_time_s": 4.0, "pop_height_m": 0.81,
                "max_axis_tilt_deg": 114.5, "min_tuck_radius_m": 0.31,
                "landing_knee_flex_deg": 141.0, "rotation_decay_dps2": -12.0,
                "mean_confidence": 0.95,
            },
        }],
    }


SEGMENTS = [{"segment": 2, "frames": 132, "net_spin_deg": 650.8, "peak_spin_deg": 654.5,
             "total_rotation_deg": 1152.3, "axis_spin_deg": 962.0, "spin_path_deg": 1111.7,
             "spin_axis_tilt_deg": 61.3, "max_tilt_deg": 114.5, "median_quality": 0.949,
             "camera_rotation_deg": 153.3}]


def _good(**over) -> dict:
    reply = {
        "verdict": "Measured 1111.7 degrees of signed path; the count itself is unvalidated.",
        "headline_metric": {"label": "Spin (signed path)", "value": 1111.7, "unit": "°",
                            "event_id": "air-1"},
        "findings": [{"severity": "good", "title": "Strong cork", "detail": "Tilt 114.5°."}],
        "cues": ["Absorb earlier on the landing."],
    }
    reply.update(over)
    return reply


# What the model sees -------------------------------------------------------

def test_prompt_counts_spin_path_and_never_axis_spin():
    data = C.measurements(_session(), SEGMENTS)
    text = json.dumps(C.build_messages(data))
    assert "spin_path_deg" in text and "1111.7" in text
    assert "axis_spin_deg" not in text and "962.0" not in text
    # Vertical yaw degenerates when inverted; it is not offered as a count either.
    assert "net_spin_deg" not in text and "650.8" not in text


def test_lengths_are_never_named_in_metres():
    data = C.measurements(_session())
    event = data["events"][0]
    assert event["pop_height_uncalibrated"] == 0.81
    assert not [k for k in event if k.endswith("_m")]
    assert "Never express them in metres" in C.build_messages(data)[0]["content"]


def test_rates_are_clip_timebase_unless_slowmo_was_set():
    unset = C.measurements(_session(slowmo=1.0))
    assert "peak_rate_rpm_clip_timebase" in unset["events"][0]
    assert "slowmo_factor was NOT set" in C.build_messages(unset)[0]["content"]

    known = C.measurements(_session(slowmo=4.0))
    assert "peak_rate_rpm_real_time" in known["events"][0]
    system = C.build_messages(known)[0]["content"]
    assert "x4.0" in system and "NOT set" not in system


def test_prompt_carries_every_validation_caveat():
    system = C.build_messages(C.measurements(_session(metrics="display-shim")))[0]["content"]
    assert "UNVALIDATED" in system
    assert "camera cut was not measured" in system
    assert "was NOT subtracted" in system
    assert "frontside/backside" in system.lower()


def test_focus_event_is_requested():
    user = C.build_messages(C.measurements(_session(), event_id="air-1"))[1]["content"]
    assert user.startswith("Coach the air air-1")


# What it may send back -----------------------------------------------------

def _validate(reply: dict) -> dict:
    return C.validate_report(reply, C.measurements(_session(), SEGMENTS), "nvidia/test")


def test_a_valid_reply_becomes_a_coach_report():
    report = _validate(_good())
    assert report["source"] == "nemotron" and report["model"] == "nvidia/test"
    assert report["headline_metric"]["value"] == 1111.7


@pytest.mark.parametrize("reply, why", [
    (_good(source="nemotron"), "unexpected"),
    ({k: v for k, v in _good().items() if k != "cues"}, "missing"),
    (_good(findings=[{"severity": "fine", "title": "x", "detail": "y"}]), "severity"),
    (_good(findings=[]), "findings"),
    (_good(cues=["ok", 3]), "cues"),
    (_good(verdict="  "), "verdict"),
    (_good(headline_metric={"label": "x", "value": 1111.7, "unit": "°", "event_id": "air-9"}),
     "event_id"),
    (_good(headline_metric={"label": "x", "value": 1080, "unit": "°", "event_id": "air-1"}),
     "not one of the measured"),
    (_good(headline_metric={"label": "Pop", "value": 0.81, "unit": "m", "event_id": "air-1"}),
     "uncalibrated length"),
    (_good(headline_metric={"label": "x", "value": "1111.7", "unit": "°", "event_id": "air-1"}),
     "must be a number"),
    (_good(headline_metric=None), "only be null"),
])
def test_schema_mismatches_are_rejected(reply, why):
    with pytest.raises(C.CoachSchemaError, match=why):
        _validate(reply)


def test_reply_parsing_tolerates_fences_and_think_tags():
    text = "<think>hmm {not json}</think>\n```json\n" + json.dumps(_good()) + "\n```"
    assert C.parse_reply(text)["verdict"].startswith("Measured")
    with pytest.raises(C.CoachSchemaError, match="no JSON"):
        C.parse_reply("I think it was a 1080.")


def test_one_retry_with_the_reason_then_success():
    sent = []
    replies = iter(["not json", json.dumps(_good())])

    def chat(messages):
        sent.append(messages)
        return next(replies)

    report = C.report(C.measurements(_session()), chat, "nvidia/test")
    assert report["source"] == "nemotron"
    assert len(sent) == 2
    assert "rejected" in sent[1][-1]["content"]


def test_two_misses_fail_loudly():
    calls = []

    def chat(messages):
        calls.append(1)
        return json.dumps(_good(findings=[]))

    with pytest.raises(C.CoachSchemaError, match="twice"):
        C.report(C.measurements(_session()), chat, "nvidia/test")
    assert len(calls) == 2


def test_unknown_event_is_refused_before_any_call():
    def chat(messages):
        raise AssertionError("must not be called")

    with pytest.raises(ValueError, match="unknown event"):
        C.coach_session(_session(), "air-7", segments=[], chat=chat)


def test_model_id_is_configurable(monkeypatch):
    monkeypatch.delenv("SNOW_COACH_MODEL", raising=False)
    assert C.model_id() == C.DEFAULT_MODEL
    monkeypatch.setenv("SNOW_COACH_MODEL", "nvidia/other")
    assert C.model_id() == "nvidia/other"


# The HTTP surface ----------------------------------------------------------

@pytest.fixture
def server(tmp_path):
    (tmp_path / "run1").mkdir()
    (tmp_path / "index.json").write_text(json.dumps({"sessions": [{"name": "run1"}]}))
    (tmp_path / "run1" / "session.json").write_text(json.dumps(_session()))
    (tmp_path / "run1" / "clip.mp4").write_bytes(bytes(range(256)) * 4)

    def fake_coach(session, event_id):
        if event_id == "boom":
            raise C.CoachSchemaError("invalid twice")
        return C.report(C.measurements(session, [], event_id),
                        lambda m: json.dumps(_good()), "nvidia/test")

    httpd = serve.ThreadingHTTPServer(("127.0.0.1", 0), serve.make_handler(tmp_path, fake_coach))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def _post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=5)


def test_server_routes(server):
    assert json.load(urllib.request.urlopen(f"{server}/index.json"))["sessions"][0]["name"] == "run1"
    assert json.load(urllib.request.urlopen(f"{server}/run1/session.json"))["name"] == "run1"
    ranged = urllib.request.urlopen(urllib.request.Request(
        f"{server}/run1/clip.mp4", headers={"Range": "bytes=10-19"}))
    assert ranged.status == 206 and ranged.read() == bytes(range(10, 20))

    report = json.load(_post(f"{server}/run1/coach", {"event_id": "air-1"}))
    assert report["source"] == "nemotron"


def test_server_fails_loudly(server):
    with pytest.raises(urllib.error.HTTPError) as err:
        _post(f"{server}/run1/coach", {"event_id": "boom"})
    assert err.value.code == 502 and "invalid twice" in json.load(io.BytesIO(err.value.read()))["error"]
    with pytest.raises(urllib.error.HTTPError) as err:
        urllib.request.urlopen(f"{server}/..%2F/session.json")
    assert err.value.code == 404
