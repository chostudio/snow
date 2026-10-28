"""``snow label``: the labels it writes, the guide it shows, the routes it serves.

No network, no GPU, no models: the machine guesses are off (``compute=False``)
and only their pure helpers are tested here.
"""

from __future__ import annotations

import json
import re
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

from snow.label import guide, prefill, server
from snow.learn import labels as L

REPO = Path(__file__).resolve().parents[1]
OLD_HEADER = "clip,air,takeoff_frame,landing_frame,event,sure,labeller,notes,frames,fps,slowmo_factor\n"


# --- the CSV --------------------------------------------------------------------

def test_a_file_without_trick_columns_still_loads(tmp_path):
    path = tmp_path / "airs.csv"
    path.write_text(OLD_HEADER + "run1,1,40,90,ev/a,y,cc,,200,30,1\nrun2,0,,,,y,cc,,100,30,1\n")
    assert L.validate(L.read(path)) == []
    sources = {s.name: s for s in L.gold_sources(path)}
    assert sources["run1"].takeoff == [40] and sources["run1"].landing == [90]
    assert sources["run2"].takeoff == []


def test_trick_columns_are_checked_and_every_one_admits_unsure():
    for column, values in L.TRICK_VALUES.items():
        assert L.UNSURE in values, column
    row = {"clip": "run1", "air": "1", "takeoff_frame": "40", "landing_frame": "90",
           "event": "ev/a", "sure": "y", "spin": "1080", "direction": "frontside",
           "axis": "double_cork", "grab": "mute", "stance": "unsure", "switch": "n"}
    assert L.validate([row]) == []
    assert any("spin '1000'" in p for p in L.validate([{**row, "spin": "1000"}]))
    assert any("sure 'maybe'" in p for p in L.validate([{**row, "sure": "maybe"}]))
    no_air = {"clip": "run2", "air": "0", "takeoff_frame": "", "landing_frame": "", "grab": "indy"}
    assert any("has a trick" in p for p in L.validate([no_air]))
    assert any("whole numbers" in p for p in L.validate([{**row, "takeoff_frame": "4.5"}]))


def test_replace_clip_keeps_other_clips_and_unknown_columns(tmp_path):
    path = tmp_path / "airs.csv"
    path.write_text(OLD_HEADER.rstrip("\n") + ",mine\n"
                    "a,1,1,20,ev/a,y,cc,,,,,x\nb,1,5,30,ev/b,y,cc,,,,,y\n")
    L.replace_clip(path, "a", [
        {"clip": "a", "air": "2", "takeoff_frame": "50", "landing_frame": "70", "event": "ev/a2",
         "spin": "unsure"},
        {"clip": "a", "air": "1", "takeoff_frame": "10", "landing_frame": "30", "event": "ev/a1"},
    ])
    rows = L.read(path)
    assert [(r["clip"], r["air"]) for r in rows] == [("a", "1"), ("a", "2"), ("b", "1")]
    assert rows[2]["mine"] == "y" and rows[1]["spin"] == "unsure"
    header = path.read_text().splitlines()[0].split(",")
    assert header[: len(L.COLUMNS)] == L.COLUMNS and header[-1] == "mine"
    assert L.validate(rows) == []
    assert {s.name for s in L.gold_sources(path)} == {"a", "b"}


# --- the guide --------------------------------------------------------------------

def test_guide_escapes_markup_and_links_sibling_docs_on_github():
    out = guide.to_html("# A <b>title</b>\n\nSee [x](segmenter.md) and [y](#a-b) `<i>`.\n")
    assert "<b>title" not in out and "&lt;b&gt;" in out
    assert guide.DOCS_URL + "segmenter.md" in out
    assert 'data-anchor="a-b"' in out and "<code>&lt;i&gt;</code>" in out


def test_every_guide_link_in_the_page_has_a_section():
    html = guide.to_html((REPO / "docs" / "labelling.md").read_text())
    ids = set(re.findall(r'id="([^"]+)"', html))
    js = (REPO / "src" / "snow" / "label" / "static" / "label.js").read_text()
    wanted = set(re.findall(r'guide: "([^"]+)"', js)) | set(re.findall(r'data-guide="([a-z-]+)"', js))
    wanted |= set(re.findall(r'data-anchor="([^"]+)"', html))
    assert wanted and wanted <= ids, wanted - ids


def test_the_page_offers_exactly_the_values_the_csv_accepts():
    js = (REPO / "src" / "snow" / "label" / "static" / "label.js").read_text()
    offered = {key: tuple(json.loads(f"[{values}]")) for key, values in
               re.findall(r'key: "(\w+)".*?values: \[([^\]]*)\]', js, flags=re.S)}
    assert {k: set(v) for k, v in offered.items()} == {k: set(v) for k, v in L.TRICK_VALUES.items()}


# --- machine-guess helpers ----------------------------------------------------------

def test_spans_are_half_open_in_clip_frames_and_stop_at_cuts():
    frame_ids = np.array([10, 11, 12, 13, 14, 15, 16])
    mask = np.array([0, 1, 1, 1, 1, 0, 1], dtype=bool)
    segment = np.array([0, 0, 0, 1, 1, 1, 1])
    assert prefill.spans_from_mask(mask, frame_ids, segment) == [(11, 13), (13, 15), (16, 17)]


def test_rounding_margin_is_one_on_a_step_and_zero_halfway():
    assert prefill.rounding_margin(1080) == 1.0
    assert prefill.rounding_margin(-990) == 0.0
    assert prefill.rounding_margin(1035) == 0.5


# --- the server -------------------------------------------------------------------

@pytest.fixture
def site(tmp_path):
    clips = tmp_path / "clips"
    clips.mkdir()
    (clips / "real.mp4").write_bytes(bytes(range(256)) * 4)
    (clips / "real.meta.json").write_text(json.dumps(
        {"name": "real", "fps": 30.0, "frames": 120, "width": 64, "height": 36,
         "slowmo_factor": 4.0}))
    for other in ("real.track.mp4", "synth01_flat_360.mp4", "awstest-real.mp4"):
        (clips / other).write_bytes(b"x")
    state = server.State(clips, tmp_path / "labels" / "airs.csv", tmp_path / "cache",
                         labeller="tester", compute=False)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(state))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", state
    httpd.shutdown()


def _get(url, headers=None):
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers or {})) as r:
        return r.status, dict(r.headers), r.read()


def _post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def test_lists_real_clips_only_with_their_slow_motion(site):
    base, _ = site
    _, headers, body = _get(base + "/api/clips")
    clips = json.loads(body)["clips"]
    assert [c["name"] for c in clips] == ["real"]
    assert clips[0]["slowmo_factor"] == 4.0 and clips[0]["frames"] == 120
    assert "Access-Control-Allow-Origin" not in headers


def test_saves_what_the_gold_loader_reads(site):
    base, state = site
    got = _post(base + "/api/save/real", {"rows": [
        {"air": "1", "takeoff_frame": "30", "landing_frame": "52", "event": "ev/x", "sure": "n",
         "spin": "unsure", "grab": "indy", "notes": "line\nbreak", "frames": "999"}]})
    assert got["problems"] == []
    row = L.read(state.csv_path)[0]
    assert row["clip"] == "real" and row["labeller"] == "tester" and row["frames"] == "120"
    assert row["slowmo_factor"] == "4.0" and row["notes"] == "line break"
    (src,) = L.gold_sources(state.csv_path)
    assert (src.takeoff, src.landing, src.group, src.slowmo) == ([30], [52], "ev/x", 4.0)
    got = _post(base + "/api/save/real", {"rows": [{"air": "1", "takeoff_frame": "60",
                                                   "landing_frame": "40", "event": "e"}]})
    assert any("not after takeoff" in p for p in got["problems"])
    assert _post(base + "/api/save/real", {"rows": [{"air": "0"}]})["problems"] == []


def test_video_seeks_and_names_cannot_escape(site):
    base, _ = site
    status, headers, body = _get(base + "/video/real", {"Range": "bytes=10-19"})
    assert status == 206 and body == bytes(range(10, 20))
    assert headers["Content-Range"] == "bytes 10-19/1024"
    for bad in ("/video/..%2Fx", "/video/synth01_flat_360", "/api/clip/nope", "/static/..%2F..%2Fx"):
        with pytest.raises(urllib.error.HTTPError) as err:
            _get(base + bad)
        assert err.value.code == 404


def test_page_guide_and_a_clip_without_poses(site):
    base, _ = site
    assert b"label.js" in _get(base + "/")[2]
    assert b'id="counting-the-spin"' in _get(base + "/api/guide")[2]
    clip = json.loads(_get(base + "/api/clip/real")[2])
    assert clip["info"]["has_poses"] is False and clip["rows"] == []
    assert _post(base + "/api/reading/real", {"takeoff": 1, "landing": 9}) == {
        "geometry": None, "classifier": None}
