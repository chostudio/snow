"""Unit tests for the safe video harvester module (snow.harvest).

Mocks yt-dlp, subprocess, and network calls to verify:
- Polite rate-limiting, jitter, and concurrency locking
- Adaptive exponential backoff on HTTP 429 / bot detection
- Subtitle parsing (WebVTT, SRT, json3) and text cleaning
- Snowboard trick keyword extraction and cue clustering
- FIS event cross-referencing and query generation
- HarvestCache persistence and dry-run execution
- CLI command dispatch (snow harvest search, get, fis)
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest import mock

import pytest

from snow import config as cfg
from snow import harvest as harv
from snow import ingest as ing
from snow.cli import build_parser, main


# -----------------------------------------------------------------------------
# Rate Limiting & Backoff Tests
# -----------------------------------------------------------------------------

def test_rate_limiter_waits_when_interval_too_short():
    slept = []
    limiter = harv.PoliteRateLimiter(min_delay=5.0, max_delay=5.0, sleep_fn=slept.append)

    # First wait establishes base timestamp
    d1 = limiter.wait()
    assert d1 == 0.0

    # Immediate second wait must sleep the full delay
    d2 = limiter.wait()
    assert d2 == pytest.approx(5.0, abs=0.1)
    assert len(slept) == 1
    assert slept[0] == pytest.approx(5.0, abs=0.1)


def test_rate_limiter_no_sleep_when_interval_elapsed():
    slept = []
    limiter = harv.PoliteRateLimiter(min_delay=2.0, max_delay=2.0, sleep_fn=slept.append)

    limiter.wait()
    # Simulate passage of time
    limiter._last_request_time -= 5.0
    d = limiter.wait()
    assert d == 0.0
    assert len(slept) == 0


def test_rate_limiter_exponential_backoff_calculation():
    limiter = harv.PoliteRateLimiter(backoff_base=20.0)
    # attempt 0: ~20s + jitter(2..8)
    b0 = limiter.calculate_backoff(0)
    assert 22.0 <= b0 <= 28.0

    # attempt 1: ~40s + jitter
    b1 = limiter.calculate_backoff(1)
    assert 42.0 <= b1 <= 48.0

    # attempt 2: ~80s + jitter
    b2 = limiter.calculate_backoff(2)
    assert 82.0 <= b2 <= 88.0


def test_rate_limiter_sleep_backoff_invoked():
    slept = []
    limiter = harv.PoliteRateLimiter(backoff_base=10.0, sleep_fn=slept.append)
    limiter.sleep_backoff(attempt=1, reason="HTTP 429 Too Many Requests")
    assert len(slept) == 1
    assert 22.0 <= slept[0] <= 28.0


# -----------------------------------------------------------------------------
# yt-dlp & Subprocess Execution Tests
# -----------------------------------------------------------------------------

def test_safe_run_ytdlp_retries_on_429(tmp_path):
    lock_file = tmp_path / ".lock"
    calls = 0
    slept = []

    def mock_run(cmd, capture_output=True, text=True):
        nonlocal calls
        calls += 1
        if calls == 1:
            return subprocess.CompletedProcess(
                cmd, 1, stdout="", stderr="ERROR: HTTP Error 429: Too Many Requests"
            )
        return subprocess.CompletedProcess(cmd, 0, stdout='{"id": "ok"}\n', stderr="")

    limiter = harv.PoliteRateLimiter(
        min_delay=0.01, max_delay=0.01, backoff_base=0.05, max_retries=2, sleep_fn=slept.append
    )

    proc = harv.safe_run_ytdlp(
        ["--dump-json", "test_url"],
        limiter=limiter,
        lock_path=lock_file,
        run_fn=mock_run,
    )

    assert proc.returncode == 0
    assert calls == 2
    assert len(slept) >= 1  # Slept during backoff


def test_safe_run_ytdlp_raises_when_retries_exhausted(tmp_path):
    lock_file = tmp_path / ".lock"

    def mock_failing_run(cmd, capture_output=True, text=True):
        return subprocess.CompletedProcess(
            cmd, 1, stdout="", stderr="ERROR: Sign in to confirm you're not a bot"
        )

    limiter = harv.PoliteRateLimiter(
        min_delay=0.0, max_delay=0.0, backoff_base=0.01, max_retries=2, sleep_fn=lambda _: None
    )

    with pytest.raises(harv.HarvestRateLimitError):
        harv.safe_run_ytdlp(
            ["--dump-json", "test_url"],
            limiter=limiter,
            lock_path=lock_file,
            run_fn=mock_failing_run,
        )


def test_safe_run_ytdlp_appends_anti_ban_flags():
    recorded_cmd = []

    def mock_run(cmd, capture_output=True, text=True):
        recorded_cmd.extend(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    limiter = harv.PoliteRateLimiter(min_delay=0.0, max_delay=0.0)

    harv.safe_run_ytdlp(
        ["test_url"],
        limiter=limiter,
        cookies_from_browser="firefox",
        limit_rate="4M",
        user_agent="CustomTestAgent/1.0",
        run_fn=mock_run,
    )

    assert "--limit-rate" in recorded_cmd
    assert "4M" in recorded_cmd
    assert "--cookies-from-browser" in recorded_cmd
    assert "firefox" in recorded_cmd
    assert "--user-agent" in recorded_cmd
    assert "CustomTestAgent/1.0" in recorded_cmd


# -----------------------------------------------------------------------------
# Subtitle Parsing Tests
# -----------------------------------------------------------------------------

def test_parse_timestamp_formats():
    assert harv.parse_timestamp("00:01:23.456") == pytest.approx(83.456)
    assert harv.parse_timestamp("01:23.456") == pytest.approx(83.456)
    assert harv.parse_timestamp("00:01:23,456") == pytest.approx(83.456)  # SRT comma
    assert harv.parse_timestamp("45.5") == pytest.approx(45.5)
    assert harv.format_timestamp(83.456) == "00:01:23.456"


def test_parse_vtt_with_formatting_tags_and_deduplication():
    vtt = """WEBVTT
Kind: captions
Language: en

00:00:10.000 --> 00:00:13.500
<c>Here comes</c> Marcus Kleveland.

00:00:13.500 --> 00:00:17.000 align:start position:0%
Here comes Marcus Kleveland.

00:00:17.200 --> 00:00:21.000
Backside &amp; triple cork 1440!
"""
    cues = harv.parse_vtt(vtt)
    # Duplicate cue at 00:00:13.500 must be removed
    assert len(cues) == 2
    assert cues[0].start_s == 10.0
    assert cues[0].text == "Here comes Marcus Kleveland."
    assert cues[1].start_s == 17.2
    assert cues[1].text == "Backside & triple cork 1440!"


def test_parse_srt_format():
    srt = """1
00:01:15,200 --> 00:01:18,500
Massive jump from Su Yiming!

2
00:01:18,600 --> 00:01:22,000
Frontside 1800 indy grab!
"""
    cues = harv.parse_srt(srt)
    assert len(cues) == 2
    assert cues[0].start_s == pytest.approx(75.2)
    assert cues[0].text == "Massive jump from Su Yiming!"
    assert cues[1].start_s == pytest.approx(78.6)
    assert cues[1].text == "Frontside 1800 indy grab!"


def test_parse_json3_format():
    json3_data = {
        "events": [
            {
                "tStartMs": 12000,
                "dDurationMs": 3000,
                "segs": [{"utf8": "Look at that "}, {"utf8": "quad cork 1800!"}],
            }
        ]
    }
    cues = harv.parse_json3(json3_data)
    assert len(cues) == 1
    assert cues[0].start_s == 12.0
    assert cues[0].end_s == 15.0
    assert cues[0].text == "Look at that quad cork 1800!"


# -----------------------------------------------------------------------------
# Trick Keyword Extraction & Clustering Tests
# -----------------------------------------------------------------------------

def test_detect_tricks_in_text_combinations():
    # Backside Triple Cork 1440 Weddle
    t1 = "Unbelievable, backside triple cork 1440 weddle grab!"
    m1 = harv.detect_tricks_in_text(t1)
    assert m1.has_trick
    assert 1440 in m1.spins
    assert "triple cork" in m1.axes
    assert "backside" in m1.directions
    assert "weddle" in m1.grabs
    assert m1.describe() == "Backside Triple Cork 1440 Weddle Grab"

    # Spelled-out numbers: switch backside eighteen hundred
    t2 = "He just stomped a switch backside eighteen hundred indy!"
    m2 = harv.detect_tricks_in_text(t2)
    assert m2.has_trick
    assert 1800 in m2.spins
    assert "switch backside" in m2.directions
    assert "indy" in m2.grabs

    # Cab double cork 1080
    t3 = "Stomped the cab double cork 1080 on the super slo-mo replay."
    m3 = harv.detect_tricks_in_text(t3)
    assert 1080 in m3.spins
    assert "double cork" in m3.axes
    assert "cab" in m3.directions
    assert "super slo-mo" in m3.replay or "replay" in m3.replay


def test_scan_transcript_split_across_two_cues():
    cues = [
        harv.SubtitleCue(10.0, 12.0, "He is going for the triple cork"),
        harv.SubtitleCue(12.5, 14.5, "1440 weddle grab clean landing!"),
    ]
    mentions = harv.scan_transcript_for_tricks(cues)
    assert len(mentions) >= 1
    best = max(mentions, key=lambda m: m.match.score)
    assert 1440 in best.match.spins
    assert "triple cork" in best.match.axes
    assert "weddle" in best.match.grabs


def test_cluster_trick_windows_padding_and_merging():
    mentions = [
        harv.TrickMention(
            start_s=50.0,
            end_s=53.0,
            text="Backside 1440!",
            match=harv.detect_tricks_in_text("Backside 1440!"),
        ),
        # 12 seconds later: replay commentary
        harv.TrickMention(
            start_s=65.0,
            end_s=68.0,
            text="Look at that super slo-mo replay of the 1440 weddle grab",
            match=harv.detect_tricks_in_text("Look at that super slo-mo replay of the 1440 weddle grab"),
        ),
    ]

    candidates = harv.cluster_trick_windows(mentions, pre_pad=10.0, post_pad=15.0, merge_gap=20.0)
    # The two mentions should merge into one contiguous clip window
    assert len(candidates) == 1
    c = candidates[0]
    assert c.start_s == 40.0  # 50 - 10
    assert c.end_s == 83.0    # 68 + 15
    assert c.duration == 43.0
    assert 1440 in c.mentions[0].match.spins


# -----------------------------------------------------------------------------
# FIS Event Matching Tests
# -----------------------------------------------------------------------------

def test_search_fis_events_filter_by_place_and_season():
    events = harv.search_fis_events(query="Kreischberg", season=2021)
    assert len(events) >= 1
    ev = events[0]
    assert ev["season"] == 2021
    assert "Kreischberg" in ev["place"]
    assert len(ev["riders"]) > 0
    assert len(ev["top_tricks"]) > 0


def test_build_fis_queries_generates_broadcast_terms():
    mock_event = {
        "season": 2021,
        "place": "Kreischberg (AUT)",
        "category": "FIS SNOWBOARD WORLD CUP 2021",
        "gender": "M",
        "top_tricks": [{"rider": "MAX PARROT", "code": "fT16Wed", "score": 84.8}],
    }
    queries = harv.build_fis_queries(mock_event)
    assert any("Kreischberg" in q and "2021" in q and "Big Air" in q for q in queries)
    assert any("Max Parrot" in q for q in queries)


# -----------------------------------------------------------------------------
# Caching Tests
# -----------------------------------------------------------------------------

def test_harvest_cache_round_trip(tmp_path):
    cache = harv.HarvestCache(root=tmp_path)

    # Search cache
    search_data = [{"id": "vid123", "title": "Big Air Final", "url": "https://y.be/123", "duration": 120.0, "uploader": "FIS"}]
    cache.set_search("Kreischberg 2021", search_data)
    cached_search = cache.get_search("kreischberg 2021 ")  # case/whitespace agnostic
    assert cached_search is not None
    assert cached_search[0]["id"] == "vid123"

    # Transcript cache
    cues = [harv.SubtitleCue(1.0, 4.0, "Triple cork 1440!")]
    cache.set_transcript("vid123", cues)
    cached_cues = cache.get_transcript("vid123")
    assert cached_cues is not None
    assert cached_cues[0]["text"] == "Triple cork 1440!"


# -----------------------------------------------------------------------------
# Harvest Clip & Ingest Integration Tests
# -----------------------------------------------------------------------------

def test_harvest_clip_dry_run():
    dest = harv.harvest_clip(
        url="https://www.youtube.com/watch?v=mock123",
        name="test_dry_run_clip",
        start="00:01:10",
        duration="25",
        dry_run=True,
    )
    assert dest == cfg.clip_path("test_dry_run_clip")


def test_harvest_clip_mocked_execution(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "CLIPS_DIR", tmp_path / "clips")
    from snow import meta as mt
    monkeypatch.setattr(mt, "CLIPS_DIR", tmp_path / "clips")
    (tmp_path / "clips").mkdir(parents=True, exist_ok=True)

    # Mock download_video_section
    fake_src = tmp_path / "clips" / "_harvest_mock.src.mkv"
    fake_src.write_bytes(b"dummy mkv data")

    def mock_download(*args, **kwargs):
        return fake_src

    monkeypatch.setattr(harv, "download_video_section", mock_download)

    # Mock ingest.transcode
    fake_dest = cfg.CLIPS_DIR / "mock_trick.mp4"
    fake_dest.write_bytes(b"dummy mp4 data")

    fake_info = ing.VideoInfo(
        path=fake_dest,
        codec="h264",
        pix_fmt="yuv420p",
        width=1920,
        height=1080,
        fps=30.0,
        frames=600,
        duration=20.0,
        size_bytes=1000000,
    )
    monkeypatch.setattr(ing, "transcode", lambda *a, **k: fake_info)
    monkeypatch.setattr(ing, "frame_intervals_are_constant", lambda *a, **k: (True, "constant 30.000 fps"))

    out_path = harv.harvest_clip(
        url="https://youtube.com/watch?v=mock",
        name="mock_trick",
        start="00:01:00",
        duration="20",
        extra_meta={"trick": "Backside 1440"},
    )

    assert out_path == fake_dest
    meta_file = cfg.CLIPS_DIR / "mock_trick.meta.json"
    assert meta_file.exists()
    meta = json.loads(meta_file.read_text())
    assert meta["name"] == "mock_trick"
    assert meta["extra"]["harvest"]["trick"] == "Backside 1440"


# -----------------------------------------------------------------------------
# CLI Command Tests
# -----------------------------------------------------------------------------

def test_cli_harvest_search_dry_run():
    ret = main(["harvest", "search", "X Games 2024", "--dry-run", "--limit", "2"])
    assert ret == 0


def test_cli_harvest_fis_command():
    ret = main(["harvest", "fis", "Kreischberg", "--season", "2021"])
    assert ret == 0


def test_cli_harvest_get_dry_run():
    ret = main([
        "harvest", "get", "https://www.youtube.com/watch?v=mock",
        "--name", "test_trick",
        "--start", "00:01:23",
        "--duration", "20",
        "--dry-run",
    ])
    assert ret == 0
