"""Safe video harvester and downloader for broadcast snowboard footage.

Designed to prevent IP bans, HTTP 429 rate limits, and bot detection:
1. Polite rate-limiting with configurable randomized jitter (5-15s).
2. Adaptive exponential backoff on HTTP 429 / Too Many Requests.
3. Bandwidth throttling via yt-dlp `--limit-rate` (default 3M).
4. Browser impersonation & cookie extraction (`--cookies-from-browser` / `--cookies`).
5. Process serialization via lockfile to guarantee single-threaded downloads.
6. Transcript-first discovery: downloads and scans subtitles for commentator trick
   callouts (spins, inverts, grabs) before downloading video media.
7. Targeted timestamp-based downloading (`--download-sections`) to download only
   relevant trick windows rather than full 2-hour broadcasts.
8. Cross-referencing FIS Big Air events in `scoring/data/jumps.csv` to generate
   optimal broadcast search queries.
9. Full integration with `snow ingest` (transcoding to H.264/yuv420p CFR, metadata).
10. Query & transcript caching in `data/harvest/` and `--dry-run` preview mode.
"""

from __future__ import annotations

import contextlib
import csv
import dataclasses
import hashlib
import html
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

from . import config as cfg
from . import ingest as ing
from . import meta as mt

# Try loading FIS trick decoder if available
try:
    if str(cfg.REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(cfg.REPO_ROOT))
    from scoring import tricks as fis_tricks
except ImportError:
    fis_tricks = None


# Default rate limiting and backoff settings
DEFAULT_MIN_DELAY = 5.0
DEFAULT_MAX_DELAY = 15.0
DEFAULT_LIMIT_RATE = "3M"
DEFAULT_BACKOFF_BASE = 30.0
DEFAULT_MAX_RETRIES = 4

# Standard desktop browser user-agent to avoid automated scraper blacklists
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)


class HarvestError(RuntimeError):
    """Base error for harvester failures."""
    pass


class HarvestRateLimitError(HarvestError):
    """Raised when rate limits (HTTP 429) persist after backoff retries."""
    pass


# -----------------------------------------------------------------------------
# Rate Limiting & Concurrency Control
# -----------------------------------------------------------------------------

@contextlib.contextmanager
def file_lock(lock_path: Path) -> Iterator[None]:
    """Advisory file lock to serialize downloads across processes.

    Guarantees that multiple concurrent invocations never hammer YouTube
    simultaneously from the same host.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(lock_path, "a+")
    try:
        try:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        except (ImportError, OSError):
            pass  # Fall back gracefully if unsupported
        yield
    finally:
        try:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except (ImportError, OSError):
            pass
        f.close()


class PoliteRateLimiter:
    """Manages inter-request delays and exponential backoff."""

    def __init__(
        self,
        min_delay: float = DEFAULT_MIN_DELAY,
        max_delay: float = DEFAULT_MAX_DELAY,
        backoff_base: float = DEFAULT_BACKOFF_BASE,
        max_retries: int = DEFAULT_MAX_RETRIES,
        sleep_fn=time.sleep,
    ) -> None:
        self.min_delay = max(0.0, min_delay)
        self.max_delay = max(self.min_delay, max_delay)
        self.backoff_base = backoff_base
        self.max_retries = max_retries
        self._sleep = sleep_fn
        self._last_request_time: float = 0.0

    def wait(self, min_delay: float | None = None, max_delay: float | None = None) -> float:
        """Wait with randomized jitter before dispatching a network call."""
        lo = self.min_delay if min_delay is None else min_delay
        hi = self.max_delay if max_delay is None else max_delay
        if hi <= 0:
            self._last_request_time = time.monotonic()
            return 0.0

        target_delay = random.uniform(lo, hi)
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if self._last_request_time > 0 and elapsed < target_delay:
            to_sleep = target_delay - elapsed
            self._sleep(to_sleep)
            self._last_request_time = time.monotonic()
            return to_sleep

        self._last_request_time = time.monotonic()
        return 0.0

    def calculate_backoff(self, attempt: int) -> float:
        """Compute exponential backoff with jitter: base * 2^attempt + jitter."""
        jitter = random.uniform(2.0, 8.0)
        return (self.backoff_base * (2 ** attempt)) + jitter

    def sleep_backoff(self, attempt: int, reason: str = "") -> float:
        """Sleep during backoff and log notification."""
        delay = self.calculate_backoff(attempt)
        msg = f"[polite-scraper] Rate limit encountered ({reason or '429 / bot detection'})."
        msg += f" Backing off for {delay:.1f}s (retry {attempt + 1}/{self.max_retries})..."
        print(msg, file=sys.stderr)
        self._sleep(delay)
        self._last_request_time = time.monotonic()
        return delay


# -----------------------------------------------------------------------------
# Caching Subsystem
# -----------------------------------------------------------------------------

class HarvestCache:
    """Caches YouTube searches and transcripts under data/harvest/."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or (cfg.DATA_DIR / "harvest")
        self.search_dir = self.root / "cache"
        self.transcripts_dir = self.root / "transcripts"
        self.metadata_dir = self.root / "metadata"
        self.ensure_dirs()

    def ensure_dirs(self) -> None:
        for d in (self.search_dir, self.transcripts_dir, self.metadata_dir):
            d.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _query_key(query: str) -> str:
        h = hashlib.sha256(query.strip().lower().encode("utf-8")).hexdigest()[:16]
        slug = re.sub(r"[^a-zA-Z0-9_-]", "_", query.strip().lower())[:24]
        return f"{slug}_{h}.json"

    def get_search(self, query: str) -> list[dict] | None:
        path = self.search_dir / self._query_key(query)
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return None
        return None

    def set_search(self, query: str, results: list[dict]) -> None:
        path = self.search_dir / self._query_key(query)
        path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    def get_transcript(self, video_id: str) -> list[dict] | None:
        path = self.transcripts_dir / f"{video_id}.json"
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return None
        return None

    def set_transcript(self, video_id: str, cues: list[SubtitleCue]) -> None:
        path = self.transcripts_dir / f"{video_id}.json"
        data = [asdict(c) for c in cues]
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def get_metadata(self, video_id: str) -> dict | None:
        path = self.metadata_dir / f"{video_id}.json"
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return None
        return None

    def set_metadata(self, video_id: str, meta: dict) -> None:
        path = self.metadata_dir / f"{video_id}.json"
        path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


# -----------------------------------------------------------------------------
# Subprocess / Safe yt-dlp Execution
# -----------------------------------------------------------------------------

def _is_rate_limited(proc: subprocess.CompletedProcess) -> tuple[bool, str]:
    """Examine process output for rate-limiting, 429, or bot detection signals."""
    combined = (proc.stderr or "") + "\n" + (proc.stdout or "")
    signals = [
        "HTTP Error 429",
        "Too Many Requests",
        "Sign in to confirm you're not a bot",
        "rate limit",
        "The read operation timed out",
        "blocked by cloudflare",
        "ip banned",
        "temporary ban",
    ]
    for s in signals:
        if s.lower() in combined.lower():
            return True, s
    return False, ""


def safe_run_ytdlp(
    args: list[str],
    limiter: PoliteRateLimiter | None = None,
    cookies_from_browser: str | None = None,
    cookies: str | Path | None = None,
    limit_rate: str | None = None,
    user_agent: str | None = DEFAULT_USER_AGENT,
    lock_path: Path | None = None,
    dry_run: bool = False,
    run_fn=subprocess.run,
) -> subprocess.CompletedProcess:
    """Execute yt-dlp with rate limiting, cookies, and retry backoff."""
    limiter = limiter or PoliteRateLimiter()
    lock_file = lock_path or (cfg.DATA_DIR / "harvest" / ".lock")

    cmd = [ing._require("yt-dlp")]
    if limit_rate:
        cmd += ["--limit-rate", limit_rate]
    if cookies_from_browser:
        cmd += ["--cookies-from-browser", cookies_from_browser]
    elif cookies:
        cmd += ["--cookies", str(cookies)]
    if user_agent:
        cmd += ["--user-agent", user_agent]
    cmd += args

    if dry_run:
        print(f"[dry-run] Would execute: {' '.join(cmd)}")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    with file_lock(lock_file):
        for attempt in range(limiter.max_retries + 1):
            limiter.wait()
            res = run_fn(cmd, capture_output=True, text=True)
            limited, reason = _is_rate_limited(res)
            if not limited and res.returncode == 0:
                return res

            if limited:
                if attempt < limiter.max_retries:
                    limiter.sleep_backoff(attempt, reason)
                    continue
                raise HarvestRateLimitError(
                    f"yt-dlp encountered rate limit/bot block after {limiter.max_retries} retries: {reason}\n"
                    f"{res.stderr[-1000:]}"
                )

            # Some errors are not rate limits (e.g. video unavailable, format not found)
            return res

    return res  # Fallback


# -----------------------------------------------------------------------------
# Subtitle & Transcript Parsing
# -----------------------------------------------------------------------------

@dataclass
class SubtitleCue:
    start_s: float
    end_s: float
    text: str

    @property
    def start_ts(self) -> str:
        return format_timestamp(self.start_s)

    @property
    def end_ts(self) -> str:
        return format_timestamp(self.end_s)


def parse_timestamp(ts: str) -> float:
    """Parse HH:MM:SS.mmm or MM:SS.mmm or SS.mmm into float seconds."""
    ts = ts.strip().replace(",", ".")
    parts = ts.split(":")
    try:
        if len(parts) == 3:
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        return float(parts[0])
    except ValueError:
        return 0.0


def format_timestamp(seconds: float) -> str:
    """Format seconds into HH:MM:SS (or HH:MM:SS.mmm)."""
    s = max(0.0, seconds)
    hrs = int(s // 3600)
    mins = int((s % 3600) // 60)
    secs = s % 60
    return f"{hrs:02d}:{mins:02d}:{secs:06.3f}"


def clean_subtitle_text(text: str) -> str:
    """Strip tags, timing codes, HTML entities, and collapse whitespace."""
    # Strip HTML and VTT formatting tags like <c>, </c>, <00:01:23.456>
    t = re.sub(r"<[^>]+>", " ", text)
    t = html.unescape(t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def parse_vtt(content: str) -> list[SubtitleCue]:
    """Parse WebVTT content into cleaned SubtitleCue objects."""
    cues: list[SubtitleCue] = []
    ts_pattern = re.compile(
        r"(\d+:\d+(?::\d+)?(?:[.,]\d+)?)\s*-->\s*(\d+:\d+(?::\d+)?(?:[.,]\d+)?)"
    )
    blocks = re.split(r"\n\s*\n", content)
    last_text = ""

    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        for i, line in enumerate(lines):
            m = ts_pattern.search(line)
            if m:
                start = parse_timestamp(m.group(1))
                end = parse_timestamp(m.group(2))
                raw_text = " ".join(lines[i + 1:])
                text = clean_subtitle_text(raw_text)
                # Deduplicate rolling auto-caption overlaps
                if text and text != last_text:
                    cues.append(SubtitleCue(start_s=start, end_s=end, text=text))
                    last_text = text
                break

    return cues


def parse_srt(content: str) -> list[SubtitleCue]:
    """Parse SubRip (.srt) subtitle content."""
    cues: list[SubtitleCue] = []
    blocks = re.split(r"\n\s*\n", content.strip())
    ts_pattern = re.compile(
        r"(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})"
    )
    last_text = ""

    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        ts_idx = -1
        for i, line in enumerate(lines):
            if ts_pattern.search(line):
                ts_idx = i
                break
        if ts_idx >= 0:
            m = ts_pattern.search(lines[ts_idx])
            if m:
                start = parse_timestamp(m.group(1))
                end = parse_timestamp(m.group(2))
                raw_text = " ".join(lines[ts_idx + 1:])
                text = clean_subtitle_text(raw_text)
                if text and text != last_text:
                    cues.append(SubtitleCue(start_s=start, end_s=end, text=text))
                    last_text = text

    return cues


def parse_json3(data: str | dict) -> list[SubtitleCue]:
    """Parse YouTube json3 caption format."""
    obj = json.loads(data) if isinstance(data, str) else data
    events = obj.get("events", [])
    cues: list[SubtitleCue] = []
    last_text = ""

    for ev in events:
        start_ms = ev.get("tStartMs", 0)
        dur_ms = ev.get("dDurationMs", 0)
        segs = ev.get("segs", [])
        raw = "".join(s.get("utf8", "") for s in segs)
        text = clean_subtitle_text(raw)
        if text and text != "\n" and text != last_text:
            start_s = start_ms / 1000.0
            end_s = (start_ms + dur_ms) / 1000.0
            cues.append(SubtitleCue(start_s=start_s, end_s=end_s, text=text))
            last_text = text

    return cues


def parse_subtitles(content: str, ext_hint: str = "") -> list[SubtitleCue]:
    """Auto-detect format and parse subtitle text into cues."""
    stripped = content.strip()
    if stripped.startswith("WEBVTT") or ext_hint.lower() == "vtt":
        return parse_vtt(stripped)
    if stripped.startswith("{") or ext_hint.lower() == "json3":
        return parse_json3(stripped)
    return parse_srt(stripped)


# -----------------------------------------------------------------------------
# Trick Detection & Keyword Extraction
# -----------------------------------------------------------------------------

SPIN_MAP = {
    # Numerical
    "180": 180, "360": 360, "540": 540, "720": 720, "900": 900,
    "1080": 1080, "1260": 1260, "1440": 1440, "1620": 1620, "1800": 1800,
    "1980": 1980, "2160": 2160,
    # Spelled out
    "one eighty": 180, "three sixty": 360, "five forty": 540,
    "seven twenty": 720, "nine hundred": 900, "ten eighty": 1080,
    "twelve sixty": 1260, "fourteen forty": 1440, "sixteen twenty": 1620,
    "eighteen hundred": 1800, "nineteen eighty": 1980, "twenty-one sixty": 2160,
    # Commentator shorthand (often "eighteen", "sixteen", "fourteen", "twelve", "ten" when accompanied by cork/spin)
    "eighteen": 1800, "sixteen": 1620, "fourteen": 1440, "twelve": 1260,
    "ten": 1080, "nine": 900, "seven": 720,
}

# Invert / axis descriptors sorted by length descending
AXES = [
    "quadruple cork", "quad cork", "triple cork", "double cork", "single cork",
    "double underflip", "underflip", "double rodeo", "rodeo",
    "flat spin", "flatspin", "misty flip", "misty", "bloody dracula",
    "wildcat", "tamedog", "haakon flip", "mctwist", "mc twist", "crippler", "cork",
]

# Stances / rotation directions
DIRECTIONS = [
    "switch backside", "switch frontside", "frontside", "backside",
    "caballero", "half cab", "cab", "switch",
]

# Snowboard grabs
GRABS = [
    "indy grab", "indy", "mute grab", "mute", "weddle grab", "weddle",
    "japan air", "japan", "stalefish", "stale fish", "melon grab", "melon",
    "nose grab", "nosegrab", "tail grab", "tailgrab", "crailt", "crail",
    "roast beef", "chicken salad", "seatbelt", "taipan", "method air", "method",
    "truckdriver", "canadian bacon", "lien", "bloody mary",
]

REPLAY_CUES = [
    "super slo-mo", "slow motion", "replay", "instant replay",
    "let's look at the replay", "here it is again", "on the replay",
    "stomped it", "stomped", "clean landing", "put it down", "huge air",
]


@dataclass
class TrickMatch:
    spins: list[int] = field(default_factory=list)
    axes: list[str] = field(default_factory=list)
    directions: list[str] = field(default_factory=list)
    grabs: list[str] = field(default_factory=list)
    replay: list[str] = field(default_factory=list)
    score: float = 0.0

    @property
    def has_trick(self) -> bool:
        # A valid trick needs at least a spin or an axis or a grab paired with rotation
        return bool(self.spins or (self.axes and self.axes != ["cork"]) or (self.grabs and self.directions))

    def describe(self) -> str:
        parts = []
        if self.directions:
            parts.append(self.directions[0].title())
        if self.axes:
            parts.append(self.axes[0].title())
        if self.spins:
            parts.append(str(self.spins[0]))
        if self.grabs:
            parts.append(f"{self.grabs[0].title()} Grab" if "grab" not in self.grabs[0] else self.grabs[0].title())
        return " ".join(parts) or "Trick Mention"


def detect_tricks_in_text(text: str) -> TrickMatch:
    """Scan string for snowboarding trick keywords and compute confidence score."""
    low = text.lower()
    match = TrickMatch()

    # Match spins
    # Only match shorthand numbers like "eighteen", "sixteen", etc. if cork/spin/trick is in text
    has_cork_or_spin = bool(re.search(r"\b(cork|spin|rotation|degrees|flip|air|trick)\b", low))

    for phrase, val in SPIN_MAP.items():
        if phrase in ("eighteen", "sixteen", "fourteen", "twelve", "ten", "nine", "seven"):
            if not has_cork_or_spin:
                continue
        pattern = r"\b" + re.escape(phrase) + r"\b"
        if re.search(pattern, low):
            if val not in match.spins:
                match.spins.append(val)

    # Match axes (longest first)
    for axis in sorted(AXES, key=len, reverse=True):
        pattern = r"\b" + re.escape(axis) + r"\b"
        if re.search(pattern, low):
            # Avoid matching 'cork' if 'triple cork' already matched
            if not any(axis in existing for existing in match.axes):
                match.axes.append(axis)

    # Match directions
    for direct in sorted(DIRECTIONS, key=len, reverse=True):
        pattern = r"\b" + re.escape(direct) + r"\b"
        if re.search(pattern, low):
            if not any(direct in existing for existing in match.directions):
                match.directions.append(direct)

    # Match grabs
    for grab in sorted(GRABS, key=len, reverse=True):
        pattern = r"\b" + re.escape(grab) + r"\b"
        if re.search(pattern, low):
            # Canonicalize grab name
            clean_grab = grab.replace(" grab", "").replace(" air", "")
            if not any(clean_grab in existing for existing in match.grabs):
                match.grabs.append(clean_grab)

    # Match replay cues
    for cue in REPLAY_CUES:
        if re.search(r"\b" + re.escape(cue) + r"\b", low):
            match.replay.append(cue)

    # Calculate confidence score
    score = 0.0
    if match.spins:
        score += 2.0
    if match.axes:
        score += 2.5
    if match.directions:
        score += 1.0
    if match.grabs:
        score += 1.5
    if match.replay:
        score += 0.5

    # Bonus for full combinations (e.g. Backside Triple Cork 1440 Weddle)
    if match.spins and match.axes:
        score += 2.0
    if match.spins and match.grabs:
        score += 1.5

    match.score = score
    return match


@dataclass
class TrickMention:
    start_s: float
    end_s: float
    text: str
    match: TrickMatch


@dataclass
class TrickCandidate:
    start_s: float
    end_s: float
    duration: float
    trick_name: str
    score: float
    transcript_excerpt: str
    mentions: list[TrickMention]

    @property
    def start_ts(self) -> str:
        return format_timestamp(self.start_s)

    @property
    def end_ts(self) -> str:
        return format_timestamp(self.end_s)


def scan_transcript_for_tricks(cues: list[SubtitleCue]) -> list[TrickMention]:
    """Scan transcript cues with a 2-cue sliding window to detect tricks."""
    mentions: list[TrickMention] = []

    # Single-cue inspection
    for cue in cues:
        m = detect_tricks_in_text(cue.text)
        if m.has_trick:
            mentions.append(TrickMention(cue.start_s, cue.end_s, cue.text, m))

    # Two-cue window for tricks spanning two consecutive subtitles
    for i in range(len(cues) - 1):
        c1, c2 = cues[i], cues[i + 1]
        # Only combine if cues are adjacent in time (< 4s apart)
        if 0 <= (c2.start_s - c1.end_s) <= 4.0:
            joined = f"{c1.text} {c2.text}"
            m = detect_tricks_in_text(joined)
            if m.has_trick:
                # If neither cue individually caught the combo, add the combined mention
                m1 = detect_tricks_in_text(c1.text)
                m2 = detect_tricks_in_text(c2.text)
                if m.score > max(m1.score, m2.score):
                    mentions.append(TrickMention(c1.start_s, c2.end_s, joined, m))

    # Deduplicate overlapping mentions that describe the exact same time
    unique: list[TrickMention] = []
    for mention in sorted(mentions, key=lambda x: (x.start_s, -x.match.score)):
        if not any(abs(mention.start_s - u.start_s) < 2.0 and mention.match.describe() == u.match.describe() for u in unique):
            unique.append(mention)

    return sorted(unique, key=lambda x: x.start_s)


def cluster_trick_windows(
    mentions: list[TrickMention],
    pre_pad: float = 12.0,
    post_pad: float = 16.0,
    merge_gap: float = 20.0,
) -> list[TrickCandidate]:
    """Cluster nearby trick mentions into downloadable video sections.

    Broadcasters call the trick on landing or during the slow-mo replay ~10-25s later.
    Padding and clustering ensure both the live hit and the slow-mo replay are captured.
    """
    if not mentions:
        return []

    clusters: list[list[TrickMention]] = []
    current: list[TrickMention] = [mentions[0]]

    for m in mentions[1:]:
        # If this mention is close to the previous mention in the current cluster
        prev = current[-1]
        if m.start_s - prev.end_s <= merge_gap:
            current.append(m)
        else:
            clusters.append(current)
            current = [m]
    if current:
        clusters.append(current)

    candidates: list[TrickCandidate] = []
    for cl in clusters:
        min_start = max(0.0, min(m.start_s for m in cl) - pre_pad)
        max_end = max(m.end_s for m in cl) + post_pad
        duration = max_end - min_start

        # Best trick description based on highest scoring mention
        best_mention = max(cl, key=lambda m: m.match.score)
        name = best_mention.match.describe()
        total_score = max(m.match.score for m in cl)

        # Build transcript excerpt
        excerpt = " / ".join(m.text for m in cl)

        candidates.append(
            TrickCandidate(
                start_s=round(min_start, 2),
                end_s=round(max_end, 2),
                duration=round(duration, 2),
                trick_name=name,
                score=total_score,
                transcript_excerpt=excerpt,
                mentions=cl,
            )
        )

    return candidates


# -----------------------------------------------------------------------------
# Cross-Referencing FIS Events
# -----------------------------------------------------------------------------

def load_fis_jumps(csv_path: Path | None = None) -> list[dict]:
    """Load scoring/data/jumps.csv with fallback."""
    path = csv_path or (cfg.REPO_ROOT / "scoring" / "data" / "jumps.csv")
    if not path.exists():
        return []

    jumps: list[dict] = []
    with open(path, mode="r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            jumps.append(row)
    return jumps


def search_fis_events(
    query: str = "",
    season: int | None = None,
    gender: str | None = None,
    rider: str | None = None,
    csv_path: Path | None = None,
) -> list[dict]:
    """Search FIS jumps and group by event with decoded tricks and top riders."""
    jumps = load_fis_jumps(csv_path)
    if not jumps:
        return []

    q_lower = query.lower().strip()
    rider_lower = rider.lower().strip() if rider else None

    # Group jumps by unique event key (season, place, category, gender)
    events: dict[str, dict[str, Any]] = {}

    for row in jumps:
        r_season = int(row.get("season", 0) or 0)
        r_place = row.get("place", "")
        r_cat = row.get("category", "")
        r_gender = row.get("gender", "")
        r_name = row.get("name", "")
        r_trick = row.get("trick", "")
        r_date = row.get("date", "")
        try:
            r_score = float(row.get("score", 0.0) or 0.0)
        except ValueError:
            r_score = 0.0

        if season and r_season != season:
            continue
        if gender and r_gender.upper() != gender.upper():
            continue
        if rider_lower and rider_lower not in r_name.lower():
            continue
        if q_lower:
            combined = f"{r_season} {r_place} {r_cat} {r_name} {r_date}".lower()
            if q_lower not in combined:
                continue

        key = f"{r_season}_{r_place}_{r_gender}_{r_date}"
        if key not in events:
            events[key] = {
                "season": r_season,
                "place": r_place,
                "category": r_cat,
                "gender": r_gender,
                "date": r_date,
                "riders": set(),
                "top_tricks": [],
                "jumps_count": 0,
            }

        ev = events[key]
        ev["jumps_count"] += 1
        ev["riders"].add(r_name)
        ev["top_tricks"].append((r_name, r_trick, r_score))

    results = []
    for ev in events.values():
        ev["riders"] = sorted(ev["riders"])
        # Sort top tricks by score descending
        sorted_tricks = sorted(ev["top_tricks"], key=lambda x: x[2], reverse=True)
        # Keep unique top trick performances
        seen = set()
        unique_tricks = []
        for name, trick, score in sorted_tricks:
            if (name, trick) not in seen:
                seen.add((name, trick))
                decoded = fis_tricks.decode(trick) if fis_tricks else {}
                unique_tricks.append({
                    "rider": name,
                    "code": trick,
                    "score": score,
                    "decoded": decoded,
                })
        ev["top_tricks"] = unique_tricks[:8]
        results.append(ev)

    # Sort events by season desc, place
    return sorted(results, key=lambda x: (x["season"], x["place"]), reverse=True)


def build_fis_queries(event: dict) -> list[str]:
    """Generate high-yield YouTube search queries for an FIS event."""
    season = event.get("season", "")
    place_clean = re.sub(r"\s*\([^)]*\)", "", event.get("place", ""))  # "Kreischberg (AUT)" -> "Kreischberg"
    gender_full = "Men" if event.get("gender") == "M" else "Women"

    queries = [
        f"FIS Snowboard World Cup {season} {place_clean} Big Air {gender_full} Final",
        f"FIS Snowboard {place_clean} {season} Big Air replay",
        f"Snowboard Big Air {season} {place_clean} Final highlights",
    ]

    # Add rider-specific query for top scoring rider
    if event.get("top_tricks"):
        top_rider = event["top_tricks"][0]["rider"].title()
        queries.append(f"{top_rider} {place_clean} {season} Big Air")

    return queries


# -----------------------------------------------------------------------------
# YouTube Discovery & Subtitle Harvester
# -----------------------------------------------------------------------------

@dataclass
class SearchResult:
    id: str
    title: str
    url: str
    duration: float
    uploader: str
    description: str = ""

    @property
    def duration_str(self) -> str:
        mins = int(self.duration // 60)
        secs = int(self.duration % 60)
        return f"{mins}:{secs:02d}"


def search_youtube(
    query: str,
    limit: int = 5,
    limiter: PoliteRateLimiter | None = None,
    cache: HarvestCache | None = None,
    cookies_from_browser: str | None = None,
    cookies: str | Path | None = None,
    dry_run: bool = False,
    no_cache: bool = False,
) -> list[SearchResult]:
    """Search YouTube politely, caching results under data/harvest/cache."""
    cache = cache or HarvestCache()
    limiter = limiter or PoliteRateLimiter()

    if not no_cache:
        cached = cache.get_search(query)
        if cached is not None:
            return [SearchResult(**r) for r in cached[:limit]]

    if dry_run:
        print(f"[dry-run] Would search YouTube for: '{query}' (limit={limit})")
        return []

    # Safe ytsearch invocation
    search_spec = f"ytsearch{limit}:{query}"
    args = [
        "--dump-json",
        "--flat-playlist",
        "--default-search", "ytsearch",
        "--no-warnings",
        search_spec,
    ]

    proc = safe_run_ytdlp(
        args,
        limiter=limiter,
        cookies_from_browser=cookies_from_browser,
        cookies=cookies,
        dry_run=False,
    )

    if proc.returncode != 0 and not proc.stdout:
        raise HarvestError(f"yt-dlp search failed:\n{proc.stderr.strip()[-1000:]}")

    results: list[SearchResult] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
            vid = item.get("id", "")
            title = item.get("title", "")
            url = item.get("url") or item.get("webpage_url") or f"https://www.youtube.com/watch?v={vid}"
            duration = float(item.get("duration") or 0.0)
            uploader = item.get("uploader") or item.get("channel") or ""
            desc = item.get("description", "")
            if vid and title:
                results.append(SearchResult(
                    id=vid, title=title, url=url, duration=duration,
                    uploader=uploader, description=desc[:300],
                ))
        except json.JSONDecodeError:
            continue

    # Save to cache
    cache.set_search(query, [asdict(r) for r in results])
    return results


def fetch_transcript(
    url_or_id: str,
    limiter: PoliteRateLimiter | None = None,
    cache: HarvestCache | None = None,
    cookies_from_browser: str | None = None,
    cookies: str | Path | None = None,
    dry_run: bool = False,
    no_cache: bool = False,
) -> list[SubtitleCue]:
    """Download and extract subtitles/auto-captions without downloading video media."""
    cache = cache or HarvestCache()
    limiter = limiter or PoliteRateLimiter()

    # Extract ID
    video_id = url_or_id
    if "v=" in url_or_id:
        video_id = url_or_id.split("v=")[1].split("&")[0]
    elif "/" in url_or_id:
        video_id = url_or_id.rstrip("/").split("/")[-1]

    if not no_cache:
        cached = cache.get_transcript(video_id)
        if cached is not None:
            return [SubtitleCue(**c) for c in cached]

    if dry_run:
        print(f"[dry-run] Would fetch transcript for {url_or_id} without downloading video.")
        return []

    # Temporary directory for subtitle download
    temp_dir = cache.root / "tmp_subs"
    temp_dir.mkdir(parents=True, exist_ok=True)
    out_template = str(temp_dir / f"{video_id}.%(ext)s")

    args = [
        "--skip-download",
        "--write-sub",
        "--write-auto-sub",
        "--sub-lang", "en.*,en",
        "--sub-format", "vtt/srv3/ttml/best",
        "-o", out_template,
        url_or_id if url_or_id.startswith("http") else f"https://www.youtube.com/watch?v={url_or_id}",
    ]

    proc = safe_run_ytdlp(
        args,
        limiter=limiter,
        cookies_from_browser=cookies_from_browser,
        cookies=cookies,
        dry_run=False,
    )

    # Locate downloaded subtitle file
    sub_files = list(temp_dir.glob(f"{video_id}.*"))
    cues: list[SubtitleCue] = []

    for sf in sub_files:
        if sf.suffix in (".vtt", ".srt", ".srv3", ".ttml", ".json3"):
            try:
                content = sf.read_text(encoding="utf-8", errors="replace")
                cues = parse_subtitles(content, ext_hint=sf.suffix.lstrip("."))
                if cues:
                    break
            except Exception:
                continue

    # Cleanup temp directory
    shutil.rmtree(temp_dir, ignore_errors=True)

    # Cache result even if empty so we don't repeat calls for captionless videos
    cache.set_transcript(video_id, cues)
    return cues


# -----------------------------------------------------------------------------
# Video Harvesting & Ingest Integration
# -----------------------------------------------------------------------------

def download_video_section(
    url: str,
    dest_stem: Path,
    start: str | None = None,
    duration: str | None = None,
    limit_rate: str = DEFAULT_LIMIT_RATE,
    limiter: PoliteRateLimiter | None = None,
    cookies_from_browser: str | None = None,
    cookies: str | Path | None = None,
    dry_run: bool = False,
) -> Path | None:
    """Download a video (or specific section) using yt-dlp.

    Uses `--download-sections "*start-end"` to minimize network transfer
    and avoid bot flagging.
    """
    if dry_run:
        print(f"[dry-run] Would download section of {url} (start={start}, duration={duration}, rate={limit_rate})")
        return dest_stem.with_suffix(".mp4")

    dest_stem.parent.mkdir(parents=True, exist_ok=True)
    out_pattern = str(dest_stem) + ".src.%(ext)s"

    # Clean old sources
    for stale in dest_stem.parent.glob(dest_stem.name + ".src.*"):
        with contextlib.suppress(OSError):
            stale.unlink()

    args = [
        "-f", "bv*[vcodec^=avc1]+ba/bv*+ba/b",
        "--merge-output-format", "mkv",
        "-o", out_pattern,
    ]

    # If start and duration are specified, calculate section window
    if start is not None and duration is not None:
        start_s = parse_timestamp(start)
        dur_s = float(duration)
        end_s = start_s + dur_s
        args += ["--download-sections", f"*{format_timestamp(start_s)}-{format_timestamp(end_s)}"]

    proc = safe_run_ytdlp(
        args + [url],
        limiter=limiter,
        cookies_from_browser=cookies_from_browser,
        cookies=cookies,
        limit_rate=limit_rate,
        dry_run=False,
    )

    if proc.returncode != 0:
        raise HarvestError(f"yt-dlp download failed:\n{proc.stderr.strip()[-1000:]}")

    files = sorted(dest_stem.parent.glob(dest_stem.name + ".src.*"))
    if not files:
        raise HarvestError(f"yt-dlp reported success but produced no source file at {dest_stem}")
    return files[0]


def harvest_clip(
    url: str,
    name: str,
    start: str | None = None,
    duration: str | None = None,
    fps: float = 30.0,
    slowmo_factor: float = 1.0,
    limit_rate: str = DEFAULT_LIMIT_RATE,
    limiter: PoliteRateLimiter | None = None,
    cookies_from_browser: str | None = None,
    cookies: str | Path | None = None,
    dry_run: bool = False,
    extra_meta: dict | None = None,
) -> Path | None:
    """Full harvest pipeline: download section -> transcode to NIM spec -> save ClipMeta."""
    dest = cfg.clip_path(name)
    if dry_run:
        print(f"[dry-run] Would harvest clip '{name}' -> {dest}")
        print(f"          Source: {url}")
        print(f"          Window: start={start}, duration={duration}, slowmo={slowmo_factor}x")
        print(f"          Rate limit: {limit_rate}")
        return dest

    cfg.ensure_dirs()
    temp_stem = cfg.CLIPS_DIR / f"_harvest_{name}"

    # Download polite section
    src_file = download_video_section(
        url=url,
        dest_stem=temp_stem,
        start=start,
        duration=duration,
        limit_rate=limit_rate,
        limiter=limiter,
        cookies_from_browser=cookies_from_browser,
        cookies=cookies,
        dry_run=False,
    )
    if not src_file or not src_file.exists():
        raise HarvestError(f"Failed to download source video for clip '{name}'")

    try:
        # Transcode to NVIDIA NIM-compliant H.264 / CFR video
        # If section download was already cut, transcode handles remux/CFR
        info = ing.transcode(
            src=src_file,
            dest=dest,
            fps=fps,
            # If section download was performed, ffmpeg start seek might be 0
            start=None if "--download-sections" in sys.argv else start,
            duration=duration if "--download-sections" not in sys.argv else None,
        )

        cfr_ok, detail = ing.frame_intervals_are_constant(dest)
        if not cfr_ok:
            print(f"  warning: CFR check flagged: {detail}", file=sys.stderr)

        # Write metadata
        clip = mt.ClipMeta(
            name=name,
            source=url,
            start=start,
            duration=duration,
            fps=info.fps,
            frames=info.frames,
            width=info.width,
            height=info.height,
            slowmo_factor=slowmo_factor,
            extra={"harvest": extra_meta or {}},
        )
        clip.save()
        return dest
    finally:
        # Cleanup temp source files
        if src_file.exists() and src_file.parent == cfg.CLIPS_DIR:
            with contextlib.suppress(OSError):
                src_file.unlink()
