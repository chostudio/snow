"""Download official FIS big air result sheets.

The crawl walks the public FIS site the way a person would: season calendar
-> event page -> race page -> the race's "Results - ..." PDF attachments.
Every page and PDF is cached under `data/raw/`, so a re-run only fetches what
is new, and requests are spaced out (`--delay`) to stay polite.

The `medias*.fis-ski.com` PDF mirror that search engines index is not used:
it returned 502 for every request on 2026-10-28. The same sheets are served
from www.fis-ski.com as "competition attachments", which is what the race
page links to.

    uv run python fis.py --seasons 2019-2026
"""

from __future__ import annotations

import argparse
import html
import json
import re
import time
import urllib.request
from pathlib import Path

BASE = "https://www.fis-ski.com/DB"
HERE = Path(__file__).parent
RAW = HERE / "data" / "raw"
# Identify ourselves plainly; no browser impersonation.
UA = "snow-scoring/0.1 (research; github.com/chostudio/snow)"

_last_request = 0.0


def fetch(url: str, dest: Path, delay: float) -> bytes:
    """GET `url` once, caching the body at `dest`."""
    global _last_request
    if dest.exists():
        return dest.read_bytes()
    wait = delay - (time.monotonic() - _last_request)
    if wait > 0:
        time.sleep(wait)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = resp.read()
    _last_request = time.monotonic()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(body)
    return body


def calendar_events(season: int, delay: float) -> list[int]:
    url = (f"{BASE}/snowboard/park-and-pipe/calendar-results.html?eventselection=results"
           f"&seasoncode={season}&categorycode=&disciplinecode=BA&gendercode=")
    page = fetch(url, RAW / "pages" / f"calendar-{season}.html", delay).decode("utf-8", "replace")
    return sorted({int(e) for e in re.findall(r"eventid=(\d+)", page)})


def event_races(season: int, event: int, delay: float) -> list[int]:
    url = f"{BASE}/general/event-details.html?sectorcode=SB&eventid={event}&seasoncode={season}"
    page = fetch(url, RAW / "pages" / f"event-{event}.html", delay).decode("utf-8", "replace")
    # Each race row reads "30 Sep 2024 6216 Big Air Big Air ANC M ..." just after its
    # link, so slopestyle/halfpipe races are skipped without fetching their pages.
    races = set()
    for m in re.finditer(r"results\.html\?sectorcode=SB&raceid=(\d+)", page):
        row = re.sub(r"<[^>]+>", " ", page[m.end():m.end() + 1500])
        if "Big Air" in row:
            races.add(int(m.group(1)))
    return sorted(races)


def race_info(race: int, delay: float) -> dict:
    """Discipline label and result attachments for one race."""
    url = f"{BASE}/general/results.html?sectorcode=SB&raceid={race}"
    page = fetch(url, RAW / "pages" / f"race-{race}.html", delay).decode("utf-8", "replace")
    # The race picker marks the current race: "19.10.2024 - Women's Big Air | WC".
    m = re.search(rf'<option value="{race}"\s*selected>([^<]*)</option>', page)
    label = html.unescape(m.group(1)).strip() if m else ""
    attachments = re.findall(
        r'data-link="(https://www\.fis-ski\.com/DB/v2/download/competition-attachment/[^"]+\.pdf)"'
        r'[^>]*?data-ga-download="([^"]*)"', page, re.S)
    return {"race": race, "label": label,
            "results": [(u, t) for u, t in attachments if t.startswith("Results")]}


def crawl(seasons: list[int], delay: float) -> list[dict]:
    index = []
    for season in seasons:
        events = calendar_events(season, delay)
        print(f"season {season}: {len(events)} events")
        for event in events:
            for race in event_races(season, event, delay):
                info = race_info(race, delay)
                if "Big Air" not in info["label"]:
                    continue
                for url, title in info["results"]:
                    pdf = RAW / "pdf" / f"{race}-{url.rsplit('/', 1)[1]}"
                    try:
                        fetch(url, pdf, delay)
                    except OSError as e:
                        print(f"  race {race} {title}: {e}")
                        continue
                    index.append({"season": season, "event": event, "race": race,
                                  "label": info["label"], "title": title,
                                  "pdf": str(pdf.relative_to(HERE))})
                print(f"  event {event} race {race}: {info['label']} ({len(info['results'])} result sheets)")
    out = HERE / "data" / "index.json"
    out.write_text(json.dumps(index, indent=1))
    print(f"{len(index)} result sheets -> {out.relative_to(HERE)}")
    return index


def _seasons(spec: str) -> list[int]:
    lo, _, hi = spec.partition("-")
    return list(range(int(lo), int(hi or lo) + 1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seasons", default="2019-2026", help="FIS season codes, e.g. 2025 or 2019-2026")
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between requests")
    args = ap.parse_args()
    crawl(_seasons(args.seasons), args.delay)
