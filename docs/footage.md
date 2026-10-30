# Finding and preparing footage

Covers `src/snow/harvest.py`, `src/snow/scout.py`, and `src/snow/ingest.py` — discovering broadcast
footage safely without IP bans, finding the usable slow-motion replays, and re-encoding for the NIM.

## Safe video harvesting (`snow harvest`)

Downloading broadcast footage at scale easily triggers IP bans, HTTP 429 rate limits, and YouTube bot
detection challenges. `snow harvest` provides a polite, anti-ban harvesting pipeline with transcript-first
filtering, cross-referencing against official FIS event results, and automated ingest integration.

### Anti-ban / polite scraping protections

1. **Randomized request jitter**: Enforces a configurable 5–15 second random delay (`--min-delay`, `--max-delay`)
   between all network requests.
2. **Adaptive backoff on 429**: Catches HTTP 429 "Too Many Requests" and bot challenge signals, applying
   exponential backoff ($base \times 2^{attempt} + jitter$, default starting at 30s) across up to 4 retries.
3. **Bandwidth throttling**: Constrains download rates using yt-dlp `--limit-rate` (default `3M`, configurable via
   `--limit-rate 5M`) to blend into normal residential streaming profiles.
4. **Browser impersonation & cookies**: Supports `--cookies-from-browser <browser>` (e.g. `chrome`, `firefox`,
   `safari`) and `--cookies <file>` alongside desktop User-Agent headers to avoid bot detection.
5. **Serialized downloads**: Uses an advisory file lock (`data/harvest/.lock`) so multiple CLI processes
   or agent jobs never hammer video platforms in parallel from the same IP.
6. **Local query & transcript caching**: Caches metadata, search results, and subtitles in `data/harvest/`
   so identical queries and videos never hit the network twice.
7. **`--dry-run` preview**: Allows testing queries, viewing detected trick timestamps, and checking download
   commands without initiating media downloads.

### 1. Discover broadcast replays

Search YouTube for competition broadcasts with polite rate limiting and subtitle extraction:

```bash
# Search for broadcast replays with dry-run preview
snow harvest search "X Games 2024 snowboard big air" --dry-run

# Search and inspect transcripts for commentator trick callouts
snow harvest search "FIS Snowboard World Cup 2021 Kreischberg Big Air Final" --extract-tricks
```

### 2. Transcript-first trick filtering

Rather than downloading a 2-hour broadcast, `snow harvest` pulls auto-captions or subtitles first, scans the text
for commentator trick callouts, and targets only the trick and replay timestamps.

Trick scanning identifies:
* **Spin counts**: 720, 900, 1080, 1260, 1440, 1620, 1800, 1980, 2160 (including spelled-out forms like "fourteen forty")
* **Inverts / axes**: quad cork, triple cork, double cork, underflip, rodeo, flat spin, misty
* **Directions**: frontside, backside, switch, cab, caballero
* **Grabs**: indy, mute, weddle, stalefish, japan, melon, nosegrab, tailgrab, crail, chicken salad

Nearby mentions (e.g. live hit + slow-mo replay 15 seconds later) are automatically clustered into a single
candidate clip window with padding:

```bash
# Preview trick callouts found in commentator audio
snow harvest get "https://www.youtube.com/watch?v=..." --dry-run

# Cut and ingest all detected tricks into data/clips/
snow harvest get "https://www.youtube.com/watch?v=..." --name xgames24 --auto-cut

# Or cut a specific window with cookies and rate limiting
snow harvest get "https://www.youtube.com/watch?v=..." --name run1 --start 00:01:25 --duration 28 \
    --cookies-from-browser chrome --limit-rate 3M
```

`snow harvest get` downloads only the target section using yt-dlp `--download-sections`, transcodes to
NIM-compliant H.264 CFR video, verifies constant frame rate, and writes `data/clips/<name>.meta.json` with
harvest provenance.

### 3. Cross-referencing FIS event results

`snow harvest fis` cross-references official competition results from `scoring/data/jumps.csv` (covering 2019–2026
World Cups, World Championships, and Olympic qualifiers). It extracts recorded tricks, top scores, and generates
exact broadcast search queries:

```bash
# Look up big air events at Kreischberg
snow harvest fis Kreischberg

# Filter by season and rider
snow harvest fis --season 2021 --rider "Gasser"

# Cross-reference and immediately search YouTube for footage
snow harvest fis "Chur" --season 2024 --search --extract-tricks
```

---

## Why scouting exists

Televised runs are mostly useless for pose estimation: wide shots where the
rider is 60 px tall, crowd cutaways, interviews, sponsor bumpers. The usable
material is the **super-slow-motion replay** — a high-speed camera, the rider
large and sharp, motion slow enough that nothing smears.

A 73-second edit can hold 22 shots of which two are worth running. `snow
scout` finds them in about 25 seconds.

```bash
snow scout "https://www.youtube.com/watch?v=..." --name olympics
```

```
  #        start     len    flow   rel   dup  subj  body  seen  suggested command
 13 00:00:34.935    2.8s    0.09   3.0   28%  0.58   33%  100%  --start ... --duration 2.8   SLOW DUP28% SHORT
 12 00:00:28.795    6.1s    0.18   1.4    2%  0.21  100%  100%  --start ... --duration 6.1
```

| column | meaning |
|---|---|
| `flow` / `rel` | median optical flow, and how much slower than this video's own median shot. `rel > 2` means slow motion |
| `dup` | fraction of near-identical frames — slow motion made by *duplicating* frames rather than by a high-speed camera |
| `subj` | median height of the largest person, as a fraction of frame height |
| `body` | fraction of frames where that person shows **hips and ankles** |
| `seen` | fraction of sampled frames containing a person |

`SHORT` marks shots under 120 frames, because the NIM buffers about that many
before emitting anything. Widen the window around one rather than sending it
alone.

### The close-up trap

`subj` alone is a trap: "a person fills the frame" is equally true of a
talking-head interview. The first version of scouting ranked a **face
close-up** as the best shot in the video.

So scouting runs a **pose** model (`yolo11s-pose.pt`), not a plain detector,
and scores `body` — the fraction of frames where COCO hips (11/12) *and*
ankles (15/16) are visible above 0.5 confidence. A shot only ranks if the
subject is a whole body.

A follow-on bug worth remembering: after switching the model, `body` read 0%
everywhere, because the CLI still defaulted `--weights yolo11s.pt` and a
detector has no keypoints to score. The default is now the pose model, and
passing a non-pose model raises a loud `ValueError` rather than silently
reporting zeros. **A check that cannot fail is not a check** — if you add a
metric, make its unavailable state an error, not a zero.

### Cut detection

Cuts come from whole-frame difference on a 320×180 downscale
(`--cut-threshold`, default 22.0), not from feature-matching quality.

An earlier version inferred a cut whenever feature or inlier count dropped,
which shattered a continuous shot into 43 segments. A frame that cannot be
solved now carries identity and is flagged unreliable instead of ending the
segment. That change took the same clip from 43 segments to 17, with the main
shot contiguous. A few single-frame segments still survive whip-pans and
camera flashes; they are cosmetic.

## Slow motion corrupts rates, not totals

Broadcast slow motion is shot fast and **conformed** to 30 fps. The file
claims 30 fps and the timestamps lie about how much real time passed.

- **Totals are fine.** It is the same turn, just sampled more often. 720° is
  still 720°.
- **Every per-second number is wrong** by the slow-motion factor. Degrees per
  second, RPM, air time.

Tell `ingest`:

```bash
snow ingest <src> trick --start 00:00:34 --duration 8 --slowmo-factor 4
```

It is stored in `data/clips/<name>.meta.json` as `slowmo_factor`, and
`real_fps = fps * slowmo_factor` flows through to `features` and the web
exporter.

**The factor cannot be recovered from the file.** A high `dup` in scout
implies frame doubling and gives you a rough estimate; a genuine high-speed
camera leaves no trace at all. If you do not know it, leave it at 1.0 and
treat every rate as a lower bound — and say so in anything user-facing. This
is exactly the kind of silent scale error that makes a confident wrong number.

## Ingest

```bash
snow ingest <url|file> run1 --start 00:00:34 --duration 24
```

Downloads (yt-dlp), cuts, re-encodes to H.264/4:2:0/8-bit CFR, caps height at
1080 and size under the hosted limit, then **verifies its own output**.

Verification re-probes the written file and checks each NIM requirement
separately rather than trusting ffmpeg's exit code. Constant frame rate is
checked from **real per-frame timestamps**, because a stream labelled CFR that
dropped a frame is precisely what the NIM rejects — and that is invisible in
the container metadata.

### yt-dlp goes stale fast

YouTube breaks old versions within months; a 17-month-old Homebrew copy failed
with SABR errors. `yt-dlp>=2025.10` is therefore a project dependency, and
`ingest._require()` prefers a tool installed **beside the running interpreter**
over whatever is on `PATH`, so the venv's pinned copy wins. `snow doctor`
resolves tools the same way, so it reports the binary that will actually run.

A user's Homebrew install is left alone.
