# snow

Measure what a snowboarder's body actually did, from video.

Point it at a run — a YouTube clip, an Olympic broadcast — and it finds the
usable shots, tracks the rider, extracts a 77-joint 3D skeleton for every
frame, subtracts the broadcast camera's own motion, and gives you the
rotation: how far the rider turned, about which axis, and how far off vertical
that axis was.

3D pose comes from the [NVIDIA 3D Body Pose NIM](https://build.nvidia.com/nvidia/body-pose),
running on NVIDIA's hosted endpoint. **No local GPU needed.** Everything else
runs on your machine.

```
source video
    │
    ├─ scout ──────── find the slow-motion, full-body shots worth using
    │
    └─ ingest ─────── cut + re-encode to what the NIM accepts
         └─ track ─── YOLO/ByteTrack → the NIM's bbox annotation
              ├─ crop ──── optional: reframe onto the rider
              └─ pose ──── the NIM → 77-joint 3D skeleton per frame
                   ├─ features ── camera-compensated rotation → CSV/Parquet
                   ├─ usd ─────── UsdSkelAnimation for Omniverse
                   └─ web ─────── a browser dashboard to review it
```

## Setup

```bash
scripts/setup.sh          # venv (python 3.12), deps, vendored NVIDIA client
cp .env.example .env      # then paste your key from build.nvidia.com
source .venv/bin/activate
snow doctor               # should end with "all good"
```

The NVIDIA client supports Python 3.10–3.12 only, so the venv is pinned to
3.12 regardless of your system Python.

## Try it

```bash
snow scout "https://www.youtube.com/watch?v=..." --name olympics
```

Scout lists the shots in the video, ranked. Most televised snowboarding is
useless for pose — wide shots where the rider is 60 px tall, crowd cutaways,
interviews. What you want is the **super-slow-motion replay**: high-speed
camera, rider large and sharp. Scout flags those.

Pick a shot and run it:

```bash
snow ingest <url|file> run1 --start 00:00:34 --duration 24
snow track  run1 --preview --show-candidates
snow pose   run1 --overlay
snow features run1
```

`--overlay` writes a video with the skeleton drawn on the rider, which is the
fastest way to see whether any of it worked.

Then review it in the browser:

```bash
cd web && npm ci
../.venv/bin/python tools/export_session.py run1
npm run dev
```

<small>A 24-second 1080p clip is about 19 seconds of inference.</small>

## Two things that will bite you

**Broadcast cuts shatter tracking.** Every camera cut makes the tracker issue
the rider a new id, so automatic selection covered only 35% of a test clip.
Run `--show-candidates`, watch the preview, and stitch the ids yourself:

```bash
snow track run1 --merge "1,8,945,1097,1738,1842" --preview   # 35% -> 73%
```

It is manual on purpose — guessing which track is the same person across a cut
can silently fuse two riders into one.

**Slow motion lies about time.** Broadcast slow motion is shot fast and
conformed to 30 fps, so the file claims 30 fps and the timestamps are wrong.
Totals are fine (720° is still 720°) but every per-second number is off by the
slow-motion factor. If you know it, pass it:

```bash
snow ingest <src> trick --start 00:00:34 --duration 8 --slowmo-factor 4
```

It cannot be recovered from the file.

## How much to trust the numbers

Short version: **angles and comparisons yes, absolute spin counts and metres
not yet.**

- On synthetic motion the maths is exact — a known 1080 about a known axis
  comes back as 1080, within 2°.
- On real footage, absolute spin counts are **unvalidated**. On a clip from a
  video titled *"Frontside Double Cork 1080"* the pipeline reads 668°. It has
  not been established that the shot the scout picked is the trick the title
  names, so that is an open question rather than a known error rate.
- Metric scale is uncalibrated — the focal length is assumed. Exported
  skeletons come out about 1.2 m tall. Angles and ratios hold; metres do not.

[docs/validation.md](docs/validation.md) is the honest, complete version.
Read it before you quote a number.

## Going deeper

| | |
|---|---|
| [docs/footage.md](docs/footage.md) | finding usable shots, slow motion, encoding rules |
| [docs/tracking.md](docs/tracking.md) | the bbox format, stitching tracks, reframing safely |
| [docs/nim.md](docs/nim.md) | the NIM: endpoint, input rules, output format, self-hosting |
| [docs/rotation.md](docs/rotation.md) | camera compensation and the three rotation measures |
| [docs/usd.md](docs/usd.md) | USD / Omniverse export |
| [docs/validation.md](docs/validation.md) | what is proven and what is not |
| [web/README.md](web/README.md) | the review dashboard |
| [AGENTS.md](AGENTS.md) | architecture map, for agents working on the code |

## Layout

```
src/snow/           the pipeline, one module per stage
web/                review dashboard (React) + session exporter
docs/               the deep dives above
tests/              27 tests, ~3s, no network and no GPU
scripts/setup.sh    venv, deps, vendored NVIDIA client
vendor/             upstream NVIDIA gRPC client (cloned, gitignored)
data/               clips, tracks, poses (gitignored)
```

## Next

Trick segmentation — splitting a run into individual hits — and the ML work on
top of the extracted features. Closing the validation question above comes
first, though: every downstream label depends on the spin count being right.
