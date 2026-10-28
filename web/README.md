# web — Snow VAR

A review dashboard for the pose output: video with the skeleton drawn on it, a
synced telemetry stack, a 3D pose inspector, a coaching panel and a printable
trick breakdown.

```bash
cd web
npm install
../.venv/bin/python tools/export_session.py run1   # build a session bundle
npm run dev                                        # http://localhost:5173
```

The exporter needs a finished `snow run`, i.e. `data/poses/<name>.json` and
`data/clips/<name>.mp4`. It writes `public/sessions/<name>/` (gitignored —
clips and poses never get committed).

## How it attaches to the pipeline

The UI never touches `src/snow/`. It reads one JSON bundle per clip, and
`src/api.ts` is the only file that knows where bundles come from:

| route | serves |
|---|---|
| `GET /index.json` | the list of sessions |
| `GET /<name>/session.json` | the bundle (`src/types.ts` is the contract) |
| `GET /<name>/clip.mp4` | the video |
| `POST /<name>/coach` *(optional)* | a `CoachReport` for one event, from Nemotron |

Static files under `public/sessions/` are the default. `snow serve` serves the
same routes plus the live coach; point `VITE_API_BASE` at it and nothing else
in the UI changes:

```bash
snow serve                                          # repo root, port 8000
VITE_API_BASE=http://127.0.0.1:8000 npm run dev
```

Two header chips say honestly where a number came from, and flip to live only
when the real thing produced it:

- **Kinematics**: without `snow features`, `tools/export_session.py` derives
  rotation from the rider's body axis with the camera **not** subtracted, and
  the chip stays a stand-in. With `data/poses/<name>.features.csv` it uses the
  camera-compensated `spin_path_deg` and the chip reads *Camera-compensated
  kinematics*.
- **Coach**: the bundle carries a rule-based template, badged *Coach
  (template)*. **Re-run analysis** posts to `POST /<name>/coach`; with
  `snow serve` behind it a Nemotron model on the NVIDIA API answers, the
  report is schema-checked, and the chip reads *Nemotron coach*. If the model
  fails or returns an invalid report the panel shows the error rather than a
  template. `export_session.py run1 --coach nemotron` bakes a real report into
  the bundle so the static build has one too. See
  [../docs/coach.md](../docs/coach.md).

## What the numbers mean

Rotation is integrated about the rider's **own** body axis (hip axis projected
perpendicular to the hip→shoulder vector), frame to frame, and never across a
camera cut — a broadcast camera pans with the rider, so root yaw in camera
coordinates would mix the camera's own rotation in. Rates use real-time fps
from `ClipMeta.slowmo_factor`, so conformed slow motion cannot inflate them.

Metric scale is uncalibrated: angles and ratios hold, absolute metres do not.

**Absolute spin counts are unvalidated.** The UI prints a "called spin" by
rounding the measured rotation to the nearest 180°, and on synthetic motion the
measurement is exact — but on the one real clip whose trick is named by its
source (`trick`, from a video titled *Frontside Double Cork 1080*) the pipeline
reads well short of 1080°. It is not established that the scouted shot is the
trick the title names, so that is an open discrepancy rather than a known error
rate. Read the rotation curve and the axis tilt, which are what the pose
actually supports; treat the big number at the top as provisional.

Air detection gates on a fraction of the clip's **own** peak rotation rate, not
an absolute RPM, because conformed slow motion scales every real-world rate by
a factor the file does not record. An absolute gate clipped a continuous 600°
slow-motion air down to the 142° sliver that happened to exceed it.
The air is found on the unsigned rotation rate and followed across pose holes
of up to 20 frames; its sense is read once per air from net vertical yaw, and
a turn the other way at either end (a wind-up) is trimmed off. Reading the
sense frame by frame split synthetic corks wherever the pipeline read a
stretch backwards ([../docs/validation.md](../docs/validation.md)).
The report labels the one length that is camera-relative rather than quietly
printing it as a pop height.

## Layout

```
tools/export_session.py   data/ -> public/sessions/<name>/   (the only Python here)
src/api.ts                where bundles come from
src/types.ts              the bundle contract
src/lib/                  scales, formatting, skeleton helpers, the video clock
src/components/           stage, timeline, tiles, coach, inspector, report
```

Charts are hand-rolled SVG — no chart library, no WebGL. The 3D inspector
projects to a 2D canvas, so there is no GPU context to lose mid-demo.

Keyboard: space toggles playback, ←/→ step a frame, shift+←/→ step ten.

## Known

`npm audit` reports a moderate esbuild advisory (GHSA-67mh-4wv8-2f99) reached
through vite. It lets a website you visit while `npm run dev` is running read
responses from the dev server; it does not affect `npm run build` or anything
served from `dist/`. Vite 5.4.21 is the newest release on this major and still
bundles the flagged esbuild, so clearing it needs a vite 8 upgrade — not worth
the breakage for a local review tool. Don't browse untrusted sites with the dev
server up, or run `npm run preview` instead.
