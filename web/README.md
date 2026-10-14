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
| `POST /<name>/coach` *(optional)* | a `CoachReport` for one event |

Static files under `public/sessions/` are the default. Set `VITE_API_BASE` to
point at a real HTTP surface instead — nothing else in the UI changes.

Two stages are honestly marked as stand-ins in the header, and flip themselves
to live when the real thing lands:

- **Kinematics** — `tools/export_session.py` derives rotation, rates, axis tilt
  and air segmentation for display. If `data/analysis/<name>.json` exists, its
  `events` are used instead and the chip reads `snow-analyze`.
- **Nemotron reasoning** — the coach panel renders a template over measured
  numbers and says so. Implement `POST /<name>/coach` returning a `CoachReport`
  and the panel renders that instead, badged with the model name.

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
