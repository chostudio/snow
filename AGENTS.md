# AGENTS.md

Orientation for agents working on this repo. This file is a **map**: what the
pieces are, how they connect, and which invariants must not be broken. It
deliberately does not explain any stage in depth — each one has its own doc,
linked below. Read this file, then read only the doc for the part you are
touching.

## What this is

A pipeline that turns snowboarding video into measured body rotation. Source
video goes in; per-frame 3D skeletons, camera-compensated rotation features, a
USD animation and a browser review dashboard come out.

3D pose comes from the **NVIDIA 3D Body Pose NIM**, called over gRPC on
NVIDIA's hosted NVCF endpoint. There is no local GPU and no model in this repo.
Everything around the NIM — shot selection, encoding, tracking, framing,
rotation maths, export, UI — is ours. The dashboard's coach is a **Nemotron**
model on NVIDIA's chat completions API, shown only the measured numbers
(`src/snow/coach.py`, [docs/coach.md](docs/coach.md)).

## The shape of it

```
source video
    │
    ├─ scout ──────── rank shots; find the slow-motion, full-body ones
    │
    └─ ingest ─────── cut + re-encode to exactly what the NIM accepts
         └─ track ─── YOLO/ByteTrack → the NIM's bbox annotation file
              ├─ crop ──── optional: reframe onto the rider
              └─ pose ──── the NIM → 77-joint 3D skeleton per frame
                   ├─ features ── camera-compensated rotation → CSV/Parquet
                   ├─ usd ─────── UsdSkelAnimation for Omniverse, camera removed
                   └─ web/tools/export_session.py → the dashboard's bundle
                        └─ coach / serve ── Nemotron over the bundle's numbers
```

Stages talk to each other **only through files in `data/`**, keyed by a clip
`name`. Nothing holds state in memory across stages, and any stage can be re-run
alone. That is the main reason the pipeline is debuggable, so preserve it.

## Where everything lives

| Module | Does | Deep dive |
|---|---|---|
| `src/snow/scout.py` | split a broadcast into shots, rank them for pose work | [docs/footage.md](docs/footage.md) |
| `src/snow/ingest.py` | download, cut, re-encode, verify the NIM's input rules | [docs/footage.md](docs/footage.md) |
| `src/snow/track.py` | detect/track the rider, stitch its fragments, re-find it upside down, write the bbox annotation | [docs/tracking.md](docs/tracking.md) |
| `src/snow/crop.py` | reframe onto the rider, with correct intrinsics | [docs/tracking.md](docs/tracking.md) |
| `src/snow/pose.py` | drive the NIM, summarise its output | [docs/nim.md](docs/nim.md) |
| `src/snow/_nim_launcher.py` | keep the API key out of the process list | [docs/nim.md](docs/nim.md) |
| `src/snow/camera.py` | camera rotation from a background homography | [docs/rotation.md](docs/rotation.md) |
| `src/snow/features.py` | subtract the camera, integrate what is left | [docs/rotation.md](docs/rotation.md) |
| `src/snow/usd.py` | UsdSkelAnimation export | [docs/usd.md](docs/usd.md) |
| `src/snow/skeleton.py` | Nova-77 parent hierarchy. 19 lines, read it directly | — |
| `src/snow/meta.py` | per-clip metadata (`fps`, `slowmo_factor`, crop plan) | — |
| `src/snow/config.py` | paths, endpoint settings, `.env` loading | — |
| `src/snow/coach.py` | Nemotron on the NVIDIA API over a bundle's measurements; schema-checked `CoachReport` | [docs/coach.md](docs/coach.md) |
| `src/snow/serve.py` | `snow serve`: the dashboard's routes plus `POST /<name>/coach`, stdlib only | [docs/coach.md](docs/coach.md) |
| `src/snow/label/` | `snow label`: the hand-labelling page (stdlib server, plain JS in `static/`), machine pre-fills in `prefill.py`, the guide rendered from `docs/labelling.md` | [docs/labelling.md](docs/labelling.md) |
| `src/snow/cli.py` | argparse surface for every stage | — |
| `src/snow/synth/` | synthetic tricks with exact labels: Nova-77 rider in Blender, renders, comparison | [docs/synthetic.md](docs/synthetic.md) |
| `src/snow/learn/` | the learned half: model A, the per-frame airborne segmenter (`snow segmenter`), and model B, the trick classifier (`snow classify`) | [docs/segmenter.md](docs/segmenter.md), [docs/classifier.md](docs/classifier.md) |
| `web/` | review dashboard + session exporter | [web/README.md](web/README.md) |

Cross-cutting: **[docs/validation.md](docs/validation.md)** — what is actually
proven, what is not, and which claims you may not make. Read it before you
write anything user-facing about accuracy.

**[docs/training.md](docs/training.md)** is the design for the learned half
(airborne segmenter, trick classifier, judging score). Models A and B are
built, both trained and measured on synthetic renders only. The segmenter
([docs/segmenter.md](docs/segmenter.md)) is opt-in via `snow features
--segmenter` and **not to be enabled on real footage** until a hand-labelled
gold set exists. The classifier ([docs/classifier.md](docs/classifier.md))
reads the turn's sense and the stance well and does not beat the geometric
spin count or axis, so the dashboard's numbers stay geometric. The judge is
not built. Its
**Synthetic ground truth in Omniverse** section is now built, as
`snow synth` ([docs/synthetic.md](docs/synthetic.md)): parametric tricks on a
rider rigged as the NIM's own 77-joint skeleton, rendered and run through the
unchanged pipeline. The published table comes from Blender renders; the
Replicator renderer is written but has not run on an RTX machine yet.

## Artifacts on disk

Everything under `data/` is gitignored. For a clip named `run1`:

| Path | Written by | Holds |
|---|---|---|
| `data/clips/run1.mp4` | ingest | the clip, H.264/CFR, under 50 MB |
| `data/clips/run1.meta.json` | ingest | fps, `slowmo_factor`, crop plan, `extra` |
| `data/clips/run1.tracks.json` | track | raw tracker output, cached |
| `data/clips/run1.bbox.txt` | track | the NIM's annotation (format is strict) |
| `data/clips/run1.track.mp4` | track | `--preview` overlay |
| `data/clips/run1.zoom.mp4`, `.zoom.bbox.txt`, `.crop.json` | crop | reframed clip, remapped boxes, the plan |
| `data/poses/run1.json` | pose | **JSON Lines**, one object per frame |
| `data/poses/run1.overlay.mp4` | pose | `--overlay` render |
| `data/poses/run1.camera.npz` | features, usd | cached camera track |
| `data/poses/run1.features.{csv,parquet,json}` | features | the measurements |
| `data/poses/run1.usda` | usd | the skeletal animation |
| `web/public/sessions/run1/` | `export_session.py` | `session.json` + `clip.mp4`; `--coach nemotron` bakes in a real coach report |
| `data/synth/<clip>/` | `snow synth` | trick, exact labels, ground truth, scene USD, render |
| `data/poses/<clip>-oracle.*` | `snow synth prepare` | ground truth in the NIM's schema, through `features` |
| `data/models/segmenter/` | `snow segmenter` | dataset bundles, evaluation runs; weights also in S3 |
| `data/weights/segmenter.pt` | you, by hand | segmenter weights `snow features` picks up by default; absent, the old heuristic runs |
| `data/models/classifier/` | `snow classify` | dataset, `classifier.pt` (weights in S3, never committed), cv and split reports |
| `data/labels/real_airs.csv` | `snow label` | the human gold set: airs and factored tricks (`learn/labels.py` format) |
| `data/labels/prefill/<clip>.json` | `snow label` | cached machine guesses, keyed by input sizes and mtimes; safe to delete |

## Invariants

Break one of these and the output is wrong in a way that looks plausible.
Each is load-bearing and each is explained in the linked doc.

1. **The NIM does not detect people.** You supply tracked boxes per person per
   frame. Its input rules are strict and its errors are unhelpful
   (`INVALID_ARGUMENT`, `INTERNAL`). → `docs/nim.md`
2. **`data/poses/<name>.json` is JSON Lines, not JSON**, despite the
   extension, which the upstream client enforces. → `docs/nim.md`
3. **The NIM's camera frame is X right, Y DOWN, Z forward** (OpenCV). World up
   is `(0, -1, 0)`. Verified from data, not assumed. → `docs/rotation.md`
4. **`joint_rotations[0]` *is* `root_pose.rotation`**, and `rest_pose[0]` is
   the origin — so the root must not be transformed twice. → `docs/usd.md`
5. **A crop window that pans or zooms moves the principal point**, and a
   *time-varying* intrinsic error reads as rotation the rider never performed.
   Windows are static and fixed-size by default. → `docs/tracking.md`
6. **Rotation is never integrated across a camera cut.** The turn that
   happened off-camera was not measured, so it is not counted. Every stage
   that accumulates respects segment boundaries. → `docs/rotation.md`
7. **Metric scale is uncalibrated, and so is camera rotation.** The focal
   length is assumed, so body angles and ratios hold and absolute metres do
   not; on a long lens `camera_rotation_deg` is over-read several times. Never
   print a length, or a camera pan, as though it were real. → `docs/validation.md`
8. **Slow motion corrupts rates, not totals.** Conformed slow motion claims 30
   fps and the timestamps lie; the factor cannot be recovered from the file.
   Anything per-second is wrong by that factor unless `slowmo_factor` is set.
   → `docs/footage.md`
9. **Absolute spin counts are unvalidated on real footage.** The headline
   is `spin_path_deg`: it names 20 of 20 synthetic tricks on exact poses,
   but through the pipeline corks failed, because the tracker dropped the
   inverted rider (the NIM reads them fine when boxed). Stitching and rotated
   recovery in `track.py` fixed the boxes, and the dashboard's air detector
   now reads each air's sense once: the pipeline names 13 of 20 (7 of 13
   corks, from 1); most of the rest over-read a step from pose jitter. On
   the 285-clip training set it names only 116 of 281, and 17 of 69 at 4x
   slow motion: the 20 were a narrow sample.
   `axis_spin_deg` undercounts corks by definition; never promote it back.
   → `docs/validation.md`, `docs/rotation.md`, `docs/tracking.md`

## Secrets

The NVIDIA API key lives in `.env` at the repo root, `chmod 600`, gitignored.
**Never commit it, never print it in full, never pass it as a CLI argument.**

The upstream client only accepts `--api-key` on the command line, which would
expose it in the process list to every user on the machine.
`src/snow/_nim_launcher.py` works around this by injecting it into `sys.argv`
after the process has started. Keep that property.

The coach (`src/snow/coach.py`) uses the same key against NVIDIA's chat
completions API, read from the environment and sent only in the
`Authorization` header. It is never logged or put in an error message.

Before any commit, grep the staged diff:

```bash
git diff --cached | grep -niE "nvapi-|api[_-]key|secret|bearer"
```

## Working here

```bash
source .venv/bin/activate
snow doctor                 # tools, packages, vendored client, endpoint, key
python -m pytest -q         # all tests, ~20s, no network and no GPU
```

The venv is pinned to **Python 3.12**: the NVIDIA client supports 3.10–3.12
only, and the system Python here is 3.14. `numpy` is pinned `<2` to match the
client, which forces `pyarrow<17`.

`vendor/nim-clients/` is the upstream NVIDIA gRPC client, cloned by
`scripts/setup.sh` and gitignored. Its sample assets are git-lfs, so the clone
skips smudging; do not add `git-lfs` as a requirement.

For the web app: `cd web && npm ci && npm run dev`. It builds under strict
TypeScript and must stay console-clean.

### Testing

Tests are fast, hermetic and run without the NIM. The suite is the real
specification for these things:

- `tests/test_bbox_format.py` — parses our annotation with **NVIDIA's own
  reader**, not a reimplementation of it.
- `tests/test_rotation.py` — synthetic motion with a known answer: a rider
  turning a known amount about a known axis, and camera compensation in both
  directions.
- `tests/test_crop_and_usd.py` — crop coordinate round-trips, and the USD is
  validated with **USD's own skinning resolver** when `usd-core` is installed.
- `tests/test_usd_camera.py` — a rider fixed in the world under a panning
  camera and a cut: the exported root must hold still within each shot and
  never carry one shot's pan into the next.
- `tests/test_camera_cuts.py` — a whip pan over high-contrast texture is one
  shot, and a cut to a different scene is two.
- `tests/test_track_recovery.py` — rotated boxes map back exactly; stitching
  joins the rider's fragments but not a bystander or across a cut; a
  recovery chain that wanders onto a spectator is dropped whole.
- `tests/test_air_events.py` — the dashboard's air detector keeps one trick
  one air across a pose hole, a few flip frames and a stretch read
  backwards, and leaves a wind-up out of it.
- `tests/test_synth.py` — synthetic tricks really have the inversions their
  names claim, labels are measured from motion, and ground truth in the NIM's
  schema obeys the same forward kinematics as the NIM's own output.
- `tests/test_coach.py` — what Nemotron is shown (the spin count is
  `spin_path_deg`, never `axis_spin_deg`; lengths uncalibrated; rates in the
  clip's timebase unless `slowmo_factor` was set) and what it may send back
  (exact `CoachReport` shape, one retry, then a loud failure). No network.
- `tests/test_segmenter.py` — the segmenter's inputs are scale-free and
  identical for training and serving, a mirror changes only what it should,
  no window or interpolation crosses a cut, folds never split a group,
  oracle skeletons are refused, and without weights `snow features` runs
  the old heuristic. Torch-dependent cases skip without torch.
- `tests/test_classifier.py` — the classifier's air encoding: a mirror
  reverses the turn and flips sense and stance but not frontside/backside, slow
  motion encodes like real time, an air is never read across a cut, and
  prediction is mirror-equivariant.
- `tests/test_labelling.py` — `snow label`: an old labels CSV still loads,
  every trick field admits `unsure` and the page offers exactly the values the
  CSV accepts, a save keeps other clips' rows and unknown columns and is what
  `gold_sources` reads, every guide link the page uses has a section, and the
  server lists real clips only, seeks the video and refuses other paths.

To label real footage: `snow label` ([docs/labelling.md](docs/labelling.md)).
Machine guesses on that page come from the rate detector, the segmenter and
the classifier; they are never written to the CSV unless the labeller adopts
and corrects them, and the weights are read from `data/models/`, never
installed into `data/weights/`.

When you change measurement code, add the synthetic case that pins the
property, not a regression test against whatever number it printed today.

## If you are changing something

- **A measurement** → `docs/rotation.md` first, then
  `tests/test_rotation.py`. The intuitive version of the spin maths is wrong
  in a specific way the doc spells out.
- **Anything the NIM sees** → `docs/nim.md`. Its input rules are unforgiving
  and its errors do not say what is wrong.
- **Framing or intrinsics** → `docs/tracking.md`. The principal-point hazard
  is subtle and silently produces fake rotation.
- **A user-facing number** → `docs/validation.md`. Say what is measured and
  what is assumed; this project's main risk is a confident wrong number.
