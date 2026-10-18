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
rotation maths, export, UI — is ours.

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
                   ├─ usd ─────── UsdSkelAnimation for Omniverse
                   └─ web/tools/export_session.py → the dashboard's bundle
```

Stages talk to each other **only through files in `data/`**, keyed by a clip
`name`. Nothing holds state in memory across stages, and any stage can be re-run
alone. That is the main reason the pipeline is debuggable, so preserve it.

## Where everything lives

| Module | Does | Deep dive |
|---|---|---|
| `src/snow/scout.py` | split a broadcast into shots, rank them for pose work | [docs/footage.md](docs/footage.md) |
| `src/snow/ingest.py` | download, cut, re-encode, verify the NIM's input rules | [docs/footage.md](docs/footage.md) |
| `src/snow/track.py` | detect/track the rider, write the bbox annotation | [docs/tracking.md](docs/tracking.md) |
| `src/snow/crop.py` | reframe onto the rider, with correct intrinsics | [docs/tracking.md](docs/tracking.md) |
| `src/snow/pose.py` | drive the NIM, summarise its output | [docs/nim.md](docs/nim.md) |
| `src/snow/_nim_launcher.py` | keep the API key out of the process list | [docs/nim.md](docs/nim.md) |
| `src/snow/camera.py` | camera rotation from a background homography | [docs/rotation.md](docs/rotation.md) |
| `src/snow/features.py` | subtract the camera, integrate what is left | [docs/rotation.md](docs/rotation.md) |
| `src/snow/usd.py` | UsdSkelAnimation export | [docs/usd.md](docs/usd.md) |
| `src/snow/skeleton.py` | Nova-77 parent hierarchy. 19 lines, read it directly | — |
| `src/snow/meta.py` | per-clip metadata (`fps`, `slowmo_factor`, crop plan) | — |
| `src/snow/config.py` | paths, endpoint settings, `.env` loading | — |
| `src/snow/cli.py` | argparse surface for every stage | — |
| `web/` | review dashboard + session exporter | [web/README.md](web/README.md) |

Cross-cutting: **[docs/validation.md](docs/validation.md)** — what is actually
proven, what is not, and which claims you may not make. Read it before you
write anything user-facing about accuracy.

Planned, not built: **[docs/training.md](docs/training.md)** — the design for
the learned half (airborne segmenter, trick classifier, judging score). Nothing in it exists
in the repo yet; do not treat it as describing current behaviour. Its
**Synthetic ground truth in Omniverse** section is the agreed next direction:
render parametric tricks with Replicator, run them through the unchanged
pipeline, and use the exact labels to validate spin counts and to train on
NIM output (never on clean rig skeletons).

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
| `data/poses/run1.camera.npz` | features | cached camera track |
| `data/poses/run1.features.{csv,parquet,json}` | features | the measurements |
| `data/poses/run1.usda` | usd | the skeletal animation |
| `web/public/sessions/run1/` | `export_session.py` | `session.json` + `clip.mp4` |

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
7. **Metric scale is uncalibrated.** The focal length is assumed, so angles
   and ratios hold and absolute metres do not. Never print a length as though
   it were real. → `docs/validation.md`
8. **Slow motion corrupts rates, not totals.** Conformed slow motion claims 30
   fps and the timestamps lie; the factor cannot be recovered from the file.
   Anything per-second is wrong by that factor unless `slowmo_factor` is set.
   → `docs/footage.md`
9. **Absolute spin counts are unvalidated on real footage.** Synthetic motion
   is exact; one real clip shows an open discrepancy. → `docs/validation.md`

## Secrets

The NVIDIA API key lives in `.env` at the repo root, `chmod 600`, gitignored.
**Never commit it, never print it in full, never pass it as a CLI argument.**

The upstream client only accepts `--api-key` on the command line, which would
expose it in the process list to every user on the machine.
`src/snow/_nim_launcher.py` works around this by injecting it into `sys.argv`
after the process has started. Keep that property.

Before any commit, grep the staged diff:

```bash
git diff --cached | grep -niE "nvapi-|api[_-]key|secret|bearer"
```

## Working here

```bash
source .venv/bin/activate
snow doctor                 # tools, packages, vendored client, endpoint, key
python -m pytest -q         # 27 tests, ~3s, no network and no GPU
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
specification for three things:

- `tests/test_bbox_format.py` — parses our annotation with **NVIDIA's own
  reader**, not a reimplementation of it.
- `tests/test_rotation.py` — synthetic motion with a known answer: a rider
  turning a known amount about a known axis, and camera compensation in both
  directions.
- `tests/test_crop_and_usd.py` — crop coordinate round-trips, and the USD is
  validated with **USD's own skinning resolver** when `usd-core` is installed.

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
