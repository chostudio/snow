# The NVIDIA 3D Body Pose NIM

Covers `src/snow/pose.py` and `src/snow/_nim_launcher.py` — the external
dependency this whole pipeline is built around.

## What it is and is not

A gRPC service that takes a video plus **boxes you supply** and returns a
77-joint 3D skeleton per person per frame.

**It does not detect or track people.** That is the single most important fact
about it. You hand it a bbox annotation file naming every person in every
frame; it fills in skeletons. Everything in `src/snow/track.py` exists to
produce that file.

## Running it without a GPU

Inference runs on NVIDIA's hosted NVCF endpoint. The defaults in
`src/snow/config.py`:

```
target       grpc.nvcf.nvidia.com:443
function-id  254b3621-bb66-4b7b-9c89-621c352b4d91
mode         preview (NVCF)
```

The function id came from `build.nvidia.com`'s page source (`nvcfFunctionId`,
namespace `qc69jvmznzxy`) — the page is JS-rendered, so fetching it as a
document returns nothing useful; curl the raw HTML and read the attribute.

Use of the hosted endpoint is governed by the NVIDIA API Trial Terms of
Service.

Two traps:

- **Preview mode opens a secure channel with system roots automatically.** Do
  not also pass `--ssl-mode TLS`; that makes the client demand a root
  certificate file it does not need.
- **The hosted endpoint takes mp4/H.264 only, 50 MB maximum.** The cap is on
  the uploaded file, so `ingest` targets 46 MB to leave margin rather than
  failing after a long upload.

Override any of it with `SNOW_NIM_TARGET`, `SNOW_NIM_FUNCTION_ID`,
`SNOW_NIM_PREVIEW`.

## The API key

Lives in `.env` at the repo root. Never commit it, never print it in full,
never put it on a command line.

The upstream client only accepts its key as `--api-key`, which places the
secret in the process list where every user on the machine can read it.
`src/snow/_nim_launcher.py` avoids that: it reads `SNOW_NIM_API_KEY` from the
environment and appends it to `sys.argv` **after** the process has started,
then runs the vendored script with
`runpy.run_path(..., run_name="__main__")`. The key never appears in the real
argv.

`runpy` rather than `python script.py` for a second reason: the latter would
put the script's own directory first on `sys.path`, and the client does a bare
`from config import ...` that must resolve to *its* config, not ours. The
launcher inserts that directory explicitly.

## Input requirements

The NIM is strict and its errors do not say what is wrong.

| Requirement | Failure if violated |
|---|---|
| H.264, 4:2:0 8-bit, MP4 | `INVALID_ARGUMENT` |
| **Constant frame rate** — even one dropped frame | rejected |
| ≤ 50 MB (hosted) | rejected |
| Boxes reference only decoded frames | `INTERNAL` |
| Positive box dimensions, or exactly `-1 -1 -1 -1` | rejected |

`ingest` re-probes its own output and checks each property rather than
trusting the encoder. CFR in particular is verified from **real per-frame
timestamps**, because a stream *labelled* CFR that dropped a frame is exactly
the case the NIM rejects. See [footage.md](footage.md).

The bbox annotation format is in [tracking.md](tracking.md).

## Output

`data/poses/<name>.json` is **JSON Lines** — one object per frame, not a JSON
document. The `.json` extension is required by the upstream client even though
it writes JSONL; `src/snow/config.py:pose_path` carries that comment so nobody
"fixes" it.

```json
{"frame_id": 100, "detections": [{"tracking_id": 1, "bbox": [x, y, w, h],
 "keypoints_2d": [[x, y], ...], "keypoints_confidence": [c, ...],
 "keypoints_3d": [[x, y, z], ...], "rest_pose": [[x, y, z], ...],
 "joint_rotations": [[x, y, z, w], ...],
 "root_pose": {"translation": [x, y, z], "rotation": [x, y, z, w]}}]}
```

- `keypoints_2d` — source pixels
- `keypoints_3d` — **camera coordinates, metres**, X right, **Y down**, Z forward
- `joint_rotations` — quaternions `(x, y, z, w)`, local, against `rest_pose`
- Every per-joint array is length **77** (Nova-77)

Two relationships that matter downstream, both confirmed from real output:

- `joint_rotations[0]` **is exactly** `root_pose.rotation`
- `rest_pose[0]` is `(0, 0, 0)`

So the root is already accounted for and must not be transformed a second
time. See [usd.md](usd.md).

Joint indices worth knowing: `0` root/pelvis, `6` neck, `67`/`72` the hips
(0.221 m apart, near-perpendicular to the spine), `14`/`42` wrists, `71`/`76`
feet. The parent hierarchy is `src/snow/skeleton.py`, 19 lines.

## Latency

The server buffers about **120 frames** before emitting anything. This is
latency, not loss — 720 frames in gives poses for 720 frames out. It does mean
a very short clip appears to hang, which is why `scout` flags shots under 120
frames as `SHORT`: widen the window rather than sending one alone.

Measured throughput: a 720-frame 1080p clip took **19 s**; a 312-frame clip
took 9 s.

## Focal length

With `focal_length = 0` the NIM assumes **`f = hypot(width, height)`** — the
image diagonal — and puts the principal point at the image centre.

This was measured, not documented: fitted **2206.1 px** against **2202.9**
predicted on a 1920×1080 clip, over 423 frames, IQR 2182–2229.

Everything about metric scale follows from this assumption, and it is why
absolute metres are untrustworthy. `crop` passes a correctly scaled focal
length explicitly — see [tracking.md](tracking.md).

## Self-hosting

The 50 MB cap is the main reason to. Requirements:

- Linux/amd64, NVIDIA GPU of **compute capability 8.0+** with **24 GB** VRAM
- driver 570.26+
- `nvcr.io/nim/nvidia/body-pose:latest`, gRPC on :8001

AWS `g5.xlarge` (A10G) or `g6.xlarge` (L4). **Not `g4dn`/T4** — unsupported
compute capability.

```bash
SNOW_NIM_PREVIEW=0 SNOW_NIM_TARGET=127.0.0.1:8001 snow pose run1
```
