# snow

Extract 3D body pose from snowboarding video using the
[NVIDIA 3D Body Pose NIM](https://build.nvidia.com/nvidia/body-pose), as the
first stage of a trick/rotation analysis pipeline.

No local GPU required — inference runs on NVIDIA's hosted NVCF endpoint.
Detection and tracking run locally on CPU/MPS.

```
YouTube / file ──▶ ingest ──▶ track ──▶ pose ──▶ data/poses/<name>.json
                   (ffmpeg)   (YOLO+     (NIM,      77-joint 3D skeleton
                              ByteTrack)  gRPC)      per frame, per rider
```

## Why there are three stages

The NIM **does not detect or track people**. It takes a video *plus* a tracked
bounding box per person per frame, and it is strict about both:

| Requirement | Enforced how |
|---|---|
| H.264, 4:2:0 8-bit, MP4 | request rejected `INVALID_ARGUMENT` |
| **Constant frame rate** | rejected — even one dropped frame in a CFR stream |
| ≤ 50 MB (hosted endpoint) | rejected |
| boxes reference only decoded frames | fails `INTERNAL` |
| positive box dimensions, or exactly `-1 -1 -1 -1` | rejected |

YouTube footage is VP9/AV1 and usually variable frame rate, so `ingest`
re-encodes and then re-probes the result to confirm each property rather than
trusting the encoder.

## Setup

```bash
scripts/setup.sh          # venv (python 3.12), deps, vendored NVIDIA client
cp .env.example .env      # then paste your key from build.nvidia.com
source .venv/bin/activate
snow doctor               # verifies tools, packages, credentials
```

The NVIDIA client supports Python 3.10–3.12 only, so the venv is pinned to
3.12 regardless of the system Python.

## Use

```bash
# 1. cut a clip and make it NIM-legal
snow ingest "https://www.youtube.com/watch?v=..." run1 --start 00:00:34 --duration 24

# 2. find the rider  (--preview writes a video with the boxes drawn)
snow track run1 --preview --show-candidates

# 3. 3D pose  (--overlay renders the skeleton onto the clip)
snow pose run1 --overlay

# or all three
snow run "<url>" run1 --start 00:00:34 --duration 24 --preview --overlay
```

### Broadcast footage needs its tracks stitched

A televised run is cut between several cameras, and every cut makes ByteTrack
issue the rider a **new track id**, so auto-selection alone typically covers
only ~35% of the clip. `--show-candidates` lists the strongest raw tracks:

```
   8:  221 frames, 32-252, 27024px^2      <- rider, camera 1
1097:  131 frames, 457-587, 62289px^2     <- rider, camera 3
   1:   31 frames, 0-30, 432058px^2       <- rider, drop-in close-up
```

Watch the preview, then stitch them into one body:

```bash
snow track run1 --merge "1,8,945,1097,1738,1842" --preview
```

Detection results are cached in `data/clips/<name>.tracks.json`, so re-running
`track` with different selections is instant. `--retrack` forces a re-detect.

Tuning worth knowing: a rider far up the pipe is small in frame — raise
`--imgsz 1920` and lower `--conf 0.12`. Default `--max-bodies 1` keeps the
rider alone; the NIM allows up to 50.

## Output

`data/poses/<name>.json` is JSON Lines, one object per frame (the `.json`
extension is required by the upstream client even though it writes JSONL):

```json
{"frame_id": 100, "detections": [{"tracking_id": 1, "bbox": [x, y, w, h],
 "keypoints_2d": [[x, y], ...], "keypoints_confidence": [c, ...],
 "keypoints_3d": [[x, y, z], ...], "rest_pose": [[x, y, z], ...],
 "joint_rotations": [[x, y, z, w], ...],
 "root_pose": {"translation": [x, y, z], "rotation": [x, y, z, w]}}]}
```

Every per-joint array is length 77, on the Nova-77 skeleton whose parent
hierarchy is in `src/snow/skeleton.py`. `keypoints_3d` is in **camera
coordinates, metres**; `joint_rotations` are local quaternions `(x, y, z, w)`
relative to the rest pose.

## Reading rotation out of this — the caveat that matters

`root_pose.rotation` is expressed **in camera coordinates**, and a broadcast
camera pans to follow the rider. A 1080 will therefore *not* integrate to 1080°
of root yaw: the camera's own rotation is mixed in. Before trusting any spin
count you need either

- camera motion estimated from the background (feature tracking / homography)
  and subtracted, or
- rotation measured about the rider's **own** body axis from `keypoints_3d`,
  which is invariant to camera yaw.

Metric scale is also uncalibrated — it comes from an assumed pinhole focal
length (`--focal-length`, 0 derives it from frame size). Angles and ratios are
trustworthy; absolute metres are not.

## Self-hosting instead

The hosted endpoint caps input at 50 MB. To run the container yourself you need
Linux/amd64 and an NVIDIA GPU of compute capability 8.0+ with 24 GB
(AWS `g5.xlarge`/A10G or `g6.xlarge`/L4; **not** `g4dn`/T4, which is
unsupported). Then point the pipeline at it:

```bash
SNOW_NIM_PREVIEW=0 SNOW_NIM_TARGET=127.0.0.1:8001 snow pose run1
```

## Layout

```
src/snow/ingest.py    download + re-encode + verify NIM input rules
src/snow/track.py     YOLO/ByteTrack -> the NIM's bbox annotation
src/snow/pose.py      drives the NIM, summarises the result
src/snow/skeleton.py  Nova-77 joint hierarchy
vendor/nim-clients/   upstream NVIDIA gRPC client (cloned by setup.sh)
data/                 clips, tracks, poses — all gitignored
```

## Next

Stage 2 is feature extraction for the DS/ML work: camera-compensated rotation,
per-trick segmentation, grab detection from `joint_rotations`. The Nova-77
hierarchy in `skeleton.py` is also what a USD `SkelAnimation` export needs if
the poses get driven into Omniverse.
