"""Fetch a clip and re-encode it into what the 3D Body Pose NIM accepts.

The NIM is strict and fails the whole request on input it dislikes:

* MP4/WebM/MKV, H.264 (the hosted endpoint is H.264-only), 4:2:0 8-bit SDR.
* Constant frame rate, *enforced* — one irregular frame interval and the
  request comes back ``INVALID_ARGUMENT``. YouTube footage is routinely VFR.
* Hosted endpoint: 50 MB per file.

So this module does not trust the encoder; it re-probes the finished file and
checks each of those properties directly.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .config import HOSTED_MAX_BYTES, HOSTED_TARGET_BYTES


class IngestError(RuntimeError):
    pass


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def _require(tool: str) -> str:
    """Locate a tool, preferring one installed beside the running interpreter.

    yt-dlp in particular goes stale fast — YouTube breaks old versions within
    months — so the pinned copy in this project's venv wins over whatever
    happens to be on PATH.
    """
    local = Path(sys.executable).parent / tool
    if local.exists():
        return str(local)
    path = shutil.which(tool)
    if not path:
        raise IngestError(f"{tool} not found on PATH")
    return path


@dataclass
class VideoInfo:
    path: Path
    codec: str
    pix_fmt: str
    width: int
    height: int
    fps: float
    frames: int
    duration: float
    size_bytes: int

    @property
    def size_mb(self) -> float:
        return self.size_bytes / 1024 / 1024


def probe(path: Path) -> VideoInfo:
    """Probe a video, decoding it fully to get an exact frame count.

    ``nb_frames`` from the container header is often wrong or absent;
    ``-count_frames`` decodes and is the number the NIM will also see.
    """
    ffprobe = _require("ffprobe")
    out = _run([
        ffprobe, "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries",
        "stream=codec_name,pix_fmt,width,height,avg_frame_rate,nb_read_frames,duration",
        "-show_entries", "format=duration",
        "-of", "json", str(path),
    ])
    if out.returncode != 0:
        raise IngestError(f"ffprobe failed on {path}: {out.stderr.strip()}")
    data = json.loads(out.stdout)
    if not data.get("streams"):
        raise IngestError(f"{path}: no video stream")
    s = data["streams"][0]

    num, _, den = s.get("avg_frame_rate", "0/1").partition("/")
    fps = float(num) / float(den) if den and float(den) else 0.0
    duration = float(
        s.get("duration") or data.get("format", {}).get("duration") or 0.0
    )
    return VideoInfo(
        path=path,
        codec=s.get("codec_name", "?"),
        pix_fmt=s.get("pix_fmt", "?"),
        width=int(s.get("width", 0)),
        height=int(s.get("height", 0)),
        fps=fps,
        frames=int(s.get("nb_read_frames") or 0),
        duration=duration,
        size_bytes=path.stat().st_size,
    )


def frame_intervals_are_constant(path: Path, tolerance: float = 0.002) -> tuple[bool, str]:
    """Check the property the NIM enforces: every frame interval the same.

    Reads real per-frame presentation timestamps rather than trusting the
    stream's advertised frame rate, because a CFR-labelled stream with a
    dropped frame is exactly the case the NIM rejects.
    """
    out = _run([
        _require("ffprobe"), "-v", "error", "-select_streams", "v:0",
        "-show_entries", "frame=pts_time",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ])
    if out.returncode != 0:
        return False, f"ffprobe failed: {out.stderr.strip()}"
    times = [
        float(tok)
        for tok in out.stdout.replace(",", " ").split()
        if tok not in ("", "N/A")
    ]
    if len(times) < 3:
        return False, f"only {len(times)} frames with timestamps"
    times.sort()
    deltas = [b - a for a, b in zip(times, times[1:])]
    lo, hi = min(deltas), max(deltas)
    if hi - lo > tolerance:
        worst = max(range(len(deltas)), key=lambda i: abs(deltas[i] - deltas[0]))
        return False, (
            f"frame intervals vary from {lo:.5f}s to {hi:.5f}s "
            f"(first irregular gap near frame {worst})"
        )
    return True, f"constant {1 / deltas[0]:.3f} fps over {len(times)} frames"


def download(url: str, dest_stem: Path) -> Path:
    """Download a video with yt-dlp. Returns the downloaded file."""
    ytdlp = _require("yt-dlp")
    dest_stem.parent.mkdir(parents=True, exist_ok=True)
    for stale in dest_stem.parent.glob(dest_stem.name + ".src.*"):
        stale.unlink()
    out = _run([
        ytdlp,
        # Prefer an H.264 source so the transcode is a remux-grade operation
        # rather than a VP9/AV1 -> H.264 generation loss.
        "-f", "bv*[vcodec^=avc1]+ba/bv*+ba/b",
        "--merge-output-format", "mkv",
        "-o", str(dest_stem) + ".src.%(ext)s",
        url,
    ])
    if out.returncode != 0:
        raise IngestError(f"yt-dlp failed:\n{out.stderr.strip()[-2000:]}")
    files = sorted(dest_stem.parent.glob(dest_stem.name + ".src.*"))
    if not files:
        raise IngestError("yt-dlp reported success but produced no file")
    return files[0]


def transcode(
    src: Path,
    dest: Path,
    fps: float = 30.0,
    start: str | None = None,
    duration: str | None = None,
    height: int | None = 1080,
    crf: int = 23,
    max_bytes: int = HOSTED_MAX_BYTES,
    target_bytes: int = HOSTED_TARGET_BYTES,
) -> VideoInfo:
    """Re-encode to H.264 / yuv420p / constant frame rate, under the size cap.

    Tries CRF first for quality; if that overshoots the size cap, re-encodes at
    a bitrate computed from the clip duration, which lands predictably.
    """
    ffmpeg = _require("ffmpeg")
    dest.parent.mkdir(parents=True, exist_ok=True)

    def base_cmd() -> list[str]:
        cmd = [ffmpeg, "-y", "-v", "error"]
        # Before -i: fast seek. Re-encoding anyway, so accuracy is preserved.
        if start:
            cmd += ["-ss", start]
        cmd += ["-i", str(src)]
        if duration:
            cmd += ["-t", duration]
        filters = [f"fps={fps}"]
        if height:
            # Even dimensions are required by yuv420p.
            filters.append(f"scale=-2:'min({height},ih)'")
        cmd += [
            "-vf", ",".join(filters),
            "-fps_mode", "cfr",
            "-c:v", "libx264",
            "-profile:v", "high",
            "-pix_fmt", "yuv420p",
            "-g", str(int(round(fps * 2))),
            "-an",
            "-movflags", "+faststart",
        ]
        return cmd

    out = _run(base_cmd() + ["-crf", str(crf), str(dest)])
    if out.returncode != 0:
        raise IngestError(f"ffmpeg failed:\n{out.stderr.strip()[-2000:]}")

    info = probe(dest)
    if info.size_bytes > max_bytes:
        if info.duration <= 0:
            raise IngestError("cannot size-target a clip of unknown duration")
        # Leave ~3% headroom for container overhead.
        bitrate = int(target_bytes * 8 / info.duration * 0.97)
        if bitrate < 200_000:
            raise IngestError(
                f"clip is {info.duration:.0f}s; fitting it under "
                f"{max_bytes / 1024 / 1024:.0f} MB needs {bitrate / 1000:.0f} kbps, "
                "which is too low to be useful. Use --duration to cut a shorter clip."
            )
        out = _run(
            base_cmd()
            + ["-b:v", str(bitrate), "-maxrate", str(int(bitrate * 1.5)),
               "-bufsize", str(bitrate * 2), str(dest)]
        )
        if out.returncode != 0:
            raise IngestError(f"ffmpeg (bitrate pass) failed:\n{out.stderr.strip()[-2000:]}")
        info = probe(dest)

    verify(info, max_bytes=max_bytes)
    return info


def verify(info: VideoInfo, max_bytes: int = HOSTED_MAX_BYTES) -> None:
    """Fail loudly here rather than letting the NIM reject the request."""
    problems: list[str] = []
    if info.codec != "h264":
        problems.append(f"codec is {info.codec}, hosted endpoint accepts H.264 only")
    if info.pix_fmt != "yuv420p":
        problems.append(f"pixel format is {info.pix_fmt}, must be yuv420p (4:2:0 8-bit)")
    if info.frames < 1:
        problems.append("no decodable frames")
    if info.size_bytes > max_bytes:
        problems.append(
            f"{info.size_mb:.1f} MB exceeds the {max_bytes / 1024 / 1024:.0f} MB limit"
        )
    ok, detail = frame_intervals_are_constant(info.path)
    if not ok:
        problems.append(f"not constant frame rate: {detail}")
    if problems:
        raise IngestError(
            f"{info.path.name} is not acceptable to the NIM:\n  - "
            + "\n  - ".join(problems)
        )
