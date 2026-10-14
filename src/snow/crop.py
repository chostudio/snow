"""Crop and zoom onto the rider before pose estimation.

In a wide halfpipe shot the rider is 60-150 px tall in a 1920x1080 frame.
Two things cost accuracy there, and cropping fixes both:

* **Bitrate.** An encoder spends its bits on the pipe wall and the crowd. The
  rider — the only part that matters — gets a few hundred bytes a frame and
  arrives smeared. Re-encoding a tight window spends the whole budget on them.
* **Source resolution.** The hosted endpoint caps input at 50 MB, which forces
  a wide 4K clip down to 1080p and throws away real detail. Cropping first
  keeps the rider's native pixels and still fits.

The crop window is a **fixed size**, and by default it does not move either.
Both constraints exist because the NIM assumes a pinhole camera whose
principal point is the centre of the image it is given:

* A window that **zoomed** would change the effective focal length every
  frame, and the returned 3D is expressed in camera coordinates that assume
  a single one.
* A window that **pans** moves the true principal point relative to the crop
  centre every frame. The NIM cannot know this, so it back-projects along
  slightly wrong rays, and because the error *changes over time* it shows up
  as rotation the rider never performed — fatal for the one measurement this
  pipeline exists to produce. A static window makes the offset constant, so
  it biases absolute orientation slightly and leaves frame-to-frame rotation
  untouched. ``--pan`` re-enables panning for shots where the rider simply
  will not fit, and the offset is recorded per frame so ``features.py`` can
  correct for it.

Measured empirically from NIM output: when ``focal_length`` is 0 the server
assumes ``f = hypot(width, height)``, the image diagonal (2206 px fitted vs
2203 predicted on a 1920x1080 clip, over 423 frames). Cropping by a factor
``z`` and rescaling therefore needs ``f = hypot(W, H) * out_w / crop_w``
passed explicitly, or the depths come back wrong by that factor.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .ingest import _require, probe
from .track import ABSENT


class CropError(RuntimeError):
    pass


@dataclass
class CropPlan:
    crop_w: int
    crop_h: int
    out_w: int
    out_h: int
    centers: np.ndarray  # (frames, 2), float, clamped window centre per frame
    source_focal: float
    frames: int
    source_w: int = 0
    source_h: int = 0
    panned: bool = False

    @property
    def scale(self) -> float:
        return self.out_w / self.crop_w

    @property
    def focal_length(self) -> float:
        return self.source_focal * self.scale

    def origin(self, frame_id: int) -> tuple[float, float]:
        cx, cy = self.centers[frame_id]
        return cx - self.crop_w / 2.0, cy - self.crop_h / 2.0

    def principal_offset(self, frame_id: int) -> tuple[float, float]:
        """Where the true principal point lands in the output image.

        The NIM assumes (out_w/2, out_h/2). This returns the offset from that
        assumption, in output pixels, which is what a correction needs.
        """
        ox, oy = self.origin(frame_id)
        true_x = (self.source_w / 2.0 - ox) * self.scale
        true_y = (self.source_h / 2.0 - oy) * self.scale
        return true_x - self.out_w / 2.0, true_y - self.out_h / 2.0


def read_bbox_file(path: Path) -> tuple[int, dict[int, dict[int, tuple]]]:
    """Parse our own annotation back into {tracking_id: {frame: box}}."""
    lines = path.read_text().strip().split("\n")
    bodies = int(lines[0].split()[0])
    out: dict[int, dict[int, tuple]] = {i: {} for i in range(1, bodies + 1)}
    for line in lines[1:]:
        fields = line.split()
        if len(fields) != 6:
            continue
        frame_id, tracking_id = int(fields[0]), int(fields[1])
        x, y, w, h = (float(v) for v in fields[2:])
        if (x, y, w, h) == (-1.0, -1.0, -1.0, -1.0):
            continue
        out.setdefault(tracking_id, {})[frame_id] = (x, y, w, h)
    return bodies, out


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average, edge-padded, applied per column."""
    if window <= 1:
        return values
    pad = window // 2
    kernel = np.ones(window) / window
    out = np.empty_like(values)
    for col in range(values.shape[1]):
        padded = np.pad(values[:, col], pad, mode="edge")
        out[:, col] = np.convolve(padded, kernel, mode="same")[pad : pad + len(values)]
    return out


def plan(
    video: Path,
    bbox: Path,
    fill: float = 0.5,
    out_height: int = 0,
    smooth_window: int = 15,
    min_zoom: float = 1.0,
    pan: bool = False,
) -> CropPlan:
    """Plan a fixed-size crop window.

    ``out_height`` of 0 keeps the crop at its native pixel scale, which is
    usually what you want: the rider's real pixels are preserved and the whole
    bitrate goes to them. A larger value only interpolates — it adds no
    detail, though it does give the model's internal ROI resize more to work
    with.
    """
    info = probe(video)
    bodies, tracks = read_bbox_file(bbox)
    frames = info.frames

    # Per-frame union of every tracked body.
    centers = np.full((frames, 2), np.nan)
    spans = []
    for frame_id in range(frames):
        present = [t[frame_id] for t in tracks.values() if frame_id in t]
        if not present:
            continue
        x0 = min(b[0] for b in present)
        y0 = min(b[1] for b in present)
        x1 = max(b[0] + b[2] for b in present)
        y1 = max(b[1] + b[3] for b in present)
        centers[frame_id] = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
        spans.append(max(x1 - x0, y1 - y0))
    if not spans:
        raise CropError(f"{bbox.name}: no boxes to crop around")

    # One window size for the whole clip, from a high percentile so the
    # biggest moments still fit rather than clipping the rider's limbs. A
    # static window must additionally span wherever the rider travelled.
    subject = float(np.percentile(spans, 95))
    if not pan:
        valid_centers = centers[~np.isnan(centers[:, 0])]
        travel_x = float(np.ptp(valid_centers[:, 0])) if len(valid_centers) else 0.0
        travel_y = float(np.ptp(valid_centers[:, 1])) if len(valid_centers) else 0.0
        subject = max(subject, travel_x + subject, travel_y + subject)
    aspect = info.width / info.height
    crop_h = subject / max(fill, 0.05)
    crop_w = crop_h * aspect
    # Never exceed the frame, and keep the source aspect ratio.
    shrink = min(info.width / crop_w, info.height / crop_h, 1.0)
    crop_w, crop_h = crop_w * shrink, crop_h * shrink
    crop_w, crop_h = int(round(crop_w / 2) * 2), int(round(crop_h / 2) * 2)

    if crop_w <= 0 or crop_h <= 0:
        raise CropError("computed a degenerate crop window")

    out_h = int(round((out_height or crop_h) / 2) * 2)
    out_w = int(round(out_h * aspect / 2) * 2)

    # The number that matters is the final resampling scale, not how much of
    # the frame was kept: a tight crop written out small still throws the
    # rider's pixels away.
    scale = out_w / crop_w
    if scale < min_zoom:
        raise CropError(
            f"this crop would scale the rider by {scale:.2f}x, below the "
            f"{min_zoom:.2f}x minimum — it would lose detail rather than keep "
            f"it. Raise --out-height (crop window is {crop_w}x{crop_h}) or "
            "skip this stage."
        )
    if crop_w > 0.92 * info.width:
        raise CropError(
            f"the rider already spans {subject / info.height:.0%} of the frame "
            f"height; the crop would keep {crop_w / info.width:.0%} of the "
            "width, which is not worth re-encoding. Skip this stage."
        )

    valid = ~np.isnan(centers[:, 0])
    if not valid.any():
        raise CropError("no valid centres")
    index = np.arange(frames)
    if pan:
        # Hold the last known centre across gaps, then smooth the pan.
        for col in range(2):
            centers[:, col] = np.interp(index, index[valid], centers[valid, col])
        centers = _smooth(centers, smooth_window)
    else:
        middle = centers[valid].mean(axis=0)
        centers = np.tile(middle, (frames, 1))

    # Keep the window inside the frame.
    centers[:, 0] = np.clip(centers[:, 0], crop_w / 2.0, info.width - crop_w / 2.0)
    centers[:, 1] = np.clip(centers[:, 1], crop_h / 2.0, info.height - crop_h / 2.0)

    return CropPlan(
        crop_w=crop_w, crop_h=crop_h, out_w=out_w, out_h=out_h,
        centers=centers, source_focal=float(np.hypot(info.width, info.height)),
        frames=frames, source_w=info.width, source_h=info.height, panned=pan,
    )


def render(video: Path, dest: Path, plan: CropPlan, fps: float) -> None:
    """Write the zoomed clip straight into x264, so it stays NIM-legal."""
    ffmpeg = _require("ffmpeg")
    dest.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        [
            ffmpeg, "-y", "-v", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{plan.out_w}x{plan.out_h}", "-r", f"{fps}",
            "-i", "-",
            "-c:v", "libx264", "-profile:v", "high", "-crf", "20",
            "-pix_fmt", "yuv420p", "-fps_mode", "cfr",
            "-movflags", "+faststart", "-an", str(dest),
        ],
        stdin=subprocess.PIPE,
    )
    cap = cv2.VideoCapture(str(video))
    frame_id = 0
    try:
        while frame_id < plan.frames:
            ok, frame = cap.read()
            if not ok:
                break
            x0, y0 = plan.origin(frame_id)
            x0i, y0i = int(round(x0)), int(round(y0))
            window = frame[y0i : y0i + plan.crop_h, x0i : x0i + plan.crop_w]
            if window.shape[0] != plan.crop_h or window.shape[1] != plan.crop_w:
                window = cv2.copyMakeBorder(
                    window, 0, plan.crop_h - window.shape[0],
                    0, plan.crop_w - window.shape[1], cv2.BORDER_REPLICATE,
                )
            resized = cv2.resize(
                window, (plan.out_w, plan.out_h), interpolation=cv2.INTER_CUBIC
            )
            proc.stdin.write(resized.tobytes())
            frame_id += 1
    finally:
        cap.release()
        proc.stdin.close()
        proc.wait()
    if proc.returncode != 0:
        raise CropError(f"ffmpeg failed while writing {dest}")


def remap_bbox(bbox: Path, dest: Path, plan: CropPlan) -> int:
    """Rewrite the annotation in cropped-image coordinates."""
    bodies, tracks = read_bbox_file(bbox)
    ids = sorted(tracks)
    lines = [str(len(ids))]
    kept = 0
    for frame_id in range(plan.frames):
        ox, oy = plan.origin(frame_id)
        for tracking_id in ids:
            box = tracks[tracking_id].get(frame_id)
            if box is None:
                lines.append(f"{frame_id} {tracking_id} {ABSENT}")
                continue
            x, y, w, h = box
            nx, ny = (x - ox) * plan.scale, (y - oy) * plan.scale
            nw, nh = w * plan.scale, h * plan.scale
            # Clip into the new frame; drop anything that fell outside it.
            x0, y0 = max(0.0, nx), max(0.0, ny)
            x1 = min(float(plan.out_w), nx + nw)
            y1 = min(float(plan.out_h), ny + nh)
            if x1 - x0 < 2.0 or y1 - y0 < 2.0:
                lines.append(f"{frame_id} {tracking_id} {ABSENT}")
                continue
            lines.append(
                f"{frame_id} {tracking_id} {x0:.6f} {y0:.6f} {x1 - x0:.6f} {y1 - y0:.6f}"
            )
            kept += 1
    dest.write_text("\n".join(lines) + "\n")
    return kept


def save_plan(plan: CropPlan, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        json.dumps(
            {
                "crop_w": plan.crop_w, "crop_h": plan.crop_h,
                "out_w": plan.out_w, "out_h": plan.out_h,
                "scale": plan.scale,
                "source_focal": plan.source_focal,
                "focal_length": plan.focal_length,
                "frames": plan.frames,
                "panned": plan.panned,
                "source_w": plan.source_w,
                "source_h": plan.source_h,
                "origins": [list(map(float, plan.origin(f))) for f in range(plan.frames)],
                "principal_offsets": [
                    list(map(float, plan.principal_offset(f)))
                    for f in range(plan.frames)
                ],
            }
        )
    )


def load_plan(src: Path) -> dict:
    return json.loads(src.read_text())
