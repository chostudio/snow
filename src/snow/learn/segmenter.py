"""Model A: per-frame airborne segmenter (docs/training.md).

A small non-causal temporal CNN (or a BiGRU) over fixed windows of
``WINDOW`` frames, one logit per frame. At inference every camera segment is
covered by overlapping windows whose logits are averaged; no window ever
crosses a cut (invariant 6), and a segment shorter than a window is padded
with frames marked absent.

Imports torch, so nothing on the default ``snow features`` path imports this
module unless weights exist (see ``features.build``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import inputs as I
from .events import MERGE_GAP_FRAMES, MIN_AIR_FRAMES, clean_mask

WINDOW = 64      # about 2 s of playback at 30 fps
STRIDE = 8       # inference overlap


def dilations_for(window: int) -> tuple[int, ...]:
    """Doubling dilations until the receptive field (1 + 4 * sum) spans the window."""
    out = [1]
    while 1 + 4 * sum(out) < window:
        out.append(out[-1] * 2)
    return tuple(out)


class TCN(nn.Module):
    """Residual dilated 1D convolutions, two per block, kernel 3.

    Receptive field ``1 + 4 * sum(dilations)``: 61 frames for (1, 2, 4, 8).
    """

    def __init__(self, channels: int, width: int = 64, dilations=(1, 2, 4, 8),
                 dropout: float = 0.1):
        super().__init__()
        self.inp = nn.Conv1d(channels, width, 1)
        self.blocks = nn.ModuleList()
        for d in dilations:
            self.blocks.append(nn.Sequential(
                nn.Conv1d(width, width, 3, padding=d, dilation=d),
                nn.GroupNorm(8, width), nn.GELU(), nn.Dropout(dropout),
                nn.Conv1d(width, width, 3, padding=d, dilation=d),
                nn.GroupNorm(8, width), nn.GELU(),
            ))
        self.out = nn.Conv1d(width, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, C) -> (B, T)
        h = self.inp(x.transpose(1, 2))
        for block in self.blocks:
            h = h + block(h)
        return self.out(h)[:, 0]


class BiGRU(nn.Module):
    def __init__(self, channels: int, width: int = 64, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.inp = nn.Linear(channels, width)
        self.gru = nn.GRU(width, width, num_layers=layers, batch_first=True,
                          bidirectional=True, dropout=dropout)
        self.out = nn.Linear(2 * width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(torch.relu(self.inp(x)))
        return self.out(h)[..., 0]


ARCHS = {"tcn": TCN, "bigru": BiGRU}


def build_model(arch: str, channels: int = I.NUM_CHANNELS, window: int = WINDOW,
                **kwargs) -> nn.Module:
    if arch == "tcn":
        kwargs.setdefault("dilations", dilations_for(window))
    return ARCHS[arch](channels, **kwargs)


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@dataclass
class Segmenter:
    """A trained model plus the normalisation it was trained with."""

    model: nn.Module
    arch: str
    mean: np.ndarray
    std: np.ndarray
    threshold: float = 0.5
    window: int = WINDOW
    info: dict = field(default_factory=dict)

    def save(self, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "arch": self.arch, "state": self.model.state_dict(),
            "mean": self.mean, "std": self.std, "threshold": self.threshold,
            "window": self.window, "channels": list(I.CHANNELS), "info": self.info,
        }, dest)

    @classmethod
    def load(cls, src: Path) -> "Segmenter":
        blob = torch.load(src, map_location="cpu", weights_only=False)
        if blob["channels"] != list(I.CHANNELS):
            raise ValueError(f"{src}: trained on different input channels; retrain")
        model = build_model(blob["arch"], window=int(blob["window"]))
        model.load_state_dict(blob["state"])
        model.eval()
        return cls(model=model, arch=blob["arch"], mean=np.asarray(blob["mean"]),
                   std=np.asarray(blob["std"]), threshold=float(blob["threshold"]),
                   window=int(blob["window"]), info=blob.get("info", {}))

    def normalise(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)

    @torch.no_grad()
    def logits(self, x: np.ndarray, segment: np.ndarray) -> np.ndarray:
        """Per-frame logits for one clip's raw channels, segment by segment."""
        self.model.eval()
        out = np.zeros(len(x), dtype=np.float32)
        for seg in np.unique(segment):
            rows = np.flatnonzero(segment == seg)
            out[rows] = segment_logits(self.model, self.normalise(x[rows]), self.window,
                                       pad_row=self.pad_row)
        return out

    @property
    def pad_row(self) -> np.ndarray:
        """An absent frame (all raw channels zero), normalised."""
        return self.normalise(np.zeros(I.NUM_CHANNELS, dtype=np.float32))

    def probabilities(self, x: np.ndarray, segment: np.ndarray) -> np.ndarray:
        """P(airborne) per frame; NaN where no body was seen (``inputs.observed``)."""
        prob = 1.0 / (1.0 + np.exp(-self.logits(x, segment)))
        prob[~I.observed(x[:, I.PRESENT], segment)] = np.nan
        return prob

    def mask(self, prob: np.ndarray, segment: np.ndarray) -> np.ndarray:
        return clean_mask(np.nan_to_num(prob) >= self.threshold, segment,
                          min_frames=MIN_AIR_FRAMES, merge_gap=MERGE_GAP_FRAMES)


def pad_window(x: np.ndarray, window: int,
               pad_row: np.ndarray | None = None) -> tuple[np.ndarray, int]:
    """Pad a short segment to one window with absent frames (``pad_row``)."""
    n = len(x)
    if n >= window:
        return x, n
    padded = np.zeros((window, x.shape[1]), dtype=np.float32)
    if pad_row is not None:
        padded[:] = pad_row
    padded[:n] = x
    return padded, n


def window_starts(n: int, window: int, stride: int = STRIDE) -> list[int]:
    if n <= window:
        return [0]
    starts = list(range(0, n - window + 1, stride))
    if starts[-1] != n - window:
        starts.append(n - window)
    return starts


@torch.no_grad()
def segment_logits(model: nn.Module, x: np.ndarray, window: int = WINDOW,
                   stride: int = STRIDE, pad_row: np.ndarray | None = None) -> np.ndarray:
    """Average of overlapping window logits over one segment's normalised rows."""
    padded, n = pad_window(x, window, pad_row)
    starts = window_starts(len(padded), window, stride)
    batch = torch.from_numpy(np.stack([padded[s:s + window] for s in starts]))
    pred = model(batch).cpu().numpy()
    total = np.zeros(len(padded))
    count = np.zeros(len(padded))
    for s, p in zip(starts, pred):
        total[s:s + window] += p
        count[s:s + window] += 1
    return (total / np.maximum(count, 1))[:n].astype(np.float32)
