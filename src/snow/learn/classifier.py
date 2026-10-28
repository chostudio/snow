"""Model B, the factored trick classifier (docs/training.md).

For one air: how far the rider turned (a multiple of 180), which way
(clockwise or counter-clockwise about world up), the stance (so frontside or
backside can be composed), the axis (flat, cork, double cork) and the grab.
Separate heads over one shared encoder, so a head whose data is too thin can be
dropped without taking the others with it.

The encoder is a small dilated temporal CNN over the air resampled to
``airs.SEQ_LEN`` steps, with attention pooling. Well under a million
parameters; minutes to train.

Prediction averages the air and its mirror image, with the mirrored sense and
stance swapped back, so the model's answer about the turn's sense cannot
depend on which way the camera happened to face.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import config as cfg
from . import airs
from . import dataset as ds

WEIGHTS = cfg.DATA_DIR / "models" / "classifier" / "classifier.pt"
HEAD_NAMES = tuple(ds.HEADS)
# Heads whose label a mirror swaps: their two logits are exchanged when the
# mirrored copy's prediction is folded back in.
SWAPPED_BY_MIRROR = ("sense", "stance")


def _torch():
    import torch  # deferred: the rest of the package must import without it

    return torch


def build_model(channels: int = airs.N_CHANNELS, width: int = 64, dropout: float = 0.2):
    torch = _torch()
    nn = torch.nn

    class Block(nn.Module):
        def __init__(self, dilation: int):
            super().__init__()
            self.conv = nn.Conv1d(width, width, 5, padding=2 * dilation, dilation=dilation)
            self.norm = nn.GroupNorm(8, width)
            self.drop = nn.Dropout(dropout)

        def forward(self, x):
            return x + self.drop(torch.nn.functional.gelu(self.norm(self.conv(x))))

    class TrickNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Conv1d(channels, width, 1)
            self.blocks = nn.Sequential(*[Block(d) for d in (1, 2, 4, 8, 1)])
            self.attn = nn.Conv1d(width, 1, 1)
            self.heads = nn.ModuleDict({
                name: nn.Linear(2 * width, len(classes)) for name, classes in ds.HEADS.items()
            })
            # Spin in turns, as an auxiliary ordinal signal for the rotation head.
            self.turns = nn.Linear(2 * width, 1)

        def forward(self, x):
            h = self.blocks(self.inp(x))
            weights = torch.softmax(self.attn(h), dim=-1)
            pooled = torch.cat([(h * weights).sum(-1), h.mean(-1)], dim=1)
            out = {name: head(pooled) for name, head in self.heads.items()}
            out["turns"] = self.turns(pooled).squeeze(-1)
            return out

    return TrickNet()


# --- augmentation bank ---------------------------------------------------------

def _encode_variants(args) -> tuple[np.ndarray, np.ndarray]:
    """``count`` augmented encodings of one air and their (possibly mirrored) labels."""
    track, gt, labels, count, seed = args
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    plain = [airs.encode(track, *gt), airs.encode(airs.mirror(track), *gt)]
    for k, x in enumerate(plain):
        if x is not None:
            xs.append(x)
            ys.append(_label_row(ds.mirrored_labels(labels) if k else labels))
    tries = 0
    while len(xs) < count and tries < 3 * count:
        tries += 1
        x, mirrored = airs.augment(track, *gt, rng)
        if x is None:
            continue
        xs.append(x)
        ys.append(_label_row(ds.mirrored_labels(labels) if mirrored else labels))
    return np.stack(xs), np.stack(ys)


def _label_row(labels: dict[str, int]) -> np.ndarray:
    return np.array([labels[h] for h in HEAD_NAMES], dtype=np.int64)


def bank(records: list[ds.Air], per_air: int = 128, seed: int = 0,
         workers: int = 4) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Augmented encodings per clip name, built once and shared by every fold."""
    jobs = [(a.track, a.gt, a.labels, per_air, seed + i) for i, a in enumerate(records)]
    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            results = list(pool.map(_encode_variants, jobs))
    else:
        results = [_encode_variants(j) for j in jobs]
    return {a.name: r for a, r in zip(records, results)}


# --- training ------------------------------------------------------------------

@dataclass
class TrainConfig:
    epochs: int = 40
    batch: int = 64
    lr: float = 2e-3
    weight_decay: float = 1e-2
    width: int = 64
    dropout: float = 0.2
    turns_weight: float = 0.5
    seed: int = 0


def train(train_names: list[str], variants: dict, config: TrainConfig | None = None,
          device: str | None = None, val: tuple[np.ndarray, np.ndarray] | None = None,
          log=print) -> dict:
    """Fit on the banked variants of ``train_names``. Returns a checkpoint dict."""
    torch = _torch()
    config = config or TrainConfig()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config.seed)
    x = np.concatenate([variants[n][0] for n in train_names])
    y = np.concatenate([variants[n][1] for n in train_names])
    mean = x.mean(axis=(0, 2), keepdims=True)[0]
    std = x.std(axis=(0, 2), keepdims=True)[0] + 1e-3
    xt = torch.tensor((x - mean) / std, dtype=torch.float32, device=device)
    yt = torch.tensor(y, device=device)
    turns = torch.tensor([ds.ROTATIONS[i] / 360.0 for i in y[:, 0]], device=device)

    model = build_model(width=config.width, dropout=config.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr,
                            weight_decay=config.weight_decay)
    steps = config.epochs * max(1, int(np.ceil(len(xt) / config.batch)))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=config.lr, total_steps=steps)
    ce = torch.nn.CrossEntropyLoss()
    gen = torch.Generator(device="cpu").manual_seed(config.seed)
    start = time.time()
    for epoch in range(config.epochs):
        model.train()
        order = torch.randperm(len(xt), generator=gen).to(device)
        total = 0.0
        for i in range(0, len(order), config.batch):
            idx = order[i:i + config.batch]
            out = model(xt[idx])
            loss = sum(ce(out[h], yt[idx, k]) for k, h in enumerate(HEAD_NAMES))
            loss = loss + config.turns_weight * torch.nn.functional.smooth_l1_loss(
                out["turns"], turns[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += float(loss) * len(idx)
        if log and (epoch % 10 == 9 or epoch == config.epochs - 1):
            msg = f"    epoch {epoch + 1:3d}  loss {total / len(xt):.3f}"
            if val is not None:
                acc = _accuracy(model, mean, std, *val, device)
                msg += "  val " + " ".join(f"{h} {a:.2f}" for h, a in acc.items())
            log(msg + f"  ({time.time() - start:.0f}s)")
    return {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "mean": mean, "std": std, "config": config.__dict__,
        "heads": {h: list(c) for h, c in ds.HEADS.items()},
        "channels": list(airs.CHANNELS), "seq_len": airs.SEQ_LEN,
        "train": sorted(train_names),
    }


def _accuracy(model, mean, std, x, y, device) -> dict[str, float]:
    torch = _torch()
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor((x - mean) / std, dtype=torch.float32, device=device))
    return {h: float((out[h].argmax(1).cpu().numpy() == y[:, k]).mean())
            for k, h in enumerate(HEAD_NAMES)}


# --- inference -----------------------------------------------------------------

class Classifier:
    """A trained checkpoint, ready to name airs."""

    def __init__(self, checkpoint: dict, device: str = "cpu"):
        torch = _torch()
        c = checkpoint["config"]
        if checkpoint.get("channels") != list(airs.CHANNELS):
            raise ValueError("checkpoint was trained on a different input encoding")
        self.model = build_model(width=c["width"], dropout=c["dropout"])
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(device).eval()
        self.mean, self.std = checkpoint["mean"], checkpoint["std"]
        self.device = device
        self.torch = torch

    @classmethod
    def load(cls, path: Path = WEIGHTS, device: str = "cpu") -> "Classifier":
        torch = _torch()
        return cls(torch.load(path, map_location=device, weights_only=False), device)

    def probabilities(self, encoded: list[np.ndarray],
                      mirrored: list[np.ndarray] | None = None) -> dict[str, np.ndarray]:
        """Per-head class probabilities for encoded airs, mirror-averaged."""
        torch = self.torch
        with torch.no_grad():
            def run(batch):
                x = torch.tensor((np.stack(batch) - self.mean) / self.std,
                                 dtype=torch.float32, device=self.device)
                return {h: torch.softmax(v, -1).cpu().numpy()
                        for h, v in self.model(x).items() if h in ds.HEADS}
            probs = run(encoded)
            if mirrored is not None:
                back = run(mirrored)
                for h in HEAD_NAMES:
                    other = back[h][:, ::-1] if h in SWAPPED_BY_MIRROR else back[h]
                    probs[h] = (probs[h] + other) / 2
        return probs

    def classify(self, track: airs.Track, start: int, end: int) -> dict | None:
        """One air's trick, per head, with a composed name. None if unposed."""
        x = airs.encode(track, start, end)
        xm = airs.encode(airs.mirror(track), start, end)
        if x is None or xm is None:
            return None
        probs = self.probabilities([x], [xm])
        return describe({h: p[0] for h, p in probs.items()})


def describe(probs: dict[str, np.ndarray]) -> dict:
    """Head outputs as labels with confidences, and a composed trick name."""
    out = {}
    for head, classes in ds.HEADS.items():
        k = int(np.argmax(probs[head]))
        out[head] = {"label": classes[k], "p": round(float(probs[head][k]), 3)}
    out["direction"] = ds.direction(ds.SENSES.index(out["sense"]["label"]),
                                    ds.STANCES.index(out["stance"]["label"]))
    axis = {"flat": "", "cork": "cork ", "double_cork": "double cork "}[out["axis"]["label"]]
    grab = out["grab"]["label"]
    out["name"] = (f"{out['direction']} {axis}{out['rotation']['label']}"
                   + ("" if grab == "none" else f" {grab}"))
    return out


def save(checkpoint: dict, dest: Path = WEIGHTS) -> Path:
    torch = _torch()
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, dest)
    meta = {k: checkpoint[k] for k in ("config", "heads", "seq_len", "train")}
    dest.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    return dest


def available(path: Path = WEIGHTS) -> bool:
    """True when weights exist and torch imports; callers fall back otherwise."""
    if not path.exists():
        return False
    try:
        _torch()
    except ImportError:
        return False
    return True
