"""Per-clip metadata, written by ingest and read by every later stage.

The slow-motion factor lives here because it cannot be recovered from the
file: broadcast slow motion is shot at a high frame rate and *conformed* to a
normal one, so the container says 30 fps and the timestamps lie about how much
real time passed between frames. Total rotation is unaffected — it is just
more samples of the same turn — but every rate (deg/s, angular velocity) is
wrong by exactly this factor.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import CLIPS_DIR


def meta_path(name: str) -> Path:
    return CLIPS_DIR / f"{name}.meta.json"


@dataclass
class ClipMeta:
    name: str
    source: str = ""
    start: str | None = None
    duration: str | None = None
    fps: float = 30.0
    frames: int = 0
    width: int = 0
    height: int = 0
    # Playback is this many times slower than reality. 1.0 = real time,
    # 4.0 = a 120 fps camera conformed to 30 fps.
    slowmo_factor: float = 1.0
    # Set by the crop stage.
    crop: dict | None = None
    extra: dict = field(default_factory=dict)

    @property
    def real_fps(self) -> float:
        """Frames per second of *real* time, which is what rates need."""
        return self.fps * self.slowmo_factor

    @property
    def dt(self) -> float:
        """Real seconds between consecutive frames."""
        return 1.0 / self.real_fps if self.real_fps else 0.0

    def save(self) -> Path:
        path = meta_path(self.name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))
        return path


def load(name: str) -> ClipMeta:
    path = meta_path(name)
    if not path.exists():
        raise FileNotFoundError(
            f"no metadata for clip '{name}' at {path} — re-run `snow ingest`"
        )
    data = json.loads(path.read_text())
    known = {f for f in ClipMeta.__dataclass_fields__}
    return ClipMeta(**{k: v for k, v in data.items() if k in known})
