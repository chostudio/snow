"""Paths and settings, loaded from .env at the repo root."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
CLIPS_DIR = DATA_DIR / "clips"
POSES_DIR = DATA_DIR / "poses"
WEIGHTS_DIR = DATA_DIR / "weights"
VENDOR_CLIENT = REPO_ROOT / "vendor" / "nim-clients" / "body-pose" / "scripts" / "body_pose.py"

# The hosted NVCF endpoint rejects anything larger. Stay under it with margin:
# the limit is on the uploaded file, and we would rather re-encode than fail
# after a long upload.
HOSTED_MAX_BYTES = 50 * 1024 * 1024
HOSTED_TARGET_BYTES = 46 * 1024 * 1024

# Per the NIM support matrix.
MAX_BODIES = 50

load_dotenv(REPO_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    api_key: str | None
    target: str
    function_id: str
    preview: bool

    @property
    def hosted(self) -> bool:
        return self.preview

    def require_key(self) -> str:
        if not self.api_key:
            raise SystemExit(
                "NVIDIA_API_KEY is not set. Copy .env.example to .env and put your "
                "key from https://build.nvidia.com/nvidia/body-pose in it."
            )
        return self.api_key


def settings() -> Settings:
    return Settings(
        api_key=os.environ.get("NVIDIA_API_KEY") or None,
        target=os.environ.get("SNOW_NIM_TARGET", "grpc.nvcf.nvidia.com:443"),
        function_id=os.environ.get(
            "SNOW_NIM_FUNCTION_ID", "254b3621-bb66-4b7b-9c89-621c352b4d91"
        ),
        preview=os.environ.get("SNOW_NIM_PREVIEW", "1") not in ("0", "false", "False", ""),
    )


def clip_path(name: str) -> Path:
    return CLIPS_DIR / f"{name}.mp4"


def bbox_path(name: str) -> Path:
    return CLIPS_DIR / f"{name}.bbox.txt"


def pose_path(name: str) -> Path:
    # The upstream client insists on a .json extension even though it writes
    # JSON Lines.
    return POSES_DIR / f"{name}.json"


def ensure_dirs() -> None:
    for d in (CLIPS_DIR, POSES_DIR, WEIGHTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
