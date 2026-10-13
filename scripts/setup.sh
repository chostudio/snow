#!/usr/bin/env bash
# Set up the snow pipeline: venv, dependencies, and the vendored NVIDIA client.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON_VERSION=3.12   # the NVIDIA client supports 3.10-3.12 only

if ! command -v uv >/dev/null; then
  echo "uv not found — install it from https://docs.astral.sh/uv/ or create" >&2
  echo "a Python ${PYTHON_VERSION} venv yourself and run: pip install -e ." >&2
  exit 1
fi

echo "==> creating .venv (python ${PYTHON_VERSION})"
uv venv --python "${PYTHON_VERSION}"
uv pip install -e .

if [ ! -f vendor/nim-clients/body-pose/scripts/body_pose.py ]; then
  echo "==> cloning the NVIDIA AI for Media NIM client"
  mkdir -p vendor
  rm -rf vendor/nim-clients
  # The repo keeps its sample media in Git LFS, which we do not need and which
  # fails the clone outright when git-lfs is not installed.
  GIT_LFS_SKIP_SMUDGE=1 git -c filter.lfs.smudge= -c filter.lfs.process= \
    -c filter.lfs.required=false \
    clone --depth 1 -q https://github.com/NVIDIA-Maxine/nim-clients.git vendor/nim-clients
fi

if [ ! -f .env ]; then
  echo "==> creating .env — put your NVIDIA API key in it"
  cp .env.example .env
  chmod 600 .env
fi

echo
echo "done. activate with: source .venv/bin/activate"
echo "then check the setup with: snow doctor"
