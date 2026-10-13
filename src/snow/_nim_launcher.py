"""Run the vendored NVIDIA client in-process, taking the API key from the env.

The upstream client only accepts its key as ``--api-key`` on the command line,
which puts the secret in the process list for every user on the machine. This
launcher reads it from the environment and injects it into ``sys.argv`` after
the process has already started, so it never appears in the real argv.
"""

from __future__ import annotations

import os
import runpy
import sys

from .config import VENDOR_CLIENT


def main() -> None:
    if not VENDOR_CLIENT.exists():
        raise SystemExit(
            f"vendored client missing at {VENDOR_CLIENT}\nRun: scripts/setup.sh"
        )
    argv = sys.argv[1:]
    key = os.environ.pop("SNOW_NIM_API_KEY", None)
    if key:
        argv += ["--api-key", key]

    scripts_dir = VENDOR_CLIENT.parent
    # `python script.py` would put the script's directory first on sys.path;
    # runpy does not, and the client does a bare `from config import ...`.
    sys.path.insert(0, str(scripts_dir))
    sys.argv = [str(VENDOR_CLIENT), *argv]
    runpy.run_path(str(VENDOR_CLIENT), run_name="__main__")


if __name__ == "__main__":
    main()
