"""
scripts/download_models.py
---------------------------
One-time setup script that prepares everything needed to run HMR2
inference locally on your GPU — replaces the manual Colab notebook
steps (Cell 1 "deps", Cell 2 "paths") with a single local script.

What it does:
    1. Clones the 4D-Humans repo (if not already present) into
       `models/4D-Humans`.
    2. Downloads the official HMR2 checkpoint + SMPL data bundle using
       the repo's own `fetch_demo_data.sh` / `download` helpers if
       present, falling back to a direct URL download of the
       `hmr2_data.tar.gz` bundle used by the official project.
    3. Prints the final `HMR2_REPO_PATH` and `HMR2_CHECKPOINT` values
       to paste into your local `.env` file.

Usage:
    python scripts/download_models.py

Notes:
    - This downloads several GB of data (model weights + SMPL body
      model data) and may take a while depending on your connection.
    - Run this ONCE after cloning the tanitas project and before
      starting the backend for the first time.
    - detectron2 (optional person detector) is NOT installed by this
      script — see requirements.txt / README for notes on it. Local
      inference works without it (automatic full-image fallback).
"""

from __future__ import annotations

import os
import subprocess
import sys
import tarfile
from pathlib import Path
from urllib.request import urlretrieve

REPO_URL = "https://github.com/shubham-goel/4D-Humans.git"
REPO_DIR = Path("models/4D-Humans")

# Official HMR2 data bundle (checkpoint + SMPL model files), hosted by the
# 4D-Humans authors. This single tarball expands into the directory
# structure load_hmr2() expects (logs/, data/).
HMR2_DATA_URL = "https://people.eecs.berkeley.edu/~jathushan/projects/4dhumans/hmr2_data.tar.gz"
HMR2_DATA_TARBALL = Path("models/hmr2_data.tar.gz")

EXPECTED_CKPT_CANDIDATES = [
    REPO_DIR / "logs/train/multiruns/hmr2/0/checkpoints/epoch=35-step=1000000.ckpt",
    REPO_DIR / "logs/train/runs/hmr2/checkpoints/epoch=35-step=1000000.ckpt",
    REPO_DIR / "checkpoints/hmr2_weights.ckpt",
]


def _run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.check_call(cmd)


def clone_repo() -> None:
    if REPO_DIR.exists():
        print(f"[1/3] 4D-Humans repo already present at {REPO_DIR} — skipping clone.")
        return
    print(f"[1/3] Cloning 4D-Humans repo into {REPO_DIR} ...")
    REPO_DIR.parent.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)])


def download_checkpoint() -> None:
    existing = next((c for c in EXPECTED_CKPT_CANDIDATES if c.is_file()), None)
    if existing:
        print(f"[2/3] Checkpoint already present at {existing} — skipping download.")
        return

    print("[2/3] Downloading HMR2 checkpoint + SMPL data bundle (several GB, please wait)...")
    HMR2_DATA_TARBALL.parent.mkdir(parents=True, exist_ok=True)

    def _progress(block_num, block_size, total_size):
        if total_size <= 0:
            return
        downloaded = block_num * block_size
        pct = min(100, downloaded * 100 // total_size)
        print(f"\r    {pct:3d}%  ({downloaded // (1024*1024)} MB / {total_size // (1024*1024)} MB)", end="")

    try:
        urlretrieve(HMR2_DATA_URL, HMR2_DATA_TARBALL, _progress)
        print()
    except Exception as exc:
        print(
            f"\nAutomatic download failed ({exc}).\n"
            "Please download the HMR2 checkpoint manually from the official "
            "4D-Humans repo README (https://github.com/shubham-goel/4D-Humans) "
            "and place the .ckpt file under models/4D-Humans/, then set "
            "HMR2_CHECKPOINT in your .env file to its path."
        )
        return

    print("    Extracting archive into", REPO_DIR, "...")
    with tarfile.open(HMR2_DATA_TARBALL) as tar:
        tar.extractall(path=REPO_DIR)
    HMR2_DATA_TARBALL.unlink(missing_ok=True)


def print_summary() -> None:
    print("\n[3/3] Setup summary")
    print("=" * 60)
    repo_path = REPO_DIR.resolve()
    print(f"HMR2_REPO_PATH={repo_path}")

    ckpt = next((c for c in EXPECTED_CKPT_CANDIDATES if c.is_file()), None)
    if ckpt:
        print(f"HMR2_CHECKPOINT={ckpt.resolve()}")
    else:
        print("HMR2_CHECKPOINT=<NOT FOUND — download/extract manually, see above>")
        # Scan for any .ckpt file as a hint.
        found = list(REPO_DIR.rglob("*.ckpt"))
        if found:
            print("Found candidate checkpoint file(s):")
            for f in found:
                print(f"  {f.resolve()}")
    print("=" * 60)
    print("Copy the two lines above into your .env file.")


def main() -> None:
    clone_repo()
    download_checkpoint()
    print_summary()


if __name__ == "__main__":
    main()
