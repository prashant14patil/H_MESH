"""Compare SHAPY with the pipeline's circumferences for one completed reconstruction run (CLI).

Uses backend/services/shapy_shape.py (the same code as the app's SHAPY / Compare views). SHAPY weights
go in models/shapy (SHAPY download, unzipped) and its public code in models/shapy/code; both git-ignored.

Usage:
  .venv/Scripts/python.exe scripts/shapy_compare.py RUN_ID --height-cm 176 [--recompute]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from backend.services import body_measurements as bm  # noqa: E402
from backend.services import shapy_shape  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_id")
    ap.add_argument("--height-cm", type=float, required=True)
    ap.add_argument("--recompute", action="store_true", help="ignore a stored SHAPY result")
    args = ap.parse_args()
    height_cm = bm.validate_height_cm(args.height_cm)
    run_dir = ROOT / "outputs" / "reconstructions" / args.run_id
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    stored = run_dir / shapy_shape.RESULT_FILE
    if stored.is_file() and not args.recompute:
        record = json.loads(stored.read_text(encoding="utf-8"))
    else:
        video = next((ROOT / "data" / "captures" / status["video_id"]).glob("original.*"))
        record = shapy_shape.compute_for_run(run_dir, video)
    shapy = shapy_shape.summarize(record, height_cm)["measurements_cm"]
    pipeline = bm.scale_to_height(json.loads((run_dir / "measurements.json").read_text(encoding="utf-8")), height_cm)
    print(f"SHAPY on {record['frames_used']} frames of run {args.run_id[:8]}, height {height_cm:g} cm")
    print(f"{'site':16s} {'pipeline':>9s} {'SHAPY':>8s} {'diff':>7s}")
    for site, s in shapy.items():
        p = pipeline["measurements"].get(site, {}).get("tape_circumference_cm")
        diff = f"{s - p:+7.1f}" if (p is not None and s is not None) else "      -"
        print(f"{site:16s} {p if p is not None else float('nan'):9.1f} {s if s is not None else float('nan'):8.1f} {diff}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
