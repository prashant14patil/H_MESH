"""Pool tape-measure validation across reconstruction runs.

Usage:
    .venv/Scripts/python.exe scripts/tape_validation_report.py [--root outputs/reconstructions] [--json out.json]

Reads every <run>/tape_validation.json (written via POST /api/reconstruction/runs/{id}/tape) and prints
per-site bias and error between the model's centimetre circumferences and independent tape measurements.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.services.body_measurements import aggregate_tape  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="outputs/reconstructions")
    ap.add_argument("--json", help="also write the aggregate to this file")
    args = ap.parse_args()
    records = []
    for path in sorted(Path(args.root).glob("*/tape_validation.json")):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"skipping {path}: {exc}", file=sys.stderr)
    if not records:
        print(f"No tape_validation.json files under {args.root}. Record tape measurements for scaled runs first.")
        return 1
    agg = aggregate_tape(records)
    print(f"Subjects: {agg['subjects']}   clothing: {agg['clothing_counts']}")
    print(f"{'site':16s} {'n':>3s} {'bias cm':>8s} {'SD cm':>6s} {'MAE cm':>7s} {'MAE %':>6s} {'max cm':>7s}")
    for site, s in agg["per_site"].items():
        sd = f"{s['sd_error_cm']:6.2f}" if s["sd_error_cm"] is not None else "     -"
        print(f"{site:16s} {s['n']:3d} {s['mean_error_cm']:8.2f} {sd} {s['mean_abs_error_cm']:7.2f} "
              f"{s['mean_abs_error_pct']:6.2f} {s['max_abs_error_cm']:7.2f}")
    print(agg["sign_convention"])
    print(agg["caveat"])
    if args.json:
        Path(args.json).write_text(json.dumps(agg, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
