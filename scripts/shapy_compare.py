"""Compare SHAPY with the pipeline's circumferences for one completed reconstruction run (CLI).

Uses backend/services/shapy_shape.py (the same code as the app's SHAPY / Compare views). SHAPY weights
go in models/shapy (SHAPY download, unzipped) and its public code in models/shapy/code; both git-ignored.

With tape measurements of the same person (--tape, or the run's recorded tape_validation.json) both models
are scored by the same compare_to_tape() the app uses for the pipeline, so one tape session shows which
model is closer for this person and this clothing.

Usage:
  .venv/Scripts/python.exe scripts/shapy_compare.py RUN_ID --height-cm 176 [--recompute]
  .venv/Scripts/python.exe scripts/shapy_compare.py RUN_ID --height-cm 176 --tape waist=92 hip=104 chest=108
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from backend.services import body_measurements as bm  # noqa: E402
from backend.services import shapy_shape  # noqa: E402


def parse_tape(items: list[str]) -> dict:
    """['waist=92', 'hip=104.5'] -> validated {site: cm}; sites as in the app (thigh_left, calf_right, ...)."""
    tape = {}
    for item in items:
        site, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"--tape expects SITE=CM, got {item!r}")
        try:
            tape[site.strip()] = float(value)
        except ValueError:
            raise ValueError(f"--tape {item!r}: {value!r} is not a number") from None
    try:
        return bm.validate_tape(tape)
    except ValueError as exc:
        raise ValueError(f"{exc} (sites: {', '.join(bm.TAPE_SITES)})") from None


def score(pipeline_scaled: dict, shapy_scaled: dict, tape: dict) -> dict:
    """Per-site errors (model - tape) for both models and their mean absolute error on the sites both cover."""
    pipe = bm.compare_to_tape(pipeline_scaled, tape)["sites"]
    shp = bm.compare_to_tape(shapy_scaled, tape)["sites"]
    common = [s for s in tape if pipe[s].get("compared") and shp[s].get("compared")]
    out = {"sites": {s: {"pipeline": pipe[s].get("error_cm"), "shapy": shp[s].get("error_cm")} for s in tape},
           "common_sites": common}
    if common:
        out["pipeline_mae_cm"] = sum(abs(pipe[s]["error_cm"]) for s in common) / len(common)
        out["shapy_mae_cm"] = sum(abs(shp[s]["error_cm"]) for s in common) / len(common)
    return out


def _fmt(v, spec="9.1f"):
    if isinstance(v, (int, float)):
        return format(v, spec)
    return format("-", ">" + "".join(c for c in spec.split(".")[0] if c.isdigit()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_id")
    ap.add_argument("--height-cm", type=float, required=True, help="measured barefoot height (sets the scale)")
    ap.add_argument("--recompute", action="store_true", help="ignore a stored SHAPY result")
    ap.add_argument("--tape", nargs="+", metavar="SITE=CM",
                    help="tape circumferences of the same person, e.g. waist=92 hip=104 chest=108 thigh_left=58")
    args = ap.parse_args()
    height_cm = bm.validate_height_cm(args.height_cm)
    run_dir = ROOT / "outputs" / "reconstructions" / args.run_id
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))

    tape, tape_source = None, None
    if args.tape:
        try:
            tape, tape_source = parse_tape(args.tape), "--tape"
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    elif (run_dir / "tape_validation.json").is_file():
        rec = json.loads((run_dir / "tape_validation.json").read_text(encoding="utf-8"))
        tape = bm.validate_tape(rec["tape_cm"])
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(rec.get("recorded_at", 0)))
        tape_source = (f"stored tape_validation.json ({when}, clothing {rec.get('clothing')}"
                       f"{', note ' + repr(rec['note']) if rec.get('note') else ''}) - make sure it is a real measurement")

    stored = run_dir / shapy_shape.RESULT_FILE
    record = json.loads(stored.read_text(encoding="utf-8")) if stored.is_file() and not args.recompute else None
    if record is None or record.get("version", 1) < shapy_shape.RESULT_VERSION:
        videos = list((ROOT / "data" / "captures" / status["video_id"]).glob("original.*"))
        if len(videos) != 1:
            print(f"error: the run's source video is missing or ambiguous ({len(videos)} found)", file=sys.stderr)
            return 1
        record = shapy_shape.compute_for_run(run_dir, videos[0])
    shapy_scaled = bm.scale_to_height(record, height_cm)
    pipeline = bm.scale_to_height(json.loads((run_dir / "measurements.json").read_text(encoding="utf-8")), height_cm)
    result = score(pipeline, shapy_scaled, tape) if tape else None

    print(f"SHAPY on {record['frames_used']} frames of run {args.run_id[:8]}, height {height_cm:g} cm")
    if tape_source:
        print(f"tape: {tape_source}")
    head = f"{'site':16s} {'pipeline':>9s} {'SHAPY':>8s} {'diff':>7s}"
    print(head + (f" {'tape':>7s} {'pipe err':>9s} {'SHAPY err':>9s}" if tape else ""))
    for site, m in shapy_scaled["measurements"].items():
        s = m.get("tape_circumference_cm") if m.get("valid") else None
        p = (pipeline["measurements"].get(site) or {}).get("tape_circumference_cm")
        diff = f"{s - p:+7.1f}" if (p is not None and s is not None) else "      -"
        line = f"{site:16s} {_fmt(p)} {_fmt(s, '8.1f')} {diff}"
        if tape:
            err = result["sites"].get(site, {})
            line += f" {_fmt(tape.get(site), '7.1f')} {_fmt(err.get('pipeline'), '+9.1f')} {_fmt(err.get('shapy'), '+9.1f')}"
        print(line)
    if result and result["common_sites"]:
        pm, sm = result["pipeline_mae_cm"], result["shapy_mae_cm"]
        closer = "SHAPY" if sm < pm else "pipeline" if pm < sm else "neither"
        print(f"\nmean absolute error over {len(result['common_sites'])} taped site(s): "
              f"pipeline {pm:.1f} cm, SHAPY {sm:.1f} cm -> {closer} closer"
              + (f" by {abs(pm - sm):.1f} cm" if closer != "neither" else ""))
        print("(error = model - tape; one person and one recording - repeat with a second person or fitted clothing "
              "before treating this as settled)")
    elif tape:
        print("\nno taped site has a valid cross-section in both models")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
