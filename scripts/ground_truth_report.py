r"""Tape vs pipeline vs SHAPY for every measured subject (see backend/services/ground_truth_validation.py).

Usage:
  .venv\Scripts\python.exe scripts/ground_truth_report.py --sessions data/validation/sessions.csv \
      --tape data/validation/tape.csv --markdown outputs/validation/ground_truth.md --json outputs/validation/ground_truth.json
Templates: docs/validation/sessions_template.csv, docs/validation/tape_template.csv
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from backend.services import ground_truth_validation as gt  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sessions", type=Path, required=True)
    ap.add_argument("--tape", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=ROOT / "outputs" / "reconstructions")
    ap.add_argument("--target-torso-cm", type=float, default=gt.DEFAULT_TARGET_CM["torso"])
    ap.add_argument("--target-limb-cm", type=float, default=gt.DEFAULT_TARGET_CM["limb"])
    ap.add_argument("--markdown", type=Path)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    try:
        rep = gt.build(gt.read_sessions(a.sessions), gt.read_tape(a.tape), a.root,
                       {"torso": a.target_torso_cm, "limb": a.target_limb_cm})
    except gt.GroundTruthError as exc:
        print(f"input error: {exc}", file=sys.stderr)
        return 2
    text = gt.markdown(rep)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(text)
    for path, content in ((a.markdown, text), (a.json, json.dumps(rep, indent=2))):
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
