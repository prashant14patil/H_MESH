"""Measurement validation report: repeatability, capture-condition effects and accuracy against tape.

Pools every completed reconstruction under outputs/reconstructions (one row per distinct source
recording, newest run; only runs made with the same fit settings). See
backend/services/measurement_validation.py for the statistics.

Usage:
  .venv\\Scripts\\python.exe scripts/validation_report.py --subject S01 --markdown outputs/validation/report.md
  # tag an existing run with its capture conditions
  .venv\\Scripts\\python.exe scripts/validation_report.py --tag RUN_ID --subject S01 --clothing loose --distance 3 --posture a_pose_standard
  # mark a run's tape record as test data (never counted as validation)
  .venv\\Scripts\\python.exe scripts/validation_report.py --exclude-tape RUN_ID --reason "test entry"
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.services import measurement_validation as mv  # noqa: E402

LABELS = {"chest": "Chest", "waist": "Waist", "abdomen": "Abdomen", "hip": "Hip", "upper_arm_left": "Upper arm L",
          "upper_arm_right": "Upper arm R", "forearm_left": "Forearm L", "forearm_right": "Forearm R",
          "thigh_left": "Thigh L", "thigh_right": "Thigh R", "calf_left": "Calf L", "calf_right": "Calf R"}


def fmt(v, d=2):
    return "—" if v is None else f"{v:.{d}f}"


def markdown(rep: dict) -> str:
    out = [f"# Measurement validation report ({datetime.now():%d %b %Y %H:%M})", ""]
    out.append(f"Runs found {rep['runs_found']}, same fit settings {rep['runs_considered']}, "
               f"distinct recordings {rep['distinct_recordings']}, excluded tape records {rep['excluded_tape_records']}.")
    if rep.get("fit_config_note"):
        out.append(f"\n> {rep['fit_config_note']}")
    out += ["", "## Recordings", "", "| Run | Created | Subject | Clothing | Distance m | Posture | Height cm |",
            "|---|---|---|---|---|---|---|"]
    for r in rep["recordings"]:
        c = r.get("conditions") or {}
        out.append(f"| {r['run_id'][:8]} | {datetime.fromtimestamp(r['created_at']):%d %b %H:%M} | "
                   f"{c.get('subject_id', '—')} | {c.get('clothing', '—')} | {fmt(c.get('distance_m'), 1)} | "
                   f"{c.get('posture', '—')} | {fmt(r.get('height_cm'), 1)} |")
    rp = rep["repeatability"]["per_site"]
    out += ["", "## Repeatability (distinct recordings of the same subject)", "",
            "| Site | Recordings | CV % | Within-subject SD cm | RC 95% cm | Note |", "|---|---|---|---|---|---|"]
    for site in mv.SITES:
        r = rp.get(site, {})
        if not r.get("evaluable"):
            out.append(f"| {LABELS[site]} | <2 | — | — | — | needs ≥ 2 recordings of one subject |")
            continue
        out.append(f"| {LABELS[site]} | {r['recordings']} | {fmt(r['cv_pct'])} | {fmt(r.get('within_sd_cm'))} | "
                   f"{fmt(r.get('rc_cm'))} | {'low n' if r['low_sample'] else ''} |")
    out += ["", "## Capture-condition effects", ""]
    for factor, eff in rep["condition_effects"].items():
        if not eff.get("evaluable"):
            out.append(f"- **{factor}**: not evaluable ({eff['reason']}; levels seen: {', '.join(eff['levels_seen']) or 'none'}).")
            continue
        out.append(f"- **{factor}** (reference: {eff['reference']})")
        for lv, sites in eff["levels"].items():
            parts = [f"{LABELS[s]} {v['mean_diff_x_height'] * 100:+.2f}%H{' *' if v['exceeds_repeatability'] else ''}"
                     for s, v in sites.items()]
            out.append(f"  - {lv}: " + ", ".join(parts))
    out.append("\n`*` = larger than the repeatability coefficient (a real effect, not run-to-run noise).")
    ta = rep["tape_accuracy"]
    out += ["", "## Accuracy against tape (model − tape)", ""]
    if not ta.get("subjects"):
        out.append(ta.get("note", "no tape records"))
    else:
        out += [f"Subjects {ta['subjects']}; clothing {ta['clothing_counts']}.", "",
                "| Site | n | Bias cm | SD cm | MAE cm | MAE % | Max cm |", "|---|---|---|---|---|---|---|"]
        for site, s in ta["per_site"].items():
            out.append(f"| {LABELS[site]} | {s['n']} | {fmt(s['mean_error_cm'])} | {fmt(s['sd_error_cm'])} | "
                       f"{fmt(s['mean_abs_error_cm'])} | {fmt(s['mean_abs_error_pct'])} | {fmt(s['max_abs_error_cm'])} |")
        out.append(f"\n{ta['caveat']}")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=ROOT / "outputs" / "reconstructions")
    ap.add_argument("--subject", help="subject code: tags with --tag, else default subject for untagged runs")
    ap.add_argument("--fit-config", default="latest", help="'latest' (default), 'any', or an exact JSON config")
    ap.add_argument("--since", help="only runs created after this local time, e.g. 2026-11-18T14:30")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--markdown", type=Path)
    ap.add_argument("--tag", metavar="RUN_ID")
    ap.add_argument("--clothing", default="unknown")
    ap.add_argument("--distance", type=float)
    ap.add_argument("--posture", default="a_pose_standard")
    ap.add_argument("--height-cm", type=float)
    ap.add_argument("--note", default="")
    ap.add_argument("--exclude-tape", metavar="RUN_ID", nargs="+")
    ap.add_argument("--reason", default="test entry, not a real tape measurement")
    a = ap.parse_args()

    if a.tag:
        rec = mv.write_conditions(a.root / a.tag, {"subject_id": a.subject, "clothing": a.clothing,
                                                   "distance_m": a.distance, "posture": a.posture,
                                                   "height_cm": a.height_cm, "note": a.note})
        print("tagged", json.dumps(rec))
        return 0
    if a.exclude_tape:
        for run in a.exclude_tape:
            mv.exclude_tape(a.root / run, a.reason)
            print("excluded tape record of", run)
        return 0

    from backend.services.body_reconstruction import _source_key
    since = datetime.fromisoformat(a.since).timestamp() if a.since else None
    rep = mv.build_report(a.root, _source_key, since=since, default_subject=a.subject, fit_config=a.fit_config)
    text = markdown(rep)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(text)
    if a.markdown:
        a.markdown.parent.mkdir(parents=True, exist_ok=True)
        a.markdown.write_text(text, encoding="utf-8")
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
