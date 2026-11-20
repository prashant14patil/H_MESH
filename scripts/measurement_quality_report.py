r"""Per-measurement quality grades for existing runs (backend/services/measurement_quality.py).

Runs made before the pipeline saved its cross-validation fold shapes have no fold girths; --folds refits the 4
heading folds offline from diagnostics/fit_inputs.npz (pipeline defaults; ~4 fits per run). Writes
<run>/measurement_quality.json unless --dry-run.

Usage: .venv\\Scripts\\python.exe scripts/measurement_quality_report.py RUN_ID [RUN_ID ...] [--folds] [--height-cm 176]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from backend.services import measurement_quality as mq  # noqa: E402

RUNS = ROOT / "outputs" / "reconstructions"


def _json(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--folds", action="store_true", help="refit the 4 CV fold shapes offline when not saved")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--height-cm", type=float)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    smpl = None
    for rid in a.runs:
        d = next(RUNS.glob(rid + "*"))
        meas, rep = _json(d / "measurements.json"), _json(d / "reconstruction_report.json") or {}
        rq = _json(d / "reconstruction_quality_report.json")
        if rq is None and rep:  # older run: evaluate the same run-level checks from its report
            from backend.services import scan_quality as sq
            rq = sq.assess_reconstruction(rep)
        cq = _json(d / "capture_quality_report.json") or {}
        folds = rep.get("multiview_optimisation", {}).get("held_out_validation", {}).get("folds", [])
        fold_girths = None
        if a.folds or any(f.get("betas") for f in folds):
            from measurement_accuracy_study import girths, load_run, load_smpl
            smpl = smpl or load_smpl(a.device)
            if any(f.get("betas") for f in folds):
                fold_girths = [girths(smpl, np.asarray(f["betas"]), a.device) for f in folds if f.get("betas")]
            else:
                from reliability_study import jackknife
                fold_girths = jackknife(smpl, np.asarray(smpl.faces, dtype=np.int64), load_run(d.name), a.device)
        z = np.load(d / "diagnostics" / "fit_inputs.npz") if (d / "diagnostics" / "fit_inputs.npz").is_file() else None
        yaws = None
        if z is not None:
            usable = z["mask_usable"]
            fr = d / "diagnostics" / "fit_result.npz"
            if fr.is_file() and "sil_weight" in np.load(fr):
                usable = usable & (np.load(fr)["sil_weight"] > 0)
            yaws = [float(y) for y, u in zip(z["yaw"], usable) if u]
        height = a.height_cm or (_json(d / "measurements_cm.json") or {}).get("height_cm")
        out = mq.assess(meas, rep, fold_girths=fold_girths, usable_yaws=yaws, capture_verdict=cq.get("verdict"),
                        reconstruction_verdict=rq.get("verdict"), height_cm=height)
        out["fold_girths_x_height"] = fold_girths
        out["source"] = "scripts/measurement_quality_report.py (backfill)"
        print(f"{d.name[:8]} run={out['run_grade']} counts={out['counts']}")
        for s, v in out["sites"].items():
            cm = f" {v['value_cm']:.1f}±{v['precision_sd_cm']:.1f} cm" if v.get("value_cm") else ""
            fsd = f" foldSD={v['fold_sd_pct']:.2f}%" if v.get("fold_sd_pct") is not None else ""
            print(f"   {s:16s} {v['grade']:11s}{cm}{fsd} {'; '.join(v.get('reasons', []))}")
        if not a.dry_run:
            (d / "measurement_quality.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
