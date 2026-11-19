"""Plausibility check of a reconstruction against ANSUR II (4,082 tape-measured US Army men; public 2017 release).

Without the subject's own tape this gives two independent checks:

1. CONSISTENCY between girths. In real men chest, navel-waist and buttock girths are strongly linked. Each girth is
   predicted from the other two plus stature (linear regression on ANSUR men, residual SD reported). A mesh girth
   beyond ~2 SD of that prediction is a combination real men rarely have: the likely sign of a local error such as
   loose trousers inflating the hip. It does not say which value is right, only which one is out of line.
2. LANDMARK HEIGHTS. Where each girth is measured, as a fraction of stature, vs where ANSUR measured it.

Also printed: each girth's percentile among ANSUR men of similar stature (context, not accuracy).

Not done: comparing mesh breadth/depth with ANSUR breadth/depth. ANSUR breadths are caliper distances between
landmarks (girth ~ 1.17 x ellipse(breadth, depth) in ANSUR) and are not comparable with a section's full extent.

    .venv\\Scripts\\python.exe scripts/ansur_check.py RUN_ID [--height-cm 179] [--detailed]

Data: data/reference/ansur2/ANSUR_II_MALE_Public.csv from https://www.openlab.psu.edu/ansur2/ (mm; weight in hg).
Male reference only: use it for male subjects.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

ANSUR = ROOT / "data" / "reference" / "ansur2" / "ANSUR_II_MALE_Public.csv"
# our site -> (ANSUR girth column, ANSUR landmark-height column). Our "abdomen" is at navel level = ANSUR "waist".
SITES = {"chest": ("chestcircumference", "chestheight"),
         "abdomen": ("waistcircumference", "waistheightomphalion"),
         "hip": ("buttockcircumference", "buttockheight")}


def load_ansur():
    rows = list(csv.DictReader(open(ANSUR, encoding="latin-1")))
    cols = {c for v in SITES.values() for c in v} | {"stature", "weightkg"}
    return {c: np.array([float(r[c]) for r in rows]) for c in cols}


def consistency(A, girths_cm: dict, stature_cm: float) -> dict:
    """Each site's girth predicted from the other two + stature; returns prediction, residual SD and z (cm)."""
    out = {}
    for site, (col, _) in SITES.items():
        others = [s for s in SITES if s != site]
        X = np.stack([A[SITES[o][0]] for o in others] + [A["stature"], np.ones(len(A["stature"]))], 1)
        coef, *_ = np.linalg.lstsq(X, A[col], rcond=None)
        sd = float((A[col] - X @ coef).std())
        pred = float(np.array([girths_cm[o] * 10 for o in others] + [stature_cm * 10, 1.0]) @ coef)
        out[site] = {"mesh_cm": girths_cm[site], "typical_cm": pred / 10, "sd_cm": sd / 10,
                     "z": (girths_cm[site] * 10 - pred) / sd}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--height-cm", type=float, default=179.0)
    ap.add_argument("--detailed", action="store_true", help="check final_body_mesh_detailed.obj instead")
    a_ = ap.parse_args()
    import measurement_accuracy_study as mas
    from backend.services import body_measurements as bm
    from backend.services import smpl_multiview as smv
    from backend.services.body_reconstruction import stature_height

    d = next((ROOT / "outputs" / "reconstructions").glob(a_.run_id + "*"))
    smpl = mas.load_smpl("cpu")
    stature = stature_height(smpl, np.load(d / "diagnostics" / "fit_result.npz")["betas"], "cpu")
    mesh_name = "final_body_mesh_detailed.obj" if a_.detailed else "final_body_mesh.obj"
    v, f = smv.read_obj(d / mesh_name)
    meas = json.loads((d / ("measurements_detailed.json" if a_.detailed else "measurements.json")).read_text())["measurements"]
    cm = a_.height_cm / stature
    A = load_ansur()
    near = np.abs(A["stature"] - a_.height_cm * 10) <= 40
    girths, heights = {}, {}
    for site in SITES:
        m = meas[site]
        sec = bm.measure_section(v, f, np.array(m["plane_origin"]), np.array(m["plane_normal"]))
        girths[site] = sec["tape_perimeter"] * cm
        heights[site] = (m["plane_origin"][1] - v[:, 1].min()) / stature
    cons = consistency(A, girths, a_.height_cm)
    print(f"run {d.name[:8]} ({mesh_name}) at {a_.height_cm:.0f} cm; ANSUR men n={len(A['stature'])} "
          f"(similar stature n={near.sum()})")
    print(f"{'site':8s} {'mesh cm':>8s} | {'typical given other girths':>27s} {'z':>6s} | {'plane h/stature':>15s} "
          f"{'ANSUR':>6s} | {'pctile':>6s}")
    out = {}
    for site, (col, hcol) in SITES.items():
        c = cons[site]
        ansur_h = float(np.median(A[hcol] / A["stature"]))
        pct = float((A[col][near] < girths[site] * 10).mean() * 100)
        flag = "  <-- out of line" if abs(c["z"]) > 2 else ""
        print(f"{site:8s} {girths[site]:8.1f} | {c['typical_cm']:19.1f} +-{c['sd_cm']:4.1f} {c['z']:+6.1f} | "
              f"{heights[site]:15.3f} {ansur_h:6.3f} | {pct:5.0f}%{flag}")
        out[site] = {**c, "plane_height_over_stature": heights[site], "ansur_height_over_stature": ansur_h,
                     "percentile_similar_stature": pct}
    (d / "diagnostics" / f"ansur_check{'_detailed' if a_.detailed else ''}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
