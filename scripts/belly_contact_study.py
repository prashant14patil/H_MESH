"""Anterior-belly contact detail vs tape, side by side with the current default and SHAPY.

HMR2's shape (and SHAPY) underestimates a protruding belly by 4-8 cm (tape study, 20 Nov). Where a belly protrudes,
a loose shirt rests on its front: in a profile view that part of the outline is a CONVEX bulge and a true contact
point. This study lets trunk surface offsets move ONLY the front of the belly (forward-facing trunk vertices between
pelvis and spine2), pushed ONLY by convex outline points (where fabric rests on the body); waist sides and hips
cannot move. Thresholds are fixed a priori (not tuned to the tape):
  * anterior: rest-pose normal z >= 0.3, full weight at z >= 0.7;
  * height band: pelvis joint - 0.03 m .. spine2 joint;
  * contact: convex turn over +-6 outline samples (~8 cm), full weight at 15 degrees.

Decision rule (fixed before running): adopt only if the abdomen improves on ALL recordings, waist/hip errors do not
worsen by > 0.5 cm, and the mean distance to tape improves.

    .venv\\Scripts\\python.exe scripts/belly_contact_study.py RUN [RUN ...] --height-cm 179 \
        --tape waist=88-90 abdomen=95-100 hip=98-103 [--shapy RUN8=w/a/h ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import measurement_accuracy_study as mas  # noqa: E402
import tape_config_study as tcs  # noqa: E402
from backend.services import smpl_multiview as smv  # noqa: E402

K, THETA0 = 6, 15.0


def contact_weights(pts):
    p = np.asarray(pts, dtype=np.float64)
    a = p - np.roll(p, K, 0)
    b = np.roll(p, -K, 0) - p
    turn = np.degrees(np.arctan2(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0], (a * b).sum(1)))
    area = 0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])
    return np.clip(turn * np.sign(area) / THETA0, 0.0, 1.0)


def belly_region(smpl, betas, device):
    n = smv.rest_normals(smpl, betas, device)
    v = (smpl.v_template + __import__("torch").einsum("vck,k->vc", smpl.shapedirs[:, :, :10],
         __import__("torch").as_tensor(betas, dtype=smpl.v_template.dtype, device=smpl.v_template.device)))
    v = v.detach().cpu().numpy()
    j = smpl.J_regressor.detach().cpu().numpy() @ v
    trunk = smv.surface_region_weights(smpl)
    anterior = np.clip((n[:, 2] - 0.3) / 0.4, 0.0, 1.0)
    band = ((v[:, 1] >= j[0, 1] - 0.03) & (v[:, 1] <= j[6, 1])).astype(float)
    return trunk * anterior * band


def belly_offsets(smpl, views, fit, device):
    region = belly_region(smpl, fit.betas, device)
    cws = [cw if pts is None else np.asarray(cw) * contact_weights(pts)
           for pts, cw in zip(views.contours, views.contour_weights)]
    cv = smv.FitViews(**{**views.__dict__, "contour_weights": cws})
    orig = smv.surface_region_weights
    smv.surface_region_weights = lambda _smpl: region
    try:
        res = smv.fit_surface_offsets(smpl, cv, fit, smv.SurfaceConfig(), device)
    finally:
        smv.surface_region_weights = orig
    return smv.offset_vectors(res), res["offsets_cm"][region > 0.5]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--height-cm", type=float, required=True)
    ap.add_argument("--tape", nargs="+", required=True)
    ap.add_argument("--shapy", nargs="*", default=[], help="RUN8=waist/abdomen/hip in cm (from the SHAPY compare)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "accuracy_study" / "belly_contact_study.json"))
    a = ap.parse_args()
    tape = {s: tuple(float(x) for x in r.split("-")) for s, r in (t.split("=") for t in a.tape)}
    shapy = {k: dict(zip(("waist", "abdomen", "hip"), (float(x) for x in v.split("/"))))
             for k, v in (s.split("=") for s in a.shapy)}
    smpl = mas.load_smpl(a.device)
    rows = {}
    for rid in a.runs:
        run = mas.load_run(rid)
        agg, std, _ = smv.robust_beta_aggregate(run["betas_each"], run["quality"] / run["quality"].max())
        res, betas = tcs.fit(smpl, run["views"], agg, np.maximum(std, 0.25), "P02m", a.device)
        fr = smv.FitResult(betas=betas, global_orient=res.global_orient, body_pose=res.body_pose, cam_t=res.cam_t)
        off, belly_cm = belly_offsets(smpl, run["views"], fr, a.device)
        variants = {"current default": tcs.girths(smpl, betas, a.height_cm, a.device),
                    "+ belly contact": tcs.girths(smpl, betas, a.height_cm, a.device, off)}
        if rid[:8] in shapy:
            variants["SHAPY"] = shapy[rid[:8]]
        for label, g in variants.items():
            err, mae = tcs.tape_error(g, tape)
            rows.setdefault(label, {})[rid[:8]] = {"girths_cm": g, "tape_error_cm": err, "mae_cm": mae}
            print(f"{rid[:8]} {label:16s} " + " ".join(f"{s} {g[s]:.1f}" for s in tape) +
                  f" | vs tape {', '.join(f'{s} {err[s]:+.1f}' for s in tape)} | MAE {mae:.1f}"
                  + (f" | belly offsets cm p50/p95 {np.percentile(belly_cm, 50):.2f}/{np.percentile(belly_cm, 95):.2f}"
                     if label == "+ belly contact" else ""), flush=True)
    base, new = rows["current default"], rows["+ belly contact"]
    abd_better = all(abs(new[r]["tape_error_cm"]["abdomen"]) < abs(base[r]["tape_error_cm"]["abdomen"]) for r in base)
    no_worse = all(abs(new[r]["tape_error_cm"][s]) <= abs(base[r]["tape_error_cm"][s]) + 0.5
                   for r in base for s in ("waist", "hip"))
    mae_b, mae_n = np.mean([v["mae_cm"] for v in base.values()]), np.mean([v["mae_cm"] for v in new.values()])
    verdict = abd_better and no_worse and mae_n < mae_b
    print(f"\nmean MAE: current {mae_b:.2f} cm | + belly contact {mae_n:.2f} cm"
          + (f" | SHAPY {np.mean([v['mae_cm'] for v in rows['SHAPY'].values()]):.2f} cm (on {len(rows['SHAPY'])} recordings)"
             if "SHAPY" in rows else ""))
    print(f"decision rule: abdomen better on all = {abd_better}, waist/hip not worse = {no_worse}, "
          f"MAE better = {mae_n < mae_b}  ->  {'ADOPT' if verdict else 'REJECT'}")
    Path(a.out).write_text(json.dumps({"tape": tape, "rows": rows, "adopt": bool(verdict)}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
