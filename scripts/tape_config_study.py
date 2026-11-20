"""Which shape-fit configuration matches REAL tape? (The only valid selection criterion under loose clothing.)

Held-out silhouette IoU is scored against the clothed outline and rewards filling loose garments, so it picked the
worst configuration once (prior 0: 10-14 cm too large vs tape). This study refits saved runs of ONE subject under
several configurations and compares girths (stature-scaled) with the subject's tape ranges.

    .venv\\Scripts\\python.exe scripts/tape_config_study.py RUN [RUN ...] --height-cm 179 --sex male \
        --tape waist=88-90 abdomen=95-100 hip=98-103 [--detail CONFIG,...]

Caveat: one subject, self-measured ranges. It ranks configurations; it does not establish population accuracy.
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
from backend.services import body_measurements as bm  # noqa: E402
from backend.services import smpl_multiview as smv  # noqa: E402
from backend.services.body_reconstruction import stature_height  # noqa: E402

SITES = ("chest", "waist", "abdomen", "hip", "thigh_left", "calf_left", "upper_arm_left")
# name -> (HMR2 Mahalanobis prior weight, population prior weight, use male prior, coverage weight, silhouette on)
CONFIGS = {
    "D0": (0.0, 0.05, True, 1.0, True),
    "P02": (0.2, 0.05, False, 1.0, True),
    "P02m": (0.2, 0.05, True, 1.0, True),
    "IN": (0.0, 0.05, True, 0.0, True),
    "IN_P02": (0.2, 0.05, True, 0.0, True),
    "C025": (0.0, 0.05, True, 0.25, True),
    "HMR2": (None, None, False, None, False),
}


def fit(smpl, views, agg, sig, name, device):
    w_init, w_pop, male, cov, sil = CONFIGS[name]
    align = smv.fit_multiview(smpl, views, agg, smv.FitConfig(iterations=150), device, fit_shape=False)
    if not sil:
        return align, agg
    aligned = smv.FitViews(**{**views.__dict__, "global_orient": align.global_orient, "body_pose": align.body_pose,
                              "cam_t": align.cam_t})
    mean, prec = smv.sex_shape_prior("male") if male else (None, None)
    cfg = smv.FitConfig(iterations=300, w_silhouette=1.0, w_beta_init=w_init, w_beta_prior=w_pop, beta_sigma=sig,
                        sil_tau_frac=0.01, sil_coverage_weight=cov, beta_pop_mean=mean, beta_pop_precision=prec)
    res = smv.fit_multiview(smpl, aligned, agg, cfg, device)
    return res, res.betas


def girths(smpl, betas, height_cm, device, offsets=None):
    if offsets is None:
        v, _, _, _ = smv.canonical_mesh(smpl, betas, np.tile(np.eye(3), (1, 23, 1, 1)), None, device, pose="measurement")
    else:
        v, _ = smv.canonical_detailed_mesh(smpl, betas, offsets, device)
    m = bm.measure_body(v, np.asarray(smpl.faces, dtype=np.int64), smv.smpl_joints24(smpl, v))["measurements"]
    k = height_cm / stature_height(smpl, betas, device)
    return {s: (round(m[s]["tape_circumference"] * k, 1) if m.get(s, {}).get("valid") else None) for s in SITES}


def tape_error(g, tape):
    out = {}
    for s, (lo, hi) in tape.items():
        v = g.get(s)
        out[s] = None if v is None else (v - hi if v > hi else (v - lo if v < lo else 0.0))
    vals = [abs(x) for x in out.values() if x is not None]
    return out, (float(np.mean(vals)) if vals else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--height-cm", type=float, required=True)
    ap.add_argument("--sex", default="male")
    ap.add_argument("--tape", nargs="+", required=True, help="site=lo-hi (cm)")
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--detail", default="", help="configs to also evaluate with trunk surface detail")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "accuracy_study" / "tape_config_study.json"))
    a = ap.parse_args()
    tape = {}
    for t in a.tape:
        site, rng = t.split("=")
        lo, hi = (float(x) for x in rng.split("-"))
        tape[site] = (lo, hi)
    smpl = mas.load_smpl(a.device)
    detail = set(filter(None, a.detail.split(",")))
    rows = {}
    for rid in a.runs:
        run = mas.load_run(rid)
        agg, std, _ = smv.robust_beta_aggregate(run["betas_each"], run["quality"] / run["quality"].max())
        sig = np.maximum(std, 0.25)
        for name in a.configs.split(","):
            res, betas = fit(smpl, run["views"], agg, sig, name, a.device)
            variants = [(name, None)]
            if name in detail:
                fr = smv.FitResult(betas=betas, global_orient=res.global_orient, body_pose=res.body_pose, cam_t=res.cam_t)
                off = smv.offset_vectors(smv.fit_surface_offsets(smpl, run["views"], fr, smv.SurfaceConfig(), a.device))
                variants.append((name + "+detail", off))
            for label, off in variants:
                g = girths(smpl, betas, a.height_cm, a.device, off)
                err, mae = tape_error(g, tape)
                rows.setdefault(label, {})[rid[:8]] = {"girths_cm": g, "tape_error_cm": err, "mae_cm": mae}
                print(f"{rid[:8]} {label:14s} " + " ".join(f"{s} {g[s]}" for s in tape) +
                      f" | vs tape {', '.join(f'{s} {err[s]:+.1f}' for s in tape)} | MAE {mae:.1f} cm", flush=True)
    summary = {}
    for label, per in rows.items():
        maes = [v["mae_cm"] for v in per.values()]
        site_mean = {s: float(np.mean([v["girths_cm"][s] for v in per.values() if v["girths_cm"][s] is not None]))
                     for s in SITES}
        summary[label] = {"mae_cm_mean": float(np.mean(maes)), "mae_cm_per_recording": maes, "girths_cm_mean": site_mean}
    print("\nconfig          mean MAE vs tape   mean girths (" + ", ".join(tape) + ")")
    for label, s in sorted(summary.items(), key=lambda kv: kv[1]["mae_cm_mean"]):
        print(f"{label:16s} {s['mae_cm_mean']:6.1f} cm        " + " / ".join(f"{s['girths_cm_mean'][x]:.1f}" for x in tape))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"tape": tape, "height_cm": a.height_cm, "rows": rows, "summary": summary},
                                      indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
