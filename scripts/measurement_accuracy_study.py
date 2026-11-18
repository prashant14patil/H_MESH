"""Offline accuracy study: refit saved reconstructions with different silhouette settings.

Every completed run saves everything the shape fit consumes (diagnostics/fit_inputs.npz), so the
fit can be repeated here without re-running video decoding, HMR2 or segmentation. For each run and
each setting this reports:

* girths as a fraction of mesh height (and in cm when --height-cm is given);
* how far the fitted body PROTRUDES outside the observed (clothed) silhouette - physically the body
  must lie inside its clothing, so protrusion is a measurable error even without tape measurements;
* how much of the clothed outline the body leaves uncovered (expected > 0 under loose clothing);
* the largest shape coefficient in population SD;
* optionally (--cv) the 4-fold held-out silhouette IoU.

With two or more runs of independent recordings of the same person (--same-subject), the spread
between recordings is the recording-to-recording repeatability of each setting.

Run:
  .venv\\Scripts\\python.exe scripts/measurement_accuracy_study.py RUN_ID [RUN_ID ...] \
      --same-subject --height-cm 174 --output outputs/accuracy_study/study.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.services import body_measurements as bm  # noqa: E402
from backend.services import smpl_multiview as smv  # noqa: E402
from backend.services.subject_masks import silhouette_target  # noqa: E402

RUNS = ROOT / "outputs" / "reconstructions"
BETA_SIGMA_FLOOR = 0.25
SITES = ["chest", "waist", "abdomen", "hip", "upper_arm_left", "upper_arm_right", "forearm_left", "forearm_right",
         "thigh_left", "thigh_right", "calf_left", "calf_right"]


def load_smpl(device):
    """HMR2's SMPL wrapper (same joint map as the live pipeline), loaded without HMR2's renderer imports."""
    cache = Path(os.environ.get("USERPROFILE", str(Path.home()))) / ".cache" / "4DHumans" / "data"
    spec = importlib.util.spec_from_file_location(
        "smpl_wrapper", ROOT / "models" / "4D-Humans" / "hmr2" / "models" / "smpl_wrapper.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    smpl = mod.SMPL(model_path=str(cache / "smpl"), gender="neutral", num_body_joints=23,
                    joint_regressor_extra=str(cache / "SMPL_to_J19.pkl"))
    return smpl.to(device).eval()


def load_run(run_id: str):
    d = RUNS / run_id
    z = np.load(d / "diagnostics" / "fit_inputs.npz")
    shape = tuple(int(x) for x in z["mask_shape"])
    masks = [np.unpackbits(z["masks_packed"][i])[:shape[0] * shape[1]].reshape(shape) if z["mask_usable"][i] else None
             for i in range(len(z["selected"]))]
    views = smv.FitViews(keypoints=z["keypoints"], weights=z["weights"], global_orient=z["global_orient"],
                         body_pose=z["body_pose"], cam_t=z["cam_t"], box_height=z["box_height"],
                         center=z["center"], focal=float(z["focal"]))
    contours, dists, cws, wts = [], [], [], []
    for i, m in enumerate(masks):
        if m is None:
            contours.append(None), dists.append(None), cws.append(None), wts.append(0.0)
            continue
        ys, xs = np.nonzero(m)
        box = [xs.min(), ys.min(), xs.max(), ys.max()]  # detector box is not saved; the mask extent is equivalent
        pts, cw, dist = silhouette_target(m, box, z["keypoints"][i])
        contours.append(pts), dists.append(dist), cws.append(cw), wts.append(1.0)
    views.contours, views.dist_maps, views.contour_weights, views.sil_weight = contours, dists, cws, np.array(wts)
    saved = dict(np.load(d / "diagnostics" / "fit_result.npz"))
    if "sil_weight" in saved and len(saved["sil_weight"]) == len(wts):
        views.sil_weight = np.minimum(views.sil_weight, saved["sil_weight"])  # same post-fit exclusions as the run
    meas = json.loads((d / "measurements.json").read_text(encoding="utf-8"))
    return {"run_id": run_id, "views": views, "masks": masks, "betas_each": z["betas_each"], "quality": z["quality"],
            "yaw": z["yaw"], "saved_betas": saved["betas"], "saved_measurements": meas}


def fit(smpl, views, betas_init, beta_sigma, coverage, device, iterations=300, beta_prior=0.0):
    align = smv.fit_multiview(smpl, views, betas_init, smv.FitConfig(iterations=150), device, fit_shape=False)
    aligned = smv.FitViews(**{**views.__dict__, "global_orient": align.global_orient, "body_pose": align.body_pose,
                              "cam_t": align.cam_t})
    cfg = smv.FitConfig(iterations=iterations, w_silhouette=1.0, w_beta_init=beta_prior, beta_sigma=beta_sigma,
                        sil_tau_frac=0.01, sil_coverage_weight=coverage)
    return smv.fit_multiview(smpl, aligned, betas_init, cfg, device)


def silhouette_errors(smpl, faces, result, views, masks, device):
    out = smv.posed_output(smpl, result.betas, result.global_orient, result.body_pose, device)
    verts = out.vertices.cpu().numpy().astype(np.float64)
    rows = []
    for i, m in enumerate(masks):
        if m is None:
            continue
        uv = smv.project_np(verts[i] + result.cam_t[i], views.focal, views.center[i])
        pred = smv.rasterize_silhouette(uv, faces, m.shape).astype(bool)
        obs = m.astype(bool)
        rows.append({"iou": float((pred & obs).sum() / max((pred | obs).sum(), 1)),
                     "protrusion_frac": float((pred & ~obs).sum() / max(pred.sum(), 1)),
                     "uncovered_frac": float((obs & ~pred).sum() / max(obs.sum(), 1))})
    return {k: float(np.mean([r[k] for r in rows])) for k in ("iou", "protrusion_frac", "uncovered_frac")}


def girths(smpl, betas, device):
    v, _, _, _ = smv.canonical_mesh(smpl, betas, np.tile(np.eye(3), (1, 23, 1, 1)), None, device, pose="measurement")
    faces = np.asarray(smpl.faces, dtype=np.int64)
    m = bm.measure_body(v, faces, smv.smpl_joints24(smpl, v))
    return {k: (x["tape_circumference_over_height"] if x["valid"] else None) for k, x in m["measurements"].items()}


def held_out_iou(smpl, faces, run, betas_init_fn, coverage, device, k=4, beta_prior=0.0):
    views, masks = run["views"], run["masks"]
    order = np.argsort(run["yaw"])
    ious = []
    for f in range(k):
        held = sorted(order[f::k].tolist())
        train = sorted(set(range(len(masks))) - set(held))
        tv = _subset(views, train)
        b0, sig = betas_init_fn(train)
        shape = fit(smpl, tv, b0, sig, coverage, device, beta_prior=beta_prior)
        hv = _subset(views, held)
        hv.contours = None
        hfit = smv.fit_multiview(smpl, hv, shape.betas, smv.FitConfig(iterations=150), device, fit_shape=False)
        ious.append(silhouette_errors(smpl, faces, hfit, hv, [masks[i] for i in held], device)["iou"])
    return float(np.mean(ious))


def _subset(views, idx):
    def pick(seq):
        return None if seq is None else [seq[i] for i in idx]
    return smv.FitViews(keypoints=views.keypoints[idx], weights=views.weights[idx],
                        global_orient=views.global_orient[idx], body_pose=views.body_pose[idx],
                        cam_t=views.cam_t[idx], box_height=views.box_height[idx], center=views.center[idx],
                        focal=views.focal, contours=pick(views.contours), dist_maps=pick(views.dist_maps),
                        contour_weights=pick(views.contour_weights),
                        sil_weight=None if views.sil_weight is None else views.sil_weight[idx])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--coverage", type=float, nargs="+", default=[1.0, 0.5, 0.25, 0.1, 0.0])
    ap.add_argument("--beta-prior", type=float, nargs="+", default=[0.0],
                    help="weights of the Mahalanobis pull toward the multi-frame HMR2 shape")
    ap.add_argument("--same-subject", action="store_true", help="runs are independent recordings of one person")
    ap.add_argument("--height-cm", type=float, default=None)
    ap.add_argument("--cv", action="store_true", help="also compute 4-fold held-out silhouette IoU (slow)")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    smpl = load_smpl(device)
    faces = np.asarray(smpl.faces, dtype=np.int64)
    report = {"created_at": time.strftime("%Y-%m-%d %H:%M:%S"), "device": device, "runs": {}, "coverage": args.coverage, "beta_prior": args.beta_prior}
    for run_id in args.runs:
        run = load_run(run_id)
        q = run["quality"] / run["quality"].max()
        agg, std, _ = smv.robust_beta_aggregate(run["betas_each"], q)
        sigma = np.maximum(std, BETA_SIGMA_FLOOR)

        def init_fn(idx, run=run, q=q):
            a, s, _ = smv.robust_beta_aggregate(run["betas_each"][idx], q[idx])
            return a, np.maximum(s, BETA_SIGMA_FLOOR)
        rows = {}
        for cov, prior in [(c, b) for b in args.beta_prior for c in args.coverage]:
            key = f"{cov}|{prior}"
            t0 = time.time()
            res = fit(smpl, run["views"], agg, sigma, cov, device, beta_prior=prior)
            row = {"coverage_weight": cov, "beta_prior_weight": prior, "betas": res.betas.tolist(),
                   "max_abs_beta_sd": float(np.abs(res.betas).max()),
                   "keypoint_error_pct_height": float(np.median(res.residual_px_after / run["views"].box_height[:, None]) * 100),
                   "silhouette": silhouette_errors(smpl, faces, res, run["views"], run["masks"], device),
                   "girths_x_height": girths(smpl, res.betas, device)}
            if cov == 1.0 and prior == 0.0:
                row["reproduces_saved_fit_max_beta_diff"] = float(np.abs(res.betas - run["saved_betas"]).max())
            if args.cv:
                row["held_out_iou"] = held_out_iou(smpl, faces, run, init_fn, cov, device, beta_prior=prior)
            row["seconds"] = round(time.time() - t0, 1)
            rows[key] = row
            print(run_id[:8], f"cov={cov:<5} prior={prior:<5}", f"protrude={row['silhouette']['protrusion_frac']:.3f}",
                  f"uncov={row['silhouette']['uncovered_frac']:.3f}", f"iou={row['silhouette']['iou']:.3f}",
                  f"maxbeta={row['max_abs_beta_sd']:.2f}",
                  " ".join(f"{k}={row['girths_x_height'][k]:.3f}" for k in ("chest", "waist", "hip", "thigh_left")
                           if row["girths_x_height"].get(k)), flush=True)
        report["runs"][run_id] = {"hmr2_aggregate_girths_x_height": girths(smpl, agg, device), "settings": rows}
    keys = list(next(iter(report["runs"].values()))["settings"])

    if args.same_subject and len(args.runs) >= 2:
        rep = {}
        for key in keys:
            per_site = {}
            for s in SITES:
                vals = [report["runs"][r]["settings"][key]["girths_x_height"].get(s) for r in args.runs]
                vals = [v for v in vals if v is not None]
                if len(vals) >= 2:
                    mean = float(np.mean(vals))
                    per_site[s] = {"mean_x_height": mean, "sd_x_height": float(np.std(vals, ddof=1)),
                                   "cv_pct": float(np.std(vals, ddof=1) / mean * 100),
                                   "range_x_height": float(max(vals) - min(vals))}
                    if args.height_cm:
                        per_site[s]["range_cm"] = per_site[s]["range_x_height"] * args.height_cm
            rep[key] = {"per_site": per_site,
                             "mean_cv_pct": float(np.mean([v["cv_pct"] for v in per_site.values()]))}
        report["between_recordings"] = rep
        for key, r in rep.items():
            print(f"between recordings cov|prior={key:<10} mean CV {r['mean_cv_pct']:.2f}%")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("wrote", args.output)


if __name__ == "__main__":
    main()
