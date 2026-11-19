"""Transfer SHAPY's sex-specific body-shape priors (CAESAR scans, SMPL-X neutral space) into SMPL-neutral space.

The pipeline fits SMPL-neutral betas. Its only shape prior was "how unusual is this body for a mixed male/female
population" (N(0, I)), and in the neutral model's coupled shape space extra volume tends to go to the female
pattern (breasts, hips). SHAPY ships the distribution of real male and female bodies in SMPL-X-neutral betas
(models/shapy/utility_files/shape_priors/{male,female}_normal.npz). This script:

1. samples bodies from that distribution (first 10 coefficients, marginal Gaussian) and builds each SMPL-X rest mesh
   from the SMPL-X model stored in the SHAPY checkpoint;
2. fits SMPL-neutral betas (+ translation) to each mesh by symmetric surface (chamfer) distance on the GPU;
3. saves mean / covariance of the fitted SMPL betas, with fit errors and a girth cross-check, to
   data/shape_priors/smpl_neutral_{male,female}_prior.npz.

    .venv\\Scripts\\python.exe scripts/build_sex_shape_priors.py [--samples 400]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import measurement_accuracy_study as mas  # noqa: E402
from backend.services import body_measurements as bm  # noqa: E402
from backend.services import shapy_shape  # noqa: E402
from backend.services import smpl_multiview as smv  # noqa: E402

OUT = ROOT / "data" / "shape_priors"
PRIORS = ROOT / "models" / "shapy" / "utility_files" / "shape_priors"


def rest(template, shapedirs, betas):
    return template[None] + torch.einsum("vck,nk->nvc", shapedirs[:, :, :betas.shape[1]], betas)


def chamfer(a, b):
    d = torch.cdist(a, b)
    return d.min(2).values.mean(1) + d.min(1).values.mean(1)


def fit_smpl_to(targets, smpl_t, smpl_s, iters=300, n_pts=3000, seed=0):
    """targets [N,Vx,3] -> SMPL betas [N,10], translations, per-sample RMS-like chamfer (m)."""
    g = torch.Generator(device=targets.device).manual_seed(seed)
    N = len(targets)
    beta = torch.zeros(N, 10, device=targets.device, requires_grad=True)
    tr = (targets.mean(1) - smpl_t.mean(0)).detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([beta, tr], lr=0.05)
    for it in range(iters):
        ia = torch.randint(0, smpl_t.shape[0], (n_pts,), generator=g, device=targets.device)
        ib = torch.randint(0, targets.shape[1], (n_pts,), generator=g, device=targets.device)
        v = rest(smpl_t, smpl_s, beta)[:, ia] + tr[:, None]
        loss = chamfer(v, targets[:, ib]).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it == int(iters * 0.7):
            for grp in opt.param_groups:
                grp["lr"] = 0.01
    with torch.no_grad():
        v = rest(smpl_t, smpl_s, beta) + tr[:, None]
        err = chamfer(v[:, ::2], targets[:, ::3]) / 2
    return beta.detach(), tr.detach(), err


def girth(v, faces, joints, site):
    m = bm.measure_body(v, faces, joints)["measurements"][site]
    return m["tape_circumference_over_height"] if m.get("valid") else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=400)
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(shapy_shape.CKPT, map_location="cpu", weights_only=True)["model"]
    sx = shapy_shape.SmplxFromCheckpoint(state, dev)
    smpl = mas.load_smpl(dev)
    smpl_t = smpl.v_template.float()
    smpl_s = smpl.shapedirs.float()[:, :, :10]
    faces = np.asarray(smpl.faces, dtype=np.int64)
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    for sex in ("male", "female"):
        z = np.load(PRIORS / f"{sex}_normal.npz")
        mu, cov = z["mean"][:10], z["covariance"][:10, :10]
        rng = np.random.default_rng(0)
        bx = np.vstack([mu[None], rng.multivariate_normal(mu, cov, size=a.samples - 1)])
        betas, errs = [], []
        for k in range(0, len(bx), a.batch):
            b = torch.as_tensor(bx[k:k + a.batch], dtype=torch.float32, device=dev)
            tgt = rest(sx.v_template, sx.shapedirs, b)
            bs, _, e = fit_smpl_to(tgt, smpl_t, smpl_s, seed=k)
            betas.append(bs.cpu().numpy()), errs.append(e.cpu().numpy())
            print(f"  {sex}: {min(k + a.batch, len(bx))}/{len(bx)}  surface error median {np.median(errs[-1]) * 1000:.1f} mm",
                  flush=True)
        B = np.concatenate(betas).astype(np.float64)
        E = np.concatenate(errs)
        # Girth cross-check on 8 samples: SMPL-X sample vs its SMPL fit, same A-pose and measurement code.
        checks = []
        for i in range(8):
            vx, jx = sx.posed(bx[i], smv.measurement_pose()[:21])
            vs, js, _, _ = smv.canonical_mesh(smpl, B[i], np.tile(np.eye(3), (1, 23, 1, 1)), None, dev, pose="measurement")
            j24 = smv.smpl_joints24(smpl, vs)
            row = {}
            for site in ("chest", "waist", "hip", "thigh_left"):
                gx, gs = girth(vx, sx.faces, jx[:24], site), girth(vs, faces, j24, site)
                row[site] = None if gx is None or gs is None else float(gs - gx)
            checks.append(row)
        diffs = {s: [c[s] for c in checks if c[s] is not None] for s in ("chest", "waist", "hip", "thigh_left")}
        rec = {"mean": B.mean(0), "covariance": np.cov(B, rowvar=False), "samples": len(B),
               "surface_error_median_m": float(np.median(E)), "surface_error_p95_m": float(np.percentile(E, 95)),
               "source": f"SHAPY {sex}_normal.npz (CAESAR, SMPL-X neutral betas[:10]) transferred by chamfer fit"}
        np.savez(OUT / f"smpl_neutral_{sex}_prior.npz", **rec)
        report[sex] = {"samples": len(B), "mean": np.round(B.mean(0), 3).tolist(),
                       "sd": np.round(B.std(0), 3).tolist(),
                       "surface_error_mm_median_p95": [round(float(np.median(E)) * 1000, 1),
                                                       round(float(np.percentile(E, 95)) * 1000, 1)],
                       "girth_over_height_diff_smpl_minus_smplx_mean_abs": {
                           s: (round(float(np.mean(np.abs(v))), 4) if v else None) for s, v in diffs.items()}}
        print(json.dumps(report[sex], indent=1), flush=True)
    (OUT / "smpl_neutral_sex_priors_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
