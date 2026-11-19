"""Multi-view reliability study: how much do girths move when the capture is degraded?

Each completed run saved everything its shape fit consumed (diagnostics/fit_inputs.npz). This script refits the
shape with the pipeline's CURRENT default settings (same FitConfig values as body_reconstruction._cfg_silhouette,
including the post-fit exclusion of views whose silhouette IoU stays low, followed by a refit) after applying a
controlled degradation to the saved observations, and reports how far every girth moves from the undegraded fit.

The degradations are image-space proxies for capture conditions; they act on what the fit consumes (subject
masks, 2D keypoints, views available, initial pose/camera), not on the raw video, so they isolate the fitting
stage. HMR2 and segmentation are NOT re-run on degraded pixels: e.g. "blur" models what a blurred frame does to
the mask outline and to keypoint precision, not what HMR2 would predict from a blurred image.

  fast rotation           fewer distinct views (24 -> 12 / 8 / 6, evenly spread in heading)
  blurry frames           horizontal mask smear (rotation motion blur) + keypoint noise in a share of views
  occlusion               a band of the mask removed and the keypoints inside it dropped, in a share of views
  missing coverage        every view within a heading sector removed (no back views / no side views)
  lighting                per-view random mask boundary shift (+-px), or a uniform shadow bleed / dark-edge loss
  not in place            per-view scale (distance) and lateral drift of the subject, camera init moved to match
  standing pose           initial per-view joint angles perturbed (initialisation robustness only)
  yaw error               initial per-view heading perturbed

Also reports a 4-fold jackknife over views (the same folds as the pipeline's cross-validation) for the baseline:
the spread of each girth across folds is the fit's own estimate of its view-sampling sensitivity.

Usage (CPU keeps clear of a busy GPU; --device cuda is faster):
  .venv\\Scripts\\python.exe scripts/reliability_study.py RUN_A RUN_B RUN_C --height-cm 176 \
      --output outputs/accuracy_study/reliability.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from backend.services import smpl_multiview as smv  # noqa: E402
from backend.services.subject_masks import silhouette_target  # noqa: E402
from measurement_accuracy_study import SITES, _subset, girths, load_run, load_smpl, silhouette_errors  # noqa: E402

RUNS = ROOT / "outputs" / "reconstructions"
BETA_SIGMA_FLOOR = 0.25
SIL_EXCLUDE_IOU = 0.75
COCO_HIPS_KNEES = [11, 12, 13, 14]


# ------------------------------------------------------------------ fitting (pipeline defaults)
def cfg_silhouette(beta_sigma):
    return smv.FitConfig(iterations=300, w_silhouette=1.0, w_beta_init=0.0, w_beta_prior=0.05,
                         sil_coverage_weight=1.0, sil_outside_form="charbonnier", fit_body_pose=True,
                         beta_sigma=beta_sigma, sil_tau_frac=0.01)


def fit_pipeline(smpl, faces, views, masks, betas_each, quality, device):
    """Stage A (pose/camera, shape frozen) + stage B (shared shape), then the pipeline's IoU-based view exclusion."""
    q = quality / quality.max()
    agg, std, _ = smv.robust_beta_aggregate(betas_each, q)
    sigma = np.maximum(std, BETA_SIGMA_FLOOR)

    def once():
        align = smv.fit_multiview(smpl, views, agg, smv.FitConfig(iterations=150), device, fit_shape=False)
        aligned = smv.FitViews(**{**views.__dict__, "global_orient": align.global_orient,
                                  "body_pose": align.body_pose, "cam_t": align.cam_t})
        return smv.fit_multiview(smpl, aligned, agg, cfg_silhouette(sigma), device)
    res = once()
    ious = per_view_iou(smpl, faces, res, views, masks, device)
    valid = [x for x in ious if x is not None]
    excluded = 0
    if valid:
        limit = max(SIL_EXCLUDE_IOU, float(np.median(valid)) - 0.10)
        for i, x in enumerate(ious):
            if x is not None and x < limit and views.sil_weight[i] > 0:
                views.sil_weight[i] = 0.0
                excluded += 1
        if excluded:
            res = once()
    return res, excluded


def per_view_iou(smpl, faces, res, views, masks, device):
    out = smv.posed_output(smpl, res.betas, res.global_orient, res.body_pose, device)
    verts = out.vertices.cpu().numpy().astype(np.float64)
    ious = []
    for i, m in enumerate(masks):
        if m is None:
            ious.append(None)
            continue
        uv = smv.project_np(verts[i] + res.cam_t[i], views.focal, views.center[i])
        pred = smv.rasterize_silhouette(uv, faces, m.shape).astype(bool)
        obs = m.astype(bool)
        ious.append(float((pred & obs).sum() / max((pred | obs).sum(), 1)))
    return ious


# ------------------------------------------------------------------ observation handling
def attach(views, masks):
    """(Re)derive silhouette targets from masks exactly as the pipeline does."""
    contours, dists, cws, wts = [], [], [], []
    for i, m in enumerate(masks):
        if m is None or m.sum() < 100:
            contours.append(None), dists.append(None), cws.append(None), wts.append(0.0)
            continue
        ys, xs = np.nonzero(m)
        pts, cw, dist = silhouette_target(m, [xs.min(), ys.min(), xs.max(), ys.max()], views.keypoints[i])
        contours.append(pts), dists.append(dist), cws.append(cw), wts.append(1.0)
    views.contours, views.dist_maps, views.contour_weights, views.sil_weight = contours, dists, cws, np.array(wts)
    return views


def copy_views(v):
    return smv.FitViews(keypoints=v.keypoints.copy(), weights=v.weights.copy(), global_orient=v.global_orient.copy(),
                        body_pose=v.body_pose.copy(), cam_t=v.cam_t.copy(), box_height=v.box_height.copy(),
                        center=v.center.copy(), focal=v.focal)


def pick_views(run, idx):
    idx = sorted(int(i) for i in idx)
    v = _subset(run["views"], idx)
    return v, [run["masks"][i] for i in idx], run["betas_each"][idx], run["quality"][idx], run["yaw"][idx]


def even_by_heading(yaw, n):
    order = np.argsort(yaw)
    return order[np.round(np.linspace(0, len(order), n, endpoint=False)).astype(int)]


def rot_y(deg):
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def scale_shift(mask, kps, s, dx, center):
    """Scale about the principal point by s and shift by dx px (subject nearer/further and off to one side)."""
    cx, cy = center
    M = np.array([[s, 0, (1 - s) * cx + dx], [0, s, (1 - s) * cy]], dtype=np.float64)
    h, w = mask.shape
    out = cv2.warpAffine(mask.astype(np.uint8), M, (w, h), flags=cv2.INTER_NEAREST)
    k = kps.copy()
    k[:, 0] = s * (kps[:, 0] - cx) + cx + dx
    k[:, 1] = s * (kps[:, 1] - cy) + cy
    return out, k


# ------------------------------------------------------------------ conditions
def build(run, cond, rng):
    """Return (views, masks, betas_each, quality, description) for one degradation."""
    n = len(run["masks"])
    allidx = np.arange(n)
    kind, arg = cond
    if kind == "views":
        v, masks, be, q, _ = pick_views(run, even_by_heading(run["yaw"], arg))
        return attach(v, masks), masks, be, q
    if kind == "drop_sector":  # heading sector (deg) with no camera coverage
        lo, hi = arg
        yaw = np.array([smv.wrap_deg(y) for y in run["yaw"]])
        keep = allidx[~((yaw >= lo) & (yaw <= hi))] if lo <= hi else allidx[~((yaw >= lo) | (yaw <= hi))]
        v, masks, be, q, _ = pick_views(run, keep)
        return attach(v, masks), masks, be, q
    v, masks, be, q, _ = pick_views(run, allidx)
    masks = [None if m is None else m.copy() for m in masks]
    hit = rng.choice(n, size=max(1, int(round(n * (arg.get("share", 1.0) if isinstance(arg, dict) else 1.0)))),
                     replace=False)
    if kind == "blur":
        k = int(arg["smear_px"])
        ker = np.ones((1, 2 * k + 1), np.uint8)  # horizontal: the body turns about a vertical axis
        for i in hit:
            if masks[i] is not None:
                masks[i] = cv2.dilate(masks[i].astype(np.uint8), ker)
            v.keypoints[i] += rng.normal(0, arg["kp_sigma_frac"] * v.box_height[i], v.keypoints[i].shape)
            v.weights[i] *= 0.6  # a blurred detection is reported with lower confidence
    elif kind == "occlude":  # a band (fraction of box height, top..bottom) hidden behind an object
        top, bot = arg["band"]
        for i in hit:
            m = masks[i]
            if m is None:
                continue
            ys = np.nonzero(m.any(1))[0]
            y0, y1 = ys.min(), ys.max()
            a, b = int(y0 + top * (y1 - y0)), int(y0 + bot * (y1 - y0))
            xs = np.nonzero(m.any(0))[0]
            xm = int(xs.mean())
            half = (xs.max() - xs.min()) // 2 + 5
            side = rng.choice([-1, 1])  # the occluder covers one half of the body
            x0, x1 = (xm - half, xm) if side < 0 else (xm, xm + half)
            m[a:b, max(x0, 0):x1] = 0
            inside = (v.keypoints[i][:, 1] >= a) & (v.keypoints[i][:, 1] <= b) & \
                     (v.keypoints[i][:, 0] >= x0) & (v.keypoints[i][:, 0] <= x1)
            v.weights[i][inside] = 0.0
    elif kind == "boundary":  # segmentation boundary moved by lighting
        for i in hit:
            if masks[i] is None:
                continue
            r = int(arg["px"]) if "px" in arg else int(rng.integers(-arg["max_px"], arg["max_px"] + 1))
            if r == 0:
                continue
            ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * abs(r) + 1, 2 * abs(r) + 1))
            masks[i] = (cv2.dilate if r > 0 else cv2.erode)(masks[i].astype(np.uint8), ker)
    elif kind == "drift":  # not rotating in place: distance and lateral position change across the turn
        order = np.argsort(run["yaw"])
        phase = np.empty(n)
        phase[order] = np.linspace(0, 2 * np.pi, n, endpoint=False)
        for i in range(n):
            s = 1.0 + arg["scale"] * np.sin(phase[i])
            dx = arg["shift_frac"] * v.box_height[i] * np.cos(phase[i] + 0.7)
            if masks[i] is not None:
                masks[i], v.keypoints[i] = scale_shift(masks[i], v.keypoints[i], s, dx, v.center[i])
            else:
                _, v.keypoints[i] = scale_shift(np.zeros((2, 2), np.uint8), v.keypoints[i], s, dx, v.center[i])
            z = v.cam_t[i, 2] / s
            v.cam_t[i, 0] = v.cam_t[i, 0] / s + dx * z / v.focal
            v.cam_t[i, 1] = v.cam_t[i, 1] / s
            v.cam_t[i, 2] = z
            v.box_height[i] *= s
    elif kind == "pose_init":
        for i in hit:
            for j in range(v.body_pose.shape[1]):
                axis = rng.normal(size=3)
                axis /= np.linalg.norm(axis)
                ang = np.deg2rad(rng.normal(0, arg["deg"]))
                K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
                R = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K
                v.body_pose[i, j] = R @ v.body_pose[i, j]
    elif kind == "yaw_init":
        for i in hit:
            v.global_orient[i] = rot_y(rng.normal(0, arg["deg"])) @ v.global_orient[i]
    else:
        raise ValueError(kind)
    return attach(v, masks), masks, be, q


CONDITIONS = {
    "baseline": ("none", None),
    "views_16": ("views", 16), "views_12": ("views", 12), "views_8": ("views", 8), "views_6": ("views", 6),
    "blur_25pct": ("blur", {"share": 0.25, "smear_px": 12, "kp_sigma_frac": 0.015}),
    "blur_50pct": ("blur", {"share": 0.5, "smear_px": 12, "kp_sigma_frac": 0.015}),
    "occlude_hip_25pct": ("occlude", {"share": 0.25, "band": (0.45, 0.65)}),
    "occlude_chest_25pct": ("occlude", {"share": 0.25, "band": (0.18, 0.38)}),
    "no_back_views": ("drop_sector", (135, -135)),
    "no_left_side_views": ("drop_sector", (45, 135)),
    "lighting_random_6px": ("boundary", {"share": 1.0, "max_px": 6}),
    "lighting_shadow_bleed_4px": ("boundary", {"share": 1.0, "px": 4}),
    "lighting_dark_edge_4px": ("boundary", {"share": 1.0, "px": -4}),
    "drift_10pct": ("drift", {"scale": 0.10, "shift_frac": 0.15}),
    "pose_init_10deg": ("pose_init", {"share": 1.0, "deg": 10}),
    "yaw_init_15deg": ("yaw_init", {"share": 1.0, "deg": 15}),
}
STOCHASTIC = {"blur", "occlude", "boundary", "pose_init", "yaw_init"}


def jackknife(smpl, faces, run, device, k=4):
    """Girths from shapes fitted on 3 of 4 heading-interleaved folds (the pipeline's CV folds)."""
    order = np.argsort(run["yaw"])
    rows = []
    for f in range(k):
        train = sorted(set(range(len(run["masks"]))) - set(order[f::k].tolist()))
        v, masks, be, q, _ = pick_views(run, train)
        res, _ = fit_pipeline(smpl, faces, attach(v, masks), masks, be, q, device)
        rows.append(girths(smpl, res.betas, device))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--seeds", type=int, default=2, help="repeats of each random degradation")
    ap.add_argument("--height-cm", type=float, default=176.0, help="only to express changes in cm")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--no-jackknife", action="store_true")
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.threads)
    smpl = load_smpl(a.device)
    faces = np.asarray(smpl.faces, dtype=np.int64)
    report = {"created_at": time.strftime("%Y-%m-%d %H:%M:%S"), "device": a.device, "height_cm": a.height_cm,
              "fit": "pipeline defaults: charbonnier outside term, population prior 0.05, HMR2 prior 0, per-view pose, "
                     "post-fit IoU exclusion + refit", "conditions": {k: str(CONDITIONS[k]) for k in a.conditions},
              "runs": {}}
    if a.output.is_file():  # resume
        report = json.loads(a.output.read_text(encoding="utf-8"))
    for run_id in a.runs:
        d = next(RUNS.glob(run_id + "*"))
        run = load_run(d.name)
        R = report["runs"].setdefault(d.name[:8], {"views": len(run["masks"]), "conditions": {}})
        for name in a.conditions:
            kind, arg = CONDITIONS[name]
            seeds = range(a.seeds) if kind in STOCHASTIC else range(1)
            for seed in seeds:
                key = f"{name}#{seed}"
                if key in R["conditions"]:
                    continue
                t0 = time.time()
                rng = np.random.default_rng(1000 + seed)
                if kind == "none":
                    v, masks, be, q, _ = pick_views(run, np.arange(len(run["masks"])))
                    v = attach(v, masks)
                else:
                    v, masks, be, q = build(run, (kind, arg), rng)
                res, excl = fit_pipeline(smpl, faces, v, masks, be, q, a.device)
                sil = silhouette_errors(smpl, faces, res, v, masks, a.device)
                R["conditions"][key] = {"views_used": len(masks), "views_excluded_after_fit": excl,
                                        "girths_x_height": girths(smpl, res.betas, a.device),
                                        "betas": res.betas.tolist(), "max_abs_beta_sd": float(np.abs(res.betas).max()),
                                        "silhouette_vs_degraded_masks": sil, "seconds": round(time.time() - t0, 1)}
                base = R["conditions"].get("baseline#0", {}).get("girths_x_height")
                msg = ""
                if base:
                    g = R["conditions"][key]["girths_x_height"]
                    dev = {s: 100 * (g[s] - base[s]) / base[s] for s in SITES if g.get(s) and base.get(s)}
                    msg = " ".join(f"{s[:5]}{dev[s]:+.1f}" for s in ("chest", "waist", "hip", "thigh_left") if s in dev)
                print(d.name[:8], f"{key:28s}", f"{time.time() - t0:5.0f}s prot={sil['protrusion_frac']:.3f}", msg,
                      flush=True)
                a.output.parent.mkdir(parents=True, exist_ok=True)
                a.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        if not a.no_jackknife and "jackknife" not in R:
            R["jackknife"] = jackknife(smpl, faces, run, a.device)
            a.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(d.name[:8], "jackknife done", flush=True)
    print("wrote", a.output)


if __name__ == "__main__":
    main()
