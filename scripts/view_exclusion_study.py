r"""Which post-fit view-exclusion rule protects girths when some masks are missing body (occlusion, dark edges)?

Rules (applied after a first full fit, then the shape is refitted without the excluded views' silhouettes):
  iou         current pipeline: IoU < max(0.75, median IoU - 0.10)
  protrusion  per-view share of the rendered body OUTSIDE its mask > max(PROT_FLOOR, median + K x MAD). Clothing
              only ever enlarges the outline, so a body sticking out of a mask means the mask is missing body.
  none        no exclusion

For the occlusion conditions the truly degraded views are known, so each rule's precision / recall is reported.
Device-consistent: the undegraded baseline is refitted on the same device as everything else here.

Usage: .venv\\Scripts\\python.exe scripts/view_exclusion_study.py RUN [RUN ...] --device cuda --output out.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from backend.services import smpl_multiview as smv  # noqa: E402
from measurement_accuracy_study import girths, load_run, load_smpl  # noqa: E402
import reliability_study as rs  # noqa: E402

RUNS = ROOT / "outputs" / "reconstructions"
PROT_FLOOR, PROT_K = 0.04, 3.0


def per_view(smpl, faces, res, views, masks, device):
    out = smv.posed_output(smpl, res.betas, res.global_orient, res.body_pose, device)
    verts = out.vertices.cpu().numpy().astype(np.float64)
    rows = []
    for i, m in enumerate(masks):
        if m is None:
            rows.append(None)
            continue
        uv = smv.project_np(verts[i] + res.cam_t[i], views.focal, views.center[i])
        p = smv.rasterize_silhouette(uv, faces, m.shape).astype(bool)
        o = m.astype(bool)
        rows.append({"iou": float((p & o).sum() / max((p | o).sum(), 1)),
                     "protrusion": float((p & ~o).sum() / max(p.sum(), 1))})
    return rows


def exclusion(rows, rule, sil_weight):
    idx = [i for i, r in enumerate(rows) if r is not None and sil_weight[i] > 0]
    if rule == "none" or not idx:
        return []
    if rule == "iou":
        v = np.array([rows[i]["iou"] for i in idx])
        lim = max(rs.SIL_EXCLUDE_IOU, float(np.median(v)) - 0.10)
        return [i for i in idx if rows[i]["iou"] < lim]
    v = np.array([rows[i]["protrusion"] for i in idx])
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med))) * 1.4826
    lim = max(PROT_FLOOR, med + PROT_K * mad)
    return [i for i in idx if rows[i]["protrusion"] > lim]


def fit_with_rule(smpl, faces, views, masks, be, q, device, rule):
    agg, std, _ = smv.robust_beta_aggregate(be, q / q.max())
    sigma = np.maximum(std, rs.BETA_SIGMA_FLOOR)

    def once():
        align = smv.fit_multiview(smpl, views, agg, smv.FitConfig(iterations=150), device, fit_shape=False)
        aligned = smv.FitViews(**{**views.__dict__, "global_orient": align.global_orient,
                                  "body_pose": align.body_pose, "cam_t": align.cam_t})
        return smv.fit_multiview(smpl, aligned, agg, rs.cfg_silhouette(sigma), device)
    sw0 = views.sil_weight.copy()
    res = once()
    rows = first = per_view(smpl, faces, res, views, masks, device)
    base_rule = "protrusion" if rule == "protrusion_iter" else rule
    rounds = 3 if rule == "protrusion_iter" else 1
    excl = []
    for _ in range(rounds):
        new = [i for i in exclusion(rows, base_rule, views.sil_weight) if i not in excl]
        if not new:
            break
        excl += new
        views.sil_weight[new] = 0.0
        res = once()
        if rounds > 1:
            rows = per_view(smpl, faces, res, views, masks, device)
    views.sil_weight = sw0
    return res, excl, first


def occluded_views(run, cond, seed):
    """Re-draw the same random view choice reliability_study.build makes for this condition and seed."""
    kind, arg = rs.CONDITIONS[cond]
    rng = np.random.default_rng(1000 + seed)
    n = len(run["masks"])
    return set(rng.choice(n, size=max(1, int(round(n * arg["share"]))), replace=False).tolist())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--conditions", nargs="+", default=["baseline", "occlude_hip_25pct", "occlude_chest_25pct",
                                                         "lighting_random_6px", "lighting_dark_edge_4px",
                                                         "blur_50pct"])
    ap.add_argument("--rules", nargs="+", default=["iou", "protrusion", "none"])
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    smpl = load_smpl(a.device)
    faces = np.asarray(smpl.faces, dtype=np.int64)
    rep = json.loads(a.output.read_text(encoding="utf-8")) if a.output.is_file() else {"runs": {}}
    rep.update(device=a.device, prot_floor=PROT_FLOOR, prot_k=PROT_K)
    for rid in a.runs:
        d = next(RUNS.glob(rid + "*"))
        run = load_run(d.name)
        R = rep["runs"].setdefault(d.name[:8], {})
        for cond in a.conditions:
            kind, arg = rs.CONDITIONS[cond]
            for seed in (range(a.seeds) if kind in rs.STOCHASTIC else range(1)):
                for rule in a.rules:
                    key = f"{cond}#{seed}|{rule}"
                    if key in R:
                        continue
                    t0 = time.time()
                    if kind == "none":
                        v, masks, be, q, _ = rs.pick_views(run, np.arange(len(run["masks"])))
                        v = rs.attach(v, masks)
                    else:
                        v, masks, be, q = rs.build(run, (kind, arg), np.random.default_rng(1000 + seed))
                    res, excl, rows = fit_with_rule(smpl, faces, v, masks, be, q, a.device, rule)
                    row = {"excluded": excl, "girths_x_height": girths(smpl, res.betas, a.device),
                           "first_fit_per_view": rows}
                    if kind == "occlude":
                        truth = occluded_views(run, cond, seed)
                        hit = set(excl) & truth
                        row.update(degraded_views=sorted(truth), true_pos=len(hit), false_pos=len(set(excl) - truth),
                                   missed=len(truth - set(excl)))
                    R[key] = row
                    base = R.get(f"baseline#0|{rule}", R.get("baseline#0|none", {})).get("girths_x_height")
                    msg = ""
                    if base:
                        g = row["girths_x_height"]
                        msg = " ".join(f"{s[:5]}{100 * (g[s] - base[s]) / base[s]:+.1f}"
                                       for s in ("chest", "waist", "hip", "thigh_left"))
                    tp = f" TP{row['true_pos']}/FP{row['false_pos']}/miss{row['missed']}" if "true_pos" in row else ""
                    print(f"{d.name[:8]} {key:36s} {time.time() - t0:4.0f}s excl={len(excl):2d}{tp} {msg}", flush=True)
                    a.output.parent.mkdir(parents=True, exist_ok=True)
                    a.output.write_text(json.dumps(rep, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
