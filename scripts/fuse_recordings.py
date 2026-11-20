"""Fit ONE body shape to several recordings of the same person (same session/clothing) and validate it on a
recording it never saw.

Each recording keeps its own per-view poses and cameras (they are independent captures); only the body shape is
shared. Random per-recording errors (mask noise, pose, frame choice) average out across recordings; a systematic
error common to all (e.g. the same loose trousers) does not.

Reports, for the pipeline's current stage A+B shape fit:
  * per-recording fits and their spread (repeatability),
  * the fused fit,
  * leave-one-recording-out: shape from the other recordings, scored on the left-out recording's silhouettes
    with its poses/cameras fitted from 2D keypoints only (the left-out silhouettes are never used for shape).

    .venv\\Scripts\\python.exe scripts/fuse_recordings.py RUN_A RUN_B [RUN_C ...] --sex male --height-cm 179 [--device cpu]
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
from backend.services import anthropometric_check as ac  # noqa: E402
from backend.services import body_measurements as bm  # noqa: E402
from backend.services import smpl_multiview as smv  # noqa: E402

SITES = ["chest", "waist", "abdomen", "hip", "thigh_left", "thigh_right", "calf_left", "upper_arm_left", "forearm_left"]


def concat(views_list):
    """One FitViews from several recordings' views (shared focal required: same device/resolution)."""
    f = {round(v.focal, 3) for v in views_list}
    if len(f) != 1:
        raise ValueError(f"recordings have different camera focal lengths {f}")
    cat = lambda name: np.concatenate([getattr(v, name) for v in views_list])  # noqa: E731
    lst = lambda name: sum([list(getattr(v, name)) for v in views_list], [])  # noqa: E731
    return smv.FitViews(keypoints=cat("keypoints"), weights=cat("weights"), global_orient=cat("global_orient"),
                        body_pose=cat("body_pose"), cam_t=cat("cam_t"), box_height=cat("box_height"),
                        center=cat("center"), focal=views_list[0].focal, contours=lst("contours"),
                        dist_maps=lst("dist_maps"), contour_weights=lst("contour_weights"),
                        sil_weight=cat("sil_weight"))


def shape_fit(smpl, views, agg, sig, sex, device):
    """Exactly the pipeline's stage A + B (body_reconstruction.fit_shared_shape and its current config)."""
    from backend.services.body_reconstruction import fit_shared_shape
    _, _, shape = fit_shared_shape(smpl, views, agg, sig, device, sex)
    return shape


def girths_cm(smpl, betas, height_cm, device):
    from backend.services.body_reconstruction import stature_height
    v, _, _, _ = smv.canonical_mesh(smpl, betas, np.tile(np.eye(3), (1, 23, 1, 1)), None, device, pose="measurement")
    m = bm.measure_body(v, np.asarray(smpl.faces, dtype=np.int64), smv.smpl_joints24(smpl, v))["measurements"]
    k = height_cm / stature_height(smpl, betas, device)
    return {s: (round(m[s]["tape_circumference"] * k, 1) if m.get(s, {}).get("valid") else None) for s in SITES}


def silhouette_iou(smpl, fit, views, masks, device):
    faces = np.asarray(smpl.faces, dtype=np.int64)
    verts = smv.posed_output(smpl, fit.betas, fit.global_orient, fit.body_pose, device).vertices.cpu().numpy()
    ious = []
    for i, m in enumerate(masks):
        if m is None:
            continue
        uv = smv.project_np(verts[i].astype(np.float64) + fit.cam_t[i], views.focal, views.center[i])
        ious.append(smv.silhouette_metrics(smv.rasterize_silhouette(uv, faces, m.shape), m, 1.0)["iou"])
    return float(np.mean(ious))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--sex", choices=["male", "female"], default=None)
    ap.add_argument("--height-cm", type=float, default=179.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "accuracy_study" / "fused_recordings.json"))
    a = ap.parse_args()
    smpl = mas.load_smpl(a.device)
    recs = [mas.load_run(r) for r in a.runs]
    for r in recs:
        r["views"].sil_weight = r["views"].sil_weight if r["views"].sil_weight is not None else np.ones(len(r["masks"]))
    res = {"runs": a.runs, "sex": a.sex, "height_cm": a.height_cm, "individual": {}, "leave_one_out": {}}

    def prior(idx):
        be = np.concatenate([recs[i]["betas_each"] for i in idx])
        q = np.concatenate([recs[i]["quality"] / recs[i]["quality"].max() for i in idx])
        agg, std, _ = smv.robust_beta_aggregate(be, q)
        return agg, np.maximum(std, 0.25)

    for i, r in enumerate(recs):
        agg, sig = prior([i])
        fit = shape_fit(smpl, r["views"], agg, sig, a.sex, a.device)
        res["individual"][a.runs[i]] = {"girths_cm": girths_cm(smpl, fit.betas, a.height_cm, a.device),
                                        "betas": fit.betas.tolist()}
        print(f"individual {a.runs[i][:8]}: {res['individual'][a.runs[i]]['girths_cm']}", flush=True)
    agg, sig = prior(range(len(recs)))
    fused = shape_fit(smpl, concat([r["views"] for r in recs]), agg, sig, a.sex, a.device)
    res["fused"] = {"girths_cm": girths_cm(smpl, fused.betas, a.height_cm, a.device), "betas": fused.betas.tolist(),
                    "views": int(sum(len(r["masks"]) for r in recs))}
    print(f"fused ({res['fused']['views']} views): {res['fused']['girths_cm']}", flush=True)
    # Export the fused body: measurement A-pose, plain face, scaled to metres from the supplied stature.
    from backend.services.body_reconstruction import stature_height
    out_dir = Path(a.out).with_suffix("")
    out_dir.mkdir(parents=True, exist_ok=True)
    v, _, _, _ = smv.canonical_mesh(smpl, fused.betas, np.tile(np.eye(3), (1, 23, 1, 1)), None, a.device,
                                    pose="measurement")
    faces = np.asarray(smpl.faces, dtype=np.int64)
    v = smv.plain_face(v, faces, smv.face_region(smpl, v))
    metres = a.height_cm / 100.0 / stature_height(smpl, fused.betas, a.device)
    smv.write_obj(out_dir / "fused_body_mesh_metric.obj", v * metres, faces,
                  f"Body shape fused from {len(recs)} recordings ({res['fused']['views']} views): "
                  + ", ".join(r[:8] for r in a.runs) + f". Metres, scaled to {a.height_cm:.1f} cm stature. "
                  "Measurement A-pose, plain face. External geometry only.")
    res["fused"]["mesh_obj"] = str(out_dir / "fused_body_mesh_metric.obj")
    for i in range(len(recs)):
        train = [j for j in range(len(recs)) if j != i]
        agg, sig = prior(train)
        sh = shape_fit(smpl, concat([recs[j]["views"] for j in train]), agg, sig, a.sex, a.device)
        hv = recs[i]["views"]
        held = smv.FitViews(**{**hv.__dict__, "contours": None})
        own_agg, own_sig = prior([i])
        scores = {}
        for name, betas in (("shape from OTHER recordings", sh.betas),
                            ("own recording's shape", np.array(res["individual"][a.runs[i]]["betas"]))):
            hf = smv.fit_multiview(smpl, held, betas, smv.FitConfig(iterations=150), a.device, fit_shape=False)
            scores[name] = silhouette_iou(smpl, hf, hv, recs[i]["masks"], a.device)
        res["leave_one_out"][a.runs[i]] = {"iou": scores, "girths_cm_from_others": girths_cm(smpl, sh.betas, a.height_cm, a.device)}
        other, own = scores["shape from OTHER recordings"], scores["own recording's shape"]
        print(f"left out {a.runs[i][:8]}: IoU with shape from other recordings {other:.4f} | own shape {own:.4f}",
              flush=True)
    G = np.array([[res["individual"][r]["girths_cm"][s] for s in SITES] for r in a.runs], dtype=float)
    res["repeatability"] = {s: {"mean": float(np.nanmean(G[:, k])), "range": float(np.nanmax(G[:, k]) - np.nanmin(G[:, k])),
                                "sd": float(np.nanstd(G[:, k], ddof=1)) if len(a.runs) > 1 else None}
                            for k, s in enumerate(SITES)}
    if a.sex:
        g = res["fused"]["girths_cm"]
        res["fused"]["plausibility"] = ac.check({s: g[s] for s in ("chest", "abdomen", "hip")}, a.height_cm, a.sex)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("repeatability across recordings (cm): " + ", ".join(
        f"{s} sd {v['sd']:.1f} range {v['range']:.1f}" for s, v in res["repeatability"].items() if v["sd"] is not None))


if __name__ == "__main__":
    main()
