"""Accuracy benchmarking for the unified 360-degree reconstruction.

What this measures, and on what evidence:

* Measurement accuracy (MAE, RMSE, % error, bias) ONLY against operator-entered reference
  measurements (tape measure + stadiometer height) in the benchmark manifest. Mesh girths are
  relative; they are converted to centimetres by the measured stature ("height-scaled"), and
  that conversion is reported as such. Without reference data the report says "not validated".
* Repeatability across *independent* recordings of the same subject (identical source files
  are de-duplicated: re-uploads are not repeat measurements).
* Surface agreement per viewpoint: exact rasterised silhouette IoU / contour distance of the
  fitted mesh against the observed subject mask, using the cross-validated held-out scores
  (each view scored by a shape fitted without it).
* Missing / extra regions: per body band, observed pixels the model misses and model pixels
  outside the observed outline.
* Single-frame HMR2 vs unified multi-view: silhouette agreement and the spread of girths
  that independent per-frame shapes would give.
* Error factors: per-view correlations of held-out IoU with sharpness, lighting, camera
  distance, rotation speed and a clothing-looseness proxy (exploratory; small samples flagged).

Nothing here fabricates accuracy: every number is computed from saved reconstruction outputs,
the source video frames, and (when present) reference measurements.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

MEASUREMENTS = ["chest", "waist", "abdomen", "hip", "upper_arm_left", "upper_arm_right", "forearm_left",
                "forearm_right", "thigh_left", "thigh_right", "calf_left", "calf_right"]
SECTORS = [("front", 0), ("front_right", 45), ("right", 90), ("back_right", 135), ("back", 180),
           ("back_left", -135), ("left", -90), ("front_left", -45)]
REGIONS = ["head", "torso_and_arms", "legs", "feet"]


# ======================================================================= pure metrics
def error_metrics(pred: dict, ref: dict) -> dict:
    """Per-measurement and aggregate errors of predictions against reference values (same units)."""
    rows = {}
    for key, r in ref.items():
        p = pred.get(key)
        if r is None or p is None or not np.isfinite(p) or r <= 0:
            continue
        err = float(p - r)
        rows[key] = {"predicted": float(p), "reference": float(r), "error": err,
                     "abs_error": abs(err), "pct_error": 100.0 * err / r}
    if not rows:
        return {"n": 0, "per_measurement": {}}
    e = np.array([v["error"] for v in rows.values()])
    pct = np.array([v["pct_error"] for v in rows.values()])
    return {"n": len(rows), "mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e ** 2).mean())),
            "bias": float(e.mean()), "mape": float(np.abs(pct).mean()), "max_abs_error": float(np.abs(e).max()),
            "per_measurement": rows}


def repeatability(values_by_subject: dict) -> dict:
    """Within-subject repeatability from >= 2 independent sessions per subject.

    ``values_by_subject`` maps subject -> list of values (one per session). Returns pooled
    within-subject SD (Sw), mean CV%, and the repeatability coefficient RC = 2.77 x Sw
    (Bland & Altman): 95% of repeat differences are expected to be below RC.
    """
    ss, dof, cvs, n_sessions, subjects = 0.0, 0, [], 0, 0
    for vals in values_by_subject.values():
        v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], dtype=np.float64)
        if len(v) < 2:
            continue
        subjects += 1
        n_sessions += len(v)
        ss += float(((v - v.mean()) ** 2).sum())
        dof += len(v) - 1
        if v.mean() != 0:
            cvs.append(float(v.std(ddof=1) / abs(v.mean()) * 100))
    if dof == 0:
        return {"evaluable": False, "reason": "needs >= 2 independent sessions of at least one subject"}
    sw = math.sqrt(ss / dof)
    return {"evaluable": True, "subjects": subjects, "sessions": n_sessions, "within_subject_sd": sw,
            "cv_pct": float(np.mean(cvs)) if cvs else None, "repeatability_coefficient": 2.77 * sw,
            "low_sample_warning": subjects < 5 or dof < 10}


def sector_of(yaw_deg: float) -> str:
    """45-degree viewing sector of an HMR2 heading (0 = facing camera, +90 = subject's right side shown)."""
    best = min(SECTORS, key=lambda s: abs(((yaw_deg - s[1]) + 180) % 360 - 180))
    return best[0]


def sector_summary(yaws, values) -> dict:
    out = {}
    for name, _ in SECTORS:
        vals = [v for y, v in zip(yaws, values) if v is not None and sector_of(y) == name]
        out[name] = ({"n": len(vals), "mean": float(np.mean(vals)), "min": float(np.min(vals))}
                     if vals else {"n": 0, "mean": None, "min": None})
    return out


def spearman(x, y) -> dict:
    from scipy.stats import spearmanr
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None and np.isfinite(a) and np.isfinite(b)]
    if len(pairs) < 5:
        return {"n": len(pairs), "rho": None, "p": None, "note": "fewer than 5 observations"}
    a, b = map(np.asarray, zip(*pairs))
    if np.all(a == a[0]) or np.all(b == b[0]):
        return {"n": len(pairs), "rho": None, "p": None, "note": "no variation"}
    res = spearmanr(a, b)
    return {"n": len(pairs), "rho": float(res.statistic), "p": float(res.pvalue),
            "note": "exploratory (n < 30)" if len(pairs) < 30 else "exploratory"}


def region_errors(pred: np.ndarray, obs: np.ndarray, keypoints: np.ndarray) -> dict:
    """Missing (observed, not covered) and extra (model outside outline) fractions per body band.

    Bands from 2D keypoints: head above the shoulders, torso_and_arms to the hips, legs to the
    ankles, feet below. Fractions are relative to the observed area in that band.
    """
    p, o = pred.astype(bool), obs.astype(bool)
    h = o.shape[0]
    shoulder = float(np.min(keypoints[[5, 6], 1]))
    hip = float(np.mean(keypoints[[11, 12], 1]))
    ankle = float(np.max(keypoints[[15, 16], 1]))
    bands = {"head": (0, shoulder), "torso_and_arms": (shoulder, hip), "legs": (hip, ankle), "feet": (ankle, h)}
    out = {}
    for name, (y0, y1) in bands.items():
        y0, y1 = int(max(0, round(y0))), int(min(h, round(y1)))
        ob, pb = o[y0:y1], p[y0:y1]
        area = ob.sum()
        out[name] = ({"missing_frac": float((ob & ~pb).sum() / area), "extra_frac": float((pb & ~ob).sum() / area)}
                     if area > 0 else {"missing_frac": None, "extra_frac": None})
    return out


def height_scaled_cm(ratio: float | None, height_cm: float | None):
    """Relative girth (x mesh height) -> centimetres using a MEASURED stature. None if either is missing."""
    if ratio is None or height_cm is None:
        return None
    return float(ratio) * float(height_cm)


def proportions(joints24: np.ndarray, height: float) -> dict:
    """Body proportions from SMPL joints, as fractions of mesh height (scale-free)."""
    J = np.asarray(joints24)

    def d(a, b):
        return float(np.linalg.norm(J[a] - J[b]))
    return {"shoulder_width": d(16, 17) / height, "hip_width": d(1, 2) / height,
            "upper_arm": (d(16, 18) + d(17, 19)) / 2 / height, "forearm": (d(18, 20) + d(19, 21)) / 2 / height,
            "thigh": (d(1, 4) + d(2, 5)) / 2 / height, "shank": (d(4, 7) + d(5, 8)) / 2 / height,
            "torso": d(0, 12) / height}


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ===================================================================== data access
class SMPLHelper:
    """Standalone SMPL (same neutral model HMR2 uses) for re-posing saved results on the GPU."""

    def __init__(self, model_path="data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl", device=None):
        import smplx
        import torch
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = smplx.SMPLLayer(model_path=model_path).to(self.device).eval()
        self.faces = np.asarray(self.model.faces, dtype=np.int64)

    def posed(self, betas, global_orient, body_pose):
        t = self.torch
        f32 = dict(dtype=t.float32, device=self.device)
        go = t.as_tensor(np.asarray(global_orient), **f32).reshape(-1, 3, 3)
        b = t.as_tensor(np.asarray(betas), **f32).reshape(-1, 10)
        if len(b) == 1:
            b = b.expand(len(go), -1)
        with t.no_grad():
            out = self.model(betas=b, global_orient=go[:, None],
                             body_pose=t.as_tensor(np.asarray(body_pose), **f32).reshape(len(go), 23, 3, 3))
        return out.vertices.cpu().numpy().astype(np.float64)

    def measurement_mesh(self, betas):
        from backend.services import smpl_multiview as smv
        pose = smv.measurement_pose()
        v = self.posed(betas, np.eye(3)[None], pose[None])[0]
        shift = np.array([(v[:, 0].min() + v[:, 0].max()) / 2, v[:, 1].min(), (v[:, 2].min() + v[:, 2].max()) / 2])
        v = v - shift
        reg = self.model.J_regressor.detach().cpu().numpy().astype(np.float64)
        return v, reg @ v


def load_session(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    fit_inputs = run_dir / "diagnostics" / "fit_inputs.npz"
    fit_result = run_dir / "diagnostics" / "fit_result.npz"
    missing = [p.name for p in (run_dir / "reconstruction_report.json", run_dir / "measurements.json", fit_inputs,
                                fit_result) if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"{run_dir.name}: missing {', '.join(missing)} (re-run the reconstruction)")
    npz = np.load(fit_inputs)
    shape = tuple(int(x) for x in npz["mask_shape"])
    masks = [np.unpackbits(npz["masks_packed"][i])[:shape[0] * shape[1]].reshape(shape) if npz["mask_usable"][i]
             else None for i in range(len(npz["selected"]))]
    return {"run_dir": run_dir,
            "report": json.loads((run_dir / "reconstruction_report.json").read_text(encoding="utf-8")),
            "selected": json.loads((run_dir / "selected_frames.json").read_text(encoding="utf-8")),
            "measurements": json.loads((run_dir / "measurements.json").read_text(encoding="utf-8")),
            "inputs": {k: npz[k] for k in npz.files if k not in ("masks_packed",)},
            "masks": masks, "fit": dict(np.load(fit_result))}


# ======================================================================== analyses
def measure_ratios(vertices, faces, joints):
    from backend.services import body_measurements as bm
    m = bm.measure_body(vertices, faces, joints)
    return {k: (v["tape_circumference_over_height"] if v["valid"] else None) for k, v in m["measurements"].items()}


def single_frame_vs_unified(session: dict, smpl: SMPLHelper, max_frames: int | None = None) -> dict:
    """Girth spread from independent per-frame HMR2 shapes vs the single unified shape."""
    betas_each = session["inputs"]["betas_each"]
    idx = range(len(betas_each)) if max_frames is None else np.linspace(0, len(betas_each) - 1, max_frames).astype(int)
    per_frame = []
    for i in idx:
        v, j = smpl.measurement_mesh(betas_each[i])
        per_frame.append(measure_ratios(v, smpl.faces, j))
    unified = {k: v.get("tape_circumference_over_height") for k, v in session["measurements"]["measurements"].items()}
    out = {}
    for k in MEASUREMENTS:
        vals = np.array([p[k] for p in per_frame if p.get(k) is not None])
        out[k] = {"unified": unified.get(k),
                  "previous_keypoint_only": session["measurements"].get("previous_method_keypoint_only", {}).get(k),
                  "single_frame_mean": float(vals.mean()) if len(vals) else None,
                  "single_frame_sd": float(vals.std(ddof=1)) if len(vals) > 1 else None,
                  "single_frame_min": float(vals.min()) if len(vals) else None,
                  "single_frame_max": float(vals.max()) if len(vals) else None,
                  "single_frame_cv_pct": float(vals.std(ddof=1) / vals.mean() * 100) if len(vals) > 1 else None,
                  "frames": int(len(vals))}
    held = session["report"]["multiview_optimisation"]["held_out_validation"]["results"]
    return {"girths_x_height": out, "per_frame_measurements": per_frame,
            "silhouette_held_out_iou": {k: v.get("iou_mean") for k, v in held.items()},
            "silhouette_held_out_contour_pct_height": {k: v.get("contour_distance_mean_pct_height")
                                                       for k, v in held.items()}}


def view_analysis(session: dict, smpl: SMPLHelper) -> dict:
    """Per-view accuracy (held-out IoU), region errors of the final fit, and per-view covariates."""
    from backend.services import smpl_multiview as smv
    inp, fit = session["inputs"], session["fit"]
    sel = session["selected"]["selected"]
    verts = smpl.posed(fit["betas"], fit["global_orient"], fit["body_pose"])
    held_iou = session["report"]["multiview_optimisation"]["held_out_validation"]["results"][
        "silhouette_keypoint_shared_shape (new)"]["per_view_iou"]
    hmr2_iou = session["report"]["multiview_optimisation"]["held_out_validation"]["results"][
        "hmr2_independent_per_frame (each frame its own shape)"]["per_view_iou"]
    rows = []
    times = np.array([s["time_sec"] for s in sel])
    yaws_unwrapped = np.degrees(np.unwrap(np.radians([s["yaw_deg"] for s in sorted(sel, key=lambda s: s["time_sec"])])))
    order = np.argsort(times)
    speed_sorted = np.abs(np.gradient(yaws_unwrapped, times[order])) if len(sel) > 2 else np.full(len(sel), np.nan)
    speed = np.empty(len(sel))
    speed[order] = speed_sorted
    for i, s in enumerate(sel):
        mask = session["masks"][i]
        row = {"frame_position": s["frame_position"], "source_frame_index": s["source_frame_index"],
               "time_sec": s["time_sec"], "yaw_deg": s["yaw_deg"], "sector": sector_of(s["yaw_deg"]),
               "iou_fit": s.get("silhouette_iou_after_fit"), "iou_held_out": held_iou[i],
               "iou_hmr2_single_frame": hmr2_iou[i] if i < len(hmr2_iou) else None,
               "sharpness": s.get("sharpness"), "rotation_speed_deg_s": float(speed[i]),
               "box_height_frac": float(inp["box_height"][i] / inp["mask_shape"][0]),
               "camera_depth_hmr2": float(inp["cam_t"][i][2])}
        if mask is not None:
            uv = smv.project_np(verts[i] + fit["cam_t"][i], float(inp["focal"]), inp["center"][i])
            pred = smv.rasterize_silhouette(uv, smpl.faces, mask.shape)
            row["regions"] = region_errors(pred, mask, inp["keypoints"][i])
            uv0 = smv.project_np(inp["hmr2_vertices"][i].astype(np.float64) + inp["cam_t"][i], float(inp["focal"]),
                                 inp["center"][i])
            hmr2_pred = smv.rasterize_silhouette(uv0, smpl.faces, mask.shape)
            # HMR2 predicts the body under the clothes; the observed outline being larger is a looseness proxy.
            row["clothing_looseness_proxy"] = float(mask.sum() / max(hmr2_pred.sum(), 1))
        rows.append(row)
    return {"views": rows,
            "sector_held_out_iou": sector_summary([r["yaw_deg"] for r in rows], [r["iou_held_out"] for r in rows]),
            "sector_single_frame_iou": sector_summary([r["yaw_deg"] for r in rows],
                                                      [r["iou_hmr2_single_frame"] for r in rows])}


def image_conditions(session: dict, video_path: Path) -> dict:
    """Lighting statistics inside the subject mask, decoded from the source video."""
    import cv2
    from backend.services.video_processor import VideoCaptureEngine
    cap, _ = VideoCaptureEngine._open_oriented_capture(str(video_path))
    out = []
    try:
        for i, s in enumerate(session["selected"]["selected"]):
            mask = session["masks"][i]
            cap.set(cv2.CAP_PROP_POS_FRAMES, s["source_frame_index"])
            ok, frame = cap.read()
            if not ok or mask is None or frame.shape[:2] != mask.shape:
                out.append(None)
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[mask > 0].astype(np.float64)
            out.append({"subject_brightness": float(gray.mean()), "subject_contrast": float(gray.std()),
                        "clipped_frac": float(((gray <= 3) | (gray >= 252)).mean())})
    finally:
        cap.release()
    return out


def factor_analysis(views: list) -> dict:
    """Exploratory per-view correlations between held-out silhouette IoU and capture factors."""
    y = [v["iou_held_out"] for v in views]
    factors = {
        "sharpness": [v.get("sharpness") for v in views],
        "subject_brightness": [(v.get("lighting") or {}).get("subject_brightness") for v in views],
        "subject_contrast": [(v.get("lighting") or {}).get("subject_contrast") for v in views],
        "clipped_frac": [(v.get("lighting") or {}).get("clipped_frac") for v in views],
        "camera_distance_box_height_frac": [v.get("box_height_frac") for v in views],
        "rotation_speed_deg_s": [v.get("rotation_speed_deg_s") for v in views],
        "clothing_looseness_proxy": [v.get("clothing_looseness_proxy") for v in views],
        "profile_view_(1-|cos yaw|)": [1 - abs(math.cos(math.radians(v["yaw_deg"]))) for v in views],
    }
    return {name: spearman(vals, y) for name, vals in factors.items()}


def session_summary(session: dict, views: list) -> dict:
    rep = session["report"]
    sel = session["selected"]
    v = rep["validation_summary"]
    times = [s["time_sec"] for s in sel["selected"]]
    light = [x["lighting"]["subject_brightness"] for x in views if x.get("lighting")]
    return {"run_id": rep["run_id"], "video_id": rep["video_id"],
            "resolution_wh": rep["video"]["displayed_size_wh"], "frames_extracted": v["frames_extracted"],
            "frames_used": v["frames_in_final_fit"], "viewpoint_distribution": v["viewpoint_distribution"],
            "rotation_deg": sel["measured_heading_rotation_deg"],
            "rotation_duration_s": max(times) - min(times) if times else None,
            "mean_rotation_speed_deg_s": (sel["measured_heading_rotation_deg"] / (max(times) - min(times))
                                          if times and max(times) > min(times) else None),
            "camera_distance_box_height_frac": float(np.median([x["box_height_frac"] for x in views])),
            "subject_brightness_median": float(np.median(light)) if light else None,
            "clothing_looseness_proxy_median": float(np.median([x["clothing_looseness_proxy"] for x in views
                                                                if "clothing_looseness_proxy" in x])),
            "held_out_iou_new": v["held_out_silhouette_iou_by_method"].get("silhouette_keypoint_shared_shape (new)"),
            "held_out_iou_previous": v["held_out_silhouette_iou_by_method"].get(
                "keypoint_only_shared_shape (previous method)"),
            "held_out_iou_hmr2_single_frame": v["held_out_silhouette_iou_by_method"].get(
                "hmr2_independent_per_frame (each frame its own shape)"),
            "landmark_error_pct_height": v["landmark_reprojection_error_pct_height"],
            "mesh_valid": v["mesh_topology_valid"], "mesh_components": v["mesh_component_count"],
            "sections_valid": v["circumference_sections_valid"]}


def mesh_reproduction_check(session: dict, smpl: SMPLHelper) -> float:
    """Max vertex difference between the exported final mesh and a re-posing of its saved betas."""
    from backend.services import smpl_multiview as smv
    v_file, _ = smv.read_obj(session["run_dir"] / "final_body_mesh.obj")
    v, _ = smpl.measurement_mesh(session["fit"]["betas"])
    return float(np.abs(v_file - v).max())
