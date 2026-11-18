"""Side-by-side comparison of completed reconstruction runs of the same recording.

    .venv\\Scripts\\python.exe scripts/compare_reconstructions.py LABEL=RUN_ID [LABEL=RUN_ID ...] --out DIR

Writes DIR/views_comparison.jpg (per method: photo | observed red vs fitted green outline | render, for
front / left / back / right), DIR/mesh_views_comparison.jpg (final A-pose mesh at 0/90/180/270 deg per method)
and DIR/metrics.json + DIR/metrics.md (fit / held-out IoU, protrusion, proportion check, girths, per view).
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

RUNS = Path(__file__).resolve().parents[1] / "outputs" / "reconstructions"
SITES = ["chest", "waist", "abdomen", "hip", "upper_arm_left", "forearm_left", "thigh_left", "calf_left"]


def _label(img, text, h=40):
    band = np.full((h, img.shape[1], 3), 255, np.uint8)
    cv2.putText(band, text, (10, int(h * 0.7)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2)
    return np.vstack([band, img])


def _fit_h(img, h):
    return cv2.resize(img, (int(round(img.shape[1] * h / img.shape[0])), h))


def _pad_w(img, w):
    return np.pad(img, ((0, 0), (0, w - img.shape[1]), (0, 0)), constant_values=255)


def metrics(run_dir: Path) -> dict:
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    rep = json.loads((run_dir / "reconstruction_report.json").read_text(encoding="utf-8"))
    res = status["result"]
    sil = rep["silhouette_fitting"]["summary_by_method"]["silhouette_keypoint_shared_shape (new)"]
    held = rep["multiview_optimisation"]["held_out_validation"]["results"]["silhouette_keypoint_shared_shape (new)"]
    views = rep.get("comparison_images", {}).get("views", {})
    return {
        "run_id": status["run_id"], "video_id": status["video_id"],
        "fit_config": rep["multiview_optimisation"].get("fit_config"),
        "fit_iou": sil.get("iou_mean"), "held_out_iou": held.get("iou_mean"),
        "fit_protrusion": sil.get("protrusion_frac_mean"), "fit_uncovered": sil.get("uncovered_frac_mean"),
        "held_out_protrusion": held.get("protrusion_frac_mean"),
        "proportion_check": rep["silhouette_fitting"].get("proportion_check"),
        "max_abs_beta": float(np.abs(rep["shared_shape"]["betas"]).max()),
        "beta_mahalanobis_change_from_hmr2": rep["shared_shape"]["beta_mahalanobis_change_from_prior"],
        "per_view": {k: (None if v is None else {"yaw": v["yaw_deg"], "fit_iou": v["iou_final_fit"],
                                                 "held_out_iou": v["iou_cross_validated_held_out"]})
                     for k, v in views.items()},
        "girth_over_height": {k: res["measurements"].get(k) for k in SITES},
        "geometry_valid": res["geometry_valid"], "warnings": [w.get("code") for w in res.get("warnings", [])],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="LABEL=RUN_ID")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pairs = [r.split("=", 1) for r in a.runs]
    dirs = {lab: next(RUNS.glob(rid + "*")) for lab, rid in pairs}

    cols = [_label(_fit_h(cv2.imread(str(d / "view_comparison.jpg")), 2000), lab) for lab, d in dirs.items()]
    cv2.imwrite(str(out / "views_comparison.jpg"), np.hstack(cols), [cv2.IMWRITE_JPEG_QUALITY, 88])
    rows = [_label(cv2.imread(str(d / "final_mesh_views.jpg")), lab) for lab, d in dirs.items()]
    w = max(r.shape[1] for r in rows)
    cv2.imwrite(str(out / "mesh_views_comparison.jpg"), np.vstack([_pad_w(r, w) for r in rows]),
                [cv2.IMWRITE_JPEG_QUALITY, 90])

    m = {lab: metrics(d) for lab, d in dirs.items()}
    (out / "metrics.json").write_text(json.dumps(m, indent=1), encoding="utf-8")
    labs = list(m)
    f = lambda v, p=3: "-" if v is None else (f"{v:.{p}f}" if isinstance(v, float) else str(v))  # noqa: E731
    lines = ["| metric | " + " | ".join(labs) + " |", "|---" * (len(labs) + 1) + "|"]
    for key, name in (("fit_iou", "fit IoU"), ("held_out_iou", "held-out IoU"), ("fit_protrusion", "fit protrusion"),
                      ("held_out_protrusion", "held-out protrusion"), ("fit_uncovered", "uncovered outline"),
                      ("max_abs_beta", "max |beta| (SD)"), ("beta_mahalanobis_change_from_hmr2", "change from HMR2")):
        lines.append(f"| {name} | " + " | ".join(f(m[l][key]) for l in labs) + " |")
    for grp in ("side_depth", "front_back_width"):
        lines.append(f"| {grp} model/obs | " + " | ".join(
            str((m[l]["proportion_check"] or {}).get(grp, {}).get("median_ratio")) for l in labs) + " |")
    for v in ("front", "left_side", "back", "right_side"):
        lines.append(f"| {v} fit / held-out IoU | " + " | ".join(
            "-" if not m[l]["per_view"].get(v) else f"{f(m[l]['per_view'][v]['fit_iou'])} / "
            f"{f(m[l]['per_view'][v]['held_out_iou'])}" for l in labs) + " |")
    for s in SITES:
        lines.append(f"| {s} / height | " + " | ".join(f(m[l]["girth_over_height"][s]) for l in labs) + " |")
    lines.append("| warnings | " + " | ".join(", ".join(m[l]["warnings"]) or "none" for l in labs) + " |")
    (out / "metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
