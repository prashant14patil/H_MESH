"""Synthetic camera geometry experiment; no human model or video is used.

Run: python scripts/run_synthetic_multiview.py --output outputs/synthetic_geometry/run1
Requires matplotlib only for plotting. Output directory must not already exist.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.services.camera_geometry import Camera
from backend.services.multiview_geometry import project_views, reprojection_errors, select_frames, triangulate

# Synthetic world units, +Y down. This is not an anatomical ground truth model.
NAMES = ("head", "neck", "left_shoulder", "right_shoulder", "left_elbow",
         "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
         "left_knee", "right_knee", "left_ankle", "right_ankle")
POINTS = np.array([[0, -.88, 0], [0, -.65, 0], [-.22, -.62, 0], [.22, -.62, 0],
                   [-.38, -.33, .05], [.38, -.33, -.05], [-.45, -.08, .09],
                   [.45, -.08, -.09], [-.14, 0, 0], [.14, 0, 0],
                   [-.16, .42, .03], [.16, .42, -.03],
                   [-.17, .83, .06], [.17, .83, .06]], dtype=float)
EDGES = [(0, 1), (1, 2), (1, 3), (2, 4), (4, 6), (3, 5), (5, 7),
         (2, 8), (3, 9), (8, 9), (8, 10), (10, 12), (9, 11), (11, 13)]
CAMERA_NAMES = ("front", "left", "back", "right")


def look_at(center, target=np.zeros(3), *, focal=800., provenance="synthetic"):
    center = np.asarray(center, dtype=float)
    forward = np.asarray(target) - center
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., -1., 0.])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward])
    return Camera(focal, focal, 320., 240., 640, 480, rotation, -rotation @ center, provenance)


def scene():
    return [look_at([0, 0, -4]), look_at([-4, 0, 0]),
            look_at([0, 0, 4]), look_at([4, 0, 0])]


def metrics(cameras, observations, confidence, reconstructed, truth, training, heldout):
    valid = np.isfinite(reconstructed).all(axis=1)
    if not valid.any():
        raise RuntimeError("No reconstructed landmarks")
    errors = reprojection_errors(cameras, reconstructed, observations)
    errors[confidence < .1] = np.nan
    d3 = np.linalg.norm(reconstructed[valid] - truth[valid], axis=1)
    return {
        "reconstructed_landmarks": int(valid.sum()),
        "missing_reconstructions": int((~valid).sum()),
        "mean_3d_error_world_units": float(d3.mean()),
        "max_3d_error_world_units": float(d3.max()),
        "train_mean_reprojection_px": float(np.nanmean(errors[training])),
        "heldout_mean_reprojection_px": float(np.nanmean(errors[heldout])) if heldout is not None else None,
    }


def plot_result(path, cameras, observations, truth, result, heldout):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15, 9))
    ax = fig.add_subplot(2, 3, 1, projection="3d")
    p = result["points"]
    ax.scatter(*truth.T, label="ground truth", c="tab:blue")
    valid = np.isfinite(p).all(axis=1)
    ax.scatter(*p[valid].T, label="reconstructed", c="tab:orange", marker="x")
    for a, b in EDGES:
        ax.plot(*truth[[a, b]].T, color="steelblue", alpha=.6)
    for i, c in enumerate(cameras):
        ax.scatter(*c.center_world, marker="^", label=f"{CAMERA_NAMES[i]} camera")
    ax.set_title("Synthetic world: cameras and landmarks")
    ax.set_xlabel("X"); ax.set_ylabel("Y down"); ax.set_zlabel("Z")
    ax.legend(fontsize=7)
    for i in range(4):
        ax = fig.add_subplot(2, 3, i + 2)
        observed = observations[i]
        ok = np.isfinite(observed).all(axis=1)
        ax.scatter(observed[ok, 0], observed[ok, 1], s=15, label="observed")
        projected = cameras[i].project(p[valid])
        ax.scatter(projected[:, 0], projected[:, 1], s=18, marker="x", label="reprojected")
        ax.set_xlim(0, 640); ax.set_ylim(480, 0)
        ax.set_title(f"{CAMERA_NAMES[i].title()} camera" + (" (held out)" if i == heldout else " (fit)"))
        ax.legend(fontsize=7)
    errors = reprojection_errors(cameras, p, observations)
    ax = fig.add_subplot(2, 3, 6)
    ax.plot(np.nanmean(errors, axis=1), marker="o")
    ax.set_xticks(range(4)); ax.set_xlabel("camera"); ax.set_ylabel("mean error (px)")
    ax.set_title("Reprojection error by view")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--noise", type=float, default=1.0, help="Gaussian pixel standard deviation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--heldout", type=int, default=3, choices=range(4))
    parser.add_argument("--frame-metadata", type=Path, help="Optional extraction metadata to review; never used as synthetic camera input")
    parser.add_argument("--select-frame", action="append", default=[], help="Explicit filename from extraction metadata; repeat as needed")
    parser.add_argument("--allow-blurry", action="store_true", help="Include manually selected blurry frames without removing their warnings")
    args = parser.parse_args()
    if not np.isfinite(args.noise) or args.noise < 0:
        parser.error("--noise must be finite and nonnegative")
    if bool(args.frame_metadata) != bool(args.select_frame):
        parser.error("--frame-metadata and at least one --select-frame must be supplied together")
    if args.allow_blurry and not args.frame_metadata:
        parser.error("--allow-blurry requires explicit frame selection")
    selection = None
    if args.frame_metadata:
        metadata = json.loads(args.frame_metadata.read_text(encoding="utf-8"))
        selection = select_frames(metadata, args.select_frame, allow_blurry=args.allow_blurry)
        selection["purpose"] = "quality review only; these frames are not synthetic experiment inputs"
    cameras = scene()
    truth = POINTS.copy()
    clean = project_views(cameras, truth)
    rng = np.random.default_rng(args.seed)
    noisy = clean + rng.normal(0, args.noise, clean.shape)
    training = [i for i in range(4) if i != args.heldout]
    confidence = np.ones(clean.shape[:2])
    missing = noisy.copy()
    missing_conf = confidence.copy()
    missing[training[0], 0:3] = np.nan
    missing_conf[training[1], 3:6] = 0.02
    # Each landmark retains at least two training observations.
    approx = [look_at(c.center_world + np.array([.08, 0, .04]), focal=840.,
                      provenance="approximate") for c in cameras]
    scenarios = [
        ("clean", cameras, clean, confidence),
        ("noisy", cameras, noisy, confidence),
        ("missing_low_confidence", cameras, missing, missing_conf),
        ("approximate_calibration", approx, noisy, confidence),
    ]
    results = {}
    reconstructions = {}
    for name, used_cameras, observations, weights in scenarios:
        result = triangulate([used_cameras[i] for i in training], observations[training],
                             weights[training], min_confidence=.1)
        results[name] = metrics(used_cameras, observations, weights, result["points"], truth, training, args.heldout)
        reconstructions[name] = result
    all_views = triangulate(cameras, noisy, confidence, min_confidence=.1)
    results["noisy_all_four_views"] = metrics(cameras, noisy, confidence, all_views["points"],
                                                truth, list(range(4)), None)
    args.output.mkdir(parents=True, exist_ok=False)
    plot_result(args.output / "noisy_views.png", cameras, noisy, truth, reconstructions["noisy"], args.heldout)
    report = {
        "description": "Synthetic stationary articulated landmark configuration and four virtual pinhole cameras; not human reconstruction.",
        "seed": args.seed, "noise_sigma_px": args.noise, "heldout_camera": args.heldout,
        "training_cameras": training, "camera_names": CAMERA_NAMES, "camera_provenance": "synthetic",
        "approximation": "focal 840 vs true 800 pixels; all camera centers shifted [0.08, 0, 0.04] world units",
        "landmark_names": NAMES, "scenarios": results,
        "frame_quality_selection": selection,
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
