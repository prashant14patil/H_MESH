"""Folder-based checkerboard calibration reports; no production routes or models."""
from __future__ import annotations

import json
import math
import uuid
from pathlib import Path

import cv2
import numpy as np

from backend.services.checkerboard_calibration import (
    CalibrationFailure, board_points, calibrate_observations, detect_corners, save_calibration,
)

EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _write_json(path, value):
    with path.open("x", encoding="utf-8") as output:
        json.dump(value, output, indent=2, allow_nan=False)


def _report(path, status, records, config=None, error=None):
    lines = ["# Checkerboard calibration report", "", f"Status: **{status}**", "",
             "Training reprojection errors are not independent validation of camera accuracy.",
             "No image was silently removed due to reprojection error.", ""]
    if error:
        lines += [f"Failure: {error}", ""]
    if config:
        quality = config["quality_assessment"]
        lines += [f"Resolution: {config['image_size'][0]} × {config['image_size'][1]}",
                  f"Detected views used: {config['view_count']}",
                  f"Overall RMS: {config['rms_reprojection_px']:.4f} px",
                  f"Quality: **{quality['status']}**",
                  f"Thresholds: warning > {quality['warning_rms_px']} px RMS; "
                  f"poor > {quality['poor_rms_px']} px RMS; high per-image > "
                  f"{quality['high_image_error_px']} px.", ""]
    lines += ["## Image results", "", "| Image | Detection | Reason | RMS (px) | High error |",
              "|---|---|---|---:|---|"]
    for record in records:
        error_value = record.get("reprojection_error_px")
        lines.append(f"| {record['filename'].replace('|', chr(92) + '|')} | {record['detection']} | "
                     f"{record.get('reason') or '-'} | "
                     f"{error_value:.4f} | {record.get('high_error', False)} |"
                     if error_value is not None else
                     f"| {record['filename'].replace('|', chr(92) + '|')} | {record['detection']} | "
                     f"{record.get('reason') or '-'} | - | - |")
    lines += ["", "## Limitations and next steps", "",
              "- Thresholds are screening heuristics, not proof of calibration quality.",
              "- Check independent held-out board views and image coverage; low training RMS can hide systematic error.",
              "- Use the same physical camera, resolution, focus, zoom and stabilization settings as capture.",
              "- Checkerboard poses are per-image board poses, not capture-camera motion.",
              "- Treat images and diagnostic artifacts as potentially sensitive."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_visualizations(run_dir, accepted, projected, config, sample_count):
    matrix = np.asarray(config["camera_matrix"], dtype=float)
    distortion = np.asarray(config["distortion"], dtype=float)
    samples = run_dir / "undistorted_samples"
    samples.mkdir()
    panels = []
    for index, ((path, points), projection) in enumerate(zip(accepted, projected)):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise CalibrationFailure(f"Image became unreadable during output generation: {path.name}")
        if index < sample_count:
            undistorted = cv2.undistort(image, matrix, distortion)
            for name, picture in (("before", image), ("after", undistorted)):
                if not cv2.imwrite(str(samples / f"{index:03d}_{name}.png"), picture):
                    raise CalibrationFailure("Unable to write undistorted sample")
        overlay = image.copy()
        for observed, predicted in zip(points.reshape(-1, 2), np.asarray(projection)):
            a, b = tuple(np.rint(observed).astype(int)), tuple(np.rint(predicted).astype(int))
            cv2.circle(overlay, a, 3, (0, 255, 0), -1)
            cv2.drawMarker(overlay, b, (0, 0, 255), cv2.MARKER_CROSS, 9, 1)
            cv2.line(overlay, a, b, (255, 200, 0), 1)
        thumb = cv2.resize(overlay, (480, 360), interpolation=cv2.INTER_AREA)
        cv2.putText(thumb, f"{index}: green detected / red projected", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
        panels.append(thumb)
    # Include every accepted image in the contact sheet.
    width = 2
    blank = np.zeros_like(panels[0])
    rows = [np.hstack((panels[i], panels[i + 1] if i + 1 < len(panels) else blank))
            for i in range(0, len(panels), width)]
    if not cv2.imwrite(str(run_dir / "reprojection_visualization.png"), np.vstack(rows)):
        raise CalibrationFailure("Unable to write reprojection visualization")


def run_calibration(input_dir, columns, rows, square_size, *, output_root=Path("outputs/fixed_camera"),
                    run_id=None, min_views=6, warning_rms_px=0.5, poor_rms_px=2.0,
                    high_image_error_px=1.0, sample_count=3, provenance="checkerboard_images"):
    """Create one exclusive run and record all image outcomes, including failed runs.

    Absolute RMS thresholds are explicit screening heuristics in pixels, not
    independent validation. No reprojection outlier is removed.
    """
    board_points(columns, rows, square_size)
    if (not isinstance(min_views, int) or min_views < 3 or not isinstance(sample_count, int)
            or sample_count < 1 or not all(math.isfinite(x) and x > 0 for x in
            (warning_rms_px, poor_rms_px, high_image_error_px)) or warning_rms_px >= poor_rms_px):
        raise CalibrationFailure("Invalid view count, sample count or quality thresholds")
    if provenance not in ("synthetic", "checkerboard_images"):
        raise CalibrationFailure("Invalid provenance")
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise CalibrationFailure(f"Input directory does not exist or is not a directory: {input_dir}")
    paths = sorted((p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS),
                   key=lambda p: (p.name.casefold(), p.name))
    if not paths:
        raise CalibrationFailure("No JPG, JPEG or PNG files found in input directory")
    run_id = run_id or uuid.uuid4().hex
    if not isinstance(run_id, str) or not run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in run_id):
        raise CalibrationFailure("Run ID must contain only letters, digits, underscore or hyphen")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / f"real_{run_id}"
    run_dir.mkdir(exist_ok=False)

    records, accepted = [], []
    size = None
    mixed_resolution = False
    failure = None
    config = None
    try:
        for path in paths:
            record = {"filename": path.name, "detection": "rejected", "reason": None,
                      "reprojection_error_px": None, "high_error": None}
            records.append(record)
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                record["reason"] = "unreadable"
                continue
            this_size = (image.shape[1], image.shape[0])
            record["image_size"] = list(this_size)
            if size is None:
                size = this_size
            if this_size != size:
                record["reason"] = "inconsistent_resolution"
                mixed_resolution = True
                continue
            try:
                points = detect_corners(image, columns, rows)
            except cv2.error:
                record["reason"] = "corner_detection_error"
                continue
            if points is None:
                record["reason"] = "checkerboard_not_detected"
                continue
            record["detection"] = "detected"
            accepted.append((path, points))
        if mixed_resolution:
            raise CalibrationFailure("Image resolutions differ; use a consistent-resolution image set")
        if len(accepted) < min_views:
            raise CalibrationFailure(f"Insufficient successful detections: {len(accepted)}; require {min_views}")
        # The existing calibrator enforces finite positive threshold: this very
        # high limit allows diagnostic reporting of a poor but computable fit.
        config = calibrate_observations(
            [points for _, points in accepted], size, columns, rows, square_size,
            provenance=provenance, min_views=min_views, max_rms_px=1e9, include_projections=True,
        )
        projected = config.pop("projected_corners_px")
        errors = iter(config["per_view_reprojection_px"])
        for record in records:
            if record["detection"] == "detected":
                record["reprojection_error_px"] = next(errors)
                record["high_error"] = record["reprojection_error_px"] > high_image_error_px
        rms = config["rms_reprojection_px"]
        status = "poor" if rms > poor_rms_px else ("review" if rms > warning_rms_px else "acceptable_screening")
        if status == "acceptable_screening" and any(r.get("high_error") is True for r in records):
            status = "review"
        config["quality_assessment"] = {"status": status, "warning_rms_px": warning_rms_px,
                                        "poor_rms_px": poor_rms_px,
                                        "high_image_error_px": high_image_error_px,
                                        "independent_validation": False}
        config["accepted_images"] = [path.name for path, _ in accepted]
        config["rejected_images"] = [{"filename": r["filename"], "reason": r["reason"]}
                                      for r in records if r["detection"] == "rejected"]
        _save_visualizations(run_dir, accepted, projected, config, sample_count)
        save_calibration(config, run_dir / "calibration.json")
    except (CalibrationFailure, cv2.error, OSError, ValueError) as exc:
        failure = str(exc)
    _write_json(run_dir / "image_results.json", {"status": "failed" if failure else "completed",
                "error": failure, "images": records})
    _report(run_dir / "calibration_report.md", "failed" if failure else config["quality_assessment"]["status"],
            records, config if not failure else None, failure)
    if failure:
        raise CalibrationFailure(f"{failure}. Diagnostic report: {run_dir}")
    return run_dir
