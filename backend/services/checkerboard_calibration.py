"""Isolated OpenCV intrinsic calibration; no models or capture routes are used."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


class CalibrationFailure(ValueError):
    """Invalid input or an unsuitable calibration."""


def board_points(columns, rows, square_size):
    if (not isinstance(columns, int) or not isinstance(rows, int)
            or columns < 3 or rows < 3 or not np.isfinite(square_size) or square_size <= 0):
        raise CalibrationFailure("Board must have at least 3x3 inner corners and positive square size")
    obj = np.zeros((rows * columns, 3), dtype=np.float32)
    obj[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2) * square_size
    return obj


def validate_config(config, image_size=None):
    """Validate serialized calibration and optionally require an exact capture resolution."""
    try:
        if config["schema_version"] != 1 or config["provenance"] not in ("synthetic", "checkerboard_images"):
            raise CalibrationFailure("Unknown calibration schema or provenance")
        width, height = config["image_size"]
        if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
            raise CalibrationFailure("Invalid image resolution")
        matrix = np.asarray(config["camera_matrix"], dtype=float)
        distortion = np.asarray(config["distortion"], dtype=float)
        rms = float(config["rms_reprojection_px"])
        errors = np.asarray(config["per_view_reprojection_px"], dtype=float)
        count = config["view_count"]
        if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
                or matrix[0, 0] <= 0 or matrix[1, 1] <= 0
                or not np.allclose(matrix[2], [0, 0, 1])
                or not np.isclose(matrix[0, 1], 0) or not np.isclose(matrix[1, 0], 0)):
            raise CalibrationFailure("Invalid camera matrix")
        if distortion.ndim != 1 or len(distortion) not in (4, 5, 8, 12, 14) or not np.isfinite(distortion).all():
            raise CalibrationFailure("Invalid distortion coefficients")
        if (type(count) is not int or count < 3 or errors.shape != (count,)
                or not np.isfinite(errors).all() or np.any(errors < 0)
                or not np.isfinite(rms) or rms < 0):
            raise CalibrationFailure("Invalid calibration error statistics")
        board = config["board"]
        board_points(board["columns"], board["rows"], board["square_size"])
        if image_size is not None and tuple(image_size) != (width, height):
            raise CalibrationFailure("Capture resolution does not match calibration")
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        if isinstance(exc, CalibrationFailure):
            raise
        raise CalibrationFailure("Malformed calibration configuration") from exc
    return config


def calibrate_observations(corners, image_size, columns, rows, square_size,
                           *, provenance="synthetic", min_views=6, max_rms_px=2.0,
                           include_projections=False):
    """Calibrate from detected inner corners. Synthetic input must be labelled synthetic."""
    obj = board_points(columns, rows, square_size)
    if provenance not in ("synthetic", "checkerboard_images"):
        raise CalibrationFailure("Explicit valid provenance required")
    if (not isinstance(min_views, int) or min_views < 3 or len(corners) < min_views
            or not np.isfinite(max_rms_px) or max_rms_px <= 0):
        raise CalibrationFailure("Insufficient views or invalid threshold")
    if (len(image_size) != 2 or any(type(v) is not int or v <= 0 for v in image_size)):
        raise CalibrationFailure("Invalid image size")
    observations = []
    for corner in corners:
        points = np.asarray(corner, dtype=np.float32).reshape(-1, 2)
        if (points.shape != (len(obj), 2) or not np.isfinite(points).all()
                or np.any(points[:, 0] < 0) or np.any(points[:, 0] >= image_size[0])
                or np.any(points[:, 1] < 0) or np.any(points[:, 1] >= image_size[1])):
            raise CalibrationFailure("Invalid or out-of-image corner observations")
        observations.append(points.reshape(-1, 1, 2))
    # Near-identical point sets cannot reliably constrain intrinsics.
    spans = np.array([p.reshape(-1, 2) for p in observations])
    max_difference = max(float(np.linalg.norm(a - b, axis=1).mean())
                         for i, a in enumerate(spans) for b in spans[i + 1:])
    if max_difference < 5.0:
        raise CalibrationFailure("Checkerboard views are too similar")
    try:
        rms, matrix, dist, rotations, translations = cv2.calibrateCamera(
            [obj.copy() for _ in observations], observations, tuple(image_size), None, None)
        normals = [cv2.Rodrigues(rvec)[0][:, 2] for rvec in rotations]
        max_tilt = max(np.degrees(np.arccos(np.clip(abs(np.dot(a, b)), -1, 1)))
                       for i, a in enumerate(normals) for b in normals[i + 1:])
        if max_tilt < 3.0:
            raise CalibrationFailure("Insufficient checkerboard tilt diversity")
        errors, projections = [], []
        for points, rvec, tvec in zip(observations, rotations, translations):
            projected, _ = cv2.projectPoints(obj, rvec, tvec, matrix, dist)
            errors.append(float(np.sqrt(np.mean(np.sum((projected - points) ** 2, axis=2)))))
            projections.append(projected.reshape(-1, 2).tolist())
    except cv2.error as exc:
        raise CalibrationFailure("OpenCV calibration failed") from exc
    config = {"schema_version": 1, "provenance": provenance,
              "image_size": list(image_size), "camera_matrix": matrix.tolist(),
              "distortion": dist.reshape(-1).tolist(), "rms_reprojection_px": float(rms),
              "per_view_reprojection_px": errors, "view_count": len(observations),
              "max_board_normal_angle_deg": float(max_tilt),
              "board": {"columns": columns, "rows": rows, "square_size": float(square_size)},
              "warnings": ["A low reprojection error alone does not validate calibration; check view diversity and held-out images."]}
    if include_projections:
        config["projected_corners_px"] = projections
    validate_config(config)
    if rms > max_rms_px:
        raise CalibrationFailure(f"Calibration RMS {rms:.3f}px exceeds {max_rms_px:.3f}px")
    return config


def detect_corners(gray, columns, rows):
    """Detect and refine checkerboard inner corners in one grayscale image."""
    found, points = cv2.findChessboardCorners(gray, (columns, rows))
    if not found:
        return None
    return cv2.cornerSubPix(gray, points, (11, 11), (-1, -1),
                            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, .001))


def calibrate_images(paths, columns, rows, square_size, *, min_views=6, max_rms_px=2.0,
                     provenance="checkerboard_images"):
    """Detect checkerboard corners in explicit image paths; never invent calibration."""
    board_points(columns, rows, square_size)
    if not paths:
        raise CalibrationFailure("No checkerboard images supplied; calibration pending")
    corners, accepted, rejected, image_size = [], [], [], None
    for raw_path in paths:
        path = Path(raw_path)
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            rejected.append({"path": str(path), "reason": "unreadable"})
            continue
        size = (image.shape[1], image.shape[0])
        if image_size is None:
            image_size = size
        if image_size != size:
            raise CalibrationFailure("Calibration images have different resolutions")
        points = detect_corners(image, columns, rows)
        if points is None:
            rejected.append({"path": str(path), "reason": "checkerboard_not_detected"})
            continue
        corners.append(points)
        accepted.append(str(path))
    if image_size is None:
        raise CalibrationFailure("No readable checkerboard images")
    result = calibrate_observations(corners, image_size, columns, rows, square_size,
                                    provenance=provenance, min_views=min_views,
                                    max_rms_px=max_rms_px)
    result["accepted_images"], result["rejected_images"] = accepted, rejected
    return result


def save_calibration(config, path):
    validate_config(config)
    with Path(path).open("x", encoding="utf-8") as output:
        json.dump(config, output, indent=2, allow_nan=False)


def load_calibration(path, image_size=None):
    with Path(path).open(encoding="utf-8") as source:
        config = json.load(source)
    return validate_config(config, image_size)


def undistort_pixels(config, pixels, image_size):
    """Return undistorted pixel coordinates in the same intrinsic pixel convention."""
    validate_config(config, image_size)
    points = np.asarray(pixels, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise CalibrationFailure("Pixels must be finite Nx2")
    matrix = np.asarray(config["camera_matrix"], dtype=float)
    return cv2.undistortPoints(points.reshape(-1, 1, 2), matrix,
                               np.asarray(config["distortion"], dtype=float), P=matrix).reshape(-1, 2)
