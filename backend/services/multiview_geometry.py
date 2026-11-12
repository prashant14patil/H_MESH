"""Synthetic/general multi-view geometry. No model or camera poses are inferred."""
from itertools import combinations

import numpy as np

from backend.services.camera_geometry import CalibrationError, Camera


def project_views(cameras, points):
    """Project [N,3] world points into V explicitly supplied cameras: [V,N,2]."""
    if len(cameras) < 1 or not all(isinstance(c, Camera) for c in cameras):
        raise ValueError("At least one valid camera is required")
    return np.stack([camera.project(points) for camera in cameras])


def _intersection(cameras, observations, weights, selected, min_angle_deg):
    origins, directions = [], []
    for idx in selected:
        origin, direction = cameras[idx].rays(observations[idx])
        origins.append(origin)
        directions.append(direction)
    origins, directions = np.asarray(origins), np.asarray(directions)
    max_angle = max(np.degrees(np.arccos(np.clip(np.abs(np.dot(a, b)), -1, 1)))
                    for a, b in combinations(directions, 2))
    if max_angle < min_angle_deg:
        raise ValueError("Insufficient triangulation angle")
    eye = np.eye(3)
    projectors = eye - directions[:, :, None] * directions[:, None, :]
    w = weights[selected]
    matrix = np.sum(w[:, None, None] * projectors, axis=0)
    rhs = np.sum(w[:, None] * np.einsum("nij,nj->ni", projectors, origins), axis=0)
    if np.linalg.cond(matrix) > 1e10:
        raise ValueError("Degenerate camera geometry")
    point = np.linalg.solve(matrix, rhs)
    if not np.isfinite(point).all():
        raise ValueError("Non-finite triangulation")
    if any(cameras[idx].world_to_camera(point)[2] <= 1e-9 for idx in selected):
        raise ValueError("Triangulated point is behind a camera")
    return point, max_angle


def triangulate(cameras, observations, confidence=None, *, min_confidence=0.1,
                min_angle_deg=1.0, outlier_threshold_px=None):
    """Triangulate corresponding landmarks.

    observations [V,N,2] may contain NaNs for missing points. confidence is
    [V,N] in [0,1]; values below min_confidence are ignored. Optional pair
    hypotheses reject reprojection outliers only when at least three views are
    available. Results expose NaN points and reasons rather than fabricating data.
    """
    cameras = list(cameras)
    if len(cameras) < 2 or not all(isinstance(c, Camera) for c in cameras):
        raise ValueError("At least two cameras are required")
    obs = np.asarray(observations, dtype=float)
    if obs.ndim != 3 or obs.shape[0] != len(cameras) or obs.shape[2] != 2:
        raise ValueError("observations must have shape [views, landmarks, 2]")
    weights = np.ones(obs.shape[:2]) if confidence is None else np.asarray(confidence, dtype=float)
    if weights.shape != obs.shape[:2] or not np.isfinite(weights).all() or np.any((weights < 0) | (weights > 1)):
        raise ValueError("confidence must be finite in [0,1] with shape [views, landmarks]")
    if not np.isfinite(min_confidence) or not 0 <= min_confidence <= 1:
        raise ValueError("min_confidence must be in [0,1]")
    if not np.isfinite(min_angle_deg) or not 0 < min_angle_deg < 90:
        raise ValueError("min_angle_deg must be between 0 and 90")
    if outlier_threshold_px is not None and (not np.isfinite(outlier_threshold_px) or outlier_threshold_px <= 0):
        raise ValueError("outlier threshold must be positive")
    points = np.full((obs.shape[1], 3), np.nan)
    inliers = np.zeros(obs.shape[:2], dtype=bool)
    reasons = []
    angles = []
    for j in range(obs.shape[1]):
        valid = np.flatnonzero(np.isfinite(obs[:, j]).all(axis=1)
                               & (weights[:, j] >= min_confidence) & (weights[:, j] > 0))
        if len(valid) < 2:
            reasons.append("insufficient_observations")
            angles.append(None)
            continue
        selected = valid
        if outlier_threshold_px is not None and len(valid) >= 3:
            best_score, best = None, None
            for pair in combinations(valid, 2):
                try:
                    candidate, _ = _intersection(cameras, obs[:, j], weights[:, j], list(pair), min_angle_deg)
                    errors = np.full(len(cameras), np.inf)
                    for idx in valid:
                        try:
                            errors[idx] = np.linalg.norm(cameras[idx].project(candidate) - obs[idx, j])
                        except CalibrationError:
                            pass
                    subset = valid[errors[valid] <= outlier_threshold_px]
                    score = (len(subset), float(weights[subset, j].sum()), -float(np.sum(errors[subset])))
                    if len(subset) >= 2 and (best_score is None or score > best_score):
                        best_score, best = score, subset
                except (ValueError, np.linalg.LinAlgError):
                    continue
            if best is None:
                reasons.append("no_consistent_hypothesis")
                angles.append(None)
                continue
            selected = best
        try:
            point, angle = _intersection(cameras, obs[:, j], weights[:, j], selected, min_angle_deg)
            points[j] = point
            inliers[selected, j] = True
            reasons.append(None)
            angles.append(float(angle))
        except (ValueError, np.linalg.LinAlgError):
            reasons.append("degenerate_or_behind_camera")
            angles.append(None)
    return {"points": points, "inliers": inliers, "reasons": reasons, "max_ray_angles_deg": angles}


def reprojection_errors(cameras, points, observations):
    """Return [views, landmarks] pixel errors; NaNs indicate unavailable values."""
    points, obs = np.asarray(points, dtype=float), np.asarray(observations, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or obs.shape != (len(cameras), len(points), 2):
        raise ValueError("Incompatible points or observations")
    errors = np.full(obs.shape[:2], np.nan)
    for i, camera in enumerate(cameras):
        valid = np.isfinite(points).all(axis=1) & np.isfinite(obs[i]).all(axis=1)
        front = np.zeros(len(points), dtype=bool)
        if valid.any():
            front[valid] = camera.world_to_camera(points[valid])[:, 2] > 1e-9
        valid &= front
        if valid.any():
            errors[i, valid] = np.linalg.norm(camera.project(points[valid]) - obs[i, valid], axis=1)
    return errors


def select_frames(metadata, selected_filenames, *, allow_blurry=False):
    """Select existing extracted frames; never alter their original quality fields."""
    frames = metadata.get("frames")
    if not isinstance(frames, list) or not isinstance(selected_filenames, (list, tuple)):
        raise ValueError("Frame metadata and explicit selection are required")
    by_name = {}
    for frame in frames:
        name = frame.get("filename")
        if not isinstance(name, str) or name in by_name or not isinstance(frame.get("is_blurry"), bool):
            raise ValueError("Invalid, duplicate or missing frame quality metadata")
        by_name[name] = frame
    if not selected_filenames or len(set(selected_filenames)) != len(selected_filenames):
        raise ValueError("Select at least one unique frame")
    if any(name not in by_name for name in selected_filenames):
        raise ValueError("Selected frame not present in metadata")
    selected = set(selected_filenames)
    accepted, excluded = [], []
    for name, frame in by_name.items():
        record = dict(frame)
        record["manually_selected"] = name in selected
        if name not in selected:
            record["exclusion_reason"] = "not_selected"
            excluded.append(record)
        elif frame["is_blurry"] and not allow_blurry:
            record["exclusion_reason"] = "blurry_requires_explicit_override"
            excluded.append(record)
        else:
            record["quality_warning"] = "blurry" if frame["is_blurry"] else None
            accepted.append(record)
    return {"accepted": accepted, "excluded": excluded,
            "counts": {"accepted": len(accepted), "excluded": len(excluded),
                       "manually_selected": len(selected_filenames),
                       "blurry_accepted": sum(f["is_blurry"] for f in accepted)}}
