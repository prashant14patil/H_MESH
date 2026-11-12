"""Experimental fixed-camera capture metadata validation.

No camera pose, metric translation, or reliable body yaw is inferred. Output
viewpoint scores are uncalibrated heuristics, not classification probabilities.
"""
from __future__ import annotations

import math
from pathlib import Path

from backend.services.multiview_geometry import select_frames
from backend.services.viewpoint_estimator import _classify_frame


def _xy(landmark, visibility_threshold):
    if not isinstance(landmark, dict):
        return None
    x, y, visibility = landmark.get("x"), landmark.get("y"), landmark.get("visibility")
    if (not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (x, y, visibility))
            or visibility < visibility_threshold or not landmark.get("visible", False)):
        return None
    return float(x), float(y)


def validate_capture(metadata, pose_summary, filenames, *, allow_blurry=False,
                     margin=0.02, visibility_threshold=0.5,
                     displacement_threshold=0.1, orientation_threshold=0.6):
    """Combine extracted-frame and pose metadata without modifying production outputs."""
    if not (0 <= margin < 0.5 and 0 <= visibility_threshold <= 1
            and displacement_threshold > 0 and 0 <= orientation_threshold <= 1):
        raise ValueError("Invalid capture-validation thresholds")
    selection = select_frames(metadata, filenames, allow_blurry=allow_blurry)
    poses = {}
    for entry in pose_summary.get("frames", []):
        name = entry.get("frame")
        if not isinstance(name, str) or name in poses:
            raise ValueError("Invalid or duplicate pose frame")
        poses[name] = entry
    records = []
    reference = None
    buckets = {key: 0 for key in ("front", "left_side", "back", "right_side", "intermediate", "unknown")}
    ordered_frames = sorted(
        selection["accepted"],
        key=lambda frame: (frame.get("timestamp_sec") is None,
                           frame.get("timestamp_sec") if isinstance(frame.get("timestamp_sec"), (int, float)) and math.isfinite(frame["timestamp_sec"]) else float("inf"),
                           frame["filename"]),
    )
    for frame in ordered_frames:
        pose = poses.get(Path(frame["filename"]).stem)
        warnings = []
        if frame["is_blurry"]:
            warnings.append("blurry_manual_override")
        result = {"filename": frame["filename"], "timestamp_sec": frame.get("timestamp_sec"),
                  "source_quality": dict(frame), "person_detected": None,
                  "full_body_visible": None, "within_capture_region": None,
                  "person_image_displacement": None,
                  "duplicate_status": ("not_rejected_by_extractor" if frame.get("diff_score") is not None
                                       else "unknown_no_comparison_score"),
                  "warnings": warnings}
        if pose is None:
            result["orientation"] = {"label": "unknown", "heuristic_score": None,
                                     "reason": "pose_metadata_missing"}
            warnings.append("pose_metadata_missing")
            buckets["unknown"] += 1
            records.append(result)
            continue
        detected = pose.get("person_detected")
        result["person_detected"] = detected if isinstance(detected, bool) else None
        result["full_body_visible"] = pose.get("full_body_visible")
        landmarks = pose.get("landmarks", {})
        names = ("nose", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                 "left_wrist", "right_wrist", "left_hip", "right_hip",
                 "left_knee", "right_knee", "left_ankle", "right_ankle")
        coords = [_xy(landmarks.get(name), visibility_threshold) for name in names]
        if any(point is not None and not (margin <= point[0] <= 1 - margin and margin <= point[1] <= 1 - margin) for point in coords):
            result["within_capture_region"] = False
        elif all(point is not None for point in coords):
            result["within_capture_region"] = True
        else:
            warnings.append("capture_region_unknown_missing_landmarks")
        if result["person_detected"] is False:
            warnings.append("person_not_detected")
        elif result["person_detected"] is None:
            warnings.append("person_detection_unknown")
        if result["full_body_visible"] is not True:
            warnings.append("full_body_visibility_unconfirmed")
        if result["within_capture_region"] is False:
            warnings.append("outside_capture_region")
        hips = [_xy(landmarks.get(name), visibility_threshold) for name in ("left_hip", "right_hip")]
        if all(p is not None for p in hips):
            center = ((hips[0][0] + hips[1][0]) / 2, (hips[0][1] + hips[1][1]) / 2)
            if reference is None:
                reference = center
            dx, dy = center[0] - reference[0], center[1] - reference[1]
            distance = math.hypot(dx, dy)
            result["person_image_displacement"] = {"dx": dx, "dy": dy, "distance": distance,
                                                    "reference": "first_accepted_frame_with_visible_hips",
                                                    "units": "normalized_image_coordinates"}
            if distance > displacement_threshold:
                warnings.append("excessive_image_displacement")
        else:
            warnings.append("person_displacement_unknown")
        try:
            raw = _classify_frame(pose)
        except (KeyError, TypeError, ValueError):
            raw = {"viewpoint": "unknown", "confidence": 0, "reason": "invalid_pose_data", "signals": {}}
        raw_label, reason, score = raw["viewpoint"], raw.get("reason"), raw["confidence"]
        # The current estimator's intermediate output specifically represents ambiguous side.
        if (raw_label not in buckets or score < orientation_threshold
                or reason == "side_view_but_left_right_ambiguous"):
            label = "unknown"
        else:
            label = raw_label
        result["orientation"] = {"label": label, "heuristic_score": score,
                                 "reason": reason, "raw_estimator": raw,
                                 "status": "unvalidated_heuristic"}
        if label == "unknown":
            warnings.append("orientation_unknown")
        buckets[label] += 1
        records.append(result)
    missing = [name for name in ("front", "left_side", "back", "right_side") if buckets[name] == 0]
    return {
        "camera": {"motion": "assumed_fixed_not_verified", "extrinsics": "not_estimated",
                   "calibration": "not_supplied"},
        "person_motion": "2D pelvis displacement proxy; not 3D translation or metric motion",
        "frames": records, "selection": selection,
        "extraction_rejected_frames": metadata.get("rejected_frames", []),
        "coverage": {"heuristic_counts": buckets, "missing_heuristic_buckets": missing,
                     "status": "unconfirmed",
                     "warning": "Viewpoint labels are unvalidated and cannot confirm 360-degree coverage."},
    }
