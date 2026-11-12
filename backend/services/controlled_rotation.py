"""Experimental fixed-camera sequence assessment; no physical yaw or reconstruction."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from backend.services.viewpoint_estimator import _classify_frame


def _point(landmarks, name, threshold):
    p = landmarks.get(name)
    if not isinstance(p, dict) or not p.get("visible"):
        return None
    try:
        values = tuple(float(p[k]) for k in ("x", "y", "z", "visibility"))
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in values) or not threshold <= values[3] <= 1:
        return None
    return values


def torso_proxy(landmarks, threshold=0.5, min_signal=0.02):
    """Signed x/depth torso-axis angle, NOT validated body yaw.

    Angles are relative to the MediaPipe left/right landmark convention.
    Relative monocular z and mirror handling are not calibrated.
    """
    pairs = []
    for left, right in (("left_shoulder", "right_shoulder"), ("left_hip", "right_hip")):
        a, b = _point(landmarks, left, threshold), _point(landmarks, right, threshold)
        if a is None or b is None:
            return None
        pairs.append((b[0] - a[0], b[2] - a[2], min(a[3], b[3])))
    dx = sum(p[0] for p in pairs) / 2
    dz = sum(p[1] for p in pairs) / 2
    if math.hypot(dx, dz) < min_signal:
        return None
    return {"degrees": math.degrees(math.atan2(dz, dx)) % 360,
            "heuristic_score": min(p[2] for p in pairs),
            "signal_x": dx, "signal_z": dz}


def analyze_proxy_sequence(angles, *, bins=8, max_step=90, min_span=300, min_bin_count=1):
    """Analyze adjacent proxy angles; missing values or jumps split trajectories."""
    if bins < 4 or max_step <= 0 or max_step >= 180 or min_span <= 0 or min_bin_count < 1:
        raise ValueError("Invalid proxy coverage thresholds")
    counts = [0] * bins
    smoothed = [None] * len(angles)
    unwrapped = [None] * len(angles)
    visits = []
    previous = None
    total = 0.0
    current = None
    segments = 0
    steps = []
    for i, angle in enumerate(angles):
        if angle is None or not math.isfinite(angle):
            previous = current = None
            continue
        angle %= 360
        bucket = int(((angle + 180 / bins) % 360) / (360 / bins))
        counts[bucket] += 1
        visits.append((i, bucket))
        if previous is None:
            segments += 1
            current = angle
        else:
            delta = (angle - previous + 180) % 360 - 180
            if abs(delta) > max_step:
                segments += 1
                current = angle
            else:
                current += delta
                total += delta
                steps.append(delta)
        unwrapped[i] = current
        previous = angle
    # Smooth only within valid adjacent samples and never interpolate missing values.
    for i, value in enumerate(unwrapped):
        if value is None:
            continue
        neighbors = [value]
        for j in (i - 1, i + 1):
            if 0 <= j < len(unwrapped) and unwrapped[j] is not None and abs(unwrapped[j] - value) <= max_step:
                neighbors.append(unwrapped[j])
        smoothed[i] = float(np.mean(neighbors) % 360)
    positive = sum(max(0, step) for step in steps)
    negative = sum(max(0, -step) for step in steps)
    direction = "unknown"
    if steps and max(positive, negative) >= 45 and min(positive, negative) <= 0.25 * max(positive, negative):
        direction = "increasing_proxy_angle" if positive > negative else "decreasing_proxy_angle"
    missing = [i for i, count in enumerate(counts) if count < min_bin_count]
    repeated = [i for i, count in enumerate(counts) if count > 1]
    return {"bin_counts": counts, "missing_bins": missing, "repeated_bins": repeated,
            "smoothed_degrees": smoothed, "unwrapped_degrees": unwrapped,
            "segments": segments, "net_proxy_degrees": total,
            "direction": direction,
            "proxy_complete": bool(not missing and segments == 1 and abs(total) >= min_span),
            "thresholds": {"bins": bins, "min_bin_count": min_bin_count,
                           "min_span_degrees": min_span, "max_step_degrees": max_step},
            "warning": "Proxy bins and direction are not validated anatomical viewpoints."}


def assess_sequence(metadata, pose_summary, *, calibration=None, person_counts=None,
                    visibility_threshold=0.5, margin=0.02, max_displacement=0.1):
    """Assess existing extracted frames; real capture suitability remains unconfirmed."""
    if not (0 <= visibility_threshold <= 1 and 0 <= margin < .5 and max_displacement > 0):
        raise ValueError("Invalid quality thresholds")
    frames = metadata.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("Extraction metadata must contain frames")
    poses = {}
    for pose in pose_summary.get("frames", []):
        name = pose.get("frame")
        if not isinstance(name, str) or name in poses:
            raise ValueError("Invalid or duplicate pose frame")
        poses[name] = pose
    camera = {"status": "pending", "fixed_camera": "assumed_not_verified"}
    if calibration is not None:
        from backend.services.checkerboard_calibration import validate_config
        info = metadata.get("video_info", {})
        validate_config(calibration, (info.get("width"), info.get("height")))
        camera["provenance"] = calibration["provenance"]
        camera["screening"] = calibration.get("quality_assessment", {}).get("status", "unknown")
        camera["status"] = ("synthetic_not_real" if calibration["provenance"] == "synthetic"
                            else "independent_validation_pending")
    counts = person_counts or {}
    ordered = sorted(frames, key=lambda f: (float(f["timestamp_sec"]), f["filename"]))
    if any(not math.isfinite(float(f["timestamp_sec"])) for f in ordered):
        raise ValueError("Invalid frame timestamp")
    output, angles = [], []
    reference = None
    for index, frame in enumerate(ordered):
        name = frame["filename"]
        if Path(name).name != name:
            raise ValueError("Invalid frame filename")
        pose = poses.get(Path(name).stem, {})
        landmarks = pose.get("landmarks") or {}
        visible = {k: v for k, v in landmarks.items() if _point(landmarks, k, visibility_threshold)}
        coords = [_point(landmarks, k, visibility_threshold) for k in visible]
        box = ([min(p[0] for p in coords), min(p[1] for p in coords),
                max(p[0] for p in coords), max(p[1] for p in coords)] if coords else None)
        hips = [_point(landmarks, key, visibility_threshold) for key in ("left_hip", "right_hip")]
        center = [(hips[0][j] + hips[1][j]) / 2 for j in range(2)] if all(hips) else None
        if reference is None and center is not None:
            reference = center
        displacement = ([center[j] - reference[j] for j in range(2)] if center is not None and reference else None)
        proxy = torso_proxy(landmarks, visibility_threshold)
        count = counts.get(name)
        if count is not None and (type(count) is not int or count < 0):
            raise ValueError("Person count must be a nonnegative integer or absent")
        reasons = []
        if pose.get("person_detected") is not True:
            reasons.append("person_not_detected_or_unknown")
        if count is None:
            reasons.append("person_count_unknown")
        elif count != 1:
            reasons.append("person_count_not_one")
        if pose.get("full_body_visible") is not True:
            reasons.append("full_body_unconfirmed")
        if not box or box[0] < margin or box[1] < margin or box[2] > 1-margin or box[3] > 1-margin:
            reasons.append("landmark_bounds_missing_or_outside")
        if frame.get("is_blurry") is not False:
            reasons.append("blurry_or_unknown")
        if displacement is None:
            reasons.append("movement_unknown")
        elif math.hypot(*displacement) > max_displacement:
            reasons.append("excessive_image_movement")
        if proxy is None:
            reasons.append("orientation_proxy_unknown")
        try:
            hint = _classify_frame(pose)
        except (KeyError, TypeError, ValueError):
            hint = {"viewpoint": "unknown", "confidence": 0, "reason": "invalid_pose_data", "signals": {}}
        angles.append(proxy["degrees"] if proxy is not None and not reasons else None)
        output.append({
            "frame_number": index + 1, "frame_number_kind": "extracted_sequence_index",
            "filename": name, "timestamp_sec": frame["timestamp_sec"],
            "person_detected": pose.get("person_detected"), "person_count": count,
            "landmarks": landmarks, "visible_landmark_count": len(visible),
            "landmark_bbox_normalized": box, "full_body_visible": pose.get("full_body_visible"),
            "blur_score": frame.get("blur_score"), "is_blurry": frame.get("is_blurry"),
            "extractor_diff_score": frame.get("diff_score"), "orientation_proxy": proxy,
            "viewpoint_hint": hint, "position_delta_normalized": displacement,
            "bbox_height_normalized": box[3] - box[1] if box else None,
            "suitable_for_further_processing": not reasons, "quality_reasons": reasons,
        })
    summary = analyze_proxy_sequence(angles)
    for item, smooth in zip(output, summary["smoothed_degrees"]):
        item["smoothed_orientation_proxy_degrees"] = smooth
    return {"camera": camera, "frames": output, "proxy_coverage": summary,
            "extraction_rejections": metadata.get("rejected_frames", []),
            "capture_completeness": "unconfirmed",
            "readiness": "review_required",
            "limitations": ["Torso x/z angle and viewpoint labels are unvalidated heuristics, not physical yaw.",
                            "Single-pose processing cannot exclude multiple people.",
                            "Image position and size changes do not establish metric 3D movement.",
                            "Fixed camera is assumed, not verified."]}
