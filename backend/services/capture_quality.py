"""Additive capture quality diagnostics; does not change extraction or inference.

Detector evidence is optional and must be supplied separately. Detector scores,
viewpoint hints and image thresholds are uncalibrated screening signals.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import hashlib
from pathlib import Path

import cv2
import numpy as np

from backend.services.viewpoint_estimator import _classify_frame
from backend.services.pose_processor import _TRACKED_LANDMARKS


@dataclass(frozen=True)
class QualityConfig:
    min_laplacian: float = 100.0
    min_roi_laplacian: float = 30.0
    min_gradient: float = 5.0
    min_brightness: float = 35.0
    max_brightness: float = 220.0
    max_clipped_fraction: float = 0.35
    min_landmark_visibility: float = 0.5
    boundary_margin: float = 0.02
    min_identity_iou: float = 0.15
    reject_blurry: bool = False

    def __post_init__(self):
        values = asdict(self)
        if any(not isinstance(v, (int, float)) or not math.isfinite(v)
               for k, v in values.items() if k != "reject_blurry"):
            raise ValueError("Quality thresholds must be finite numbers")
        if (min(self.min_laplacian, self.min_roi_laplacian, self.min_gradient) < 0
                or not 0 <= self.min_brightness < self.max_brightness <= 255
                or not 0 <= self.max_clipped_fraction <= 1
                or not 0 <= self.min_landmark_visibility <= 1
                or not 0 <= self.boundary_margin < 0.5
                or not 0 <= self.min_identity_iou <= 1
                or not isinstance(self.reject_blurry, bool)):
            raise ValueError("Invalid quality thresholds")


def _landmark_box(landmarks, config):
    points = []
    for value in landmarks.values():
        try:
            x, y, visibility = (float(value[k]) for k in ("x", "y", "visibility"))
        except (KeyError, TypeError, ValueError):
            continue
        if (all(math.isfinite(v) for v in (x, y, visibility))
                and visibility >= config.min_landmark_visibility and 0 <= x <= 1 and 0 <= y <= 1):
            points.append((x, y))
    if len(points) < 4:
        return None
    xy = np.asarray(points)
    return [float(xy[:, 0].min()), float(xy[:, 1].min()),
            float(xy[:, 0].max()), float(xy[:, 1].max())]


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    area_b = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


RCNN_METHOD = "torchvision_keypoint_rcnn"
# COCO-17 keypoint indices used for the detector full-body evidence.
_HEAD, _SHOULDERS, _HIPS, _ANKLES = (0, 1, 2, 3, 4), (5, 6), (11, 12), (15, 16)


def _paired(name):
    if name.startswith("left_"):
        return "right_" + name[5:]
    if name.startswith("right_"):
        return "left_" + name[6:]
    return None


def _one_sided_self_occlusion(landmarks, detection, config):
    """Full body inside the frame although some landmarks are hidden behind the body.

    MediaPipe 'visibility' means 'not occluded', so a profile view (far arm/leg
    behind the torso) fails the strict all-landmarks-visible rule even when the
    whole body is in frame. Accept it only when ALL of these hold:
      * every tracked landmark position lies inside the frame margins;
      * each low-visibility limb landmark has its opposite-side partner visible
        (only one side is hidden, i.e. self-occlusion, not truncation);
      * an independent Keypoint R-CNN detection of the same person has its box
        inside the frame and confident head, shoulder, hip and ankle keypoints.
    """
    margin = config.boundary_margin
    for name in _TRACKED_LANDMARKS:
        lm = landmarks.get(name)
        try:
            x, y, vis = (float(lm[k]) for k in ("x", "y", "visibility"))
        except (KeyError, TypeError, ValueError):
            return False
        if not all(math.isfinite(v) for v in (x, y, vis)) or not (margin <= x <= 1 - margin and margin <= y <= 1 - margin):
            return False
        if vis < config.min_landmark_visibility or lm.get("visible") is not True:
            partner = landmarks.get(_paired(name) or "")
            try:
                if partner is None or float(partner["visibility"]) < config.min_landmark_visibility:
                    return False
            except (KeyError, TypeError, ValueError):
                return False
    box = detection.get("box")
    conf = detection.get("keypoint_confidence")
    if not isinstance(box, (list, tuple)) or len(box) != 4 or not isinstance(conf, list) or len(conf) != 17:
        return False
    if not (box[0] >= margin / 2 and box[1] >= margin / 2 and box[2] <= 1 - margin / 2 and box[3] <= 1 - margin / 2):
        return False
    return all(max(conf[i] for i in group) >= 0.5 for group in (_HEAD, _SHOULDERS, _HIPS, _ANKLES))


def _features(image, box):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    roi = gray
    if box:
        x0, y0 = max(0, int(box[0]*w)), max(0, int(box[1]*h))
        x1, y1 = min(w, int(math.ceil(box[2]*w))), min(h, int(math.ceil(box[3]*h)))
        if x1 > x0 and y1 > y0:
            roi = gray[y0:y1, x0:x1]
    dx = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
    return {
        "frame_laplacian_variance": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "roi_laplacian_variance": float(cv2.Laplacian(roi, cv2.CV_64F).var()),
        "roi_mean_gradient": float(np.sqrt(dx * dx + dy * dy).mean()),
        "roi_mean_brightness": float(roi.mean()),
        "roi_clipped_fraction": float(((roi <= 5) | (roi >= 250)).mean()),
        "roi_source": "visible_landmark_extent" if box else "entire_frame_fallback",
    }


def analyze_frames(metadata, poses, frame_dir, *, detections=None, config=None, detector_method=None):
    """Analyze all extracted frames; optional detections keyed by filename.

    Detection records contain normalized ``box`` [x1,y1,x2,y2] and optional
    uncalibrated ``score``. No match or detector failure is unknown, never a
    verified single person. Continuity compares pose-box overlap across frames;
    it cannot establish identity or prevent a subject switch on its own.
    """
    config = config or QualityConfig()
    frames = metadata.get("frames", [])
    pose_by_name = {p.get("frame"): p for p in (poses or {}).get("frames", [])}
    detections = detections or {}
    results = []
    previous_box = None
    coverage = {}
    for frame in frames:
        name = frame.get("filename", "")
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("Invalid frame filename")
        pose = pose_by_name.get(Path(name).stem, {})
        landmarks = pose.get("landmarks") or {}
        box = _landmark_box(landmarks, config)
        warnings, rejections = [], []
        try:
            image_bytes = (Path(frame_dir) / name).read_bytes()
            image_hash = hashlib.sha256(image_bytes).hexdigest()
            image = (cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                     if image_bytes else None)
        except (OSError, cv2.error):
            image_hash, image = None, None
        features = _features(image, box) if image is not None else None
        if image is None:
            rejections.append("frame_unreadable")
        if pose.get("person_detected") is False:
            rejections.append("person_not_detected")
        elif pose.get("person_detected") is not True:
            warnings.append("person_detection_unknown")
        required_in_bounds = True
        for landmark_name in _TRACKED_LANDMARKS:
            landmark = landmarks.get(landmark_name)
            try:
                x, y, visibility = (float(landmark[key]) for key in ("x", "y", "visibility"))
                valid = (landmark.get("visible") is True and
                         all(math.isfinite(v) for v in (x, y, visibility)) and
                         visibility >= config.min_landmark_visibility and
                         config.boundary_margin <= x <= 1 - config.boundary_margin and
                         config.boundary_margin <= y <= 1 - config.boundary_margin)
            except (KeyError, TypeError, ValueError):
                valid = False
            if not valid:
                required_in_bounds = False
                break
        modern = detector_method == RCNN_METHOD
        evidence_list = detections.get(name) if isinstance(detections.get(name), list) else []
        full_body_basis = None
        if pose.get("full_body_visible") is True and box is not None and required_in_bounds:
            full_body_basis = "pose_all_required_landmarks_visible"
        elif box is not None and modern and evidence_list:
            matched = max(evidence_list, key=lambda d: _iou(box, d.get("box") or [0, 0, 0, 0]))
            if (_iou(box, matched.get("box") or [0, 0, 0, 0]) >= config.min_identity_iou
                    and _one_sided_self_occlusion(landmarks, matched, config)):
                full_body_basis = "in_frame_one_sided_self_occlusion_detector_confirmed"
                warnings.append("full_body_self_occlusion_one_side")
        if full_body_basis is None:
            warnings.append("full_body_visibility_unconfirmed")
        if frame.get("is_blurry") is not False:
            (rejections if config.reject_blurry else warnings).append("extractor_blur_warning")
        if features:
            if features["frame_laplacian_variance"] < config.min_laplacian:
                warnings.append("frame_sharpness_review")
            if (features["roi_laplacian_variance"] < config.min_roi_laplacian
                    or features["roi_mean_gradient"] < config.min_gradient):
                warnings.append("roi_sharpness_review")
            if (features["roi_mean_brightness"] < config.min_brightness
                    or features["roi_mean_brightness"] > config.max_brightness
                    or features["roi_clipped_fraction"] > config.max_clipped_fraction):
                warnings.append("lighting_review")
        evidence = detections.get(name)
        identity = "unknown"
        count = None
        overlap = None
        if isinstance(evidence, list):
            valid = [d for d in evidence if isinstance(d, dict) and isinstance(d.get("box"), (list, tuple))
                     and len(d["box"]) == 4 and all(isinstance(x, (int, float)) and math.isfinite(x) for x in d["box"])]
            count = len(valid)
            if count > 1:
                warnings.append("possible_multiple_people")
            if box and valid:
                scores = sorted((_iou(box, d["box"]) for d in valid), reverse=True)
                overlap = scores[0]
                if overlap >= config.min_identity_iou and (len(scores) == 1 or scores[0] - scores[1] > 0.1):
                    identity = "tentative_pose_detector_match"
                else:
                    warnings.append("subject_association_ambiguous")
            else:
                warnings.append("subject_association_unknown")
        else:
            warnings.append("person_count_unknown")
            warnings.append("subject_association_unknown")
        continuity = _iou(previous_box, box) if previous_box and box else None
        if continuity is not None and continuity < config.min_identity_iou:
            warnings.append("subject_continuity_uncertain")
            identity = "unknown"
        if box:
            previous_box = box
        try:
            hint = _classify_frame(pose)
        except (KeyError, ValueError, TypeError):
            hint = {"viewpoint": "unknown", "confidence": 0.0, "reason": "invalid_or_missing_pose", "signals": {}}
        raw_label = hint.get("viewpoint", "unknown")
        label = "unknown" if raw_label == "front_or_back" or hint.get("reason") == "side_view_but_left_right_ambiguous" else raw_label
        coverage[label] = coverage.get(label, 0) + 1
        # A single detector match is not a verified identity; false negatives remain possible.
        warnings.append("person_count_and_identity_unverified")
        status = "rejected" if rejections else "review" if warnings else "acceptable"
        results.append({
            "filename": name, "timestamp_sec": frame.get("timestamp_sec"),
            "source_sha256": image_hash,
            "status": status, "warnings": sorted(set(warnings)), "rejections": rejections,
            "checks": {
                "readability": "pass" if image is not None else "fail",
                "person_presence": "pass" if pose.get("person_detected") is True else ("fail" if pose.get("person_detected") is False else "unknown"),
                # Only the keypoint detector can support "one person"; HOG counts stay unknown.
                "multiple_people": ("fail" if count is not None and count > 1
                                    else "pass" if modern and count == 1 else "unknown"),
                "identity": "unknown",
                "sharpness": "unknown" if not features or frame.get("is_blurry") is None else ("warning" if frame.get("is_blurry") or "frame_sharpness_review" in warnings or "roi_sharpness_review" in warnings else "pass"),
                "lighting": "unknown" if not features else ("warning" if "lighting_review" in warnings else "pass"),
                "full_body": "pass" if full_body_basis else "unknown",
                "duplicate": "unknown" if frame.get("diff_score") is None else "warning",
                "continuity": "unknown" if continuity is None else ("warning" if "subject_continuity_uncertain" in warnings else "unknown"),
                "viewpoint": "unknown",
            },
            "recommended_for_reconstruction": False,
            "extractor_quality": {"is_blurry": frame.get("is_blurry"), "blur_score": frame.get("blur_score"),
                                  "duplicate_difference": frame.get("diff_score")},
            "image_metrics": features, "person_detected": pose.get("person_detected"),
            "person_detection_confidence": (max(float(d.get("score", 0)) for d in evidence_list)
                                            if modern and evidence_list else None),
            "detector_method": detector_method, "full_body_basis": full_body_basis,
            "mean_landmark_visibility": pose.get("detection_confidence"),
            "landmarks": landmarks, "full_body_visible": pose.get("full_body_visible"),
            "missing_regions": pose.get("missing_regions"), "landmark_bbox_normalized": box,
            "person_count_observed": count, "detector_evidence": evidence,
            "subject_association": identity, "pose_detector_overlap": overlap,
            "previous_pose_box_iou": continuity,
            "background_interference": "unknown",
            "obstruction_status": "unknown",
            "viewpoint": {"label": label, "raw_label": raw_label, "heuristic_score": hint.get("confidence"),
                          "reason": hint.get("reason"), "signals": hint.get("signals"),
                          "method": "existing shoulder-width and nose-depth heuristic", "validated": False},
        })
    rejected = metadata.get("rejected_frames", [])
    return {
        "schema_version": 1, "video_id": metadata.get("video_id"), "thresholds": asdict(config),
        "summary": {
            "extracted_frames": len(results),
            "extractor_non_blurry": sum(f["extractor_quality"]["is_blurry"] is False for f in results),
            "extractor_blurry": sum(f["extractor_quality"]["is_blurry"] is True for f in results),
            "rejected_duplicates": sum(r.get("reason") == "duplicate" for r in rejected),
            "full_body_visible": sum(f["full_body_visible"] is True for f in results),
            "possible_multiple_people": sum("possible_multiple_people" in f["warnings"] for f in results),
            "single_person_detected": sum(f["checks"]["multiple_people"] == "pass" for f in results),
            "detector_method": detector_method,
            "recommended_frames": [f["filename"] for f in results if f["recommended_for_reconstruction"]],
            "status_counts": {s: sum(f["status"] == s for f in results) for s in ("acceptable", "review", "rejected")},
        },
        "viewpoint_coverage": {"counts": coverage, "status": "unconfirmed",
                               "note": "Heuristic labels are unvalidated; score is not a probability or reconstruction accuracy."},
        "rejected_candidates": rejected, "frames": results,
        "limitations": ["Single-pose output cannot rule out bystanders.",
                        "Detector and subject identity evidence are not validated.",
                        "Lighting and sharpness thresholds are screening heuristics.",
                        "Physical yaw and 360-degree completeness are not established."],
    }
