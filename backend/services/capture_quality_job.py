"""Background, opt-in capture quality reports using existing extracted frames."""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import threading
import time
import uuid
from pathlib import Path

import cv2

from backend.services.capture_authorization import require_authorization
from backend.services.capture_quality import QualityConfig, _iou, analyze_frames
from backend.services.pose_processor import get_pose_engine
from backend.services.video_processor import get_capture_engine

logger = logging.getLogger(__name__)
ROOT = Path(os.getenv("CAPTURE_QUALITY_OUTPUT_DIR", "outputs/capture_quality"))
_jobs = {}
_lock = threading.RLock()
NMS_IOU_THRESHOLD = 0.5


def _uuid(value):
    try:
        return str(uuid.UUID(value)) == value
    except (TypeError, ValueError, AttributeError):
        return False


def _hog_observations(frame_paths):
    """HOG is a weak full-person detector, not reliable identity verification."""
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    observations = {}
    for name, path in frame_paths.items():
        image = cv2.imread(str(path))
        if image is None:
            continue
        height, width = image.shape[:2]
        scale = min(1.0, 640.0 / width)
        scaled = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else image
        if scaled.shape[0] < 128 or scaled.shape[1] < 64:
            continue
        rectangles, weights = hog.detectMultiScale(scaled, winStride=(8, 8), padding=(8, 8), scale=1.05)
        sh, sw = scaled.shape[:2]
        observations[name] = [
            {"box": [float(x/sw), float(y/sh), float((x+w)/sw), float((y+h)/sh)],
             "score": float(weight), "method": "opencv_hog_uncalibrated"}
            for (x, y, w, h), weight in zip(rectangles, weights)
        ]
    return observations


def _nms_diagnostics(detections, iou_threshold=NMS_IOU_THRESHOLD):
    """Conservative IoU NMS for diagnostics only; input remains gate evidence.

    Indices refer to the original HOG-returned list, not sliding-window hits.
    Invalid records are retained in the raw evidence but excluded from NMS.
    """
    if detections is None:
        return None
    if not isinstance(detections, list):
        raise ValueError("Detections must be a list or None")
    if not isinstance(iou_threshold, (int, float)) or not 0 <= iou_threshold <= 1:
        raise ValueError("Invalid NMS IoU threshold")
    valid, invalid = [], []
    for index, detection in enumerate(detections):
        box = detection.get("box") if isinstance(detection, dict) else None
        score = detection.get("score") if isinstance(detection, dict) else None
        if (not isinstance(box, (list, tuple)) or len(box) != 4
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in box)
                or not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1)
                or isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score)):
            invalid.append(index)
        else:
            valid.append(index)
    ordered = sorted(valid, key=lambda i: (-detections[i]["score"], i))
    kept, suppressed = [], []
    for index in ordered:
        match = next(((other, _iou(detections[index]["box"], detections[other]["box"]))
                      for other in kept if _iou(detections[index]["box"], detections[other]["box"]) > iou_threshold), None)
        if match is None:
            kept.append(index)
        else:
            suppressed.append({"index": index, "retained_index": match[0], "iou": match[1]})
    return {
        "method": "score_ordered_iou_nms",
        "iou_threshold": float(iou_threshold),
        "raw_count": len(detections),
        "filtered_count": len(kept),
        "retained_indices": kept,
        "filtered_detections": [detections[i] for i in kept],
        "suppressed": suppressed,
        "invalid_indices": invalid,
    }


def start_quality(video_id, config=None, use_hog=True):
    if not _uuid(video_id):
        raise ValueError("Invalid video ID")
    require_authorization(video_id)
    capture = get_capture_engine()
    metadata = capture.get_metadata(video_id)
    if not metadata or metadata.get("status") != "completed":
        raise ValueError("Completed frame metadata not found")
    poses = get_pose_engine().get_summary(video_id)
    if not poses or poses.get("status") != "completed":
        raise ValueError("Completed pose summary required")
    frames = metadata.get("frames", [])
    if not frames:
        raise ValueError("No extracted frames")
    paths = {}
    for frame in frames:
        name = frame.get("filename")
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("Invalid extracted frame name")
        path = capture.get_frame_path(video_id, name)
        if path is None:
            raise ValueError(f"Extracted frame unavailable: {name}")
        paths[name] = path
    pose_records = poses.get("frames", [])
    pose_by_name = {}
    for pose in pose_records:
        name = pose.get("frame")
        if not isinstance(name, str) or name in pose_by_name:
            raise ValueError("Duplicate or invalid pose frame metadata; rerun pose processing")
        pose_by_name[name] = pose
    for name, path in paths.items():
        pose = pose_by_name.get(Path(name).stem)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if not pose or pose.get("source_sha256") != digest:
            raise ValueError(f"Pose data for {name} is missing or stale; rerun pose processing")
    policy = config or QualityConfig()
    run_id = str(uuid.uuid4())
    ROOT.mkdir(parents=True, exist_ok=True)
    output = ROOT / run_id
    output.mkdir(exist_ok=False)
    job = {"run_id": run_id, "video_id": video_id, "status": "processing", "created_at": time.time()}
    with _lock:
        _jobs[run_id] = job
    def worker():
        try:
            before = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}
            for name, digest in before.items():
                if pose_by_name[Path(name).stem].get("source_sha256") != digest:
                    raise ValueError(f"Pose data for {name} changed before quality analysis; rerun pose processing")
            try:
                observations = _hog_observations(paths) if use_hog else None
                detector_status = "opencv_hog_unvalidated" if use_hog else "disabled"
            except (cv2.error, ValueError, OSError) as exc:
                logger.warning("HOG quality check failed: %s", exc)
                observations, detector_status = None, "unavailable"
            frame_dir = next(iter(paths.values())).parent
            report = analyze_frames(metadata, poses, frame_dir, detections=observations, config=policy)
            for item in report["frames"]:
                name = item["filename"]
                if item.get("source_sha256") != before[name] or hashlib.sha256(paths[name].read_bytes()).hexdigest() != before[name]:
                    raise ValueError(f"Frame {name} changed during quality analysis; re-analyze")
                item["detector_nms"] = _nms_diagnostics(observations.get(name) if observations is not None else None)
            report["run_id"] = run_id
            report["detector_status"] = detector_status
            report["limitations"].append(
                "OpenCV HOG may miss small, occluded or partial bystanders; observed count is not a verified person count."
            )
            report["limitations"].append(
                "NMS is diagnostic only: overlapping people can be suppressed and nested duplicates can remain. "
                "Warnings and crop validation use the original HOG-returned candidates."
            )
            (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
            with _lock:
                job.update(status="completed", report_url=f"/api/capture-quality/runs/{run_id}/report")
        except Exception as exc:
            logger.exception("Capture quality analysis failed")
            with _lock:
                job.update(status="failed", detail=str(exc))
    response = dict(job)
    threading.Thread(target=worker, daemon=True).start()
    return response


def get_job(run_id):
    if not _uuid(run_id):
        raise ValueError("Invalid run ID")
    with _lock:
        job = _jobs.get(run_id)
        return dict(job) if job else None


def get_report(run_id):
    if not _uuid(run_id):
        raise ValueError("Invalid run ID")
    path = ROOT / run_id / "report.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
