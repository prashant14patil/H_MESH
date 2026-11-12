"""
backend/services/pose_processor.py
-------------------------------------
Module 3: Human detection, segmentation, and pose estimation.

Processes every frame previously extracted by Module 2
(backend/services/video_processor.py) from
data/captures/{video_id}/extracted_frames/, and writes structured
per-frame outputs (segmentation mask, pose visualization, metadata)
plus an overall processing_summary.json, ready for a future HMR2
multi-view reconstruction stage to consume.

Mirrors the architecture of video_processor.py / inference_engine.py:
singleton accessor, in-memory job tracking keyed by video_id,
background-thread processing.

Uses MediaPipe Tasks (bundled model assets, downloaded automatically
on first use):
    - PoseLandmarker (33-point body pose, with per-landmark visibility)
    - ImageSegmenter (category mask -- separates person from background)

Env vars (see .env.example):
    POSE_OUTPUT_DIRNAME   - subdirectory name under each video's capture
                             folder for pose outputs (default
                             "processed_frames").
    POSE_MIN_CONFIDENCE   - minimum pose-detection confidence to accept a
                             frame as "detected" (default 0.5).
    POSE_LANDMARK_VISIBILITY_THRESHOLD - minimum per-landmark visibility
                             score to count that landmark as visible
                             (default 0.5).

Public surface:
    PoseProcessingEngine, get_pose_engine(), PoseProcessingError
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_CAPTURE_ROOT = Path(os.getenv("CAPTURE_OUTPUT_DIR", "data/captures"))
_POSE_DIRNAME = os.getenv("POSE_OUTPUT_DIRNAME", "processed_frames")
_MIN_POSE_CONFIDENCE = float(os.getenv("POSE_MIN_CONFIDENCE", "0.5"))
_LANDMARK_VIS_THRESHOLD = float(os.getenv("POSE_LANDMARK_VISIBILITY_THRESHOLD", "0.5"))

# The 13 landmarks the task calls out explicitly, mapped to MediaPipe's
# PoseLandmarker landmark indices (BlazePose 33-point topology).
_TRACKED_LANDMARKS: dict[str, int] = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}

# Skeleton connections drawn in pose_visualization.jpg (pairs of the
# tracked-landmark names above).
_SKELETON_EDGES = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]


class PoseProcessingError(RuntimeError):
    """Raised when pose/segmentation processing cannot run or fails."""


class PoseProcessingEngine:
    """
    Loads MediaPipe PoseLandmarker + ImageSegmenter once (lazily, on
    first use) and runs them over every frame extracted for a video,
    writing structured per-frame outputs.
    """

    def __init__(self) -> None:
        self.jobs: dict[str, dict[str, Any]] = {}
        self._landmarker = None
        self._segmenter = None
        self._models_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    def _video_dir(self, video_id: str) -> Path:
        return _CAPTURE_ROOT / video_id

    def _frames_dir(self, video_id: str) -> Path:
        return self._video_dir(video_id) / "extracted_frames"

    def _pose_dir(self, video_id: str) -> Path:
        return self._video_dir(video_id) / _POSE_DIRNAME

    # ------------------------------------------------------------------
    # Lazy model loading (first call only -- mirrors InferenceEngine's
    # pattern of deferring heavy imports/loads until actually needed)
    # ------------------------------------------------------------------
    def _ensure_models_loaded(self) -> None:
        if self._landmarker is not None and self._segmenter is not None:
            return

        with self._models_lock:
            if self._landmarker is not None and self._segmenter is not None:
                return

            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            model_dir = Path(os.getenv("MEDIAPIPE_MODEL_DIR", "models/mediapipe"))
            model_dir.mkdir(parents=True, exist_ok=True)

            pose_model_path = model_dir / "pose_landmarker_lite.task"
            seg_model_path = model_dir / "selfie_segmenter.tflite"

            self._download_if_missing(
                pose_model_path,
                "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
            )
            self._download_if_missing(
                seg_model_path,
                "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite",
            )

            pose_options = mp_vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(pose_model_path)),
                running_mode=mp_vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=_MIN_POSE_CONFIDENCE,
                min_pose_presence_confidence=_MIN_POSE_CONFIDENCE,
                min_tracking_confidence=_MIN_POSE_CONFIDENCE,
            )
            self._landmarker = mp_vision.PoseLandmarker.create_from_options(pose_options)

            seg_options = mp_vision.ImageSegmenterOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(seg_model_path)),
                running_mode=mp_vision.RunningMode.IMAGE,
                output_category_mask=True,
            )
            self._segmenter = mp_vision.ImageSegmenter.create_from_options(seg_options)

            logger.info("MediaPipe PoseLandmarker + ImageSegmenter loaded.")

    @staticmethod
    def _download_if_missing(path: Path, url: str) -> None:
        if path.is_file() and path.stat().st_size > 0:
            return
        import urllib.request

        logger.info("Downloading MediaPipe model asset %s -> %s", url, path)
        try:
            urllib.request.urlretrieve(url, str(path))
        except Exception as exc:  # noqa: BLE001
            raise PoseProcessingError(
                f"Failed to download MediaPipe model asset from {url}: {exc}. "
                "Check your internet connection, or manually place the file "
                f"at {path}."
            ) from exc

    # ------------------------------------------------------------------
    # Job management
    # ------------------------------------------------------------------
    def get_job(self, video_id: str):
        return self.jobs.get(video_id)

    def get_summary(self, video_id: str):
        """Load processing_summary.json for a processed video, if present."""
        summary_path = self._pose_dir(video_id) / "processing_summary.json"
        if not summary_path.is_file():
            return None
        with open(summary_path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def get_frame_result_path(self, video_id: str, frame_name: str, asset: str):
        """
        Resolve a processed-frame asset (segmentation_mask.png,
        pose_visualization.jpg, or metadata.json) to its on-disk path.
        Path-traversal safe: only bare names are allowed.
        """
        safe_frame = Path(frame_name).name
        safe_asset = Path(asset).name
        path = self._pose_dir(video_id) / safe_frame / safe_asset
        if path.is_file():
            return path
        return None

    def process_video(self, video_id: str) -> None:
        """Kick off background pose/segmentation processing for a video."""
        frames_dir = self._frames_dir(video_id)
        if not frames_dir.is_dir():
            raise PoseProcessingError(
                f"No extracted frames found for video {video_id}. "
                "Run Module 2 frame extraction first (POST /api/capture/process/{video_id})."
            )

        existing = self.jobs.get(video_id)
        if existing and existing.get("status") == "processing":
            return

        self.jobs[video_id] = {"status": "processing"}

        def _worker():
            try:
                result = self._process_frames(video_id)
                self.jobs[video_id] = {**self.jobs.get(video_id, {}), **result}
            except Exception as exc:  # noqa: BLE001
                logger.exception("Pose processing failed for video %s", video_id)
                self.jobs[video_id] = {
                    **self.jobs.get(video_id, {}),
                    "status": "failed",
                    "detail": str(exc),
                }

        threading.Thread(target=_worker, daemon=True).start()

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------
    def _process_frames(self, video_id: str) -> dict[str, Any]:
        import cv2
        import numpy as np
        import mediapipe as mp

        self._ensure_models_loaded()

        started = time.time()
        frames_dir = self._frames_dir(video_id)
        frame_paths = sorted(frames_dir.glob("frame_*.jpg"))

        if not frame_paths:
            raise PoseProcessingError(f"No frame_*.jpg files found in {frames_dir}.")

        pose_dir = self._pose_dir(video_id)
        pose_dir.mkdir(parents=True, exist_ok=True)

        per_frame_results: list[dict[str, Any]] = []
        detected_count = 0
        complete_body_count = 0
        review_count = 0

        for frame_path in frame_paths:
            frame_name = frame_path.stem  # e.g. "frame_001"
            out_dir = pose_dir / frame_name
            out_dir.mkdir(parents=True, exist_ok=True)

            frame_bgr = cv2.imread(str(frame_path))
            if frame_bgr is None:
                per_frame_results.append({
                    "frame": frame_name,
                    "person_detected": False,
                    "status": "review_required",
                    "reason": "read_failed",
                })
                review_count += 1
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            # --- Pose estimation ------------------------------------------------
            pose_result = self._landmarker.detect(mp_image)
            has_pose = bool(pose_result.pose_landmarks)

            landmarks_out: dict[str, Any] = {}
            visible_count = 0
            if has_pose:
                lm_list = pose_result.pose_landmarks[0]
                for name, idx in _TRACKED_LANDMARKS.items():
                    lm = lm_list[idx]
                    visibility = float(getattr(lm, "visibility", 0.0) or 0.0)
                    is_visible = visibility >= _LANDMARK_VIS_THRESHOLD
                    if is_visible:
                        visible_count += 1
                    landmarks_out[name] = {
                        "x": round(float(lm.x), 4),
                        "y": round(float(lm.y), 4),
                        "z": round(float(lm.z), 4),
                        "visibility": round(visibility, 4),
                        "visible": is_visible,
                    }

            full_body_visible = visible_count == len(_TRACKED_LANDMARKS)
            if has_pose:
                detected_count += 1
                if full_body_visible:
                    complete_body_count += 1

            # --- Segmentation -----------------------------------------------------
            seg_result = self._segmenter.segment(mp_image)
            category_mask = seg_result.category_mask  # mp.Image, single channel
            mask_np = category_mask.numpy_view()  # uint8, 0=background 1..=person classes typically

            # Selfie segmenter: category 0 = background, category 1 = person.
            # Build a binary 0/255 mask for a clean PNG preview.
            person_mask = (mask_np > 0).astype("uint8") * 255

            mask_path = out_dir / "segmentation_mask.png"
            cv2.imwrite(str(mask_path), person_mask)

            foreground_ratio = float((person_mask > 0).mean())
            segmentation_ok = foreground_ratio > 0.02  # at least ~2% of frame is foreground

            # --- Pose visualization overlay ---------------------------------------
            vis = frame_bgr.copy()
            h, w = vis.shape[:2]
            if has_pose:
                points_px: dict[str, tuple[int, int]] = {}
                for name, data in landmarks_out.items():
                    px = int(data["x"] * w)
                    py = int(data["y"] * h)
                    points_px[name] = (px, py)
                    color = (0, 220, 0) if data["visible"] else (0, 165, 255)
                    cv2.circle(vis, (px, py), 5, color, -1)
                for a, b in _SKELETON_EDGES:
                    if a in points_px and b in points_px:
                        cv2.line(vis, points_px[a], points_px[b], (255, 200, 0), 2)

            vis_path = out_dir / "pose_visualization.jpg"
            cv2.imwrite(str(vis_path), vis, [cv2.IMWRITE_JPEG_QUALITY, 90])

            # --- Missing / obstructed region identification ------------------------
            missing_regions = [
                name for name, data in landmarks_out.items() if not data.get("visible")
            ] if has_pose else list(_TRACKED_LANDMARKS.keys())

            needs_review = (
                not has_pose
                or not segmentation_ok
                or len(missing_regions) > 3
            )
            if needs_review:
                review_count += 1

            frame_status = "ok" if (has_pose and segmentation_ok and not needs_review) else (
                "review_required" if needs_review else "partial"
            )

            frame_metadata = {
                "frame": frame_name,
                "person_detected": has_pose,
                "detection_confidence": round(
                    float(pose_result.pose_landmarks and 1.0 or 0.0), 3
                ) if not has_pose else round(
                    sum(d["visibility"] for d in landmarks_out.values()) / len(landmarks_out), 3
                ),
                "full_body_visible": full_body_visible if has_pose else False,
                "missing_regions": missing_regions,
                "landmarks": landmarks_out,
                "segmentation": {
                    "status": "ok" if segmentation_ok else "failed",
                    "foreground_ratio": round(foreground_ratio, 4),
                },
                "view_hint": self._estimate_view_hint(landmarks_out) if has_pose else "unknown",
                "status": frame_status,
            }

            meta_path = out_dir / "metadata.json"
            with open(meta_path, "w", encoding="utf-8") as fh:
                json.dump(frame_metadata, fh, indent=2)

            per_frame_results.append(frame_metadata)

        summary = {
            "video_id": video_id,
            "status": "completed",
            "total_frames": len(frame_paths),
            "frames_with_person_detected": detected_count,
            "frames_full_body_visible": complete_body_count,
            "frames_requiring_review": review_count,
            "frames": per_frame_results,
            "processing_time_sec": round(time.time() - started, 3),
            "notes": (
                "view_hint is a best-effort heuristic based on shoulder/hip "
                "landmark geometry only -- it is NOT a reliable front/side/back "
                "classifier on its own and should be combined with other "
                "signals (e.g. frame ordering/timestamp, or a dedicated "
                "viewpoint classifier) in a later module."
            ),
        }

        summary_path = pose_dir / "processing_summary.json"
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)

        logger.info(
            "Pose processing complete for video %s: %d/%d frames with person detected, "
            "%d requiring review (%.2fs)",
            video_id, detected_count, len(frame_paths), review_count,
            summary["processing_time_sec"],
        )

        return {
            "status": "completed",
            "frame_count": len(frame_paths),
            "summary_path": str(summary_path),
        }

    @staticmethod
    def _estimate_view_hint(landmarks: dict[str, Any]) -> str:
        """
        Best-effort front/side/back heuristic based on the horizontal
        separation between left/right shoulder and hip landmarks
        (normalized image coordinates). A front or back view has wide
        shoulder/hip separation; a side (profile) view has the two
        landmarks nearly overlapping horizontally. This CANNOT
        distinguish front from back on its own (both have wide
        separation) -- that requires additional signals such as nose
        position relative to the shoulder midpoint, frame ordering, or
        a dedicated viewpoint classifier in a later module.
        """
        ls = landmarks.get("left_shoulder")
        rs = landmarks.get("right_shoulder")
        if not ls or not rs or not ls.get("visible") or not rs.get("visible"):
            return "unknown"

        shoulder_width = abs(ls["x"] - rs["x"])
        if shoulder_width < 0.06:
            return "side"
        return "front_or_back"


# ---------------------------------------------------------------------------
# Process-wide singleton
# ---------------------------------------------------------------------------
_pose_engine: PoseProcessingEngine | None = None
_pose_engine_lock = threading.Lock()


def get_pose_engine() -> PoseProcessingEngine:
    """Return the process-wide PoseProcessingEngine, creating it on first call."""
    global _pose_engine
    if _pose_engine is None:
        with _pose_engine_lock:
            if _pose_engine is None:
                _pose_engine = PoseProcessingEngine()
    return _pose_engine
