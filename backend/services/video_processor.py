"""
backend/services/video_processor.py
-------------------------------------
Local 360 video capture + frame extraction engine.

Mirrors the architecture of backend/services/inference_engine.py
(singleton accessor, in-memory job tracking, background-thread
processing) for Module 2 of the Tanita pipeline: turning an uploaded
360 RGB video into a curated set of representative frames.

Env vars (see .env.example):
    CAPTURE_OUTPUT_DIR   - root dir for captured videos/frames (default "data/captures")
    TARGET_FRAME_COUNT   - desired extracted frame count (default 30, range 24-36)
    BLUR_THRESHOLD       - min Laplacian variance considered "sharp" (default 100.0)
    DUPLICATE_THRESHOLD  - mean abs pixel diff below which a frame is a near-duplicate (default 2.0)

Public surface:
    VideoCaptureEngine, get_capture_engine(), VideoProcessingError
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_OUTPUT_ROOT = Path(os.getenv("CAPTURE_OUTPUT_DIR", "data/captures"))
_TARGET_FRAME_COUNT = int(os.getenv("TARGET_FRAME_COUNT", "30"))
_BLUR_THRESHOLD = float(os.getenv("BLUR_THRESHOLD", "100.0"))
_DUPLICATE_THRESHOLD = float(os.getenv("DUPLICATE_THRESHOLD", "2.0"))

_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


class VideoProcessingError(RuntimeError):
    """Raised when a video cannot be loaded or processed."""


class VideoCaptureEngine:
    """Manages storage + frame extraction for uploaded 360 videos."""

    def __init__(self) -> None:
        self.jobs: dict[str, dict[str, Any]] = {}

    def _video_dir(self, video_id: str) -> Path:
        return _OUTPUT_ROOT / video_id

    def _frames_dir(self, video_id: str) -> Path:
        return self._video_dir(video_id) / "extracted_frames"

    def save_video(self, video_id: str, filename: str, data: bytes) -> Path:
        """Persist uploaded video bytes to disk and register the job."""
        video_dir = self._video_dir(video_id)
        video_dir.mkdir(parents=True, exist_ok=True)

        ext = Path(filename).suffix.lower() or ".mp4"
        video_path = video_dir / ("original" + ext)
        with open(video_path, "wb") as fh:
            fh.write(data)

        self.jobs[video_id] = {
            "status": "uploaded",
            "video_path": str(video_path),
            "original_filename": filename,
        }
        logger.info("Saved uploaded video %s -> %s (%d bytes)", video_id, video_path, len(data))
        return video_path

    def get_job(self, video_id: str):
        return self.jobs.get(video_id)

    def get_metadata(self, video_id: str):
        """Load metadata.json for a processed video, if present."""
        meta_path = self._video_dir(video_id) / "metadata.json"
        if not meta_path.is_file():
            return None
        with open(meta_path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def get_frame_path(self, video_id: str, frame_name: str):
        """Resolve a frame filename to its on-disk path (path-traversal safe)."""
        safe_name = Path(frame_name).name
        frame_path = self._frames_dir(video_id) / safe_name
        if frame_path.is_file():
            return frame_path
        return None

    def process_video(self, video_id: str) -> None:
        """Kick off background frame extraction for an uploaded video."""
        job = self.jobs.get(video_id)
        if job is None:
            raise VideoProcessingError("Video %s was not found. Upload it first." % video_id)

        if job.get("status") == "processing":
            return

        self.jobs[video_id] = dict(job, status="processing")
        video_path = job["video_path"]

        def _worker():
            try:
                result = self._extract_frames(video_id, video_path)
                self.jobs[video_id] = dict(self.jobs.get(video_id, {}), **result)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Frame extraction failed for video %s", video_id)
                self.jobs[video_id] = dict(
                    self.jobs.get(video_id, {}), status="failed", detail=str(exc)
                )

        threading.Thread(target=_worker, daemon=True).start()

    def _extract_frames(self, video_id: str, video_path: str) -> dict:
        """
        Core CV pipeline: load video, sample evenly spaced candidate
        frames, score each for blur (Laplacian variance) and reject
        near-duplicates (mean absolute grayscale frame difference vs
        the previously accepted frame), then persist accepted frames
        and a metadata.json summary.
        """
        import cv2
        import numpy as np

        started = time.time()
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise VideoProcessingError(
                "OpenCV could not open video file '%s'. File may be corrupt "
                "or use an unsupported codec." % video_path
            )

        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

            if fps <= 0 or total_frames <= 0:
                raise VideoProcessingError(
                    "Could not read valid FPS/frame-count metadata. "
                    "The video may be corrupt or empty."
                )

            duration_sec = total_frames / fps
            target_count = max(1, _TARGET_FRAME_COUNT)

            # Evenly spaced candidate indices across the full video.
            if target_count > 1 and total_frames > 1:
                step = (total_frames - 1) / (target_count - 1)
                candidate_indices = sorted({int(round(i * step)) for i in range(target_count)})
            else:
                candidate_indices = [0]

            frames_dir = self._frames_dir(video_id)
            if frames_dir.exists():
                shutil.rmtree(frames_dir)
            frames_dir.mkdir(parents=True, exist_ok=True)

            accepted = []
            rejected = []
            prev_gray = None
            accepted_idx = 0

            for frame_idx in candidate_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame_bgr = cap.read()
                if not ok or frame_bgr is None:
                    rejected.append({"source_frame_index": frame_idx, "reason": "read_failed"})
                    continue

                ts = round(frame_idx / fps, 3)
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

                # Blur score: Laplacian variance. Sharp edges -> high
                # variance in the 2nd-derivative response; blurry frames
                # have fewer sharp edges and thus lower variance.
                blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                is_blurry = blur_score < _BLUR_THRESHOLD

                # Duplicate score: mean absolute grayscale difference vs
                # the last *accepted* frame. A near-static camera/pose
                # produces a tiny mean difference -> skip it so the final
                # set covers distinct viewing angles instead of repeats.
                diff_score = None
                is_duplicate = False
                if prev_gray is not None:
                    diff_score = float(np.mean(cv2.absdiff(gray, prev_gray)))
                    is_duplicate = diff_score < _DUPLICATE_THRESHOLD

                if is_duplicate:
                    rejected.append({
                        "source_frame_index": frame_idx,
                        "timestamp_sec": ts,
                        "reason": "duplicate",
                        "diff_score": round(diff_score, 3),
                    })
                    continue

                accepted_idx += 1
                fname = "frame_%03d.jpg" % accepted_idx
                cv2.imwrite(str(frames_dir / fname), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])

                accepted.append({
                    "filename": fname,
                    "source_frame_index": frame_idx,
                    "timestamp_sec": ts,
                    "blur_score": round(blur_score, 2),
                    "is_blurry": is_blurry,
                    "diff_score": round(diff_score, 3) if diff_score is not None else None,
                })
                prev_gray = gray

            metadata = {
                "video_id": video_id,
                "status": "completed",
                "video_info": {
                    "fps": round(fps, 3),
                    "width": width,
                    "height": height,
                    "total_frames": total_frames,
                    "duration_sec": round(duration_sec, 3),
                },
                "target_frame_count": target_count,
                "accepted_frame_count": len(accepted),
                "rejected_frame_count": len(rejected),
                "frames": accepted,
                "rejected_frames": rejected,
                "processing_time_sec": round(time.time() - started, 3),
            }

            meta_path = self._video_dir(video_id) / "metadata.json"
            with open(meta_path, "w", encoding="utf-8") as fh:
                json.dump(metadata, fh, indent=2)

            logger.info(
                "Extracted %d frames (%d rejected) for video %s in %.2fs",
                len(accepted), len(rejected), video_id, metadata["processing_time_sec"],
            )

            return {
                "status": "completed",
                "frame_count": len(accepted),
                "metadata_path": str(meta_path),
            }
        finally:
            cap.release()


# ---------------------------------------------------------------------------
# Process-wide singleton
# ---------------------------------------------------------------------------
_capture_engine: VideoCaptureEngine | None = None
_capture_engine_lock = threading.Lock()


def get_capture_engine() -> VideoCaptureEngine:
    """Return the process-wide VideoCaptureEngine, creating it on first call."""
    global _capture_engine
    if _capture_engine is None:
        with _capture_engine_lock:
            if _capture_engine is None:
                _capture_engine = VideoCaptureEngine()
    return _capture_engine
