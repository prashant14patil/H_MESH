"""
backend/services/video_processor.py
-------------------------------------
Local 360 video capture + frame extraction engine (Module 2).

ROOT CAUSE OF THE ORIGINAL BUG:
Browser MediaRecorder output ("Record with camera") is WebM/VP8/VP9
with irregular keyframe timing. OpenCV/FFmpeg's CAP_PROP_FPS and
CAP_PROP_FRAME_COUNT are unreliable for such files and can return
degenerate values (observed: FPS=1000.0). The previous code trusted
these blindly to build frame-index sampling targets, which collapsed
to a single index once total_frames was wrong -- yielding exactly one
frame regardless of true video length.

FIX (see _resolve_video_timing / _extract_frames below):
1. Validate FPS/frame-count against sane bounds; fall back to a
   measured walk (grab() + CAP_PROP_POS_MSEC) when implausible.
2. Sample by wall-clock TIME (seconds), not raw frame index.
3. Blur/duplicate checks are independent & non-destructive.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_OUTPUT_ROOT = Path(os.getenv("CAPTURE_OUTPUT_DIR", "data/captures"))
_TARGET_FRAME_COUNT = int(os.getenv("TARGET_FRAME_COUNT", "30"))
_MAX_EXTRACTED_FRAMES = int(os.getenv("MAX_EXTRACTED_FRAMES", str(_TARGET_FRAME_COUNT)))
_MIN_FRAME_SPACING_SEC = float(os.getenv("MIN_FRAME_SPACING_SEC", "0.15"))
_BLUR_THRESHOLD = float(os.getenv("BLUR_THRESHOLD", "100.0"))
_DUPLICATE_THRESHOLD = float(os.getenv("DUPLICATE_THRESHOLD", "2.0"))
_DUPLICATE_WINDOW = max(1, int(os.getenv("DUPLICATE_WINDOW", "3")))

_FPS_SANE_MIN = 1.0
_FPS_SANE_MAX = 240.0

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
        meta_path = self._video_dir(video_id) / "metadata.json"
        if not meta_path.is_file():
            return None
        with open(meta_path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def get_frame_path(self, video_id: str, frame_name: str):
        safe_name = Path(frame_name).name
        frame_path = self._frames_dir(video_id) / safe_name
        if frame_path.is_file():
            return frame_path
        return None

    def process_video(self, video_id: str) -> None:
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

    @staticmethod
    def _measure_duration_by_walk(cap, fallback_fps: float):
        """Walk every frame via grab()+POS_MSEC to get a real duration."""
        import cv2

        frame_count = 0
        last_ts_msec = 0.0
        while True:
            ok = cap.grab()
            if not ok:
                break
            frame_count += 1
            ts_msec = cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0
            if ts_msec > 0:
                last_ts_msec = ts_msec

        if frame_count <= 0:
            return 0.0, 0
        if last_ts_msec > 0:
            duration_sec = last_ts_msec / 1000.0
        elif fallback_fps and _FPS_SANE_MIN <= fallback_fps <= _FPS_SANE_MAX:
            duration_sec = frame_count / fallback_fps
        else:
            duration_sec = frame_count / 30.0
        return duration_sec, frame_count

    def _resolve_video_timing(self, video_path: str) -> dict:
        """Return trustworthy fps/duration/total_frames/resolution."""
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise VideoProcessingError("OpenCV could not open video file '%s'." % video_path)
        try:
            reported_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            reported_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

            fps_sane = _FPS_SANE_MIN <= reported_fps <= _FPS_SANE_MAX
            header_duration = (reported_total / reported_fps) if (fps_sane and reported_fps > 0) else 0.0
            timing_source = "header"
            needs_fallback = (not fps_sane) or reported_total <= 1 or header_duration <= 0.05

            if needs_fallback:
                m_duration, m_count = self._measure_duration_by_walk(cap, reported_fps)
                if m_count > 0 and m_duration > 0:
                    duration_sec = m_duration
                    total_frames = m_count
                    fps = round(total_frames / duration_sec, 3) if duration_sec > 0 else (reported_fps if fps_sane else 30.0)
                    timing_source = "measured_walk"
                else:
                    duration_sec = header_duration
                    total_frames = reported_total
                    fps = reported_fps if fps_sane else 30.0
            else:
                duration_sec = header_duration
                total_frames = reported_total
                fps = reported_fps

            if duration_sec <= 0 or total_frames <= 0:
                raise VideoProcessingError(
                    "Could not determine a valid duration/frame-count for this "
                    "video. The file may be corrupt, empty, or use an "
                    "unsupported codec."
                )

            return {
                "fps": round(float(fps), 3),
                "duration_sec": round(float(duration_sec), 3),
                "total_frames": int(total_frames),
                "width": width,
                "height": height,
                "timing_source": timing_source,
                "reported_fps": round(float(reported_fps), 3),
                "reported_total_frames": reported_total,
            }
        finally:
            cap.release()

    @staticmethod
    def _sample_sequentially(cap, candidate_times, fps):
        """Decode forward to each target instead of seeking unreliable WebM timestamps."""
        import cv2
        import math

        frame_index = 0
        previous_ts = None
        exhausted = False
        for target in candidate_times:
            if exhausted:
                yield target, None, None
                continue
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    exhausted = True
                    yield target, None, None
                    break
                raw_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                fallback_ts = frame_index / fps
                frame_index += 1
                raw_ts = raw_ms / 1000.0 if math.isfinite(raw_ms) and raw_ms >= 0 else None
                if raw_ts is None or (previous_ts is not None and raw_ts <= previous_ts):
                    ts = max(fallback_ts, previous_ts or 0.0)
                else:
                    ts = raw_ts
                previous_ts = ts
                if ts + 1e-6 >= target:
                    yield target, ts, frame
                    break

    def _extract_frames(self, video_id: str, video_path: str) -> dict:
        """
        Time-based sampling pipeline: resolve trustworthy timing info,
        sample candidate timestamps evenly across the real duration,
        decode each, score blur + duplicate independently (both
        reported, neither one silently drops a frame on its own
        except duplicates), and persist accepted frames + metadata.json.
        """
        import cv2
        import numpy as np

        started = time.time()
        timing = self._resolve_video_timing(video_path)
        fps = timing["fps"]
        duration_sec = timing["duration_sec"]
        width = timing["width"]
        height = timing["height"]

        target_count = max(1, _TARGET_FRAME_COUNT)
        max_frames = max(1, _MAX_EXTRACTED_FRAMES)

        if target_count > 1 and duration_sec > 0:
            interval = duration_sec / target_count
            candidate_times = [round(i * interval, 3) for i in range(target_count)]
        else:
            candidate_times = [0.0]

        frames_dir = self._frames_dir(video_id)
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise VideoProcessingError("OpenCV could not re-open video file for extraction.")

        accepted: list[dict] = []
        rejected: list[dict] = []
        recent_accepted_gray: deque = deque(maxlen=_DUPLICATE_WINDOW)
        last_accepted_ts = None
        accepted_idx = 0

        try:
            for target_ts, decoded_ts, frame_bgr in self._sample_sequentially(cap, candidate_times, fps):
                if len(accepted) >= max_frames:
                    break

                if frame_bgr is None:
                    rejected.append({
                        "timestamp_sec": target_ts,
                        "reason": "read_failed",
                    })
                    continue

                ts = round(decoded_ts, 3)

                # Enforce minimum spacing between ACCEPTED frames (a
                # safeguard independent of the sampling interval math).
                if last_accepted_ts is not None and (ts - last_accepted_ts) < _MIN_FRAME_SPACING_SEC:
                    rejected.append({
                        "timestamp_sec": ts,
                        "reason": "too_close_to_previous_accepted",
                    })
                    continue

                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

                # --- Independent check 1: blur ---
                blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                is_blurry = blur_score < _BLUR_THRESHOLD

                # --- Independent check 2: near-duplicate vs rolling window ---
                is_duplicate = False
                best_diff_score = None
                for prev_gray in recent_accepted_gray:
                    diff_score = float(np.mean(cv2.absdiff(gray, prev_gray)))
                    if best_diff_score is None or diff_score < best_diff_score:
                        best_diff_score = diff_score
                    if diff_score < _DUPLICATE_THRESHOLD:
                        is_duplicate = True
                        break

                if is_duplicate:
                    rejected.append({
                        "timestamp_sec": ts,
                        "reason": "duplicate",
                        "diff_score": round(best_diff_score, 3) if best_diff_score is not None else None,
                    })
                    continue

                # Accept. Blurry frames are KEPT (flagged), since a
                # legitimate side/back view should not be discarded just
                # for being slightly soft -- only true duplicates or
                # unreadable frames are dropped from the result set.
                accepted_idx += 1
                fname = "frame_%03d.jpg" % accepted_idx
                cv2.imwrite(str(frames_dir / fname), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])

                accepted.append({
                    "filename": fname,
                    "timestamp_sec": ts,
                    "blur_score": round(blur_score, 2),
                    "is_blurry": is_blurry,
                    "diff_score": round(best_diff_score, 3) if best_diff_score is not None else None,
                })
                recent_accepted_gray.append(gray)
                last_accepted_ts = ts

            metadata = {
                "video_id": video_id,
                "status": "completed",
                "video_info": {
                    "fps": fps,
                    "width": width,
                    "height": height,
                    "duration_sec": duration_sec,
                    "timing_source": timing["timing_source"],
                    "reported_fps": timing["reported_fps"],
                    "reported_total_frames": timing["reported_total_frames"],
                },
                "target_frame_count": target_count,
                "max_extracted_frames": max_frames,
                "candidate_frame_count": len(candidate_times),
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
                "Extracted %d frames (%d rejected, timing_source=%s) for video %s in %.2fs",
                len(accepted), len(rejected), timing["timing_source"], video_id,
                metadata["processing_time_sec"],
            )

            return {
                "status": "completed",
                "frame_count": len(accepted),
                "metadata_path": str(meta_path),
            }
        finally:
            cap.release()


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
