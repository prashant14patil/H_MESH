"""
backend/services/viewpoint_estimator.py
-----------------------------------------
Module 2/3 bridge: derives a 360-degree viewpoint coverage estimate for
a video's extracted frames, re-using the pose/landmark data already
produced by backend/services/pose_processor.py (Module 3) -- it does
NOT run MediaPipe itself and does NOT modify pose_processor.py in any
way. This keeps the heuristic logic isolated and easy to replace later
with a dedicated orientation classifier without touching the pose
pipeline.

-----------------------------------------------------------------------
What this can and cannot reliably determine
-----------------------------------------------------------------------
pose_processor.py's PoseLandmarker gives us 2D (x, y, normalized) plus
a relative z (depth) and a visibility score for each of the 33
BlazePose landmarks, per frame. From the shoulder/hip landmarks alone:

  - Small shoulder separation can suggest a side profile, but depends on
    framing, pose, occlusion and landmark accuracy; it is not validated.

  - Front vs. back CANNOT be reliably distinguished from 2D (x, y)
    landmark positions alone, because a person facing the camera and
    a person with their back to the camera produce nearly identical
    shoulder/hip x-separation. We use the MediaPipe landmark `z`
    (relative depth, more negative = closer to camera) for the nose
    vs. shoulder-midpoint as a SECONDARY heuristic: a visible,
    forward-protruding nose (closer to camera than the shoulder
    midpoint) suggests a front-facing pose. This is NOT a validated
    classifier -- MediaPipe's monocular z estimates are noisy -- so we
    only use it to nudge a "front_or_back" classification toward
    "front" or "back" when the signal is strong, and otherwise leave
    it as "front_or_back" (ambiguous) with reduced confidence.

  - Left vs. right side (when in profile) is resolved using which
    shoulder/hip landmark has higher visibility (the far-side shoulder
    is usually occluded/less visible in a true profile shot).

Every classification carries a 0..1 confidence score and frames with
missing/undetected pose are labeled "unknown" with confidence 0 --
this module never claims an accurate front/back classification it
cannot support.

Public surface:
    ViewpointEstimationError, estimate_viewpoints(video_id) -> dict
"""

from __future__ import annotations

import logging
import math
from typing import Any

from backend.services.pose_processor import get_pose_engine

logger = logging.getLogger(__name__)

# Viewpoint buckets we attempt to fill for 360-degree coverage.
_VIEWPOINT_BUCKETS = ["front", "left_side", "back", "right_side", "intermediate"]

# Shoulder separation (normalized image width) below which we treat the
# pose as a side/profile view rather than front-or-back facing.
_SIDE_SHOULDER_WIDTH_THRESHOLD = 0.08

# Minimum |nose_z - shoulder_mid_z| to trust the front/back depth nudge.
# MediaPipe z is in roughly the same scale as x (hip-depth normalized),
# so small differences are noise.
_DEPTH_NUDGE_THRESHOLD = 0.03


# Left/right cues for profile frames (any one suffices if none conflict).
_SIDE_VISIBILITY_GAP = 0.05
# left_shoulder.z - right_shoulder.z; profile frames typically show ~0.5+.
_SIDE_DEPTH_GAP = 0.15
# Nose x offset from shoulder midpoint (normalized image width).
_SIDE_NOSE_OFFSET = 0.03

# Intermediate: shoulder width / torso height below this (a frontal pose
# is ~1.1-1.3) together with a clear shoulder depth asymmetry.
_INTERMEDIATE_WIDTH_RATIO = 0.85
_INTERMEDIATE_DEPTH_GAP = 0.2


def _finite(value) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _shoulder_depth_gap(left_shoulder: dict[str, Any], right_shoulder: dict[str, Any]):
    """left z - right z (negative => left shoulder nearer the camera)."""
    lz, rz = left_shoulder.get("z"), right_shoulder.get("z")
    if not (_finite(lz) and _finite(rz)):
        return None
    return lz - rz


def _torso_height(landmarks: dict[str, Any]):
    """Vertical shoulder-mid to hip-mid distance, or None if unavailable."""
    points = [landmarks.get(n) for n in ("left_shoulder", "right_shoulder", "left_hip", "right_hip")]
    if not all(isinstance(p, dict) and p.get("visible") and _finite(p.get("y")) for p in points):
        return None
    ls, rs, lh, rh = points
    height = abs((lh["y"] + rh["y"]) / 2.0 - (ls["y"] + rs["y"]) / 2.0)
    return height if height > 0.02 else None


class ViewpointEstimationError(RuntimeError):
    """Raised when viewpoint estimation cannot run (e.g. no pose data yet)."""


def _landmark(landmarks: dict[str, Any], name: str):
    data = landmarks.get(name)
    if not isinstance(data, dict) or not data.get("visible"):
        return None
    x = data.get("x")
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or not 0 <= x <= 1:
        return None
    return data


def _classify_frame(frame_result: dict[str, Any]) -> dict[str, Any]:
    """
    Classify a single frame's viewpoint from its pose_processor.py
    landmark output. Returns a dict with keys: viewpoint, confidence,
    signals (diagnostic info), reason (when unknown).
    """
    if not frame_result.get("person_detected"):
        return {
            "viewpoint": "unknown",
            "confidence": 0.0,
            "reason": "no_person_detected",
            "signals": {},
        }

    landmarks = frame_result.get("landmarks") or {}
    if not isinstance(landmarks, dict):
        landmarks = {}
    left_shoulder = _landmark(landmarks, "left_shoulder")
    right_shoulder = _landmark(landmarks, "right_shoulder")
    left_hip = _landmark(landmarks, "left_hip")
    right_hip = _landmark(landmarks, "right_hip")
    nose = _landmark(landmarks, "nose")

    if not left_shoulder or not right_shoulder:
        # Fall back to hips if shoulders are occluded.
        if left_hip and right_hip:
            left_shoulder, right_shoulder = left_hip, right_hip
        else:
            return {
                "viewpoint": "unknown",
                "confidence": 0.0,
                "reason": "shoulders_and_hips_not_visible",
                "signals": {},
            }

    shoulder_width = abs(left_shoulder["x"] - right_shoulder["x"])
    signals: dict[str, Any] = {"shoulder_width": round(shoulder_width, 4)}

    # --- Side vs front/back ---
    if shoulder_width < _SIDE_SHOULDER_WIDTH_THRESHOLD:
        # Profile view. Determine left vs right by comparing which
        # shoulder landmark had higher visibility (near-side shoulder
        # is less occluded than the far-side one in a true profile).
        left_vis = left_shoulder.get("visibility")
        right_vis = right_shoulder.get("visibility")
        valid_visibility = all(not isinstance(v, bool) and isinstance(v, (int, float))
                               and math.isfinite(v) and 0 <= v <= 1 for v in (left_vis, right_vis))
        if valid_visibility:
            signals["left_visibility"] = left_vis
            signals["right_visibility"] = right_vis
        vis_gap = abs(left_vis - right_vis) if valid_visibility else None

        # MediaPipe usually reports ~1.0 visibility for BOTH shoulders,
        # so visibility alone almost never separates left from right.
        # Combine it with two geometric cues and require agreement:
        #   1. shoulder depth: the near-camera shoulder has the smaller z;
        #   2. nose offset: the face points toward the visible side's
        #      direction (nose left of shoulder midpoint => left side).
        votes: list[str] = []
        if vis_gap is not None and vis_gap >= _SIDE_VISIBILITY_GAP:
            votes.append("right_side" if right_vis > left_vis else "left_side")

        depth_gap = _shoulder_depth_gap(left_shoulder, right_shoulder)
        if depth_gap is not None:
            signals["shoulder_depth_gap"] = round(depth_gap, 4)
            if abs(depth_gap) >= _SIDE_DEPTH_GAP:
                votes.append("left_side" if depth_gap < 0 else "right_side")

        if nose:
            shoulder_mid_x = (left_shoulder["x"] + right_shoulder["x"]) / 2.0
            nose_offset = nose["x"] - shoulder_mid_x
            signals["nose_offset_x"] = round(nose_offset, 4)
            if abs(nose_offset) >= _SIDE_NOSE_OFFSET:
                votes.append("left_side" if nose_offset < 0 else "right_side")

        signals["side_votes"] = votes
        if not votes or len(set(votes)) > 1:
            # No cue, or cues disagree -- too ambiguous to call.
            return {
                "viewpoint": "side_ambiguous",
                "confidence": round(0.4 + min(vis_gap or 0, 0.1), 2),
                "reason": ("side_view_but_left_right_ambiguous" if not votes
                           else "side_view_left_right_cues_conflict"),
                "signals": signals,
            }

        side = votes[0]
        confidence = round(min(0.45 + 0.12 * len(votes), 0.8), 2)
        return {
            "viewpoint": side,
            "confidence": confidence,
            "reason": "side_cues_agree",
            "signals": signals,
        }

    # --- Intermediate (roughly 30-60 degrees off front/back) ---
    # Shoulders are foreshortened relative to torso height AND clearly
    # at different depths. Torso height normalizes for subject distance.
    depth_gap = _shoulder_depth_gap(left_shoulder, right_shoulder)
    torso_height = _torso_height(landmarks)
    if depth_gap is not None and torso_height:
        width_ratio = shoulder_width / torso_height
        signals["shoulder_width_to_torso"] = round(width_ratio, 4)
        signals["shoulder_depth_gap"] = round(depth_gap, 4)
        if width_ratio < _INTERMEDIATE_WIDTH_RATIO and abs(depth_gap) >= _INTERMEDIATE_DEPTH_GAP:
            return {
                "viewpoint": "intermediate",
                "confidence": 0.45,
                "reason": "foreshortened_shoulders_with_depth_asymmetry",
                "signals": signals,
            }

    # --- Front or back: use nose depth relative to shoulder midpoint as a
    # secondary, low-confidence nudge. If unavailable/ambiguous, we
    # explicitly report "front_or_back" rather than guessing. ---
    base_confidence = round(min(0.4 + (shoulder_width - _SIDE_SHOULDER_WIDTH_THRESHOLD), 0.6), 2)

    depths = [nose.get("z") if nose else None, left_shoulder.get("z"), right_shoulder.get("z")]
    if all(not isinstance(z, bool) and isinstance(z, (int, float)) and math.isfinite(z) for z in depths):
        shoulder_mid_z = (left_shoulder["z"] + right_shoulder["z"]) / 2.0
        depth_delta = shoulder_mid_z - nose["z"]  # positive => nose closer to camera
        signals["nose_z"] = round(nose["z"], 4)
        signals["shoulder_mid_z"] = round(shoulder_mid_z, 4)
        signals["depth_delta"] = round(depth_delta, 4)

        if abs(depth_delta) >= _DEPTH_NUDGE_THRESHOLD:
            if depth_delta > 0:
                return {
                    "viewpoint": "front",
                    "confidence": round(min(base_confidence + 0.15, 0.75), 2),
                    "reason": "nose_depth_nudge_front",
                    "signals": signals,
                }
            else:
                return {
                    "viewpoint": "back",
                    "confidence": round(min(base_confidence + 0.1, 0.65), 2),
                    "reason": "nose_depth_nudge_back",
                    "signals": signals,
                }

    # No reliable front/back signal -- do NOT overclaim.
    return {
        "viewpoint": "front_or_back",
        "confidence": base_confidence,
        "reason": "front_back_not_distinguishable_from_available_landmarks",
        "signals": signals,
    }


def estimate_viewpoints(video_id: str) -> dict[str, Any]:
    """
    Build a per-frame viewpoint classification + an overall coverage
    summary for a video, using the pose_processor.py results already
    computed for its extracted frames (Module 3 must have been run).

    Raises ViewpointEstimationError if pose results are not yet
    available (caller should run POST /api/pose/process/{video_id}
    first and poll to completion).
    """
    pose_engine = get_pose_engine()
    summary = pose_engine.get_summary(video_id)

    if summary is None:
        job = pose_engine.get_job(video_id)
        status = job.get("status") if job else "not_started"
        raise ViewpointEstimationError(
            f"Pose processing results not available for video {video_id} "
            f"(status: {status}). Run POST /api/pose/process/{{video_id}} "
            "and wait for completion before requesting viewpoint coverage."
        )

    frames = summary.get("frames", [])
    per_frame: list[dict[str, Any]] = []
    coverage_counts: dict[str, int] = {b: 0 for b in _VIEWPOINT_BUCKETS}
    coverage_counts["front_or_back"] = 0
    coverage_counts["side_ambiguous"] = 0
    coverage_counts["unknown"] = 0

    for frame_result in frames:
        classification = _classify_frame(frame_result)
        viewpoint = classification["viewpoint"]
        coverage_counts[viewpoint] = coverage_counts.get(viewpoint, 0) + 1
        per_frame.append({
            "frame": frame_result.get("frame"),
            "viewpoint": viewpoint,
            "confidence": classification["confidence"],
            "reason": classification["reason"],
            "signals": classification["signals"],
        })

    # Coverage is considered satisfied for a bucket if at least one
    # frame was classified into it with reasonable confidence. Since
    # front/back often can't be disambiguated, "front_or_back" counts
    # toward satisfying EITHER the front or back requirement (whichever
    # is still missing) -- but we are explicit about this in the
    # reported notes rather than pretending we classified it.
    distinct_sides = sum(1 for b in ("left_side", "right_side") if coverage_counts.get(b, 0) > 0)
    front_back_observations = coverage_counts.get("front", 0) + coverage_counts.get("back", 0) + coverage_counts.get("front_or_back", 0)

    covered_buckets = [b for b in _VIEWPOINT_BUCKETS if coverage_counts.get(b, 0) > 0]
    missing_buckets = [b for b in _VIEWPOINT_BUCKETS if coverage_counts.get(b, 0) == 0]

    sufficient_rotation_coverage = (
        distinct_sides >= 2 and front_back_observations >= 1
    )

    coverage_summary = {
        "counts": coverage_counts,
        "covered_buckets": covered_buckets,
        "missing_buckets": missing_buckets,
        "distinct_sides_observed": distinct_sides,
        "front_or_back_observations": front_back_observations,
        "sufficient_rotation_coverage": sufficient_rotation_coverage,
        "status": "unconfirmed",
        "notes": (
            "All viewpoint labels are unvalidated heuristics; scores are not probabilities. "
            "Front/back cannot be reliably distinguished from pose "
            "landmarks alone; 'front_or_back' entries are counted "
            "toward general rotation coverage, not a confirmed "
            "front/back classification. This is a heuristic placeholder "
            "pending a dedicated orientation classifier."
        ),
    }

    return {
        "video_id": video_id,
        "frames": per_frame,
        "coverage": coverage_summary,
    }
