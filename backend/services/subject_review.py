"""Explicit human subject review; no biometric identification."""
from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
import uuid
from pathlib import Path

from backend.services.capture_authorization import require_authorization
from backend.services.capture_quality_job import get_report
from backend.services.video_processor import get_capture_engine

ROOT = Path(os.getenv("SUBJECT_REVIEW_OUTPUT_DIR", "outputs/subject_reviews"))
_LOCK = threading.RLock()


def valid_id(value):
    try:
        return str(uuid.UUID(value)) == value
    except (ValueError, TypeError, AttributeError):
        return False


def validate_box(value):
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("Subject box must contain four normalized coordinates")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in value):
        raise ValueError("Subject box coordinates must be finite numbers")
    x1, y1, x2, y2 = map(float, value)
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError("Subject box must be inside the image with positive area")
    if (x2-x1)*(y2-y1) < 0.005:
        raise ValueError("Subject box is too small for review")
    return [x1, y1, x2, y2]


def validate_review_crop(box, frame):
    """Reject crops containing multiple detected candidate centers."""
    box = validate_box(box)
    try:
        landmark_box = validate_box(frame.get("landmark_bbox_normalized"))
    except ValueError as exc:
        raise ValueError("Pose landmark extent is unavailable") from exc
    if not (box[0] <= landmark_box[0] and box[1] <= landmark_box[1]
            and box[2] >= landmark_box[2] and box[3] >= landmark_box[3]):
        raise ValueError("Reviewed box must contain the pose landmark extent used for quality checks")
    inside = 0
    for detection in frame.get("detector_evidence") or []:
        try:
            candidate = validate_box(detection["box"])
        except (TypeError, KeyError, ValueError):
            continue
        x = (candidate[0] + candidate[2]) / 2
        y = (candidate[1] + candidate[3]) / 2
        if box[0] <= x <= box[2] and box[1] <= y <= box[3]:
            inside += 1
    if inside > 1:
        raise ValueError("Reviewed crop includes multiple detected people; reject or redraw")
    return box


def _iou(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def suggest(reference_box, detections):
    """Suggest candidates only; no suggestion is an approval."""
    reference_box = validate_box(reference_box)
    candidates = []
    for detection in detections or []:
        try:
            box = validate_box(detection["box"])
        except (TypeError, KeyError, ValueError):
            continue
        overlap = _iou(reference_box, box)
        distance = math.hypot((box[0]+box[2]-reference_box[0]-reference_box[2])/2,
                              (box[1]+box[3]-reference_box[1]-reference_box[3])/2)
        candidates.append({"box": box, "overlap": overlap, "center_distance": distance})
    candidates.sort(key=lambda x: (-x["overlap"], x["center_distance"]))
    ambiguous = (not candidates or candidates[0]["overlap"] < 0.15 or
                 (len(candidates) > 1 and candidates[0]["overlap"]-candidates[1]["overlap"] < 0.1))
    return {"status": "ambiguous" if ambiguous else "suggested_unconfirmed",
            "candidates": candidates, "method": "box overlap and normalized center distance; no identity verification"}


def _report(quality_run_id, video_id):
    if not valid_id(quality_run_id) or not valid_id(video_id):
        raise ValueError("Invalid recording or quality report identifier")
    require_authorization(video_id)
    report = get_report(quality_run_id)
    if not report or report.get("run_id") != quality_run_id or report.get("video_id") != video_id:
        raise ValueError("Quality report unavailable or does not match recording")
    return report


def _frame(report, video_id, filename):
    if not isinstance(filename, str) or Path(filename).name != filename or "/" in filename or "\\" in filename:
        raise ValueError("Invalid frame filename")
    frames = [f for f in report.get("frames", []) if f.get("filename") == filename]
    if len(frames) != 1:
        raise ValueError("Frame is absent or ambiguous in report")
    path = get_capture_engine().get_frame_path(video_id, filename)
    if path is None or not path.is_file():
        raise ValueError("Frame unavailable")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if not frames[0].get("source_sha256") or digest != frames[0]["source_sha256"]:
        raise ValueError("Frame changed since quality analysis; re-analyze")
    return frames[0], digest


def create_session(video_id, quality_run_id, reference_frame, bbox):
    report = _report(quality_run_id, video_id)
    _, digest = _frame(report, video_id, reference_frame)
    box = validate_box(bbox)
    authorization = require_authorization(video_id)
    session_id = str(uuid.uuid4())
    root = ROOT / session_id
    root.mkdir(parents=True, exist_ok=False)
    session = {"session_id": session_id, "video_id": video_id, "quality_run_id": quality_run_id,
               "authorization_source_sha256": authorization["source_sha256"],
               "reference_frame": reference_frame, "reference_frame_sha256": digest,
               "reference_box": box, "created_at": time.time(),
               "identity_note": "Human-selected subject; not biometric identity verification."}
    (root / "session.json").write_text(json.dumps(session, indent=2, allow_nan=False), encoding="utf-8")
    return session


def get_session(session_id):
    if not valid_id(session_id):
        raise ValueError("Invalid review session identifier")
    path = ROOT / session_id / "session.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def get_decision(session_id, filename):
    session = get_session(session_id)
    if not session:
        raise ValueError("Review session not found")
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ValueError("Invalid frame filename")
    path = ROOT / session_id / "decisions" / (filename + ".json")
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def resume_session(session_id, video_id, quality_run_id):
    """Restore only decisions whose persisted inputs still match the report and frames."""
    session = get_session(session_id)
    if not session or session.get("video_id") != video_id or session.get("quality_run_id") != quality_run_id:
        raise ValueError("Review session does not match recording and quality report")
    report = _report(quality_run_id, video_id)
    if session.get("authorization_source_sha256") != require_authorization(video_id)["source_sha256"]:
        raise ValueError("Recording authorization changed; create a new review session")
    _, reference_digest = _frame(report, video_id, session["reference_frame"])
    if reference_digest != session.get("reference_frame_sha256"):
        raise ValueError("Reference frame changed; create a new review session")
    decisions, unavailable = {}, []
    for item in report.get("frames", []):
        name = item.get("filename")
        try:
            frame, digest = _frame(report, video_id, name)
            decision = get_decision(session_id, name)
        except ValueError:
            unavailable.append(name)
            continue
        if decision is None:
            continue
        if (decision.get("session_id") != session_id
                or decision.get("recording_id") != video_id
                or decision.get("quality_run_id") != quality_run_id
                or decision.get("frame_id") != name
                or decision.get("frame_sha256") != digest
                or decision.get("reviewer_decision") not in ("confirm", "reassign", "reject")
                or decision.get("identity_status") != (
                    "rejected" if decision.get("reviewer_decision") == "reject"
                    else "human_confirmed_not_biometric")):
            unavailable.append(name)
            continue
        if decision["reviewer_decision"] != "reject":
            checks = frame.get("checks", {})
            try:
                if (frame.get("status") not in ("acceptable", "review") or frame.get("rejections")
                        or any(checks.get(key) != "pass" for key in
                               ("readability", "person_presence", "sharpness", "lighting", "full_body"))):
                    raise ValueError("Mandatory quality checks do not pass")
                validate_review_crop(decision.get("selected_subject_box"), frame)
            except (ValueError, AttributeError):
                unavailable.append(name)
                continue
        decisions[name] = decision
    return {"session": session, "decisions": decisions, "unavailable_frames": unavailable}


def frame_suggestion(session_id, filename):
    session = get_session(session_id)
    if not session:
        raise ValueError("Review session not found")
    report = _report(session["quality_run_id"], session["video_id"])
    if session.get("authorization_source_sha256") != require_authorization(session["video_id"])["source_sha256"]:
        raise ValueError("Recording authorization changed; create a new review session")
    _frame(report, session["video_id"], session["reference_frame"])
    frame, digest = _frame(report, session["video_id"], filename)
    decision = get_decision(session_id, filename)
    if decision and decision.get("frame_sha256") != digest:
        decision = None
    anchor_box = session["reference_box"]
    anchor_frame = session["reference_frame"]
    ordered = [f.get("filename") for f in report.get("frames", [])]
    index = ordered.index(filename)
    for prior in reversed(ordered[:index]):
        previous = get_decision(session_id, prior)
        if previous and previous.get("reviewer_decision") in ("confirm", "reassign"):
            _, prior_digest = _frame(report, session["video_id"], prior)
            if previous.get("frame_sha256") != prior_digest:
                raise ValueError("Previously reviewed frame changed; re-analyze")
            anchor_box = validate_box(previous["selected_subject_box"])
            anchor_frame = prior
            break
    suggestion = suggest(anchor_box, frame.get("detector_evidence"))
    suggestion["anchor_frame"] = anchor_frame
    return {"filename": filename, "frame_sha256": digest,
            "identity_status": (decision["identity_status"] if decision else
                                "subject_not_detected" if frame.get("person_detected") is False else
                                "identity_ambiguous" if suggestion["status"] == "ambiguous" else "not_reviewed"),
            "suggestion": suggestion, "decision": decision, "checks": frame.get("checks", {}),
            "warnings": frame.get("warnings", []), "rejections": frame.get("rejections", [])}


def review_frame(session_id, filename, decision, bbox=None):
    if decision not in ("confirm", "reassign", "reject"):
        raise ValueError("Decision must be confirm, reassign or reject")
    session = get_session(session_id)
    if not session:
        raise ValueError("Review session not found")
    report = _report(session["quality_run_id"], session["video_id"])
    if session.get("authorization_source_sha256") != require_authorization(session["video_id"])["source_sha256"]:
        raise ValueError("Recording authorization changed; create a new review session")
    _, reference_digest = _frame(report, session["video_id"], session["reference_frame"])
    if reference_digest != session["reference_frame_sha256"]:
        raise ValueError("Reference frame changed; create a new review session")
    frame, digest = _frame(report, session["video_id"], filename)
    box = validate_box(bbox) if decision != "reject" else None
    checks = frame.get("checks", {})
    if decision != "reject" and (frame.get("status") == "rejected" or frame.get("rejections") or
                                 any(checks.get(key) != "pass" for key in
                                     ("readability", "person_presence", "sharpness", "lighting", "full_body"))):
        raise ValueError("Frame has failed or unknown mandatory quality checks")
    if box is not None:
        box = validate_review_crop(box, frame)
    record = {"recording_id": session["video_id"], "quality_run_id": session["quality_run_id"],
              "session_id": session_id, "frame_id": filename, "frame_sha256": digest,
              "reference_frame": session["reference_frame"], "selected_subject_box": box,
              "reviewer_decision": decision, "timestamp": time.time(),
              "identity_status": "human_confirmed_not_biometric" if box else "rejected",
              "association_suggestion": suggest(session["reference_box"], frame.get("detector_evidence"))}
    folder = ROOT / session_id / "decisions"
    with _LOCK:
        folder.mkdir(exist_ok=True)
        history = folder / "history"
        history.mkdir(exist_ok=True)
        record["decision_id"] = str(uuid.uuid4())
        serialized = json.dumps(record, indent=2, allow_nan=False)
        with (history / (record["decision_id"] + ".json")).open("x", encoding="utf-8") as output:
            output.write(serialized)
        path = folder / (filename + ".json")
        tmp = folder / (record["decision_id"] + ".tmp")
        tmp.write_text(serialized, encoding="utf-8")
        os.replace(tmp, path)
    return record
