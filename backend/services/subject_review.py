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
MAX_SELECTED_FRAMES = 12
MANDATORY_CHECKS = ("readability", "person_presence", "sharpness", "lighting", "full_body")


class SessionConflict(ValueError):
    """An active session already exists; the caller must resume it or explicitly replace it."""

    def __init__(self, message, session_id):
        super().__init__(message)
        self.session_id = session_id


def _write_json(path, data):
    tmp = path.with_suffix(path.suffix + "." + uuid.uuid4().hex + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def _mandatory_pass(frame):
    checks = frame.get("checks", {}) if isinstance(frame, dict) else {}
    return (frame.get("status") in ("acceptable", "review") and not frame.get("rejections")
            and isinstance(checks, dict) and all(checks.get(key) == "pass" for key in MANDATORY_CHECKS))


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


def _session_problem(session, report, video_id, quality_run_id, authorization_sha):
    """Return why a stored session cannot be resumed, or None when it is still valid."""
    if not session or session.get("video_id") != video_id or session.get("quality_run_id") != quality_run_id:
        return "Review session does not match recording and quality report"
    if session.get("superseded_by"):
        return "Review session was replaced by a newer reference subject"
    if session.get("authorization_source_sha256") != authorization_sha:
        return "Recording authorization changed; create a new review session"
    try:
        _, digest = _frame(report, video_id, session.get("reference_frame"))
    except ValueError as exc:
        return str(exc)
    if digest != session.get("reference_frame_sha256"):
        return "Reference frame changed; create a new review session"
    return None


def _decision_count(session_id):
    folder = ROOT / session_id / "decisions"
    return len(list(folder.glob("*.json"))) if folder.is_dir() else 0


def find_sessions(video_id, quality_run_id):
    """List stored sessions for a recording/report, newest first, with the resumable one marked active."""
    report = _report(quality_run_id, video_id)
    authorization_sha = require_authorization(video_id)["source_sha256"]
    sessions = []
    if ROOT.is_dir():
        for path in ROOT.glob("*/session.json"):
            if not valid_id(path.parent.name):
                continue
            try:
                session = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if session.get("video_id") != video_id or session.get("quality_run_id") != quality_run_id:
                continue
            problem = _session_problem(session, report, video_id, quality_run_id, authorization_sha)
            sessions.append({"session_id": session.get("session_id"), "reference_frame": session.get("reference_frame"),
                             "reference_box": session.get("reference_box"), "created_at": session.get("created_at"),
                             "superseded_by": session.get("superseded_by"), "valid": problem is None,
                             "problem": problem, "decision_count": _decision_count(path.parent.name)})
    sessions.sort(key=lambda item: item.get("created_at") or 0, reverse=True)
    active = next((item["session_id"] for item in sessions if item["valid"]), None)
    return {"video_id": video_id, "quality_run_id": quality_run_id,
            "active_session_id": active, "sessions": sessions}


def create_session(video_id, quality_run_id, reference_frame, bbox, replace_existing=False):
    """Create a session; an identical request resumes, a different reference must explicitly replace."""
    report = _report(quality_run_id, video_id)
    frame, digest = _frame(report, video_id, reference_frame)
    box = validate_review_crop(bbox, frame)
    authorization = require_authorization(video_id)
    with _LOCK:
        active_id = find_sessions(video_id, quality_run_id)["active_session_id"]
        active = get_session(active_id) if active_id else None
        if active:
            same = (active.get("reference_frame") == reference_frame and
                    all(abs(a - b) < 1e-9 for a, b in zip(active.get("reference_box") or [], box)))
            if same:
                return {**active, "resumed_existing": True}
            if replace_existing is not True:
                raise SessionConflict(
                    f"Review session {active_id} is already active for this recording and quality report. "
                    "Resume it, or explicitly replace its reference subject.", active_id)
        session_id = str(uuid.uuid4())
        root = ROOT / session_id
        root.mkdir(parents=True, exist_ok=False)
        session = {"session_id": session_id, "video_id": video_id, "quality_run_id": quality_run_id,
                   "authorization_source_sha256": authorization["source_sha256"],
                   "reference_frame": reference_frame, "reference_frame_sha256": digest,
                   "reference_box": box, "created_at": time.time(),
                   "replaces_session_id": active["session_id"] if active else None,
                   "identity_note": "Human-selected subject; not biometric identity verification."}
        _write_json(root / "session.json", session)
        if active:
            # Decisions were made against the old reference, so they are not carried over.
            _write_json(ROOT / active["session_id"] / "session.json",
                        {**active, "superseded_by": session_id, "superseded_at": time.time()})
    return {**session, "resumed_existing": False}


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


def _decision_valid(decision, session_id, video_id, quality_run_id, name, digest, frame):
    """True when a persisted decision still matches its session, recording, report and frame bytes."""
    if (not isinstance(decision, dict)
            or decision.get("session_id") != session_id
            or decision.get("recording_id") != video_id
            or decision.get("quality_run_id") != quality_run_id
            or decision.get("frame_id") != name
            or decision.get("frame_sha256") != digest
            or decision.get("reviewer_decision") not in ("confirm", "reassign", "reject")
            or decision.get("identity_status") != (
                "rejected" if decision.get("reviewer_decision") == "reject"
                else "human_confirmed_not_biometric")):
        return False
    if decision["reviewer_decision"] == "reject":
        return True
    try:
        if not _mandatory_pass(frame):
            return False
        validate_review_crop(decision.get("selected_subject_box"), frame)
    except (ValueError, AttributeError):
        return False
    return True


def _confirmed(decision):
    return bool(decision) and decision.get("reviewer_decision") in ("confirm", "reassign")


def _validated_session(session_id, video_id, quality_run_id):
    session = get_session(session_id)
    if not session or session.get("video_id") != video_id or session.get("quality_run_id") != quality_run_id:
        raise ValueError("Review session does not match recording and quality report")
    report = _report(quality_run_id, video_id)
    problem = _session_problem(session, report, video_id, quality_run_id,
                               require_authorization(video_id)["source_sha256"])
    if problem:
        raise ValueError(problem)
    return session, report


def _selection_path(session_id):
    return ROOT / session_id / "selection.json"


def resume_session(session_id, video_id, quality_run_id):
    """Restore only decisions and selections whose persisted inputs still match the report and frames."""
    session, report = _validated_session(session_id, video_id, quality_run_id)
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
        if not _decision_valid(decision, session_id, video_id, quality_run_id, name, digest, frame):
            unavailable.append(name)
            continue
        decisions[name] = decision
    selection, dropped = {"frames": [], "acknowledged": []}, []
    path = _selection_path(session_id)
    if path.is_file():
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            stored = {}
        for name in stored.get("frames") or []:
            (selection["frames"] if _confirmed(decisions.get(name)) else dropped).append(name)
        # An acknowledgement only survives while the decision and frame bytes it covered are unchanged.
        for ack in stored.get("acknowledged") or []:
            decision = decisions.get(ack.get("filename"))
            if (ack.get("filename") in selection["frames"] and _confirmed(decision)
                    and ack.get("decision_id") == decision.get("decision_id")
                    and ack.get("frame_sha256") == decision.get("frame_sha256")):
                selection["acknowledged"].append(ack["filename"])
    return {"session": session, "decisions": decisions, "unavailable_frames": unavailable,
            "selection": selection, "dropped_selection": dropped}


def save_selection(session_id, frames, acknowledged):
    """Persist the operator's frame selection; only subject-confirmed, quality-passed frames are accepted."""
    session = get_session(session_id)
    if not session:
        raise ValueError("Review session not found")
    if (not isinstance(frames, list) or not isinstance(acknowledged, list)
            or any(not isinstance(name, str) for name in frames + acknowledged)):
        raise ValueError("Selection must be lists of frame filenames")
    if len(frames) > MAX_SELECTED_FRAMES:
        raise ValueError(f"Select at most {MAX_SELECTED_FRAMES} frames")
    if len(set(frames)) != len(frames) or len(set(acknowledged)) != len(acknowledged):
        raise ValueError("Duplicate frame in selection")
    if not set(acknowledged) <= set(frames):
        raise ValueError("Warnings can only be acknowledged for selected frames")
    video_id, quality_run_id = session["video_id"], session["quality_run_id"]
    _, report = _validated_session(session_id, video_id, quality_run_id)
    acks = []
    for name in frames:
        frame, digest = _frame(report, video_id, name)
        decision = get_decision(session_id, name)
        if not (_confirmed(decision) and
                _decision_valid(decision, session_id, video_id, quality_run_id, name, digest, frame)):
            raise ValueError(f"Frame {name} is not subject-confirmed in this session")
        if name in acknowledged:
            acks.append({"filename": name, "decision_id": decision.get("decision_id"), "frame_sha256": digest})
    record = {"session_id": session_id, "frames": frames, "acknowledged": acks, "updated_at": time.time()}
    with _LOCK:
        _write_json(_selection_path(session_id), record)
    return {"frames": frames, "acknowledged": [ack["filename"] for ack in acks], "updated_at": record["updated_at"]}


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
    if session.get("superseded_by"):
        raise ValueError("Review session was replaced by a newer reference subject; use the active session")
    frame, digest = _frame(report, session["video_id"], filename)
    box = validate_box(bbox) if decision != "reject" else None
    if decision != "reject" and not _mandatory_pass(frame):
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
        selection_path = _selection_path(session_id)
        if selection_path.is_file():
            # A changed decision invalidates its acknowledgement; a rejection also removes the selection.
            try:
                stored = json.loads(selection_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                stored = None
            if stored and filename in (stored.get("frames") or []):
                if decision == "reject":
                    stored["frames"] = [name for name in stored["frames"] if name != filename]
                stored["acknowledged"] = [ack for ack in stored.get("acknowledged") or []
                                          if ack.get("filename") != filename]
                stored["updated_at"] = time.time()
                _write_json(selection_path, stored)
    return record
