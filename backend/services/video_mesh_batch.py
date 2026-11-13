"""Background video-frame inference batches, using the existing inference engine."""
from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
import re
import shutil
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from backend.services import inference_engine
from backend.services.video_processor import get_capture_engine
from backend.services.inference_engine import get_engine
from backend.services.capture_authorization import require_authorization
from backend.services.capture_quality_job import get_report
from backend.services.capture_quality import QualityConfig
from backend.services.subject_review import get_session, get_decision, validate_review_crop
from PIL import Image
from backend.utils.validators import validate_image

_RUN_ROOT = Path(os.getenv("VIDEO_MESH_OUTPUT_DIR", "outputs/video_mesh_ui"))
_FRAME = re.compile(r"frame_\d+\.(?:jpg|jpeg|png)", re.IGNORECASE)
_MAX_FRAMES = 12
_TIMEOUT = float(os.getenv("VIDEO_MESH_FRAME_TIMEOUT_SEC", "300"))
_LOCK = threading.RLock()
_BATCH_LOCK = threading.Lock()
_ACTIVE = {}


def valid_uuid(value):
    try:
        return str(uuid.UUID(value)) == value
    except (ValueError, TypeError, AttributeError):
        return False


def _root(run_id):
    if not valid_uuid(run_id):
        raise ValueError("Invalid run ID")
    return _RUN_ROOT / run_id


def _persist(run):
    root = _root(run["run_id"])
    temp = root / "status.tmp"
    temp.write_text(json.dumps(run, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temp, root / "status.json")


def _validated_crop(name, data, quality, decision):
    """Apply the same image and reviewed-crop validation for preflight and submission."""
    validate_image(name, data)
    box = validate_review_crop(decision.get("selected_subject_box"), quality)
    with Image.open(io.BytesIO(data)) as image:
        image = image.convert("RGB")
        width, height = image.size
        pixels = (math.floor(box[0] * width), math.floor(box[1] * height),
                  math.ceil(box[2] * width), math.ceil(box[3] * height))
        crop = image.crop(pixels)
        if crop.width < 32 or crop.height < 32:
            raise ValueError(f"Reviewed crop for {name} is too small")
        buffer = io.BytesIO()
        crop.save(buffer, format="PNG")
    return box, pixels, buffer.getvalue()


def _prepare_batch(video_id, names, quality_run_id=None, reviewed_frames=None, review_session_id=None, *, initialize_engine=True):
    if not valid_uuid(video_id):
        raise ValueError("Invalid video ID")
    authorization = require_authorization(video_id)
    if not isinstance(names, list) or not 1 <= len(names) <= _MAX_FRAMES:
        raise ValueError("Select between 1 and 12 frames")
    if any(not isinstance(n, str) or not _FRAME.fullmatch(n) for n in names) or len({n.lower() for n in names}) != len(names):
        raise ValueError("Invalid or duplicate frame names")
    if len({Path(n).stem.lower() for n in names}) != len(names):
        raise ValueError("Frame names have colliding mesh stems")
    capture = get_capture_engine()
    metadata = capture.get_metadata(video_id)
    if not metadata or metadata.get("status") != "completed":
        raise ValueError("Completed extraction metadata not found")
    allowed = {f.get("filename"): f for f in metadata.get("frames", [])}
    if any(n not in allowed for n in names):
        raise ValueError("Selected frame is not in extraction metadata")
    if not valid_uuid(quality_run_id):
        raise ValueError("A completed quality report is required before reconstruction")
    report = get_report(quality_run_id)
    if not report or report.get("video_id") != video_id or report.get("run_id") != quality_run_id:
        raise ValueError("Quality report unavailable or does not belong to this video")
    if report.get("thresholds") != asdict(QualityConfig()):
        raise ValueError("Reconstruction requires a quality report using the default server screening policy")
    if not isinstance(reviewed_frames, list) or set(reviewed_frames) != set(names) or len(reviewed_frames) != len(names):
        raise ValueError("Explicit manual review is required for every selected frame")
    session = get_session(review_session_id) if valid_uuid(review_session_id) else None
    if not session or session.get("video_id") != video_id or session.get("quality_run_id") != quality_run_id:
        raise ValueError("A matching subject review session is required")
    if session.get("authorization_source_sha256") != authorization["source_sha256"]:
        raise ValueError("Recording authorization changed; create a new review session")
    reference = capture.get_frame_path(video_id, session.get("reference_frame", ""))
    if not reference or hashlib.sha256(reference.read_bytes()).hexdigest() != session.get("reference_frame_sha256"):
        raise ValueError("Reference frame changed; create a new review session")
    quality_frames = {f.get("filename"): f for f in report.get("frames", [])}
    snapshots = {}
    crops = {}
    review_records = {}
    for name in names:
        quality = quality_frames.get(name, {})
        checks = quality.get("checks", {})
        warnings = quality.get("warnings", [])
        if (quality.get("status") not in ("acceptable", "review") or quality.get("rejections")
                or not isinstance(checks, dict)
                or any(checks.get(key) != "pass" for key in
                       ("readability", "person_presence", "sharpness", "lighting", "full_body"))):
            raise ValueError(f"Frame {name} has failed or unknown mandatory quality checks")
        decision = get_decision(review_session_id, name)
        if (not decision or decision.get("reviewer_decision") not in ("confirm", "reassign")
                or decision.get("identity_status") != "human_confirmed_not_biometric"
                or decision.get("recording_id") != video_id
                or decision.get("quality_run_id") != quality_run_id
                or decision.get("session_id") != review_session_id
                or decision.get("frame_id") != name):
            raise ValueError(f"Frame {name} needs an explicit subject review")
        source = capture.get_frame_path(video_id, name)
        if source is None:
            raise ValueError(f"Frame unavailable: {name}")
        data = source.read_bytes()
        if not quality.get("source_sha256") or hashlib.sha256(data).hexdigest() != quality["source_sha256"]:
            raise ValueError(f"Frame {name} changed since quality analysis; re-analyze")
        if hashlib.sha256(data).hexdigest() != decision.get("frame_sha256"):
            raise ValueError(f"Frame {name} changed since subject review; re-review")
        box, pixels, crop_bytes = _validated_crop(name, data, quality, decision)
        snapshots[name] = data
        crops[name] = crop_bytes
        review_records[name] = {"decision_id": decision.get("decision_id"),
                                "reviewer_decision": decision["reviewer_decision"],
                                "bbox_normalized": box, "bbox_pixels": list(pixels),
                                "source_sha256": decision["frame_sha256"]}
    engine = get_engine() if initialize_engine else inference_engine._engine
    if engine is None or not engine.is_ready:
        raise RuntimeError("Inference engine is not ready")
    return allowed, snapshots, crops, review_records, engine


def check_readiness(video_id, names, quality_run_id=None, reviewed_frames=None, review_session_id=None):
    """Collect independent blockers; submission's full gate remains authoritative."""
    blockers, unevaluated = [], []

    def add(message):
        if message not in blockers:
            blockers.append(message)

    valid_video = valid_uuid(video_id)
    if not valid_video:
        add("Invalid recording ID")
    authorization = None
    capture = None
    metadata = None
    if valid_video:
        try:
            authorization = require_authorization(video_id)
        except (ValueError, OSError) as exc:
            add(str(exc))
        try:
            capture = get_capture_engine()
            metadata = capture.get_metadata(video_id)
        except (ValueError, OSError) as exc:
            add(f"Capture metadata unavailable: {exc}")
    if not metadata or metadata.get("status") != "completed":
        add("Completed extraction metadata not found")
        metadata = None

    valid_names = isinstance(names, list) and 1 <= len(names) <= _MAX_FRAMES
    if not valid_names:
        add("Select between 1 and 12 frames")
    elif (any(not isinstance(n, str) or not _FRAME.fullmatch(n) for n in names)
          or len({n.lower() for n in names}) != len(names)
          or len({Path(n).stem.lower() for n in names}) != len(names)):
        add("Invalid, duplicate or colliding frame names")
        valid_names = False

    report = None
    if not valid_uuid(quality_run_id):
        add("A completed quality report is required before reconstruction")
    else:
        try:
            report = get_report(quality_run_id)
        except (ValueError, OSError) as exc:
            add(f"Quality report unavailable: {exc}")
        if not report or report.get("video_id") != video_id or report.get("run_id") != quality_run_id:
            add("Quality report unavailable or does not belong to this video")
            report = None
        elif report.get("thresholds") != asdict(QualityConfig()):
            add("Reconstruction requires a quality report using the default server screening policy")

    if not isinstance(reviewed_frames, list):
        add("Explicit warning acknowledgement is required for every selected frame")
        reviewed_frames = []
    if valid_names:
        if len(set(reviewed_frames)) != len(reviewed_frames) or set(reviewed_frames) != set(names):
            add("Explicit warning acknowledgement is required for every selected frame")

    session = None
    if valid_uuid(review_session_id):
        try:
            session = get_session(review_session_id)
        except (ValueError, OSError) as exc:
            add(f"Subject review session unavailable: {exc}")
    if not session or session.get("video_id") != video_id or session.get("quality_run_id") != quality_run_id:
        add("A matching subject review session and reference subject are required")
        session = None
    else:
        if authorization is None:
            unevaluated.append("Session authorization source binding cannot be checked without a valid attestation")
        elif session.get("authorization_source_sha256") != authorization["source_sha256"]:
            add("Recording authorization changed; create a new review session")
        if capture is None:
            unevaluated.append("Reference-frame hash cannot be checked without capture metadata")
        else:
            try:
                reference = capture.get_frame_path(video_id, session.get("reference_frame", ""))
                if not reference or hashlib.sha256(reference.read_bytes()).hexdigest() != session.get("reference_frame_sha256"):
                    add("Reference frame changed or unavailable; create a new review session")
            except (ValueError, OSError):
                add("Reference frame changed or unavailable; create a new review session")

    if valid_names:
        allowed = {f.get("filename") for f in metadata.get("frames", [])} if metadata else set()
        qualities = {f.get("filename"): f for f in report.get("frames", [])} if report else {}
        if metadata is None:
            unevaluated.append("Selected-frame membership cannot be checked without capture metadata")
        if report is None:
            unevaluated.append("Selected-frame quality and report hashes cannot be checked without a matching report")
        if session is None:
            unevaluated.append("Selected-frame decisions and reviewed crops cannot be checked without a matching session")
        for name in names:
            if metadata and name not in allowed:
                add(f"Frame {name} is not in extraction metadata")
            quality = qualities.get(name)
            if report:
                checks = quality.get("checks", {}) if quality else {}
                if (not quality or quality.get("status") not in ("acceptable", "review") or quality.get("rejections")
                        or not isinstance(checks, dict) or any(checks.get(key) != "pass" for key in
                            ("readability", "person_presence", "sharpness", "lighting", "full_body"))):
                    add(f"Frame {name} has failed or unknown mandatory quality checks")
            decision = None
            if session:
                try:
                    decision = get_decision(review_session_id, name)
                except (ValueError, OSError):
                    pass
                if (not decision or decision.get("reviewer_decision") not in ("confirm", "reassign")
                        or decision.get("identity_status") != "human_confirmed_not_biometric"
                        or decision.get("recording_id") != video_id or decision.get("quality_run_id") != quality_run_id
                        or decision.get("session_id") != review_session_id or decision.get("frame_id") != name):
                    add(f"Frame {name} needs an explicit subject review")
                    decision = None
            data = None
            if capture:
                try:
                    path = capture.get_frame_path(video_id, name)
                    if not path:
                        add(f"Frame {name} is unavailable")
                    else:
                        data = path.read_bytes()
                        digest = hashlib.sha256(data).hexdigest()
                        if quality and digest != quality.get("source_sha256"):
                            add(f"Frame {name} changed since quality analysis; re-analyze")
                        if decision and digest != decision.get("frame_sha256"):
                            add(f"Frame {name} changed since subject review; re-review")
                except (ValueError, OSError):
                    add(f"Frame {name} is unavailable")
            if data is None:
                unevaluated.append(f"Frame {name}: image validation cannot be checked without frame bytes")
            else:
                try:
                    validate_image(name, data)
                except ValueError as exc:
                    add(f"Frame {name}: image validation failed: {exc}")
            if quality and decision and data is not None:
                try:
                    _validated_crop(name, data, quality, decision)
                except (ValueError, OSError) as exc:
                    add(f"Frame {name}: image or crop validation failed: {exc}")
            else:
                unevaluated.append(f"Frame {name}: reviewed crop and crop size cannot be checked without frame bytes, quality report and valid decision")
    else:
        unevaluated.append("Per-frame quality, decisions, hashes and crops require a valid frame selection")

    engine = inference_engine._engine
    if engine is None or not engine.is_ready:
        add("Inference engine is not ready")

    # Diagnostics can never override the authoritative gate.
    if not blockers and not unevaluated:
        try:
            _prepare_batch(video_id, names, quality_run_id, reviewed_frames, review_session_id, initialize_engine=False)
        except (ValueError, RuntimeError, OSError) as exc:
            add(str(exc))
    return {"ready": not blockers and not unevaluated, "blockers": blockers, "unevaluated": unevaluated}


def start_batch(video_id, names, quality_run_id=None, reviewed_frames=None, review_session_id=None):
    allowed, snapshots, crops, review_records, engine = _prepare_batch(
        video_id, names, quality_run_id, reviewed_frames, review_session_id
    )
    run_id = str(uuid.uuid4())
    _RUN_ROOT.mkdir(parents=True, exist_ok=True)
    root = _root(run_id)
    root.mkdir(exist_ok=False)
    (root / "frames").mkdir()
    (root / "crops").mkdir()
    (root / "meshes").mkdir()
    frames = []
    # Snapshot selected images before accepting a job, preserving the run's source.
    try:
        for name in names:
            data = snapshots[name]
            (root / "frames" / name).write_bytes(data)
            (root / "crops" / (Path(name).stem + ".png")).write_bytes(crops[name])
            frames.append({"filename": name, "review": review_records[name],
                           "timestamp_sec": allowed[name].get("timestamp_sec"),
                           "is_blurry": allowed[name].get("is_blurry"),
                           "blur_score": allowed[name].get("blur_score"), "status": "queued"})
        run = {"run_id": run_id, "video_id": video_id, "status": "processing",
               "quality_run_id": quality_run_id, "review_session_id": review_session_id,
               "manual_reviewed_frames": reviewed_frames,
               "identity_note": "Human-confirmed subject crop; not biometric identity verification.",
               "frames": frames, "created_at": time.time(),
               "note": "Independent single-image predictions; not a unified reconstruction."}
        with _LOCK:
            _ACTIVE[run_id] = run
            _persist(run)
            response = copy.deepcopy(run)
            threading.Thread(target=_serialized_worker, args=(run_id, engine), daemon=True).start()
    except Exception:
        with _LOCK:
            _ACTIVE.pop(run_id, None)
        shutil.rmtree(root, ignore_errors=True)
        raise
    return response


def _serialized_worker(run_id, engine):
    with _BATCH_LOCK:
        _worker(run_id, engine)


def _worker(run_id, engine):
    root = _root(run_id)
    with _LOCK:
        run = _ACTIVE[run_id]
    try:
        for frame in run["frames"]:
            start = time.monotonic()
            try:
                with _LOCK:
                    frame["status"] = "processing"
                    _persist(run)
                job_id = str(uuid.uuid4())
                engine.submit_job(job_id=job_id, image_bytes=(root / "crops" / (Path(frame["filename"]).stem + ".png")).read_bytes())
                deadline = time.monotonic() + _TIMEOUT
                while time.monotonic() < deadline:
                    result = engine.get_job(job_id)
                    if result and result.get("status") == "completed":
                        break
                    if result and result.get("status") in ("failed", "error"):
                        raise RuntimeError(result.get("detail", "Inference failed"))
                    time.sleep(.25)
                else:
                    raise TimeoutError("Inference timed out")
                target = root / "meshes" / (Path(frame["filename"]).stem + ".obj")
                with target.open("xb") as dest, Path(result["mesh_path"]).open("rb") as src:
                    shutil.copyfileobj(src, dest)
                with _LOCK:
                    frame.update(status="completed", metadata=result.get("metadata", {}),
                                 mesh_url=f"/api/video-mesh/runs/{run_id}/mesh/{frame['filename']}")
            except Exception as exc:
                with _LOCK:
                    frame.update(status="failed", detail=str(exc))
            finally:
                with _LOCK:
                    frame["processing_time_sec"] = round(time.monotonic() - start, 3)
                    _persist(run)
        with _LOCK:
            run["status"] = "completed" if all(f["status"] == "completed" for f in run["frames"]) else "completed_with_errors"
            run["completed_at"] = time.time()
            _persist(run)
    finally:
        with _LOCK:
            _ACTIVE.pop(run_id, None)


def get_run(run_id):
    root = _root(run_id)
    with _LOCK:
        if run_id in _ACTIVE:
            return copy.deepcopy(_ACTIVE[run_id])
        path = root / "status.json"
        if not path.is_file():
            return None
        run = json.loads(path.read_text(encoding="utf-8"))
    if run.get("status") == "processing":
        run["status"] = "interrupted"
        run["detail"] = "The server stopped before this batch finished."
    return run


def get_asset(run_id, filename, kind):
    if not _FRAME.fullmatch(filename):
        raise ValueError("Invalid frame name")
    run = get_run(run_id)
    if not run or filename not in {f["filename"] for f in run["frames"]}:
        return None
    if kind == "frame":
        path = _root(run_id) / "frames" / filename
    elif kind == "mesh":
        frame = next(f for f in run["frames"] if f["filename"] == filename)
        if frame["status"] != "completed":
            return None
        path = _root(run_id) / "meshes" / (Path(filename).stem + ".obj")
    else:
        raise ValueError("Invalid asset kind")
    return path if path.is_file() else None
