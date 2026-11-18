"""Explicit experimental whole-frame HMR2 batches; not subject-identity review."""
from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from backend.services import video_mesh_batch as batch
from backend.services.capture_authorization import require_authorization
from backend.services.capture_quality import QualityConfig
from backend.services.capture_quality_job import get_report
from backend.services.video_processor import get_capture_engine
from backend.utils.validators import validate_image

_MANDATORY = ("readability", "person_presence", "sharpness", "lighting", "full_body")


def start_experimental(video_id, quality_run_id, unknown_inspections):
    if not batch.valid_uuid(video_id) or not batch.valid_uuid(quality_run_id):
        raise ValueError("Invalid recording or quality report ID")
    authorization = require_authorization(video_id)
    capture = get_capture_engine()
    metadata = capture.get_metadata(video_id)
    if not metadata or metadata.get("status") != "completed":
        raise ValueError("Completed extraction metadata not found")
    report = get_report(quality_run_id)
    if not report or report.get("video_id") != video_id or report.get("run_id") != quality_run_id:
        raise ValueError("Quality report unavailable or does not belong to this video")
    if report.get("thresholds") != asdict(QualityConfig()):
        raise ValueError("Experimental inference requires the default server screening policy")

    allowed = {}
    for frame in metadata.get("frames", []):
        name = frame.get("filename")
        if not isinstance(name, str) or not batch._FRAME.fullmatch(name) or name in allowed:
            raise ValueError("Invalid or duplicate extraction frame")
        allowed[name] = frame
    qualities = {}
    for frame in report.get("frames", []):
        name = frame.get("filename")
        if name not in allowed or name in qualities:
            raise ValueError("Invalid or duplicate quality report frame")
        qualities[name] = frame

    inspections = {}
    for item in unknown_inspections:
        name = item["filename"]
        if name in inspections or name not in qualities:
            raise ValueError("Invalid or duplicate unknown frame inspection")
        if item["inspected"] is not True or item["warning_acknowledged"] is not True:
            raise ValueError(f"Frame {name} requires explicit inspection and warning acknowledgement")
        inspections[name] = item

    selected = []
    for name in allowed:
        quality = qualities.get(name)
        if not quality:
            continue
        checks = quality.get("checks")
        if (quality.get("status") not in ("acceptable", "review") or quality.get("rejections")
                or not isinstance(checks, dict)):
            if name in inspections:
                raise ValueError(f"Frame {name} failed quality screening")
            continue
        values = [checks.get(key) for key in _MANDATORY]
        if all(value == "pass" for value in values):
            if name in inspections:
                raise ValueError(f"Frame {name} is not an unknown-quality frame")
            selected.append(name)
        elif all(value in ("pass", "unknown") for value in values) and "unknown" in values:
            if name in inspections:
                selected.append(name)
        elif name in inspections:
            raise ValueError(f"Frame {name} failed or has invalid mandatory quality checks")

    if not selected:
        raise ValueError("No eligible frames selected")
    if len(selected) > 300 or len({Path(n).stem.lower() for n in selected}) != len(selected):
        raise ValueError("Too many frames or colliding mesh names")
    snapshots = {}
    for name in selected:
        path = capture.get_frame_path(video_id, name)
        if path is None:
            raise ValueError(f"Frame unavailable: {name}")
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if not qualities[name].get("source_sha256") or digest != qualities[name]["source_sha256"]:
            raise ValueError(f"Frame {name} changed since quality analysis; re-analyze")
        if name in inspections and inspections[name]["source_sha256"] != digest:
            raise ValueError(f"Frame {name} changed since operator inspection")
        validate_image(name, data)
        snapshots[name] = data

    engine = batch.get_engine()
    if not engine.is_ready:
        raise RuntimeError(f"Inference engine is not ready: {engine.load_error or 'model loading'}")

    run_id = str(uuid.uuid4())
    batch._RUN_ROOT.mkdir(parents=True, exist_ok=True)
    root = batch._root(run_id)
    root.mkdir(exist_ok=False)
    try:
        (root / "frames").mkdir()
        (root / "crops").mkdir()
        (root / "meshes").mkdir()
        frames = []
        for name in selected:
            (root / "frames" / name).write_bytes(snapshots[name])
            # The shared worker reads this file. It is the whole image, not a reviewed crop.
            (root / "crops" / (Path(name).stem + ".png")).write_bytes(snapshots[name])
            frames.append({"filename": name, "status": "queued",
                           "timestamp_sec": allowed[name].get("timestamp_sec"),
                           "quality_checks": qualities[name]["checks"],
                           "quality_warnings": qualities[name].get("warnings", []),
                           "source_sha256": qualities[name]["source_sha256"],
                           "unknown_inspection": inspections.get(name)})
        run = {"run_id": run_id, "video_id": video_id, "quality_run_id": quality_run_id,
               "mode": "experimental_whole_frame", "status": "processing",
               "authorization_source_sha256": authorization["source_sha256"],
               "experimental_risk_acknowledged": True,
               "identity_note": "No subject review or identity verification; detector may select another person or use the whole image.",
               "note": "Independent experimental HMR2 predictions, not a unified scan or validated measurements.",
               "frames": frames, "created_at": time.time()}
        with batch._LOCK:
            batch._ACTIVE[run_id] = run
            batch._persist(run)
            response = batch.copy.deepcopy(run)
            batch.threading.Thread(target=batch._serialized_worker, args=(run_id, engine), daemon=True).start()
        return response
    except Exception:
        with batch._LOCK:
            batch._ACTIVE.pop(run_id, None)
        batch.shutil.rmtree(root, ignore_errors=True)
        raise
