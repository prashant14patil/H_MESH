"""Background video-frame inference batches, using the existing inference engine."""
from __future__ import annotations

import copy
import json
import os
import re
import shutil
import threading
import time
import uuid
from pathlib import Path

from backend.services.video_processor import get_capture_engine
from backend.services.inference_engine import get_engine
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


def start_batch(video_id, names):
    if not valid_uuid(video_id):
        raise ValueError("Invalid video ID")
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
    engine = get_engine()
    if not engine.is_ready:
        raise RuntimeError("Inference engine is not ready")
    run_id = str(uuid.uuid4())
    _RUN_ROOT.mkdir(parents=True, exist_ok=True)
    root = _root(run_id)
    root.mkdir(exist_ok=False)
    (root / "frames").mkdir()
    (root / "meshes").mkdir()
    frames = []
    # Snapshot selected images before accepting a job, preserving the run's source.
    try:
        for name in names:
            source = capture.get_frame_path(video_id, name)
            if source is None:
                raise ValueError(f"Frame unavailable: {name}")
            data = source.read_bytes()
            validate_image(name, data)
            (root / "frames" / name).write_bytes(data)
            frames.append({"filename": name, "timestamp_sec": allowed[name].get("timestamp_sec"),
                           "is_blurry": allowed[name].get("is_blurry"),
                           "blur_score": allowed[name].get("blur_score"), "status": "queued"})
        run = {"run_id": run_id, "video_id": video_id, "status": "processing",
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
                engine.submit_job(job_id=job_id, image_bytes=(root / "frames" / frame["filename"]).read_bytes())
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
