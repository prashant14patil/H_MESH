"""End-to-end 360-degree video -> ONE unified SMPL body mesh.

Pipeline (each stage reports progress and records rejections):
  1. decode   oriented frames (display matrix applied exactly once, verified)
  2. detect   Keypoint R-CNN person boxes + 2D keypoints
  3. confirm  operator picks the subject when several people are present
  4. track    single-subject tracking, identity-switch / loss detection
  5. hmr2     real HMR2 inference on every tracked, usable frame
  6. views    heading from HMR2 global orientation, cross-checked with 2D cues
  7. select   sharp, unoccluded, de-duplicated frames spread over the rotation
  8. fit      shared SMPL shape + per-frame pose/camera multi-view optimisation
  9. export   canonical upright mesh (OBJ/PLY) + reports + diagnostics

Individual per-frame meshes are written only as diagnostics.
"""
from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np

from backend.services import smpl_multiview as smv
from backend.services.capture_authorization import require_authorization
from backend.services.person_tracking import (
    COCO_NAMES, Detection, TrackerConfig, candidate_summary, get_shared_detector, keypoint_confidence01,
    track_sequence)
from backend.services import video_processor

logger = logging.getLogger(__name__)

_ROOT = Path(os.getenv("RECONSTRUCTION_OUTPUT_DIR", "outputs/reconstructions"))
N_SAMPLES = int(os.getenv("RECON_SAMPLES", "120"))
K_SELECT = int(os.getenv("RECON_SELECT", "24"))
MIN_SUBJECT_HEIGHT_FRAC = 0.25   # a "candidate subject" spans >= 25% of image height
YAW_BIN_DEG = 15.0
HMR2_FOCAL = 5000.0
_RUN_LOCK = threading.Lock()
_RUNS: dict[str, dict] = {}
_EVENTS: dict[str, threading.Event] = {}
_STATE_LOCK = threading.RLock()

STAGES = [("decode", 0.10), ("detect", 0.16), ("confirm", 0.01), ("track", 0.02), ("hmr2", 0.22),
          ("views", 0.02), ("select", 0.02), ("masks", 0.08), ("fit", 0.27), ("export", 0.10)]
ASSET_FILES = {"final_body_mesh.obj", "final_body_mesh.ply", "reconstruction_report.json",
               "selected_frames.json", "tracking_report.json", "final_mesh_views.jpg",
               "contact_sheet.jpg", "orientation_check.jpg", "candidates_overview.jpg",
               "measurements.json", "measurement_rings.obj", "measurements_overlay.jpg",
               "view_comparison.jpg", "mesh_turntable.gif", "silhouette_masks.jpg",
               "measurements_cm.json", "final_body_mesh_metric.obj", "tape_validation.json"}
SIL_WEIGHT = float(os.getenv("RECON_SIL_WEIGHT", "1.0"))
BETA_SIGMA_FLOOR = 0.25      # minimum per-coefficient spread used by the shape prior
# Silhouette "stay inside the outline" term (see smpl_multiview.FitConfig.sil_outside_form). The legacy term
# saturated and was diluted by interior vertices, so the body protruded 7-8% outside its own clothed outline;
# the HMR2 shape prior below was then raised to hide that, which pinned the shape to HMR2's estimate and cut
# profile depth ~20%. Evidence: docs/RECONSTRUCTION_ACCURACY_INVESTIGATION.md.
#   original (16 Nov):  RECON_SIL_OUTSIDE=legacy RECON_BETA_PRIOR=0.02
#   committed:          RECON_SIL_OUTSIDE=legacy RECON_BETA_PRIOR=0
#   pre-fix working:    RECON_SIL_OUTSIDE=legacy RECON_BETA_PRIOR=0.2
SIL_OUTSIDE_FORM = os.getenv("RECON_SIL_OUTSIDE", "charbonnier")
# Mahalanobis pull toward the multi-frame HMR2 shape. Its sigma is HMR2's frame-to-frame JITTER, not the
# uncertainty of HMR2's shape, so even small weights pin the result to HMR2's (average-biased) estimate.
BETA_PRIOR_WEIGHT = float(os.getenv("RECON_BETA_PRIOR", "0.0"))
# Population prior: SMPL betas are in population standard deviations, so N(0, I) is the statistically
# meaningful regulariser of how unusual a body shape is.
POP_PRIOR_WEIGHT = float(os.getenv("RECON_POP_PRIOR", "0.05"))
# Whether stage B may bend each view's joints to match that view's silhouette (see FitConfig.fit_body_pose).
FIT_BODY_POSE = os.getenv("RECON_FIT_BODY_POSE", "1") == "1"
PROTRUSION_WARN = 0.03       # mean share of the rendered body outside the clothed outline
# Fit IoU minus held-out IoU. ~0.06 is typical: per-view joints absorb part of each view's silhouette (the
# exported mesh uses only the shared shape, so this does not change it). Locking pose closes the gap but makes
# the shape absorb keypoint-pose errors instead (coefficients up to 3.8 SD), so only abnormal gaps are flagged.
FIT_HELD_GAP_WARN = 0.08
# Weight of the silhouette "fill the outline" term (1 = symmetric). Kept at 1: with the shape prior on, lowering it
# changed girths by < 1% in the study; exposed for further experiments.
SIL_COVERAGE_WEIGHT = float(os.getenv("RECON_SIL_COVERAGE", "1.0"))
SIL_EXCLUDE_IOU = 0.75       # a view whose fitted silhouette IoU stays below this is unreliable
SIL_TAU_FRAC = float(os.getenv("RECON_SIL_TAU", "0.01"))  # robust silhouette scale (fraction of box height)
CV_FOLDS = 4
HELD_OUT_IOU_TOLERANCE = 0.03


class RunNotFoundError(Exception):
    """No reconstruction with this ID exists."""


class ReconstructionError(RuntimeError):
    """A condition that makes a trustworthy result impossible (with an actionable message)."""


# ------------------------------------------------------------- run plumbing
def _valid_uuid(value) -> bool:
    try:
        return str(uuid.UUID(value)) == value
    except (ValueError, TypeError, AttributeError):
        return False


def _run_dir(run_id: str) -> Path:
    if not _valid_uuid(run_id):
        raise ValueError("Invalid run ID")
    return _ROOT / run_id


def _persist(run: dict) -> None:
    root = _run_dir(run["run_id"])
    root.mkdir(parents=True, exist_ok=True)
    tmp = root / "status.tmp"
    tmp.write_text(json.dumps(run, indent=2, allow_nan=False, default=_json_default), encoding="utf-8")
    os.replace(tmp, root / "status.json")


def _json_default(obj):
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)}")


def _clean(obj):
    """Replace NaN/inf so strict JSON never fails."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, (float, np.floating)):
        return float(obj) if math.isfinite(obj) else None
    return obj


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(_clean(data), indent=2, allow_nan=False, default=_json_default), encoding="utf-8")


def _progress(run: dict, stage: str, frac: float, message: str) -> None:
    done = 0.0
    for name, weight in STAGES:
        if name == stage:
            done += weight * max(0.0, min(1.0, frac))
            break
        done += weight
    with _STATE_LOCK:
        run["stage"] = stage
        run["progress"] = round(min(done, 0.999), 4)
        run["message"] = message
        _persist(run)


def _warn(run: dict, code: str, message: str) -> None:
    with _STATE_LOCK:
        run.setdefault("warnings", []).append({"code": code, "message": message})
    logger.warning("[%s] %s: %s", run["run_id"][:8], code, message)


def get_run(run_id: str):
    with _STATE_LOCK:
        run = _RUNS.get(run_id)
        if run is not None:
            return json.loads(json.dumps(_clean(run), default=_json_default))
    path = _run_dir(run_id) / "status.json"
    if path.is_file():
        run = json.loads(path.read_text(encoding="utf-8"))
        if run.get("status") in ("running", "awaiting_person_selection"):
            # Not owned by this process: the backend restarted (e.g. --reload) and the worker thread died.
            _mark_interrupted(run)
        return run
    return None


def _mark_interrupted(run: dict) -> None:
    run.update(status="failed", message="Interrupted: the backend restarted while this reconstruction was running. "
                                        "Start the reconstruction again.",
               error={"code": "interrupted_by_backend_restart",
                      "message": "The backend restarted while this reconstruction was running "
                                 f"(last stage: {run.get('stage')}). Start the reconstruction again."})
    with _STATE_LOCK:
        _persist(run)


def mark_orphaned_runs() -> list[str]:
    """At startup: runs left 'running' on disk by a previous process can never finish; mark them interrupted.

    Without this they keep showing as running in listings until someone opens them. Runs owned by
    this process (in memory) are never touched.
    """
    marked = []
    if not _ROOT.is_dir():
        return marked
    for status_path in _ROOT.glob("*/status.json"):
        try:
            run = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        run_id = run.get("run_id")
        if (run.get("status") in ("running", "awaiting_person_selection") and _valid_uuid(run_id)
                and status_path.parent.name == run_id):
            with _STATE_LOCK:
                if run_id in _RUNS:
                    continue
            _mark_interrupted(run)
            marked.append(run_id)
    if marked:
        logger.warning("Marked %d reconstruction run(s) interrupted by an earlier backend restart", len(marked))
    return marked


def list_completed_runs(limit: int = 50) -> list[dict]:
    """Completed runs that have measurements, newest first: the history the Body Avatar compares against.

    Runs of byte-identical source videos are flagged with the same ``source_key`` so the UI can tell a
    re-upload apart from a new recording (a re-upload is not a new measurement).
    """
    rows = []
    if not _ROOT.is_dir():
        return rows
    for status_path in _ROOT.glob("*/status.json"):
        try:
            run = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        res = run.get("result") or {}
        if run.get("status") != "completed" or not res.get("measurements") or not _valid_uuid(run.get("run_id")):
            continue
        metric = res.get("metric_scale") or {}
        rows.append({"run_id": run["run_id"], "video_id": run.get("video_id"), "created_at": run.get("created_at"),
                     "mesh_obj": res.get("mesh_obj"), "measurements": res.get("measurements"),
                     "height_cm": metric.get("height_cm"), "measurements_cm": metric.get("measurements_cm"),
                     "source_key": _source_key(run.get("video_id"))})
    rows.sort(key=lambda r: r.get("created_at") or 0, reverse=True)
    return rows[:limit]


def _source_key(video_id) -> str | None:
    """Cheap content fingerprint of the uploaded video (first and last MiB plus file size)."""
    try:
        src = next(p for p in video_processor._OUTPUT_ROOT.joinpath(str(video_id)).glob("original.*") if p.is_file())
    except (StopIteration, OSError):
        return None
    import hashlib
    h = hashlib.sha256()
    size = src.stat().st_size
    with open(src, "rb") as fh:
        h.update(fh.read(1 << 20))
        fh.seek(max(0, size - (1 << 20)))
        h.update(fh.read(1 << 20))
    h.update(str(size).encode())
    return h.hexdigest()[:16]


def get_asset(run_id: str, name: str):
    root = _run_dir(run_id)
    if name in ASSET_FILES:
        path = root / name
    elif name.startswith("diagnostics__") and name.endswith((".jpg", ".obj")) and "/" not in name and "\\" not in name:
        path = root / "diagnostics" / name[len("diagnostics__"):]
    elif name.startswith("candidate_") and name.endswith(".jpg") and "/" not in name:
        path = root / name
    else:
        return None
    return path if path.is_file() else None


def start_run(video_id: str, target_box_norm: list | None = None, height_cm: float | None = None) -> dict:
    if not _valid_uuid(video_id):
        raise ValueError("Invalid recording ID")
    if height_cm is not None:
        from backend.services.body_measurements import validate_height_cm
        height_cm = validate_height_cm(height_cm)
    require_authorization(video_id)  # genuine consent gate is preserved
    engine = video_processor.get_capture_engine()
    meta = engine.get_metadata(video_id)
    if not meta or meta.get("status") != "completed":
        raise ValueError("Recording has not finished upload/extraction processing")
    sources = [p for p in video_processor._OUTPUT_ROOT.joinpath(video_id).glob("original.*") if p.is_file()]
    if len(sources) != 1:
        raise ValueError("Recording source is missing or ambiguous")
    if target_box_norm is not None:
        if (not isinstance(target_box_norm, list) or len(target_box_norm) != 4
                or not all(isinstance(v, (int, float)) and 0 <= v <= 1 for v in target_box_norm)):
            raise ValueError("target_box_norm must be four numbers in [0, 1]")
    with _STATE_LOCK:
        for existing in _RUNS.values():
            if existing.get("status") in ("running", "awaiting_person_selection"):
                raise RuntimeError("Another reconstruction is already running")
        run_id = str(uuid.uuid4())
        run = {"run_id": run_id, "video_id": video_id, "status": "running", "stage": "decode",
               "progress": 0.0, "message": "Queued", "warnings": [], "candidates": [],
               "created_at": time.time(), "result": None, "error": None, "requested_height_cm": height_cm}
        _RUNS[run_id] = run
        _EVENTS[run_id] = threading.Event()
        _persist(run)
    thread = threading.Thread(target=_worker, args=(run, sources[0], meta, target_box_norm), daemon=True)
    thread.start()
    return get_run(run_id)


def select_person(run_id: str, candidate_id: int) -> dict:
    with _STATE_LOCK:
        run = _RUNS.get(run_id)
        if run is None or run.get("status") != "awaiting_person_selection":
            raise ValueError("Run is not waiting for a person selection")
        ids = [c["id"] for c in run["candidates"]]
        if candidate_id not in ids:
            raise ValueError("Unknown candidate")
        run["selected_candidate"] = candidate_id
        run["status"] = "running"
        run["message"] = "Subject confirmed by operator"
        _persist(run)
        _EVENTS[run_id].set()
    return get_run(run_id)


def _scale_run(run: dict, height_cm) -> dict:
    """Write cm measurements + a metre-scaled OBJ for a completed run (caller holds _STATE_LOCK)."""
    from backend.services import body_measurements as bm
    height_cm = bm.validate_height_cm(height_cm)
    if run.get("status") != "completed" or not run.get("result"):
        raise ValueError("Metric scale can only be applied to a completed reconstruction")
    root = _run_dir(run["run_id"])
    mpath, opath = root / "measurements.json", root / "final_body_mesh.obj"
    if not mpath.is_file() or not opath.is_file():
        raise ValueError("This run has no measurements.json/final_body_mesh.obj to scale")
    scaled = bm.scale_to_height(json.loads(mpath.read_text(encoding="utf-8")), height_cm)
    verts, faces = smv.read_obj(opath)
    if verts.ndim != 2 or len(verts) == 0 or not np.isfinite(verts).all():
        raise ValueError("final_body_mesh.obj could not be read")
    metres_per_unit = scaled["cm_per_mesh_unit"] / 100.0
    tmp = root / "final_body_mesh_metric.tmp"
    smv.write_obj(tmp, verts * metres_per_unit, faces,
                  header=(f"Unified SMPL body mesh scaled to a supplied barefoot height of {height_cm:.1f} cm.\n"
                          "Units: metres. Frame: y up, +z front, feet on y=0. Pose: measurement A-pose.\n"
                          "Absolute size comes only from the supplied height. "
                          "External geometry only - not a Tanita BIA/BCM measurement."))
    os.replace(tmp, root / "final_body_mesh_metric.obj")
    scaled["mesh_file"] = "final_body_mesh_metric.obj"
    scaled["mesh_units"] = "metres"
    scaled["applied_at"] = time.time()
    _write_json(root / "measurements_cm.json", scaled)
    run["result"]["metric_scale"] = {
        "height_cm": height_cm, "cm_per_mesh_unit": scaled["cm_per_mesh_unit"],
        "relative_scale_uncertainty": scaled["relative_scale_uncertainty"],
        "measurements_cm": {k: m.get("tape_circumference_cm") for k, m in scaled["measurements"].items()},
        "measurements_file": "measurements_cm.json", "mesh_obj": "final_body_mesh_metric.obj"}
    run["result"]["scale_status"] = "metric_from_supplied_height"
    tape_path = root / "tape_validation.json"
    if tape_path.is_file():  # a corrected height changes every model value: keep the comparison current
        previous = json.loads(tape_path.read_text(encoding="utf-8"))
        _write_tape(run, root, scaled, previous["tape_cm"], previous.get("clothing", "unknown"),
                    previous.get("note", ""), previous.get("recorded_at"))
    _persist(run)
    return run


def _write_tape(run: dict, root: Path, scaled: dict, tape_cm: dict, clothing: str, note: str,
                recorded_at=None) -> None:
    from backend.services import body_measurements as bm
    comparison = bm.compare_to_tape(scaled, tape_cm)
    record = {"run_id": run["run_id"], "video_id": run.get("video_id"), "height_cm": scaled["height_cm"],
              "clothing": clothing, "note": note, "tape_cm": bm.validate_tape(tape_cm),
              "recorded_at": recorded_at or time.time(), "compared_at": time.time(),
              "protocol": bm.TAPE_PROTOCOL, **comparison}
    _write_json(root / "tape_validation.json", record)
    run["result"]["tape_validation"] = {"file": "tape_validation.json", "clothing": clothing,
                                        "summary": comparison["summary"],
                                        "sites": {k: {kk: v.get(kk) for kk in ("tape_cm", "model_cm", "error_cm", "error_pct")}
                                                  for k, v in comparison["sites"].items()}}


def _load_run_for_update(run_id: str) -> dict:
    """Caller holds _STATE_LOCK. Returns the live run dict, or the persisted one for runs from earlier sessions."""
    root = _run_dir(run_id)
    run = _RUNS.get(run_id)
    if run is None:
        path = root / "status.json"
        if not path.is_file():
            raise RunNotFoundError("Run not found")
        run = json.loads(path.read_text(encoding="utf-8"))
    return run


def apply_height_scale(run_id: str, height_cm) -> dict:
    """Set absolute scale on a finished run from the subject's known barefoot height (no re-fit)."""
    with _STATE_LOCK:
        _scale_run(_load_run_for_update(run_id), height_cm)
    return get_run(run_id)


def record_tape_measurements(run_id: str, tape_cm: dict, clothing: str = "unknown", note: str = "") -> dict:
    """Store independent tape measurements for a scaled run and compare them with the model's cm values."""
    from backend.services import body_measurements as bm
    if clothing not in bm.CLOTHING:
        raise ValueError(f"clothing must be one of {', '.join(bm.CLOTHING)}")
    note = str(note or "")[:500]
    with _STATE_LOCK:
        run = _load_run_for_update(run_id)
        if run.get("status") != "completed" or not run.get("result"):
            raise ValueError("Tape measurements can only be recorded for a completed reconstruction")
        root = _run_dir(run_id)
        scaled_path = root / "measurements_cm.json"
        if not scaled_path.is_file():
            raise ValueError("Apply the subject's height first: tape values are compared in centimetres")
        _write_tape(run, root, json.loads(scaled_path.read_text(encoding="utf-8")), tape_cm, clothing, note)
        _persist(run)
    return get_run(run_id)


def _worker(run, source: Path, meta: dict, target_box_norm):
    with _RUN_LOCK:
        try:
            result = _pipeline(run, source, meta, target_box_norm)
            with _STATE_LOCK:
                run["result"] = result
                run["status"] = "completed"
                run["stage"] = "done"
                run["progress"] = 1.0
                run["message"] = "Unified body mesh ready"
                _persist(run)
            if run.get("requested_height_cm") is not None:
                try:
                    with _STATE_LOCK:
                        _scale_run(run, run["requested_height_cm"])
                except (ValueError, OSError) as exc:  # the mesh itself is still valid
                    _warn(run, "metric_scale_failed", f"Could not apply the supplied height: {exc}")
                    with _STATE_LOCK:
                        _persist(run)
        except ReconstructionError as exc:
            _fail(run, str(exc), "reconstruction_blocked")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Reconstruction %s crashed", run["run_id"])
            _fail(run, f"Unexpected error: {exc}", "internal_error")


def _fail(run, message, code):
    with _STATE_LOCK:
        run["status"] = "failed"
        run["error"] = {"code": code, "message": message}
        run["message"] = message
        _persist(run)


# ----------------------------------------------------------------- stage 1
def decode_frames(source: Path, meta: dict, work: Path, n_samples: int, run) -> tuple[list[dict], dict]:
    """Decode evenly spaced, correctly oriented frames and verify orientation."""
    cap, rotation = video_processor.VideoCaptureEngine._open_oriented_capture(str(source))
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or float((meta.get("video_info") or {}).get("fps") or 30.0)
        if total <= 0:
            raise ReconstructionError("Video reports no frames; re-record or re-upload the video.")
        # Independent check that rotation is applied exactly once.
        raw = cv2.VideoCapture(str(source), cv2.CAP_FFMPEG)
        raw.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
        ok, raw_frame = raw.read()
        raw.release()
        if not ok:
            raise ReconstructionError("Could not decode the first video frame.")
        rh, rw = raw_frame.shape[:2]
        rot = int(round(rotation)) % 360
        expected_hw = (rw, rh) if rot in (90, 270) else (rh, rw)
        stride = max(1, total // n_samples)
        frames, corrupt, kept = [], [], 0
        prev_small = None
        work.mkdir(parents=True, exist_ok=True)
        idx = 0
        orientation = {"orientation_meta_deg": rot, "raw_decoded_hw": [rh, rw], "expected_displayed_hw": list(expected_hw),
                       "orientation_applied": "ffmpeg_decoder_auto_once"}
        consecutive_fail = 0
        while idx < total:
            if idx % stride != 0:
                if not cap.grab():
                    consecutive_fail += 1
                    if consecutive_fail > 30:
                        break
                else:
                    consecutive_fail = 0
                idx += 1
                continue
            ok, frame = cap.read()
            if not ok or frame is None:
                corrupt.append({"source_frame_index": idx, "reason": "read_failed"})
                consecutive_fail += 1
                idx += 1
                if consecutive_fail > 30:
                    break
                continue
            consecutive_fail = 0
            if tuple(frame.shape[:2]) != tuple(expected_hw):
                raise ReconstructionError(
                    f"Decoded frame {frame.shape[:2]} does not match display orientation {expected_hw}; "
                    "rotation metadata was not applied exactly once.")
            if float(frame.std()) < 2.0:
                corrupt.append({"source_frame_index": idx, "reason": "blank_or_uniform"})
                idx += 1
                continue
            small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (54, 96)).astype(np.float32)
            duplicate = prev_small is not None and float(np.abs(small - prev_small).mean()) < 0.35
            prev_small = small
            path = work / f"s_{kept:04d}.jpg"
            cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 97])
            frames.append({"pos": kept, "source_frame_index": idx, "time_sec": idx / fps,
                           "path": str(path), "duplicate_of_previous": duplicate})
            kept += 1
            idx += 1
            if kept % 10 == 0:
                _progress(run, "decode", idx / total, f"Decoded {kept} frames")
        orientation.update({"decoded_hw": list(expected_hw), "total_frames": total, "fps": fps,
                            "sample_stride": stride, "frames_decoded": kept,
                            "corrupt_or_missing": corrupt,
                            "duplicates_flagged": sum(f["duplicate_of_previous"] for f in frames)})
    finally:
        cap.release()
    if len(frames) < 10:
        raise ReconstructionError("Fewer than 10 usable frames could be decoded from the video.")
    return frames, orientation


# ------------------------------------------------------------ quality helpers
def keypoint_confidence(det: Detection) -> np.ndarray:
    """Map Keypoint R-CNN heat-map logits to [0, 1] weights."""
    return keypoint_confidence01(det.keypoint_scores)


def sharpness(bgr, box) -> float:
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    crop = bgr[max(0, y0):y1, max(0, x0):x1]
    if crop.size == 0:
        return 0.0
    scale = 512.0 / max(crop.shape[0], 1)
    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def frame_usability(det: Detection, image_wh) -> list[str]:
    """Reasons a tracked frame cannot be used for reconstruction (empty = usable)."""
    w, h = image_wh
    x0, y0, x1, y1 = det.box
    reasons = []
    mx, my = 0.004 * w, 0.004 * h
    if x0 < mx or y0 < my or x1 > w - mx or y1 > h - my:
        reasons.append("body_touches_image_border")
    conf = keypoint_confidence(det)
    if (conf > 0.25).sum() < 13:
        reasons.append("heavy_occlusion_or_low_keypoint_confidence")
    for name in ("left_ankle", "right_ankle", "left_hip", "right_hip", "left_shoulder", "right_shoulder"):
        if conf[COCO_NAMES.index(name)] < 0.15:
            reasons.append(f"missing_{name}")
            break
    return reasons


def shoulder_cue(det: Detection) -> float:
    """Signed shoulder separation / box height: >0 anatomical-left shoulder is image-right (facing camera)."""
    ls, rs = det.keypoints[COCO_NAMES.index("left_shoulder")], det.keypoints[COCO_NAMES.index("right_shoulder")]
    return float((ls[0] - rs[0]) / max(det.height, 1.0))


def verify_view(yaw_deg: float, cue: float) -> tuple[str, str]:
    """Cross-check HMR2's heading against the 2D shoulder cue.

    Returns (status, note). 'agree' means two independent cues are consistent;
    otherwise the label stays an unverified estimate.
    """
    c = math.cos(math.radians(yaw_deg))
    expected_sep = 0.25 * c  # frontal shoulder width ~0.25 of body height
    if abs(c) >= 0.5:
        if np.sign(cue) == np.sign(c) and abs(cue) >= 0.4 * abs(expected_sep):
            return "agree", "front/back confirmed by shoulder ordering"
        return "conflict", "HMR2 heading and 2D shoulder ordering disagree"
    if abs(cue) <= 0.14:
        return "agree", "profile confirmed by narrow shoulder separation"
    return "conflict", "HMR2 says profile but shoulders are wide in 2D"


# ----------------------------------------------------------------- HMR2 step
def run_hmr2_on_frame(engine, bgr, det: Detection) -> dict:
    import torch
    from hmr2.datasets.vitdet_dataset import ViTDetDataset
    from hmr2.utils import recursive_to

    h, w = bgr.shape[:2]
    box = det.box.astype(np.float32)
    ds = ViTDetDataset(engine.model_cfg, bgr, box[None])
    batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)))
    center = batch["box_center"][0].numpy().astype(np.float64)
    size = float(batch["box_size"][0])
    batch = recursive_to(batch, engine.device)
    from backend.services.inference_engine import gpu_slot
    with torch.no_grad(), gpu_slot():  # shares the GPU with single-photo jobs one forward pass at a time
        out = engine.model(batch)
    params = out["pred_smpl_params"]
    data = {
        "betas": params["betas"][0].cpu().numpy().astype(np.float64),
        "body_pose": params["body_pose"][0].cpu().numpy().astype(np.float64).reshape(23, 3, 3),
        "global_orient": params["global_orient"][0].cpu().numpy().astype(np.float64).reshape(3, 3),
        "pred_cam": out["pred_cam"][0].cpu().numpy().astype(np.float64),
        "vertices": out["pred_vertices"][0].cpu().numpy().astype(np.float64),
        "box_center": center, "box_size": size,
    }
    for key in ("betas", "body_pose", "global_orient", "pred_cam", "vertices"):
        if not np.isfinite(data[key]).all():
            raise ValueError(f"non-finite {key}")
    for key in ("body_pose", "global_orient"):
        r = data[key].reshape(-1, 3, 3)
        if (not np.allclose(r @ r.transpose(0, 2, 1), np.eye(3), atol=1e-2)
                or not np.allclose(np.linalg.det(r), 1.0, atol=1e-2)):
            raise ValueError(f"{key} not valid rotation matrices")
    if data["pred_cam"][0] <= 0:
        raise ValueError("non-positive HMR2 camera scale")
    return data


def _wait_for_engine(engine, run, timeout=600):
    start = time.time()
    while engine._loading or not (engine.is_ready or engine.load_error):
        if time.time() - start > timeout:
            raise ReconstructionError("HMR2 model did not finish loading in time.")
        time.sleep(0.5)
    if not engine.is_ready:
        raise ReconstructionError(f"HMR2 model is not available: {engine.load_error}")


# ---------------------------------------------------------------- the pipeline
def _pipeline(run, source: Path, meta: dict, target_box_norm) -> dict:
    from backend.services.inference_engine import get_engine

    root = _run_dir(run["run_id"])
    work = root / "work"
    diag = root / "diagnostics"
    diag.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    timings = {}
    _gpu_reset()

    # 1 ------------------------------------------------------------- decode
    _progress(run, "decode", 0.0, "Decoding video with display orientation applied")
    frames, orientation = decode_frames(source, meta, work, N_SAMPLES, run)
    H, W = orientation["decoded_hw"]
    timings["decode"] = time.time() - t0

    # 2 ------------------------------------------------------------- detect
    t1 = time.time()
    dets: list[list[Detection]] = []
    for i, fr in enumerate(frames):
        bgr = cv2.imread(fr["path"])
        if bgr is None:
            raise ReconstructionError(f"Decoded frame {fr['path']} is unreadable.")
        dets.append(get_shared_detector().detect(bgr))
        if i % 5 == 0:
            _progress(run, "detect", i / len(frames), f"Detecting people ({i + 1}/{len(frames)})")
    timings["detect"] = time.time() - t1
    if not any(dets):
        raise ReconstructionError("No person was detected in the video. Check that the full body is visible and well lit.")

    # 3 -------------------------------------------------------------- confirm
    sel_pos = next((i for i, d in enumerate(dets)
                    if any(x.height / H >= MIN_SUBJECT_HEIGHT_FRAC for x in d)), None)
    if sel_pos is None:
        raise ReconstructionError("No person large enough to be the subject was found (subject must span at least 25% of the frame height).")
    cands = [d for d in dets[sel_pos] if d.height / H >= MIN_SUBJECT_HEIGHT_FRAC]
    seed = None
    selection = {"mode": None, "frame_position": sel_pos,
                 "source_frame_index": frames[sel_pos]["source_frame_index"]}
    if target_box_norm is not None:
        box = np.array([target_box_norm[0] * W, target_box_norm[1] * H, target_box_norm[2] * W, target_box_norm[3] * H])
        best = max(cands, key=lambda d: _iou(d.box, box))
        if _iou(best.box, box) < 0.4:
            raise ReconstructionError("The selected person box does not match any detected person; re-select the subject.")
        seed, selection["mode"] = best, "operator_box"
    elif len(cands) == 1:
        seed, selection["mode"] = cands[0], "auto_single_person"
    else:
        bgr = cv2.imread(frames[sel_pos]["path"])
        overview = bgr.copy()
        run_cands = []
        for k, d in enumerate(sorted(cands, key=lambda d: d.box[0])):
            x0, y0, x1, y1 = [int(v) for v in d.box]
            cv2.rectangle(overview, (x0, y0), (x1, y1), (0, 200, 255), 6)
            cv2.putText(overview, str(k + 1), (x0 + 10, y0 + 70), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 200, 255), 6)
            thumb = bgr[max(0, y0):y1, max(0, x0):x1]
            thumb = cv2.resize(thumb, None, fx=360 / max(thumb.shape[0], 1), fy=360 / max(thumb.shape[0], 1))
            cv2.imwrite(str(root / f"candidate_{k}.jpg"), thumb)
            info = candidate_summary(d, (W, H))
            info.update({"id": k, "label": f"Person {k + 1}", "thumbnail": f"candidate_{k}.jpg"})
            run_cands.append(info)
        cv2.imwrite(str(root / "candidates_overview.jpg"), cv2.resize(overview, None, fx=0.4, fy=0.4))
        ordered = sorted(cands, key=lambda d: d.box[0])
        with _STATE_LOCK:
            run["candidates"] = run_cands
            run["status"] = "awaiting_person_selection"
            run["message"] = f"{len(cands)} people detected - confirm which person to reconstruct"
        _progress(run, "confirm", 0.0, run["message"])
        if not _EVENTS[run["run_id"]].wait(timeout=3600):
            raise ReconstructionError("Timed out waiting for the operator to confirm the subject.")
        seed, selection["mode"] = ordered[run["selected_candidate"]], "operator_selection"

    # 4 --------------------------------------------------------------- track
    _progress(run, "track", 0.0, "Tracking the confirmed subject")
    records, events = track_sequence(dets, sel_pos, seed, TrackerConfig())
    tracked = [r for r in records if r.status == "tracked"]
    other_people = sum(max(0, r.n_detections - 1) for r in records)
    if len(tracked) < 0.6 * len(frames):
        raise ReconstructionError(
            f"The subject could be tracked in only {len(tracked)}/{len(frames)} frames "
            "(identity could not be kept consistently). Re-record with only the subject in view, or re-select the person.")
    switches = [e for e in events if e["type"] == "possible_identity_switch"]
    if switches:
        _warn(run, "possible_identity_switch",
              f"{len(switches)} frame(s) were rejected because a nearby detection looked like a different person.")

    # per-frame usability
    usable_positions, rejected = [], []
    for r in records:
        reasons = []
        if r.status != "tracked":
            reasons.append(f"tracking_{r.status}")
        else:
            reasons += frame_usability(r.detection, (W, H))
        if frames[r.index]["duplicate_of_previous"]:
            reasons.append("duplicate_of_previous_frame")
        if reasons:
            rejected.append({"frame_position": r.index, "source_frame_index": frames[r.index]["source_frame_index"],
                             "reasons": reasons})
        else:
            usable_positions.append(r.index)

    # orientation sanity: head must be above feet in the displayed frames
    upright_votes = []
    for r in tracked:
        k = r.detection.keypoints
        head = k[0][1]
        ankle = (k[COCO_NAMES.index("left_ankle")][1] + k[COCO_NAMES.index("right_ankle")][1]) / 2
        hip = (k[COCO_NAMES.index("left_hip")][1] + k[COCO_NAMES.index("right_hip")][1]) / 2
        upright_votes.append(head < hip < ankle)
    upright_frac = float(np.mean(upright_votes)) if upright_votes else 0.0
    orientation["head_above_feet_fraction"] = upright_frac
    if upright_frac < 0.9:
        raise ReconstructionError(
            f"Frame orientation check failed: head is above the feet in only {upright_frac:.0%} of tracked frames. "
            "The video rotation metadata is probably wrong; re-export the video upright.")
    _save_orientation_check(root, frames, records, W, H)

    # 5 --------------------------------------------------------------- HMR2
    engine = get_engine()
    _progress(run, "hmr2", 0.0, "Waiting for HMR2 model")
    _wait_for_engine(engine, run)
    t2 = time.time()
    per_frame = {}
    hmr2_failures = []
    for n, pos in enumerate(usable_positions):
        bgr = cv2.imread(frames[pos]["path"])
        det = records[pos].detection
        try:
            h2 = run_hmr2_on_frame(engine, bgr, det)
        except Exception as exc:  # noqa: BLE001
            hmr2_failures.append({"frame_position": pos, "source_frame_index": frames[pos]["source_frame_index"],
                                  "reason": f"hmr2_failed: {exc}"})
            continue
        h2["sharpness"] = sharpness(bgr, det.box)
        per_frame[pos] = h2
        if n % 3 == 0:
            _progress(run, "hmr2", n / max(len(usable_positions), 1), f"HMR2 inference ({n + 1}/{len(usable_positions)})")
    timings["hmr2"] = time.time() - t2
    if len(per_frame) < 8:
        raise ReconstructionError(f"HMR2 produced valid output for only {len(per_frame)} frames; cannot reconstruct reliably.")
    for f in hmr2_failures:
        rejected.append({"frame_position": f["frame_position"], "source_frame_index": f["source_frame_index"],
                         "reasons": [f["reason"]]})

    # 6 --------------------------------------------------------------- views
    _progress(run, "views", 0.0, "Estimating viewpoints")
    med_sharp = float(np.median([v["sharpness"] for v in per_frame.values()]))
    view_info = {}
    for pos, h2 in per_frame.items():
        yaw = smv.heading_yaw_deg(h2["global_orient"])
        status, note = verify_view(yaw, shoulder_cue(records[pos].detection))
        view_info[pos] = {"yaw_deg": yaw, "label": smv.view_label(yaw), "verification": status,
                          "verification_note": note, "shoulder_cue": shoulder_cue(records[pos].detection)}
    ordered = sorted(per_frame)
    unwrapped = np.unwrap(np.radians([view_info[p]["yaw_deg"] for p in ordered]))
    total_rotation = float(abs(np.degrees(unwrapped[-1] - unwrapped[0])))
    max_step = float(np.degrees(np.max(np.abs(np.diff(unwrapped))))) if len(unwrapped) > 1 else 0.0
    n_conflict = sum(v["verification"] == "conflict" for v in view_info.values())
    if n_conflict > 0.25 * len(view_info):
        _warn(run, "viewpoint_conflicts",
              f"{n_conflict}/{len(view_info)} frames have heading estimates that disagree with 2D shoulder cues; "
              "viewpoint labels are unverified estimates.")

    # 7 -------------------------------------------------------------- select
    _progress(run, "select", 0.0, "Selecting frames across the rotation")
    selected, not_selected = select_frames(per_frame, view_info, med_sharp, records, K_SELECT)
    for pos in not_selected:
        rejected.append({"frame_position": pos, "source_frame_index": frames[pos]["source_frame_index"],
                         "reasons": [not_selected[pos]]})
    counts = {lab: sum(1 for p in selected if view_info[p]["label"] == lab)
              for lab in ("front", "back", "left_side", "right_side")}
    coverage_ok = all(c >= 2 for c in counts.values())
    if not coverage_ok:
        missing = [k for k, c in counts.items() if c < 2]
        _warn(run, "insufficient_viewpoints",
              f"Too few frames for: {', '.join(missing)}. Rotate fully (360 degrees) at a steady pace so every side is visible.")
    if len(selected) < 6:
        raise ReconstructionError(
            f"Only {len(selected)} suitable frames remain after quality filtering; at least 6 spread over the rotation are required.")
    if total_rotation < 270:
        _warn(run, "incomplete_rotation", f"Measured heading change is {total_rotation:.0f} degrees (< 270); the scan may not cover a full turn.")

    # 8 --------------------------------------------------------------- masks
    _progress(run, "masks", 0.0, "Segmenting the subject's silhouette in each selected frame")
    t_m = time.time()
    masks = compute_masks(frames, records, selected, run)
    timings["masks"] = time.time() - t_m
    usable_masks = sum(1 for m in masks.values() if m.usable)
    if usable_masks < 4:
        _warn(run, "few_reliable_silhouettes",
              f"Only {usable_masks}/{len(selected)} silhouettes passed the mask checks; the shape is constrained "
              "mostly by keypoints and the HMR2 prior.")

    # 9 ----------------------------------------------------------------- fit
    _progress(run, "fit", 0.0, "Optimising shared body shape")
    t3 = time.time()
    fit_out = fit_unified(engine, selected, per_frame, records, view_info, (W, H), run, masks)
    timings["fit"] = time.time() - t3

    # 10 ------------------------------------------------------------- export
    _progress(run, "export", 0.0, "Exporting unified mesh")
    result = export_outputs(engine, run, root, frames, records, per_frame, view_info, selected, fit_out,
                            rejected, events, orientation, selection, (W, H), counts, coverage_ok,
                            total_rotation, max_step, other_people, timings, t0, hmr2_failures, med_sharp, meta,
                            masks)
    # remove bulky decoded frames
    for f in work.glob("*.jpg"):
        f.unlink()
    try:
        work.rmdir()
    except OSError:
        pass
    return result


def _iou(a, b):
    from backend.services.person_tracking import iou_xyxy
    return iou_xyxy(a, b)


# ----------------------------------------------------------- frame selection
def select_frames(per_frame: dict, view_info: dict, med_sharp: float, records, k: int):
    """Pick <=k frames: best per 15-degree heading bin, sharp, high keypoint confidence."""
    reasons = {}
    candidates = []
    for pos, h2 in per_frame.items():
        if h2["sharpness"] < 0.35 * med_sharp:
            reasons[pos] = "blurry"
            continue
        conf = float(keypoint_confidence(records[pos].detection).mean())
        score = 0.5 * min(h2["sharpness"] / max(med_sharp, 1e-6), 2.0) / 2.0 + 0.5 * conf
        if view_info[pos]["verification"] == "agree":
            score += 0.25
        candidates.append((pos, score))
    bins: dict[int, tuple[int, float]] = {}
    for pos, score in candidates:
        b = int(((view_info[pos]["yaw_deg"] + 180.0) % 360.0) // YAW_BIN_DEG)
        if b not in bins or score > bins[b][1]:
            if b in bins:
                reasons[bins[b][0]] = "redundant_viewpoint_lower_quality"
            bins[b] = (pos, score)
        else:
            reasons[pos] = "redundant_viewpoint_lower_quality"
    chosen = sorted(bins.values(), key=lambda t: -t[1])
    keep = chosen[:k]
    for pos, _ in chosen[k:]:
        reasons[pos] = "exceeds_frame_budget_lower_quality"
    selected = sorted(p for p, _ in keep)
    return selected, {p: r for p, r in reasons.items() if p not in selected}


# --------------------------------------------------------------------- fitting
def _frame_arrays(per_frame, records, selected, image_wh):
    W, H = image_wh
    focal = HMR2_FOCAL / 224.0 * max(W, H)
    kps, wts, go, pose, cam, bh = [], [], [], [], [], []
    for pos in selected:
        det, h2 = records[pos].detection, per_frame[pos]
        kps.append(det.keypoints)
        w = keypoint_confidence(det).copy()
        w[:5] *= 0.5  # facial keypoints are less reliable across views
        wts.append(w)
        go.append(h2["global_orient"])
        pose.append(h2["body_pose"])
        cam.append(smv.crop_cam_to_full(h2["pred_cam"], h2["box_center"], h2["box_size"], (W, H), focal))
        bh.append(det.height)
    return smv.FitViews(keypoints=np.array(kps), weights=np.array(wts), global_orient=np.array(go),
                        body_pose=np.array(pose), cam_t=np.array(cam), box_height=np.array(bh),
                        center=np.tile([W / 2.0, H / 2.0], (len(selected), 1)), focal=focal)


def _subset(views: smv.FitViews, idx):
    idx = list(idx)

    def pick(seq):
        return None if seq is None else [seq[i] for i in idx]
    return smv.FitViews(keypoints=views.keypoints[idx], weights=views.weights[idx],
                        global_orient=views.global_orient[idx], body_pose=views.body_pose[idx],
                        cam_t=views.cam_t[idx], box_height=views.box_height[idx],
                        center=views.center[idx], focal=views.focal,
                        contours=pick(views.contours), dist_maps=pick(views.dist_maps),
                        contour_weights=pick(views.contour_weights),
                        sil_weight=None if views.sil_weight is None else views.sil_weight[idx])


def _with_params(views: smv.FitViews, fit: smv.FitResult) -> smv.FitViews:
    """Copy of ``views`` whose initial (and prior) pose/camera are a previous fit's result."""
    return smv.FitViews(keypoints=views.keypoints, weights=views.weights, global_orient=fit.global_orient,
                        body_pose=fit.body_pose, cam_t=fit.cam_t, box_height=views.box_height,
                        center=views.center, focal=views.focal, contours=views.contours,
                        dist_maps=views.dist_maps, contour_weights=views.contour_weights,
                        sil_weight=views.sil_weight)


# ------------------------------------------------------------------- masks
def compute_masks(frames, records, selected, run) -> dict:
    """Subject silhouettes for the selected frames (Mask R-CNN on GPU, voted with two CPU sources)."""
    from backend.services.subject_masks import get_shared_segmenter
    seg = get_shared_segmenter()
    out = {}
    try:
        for n, pos in enumerate(selected):
            bgr = cv2.imread(frames[pos]["path"])
            det = records[pos].detection
            out[pos] = seg.segment(bgr, det.box, det.keypoints)
            _progress(run, "masks", (n + 1) / len(selected), f"Segmenting silhouettes ({n + 1}/{len(selected)})")
    finally:
        seg.release()  # keep GPU memory for the optimiser
    return out


def _attach_silhouettes(views: smv.FitViews, selected, records, masks) -> None:
    from backend.services.subject_masks import silhouette_target
    contours, dists, cws, weights = [], [], [], []
    for pos in selected:
        m = masks.get(pos)
        if m is None or not m.usable:
            contours.append(None), dists.append(None), cws.append(None), weights.append(0.0)
            continue
        det = records[pos].detection
        pts, cw, dist = silhouette_target(m.mask, det.box, det.keypoints)
        contours.append(pts), dists.append(dist), cws.append(cw), weights.append(1.0)
    views.contours, views.dist_maps, views.contour_weights = contours, dists, cws
    views.sil_weight = np.array(weights)


# ---------------------------------------------------------------- metrics
def silhouette_scores(smpl, faces, betas, go, pose, cam, views: smv.FitViews, mask_list, device) -> list:
    """Exact rasterised silhouette metrics of posed shared-shape meshes against observed masks."""
    out = smv.posed_output(smpl, betas, go, pose, device)
    verts = out.vertices.cpu().numpy().astype(np.float64)
    scores = []
    for i, mask in enumerate(mask_list):
        if mask is None:
            scores.append(None)
            continue
        uv = smv.project_np(verts[i] + cam[i], views.focal, views.center[i])
        pred = smv.rasterize_silhouette(uv, faces, mask.shape)
        s = smv.silhouette_metrics(pred, mask, float(views.box_height[i]))
        s["torso_width_ratio"] = smv.torso_width_ratios(pred, mask, views.keypoints[i])
        scores.append(s)
    return scores


def proportion_check(scores, yaws) -> dict:
    """Median model/observed torso width per torso level, separately for frontal and profile views.

    Profile views measure body DEPTH, frontal views WIDTH. Against a clothed outline a correct body sits at
    <= 1 everywhere (protruding is impossible) and close to 1 where clothing is tight.
    """
    groups = {"front_back_width": lambda y: abs(y) < 30 or abs(y) > 150, "side_depth": lambda y: 60 < abs(y) < 120}
    out = {"levels_shoulder0_hip1": list(smv.TORSO_LEVELS)}
    for name, sel in groups.items():
        rows = [s["torso_width_ratio"] for s, y in zip(scores, yaws) if s and sel(y) and s.get("torso_width_ratio")]
        arr = np.array([[np.nan if v is None else v for v in r] for r in rows], dtype=float) if rows else None
        out[name] = {"views": len(rows),
                     "median_ratio": (np.nanmedian(arr, 0).round(3).tolist() if arr is not None else None)}
    return out


def per_frame_hmr2_scores(per_frame, selected, idx, views, mask_list, faces) -> list:
    """Each frame's own independent HMR2 mesh (own shape, pose and camera) - the per-frame baseline."""
    scores = []
    for i in idx:
        mask = mask_list[i]
        if mask is None:
            scores.append(None)
            continue
        uv = smv.project_np(per_frame[selected[i]]["vertices"] + views.cam_t[i], views.focal, views.center[i])
        scores.append(smv.silhouette_metrics(smv.rasterize_silhouette(uv, faces, mask.shape), mask,
                                             float(views.box_height[i])))
    return scores


def _summary(scores):
    vals = [s for s in scores if s]
    if not vals:
        return {"views_scored": 0}
    iou = np.array([s["iou"] for s in vals])
    cd = np.array([s["contour_mean_pct_height"] for s in vals if s["contour_mean_pct_height"] is not None])
    return {"views_scored": len(vals), "iou_mean": float(iou.mean()), "iou_median": float(np.median(iou)),
            "iou_min": float(iou.min()),
            # Body outside its own clothed outline (always an error) vs outline left unfilled (may be clothing).
            "protrusion_frac_mean": float(np.mean([s.get("protrusion_frac", np.nan) for s in vals])),
            "uncovered_frac_mean": float(np.mean([s.get("uncovered_frac", np.nan) for s in vals])),
            "contour_distance_mean_pct_height": float(cd.mean()) if len(cd) else None,
            "area_ratio_pred_over_obs_median": float(np.median([s["area_ratio_pred_over_obs"] for s in vals]))}


def _kp_error_pct(fit: smv.FitResult, views: smv.FitViews):
    return float(np.median(fit.residual_px_after / views.box_height[:, None]) * 100)


def cv_folds(selected, view_info, k=4):
    """K interleaved folds over heading, so every fold's held-out views are spread around the body.

    Yields (train_idx, held_idx, nearest_train_gap_deg per held view). Held-out views are never
    used to estimate the shape they validate.
    """
    order = sorted(range(len(selected)), key=lambda i: view_info[selected[i]]["yaw_deg"])
    for f in range(k):
        held = sorted(order[f::k])
        train = sorted(set(range(len(selected))) - set(held))
        if len(train) < 3 or not held:
            continue
        gaps = [min(abs(smv.wrap_deg(view_info[selected[h]]["yaw_deg"] - view_info[selected[t]]["yaw_deg"]))
                    for t in train) for h in held]
        yield train, held, gaps


def _cfg_align():
    return smv.FitConfig(iterations=150)


def _cfg_keypoint_shape():
    """The previous method (kept for comparison): shape from 2D keypoints only."""
    return smv.FitConfig()


def _cfg_silhouette(beta_sigma):
    return smv.FitConfig(iterations=300, w_silhouette=SIL_WEIGHT, w_beta_init=BETA_PRIOR_WEIGHT,
                         w_beta_prior=POP_PRIOR_WEIGHT, sil_coverage_weight=SIL_COVERAGE_WEIGHT,
                         sil_outside_form=SIL_OUTSIDE_FORM, fit_body_pose=FIT_BODY_POSE,
                         beta_sigma=beta_sigma, sil_tau_frac=SIL_TAU_FRAC)


def fit_config_record() -> dict:
    """The exact shape-fit settings of this run, saved in the report so any result can be reproduced later."""
    cfg = _cfg_silhouette(np.ones(10))
    return {"silhouette_weight": cfg.w_silhouette, "outside_term": cfg.sil_outside_form,
            "inside_weight": cfg.sil_inside_weight, "coverage_weight": cfg.sil_coverage_weight,
            "tau_frac_of_box_height": cfg.sil_tau_frac, "hmr2_shape_prior_weight": cfg.w_beta_init,
            "population_shape_prior_weight": cfg.w_beta_prior, "per_view_body_pose_optimised": cfg.fit_body_pose,
            "iterations": cfg.iterations, "beta_sigma_floor": BETA_SIGMA_FLOOR,
            "env_overrides": {k: os.environ[k] for k in sorted(os.environ) if k.startswith("RECON_")}}


def fit_shared_shape(smpl, views, betas_init, beta_sigma, device):
    """Two stages: (A) per-view pose/camera with the HMR2 shape frozen, (B) ONE shared shape
    with per-view pose/camera against keypoints + silhouettes under a Mahalanobis shape prior."""
    align = smv.fit_multiview(smpl, views, betas_init, _cfg_align(), device, fit_shape=False)
    aligned = _with_params(views, align)
    shape = smv.fit_multiview(smpl, aligned, betas_init, _cfg_silhouette(beta_sigma), device)
    return align, aligned, shape


def fit_unified(engine, selected, per_frame, records, view_info, image_wh, run, masks=None) -> dict:
    device = engine.device
    smpl = engine.model.smpl
    faces = np.asarray(engine.smpl_faces, dtype=np.int64)
    masks = masks or {}
    views = _frame_arrays(per_frame, records, selected, image_wh)
    _attach_silhouettes(views, selected, records, masks)
    mask_list = [masks[p].mask if p in masks and masks[p].usable else None for p in selected]
    betas_each = np.array([per_frame[p]["betas"] for p in selected])
    betas_med = np.median(betas_each, axis=0)
    quality = np.array([min(per_frame[p]["sharpness"], 1e9) for p in selected])
    betas_agg, betas_std, beta_outliers = smv.robust_beta_aggregate(betas_each, quality / quality.max())
    beta_sigma = np.maximum(betas_std, BETA_SIGMA_FLOOR)
    _save_fit_inputs(_run_dir(run["run_id"]) / "diagnostics" / "fit_inputs.npz", views, selected, view_info,
                     betas_each, quality, mask_list, per_frame)

    # Sanity check of the HMR2 -> full-image camera conversion on unmodified HMR2 output.
    res0 = smv.keypoint_residuals(smpl, views, betas_each.mean(0), views.global_orient, views.body_pose,
                                  views.cam_t, device).cpu().numpy()
    init_rel = float(np.median(res0 / views.box_height[:, None]))
    if init_rel > 0.12:
        raise ReconstructionError(
            f"HMR2 projection does not match detected keypoints (median error {init_rel:.1%} of body height); "
            "the camera conversion or tracking is inconsistent.")

    # --- full fit (all selected views) -------------------------------------------------
    _progress(run, "fit", 0.05, "Aligning per-view pose and camera with the HMR2 shape frozen")
    align, aligned, full = fit_shared_shape(smpl, views, betas_agg, beta_sigma, device)
    _progress(run, "fit", 0.3, "Checking silhouette agreement per view")
    full_scores = silhouette_scores(smpl, faces, full.betas, full.global_orient, full.body_pose, full.cam_t,
                                    views, mask_list, device)
    ious = np.array([s["iou"] for s in full_scores if s])
    excluded_after_fit = []
    if len(ious):
        limit = max(SIL_EXCLUDE_IOU, float(np.median(ious)) - 0.10)
        for i, s in enumerate(full_scores):
            if s and s["iou"] < limit:
                excluded_after_fit.append({"frame_position": selected[i], "iou": s["iou"],
                                           "reason": f"fitted silhouette IoU {s['iou']:.2f} < {limit:.2f}"})
                views.sil_weight[i] = 0.0
    if excluded_after_fit:
        _progress(run, "fit", 0.35, f"Refitting without {len(excluded_after_fit)} unreliable silhouette(s)")
        align, aligned, full = fit_shared_shape(smpl, views, betas_agg, beta_sigma, device)
        full_scores = silhouette_scores(smpl, faces, full.betas, full.global_orient, full.body_pose, full.cam_t,
                                        views, mask_list, device)

    # Anatomical plausibility: SMPL shape coefficients are in standard deviations of the body-shape
    # population; values beyond +-3 SD mean the silhouette is pushing the body model outside it.
    if np.abs(full.betas).max() > 3.0:
        _warn(run, "implausible_shape_coefficients",
              f"Shared shape coefficient reaches {np.abs(full.betas).max():.1f} SD (> 3): the silhouette may be "
              "distorting the body (loose clothing, mask error). Inspect the comparison images.")

    # Previous method on the same views, for a like-for-like comparison.
    _progress(run, "fit", 0.45, "Fitting the previous keypoint-only shape for comparison")
    kp_only = smv.fit_multiview(smpl, views, betas_med, _cfg_keypoint_shape(), device)
    kp_scores = silhouette_scores(smpl, faces, kp_only.betas, kp_only.global_orient, kp_only.body_pose,
                                  kp_only.cam_t, views, mask_list, device)
    agg_scores = silhouette_scores(smpl, faces, betas_agg, align.global_orient, align.body_pose, align.cam_t,
                                   views, mask_list, device)
    hmr2_scores = per_frame_hmr2_scores(per_frame, selected, range(len(selected)), views, mask_list, faces)

    # --- cross-validated held-out evaluation -------------------------------------------
    # K folds over heading: each view is held out once. Shapes are estimated from the other
    # folds only; the held-out view then gets its pose and camera from its 2D keypoints with
    # the shape frozen. Its silhouette is never used for fitting, so the silhouette IoU there
    # is independent evidence of how well each shared shape generalises to unseen views.
    new_key, old_key = "silhouette_keypoint_shared_shape (new)", "keypoint_only_shared_shape (previous method)"
    agg_key, pf_key = "hmr2_aggregate_shape", "hmr2_independent_per_frame (each frame its own shape)"
    held_scores = {k: [None] * len(selected) for k in (agg_key, old_key, new_key, pf_key)}
    held_kp = {k: [] for k in (agg_key, old_key, new_key)}
    folds_info = []
    for k, (train, held, gaps) in enumerate(cv_folds(selected, view_info, CV_FOLDS)):
        _progress(run, "fit", 0.55 + 0.4 * k / CV_FOLDS, f"Cross-validating shapes (fold {k + 1}/{CV_FOLDS})")
        tv, hv = _subset(views, train), _subset(views, held)
        hv.contours = None  # held-out silhouettes are evaluation-only
        t_agg, t_std, _ = smv.robust_beta_aggregate(betas_each[train], quality[train] / quality.max())
        _, _, t_sil = fit_shared_shape(smpl, tv, t_agg, np.maximum(t_std, BETA_SIGMA_FLOOR), device)
        t_kp = smv.fit_multiview(smpl, tv, np.median(betas_each[train], axis=0), _cfg_keypoint_shape(), device)
        held_masks = [mask_list[i] for i in held]
        for name, b in ((agg_key, t_agg), (old_key, t_kp.betas), (new_key, t_sil.betas)):
            hfit = smv.fit_multiview(smpl, hv, b, _cfg_align(), device, fit_shape=False)
            sc = silhouette_scores(smpl, faces, b, hfit.global_orient, hfit.body_pose, hfit.cam_t, hv, held_masks,
                                   device)
            for j, i in enumerate(held):
                held_scores[name][i] = sc[j]
            held_kp[name].append(_kp_error_pct(hfit, hv))
        for j, s in zip(held, per_frame_hmr2_scores(per_frame, selected, held, views, mask_list, faces)):
            held_scores[pf_key][j] = s
        folds_info.append({"fold": k, "views_train": [selected[i] for i in train],
                           "views_held_out": [selected[i] for i in held],
                           "held_out_nearest_train_heading_gap_deg": gaps})
    held_results = {}
    for name, sc in held_scores.items():
        held_results[name] = {**_summary(sc), "per_view_iou": [s["iou"] if s else None for s in sc]}
        if name in held_kp:
            held_results[name]["keypoint_error_pct_height"] = float(np.mean(held_kp[name]))
    new_iou, old_iou = held_results[new_key].get("iou_mean"), held_results[old_key].get("iou_mean")
    # Held-out IoU is scored against the CLOTHED outline, which rewards a body that fills loose garments, so a
    # shape-prior fit may score slightly lower without being less accurate. Warn only on a clear drop.
    if new_iou is not None and old_iou is not None and new_iou < old_iou - HELD_OUT_IOU_TOLERANCE:
        _warn(run, "silhouette_shape_not_better_on_held_out",
              f"On held-out views the silhouette-fitted shape (IoU {new_iou:.3f}) did not beat the keypoint-only "
              f"shape (IoU {old_iou:.3f}).")
    paired = [(a["iou"], b["iou"]) for a, b in zip(held_scores[new_key], held_scores[old_key]) if a and b]
    heldout = {
        "protocol": f"{CV_FOLDS}-fold cross-validation over heading: each view held out once; shapes estimated from "
                    "the other folds only; held-out views fitted for pose/camera from 2D keypoints with the shape "
                    "frozen; held-out silhouettes never used for fitting",
        "folds": folds_info,
        "views_train": [], "views_held_out": list(selected),
        "held_out_masks_usable": sum(m is not None for m in mask_list),
        "results": held_results,
        "paired_new_vs_previous": {"views": len(paired),
                                   "new_better": sum(a > b for a, b in paired),
                                   "mean_iou_difference": float(np.mean([a - b for a, b in paired])) if paired else None},
        # Backwards-compatible keypoint numbers (as reported by the previous version).
        "median_error_pct_height_median_shape": held_results[agg_key]["keypoint_error_pct_height"],
        "median_error_pct_height_shared_shape": held_results[new_key]["keypoint_error_pct_height"],
    }
    # --- consistency / proportion checks that IoU cannot make ---------------------------------
    yaws = [view_info[p]["yaw_deg"] for p in selected]
    proportions = proportion_check(full_scores, yaws)
    fit_summary = _summary(full_scores)
    prot = fit_summary.get("protrusion_frac_mean")
    if prot is not None and prot > PROTRUSION_WARN:
        _warn(run, "body_protrudes_outside_outline",
              f"The fitted body lies {prot:.1%} outside its own (clothed) silhouette on average; a body cannot be "
              "larger than its clothing, so check masks and camera alignment.")
    side = (proportions.get("side_depth") or {}).get("median_ratio") or []
    if any(v is not None and v > 1.05 for v in side):
        _warn(run, "body_depth_exceeds_profile",
              f"Body depth exceeds the observed profile outline (model/observed {side}).")
    fit_iou, held_iou = fit_summary.get("iou_mean"), held_results[new_key].get("iou_mean")
    if fit_iou is not None and held_iou is not None and fit_iou - held_iou > FIT_HELD_GAP_WARN:
        _warn(run, "per_view_overfit",
              f"Fit IoU {fit_iou:.3f} exceeds held-out IoU {held_iou:.3f} by more than {FIT_HELD_GAP_WARN:.2f}: "
              "per-view parameters are absorbing silhouette error that the shared shape does not explain.")
    _progress(run, "fit", 0.95, "Shape fusion complete")
    # Fitted parameters, so benchmarks can re-pose and re-score the result without re-running inference.
    np.savez_compressed(_run_dir(run["run_id"]) / "diagnostics" / "fit_result.npz",
                        betas=full.betas, global_orient=full.global_orient, body_pose=full.body_pose,
                        cam_t=full.cam_t, keypoint_only_betas=kp_only.betas, betas_aggregate=betas_agg,
                        sil_weight=views.sil_weight if views.sil_weight is not None else np.zeros(len(selected)))
    bh = views.box_height[:, None]
    return {"views": views, "fit": full, "align": align, "kp_only": kp_only, "betas_each": betas_each,
            "betas_median": betas_med, "betas_aggregate": betas_agg, "beta_sigma": beta_sigma,
            "beta_outlier_frames": [selected[i] for i in beta_outliers],
            "init_error_pct_height": init_rel * 100, "heldout": heldout, "mask_list": mask_list,
            "proportions": proportions, "fit_config": fit_config_record(),
            "scores": {"silhouette_keypoint_shared_shape (new)": full_scores,
                       "keypoint_only_shared_shape (previous method)": kp_scores,
                       "hmr2_aggregate_shape": agg_scores,
                       "hmr2_independent_per_frame (each frame its own shape)": hmr2_scores},
            "excluded_after_fit": excluded_after_fit,
            "error_before_pct": float(np.median(align.residual_px_before / bh) * 100),
            "error_after_pct": float(np.median(full.residual_px_after / bh) * 100),
            "kp_only_error_after_pct": float(np.median(kp_only.residual_px_after / bh) * 100)}


# --------------------------------------------------------------------- export
def _upright_from_hmr2(vertices):
    """HMR2 camera frame (y down, z away) -> y-up for viewing diagnostics."""
    return vertices * np.array([1.0, -1.0, -1.0])


def _save_orientation_check(root, frames, records, W, H):
    tracked = [r for r in records if r.status == "tracked"]
    picks = [tracked[0], tracked[len(tracked) // 2], tracked[-1]]
    tiles = []
    for r in picks:
        img = cv2.imread(frames[r.index]["path"])
        x0, y0, x1, y1 = [int(v) for v in r.detection.box]
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 6)
        cv2.putText(img, f"t={frames[r.index]['time_sec']:.1f}s", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 6)
        tiles.append(cv2.resize(img, (360, int(360 * H / W))))
    cv2.imwrite(str(root / "orientation_check.jpg"), np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 90])


def render_mesh_views(vertices, faces, size=420):
    """Flat-shaded orthographic renders at heading 0/90/180/270 (verification aid, no GL needed)."""
    tiles = []
    ymin, ymax = vertices[:, 1].min(), vertices[:, 1].max()
    scale = size * 0.9 / (ymax - ymin)
    for yaw in (0, 90, 180, 270):
        a = math.radians(yaw)
        rot = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
        v = vertices @ rot.T
        img = np.full((size, size, 3), 245, np.uint8)
        px = np.stack([size / 2 + v[:, 0] * scale, size * 0.95 - (v[:, 1] - ymin) * scale], 1)
        tri = v[faces]
        normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
        facing = normal[:, 2] > 0
        depth = tri[:, :, 2].mean(1)
        for fi in np.argsort(depth):
            if not facing[fi]:
                continue
            shade = int(70 + 150 * max(normal[fi, 2], 0))
            cv2.fillConvexPoly(img, px[faces[fi]].astype(np.int32), (shade, shade + 10, shade - 20 if shade > 20 else 0))
        cv2.putText(img, f"{yaw} deg", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2)
        tiles.append(img)
    return np.hstack(tiles)


def export_outputs(engine, run, root, frames, records, per_frame, view_info, selected, fit_out, rejected,
                   events, orientation, selection, image_wh, counts, coverage_ok, total_rotation, max_step,
                   other_people, timings, t0, hmr2_failures, med_sharp, meta, masks=None):
    from backend.services import body_measurements as bm
    device = engine.device
    smpl = engine.model.smpl
    faces = np.asarray(engine.smpl_faces, dtype=np.int64)
    fit = fit_out["fit"]
    views = fit_out["views"]
    masks = masks or {}
    W, H = image_wh

    quality_w = np.array([min(per_frame[p]["sharpness"] / max(med_sharp, 1e-6), 2.0) * keypoint_confidence(records[p].detection).mean()
                          for p in selected])
    # The final mesh is the shared shape in a standard measurement A-pose (arms clear of the torso,
    # thighs apart): body shape is defined in SMPL's rest pose and circumference sections need
    # separated limbs. The mean observed pose is kept as a diagnostic.
    verts, joints, _, spread = smv.canonical_mesh(smpl, fit.betas, fit.body_pose, quality_w, device,
                                                  pose="measurement")
    mean_verts, _, _, _ = smv.canonical_mesh(smpl, fit.betas, fit.body_pose, quality_w, device, pose="mean_observed")
    kp_verts, _, _, _ = smv.canonical_mesh(smpl, fit_out["kp_only"].betas, fit.body_pose, quality_w, device,
                                           pose="measurement")

    header = ("Unified SMPL body mesh (HMR2 initialisation + ONE shared shape fitted to multi-view keypoints and silhouettes)\n"
              "Pose: standard measurement A-pose (arms 45 deg below horizontal).  Frame: y up, +z front, feet on y=0.\n"
              "Scale: RELATIVE / ESTIMATED (SMPL units; not metrically validated).  Shape follows the CLOTHED silhouette.\n"
              "External geometry only - not a Tanita BIA/BCM measurement.")
    obj_path, ply_path = root / "final_body_mesh.obj", root / "final_body_mesh.ply"
    smv.write_obj(obj_path, verts, faces, header)
    smv.write_ply(ply_path, verts, faces)

    # Re-read the file we wrote and validate THAT, not the in-memory array.
    rv, rf = smv.read_obj(obj_path)
    validation = smv.validate_mesh(rv, rf, joints)
    validation["validated_file"] = obj_path.name
    validation["roundtrip_max_abs_error"] = float(np.abs(rv - verts).max())
    if not validation["valid"]:
        raise ReconstructionError("The unified mesh failed geometry validation: " + json.dumps(_clean(validation))[:400])

    cv2.imwrite(str(root / "final_mesh_views.jpg"), render_mesh_views(rv, rf), [cv2.IMWRITE_JPEG_QUALITY, 92])
    _write_turntable(root / "mesh_turntable.gif", rv, rf)

    # ----- circumferences on the exported mesh file -----
    _progress(run, "export", 0.4, "Measuring circumferences on the final mesh")
    j24 = smv.smpl_joints24(smpl, rv)
    measurements = bm.measure_body(rv, rf, j24)
    bm.write_rings_obj(root / "measurement_rings.obj", measurements)
    kp_measure = bm.measure_body(kp_verts, faces, smv.smpl_joints24(smpl, kp_verts))
    _write_measurement_overlay(root / "measurements_overlay.jpg", rv, rf, measurements)
    measurement_json = bm.strip_rings(measurements)
    measurement_json["scale_note"] = ("Values are in relative mesh units and as a fraction of mesh height. No camera "
                                      "calibration or known-size reference exists, so they are NOT centimetres. The "
                                      "shape follows the clothed silhouette (loose clothing inflates girths).")
    measurement_json["previous_method_keypoint_only"] = {
        k: m.get("tape_circumference_over_height") for k, m in kp_measure["measurements"].items()}
    _write_json(root / "measurements.json", measurement_json)

    # ----- diagnostics: per-frame meshes are NOT the result -----
    _progress(run, "export", 0.6, "Writing comparison images")
    diag = root / "diagnostics"
    for i, pos in enumerate(selected):
        src = frames[pos]["source_frame_index"]
        smv.write_obj(diag / f"hmr2_frame_{src:05d}.obj", _upright_from_hmr2(per_frame[pos]["vertices"]), faces,
                      "DIAGNOSTIC single-frame HMR2 mesh (not the final result); flipped to y-up")
    smv.write_obj(diag / "mean_observed_pose_mesh.obj", mean_verts, faces,
                  "DIAGNOSTIC: same shared shape in the mean observed pose")
    smv.write_obj(diag / "previous_method_keypoint_only_mesh.obj", kp_verts, faces,
                  "DIAGNOSTIC: previous keypoint-only shared shape, same measurement pose (for comparison)")
    _write_overlays(engine, diag, frames, records, selected, view_info, fit_out, faces, image_wh)
    _write_contact_sheet(root, frames, records, selected, view_info, image_wh)
    comparison = _write_view_comparison(root / "view_comparison.jpg", engine, frames, records, selected, view_info,
                                        fit_out, faces)
    _write_mask_sheet(root / "silhouette_masks.jpg", frames, records, selected, view_info, masks)
    gpu = _gpu_report(engine)
    if any(gpu["devices"][k] == "cpu" for k in ("hmr2", "person_detection", "segmentation_mask_rcnn", "optimisation")):
        _warn(run, "cpu_fallback", "A GPU-capable stage ran on the CPU: " + json.dumps(gpu["devices"]))

    # ----- reports -----
    tracked = [r for r in records if r.status == "tracked"]
    tracking_report = {
        "subject_selection": selection, "frames_sampled": len(records), "frames_tracked": len(tracked),
        "frames_lost": sum(r.status == "lost" for r in records),
        "frames_ambiguous": sum(r.status == "ambiguous" for r in records),
        "frames_no_detection": sum(r.status == "no_detection" for r in records),
        "other_person_detections": other_people,
        "identity_events": events,
        "identity_switch_events": [e for e in events if e["type"] == "possible_identity_switch"],
        "consistent_single_subject": not any(e["type"] == "possible_identity_switch" for e in events),
        "method": "Keypoint R-CNN detection + IoU/centre/size/HSV-appearance association with strict re-identification",
        "orientation": orientation,
        "frames": [{"frame_position": r.index, "source_frame_index": frames[r.index]["source_frame_index"],
                    "time_sec": frames[r.index]["time_sec"], "status": r.status, "box_xyxy": r.box,
                    "detector_score": r.score, "iou_prev": r.iou_prev, "appearance_distance": r.appearance,
                    "cost": r.cost, "n_detections": r.n_detections, "note": r.note} for r in records],
    }
    _write_json(root / "tracking_report.json", tracking_report)

    selected_report = {
        "selected_count": len(selected), "selection_policy": f"best frame per {YAW_BIN_DEG:.0f}-degree heading bin; sharpness >= 35% of median; "
                                                           "full body inside frame; unoccluded; no duplicates",
        "viewpoint_note": "Heading comes from HMR2's global orientation and is cross-checked against 2D shoulder ordering; "
                          "only entries with verification='agree' are corroborated by a second cue.",
        "coverage": counts, "coverage_sufficient": coverage_ok,
        "measured_heading_rotation_deg": total_rotation, "largest_heading_step_deg": max_step,
        "selected": [_selected_entry(i, p, frames, per_frame, view_info, fit, views, fit_out, masks)
                     for i, p in enumerate(selected)],
        "view_diversity_check": _view_diversity(selected, view_info, masks, records),
        "rejected": rejected,
        "rejected_reason_counts": _count_reasons(rejected),
    }
    _write_json(root / "selected_frames.json", selected_report)

    betas_each = fit_out["betas_each"]
    beta_spread = betas_each.std(0)
    height = float(verts[:, 1].max())
    has_calibration = False
    scores = fit_out["scores"]
    sil_summary = {name: _summary(s) for name, s in scores.items()}
    usable = [p for p in selected if masks.get(p) is not None and masks[p].usable]
    report = {
        "schema_version": 2, "run_id": run["run_id"], "video_id": run["video_id"],
        "architecture": [
            "Oriented decode (display matrix applied once, verified)",
            "Keypoint R-CNN person detection + single-subject tracker with identity-switch detection",
            "Real HMR2 (ViT-H) inference per frame -> SMPL betas, pose, global orientation, camera",
            "Heading estimation from HMR2 global orientation, cross-checked with 2D shoulder cue",
            "Frame selection: one sharp, unoccluded frame per heading bin",
            "Subject silhouettes: Mask R-CNN instance mask of the tracked subject (GPU), GrabCut and MediaPipe "
            "segmenter, combined by majority vote; source agreement gates each silhouette",
            "Shape prior: robust (outlier-rejecting, quality-weighted) median of per-frame HMR2 betas",
            "Stage A: per-view pose/global orientation/camera fitted to 2D keypoints with the shape frozen",
            "Stage B: ONE shared beta vector + per-view orientation/camera (and body pose unless locked) fitted to 2D "
            "keypoints AND silhouettes: non-saturating 'stay inside the outline' term + robust contour coverage; "
            "settings in fit_config",
            "Unreliable silhouettes (low fitted IoU) are excluded and the shape refitted",
            "Held-out validation: shapes from train views only, scored by silhouette IoU on unseen views",
            "Final mesh: shared betas in a standard measurement A-pose, identity orientation (upright, +z front)",
            "Circumferences: plane sections of the final mesh at SMPL-joint anatomical references"],
        "counts": {"frames_decoded": len(frames), "frames_tracked": len(tracked), "frames_hmr2_ok": len(per_frame),
                   "frames_selected": len(selected), "frames_rejected_total": len(rejected),
                   "rejected_reason_counts": _count_reasons(rejected),
                   "corrupt_or_missing": orientation["corrupt_or_missing"], "duplicates_flagged": orientation["duplicates_flagged"]},
        "video": {"source_fps": orientation["fps"], "source_total_frames": orientation["total_frames"],
                  "displayed_size_wh": [W, H], "orientation": orientation},
        "hmr2": {"inference": "real HMR2 checkpoint on " + str(engine.device), "frames_ok": len(per_frame),
                 "frames_failed": hmr2_failures, "per_frame_beta_std": beta_spread,
                 "model_betas_dim": int(betas_each.shape[1])},
        "shared_shape": {"betas": fit.betas, "betas_median_initial": fit_out["betas_median"],
                         "betas_robust_aggregate_prior": fit_out["betas_aggregate"],
                         "beta_prior_sigma": fit_out["beta_sigma"],
                         "beta_outlier_frames_excluded_from_prior": fit_out["beta_outlier_frames"],
                         "beta_l2_change": float(np.linalg.norm(fit.betas - fit_out["betas_median"])),
                         "beta_l2_change_from_prior": float(np.linalg.norm(fit.betas - fit_out["betas_aggregate"])),
                         "beta_mahalanobis_change_from_prior": float(np.sqrt(np.mean(
                             ((fit.betas - fit_out["betas_aggregate"]) / fit_out["beta_sigma"]) ** 2))),
                         "previous_method_keypoint_only_betas": fit_out["kp_only"].betas,
                         "single_beta_vector_for_all_frames": True,
                         "optimiser_loss_history": fit.loss_history},
        "multiview_optimisation": {
            "succeeded": True, "views": len(selected),
            "initial_hmr2_error_pct_height": fit_out["init_error_pct_height"],
            "median_keypoint_error_pct_height_before": fit_out["error_before_pct"],
            "median_keypoint_error_pct_height_after": fit_out["error_after_pct"],
            "median_keypoint_error_pct_height_previous_method": fit_out["kp_only_error_after_pct"],
            "held_out_validation": fit_out["heldout"],
            "camera_model": "shared pinhole with HMR2's assumed focal length (5000/224*max(W,H)); true intrinsics unknown",
            "terms": ["Geman-McClure 2D keypoint reprojection", "pose prior to stage-A pose",
                      "population shape prior (betas ~ N(0, I))", "optional Mahalanobis prior to robust HMR2 shape",
                      "constant camera distance",
                      "silhouette outside-distance (distance transform; Charbonnier, per contour sample)",
                      "silhouette contour coverage (robust chamfer)"],
            "fit_config": fit_out["fit_config"],
            "silhouette_term": "used" if usable else "not used (no reliable silhouettes)"},
        "silhouette_fitting": {
            "views_with_reliable_mask": len(usable),
            "views_excluded_by_mask_checks": [{"frame_position": p, "reasons": masks[p].reasons}
                                              for p in selected if masks.get(p) is not None and not masks[p].usable],
            "views_excluded_after_fit": fit_out["excluded_after_fit"],
            "metric": "exact rasterised silhouette IoU and symmetric contour distance (% of subject box height) "
                      "on the optimisation views",
            "summary_by_method": sil_summary,
            "proportion_check": fit_out["proportions"],
            "metric_notes": [
                "Fit IoU uses per-view orientation/camera (and pose) optimised on that very silhouette; held-out IoU "
                "does not. A large gap means per-view parameters absorb error the shared shape does not explain.",
                "protrusion_frac: share of the rendered body OUTSIDE the clothed outline - always an error.",
                "uncovered_frac: share of the outline the body leaves empty - expected under loose clothing.",
                "proportion_check: model/observed torso width per level; profile views measure body depth.",
                "None of these is ground truth: absolute 3D accuracy needs tape or a 3D scan."],
            "note": "Silhouettes include clothing, hair and shoes; IoU measures agreement with the clothed outline."},
        "canonical_pose": {"method": "standard measurement A-pose (arms 45 deg below horizontal, legs abducted 5 deg); "
                                     "the observed poses are explained per view and do not change the shape",
                           "observed_pose_spread_deg_per_joint": spread, "joints": 23,
                           "mean_angular_spread_deg_per_joint": spread,
                           "mean_observed_pose_mesh": "diagnostics/mean_observed_pose_mesh.obj"},
        "measurements": {"file": "measurements.json", "valid": measurements["valid_count"],
                         "total": measurements["total"], "units": measurements["units"],
                         "tape_circumference_over_height": {
                             k: m.get("tape_circumference_over_height") for k, m in measurements["measurements"].items()},
                         "cross_section_validity": {k: m["valid"] for k, m in measurements["measurements"].items()}},
        "scale": {"status": "relative_estimated_not_metric", "calibration_or_known_reference_found": has_calibration,
                  "mesh_height_in_smpl_units": height,
                  "note": "No camera calibration or known-size reference exists for this recording, and a single RGB camera "
                          "with unknown subject distance cannot fix absolute size. The SMPL unit height above is NOT a measured "
                          "body height; circumferences are in the same relative units and as a fraction of mesh height, "
                          "not centimetres. Nothing here measures body fat, muscle or other internal composition."},
        "scope": "External body geometry only. This is not a Tanita BIA/BCM body-composition measurement and nothing here "
                 "measures internal composition.",
        "geometry_validation": validation,
        "validation_summary": {
            "frames_extracted": len(frames), "frames_tracked": len(tracked), "frames_selected_for_optimisation":
                len(fit_out["heldout"]["views_train"]), "frames_held_out_for_validation": len(fit_out["heldout"]["views_held_out"]),
            "frames_in_final_fit": len(selected), "frames_excluded": len(rejected), "viewpoint_distribution": counts,
            "shared_shape_optimisation": "succeeded",
            "silhouette_iou_mean_final_fit": sil_summary.get("silhouette_keypoint_shared_shape (new)", {}).get("iou_mean"),
            "silhouette_contour_error_pct_height_final_fit":
                sil_summary.get("silhouette_keypoint_shared_shape (new)", {}).get("contour_distance_mean_pct_height"),
            "landmark_reprojection_error_pct_height": fit_out["error_after_pct"],
            "held_out_silhouette_iou_by_method": {k: v.get("iou_mean") for k, v in fit_out["heldout"]["results"].items()},
            "mesh_topology_valid": validation["valid"], "mesh_watertight": validation.get("watertight"),
            "mesh_upright": validation.get("upright"), "mesh_facing_plus_z": validation.get("facing_plus_z"),
            "mesh_component_count": validation.get("connected_components"),
            "circumference_sections_valid": f"{measurements['valid_count']}/{measurements['total']}"},
        "comparison_images": comparison,
        "gpu": gpu,
        "outputs": {"final_body_mesh.obj": obj_path.stat().st_size, "final_body_mesh.ply": ply_path.stat().st_size,
                    "selected_frames.json": True, "tracking_report.json": True, "measurements.json": True,
                    "measurement_rings.obj": True, "view_comparison.jpg": True, "mesh_turntable.gif": True,
                    "diagnostics_dir": "diagnostics/ (single-frame meshes and overlays; NOT the final result)"},
        "limitations": [
            "Absolute scale is unknown (no calibration or known-size reference): all sizes are relative.",
            "SMPL is a minimally-clothed body model fitted to the CLOTHED silhouette; loose clothing inflates girth.",
            "Shape is limited to SMPL's 10 shape coefficients; no per-vertex surface refinement is applied.",
            "Camera intrinsics are HMR2's assumed focal length, not calibrated values.",
            "Viewpoint labels come from HMR2 heading cross-checked with 2D cues; they are not ground truth."],
        "warnings": run.get("warnings", []),
        "timings_sec": {**timings, "total": time.time() - t0},
        "consent": {"authorization_checked": True, "note": "Operator attestation recorded at upload; not independently verified."},
    }
    _write_json(root / "reconstruction_report.json", report)

    new = fit_out["heldout"]["results"].get("silhouette_keypoint_shared_shape (new)", {})
    old = fit_out["heldout"]["results"].get("keypoint_only_shared_shape (previous method)", {})
    return {"mesh_obj": "final_body_mesh.obj", "mesh_ply": "final_body_mesh.ply",
            "vertex_count": validation["vertex_count"], "face_count": validation["face_count"],
            "geometry_valid": validation["valid"], "scale_status": "relative_estimated_not_metric",
            "frames_selected": len(selected), "frames_tracked": len(tracked),
            "coverage_sufficient": coverage_ok, "consistent_single_subject": tracking_report["consistent_single_subject"],
            "views_preview": "final_mesh_views.jpg", "warnings": run.get("warnings", []),
            "silhouette_views": len(usable),
            "silhouette_iou_fit": sil_summary.get("silhouette_keypoint_shared_shape (new)", {}).get("iou_mean"),
            "held_out_iou": new.get("iou_mean"), "held_out_iou_previous_method": old.get("iou_mean"),
            "silhouette_protrusion_fit": sil_summary.get("silhouette_keypoint_shared_shape (new)", {}).get(
                "protrusion_frac_mean"),
            "held_out_protrusion": new.get("protrusion_frac_mean"),
            "proportion_check": fit_out["proportions"],
            "held_out_views": len(fit_out["heldout"]["views_held_out"]),
            "keypoint_error_pct_height": fit_out["error_after_pct"],
            "measurements_valid": f"{measurements['valid_count']}/{measurements['total']}",
            "measurements": {k: m.get("tape_circumference_over_height") for k, m in measurements["measurements"].items()},
            "pose": "measurement_a_pose", "turntable": "mesh_turntable.gif", "comparison": "view_comparison.jpg",
            "measurements_overlay": "measurements_overlay.jpg", "gpu": gpu.get("device_name")}


def _count_reasons(rejected):
    counts: dict[str, int] = {}
    for item in rejected:
        for reason in item["reasons"]:
            key = reason.split(":")[0]
            counts[key] = counts.get(key, 0) + 1
    return counts


def _write_contact_sheet(root, frames, records, selected, view_info, image_wh):
    W, H = image_wh
    order = sorted(selected, key=lambda p: view_info[p]["yaw_deg"])
    tiles = []
    for pos in order:
        img = cv2.imread(frames[pos]["path"])
        x0, y0, x1, y1 = [int(v) for v in records[pos].detection.box]
        pad = int(0.05 * (y1 - y0))
        crop = img[max(0, y0 - pad):y1 + pad, max(0, x0 - 4 * pad):x1 + 4 * pad]
        crop = cv2.resize(crop, (170, 340))
        label = f"{view_info[pos]['yaw_deg']:+.0f} {view_info[pos]['label'][:5]}"
        cv2.putText(crop, label, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        tiles.append(crop)
    cols = 12
    rows = [np.hstack(tiles[i:i + cols] + [np.full_like(tiles[0], 255)] * (cols - len(tiles[i:i + cols])))
            for i in range(0, len(tiles), cols)]
    cv2.imwrite(str(root / "contact_sheet.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])


def _write_overlays(engine, diag, frames, records, selected, view_info, fit_out, faces, image_wh):
    """Project the optimised shared-shape SMPL onto source frames (visual verification of the fit)."""
    import torch
    smpl = engine.model.smpl
    views, fit = fit_out["views"], fit_out["fit"]
    f32 = dict(dtype=torch.float32, device=engine.device)
    picks = {}
    for lab in ("front", "left_side", "back", "right_side"):
        c = [i for i, p in enumerate(selected) if view_info[p]["label"] == lab]
        if c:
            target = {"front": 0, "left_side": -90, "back": 180, "right_side": 90}[lab]
            picks[lab] = min(c, key=lambda i: abs(smv.wrap_deg(view_info[selected[i]]["yaw_deg"] - target)))
    for lab, i in picks.items():
        pos = selected[i]
        with torch.no_grad():
            out = smpl(betas=torch.as_tensor(fit.betas, **f32)[None],
                       global_orient=torch.as_tensor(fit.global_orient[i], **f32)[None, None],
                       body_pose=torch.as_tensor(fit.body_pose[i], **f32)[None], pose2rot=False)
            cam = torch.as_tensor(fit.cam_t[i], **f32)
            uv = smv.project((out.vertices + cam)[:1], views.focal,
                             torch.as_tensor(views.center[i:i + 1], **f32))[0].cpu().numpy()
        img = cv2.imread(frames[pos]["path"])
        for u, v in uv[::3]:
            cv2.circle(img, (int(u), int(v)), 2, (0, 255, 0), -1)
        for (u, v) in records[pos].detection.keypoints:
            cv2.circle(img, (int(u), int(v)), 8, (0, 0, 255), -1)
        cv2.imwrite(str(diag / f"fit_overlay_{lab}.jpg"), cv2.resize(img, None, fx=0.5, fy=0.5), [cv2.IMWRITE_JPEG_QUALITY, 90])


# ------------------------------------------------------- per-frame reporting
def _selected_entry(i, p, frames, per_frame, view_info, fit, views, fit_out, masks):
    fold = next((f["fold"] for f in fit_out["heldout"]["folds"] if p in f["views_held_out"]), None)
    cv_iou = fit_out["heldout"]["results"]["silhouette_keypoint_shared_shape (new)"]["per_view_iou"][i]
    m = masks.get(p)
    yaw = view_info[p]["yaw_deg"]
    lo = math.floor((yaw + 180.0) / YAW_BIN_DEG) * YAW_BIN_DEG - 180.0
    sil_used = bool(views.sil_weight is not None and views.sil_weight[i] > 0)
    if sil_used:
        contribution = "keypoints + silhouette constrain the shared shape"
    elif m is not None and not m.usable:
        contribution = "keypoints only (silhouette rejected: " + "; ".join(m.reasons) + ")"
    else:
        contribution = "keypoints only (silhouette excluded after fit: low agreement with the shared shape)"
    score = fit_out["scores"]["silhouette_keypoint_shared_shape (new)"][i]
    return {"frame_position": p, "source_frame_index": frames[p]["source_frame_index"],
            "time_sec": frames[p]["time_sec"], "yaw_deg": yaw,
            "view": view_info[p]["label"], "view_verification": view_info[p]["verification"],
            "verification_note": view_info[p]["verification_note"],
            "sharpness": per_frame[p]["sharpness"],
            "diagnostic_mesh": f"hmr2_frame_{frames[p]['source_frame_index']:05d}.obj",
            "why_selected": f"sharpest well-tracked frame in heading bin [{lo:.0f}, {lo + YAW_BIN_DEG:.0f}) deg",
            "role": "final fit; held out in cross-validation fold " + str(fold),
            "cross_validated_silhouette_iou": cv_iou,
            "contribution": contribution,
            "silhouette_mask": None if m is None else {"usable": m.usable, "source_agreement": m.agreement,
                                                       "box_fill": m.box_fill, "reasons": m.reasons},
            "silhouette_iou_after_fit": score["iou"] if score else None,
            "keypoint_residual_pct_height_after_fit": float(np.median(fit.residual_px_after[i] / views.box_height[i]) * 100)}


def _view_diversity(selected, view_info, masks, records):
    """Image evidence for view diversity: silhouette torso width should shrink toward profile views.

    Spearman correlation between |cos(heading)| and silhouette torso width / height. A clearly
    positive value means the HMR2 headings are consistent with what the images show.
    """
    xs, ys = [], []
    for p in selected:
        m = masks.get(p)
        if m is None or not m.usable:
            continue
        k = records[p].detection.keypoints
        y0 = int(min(k[5, 1], k[6, 1]))
        y1 = int(max(k[11, 1], k[12, 1]))
        if y1 <= y0:
            continue
        widths = m.mask[y0:y1].sum(1)
        xs.append(abs(math.cos(math.radians(view_info[p]["yaw_deg"]))))
        ys.append(float(np.median(widths)) / max(records[p].detection.height, 1.0))
    if len(xs) < 5:
        return {"evaluated": False, "reason": "fewer than 5 usable silhouettes"}
    rx, ry = np.argsort(np.argsort(xs)), np.argsort(np.argsort(ys))
    rho = float(np.corrcoef(rx, ry)[0, 1])
    return {"evaluated": True, "spearman_abs_cos_heading_vs_torso_width": rho, "views": len(xs),
            "interpretation": ("positive: frontal/back views have wider torso silhouettes than profiles, consistent "
                               "with the HMR2 headings") if rho > 0.3 else
                              "weak or negative: headings and silhouette widths disagree; treat view labels with caution"}


# ------------------------------------------------------------- rendering
def render_camera_view(verts_cam, faces, focal, center, crop, size_h=480):
    """Flat-shaded render of a posed mesh through the fitted camera, cropped like the source image."""
    x0, y0, x1, y1 = crop
    s = size_h / float(y1 - y0)
    w = int(round((x1 - x0) * s))
    img = np.full((size_h, w, 3), 235, np.uint8)
    uv = smv.project_np(verts_cam, focal, center)
    px = ((uv - [x0, y0]) * s * 4).astype(np.int32)
    tri = verts_cam[faces]
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    facing = normal[:, 2] < 0  # camera looks along +z
    for fi in np.argsort(-tri[:, :, 2].mean(1)):
        if facing[fi]:
            shade = int(60 + 160 * max(-normal[fi, 2], 0))
            cv2.fillConvexPoly(img, px[faces[fi]], (shade, shade + 10, max(shade - 20, 0)), shift=2)
    return img


def _crop_box(box, shape, pad=0.06):
    h, w = shape[:2]
    x0, y0, x1, y1 = box
    p = pad * (y1 - y0)
    return int(max(0, x0 - 2 * p)), int(max(0, y0 - p)), int(min(w, x1 + 2 * p)), int(min(h, y1 + p))


def _write_view_comparison(path, engine, frames, records, selected, view_info, fit_out, faces) -> dict:
    """Original | observed (red) vs fitted shared-shape (green) outline | render, for front/left/back/right."""
    smpl = engine.model.smpl
    fit, views = fit_out["fit"], fit_out["views"]
    mask_list = fit_out["mask_list"]
    cv_iou = fit_out["heldout"]["results"]["silhouette_keypoint_shared_shape (new)"]["per_view_iou"]
    out = smv.posed_output(smpl, fit.betas, fit.global_orient, fit.body_pose, engine.device)
    verts = out.vertices.cpu().numpy().astype(np.float64)
    rows, info = [], {}
    for lab, target in (("front", 0), ("left_side", -90), ("back", 180), ("right_side", 90)):
        cands = [i for i, p in enumerate(selected) if view_info[p]["label"] == lab]
        if not cands:
            info[lab] = None
            continue
        i = min(cands, key=lambda k: abs(smv.wrap_deg(view_info[selected[k]]["yaw_deg"] - target)))
        pos = selected[i]
        img = cv2.imread(frames[pos]["path"])
        crop = _crop_box(records[pos].detection.box, img.shape)
        vc = verts[i] + fit.cam_t[i]
        uv = smv.project_np(vc, views.focal, views.center[i])
        pred = smv.rasterize_silhouette(uv, faces, img.shape[:2])
        over = img.copy()
        score = None
        if mask_list[i] is not None:
            cnt, _ = cv2.findContours(mask_list[i], cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(over, cnt, -1, (0, 0, 255), 4)
            score = smv.silhouette_metrics(pred, mask_list[i], float(views.box_height[i]))
        cnt, _ = cv2.findContours(pred, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(over, cnt, -1, (0, 220, 0), 4)
        x0, y0, x1, y1 = crop
        tiles = [cv2.resize(t[y0:y1, x0:x1], None, fx=480 / (y1 - y0), fy=480 / (y1 - y0)) for t in (img, over)]
        render = render_camera_view(vc, faces, views.focal, views.center[i], crop)
        tiles.append(cv2.resize(render, (tiles[0].shape[1], tiles[0].shape[0])))
        row = np.hstack(tiles)
        label = (f"{lab}  yaw {view_info[pos]['yaw_deg']:+.0f}  " + (f"fit IoU {score['iou']:.3f}" if score else "no mask")
                 + (f"  held-out IoU {cv_iou[i]:.3f}" if cv_iou[i] is not None else ""))
        rows.append((row, label))
        info[lab] = {"frame_position": pos, "yaw_deg": view_info[pos]["yaw_deg"],
                     "iou_final_fit": score["iou"] if score else None, "iou_cross_validated_held_out": cv_iou[i]}
    if rows:
        width = max(r.shape[1] for r, _ in rows)
        padded = []
        for r, label in rows:
            r = np.pad(r, ((34, 0), (0, width - r.shape[1]), (0, 0)), constant_values=255)  # label band on top
            cv2.putText(r, label, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2)
            padded.append(r)
        cv2.imwrite(str(path), np.vstack(padded), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return {"file": path.name, "columns": ["original", "observed silhouette (red) vs fitted shared shape (green)",
                                           "rendered fitted mesh"], "views": info}


def _render_turn(vertices, faces, yaw, size=360):
    a = math.radians(yaw)
    rot = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    v = vertices @ rot.T
    ymin, ymax = vertices[:, 1].min(), vertices[:, 1].max()
    scale = size * 0.9 / (ymax - ymin)
    img = np.full((size, size, 3), 245, np.uint8)
    px = (np.stack([size / 2 + v[:, 0] * scale, size * 0.95 - (v[:, 1] - ymin) * scale], 1) * 4).astype(np.int32)
    tri = v[faces]
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    for fi in np.argsort(tri[:, :, 2].mean(1)):
        if normal[fi, 2] > 0:
            shade = int(70 + 150 * normal[fi, 2])
            cv2.fillConvexPoly(img, px[faces[fi]], (shade, shade + 10, max(shade - 20, 0)), shift=2)
    return img


def _write_turntable(path, vertices, faces, steps=36):
    from PIL import Image
    images = [Image.fromarray(cv2.cvtColor(_render_turn(vertices, faces, k * 360 / steps), cv2.COLOR_BGR2RGB))
              for k in range(steps)]
    images[0].save(path, save_all=True, append_images=images[1:], duration=90, loop=0)


def _write_measurement_overlay(path, vertices, faces, measurements, size=520):
    """Front and side renders with each valid measurement ring drawn on the mesh."""
    colors = [(0, 0, 230), (0, 140, 255), (0, 200, 200), (0, 180, 0), (200, 120, 0), (200, 0, 160)]
    tiles = []
    ymin, ymax = vertices[:, 1].min(), vertices[:, 1].max()
    scale = size * 0.9 / (ymax - ymin)
    for yaw in (0, 90):
        img = _render_turn(vertices, faces, yaw, size)
        a = math.radians(yaw)
        rot = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
        for k, (name, m) in enumerate(measurements["measurements"].items()):
            if not m.get("valid"):
                continue
            ring = np.asarray(m["ring"]) @ rot.T
            pts = np.stack([size / 2 + ring[:, 0] * scale, size * 0.95 - (ring[:, 1] - ymin) * scale], 1)
            cv2.polylines(img, [pts.astype(np.int32)], True, colors[k % len(colors)], 2)
        tiles.append(img)
    legend = np.full((size, 300, 3), 255, np.uint8)
    for k, (name, m) in enumerate(measurements["measurements"].items()):
        text = (f"{name}: {m['tape_circumference_over_height']:.3f} x H" if m.get("valid") else f"{name}: invalid")
        cv2.putText(legend, text, (8, 24 + 22 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[k % len(colors)], 1)
    cv2.putText(legend, "relative units (x mesh height)", (8, size - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (60, 60, 60), 1)
    cv2.imwrite(str(path), np.hstack(tiles + [legend]), [cv2.IMWRITE_JPEG_QUALITY, 92])


def _write_mask_sheet(path, frames, records, selected, view_info, masks):
    tiles = []
    for pos in sorted(selected, key=lambda p: view_info[p]["yaw_deg"]):
        m = masks.get(pos)
        if m is None:
            continue
        img = cv2.imread(frames[pos]["path"])
        cnt, _ = cv2.findContours(m.mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, cnt, -1, (0, 200, 0) if m.usable else (0, 0, 255), 5)
        x0, y0, x1, y1 = _crop_box(records[pos].detection.box, img.shape)
        t = cv2.resize(img[y0:y1, x0:x1], (170, 340))
        cv2.putText(t, f"{view_info[pos]['yaw_deg']:+.0f} a{m.agreement:.2f}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 0, 255), 1)
        tiles.append(t)
    if not tiles:
        return
    cols = 12
    rows = [np.hstack(tiles[i:i + cols] + [np.full_like(tiles[0], 255)] * (cols - len(tiles[i:i + cols])))
            for i in range(0, len(tiles), cols)]
    cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])


# --------------------------------------------------------------------- GPU
def _gpu_reset():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:  # noqa: BLE001
        pass


def _gpu_report(engine) -> dict:
    import torch
    from backend.services.person_tracking import get_shared_detector
    from backend.services.subject_masks import get_shared_segmenter
    cuda = torch.cuda.is_available()
    seg = get_shared_segmenter()
    return {"cuda_available": cuda, "torch": torch.__version__,
            "device_name": torch.cuda.get_device_name(0) if cuda else None,
            "peak_memory_allocated_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1) if cuda else None,
            "devices": {"hmr2": str(engine.device), "person_detection": str(get_shared_detector()._device),
                        "segmentation_mask_rcnn": str(seg._device), "optimisation": str(engine.device),
                        "segmentation_refinement": "cpu (GrabCut, MediaPipe TFLite)",
                        "rasterisation_and_measurement": "cpu"}}


def _save_fit_inputs(path, views, selected, view_info, betas_each, quality, mask_list, per_frame):
    """Everything the shape fit consumes, so a fit can be audited or reproduced without re-running inference."""
    shape = next((m.shape for m in mask_list if m is not None), (1, 1))
    packed = np.stack([np.packbits(m if m is not None else np.zeros(shape, np.uint8), axis=None) for m in mask_list])
    np.savez_compressed(path, keypoints=views.keypoints, weights=views.weights, global_orient=views.global_orient,
                        body_pose=views.body_pose, cam_t=views.cam_t, box_height=views.box_height,
                        center=views.center, focal=views.focal, selected=np.array(selected),
                        yaw=np.array([view_info[p]["yaw_deg"] for p in selected]), betas_each=betas_each,
                        quality=quality, mask_shape=np.array(shape), masks_packed=packed,
                        mask_usable=np.array([m is not None for m in mask_list]),
                        hmr2_vertices=np.array([per_frame[p]["vertices"] for p in selected], dtype=np.float32))
