"""Unified 360-degree reconstruction: one video in, one body mesh out."""
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend.services import body_reconstruction as recon
from backend.services import measurement_validation as mv

router = APIRouter()


class StartRequest(BaseModel):
    video_id: str
    # Optional: normalised [x0, y0, x1, y1] of the intended person on the selection frame.
    target_box_norm: list[float] | None = Field(default=None, min_length=4, max_length=4)
    # Optional barefoot standing height; when given, cm measurements are written on completion.
    height_cm: float | None = None
    # Optional: "male" / "female" selects a sex-specific body-shape prior (fat distribution); empty = unspecified.
    sex: str | None = None


class PersonRequest(BaseModel):
    candidate_id: int


class ScaleRequest(BaseModel):
    height_cm: float


class TapeRequest(BaseModel):
    # Independent tape circumferences in cm, keyed by site (chest, waist, hip, thigh_left, ...).
    tape_cm: dict[str, float | None]
    clothing: str = "unknown"   # minimal | tight | loose | unknown — what the subject wore in the video
    note: str = Field(default="", max_length=500)


@router.post("/runs", status_code=status.HTTP_202_ACCEPTED)
def start(request: StartRequest):
    try:
        return recon.start_run(request.video_id, request.target_box_norm, request.height_cm, request.sex)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/runs")
def list_runs(limit: int = 50):
    """Completed runs with measurements, newest first (history for the Body Avatar comparison)."""
    return {"runs": recon.list_completed_runs(max(1, min(limit, 200)))}


@router.get("/runs/{run_id}")
def run_status(run_id: str):
    try:
        run = recon.get_run(run_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@router.post("/runs/{run_id}/person")
def confirm_person(run_id: str, request: PersonRequest):
    try:
        return recon.select_person(run_id, request.candidate_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/runs/{run_id}/scale")
def apply_scale(run_id: str, request: ScaleRequest):
    """Convert a completed run's measurements to centimetres from the subject's known height."""
    try:
        return recon.apply_height_scale(run_id, request.height_cm)
    except recon.RunNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/runs/{run_id}/tape")
def record_tape(run_id: str, request: TapeRequest):
    """Compare a scaled run's circumferences with independent tape measurements of the same subject."""
    try:
        return recon.record_tape_measurements(run_id, request.tape_cm, request.clothing, request.note)
    except recon.RunNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


class ConditionsRequest(BaseModel):
    # Capture conditions for validation: who (a code, not a name), clothing, camera distance, posture.
    subject_id: str
    clothing: str = "unknown"          # minimal | tight | loose | unknown
    distance_m: float | None = None    # camera to subject
    posture: str = "a_pose_standard"   # a_pose_standard | arms_lower | arms_higher | relaxed_slouch | other
    height_cm: float | None = None     # stadiometer height, if measured
    note: str = Field(default="", max_length=500)


@router.get("/runs/{run_id}/conditions")
def get_conditions(run_id: str):
    """Capture conditions recorded for a run (empty object when none were recorded)."""
    try:
        run_dir = recon._run_dir(run_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return mv._read_json(run_dir / mv.CONDITIONS_FILE) or {}


@router.post("/runs/{run_id}/conditions")
def record_conditions(run_id: str, request: ConditionsRequest):
    """Tag a completed run with its capture conditions (used by scripts/validation_report.py)."""
    try:
        run_dir = recon._run_dir(run_id)
        if not (run_dir / "status.json").is_file():
            raise HTTPException(status_code=404, detail="Run not found")
        return mv.write_conditions(run_dir, request.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/validation/report")
def validation_report(subject: str | None = None, fit_config: str = "latest"):
    """Repeatability, condition effects and tape accuracy pooled over all completed runs."""
    return mv.build_report(recon._ROOT, recon._source_key, default_subject=subject, fit_config=fit_config)


@router.get("/runs/{run_id}/files/{name}")
def run_file(run_id: str, name: str):
    try:
        path = recon.get_asset(run_id, name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if path is None:
        raise HTTPException(status_code=404, detail="File not found")
    media = {".obj": "text/plain", ".ply": "application/octet-stream", ".json": "application/json",
             ".jpg": "image/jpeg", ".gif": "image/gif"}.get(path.suffix)
    return FileResponse(path, media_type=media, filename=path.name if path.suffix in (".obj", ".ply") else None)
