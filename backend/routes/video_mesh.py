"""Endpoints for batches of independent single-frame meshes."""
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, StrictBool

from backend.services.video_mesh_batch import check_readiness, get_asset, get_run, start_batch

router = APIRouter()


class ReadinessRequest(BaseModel):
    video_id: str
    frames: list[str]
    quality_run_id: str | None = None
    reviewed_frames: list[str] = Field(default_factory=list)
    review_session_id: str | None = None


class BatchRequest(BaseModel):
    video_id: str
    frames: list[str]
    quality_run_id: str
    reviewed_frames: list[str]
    review_session_id: str


class ConfirmedBatchRequest(BatchRequest):
    final_confirmation: StrictBool = False


@router.post("/readiness")
def readiness(request: ReadinessRequest):
    return check_readiness(request.video_id, request.frames, request.quality_run_id,
                           request.reviewed_frames, request.review_session_id)


@router.post("/runs", status_code=status.HTTP_202_ACCEPTED)
def create_run(request: ConfirmedBatchRequest):
    if request.final_confirmation is not True:
        raise HTTPException(status_code=400, detail="Explicit final confirmation is required")
    try:
        return start_batch(request.video_id, request.frames, request.quality_run_id, request.reviewed_frames, request.review_session_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/runs/{run_id}")
def run_status(run_id: str):
    try:
        run = get_run(run_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@router.get("/runs/{run_id}/{kind}/{filename}")
def run_asset(run_id: str, kind: str, filename: str):
    try:
        path = get_asset(run_id, filename, kind)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if path is None:
        raise HTTPException(status_code=404, detail="Asset not found")
    return FileResponse(path, media_type="application/octet-stream" if kind == "mesh" else None)
