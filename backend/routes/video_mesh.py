"""Endpoints for batches of independent single-frame meshes."""
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.services.video_mesh_batch import get_asset, get_run, start_batch

router = APIRouter()


class BatchRequest(BaseModel):
    video_id: str
    frames: list[str]


@router.post("/runs", status_code=status.HTTP_202_ACCEPTED)
def create_run(request: BatchRequest):
    try:
        return start_batch(request.video_id, request.frames)
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
