"""Unified 360-degree reconstruction: one video in, one body mesh out."""
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend.services import body_reconstruction as recon

router = APIRouter()


class StartRequest(BaseModel):
    video_id: str
    # Optional: normalised [x0, y0, x1, y1] of the intended person on the selection frame.
    target_box_norm: list[float] | None = Field(default=None, min_length=4, max_length=4)


class PersonRequest(BaseModel):
    candidate_id: int


@router.post("/runs", status_code=status.HTTP_202_ACCEPTED)
def start(request: StartRequest):
    try:
        return recon.start_run(request.video_id, request.target_box_norm)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


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
