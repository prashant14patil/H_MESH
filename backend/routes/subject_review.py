"""Manual subject review endpoints."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, StrictBool

from backend.services import subject_review
from backend.services.video_mesh_batch import list_runs

router = APIRouter()


class CreateSession(BaseModel):
    video_id: str
    quality_run_id: str
    reference_frame: str
    bbox: list[float]
    replace_existing: StrictBool = False


class ReviewDecision(BaseModel):
    filename: str
    decision: str
    bbox: list[float] | None = None


class Selection(BaseModel):
    frames: list[str] = Field(default_factory=list)
    acknowledged: list[str] = Field(default_factory=list)


@router.get("/sessions")
def list_sessions(video_id: str, quality_run_id: str):
    """Find stored sessions so the UI can resume without a hand-typed ID."""
    try:
        return subject_review.find_sessions(video_id, quality_run_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/sessions", status_code=201)
def create_session(request: CreateSession):
    try:
        return subject_review.create_session(
            request.video_id, request.quality_run_id, request.reference_frame, request.bbox,
            replace_existing=request.replace_existing,
        )
    except subject_review.SessionConflict as exc:
        raise HTTPException(status_code=409, detail={"message": str(exc), "active_session_id": exc.session_id}) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/sessions/{session_id}")
def get_session(session_id: str):
    try:
        result = subject_review.get_session(session_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Review session not found")
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/sessions/{session_id}/resume")
def resume_session(session_id: str, video_id: str, quality_run_id: str):
    try:
        result = subject_review.resume_session(session_id, video_id, quality_run_id)
        result["mesh_runs"] = list_runs(video_id, session_id)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.put("/sessions/{session_id}/selection")
def save_selection(session_id: str, request: Selection):
    try:
        return subject_review.save_selection(session_id, request.frames, request.acknowledged)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/sessions/{session_id}/suggestions/{filename}")
def get_suggestion(session_id: str, filename: str):
    try:
        return subject_review.frame_suggestion(session_id, filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/sessions/{session_id}/decisions")
def create_decision(session_id: str, request: ReviewDecision):
    try:
        return subject_review.review_frame(session_id, request.filename, request.decision, request.bbox)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/sessions/{session_id}/decisions/{filename}")
def get_decision(session_id: str, filename: str):
    try:
        result = subject_review.get_decision(session_id, filename)
        if result is None:
            raise HTTPException(status_code=404, detail="Frame not reviewed")
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
