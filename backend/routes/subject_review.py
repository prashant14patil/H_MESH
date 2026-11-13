"""Manual subject review endpoints."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.services import subject_review

router = APIRouter()


class CreateSession(BaseModel):
    video_id: str
    quality_run_id: str
    reference_frame: str
    bbox: list[float]


class ReviewDecision(BaseModel):
    filename: str
    decision: str
    bbox: list[float] | None = None


@router.post("/sessions", status_code=201)
def create_session(request: CreateSession):
    try:
        return subject_review.create_session(
            request.video_id, request.quality_run_id, request.reference_frame, request.bbox
        )
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
        return subject_review.resume_session(session_id, video_id, quality_run_id)
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
