"""Opt-in, unvalidated capture quality report endpoints."""
from dataclasses import fields

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from backend.services.capture_quality import QualityConfig
from backend.services.capture_quality_job import get_job, get_report, start_quality

router = APIRouter()


class QualityRequest(BaseModel):
    video_id: str
    use_hog: bool = True
    thresholds: dict = Field(default_factory=dict)


@router.post("/runs", status_code=status.HTTP_202_ACCEPTED)
def create_run(request: QualityRequest):
    try:
        names = {field.name for field in fields(QualityConfig)}
        if set(request.thresholds) - names:
            raise ValueError("Unknown quality threshold")
        config = QualityConfig(**request.thresholds)
        return start_quality(request.video_id, config=config, use_hog=request.use_hog)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/runs/{run_id}")
def status_for_run(run_id: str):
    try:
        job = get_job(run_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if job is None:
        raise HTTPException(status_code=404, detail="Quality run not found")
    return job


@router.get("/runs/{run_id}/report")
def report_for_run(run_id: str):
    try:
        report = get_report(run_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if report is None:
        raise HTTPException(status_code=404, detail="Quality report not available")
    return report
