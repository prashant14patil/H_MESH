"""
backend/routes/capture.py
---------------------------
API routes for 360 video capture + frame extraction (Module 1 & 2).

These endpoints never touch the HMR2 inference engine — they only
handle uploading a 360 RGB video, running CV-based frame extraction
via backend.services.video_processor, and exposing the results so the
frontend can display a frame gallery with quality indicators. This is
intentionally kept independent of backend/routes/inference.py so a
later module can wire extracted frames into HMR2 without touching this
code.

Endpoints:
    POST /api/capture/upload
        Upload an MP4/MOV video, validate it, return a video_id.

    POST /api/capture/process/{video_id}
        Kick off frame extraction for a previously uploaded video.

    GET  /api/capture/status/{video_id}
        Poll the status of an upload/extraction job.

    GET  /api/capture/frames/{video_id}
        Return the full metadata.json (video info + per-frame scores).

    GET  /api/capture/frame/{video_id}/{frame_name}
        Serve a single extracted frame image (for gallery thumbnails).
"""

import logging
import uuid

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, StrictBool

from backend.services.capture_authorization import attest, require_authorization
from fastapi.responses import FileResponse, JSONResponse

from backend.services.video_processor import VideoProcessingError, get_capture_engine
from backend.utils.video_validators import VideoValidationError, validate_video

logger = logging.getLogger(__name__)

router = APIRouter()


class AuthorizationRequest(BaseModel):
    authorized: StrictBool
    consent: StrictBool


# ---------------------------------------------------------------------------
# POST /api/capture/upload
# ---------------------------------------------------------------------------
@router.post(
    "/upload",
    summary="Upload a 360 capture video",
    status_code=status.HTTP_201_CREATED,
)
async def upload_video(
    video: UploadFile = File(..., description="360 body-scan video (MP4 or MOV)"),
    authorized: bool = Form(False),
    consent: bool = Form(False),
) -> JSONResponse:
    """
    Accepts an uploaded 360 video, validates its container/size, and
    persists it to disk under a fresh video_id.

    Returns
    -------
    201 Created
        ``{ "video_id": "<uuid>", "status": "uploaded", "filename": "<name>" }``

    Errors
    ------
    400 – Video failed validation (wrong container, too large, etc.)
    """
    if not authorized or not consent:
        raise HTTPException(status_code=400, detail="Explicit authorization and consent confirmation are required")
    video_bytes = await video.read()

    try:
        validate_video(filename=video.filename, data=video_bytes)
    except VideoValidationError as exc:
        logger.warning("Video validation failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    video_id = str(uuid.uuid4())
    engine = get_capture_engine()
    engine.save_video(video_id=video_id, filename=video.filename, data=video_bytes)
    attest(video_id, authorized, consent)

    logger.info("Uploaded capture video %s (file: %s)", video_id, video.filename)

    return JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content={"video_id": video_id, "status": "uploaded", "filename": video.filename},
    )


@router.post("/{video_id}/authorization")
def confirm_authorization(video_id: str, request: AuthorizationRequest):
    try:
        return attest(video_id, request.authorized, request.consent)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{video_id}/authorization")
def authorization_status(video_id: str):
    try:
        return require_authorization(video_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# POST /api/capture/process/{video_id}
# ---------------------------------------------------------------------------
@router.post(
    "/process/{video_id}",
    summary="Start frame extraction for an uploaded video",
    status_code=status.HTTP_202_ACCEPTED,
)
async def process_video(video_id: str) -> JSONResponse:
    """
    Kicks off background frame extraction (FPS/resolution/duration
    parsing, evenly-spaced sampling, blur + duplicate detection,
    frame_NNN.jpg storage, metadata.json generation) for a video that
    was previously uploaded via POST /api/capture/upload.

    Returns
    -------
    202 Accepted
        ``{ "video_id": "<uuid>", "status": "processing" }``

    Errors
    ------
    404 – video_id not recognised (not uploaded, or server restarted).
    """
    engine = get_capture_engine()

    try:
        require_authorization(video_id)
        engine.process_video(video_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except VideoProcessingError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))

    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={"video_id": video_id, "status": "processing"},
    )


# ---------------------------------------------------------------------------
# GET /api/capture/status/{video_id}
# ---------------------------------------------------------------------------
@router.get(
    "/status/{video_id}",
    summary="Poll the status of an upload/extraction job",
)
async def get_capture_status(video_id: str) -> JSONResponse:
    """
    Returns the current status of a capture job: "uploaded",
    "processing", "completed", or "failed".

    Errors
    ------
    404 – video_id not recognised.
    """
    engine = get_capture_engine()
    job = engine.get_job(video_id)

    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video {video_id} not found.",
        )

    response_body = {"video_id": video_id, **job}
    # Never expose server-side file paths to the client.
    response_body.pop("video_path", None)
    response_body.pop("metadata_path", None)

    return JSONResponse(status_code=status.HTTP_200_OK, content=response_body)


# ---------------------------------------------------------------------------
# GET /api/capture/frames/{video_id}
# ---------------------------------------------------------------------------
@router.get(
    "/frames/{video_id}",
    summary="Get extracted frame metadata (video info, timestamps, quality scores)",
)
async def get_frames(video_id: str) -> JSONResponse:
    """
    Returns the full metadata.json written by frame extraction:
    video info (fps/resolution/duration), per-frame timestamps and
    blur/duplicate scores, and overall processing status.

    Each frame entry's ``filename`` can be fetched via
    GET /api/capture/frame/{video_id}/{filename}.

    Errors
    ------
    404 – video_id not recognised, or extraction not yet completed.
    """
    engine = get_capture_engine()
    metadata = engine.get_metadata(video_id)

    if metadata is None:
        job = engine.get_job(video_id)
        if job is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Video {video_id} not found.",
            )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Frames for video {video_id} not ready (status: {job.get('status')}).",
        )

    return JSONResponse(status_code=status.HTTP_200_OK, content=metadata)


# ---------------------------------------------------------------------------
# GET /api/capture/frame/{video_id}/{frame_name}
# ---------------------------------------------------------------------------
@router.get(
    "/frame/{video_id}/{frame_name}",
    summary="Download a single extracted frame image",
)
async def get_frame_image(video_id: str, frame_name: str):
    """
    Serves a single extracted frame (e.g. ``frame_001.jpg``) as a JPEG
    image, used by the frontend's frame gallery.

    Errors
    ------
    404 – video_id or frame_name not found on disk.
    """
    engine = get_capture_engine()
    frame_path = engine.get_frame_path(video_id, frame_name)

    if frame_path is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Frame '{frame_name}' not found for video {video_id}.",
        )

    return FileResponse(path=str(frame_path), media_type="image/jpeg", filename=frame_name)
