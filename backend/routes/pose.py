"""
backend/routes/pose.py
-------------------------
API routes for Module 3: Human detection, segmentation, and pose
estimation.

These endpoints operate on frames already extracted by Module 2
(backend/services/video_processor.py) and never touch the HMR2
inference engine. Kept independent so Modules 1/2 and the single-photo
HMR2 workflow remain completely unaffected.

Endpoints:
    POST /api/pose/process/{video_id}
        Start pose/segmentation processing for a video's extracted frames.

    GET  /api/pose/status/{video_id}
        Poll the status of a pose-processing job.

    GET  /api/pose/results/{video_id}
        Return the full processing_summary.json (per-frame landmarks,
        segmentation status, detection confidence, review flags).

    GET  /api/pose/frame/{video_id}/{frame_name}/{asset}
        Serve a single processed-frame asset: segmentation_mask.png,
        pose_visualization.jpg, or metadata.json.
"""

import logging

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse, JSONResponse

from backend.services.pose_processor import PoseProcessingError, get_pose_engine

logger = logging.getLogger(__name__)

router = APIRouter()

_ASSET_MEDIA_TYPES = {
    "segmentation_mask.png": "image/png",
    "pose_visualization.jpg": "image/jpeg",
    "metadata.json": "application/json",
}


# ---------------------------------------------------------------------------
# POST /api/pose/process/{video_id}
# ---------------------------------------------------------------------------
@router.post(
    "/process/{video_id}",
    summary="Start human detection, segmentation, and pose estimation for a video",
    status_code=status.HTTP_202_ACCEPTED,
)
async def process_pose(video_id: str) -> JSONResponse:
    """
    Kicks off background processing of every frame previously extracted
    for ``video_id`` (via POST /api/capture/process/{video_id}): pose
    landmark detection, human segmentation, and structured per-frame
    output generation under
    ``data/captures/{video_id}/processed_frames/``.

    Returns
    -------
    202 Accepted
        ``{ "video_id": "<uuid>", "status": "processing" }``

    Errors
    ------
    404 – No extracted frames found for this video_id (run Module 2 first).
    """
    engine = get_pose_engine()

    try:
        engine.process_video(video_id)
    except PoseProcessingError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))

    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={"video_id": video_id, "status": "processing"},
    )


# ---------------------------------------------------------------------------
# GET /api/pose/status/{video_id}
# ---------------------------------------------------------------------------
@router.get(
    "/status/{video_id}",
    summary="Poll the status of a pose-processing job",
)
async def get_pose_status(video_id: str) -> JSONResponse:
    """
    Returns the current status of a pose-processing job: "processing",
    "completed", or "failed".

    Errors
    ------
    404 – video_id not recognised (pose processing never started).
    """
    engine = get_pose_engine()
    job = engine.get_job(video_id)

    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No pose-processing job found for video {video_id}.",
        )

    response_body = {"video_id": video_id, **job}
    response_body.pop("summary_path", None)  # never expose server file paths

    return JSONResponse(status_code=status.HTTP_200_OK, content=response_body)


# ---------------------------------------------------------------------------
# GET /api/pose/results/{video_id}
# ---------------------------------------------------------------------------
@router.get(
    "/results/{video_id}",
    summary="Get the full pose/segmentation processing summary",
)
async def get_pose_results(video_id: str) -> JSONResponse:
    """
    Returns the full processing_summary.json: per-frame person
    detection, pose landmarks (with visibility), segmentation status,
    missing-region flags, and review requirements.

    Errors
    ------
    404 – video_id not recognised, or processing not yet completed.
    """
    engine = get_pose_engine()
    summary = engine.get_summary(video_id)

    if summary is None:
        job = engine.get_job(video_id)
        if job is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No pose-processing job found for video {video_id}.",
            )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Pose results for video {video_id} not ready (status: {job.get('status')}).",
        )

    return JSONResponse(status_code=status.HTTP_200_OK, content=summary)


# ---------------------------------------------------------------------------
# GET /api/pose/frame/{video_id}/{frame_name}/{asset}
# ---------------------------------------------------------------------------
@router.get(
    "/frame/{video_id}/{frame_name}/{asset}",
    summary="Download a processed-frame asset (mask, pose visualization, or metadata)",
)
async def get_pose_frame_asset(video_id: str, frame_name: str, asset: str):
    """
    Serves one of the per-frame outputs written under
    ``processed_frames/{frame_name}/``:

    - ``segmentation_mask.png``
    - ``pose_visualization.jpg``
    - ``metadata.json``

    Errors
    ------
    400 – unrecognised asset name.
    404 – video_id, frame_name, or asset not found on disk.
    """
    if asset not in _ASSET_MEDIA_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown asset '{asset}'. Expected one of: {', '.join(_ASSET_MEDIA_TYPES)}.",
        )

    engine = get_pose_engine()
    asset_path = engine.get_frame_result_path(video_id, frame_name, asset)

    if asset_path is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Asset '{asset}' not found for frame '{frame_name}' of video {video_id}.",
        )

    return FileResponse(
        path=str(asset_path),
        media_type=_ASSET_MEDIA_TYPES[asset],
        filename=f"{frame_name}_{asset}",
    )
