"""
backend/routes/inference.py
----------------------------
API routes for HMR2 inference — LOCAL GPU VERSION.

Unlike the original Colab-based POC, these endpoints never make an
outbound HTTP call. All inference happens in-process via
`backend.services.inference_engine.get_engine()`, running on the
local machine's GPU (or CPU fallback).

Endpoints:
    POST  /api/inference/submit
        Accept a full-body RGB image upload, validate it, submit it to
        the local inference engine, and return a job identifier.

    GET   /api/inference/status/{job_id}
        Poll the status of a previously submitted inference job.

    GET   /api/inference/mesh/{job_id}
        Return the OBJ mesh generated for a completed job.
"""

import logging
import os
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse

from backend.services.inference_engine import EngineNotReadyError, get_engine
from backend.utils.validators import ValidationError, validate_image

logger = logging.getLogger(__name__)

router = APIRouter()

# Local directory where generated meshes are written by the engine.
_MESH_DIR = Path(os.getenv("MESH_OUTPUT_DIR", "outputs/meshes"))
_MESH_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# POST /api/inference/submit
# ---------------------------------------------------------------------------
@router.post(
    "/submit",
    summary="Submit an image for 3D mesh reconstruction",
    status_code=status.HTTP_202_ACCEPTED,
)
async def submit_inference(
    image: UploadFile = File(..., description="Full-body RGB photograph (JPEG or PNG)"),
) -> JSONResponse:
    """
    Accepts an uploaded image, validates it, and submits it to the local
    HMR2 inference engine for background processing on the GPU.

    Returns
    -------
    202 Accepted
        ``{ "job_id": "<uuid>", "status": "queued" }``

    Errors
    ------
    400  – Image failed validation (wrong type, too large, etc.)
    503  – The inference engine failed to load (missing checkpoint, etc.)
    """
    image_bytes = await image.read()

    try:
        validate_image(filename=image.filename, data=image_bytes)
    except ValidationError as exc:
        logger.warning("Image validation failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    job_id = str(uuid.uuid4())
    logger.info("Submitting local inference job %s (file: %s)", job_id, image.filename)

    try:
        engine = get_engine()
        engine.submit_job(job_id=job_id, image_bytes=image_bytes)
    except EngineNotReadyError as exc:
        logger.error("Inference engine not ready for job %s: %s", job_id, exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Inference engine not ready: {exc}",
        )

    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={"job_id": job_id, "status": "queued"},
    )


# ---------------------------------------------------------------------------
# GET /api/inference/status/{job_id}
# ---------------------------------------------------------------------------
@router.get(
    "/status/{job_id}",
    summary="Poll the status of an inference job",
)
async def get_inference_status(job_id: str) -> JSONResponse:
    """
    Returns the current status of a job tracked by the local engine.

    Returns
    -------
    200 OK – processing::

        { "job_id": "<uuid>", "status": "processing" }

    200 OK – completed::

        {
            "job_id": "<uuid>",
            "status": "completed",
            "mesh_url": "/api/inference/mesh/<uuid>",
            "metadata": { "vertices": 6890, "faces": 13776, "body_model": "SMPL" }
        }

    Errors
    ------
    404  – Job ID not recognised.
    """
    logger.info("Status request for job %s", job_id)

    engine = get_engine()
    info = engine.get_job(job_id)

    if info is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found.",
        )

    response_body: dict = {"job_id": job_id, **info}

    if info.get("status") == "completed":
        response_body["mesh_url"] = f"/api/inference/mesh/{job_id}"
        response_body.pop("mesh_path", None)  # never expose server file paths

    return JSONResponse(status_code=status.HTTP_200_OK, content=response_body)


# ---------------------------------------------------------------------------
# GET /api/inference/mesh/{job_id}
# ---------------------------------------------------------------------------
@router.get(
    "/mesh/{job_id}",
    summary="Download the reconstructed OBJ mesh",
)
async def get_mesh(job_id: str):
    """
    Returns the reconstructed OBJ mesh file for a completed job.

    The mesh was written locally by the inference engine to
    ``outputs/meshes/{job_id}.obj`` — no remote fetch is needed.

    Returns
    -------
    200 OK – OBJ file as ``application/octet-stream`` attachment.

    Errors
    ------
    404  – Job not found, mesh file missing, or not yet complete.
    """
    local_mesh = _MESH_DIR / f"{job_id}.obj"

    if local_mesh.exists():
        logger.info("Serving mesh for job %s", job_id)
        return FileResponse(
            path=str(local_mesh),
            media_type="application/octet-stream",
            filename=f"{job_id}.obj",
        )

    engine = get_engine()
    info = engine.get_job(job_id)

    if info is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found.",
        )

    if info.get("status") != "completed":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Mesh for job {job_id} not ready (status: {info.get('status')}).",
        )

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Mesh file for job {job_id} is missing on disk.",
    )
