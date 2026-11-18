"""SHAPY as an alternative / comparison measurement model for completed reconstruction runs."""
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backend.services import body_reconstruction as recon
from backend.services import shapy_shape
from backend.services import video_processor

router = APIRouter()


def _completed_run(run_id: str) -> dict:
    try:
        run = recon.get_run(run_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if run.get("status") != "completed":
        raise HTTPException(status_code=400, detail="SHAPY needs a completed reconstruction")
    return run


def _height(run: dict):
    return ((run.get("result") or {}).get("metric_scale") or {}).get("height_cm")


def _load(run_id: str):
    path = recon._run_dir(run_id) / shapy_shape.RESULT_FILE
    if not path.is_file():
        return None
    import json
    return json.loads(path.read_text(encoding="utf-8"))


@router.get("/models")
def models():
    """Measurement models this backend can offer."""
    return {"models": [
        {"id": "pipeline", "label": "Pipeline (HMR2 + multi-view fit)", "available": True},
        {"id": "shapy", "label": "SHAPY (shape-focused regressor)", "available": shapy_shape.available()},
    ]}


@router.get("/runs/{run_id}/shapy")
def get_shapy(run_id: str):
    run = _completed_run(run_id)
    record = _load(run_id)
    if record is None or record.get("version", 1) < shapy_shape.RESULT_VERSION:
        return {"computed": False, "available": shapy_shape.available()}
    return {"computed": True, "available": shapy_shape.available(), **shapy_shape.summarize(record, _height(run))}


@router.post("/runs/{run_id}/shapy")
def compute_shapy(run_id: str, recompute: bool = False):
    """Compute (once) SHAPY measurements for a completed run; cm follow the run's current height."""
    run = _completed_run(run_id)
    record = None if recompute else _load(run_id)
    if record is not None and record.get("version", 1) < shapy_shape.RESULT_VERSION:
        record = None  # stored with an older method: recompute
    if record is None:
        if not shapy_shape.available():
            raise HTTPException(status_code=503, detail="SHAPY model files are not installed on this backend")
        sources = [p for p in video_processor._OUTPUT_ROOT.joinpath(run["video_id"]).glob("original.*") if p.is_file()]
        if len(sources) != 1:
            raise HTTPException(status_code=400, detail="The run's source video is missing or ambiguous")
        try:
            record = shapy_shape.compute_for_run(recon._run_dir(run_id), sources[0])
        except (ValueError, FileNotFoundError, KeyError) as exc:
            raise HTTPException(status_code=400, detail=f"SHAPY could not process this run: {exc}") from exc
    return {"computed": True, "available": True, **shapy_shape.summarize(record, _height(run))}


@router.get("/runs/{run_id}/shapy/mesh")
def shapy_mesh(run_id: str):
    try:
        path = recon._run_dir(run_id) / shapy_shape.MESH_FILE
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not path.is_file():
        raise HTTPException(status_code=404, detail="SHAPY mesh not computed for this run")
    return FileResponse(path, media_type="text/plain", filename="shapy_body_mesh.obj")
