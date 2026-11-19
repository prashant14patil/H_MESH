"""
Assessments (camera reconstruction + optional BCM data) and the BCM integration layer.

    GET    /api/assessments                         list (optionally ?person_id=)
    GET    /api/assessments/persons                 person codes with assessment / BCM counts
    GET    /api/assessments/compare?before=&after=  field-by-field change between two assessments
    GET    /api/assessments/persons/{person_id}/history
    GET    /api/assessments/{id}                    unified assessment (camera + bcm|null)
    PUT    /api/assessments/{id}/person             assign to a person code
    GET    /api/assessments/{id}/bcm                current BCM record (404 when none)
    PUT    /api/assessments/{id}/bcm                create / edit a manual record (new revision)
    DELETE /api/assessments/{id}/bcm                remove (kept in the history log)
    POST   /api/assessments/{id}/bcm/demo           attach DEMO DATA
    POST   /api/assessments/{id}/bcm/import         import a bcm-json-v1 document
    GET    /api/assessments/{id}/bcm/history        audit log of revisions
    GET    /api/bcm/schema                          parameters, units and profiles
    GET    /api/bcm/devices                         device-integration status (not connected)
"""
from fastapi import APIRouter, HTTPException, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from backend.services import assessments as svc
from backend.services.bcm import devices, records, schema

router = APIRouter()
bcm_router = APIRouter()


class PersonRequest(BaseModel):
    person_id: str = Field(min_length=1, max_length=40)


class BCMRequest(BaseModel):
    profile: str | None = None
    device_model: str | None = Field(default=None, max_length=60)
    measured_at: str | None = None
    # {parameter_key: {"value": number|null, "unit": str}}; blank / null values mean "not measured".
    values: dict = Field(default_factory=dict)
    note: str = Field(default="", max_length=500)


class DemoRequest(BaseModel):
    sample_set: int = Field(default=1, ge=1, le=3)
    profile: str = "generic_segmental"


class ImportRequest(BaseModel):
    format: str = "bcm-json-v1"
    filename: str | None = None
    payload: dict


def _call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except svc.AssessmentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except records.BCMValidationError as exc:
        raise HTTPException(status_code=422, detail={"message": "Invalid BCM data", "errors": exc.errors}) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("")
def list_assessments(person_id: str | None = None, limit: int = 100):
    return {"assessments": svc.list_assessments(person_id, max(1, min(limit, 200)))}


@router.get("/persons")
def list_persons():
    return {"persons": svc.list_persons()}


@router.get("/compare")
def compare(before: str, after: str):
    return _call(svc.compare, before, after)


@router.get("/persons/{person_id}/history")
def person_history(person_id: str):
    return svc.person_history(person_id)


@router.get("/{assessment_id}")
def get_assessment(assessment_id: str):
    return _call(svc.get_assessment, assessment_id)


@router.put("/{assessment_id}/person")
def set_person(assessment_id: str, request: PersonRequest):
    return _call(svc.set_person, assessment_id, request.person_id)


@router.get("/{assessment_id}/bcm")
def get_bcm(assessment_id: str):
    rec = _call(svc.get_assessment, assessment_id)["bcm"]
    if rec is None:
        # Not an error for the app: camera-only assessments are normal. The body says so explicitly.
        return JSONResponse(status_code=404, content={"detail": "No BCM data for this assessment", "bcm": None})
    return rec


@router.put("/{assessment_id}/bcm")
def save_bcm(assessment_id: str, request: BCMRequest):
    return _call(svc.save_bcm, assessment_id, request.model_dump(), "manual")


@router.delete("/{assessment_id}/bcm", status_code=status.HTTP_204_NO_CONTENT)
def remove_bcm(assessment_id: str):
    if _call(svc.remove_bcm, assessment_id) is None:
        raise HTTPException(status_code=404, detail="No BCM data for this assessment")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/{assessment_id}/bcm/demo")
def save_demo(assessment_id: str, request: DemoRequest):
    return _call(svc.save_demo_bcm, assessment_id, request.sample_set, request.profile)


@router.post("/{assessment_id}/bcm/import")
def import_bcm(assessment_id: str, request: ImportRequest):
    return _call(svc.import_bcm, assessment_id, request.format, request.payload, request.filename)


@router.get("/{assessment_id}/bcm/history")
def bcm_history(assessment_id: str):
    return {"events": _call(svc.bcm_history, assessment_id)}


@bcm_router.get("/schema")
def bcm_schema():
    return schema.describe()


@bcm_router.get("/devices")
def bcm_devices():
    return devices.integration_status()
