"""BCM integration layer: optional BCM data on camera assessments (no hardware, no model, no real BCM data needed)."""
import json
import uuid

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.routes.assessments import bcm_router, router
from backend.services import assessments as svc
from backend.services import body_reconstruction as br
from backend.services.bcm import demo, devices, records, schema, store

CM = {"chest": 100.0, "waist": 85.0, "abdomen": 90.0, "hip": 102.0, "upper_arm_left": 31.0, "upper_arm_right": 31.5,
      "forearm_left": 26.0, "forearm_right": 26.2, "thigh_left": 56.0, "thigh_right": 56.4, "calf_left": 37.0,
      "calf_right": 37.1}


def _make_run(root, created_at, scaled=True, waist=85.0):
    run_id = str(uuid.uuid4())
    d = root / run_id
    d.mkdir(parents=True)
    cm = {**CM, "waist": waist}
    result = {"mesh_obj": "final_body_mesh.obj", "measurements_valid": True, "silhouette_iou_fit": 0.9,
              "held_out_iou": 0.86, "keypoint_error_pct_height": 1.2, "frames_selected": 24,
              "measurements": {k: v / 175.0 for k, v in cm.items()}}
    if scaled:
        result["metric_scale"] = {"height_cm": 175.0, "relative_scale_uncertainty": 0.0057, "measurements_cm": cm,
                                  "mesh_obj": "final_body_mesh_metric.obj"}
    (d / "status.json").write_text(json.dumps({"run_id": run_id, "video_id": "v", "status": "completed",
                                               "created_at": created_at, "result": result}))
    (d / "measurements.json").write_text(json.dumps({"measurements": {k: {"valid": True} for k in cm}}))
    return run_id


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(br, "_ROOT", tmp_path)
    app = FastAPI()
    app.include_router(router, prefix="/api/assessments")
    app.include_router(bcm_router, prefix="/api/bcm")
    c = TestClient(app)
    c.root = tmp_path
    return c


MANUAL = {"profile": "generic_whole_body", "device_model": "entered by operator", "measured_at": "2026-11-19T09:30",
          "values": {"weight": {"value": 80, "unit": "kg"}, "body_fat_pct": {"value": 25, "unit": "%"},
                     "fat_mass": {"value": 20, "unit": "kg"}, "muscle_mass": {"value": 56.5, "unit": "kg"}}}


# 1. camera-only -----------------------------------------------------------------------------------------------
def test_camera_only_assessment_has_full_camera_block_and_null_bcm(client):
    rid = _make_run(client.root, 1_700_000_000)
    a = client.get(f"/api/assessments/{rid}").json()
    assert a["assessment_id"] == rid and a["bcm"] is None
    assert a["camera"]["available"] and a["camera"]["unit"] == "cm"
    assert a["camera"]["measurements_cm"]["waist"] == 85.0 and a["camera"]["quality"]["held_out_iou"] == 0.86
    r = client.get(f"/api/assessments/{rid}/bcm")
    assert r.status_code == 404 and r.json()["bcm"] is None
    listed = client.get("/api/assessments").json()["assessments"]
    assert listed[0]["bcm"] == {"available": False}


def test_unscaled_camera_assessment_still_listed_in_ratios(client):
    rid = _make_run(client.root, 1_700_000_000, scaled=False)
    cam = client.get(f"/api/assessments/{rid}").json()["camera"]
    assert cam["unit"] == "ratio_of_height" and cam["measurements_cm"] is None
    assert cam["measurements_ratio"]["waist"] == pytest.approx(85 / 175)


def test_incomplete_or_unknown_run_is_not_an_assessment(client):
    assert client.get(f"/api/assessments/{uuid.uuid4()}").status_code == 404
    assert client.get("/api/assessments/not-a-uuid").status_code == 400


# 2. manual entry ----------------------------------------------------------------------------------------------
def test_manual_entry_stores_units_originals_timestamp_and_source(client):
    rid = _make_run(client.root, 1_700_000_000)
    client.put(f"/api/assessments/{rid}/person", json={"person_id": "S01"})
    body = {**MANUAL, "values": {**MANUAL["values"], "weight": {"value": 176.4, "unit": "lb"},
                                 "bmr": {"value": 7531, "unit": "kJ"}}}
    rec = client.put(f"/api/assessments/{rid}/bcm", json=body).json()
    assert rec["assessment_id"] == rid and rec["person_id"] == "S01" and rec["revision"] == 1
    assert rec["source"] == {"type": "manual", "is_demo": False, "label": "Manually entered",
                             "device_model": "entered by operator"}
    w = rec["values"]["weight"]
    assert w["unit"] == "kg" and w["value"] == pytest.approx(80.01, abs=0.01) and w["original"] == {"value": 176.4, "unit": "lb"}
    assert rec["values"]["bmr"]["value"] == pytest.approx(1800, abs=1) and rec["values"]["bmr"]["original"]["unit"] == "kJ"
    assert rec["measured_at"] == "2026-11-19T09:30" and rec["recorded_at"]
    # The camera block is unchanged by adding BCM data.
    a = client.get(f"/api/assessments/{rid}").json()
    assert a["camera"]["measurements_cm"]["waist"] == 85.0 and a["bcm"]["record_id"] == rec["record_id"]


def test_manual_entry_validation_errors_are_per_field(client):
    rid = _make_run(client.root, 1_700_000_000)
    bad = {"profile": "weight_only", "values": {"weight": {"value": 900, "unit": "kg"},
                                                "body_fat_pct": {"value": 20, "unit": "%"},
                                                "bogus": {"value": 1, "unit": "kg"}}}
    r = client.put(f"/api/assessments/{rid}/bcm", json=bad)
    assert r.status_code == 422
    errors = r.json()["detail"]["errors"]
    assert "plausible range" in errors["weight"] and "not reported" in errors["body_fat_pct"] and errors["bogus"]
    r = client.put(f"/api/assessments/{rid}/bcm", json={"values": {"weight": {"value": 80, "unit": "stone"}}})
    assert "unit must be" in r.json()["detail"]["errors"]["weight"]
    r = client.put(f"/api/assessments/{rid}/bcm", json={"values": {}})
    assert "at least one" in r.json()["detail"]["errors"]["values"]
    r = client.put(f"/api/assessments/{rid}/bcm", json={"values": {"weight": 80}, "measured_at": "2999-01-01T00:00"})
    assert r.json()["detail"]["errors"]["measured_at"] == "is in the future"
    assert not (client.root / rid / store.RECORD_FILE).exists()


def test_consistency_problems_are_warnings_not_corrections(client):
    rid = _make_run(client.root, 1_700_000_000)
    body = {"values": {"weight": 80, "body_fat_pct": 25, "fat_mass": 30}}
    rec = client.put(f"/api/assessments/{rid}/bcm", json=body).json()
    assert rec["values"]["fat_mass"]["value"] == 30
    assert [w["code"] for w in rec["warnings"]] == ["fat_mass_mismatch"]


# 3. demo data -------------------------------------------------------------------------------------------------
def test_demo_data_is_always_flagged(client):
    rid = _make_run(client.root, 1_700_000_000)
    rec = client.post(f"/api/assessments/{rid}/bcm/demo", json={"sample_set": 2}).json()
    assert rec["source"]["type"] == "demo" and rec["source"]["is_demo"] is True
    assert "DEMO DATA" in rec["source"]["label"] and "not a Tanita measurement" in rec["source"]["label"]
    assert "DEMO DATA" in rec["note"] and rec["source"]["demo_set"] == 2
    assert rec["values"]["weight"]["value"] == 77.0 and "segment_fat_pct_trunk" in rec["values"]
    assert rec["warnings"] == []          # the sample sets are internally consistent
    listed = client.get("/api/assessments").json()["assessments"][0]["bcm"]
    assert listed["is_demo"] is True
    assert client.post(f"/api/assessments/{rid}/bcm/demo", json={"sample_set": 9}).status_code == 422


def test_demo_payload_respects_profile():
    p = demo.demo_payload(1, "generic_whole_body")
    assert set(p["values"]) <= set(schema.PROFILES["generic_whole_body"].parameters)
    assert set(demo.demo_payload(1, "weight_only")["values"]) == {"weight"}


# 4. editing ---------------------------------------------------------------------------------------------------
def test_edit_creates_revision_and_keeps_previous_in_history(client):
    rid = _make_run(client.root, 1_700_000_000)
    first = client.put(f"/api/assessments/{rid}/bcm", json=MANUAL).json()
    edited = {**MANUAL, "values": {**MANUAL["values"], "weight": {"value": 81, "unit": "kg"},
                                   "fat_mass": {"value": 20.3, "unit": "kg"}}}
    second = client.put(f"/api/assessments/{rid}/bcm", json=edited).json()
    assert second["record_id"] == first["record_id"] and second["revision"] == 2
    assert second["recorded_at"] == first["recorded_at"] and second["values"]["weight"]["value"] == 81
    events = client.get(f"/api/assessments/{rid}/bcm/history").json()["events"]
    assert [e["record"]["revision"] for e in events] == [1, 2]
    assert events[0]["record"]["values"]["weight"]["value"] == 80
    # Demo replaced by a manual record: the record is no longer demo.
    client.post(f"/api/assessments/{rid}/bcm/demo", json={})
    rec = client.put(f"/api/assessments/{rid}/bcm", json=MANUAL).json()
    assert rec["source"]["is_demo"] is False


def test_remove_keeps_audit_trail(client):
    rid = _make_run(client.root, 1_700_000_000)
    client.put(f"/api/assessments/{rid}/bcm", json=MANUAL)
    assert client.delete(f"/api/assessments/{rid}/bcm").status_code == 204
    assert client.get(f"/api/assessments/{rid}").json()["bcm"] is None
    assert client.delete(f"/api/assessments/{rid}/bcm").status_code == 404
    events = client.get(f"/api/assessments/{rid}/bcm/history").json()["events"]
    assert [e["event"] for e in events] == ["saved", "removed"]


# 5. partially missing -----------------------------------------------------------------------------------------
def test_partial_record_lists_missing_parameters(client):
    rid = _make_run(client.root, 1_700_000_000)
    body = {"profile": "generic_segmental",
            "values": {"weight": {"value": 70, "unit": "kg"}, "body_fat_pct": {"value": None, "unit": "%"},
                       "fat_mass": {"value": "", "unit": "kg"}, "segment_fat_pct_trunk": {"value": "21,5", "unit": "%"}}}
    rec = client.put(f"/api/assessments/{rid}/bcm", json=body).json()
    assert set(rec["values"]) == {"weight", "segment_fat_pct_trunk"}
    assert rec["values"]["segment_fat_pct_trunk"]["value"] == 21.5
    assert "body_fat_pct" in rec["missing"] and "fat_mass" in rec["missing"] and "weight" not in rec["missing"]
    assert len(rec["missing"]) == len(schema.PROFILES["generic_segmental"].parameters) - 2


# 6 + 7. multiple assessments and comparison --------------------------------------------------------------------
def test_person_history_and_comparison_with_and_without_bcm(client):
    r1 = _make_run(client.root, 1_700_000_000, waist=88.0)
    r2 = _make_run(client.root, 1_700_100_000, waist=86.0)
    r3 = _make_run(client.root, 1_700_200_000, waist=85.0)
    other = _make_run(client.root, 1_700_300_000)
    for rid in (r1, r2, r3):
        client.put(f"/api/assessments/{rid}/person", json={"person_id": "S01"})
    client.put(f"/api/assessments/{other}/person", json={"person_id": "S02"})

    h = client.get("/api/assessments/persons/S01/history").json()
    assert [a["assessment_id"] for a in h["assessments"]] == [r1, r2, r3]
    assert h["bcm_series"] == [] and h["bcm_trend_available"] is False   # no BCM -> no trend
    assert len(h["camera_series"]) == 3

    cmp = client.get(f"/api/assessments/compare?before={r1}&after={r2}").json()
    waist = next(r for r in cmp["camera"]["rows"] if r["site"] == "waist")
    assert cmp["camera"]["unit"] == "cm" and waist["delta"] == pytest.approx(-2.0) and cmp["same_person"]
    assert cmp["bcm"]["available"] is False and cmp["bcm"]["rows"] == [] and "Neither" in cmp["bcm"]["message"]

    client.put(f"/api/assessments/{r2}/bcm", json=MANUAL)
    h = client.get("/api/assessments/persons/S01/history").json()
    assert len(h["bcm_series"]) == 1 and h["bcm_trend_available"] is False   # one record is not a trend
    cmp = client.get(f"/api/assessments/compare?before={r1}&after={r2}").json()
    assert cmp["bcm"]["available"] is False and all(r["delta"] is None for r in cmp["bcm"]["rows"])
    assert "Only one" in cmp["bcm"]["message"]

    client.post(f"/api/assessments/{r3}/bcm/demo", json={"sample_set": 1, "profile": "generic_whole_body"})
    h = client.get("/api/assessments/persons/S01/history").json()
    assert h["bcm_trend_available"] is True and [s["is_demo"] for s in h["bcm_series"]] == [False, True]
    cmp = client.get(f"/api/assessments/compare?before={r2}&after={r3}").json()
    weight = next(r for r in cmp["bcm"]["rows"] if r["key"] == "weight")
    assert cmp["bcm"]["available"] and cmp["bcm"]["demo_involved"] and weight["delta"] == pytest.approx(-2.0)

    persons = {p["person_id"]: p for p in client.get("/api/assessments/persons").json()["persons"]}
    assert persons["S01"]["assessments"] == 3 and persons["S01"]["with_bcm"] == 2 and persons["S02"]["with_bcm"] == 0
    only = client.get("/api/assessments?person_id=S02").json()["assessments"]
    assert [a["assessment_id"] for a in only] == [other]


def test_mixed_scaled_and_unscaled_compare_in_ratios(client):
    r1 = _make_run(client.root, 1_700_000_000, scaled=False)
    r2 = _make_run(client.root, 1_700_100_000)
    cmp = client.get(f"/api/assessments/compare?before={r1}&after={r2}").json()
    assert cmp["camera"]["unit"] == "ratio_of_height"


def test_person_change_keeps_conditions_and_syncs_bcm(client):
    rid = _make_run(client.root, 1_700_000_000)
    (client.root / rid / "capture_conditions.json").write_text(json.dumps(
        {"subject_id": "S01", "clothing": "tight", "posture": "a_pose_standard", "distance_m": 2.5}))
    client.put(f"/api/assessments/{rid}/bcm", json=MANUAL)
    a = client.put(f"/api/assessments/{rid}/person", json={"person_id": "S07"}).json()
    assert a["person_id"] == "S07" and a["bcm"]["person_id"] == "S07"
    cond = json.loads((client.root / rid / "capture_conditions.json").read_text())
    assert cond["clothing"] == "tight" and cond["distance_m"] == 2.5
    assert client.put(f"/api/assessments/{rid}/person", json={"person_id": "Jane <Doe>"}).status_code == 400


# import / device / schema --------------------------------------------------------------------------------------
def test_import_preserves_original_document(client):
    rid = _make_run(client.root, 1_700_000_000)
    doc = {"profile": "generic_whole_body", "measured_at": "2026-11-18T08:00", "values": {"weight": {"value": 72.4, "unit": "kg"}},
           "extra_field_from_exporter": "kept"}
    rec = client.post(f"/api/assessments/{rid}/bcm/import", json={"filename": "export.json", "payload": doc}).json()
    assert rec["source"]["type"] == "import" and rec["source"]["original_payload"] == doc
    assert rec["source"]["imported_filename"] == "export.json"
    r = client.post(f"/api/assessments/{rid}/bcm/import", json={"format": "tanita-csv", "payload": doc})
    assert r.status_code == 422


def test_device_integration_is_reported_as_not_connected(client):
    s = client.get("/api/bcm/devices").json()
    assert s["state"] == "not_connected" and s["planned"] is True and s["adapters"] == []
    with pytest.raises(TypeError):
        devices.BCMDeviceAdapter()   # abstract: no implementation exists


def test_schema_describes_profiles_and_units(client):
    s = client.get("/api/bcm/schema").json()
    keys = {p["key"]: p for p in s["parameters"]}
    assert keys["weight"]["units"] == ["kg", "lb"] and keys["bmr"]["unit"] == "kcal"
    assert all(not p["verified_device_spec"] for p in s["profiles"])   # no invented device specifications
    with pytest.raises(ValueError):
        schema.register_profile(schema.Profile("x", "x", "x", ("not_a_parameter",)))


def test_build_record_rejects_unknown_profile():
    with pytest.raises(records.BCMValidationError) as exc:
        records.build_record({"profile": "nope", "values": {"weight": 70}}, assessment_id="a", person_id=None,
                             source_type="manual")
    assert "profile" in exc.value.errors


def test_assessment_service_never_writes_reconstruction_files(client):
    rid = _make_run(client.root, 1_700_000_000)
    before = (client.root / rid / "status.json").read_text()
    client.put(f"/api/assessments/{rid}/bcm", json=MANUAL)
    client.post(f"/api/assessments/{rid}/bcm/demo", json={})
    client.delete(f"/api/assessments/{rid}/bcm")
    assert (client.root / rid / "status.json").read_text() == before
    assert svc.get_assessment(rid)["camera"]["measurements_cm"] == CM
