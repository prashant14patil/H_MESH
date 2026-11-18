"""Metric scale from an operator-supplied height (no GPU/model needed)."""
import json

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.routes.reconstruction import router
from backend.services import body_measurements as bm
from backend.services import body_reconstruction as br
from backend.services import smpl_multiview as smv

RUN_ID = "00000000-0000-4000-8000-0000000000aa"


def _measurements(mesh_height=1.75):
    return {"units": "mesh units", "mesh_height": mesh_height, "valid_count": 2, "total": 3,
            "measurements": {
                "waist": {"valid": True, "tape_circumference": 0.80, "contour_perimeter": 0.81, "definition": "w"},
                "hip": {"valid": True, "tape_circumference": 1.00, "contour_perimeter": 1.02, "definition": "h"},
                "calf_left": {"valid": False, "reason": "no valid closed cross-section"}}}


def test_scale_to_height_is_uniform_and_reports_uncertainty():
    out = bm.scale_to_height(_measurements(1.75), 175.0)
    assert out["cm_per_mesh_unit"] == pytest.approx(100.0)
    assert out["measurements"]["waist"]["tape_circumference_cm"] == pytest.approx(80.0)
    assert out["measurements"]["hip"]["contour_perimeter_cm"] == pytest.approx(102.0)
    assert out["measurements"]["calf_left"]["valid"] is False
    assert out["valid_count"] == 2 and out["total"] == 3
    # +-1 cm of stature on 175 cm moves a 100 cm girth by ~0.57 cm.
    assert out["measurements"]["hip"]["height_uncertainty_contribution_cm"] == pytest.approx(0.57, abs=0.01)
    # Ratio to height is preserved by the conversion.
    half = bm.scale_to_height(_measurements(1.75), 87.5 * 2)
    assert half["measurements"]["waist"]["tape_circumference_cm"] / 175.0 == pytest.approx(0.80 / 1.75, abs=1e-3)


@pytest.mark.parametrize("bad", [None, "170", True, float("nan"), float("inf"), 50, 300])
def test_height_validation_rejects_implausible_values(bad):
    with pytest.raises(ValueError):
        bm.validate_height_cm(bad)


def test_scale_requires_valid_mesh_height():
    with pytest.raises(ValueError):
        bm.scale_to_height(_measurements(0.0), 170)


@pytest.fixture()
def finished_run(tmp_path, monkeypatch):
    monkeypatch.setattr(br, "_ROOT", tmp_path)
    monkeypatch.setattr(br, "_RUNS", {})
    root = tmp_path / RUN_ID
    root.mkdir()
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1.75, 0], [0, 0, 1]], dtype=float)
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    smv.write_obj(root / "final_body_mesh.obj", verts, faces)
    (root / "measurements.json").write_text(json.dumps(_measurements(1.75)))
    (root / "status.json").write_text(json.dumps(
        {"run_id": RUN_ID, "status": "completed", "stage": "done", "result": {"mesh_obj": "final_body_mesh.obj"}}))
    return root


@pytest.fixture()
def client():
    app = FastAPI()
    app.include_router(router, prefix="/api/reconstruction")
    return TestClient(app)


def test_api_scales_completed_run_from_disk(client, finished_run):
    r = client.post(f"/api/reconstruction/runs/{RUN_ID}/scale", json={"height_cm": 182})
    assert r.status_code == 200, r.text
    ms = r.json()["result"]["metric_scale"]
    assert ms["height_cm"] == 182 and ms["measurements_cm"]["hip"] == pytest.approx(104.0, abs=0.1)
    assert r.json()["result"]["scale_status"] == "metric_from_supplied_height"
    # Metric OBJ is in metres and its height equals the supplied stature.
    v, f = smv.read_obj(finished_run / "final_body_mesh_metric.obj")
    assert v[:, 1].max() - v[:, 1].min() == pytest.approx(1.82, abs=1e-4) and len(f) == 4
    saved = json.loads((finished_run / "measurements_cm.json").read_text())
    assert saved["measurements"]["waist"]["tape_circumference_cm"] == pytest.approx(83.2, abs=0.1)
    assert json.loads((finished_run / "status.json").read_text())["result"]["metric_scale"]["height_cm"] == 182
    # The relative originals are untouched and the new files are downloadable.
    assert json.loads((finished_run / "measurements.json").read_text())["mesh_height"] == 1.75
    assert client.get(f"/api/reconstruction/runs/{RUN_ID}/files/measurements_cm.json").status_code == 200
    assert client.get(f"/api/reconstruction/runs/{RUN_ID}/files/final_body_mesh_metric.obj").status_code == 200


def test_api_scale_can_be_reapplied_with_corrected_height(client, finished_run):
    client.post(f"/api/reconstruction/runs/{RUN_ID}/scale", json={"height_cm": 160})
    r = client.post(f"/api/reconstruction/runs/{RUN_ID}/scale", json={"height_cm": 175})
    assert r.json()["result"]["metric_scale"]["measurements_cm"]["waist"] == pytest.approx(80.0)


def test_api_scale_rejects_bad_input_and_unfinished_runs(client, finished_run):
    url = f"/api/reconstruction/runs/{RUN_ID}/scale"
    assert client.post(url, json={"height_cm": 20}).status_code == 400
    assert client.post("/api/reconstruction/runs/not-a-uuid/scale", json={"height_cm": 170}).status_code == 400
    assert client.post("/api/reconstruction/runs/00000000-0000-4000-8000-000000000000/scale",
                       json={"height_cm": 170}).status_code == 404
    status = json.loads((finished_run / "status.json").read_text())
    status["status"] = "failed"
    (finished_run / "status.json").write_text(json.dumps(status))
    r = client.post(url, json={"height_cm": 170})
    assert r.status_code == 400 and "completed" in r.json()["detail"]
    assert not (finished_run / "measurements_cm.json").exists()


def test_start_rejects_implausible_height_before_anything_runs(client, monkeypatch):
    called = []
    monkeypatch.setattr(br, "require_authorization", lambda v: called.append(v))
    r = client.post("/api/reconstruction/runs",
                    json={"video_id": "00000000-0000-4000-8000-000000000000", "height_cm": 400})
    assert r.status_code == 400 and "height_cm" in r.json()["detail"] and not called


# ------------------------------------------------- tape-measure validation
def test_compare_to_tape_signs_and_summary():
    scaled = bm.scale_to_height(_measurements(1.75), 175.0)  # waist 80, hip 100, calf invalid
    out = bm.compare_to_tape(scaled, {"waist": 76.0, "hip": 104.0, "calf_left": 36.0})
    assert out["sites"]["waist"]["error_cm"] == pytest.approx(4.0)      # model larger
    assert out["sites"]["hip"]["error_pct"] == pytest.approx(-3.85, abs=0.01)
    assert out["sites"]["calf_left"]["compared"] is False
    s = out["summary"]
    assert s["sites_compared"] == 2 and s["mean_error_cm"] == pytest.approx(0.0) and s["mean_abs_error_cm"] == pytest.approx(4.0)


@pytest.mark.parametrize("bad", [{}, {"neck": 38}, {"waist": 5}, {"waist": "80"}, {"waist": float("nan")}, {"waist": None}])
def test_validate_tape_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        bm.validate_tape(bad)


def test_aggregate_dedupes_recordings_and_splits_by_clothing():
    def rec(video, t, clothing, waist_err):
        return {"video_id": video, "recorded_at": t, "clothing": clothing,
                "sites": {"waist": {"compared": True, "error_cm": waist_err, "error_pct": waist_err},
                          "hip": {"compared": False}}}
    agg = bm.aggregate_tape([rec("a", 1, "loose", 9.0), rec("a", 2, "tight", 1.0),  # same subject: keep t=2
                             rec("b", 1, "tight", 3.0), rec("c", 1, "loose", 6.0)])
    assert agg["subjects"] == 3 and agg["clothing_counts"]["tight"] == 2
    w = agg["per_site"]["waist"]
    assert w["n"] == 3 and w["mean_error_cm"] == pytest.approx(10 / 3, abs=0.01)
    assert agg["per_site_by_clothing"]["waist"]["tight"]["mean_error_cm"] == pytest.approx(2.0)
    assert agg["per_site_by_clothing"]["waist"]["loose"]["n"] == 1
    assert "hip" not in agg["per_site"]


def test_api_tape_requires_scale_then_compares_and_follows_height_change(client, finished_run):
    url = f"/api/reconstruction/runs/{RUN_ID}/tape"
    body = {"tape_cm": {"waist": 78.0, "hip": 100.0}, "clothing": "tight", "note": "measured twice, averaged"}
    r = client.post(url, json=body)
    assert r.status_code == 400 and "height" in r.json()["detail"].lower()
    client.post(f"/api/reconstruction/runs/{RUN_ID}/scale", json={"height_cm": 175})
    r = client.post(url, json=body)
    assert r.status_code == 200, r.text
    tv = r.json()["result"]["tape_validation"]
    assert tv["sites"]["waist"]["error_cm"] == pytest.approx(2.0) and tv["clothing"] == "tight"
    saved = json.loads((finished_run / "tape_validation.json").read_text())
    assert saved["note"] == "measured twice, averaged" and "waist" in saved["protocol"]
    # Correcting the height re-scores the stored tape values instead of leaving a stale comparison.
    r = client.post(f"/api/reconstruction/runs/{RUN_ID}/scale", json={"height_cm": 170})
    tv = r.json()["result"]["tape_validation"]
    assert tv["sites"]["waist"]["model_cm"] == pytest.approx(77.7, abs=0.1)
    assert json.loads((finished_run / "tape_validation.json").read_text())["height_cm"] == 170
    assert client.get(f"/api/reconstruction/runs/{RUN_ID}/files/tape_validation.json").status_code == 200


def test_api_tape_rejects_bad_payloads(client, finished_run):
    client.post(f"/api/reconstruction/runs/{RUN_ID}/scale", json={"height_cm": 175})
    url = f"/api/reconstruction/runs/{RUN_ID}/tape"
    assert client.post(url, json={"tape_cm": {"neck": 40}}).status_code == 400
    assert client.post(url, json={"tape_cm": {"waist": 80}, "clothing": "pyjamas"}).status_code == 400
    assert client.post(url, json={"tape_cm": {"waist": 900}}).status_code == 400
    assert client.post("/api/reconstruction/runs/00000000-0000-4000-8000-000000000000/tape",
                       json={"tape_cm": {"waist": 80}}).status_code == 404
    assert not (finished_run / "tape_validation.json").exists()


def test_report_script_runs_on_recorded_tape(client, finished_run, tmp_path):
    import subprocess, sys
    client.post(f"/api/reconstruction/runs/{RUN_ID}/scale", json={"height_cm": 175})
    client.post(f"/api/reconstruction/runs/{RUN_ID}/tape", json={"tape_cm": {"waist": 78.0}, "clothing": "minimal"})
    out = subprocess.run([sys.executable, "scripts/tape_validation_report.py", "--root", str(finished_run.parent)],
                         capture_output=True, text=True, check=True).stdout
    assert "Subjects: 1" in out and "waist" in out and "2.00" in out
