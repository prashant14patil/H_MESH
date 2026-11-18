"""SHAPY measurement-model plumbing (no SHAPY weights or GPU needed)."""
import json

import cv2
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.routes.shapy import router
from backend.services import body_reconstruction as br
from backend.services import shapy_shape

RUN_ID = "00000000-0000-4000-8000-0000000000bb"


def test_crop_centres_box_and_normalises():
    frame = np.zeros((400, 200, 3), np.uint8)
    frame[100:300, 50:150] = (0, 0, 255)  # red (BGR) block = subject box
    crop = shapy_shape.crop_for_shapy(frame, (50, 100, 150, 300))
    assert crop.shape == (256, 256, 3)
    rgb = crop * shapy_shape.STD + shapy_shape.MEAN
    assert rgb[128, 128, 0] == pytest.approx(1.0, abs=1e-3) and rgb[128, 128, 2] == pytest.approx(0.0, abs=1e-3)
    # box height 200 -> crop side 240 -> box spans 200/240 of the 256 crop vertically
    red_rows = np.where(rgb[:, 128, 0] > 0.5)[0]
    assert (red_rows.max() - red_rows.min() + 1) == pytest.approx(256 * 200 / 240, abs=2)


def test_mask_box_scales_to_frame_and_rejects_empty():
    mask = np.zeros((100, 50), np.uint8)
    mask[20:80, 10:40] = 1
    assert shapy_shape.mask_box(mask, (200, 100)) == (20.0, 40.0, 80.0, 160.0)
    assert shapy_shape.mask_box(np.zeros((100, 50), np.uint8), (200, 100)) is None


def test_read_display_frames_refuses_misoriented_frames(tmp_path):
    path = tmp_path / "v.avi"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 32))
    for _ in range(3):
        w.write(np.full((32, 64, 3), 128, np.uint8))
    w.release()
    assert shapy_shape.read_display_frames(path, [0, 2], (32, 64)).keys() == {0, 2}
    with pytest.raises(ValueError, match="does not match"):
        shapy_shape.read_display_frames(path, [0], (64, 32))


def _record():
    return {"mesh_height": 1.7, "valid_count": 2, "total": 3, "model": "SHAPY", "frames_used": 5,
            "measurements": {"waist": {"valid": True, "tape_circumference": 0.85, "tape_circumference_over_height": 0.5},
                             "hip": {"valid": True, "tape_circumference": 1.02, "tape_circumference_over_height": 0.6},
                             "calf_left": {"valid": False}}}


def test_summarize_follows_height():
    s = shapy_shape.summarize(_record(), None)
    assert s["measurements"]["waist"] == 0.5 and "measurements_cm" not in s
    s = shapy_shape.summarize(_record(), 170)
    assert s["measurements_cm"]["waist"] == pytest.approx(85.0) and s["measurements_cm"]["calf_left"] is None


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(br, "_ROOT", tmp_path)
    monkeypatch.setattr(br, "_RUNS", {})
    (tmp_path / RUN_ID).mkdir()
    app = FastAPI()
    app.include_router(router, prefix="/api/reconstruction")
    return TestClient(app)


def _status(tmp_path, status="completed", height=None):
    result = {"mesh_obj": "final_body_mesh.obj"}
    if height:
        result["metric_scale"] = {"height_cm": height}
    (tmp_path / RUN_ID / "status.json").write_text(json.dumps(
        {"run_id": RUN_ID, "video_id": "00000000-0000-4000-8000-0000000000cc", "status": status, "result": result}))


def test_api_reports_models_and_uncomputed_state(client, tmp_path):
    ids = {m["id"] for m in client.get("/api/reconstruction/models").json()["models"]}
    assert ids == {"pipeline", "shapy"}
    _status(tmp_path)
    body = client.get(f"/api/reconstruction/runs/{RUN_ID}/shapy").json()
    assert body["computed"] is False
    assert client.get(f"/api/reconstruction/runs/{RUN_ID}/shapy/mesh").status_code == 404


def test_api_returns_stored_result_in_cm_of_current_height(client, tmp_path):
    _status(tmp_path, height=176)
    (tmp_path / RUN_ID / shapy_shape.RESULT_FILE).write_text(json.dumps(_record()))
    body = client.post(f"/api/reconstruction/runs/{RUN_ID}/shapy").json()  # stored: no recompute
    assert body["computed"] and body["height_cm"] == 176
    assert body["measurements_cm"]["hip"] == pytest.approx(105.6, abs=0.1)


def test_api_rejects_unfinished_or_unknown_runs_and_missing_model(client, tmp_path, monkeypatch):
    assert client.get("/api/reconstruction/runs/00000000-0000-4000-8000-000000000000/shapy").status_code == 404
    assert client.get("/api/reconstruction/runs/bad-id/shapy").status_code == 400
    _status(tmp_path, status="failed")
    assert client.post(f"/api/reconstruction/runs/{RUN_ID}/shapy").status_code == 400
    _status(tmp_path)
    monkeypatch.setattr(shapy_shape, "available", lambda: False)
    r = client.post(f"/api/reconstruction/runs/{RUN_ID}/shapy")
    assert r.status_code == 503 and "not installed" in r.json()["detail"]
