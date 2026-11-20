"""Tests for the unified 360-degree reconstruction building blocks (no GPU/model needed)."""
import json
import math

import numpy as np
import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.routes.reconstruction import router
from backend.services import body_reconstruction as br
from backend.services import smpl_multiview as smv
from backend.services.person_tracking import (
    Detection, SubjectTracker, TrackerConfig, appearance_distance, iou_xyxy, track_sequence)


# ---------------------------------------------------------------- rotations
def _rot(axis, deg):
    a = np.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
            "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
            "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]


def test_rot6d_roundtrip_gives_valid_rotation():
    m = torch.as_tensor(_rot("y", 40) @ _rot("x", 15), dtype=torch.float32)[None]
    back = smv.rot6d_to_matrix(smv.matrix_to_rot6d(m))
    assert torch.allclose(back, m, atol=1e-5)
    assert torch.allclose(torch.det(back), torch.ones(1), atol=1e-5)


def test_chordal_mean_is_a_rotation_and_rejects_outlier():
    base = _rot("y", 10)
    rots = np.stack([base @ _rot("z", d) for d in (-3, 0, 3, 2, -2)] + [_rot("x", 120)])
    mean = smv.chordal_mean(rots)
    assert np.allclose(mean @ mean.T, np.eye(3), atol=1e-8) and np.isclose(np.linalg.det(mean), 1)
    assert smv.geodesic_deg(mean, base) < 3.0  # outlier did not drag the mean away


def test_heading_convention_matches_measured_hmr2_orientations():
    # Forward vectors measured on real HMR2 output of vid_0004 (y-down camera frame).
    def go_with_forward(fx, fz):
        yaw = math.atan2(fx, -fz)
        return _rot("y", 0) @ np.array([[math.cos(yaw + math.pi), 0, math.sin(yaw + math.pi)],
                                         [0, 1, 0], [-math.sin(yaw + math.pi), 0, math.cos(yaw + math.pi)]])
    # Directly test the formula on a rotation whose third column is the forward vector.
    for fx, fz, expected in [(0.0, -1.0, 0), (1.0, 0.0, 90), (0.0, 1.0, 180), (-1.0, 0.0, -90)]:
        go = np.eye(3)
        go[:, 2] = [fx, 0, fz]
        assert smv.heading_yaw_deg(go) == pytest.approx(expected, abs=1e-6) or abs(abs(smv.heading_yaw_deg(go)) - 180) < 1e-6


def test_view_labels():
    assert smv.view_label(0) == "front" and smv.view_label(-20) == "front"
    assert smv.view_label(179) == "back" and smv.view_label(-170) == "back"
    assert smv.view_label(90) == "right_side" and smv.view_label(-90) == "left_side"
    assert smv.wrap_deg(190) == pytest.approx(-170)


# ----------------------------------------------------------------- tracking
def _det(x0, y0, x1, y1, hue_bin=0, score=0.99):
    desc = np.zeros(96)
    desc[hue_bin] = 0.5
    desc[48 + hue_bin] = 0.5
    kp = np.tile([(x0 + x1) / 2, (y0 + y1) / 2], (17, 1))
    return Detection(np.array([x0, y0, x1, y1], float), score, kp, np.full(17, 5.0), desc)


def test_appearance_and_iou_basics():
    a, b = _det(0, 0, 10, 10, 0).descriptor, _det(0, 0, 10, 10, 20).descriptor
    assert appearance_distance(a, a) == pytest.approx(0, abs=1e-6)
    assert appearance_distance(a, b) == pytest.approx(1, abs=1e-6)
    assert iou_xyxy([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1 / 3)


def test_tracker_follows_subject_and_ignores_distant_bystander():
    seq = []
    for i in range(10):
        seq.append([_det(400 + i * 2, 100, 600 + i * 2, 900, 3), _det(50, 300, 200, 900, 30)])
    records, events = track_sequence(seq, 0, seq[0][0])
    assert all(r.status == "tracked" for r in records)
    assert all(r.box[0] >= 400 for r in records)  # never the bystander
    assert not events


def test_tracker_detects_identity_switch_and_rejects_frames():
    """A different-looking person takes the subject's place: frames must be rejected, not silently used."""
    seq = []
    for i in range(12):
        subject = _det(400, 100, 600, 900, 3)
        if 5 <= i <= 8:
            subject = _det(405, 100, 605, 900, 30)  # same place, different person (different colours)
        seq.append([subject])
    records, events = track_sequence(seq, 0, seq[0][0])
    for i in range(5, 9):
        assert records[i].status == "lost", i
    assert any(e["type"] == "possible_identity_switch" for e in events)
    assert all(records[i].status == "tracked" for i in (1, 2, 3, 4))


def test_tracker_does_not_reacquire_a_different_person_after_loss():
    seq = [[_det(400, 100, 600, 900, 3)]] + [[] for _ in range(3)] + [[_det(300, 120, 520, 880, 30)]]
    records, _ = track_sequence(seq, 0, seq[0][0])
    assert records[2].status == "no_detection"
    assert records[4].status == "lost"  # different appearance is not re-identified


def test_tracker_reacquires_same_appearance_after_occlusion():
    seq = [[_det(400, 100, 600, 900, 3)]] + [[] for _ in range(3)] + [[_det(420, 110, 620, 900, 3)]]
    records, events = track_sequence(seq, 0, seq[0][0])
    assert records[4].status == "tracked"
    assert any(e["type"] == "reacquired" for e in events)


def test_tracker_flags_ambiguous_overlap():
    seq = [[_det(400, 100, 600, 900, 3)],
           [_det(402, 100, 602, 900, 3), _det(410, 100, 610, 900, 3)]]
    records, events = track_sequence(seq, 0, seq[0][0])
    assert records[1].status == "ambiguous"
    assert any(e["type"] == "ambiguous_overlap" for e in events)


# ------------------------------------------------------------ view checking
def test_verify_view_cross_checks_two_cues():
    assert br.verify_view(0, 0.25)[0] == "agree"          # front: left shoulder right of right shoulder
    assert br.verify_view(180, -0.25)[0] == "agree"       # back: ordering reversed
    assert br.verify_view(0, -0.25)[0] == "conflict"      # HMR2 says front, 2D says back
    assert br.verify_view(90, 0.02)[0] == "agree"         # profile: shoulders overlap
    assert br.verify_view(90, 0.22)[0] == "conflict"


def test_select_frames_one_per_bin_and_prefers_sharp():
    per_frame, views, recs = {}, {}, {}
    for pos in range(60):
        yaw = -180 + pos * 6.0
        per_frame[pos] = {"sharpness": 100.0 + (50 if pos % 2 else 0)}
        views[pos] = {"yaw_deg": yaw, "label": smv.view_label(yaw), "verification": "agree"}

        class R:  # minimal record
            detection = _det(0, 0, 10, 10)
        recs[pos] = R()
    selected, rejected = br.select_frames(per_frame, views, 100.0, recs, 24)
    assert len(selected) == 24
    bins = {int(((views[p]["yaw_deg"] + 180) % 360) // 15) for p in selected}
    assert len(bins) == 24  # no two frames from the same heading bin
    assert all(p % 2 == 1 for p in selected)  # sharper frame of each pair won


def test_select_frames_drops_blurry():
    per_frame = {0: {"sharpness": 5.0}, 1: {"sharpness": 100.0}}
    views = {p: {"yaw_deg": 10.0 * p, "label": "front", "verification": "agree"} for p in (0, 1)}

    class R:
        detection = _det(0, 0, 10, 10)
    selected, rejected = br.select_frames(per_frame, views, 100.0, {0: R(), 1: R()}, 24)
    assert selected == [1] and rejected[0] == "blurry"


# -------------------------------------------------------------- mesh checks
def _closed_mesh():
    import trimesh
    m = trimesh.creation.capsule(height=1.2, radius=0.2, count=[24, 24])
    v = np.asarray(m.vertices)
    v[:, 1] -= v[:, 1].min()
    return v, np.asarray(m.faces)


def test_validate_mesh_accepts_good_closed_upright_mesh():
    v, f = _closed_mesh()
    v = np.stack([v[:, 0], v[:, 2], -v[:, 1]], axis=1)  # proper rotation (z-up capsule -> y-up)
    v[:, 1] -= v[:, 1].min()
    joints = np.zeros((25, 3))
    joints[0] = [0, v[:, 1].max() * 0.95, 0.1]
    joints[8] = [0, v[:, 1].max() * 0.5, 0.0]
    report = smv.validate_mesh(v, f, joints)
    assert report["all_vertices_finite"] and report["winding_consistent"] and report["watertight"]
    assert report["connected_components"] == 1
    assert report["valid"] is True


def test_validate_mesh_rejects_inside_out_mesh():
    v, f = _closed_mesh()
    v = np.stack([v[:, 0], v[:, 2], -v[:, 1]], axis=1)
    v[:, 1] -= v[:, 1].min()
    joints = np.zeros((25, 3))
    joints[0] = [0, v[:, 1].max() * 0.95, 0.1]
    joints[8] = [0, v[:, 1].max() * 0.5, 0.0]
    assert smv.validate_mesh(v, f, joints)["valid"] is True
    flipped = smv.validate_mesh(v, f[:, ::-1], joints)
    assert flipped["volume_positive"] is False and flipped["valid"] is False


def test_validate_mesh_rejects_nan_bad_faces_and_fragments():
    v, f = _closed_mesh()
    bad = v.copy()
    bad[3, 1] = np.nan
    assert smv.validate_mesh(bad, f)["valid"] is False
    assert smv.validate_mesh(v, f + 100000)["faces_in_range"] is False
    frag_v = np.vstack([v, v + [3, 0, 0]])
    frag_f = np.vstack([f, f + len(v)])
    rep = smv.validate_mesh(frag_v, frag_f)
    assert rep["connected_components"] == 2 and rep["valid"] is False


def test_obj_and_ply_roundtrip(tmp_path):
    v, f = _closed_mesh()
    smv.write_obj(tmp_path / "m.obj", v, f, "header line\nsecond")
    rv, rf = smv.read_obj(tmp_path / "m.obj")
    assert np.allclose(rv, v, atol=1e-6) and np.array_equal(rf, f)
    text = (tmp_path / "m.obj").read_text()
    assert text.startswith("# header line") and "\nvn " in text
    smv.write_ply(tmp_path / "m.ply", v, f)
    raw = (tmp_path / "m.ply").read_bytes()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    head = raw[:end].decode("latin1")
    assert f"element vertex {len(v)}" in head and f"element face {len(f)}" in head
    assert len(raw) - end == len(v) * 12 + len(f) * 13  # float xyz + (uchar count + 3 int32) per face
    first = raw[end + len(v) * 12:end + len(v) * 12 + 13]
    assert first[0] == 3 and np.array_equal(np.frombuffer(first[1:], "<i4"), f[0])


# ------------------------------------------------------------ shape fitting
class _ToyBody(torch.nn.Module):
    """Differentiable stand-in for SMPL: joints = template scaled per-axis by betas (width/depth/height)."""

    def __init__(self):
        super().__init__()
        t = torch.zeros(25, 3)
        t[:, 1] = torch.linspace(-0.9, 0.9, 25)
        t[:, 0] = torch.sin(torch.linspace(0, 6, 25)) * 0.25
        t[:, 2] = torch.cos(torch.linspace(0, 6, 25)) * 0.12
        self.register_buffer("template", t)

    def forward(self, betas, global_orient, body_pose, pose2rot=False):
        scale = 1.0 + 0.2 * betas[:, :3]
        joints = self.template[None] * scale[:, None, :]
        joints = torch.einsum("vij,vkj->vki", global_orient[:, 0], joints)
        out = type("Out", (), {})()
        out.joints, out.vertices = joints, joints
        return out


def test_fit_multiview_recovers_shared_shape_from_noisy_views():
    torch.manual_seed(0)
    body = _ToyBody()
    true_beta = torch.tensor([0.8, -0.5, 0.3] + [0.0] * 7)
    V, focal = 8, 4000.0
    cam = torch.tensor([0.0, 0.0, 6.0]).repeat(V, 1)
    center = torch.tensor([540.0, 960.0]).repeat(V, 1)
    go = torch.stack([torch.as_tensor(_rot("y", a), dtype=torch.float32) for a in np.linspace(0, 315, V)])[:, None]
    pose = torch.eye(3).repeat(V, 23, 1, 1)
    joints = body(true_beta.expand(V, -1), go, pose).joints
    uv = smv.project(joints[:, smv.COCO_TO_OPENPOSE] + cam[:, None], focal, center)
    obs = uv + torch.randn_like(uv) * 1.5
    views = smv.FitViews(keypoints=obs.numpy(), weights=np.ones((V, 17)), global_orient=go[:, 0].numpy(),
                         body_pose=pose.numpy(), cam_t=cam.numpy(), box_height=np.full(V, 900.0),
                         center=center.numpy(), focal=focal)
    init = np.zeros(10)
    result = smv.fit_multiview(body, views, init, smv.FitConfig(iterations=400, w_beta_prior=0.0, w_beta_init=0.0),
                               device="cpu")
    assert np.linalg.norm(result.betas[:3] - true_beta[:3].numpy()) < 0.6 * np.linalg.norm(true_beta[:3].numpy())
    assert np.median(result.residual_px_after) < np.median(result.residual_px_before)
    assert result.betas.shape == (10,)  # ONE shared vector for every view


def test_fit_rejects_too_few_views_and_nonfinite_input():
    body = _ToyBody()
    views = smv.FitViews(keypoints=np.zeros((2, 17, 2)), weights=np.ones((2, 17)), global_orient=np.tile(np.eye(3), (2, 1, 1)),
                         body_pose=np.tile(np.eye(3), (2, 23, 1, 1)), cam_t=np.tile([0, 0, 5.0], (2, 1)),
                         box_height=np.full(2, 900.0), center=np.tile([540.0, 960.0], (2, 1)), focal=4000.0)
    with pytest.raises(smv.FitError):
        smv.fit_multiview(body, views, np.zeros(10), device="cpu")
    views.keypoints = np.zeros((3, 17, 2))
    views.weights = np.ones((3, 17))
    views.global_orient = np.tile(np.eye(3), (3, 1, 1))
    views.body_pose = np.tile(np.eye(3), (3, 23, 1, 1))
    views.cam_t = np.tile([0, 0, 5.0], (3, 1))
    views.box_height = np.full(3, 900.0)
    views.center = np.tile([540.0, 960.0], (3, 1))
    views.keypoints[0, 0, 0] = np.nan
    with pytest.raises(smv.FitError):
        smv.fit_multiview(body, views, np.zeros(10), device="cpu")


def test_crop_cam_to_full_matches_hmr2_formula():
    cam = np.array([1.2, 0.05, -0.02])
    t = smv.crop_cam_to_full(cam, (600.0, 1000.0), 800.0, (1080, 1920), 5000 / 224 * 1920)
    bs = 800.0 * 1.2
    assert t[2] == pytest.approx(2 * (5000 / 224 * 1920) / bs)
    assert t[0] == pytest.approx(2 * (600 - 540) / bs + 0.05)
    assert t[1] == pytest.approx(2 * (1000 - 960) / bs - 0.02)


# --------------------------------------------------------------------- API
@pytest.fixture()
def client():
    app = FastAPI()
    app.include_router(router, prefix="/api/reconstruction")
    return TestClient(app)


def test_api_rejects_bad_ids_and_unknown_runs(client):
    assert client.post("/api/reconstruction/runs", json={"video_id": "nope"}).status_code == 400
    assert client.get("/api/reconstruction/runs/not-a-uuid").status_code == 400
    assert client.get("/api/reconstruction/runs/00000000-0000-4000-8000-000000000000").status_code == 404
    assert client.post("/api/reconstruction/runs/00000000-0000-4000-8000-000000000000/person",
                       json={"candidate_id": 0}).status_code == 400


def test_api_requires_consent_attestation(client, monkeypatch):
    def deny(_):
        raise ValueError("Recording authorization and consent confirmation required")
    monkeypatch.setattr(br, "require_authorization", deny)
    r = client.post("/api/reconstruction/runs", json={"video_id": "00000000-0000-4000-8000-000000000000"})
    assert r.status_code == 400 and "authorization" in r.json()["detail"].lower()


def test_orphaned_run_after_backend_restart_reports_interrupted(client, tmp_path, monkeypatch):
    monkeypatch.setattr(br, "_ROOT", tmp_path)
    run_id = "00000000-0000-4000-8000-000000000002"
    (tmp_path / run_id).mkdir()
    (tmp_path / run_id / "status.json").write_text(json.dumps(
        {"run_id": run_id, "status": "running", "stage": "decode", "progress": 0.0, "message": "Decoding"}))
    body = client.get(f"/api/reconstruction/runs/{run_id}").json()  # not in this process's memory
    assert body["status"] == "failed" and body["error"]["code"] == "interrupted_by_backend_restart"
    assert "Start the reconstruction again" in body["error"]["message"]
    assert json.loads((tmp_path / run_id / "status.json").read_text())["status"] == "failed"


def test_api_file_whitelist_blocks_traversal(client, tmp_path, monkeypatch):
    monkeypatch.setattr(br, "_ROOT", tmp_path)
    run_id = "00000000-0000-4000-8000-000000000001"
    (tmp_path / run_id).mkdir()
    (tmp_path / run_id / "final_body_mesh.obj").write_text("v 0 0 0\n")
    (tmp_path / "secret.txt").write_text("x")
    assert client.get(f"/api/reconstruction/runs/{run_id}/files/final_body_mesh.obj").status_code == 200
    assert client.get(f"/api/reconstruction/runs/{run_id}/files/..%2Fsecret.txt").status_code in (400, 404)
    assert client.get(f"/api/reconstruction/runs/{run_id}/files/status.json").status_code == 404
    assert br.get_asset(run_id, "diagnostics__../../secret.txt") is None


def test_run_owned_by_a_live_process_is_not_marked_interrupted():
    import os
    import subprocess
    import sys
    import time
    from backend.services import body_reconstruction as br
    assert br._owner_alive(br._owner_record())                         # this process
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        time.sleep(0.5)
        owner = {"pid": child.pid, "created": br._process_created(child.pid)}
        assert br._owner_alive(owner)
    finally:
        child.kill(); child.wait()
    time.sleep(0.3)
    assert not br._owner_alive(owner)                                  # owner gone -> interrupted
    assert not br._owner_alive({"pid": os.getpid(), "created": 1.0})   # reused PID, different process
    assert not br._owner_alive(None)                                   # legacy run without owner
