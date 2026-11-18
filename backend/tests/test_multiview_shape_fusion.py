"""Tests for multi-view shape fusion: silhouette fitting, shape aggregation, masks and circumferences."""
import math
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from backend.services import body_measurements as bm
from backend.services import smpl_multiview as smv
from backend.services import subject_masks as sm

SMPL_PKL = Path("data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl")


def _rot_y(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])


class _Ellipsoid(torch.nn.Module):
    """Differentiable stand-in for SMPL: a point-sampled ellipsoid whose radii are set by betas."""

    def __init__(self):
        super().__init__()
        g = torch.linspace(0, math.pi, 24)
        a = torch.linspace(0, 2 * math.pi, 48)
        th, ph = torch.meshgrid(g, a, indexing="ij")
        unit = torch.stack([th.sin() * ph.cos(), th.cos(), th.sin() * ph.sin()], -1).reshape(-1, 3)
        self.register_buffer("unit", unit)

    def forward(self, betas, global_orient, body_pose, pose2rot=False):
        radii = torch.stack([0.25 * (1 + 0.3 * betas[:, 0]), 0.9 * (1 + 0.1 * betas[:, 1]),
                             0.15 * (1 + 0.3 * betas[:, 2])], -1)
        pts = self.unit[None] * radii[:, None, :]
        pts = torch.einsum("vij,vkj->vki", global_orient[:, 0], pts)
        out = type("Out", (), {})()
        out.vertices = pts
        # 25 "joints": a fixed subset of surface points, so keypoints alone barely see the radii.
        out.joints = pts[:, ::len(self.unit) // 25][:, :25] * 0 + pts[:, :1] * 0
        return out


def _ellipsoid_views(true_beta, yaws, focal=4000.0, size=(600, 1000)):
    body = _Ellipsoid()
    V = len(yaws)
    go = torch.stack([torch.as_tensor(_rot_y(a), dtype=torch.float32) for a in yaws])[:, None]
    verts = body(torch.as_tensor(true_beta, dtype=torch.float32)[None].expand(V, -1), go, None).vertices
    cam = np.tile([0.0, 0.0, 8.0], (V, 1))
    center = np.tile([size[0] / 2, size[1] / 2], (V, 1))
    contours, dists, masks = [], [], []
    for i in range(V):
        uv = smv.project_np(verts[i].numpy() + cam[i], focal, center[i])
        mask = np.zeros((size[1], size[0]), np.uint8)
        cv2.fillConvexPoly(mask, cv2.convexHull(uv.astype(np.int32)), 1)
        ys, xs = np.nonzero(mask)
        box = [xs.min(), ys.min(), xs.max(), ys.max()]
        pts, w, dist = sm.silhouette_target(mask, box)
        contours.append(pts)
        dists.append(dist)
        masks.append(mask)
    views = smv.FitViews(keypoints=np.zeros((V, 17, 2)), weights=np.zeros((V, 17)), global_orient=go[:, 0].numpy(),
                         body_pose=np.tile(np.eye(3), (V, 23, 1, 1)), cam_t=cam,
                         box_height=np.full(V, float(masks[0].any(1).sum())), center=center, focal=focal,
                         contours=contours, dist_maps=dists)
    return body, views, masks


def test_silhouette_fit_recovers_width_and_depth_that_keypoints_cannot_see():
    torch.manual_seed(0)
    true_beta = np.array([1.0, 0.0, -1.0] + [0.0] * 7)
    body, views, _ = _ellipsoid_views(true_beta, np.linspace(0, 315, 8))
    cfg = smv.FitConfig(iterations=300, w_keypoints=0.0, w_beta_prior=0.0, w_beta_init=0.0, w_silhouette=1.0,
                        w_pose_prior=50.0, w_depth=50.0, lr_beta=0.05)
    result = smv.fit_multiview(body, views, np.zeros(10), cfg, device="cpu")
    # Width needs frontal views, depth needs profile views: both are recovered from one shared beta.
    assert abs(result.betas[0] - 1.0) < 0.25
    assert abs(result.betas[2] + 1.0) < 0.25
    assert result.loss_history[-1]["silhouette"] < result.loss_history[0]["silhouette"]


def test_silhouette_from_a_single_view_cannot_recover_depth():
    true_beta = np.array([1.0, 0.0, -1.0] + [0.0] * 7)
    body, views, _ = _ellipsoid_views(true_beta, [0, 0, 0])
    cfg = smv.FitConfig(iterations=300, w_keypoints=0.0, w_beta_prior=0.0, w_beta_init=0.0, w_silhouette=1.0,
                        w_pose_prior=50.0, w_depth=50.0, lr_beta=0.05)
    result = smv.fit_multiview(body, views, np.zeros(10), cfg, device="cpu")
    assert abs(result.betas[0] - 1.0) < 0.25          # frontal width is observable
    assert abs(result.betas[2] + 1.0) > 0.5           # depth is not: multiple views are required


def test_excluded_silhouette_views_do_not_influence_shape():
    true_beta = np.array([1.0, 0.0, 0.0] + [0.0] * 7)
    body, views, _ = _ellipsoid_views(true_beta, [0, 90, 180, 270])
    views.sil_weight = np.zeros(4)
    cfg = smv.FitConfig(iterations=100, w_keypoints=0.0, w_beta_prior=0.0, w_beta_init=0.0, w_silhouette=1.0)
    result = smv.fit_multiview(body, views, np.zeros(10), cfg, device="cpu")
    assert np.allclose(result.betas, 0.0, atol=1e-6)


def test_beta_sigma_prior_limits_drift_per_coefficient():
    true_beta = np.array([2.0, 0.0, 0.0] + [0.0] * 7)
    body, views, _ = _ellipsoid_views(true_beta, [0, 90, 180, 270])
    sigma = np.full(10, 1.0)
    sigma[0] = 0.05
    cfg = smv.FitConfig(iterations=300, w_keypoints=0.0, w_beta_prior=0.0, w_beta_init=1.0, w_silhouette=1.0,
                        beta_sigma=sigma, lr_beta=0.05)
    tight = smv.fit_multiview(body, views, np.zeros(10), cfg, device="cpu")
    cfg.beta_sigma = np.full(10, 10.0)
    loose = smv.fit_multiview(body, views, np.zeros(10), cfg, device="cpu")
    assert abs(tight.betas[0]) < 0.5 * abs(loose.betas[0])


def test_robust_beta_aggregate_rejects_outlier_frame():
    rng = np.random.RandomState(0)
    b = rng.normal(0.5, 0.05, size=(15, 10))
    b[3] += 4.0
    agg, std, outliers = smv.robust_beta_aggregate(b)
    assert list(outliers) == [3]
    assert np.allclose(agg, 0.5, atol=0.06)
    assert (std < 0.2).all()


def test_rasterized_silhouette_metrics():
    uv = np.array([[10, 10], [60, 10], [60, 90], [10, 90]], float)
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    pred = smv.rasterize_silhouette(uv, faces, (100, 100))
    assert abs(pred.sum() - 51 * 81) < 200
    m = smv.silhouette_metrics(pred, pred, 80.0)
    assert m["iou"] == 1.0 and m["contour_mean_pct_height"] == 0.0
    shifted = smv.rasterize_silhouette(uv + [10, 0], faces, (100, 100))
    m2 = smv.silhouette_metrics(pred, shifted, 80.0)
    assert 0.6 < m2["iou"] < 0.8 and m2["contour_mean_pct_height"] > 1.0


def test_majority_vote_and_cleanup_keep_subject_component():
    a = np.zeros((200, 100), np.uint8)
    a[20:180, 30:70] = 1
    b, c = a.copy(), a.copy()
    b[100:110, 70:95] = 1                      # one source leaks into the background
    c[50:60, 40:60] = 0                        # one source drops a region (e.g. a print)
    voted = sm.majority_vote([a, b, c])
    assert voted[100:110, 75:95].sum() == 0 and voted[50:60, 40:60].all()
    blob = voted.copy()
    blob[2:22, 2:22] = 1                       # stray blob away from the subject
    kps = np.array([[50, 40], [50, 160]], float)
    cleaned, comps = sm.clean_mask(blob, kps, box_height=160)
    assert cleaned[2:22, 2:22].sum() == 0 and cleaned[100, 50] == 1 and comps == 2


def test_silhouette_target_downweights_hair_and_shoes():
    mask = np.zeros((400, 200), np.uint8)
    mask[20:380, 60:140] = 1
    kps = np.zeros((17, 2))
    kps[[5, 6], 1] = 100
    kps[[15, 16], 1] = 350
    pts, w, (dist, origin) = sm.silhouette_target(mask, [60, 20, 140, 380], kps)
    assert len(pts) > 0 and w.min() == 0.3 and w.max() == 1.0
    assert (w[pts[:, 1] < 90] == 0.3).all() and (w[(pts[:, 1] > 120) & (pts[:, 1] < 340)] == 1.0).all()
    assert dist.min() == 0 and dist.max() > 0


# ---------------------------------------------------------- circumferences
def _cylinder(radius=0.1, height=1.0, sections=256):
    import trimesh
    return trimesh.creation.cylinder(radius=radius, height=height, sections=sections)


def test_section_of_cylinder_matches_circumference():
    mesh = _cylinder()
    m = bm.measure_section(mesh.vertices, mesh.faces, [0, 0, 0.1], [0, 0, 1])
    assert m["valid"] and m["closed_loops"] == 1
    assert abs(m["perimeter"] - 2 * math.pi * 0.1) < 1e-3
    assert abs(m["tape_perimeter"] - m["perimeter"]) < 1e-6


def test_section_selects_loop_enclosing_reference_and_tape_bridges_concavity():
    import trimesh
    a = _cylinder(0.1).apply_translation([-0.15, 0, 0])
    b = _cylinder(0.1).apply_translation([0.15, 0, 0])
    both = trimesh.util.concatenate([a, b])
    left = bm.measure_section(both.vertices, both.faces, [-0.15, 0, 0], [0, 0, 1])
    assert left["valid"] and left["loops_in_plane"] == 2
    assert abs(left["perimeter"] - 2 * math.pi * 0.1) < 1e-3
    gap = bm.measure_section(both.vertices, both.faces, [0, 0, 0], [0, 0, 1])
    assert not gap["valid"]  # the point between two legs is not inside a body section
    hull = bm._hull(np.array([[0, 0], [2, 0], [2, 2], [1, 0.5], [0, 2]], float))
    assert len(hull) == 4  # the concave vertex is bridged like a tape measure


@pytest.mark.skipif(not SMPL_PKL.is_file(), reason="SMPL model file not available")
def test_measurement_pose_mesh_gives_valid_symmetric_circumferences():
    import smplx
    smpl = smplx.SMPLLayer(model_path=str(SMPL_PKL))
    faces = np.asarray(smpl.faces, dtype=np.int64)
    verts, joints, pose, _ = smv.canonical_mesh(smpl, np.zeros(10), np.tile(np.eye(3), (3, 23, 1, 1)), None,
                                                device="cpu", pose="measurement")
    report = smv.validate_mesh(verts, faces)
    assert report["upright"] and report["connected_components"] == 1
    j24 = smv.smpl_joints24(smpl, verts)
    assert j24[16, 1] > j24[18, 1] > j24[20, 1]  # A-pose: elbows and wrists below the shoulders
    result = bm.measure_body(verts, faces, j24)
    ms = result["measurements"]
    assert result["valid_count"] == result["total"], {k: m.get("reason") for k, m in ms.items() if not m["valid"]}
    for limb in ("upper_arm", "forearm", "thigh", "calf"):
        left, right = ms[f"{limb}_left"]["tape_circumference"], ms[f"{limb}_right"]["tape_circumference"]
        assert abs(left - right) / left < 0.03
    assert ms["waist"]["tape_circumference"] < ms["hip"]["tape_circumference"]
    assert ms["thigh_left"]["tape_circumference"] > ms["calf_left"]["tape_circumference"]
    # SMPL's mean body is ~1.7 units tall: torso girths must be plausible fractions of height.
    for name in ("chest", "waist", "hip"):
        assert 0.35 < ms[name]["tape_circumference_over_height"] < 0.75
    # A fatter body (beta[1] is the main girth direction of the neutral model) has a larger waist.
    betas = np.zeros(10)
    betas[1] = -2.0
    v2, _, _, _ = smv.canonical_mesh(smpl, betas, np.tile(np.eye(3), (3, 23, 1, 1)), None, device="cpu",
                                     pose="measurement")
    w2 = bm.measure_body(v2, faces, smv.smpl_joints24(smpl, v2))["measurements"]["waist"]
    assert w2["valid"] and w2["tape_circumference_over_height"] != ms["waist"]["tape_circumference_over_height"]


def test_cv_folds_hold_out_every_view_once_and_never_train_on_it():
    from backend.services import body_reconstruction as br
    selected = list(range(100, 123))
    view_info = {p: {"yaw_deg": float(-180 + 15.5 * k)} for k, p in enumerate(selected)}
    held_all = []
    for train, held, gaps in br.cv_folds(selected, view_info, 4):
        assert not set(train) & set(held)
        assert len(gaps) == len(held) and min(gaps) > 0
        held_all += held
    assert sorted(held_all) == list(range(len(selected)))
