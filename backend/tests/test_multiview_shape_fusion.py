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


def _fit_ellipsoid(views, form):
    torch.manual_seed(0)
    cfg = smv.FitConfig(iterations=300, w_keypoints=0.0, w_beta_prior=0.0, w_beta_init=0.0, w_silhouette=1.0,
                        w_pose_prior=50.0, w_depth=50.0, lr_beta=0.05, sil_outside_form=form)
    return smv.fit_multiview(_Ellipsoid(), views, np.zeros(10), cfg, device="cpu")


def test_inside_term_recovers_depth_at_least_as_well_as_legacy():
    """Exact outlines: the corrected outside term must not lose the depth the profile views carry."""
    true_beta = np.array([1.0, 0.0, -1.0] + [0.0] * 7)
    _, views, _ = _ellipsoid_views(true_beta, np.linspace(0, 315, 8))
    new, old = _fit_ellipsoid(views, "charbonnier"), _fit_ellipsoid(views, "legacy")
    assert abs(new.betas[2] + 1.0) < 0.25 and abs(new.betas[0] - 1.0) < 0.25
    assert abs(new.betas[2] + 1.0) <= abs(old.betas[2] + 1.0) + 0.02


def test_inside_term_limits_inflation_under_loose_outline():
    """Outline grown by a 'garment': the body must inflate less with the corrected (non-saturating) term."""
    true_beta = np.array([1.0, 0.0, -1.0] + [0.0] * 7)
    _, views, masks = _ellipsoid_views(true_beta, np.linspace(0, 315, 8))
    kernel = np.ones((31, 31), np.uint8)
    for i, m in enumerate(masks):
        loose = cv2.dilate(m, kernel)
        ys, xs = np.nonzero(loose)
        views.contours[i], _, views.dist_maps[i] = sm.silhouette_target(loose, [xs.min(), ys.min(), xs.max(), ys.max()])
    new, old = _fit_ellipsoid(views, "charbonnier"), _fit_ellipsoid(views, "legacy")
    err = lambda b: abs(b[0] - 1.0) + abs(b[2] + 1.0)  # noqa: E731
    assert err(new.betas) < err(old.betas)


def test_silhouette_metrics_split_protrusion_and_uncovered():
    obs = np.zeros((100, 100), np.uint8); obs[20:80, 20:80] = 1
    pred = np.zeros_like(obs); pred[10:70, 20:80] = 1      # 10 rows above the outline, 10 rows short below
    m = smv.silhouette_metrics(pred, obs, 60.0)
    assert m["protrusion_frac"] == pytest.approx(10 / 60)
    assert m["uncovered_frac"] == pytest.approx(10 / 60)


def test_torso_width_ratios_measure_width_per_level():
    obs = np.zeros((200, 200), np.uint8); obs[20:180, 60:140] = 1   # 80 px wide torso
    pred = np.zeros_like(obs); pred[20:180, 70:130] = 1             # 60 px wide model
    kp = np.zeros((17, 2)); kp[[5, 6]] = [[80, 40], [120, 40]]; kp[[11, 12]] = [[85, 150], [115, 150]]
    r = smv.torso_width_ratios(pred, obs, kp)
    assert all(v == pytest.approx(0.75) for v in r)


def test_locked_body_pose_is_not_changed_by_the_fit():
    true_beta = np.array([1.0, 0.0, 0.0] + [0.0] * 7)
    _, views, _ = _ellipsoid_views(true_beta, [0, 90, 180, 270])
    cfg = smv.FitConfig(iterations=20, w_keypoints=0.0, w_silhouette=1.0, fit_body_pose=False)
    r = smv.fit_multiview(_Ellipsoid(), views, np.zeros(10), cfg, device="cpu")
    assert np.allclose(r.body_pose, views.body_pose, atol=1e-5)


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


# ------------------------------------------------------------------ surface offsets (SMPL+D)
def _smpl_cpu():
    import importlib.util, os
    cache = Path(os.environ.get("USERPROFILE", str(Path.home()))) / ".cache" / "4DHumans" / "data"
    if not (cache / "smpl").exists():
        pytest.skip("SMPL model files not available")
    spec = importlib.util.spec_from_file_location("smpl_wrapper", Path("models/4D-Humans/hmr2/models/smpl_wrapper.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.SMPL(model_path=str(cache / "smpl"), gender="neutral", num_body_joints=23,
                    joint_regressor_extra=str(cache / "SMPL_to_J19.pkl")).eval()


def test_posed_vertices_without_offsets_equals_model_forward():
    smpl = _smpl_cpu()
    betas = np.linspace(-1, 1, 10)
    go = np.stack([_rot_y(a) for a in (0, 90)])
    pose = np.tile(smv.measurement_pose(), (2, 1, 1, 1))
    ref = smv.posed_output(smpl, betas, go, pose, "cpu").vertices
    got = smv.posed_vertices(smpl, torch.as_tensor(betas, dtype=torch.float32), torch.as_tensor(go, dtype=torch.float32),
                             torch.as_tensor(pose, dtype=torch.float32))
    assert float((ref - got).abs().max()) < 1e-5


def test_surface_offsets_recover_a_flank_bulge_that_betas_cannot():
    """Ground truth: SMPL body + 2.5 cm flank bulge, seen in 8 silhouettes. The offset stage must recover the
    bulge at the flanks and leave the front of the trunk (no bulge there) essentially unchanged."""
    torch.manual_seed(0)
    smpl = _smpl_cpu()
    faces = np.asarray(smpl.faces, dtype=np.int64)
    betas = np.zeros(10)
    T = smpl.v_template.numpy().astype(np.float64)
    J = smpl.J_regressor.numpy() @ T
    N = smv.rest_normals(smpl, betas, "cpu")
    rel = T - J[3]
    ang = np.degrees(np.arctan2(np.abs(rel[:, 0]), rel[:, 2]))
    y0 = J[0, 1] + 0.35 * (J[6, 1] - J[0, 1])
    trunk = (np.abs(rel[:, 0]) < 0.22) & (T[:, 1] > J[0, 1] - 0.05) & (T[:, 1] < J[6, 1] + 0.05)
    bump_w = trunk * np.exp(-((T[:, 1] - y0) / 0.05) ** 2) * np.exp(-((ang - 120) / 30) ** 2)
    true_off = 0.025 * bump_w[:, None] * N
    yaws = np.linspace(0, 315, 8)
    V = len(yaws)
    go = np.stack([_rot_y(a) for a in yaws])
    pose = np.tile(np.eye(3), (V, 23, 1, 1))
    cam = np.tile([0.0, 0.3, 6.0], (V, 1))
    focal, size = 2500.0, (700, 1000)
    center = np.tile([size[0] / 2, size[1] / 2], (V, 1))
    verts = smv.posed_vertices(smpl, torch.zeros(10), torch.as_tensor(go, dtype=torch.float32),
                               torch.as_tensor(pose, dtype=torch.float32),
                               torch.as_tensor(true_off, dtype=torch.float32)).numpy().astype(np.float64)
    contours, dists = [], []
    for i in range(V):
        uv = smv.project_np(verts[i] + cam[i], focal, center[i])
        mask = smv.rasterize_silhouette(uv, faces, (size[1], size[0]))
        ys, xs = np.nonzero(mask)
        pts, _, dist = sm.silhouette_target(mask, [xs.min(), ys.min(), xs.max(), ys.max()])
        contours.append(pts), dists.append(dist)
    views = smv.FitViews(keypoints=np.zeros((V, 17, 2)), weights=np.zeros((V, 17)), global_orient=go, body_pose=pose,
                         cam_t=cam, box_height=np.full(V, 800.0), center=center, focal=focal,
                         contours=contours, dist_maps=dists)
    fit = smv.FitResult(betas=betas, global_orient=go, body_pose=pose, cam_t=cam)
    res = smv.fit_surface_offsets(smpl, views, fit, smv.SurfaceConfig(iterations=200), device="cpu")
    est = res["offsets_cm"]
    bulge = bump_w > 0.6
    front = trunk & (ang < 40) & (np.abs(T[:, 1] - y0) < 0.05)
    assert est[bulge].mean() > 1.0                    # recovers most of the 2.5 cm flank bulge
    assert abs(est[front].mean()) < 0.5               # adds nothing where the body had no bulge


def test_canonical_detailed_mesh_passes_geometry_validation():
    smpl = _smpl_cpu()
    off = np.zeros((6890, 3)); off[:, 2] = 0.005 * smv.surface_region_weights(smpl)   # small trunk offsets
    v, joints = smv.canonical_detailed_mesh(smpl, np.zeros(10), off, device="cpu")
    report = smv.validate_mesh(v, np.asarray(smpl.faces, dtype=np.int64), joints)
    assert report["valid"], report


def test_plain_face_flattens_features_but_keeps_height_and_trunk():
    smpl = _smpl_cpu()
    faces = np.asarray(smpl.faces, dtype=np.int64)
    v = smpl.v_template.numpy().astype(np.float64)
    region = smv.face_region(smpl, v)
    out = smv.plain_face(v, faces, region)
    nose = np.argmax(np.where(region, v[:, 2], -np.inf))           # most forward face vertex
    assert out[nose, 2] < v[nose, 2] - 0.005                        # nose flattened by > 5 mm
    assert np.allclose(out[~region], v[~region])                    # nothing outside the face moves
    assert out[:, 1].max() == pytest.approx(v[:, 1].max())          # height (crown) unchanged


def test_sex_shape_prior_loads_and_neutral_means_none():
    for sex in ("male", "female"):
        mean, prec = smv.sex_shape_prior(sex)
        assert mean.shape == (10,) and prec.shape == (10, 10)
        assert np.all(np.linalg.eigvalsh(prec) > 0)              # valid Gaussian precision
    assert smv.sex_shape_prior(None) == (None, None)
    m, _ = smv.sex_shape_prior("male"); f, _ = smv.sex_shape_prior("female")
    assert np.linalg.norm(m - f) > 0.5                              # the two populations really differ


def test_validate_sex_accepts_only_known_values():
    from backend.services.body_reconstruction import _validate_sex
    assert _validate_sex(None) is None and _validate_sex("") is None and _validate_sex("unspecified") is None
    assert _validate_sex(" Male ") == "male"
    with pytest.raises(ValueError):
        _validate_sex("other")


def test_male_prior_pulls_shape_toward_male_mean_when_data_is_silent():
    """No keypoints/silhouettes: the only shape information is the prior, so the fit lands on its mean."""
    smpl = _smpl_cpu()
    mean, prec = smv.sex_shape_prior("male")
    V = 3
    views = smv.FitViews(keypoints=np.zeros((V, 17, 2)), weights=np.zeros((V, 17)),
                         global_orient=np.tile(np.eye(3), (V, 1, 1)), body_pose=np.tile(np.eye(3), (V, 23, 1, 1)),
                         cam_t=np.tile([0.0, 0.0, 5.0], (V, 1)), box_height=np.full(V, 500.0),
                         center=np.tile([250.0, 250.0], (V, 1)), focal=1000.0)
    cfg = smv.FitConfig(iterations=300, w_keypoints=0.0, w_beta_init=0.0, w_beta_prior=1.0, lr_beta=0.05,
                        beta_pop_mean=mean, beta_pop_precision=prec)
    r = smv.fit_multiview(smpl, views, np.zeros(10), cfg, device="cpu")
    assert np.linalg.norm(r.betas - mean) < 0.25 * np.linalg.norm(mean)


def test_restate_to_stature_changes_reference_not_girths():
    from backend.services.body_reconstruction import restate_to_stature
    m = {"mesh_height": 1.70, "measurements": {"waist": {"tape_circumference": 0.85, "tape_circumference_over_height": 0.5},
                                               "bad": {"valid": False}}}
    restate_to_stature(m, 1.72)
    assert m["mesh_height"] == 1.72 and m["mesh_height_measurement_pose"] == 1.70
    assert m["measurements"]["waist"]["tape_circumference"] == 0.85
    assert m["measurements"]["waist"]["tape_circumference_over_height"] == pytest.approx(0.85 / 1.72)


def test_stature_height_is_taller_than_measurement_a_pose():
    from backend.services.body_reconstruction import stature_height
    smpl = _smpl_cpu()
    v, *_ = smv.canonical_mesh(smpl, np.zeros(10), np.tile(np.eye(3), (1, 23, 1, 1)), None, "cpu", pose="measurement")
    s = stature_height(smpl, np.zeros(10), "cpu")
    assert 1.0 < s / np.ptp(v[:, 1]) < 1.01        # feet together is slightly taller than legs abducted 5 deg


def test_rest_volume_ratio_matches_closed_mesh_volume():
    import trimesh
    smpl = _smpl_cpu()
    b = np.linspace(-1, 1, 10)
    ratio = float(smv.rest_volume_ratio(smpl, torch.as_tensor(b, dtype=torch.float32),
                                        torch.as_tensor(np.asarray(smpl.faces, dtype=np.int64))))
    v = (smpl.v_template + torch.einsum("vck,k->vc", smpl.shapedirs[:, :, :10], torch.as_tensor(b, dtype=torch.float32))).numpy()
    ref = trimesh.Trimesh(v, np.asarray(smpl.faces), process=False).volume / np.ptp(v[:, 1]) ** 3
    assert ratio == pytest.approx(ref, rel=1e-4)


def test_volume_constraint_reaches_target_when_data_is_silent():
    """No image evidence: the shape must move to the volume a given weight implies (within the 3% sigma)."""
    smpl = _smpl_cpu()
    target = smv.volume_ratio_target(110.0, 175.0)          # a heavy body: well above the mean SMPL volume
    faces_t = torch.as_tensor(np.asarray(smpl.faces, dtype=np.int64))
    start = float(smv.rest_volume_ratio(smpl, torch.zeros(10), faces_t))
    V = 3
    views = smv.FitViews(keypoints=np.zeros((V, 17, 2)), weights=np.zeros((V, 17)),
                         global_orient=np.tile(np.eye(3), (V, 1, 1)), body_pose=np.tile(np.eye(3), (V, 23, 1, 1)),
                         cam_t=np.tile([0.0, 0.0, 5.0], (V, 1)), box_height=np.full(V, 500.0),
                         center=np.tile([250.0, 250.0], (V, 1)), focal=1000.0)
    cfg = smv.FitConfig(iterations=300, w_keypoints=0.0, w_beta_init=0.0, w_beta_prior=0.002, lr_beta=0.05,
                        target_volume_ratio=target)
    r = smv.fit_multiview(smpl, views, np.zeros(10), cfg, device="cpu")
    end = float(smv.rest_volume_ratio(smpl, torch.as_tensor(r.betas, dtype=torch.float32), faces_t))
    assert abs(end - target) < 0.05 * target < abs(start - target)


def test_validate_weight_needs_height_and_sane_range():
    from backend.services.body_reconstruction import _validate_weight
    assert _validate_weight(None, None) is None
    assert _validate_weight(80, 179) == 80.0
    with pytest.raises(ValueError):
        _validate_weight(80, None)          # volume is relative to stature
    with pytest.raises(ValueError):
        _validate_weight(10, 179)


def test_anthropometric_check_flags_only_the_out_of_line_girth():
    from backend.services import anthropometric_check as ac
    if not ac.available("male"):
        pytest.skip("ANSUR II reference data not downloaded")
    typical = ac.check({"chest": 106.0, "abdomen": 94.0, "hip": 102.0}, 176.0, "male")
    assert not any(v["out_of_line"] for v in typical["sites"].values())
    inflated = ac.check({"chest": 114.0, "abdomen": 103.7, "hip": 116.8}, 179.0, "male")   # run 6303a2d5
    assert inflated["sites"]["hip"]["out_of_line"] and inflated["sites"]["hip"]["z"] > 2
    assert not inflated["sites"]["chest"]["out_of_line"]
    assert ac.check({"chest": 100.0, "abdomen": 90.0, "hip": 100.0}, 170.0, None) is None


def test_mass_check_flags_a_body_heavier_than_the_person():
    from backend.services.body_reconstruction import mass_check
    smpl = _smpl_cpu()
    b = np.zeros(10)
    same = mass_check(smpl, b, 1.0, 172.0, "cpu")                       # find the mean body's implied mass first
    implied = same["implied_mass_kg"]
    ok = mass_check(smpl, b, implied, 172.0, "cpu")
    assert not ok["significant"] and abs(ok["relative_difference"]) < 0.01
    heavy = mass_check(smpl, b, implied / 1.15, 172.0, "cpu")           # body 15% heavier than the person
    assert heavy["significant"] and heavy["relative_difference"] == pytest.approx(0.15, abs=0.01)
    assert heavy["expected_girth_bias_pct"] == pytest.approx(7.2, abs=0.3)


def _male_girth_prior(smpl, betas, weight=1.0):
    from backend.services import anthropometric_check as ac
    if not ac.available("male"):
        pytest.skip("ANSUR II reference data not downloaded")
    return ac.build_girth_prior(smpl, betas, "male", 175.0, weight)


def test_differentiable_girths_equal_measurement_girths():
    smpl = _smpl_cpu()
    b = np.array([0.5, -0.3, 0.8, 0.2, 0, 0, 0, 0, 0, 0])
    gp = _male_girth_prior(smpl, b)
    g = smv.girths_mm(smpl, torch.as_tensor(b, dtype=torch.float32), gp, "cpu")
    v, *_ = smv.canonical_mesh(smpl, b, np.tile(np.eye(3), (1, 23, 1, 1)), None, "cpu", pose="measurement")
    m = bm.measure_body(v, np.asarray(smpl.faces), smv.smpl_joints24(smpl, v))["measurements"]
    rest_h = float(np.ptp((smpl.v_template + torch.einsum("vck,k->vc", smpl.shapedirs[:, :, :10],
                                                          torch.as_tensor(b, dtype=torch.float32))).numpy()[:, 1]))
    for site in ("chest", "abdomen", "hip"):
        assert float(g[site]) == pytest.approx(m[site]["tape_circumference"] / rest_h * 1750.0, rel=1e-4)


def test_girth_prior_reduces_inconsistency_when_images_are_silent():
    smpl = _smpl_cpu()
    start = np.array([1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    gp = _male_girth_prior(smpl, start, weight=1.0)
    before = float(smv.girth_prior_loss(smpl, torch.as_tensor(start, dtype=torch.float32), gp, "cpu"))
    V = 3
    views = smv.FitViews(keypoints=np.zeros((V, 17, 2)), weights=np.zeros((V, 17)),
                         global_orient=np.tile(np.eye(3), (V, 1, 1)), body_pose=np.tile(np.eye(3), (V, 23, 1, 1)),
                         cam_t=np.tile([0.0, 0.0, 5.0], (V, 1)), box_height=np.full(V, 500.0),
                         center=np.tile([250.0, 250.0], (V, 1)), focal=1000.0)
    cfg = smv.FitConfig(iterations=150, w_keypoints=0.0, w_beta_init=0.0, w_beta_prior=0.0, lr_beta=0.02, girth_prior=gp)
    r = smv.fit_multiview(smpl, views, start, cfg, device="cpu")
    after = float(smv.girth_prior_loss(smpl, torch.as_tensor(r.betas, dtype=torch.float32), gp, "cpu"))
    assert after < 0.5 * before
