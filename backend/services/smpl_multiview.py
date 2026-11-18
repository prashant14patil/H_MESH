"""Multi-view SMPL fitting, canonical mesh construction, validation and export.

Conventions (all verified against real HMR2 output, see tests):
* HMR2 works in a y-down camera frame: image x right, y down, z away from the
  camera. A person facing the camera has forward vector (0, 0, -1).
* Heading (yaw) = atan2(fwd_x, -fwd_z) in degrees: 0 = facing camera,
  +90 = facing image-right, +-180 = facing away.
* Canonical output mesh: SMPL frame, y up, +z forward, feet on y = 0.

The body model passed in must be a differentiable SMPL accepting
``betas, global_orient, body_pose, pose2rot=False`` and returning ``.vertices``
and ``.joints`` whose first 25 entries are OpenPose BODY_25 ordered (HMR2's
wrapper does this).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

# COCO-17 index -> OpenPose BODY_25 index.
COCO_TO_OPENPOSE = [0, 16, 15, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11]
OPENPOSE_PELVIS = 8
OPENPOSE_NOSE = 0


class FitError(RuntimeError):
    pass


# ------------------------------------------------------------ rotation math
def rot6d_to_matrix(x: torch.Tensor) -> torch.Tensor:
    """HMR2 6D convention: matrix columns 0 and 1, Gram-Schmidt orthonormalised."""
    x = x.reshape(-1, 3, 2)
    a1, a2 = x[:, :, 0], x[:, :, 1]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = torch.nn.functional.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def matrix_to_rot6d(m: torch.Tensor) -> torch.Tensor:
    return m[..., :, :2].reshape(*m.shape[:-2], 6)


def project_to_so3(m: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(m)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    return r


def geodesic_deg(a: np.ndarray, b: np.ndarray) -> float:
    cos = (np.trace(a.T @ b) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


def chordal_mean(rotations: np.ndarray, weights: np.ndarray | None = None,
                 outlier_deg: float = 45.0) -> np.ndarray:
    """Chordal L2 mean of rotation matrices with one outlier-rejection pass.

    This is a valid rotation average (SVD projection), not an element-wise
    average of matrices or vertices.
    """
    rotations = np.asarray(rotations, dtype=np.float64)
    w = np.ones(len(rotations)) if weights is None else np.asarray(weights, dtype=np.float64)
    mean = project_to_so3((rotations * w[:, None, None]).sum(0))
    keep = np.array([geodesic_deg(r, mean) <= outlier_deg for r in rotations])
    if keep.sum() >= max(2, len(rotations) // 2):
        mean = project_to_so3((rotations[keep] * w[keep, None, None]).sum(0))
    return mean


def heading_yaw_deg(global_orient: np.ndarray) -> float:
    """Body heading from an HMR2 global orientation matrix (see module docstring)."""
    fwd = np.asarray(global_orient, dtype=np.float64) @ np.array([0.0, 0.0, 1.0])
    return float(np.degrees(np.arctan2(fwd[0], -fwd[2])))


def wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def view_label(yaw_deg: float) -> str:
    """Label by which side of the body faces the camera (heading-derived)."""
    a = wrap_deg(yaw_deg)
    if abs(a) <= 35:
        return "front"
    if abs(a) >= 145:
        return "back"
    # Facing image-right (+yaw) shows the subject's right side to the camera.
    return "right_side" if a > 0 else "left_side"


# ---------------------------------------------------------- camera helpers
def project(points_cam: torch.Tensor, focal: float, center: torch.Tensor) -> torch.Tensor:
    """Pinhole projection. points_cam [V,J,3] already includes camera translation."""
    z = points_cam[..., 2].clamp(min=1e-3)
    u = focal * points_cam[..., 0] / z + center[:, None, 0]
    v = focal * points_cam[..., 1] / z + center[:, None, 1]
    return torch.stack([u, v], dim=-1)


def crop_cam_to_full(pred_cam, box_center, box_size, image_wh, focal_full):
    """HMR2 crop camera -> full-image camera translation (hmr2.utils.renderer)."""
    w2, h2 = image_wh[0] / 2.0, image_wh[1] / 2.0
    bs = float(box_size) * float(pred_cam[0]) + 1e-9
    tz = 2 * focal_full / bs
    tx = 2 * (box_center[0] - w2) / bs + float(pred_cam[1])
    ty = 2 * (box_center[1] - h2) / bs + float(pred_cam[2])
    return np.array([tx, ty, tz], dtype=np.float64)


# ------------------------------------------------------------------- fitting
@dataclass
class FitViews:
    keypoints: np.ndarray        # [V,17,2] detector pixels
    weights: np.ndarray          # [V,17]
    global_orient: np.ndarray    # [V,3,3] initial
    body_pose: np.ndarray        # [V,23,3,3] initial
    cam_t: np.ndarray            # [V,3] initial full-image translation
    box_height: np.ndarray       # [V] pixels
    center: np.ndarray           # [V,2] principal point
    focal: float
    contours: list | None = None  # optional per-view [N,2] silhouette contour points
    dist_maps: list | None = None  # optional per-view (outside-distance map, (ox, oy, s))
    contour_weights: list | None = None  # optional per-view [N] weights for contour points
    sil_weight: np.ndarray | None = None  # optional per-view silhouette reliability (0 = excluded)


@dataclass
class FitConfig:
    iterations: int = 300
    lr_pose: float = 0.01
    lr_beta: float = 0.02
    lr_cam: float = 0.01
    sigma_frac: float = 0.02      # keypoint noise scale as a fraction of box height
    w_keypoints: float = 1.0
    w_pose_prior: float = 4.0     # keep per-view pose near HMR2's estimate
    w_beta_prior: float = 0.002   # shrink betas toward the mean body
    w_beta_init: float = 0.02     # weak pull to the multi-frame median
    w_depth: float = 2.0          # people rotating in place keep ~constant distance
    w_silhouette: float = 0.0
    sil_tau_frac: float = 0.01    # silhouette robust scale as a fraction of box height
    beta_sigma: np.ndarray | None = None  # per-coefficient scale for the pull to betas_init (Mahalanobis)
    seed: int = 0


@dataclass
class FitResult:
    betas: np.ndarray
    global_orient: np.ndarray
    body_pose: np.ndarray
    cam_t: np.ndarray
    loss_history: list = field(default_factory=list)
    residual_px_before: np.ndarray | None = None
    residual_px_after: np.ndarray | None = None


def _smpl_joints(smpl, betas, go, pose):
    out = smpl(betas=betas, global_orient=go[:, None], body_pose=pose, pose2rot=False)
    return out.vertices, out.joints


def keypoint_residuals(smpl, views: FitViews, betas, go, pose, cam_t, device) -> torch.Tensor:
    """Per-view, per-joint pixel residual norm [V,17] for given parameters."""
    V = len(views.keypoints)
    b = torch.as_tensor(betas, dtype=torch.float32, device=device).expand(V, -1)
    _, joints = _smpl_joints(smpl, b, torch.as_tensor(go, dtype=torch.float32, device=device),
                             torch.as_tensor(pose, dtype=torch.float32, device=device))
    cam = torch.as_tensor(cam_t, dtype=torch.float32, device=device)
    pts = joints[:, COCO_TO_OPENPOSE] + cam[:, None, :]
    uv = project(pts, views.focal, torch.as_tensor(views.center, dtype=torch.float32, device=device))
    obs = torch.as_tensor(views.keypoints, dtype=torch.float32, device=device)
    return (uv - obs).norm(dim=-1)


def _robust(x):
    """Geman-McClure: quadratic near 0, saturates at 1 so bad mask pixels cannot dominate."""
    return x ** 2 / (1 + x ** 2)


class _SilhouetteTargets:
    """Per-view silhouette tensors, moved to the device once per fit."""

    def __init__(self, views: FitViews, device):
        f32 = dict(dtype=torch.float32, device=device)
        self.items = []
        n = len(views.contours or [])
        for i in range(n):
            contour = views.contours[i]
            weight = 1.0 if views.sil_weight is None else float(views.sil_weight[i])
            if contour is None or len(contour) == 0 or weight <= 0:
                self.items.append(None)
                continue
            cw = (views.contour_weights[i] if views.contour_weights is not None else np.ones(len(contour)))
            dist = (views.dist_maps or [None] * n)[i]
            dm = None
            if dist is not None:
                arr, origin = dist
                dm = (torch.as_tensor(np.asarray(arr), **f32), origin)
            self.items.append({"contour": torch.as_tensor(contour, **f32), "cw": torch.as_tensor(cw, **f32),
                               "dist": dm, "weight": weight})


def _silhouette_loss(vertices, cam, views: FitViews, device, tau_frac: float = 0.01, targets=None):
    """Symmetric silhouette agreement using projected SMPL vertices.

    * outside term: projected vertices that fall outside the observed mask are penalised by
      their distance to the mask (from a distance transform) -> the body may not be too big;
    * coverage term: every observed contour point must have a projected vertex nearby
      (chamfer) -> the body may not be too small.

    Both are robust (Geman-McClure) at a scale of ``tau_frac`` x box height. Without a
    differentiable rasteriser this vertex-based form is a practical surrogate of silhouette
    IoU; the exact rasterised IoU is reported separately by :func:`silhouette_metrics`.
    """
    if not views.contours:
        return vertices.new_zeros(())
    targets = targets or _SilhouetteTargets(views, device)
    total = vertices.new_zeros(())
    pts = vertices + cam[:, None, :]
    uv = project(pts, views.focal, torch.as_tensor(views.center, dtype=torch.float32, device=device))
    weight_sum = 0.0
    for i, item in enumerate(targets.items):
        if item is None:
            continue
        tau = tau_frac * float(views.box_height[i])
        d = torch.cdist(item["contour"], uv[i]).min(dim=1).values
        loss = (item["cw"] * _robust(d / tau)).sum() / item["cw"].sum().clamp(min=1e-6)
        if item["dist"] is not None:
            dm, (ox, oy, s) = item["dist"]
            h, w = dm.shape
            gx = (uv[i, :, 0] - ox) / s / (w - 1) * 2 - 1
            gy = (uv[i, :, 1] - oy) / s / (h - 1) * 2 - 1
            grid = torch.stack([gx, gy], dim=-1)[None, None]
            outside = torch.nn.functional.grid_sample(dm[None, None], grid, align_corners=True,
                                                      padding_mode="border")[0, 0, 0] * s
            loss = loss + _robust(outside / tau).mean()
        total = total + item["weight"] * loss
        weight_sum += item["weight"]
    return total / max(weight_sum, 1e-6)


def fit_multiview(smpl, views: FitViews, betas_init: np.ndarray, cfg: FitConfig | None = None,
                  device: str = "cuda", fit_shape: bool = True) -> FitResult:
    """Shared shape + per-view pose/camera optimisation against 2D keypoints.

    * ONE beta vector shared by every view (``fit_shape``); per-view
      ``global_orient``/``body_pose``/camera translation are free but anchored
      to HMR2's estimate by a prior.
    * Camera focal length and principal point are shared and fixed (HMR2's
      assumed intrinsics); the true camera is unknown.
    """
    cfg = cfg or FitConfig()
    V = len(views.keypoints)
    if V < 3:
        raise FitError("At least 3 views are required for shared-shape optimisation")
    for name in ("keypoints", "weights", "global_orient", "body_pose", "cam_t"):
        if not np.isfinite(getattr(views, name)).all():
            raise FitError(f"Non-finite values in {name}")
    torch.manual_seed(cfg.seed)
    f32 = dict(dtype=torch.float32, device=device)
    obs = torch.as_tensor(views.keypoints, **f32)
    w = torch.as_tensor(views.weights, **f32)
    sigma = torch.as_tensor(views.box_height, **f32)[:, None] * cfg.sigma_frac
    center = torch.as_tensor(views.center, **f32)
    go0 = torch.as_tensor(views.global_orient, **f32)
    pose0 = torch.as_tensor(views.body_pose, **f32)
    cam0 = torch.as_tensor(views.cam_t, **f32)
    beta0 = torch.as_tensor(betas_init, **f32)

    beta = beta0.clone().requires_grad_(fit_shape)
    go6 = matrix_to_rot6d(go0).clone().requires_grad_(True)
    pose6 = matrix_to_rot6d(pose0).clone().requires_grad_(True)
    cam = cam0.clone().requires_grad_(True)
    groups = [{"params": [go6, pose6], "lr": cfg.lr_pose}, {"params": [cam], "lr": cfg.lr_cam}]
    if fit_shape:
        groups.append({"params": [beta], "lr": cfg.lr_beta})
    opt = torch.optim.Adam(groups)
    cam_z_ref = cam0[:, 2].median()
    beta_sigma = (torch.as_tensor(cfg.beta_sigma, **f32) if cfg.beta_sigma is not None
                  else torch.ones_like(beta0))
    use_sil = cfg.w_silhouette > 0 and bool(views.contours)
    sil_targets = _SilhouetteTargets(views, device) if use_sil else None

    def forward():
        go = rot6d_to_matrix(go6.reshape(-1, 6))
        pose = rot6d_to_matrix(pose6.reshape(-1, 6)).reshape(V, 23, 3, 3)
        verts, joints = _smpl_joints(smpl, beta.expand(V, -1), go, pose)
        return go, pose, verts, joints

    history = []
    with torch.no_grad():
        _, _, _, j0 = forward()
        before = (project(j0[:, COCO_TO_OPENPOSE] + cam[:, None, :], views.focal, center) - obs).norm(dim=-1)

    for it in range(cfg.iterations):
        opt.zero_grad()
        go, pose, verts, joints = forward()
        uv = project(joints[:, COCO_TO_OPENPOSE] + cam[:, None, :], views.focal, center)
        r = (uv - obs).norm(dim=-1) / sigma
        robust = r ** 2 / (1 + r ** 2)  # Geman-McClure
        l_kp = (w * robust).sum() / w.sum().clamp(min=1e-6)
        l_pose = ((go - go0) ** 2).sum((-1, -2)).mean() + ((pose - pose0) ** 2).sum((-1, -2)).mean()
        l_beta = (beta ** 2).mean()
        l_beta_init = (((beta - beta0) / beta_sigma) ** 2).mean()
        l_depth = (((cam[:, 2] - cam_z_ref) / cam_z_ref) ** 2).mean()
        loss = (cfg.w_keypoints * l_kp + cfg.w_pose_prior * l_pose + cfg.w_beta_prior * l_beta
                + cfg.w_beta_init * l_beta_init + cfg.w_depth * l_depth)
        l_sil = None
        if use_sil:
            l_sil = _silhouette_loss(verts, cam, views, device, cfg.sil_tau_frac, sil_targets)
            loss = loss + cfg.w_silhouette * l_sil
        if not torch.isfinite(loss):
            raise FitError("Optimisation diverged (non-finite loss)")
        loss.backward()
        opt.step()
        if it % 25 == 0 or it == cfg.iterations - 1:
            history.append({"iteration": it, "loss": float(loss), "keypoint": float(l_kp),
                            "pose_prior": float(l_pose), "depth": float(l_depth),
                            "beta_prior": float(l_beta_init),
                            "silhouette": float(l_sil) if l_sil is not None else None})

    with torch.no_grad():
        go, pose, _, joints = forward()
        after = (project(joints[:, COCO_TO_OPENPOSE] + cam[:, None, :], views.focal, center) - obs).norm(dim=-1)
    return FitResult(betas=beta.detach().cpu().numpy(), global_orient=go.detach().cpu().numpy(),
                     body_pose=pose.detach().cpu().numpy(), cam_t=cam.detach().cpu().numpy(),
                     loss_history=history, residual_px_before=before.cpu().numpy(),
                     residual_px_after=after.cpu().numpy())


# ------------------------------------------------------- shape aggregation
def robust_beta_aggregate(betas_each: np.ndarray, weights: np.ndarray | None = None, z_max: float = 6.0):
    """Weighted per-coefficient median of per-frame HMR2 betas after rejecting outlier frames.

    A frame is an outlier when its beta vector's distance from the median exceeds ``z_max``
    robust standard deviations (MAD of those distances). Returns (aggregate, robust_std, outlier_idx).
    The robust std (per coefficient, floored) is used as the Mahalanobis scale of the shape prior.
    """
    b = np.asarray(betas_each, dtype=np.float64)
    w = np.ones(len(b)) if weights is None else np.asarray(weights, dtype=np.float64)
    med = np.median(b, axis=0)
    dist = np.linalg.norm(b - med, axis=1)
    mad = np.median(np.abs(dist - np.median(dist))) * 1.4826 + 1e-9
    z = (dist - np.median(dist)) / mad
    outliers = np.where(z > z_max)[0]
    keep = np.setdiff1d(np.arange(len(b)), outliers)
    if len(keep) < max(3, len(b) // 2):
        keep, outliers = np.arange(len(b)), np.array([], int)
    agg = np.array([_weighted_median(b[keep, k], w[keep]) for k in range(b.shape[1])])
    robust_std = 1.4826 * np.median(np.abs(b[keep] - agg), axis=0)
    return agg, robust_std, outliers


def _weighted_median(values, weights):
    order = np.argsort(values)
    v, w = np.asarray(values)[order], np.asarray(weights)[order]
    c = np.cumsum(w)
    return float(v[np.searchsorted(c, 0.5 * c[-1])])


# ---------------------------------------------------- posing + projection
def measurement_pose(shoulder_down_deg: float = 45.0, hip_abduct_deg: float = 5.0) -> np.ndarray:
    """Standard A-pose used for the final mesh and circumference analysis.

    SMPL rest pose with arms lowered ``shoulder_down_deg`` from horizontal and legs abducted
    slightly, so arms do not touch the torso and thighs do not touch each other: horizontal
    torso sections and limb sections then separate cleanly. Body shape (betas) is defined in
    SMPL's rest pose, so this pose shows the shared shape with minimal pose deformation.
    """
    pose = np.tile(np.eye(3), (23, 1, 1))

    def rz(deg):
        a = math.radians(deg)
        return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    # body_pose index = SMPL joint - 1: L_hip 1, R_hip 2, L_shoulder 16, R_shoulder 17.
    pose[0], pose[1] = rz(hip_abduct_deg), rz(-hip_abduct_deg)
    pose[15], pose[16] = rz(-shoulder_down_deg), rz(shoulder_down_deg)
    return pose


def posed_output(smpl, betas, global_orient, body_pose, device):
    f32 = dict(dtype=torch.float32, device=device)
    go = torch.as_tensor(np.asarray(global_orient), **f32).reshape(-1, 3, 3)
    pose = torch.as_tensor(np.asarray(body_pose), **f32).reshape(len(go), 23, 3, 3)
    b = torch.as_tensor(np.asarray(betas), **f32).reshape(1, -1).expand(len(go), -1)
    with torch.no_grad():
        out = smpl(betas=b, global_orient=go[:, None], body_pose=pose, pose2rot=False)
    return out


def smpl_joints24(smpl, vertices: np.ndarray) -> np.ndarray:
    """SMPL's native 24 joints (J_regressor) for given vertices [6890,3]."""
    reg = smpl.J_regressor.detach().cpu().numpy().astype(np.float64)
    return reg @ np.asarray(vertices, dtype=np.float64)


def project_np(points_cam: np.ndarray, focal: float, center) -> np.ndarray:
    z = np.clip(points_cam[:, 2], 1e-3, None)
    return np.stack([focal * points_cam[:, 0] / z + center[0], focal * points_cam[:, 1] / z + center[1]], 1)


def rasterize_silhouette(uv: np.ndarray, faces: np.ndarray, shape_hw, scale: float = 1.0) -> np.ndarray:
    """Exact (non-differentiable) silhouette of a projected triangle mesh."""
    import cv2
    h, w = shape_hw
    out = np.zeros((int(round(h * scale)), int(round(w * scale))), np.uint8)
    pts = np.round(uv * scale * 4).astype(np.int32)  # 2 bits of sub-pixel precision
    for tri in pts[np.asarray(faces)]:
        cv2.fillConvexPoly(out, tri, 1, lineType=cv2.LINE_8, shift=2)
    return out


def silhouette_metrics(pred: np.ndarray, obs: np.ndarray, box_height: float) -> dict:
    """IoU plus symmetric contour distance (as % of subject box height) between two masks."""
    import cv2
    p, o = pred.astype(bool), obs.astype(bool)
    union = np.logical_or(p, o).sum()
    iou = float(np.logical_and(p, o).sum() / union) if union else 0.0

    def contour_dist(a, b):
        cnt, _ = cv2.findContours(a.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        if not cnt:
            return None
        pts = np.concatenate([c[:, 0, :] for c in cnt], 0)
        edge = np.ones(b.shape, np.uint8)
        cb, _ = cv2.findContours(b.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        if not cb:
            return None
        cv2.drawContours(edge, cb, -1, 0, 1)
        dt = cv2.distanceTransform(edge, cv2.DIST_L2, 5)
        return dt[np.clip(pts[:, 1], 0, b.shape[0] - 1), np.clip(pts[:, 0], 0, b.shape[1] - 1)]
    d1, d2 = contour_dist(p, o), contour_dist(o, p)
    if d1 is None or d2 is None:
        return {"iou": iou, "contour_mean_pct_height": None, "contour_p90_pct_height": None,
                "area_ratio_pred_over_obs": float(p.sum() / max(o.sum(), 1))}
    d = np.concatenate([d1, d2])
    return {"iou": iou, "contour_mean_pct_height": float(d.mean() / box_height * 100),
            "contour_p90_pct_height": float(np.percentile(d, 90) / box_height * 100),
            "area_ratio_pred_over_obs": float(p.sum() / max(o.sum(), 1))}


# ------------------------------------------------------------ canonical mesh
def canonical_mesh(smpl, betas: np.ndarray, body_poses: np.ndarray, weights: np.ndarray | None,
                   device: str = "cuda", pose: str = "mean_observed"):
    """Build the single unified mesh.

    Shared betas, identity global orientation (SMPL canonical: upright, +z forward), feet on
    the ground plane. ``pose='mean_observed'`` uses the robust chordal mean of the per-view
    body poses; ``pose='measurement'`` uses :func:`measurement_pose` (standard A-pose).
    Returns (vertices[6890,3], joints, pose_used[23,3,3], pose_spread_deg[23]).
    """
    mean_pose = np.zeros((23, 3, 3))
    spread = np.zeros(23)
    for j in range(23):
        mean_pose[j] = chordal_mean(body_poses[:, j], weights)
        spread[j] = float(np.mean([geodesic_deg(r, mean_pose[j]) for r in body_poses[:, j]]))
    used = measurement_pose() if pose == "measurement" else mean_pose
    f32 = dict(dtype=torch.float32, device=device)
    with torch.no_grad():
        out = smpl(betas=torch.as_tensor(betas, **f32)[None],
                   global_orient=torch.eye(3, **f32)[None, None],
                   body_pose=torch.as_tensor(used, **f32)[None], pose2rot=False)
    verts = out.vertices[0].cpu().numpy().astype(np.float64)
    joints = out.joints[0].cpu().numpy().astype(np.float64)
    shift = np.array([(verts[:, 0].min() + verts[:, 0].max()) / 2, verts[:, 1].min(),
                      (verts[:, 2].min() + verts[:, 2].max()) / 2])
    return verts - shift, joints - shift, used, spread


# ------------------------------------------------------- validation + export
def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = vertices[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    vn = np.zeros_like(vertices)
    for k in range(3):
        np.add.at(vn, faces[:, k], fn)
    norm = np.linalg.norm(vn, axis=1, keepdims=True)
    return vn / np.maximum(norm, 1e-12)


def validate_mesh(vertices: np.ndarray, faces: np.ndarray, joints: np.ndarray | None = None) -> dict:
    """Objective geometry/topology checks; every field is measured, not assumed."""
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces)
    report: dict = {"vertex_count": int(len(v)), "face_count": int(len(f))}
    report["all_vertices_finite"] = bool(np.isfinite(v).all())
    report["faces_in_range"] = bool(f.size and f.min() >= 0 and f.max() < len(v))
    report["no_degenerate_faces"] = False
    ok = report["all_vertices_finite"] and report["faces_in_range"]
    if ok:
        tri = v[f]
        area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        report["degenerate_face_count"] = int((area < 1e-12).sum())
        report["no_degenerate_faces"] = report["degenerate_face_count"] == 0
        report["surface_area_m2_if_metric"] = float(area.sum())
        ext = v.max(0) - v.min(0)
        report["extent_xyz"] = ext.tolist()
        report["upright"] = bool(ext[1] > ext[0] and ext[1] > ext[2]
                                 and abs(v[:, 1].min()) < 1e-6)
        if joints is not None:
            head_y = float(joints[OPENPOSE_NOSE, 1])
            pelvis_y = float(joints[OPENPOSE_PELVIS, 1])
            report["head_above_pelvis"] = bool(head_y > pelvis_y > 0)
            report["facing_plus_z"] = bool(joints[OPENPOSE_NOSE, 2] > joints[OPENPOSE_PELVIS, 2])
        try:
            import trimesh
            mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
            report["winding_consistent"] = bool(mesh.is_winding_consistent)
            report["watertight"] = bool(mesh.is_watertight)
            report["connected_components"] = int(len(mesh.split(only_watertight=False)))
            report["euler_number"] = int(mesh.euler_number)
            report["volume_positive"] = bool(mesh.volume > 0) if mesh.is_watertight else None
            report["self_intersection_probe"] = _self_intersection_probe(mesh)
        except Exception as exc:  # noqa: BLE001
            report["topology_check_error"] = str(exc)
    report["valid"] = bool(
        ok and report["no_degenerate_faces"] and report.get("winding_consistent") is True
        and report.get("connected_components") == 1 and report.get("upright") is True
        and report.get("volume_positive") is True
        and report.get("head_above_pelvis") is True
        and report.get("self_intersection_probe", {}).get("severe") is False)
    return report


def _self_intersection_probe(mesh) -> dict:
    """Proxy test: points just inside the surface must remain inside the mesh.

    A vertex pushed 3 mm inward along its normal that falls OUTSIDE the closed
    mesh sits behind another surface, i.e. a penetration. Thin structures
    (ears, fingertips) legitimately fail, so only a large fraction is 'severe'.
    This is a probe, not an exact triangle-triangle intersection test.
    """
    if not mesh.is_watertight:
        return {"evaluated": False, "reason": "mesh not watertight", "severe": None}
    normals = mesh.vertex_normals
    probe = mesh.vertices - 0.003 * normals
    inside = mesh.contains(probe)
    outside_fraction = float(1.0 - inside.mean())
    return {"evaluated": True, "outside_fraction": outside_fraction, "severe": bool(outside_fraction > 0.05),
            "method": "inward-offset containment probe (not exact triangle intersection)"}


def write_obj(path, vertices, faces, header: str = "") -> None:
    normals = vertex_normals(np.asarray(vertices, dtype=np.float64), np.asarray(faces))
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for line in header.splitlines():
            fh.write(f"# {line}\n")
        for x, y, z in vertices:
            fh.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for x, y, z in normals:
            fh.write(f"vn {x:.6f} {y:.6f} {z:.6f}\n")
        for a, b, c in (np.asarray(faces) + 1):
            fh.write(f"f {a}//{a} {b}//{b} {c}//{c}\n")


def write_ply(path, vertices, faces) -> None:
    v = np.asarray(vertices, dtype="<f4")
    f = np.asarray(faces, dtype="<i4")
    with open(path, "wb") as fh:
        fh.write((f"ply\nformat binary_little_endian 1.0\ncomment SMPL unified body mesh\n"
                  f"element vertex {len(v)}\nproperty float x\nproperty float y\nproperty float z\n"
                  f"element face {len(f)}\nproperty list uchar int vertex_indices\nend_header\n").encode())
        fh.write(v.tobytes())
        face_dtype = np.dtype([("n", "u1"), ("i", "<i4", (3,))])
        arr = np.empty(len(f), dtype=face_dtype)
        arr["n"] = 3
        arr["i"] = f
        fh.write(arr.tobytes())


def read_obj(path):
    verts, faces = [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("v "):
                verts.append([float(t) for t in line.split()[1:4]])
            elif line.startswith("f "):
                faces.append([int(t.split("/")[0]) - 1 for t in line.split()[1:4]])
    return np.asarray(verts), np.asarray(faces)
