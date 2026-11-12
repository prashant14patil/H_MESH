"""Experimental, model-agnostic shared-shape fitting primitives.

No model assets are loaded here. The caller supplies a differentiable body_model
accepting betas, global_orient and body_pose rotation matrices (pose2rot=False)
and returning an object with a ``joints`` tensor. Coordinates are original-image
pixels; rotations and translations are in the model's camera convention.

This is a feasibility tool, not a calibrated camera model or a medical device.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence
import math

import torch


class FitError(RuntimeError):
    """Invalid data or unsuccessful optimization."""


@dataclass
class Views:
    """Per-view data. Poses are fixed initial rotations; camera translations vary."""

    observations: torch.Tensor  # [views, joints, 2], image pixels
    weights: torch.Tensor  # [views, joints], nonnegative confidence
    global_orient: torch.Tensor  # [views, 1, 3, 3]
    body_pose: torch.Tensor  # [views, joints, 3, 3]
    camera_translation: torch.Tensor  # [views, 3], camera space
    focal_length: torch.Tensor  # [views, 2], pixels
    camera_center: torch.Tensor  # [views, 2], pixels
    image_size: torch.Tensor  # [views, 2], width, height in pixels


def _validate(views: Views, beta: torch.Tensor, indices: Sequence[int], min_views: int) -> None:
    if beta.ndim != 1 or beta.numel() == 0 or not torch.isfinite(beta).all():
        raise FitError("Invalid initial shape parameters")
    if views.observations.ndim != 3 or views.observations.shape[-1] != 2:
        raise FitError("Observations must have shape [views, joints, 2]")
    n, joints, _ = views.observations.shape
    if n < min_views or joints != len(indices) or not indices or any(i < 0 for i in indices):
        raise FitError("Insufficient views or invalid joint mapping")
    expected = {
        "weights": (n, joints), "global_orient": (n, 1, 3, 3),
        "camera_translation": (n, 3), "focal_length": (n, 2),
        "camera_center": (n, 2), "image_size": (n, 2),
    }
    for name, shape in expected.items():
        value = getattr(views, name)
        if value.shape != shape or not torch.isfinite(value).all():
            raise FitError(f"Invalid {name}")
    if (views.body_pose.ndim != 4 or views.body_pose.shape[0] != n
            or views.body_pose.shape[-2:] != (3, 3)
            or not torch.isfinite(views.body_pose).all()):
        raise FitError("Invalid body pose rotations")
    if (views.weights < 0).any() or (views.focal_length <= 0).any() or (views.image_size <= 0).any():
        raise FitError("Negative weights or invalid camera intrinsics")
    valid = torch.isfinite(views.observations).all(dim=-1) & (views.weights > 0)
    if ((valid.sum(dim=1) >= 4).sum().item() < min_views):
        raise FitError(f"At least {min_views} views with four valid landmarks are required")
    if (views.camera_translation[:, 2] <= 0).any():
        raise FitError("Camera depth must be positive")


def _residuals(body_model: Callable, views: Views, beta: torch.Tensor,
               camera: torch.Tensor, indices: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
    n = views.observations.shape[0]
    output = body_model(
        betas=beta.unsqueeze(0).expand(n, -1),
        global_orient=views.global_orient, body_pose=views.body_pose, pose2rot=False,
    )
    joints = output.joints
    if joints.ndim != 3 or joints.shape[0] != n or joints.shape[-1] != 3 or max(indices) >= joints.shape[1]:
        raise FitError("Body model returned incompatible joints")
    points = joints[:, list(indices), :] + camera[:, None, :]
    if not torch.isfinite(points).all() or (points[..., 2] <= 1e-4).any():
        raise FitError("Non-finite or nonpositive camera-space depth")
    projected = points[..., :2] / points[..., 2:] * views.focal_length[:, None, :] + views.camera_center[:, None, :]
    valid = torch.isfinite(views.observations).all(dim=-1) & (views.weights > 0)
    clean = torch.where(valid[..., None], views.observations, projected.detach())
    return projected - clean, torch.where(valid, views.weights, torch.zeros_like(views.weights))


def _objective(body_model: Callable, views: Views, beta: torch.Tensor, camera: torch.Tensor,
               indices: Sequence[int], beta_initial: torch.Tensor, camera_initial: torch.Tensor,
               shape_prior: float, camera_prior: float, huber_delta: float) -> torch.Tensor:
    residual, weights = _residuals(body_model, views, beta, camera, indices)
    # Normalize by image diagonal to avoid resolution-dependent regularization.
    diagonal = torch.linalg.vector_norm(views.image_size, dim=-1).clamp_min(1)
    norm = torch.linalg.vector_norm(residual, dim=-1) / diagonal[:, None]
    robust = torch.where(norm <= huber_delta, 0.5 * norm.square(),
                         huber_delta * (norm - 0.5 * huber_delta))
    per_view = (robust * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1e-8)
    active = (weights > 0).sum(dim=1) >= 4
    return (per_view[active].mean()
            + shape_prior * beta.square().mean()
            + camera_prior * (camera - camera_initial).square().mean())


def reprojection_metrics(body_model: Callable, views: Views, beta: torch.Tensor,
                         camera: torch.Tensor, indices: Sequence[int]) -> dict:
    """Report pixel and image-diagonal-normalized errors, excluding missing points."""
    _validate(views, beta, indices, 1)
    with torch.no_grad():
        residual, weights = _residuals(body_model, views, beta, camera, indices)
        errors = torch.linalg.vector_norm(residual, dim=-1)
        valid = weights > 0
        counts = valid.sum(dim=1)
        per_view = []
        for i in range(errors.shape[0]):
            if counts[i].item() == 0:
                per_view.append(None)
            else:
                mean = errors[i][valid[i]].mean()
                per_view.append({
                    "valid_landmarks": int(counts[i].item()),
                    "mean_px": float(mean.item()),
                    "mean_normalized": float((mean / torch.linalg.vector_norm(views.image_size[i])).item()),
                })
        return {
            "mean_px": float(errors[valid].mean().item()) if valid.any() else None,
            "per_view": per_view,
        }


def fit_shared_shape(body_model: Callable, views: Views, beta_initial: torch.Tensor,
                     indices: Sequence[int], *, steps: int = 200, learning_rate: float = 0.01,
                     shape_prior: float = 0.001, camera_prior: float = 0.01,
                     huber_delta: float = 0.03, min_improvement: float = 1e-8,
                     min_shape_change: float = 1e-6) -> dict:
    """Fit one beta across at least two views; fixed poses, regularized cameras.

    Reject no-op/non-finite fits rather than presenting initialization as success.
    Loss improvements are training diagnostics, not evidence of anatomical accuracy.
    """
    _validate(views, beta_initial, indices, 2)
    if (steps < 1 or learning_rate <= 0 or shape_prior < 0 or camera_prior < 0
            or huber_delta <= 0 or not all(math.isfinite(x) for x in
                (learning_rate, shape_prior, camera_prior, huber_delta, min_improvement, min_shape_change))
            or min_improvement < 0 or min_shape_change < 0):
        raise FitError("Invalid optimization configuration")
    initial = beta_initial.detach().clone()
    camera_initial = views.camera_translation.detach().clone()
    beta = torch.nn.Parameter(initial.clone())
    camera = torch.nn.Parameter(camera_initial.clone())
    optimizer = torch.optim.Adam([beta, camera], lr=learning_rate)

    def loss():
        return _objective(body_model, views, beta, camera, indices, initial, camera_initial,
                          shape_prior, camera_prior, huber_delta)

    with torch.no_grad():
        initial_loss = float(loss().item())
        initial_data_loss = float(_objective(
            body_model, views, beta, camera, indices, initial, camera_initial,
            0.0, 0.0, huber_delta,
        ).item())
    best_loss = initial_loss
    best_beta, best_camera = initial.clone(), camera_initial.clone()
    max_grad = 0.0
    for _ in range(steps):
        optimizer.zero_grad()
        value = loss()
        if not torch.isfinite(value):
            raise FitError("Optimization produced non-finite loss")
        value.backward()
        if beta.grad is None or not torch.isfinite(beta.grad).all() or camera.grad is None or not torch.isfinite(camera.grad).all():
            raise FitError("Missing or non-finite optimization gradients")
        max_grad = max(max_grad, float(beta.grad.norm().item()))
        optimizer.step()
        with torch.no_grad():
            current = float(loss().item())
            if not math.isfinite(current):
                raise FitError("Optimization produced non-finite loss")
            if current < best_loss:
                best_loss = current
                best_beta, best_camera = beta.detach().clone(), camera.detach().clone()
    shape_change = float(torch.linalg.vector_norm(best_beta - initial).item())
    with torch.no_grad():
        final_data_loss = float(_objective(
            body_model, views, best_beta, best_camera, indices, initial, camera_initial,
            0.0, 0.0, huber_delta,
        ).item())
    if (initial_loss - best_loss < min_improvement
            or initial_data_loss - final_data_loss < min_improvement
            or shape_change < min_shape_change or max_grad == 0):
        raise FitError("Optimization did not produce a measurable shared-shape and landmark-data improvement")
    return {
        "beta": best_beta,
        "camera_translation": best_camera,
        "initial_loss": initial_loss,
        "final_loss": best_loss,
        "initial_landmark_loss": initial_data_loss,
        "final_landmark_loss": final_data_loss,
        "shape_change_l2": shape_change,
        "max_shape_gradient": max_grad,
        "steps": steps,
        "status": "optimized",
    }


def fit_heldout_camera(body_model: Callable, views: Views, beta: torch.Tensor,
                       indices: Sequence[int], *, steps: int = 100,
                       learning_rate: float = 0.01, camera_prior: float = 0.01,
                       huber_delta: float = 0.03) -> torch.Tensor:
    """Fit only held-out camera translations; shape and pose stay frozen."""
    _validate(views, beta, indices, 1)
    if steps < 1 or learning_rate <= 0 or camera_prior < 0 or huber_delta <= 0:
        raise FitError("Invalid held-out optimization configuration")
    camera_initial = views.camera_translation.detach().clone()
    camera = torch.nn.Parameter(camera_initial.clone())
    optimizer = torch.optim.Adam([camera], lr=learning_rate)
    beta = beta.detach()
    best_loss = float("inf")
    best = camera_initial.clone()
    for _ in range(steps):
        optimizer.zero_grad()
        loss = _objective(body_model, views, beta, camera, indices, beta, camera_initial,
                          0.0, camera_prior, huber_delta)
        if not torch.isfinite(loss):
            raise FitError("Held-out camera optimization produced non-finite loss")
        if float(loss.item()) < best_loss:
            best_loss = float(loss.item())
            best = camera.detach().clone()
        loss.backward()
        if camera.grad is None or not torch.isfinite(camera.grad).all():
            raise FitError("Invalid held-out camera gradient")
        optimizer.step()
    return best
