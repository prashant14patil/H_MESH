"""Synthetic shared-shape tests; no licensed body models or data are used."""
import math
import unittest
from types import SimpleNamespace

import torch

from backend.services.shared_shape_fitter import (
    FitError, Views, _residuals, fit_heldout_camera, fit_shared_shape, reprojection_metrics,
)


class SyntheticBody:
    """Shape affects width and height; global rotations produce distinct views."""

    def __init__(self):
        self.template = torch.tensor([
            [-0.3, -0.8, 0.0], [0.3, -0.8, 0.0],
            [-0.4, -0.3, 0.1], [0.4, -0.3, -0.1],
            [-0.2, 0.4, 0.1], [0.2, 0.4, -0.1],
            [-0.2, 0.9, 0.0], [0.2, 0.9, 0.0],
        ])

    def __call__(self, *, betas, global_orient, body_pose, pose2rot):
        points = self.template.to(betas).unsqueeze(0).expand(betas.shape[0], -1, -1)
        scale = torch.stack((1 + betas[:, 0], 1 + betas[:, 1],
                             torch.ones_like(betas[:, 0])), dim=-1)
        points = points * scale[:, None, :]
        return SimpleNamespace(joints=torch.einsum("bij,bkj->bki", global_orient[:, 0], points))


def make_views(angles, true_beta=(0.3, -0.15)):
    n = len(angles)
    rotations = torch.tensor([
        [[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]]
        for a in angles
    ], dtype=torch.float32).unsqueeze(1)
    pose = torch.eye(3).reshape(1, 1, 3, 3).repeat(n, 1, 1, 1)
    camera = torch.tensor([[0.0, 0.0, 4.0]]).repeat(n, 1)
    focal = torch.tensor([[500.0, 500.0]]).repeat(n, 1)
    center = torch.tensor([[320.0, 240.0]]).repeat(n, 1)
    size = torch.tensor([[640.0, 480.0]]).repeat(n, 1)
    model = SyntheticBody()
    points = model(betas=torch.tensor(true_beta).repeat(n, 1),
                   global_orient=rotations, body_pose=pose, pose2rot=False).joints
    points = points + camera[:, None, :]
    obs = points[..., :2] / points[..., 2:] * focal[:, None, :] + center[:, None, :]
    return Views(obs, torch.ones(n, 8), rotations, pose, camera, focal, center, size)


class SharedShapeTests(unittest.TestCase):
    def setUp(self):
        self.model = SyntheticBody()
        self.indices = list(range(8))
        self.beta = torch.zeros(2)

    def test_shared_fit_improves_heldout_reprojection(self):
        training = make_views([0.0, 0.8, 1.5])
        heldout = make_views([2.3])
        result = fit_shared_shape(self.model, training, self.beta, self.indices,
                                  steps=350, learning_rate=0.02, shape_prior=1e-6,
                                  camera_prior=0.5)
        self.assertEqual(result["status"], "optimized")
        self.assertLess(result["final_loss"], result["initial_loss"])
        self.assertGreater(result["shape_change_l2"], 1e-4)
        baseline_cam = fit_heldout_camera(self.model, heldout, self.beta, self.indices,
                                          camera_prior=0.5)
        fitted_cam = fit_heldout_camera(self.model, heldout, result["beta"], self.indices,
                                        camera_prior=0.5)
        before = reprojection_metrics(self.model, heldout, self.beta, baseline_cam, self.indices)
        after = reprojection_metrics(self.model, heldout, result["beta"], fitted_cam, self.indices)
        self.assertLess(after["mean_px"], before["mean_px"])

    def test_repeatability_across_synthetic_view_selections(self):
        estimates = []
        for angles in ([0.0, 0.8, 1.5], [0.3, 1.1, 2.0]):
            result = fit_shared_shape(self.model, make_views(angles), self.beta, self.indices,
                                      steps=350, learning_rate=0.02, shape_prior=1e-6,
                                      camera_prior=0.5)
            estimates.append(result["beta"])
        self.assertLess(float(torch.linalg.vector_norm(estimates[0] - estimates[1])), 0.02)

    def test_each_view_contributes_to_shared_beta_gradient(self):
        views = make_views([0.0, 0.8, 1.5])
        beta = torch.zeros(2, requires_grad=True)
        residual, weights = _residuals(
            self.model, views, beta, views.camera_translation, self.indices,
        )
        for view in range(3):
            view_loss = (residual[view].square() * weights[view, :, None]).mean()
            gradient = torch.autograd.grad(view_loss, beta, retain_graph=True)[0]
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.norm()), 1e-6)

    def test_missing_and_low_confidence_points(self):
        views = make_views([0.0, 1.0])
        views.observations[0, 0] = float("nan")
        views.weights[1, 1] = 0
        result = fit_shared_shape(self.model, views, self.beta, self.indices,
                                  steps=80, shape_prior=1e-6)
        self.assertTrue(torch.isfinite(result["beta"]).all())

    def test_insufficient_observations_and_views_fail(self):
        views = make_views([0.0, 1.0])
        views.weights[1] = 0
        with self.assertRaises(FitError):
            fit_shared_shape(self.model, views, self.beta, self.indices)
        with self.assertRaises(FitError):
            fit_shared_shape(self.model, make_views([0.0]), self.beta, self.indices)

    def test_invalid_initialization_and_noop_fail(self):
        views = make_views([0.0, 1.0])
        with self.assertRaises(FitError):
            fit_shared_shape(self.model, views, torch.tensor([float("nan"), 0]), self.indices)
        with self.assertRaises(FitError):
            fit_shared_shape(self.model, views, self.beta, self.indices, steps=1,
                             min_improvement=1e6)

    def test_invalid_camera_depth_fails(self):
        views = make_views([0.0, 1.0])
        views.camera_translation[:, 2] = -1
        with self.assertRaises(FitError):
            fit_shared_shape(self.model, views, self.beta, self.indices)


if __name__ == "__main__":
    unittest.main()
