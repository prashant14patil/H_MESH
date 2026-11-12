"""Synthetic multi-view geometry and frame-selection tests."""
import unittest

import numpy as np

from backend.services.multiview_geometry import project_views, reprojection_errors, select_frames, triangulate
from scripts.run_synthetic_multiview import POINTS, scene


class MultiViewTests(unittest.TestCase):
    def setUp(self):
        self.cameras = scene()
        self.obs = project_views(self.cameras, POINTS)

    def test_exact_reconstruction_and_heldout_view(self):
        result = triangulate(self.cameras[:3], self.obs[:3])
        np.testing.assert_allclose(result["points"], POINTS, atol=1e-9)
        errors = reprojection_errors(self.cameras, result["points"], self.obs)
        self.assertLess(float(np.nanmax(errors[3])), 1e-8)
        self.assertTrue(result["inliers"].all())

    def test_noise_and_missing_confidence(self):
        rng = np.random.default_rng(42)
        noisy = self.obs + rng.normal(0, 1.0, self.obs.shape)
        confidence = np.ones(self.obs.shape[:2])
        noisy[0, 0] = np.nan
        confidence[1, 1] = 0.01
        confidence[0:3, 2] = 0
        result = triangulate(self.cameras, noisy, confidence)
        self.assertTrue(np.isnan(result["points"][2]).all())
        self.assertEqual(result["reasons"][2], "insufficient_observations")
        valid = np.isfinite(result["points"]).all(axis=1)
        self.assertLess(np.mean(np.linalg.norm(result["points"][valid] - POINTS[valid], axis=1)), .03)
        self.assertFalse(result["inliers"][0, 0])
        self.assertFalse(result["inliers"][1, 1])

    def test_rejects_obvious_outlier(self):
        obs = self.obs.copy()
        obs[1, 4] += [100, -120]
        result = triangulate(self.cameras, obs, outlier_threshold_px=3)
        self.assertFalse(result["inliers"][1, 4])
        np.testing.assert_allclose(result["points"][4], POINTS[4], atol=1e-8)

    def test_degenerate_and_invalid_input(self):
        result = triangulate([self.cameras[0], self.cameras[0]], self.obs[[0, 0]])
        self.assertTrue(np.isnan(result["points"]).all())
        self.assertTrue(all(reason == "degenerate_or_behind_camera" for reason in result["reasons"]))
        with self.assertRaises(ValueError):
            triangulate(self.cameras, self.obs, np.full(self.obs.shape[:2], 1.1))

    def test_manual_selection_preserves_blur_and_counts(self):
        metadata = {"frames": [
            {"filename": "a.jpg", "is_blurry": True, "blur_score": 12, "timestamp_sec": 1},
            {"filename": "b.jpg", "is_blurry": False, "blur_score": 200, "timestamp_sec": 2},
            {"filename": "c.jpg", "is_blurry": True, "blur_score": 14, "timestamp_sec": 3},
        ]}
        default = select_frames(metadata, ["a.jpg", "b.jpg"])
        self.assertEqual(default["counts"], {"accepted": 1, "excluded": 2, "manually_selected": 2, "blurry_accepted": 0})
        override = select_frames(metadata, ["a.jpg", "b.jpg"], allow_blurry=True)
        self.assertEqual(override["counts"]["blurry_accepted"], 1)
        self.assertTrue(override["accepted"][0]["is_blurry"])
        self.assertEqual(override["accepted"][0]["quality_warning"], "blurry")
        self.assertEqual(metadata["frames"][0]["blur_score"], 12)
        self.assertNotIn("quality_warning", metadata["frames"][0])
        with self.assertRaises(ValueError):
            select_frames(metadata, ["missing.jpg"])


if __name__ == "__main__":
    unittest.main()
