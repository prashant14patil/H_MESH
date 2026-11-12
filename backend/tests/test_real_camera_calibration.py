"""End-to-end tests with generated checkerboards; no real camera or model assets."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from backend.services.checkerboard_calibration import CalibrationFailure, calibrate_observations
from backend.services.real_camera_calibration import run_calibration
from backend.tests.test_fixed_camera_validation import synthetic_corners


def render_images(folder):
    board = np.zeros((560, 800), np.uint8)
    for row in range(7):
        for col in range(10):
            if (row + col) % 2 == 0:
                board[row*80:(row+1)*80, col*80:(col+1)*80] = 255
    source = np.float32([[0, 0], [799, 0], [799, 559], [0, 559]])
    k = np.array([[800., 0., 320.], [0., 810., 240.], [0., 0., 1.]])
    outer = np.float32([[-.025, -.025, 0], [.225, -.025, 0], [.225, .15, 0], [-.025, .15, 0]])
    for i in range(10):
        rotation = np.array([-.22 + .05*i, .16*np.sin(i), .05*np.cos(i)])
        translation = np.array([-.14 + .025*i, -.07 + .015*(i % 3), .9 + .04*(i % 4)])
        target, _ = cv2.projectPoints(outer, rotation, translation, k, np.zeros(5))
        h = cv2.getPerspectiveTransform(source, target.reshape(4, 2))
        image = cv2.warpPerspective(board, h, (640, 480), borderValue=127)
        suffix = (".png", ".jpg", ".jpeg")[i % 3]
        if not cv2.imwrite(str(folder / f"board_{i:02d}{suffix}"), image):
            raise RuntimeError("Synthetic image write failed")


class RealCameraWorkflowTests(unittest.TestCase):
    def test_folder_success_outputs_and_exclusive_run(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            source = base / "images"
            source.mkdir()
            render_images(source)
            (source / "bad.jpg").write_bytes(b"not an image")
            cv2.imwrite(str(source / "blank.png"), np.full((480, 640), 127, np.uint8))
            output = run_calibration(source, 9, 6, .025, output_root=base / "runs",
                                     run_id="test", provenance="synthetic",
                                     high_image_error_px=1e-10)
            config = json.loads((output / "calibration.json").read_text(encoding="utf-8"))
            results = json.loads((output / "image_results.json").read_text(encoding="utf-8"))
            self.assertEqual(config["provenance"], "synthetic")
            self.assertEqual(config["view_count"], 10)
            self.assertFalse(config["quality_assessment"]["independent_validation"])
            records = {r["filename"]: r for r in results["images"]}
            self.assertEqual(records["bad.jpg"]["reason"], "unreadable")
            self.assertEqual(records["blank.png"]["reason"], "checkerboard_not_detected")
            used = [r for r in records.values() if r["detection"] == "detected"]
            self.assertEqual(len(used), len(config["per_view_reprojection_px"]))
            self.assertTrue(any(r["high_error"] for r in used))
            self.assertEqual(config["quality_assessment"]["status"], "review")
            np.testing.assert_allclose([r["reprojection_error_px"] for r in used],
                                       config["per_view_reprojection_px"])
            self.assertTrue((output / "calibration_report.md").is_file())
            self.assertTrue((output / "reprojection_visualization.png").is_file())
            self.assertEqual(len(list((output / "undistorted_samples").glob("*.png"))), 6)
            before = (output / "calibration.json").read_bytes()
            with self.assertRaises(FileExistsError):
                run_calibration(source, 9, 6, .025, output_root=base / "runs",
                                run_id="test", provenance="synthetic")
            self.assertEqual((output / "calibration.json").read_bytes(), before)

    def test_failure_reports_insufficient_and_inconsistent(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            folder = base / "images"
            folder.mkdir()
            cv2.imwrite(str(folder / "blank.png"), np.zeros((480, 640), np.uint8))
            with self.assertRaises(CalibrationFailure):
                run_calibration(folder, 9, 6, .025, output_root=base / "runs", run_id="few")
            failure = json.loads((base / "runs/real_few/image_results.json").read_text(encoding="utf-8"))
            self.assertEqual(failure["status"], "failed")
            self.assertEqual(failure["images"][0]["reason"], "checkerboard_not_detected")
            self.assertTrue((base / "runs/real_few/calibration_report.md").exists())
            cv2.imwrite(str(folder / "other.jpg"), np.zeros((720, 1280), np.uint8))
            with self.assertRaises(CalibrationFailure):
                run_calibration(folder, 9, 6, .025, output_root=base / "runs", run_id="mixed")
            failure = json.loads((base / "runs/real_mixed/image_results.json").read_text(encoding="utf-8"))
            self.assertEqual(failure["images"][1]["reason"], "inconsistent_resolution")
            with self.assertRaises(CalibrationFailure):
                run_calibration(base / "missing", 9, 6, .025, output_root=base / "runs")

    def test_cli_help_and_missing_folder(self):
        help_result = subprocess.run([sys.executable, "-m", "backend.tools.calibrate_camera", "--help"],
                                     capture_output=True, text=True)
        self.assertEqual(help_result.returncode, 0)
        self.assertIn("INNER corners", help_result.stdout)
        with tempfile.TemporaryDirectory() as directory:
            missing = str(Path(directory) / "missing")
            result = subprocess.run([sys.executable, "-m", "backend.tools.calibrate_camera",
                                     "--input", missing, "--columns", "9", "--rows", "6",
                                     "--square-size", "0.025"], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("does not exist", result.stderr)

    def test_reprojection_error_matches_projected_points(self):
        corners, _, _ = synthetic_corners()
        result = calibrate_observations(corners, (640, 480), 9, 6, .025, include_projections=True)
        for observed, predicted, error in zip(corners, result["projected_corners_px"],
                                               result["per_view_reprojection_px"]):
            residual = np.asarray(predicted) - observed.reshape(-1, 2)
            calculated = np.sqrt(np.mean(np.sum(residual ** 2, axis=1)))
            self.assertAlmostEqual(calculated, error, delta=1e-8)


if __name__ == "__main__":
    unittest.main()
