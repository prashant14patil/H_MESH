"""Synthetic calibration and fixed-camera metadata tests; no model assets."""
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from backend.services.checkerboard_calibration import (
    CalibrationFailure, board_points, calibrate_images, calibrate_observations,
    load_calibration, save_calibration, undistort_pixels,
)
from backend.services.fixed_camera_validation import validate_capture


def synthetic_corners():
    obj = board_points(9, 6, .025)
    matrix = np.array([[800., 0., 320.], [0., 810., 240.], [0., 0., 1.]])
    dist = np.array([-.08, .02, .001, -.002, 0.])
    corners = []
    for i in range(10):
        rvec = np.array([-.22 + .05*i, .16*np.sin(i), .05*np.cos(i)])
        tvec = np.array([-.14 + .025*i, -.07 + .015*(i % 3), .9 + .04*(i % 4)])
        projected, _ = cv2.projectPoints(obj, rvec, tvec, matrix, dist)
        corners.append(projected)
    return corners, matrix, dist


def pose(name, shift=0., side=False, clipped=False):
    names = ("nose", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
             "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
             "right_knee", "left_ankle", "right_ankle")
    landmarks = {n: {"x": .5 + shift, "y": .5, "z": 0., "visibility": .95, "visible": True}
                 for n in names}
    landmarks["left_shoulder"]["x"] = .49 + shift if side else .35 + shift
    landmarks["right_shoulder"]["x"] = .51 + shift if side else .65 + shift
    landmarks["nose"]["z"] = -.15
    if clipped:
        landmarks["left_ankle"]["y"] = 1.01
    return {"frame": name, "person_detected": True, "full_body_visible": True, "landmarks": landmarks}


class CalibrationTests(unittest.TestCase):
    def test_synthetic_calibration_save_load_and_undistort(self):
        corners, matrix, dist = synthetic_corners()
        config = calibrate_observations(corners, (640, 480), 9, 6, .025)
        self.assertEqual(config["provenance"], "synthetic")
        self.assertLess(config["rms_reprojection_px"], .01)
        np.testing.assert_allclose(np.array(config["camera_matrix"])[:2, :2], matrix[:2, :2], atol=1.)
        points = np.array([[.2, .1, 1.], [-.25, .15, 1.]])
        distorted, _ = cv2.projectPoints(points, np.zeros(3), np.zeros(3), matrix, dist)
        undistorted = undistort_pixels(config, distorted.reshape(-1, 2), (640, 480))
        expected = points[:, :2] * [800, 810] + [320, 240]
        np.testing.assert_allclose(undistorted, expected, atol=.1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            save_calibration(config, path)
            self.assertEqual(load_calibration(path, (640, 480)), config)
            with self.assertRaises(CalibrationFailure):
                load_calibration(path, (1280, 720))
            with self.assertRaises(FileExistsError):
                save_calibration(config, path)

    def test_rendered_checkerboard_images(self):
        board = np.zeros((7 * 80, 10 * 80), np.uint8)
        for row in range(7):
            for col in range(10):
                if (row + col) % 2 == 0:
                    board[row * 80:(row + 1) * 80, col * 80:(col + 1) * 80] = 255
        source = np.float32([[0, 0], [799, 0], [799, 559], [0, 559]])
        matrix = np.array([[800., 0., 320.], [0., 810., 240.], [0., 0., 1.]])
        outer = np.float32([[-.025, -.025, 0], [.225, -.025, 0], [.225, .15, 0], [-.025, .15, 0]])
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for i in range(10):
                rotation = np.array([-.22 + .05 * i, .16 * np.sin(i), .05 * np.cos(i)])
                translation = np.array([-.14 + .025 * i, -.07 + .015 * (i % 3), .9 + .04 * (i % 4)])
                target, _ = cv2.projectPoints(outer, rotation, translation, matrix, np.zeros(5))
                homography = cv2.getPerspectiveTransform(source, target.reshape(4, 2))
                image = cv2.warpPerspective(board, homography, (640, 480), borderValue=127)
                path = str(Path(directory) / f"board_{i}.png")
                self.assertTrue(cv2.imwrite(path, image))
                paths.append(path)
            result = calibrate_images(paths, 9, 6, .025, provenance="synthetic")
            self.assertEqual(result["provenance"], "synthetic")
            self.assertGreaterEqual(result["view_count"], 6)
            self.assertLess(result["rms_reprojection_px"], 2.)

    def test_missing_degenerate_and_invalid_calibration(self):
        with self.assertRaises(CalibrationFailure):
            calibrate_images([], 9, 6, .025)
        corners, _, _ = synthetic_corners()
        with self.assertRaises(CalibrationFailure):
            calibrate_observations([corners[0]] * 6, (640, 480), 9, 6, .025)
        with self.assertRaises(CalibrationFailure):
            calibrate_observations(corners[:2], (640, 480), 9, 6, .025)


class FixedCameraTests(unittest.TestCase):
    def test_quality_motion_and_ambiguous_orientation(self):
        metadata = {"frames": [
            {"filename": "frame_002.jpg", "timestamp_sec": 1., "is_blurry": True, "blur_score": 20., "diff_score": 8.},
            {"filename": "frame_001.jpg", "timestamp_sec": 0., "is_blurry": False, "blur_score": 200., "diff_score": None},
            {"filename": "frame_003.jpg", "timestamp_sec": 2., "is_blurry": False, "blur_score": 200., "diff_score": 5.},
        ], "rejected_frames": [{"reason": "duplicate", "timestamp_sec": .5}]}
        poses = {"frames": [pose("frame_001"), pose("frame_002", shift=.2, side=True, clipped=True)]}
        names = [f["filename"] for f in metadata["frames"]]
        result = validate_capture(metadata, poses, names, allow_blurry=True)
        a, b, c = result["frames"]
        self.assertEqual(a["filename"], "frame_001.jpg")
        self.assertEqual(a["duplicate_status"], "unknown_no_comparison_score")
        self.assertEqual(b["orientation"]["label"], "unknown")
        self.assertEqual(b["orientation"]["raw_estimator"]["reason"], "side_view_but_left_right_ambiguous")
        self.assertIn("blurry_manual_override", b["warnings"])
        self.assertIn("outside_capture_region", b["warnings"])
        self.assertIn("excessive_image_displacement", b["warnings"])
        self.assertEqual(c["orientation"]["label"], "unknown")
        self.assertEqual(result["camera"]["motion"], "assumed_fixed_not_verified")
        self.assertEqual(result["coverage"]["status"], "unconfirmed")
        self.assertEqual(len(result["extraction_rejected_frames"]), 1)
        no_override = validate_capture(metadata, poses, names)
        self.assertEqual(no_override["selection"]["counts"]["accepted"], 2)

    def test_missing_landmarks_are_unknown(self):
        p = pose("frame_001")
        del p["landmarks"]["left_ankle"]
        meta = {"frames": [{"filename": "frame_001.jpg", "is_blurry": False}]}
        result = validate_capture(meta, {"frames": [p]}, ["frame_001.jpg"])
        self.assertIsNone(result["frames"][0]["within_capture_region"])
        self.assertIn("capture_region_unknown_missing_landmarks", result["frames"][0]["warnings"])


if __name__ == "__main__":
    unittest.main()
