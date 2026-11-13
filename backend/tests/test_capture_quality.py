"""Synthetic quality-policy tests: no real detector or model assets."""
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from backend.services.capture_quality import QualityConfig, analyze_frames


def fixture_pose(name, detected=True):
    coords = {
        "nose": (.50, .15), "left_shoulder": (.40, .25), "right_shoulder": (.60, .25),
        "left_elbow": (.35, .40), "right_elbow": (.65, .40),
        "left_wrist": (.30, .55), "right_wrist": (.70, .55),
        "left_hip": (.43, .55), "right_hip": (.57, .55),
        "left_knee": (.42, .70), "right_knee": (.58, .70),
        "left_ankle": (.41, .88), "right_ankle": (.59, .88),
    }
    landmarks = {k: {"x": x, "y": y, "z": -.1 if k == "nose" else 0.0,
                     "visible": True, "visibility": .95} for k, (x, y) in coords.items()}
    return {"frame": name, "person_detected": detected, "full_body_visible": detected,
            "detection_confidence": .95, "landmarks": landmarks, "missing_regions": []}


class QualityTests(unittest.TestCase):
    def test_blur_lighting_bystander_missing_evidence_and_coverage(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            sharp = np.indices((240, 320)).sum(axis=0) % 2 * 180 + 35
            sharp = np.asarray(sharp, dtype=np.uint8)
            blurred = cv2.GaussianBlur(sharp, (9, 9), 3)
            images = [sharp, blurred, sharp, np.zeros_like(sharp)]
            frames = []
            poses = []
            for i, image in enumerate(images, 1):
                name = f"frame_{i:03d}.jpg"
                self.assertTrue(cv2.imwrite(str(folder / name), image))
                frames.append({"filename": name, "timestamp_sec": i, "is_blurry": i == 2,
                               "blur_score": 5 if i == 2 else 120, "diff_score": None if i == 1 else 5})
                poses.append(fixture_pose(Path(name).stem))
            detections = {
                "frame_001.jpg": [{"box": [.25, .1, .75, .95]}],
                "frame_002.jpg": [{"box": [.25, .1, .75, .95]}],
                "frame_003.jpg": [{"box": [.25, .1, .75, .95]}, {"box": [.0, .1, .2, .9]}],
            }
            report = analyze_frames({"video_id": "synthetic", "frames": frames,
                                     "rejected_frames": [{"reason": "duplicate"}]},
                                    {"frames": poses}, folder, detections=detections)
            self.assertEqual(report["summary"]["extracted_frames"], 4)
            self.assertEqual(report["summary"]["rejected_duplicates"], 1)
            self.assertIn("extractor_blur_warning", report["frames"][1]["warnings"])
            self.assertIn("possible_multiple_people", report["frames"][2]["warnings"])
            self.assertIn("person_count_unknown", report["frames"][3]["warnings"])
            self.assertIn("lighting_review", report["frames"][3]["warnings"])
            self.assertEqual(report["viewpoint_coverage"]["status"], "unconfirmed")
            self.assertTrue(all(not frame["recommended_for_reconstruction"] for frame in report["frames"]))
            self.assertEqual(report["summary"]["status_counts"]["acceptable"], 0)

    def test_missing_pose_and_unreadable_are_not_accepted(self):
        with tempfile.TemporaryDirectory() as temp:
            report = analyze_frames({"frames": [{"filename": "frame_001.jpg", "is_blurry": False}]},
                                    {"frames": []}, temp)
            self.assertEqual(report["frames"][0]["status"], "rejected")
            self.assertIn("person_detection_unknown", report["frames"][0]["warnings"])
            self.assertEqual(report["frames"][0]["viewpoint"]["label"], "unknown")

    def test_ambiguous_side_and_subject_discontinuity(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            image = np.full((240, 320, 3), 127, dtype=np.uint8)
            frames = []
            poses = []
            for index in (1, 2):
                name = f"frame_{index:03d}.jpg"
                self.assertTrue(cv2.imwrite(str(folder / name), image))
                frames.append({"filename": name, "timestamp_sec": index, "is_blurry": False})
                pose = fixture_pose(Path(name).stem)
                if index == 1:
                    pose["landmarks"]["left_shoulder"]["x"] = .49
                    pose["landmarks"]["right_shoulder"]["x"] = .51
                else:
                    for point in pose["landmarks"].values():
                        point["x"] = min(.99, point["x"] + .4)
                poses.append(pose)
            report = analyze_frames({"frames": frames}, {"frames": poses}, folder)
            self.assertEqual(report["frames"][0]["viewpoint"]["label"], "unknown")
            self.assertEqual(report["frames"][0]["viewpoint"]["raw_label"], "intermediate")
            self.assertIn("subject_continuity_uncertain", report["frames"][1]["warnings"])
            self.assertFalse(report["frames"][1]["recommended_for_reconstruction"])

    def test_visible_landmark_outside_image_is_not_full_body_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            cv2.imwrite(str(folder / "frame_001.jpg"), np.full((240, 320, 3), 127, dtype=np.uint8))
            pose = fixture_pose("frame_001")
            pose["landmarks"]["left_ankle"]["x"] = 1.2
            report = analyze_frames(
                {"frames": [{"filename": "frame_001.jpg", "is_blurry": False}]},
                {"frames": [pose]}, folder,
            )
            self.assertEqual(report["frames"][0]["checks"]["full_body"], "unknown")
            self.assertIn("full_body_visibility_unconfirmed", report["frames"][0]["warnings"])

    def test_threshold_validation(self):
        with self.assertRaises(ValueError):
            QualityConfig(min_brightness=240, max_brightness=20)
        with self.assertRaises(ValueError):
            QualityConfig(min_gradient=float("nan"))


if __name__ == "__main__":
    unittest.main()
