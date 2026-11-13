"""Background quality job tests using synthetic fixtures."""
import hashlib
import tempfile
import time
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

import backend.services.capture_quality_job as jobs
from backend.services.subject_review import validate_review_crop
from backend.tests.test_capture_quality import fixture_pose


class Capture:
    def __init__(self, path):
        self.path = path

    def get_metadata(self, video_id):
        return {"video_id": video_id, "status": "completed",
                "frames": [{"filename": "frame_001.jpg", "timestamp_sec": 0,
                            "is_blurry": False, "blur_score": 150}],
                "rejected_frames": []}

    def get_frame_path(self, video_id, filename):
        return self.path / filename


class Pose:
    def __init__(self, path):
        self.digest = hashlib.sha256((path / "frame_001.jpg").read_bytes()).hexdigest()

    def get_summary(self, video_id):
        frame = fixture_pose("frame_001")
        frame["source_sha256"] = self.digest
        return {"status": "completed", "frames": [frame]}


@patch.object(jobs, "require_authorization", lambda video_id: None)
class JobTests(unittest.TestCase):
    def test_nms_is_deterministic_and_conservative(self):
        detections = [
            {"box": [.1, .1, .9, .9], "score": .5},
            {"box": [.1, .1, .9, .9], "score": .9},
            {"box": [.12, .12, .88, .88], "score": .8},
            {"box": [.3, .3, .5, .5], "score": .7},  # nested, low IoU
            {"box": [.0, .0, .2, .2], "score": .6},  # separate
            {"box": [.55, .1, .99, .9], "score": .4},  # overlapping, IoU below threshold
            {"box": [0, 0, 0, 1], "score": .3},
            {"box": [0, 0, 1, 1], "score": float("nan")},
        ]
        result = jobs._nms_diagnostics(detections)
        self.assertEqual(result["raw_count"], 8)
        self.assertEqual(result["retained_indices"], [1, 3, 4, 5])
        self.assertEqual(result["filtered_count"], 4)
        self.assertEqual([s["index"] for s in result["suppressed"]], [2, 0])
        self.assertTrue(all(s["retained_index"] == 1 for s in result["suppressed"]))
        self.assertEqual(result["invalid_indices"], [6, 7])
        self.assertEqual(len(detections), 8)
        self.assertIsNone(jobs._nms_diagnostics(None))
        self.assertEqual(jobs._nms_diagnostics([])["filtered_count"], 0)
        equal = [{"box": [.1, .1, .9, .9], "score": 1}] * 2
        self.assertEqual(jobs._nms_diagnostics(equal)["retained_indices"], [0])
        with self.assertRaises(ValueError):
            jobs._nms_diagnostics([], float("nan"))

    def test_nms_does_not_change_raw_quality_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cv2.imwrite(str(root / "frame_001.jpg"), np.full((160, 160, 3), 127, dtype=np.uint8))
            detections = [{"box": [.1, .1, .9, .9], "score": .9},
                          {"box": [.1, .1, .9, .9], "score": .7}]
            with patch.object(jobs, "ROOT", root / "reports"), \
                 patch.object(jobs, "get_capture_engine", return_value=Capture(root)), \
                 patch.object(jobs, "get_pose_engine", return_value=Pose(root)), \
                 patch.object(jobs, "_hog_observations", return_value={"frame_001.jpg": detections}):
                initial = jobs.start_quality(str(uuid.uuid4()))
                end = time.monotonic() + 5
                while time.monotonic() < end:
                    state = jobs.get_job(initial["run_id"])
                    if state["status"] != "processing":
                        break
                    time.sleep(.01)
                self.assertEqual(state["status"], "completed")
                frame = jobs.get_report(initial["run_id"])["frames"][0]
                self.assertEqual(frame["detector_evidence"], detections)
                self.assertEqual(frame["person_count_observed"], 2)
                self.assertIn("possible_multiple_people", frame["warnings"])
                self.assertEqual(frame["detector_nms"]["filtered_count"], 1)
                self.assertEqual(frame["detector_nms"]["raw_count"], 2)
                with self.assertRaisesRegex(ValueError, "multiple detected"):
                    validate_review_crop([0, 0, 1, 1], frame)

    def test_job_persists_report_without_detector(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cv2.imwrite(str(root / "frame_001.jpg"), np.full((160, 160, 3), 127, dtype=np.uint8))
            with patch.object(jobs, "ROOT", root / "reports"), \
                 patch.object(jobs, "get_capture_engine", return_value=Capture(root)), \
                 patch.object(jobs, "get_pose_engine", return_value=Pose(root)):
                initial = jobs.start_quality(str(uuid.uuid4()), use_hog=False)
                self.assertEqual(initial["status"], "processing")
                end = time.monotonic() + 5
                while time.monotonic() < end:
                    state = jobs.get_job(initial["run_id"])
                    if state["status"] != "processing":
                        break
                    time.sleep(.01)
                self.assertEqual(state["status"], "completed")
                report = jobs.get_report(initial["run_id"])
                self.assertEqual(report["summary"]["extracted_frames"], 1)
                self.assertEqual(report["frames"][0]["status"], "review")
                self.assertEqual(report["detector_status"], "disabled")
                self.assertEqual(report["viewpoint_coverage"]["status"], "unconfirmed")

    def test_stale_pose_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "frame_001.jpg"
            cv2.imwrite(str(path), np.full((160, 160, 3), 127, dtype=np.uint8))
            pose = Pose(root)
            cv2.imwrite(str(path), np.full((160, 160, 3), 40, dtype=np.uint8))
            with patch.object(jobs, "ROOT", root / "reports"), \
                 patch.object(jobs, "get_capture_engine", return_value=Capture(root)), \
                 patch.object(jobs, "get_pose_engine", return_value=pose):
                with self.assertRaisesRegex(ValueError, "stale"):
                    jobs.start_quality(str(uuid.uuid4()), use_hog=False)

    def test_missing_pose_and_bad_id(self):
        with patch.object(jobs, "get_capture_engine", return_value=Capture(Path("."))), \
             patch.object(jobs, "get_pose_engine") as pose:
            pose.return_value.get_summary.return_value = None
            with self.assertRaisesRegex(ValueError, "pose summary"):
                jobs.start_quality(str(uuid.uuid4()))
        with self.assertRaises(ValueError):
            jobs.get_report("../bad")


if __name__ == "__main__":
    unittest.main()
