"""Subject-review contract tests with synthetic images and detections."""
import io
import tempfile
import time
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import backend.services.subject_review as review
import backend.services.video_mesh_batch as batch
from backend.tests.test_video_mesh_batch import FakeCapture, FakeEngine, fake_report


@patch.object(review, "require_authorization", lambda video_id: {"source_sha256": "synthetic"})
@patch.object(batch, "require_authorization", lambda video_id: {"source_sha256": "synthetic"})
class SubjectReviewTests(unittest.TestCase):
    def test_suggestions_are_never_approvals(self):
        ref = [.1, .1, .4, .9]
        self.assertEqual(review.suggest(ref, [])["status"], "ambiguous")
        self.assertEqual(review.suggest(ref, [{"box": ref}, {"box": ref}])["status"], "ambiguous")
        switched = review.suggest(ref, [{"box": [.6, .1, .9, .9]}])
        self.assertEqual(switched["status"], "ambiguous")
        candidate = review.suggest(ref, [{"box": ref}, {"box": [.6, .1, .9, .9]}])
        self.assertEqual(candidate["status"], "suggested_unconfirmed")

    def test_review_history_rejection_and_stale_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            capture = FakeCapture(root)
            video_id, quality_id = str(uuid.uuid4()), str(uuid.uuid4())
            report = fake_report(capture, video_id, quality_id)
            report["frames"][1]["detector_evidence"] = [
                {"box": [.1, .1, .9, .9]}, {"box": [.1, .1, .9, .9]}
            ]
            with patch.object(review, "ROOT", root / "reviews"), \
                 patch.object(review, "get_capture_engine", return_value=capture), \
                 patch.object(review, "get_report", return_value=report):
                session = review.create_session(video_id, quality_id, "frame_001.jpg", [.1, .1, .9, .9])
                sid = session["session_id"]
                self.assertEqual(review.frame_suggestion(sid, "frame_002.jpg")["identity_status"], "identity_ambiguous")
                self.assertIsNone(review.get_decision(sid, "frame_002.jpg"))
                with self.assertRaisesRegex(ValueError, "multiple detected"):
                    review.review_frame(sid, "frame_002.jpg", "confirm", [.1, .1, .9, .9])
                report["frames"][1]["detector_evidence"] = [{"box": [.1, .1, .9, .9]}]
                first = review.review_frame(sid, "frame_002.jpg", "confirm", [.1, .1, .9, .9])
                self.assertEqual(first["identity_status"], "human_confirmed_not_biometric")
                self.assertEqual(first["recording_id"], video_id)
                self.assertEqual(first["frame_sha256"], report["frames"][1]["source_sha256"])
                second = review.review_frame(sid, "frame_002.jpg", "reassign", [.05, .05, .95, .95])
                self.assertNotEqual(first["decision_id"], second["decision_id"])
                self.assertEqual(len(list((root / "reviews" / sid / "decisions" / "history").glob("*.json"))), 2)
                self.assertEqual(review.review_frame(sid, "frame_002.jpg", "reject")["identity_status"], "rejected")
                with self.assertRaisesRegex(ValueError, "landmark"):
                    review.review_frame(sid, "frame_002.jpg", "confirm", [.3, .3, .7, .7])
                Image.new("RGB", (100, 100), "blue").save(root / "frame_002.jpg")
                with self.assertRaisesRegex(ValueError, "changed"):
                    review.review_frame(sid, "frame_002.jpg", "confirm", [.1, .1, .9, .9])

    def test_persisted_review_is_used_to_submit_exact_crop(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            capture = FakeCapture(root)
            engine = FakeEngine(root)
            engine.counter = 1  # Make this fixture's next job succeed.
            video_id, quality_id = str(uuid.uuid4()), str(uuid.uuid4())
            report = fake_report(capture, video_id, quality_id)
            with patch.object(review, "ROOT", root / "reviews"), \
                 patch.object(review, "get_capture_engine", return_value=capture), \
                 patch.object(review, "get_report", return_value=report), \
                 patch.object(batch, "_RUN_ROOT", root / "runs"), \
                 patch.object(batch, "get_capture_engine", return_value=capture), \
                 patch.object(batch, "get_report", return_value=report), \
                 patch.object(batch, "get_engine", return_value=engine):
                session = review.create_session(video_id, quality_id, "frame_001.jpg", [.1, .1, .9, .9])
                sid = session["session_id"]
                decision = review.review_frame(sid, "frame_001.jpg", "confirm", [.1, .1, .9, .9])
                run = batch.start_batch(video_id, ["frame_001.jpg"], quality_id, ["frame_001.jpg"], sid)
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    result = batch.get_run(run["run_id"])
                    if result["status"] != "processing":
                        break
                    time.sleep(.01)
                self.assertEqual(result["status"], "completed")
                self.assertEqual(result["frames"][0]["status"], "completed")
                self.assertEqual(result["frames"][0]["review"]["decision_id"], decision["decision_id"])
                self.assertEqual(Image.open(io.BytesIO(engine.inputs[0])).size, (80, 80))
                self.assertTrue(batch.get_asset(run["run_id"], "frame_001.jpg", "mesh").is_file())

    def test_resume_restores_only_matching_current_decisions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            capture = FakeCapture(root)
            video_id, quality_id = str(uuid.uuid4()), str(uuid.uuid4())
            report = fake_report(capture, video_id, quality_id)
            with patch.object(review, "ROOT", root / "reviews"), \
                 patch.object(review, "get_capture_engine", return_value=capture), \
                 patch.object(review, "get_report", return_value=report):
                session = review.create_session(video_id, quality_id, "frame_001.jpg", [.1, .1, .9, .9])
                sid = session["session_id"]
                decision = review.review_frame(sid, "frame_002.jpg", "confirm", [.1, .1, .9, .9])
                resumed = review.resume_session(sid, video_id, quality_id)
                self.assertEqual(resumed["decisions"]["frame_002.jpg"]["decision_id"], decision["decision_id"])
                self.assertEqual(resumed["unavailable_frames"], [])
                report["frames"][1]["checks"]["sharpness"] = "warning"
                self.assertNotIn("frame_002.jpg", review.resume_session(sid, video_id, quality_id)["decisions"])
                report["frames"][1]["checks"]["sharpness"] = "pass"
                report["frames"][1]["detector_evidence"] = [
                    {"box": [.1, .1, .9, .9]}, {"box": [.2, .2, .8, .8]}]
                self.assertNotIn("frame_002.jpg", review.resume_session(sid, video_id, quality_id)["decisions"])
                report["frames"][1]["detector_evidence"] = []
                with self.assertRaisesRegex(ValueError, "does not match"):
                    review.resume_session(sid, str(uuid.uuid4()), quality_id)
                Image.new("RGB", (100, 100), "blue").save(root / "frame_002.jpg")
                resumed = review.resume_session(sid, video_id, quality_id)
                self.assertNotIn("frame_002.jpg", resumed["decisions"])
                self.assertIn("frame_002.jpg", resumed["unavailable_frames"])
                Image.new("RGB", (100, 100), "green").save(root / "frame_001.jpg")
                with self.assertRaisesRegex(ValueError, "changed"):
                    review.resume_session(sid, video_id, quality_id)

    def test_missing_pose_and_quality_warning_cannot_be_approved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            capture = FakeCapture(root)
            video_id, quality_id = str(uuid.uuid4()), str(uuid.uuid4())
            report = fake_report(capture, video_id, quality_id)
            with patch.object(review, "ROOT", root / "reviews"), \
                 patch.object(review, "get_capture_engine", return_value=capture), \
                 patch.object(review, "get_report", return_value=report):
                session = review.create_session(video_id, quality_id, "frame_001.jpg", [.1, .1, .9, .9])
                sid = session["session_id"]
                report["frames"][1]["checks"]["sharpness"] = "warning"
                with self.assertRaisesRegex(ValueError, "mandatory"):
                    review.review_frame(sid, "frame_002.jpg", "confirm", [.1, .1, .9, .9])
                report["frames"][1]["checks"]["sharpness"] = "pass"
                report["frames"][1]["landmark_bbox_normalized"] = None
                with self.assertRaisesRegex(ValueError, "unavailable"):
                    review.review_frame(sid, "frame_002.jpg", "confirm", [.1, .1, .9, .9])


if __name__ == "__main__":
    unittest.main()
