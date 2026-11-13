"""Recording authorization tests using temporary synthetic data only."""
import json
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from backend.services import capture_authorization as auth
from backend.services import capture_quality_job as quality
from backend.services import subject_review as review
from backend.services import video_mesh_batch as batch
from backend.services import video_processor
from backend.tests.test_video_mesh_batch import FakeCapture, FakeEngine, fake_report, fake_session, fake_decision


class AuthorizationTests(unittest.TestCase):
    def test_explicit_attestation_is_bound_to_recording_and_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video_id = str(uuid.uuid4())
            source_dir = root / video_id
            source_dir.mkdir()
            source = source_dir / "original.mp4"
            source.write_bytes(b"synthetic source")
            with patch.object(video_processor, "_OUTPUT_ROOT", root):
                with self.assertRaisesRegex(ValueError, "required"):
                    auth.require_authorization(video_id)
                for authorized, consent in [(False, True), (True, False), (1, True), (True, None)]:
                    with self.assertRaises(ValueError):
                        auth.attest(video_id, authorized, consent)
                record = auth.attest(video_id, True, True)
                self.assertEqual(auth.require_authorization(video_id), record)
                self.assertIn("not authenticated", record["note"])
                other = str(uuid.uuid4())
                (root / other).mkdir()
                (root / other / "original.mp4").write_bytes(b"synthetic source")
                (root / other / "authorization.json").write_text(json.dumps(record))
                with self.assertRaisesRegex(ValueError, "invalid"):
                    auth.require_authorization(other)
                source.write_bytes(b"changed source")
                with self.assertRaisesRegex(ValueError, "changed"):
                    auth.require_authorization(video_id)
                with self.assertRaises(ValueError):
                    auth.require_authorization("../invalid")

    def test_attestation_does_not_bypass_review_and_stale_source_blocks_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video_id, run_id, session_id = (str(uuid.uuid4()) for _ in range(3))
            capture_dir = root / video_id
            capture_dir.mkdir()
            source = capture_dir / "original.mp4"
            source.write_bytes(b"synthetic video")
            capture = FakeCapture(capture_dir)
            engine = FakeEngine(root)
            report = fake_report(capture, video_id, run_id)
            session = fake_session(capture, video_id, run_id, session_id)
            name = "frame_001.jpg"
            decision = fake_decision(capture, video_id, run_id, session_id, name)
            with patch.object(video_processor, "_OUTPUT_ROOT", root), \
                 patch.object(batch, "get_capture_engine", return_value=capture), \
                 patch.object(batch, "get_engine", return_value=engine), \
                 patch.object(batch, "get_report", return_value=report), \
                 patch.object(batch, "get_session", return_value=session), \
                 patch.object(batch, "get_decision", return_value=None):
                with self.assertRaisesRegex(ValueError, "authorization"):
                    batch.start_batch(video_id, [name], run_id, [name], session_id)
                authorization = auth.attest(video_id, True, True)
                with self.assertRaisesRegex(ValueError, "authorization changed"):
                    batch.start_batch(video_id, [name], run_id, [name], session_id)
                session["authorization_source_sha256"] = authorization["source_sha256"]
                with self.assertRaisesRegex(ValueError, "explicit subject review"):
                    batch.start_batch(video_id, [name], run_id, [name], session_id)
                with patch.object(batch, "get_decision", return_value=decision):
                    report["frames"][0]["checks"]["full_body"] = "unknown"
                    with self.assertRaisesRegex(ValueError, "quality"):
                        batch.start_batch(video_id, [name], run_id, [name], session_id)
                    report["frames"][0]["checks"]["full_body"] = "pass"
                    with patch.object(batch.inference_engine, "_engine", None), \
                         patch.object(batch, "get_engine") as get_engine:
                        readiness = batch.check_readiness(video_id, [name], run_id, [name], session_id)
                        self.assertFalse(readiness["ready"])
                        self.assertIn("not ready", readiness["blockers"][0])
                        get_engine.assert_not_called()
                    with patch.object(batch.inference_engine, "_engine", engine), \
                         patch.object(batch, "_RUN_ROOT", root / "runs"):
                        readiness = batch.check_readiness(video_id, [name], run_id, [name], session_id)
                        self.assertTrue(readiness["ready"])
                        self.assertFalse((root / "runs").exists())
                        self.assertEqual(engine.counter, 0)
                    source.write_bytes(b"changed video")
                    with self.assertRaisesRegex(ValueError, "source has changed"):
                        batch.start_batch(video_id, [name], run_id, [name], session_id)
                self.assertEqual(engine.counter, 0)

    def test_reattested_changed_source_cannot_reuse_review_session(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video_id, run_id = str(uuid.uuid4()), str(uuid.uuid4())
            capture_dir = root / video_id
            capture_dir.mkdir()
            source = capture_dir / "original.mp4"
            source.write_bytes(b"original synthetic video")
            capture = FakeCapture(capture_dir)
            report = fake_report(capture, video_id, run_id)
            with patch.object(video_processor, "_OUTPUT_ROOT", root), \
                 patch.object(review, "ROOT", root / "reviews"), \
                 patch.object(review, "get_capture_engine", return_value=capture), \
                 patch.object(review, "get_report", return_value=report):
                auth.attest(video_id, True, True)
                session = review.create_session(video_id, run_id, "frame_001.jpg", [.1, .1, .9, .9])
                source.write_bytes(b"changed synthetic video")
                auth.attest(video_id, True, True)
                with self.assertRaisesRegex(ValueError, "authorization changed"):
                    review.resume_session(session["session_id"], video_id, run_id)
                with self.assertRaisesRegex(ValueError, "authorization changed"):
                    review.review_frame(session["session_id"], "frame_001.jpg", "confirm", [.1, .1, .9, .9])

    def test_readiness_reports_missing_authorization_and_consent_without_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video_id, run_id = str(uuid.uuid4()), str(uuid.uuid4())
            folder = root / video_id
            folder.mkdir()
            (folder / "original.mp4").write_bytes(b"synthetic video")
            capture = FakeCapture(folder)
            report = fake_report(capture, video_id, run_id)
            with patch.object(video_processor, "_OUTPUT_ROOT", root), \
                 patch.object(batch, "_RUN_ROOT", root / "runs"), \
                 patch.object(batch, "get_capture_engine", return_value=capture), \
                 patch.object(batch, "get_report", return_value=report), \
                 patch.object(batch.inference_engine, "_engine", FakeEngine(root)):
                for authorized, consent in ((False, True), (True, False)):
                    (folder / "authorization.json").write_text(json.dumps({
                        "video_id": video_id, "source_sha256": auth._source_hash(video_id),
                        "authorized": authorized, "consent_confirmed": consent,
                    }), encoding="utf-8")
                    result = batch.check_readiness(video_id, ["frame_001.jpg"], run_id, [], None)
                    self.assertFalse(result["ready"])
                    self.assertTrue(any("authorization" in item for item in result["blockers"]))
                    self.assertTrue(any("acknowledgement" in item for item in result["blockers"]))
                    self.assertTrue(any("reference subject" in item for item in result["blockers"]))
                    self.assertTrue(any("decisions" in item for item in result["unevaluated"]))
                    self.assertFalse((root / "runs").exists())

    def test_persisted_review_preflight_and_submission_revalidate_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video_id, run_id = str(uuid.uuid4()), str(uuid.uuid4())
            capture_dir = root / video_id
            capture_dir.mkdir()
            video = capture_dir / "original.mp4"
            video.write_bytes(b"synthetic video")
            capture = FakeCapture(capture_dir)
            report = fake_report(capture, video_id, run_id)
            engine = FakeEngine(root)
            name = "frame_002.jpg"
            with patch.object(video_processor, "_OUTPUT_ROOT", root), \
                 patch.object(review, "ROOT", root / "reviews"), \
                 patch.object(review, "get_capture_engine", return_value=capture), \
                 patch.object(review, "get_report", return_value=report), \
                 patch.object(batch, "get_capture_engine", return_value=capture), \
                 patch.object(batch, "get_report", return_value=report), \
                 patch.object(batch, "_RUN_ROOT", root / "runs"), \
                 patch.object(batch.inference_engine, "_engine", engine):
                auth.attest(video_id, True, True)
                session = review.create_session(video_id, run_id, "frame_001.jpg", [.1, .1, .9, .9])
                sid = session["session_id"]
                decision = review.review_frame(sid, name, "confirm", [.1, .1, .9, .9])
                self.assertEqual(decision["recording_id"], video_id)
                self.assertEqual(decision["frame_sha256"], report["frames"][1]["source_sha256"])
                self.assertEqual(review.resume_session(sid, video_id, run_id)["decisions"][name]["decision_id"],
                                 decision["decision_id"])
                self.assertFalse(batch.check_readiness(video_id, [name], run_id, [], sid)["ready"])
                self.assertTrue(batch.check_readiness(video_id, [name], run_id, [name], sid)["ready"])
                frame = capture_dir / name
                original_frame = frame.read_bytes()
                frame.write_bytes(b"changed frame")
                with self.assertRaisesRegex(ValueError, "changed"):
                    batch.start_batch(video_id, [name], run_id, [name], sid)
                frame.write_bytes(original_frame)
                video.write_bytes(b"changed video")
                with self.assertRaisesRegex(ValueError, "source has changed"):
                    batch.start_batch(video_id, [name], run_id, [name], sid)
                self.assertEqual(engine.counter, 0)
                self.assertFalse((root / "runs").exists())

    def test_missing_attestation_blocks_processing_review_and_batch(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(video_processor, "_OUTPUT_ROOT", Path(directory)):
            video_id, run_id = str(uuid.uuid4()), str(uuid.uuid4())
            with patch.object(quality, "get_capture_engine") as capture:
                with self.assertRaisesRegex(ValueError, "authorization"):
                    quality.start_quality(video_id)
                capture.assert_not_called()
            with patch.object(review, "get_report") as report:
                with self.assertRaisesRegex(ValueError, "authorization"):
                    review.create_session(video_id, run_id, "frame_001.jpg", [.1, .1, .9, .9])
                report.assert_not_called()
            with patch.object(batch, "get_engine") as engine:
                with self.assertRaisesRegex(ValueError, "authorization"):
                    batch.start_batch(video_id, ["frame_001.jpg"], run_id, ["frame_001.jpg"], str(uuid.uuid4()))
                engine.assert_not_called()


if __name__ == "__main__":
    unittest.main()
