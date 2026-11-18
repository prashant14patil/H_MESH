"""Experimental gate tests; these fixtures never run real inference."""
import tempfile
import time
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException

from backend.routes.video_mesh import ExperimentalBatchRequest, create_experimental_run
from backend.services import experimental_video_mesh as experimental
from backend.services import video_mesh_batch as batch
from backend.tests.test_video_mesh_batch import FakeCapture, FakeEngine, fake_report


class ExperimentalTests(unittest.TestCase):
    def test_route_requires_explicit_experimental_confirmation(self):
        request = ExperimentalBatchRequest(video_id=str(uuid.uuid4()), quality_run_id=str(uuid.uuid4()))
        with self.assertRaises(HTTPException) as error:
            create_experimental_run(request)
        self.assertEqual(error.exception.status_code, 400)

    def test_default_and_unknown_gates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture = FakeCapture(root)
            engine = FakeEngine(root)
            video_id, quality_id = str(uuid.uuid4()), str(uuid.uuid4())
            report = fake_report(capture, video_id, quality_id)
            unknown = report["frames"][1]
            unknown["checks"]["full_body"] = "unknown"
            inspection = {"filename": unknown["filename"], "source_sha256": unknown["source_sha256"],
                          "inspected": True, "warning_acknowledged": True}
            with patch.object(batch, "_RUN_ROOT", root / "runs"), \
                 patch.object(batch, "_EXPERIMENTAL_OUTPUT_ROOT", root / "exports"), \
                 patch.object(experimental, "require_authorization", return_value={"source_sha256": "synthetic"}), \
                 patch.object(experimental, "get_capture_engine", return_value=capture), \
                 patch.object(experimental, "get_report", return_value=report), \
                 patch.object(batch, "get_engine", return_value=engine):
                with self.assertRaisesRegex(ValueError, "inspection"):
                    experimental.start_experimental(video_id, quality_id, [{**inspection, "inspected": False}])
                with self.assertRaisesRegex(ValueError, "inspection"):
                    experimental.start_experimental(video_id, quality_id, [{**inspection, "warning_acknowledged": False}])
                with self.assertRaisesRegex(ValueError, "changed since operator inspection"):
                    experimental.start_experimental(video_id, quality_id, [{**inspection, "source_sha256": "bad"}])
                unknown["checks"]["sharpness"] = "fail"
                with self.assertRaisesRegex(ValueError, "failed"):
                    experimental.start_experimental(video_id, quality_id, [inspection])
                unknown["checks"]["sharpness"] = "pass"
                first = experimental.start_experimental(video_id, quality_id, [])
                self.assertEqual([f["filename"] for f in first["frames"]], ["frame_001.jpg"])
                self.assertNotIn("review_session_id", first)
                self.wait(first["run_id"])
                second = experimental.start_experimental(video_id, quality_id, [inspection])
                self.assertEqual(len(second["frames"]), 2)
                result = self.wait(second["run_id"])
                self.assertEqual([f["status"] for f in result["frames"]], ["completed", "failed"])
                self.assertTrue(Path(result["frames"][0]["export_path"]).is_file())
                self.assertEqual(engine.inputs[0], (root / "frame_001.jpg").read_bytes())

    def test_authorization_and_hash_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture = FakeCapture(root)
            video_id, quality_id = str(uuid.uuid4()), str(uuid.uuid4())
            report = fake_report(capture, video_id, quality_id)
            with patch.object(experimental, "require_authorization", side_effect=ValueError("consent required")):
                with self.assertRaisesRegex(ValueError, "consent"):
                    experimental.start_experimental(video_id, quality_id, [])
            with patch.object(experimental, "require_authorization", return_value={"source_sha256": "synthetic"}), \
                 patch.object(experimental, "get_capture_engine", return_value=capture), \
                 patch.object(experimental, "get_report", return_value=report):
                (root / "frame_001.jpg").write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "changed since quality"):
                    experimental.start_experimental(video_id, quality_id, [])

    def wait(self, run_id):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            run = batch.get_run(run_id)
            if run["status"] != "processing":
                return run
            time.sleep(.02)
        self.fail("Batch did not finish")


if __name__ == "__main__":
    unittest.main()
