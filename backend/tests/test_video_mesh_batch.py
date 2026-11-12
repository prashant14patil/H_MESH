"""Synthetic tests for the isolated video mesh batch service."""
import tempfile
import time
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import backend.services.video_mesh_batch as batch


class FakeCapture:
    def __init__(self, root):
        self.root = root
        self.frames = [
            {"filename": "frame_001.jpg", "timestamp_sec": 0.0, "is_blurry": False},
            {"filename": "frame_002.jpg", "timestamp_sec": 1.0, "is_blurry": True},
        ]
        for frame in self.frames:
            (root / frame["filename"]).write_bytes(b"\xff\xd8\xffsynthetic")

    def get_metadata(self, video_id):
        return {"status": "completed", "frames": self.frames}

    def get_frame_path(self, video_id, filename):
        path = self.root / filename
        return path if path.exists() else None


class FakeEngine:
    is_ready = True

    def __init__(self, root):
        self.root = root
        self.jobs = {}
        self.counter = 0

    def submit_job(self, job_id, image_bytes):
        self.counter += 1
        if self.counter % 2 == 1:
            self.jobs[job_id] = {"status": "failed", "detail": "test failure"}
        else:
            mesh = self.root / (job_id + ".obj")
            mesh.write_text("v 0 0 0\n", encoding="utf-8")
            self.jobs[job_id] = {"status": "completed", "mesh_path": str(mesh),
                                 "metadata": {"vertices": 6890, "faces": 13776}}

    def get_job(self, job_id):
        return self.jobs.get(job_id)


class BatchTests(unittest.TestCase):
    def test_partial_success_snapshots_and_previous_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture = FakeCapture(root)
            engine = FakeEngine(root)
            with patch.object(batch, "_RUN_ROOT", root / "runs"), \
                 patch.object(batch, "get_capture_engine", return_value=capture), \
                 patch.object(batch, "get_engine", return_value=engine):
                video_id = str(uuid.uuid4())
                names = [f["filename"] for f in capture.frames]
                first = batch.start_batch(video_id, names)
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    result = batch.get_run(first["run_id"])
                    if result["status"] != "processing":
                        break
                    time.sleep(.02)
                self.assertEqual(result["status"], "completed_with_errors")
                self.assertEqual([f["status"] for f in result["frames"]], ["failed", "completed"])
                mesh = batch.get_asset(first["run_id"], names[1], "mesh")
                self.assertTrue(mesh.is_file())
                self.assertIsNone(batch.get_asset(first["run_id"], names[0], "mesh"))
                self.assertEqual(batch.get_asset(first["run_id"], names[1], "frame").read_bytes(),
                                 b"\xff\xd8\xffsynthetic")
                previous = mesh.read_bytes()
                second = batch.start_batch(video_id, names)
                self.assertNotEqual(first["run_id"], second["run_id"])
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline and batch.get_run(second["run_id"])["status"] == "processing":
                    time.sleep(.02)
                self.assertEqual(mesh.read_bytes(), previous)
                self.assertEqual(batch.get_run(first["run_id"])["status"], "completed_with_errors")

    def test_invalid_identifiers_and_selection(self):
        with self.assertRaises(ValueError):
            batch.start_batch("../bad", ["frame_001.jpg"])
        video_id = str(uuid.uuid4())
        for names in ([], ["../frame_001.jpg"], ["frame_001.jpg"] * 2,
                      ["frame_001.jpg", "frame_001.png"], ["frame_001.jpg"] * 13):
            with self.subTest(names=names), self.assertRaises(ValueError):
                batch.start_batch(video_id, names)
        with self.assertRaises(ValueError):
            batch.get_asset("../bad", "frame_001.jpg", "mesh")


if __name__ == "__main__":
    unittest.main()
