"""Synthetic tests for the isolated video mesh batch service."""
import hashlib
import io
import tempfile
import time
import unittest
import uuid
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from PIL import Image
import backend.services.video_mesh_batch as batch
from backend.services.capture_quality import QualityConfig


class FakeCapture:
    def __init__(self, root):
        self.root = root
        self.frames = [
            {"filename": "frame_001.jpg", "timestamp_sec": 0.0, "is_blurry": False},
            {"filename": "frame_002.jpg", "timestamp_sec": 1.0, "is_blurry": False},
        ]
        for frame in self.frames:
            Image.new("RGB", (100, 100), (120, 80, 40)).save(root / frame["filename"])

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
        self.inputs = []

    def submit_job(self, job_id, image_bytes):
        self.inputs.append(image_bytes)
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


def fake_report(capture, video_id, run_id):
    return {"video_id": video_id, "run_id": run_id, "thresholds": asdict(QualityConfig()), "frames": [
        {"filename": f["filename"], "status": "review", "rejections": [], "warnings": [],
         "source_sha256": hashlib.sha256((capture.root / f["filename"]).read_bytes()).hexdigest(),
         "checks": {key: "pass" for key in
                    ("readability", "person_presence", "sharpness", "lighting", "full_body")},
         "landmark_bbox_normalized": [.2, .2, .8, .8],
         "person_count_observed": 1, "subject_association": "tentative_pose_detector_match"}
        for f in capture.frames
    ]}


def fake_session(capture, video_id, quality_id, session_id):
    name = capture.frames[0]["filename"]
    return {"session_id": session_id, "video_id": video_id, "quality_run_id": quality_id,
            "authorization_source_sha256": "synthetic",
            "reference_frame": name,
            "reference_frame_sha256": hashlib.sha256((capture.root / name).read_bytes()).hexdigest()}


def fake_decision(capture, video_id, quality_id, session_id, name):
    return {"decision_id": str(uuid.uuid4()), "recording_id": video_id,
            "quality_run_id": quality_id, "session_id": session_id, "frame_id": name,
            "frame_sha256": hashlib.sha256((capture.root / name).read_bytes()).hexdigest(),
            "reviewer_decision": "confirm", "identity_status": "human_confirmed_not_biometric",
            "selected_subject_box": [.1, .1, .9, .9]}


@patch.object(batch, "require_authorization", lambda video_id: {"source_sha256": "synthetic"})
class BatchTests(unittest.TestCase):
    def test_partial_success_snapshots_crops_and_previous_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture = FakeCapture(root)
            engine = FakeEngine(root)
            video_id, quality_id, session_id = (str(uuid.uuid4()) for _ in range(3))
            report = fake_report(capture, video_id, quality_id)
            session = fake_session(capture, video_id, quality_id, session_id)
            names = [f["filename"] for f in capture.frames]
            decisions = {name: fake_decision(capture, video_id, quality_id, session_id, name) for name in names}
            with patch.object(batch, "_RUN_ROOT", root / "runs"), \
                 patch.object(batch, "get_capture_engine", return_value=capture), \
                 patch.object(batch, "get_engine", return_value=engine), \
                 patch.object(batch, "get_report", return_value=report), \
                 patch.object(batch, "get_session", return_value=session), \
                 patch.object(batch, "get_decision", side_effect=lambda sid, name: decisions[name]):
                first = batch.start_batch(video_id, names, quality_id, names, session_id)
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
                self.assertEqual(batch.get_asset(first["run_id"], names[1], "frame").read_bytes(),
                                 (root / names[1]).read_bytes())
                self.assertEqual(Image.open(io.BytesIO(engine.inputs[0])).size, (80, 80))
                self.assertEqual(engine.inputs[0][:8], b"\x89PNG\r\n\x1a\n")
                previous = mesh.read_bytes()
                second = batch.start_batch(video_id, names, quality_id, names, session_id)
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline and batch.get_run(second["run_id"])["status"] == "processing":
                    time.sleep(.02)
                self.assertNotEqual(first["run_id"], second["run_id"])
                self.assertEqual(mesh.read_bytes(), previous)

    def test_gate_rejects_missing_review_failed_quality_and_stale_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture = FakeCapture(root)
            engine = FakeEngine(root)
            video_id, quality_id, session_id = (str(uuid.uuid4()) for _ in range(3))
            report = fake_report(capture, video_id, quality_id)
            session = fake_session(capture, video_id, quality_id, session_id)
            name = capture.frames[0]["filename"]
            decision = fake_decision(capture, video_id, quality_id, session_id, name)
            with patch.object(batch, "_RUN_ROOT", root / "runs"), \
                 patch.object(batch, "get_capture_engine", return_value=capture), \
                 patch.object(batch, "get_engine", return_value=engine), \
                 patch.object(batch, "get_report", return_value=report), \
                 patch.object(batch, "get_session", return_value=session), \
                 patch.object(batch, "get_decision", return_value=decision):
                with self.assertRaises(ValueError):
                    batch.start_batch(video_id, [name])
                with self.assertRaises(ValueError):
                    batch.start_batch(video_id, [name], quality_id, [], session_id)
                decision["reviewer_decision"] = "reject"
                with self.assertRaisesRegex(ValueError, "review"):
                    batch.start_batch(video_id, [name], quality_id, [name], session_id)
                decision["reviewer_decision"] = "confirm"
                report["thresholds"]["min_laplacian"] = 0
                with self.assertRaisesRegex(ValueError, "default server screening policy"):
                    batch.start_batch(video_id, [name], quality_id, [name], session_id)
                report["thresholds"] = asdict(QualityConfig())
                report["frames"][0]["detector_evidence"] = [
                    {"box": [.1, .1, .9, .9]}, {"box": [.2, .2, .8, .8]}
                ]
                with self.assertRaisesRegex(ValueError, "multiple detected"):
                    batch.start_batch(video_id, [name], quality_id, [name], session_id)
                report["frames"][0]["detector_evidence"] = []
                report["frames"][0]["checks"]["sharpness"] = "warning"
                with self.assertRaisesRegex(ValueError, "quality"):
                    batch.start_batch(video_id, [name], quality_id, [name], session_id)
                report["frames"][0]["checks"]["sharpness"] = "pass"
                Image.new("RGB", (100, 100), "blue").save(root / name)
                with self.assertRaisesRegex(ValueError, "changed"):
                    batch.start_batch(video_id, [name], quality_id, [name], session_id)
                self.assertEqual(engine.counter, 0)

    def test_readiness_reports_multiple_blockers_and_is_non_mutating(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture = FakeCapture(root)
            video_id, quality_id = str(uuid.uuid4()), str(uuid.uuid4())
            report = fake_report(capture, video_id, quality_id)
            report["frames"][0]["checks"]["full_body"] = "unknown"
            with patch.object(batch, "_RUN_ROOT", root / "runs"), \
                 patch.object(batch, "get_capture_engine", return_value=capture), \
                 patch.object(batch, "get_report", return_value=report), \
                 patch.object(batch.inference_engine, "_engine", None), \
                 patch.object(batch, "get_engine") as get_engine:
                result = batch.check_readiness(video_id, ["frame_001.jpg"], quality_id, [], None)
                self.assertFalse(result["ready"])
                self.assertTrue(any("acknowledgement" in b for b in result["blockers"]))
                self.assertTrue(any("reference subject" in b for b in result["blockers"]))
                self.assertTrue(any("mandatory quality" in b for b in result["blockers"]))
                self.assertTrue(any("not ready" in b for b in result["blockers"]))
                self.assertTrue(any("decisions" in b for b in result["unevaluated"]))
                self.assertFalse((root / "runs").exists())
                get_engine.assert_not_called()

    def test_readiness_reports_crop_size_alongside_other_blockers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture = FakeCapture(root)
            Image.new("RGB", (20, 20), "blue").save(root / "frame_002.jpg")
            video_id, quality_id, session_id = (str(uuid.uuid4()) for _ in range(3))
            report = fake_report(capture, video_id, quality_id)
            report["frames"][0]["checks"]["sharpness"] = "warning"
            session = fake_session(capture, video_id, quality_id, session_id)
            decision = fake_decision(capture, video_id, quality_id, session_id, "frame_002.jpg")
            engine = FakeEngine(root)
            with patch.object(batch, "_RUN_ROOT", root / "runs"), \
                 patch.object(batch, "get_capture_engine", return_value=capture), \
                 patch.object(batch, "get_report", return_value=report), \
                 patch.object(batch, "get_session", return_value=session), \
                 patch.object(batch, "get_decision", side_effect=lambda sid, name: decision if name == "frame_002.jpg" else None), \
                 patch.object(batch.inference_engine, "_engine", engine):
                result = batch.check_readiness(video_id, ["frame_001.jpg", "frame_002.jpg"],
                                               quality_id, [], session_id)
                self.assertFalse(result["ready"])
                self.assertTrue(any("acknowledgement" in b for b in result["blockers"]))
                self.assertTrue(any("frame_001.jpg" in b and "quality" in b for b in result["blockers"]))
                self.assertTrue(any("frame_001.jpg" in b and "subject review" in b for b in result["blockers"]))
                self.assertTrue(any("frame_002.jpg" in b and "too small" in b for b in result["blockers"]))
                self.assertTrue(any("frame_001.jpg" in b and "crop size" in b for b in result["unevaluated"]))
                self.assertFalse((root / "runs").exists())
                self.assertEqual(engine.counter, 0)

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
