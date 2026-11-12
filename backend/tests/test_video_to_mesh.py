"""Synthetic tests for batch inference client."""
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from scripts.video_to_mesh import run_batch, validate_obj


def fake_obj():
    return ("\n".join(f"v {i / 100} {(i % 10) / 10} 0" for i in range(100)) +
            "\n" + "\n".join("f 1 2 3" for _ in range(100)) + "\n").encode()


class Response:
    def __init__(self, data=None, content=b""):
        self.data, self.content = data, content

    def raise_for_status(self):
        pass

    def json(self):
        return self.data


class FakeSession:
    def __init__(self):
        self.jobs = 0

    def post(self, url, files, timeout):
        self.jobs += 1
        return Response({"job_id": str(self.jobs)})

    def get(self, url, timeout):
        if "/status/" in url:
            job = url.rsplit("/", 1)[-1]
            if job == "1":
                return Response({"status": "failed", "detail": "synthetic failure"})
            return Response({"status": "completed", "mesh_url": "/api/inference/mesh/" + job})
        return Response(content=fake_obj())


class BatchTests(unittest.TestCase):
    def test_partial_failure_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames = root / "capture" / "extracted_frames"
            frames.mkdir(parents=True)
            for i in (1, 2):
                cv2.imwrite(str(frames / f"frame_{i:03d}.jpg"), np.zeros((40, 40, 3), dtype=np.uint8))
            manifest = {"extracted_frame_count": 2, "frames": [
                {"filename": "frame_001.jpg", "label": "unknown", "reason": "test"},
                {"filename": "frame_002.jpg", "label": "unknown", "reason": "test"},
            ]}
            output = run_batch(root / "capture", manifest, "http://test", root / "out",
                               "test", session=FakeSession())
            result = json.loads((output / "results.json").read_text(encoding="utf-8"))
            self.assertEqual(result["successful_meshes"], 1)
            self.assertEqual([x["status"] for x in result["frames"]], ["failed", "completed"])
            self.assertTrue((output / "meshes" / "frame_002_unknown.obj").exists())
            self.assertTrue((output / "previews" / "frame_002_unknown.jpg").exists())
            with self.assertRaises(FileExistsError):
                run_batch(root / "capture", manifest, "http://test", root / "out",
                          "test", session=FakeSession())

    def test_invalid_obj(self):
        with self.assertRaises(ValueError):
            validate_obj(b"v nan 0 0\nf 1 2 3\n")
        with self.assertRaises(ValueError):
            validate_obj(fake_obj().replace(b"f 1 2 3", b"f 1 2 999"))

    def test_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = {"frames": [{"filename": "../frame_001.jpg", "label": "x", "reason": "test"}]}
            with self.assertRaises(ValueError):
                run_batch(directory, manifest, "http://test")


if __name__ == "__main__":
    unittest.main()
