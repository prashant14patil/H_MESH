"""HTTP tests against an ephemeral server with a fake inference engine."""
import socket
import tempfile
import threading
import time
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import requests
import uvicorn
from fastapi import FastAPI

import backend.services.video_mesh_batch as batch
from backend.routes.video_mesh import router
from backend.tests.test_video_mesh_batch import FakeCapture, FakeEngine


class VideoMeshApiTests(unittest.TestCase):
    def test_run_status_assets_and_input_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture, engine = FakeCapture(root), FakeEngine(root)
            app = FastAPI()
            app.include_router(router, prefix="/api/video-mesh")
            sock = socket.socket()
            sock.bind(("127.0.0.1", 0))
            sock.listen(128)
            base = f"http://127.0.0.1:{sock.getsockname()[1]}/api/video-mesh"
            server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="off"))
            thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
            with patch.object(batch, "_RUN_ROOT", root / "runs"), \
                 patch.object(batch, "get_capture_engine", return_value=capture), \
                 patch.object(batch, "get_engine", return_value=engine):
                thread.start()
                try:
                    deadline = time.monotonic() + 5
                    while not server.started and time.monotonic() < deadline:
                        time.sleep(.02)
                    self.assertTrue(server.started)
                    invalid = requests.post(base + "/runs", json={
                        "video_id": "../invalid", "frames": ["frame_001.jpg"]}, timeout=5)
                    self.assertEqual(invalid.status_code, 400)
                    started = requests.post(base + "/runs", json={
                        "video_id": str(uuid.uuid4()),
                        "frames": ["frame_001.jpg", "frame_002.jpg"]}, timeout=5)
                    self.assertEqual(started.status_code, 202)
                    run_id = started.json()["run_id"]
                    deadline = time.monotonic() + 5
                    while time.monotonic() < deadline:
                        response = requests.get(f"{base}/runs/{run_id}", timeout=5)
                        self.assertEqual(response.status_code, 200)
                        result = response.json()
                        if result["status"] != "processing":
                            break
                        time.sleep(.02)
                    self.assertEqual(result["status"], "completed_with_errors")
                    self.assertEqual([f["status"] for f in result["frames"]], ["failed", "completed"])
                    self.assertEqual(requests.get(f"{base}/runs/{run_id}/frame/frame_002.jpg", timeout=5).status_code, 200)
                    self.assertEqual(requests.get(f"{base}/runs/{run_id}/mesh/frame_001.jpg", timeout=5).status_code, 404)
                    mesh = requests.get(f"{base}/runs/{run_id}/mesh/frame_002.jpg", timeout=5)
                    self.assertEqual(mesh.status_code, 200)
                    self.assertTrue(mesh.content.startswith(b"v "))
                    self.assertEqual(requests.get(f"{base}/runs/not-a-uuid", timeout=5).status_code, 400)
                finally:
                    server.should_exit = True
                    thread.join(timeout=5)
                    sock.close()


if __name__ == "__main__":
    unittest.main()
