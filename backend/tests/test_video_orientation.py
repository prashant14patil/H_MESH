"""Video decoder orientation and stale pose regression tests."""
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from backend.services import pose_processor, video_processor


class FakeCapture:
    def __init__(self, angle, supported=True):
        self.angle = angle
        self.auto = 0
        self.supported = supported
        self.released = False

    def isOpened(self):
        return True

    def get(self, prop):
        if prop == cv2.CAP_PROP_ORIENTATION_META:
            return self.angle
        if prop == cv2.CAP_PROP_ORIENTATION_AUTO:
            return self.auto
        return 0

    def set(self, prop, value):
        if not self.supported:
            return False
        self.auto = value
        return True

    def release(self):
        self.released = True


class VideoOrientationTests(unittest.TestCase):
    def test_decoder_autorotation_enabled_once_for_all_rotations(self):
        for angle in (0, 90, 180, 270):
            with self.subTest(angle=angle):
                cap = FakeCapture(angle)
                with patch.object(cv2, "VideoCapture", return_value=cap) as create:
                    result, rotation = video_processor.VideoCaptureEngine._open_oriented_capture("test.mov")
                create.assert_called_once_with("test.mov", cv2.CAP_FFMPEG)
                self.assertIs(result, cap)
                self.assertEqual(rotation, angle)
                self.assertEqual(cap.auto, 1)

    def test_unsupported_autorotation_fails_closed(self):
        cap = FakeCapture(90, supported=False)
        with patch.object(cv2, "VideoCapture", return_value=cap):
            with self.assertRaisesRegex(video_processor.VideoProcessingError, "autorotation"):
                video_processor.VideoCaptureEngine._open_oriented_capture("test.mov")
        self.assertTrue(cap.released)

    def test_unrotated_real_video_preserves_pixel_orientation_and_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "source.avi"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (80, 120))
            self.assertTrue(writer.isOpened())
            for i in range(10):
                image = np.zeros((120, 80, 3), np.uint8)
                image[:40, :] = (0, 0, 255)
                image[80:, :] = (255, 0, 0)
                cv2.putText(image, str(i), (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                writer.write(image)
            writer.release()
            with patch.object(video_processor, "_OUTPUT_ROOT", root / "captures"), \
                 patch.object(video_processor, "_TARGET_FRAME_COUNT", 1):
                engine = video_processor.VideoCaptureEngine()
                engine._extract_frames("test", str(video))
                meta = engine.get_metadata("test")
                frame = meta["frames"][0]
                path = engine.get_frame_path("test", frame["filename"])
                image = cv2.imread(str(path))
                self.assertEqual(image.shape[:2], (120, 80))
                self.assertGreater(int(image[10, 10, 2]), int(image[10, 10, 0]))
                self.assertGreater(int(image[110, 10, 0]), int(image[110, 10, 2]))
                self.assertEqual(frame["source_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
                self.assertEqual((meta["video_info"]["width"], meta["video_info"]["height"]), (80, 120))

    def test_ffmpeg_display_matrix_fixtures_match_autorotated_pixels(self):
        ffmpeg = os.environ.get("TEST_FFMPEG") or shutil.which("ffmpeg")
        if not ffmpeg:
            self.skipTest("ffmpeg is required for display-matrix integration fixtures")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = np.zeros((96, 160, 3), np.uint8)
            image[:48, :80] = (0, 0, 255)
            image[:48, 80:] = (0, 255, 0)
            image[48:, :80] = (255, 0, 0)
            image[48:, 80:] = (0, 255, 255)
            cv2.imwrite(str(root / "input.png"), image)
            source = root / "source.mp4"
            subprocess.run([ffmpeg, "-v", "error", "-loop", "1", "-i", str(root / "input.png"),
                            "-t", "1", "-r", "10", "-pix_fmt", "yuv420p", "-c:v", "libx264",
                            str(source)], check=True)
            for angle in (0, 90, 180, 270):
                with self.subTest(angle=angle):
                    video = root / f"rotation-{angle}.mp4"
                    subprocess.run([ffmpeg, "-v", "error", "-i", str(source), "-c", "copy",
                                    "-metadata:s:v:0", f"rotate={angle}", str(video)], check=True)
                    reference = root / f"reference-{angle}.png"
                    subprocess.run([ffmpeg, "-v", "error", "-i", str(video), "-frames:v", "1",
                                    str(reference)], check=True)
                    expected = cv2.imread(str(reference))
                    with patch.object(video_processor, "_OUTPUT_ROOT", root / "captures"), \
                         patch.object(video_processor, "_TARGET_FRAME_COUNT", 1):
                        engine = video_processor.VideoCaptureEngine()
                        engine._extract_frames(str(angle), str(video))
                        meta = engine.get_metadata(str(angle))
                        actual = cv2.imread(str(engine.get_frame_path(str(angle), "frame_001.jpg")))
                    self.assertEqual(actual.shape, expected.shape)
                    self.assertLess(np.mean(np.abs(actual.astype(float) - expected.astype(float))), 5)
                    self.assertEqual((meta["video_info"]["width"], meta["video_info"]["height"]),
                                     (actual.shape[1], actual.shape[0]))

    def test_stale_pose_summary_and_assets_are_not_served(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "video"
            frames = folder / "extracted_frames"
            pose = folder / "processed_frames"
            frames.mkdir(parents=True)
            (pose / "frame_001").mkdir(parents=True)
            data = b"new frame"
            (frames / "frame_001.jpg").write_bytes(data)
            digest = hashlib.sha256(data).hexdigest()
            (folder / "metadata.json").write_text(json.dumps({"frames": [
                {"filename": "frame_001.jpg", "source_sha256": digest}]}))
            summary = {"frames": [{"frame": "frame_001", "source_sha256": "old"}]}
            (pose / "processing_summary.json").write_text(json.dumps(summary))
            (pose / "frame_001" / "metadata.json").write_text("{}")
            with patch.object(pose_processor, "_CAPTURE_ROOT", root):
                engine = pose_processor.PoseProcessingEngine()
                self.assertIsNone(engine.get_summary("video"))
                self.assertIsNone(engine.get_frame_result_path("video", "frame_001", "metadata.json"))
                summary["frames"][0]["source_sha256"] = digest
                (pose / "processing_summary.json").write_text(json.dumps(summary))
                self.assertEqual(engine.get_summary("video"), summary)
                self.assertIsNotNone(engine.get_frame_result_path("video", "frame_001", "metadata.json"))


if __name__ == "__main__":
    unittest.main()
