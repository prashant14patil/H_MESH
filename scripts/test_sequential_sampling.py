"""Regression tests for forward-only temporal frame sampling.

Run from the project root: python -m unittest scripts.test_sequential_sampling
"""
import unittest

from backend.services.video_processor import VideoCaptureEngine


class FakeCapture:
    def __init__(self, timestamps_ms):
        self.timestamps = timestamps_ms
        self.index = -1
        self.seeks = 0

    def read(self):
        self.index += 1
        if self.index >= len(self.timestamps):
            return False, None
        return True, self.index

    def get(self, prop):
        return self.timestamps[self.index]

    def set(self, prop, value):
        self.seeks += 1
        raise AssertionError("Sequential sampling must not seek")


class SequentialSamplingTests(unittest.TestCase):
    def test_uses_actual_timestamps_without_seeking(self):
        cap = FakeCapture([0, 100, 240, 310, 500, 610, 900])
        samples = list(VideoCaptureEngine._sample_sequentially(cap, [0, 0.3, 0.6, 0.9], 10))
        self.assertEqual([s[2] for s in samples], [0, 3, 5, 6])
        self.assertEqual([s[1] for s in samples], [0, 0.31, 0.61, 0.9])
        self.assertEqual(cap.seeks, 0)

    def test_reports_all_remaining_candidates_at_eof(self):
        cap = FakeCapture([0, 100])
        samples = list(VideoCaptureEngine._sample_sequentially(cap, [0, 0.5, 1.0], 10))
        self.assertEqual(samples[0], (0, 0.0, 0))
        self.assertEqual(samples[1:], [(0.5, None, None), (1.0, None, None)])

    def test_falls_back_when_timestamps_are_unavailable(self):
        cap = FakeCapture([0, 0, 0, 0, 0])
        samples = list(VideoCaptureEngine._sample_sequentially(cap, [0, 0.2, 0.4], 10))
        self.assertEqual([s[2] for s in samples], [0, 2, 4])
        self.assertEqual([s[1] for s in samples], [0, 0.2, 0.4])


if __name__ == "__main__":
    unittest.main()
