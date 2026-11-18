"""Unvalidated viewpoint hints must preserve side ambiguity."""
import unittest
from unittest.mock import patch

from backend.services import viewpoint_estimator as viewpoint


def landmark(x, visibility=.99, z=0):
    return {"x": x, "y": .3, "z": z, "visibility": visibility, "visible": True}


class ViewpointTests(unittest.TestCase):
    def test_profile_with_equal_visibility_is_not_intermediate_or_direction(self):
        frame = {"person_detected": True, "frame": "frame_001", "landmarks": {
            "left_shoulder": landmark(.49), "right_shoulder": landmark(.51),
            "nose": landmark(.5, z=-.2),
        }}
        result = viewpoint._classify_frame(frame)
        self.assertEqual(result["viewpoint"], "side_ambiguous")
        self.assertEqual(result["reason"], "side_view_but_left_right_ambiguous")
        engine = unittest.mock.Mock()
        engine.get_summary.return_value = {"frames": [frame]}
        with patch.object(viewpoint, "get_pose_engine", return_value=engine):
            coverage = viewpoint.estimate_viewpoints("video")["coverage"]
        self.assertEqual(coverage["counts"]["side_ambiguous"], 1)
        self.assertEqual(coverage["counts"]["left_side"], 0)
        self.assertEqual(coverage["counts"]["right_side"], 0)
        self.assertEqual(coverage["status"], "unconfirmed")

    def _profile(self, left_z, right_z, nose_x):
        # Real MediaPipe output: both shoulders report ~1.0 visibility in profile.
        return {"person_detected": True, "landmarks": {
            "left_shoulder": landmark(.49, visibility=1.0, z=left_z),
            "right_shoulder": landmark(.51, visibility=.999, z=right_z),
            "nose": landmark(nose_x, z=-.15),
        }}

    def test_profile_side_resolved_from_depth_and_nose_despite_equal_visibility(self):
        left = viewpoint._classify_frame(self._profile(-.5, .2, .41))
        right = viewpoint._classify_frame(self._profile(.2, -.5, .60))
        self.assertEqual(left["viewpoint"], "left_side")
        self.assertEqual(right["viewpoint"], "right_side")

    def test_conflicting_side_cues_stay_ambiguous(self):
        result = viewpoint._classify_frame(self._profile(-.5, .2, .60))
        self.assertEqual(result["viewpoint"], "side_ambiguous")
        self.assertEqual(result["reason"], "side_view_left_right_cues_conflict")

    def test_foreshortened_turning_pose_is_intermediate(self):
        frame = {"person_detected": True, "landmarks": {
            "left_shoulder": {**landmark(.59, z=-.43), "y": .30},
            "right_shoulder": {**landmark(.49, z=.20), "y": .30},
            "left_hip": {**landmark(.57), "y": .48},
            "right_hip": {**landmark(.50), "y": .48},
            "nose": landmark(.47, z=-.4),
        }}
        self.assertEqual(viewpoint._classify_frame(frame)["viewpoint"], "intermediate")

    def test_missing_pose_remains_unknown(self):
        self.assertEqual(viewpoint._classify_frame({"person_detected": False})["viewpoint"], "unknown")
        self.assertEqual(viewpoint._classify_frame({"person_detected": True, "landmarks": {}})["viewpoint"], "unknown")


if __name__ == "__main__":
    unittest.main()
