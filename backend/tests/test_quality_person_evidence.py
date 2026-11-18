"""Person-count and full-body evidence from the keypoint detector (synthetic, no model needed)."""
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.services import capture_quality_job as job
from backend.services.capture_quality import RCNN_METHOD, analyze_frames
from backend.tests.test_capture_quality import fixture_pose

BOX = [.25, .08, .75, .95]


def rcnn(box=BOX, conf=1.0, score=0.99):
    return {"box": list(box), "score": score, "method": RCNN_METHOD,
            "keypoints": [[.5, .5]] * 17, "keypoint_confidence": [conf] * 17}


def side_view_pose(name, hide=("right_elbow", "right_wrist", "right_knee", "right_ankle")):
    pose = fixture_pose(name)
    pose["full_body_visible"] = False
    for key in hide:
        pose["landmarks"][key].update(visibility=0.05, visible=False)
    return pose


def run(poses, detections, method):
    with tempfile.TemporaryDirectory() as temp:
        folder = Path(temp)
        frames = []
        sharp = np.asarray(np.indices((240, 320)).sum(axis=0) % 2 * 180 + 35, dtype=np.uint8)
        for pose in poses:
            name = pose["frame"] + ".jpg"
            assert cv2.imwrite(str(folder / name), sharp)
            frames.append({"filename": name, "is_blurry": False, "blur_score": 200, "diff_score": None})
        return analyze_frames({"frames": frames}, {"frames": poses}, folder, detections=detections,
                              detector_method=method)["frames"]


def test_single_keypoint_detection_passes_person_count_but_not_identity():
    f = run([fixture_pose("frame_001")], {"frame_001.jpg": [rcnn()]}, RCNN_METHOD)[0]
    assert f["checks"]["multiple_people"] == "pass"
    assert f["checks"]["identity"] == "unknown"           # never auto-verified
    assert "person_count_and_identity_unverified" in f["warnings"]
    assert f["person_detection_confidence"] == pytest.approx(0.99)


def test_two_keypoint_detections_fail_person_count():
    f = run([fixture_pose("frame_001")], {"frame_001.jpg": [rcnn(), rcnn([.0, .1, .2, .9])]}, RCNN_METHOD)[0]
    assert f["checks"]["multiple_people"] == "fail"
    assert "possible_multiple_people" in f["warnings"]


def test_hog_single_candidate_never_proves_one_person():
    f = run([fixture_pose("frame_001")], {"frame_001.jpg": [{"box": BOX, "score": 1.2}]}, "opencv_hog_uncalibrated")[0]
    assert f["checks"]["multiple_people"] == "unknown"


def test_side_view_self_occlusion_passes_with_detector_confirmation():
    f = run([side_view_pose("frame_001")], {"frame_001.jpg": [rcnn()]}, RCNN_METHOD)[0]
    assert f["checks"]["full_body"] == "pass"
    assert f["full_body_basis"] == "in_frame_one_sided_self_occlusion_detector_confirmed"
    assert "full_body_self_occlusion_one_side" in f["warnings"]


@pytest.mark.parametrize("case", ["no_detector", "hog_only", "both_sides_hidden", "body_cut_off",
                                  "feet_not_detected", "detector_box_at_edge"])
def test_full_body_stays_unknown_without_sufficient_evidence(case):
    pose = side_view_pose("frame_001")
    dets = [rcnn()]
    method = RCNN_METHOD
    if case == "no_detector":
        dets, method = None, None
    elif case == "hog_only":
        dets, method = [{"box": BOX, "score": 1.0}], "opencv_hog_uncalibrated"
    elif case == "both_sides_hidden":   # left AND right ankle hidden: could be truncation
        pose = side_view_pose("frame_001", hide=("right_ankle", "left_ankle"))
    elif case == "body_cut_off":        # a landmark lies outside the frame margin
        pose["landmarks"]["right_ankle"]["y"] = 0.995
    elif case == "feet_not_detected":
        d = rcnn()
        d["keypoint_confidence"][15] = d["keypoint_confidence"][16] = 0.1
        dets = [d]
    elif case == "detector_box_at_edge":
        dets = [rcnn([.25, .08, .75, 1.0])]
    f = run([pose], {"frame_001.jpg": dets} if dets is not None else None, method)[0]
    assert f["checks"]["full_body"] == "unknown", case
    assert "full_body_visibility_unconfirmed" in f["warnings"]


def test_person_observations_prefers_rcnn_and_reports_fallback(monkeypatch):
    monkeypatch.setattr(job, "_keypoint_rcnn_observations", lambda paths: {"a.jpg": [rcnn()]})
    obs, status, method = job._person_observations({"a.jpg": Path("a.jpg")}, True)
    assert method == RCNN_METHOD and status == "keypoint_rcnn_unvalidated"

    def broken(paths):
        raise RuntimeError("weights download failed")
    monkeypatch.setattr(job, "_keypoint_rcnn_observations", broken)
    monkeypatch.setattr(job, "_hog_observations", lambda paths: {"a.jpg": []})
    obs, status, method = job._person_observations({"a.jpg": Path("a.jpg")}, True)
    assert method == job.HOG_METHOD and "keypoint_rcnn_unavailable" in status and "weights download failed" in status
    assert job._person_observations({}, False) == (None, "disabled", None)
