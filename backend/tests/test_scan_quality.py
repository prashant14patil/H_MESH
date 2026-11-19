import numpy as np
import pytest

from backend.services import scan_quality as sq
from backend.services import scan_report


def _turn(n=60, seconds=20.0, total=360.0, start=0.0):
    t = np.linspace(0, seconds, n)
    return t, sq.wrap_deg(start + np.linspace(0, total, n, endpoint=False))


def test_trajectory_measures_full_turn_speed_and_direction():
    t, y = _turn()
    tr = sq.rotation_trajectory(t, y)
    assert tr["total_rotation_deg"] == pytest.approx(354, abs=2)
    assert tr["direction"].startswith("counter_clockwise")
    assert tr["estimated_seconds_per_full_turn"] == pytest.approx(20.3, rel=0.05)
    assert tr["outliers"] == [] and tr["reversal_steps"] == 0


def test_trajectory_flags_front_back_flip_and_it_cannot_fake_rotation():
    t, y = _turn()
    y = y.copy()
    y[20] = sq.wrap_deg(y[20] + 180)
    tr = sq.rotation_trajectory(t, y)
    assert tr["outliers"] == [20]
    assert tr["outlier_mask"][20] and abs(tr["residual_deg"][20]) > 170
    assert tr["total_rotation_deg"] == pytest.approx(354, abs=2)
    # A half turn with flips must not look like a full turn.
    t2, y2 = _turn(total=180)
    y2 = y2.copy()
    y2[[10, 30, 50]] = sq.wrap_deg(y2[[10, 30, 50]] + 180)
    assert sq.rotation_trajectory(t2, y2)["total_rotation_deg"] < 200


def test_trajectory_keeps_callers_order():
    t, y = _turn(n=30)
    perm = np.random.default_rng(0).permutation(30)
    y = y.copy()
    y[7] = sq.wrap_deg(y[7] + 180)
    tr = sq.rotation_trajectory(t[perm], y[perm])
    assert [int(perm[i]) for i in np.flatnonzero(tr["outlier_mask"])] == [7]


def test_coverage_finds_missing_back_arc():
    yaws = [a for a in range(-180, 180, 10) if not (a >= 120 or a <= -130)]
    cov = sq.angular_coverage(yaws)
    assert cov["largest_gap_deg"] == pytest.approx(130, abs=1)
    assert abs(cov["largest_gap_centre_deg"]) > 160
    assert cov["missing_arcs_deg"] and cov["coverage_fraction"] < 0.75
    assert sq.describe_heading(cov["largest_gap_centre_deg"]) == "back"
    full = sq.angular_coverage(range(-180, 180, 10))
    assert full["missing_arcs_deg"] == [] and full["largest_gap_deg"] == pytest.approx(10)


def _capture(**over):
    t, y = _turn()
    tr = sq.rotation_trajectory(t, y)
    boxes = [[400, 100, 700, 1800]] * 60
    kw = dict(image_wh=(1080, 1920), frames_decoded=60, tracked_boxes=boxes,
              tracking={"frames_tracked": 60, "identity_switches": 0, "other_person_detections": 0},
              usable_frames=55, lighting=[{"mean": 120, "std": 45, "dark_frac": 0.0, "bright_frac": 0.01}] * 60,
              bg_shift=[None] + [0.001] * 59, trajectory=tr, coverage_usable=sq.angular_coverage(y),
              coverage_selected=sq.angular_coverage(y[::3]), selected=20, usable_masks=18, blurry=2, hmr2_frames=55)
    kw.update(over)
    return sq.assess_capture(**kw)


def test_good_capture_passes_without_feedback():
    q = _capture()
    assert q["verdict"] == "good", q
    assert q["feedback"] == []


def test_poor_capture_gives_actionable_feedback():
    t, y = _turn(total=200, seconds=4)
    tr = sq.rotation_trajectory(t, y)
    q = _capture(trajectory=tr, coverage_usable=sq.angular_coverage(y),
                 lighting=[{"mean": 30, "std": 10, "dark_frac": 0.3, "bright_frac": 0.0}] * 60,
                 bg_shift=[None] + list(np.linspace(0, 0.05, 59)),
                 tracked_boxes=[[400, 900, 500, 1200]] * 60, selected=5, usable_masks=2)
    assert q["verdict"] == "insufficient"
    c = q["checks"]
    for name in ("angular_coverage", "resolution", "lighting", "camera_consistency", "useful_frames"):
        assert c[name]["status"] == "fail", name
        assert c[name]["feedback"]
    assert c["rotation_steadiness"]["status"] == "warn"
    text = " ".join(q["feedback"])
    assert "tripod" in text and "closer" in text and "360" in text


def test_lighting_stats_and_background_motion():
    img = np.full((400, 300, 3), 128, np.uint8)
    s = sq.lighting_stats(img, [50, 50, 250, 350])
    assert s["mean"] == pytest.approx(128) and s["dark_frac"] == 0
    rng = np.random.default_rng(1)
    scene = (rng.random((400, 600)) * 255).astype(np.uint8)
    scene = np.dstack([scene] * 3)
    bm = sq.BackgroundMotion(width=300)
    assert bm.shift_frac(scene, [250, 100, 350, 380]) is None
    assert bm.shift_frac(scene, [250, 100, 350, 380]) < 0.002
    moved = np.roll(scene, 30, axis=1)
    assert bm.shift_frac(moved, [250, 100, 350, 380]) == pytest.approx(30 / 600, abs=0.01)


def test_reconstruction_assessment_and_html(tmp_path):
    report = {
        "run_id": "r", "video_id": "v",
        "geometry_validation": {"valid": True, "watertight": True, "vertex_count": 6890, "face_count": 13776},
        "validation_summary": {"landmark_reprojection_error_pct_height": 1.2, "frames_in_final_fit": 20,
                               "frames_extracted": 120, "circumference_sections_valid": "12/12"},
        "silhouette_fitting": {"summary_by_method": {"silhouette_keypoint_shared_shape (new)": {
            "iou_mean": 0.88, "protrusion_frac_mean": 0.016}},
            "proportion_check": {"side_depth": {"median_ratio": [0.95, 0.97]},
                                 "front_back_width": {"median_ratio": [0.93, 0.96]}}},
        "multiview_optimisation": {"held_out_validation": {"results": {
            "silhouette_keypoint_shared_shape (new)": {"iou_mean": 0.82, "protrusion_frac_mean": 0.02}}}},
        "shared_shape": {"betas": [0.5, -1.0, 0.2]}, "measurements": {"valid": 12, "total": 12},
        "timings_sec": {"total": 300.0}, "limitations": ["relative scale"], "warnings": []}
    q = sq.assess_reconstruction(report)
    assert q["verdict"] == "consistent", q
    report["silhouette_fitting"]["summary_by_method"]["silhouette_keypoint_shared_shape (new)"]["protrusion_frac_mean"] = 0.08
    assert sq.assess_reconstruction(report)["checks"]["outline_protrusion"]["status"] == "fail"

    import json
    (tmp_path / "reconstruction_report.json").write_text(json.dumps(report))
    (tmp_path / "reconstruction_quality_report.json").write_text(json.dumps(q))
    (tmp_path / "capture_quality_report.json").write_text(json.dumps(_capture()))
    (tmp_path / "measurements.json").write_text(json.dumps({"mesh_height": 1.7, "measurements": {
        "chest": {"valid": True, "tape_circumference": 1.0, "tape_circumference_over_height": 0.588}},
        "previous_method_keypoint_only": {"chest": 0.6}}))
    page = scan_report.write_html(tmp_path).read_text(encoding="utf-8")
    assert "0.588" in page and "relative" in page and "<script" not in page
    assert "Capture quality" in page and "Reconstruction quality" in page


def test_drift_uses_anatomical_anchors_not_box_edges():
    # Arms swing the box from narrow (profile) to wide (frontal) and the lowest foot steps: no real drift.
    boxes = [[540 - w / 2, 500, 540 + w / 2, 1620 + 40 * (i % 2)] for i, w in enumerate([240, 880] * 30)]
    still = [(540.0 + (i % 3), 1600.0 + (i % 2) * 8) for i in range(60)]
    q = _capture(tracked_boxes=boxes, anchors=still)
    cam = q["checks"]["camera_consistency"]
    assert cam["status"] == "pass", cam
    assert cam["value"]["anchor_source"] == "keypoints"
    walking = [(300.0 + 8 * i, 1600.0) for i in range(60)]  # hips walk ~470 px sideways
    assert _capture(tracked_boxes=boxes, anchors=walking)["checks"]["camera_consistency"]["status"] == "warn"


def test_unseen_side_prevents_consistent_reconstruction_verdict():
    report = {"geometry_validation": {"valid": True, "watertight": True},
              "validation_summary": {"landmark_reprojection_error_pct_height": 0.6},
              "silhouette_fitting": {"summary_by_method": {"silhouette_keypoint_shared_shape (new)": {
                  "protrusion_frac_mean": 0.015}}},
              "multiview_optimisation": {"held_out_validation": {"results": {
                  "silhouette_keypoint_shared_shape (new)": {"iou_mean": 0.82}}}},
              "shared_shape": {"betas": [0.9]}, "measurements": {"valid": 12, "total": 12}}
    assert sq.assess_reconstruction(report)["verdict"] == "consistent"
    report["capture_quality"] = {"checks": {"angular_coverage": "fail"}}
    q = sq.assess_reconstruction(report)
    assert q["verdict"] == "unreliable" and q["checks"]["view_coverage"]["feedback"]
