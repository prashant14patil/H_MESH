"""Ground-truth (tape vs pipeline vs SHAPY) arithmetic and input checks. Values here are synthetic test inputs."""
import json

import pytest

from backend.services import ground_truth_validation as gt


def _run(root, run_id, pipe, shapy=None, status="completed"):
    d = root / run_id
    d.mkdir()
    (d / "status.json").write_text(json.dumps({"status": status, "video_id": "v"}))
    (d / "measurements.json").write_text(json.dumps({"measurements": {
        s: {"valid": v is not None, "tape_circumference_over_height": v} for s, v in pipe.items()}}))
    if shapy:
        (d / "shapy_measurements.json").write_text(json.dumps({"measurements": {
            s: {"valid": True, "tape_circumference_over_height": v} for s, v in shapy.items()}}))
    return d


def _csv(path, text):
    path.write_text(text, encoding="utf-8")
    return path


def test_errors_use_stadiometer_height_and_mean_of_readings(tmp_path):
    _run(tmp_path, "run-a", {"waist": 0.5, "hip": None}, {"waist": 0.48})
    sessions = gt.read_sessions(_csv(tmp_path / "s.csv", "subject_id,stadiometer_height_cm,clothing,run_ids\n"
                                                         "S01,180,tight,run-a\n"))
    tape = gt.read_tape(_csv(tmp_path / "t.csv", "subject_id,site,reading_1_cm,reading_2_cm\n"
                                                 "S01,waist,87.8,88.2\nS01,hip,100,100.4\n"))
    rep = gt.build(sessions, tape, tmp_path)
    w = next(r for r in rep["rows"] if r["site"] == "waist")
    assert w["tape_cm"] == 88.0
    assert w["pipeline_cm"] == 90.0 and w["pipeline_error_cm"] == 2.0 and w["pipeline_error_pct"] == pytest.approx(2.27)
    assert w["shapy_cm"] == pytest.approx(86.4) and w["shapy_error_cm"] == pytest.approx(-1.6)
    h = next(r for r in rep["rows"] if r["site"] == "hip")
    assert h["pipeline_cm"] is None and "pipeline_error_cm" not in h  # invalid section: never compared
    assert rep["per_site"]["waist"]["pipeline"]["claimable"] is False  # n = 1
    assert rep["per_site"]["waist"]["pipeline"]["meets_target"] is None
    assert "preliminary" in rep["status"]
    assert rep["per_site"]["waist"]["measurer_tem_cm"] == pytest.approx(0.28, abs=0.01)


def test_no_tape_reports_no_accuracy(tmp_path):
    _run(tmp_path, "run-a", {"waist": 0.5})
    sessions = gt.read_sessions(_csv(tmp_path / "s.csv", "subject_id,stadiometer_height_cm,clothing,run_ids\n"
                                                         "S01,180,tight,run-a\n"))
    rep = gt.build(sessions, {}, tmp_path)
    assert rep["rows"] == [] and "no accuracy" in rep["status"]
    assert "No tape measurements" in gt.markdown(rep)


def test_readings_far_apart_request_a_third_and_three_use_median(tmp_path):
    tape = gt.read_tape(_csv(tmp_path / "t.csv", "subject_id,site,reading_1_cm,reading_2_cm,reading_3_cm\n"
                                                 "S01,calf_left,36.0,37.0,\nS01,chest,100,104,101\n"))
    assert tape["S01"]["calf_left"]["needs_third"] is True
    assert tape["S01"]["chest"]["tape_cm"] == 101 and tape["S01"]["chest"]["needs_third"] is False


@pytest.mark.parametrize("text", ["subject_id,site,reading_1_cm,reading_2_cm\nS01,neck,30,30\n",
                                  "subject_id,site,reading_1_cm,reading_2_cm\nS01,waist,88,\n",
                                  "subject_id,site,reading_1_cm,reading_2_cm\nS01,waist,8,8\n"])
def test_bad_tape_rows_are_rejected(tmp_path, text):
    with pytest.raises(gt.GroundTruthError):
        gt.read_tape(_csv(tmp_path / "t.csv", text))


def test_paired_comparison_and_claimable_with_five_subjects(tmp_path):
    rows_s, rows_t = ["subject_id,stadiometer_height_cm,clothing,run_ids"], ["subject_id,site,reading_1_cm,reading_2_cm"]
    for i in range(5):
        _run(tmp_path, f"run-{i}", {"waist": 0.5}, {"waist": 0.47})
        rows_s.append(f"S{i},180,minimal,run-{i}")
        rows_t.append(f"S{i},waist,{89 + i * 0.1},{89 + i * 0.1}")
    rep = gt.build(gt.read_sessions(_csv(tmp_path / "s.csv", "\n".join(rows_s))),
                   gt.read_tape(_csv(tmp_path / "t.csv", "\n".join(rows_t))), tmp_path)
    p = rep["per_site"]["waist"]
    assert p["pipeline"]["n"] == 5 and p["pipeline"]["claimable"] is True
    assert p["pipeline"]["meets_target"] is True  # |bias| ~0.8 cm <= 2 cm torso target
    assert p["shapy"]["meets_target"] is False   # 84.6 vs ~89.2: bias ~ -4.6 cm
    assert p["paired_abs_error_pipeline_minus_shapy_cm"]["pipeline_closer_count"] == 5
    assert rep["status"] == "ok"
