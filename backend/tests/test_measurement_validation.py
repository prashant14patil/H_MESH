"""Validation statistics on synthetic run folders (known answers)."""
import json
import math

import pytest

from backend.services import body_measurements as bm
from backend.services import measurement_validation as mv


def make_run(root, run_id, video, created, waist, *, conditions=None, tape=None, fit_config=None, height=None):
    d = root / run_id
    d.mkdir()
    result = {"measurements": {"waist": waist}}
    if height:
        result["metric_scale"] = {"height_cm": height}
    (d / "status.json").write_text(json.dumps({"run_id": run_id, "video_id": video, "status": "completed",
                                               "created_at": created, "result": result}))
    meas = {s: {"valid": False} for s in mv.SITES}
    meas["waist"] = {"valid": True, "tape_circumference_over_height": waist}
    (d / "measurements.json").write_text(json.dumps({"measurements": meas}))
    if fit_config is not None:
        (d / "reconstruction_report.json").write_text(json.dumps({"multiview_optimisation": {"fit_config": fit_config}}))
    if conditions:
        (d / mv.CONDITIONS_FILE).write_text(json.dumps(conditions))
    if tape:
        (d / "tape_validation.json").write_text(json.dumps(tape))
    return d


def test_reruns_of_one_recording_are_not_repeat_measurements(tmp_path):
    make_run(tmp_path, "a1", "vidA", 1, 0.50)
    make_run(tmp_path, "a2", "vidA", 2, 0.51)  # re-run of the same video: newest wins
    make_run(tmp_path, "b1", "vidB", 3, 0.53)
    rep = mv.build_report(tmp_path, default_subject="S01", fit_config="any")
    assert rep["distinct_recordings"] == 2
    r = rep["repeatability"]["per_site"]["waist"]
    assert r["recordings"] == 2
    assert math.isclose(r["within_sd_x_height"], abs(0.53 - 0.51) / math.sqrt(2), rel_tol=1e-9)
    assert math.isclose(r["rc_x_height"], 2.77 * r["within_sd_x_height"])


def test_identical_files_under_different_video_ids_collapse(tmp_path):
    make_run(tmp_path, "a1", "upload1", 1, 0.50)
    make_run(tmp_path, "a2", "upload2", 2, 0.52)
    rep = mv.build_report(tmp_path, source_key=lambda v: "same-bytes", default_subject="S01", fit_config="any")
    assert rep["distinct_recordings"] == 1
    assert not rep["repeatability"]["per_site"]["waist"]["evaluable"]


def test_runs_with_other_fit_settings_are_not_pooled(tmp_path):
    make_run(tmp_path, "old", "v1", 1, 0.60, fit_config={"prior": 0})
    make_run(tmp_path, "n1", "v2", 2, 0.48, fit_config={"prior": 0.2})
    make_run(tmp_path, "n2", "v3", 3, 0.49, fit_config={"prior": 0.2})
    rep = mv.build_report(tmp_path, default_subject="S01")
    assert rep["fit_config"] == {"prior": 0.2}
    assert rep["distinct_recordings"] == 2


def test_condition_effect_against_reference_and_repeatability(tmp_path):
    cond = lambda clothing: {"subject_id": "S01", "clothing": clothing}
    make_run(tmp_path, "m1", "v1", 1, 0.470, conditions=cond("minimal"))
    make_run(tmp_path, "m2", "v2", 2, 0.474, conditions=cond("minimal"))
    make_run(tmp_path, "l1", "v3", 3, 0.520, conditions=cond("loose"))
    rep = mv.build_report(tmp_path, fit_config="any")
    eff = rep["condition_effects"]["clothing"]
    assert eff["reference"] == "minimal"
    loose = eff["levels"]["loose"]["waist"]
    assert math.isclose(loose["mean_diff_x_height"], 0.520 - 0.472, abs_tol=1e-12)
    assert loose["exceeds_repeatability"] is True
    assert not rep["condition_effects"]["distance"]["evaluable"]
    # noise comes from the two minimal-clothing recordings only, not from the loose one
    assert rep["repeatability"]["per_site"]["waist"]["recordings"] == 2


def test_excluded_tape_records_are_ignored(tmp_path):
    real = {"run_id": "r", "video_id": "v1", "clothing": "minimal", "recorded_at": 1,
            "sites": {"waist": {"compared": True, "error_cm": 2.0, "error_pct": 2.4}}}
    fake = {**real, "video_id": "v2", "excluded": True,
            "sites": {"waist": {"compared": True, "error_cm": 0.1, "error_pct": 0.1}}}
    make_run(tmp_path, "r1", "v1", 1, 0.48, tape=real)
    d = make_run(tmp_path, "r2", "v2", 2, 0.48, tape={**fake, "excluded": False})
    mv.exclude_tape(d, "test entry")
    rep = mv.build_report(tmp_path, fit_config="any")
    assert rep["excluded_tape_records"] == 1
    assert rep["tape_accuracy"]["subjects"] == 1
    assert rep["tape_accuracy"]["per_site"]["waist"]["mean_error_cm"] == 2.0
    assert bm.aggregate_tape([fake])["subjects"] == 0


@pytest.mark.parametrize("payload", [{}, {"subject_id": "x" * 41}, {"subject_id": "S1", "clothing": "jeans"},
                                     {"subject_id": "S1", "distance_m": 0.1}, {"subject_id": "S1", "posture": "sit"},
                                     {"subject_id": "S1", "name": "Bob"}])
def test_invalid_conditions_rejected(payload):
    with pytest.raises(ValueError):
        mv.validate_conditions(payload)


def test_conditions_require_a_completed_run(tmp_path):
    d = tmp_path / "x"
    d.mkdir()
    (d / "status.json").write_text(json.dumps({"status": "running"}))
    with pytest.raises(ValueError):
        mv.write_conditions(d, {"subject_id": "S01"})
