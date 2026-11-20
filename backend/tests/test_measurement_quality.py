"""Per-measurement quality grading (backend/services/measurement_quality.py) and the search-edge flag."""
import numpy as np
import pytest

from backend.services import body_measurements as bm
from backend.services import measurement_quality as mq


def _meas(**over):
    m = {s: {"valid": True, "tape_circumference_over_height": 0.5, "extremum_at_search_edge": False} for s in mq.SITES}
    for k, v in over.items():
        m[k] = {**m[k], **v}
    return {"measurements": m}


def _report(front=(0.95, 0.95, 0.95, 0.95), side=(0.95, 0.95, 0.95, 0.95)):
    return {"silhouette_fitting": {"proportion_check": {"front_back_width": {"median_ratio": list(front)},
                                                        "side_depth": {"median_ratio": list(side)}}}}


EVEN = list(range(0, 360, 15))  # 24 views, 15 deg apart


def _assess(*a, **k):
    k.setdefault("reconstruction_verdict", "consistent")
    return mq.assess(*a, **k)


def test_missing_evidence_is_never_good():
    r = mq.assess(_meas(), {}, usable_yaws=EVEN)  # older run: no verdict, no proportion check
    assert r["run_grade"] == mq.FAIR
    assert r["sites"]["waist"]["grade"] == mq.FAIR
    assert any("not available" in x for x in r["sites"]["calf_left"]["reasons"])


def test_clean_run_is_good_with_repeatability_precision():
    r = _assess(_meas(), _report(), usable_yaws=EVEN, height_cm=176)
    assert all(s["grade"] == mq.GOOD for s in r["sites"].values())
    w = r["sites"]["waist"]
    assert w["precision_sd_pct"] == pytest.approx(mq.REPEATABILITY_SD_PCT["waist"])
    expected_cm = 88.0 * np.hypot(mq.REPEATABILITY_SD_PCT["waist"] / 100, 1.0 / 176)  # repeatability + 1 cm height
    assert w["value_cm"] == 88.0 and w["precision_sd_cm"] == pytest.approx(expected_cm, abs=0.06)
    assert "They are not accuracy" in r["statement"]
    assert r["calibration"]["calibrated"] is False  # no fit_config in this report -> default values, flagged


def test_calibration_follows_the_runs_shape_prior():
    rep = {**_report(), "multiview_optimisation": {"fit_config": {"hmr2_shape_prior_weight": 0.0}}}
    r = _assess(_meas(), rep, usable_yaws=EVEN)
    assert r["calibration"] == {"hmr2_shape_prior_weight": 0.0, "calibrated": True, "note": None}
    assert r["sites"]["hip"]["precision_sd_pct"] == pytest.approx(mq.CALIBRATION[0.0]["repeatability"]["hip"])
    # A fold SD of 1% is typical at prior 0 but > 3x typical at prior 0.2 for the chest.
    folds = [{s: 0.5 for s in mq.SITES} for _ in range(4)]
    for f, v in zip(folds, (0.495, 0.500, 0.505, 0.510)):
        f["chest"] = v
    assert _assess(_meas(), rep, fold_girths=folds, usable_yaws=EVEN)["sites"]["chest"]["grade"] == mq.GOOD
    rep02 = {**_report(), "multiview_optimisation": {"fit_config": {"hmr2_shape_prior_weight": 0.2}}}
    assert _assess(_meas(), rep02, fold_girths=folds, usable_yaws=EVEN)["sites"]["chest"]["grade"] == mq.POOR


def test_invalid_section_and_edge_hits():
    r = _assess(_meas(hip={"valid": False, "reason": "merged"}, calf_left={"extremum_at_search_edge": True},
                        forearm_left={"extremum_at_search_edge": True}), _report(), usable_yaws=EVEN)
    assert r["sites"]["hip"]["grade"] == mq.UNAVAILABLE
    assert r["sites"]["calf_left"]["grade"] == mq.FAIR
    assert r["sites"]["forearm_left"]["grade"] == mq.GOOD and r["sites"]["forearm_left"]["notes"]


def test_loose_outline_and_protrusion_grade_torso_levels():
    r = _assess(_meas(), _report(front=(0.95, 0.95, 0.95, 0.80), side=(1.08, 0.95, 0.95, 0.95)), usable_yaws=EVEN)
    assert r["sites"]["hip"]["grade"] == mq.FAIR and "loose" in r["sites"]["hip"]["reasons"][0]
    assert r["sites"]["chest"]["grade"] == mq.POOR
    assert r["sites"]["waist"]["grade"] == mq.GOOD


def test_fold_instability_inflates_precision_and_downgrades():
    folds = [{s: 0.5 for s in mq.SITES} for _ in range(4)]
    for f, v in zip(folds, (0.48, 0.50, 0.52, 0.54)):  # waist SD ~5% of mean
        f["waist"] = v
    r = _assess(_meas(), _report(), fold_girths=folds, usable_yaws=EVEN)
    w = r["sites"]["waist"]
    assert w["grade"] == mq.POOR and w["fold_sd_pct"] > 2 * mq.REPEATABILITY_SD_PCT["waist"]
    assert w["precision_sd_pct"] > mq.REPEATABILITY_SD_PCT["waist"]
    assert r["sites"]["chest"]["grade"] == mq.GOOD


@pytest.mark.parametrize("yaws,grade", [(list(range(0, 360, 45)), mq.GOOD),         # 8 views, 45 deg gaps
                                        (list(range(0, 360, 60)), mq.FAIR),         # 6 views
                                        ([-30, 0, 30, 60, 90, 120, 150, 180], mq.POOR)])  # 150 deg unseen
def test_coverage_gates(yaws, grade):
    assert _assess(_meas(), _report(), usable_yaws=yaws)["sites"]["waist"]["grade"] == grade


def test_run_verdicts_cap_every_site():
    assert _assess(_meas(), _report(), reconstruction_verdict="unreliable")["run_grade"] == mq.POOR
    r = _assess(_meas(), _report(), capture_verdict="acceptable_with_warnings")
    assert {s["grade"] for s in r["sites"].values()} == {mq.FAIR}


def test_search_edge_flag_on_a_tapering_cylinder():
    # A cone-like column: girth shrinks with height, so a "max" search over heights ends at the lowest plane.
    ang = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    rings = [np.c_[(1.0 - 0.05 * k) * np.cos(ang), np.full(24, float(k)), (1.0 - 0.05 * k) * np.sin(ang)]
             for k in range(6)]
    v = np.vstack(rings)
    f = []
    for k in range(5):
        for i in range(24):
            a, b, c, d = k * 24 + i, k * 24 + (i + 1) % 24, (k + 1) * 24 + i, (k + 1) * 24 + (i + 1) % 24
            f += [[a, b, c], [b, d, c]]
    f = np.array(f)

    def plane(t):
        p = np.array([0.0, t, 0.0])
        return p, np.array([0.0, 1.0, 0.0]), p
    ts = np.linspace(0.5, 4.5, 9)
    e = bm._entry(bm._search(v, f, plane, ts, "max"), "test", 5.0, None, ts=ts)
    assert e["valid"] and e["extremum_at_search_edge"] and e["search_parameter"] == pytest.approx(0.5)
    e2 = bm._entry(bm._search(v, f, plane, ts, "max"), "test", 5.0, None, ts=[2.0])
    assert e2["extremum_at_search_edge"] is False  # a single fixed plane is never an "edge" search
