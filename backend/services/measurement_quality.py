"""Per-measurement quality grade and precision estimate for one reconstruction (no ground truth involved).

Each girth gets a grade (good / fair / poor / unavailable), the reasons, and a ±1 SD PRECISION estimate (how much
the value would move on a repeat recording), never an accuracy claim. Inputs are what every run already
produces, plus the girths of the cross-validation fold shapes:

1. Run-level caps: scan_quality's capture verdict and reconstruction verdict. An unreliable reconstruction
   makes every girth poor; usable-with-warnings caps them at fair.
2. Cross-section: invalid -> unavailable; a min/max search that ended on the edge of its anatomical zone ->
   the value is a boundary reading, not the landmark (fair). The forearm always ends at the elbow-side edge on
   SMPL (no muscle belly in the model; location audit): reported as a model limitation, not penalised per run.
3. Fold stability: the pipeline's 4 heading-interleaved CV folds each fit a shape from 3/4 of the views. The SD
   of a girth across those shapes is the fit's own sensitivity to which views it saw. Thresholds are set
   against the between-recording repeatability measured in docs/MEASUREMENT_RELIABILITY_VALIDATION_REPORT.md.
4. Silhouette evidence (torso only): model/observed torso width (front) and depth (profile) at the site's level.
   Far below 1 = the clothed outline is much wider than the body there (loose garment): the girth is then
   limited by the shape model, not observed (fair). Above 1.05 = body outside its own outline (poor).
5. Coverage: largest heading gap between views with a usable silhouette, and their number.

Precision (±1 SD, %) = sqrt(between-recording SD^2 + max(0, fold SD^2 - typical fold SD^2)), i.e. the measured
repeatability, inflated when this run's fold spread is larger than is typical. A height uncertainty adds
d/H to every girth when centimetres are requested.
"""
from __future__ import annotations

import math

import numpy as np

SITES = ("chest", "waist", "abdomen", "hip", "upper_arm_left", "upper_arm_right", "forearm_left",
         "forearm_right", "thigh_left", "thigh_right", "calf_left", "calf_right")
GOOD, FAIR, POOR, UNAVAILABLE = "good", "fair", "poor", "unavailable"
_RANK = {GOOD: 0, FAIR: 1, POOR: 2, UNAVAILABLE: 3}

# Between-recording within-subject SD (% of girth) and typical fold SD (%), current default fit, 3 recordings of
# one subject (reliability study, 19 Nov 2026). n = 3: provisional, to be replaced as recordings accumulate.
REPEATABILITY_SD_PCT = {"chest": 1.19, "waist": 1.61, "abdomen": 1.55, "hip": 0.85, "upper_arm": 1.62,
                        "forearm": 0.93, "thigh": 1.01, "calf": 0.80}
TYPICAL_FOLD_SD_PCT = {"chest": 0.5, "waist": 0.6, "abdomen": 0.6, "hip": 0.5, "upper_arm": 0.6,
                       "forearm": 0.3, "thigh": 0.5, "calf": 0.3}
# Fold SD above FOLD_FAIR x repeatability SD: this run is less stable than re-recording the person -> fair;
# above FOLD_POOR x: the girth depends on which views were used -> poor.
FOLD_FAIR, FOLD_POOR = 1.0, 2.0
# Torso evidence: model/observed width or depth at the site's level.
LOOSE_RATIO, PROTRUDE_RATIO = 0.85, 1.05
TORSO_LEVEL_INDEX = {"chest": 0, "waist": 1, "abdomen": 2, "hip": 3}  # smpl_multiview.TORSO_LEVELS order
# Coverage: largest heading gap (deg) between usable silhouettes and their minimum number.
GAP_FAIR_DEG, GAP_POOR_DEG = 60.0, 100.0
MIN_VIEWS_FAIR, MIN_VIEWS_POOR = 8, 6
EDGE_EXEMPT = {"forearm"}  # structural on SMPL (see module docstring)


def site_group(site: str) -> str:
    return "upper_arm" if site.startswith("upper_arm") else site.split("_")[0]


def _worse(a, b):
    return a if _RANK[a] >= _RANK[b] else b


def heading_coverage(yaws) -> dict:
    y = np.sort(np.mod(np.asarray([v for v in yaws if v is not None and math.isfinite(v)], float), 360.0))
    if len(y) == 0:
        return {"views": 0, "largest_gap_deg": 360.0}
    gaps = np.diff(np.concatenate([y, [y[0] + 360.0]]))
    return {"views": int(len(y)), "largest_gap_deg": float(gaps.max())}


def fold_spread(fold_girths: list[dict] | None) -> dict:
    """SD (% of mean) of each site's girth/height across the CV fold shapes."""
    out = {}
    for s in SITES:
        v = np.array([f.get(s) for f in (fold_girths or []) if f.get(s)], float)
        out[s] = float(100 * v.std(ddof=1) / v.mean()) if len(v) >= 3 else None
    return out


def assess(measurements: dict, report: dict, *, fold_girths: list[dict] | None = None,
           usable_yaws=None, capture_verdict: str | None = None, reconstruction_verdict: str | None = None,
           height_cm: float | None = None, height_uncertainty_cm: float = 1.0) -> dict:
    meas = measurements.get("measurements", {})
    pc = ((report.get("silhouette_fitting") or {}).get("proportion_check") or {})
    front = (pc.get("front_back_width") or {}).get("median_ratio") or []
    side = (pc.get("side_depth") or {}).get("median_ratio") or []
    folds = fold_spread(fold_girths)
    cov = heading_coverage(usable_yaws) if usable_yaws is not None else None

    run_grade, run_reasons = GOOD, []
    if reconstruction_verdict is None:  # missing evidence is never a pass
        run_grade, run_reasons = FAIR, ["run-level reconstruction checks not available"]
    elif reconstruction_verdict == "unreliable":
        run_grade, run_reasons = POOR, run_reasons + ["reconstruction checks failed (reconstruction_quality_report)"]
    elif reconstruction_verdict == "usable_with_warnings":
        run_grade, run_reasons = FAIR, run_reasons + ["reconstruction checks raised warnings"]
    if capture_verdict == "insufficient":  # scan_quality.assess_capture verdicts
        run_grade = POOR
        run_reasons.append("capture checks failed (capture_quality_report)")
    elif capture_verdict == "acceptable_with_warnings":
        run_grade = _worse(run_grade, FAIR)
        run_reasons.append("capture checks raised warnings")
    if cov:
        if cov["views"] < MIN_VIEWS_POOR or cov["largest_gap_deg"] > GAP_POOR_DEG:
            run_grade = POOR
            run_reasons.append(f"camera coverage too sparse ({cov['views']} usable views, largest heading gap "
                               f"{cov['largest_gap_deg']:.0f} deg)")
        elif cov["views"] < MIN_VIEWS_FAIR or cov["largest_gap_deg"] > GAP_FAIR_DEG:
            run_grade = _worse(run_grade, FAIR)
            run_reasons.append(f"camera coverage thin ({cov['views']} usable views, largest heading gap "
                               f"{cov['largest_gap_deg']:.0f} deg)")

    sites, rel_height = {}, (height_uncertainty_cm / height_cm) if height_cm else 0.0
    for s in SITES:
        m = meas.get(s) or {}
        g = site_group(s)
        if not m.get("valid"):
            sites[s] = {"grade": UNAVAILABLE, "reasons": [m.get("reason") or "no valid cross-section"]}
            continue
        grade, reasons, notes = run_grade, list(run_reasons), []
        if m.get("extremum_at_search_edge"):
            if g in EDGE_EXEMPT:
                notes.append("SMPL forearm has no muscle-belly maximum: value is the girth just below the elbow")
            else:
                grade = _worse(grade, FAIR)
                reasons.append("min/max search ended on the edge of its anatomical zone (boundary value)")
        fsd = folds.get(s)
        rep = REPEATABILITY_SD_PCT[g]
        if fsd is not None:
            if fsd > FOLD_POOR * rep:
                grade = POOR
                reasons.append(f"unstable across view subsets (fold SD {fsd:.1f}% > {FOLD_POOR:g}x repeatability)")
            elif fsd > FOLD_FAIR * rep:
                grade = _worse(grade, FAIR)
                reasons.append(f"less stable across view subsets than between recordings (fold SD {fsd:.1f}%)")
        if s in TORSO_LEVEL_INDEX:
            k = TORSO_LEVEL_INDEX[s]
            ratios = {n: (r[k] if len(r) > k else None) for n, r in (("width", front), ("depth", side))}
            got = {n: v for n, v in ratios.items() if v is not None}
            if any(v > PROTRUDE_RATIO for v in got.values()):
                grade = POOR
                reasons.append("body extends outside its own outline at this level ("
                               + ", ".join(f"{n} {v:.2f}" for n, v in got.items()) + ")")
            elif any(v < LOOSE_RATIO for v in got.values()):
                grade = _worse(grade, FAIR)
                reasons.append("outline much wider than the body at this level (loose clothing?): girth set by "
                               "the shape model, not observed (" + ", ".join(f"{n} {v:.2f}" for n, v in got.items()) + ")")
            if not got:
                grade = _worse(grade, FAIR)
                reasons.append("no frontal/profile outline evidence at this level")
        else:
            notes.append("limb girths have no direct per-level outline check")
        excess = max(0.0, (fsd or 0.0) ** 2 - TYPICAL_FOLD_SD_PCT[g] ** 2)
        sd_pct = math.sqrt(rep ** 2 + excess)
        entry = {"grade": grade, "reasons": reasons, "notes": notes, "fold_sd_pct": fsd,
                 "precision_sd_pct": round(sd_pct, 2)}
        ratio = m.get("tape_circumference_over_height")
        if height_cm and ratio:
            cm = ratio * height_cm
            entry["value_cm"] = round(cm, 1)
            entry["precision_sd_cm"] = round(cm * math.sqrt((sd_pct / 100) ** 2 + rel_height ** 2), 1)
        sites[s] = entry
    counts = {k: sum(1 for v in sites.values() if v["grade"] == k) for k in _RANK}
    return {"sites": sites, "run_grade": run_grade, "run_reasons": run_reasons, "coverage": cov,
            "fold_spread_pct": folds, "counts": counts,
            "statement": "Grades and ±SD describe internal consistency and repeatability (precision). They are not "
                         "accuracy: no tape or 3D-scan ground truth is involved."}
