"""Independent ground-truth validation: tape measure vs pipeline vs SHAPY.

Ground truth is the measuring tape, never a model. Neither the pipeline nor SHAPY is used as a reference for the
other; their disagreement is reported only as a disagreement.

Inputs (CSV, one file each; templates in docs/validation/):

* **sessions** - one row per subject visit: ``subject_id, session_date, stadiometer_height_cm, clothing, measurer,
  run_ids`` (``run_ids``: one or more reconstruction run IDs, ``;``-separated, recorded BEFORE the tape is taken,
  so the model values cannot be tuned to the tape afterwards).
* **tape** - one row per subject x site: ``subject_id, site, reading_1_cm, reading_2_cm[, reading_3_cm]``.
  Two readings per site are required; a third is taken when the first two differ by more than the re-measure
  tolerance. The tape value is the mean of the readings (median of three).

Outputs per subject x site: tape, pipeline, SHAPY, error (model - tape) in cm and %, absolute errors. Per site
and method: n, bias, SD, MAE, MAPE, 95% limits of agreement, worst error; the measurer's technical error of
measurement (TEM) from the repeat readings, which is the floor no camera method can be shown to beat; and a
paired pipeline-vs-SHAPY absolute-error difference. Nothing is reported for a site with no tape data, and every
number carries its n.

Model centimetres = (girth / mesh height) x stadiometer height, read straight from each run's measurements.json /
shapy_measurements.json, so the comparison does not depend on whatever height was typed into the app.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

SITES = ("chest", "waist", "abdomen", "hip", "upper_arm_left", "upper_arm_right", "forearm_left",
         "forearm_right", "thigh_left", "thigh_right", "calf_left", "calf_right")
TORSO = {"chest", "waist", "abdomen", "hip"}
CLOTHING = ("minimal", "tight", "loose", "unknown")
# Two readings further apart than this need a third (cm). Typical trained-measurer repeatability is a few mm on
# limbs and ~1 cm on the torso; these are re-measure triggers, not accuracy claims.
REMEASURE_TOLERANCE_CM = {"torso": 1.0, "limb": 0.5}
# Proposed acceptance targets for |bias| and MAE (cm); to be agreed for the product, not derived here.
DEFAULT_TARGET_CM = {"torso": 2.0, "limb": 1.0}
METHODS = ("pipeline", "shapy")
MIN_N_FOR_CLAIM = 5


class GroundTruthError(ValueError):
    """Malformed ground-truth input."""


def _f(value, what):
    if value is None or str(value).strip() == "":
        return None
    try:
        x = float(value)
    except ValueError as exc:
        raise GroundTruthError(f"{what}: {value!r} is not a number") from exc
    if not math.isfinite(x):
        raise GroundTruthError(f"{what}: not finite")
    return x


def read_sessions(path: Path) -> dict:
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for i, row in enumerate(csv.DictReader(fh), start=2):
            sid = (row.get("subject_id") or "").strip()
            if not sid or sid.startswith("#"):
                continue
            h = _f(row.get("stadiometer_height_cm"), f"sessions line {i} height")
            if h is None or not 100 <= h <= 250:
                raise GroundTruthError(f"sessions line {i}: stadiometer_height_cm must be 100-250")
            clothing = (row.get("clothing") or "unknown").strip()
            if clothing not in CLOTHING:
                raise GroundTruthError(f"sessions line {i}: clothing must be one of {', '.join(CLOTHING)}")
            runs = [r.strip() for r in (row.get("run_ids") or "").replace(",", ";").split(";") if r.strip()]
            if not runs:
                raise GroundTruthError(f"sessions line {i}: run_ids is empty")
            if sid in out:
                raise GroundTruthError(f"sessions line {i}: subject {sid} listed twice (use one row per subject)")
            out[sid] = {"subject_id": sid, "height_cm": h, "clothing": clothing, "run_ids": runs,
                        "measurer": (row.get("measurer") or "").strip(), "date": (row.get("session_date") or "").strip()}
    return out


def read_tape(path: Path) -> dict:
    """{subject: {site: {"readings": [...], "tape_cm": value, "needs_third": bool}}}"""
    out: dict = {}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for i, row in enumerate(csv.DictReader(fh), start=2):
            sid = (row.get("subject_id") or "").strip()
            if not sid or sid.startswith("#"):
                continue
            site = (row.get("site") or "").strip()
            if site not in SITES:
                raise GroundTruthError(f"tape line {i}: unknown site {site!r}")
            reads = [_f(row.get(f"reading_{k}_cm"), f"tape line {i} reading {k}") for k in (1, 2, 3)]
            reads = [r for r in reads if r is not None]
            if len(reads) < 2:
                raise GroundTruthError(f"tape line {i}: at least two readings are required")
            if any(not 10 <= r <= 250 for r in reads):
                raise GroundTruthError(f"tape line {i}: readings must be 10-250 cm")
            if site in out.get(sid, {}):
                raise GroundTruthError(f"tape line {i}: {sid}/{site} given twice")
            tol = REMEASURE_TOLERANCE_CM["torso" if site in TORSO else "limb"]
            needs_third = len(reads) == 2 and abs(reads[0] - reads[1]) > tol
            value = float(np.median(reads)) if len(reads) >= 3 else float(np.mean(reads))
            out.setdefault(sid, {})[site] = {"readings": reads, "tape_cm": value, "needs_third": needs_third}
    return out


def _json(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def model_cm(run_dir: Path, height_cm: float) -> dict:
    """Pipeline and SHAPY girths of one run in cm at the stadiometer height (None where unavailable/invalid)."""
    out = {m: {} for m in METHODS}
    meas = _json(run_dir / "measurements.json") or {}
    for s, m in (meas.get("measurements") or {}).items():
        r = m.get("tape_circumference_over_height") if m.get("valid") else None
        out["pipeline"][s] = r * height_cm if r else None
    shapy = _json(run_dir / "shapy_measurements.json") or {}
    for s, m in (shapy.get("measurements") or {}).items():
        r = m.get("tape_circumference_over_height") if m.get("valid") else None
        out["shapy"][s] = r * height_cm if r else None
    status = _json(run_dir / "status.json") or {}
    report = _json(run_dir / "reconstruction_report.json") or {}
    out["meta"] = {"run_id": run_dir.name, "status": status.get("status"), "video_id": status.get("video_id"),
                   "fit_config": (report.get("multiview_optimisation") or {}).get("fit_config"),
                   "shapy_available": bool(shapy), "quality": _json(run_dir / "measurement_quality.json")}
    return out


def tem(pairs) -> float | None:
    """Technical error of measurement from duplicate readings: sqrt(sum d^2 / 2n)."""
    d = np.array([a - b for a, b in pairs], dtype=float)
    return float(math.sqrt((d ** 2).sum() / (2 * len(d)))) if len(d) else None


def _stats(errs, tapes):
    e, t = np.asarray(errs, float), np.asarray(tapes, float)
    if not len(e):
        return None
    sd = float(e.std(ddof=1)) if len(e) > 1 else None
    return {"n": int(len(e)), "bias_cm": round(float(e.mean()), 2), "sd_cm": None if sd is None else round(sd, 2),
            "mae_cm": round(float(np.abs(e).mean()), 2), "mape_pct": round(float(np.abs(100 * e / t).mean()), 2),
            "max_abs_cm": round(float(np.abs(e).max()), 2),
            "loa_95_cm": None if sd is None else [round(float(e.mean() - 1.96 * sd), 2),
                                                 round(float(e.mean() + 1.96 * sd), 2)],
            "claimable": bool(len(e) >= MIN_N_FOR_CLAIM)}


def build(sessions: dict, tape: dict, runs_root: Path, targets: dict | None = None) -> dict:
    targets = targets or DEFAULT_TARGET_CM
    rows, problems = [], []
    for sid, sess in sessions.items():
        t = tape.get(sid)
        if not t:
            problems.append(f"{sid}: no tape readings")
            continue
        for run_id in sess["run_ids"]:
            d = next(iter(sorted(Path(runs_root).glob(run_id + "*"))), None)
            if d is None:
                problems.append(f"{sid}: run {run_id} not found")
                continue
            mc = model_cm(d, sess["height_cm"])
            if mc["meta"]["status"] != "completed":
                problems.append(f"{sid}: run {d.name[:8]} is not completed")
                continue
            if not mc["meta"]["shapy_available"]:
                problems.append(f"{sid}: run {d.name[:8]} has no SHAPY result (POST /runs/{{id}}/shapy or "
                                "scripts/shapy_compare.py)")
            for site, tr in t.items():
                row = {"subject_id": sid, "run_id": d.name, "site": site, "clothing": sess["clothing"],
                       "tape_cm": round(tr["tape_cm"], 2), "tape_readings": tr["readings"],
                       "tape_needs_third_reading": tr["needs_third"]}
                for m in METHODS:
                    v = mc[m].get(site)
                    row[f"{m}_cm"] = None if v is None else round(v, 2)
                    if v is not None:
                        err = v - tr["tape_cm"]
                        row[f"{m}_error_cm"] = round(err, 2)
                        row[f"{m}_abs_error_cm"] = round(abs(err), 2)
                        row[f"{m}_error_pct"] = round(100 * err / tr["tape_cm"], 2)
                if row.get("pipeline_cm") is not None and row.get("shapy_cm") is not None:
                    row["pipeline_minus_shapy_cm"] = round(row["pipeline_cm"] - row["shapy_cm"], 2)
                q = ((mc["meta"]["quality"] or {}).get("sites") or {}).get(site) or {}
                row["pipeline_quality_grade"] = q.get("grade")
                rows.append(row)
            if tr_flags := [s for s, tr in t.items() if tr["needs_third"]]:
                problems.append(f"{sid}: readings differ beyond tolerance at {', '.join(tr_flags)}: take a third")
    # One run per subject for accuracy statistics (the first listed), so re-runs do not weight a subject twice.
    primary = {s["subject_id"]: s["run_ids"][0] for s in sessions.values()}
    prim = [r for r in rows if r["run_id"].startswith(primary[r["subject_id"]])]
    per_site = {}
    for site in SITES:
        sr = [r for r in prim if r["site"] == site]
        if not sr:
            continue
        kind = "torso" if site in TORSO else "limb"
        entry = {"target_cm": targets[kind]}
        for m in METHODS:
            ok = [r for r in sr if r.get(f"{m}_error_cm") is not None]
            st = _stats([r[f"{m}_error_cm"] for r in ok], [r["tape_cm"] for r in ok])
            if st:
                st["meets_target"] = (abs(st["bias_cm"]) <= targets[kind] and st["mae_cm"] <= targets[kind]) \
                    if st["claimable"] else None
                for c in CLOTHING:
                    cc = [r for r in ok if r["clothing"] == c]
                    if cc:
                        st.setdefault("by_clothing", {})[c] = _stats([r[f"{m}_error_cm"] for r in cc],
                                                                      [r["tape_cm"] for r in cc])
            entry[m] = st
        both = [r for r in sr if r.get("pipeline_abs_error_cm") is not None and r.get("shapy_abs_error_cm") is not None]
        if both:
            diff = np.array([r["pipeline_abs_error_cm"] - r["shapy_abs_error_cm"] for r in both])
            entry["paired_abs_error_pipeline_minus_shapy_cm"] = {
                "n": len(both), "mean": round(float(diff.mean()), 2),
                "pipeline_closer_count": int((diff < 0).sum()), "shapy_closer_count": int((diff > 0).sum()),
                "note": "negative = pipeline closer to tape" + ("" if len(both) >= MIN_N_FOR_CLAIM else
                                                                "; too few subjects to conclude")}
        pairs = [(r["tape_readings"][0], r["tape_readings"][1]) for r in sr]
        entry["measurer_tem_cm"] = round(tem(pairs), 2) if pairs else None
        per_site[site] = entry
    return {"subjects_with_tape": len([s for s in sessions if s in tape]), "rows": rows, "per_site": per_site,
            "problems": problems, "targets_cm": targets, "min_n_for_claim": MIN_N_FOR_CLAIM,
            "sign_convention": "error = model - tape (positive: model larger than tape)",
            "status": "no ground truth recorded: no accuracy can be stated" if not rows else
                      ("accuracy statistics are preliminary (fewer than "
                       f"{MIN_N_FOR_CLAIM} subjects)" if len(primary) < MIN_N_FOR_CLAIM else "ok")}


def markdown(rep: dict) -> str:
    out = ["# Ground-truth validation: tape vs pipeline vs SHAPY", "", f"**Status:** {rep['status']}.", "",
           f"Subjects with tape: {rep['subjects_with_tape']} · {rep['sign_convention']} · "
           f"statistics marked claimable only with n ≥ {rep['min_n_for_claim']}.", ""]
    if rep["problems"]:
        out += ["## Input problems", ""] + [f"- {p}" for p in rep["problems"]] + [""]
    if not rep["rows"]:
        return "\n".join(out + ["No tape measurements have been recorded, so no accuracy is reported."]) + "\n"
    out += ["## Per subject", "",
            "| Subject | Run | Site | Tape cm | Pipeline cm | Err cm | Err % | SHAPY cm | Err cm | Err % | Quality |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]

    def f(v, d=1):
        return "—" if v is None else f"{v:.{d}f}"
    for r in rep["rows"]:
        out.append(f"| {r['subject_id']} | {r['run_id'][:8]} | {r['site']} | {f(r['tape_cm'])} | {f(r.get('pipeline_cm'))} | "
                   f"{f(r.get('pipeline_error_cm'))} | {f(r.get('pipeline_error_pct'))} | {f(r.get('shapy_cm'))} | "
                   f"{f(r.get('shapy_error_cm'))} | {f(r.get('shapy_error_pct'))} | {r.get('pipeline_quality_grade') or '—'} |")
    out += ["", "## Per site", "",
            "| Site | n | Method | Bias cm | SD cm | MAE cm | MAPE % | 95% LoA cm | Max cm | Target | Measurer TEM cm |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for site, e in rep["per_site"].items():
        for m in METHODS:
            s = e.get(m)
            if not s:
                continue
            loa = "—" if not s["loa_95_cm"] else f"{s['loa_95_cm'][0]:+.1f} … {s['loa_95_cm'][1]:+.1f}"
            meets = {True: "met", False: "not met", None: "n too small"}[s["meets_target"]]
            out.append(f"| {site} | {s['n']} | {m} | {s['bias_cm']:+.2f} | {f(s['sd_cm'], 2)} | {s['mae_cm']:.2f} | "
                       f"{s['mape_pct']:.1f} | {loa} | {s['max_abs_cm']:.1f} | ±{e['target_cm']:g} {meets} | "
                       f"{f(e['measurer_tem_cm'], 2)} |")
    return "\n".join(out) + "\n"
