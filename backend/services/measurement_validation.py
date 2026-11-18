"""Validation of body circumferences: capture conditions, repeatability, condition effects, tape accuracy.

Per completed run the operator can record the capture conditions (who, clothing, camera distance,
posture) in ``capture_conditions.json``. :func:`build_report` then pools every completed run:

* **Recordings, not runs.** Runs of byte-identical source videos (re-uploads, re-runs) collapse to the
  latest run of that source; only distinct recordings count as repeat measurements.
* **Repeatability** per site: within-subject SD over a subject's recordings, CV, and the repeatability
  coefficient RC = 2.77 x Sw (Bland-Altman: 95% of repeat differences of an unchanged body are below RC).
* **Condition effects**: for clothing / distance / posture, the mean within-subject difference of each
  level from the reference level, compared with RC (an effect smaller than RC is not distinguishable
  from run-to-run noise).
* **Accuracy against tape**: model - tape per site (bias, SD, MAE, worst), from tape_validation.json
  records that are not excluded (test entries are excluded with :func:`exclude_tape`).

Girths are compared as fractions of mesh height (scale-free) and converted to cm with the subject's
supplied height where one exists, so repeatability is not mixed up with height-entry differences.
Nothing is extrapolated: every statistic carries its n, and n < 3 is flagged.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

SITES = ("chest", "waist", "abdomen", "hip", "upper_arm_left", "upper_arm_right", "forearm_left",
         "forearm_right", "thigh_left", "thigh_right", "calf_left", "calf_right")
CLOTHING = ("minimal", "tight", "loose", "unknown")
POSTURES = ("a_pose_standard", "arms_lower", "arms_higher", "relaxed_slouch", "other")
REFERENCE_LEVEL = {"clothing": "minimal", "posture": "a_pose_standard", "distance": "near"}
DISTANCE_BINS = ((0.0, 2.5, "near"), (2.5, 3.5, "mid"), (3.5, 99.0, "far"))
CONDITIONS_FILE = "capture_conditions.json"


# ------------------------------------------------------------------ conditions
def validate_conditions(payload: dict) -> dict:
    """Operator-entered capture conditions. Unknown keys are rejected; every field is optional except subject."""
    if not isinstance(payload, dict):
        raise ValueError("conditions must be an object")
    allowed = {"subject_id", "clothing", "distance_m", "posture", "note", "height_cm"}
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"unknown condition field(s): {', '.join(sorted(unknown))}")
    subject = str(payload.get("subject_id") or "").strip()
    if not subject or len(subject) > 40 or not all(c.isalnum() or c in "-_ ." for c in subject):
        raise ValueError("subject_id is required: 1-40 letters, digits, space, '-', '_' or '.' (use a code, not a name)")
    clothing = payload.get("clothing", "unknown")
    if clothing not in CLOTHING:
        raise ValueError(f"clothing must be one of {', '.join(CLOTHING)}")
    posture = payload.get("posture", "a_pose_standard")
    if posture not in POSTURES:
        raise ValueError(f"posture must be one of {', '.join(POSTURES)}")
    distance = payload.get("distance_m")
    if distance is not None:
        if isinstance(distance, bool) or not isinstance(distance, (int, float)) or not math.isfinite(distance) \
                or not 0.5 <= float(distance) <= 10.0:
            raise ValueError("distance_m must be a number between 0.5 and 10 (camera to subject, metres)")
        distance = float(distance)
    height = payload.get("height_cm")
    if height is not None:
        from backend.services.body_measurements import validate_height_cm
        height = validate_height_cm(height)
    return {"subject_id": subject, "clothing": clothing, "posture": posture, "distance_m": distance,
            "height_cm": height, "note": str(payload.get("note") or "")[:500]}


def write_conditions(run_dir: Path, payload: dict) -> dict:
    run_dir = Path(run_dir)
    status = _read_json(run_dir / "status.json")
    if not status or status.get("status") != "completed":
        raise ValueError("Capture conditions can only be recorded for a completed reconstruction")
    record = {**validate_conditions(payload), "run_id": status.get("run_id"), "video_id": status.get("video_id"),
              "recorded_at": time.time()}
    (run_dir / CONDITIONS_FILE).write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def exclude_tape(run_dir: Path, reason: str) -> dict:
    """Mark a run's tape record as not validation evidence (e.g. test entries). Reversible: the values stay."""
    path = Path(run_dir) / "tape_validation.json"
    record = _read_json(path)
    if record is None:
        raise ValueError(f"{Path(run_dir).name}: no tape_validation.json")
    record.update(excluded=True, excluded_reason=str(reason)[:300], excluded_at=time.time())
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    status_path = Path(run_dir) / "status.json"
    status = _read_json(status_path)
    if status and isinstance(status.get("result"), dict) and status["result"].get("tape_validation"):
        status["result"]["tape_validation"]["excluded"] = True  # the UI must not show it as a tape check
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    return record


# ------------------------------------------------------------------ collection
def _read_json(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def collect_runs(root: Path, source_key=None, since: float | None = None) -> list[dict]:
    """Completed runs with measurements, one row each (no de-duplication yet)."""
    rows = []
    for status_path in Path(root).glob("*/status.json"):
        status = _read_json(status_path)
        if not status or status.get("status") != "completed":
            continue
        run_dir = status_path.parent
        meas = _read_json(run_dir / "measurements.json")
        if not meas or "measurements" not in meas:
            continue
        created = status.get("created_at") or 0
        if since is not None and created < since:
            continue
        cond = _read_json(run_dir / CONDITIONS_FILE) or {}
        metric = (status.get("result") or {}).get("metric_scale") or {}
        tape = _read_json(run_dir / "tape_validation.json")
        report = _read_json(run_dir / "reconstruction_report.json") or {}
        fit_config = (report.get("multiview_optimisation") or {}).get("fit_config")
        rows.append({
            "run_id": status.get("run_id"), "video_id": status.get("video_id"), "created_at": created,
            "source_key": source_key(status.get("video_id")) if source_key else status.get("video_id"),
            "ratios": {k: (m.get("tape_circumference_over_height") if m.get("valid") else None)
                       for k, m in meas["measurements"].items()},
            "height_cm": cond.get("height_cm") or metric.get("height_cm"),
            "conditions": cond or None,
            "tape": tape if tape and not tape.get("excluded") else None,
            "tape_excluded": bool(tape and tape.get("excluded")),
            # Exact shape-fit settings (recorded from Nov 2026 on). Girths from different settings are not
            # repeat measurements of each other, so the report never pools them.
            "fit_config": json.dumps(fit_config, sort_keys=True) if fit_config else None,
        })
    return rows


def latest_per_recording(rows: list[dict]) -> list[dict]:
    """One row per distinct source recording: the most recent run (re-runs and re-uploads are not repeats)."""
    best = {}
    for r in rows:
        key = r["source_key"] or r["run_id"]
        if key not in best or r["created_at"] > best[key]["created_at"]:
            # Conditions/tape recorded on an older run of the same recording still describe it.
            prev = best.get(key)
            if prev:
                r = {**r, "conditions": r["conditions"] or prev["conditions"], "tape": r["tape"] or prev["tape"],
                     "height_cm": r["height_cm"] or prev["height_cm"]}
            best[key] = r
        else:
            cur = best[key]
            best[key] = {**cur, "conditions": cur["conditions"] or r["conditions"], "tape": cur["tape"] or r["tape"],
                         "height_cm": cur["height_cm"] or r["height_cm"]}
    return sorted(best.values(), key=lambda r: r["created_at"])


# ------------------------------------------------------------------ statistics
def _subject(r, default):
    return (r.get("conditions") or {}).get("subject_id") or default


def _condition_key(r):
    c = r.get("conditions") or {}
    return (c.get("clothing") or "unknown", _level(r, "distance") or "unknown", c.get("posture") or "unknown")


def repeatability(recordings: list[dict], default_subject: str | None = None) -> dict:
    """Per-site within-subject SD / CV / RC over distinct recordings, pooled across subjects.

    Only recordings of the same subject under the SAME conditions are compared: otherwise a real
    clothing/distance/posture effect would inflate the noise estimate and then hide inside it.
    """
    by_subject = {}
    for r in recordings:
        s = _subject(r, default_subject)
        if s:
            by_subject.setdefault((s, _condition_key(r)), []).append(r)
    out = {}
    for site in SITES:
        ss, dof, cvs, sd_cm_terms = 0.0, 0, [], []
        n_rec = 0
        for recs in by_subject.values():
            vals = np.array([r["ratios"].get(site) for r in recs if r["ratios"].get(site) is not None], dtype=float)
            if len(vals) < 2:
                continue
            n_rec += len(vals)
            ss += float(((vals - vals.mean()) ** 2).sum())
            dof += len(vals) - 1
            cvs.append(float(vals.std(ddof=1) / vals.mean() * 100))
            heights = [r["height_cm"] for r in recs if r.get("height_cm")]
            if heights:
                sd_cm_terms.append((float(vals.std(ddof=1)) * float(np.median(heights)), len(vals) - 1))
        if dof == 0:
            out[site] = {"evaluable": False}
            continue
        sw = math.sqrt(ss / dof)
        row = {"evaluable": True, "subjects": sum(1 for recs in by_subject.values()
                                                  if sum(r["ratios"].get(site) is not None for r in recs) >= 2),
               "recordings": n_rec, "dof": dof, "within_sd_x_height": sw, "cv_pct": float(np.mean(cvs)),
               "rc_x_height": 2.77 * sw, "low_sample": dof < 10}
        if sd_cm_terms:
            sw_cm = math.sqrt(sum(sd ** 2 * d for sd, d in sd_cm_terms) / sum(d for _, d in sd_cm_terms))
            row.update(within_sd_cm=sw_cm, rc_cm=2.77 * sw_cm)
        out[site] = row
    return {"subjects": len({s for s, _ in by_subject}), "groups": len(by_subject), "per_site": out}


def _level(r, factor):
    c = r.get("conditions") or {}
    if factor == "distance":
        d = c.get("distance_m")
        if d is None:
            return None
        return next(name for lo, hi, name in DISTANCE_BINS if lo <= d < hi)
    return c.get(factor) if c.get(factor) not in (None, "unknown") else None


def condition_effects(recordings: list[dict], rep: dict, default_subject: str | None = None) -> dict:
    """Within-subject mean difference of each condition level from the reference level, per site."""
    out = {}
    for factor in ("clothing", "distance", "posture"):
        levels = {}
        for r in recordings:
            lv, s = _level(r, factor), _subject(r, default_subject)
            if lv and s:
                levels.setdefault(lv, {}).setdefault(s, []).append(r)
        ref = REFERENCE_LEVEL[factor] if REFERENCE_LEVEL[factor] in levels else (sorted(levels)[0] if levels else None)
        if not ref or len(levels) < 2:
            out[factor] = {"evaluable": False, "levels_seen": sorted(levels),
                           "reason": "needs recordings of the same subject at two or more levels"}
            continue
        res = {}
        for lv, subjects in levels.items():
            if lv == ref:
                continue
            per_site = {}
            for site in SITES:
                diffs = []
                for s, recs in subjects.items():
                    a = [r["ratios"].get(site) for r in recs if r["ratios"].get(site) is not None]
                    b = [r["ratios"].get(site) for r in levels[ref].get(s, []) if r["ratios"].get(site) is not None]
                    if a and b:
                        diffs.append(float(np.mean(a) - np.mean(b)))
                if not diffs:
                    continue
                mean = float(np.mean(diffs))
                rc = (rep["per_site"].get(site) or {}).get("rc_x_height")
                per_site[site] = {"subjects": len(diffs), "mean_diff_x_height": mean,
                                  "exceeds_repeatability": None if rc is None else abs(mean) > rc}
            res[lv] = per_site
        out[factor] = {"evaluable": True, "reference": ref, "levels": res}
    return out


def tape_accuracy(recordings: list[dict]) -> dict:
    from backend.services.body_measurements import aggregate_tape
    records = [{**r["tape"], "source_key": r["source_key"]} for r in recordings if r.get("tape")]
    if not records:
        return {"subjects": 0, "note": "no non-excluded tape records: accuracy in cm is not yet established"}
    return aggregate_tape(records)


def select_fit_config(rows: list[dict], fit_config: str = "latest") -> tuple[list[dict], str | None]:
    """Keep runs made with one fit configuration: 'latest' = that of the newest run that records one,
    'any' = no filter (mixes code versions; only for exploration), or an exact JSON string."""
    if fit_config == "any":
        return rows, None
    if fit_config == "latest":
        tagged = [r for r in rows if r["fit_config"]]
        if not tagged:
            return rows, None  # nothing records its config yet: cannot separate versions
        fit_config = max(tagged, key=lambda r: r["created_at"])["fit_config"]
    return [r for r in rows if r["fit_config"] == fit_config], fit_config


def build_report(root: Path, source_key=None, since: float | None = None, default_subject: str | None = None,
                 fit_config: str = "latest") -> dict:
    all_rows = collect_runs(root, source_key, since)
    rows, used_config = select_fit_config(all_rows, fit_config)
    recordings = latest_per_recording(rows)
    rep = repeatability(recordings, default_subject)
    return {"generated_at": time.time(), "runs_found": len(all_rows), "runs_considered": len(rows),
            "fit_config": json.loads(used_config) if used_config else None,
            "fit_config_note": None if used_config else (
                "runs do not record their fit settings, so results from different code versions may be pooled; "
                "pass since= to restrict to runs made after the last fitting change"),
            "distinct_recordings": len(recordings),
            "excluded_tape_records": sum(r["tape_excluded"] for r in rows),
            "recordings": [{k: r[k] for k in ("run_id", "video_id", "source_key", "created_at", "height_cm",
                                              "conditions")} for r in recordings],
            "repeatability": rep, "condition_effects": condition_effects(recordings, rep, default_subject),
            "tape_accuracy": tape_accuracy(recordings)}
