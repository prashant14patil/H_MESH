"""
Unified assessments: one camera reconstruction + optional BCM data, keyed by the same assessment ID and person ID.

* assessment ID = the reconstruction run ID (outputs/reconstructions/<run_id>/).
* person ID     = the subject code recorded in the run's capture conditions (capture_conditions.json), the same
                  code the validation report already groups by. It is a code, never a name.
* camera block  = read from the completed run; always present for a listed assessment.
* bcm block     = ``None`` when no BCM data was added. Nothing in the camera block depends on it.

This module only reads the reconstruction outputs; the pipeline that produces them is unchanged.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from backend.services import body_reconstruction as recon
from backend.services import measurement_validation as mv
from backend.services.bcm import demo, records, schema, store

CAMERA_SITES = ("chest", "waist", "abdomen", "hip", "upper_arm_left", "upper_arm_right", "forearm_left",
                "forearm_right", "thigh_left", "thigh_right", "calf_left", "calf_right")


class AssessmentNotFoundError(Exception):
    """No completed camera assessment with this ID."""


# ------------------------------------------------------------------ helpers
def _iso(ts) -> str | None:
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds")
    return ts if isinstance(ts, str) else None


def _load_completed(run_id: str) -> tuple[dict, object]:
    run_dir = recon._run_dir(run_id)          # raises ValueError for a malformed ID
    run = recon.get_run(run_id)
    if run is None or run.get("status") != "completed" or not (run.get("result") or {}).get("measurements"):
        raise AssessmentNotFoundError("No completed reconstruction with measurements for this assessment ID")
    return run, run_dir


def _person(run_dir) -> str | None:
    cond = mv._read_json(run_dir / mv.CONDITIONS_FILE) or {}
    return cond.get("subject_id") or None


def _site_validity(run_dir) -> dict:
    data = mv._read_json(run_dir / "measurements.json") or {}
    return {k: bool(v.get("valid")) for k, v in (data.get("measurements") or {}).items() if isinstance(v, dict)}


def camera_block(run: dict, run_dir) -> dict:
    res = run.get("result") or {}
    metric = res.get("metric_scale") or {}
    cm = metric.get("measurements_cm") or None
    ratios = res.get("measurements") or {}
    valid = _site_validity(run_dir)
    warnings = res.get("warnings") or run.get("warnings") or []
    return {
        "available": True,
        "method": "360° video -> multi-view SMPL fit (HMR2 initialisation)",
        "mesh_obj": metric.get("mesh_obj") or res.get("mesh_obj"),
        "unit": "cm" if cm else "ratio_of_height",
        "height_cm": metric.get("height_cm"),
        "relative_scale_uncertainty": metric.get("relative_scale_uncertainty"),
        "measurements_cm": {k: cm.get(k) for k in CAMERA_SITES if k in cm} if cm else None,
        "measurements_ratio": {k: ratios.get(k) for k in CAMERA_SITES if k in ratios},
        "site_valid": valid,
        "quality": {
            "measurements_valid": res.get("measurements_valid"),
            "geometry_valid": res.get("geometry_valid"),
            "coverage_sufficient": res.get("coverage_sufficient"),
            "consistent_single_subject": res.get("consistent_single_subject"),
            "frames_selected": res.get("frames_selected"),
            "silhouette_iou_fit": res.get("silhouette_iou_fit"),
            "held_out_iou": res.get("held_out_iou"),
            "keypoint_error_pct_height": res.get("keypoint_error_pct_height"),
            "warning_count": len(warnings),
            "warnings": [w.get("message") if isinstance(w, dict) else str(w) for w in warnings][:20],
        },
    }


def _bcm_summary(rec: dict | None) -> dict:
    if not rec:
        return {"available": False}
    return {"available": True, "is_demo": rec["source"].get("is_demo", False), "source": rec["source"]["type"],
            "measured_at": rec.get("measured_at"), "value_count": len(rec.get("values") or {}),
            "profile": rec.get("profile", {}).get("key")}


# ------------------------------------------------------------------ reads
def get_assessment(run_id: str) -> dict:
    run, run_dir = _load_completed(run_id)
    return {
        "assessment_id": run_id,
        "person_id": _person(run_dir),
        "assessed_at": _iso(run.get("created_at")),
        "camera": camera_block(run, run_dir),
        "bcm": store.read(run_dir),
    }


def list_assessments(person_id: str | None = None, limit: int = 100) -> list[dict]:
    """Completed camera assessments, newest first, with person and a BCM availability summary."""
    rows = []
    for r in recon.list_completed_runs(200):
        run_dir = recon._run_dir(r["run_id"])
        pid = _person(run_dir)
        if person_id is not None and pid != person_id:
            continue
        rows.append({"assessment_id": r["run_id"], "person_id": pid, "assessed_at": _iso(r.get("created_at")),
                     "created_at": r.get("created_at"), "height_cm": r.get("height_cm"),
                     "camera_unit": "cm" if r.get("measurements_cm") else "ratio_of_height",
                     "source_key": r.get("source_key"), "bcm": _bcm_summary(store.read(run_dir))})
    return rows[:limit]


def list_persons() -> list[dict]:
    counts: dict[str, dict] = {}
    for a in list_assessments(limit=200):
        if not a["person_id"]:
            continue
        p = counts.setdefault(a["person_id"], {"person_id": a["person_id"], "assessments": 0, "with_bcm": 0,
                                               "latest": a["assessed_at"]})
        p["assessments"] += 1
        p["with_bcm"] += int(a["bcm"]["available"])
    return sorted(counts.values(), key=lambda p: p["person_id"])


# ------------------------------------------------------------------ person
def set_person(run_id: str, person_id: str) -> dict:
    """Assign the assessment to a person (subject code). Keeps any other recorded capture conditions."""
    _, run_dir = _load_completed(run_id)
    existing = mv._read_json(run_dir / mv.CONDITIONS_FILE) or {}
    payload = {k: existing[k] for k in ("clothing", "distance_m", "posture", "note", "height_cm") if k in existing}
    payload["subject_id"] = person_id
    cond = mv.write_conditions(run_dir, payload)
    rec = store.read(run_dir)
    if rec is not None and rec.get("person_id") != cond["subject_id"]:
        rec["person_id"] = cond["subject_id"]
        store.write(run_dir, rec)
    return get_assessment(run_id)


# ------------------------------------------------------------------ BCM writes
def save_bcm(run_id: str, payload: dict, source_type: str = "manual", source_extra: dict | None = None) -> dict:
    """Create or replace (new revision) the assessment's BCM record. The previous revision stays in the history."""
    _, run_dir = _load_completed(run_id)
    previous = store.read(run_dir)
    record = records.build_record(payload, assessment_id=run_id, person_id=_person(run_dir),
                                  source_type=source_type, previous=previous, source_extra=source_extra)
    return store.write(run_dir, record)


def save_demo_bcm(run_id: str, sample_set: int = 1, profile: str = "generic_segmental") -> dict:
    return save_bcm(run_id, demo.demo_payload(sample_set, profile), "demo", {"demo_set": sample_set})


def import_bcm(run_id: str, fmt: str, payload: dict, filename: str | None = None) -> dict:
    """Import a record in the documented bcm-json-v1 format, keeping the file's content verbatim."""
    if fmt not in schema.IMPORT_FORMATS:
        raise records.BCMValidationError({"format": f"must be one of {', '.join(schema.IMPORT_FORMATS)}"})
    if not isinstance(payload, dict):
        raise records.BCMValidationError({"payload": "must be an object"})
    if len(json.dumps(payload)) > 200_000:
        raise records.BCMValidationError({"payload": "is too large"})
    extra = {"import_format": fmt, "imported_filename": records._clean_text(filename, 120) or None,
             "original_payload": payload}
    body = {k: payload.get(k) for k in ("profile", "device_model", "measured_at", "values", "note")}
    return save_bcm(run_id, body, "import", extra)


def remove_bcm(run_id: str) -> dict | None:
    _, run_dir = _load_completed(run_id)
    return store.remove(run_dir, records._now_iso())


def bcm_history(run_id: str) -> list[dict]:
    _, run_dir = _load_completed(run_id)
    return store.history(run_dir)


# ------------------------------------------------------------------ history / comparison
def _camera_values(cam: dict) -> tuple[str, dict]:
    if cam.get("measurements_cm"):
        return "cm", cam["measurements_cm"]
    return "ratio_of_height", cam.get("measurements_ratio") or {}


def compare(base_id: str, other_id: str) -> dict:
    """Field-by-field change from ``base_id`` (earlier) to ``other_id`` (later).

    Camera deltas are in cm only when both assessments are height-scaled, otherwise in ratios of height. A BCM delta
    exists only where BOTH assessments have that BCM value; demo values are flagged so they are never read as real.
    """
    a, b = get_assessment(base_id), get_assessment(other_id)
    ua, va = _camera_values(a["camera"])
    ub, vb = _camera_values(b["camera"])
    unit = "cm" if ua == ub == "cm" else "ratio_of_height"
    if unit == "ratio_of_height":
        va, vb = a["camera"]["measurements_ratio"], b["camera"]["measurements_ratio"]
    camera_rows = []
    for site in CAMERA_SITES:
        x, y = va.get(site), vb.get(site)
        camera_rows.append({"site": site, "before": x, "after": y,
                            "delta": (y - x) if isinstance(x, (int, float)) and isinstance(y, (int, float)) else None})

    bcm_a, bcm_b = a["bcm"], b["bcm"]
    bcm_rows = []
    if bcm_a or bcm_b:
        keys = [k for k in schema.PARAMETERS if k in (bcm_a or {}).get("values", {}) or k in (bcm_b or {}).get("values", {})]
        for k in keys:
            x = ((bcm_a or {}).get("values", {}).get(k) or {}).get("value")
            y = ((bcm_b or {}).get("values", {}).get(k) or {}).get("value")
            bcm_rows.append({"key": k, "unit": schema.PARAMETERS[k].unit, "before": x, "after": y,
                             "delta": (y - x) if x is not None and y is not None else None})
    demo_involved = bool((bcm_a and bcm_a["source"].get("is_demo")) or (bcm_b and bcm_b["source"].get("is_demo")))
    return {
        "before": {"assessment_id": base_id, "assessed_at": a["assessed_at"], "person_id": a["person_id"],
                   "bcm": _bcm_summary(bcm_a)},
        "after": {"assessment_id": other_id, "assessed_at": b["assessed_at"], "person_id": b["person_id"],
                  "bcm": _bcm_summary(bcm_b)},
        "same_person": bool(a["person_id"]) and a["person_id"] == b["person_id"],
        "camera": {"unit": unit, "rows": camera_rows},
        "bcm": {"available": bool(bcm_a and bcm_b), "demo_involved": demo_involved, "rows": bcm_rows,
                "message": None if (bcm_a and bcm_b) else
                ("Neither assessment has BCM data." if not (bcm_a or bcm_b)
                 else "Only one assessment has BCM data, so no BCM change can be computed.")},
    }


def person_history(person_id: str) -> dict:
    """All assessments of one person, oldest first. BCM series only include assessments that have BCM data."""
    rows = sorted(list_assessments(person_id=person_id, limit=200), key=lambda r: r.get("created_at") or 0)
    series_camera, series_bcm = [], []
    for r in rows:
        full = get_assessment(r["assessment_id"])
        unit, vals = _camera_values(full["camera"])
        series_camera.append({"assessment_id": r["assessment_id"], "assessed_at": r["assessed_at"],
                              "unit": unit, "values": vals})
        if full["bcm"]:
            series_bcm.append({"assessment_id": r["assessment_id"], "assessed_at": r["assessed_at"],
                               "measured_at": full["bcm"].get("measured_at"),
                               "is_demo": full["bcm"]["source"].get("is_demo", False),
                               "values": {k: v["value"] for k, v in full["bcm"]["values"].items()}})
    return {"person_id": person_id, "assessments": rows, "camera_series": series_camera,
            "bcm_series": series_bcm,
            # A trend needs at least two BCM records; with fewer, the UI shows no BCM trend at all.
            "bcm_trend_available": len(series_bcm) >= 2}
