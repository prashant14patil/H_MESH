"""
Validation and normalisation of BCM records.

Every value keeps the reading exactly as entered or imported (``original``: value + unit) next to the value converted
to the parameter's canonical unit. Parameters the profile supports but the record does not contain are listed in
``missing``; nothing is ever estimated or filled in. Internal-consistency problems (e.g. fat mass that does not match
weight x body-fat %) are reported as warnings, never silently corrected.
"""
from __future__ import annotations

import math
import time
import uuid
from datetime import datetime, timezone

from backend.services.bcm import schema

MAX_NOTE = 500
MAX_DEVICE_MODEL = 60
# Tolerances for the consistency checks: device rounding of displayed values is ~0.1 kg / 0.1 %.
MASS_TOLERANCE_KG = 1.0


class BCMValidationError(ValueError):
    """Invalid BCM input. ``errors`` maps a field (parameter key or top-level field) to a message."""

    def __init__(self, errors: dict):
        self.errors = dict(errors)
        super().__init__("; ".join(f"{k}: {v}" for k, v in self.errors.items()))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_measured_at(value) -> str:
    """ISO-8601 timestamp of the BCM measurement. Naive times are kept as given (local wall-clock time)."""
    if value in (None, ""):
        return _now_iso()
    if not isinstance(value, str) or len(value) > 40:
        raise BCMValidationError({"measured_at": "must be an ISO-8601 date-time, e.g. 2026-11-19T09:30"})
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise BCMValidationError({"measured_at": "must be an ISO-8601 date-time, e.g. 2026-11-19T09:30"}) from exc
    aware = parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    if aware.timestamp() > time.time() + 36 * 3600:   # allow any time zone, reject real future dates
        raise BCMValidationError({"measured_at": "is in the future"})
    if parsed.year < 1990:
        raise BCMValidationError({"measured_at": "is implausibly old"})
    return parsed.isoformat(timespec="minutes" if parsed.second == 0 else "seconds")


def _clean_text(value, limit: int) -> str:
    text = "" if value is None else str(value)
    return "".join(ch for ch in text if ch.isprintable())[:limit].strip()


def _number(raw):
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return float(raw) if math.isfinite(raw) else None
    if isinstance(raw, str) and raw.strip():
        try:
            v = float(raw.strip().replace(",", "."))
        except ValueError:
            return None
        return v if math.isfinite(v) else None
    return None


def normalise_values(values, profile: schema.Profile) -> tuple[dict, dict]:
    """Validate raw ``{key: {value, unit} | number | null}`` against a profile. Returns (values, errors)."""
    out, errors = {}, {}
    if values is None:
        values = {}
    if not isinstance(values, dict):
        return {}, {"values": "must be an object keyed by parameter"}
    for key, raw in values.items():
        param = schema.PARAMETERS.get(key)
        if param is None:
            errors[key] = "unknown parameter"
            continue
        if isinstance(raw, dict):
            raw_value, unit = raw.get("value"), raw.get("unit") or param.unit
        else:
            raw_value, unit = raw, param.unit
        if raw_value is None or (isinstance(raw_value, str) and not raw_value.strip()):
            continue                                   # blank = not measured
        if key not in profile.parameters:
            errors[key] = f"not reported by the selected profile ({profile.label})"
            continue
        number = _number(raw_value)
        if number is None:
            errors[key] = "must be a number"
            continue
        if unit not in param.units:
            errors[key] = f"unit must be one of {', '.join(param.units)}"
            continue
        canonical = number * param.units[unit]
        if not param.minimum <= canonical <= param.maximum:
            errors[key] = f"{canonical:.{param.decimals}f} {param.unit} is outside the plausible range " \
                          f"{param.minimum:g}-{param.maximum:g} {param.unit}"
            continue
        out[key] = {"value": round(canonical, 3), "unit": param.unit, "original": {"value": number, "unit": unit}}
    return out, errors


def consistency_warnings(values: dict) -> list[dict]:
    """Cross-checks between values of one record. Warnings only: the readings are kept as reported."""
    v = {k: e["value"] for k, e in values.items()}
    warnings = []

    def warn(code, message):
        warnings.append({"code": code, "message": message})

    w = v.get("weight")
    if w is not None:
        if v.get("fat_mass") is not None and v.get("body_fat_pct") is not None:
            expected = w * v["body_fat_pct"] / 100
            if abs(expected - v["fat_mass"]) > MASS_TOLERANCE_KG:
                warn("fat_mass_mismatch", f"Fat mass {v['fat_mass']:.1f} kg does not match weight x body fat "
                                          f"({expected:.1f} kg).")
        if v.get("fat_mass") is not None and v.get("fat_free_mass") is not None:
            total = v["fat_mass"] + v["fat_free_mass"]
            if abs(total - w) > MASS_TOLERANCE_KG:
                warn("mass_sum_mismatch", f"Fat mass + fat-free mass = {total:.1f} kg, weight is {w:.1f} kg.")
        if v.get("total_body_water_kg") is not None and v.get("total_body_water_pct") is not None:
            expected = w * v["total_body_water_pct"] / 100
            if abs(expected - v["total_body_water_kg"]) > MASS_TOLERANCE_KG:
                warn("water_mismatch", f"Total body water {v['total_body_water_kg']:.1f} kg does not match "
                                       f"weight x water % ({expected:.1f} kg).")
        for key in ("fat_mass", "fat_free_mass", "muscle_mass", "total_body_water_kg"):
            if v.get(key) is not None and v[key] > w:
                warn(f"{key}_exceeds_weight", f"{schema.PARAMETERS[key].label} exceeds body weight.")
        seg_muscle = [v.get(f"segment_muscle_mass_{s}") for s in schema.SEGMENTS]
        if all(x is not None for x in seg_muscle) and sum(seg_muscle) > w:
            warn("segment_muscle_exceeds_weight", "Segmental muscle masses add up to more than body weight.")
    return warnings


def resolve_profile(key) -> schema.Profile:
    key = key or schema.DEFAULT_PROFILE
    profile = schema.PROFILES.get(key)
    if profile is None:
        raise BCMValidationError({"profile": f"unknown profile '{key}'"})
    return profile


def build_record(payload: dict, *, assessment_id: str, person_id: str | None, source_type: str,
                 previous: dict | None = None, source_extra: dict | None = None) -> dict:
    """Validate an input payload and return a complete, normalised record (raises BCMValidationError)."""
    if source_type not in schema.SOURCE_TYPES:
        raise BCMValidationError({"source": f"must be one of {', '.join(schema.SOURCE_TYPES)}"})
    if not isinstance(payload, dict):
        raise BCMValidationError({"payload": "must be an object"})
    errors = {}
    profile = resolve_profile(payload.get("profile"))
    values, value_errors = normalise_values(payload.get("values"), profile)
    errors.update(value_errors)
    try:
        measured_at = parse_measured_at(payload.get("measured_at"))
    except BCMValidationError as exc:
        errors.update(exc.errors)
        measured_at = None
    if not values and not value_errors:
        errors["values"] = "enter at least one measurement (remove the BCM data instead of saving an empty record)"
    if errors:
        raise BCMValidationError(errors)

    now = _now_iso()
    is_demo = source_type == "demo"
    source = {
        "type": source_type,
        "is_demo": is_demo,
        "label": {"manual": "Manually entered", "demo": "DEMO DATA - sample values, not a Tanita measurement",
                  "import": "Imported file", "device": "Device"}[source_type],
        "device_model": _clean_text(payload.get("device_model"), MAX_DEVICE_MODEL) or None,
        **(source_extra or {}),
    }
    return {
        "schema_version": schema.SCHEMA_VERSION,
        "record_id": previous["record_id"] if previous else str(uuid.uuid4()),
        "assessment_id": assessment_id,
        "person_id": person_id,
        "revision": (previous.get("revision", 1) + 1) if previous else 1,
        "source": source,
        "profile": {"key": profile.key, "label": profile.label, "verified_device_spec": profile.verified_device_spec},
        "measured_at": measured_at,
        "recorded_at": previous.get("recorded_at", now) if previous else now,
        "updated_at": now,
        "values": values,
        "missing": [k for k in profile.parameters if k not in values],
        "warnings": consistency_warnings(values),
        "note": _clean_text(payload.get("note"), MAX_NOTE),
    }
