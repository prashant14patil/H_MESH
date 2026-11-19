"""
BCM (body-composition monitor) data schema.

A BCM record is OPTIONAL data attached to one assessment (one reconstruction run). The camera assessment never
depends on it. This module defines:

* the parameter catalogue: every quantity a record may carry, its canonical unit, the units accepted on input and
  a plausibility range (values outside it are rejected as typos, not clipped);
* measurement profiles: which parameters a given device or assessment type reports. A record is validated against
  its profile, so a field the selected device does not provide cannot be entered by mistake.

None of the built-in profiles is a statement about what a particular Tanita model measures. They are generic field
sets for manual entry. When the manufacturer's documentation for a real device is available, add a profile with
``verified_device_spec=True`` that lists exactly the parameters that device reports (see ``register_profile``).
"""
from __future__ import annotations

from dataclasses import dataclass, field

SCHEMA_VERSION = 1

# Canonical-unit conversions: (input unit -> factor to the canonical unit). Units are stored explicitly on every value.
LB_TO_KG = 0.45359237
KJ_TO_KCAL = 1 / 4.184

SEGMENTS = ("right_arm", "left_arm", "trunk", "right_leg", "left_leg")
SEGMENT_LABELS = {"right_arm": "Right arm", "left_arm": "Left arm", "trunk": "Trunk",
                  "right_leg": "Right leg", "left_leg": "Left leg"}


@dataclass(frozen=True)
class Parameter:
    key: str
    label: str
    unit: str                         # canonical unit stored in the record
    units: dict                       # accepted input unit -> factor to canonical
    minimum: float
    maximum: float
    group: str                        # whole_body | segmental_fat | segmental_muscle
    segment: str | None = None
    decimals: int = 1
    help: str = ""


def _mass(key, label, lo, hi, group="whole_body", segment=None, help=""):
    return Parameter(key, label, "kg", {"kg": 1.0, "lb": LB_TO_KG}, lo, hi, group, segment, 1, help)


def _pct(key, label, lo, hi, group="whole_body", segment=None, help=""):
    return Parameter(key, label, "%", {"%": 1.0}, lo, hi, group, segment, 1, help)


_WHOLE_BODY = [
    _mass("weight", "Weight", 2, 350),
    _pct("body_fat_pct", "Body fat", 1, 75),
    _mass("fat_mass", "Fat mass", 0, 250),
    _mass("fat_free_mass", "Fat-free mass", 2, 200),
    _mass("muscle_mass", "Muscle mass", 2, 200,
          help="As reported by the device; manufacturers define 'muscle mass' differently."),
    _mass("bone_mass", "Bone mass (estimated)", 0.2, 10),
    Parameter("visceral_fat_rating", "Visceral fat rating", "rating", {"rating": 1.0}, 1, 60, "whole_body", None, 1,
              "Device-specific scale; ratings from different devices are not comparable."),
    _pct("total_body_water_pct", "Total body water", 20, 80),
    _mass("total_body_water_kg", "Total body water", 5, 120),
    Parameter("bmr", "Basal metabolic rate", "kcal", {"kcal": 1.0, "kJ": KJ_TO_KCAL}, 500, 6000, "whole_body", None, 0,
              "Per day, as estimated by the device."),
    Parameter("metabolic_age", "Metabolic age", "years", {"years": 1.0}, 5, 120, "whole_body", None, 0),
    Parameter("bmi", "BMI (device-reported)", "kg/m²", {"kg/m²": 1.0}, 8, 90, "whole_body", None, 1),
]

_SEGMENTAL = []
for _seg in SEGMENTS:
    _SEGMENTAL.append(_pct(f"segment_fat_pct_{_seg}", f"{SEGMENT_LABELS[_seg]} fat", 1, 80, "segmental_fat", _seg))
    _SEGMENTAL.append(_mass(f"segment_fat_mass_{_seg}", f"{SEGMENT_LABELS[_seg]} fat mass", 0, 120,
                            "segmental_fat", _seg))
    _SEGMENTAL.append(_mass(f"segment_muscle_mass_{_seg}", f"{SEGMENT_LABELS[_seg]} muscle mass", 0.2, 120,
                            "segmental_muscle", _seg))

PARAMETERS: dict[str, Parameter] = {p.key: p for p in _WHOLE_BODY + _SEGMENTAL}
WHOLE_BODY_KEYS = tuple(p.key for p in _WHOLE_BODY)
SEGMENTAL_KEYS = tuple(p.key for p in _SEGMENTAL)


@dataclass(frozen=True)
class Profile:
    key: str
    label: str
    description: str
    parameters: tuple
    verified_device_spec: bool = False   # True only for a profile taken from a manufacturer's specification
    manufacturer: str | None = None
    model: str | None = None
    notes: tuple = field(default_factory=tuple)


_GENERIC_NOTE = ("Generic field set for manual entry. It does not describe what any specific Tanita model reports: "
                 "leave fields blank that your device or assessment did not produce.")

PROFILES: dict[str, Profile] = {}


def register_profile(profile: Profile) -> Profile:
    """Add a device / assessment profile. Unknown parameter keys are a programming error."""
    unknown = [k for k in profile.parameters if k not in PARAMETERS]
    if unknown:
        raise ValueError(f"profile {profile.key}: unknown parameter(s) {', '.join(unknown)}")
    if not profile.parameters:
        raise ValueError(f"profile {profile.key}: no parameters")
    PROFILES[profile.key] = profile
    return profile


register_profile(Profile(
    "generic_whole_body", "Whole-body composition (generic)",
    "Whole-body values only: weight, fat, muscle, water, metabolic estimates.",
    WHOLE_BODY_KEYS, notes=(_GENERIC_NOTE,)))
register_profile(Profile(
    "generic_segmental", "Whole-body + segmental composition (generic)",
    "Whole-body values plus fat and muscle per arm, leg and trunk, for segmental analysers.",
    WHOLE_BODY_KEYS + SEGMENTAL_KEYS, notes=(_GENERIC_NOTE,)))
register_profile(Profile(
    "weight_only", "Weight only (scale)",
    "A plain scale reading with no composition analysis.",
    ("weight",), notes=("For a scale without body-composition analysis.",)))

DEFAULT_PROFILE = "generic_whole_body"

SOURCE_TYPES = ("manual", "demo", "import", "device")
IMPORT_FORMATS = ("bcm-json-v1",)


def describe() -> dict:
    """JSON description of the schema for the frontend form and for API clients."""
    return {
        "schema_version": SCHEMA_VERSION,
        "source_types": list(SOURCE_TYPES),
        "import_formats": list(IMPORT_FORMATS),
        "segments": [{"key": s, "label": SEGMENT_LABELS[s]} for s in SEGMENTS],
        "parameters": [
            {"key": p.key, "label": p.label, "unit": p.unit, "units": list(p.units), "min": p.minimum,
             "max": p.maximum, "group": p.group, "segment": p.segment, "decimals": p.decimals, "help": p.help}
            for p in PARAMETERS.values()
        ],
        "profiles": [
            {"key": pr.key, "label": pr.label, "description": pr.description, "parameters": list(pr.parameters),
             "verified_device_spec": pr.verified_device_spec, "manufacturer": pr.manufacturer, "model": pr.model,
             "notes": list(pr.notes)}
            for pr in PROFILES.values()
        ],
        "default_profile": DEFAULT_PROFILE,
    }
