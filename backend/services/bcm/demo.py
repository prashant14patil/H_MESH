"""
DEMO DATA for showing how BCM values appear next to camera measurements before real BCM data exists.

These are invented, round sample numbers. They are not Tanita measurements, not derived from the person in the
video and not from any device. Every record built from them carries ``source.type == "demo"`` and
``source.is_demo == True``, and the UI labels them DEMO DATA wherever they appear.

Three sample sets let a multi-assessment history be demonstrated (set 1 -> 3 reads like a gradual change).
"""
from __future__ import annotations

from backend.services.bcm import schema

DEMO_NOTE = "DEMO DATA - invented sample values for demonstration only. Not a measurement of this person."

_SETS = (
    {"weight": 78.0, "body_fat_pct": 24.0, "fat_mass": 18.7, "fat_free_mass": 59.3, "muscle_mass": 56.3,
     "bone_mass": 3.0, "visceral_fat_rating": 9, "total_body_water_pct": 54.0, "total_body_water_kg": 42.1,
     "bmr": 1700, "metabolic_age": 38, "bmi": 25.2,
     "seg": {"right_arm": (20.0, 0.9, 3.1), "left_arm": (21.0, 0.9, 3.0), "trunk": (25.0, 10.1, 30.5),
             "right_leg": (23.0, 3.4, 9.9), "left_leg": (23.0, 3.4, 9.8)}},
    {"weight": 77.0, "body_fat_pct": 23.0, "fat_mass": 17.7, "fat_free_mass": 59.3, "muscle_mass": 56.4,
     "bone_mass": 3.0, "visceral_fat_rating": 9, "total_body_water_pct": 54.5, "total_body_water_kg": 42.0,
     "bmr": 1695, "metabolic_age": 37, "bmi": 24.9,
     "seg": {"right_arm": (19.5, 0.9, 3.1), "left_arm": (20.5, 0.9, 3.0), "trunk": (24.0, 9.6, 30.6),
             "right_leg": (22.5, 3.3, 9.9), "left_leg": (22.5, 3.3, 9.8)}},
    {"weight": 76.0, "body_fat_pct": 22.0, "fat_mass": 16.7, "fat_free_mass": 59.3, "muscle_mass": 56.5,
     "bone_mass": 3.0, "visceral_fat_rating": 8, "total_body_water_pct": 55.0, "total_body_water_kg": 41.8,
     "bmr": 1690, "metabolic_age": 36, "bmi": 24.5,
     "seg": {"right_arm": (19.0, 0.8, 3.1), "left_arm": (20.0, 0.8, 3.0), "trunk": (23.0, 9.0, 30.7),
             "right_leg": (22.0, 3.2, 9.9), "left_leg": (22.0, 3.2, 9.8)}},
)

DEMO_SETS = len(_SETS)


def demo_payload(sample_set: int = 1, profile: str = "generic_segmental") -> dict:
    """Input payload for ``records.build_record`` (source type 'demo'), restricted to the profile's parameters."""
    if not isinstance(sample_set, int) or isinstance(sample_set, bool) or not 1 <= sample_set <= DEMO_SETS:
        raise ValueError(f"sample_set must be 1-{DEMO_SETS}")
    if profile not in schema.PROFILES:
        raise ValueError(f"unknown profile '{profile}'")
    data = _SETS[sample_set - 1]
    flat = {k: v for k, v in data.items() if k != "seg"}
    for seg, (fat_pct, fat_kg, muscle_kg) in data["seg"].items():
        flat[f"segment_fat_pct_{seg}"] = fat_pct
        flat[f"segment_fat_mass_{seg}"] = fat_kg
        flat[f"segment_muscle_mass_{seg}"] = muscle_kg
    allowed = schema.PROFILES[profile].parameters
    values = {k: {"value": v, "unit": schema.PARAMETERS[k].unit} for k, v in flat.items() if k in allowed}
    return {"profile": profile, "device_model": None, "values": values, "note": DEMO_NOTE}
