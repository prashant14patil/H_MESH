# BCM integration layer

The application is a standalone camera-based body-measurement platform. BCM (body-composition monitor) data is an
optional second source on the same assessment. No Tanita device, SDK or protocol is implemented. Nothing in the
camera pipeline (HMR2 / multi-view SMPL fit / SHAPY) was changed.

## Data model

| Concept | Where it lives |
|---|---|
| Assessment ID | reconstruction run ID, `outputs/reconstructions/<run_id>/` |
| Person ID | `subject_id` in `capture_conditions.json` (a code, never a name) |
| Camera block | read from the completed run (`status.json`, `measurements.json`), never modified |
| BCM record | `bcm_record.json` in the same folder; absent = no BCM data |
| BCM audit log | `bcm_history.jsonl`, append-only: every saved revision and every removal |

A BCM record (schema version 1):

```json
{
  "schema_version": 1, "record_id": "uuid", "assessment_id": "<run_id>", "person_id": "S01", "revision": 2,
  "source": {"type": "manual|demo|import|device", "is_demo": false, "label": "...", "device_model": null},
  "profile": {"key": "generic_whole_body", "label": "...", "verified_device_spec": false},
  "measured_at": "2026-11-19T09:30", "recorded_at": "...", "updated_at": "...",
  "values": {"weight": {"value": 80.01, "unit": "kg", "original": {"value": 176.4, "unit": "lb"}}},
  "missing": ["fat_mass", "..."],
  "warnings": [{"code": "fat_mass_mismatch", "message": "..."}],
  "note": ""
}
```

* Every value stores its canonical unit and the original reading as entered or imported.
* Parameters the profile supports but the record lacks are listed in `missing`. They are never estimated.
* Consistency problems (fat mass vs. weight × fat %, etc.) are warnings. The values are kept as reported.
* Profiles (`backend/services/bcm/schema.py`) limit which fields can be entered. The built-in profiles are generic
  and have `verified_device_spec: false`. When the manufacturer's documentation for a real model is available,
  register a profile listing exactly what that model reports, with `verified_device_spec=True`.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/assessments[?person_id=]` | assessment list with BCM availability |
| GET | `/api/assessments/{id}` | unified assessment: `camera` + `bcm` (or null) |
| PUT | `/api/assessments/{id}/person` | assign a person code |
| PUT | `/api/assessments/{id}/bcm` | create or edit (new revision) a manual record |
| DELETE | `/api/assessments/{id}/bcm` | remove (kept in the audit log) |
| POST | `/api/assessments/{id}/bcm/demo` | attach DEMO DATA (sample set 1-3) |
| POST | `/api/assessments/{id}/bcm/import` | import a `bcm-json-v1` document |
| GET | `/api/assessments/{id}/bcm/history` | audit log |
| GET | `/api/assessments/compare?before=&after=` | camera and BCM changes; BCM only where both have the value |
| GET | `/api/assessments/persons/{pid}/history` | per-person series; `bcm_trend_available` needs ≥ 2 BCM records |
| GET | `/api/bcm/schema` | parameters, units, profiles |
| GET | `/api/bcm/devices` | device-integration status (`not_connected`) |

Validation errors return 422 with `{"detail": {"message", "errors": {field: message}}}`.

## Import format `bcm-json-v1`

This is the application's own format, not a Tanita export format. Any tool can produce it:

```json
{"profile": "generic_whole_body", "measured_at": "2026-11-18T08:00", "device_model": "optional",
 "values": {"weight": {"value": 72.4, "unit": "kg"}, "body_fat_pct": {"value": 21.0, "unit": "%"}}, "note": ""}
```

The whole document is kept unchanged in `source.original_payload`. A converter for a real device export can be
added once that export format is known from the device's documentation.

## Demo data

`backend/services/bcm/demo.py` holds three invented, internally consistent sample sets. Records made from them
always have `source.type = "demo"` and `source.is_demo = true`, and the UI labels them **DEMO DATA** everywhere. Editing
a demo record starts from an empty form, so sample values never carry over into a real record.

## Adding a device later

1. Implement `backend/services/bcm/devices.py::BCMDeviceAdapter` (`status()`, `read_latest() -> DeviceReading`) from
   the manufacturer's documented interface. Keep the raw device output in `DeviceReading.raw`.
2. Register a verified profile for the model in `schema.py`.
3. `register_adapter(...)`, and add a route that stores `read_latest()` through
   `assessments.save_bcm(..., source_type="device")`.

The assessment structure, storage and UI need no changes for this.

## Tests

* `backend/tests/test_bcm_assessments.py`: camera-only, manual entry, demo, edit/revisions, partial records,
  multiple assessments per person, comparison with and without BCM, import, device status, and a check that the
  reconstruction files are never modified.
* `frontend/src/components/bcmModel.test.js`: form/payload round-trip, profile filtering, segment pairing, demo badge.
