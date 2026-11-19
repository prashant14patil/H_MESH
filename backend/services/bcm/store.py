"""
File storage of BCM records, inside the assessment's own run directory (outputs/reconstructions/<run_id>/).

* ``bcm_record.json``  - the current record (absent = no BCM data for this assessment).
* ``bcm_history.jsonl`` - append-only audit log: every saved revision and every removal, so an edit or a delete
  never loses the reading that was there before.

Keeping BCM next to the camera outputs means one directory holds the whole assessment; the reconstruction files
themselves are never touched.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

RECORD_FILE = "bcm_record.json"
HISTORY_FILE = "bcm_history.jsonl"
_LOCK = threading.RLock()


def read(run_dir: Path) -> dict | None:
    path = Path(run_dir) / RECORD_FILE
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        return None


def _append_history(run_dir: Path, event: dict) -> None:
    with open(Path(run_dir) / HISTORY_FILE, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, allow_nan=False) + "\n")


def write(run_dir: Path, record: dict) -> dict:
    run_dir = Path(run_dir)
    with _LOCK:
        tmp = run_dir / (RECORD_FILE + ".tmp")
        tmp.write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")
        os.replace(tmp, run_dir / RECORD_FILE)
        _append_history(run_dir, {"event": "saved", "at": record.get("updated_at"), "record": record})
    return record


def remove(run_dir: Path, at: str) -> dict | None:
    run_dir = Path(run_dir)
    with _LOCK:
        current = read(run_dir)
        if current is None:
            return None
        _append_history(run_dir, {"event": "removed", "at": at, "record": current})
        (run_dir / RECORD_FILE).unlink(missing_ok=True)
    return current


def history(run_dir: Path) -> list[dict]:
    path = Path(run_dir) / HISTORY_FILE
    if not path.is_file():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            events.append(json.loads(line))
        except ValueError:
            continue
    return events
