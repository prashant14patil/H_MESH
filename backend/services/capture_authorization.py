"""Recording-bound operator attestation; not authenticated proof of consent."""
from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from pathlib import Path

from backend.services import video_processor


def _directory(video_id):
    try:
        if str(uuid.UUID(video_id)) != video_id:
            raise ValueError
    except (TypeError, ValueError, AttributeError):
        raise ValueError("Invalid recording ID") from None
    return video_processor._OUTPUT_ROOT / video_id


def _source_hash(video_id):
    directory = _directory(video_id)
    sources = [p for p in directory.glob("original.*") if p.is_file()]
    if len(sources) != 1:
        raise ValueError("Recording source is missing or ambiguous")
    digest = hashlib.sha256()
    with sources[0].open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def attest(video_id, authorized, consent):
    if authorized is not True or consent is not True:
        raise ValueError("Explicit authorization and consent confirmation are required")
    digest = _source_hash(video_id)
    record = {"video_id": video_id, "source_sha256": digest, "authorized": True,
              "consent_confirmed": True, "confirmed_at": time.time(),
              "note": "Operator assertion; not authenticated or independently verified consent."}
    directory = _directory(video_id)
    temp = directory / (str(uuid.uuid4()) + ".attestation.tmp")
    try:
        temp.write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")
        os.replace(temp, directory / "authorization.json")
    finally:
        if temp.exists():
            temp.unlink()
    return record


def require_authorization(video_id):
    directory = _directory(video_id)
    path = directory / "authorization.json"
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise ValueError("Recording authorization and consent confirmation required") from None
    if (not isinstance(record, dict) or record.get("video_id") != video_id
            or record.get("authorized") is not True or record.get("consent_confirmed") is not True
            or record.get("source_sha256") != _source_hash(video_id)):
        raise ValueError("Recording authorization is invalid or source has changed")
    return record
