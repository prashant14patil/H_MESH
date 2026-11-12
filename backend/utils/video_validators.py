"""
backend/utils/video_validators.py
----------------------------------
Input validation helpers for uploaded 360° capture videos.

Mirrors the pattern established in backend/utils/validators.py (image
validation) but targets video containers instead of images.

Responsibilities:
- Check file extension against an allowlist (.mp4, .mov, .m4v, .webm).
- Verify the actual container type from magic bytes (the ISO-BMFF
  'ftyp' box used by MP4/QuickTime/M4V, or the EBML magic number used
  by WebM — the format produced by browser MediaRecorder output), not
  just the filename, to prevent extension spoofing.
- Enforce a configurable maximum file size.

Environment variables:
    MAX_VIDEO_SIZE_MB  – Maximum allowed upload in megabytes (default: 200).

Public surface:
    validate_video(filename, data)  – Raises VideoValidationError if invalid.
    VideoValidationError            – Subclass of ValueError for bad uploads.
"""

import logging
import os

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_MAX_SIZE_MB: float = float(os.getenv("MAX_VIDEO_SIZE_MB", "200"))
_MAX_SIZE_BYTES: int = int(_MAX_SIZE_MB * 1024 * 1024)

# Allowed extensions. MP4/MOV/M4V use the ISO Base Media File Format
# container (identified by a 'ftyp' box at byte offset 4). WebM uses the
# unrelated EBML/Matroska container (identified by a magic number at byte
# offset 0). WebM must be accepted because browser MediaRecorder output
# (the "record with camera" capture path) is WebM in virtually all
# browsers -- there is no standard way to record directly to MP4
# client-side.
_ALLOWED_EXTENSIONS: set[str] = {".mp4", ".mov", ".m4v", ".webm"}

# EBML magic number (first 4 bytes of every WebM/Matroska file).
_EBML_MAGIC: bytes = b"\x1a\x45\xdf\xa3"

# Sub-types (the 4-byte "major brand" inside the ftyp box) commonly seen
# for MP4/MOV exports from phones and browsers. This is a soft check —
# we only require the 'ftyp' marker itself to be present; the brand list
# is used purely for diagnostic logging, not hard rejection, since there
# are many valid brands (isom, mp42, qt  , M4V , MSNV, avc1, 3gp5, ...).
_KNOWN_FTYP_BRANDS = {
    b"isom", b"iso2", b"mp41", b"mp42", b"avc1", b"M4V ", b"M4A ",
    b"qt  ", b"3gp4", b"3gp5", b"3g2a", b"dash", b"MSNV",
}


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------
class VideoValidationError(ValueError):
    """Raised when an uploaded video fails validation."""


# ---------------------------------------------------------------------------
# Public validator
# ---------------------------------------------------------------------------
def validate_video(filename: str, data: bytes) -> None:
    """
    Validate an uploaded video by extension, container magic bytes, and size.

    Parameters
    ----------
    filename:
        Original filename as reported by the client (e.g. ``scan.mp4``).
    data:
        Raw file contents.

    Raises
    ------
    VideoValidationError
        If the file fails any validation check.
    """
    _check_extension(filename)
    _check_size(filename, data)
    _check_container(filename, data)
    logger.debug("Video '%s' passed all validation checks (%d bytes).", filename, len(data))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _check_extension(filename: str) -> None:
    """Reject files whose extension is not in the allowlist."""
    if not filename:
        raise VideoValidationError("No filename provided.")

    ext = _get_extension(filename)
    if ext not in _ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(_ALLOWED_EXTENSIONS))
        raise VideoValidationError(
            f"File '{filename}' has unsupported extension '{ext}'. "
            f"Allowed extensions: {allowed}."
        )


def _check_container(filename: str, data: bytes) -> None:
    """
    Verify the file looks like a valid video container by checking its
    magic bytes against the two container families this endpoint
    accepts:

    1. ISO-BMFF (MP4/MOV/M4V) — identified by a 'ftyp' box marker at
       byte offset 4:
           bytes[0:4]  = box size (big-endian uint32)
           bytes[4:8]  = box type, expected to be literally b"ftyp"
           bytes[8:12] = major brand (e.g. b"isom", b"mp42", b"qt  ")

    2. WebM/Matroska (EBML) — identified by the fixed 4-byte EBML magic
       number at byte offset 0 (0x1A45DFA3). This is the format produced
       by browser MediaRecorder output.
    """
    if not data:
        raise VideoValidationError(f"File '{filename}' is empty.")

    # WebM/Matroska check first — a single, fixed 4-byte signature.
    if data[:4] == _EBML_MAGIC:
        return

    # Fall back to the ISO-BMFF 'ftyp' box check for MP4/MOV/M4V.
    if len(data) < 12 or data[4:8] != b"ftyp":
        raise VideoValidationError(
            f"File '{filename}' does not appear to be a valid MP4/MOV/WEBM "
            "video. No recognised container header was found."
        )

    brand = data[8:12]
    if brand not in _KNOWN_FTYP_BRANDS:
        logger.info(
            "Video '%s' has an unrecognised ftyp brand %r — accepting anyway "
            "since the container marker is valid.",
            filename,
            brand,
        )


def _check_size(filename: str, data: bytes) -> None:
    """Reject files that exceed the configured size limit."""
    size_bytes = len(data)
    if size_bytes == 0:
        raise VideoValidationError(f"File '{filename}' is empty.")

    if size_bytes > _MAX_SIZE_BYTES:
        size_mb = size_bytes / (1024 * 1024)
        raise VideoValidationError(
            f"File '{filename}' is {size_mb:.1f} MB, which exceeds the "
            f"maximum allowed size of {_MAX_SIZE_MB} MB."
        )


def _get_extension(filename: str) -> str:
    """Return the lowercase file extension including the leading dot."""
    _, ext = os.path.splitext(filename)
    return ext.lower()
