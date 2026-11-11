"""
backend/utils/validators.py
----------------------------
Input validation helpers for uploaded images.

Responsibilities:
- Check file extension against an allowlist.
- Verify the actual MIME type from the file header (magic bytes),
  not just the filename, to prevent extension spoofing.
- Enforce a configurable maximum file size.

Environment variables:
    MAX_UPLOAD_SIZE_MB  – Maximum allowed upload in megabytes (default: 10).

Public surface:
    validate_image(filename, data)  – Raises ValidationError if invalid.
    ValidationError                 – Subclass of ValueError for bad uploads.

Note:
    imghdr was deprecated in Python 3.11 and removed in 3.13.
    We use magic-byte detection directly to avoid this dependency.
"""

import logging
import os

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_MAX_SIZE_MB: float = float(os.getenv("MAX_UPLOAD_SIZE_MB", "10"))
_MAX_SIZE_BYTES: int = int(_MAX_SIZE_MB * 1024 * 1024)

# Allowed extensions mapped to their magic-byte signatures.
# JPEG: FF D8 FF
# PNG:  89 50 4E 47 0D 0A 1A 0A
_ALLOWED_EXTENSIONS: dict[str, str] = {
    ".jpg": "jpeg",
    ".jpeg": "jpeg",
    ".png": "png",
}

_MAGIC_BYTES: dict[str, bytes] = {
    "jpeg": b"\xff\xd8\xff",
    "png": b"\x89PNG",
}


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------
class ValidationError(ValueError):
    """Raised when an uploaded image fails validation."""


# ---------------------------------------------------------------------------
# Public validator
# ---------------------------------------------------------------------------
def validate_image(filename: str, data: bytes) -> None:
    """
    Validate an uploaded image by extension, magic bytes, and size.

    Parameters
    ----------
    filename:
        Original filename as reported by the client (e.g. ``photo.jpg``).
    data:
        Raw file contents.

    Raises
    ------
    ValidationError
        If the file fails any validation check.

    Notes
    -----
    - Extension check prevents obviously wrong file types being submitted.
    - Magic-byte check prevents extension spoofing.
    - Size check prevents excessively large uploads from reaching the model.
    """
    _check_extension(filename)
    _check_magic_bytes(filename, data)
    _check_size(filename, data)
    logger.debug("Image '%s' passed all validation checks (%d bytes).", filename, len(data))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _check_extension(filename: str) -> None:
    """Reject files whose extension is not in the allowlist."""
    if not filename:
        raise ValidationError("No filename provided.")

    ext = _get_extension(filename)
    if ext not in _ALLOWED_EXTENSIONS:
        allowed = ", ".join(_ALLOWED_EXTENSIONS)
        raise ValidationError(
            f"File '{filename}' has unsupported extension '{ext}'. "
            f"Allowed extensions: {allowed}."
        )


def _check_magic_bytes(filename: str, data: bytes) -> None:
    """
    Detect the actual image type from magic bytes and verify it matches
    the declared extension.  Replaces the deprecated imghdr module.

    JPEG files begin with: FF D8 FF
    PNG  files begin with: 89 50 4E 47  (i.e. the ASCII bytes ".PNG")
    """
    ext = _get_extension(filename)
    expected_type = _ALLOWED_EXTENSIONS[ext]

    if not data:
        raise ValidationError(f"File '{filename}' is empty.")

    # Detect the actual format from the file header.
    detected_type: str | None = None
    for fmt, signature in _MAGIC_BYTES.items():
        if data[: len(signature)] == signature:
            detected_type = fmt
            break

    if detected_type is None:
        raise ValidationError(
            f"File '{filename}' does not appear to be a valid image. "
            "The file header could not be recognised as JPEG or PNG."
        )

    if detected_type != expected_type:
        raise ValidationError(
            f"File '{filename}' has extension '{ext}' but its content "
            f"looks like '{detected_type}'. Upload a genuine JPEG or PNG."
        )


def _check_size(filename: str, data: bytes) -> None:
    """Reject files that exceed the configured size limit."""
    size_bytes = len(data)
    if size_bytes == 0:
        raise ValidationError(f"File '{filename}' is empty.")

    if size_bytes > _MAX_SIZE_BYTES:
        size_mb = size_bytes / (1024 * 1024)
        raise ValidationError(
            f"File '{filename}' is {size_mb:.1f} MB, which exceeds the "
            f"maximum allowed size of {_MAX_SIZE_MB} MB."
        )


def _get_extension(filename: str) -> str:
    """Return the lowercase file extension including the leading dot."""
    _, ext = os.path.splitext(filename)
    return ext.lower()
