"""
validators.py — 3-layer image validation for tf-image-guard.

Validation pipeline for each image file:

    Layer 1 — Filesystem Check
        Verifies the file exists, is readable, is non-zero, and meets
        the configured min/max byte size thresholds.

    Layer 2 — Pillow Decode
        Opens and fully verifies the image with Pillow (PIL).
        ``ImageFile.LOAD_TRUNCATED_IMAGES = False`` is deliberately NOT set,
        so truncated files are caught here.
        Uses ``img.verify()`` which detects broken headers and data corruption.

    Layer 3 — TensorFlow Decode
        Reads and decodes the image using ``tf.io.read_file`` +
        ``tf.image.decode_image``. This guarantees the file will actually
        load in a ``tf.data`` pipeline. Any file that fails this step would
        silently crash TF training.

All layers are wrapped via ``safe_execute()`` so a failure at any layer
is recorded as a structured :class:`ValidationResult` rather than raising.
Execution NEVER stops.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional

from .utils import SafeResult, format_size, get_file_size, safe_execute

# ──────────────────────────────────────────────────────────────────────────────
# Enums & result types
# ──────────────────────────────────────────────────────────────────────────────

class ValidationStatus(str, Enum):
    OK = "ok"
    CORRUPTED = "corrupted"
    SKIPPED = "skipped"   # e.g., unsupported extension, already quarantined


class ErrorLayer(str, Enum):
    NONE = "none"
    FILESYSTEM = "filesystem"
    PILLOW = "pillow"
    TENSORFLOW = "tensorflow"


@dataclass
class ValidationResult:
    """The result of validating a single image file."""
    file_path: Path
    status: ValidationStatus = ValidationStatus.OK
    error_layer: ErrorLayer = ErrorLayer.NONE
    error_message: str = ""
    file_size: Optional[int] = None
    detected_format: str = ""

    @property
    def is_corrupted(self) -> bool:
        return self.status == ValidationStatus.CORRUPTED

    @property
    def is_ok(self) -> bool:
        return self.status == ValidationStatus.OK

    def to_dict(self) -> dict:
        return {
            "file_path": str(self.file_path),
            "status": self.status.value,
            "error_layer": self.error_layer.value,
            "error_message": self.error_message,
            "file_size_bytes": self.file_size,
            "file_size_human": format_size(self.file_size),
            "detected_format": self.detected_format,
        }


# ──────────────────────────────────────────────────────────────────────────────
# TF import helper — lazy, CPU-only by default
# ──────────────────────────────────────────────────────────────────────────────

def _get_tf(cpu_only: bool = True):
    """
    Lazily import TensorFlow and optionally hide all GPUs so the scanning
    process does not allocate GPU memory.
    """
    import tensorflow as tf  # noqa: PLC0415
    if cpu_only:
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:  # noqa: BLE001
            pass  # If no GPUs present, silently continue
    return tf


# ──────────────────────────────────────────────────────────────────────────────
# Individual layer validators
# ──────────────────────────────────────────────────────────────────────────────

def _validate_filesystem(
    path: Path,
    min_bytes: int = 100,
    max_bytes: Optional[int] = None,
) -> SafeResult:
    """
    Layer 1: Filesystem sanity check.

    Checks:
    - File exists and is a regular file
    - File is readable (os.access)
    - File size >= min_bytes
    - File size <= max_bytes (if set)
    """
    def _check():
        if not path.exists():
            raise FileNotFoundError(f"File does not exist: {path}")
        if not path.is_file():
            raise IsADirectoryError(f"Path is not a file: {path}")
        if not os.access(path, os.R_OK):
            raise PermissionError(f"File is not readable: {path}")

        size = os.path.getsize(path)
        if size < min_bytes:
            raise ValueError(
                f"File too small ({size} bytes, minimum is {min_bytes} bytes) — "
                "likely zero-byte or severely truncated."
            )
        if max_bytes is not None and size > max_bytes:
            raise ValueError(
                f"File too large ({format_size(size)}, maximum is {format_size(max_bytes)})."
            )
        return size

    return safe_execute(_check)


def _validate_pillow(path: Path) -> SafeResult:
    """
    Layer 2: Pillow full decode + verification.

    Uses PIL.Image.open() followed by .verify() to catch:
    - Truncated/partial images
    - Broken JPEG headers
    - Corrupted PNG chunks
    - Invalid or unsupported image data
    """
    def _check():
        from PIL import Image, ImageFile, UnidentifiedImageError  # noqa: PLC0415

        # Do NOT set LOAD_TRUNCATED_IMAGES — we want to catch truncated files
        ImageFile.LOAD_TRUNCATED_IMAGES = False

        try:
            # verify() is destructive on some formats, open fresh each time
            with Image.open(path) as img:
                fmt = img.format or ""
                img.verify()  # Raises on corruption
        except UnidentifiedImageError as exc:
            raise ValueError(f"Pillow cannot identify image format: {exc}") from exc
        except Exception:
            raise

        # Re-open after verify() because verify() leaves image in unusable state
        with Image.open(path) as img:
            fmt = img.format or ""
            img.load()  # Force full decode
        return fmt

    return safe_execute(_check)


def _validate_tensorflow(path: Path, channels: int = 0, cpu_only: bool = True) -> SafeResult:
    """
    Layer 3: TensorFlow image decode check.

    Uses tf.io.read_file + tf.image.decode_image to ensure the file
    is loadable in a real TF data pipeline. Any file that fails here
    would silently break tf.data or cause training crashes.
    """
    def _check():
        tf = _get_tf(cpu_only=cpu_only)
        raw = tf.io.read_file(str(path))
        img_tensor = tf.image.decode_image(
            raw,
            channels=channels,
            expand_animations=False,
        )
        # Force eager evaluation of the tensor
        _ = img_tensor.numpy()
        return True

    return safe_execute(_check)


# ──────────────────────────────────────────────────────────────────────────────
# Main public validator
# ──────────────────────────────────────────────────────────────────────────────

def validate_image(
    path: Path,
    config: Optional[dict] = None,
    logger: Optional[logging.Logger] = None,
) -> ValidationResult:
    """
    Run the full 3-layer validation pipeline on a single image file.

    This function NEVER raises. All exceptions from all layers are
    caught internally and translated to a :class:`ValidationResult`.

    Parameters
    ----------
    path : Path
        Absolute or relative path to the image file.
    config : dict, optional
        Merged configuration dictionary (from :func:`utils.load_config`).
        Determines which layers are active and what thresholds to apply.
    logger : logging.Logger, optional
        Logger for debug messages.

    Returns
    -------
    ValidationResult
    """
    if config is None:
        config = {}

    val_cfg = config.get("validation", {})
    tf_cfg = config.get("tensorflow", {})
    size_cfg = config.get("size_limits", {})

    min_bytes: int = size_cfg.get("min_bytes", 100) or 100
    max_bytes: Optional[int] = size_cfg.get("max_bytes", None)
    channels: int = tf_cfg.get("channels", 0) or 0
    cpu_only: bool = tf_cfg.get("cpu_only", True)

    do_filesystem: bool = val_cfg.get("filesystem_check", True)
    do_pillow: bool = val_cfg.get("pillow_decode", True)
    do_tensorflow: bool = val_cfg.get("tensorflow_decode", True)

    file_size = get_file_size(path)
    result = ValidationResult(
        file_path=path,
        file_size=file_size,
    )

    # ── Layer 1: Filesystem ──────────────────────────────────────────────────
    if do_filesystem:
        fs_result: SafeResult = _validate_filesystem(path, min_bytes=min_bytes, max_bytes=max_bytes)
        if fs_result.failed:
            msg = str(fs_result.error)
            if logger:
                logger.debug("[FILESYSTEM] FAIL %s — %s", path.name, msg)
            result.status = ValidationStatus.CORRUPTED
            result.error_layer = ErrorLayer.FILESYSTEM
            result.error_message = msg
            return result  # Early-out: no point checking further
        else:
            # Update file_size from accurate os.path.getsize call
            result.file_size = fs_result.value

    # ── Layer 2: Pillow ──────────────────────────────────────────────────────
    if do_pillow:
        pil_result: SafeResult = _validate_pillow(path)
        if pil_result.failed:
            msg = str(pil_result.error)
            if logger:
                logger.debug("[PILLOW] FAIL %s — %s", path.name, msg)
            result.status = ValidationStatus.CORRUPTED
            result.error_layer = ErrorLayer.PILLOW
            result.error_message = msg
            return result
        else:
            result.detected_format = pil_result.value or ""

    # ── Layer 3: TensorFlow ──────────────────────────────────────────────────
    if do_tensorflow:
        tf_result: SafeResult = _validate_tensorflow(path, channels=channels, cpu_only=cpu_only)
        if tf_result.failed:
            msg = str(tf_result.error)
            if logger:
                logger.debug("[TENSORFLOW] FAIL %s — %s", path.name, msg)
            result.status = ValidationStatus.CORRUPTED
            result.error_layer = ErrorLayer.TENSORFLOW
            result.error_message = msg
            return result

    # ── All layers passed ────────────────────────────────────────────────────
    if logger:
        logger.debug("[OK] %s", path.name)
    result.status = ValidationStatus.OK
    return result
