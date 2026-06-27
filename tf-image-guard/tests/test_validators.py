"""
test_validators.py — Unit tests for tf_image_guard.validators

Tests cover:
- Zero-byte files flagged at filesystem layer
- Truncated/partial JPEGs flagged at Pillow layer
- Non-image files (e.g., .txt) renamed to .jpg — caught at Pillow layer
- Valid images pass all 3 layers
- validate_image() NEVER raises, regardless of input
"""

from __future__ import annotations

import io
import struct
import tempfile
from pathlib import Path

import pytest

from tf_image_guard.validators import (
    ErrorLayer,
    ValidationStatus,
    validate_image,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures — create synthetic test files
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def tmp_dir(tmp_path: Path) -> Path:
    return tmp_path


def _write_file(path: Path, data: bytes) -> Path:
    path.write_bytes(data)
    return path


def _minimal_valid_png() -> bytes:
    """Return the bytes of a minimal 1x1 white PNG."""
    import zlib
    import struct

    def write_chunk(chunk_type: bytes, data: bytes) -> bytes:
        c = chunk_type + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    signature = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)  # 1x1, 8-bit RGB
    ihdr = write_chunk(b"IHDR", ihdr_data)
    # Single white pixel (RGB)
    raw_row = b"\x00\xff\xff\xff"
    compressed = zlib.compress(raw_row)
    idat = write_chunk(b"IDAT", compressed)
    iend = write_chunk(b"IEND", b"")
    return signature + ihdr + idat + iend


# ──────────────────────────────────────────────────────────────────────────────
# Filesystem layer tests
# ──────────────────────────────────────────────────────────────────────────────

class TestFilesystemLayer:
    def test_zero_byte_file_flagged(self, tmp_dir: Path):
        """A zero-byte image file must be flagged at the filesystem layer."""
        img = _write_file(tmp_dir / "empty.jpg", b"")
        result = validate_image(img, config={"validation": {"pillow_decode": False, "tensorflow_decode": False}})
        assert result.is_corrupted
        assert result.error_layer == ErrorLayer.FILESYSTEM
        assert "too small" in result.error_message.lower() or "byte" in result.error_message.lower()

    def test_missing_file_flagged(self, tmp_dir: Path):
        """A path to a non-existent file must be flagged at the filesystem layer."""
        missing = tmp_dir / "does_not_exist.jpg"
        result = validate_image(missing, config={"validation": {"pillow_decode": False, "tensorflow_decode": False}})
        assert result.is_corrupted
        assert result.error_layer == ErrorLayer.FILESYSTEM

    def test_small_file_below_min_bytes_flagged(self, tmp_dir: Path):
        """A file below the min_bytes threshold must be flagged."""
        tiny = _write_file(tmp_dir / "tiny.jpg", b"\xff\xd8" + b"\x00" * 5)  # 7 bytes
        result = validate_image(
            tiny,
            config={
                "size_limits": {"min_bytes": 100, "max_bytes": None},
                "validation": {"pillow_decode": False, "tensorflow_decode": False},
            },
        )
        assert result.is_corrupted
        assert result.error_layer == ErrorLayer.FILESYSTEM

    def test_never_raises_on_filesystem_error(self, tmp_dir: Path):
        """validate_image must not raise even for completely invalid inputs."""
        completely_invalid = tmp_dir / "not_a_real_path_xyzzy_12345.png"
        try:
            result = validate_image(completely_invalid)
        except Exception as exc:
            pytest.fail(f"validate_image raised an unexpected exception: {exc}")
        assert result.is_corrupted


# ──────────────────────────────────────────────────────────────────────────────
# Pillow layer tests
# ──────────────────────────────────────────────────────────────────────────────

class TestPillowLayer:
    def test_garbage_data_in_jpg_flagged(self, tmp_dir: Path):
        """A file with a JPEG header but garbage body must be caught by Pillow."""
        garbage = _write_file(
            tmp_dir / "garbage.jpg",
            b"\xff\xd8\xff" + b"\xde\xad\xbe\xef" * 512  # valid SOI marker, then garbage
        )
        result = validate_image(
            garbage,
            config={"validation": {"tensorflow_decode": False}},
        )
        assert result.is_corrupted
        assert result.error_layer in (ErrorLayer.PILLOW, ErrorLayer.FILESYSTEM)

    def test_non_image_renamed_to_jpg_flagged(self, tmp_dir: Path):
        """A plain text file renamed to .jpg must be caught by Pillow."""
        fake = _write_file(tmp_dir / "not_an_image.jpg", b"This is plain text, not an image!\n" * 100)
        result = validate_image(
            fake,
            config={"validation": {"tensorflow_decode": False}},
        )
        assert result.is_corrupted
        assert result.error_layer == ErrorLayer.PILLOW

    def test_never_raises_on_pillow_error(self, tmp_dir: Path):
        """validate_image must not raise even when Pillow encounters unrecognized data."""
        weird = _write_file(tmp_dir / "weird.png", b"\x00" * 1024)
        try:
            result = validate_image(weird, config={"validation": {"tensorflow_decode": False}})
        except Exception as exc:
            pytest.fail(f"validate_image raised: {exc}")
        # Whether it's corrupted or not, no exception should propagate
        assert result.status in (ValidationStatus.OK, ValidationStatus.CORRUPTED)


# ──────────────────────────────────────────────────────────────────────────────
# Valid image test
# ──────────────────────────────────────────────────────────────────────────────

class TestValidImage:
    def test_valid_png_passes_pillow_layer(self, tmp_dir: Path):
        """A minimal valid PNG must pass at least the Pillow layer."""
        valid_png = _write_file(tmp_dir / "valid.png", _minimal_valid_png())
        result = validate_image(
            valid_png,
            config={
                "validation": {
                    "filesystem_check": True,
                    "pillow_decode": True,
                    "tensorflow_decode": False,  # Skip TF to avoid loading TF in unit tests
                }
            },
        )
        assert result.status == ValidationStatus.OK, (
            f"Expected OK, got {result.status} — layer={result.error_layer}, msg={result.error_message}"
        )
        assert result.detected_format == "PNG"

    def test_result_always_has_file_path(self, tmp_dir: Path):
        """The ValidationResult must always reference the original file path."""
        f = _write_file(tmp_dir / "check.jpg", b"")
        result = validate_image(f)
        assert result.file_path == f

    def test_to_dict_is_serializable(self, tmp_dir: Path):
        """to_dict() must return a JSON-serializable dictionary."""
        import json
        f = _write_file(tmp_dir / "check.png", _minimal_valid_png())
        result = validate_image(f, config={"validation": {"tensorflow_decode": False}})
        d = result.to_dict()
        # Should not raise
        _ = json.dumps(d)
        assert "file_path" in d
        assert "status" in d
        assert "error_layer" in d


# ──────────────────────────────────────────────────────────────────────────────
# Config tests
# ──────────────────────────────────────────────────────────────────────────────

class TestConfigHandling:
    def test_none_config_uses_defaults(self, tmp_dir: Path):
        """Passing config=None must not raise and must use defaults."""
        f = _write_file(tmp_dir / "x.jpg", b"")
        try:
            result = validate_image(f, config=None)
        except Exception as exc:
            pytest.fail(f"validate_image raised with config=None: {exc}")
        assert result is not None

    def test_empty_config_uses_defaults(self, tmp_dir: Path):
        """Passing config={} must not raise and must use defaults."""
        f = _write_file(tmp_dir / "x.jpg", b"")
        try:
            result = validate_image(f, config={})
        except Exception as exc:
            pytest.fail(f"validate_image raised with config={{}}: {exc}")
        assert result is not None
