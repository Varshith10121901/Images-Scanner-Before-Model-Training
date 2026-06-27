"""
test_scanner.py — Unit tests for tf_image_guard.scanner

Tests cover:
- Scanning a directory with no images returns empty results without error
- Scanning a non-existent directory returns a ScanResult with scan_errors
- Mixed directory (valid + corrupted) correctly identifies corrupted files
- scan_directory() NEVER raises, regardless of how broken the input is
- ScanResult statistics are computed correctly
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import pytest

from tf_image_guard.scanner import ScanResult, scan_directory
from tf_image_guard.validators import ValidationStatus


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _minimal_valid_png() -> bytes:
    def write_chunk(t: bytes, d: bytes) -> bytes:
        c = t + d
        return struct.pack(">I", len(d)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = write_chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    idat = write_chunk(b"IDAT", zlib.compress(b"\x00\xff\xff\xff"))
    iend = write_chunk(b"IEND", b"")
    return sig + ihdr + idat + iend


NO_TF_CONFIG = {
    "validation": {
        "filesystem_check": True,
        "pillow_decode": True,
        "tensorflow_decode": False,  # Skip TF to keep tests fast
    },
    "size_limits": {"min_bytes": 1, "max_bytes": None},
    "workers": 2,
}


# ──────────────────────────────────────────────────────────────────────────────
# Non-existent / invalid directories
# ──────────────────────────────────────────────────────────────────────────────

class TestInvalidInputDirectory:
    def test_nonexistent_dir_returns_scan_errors(self):
        """scan_directory on a non-existent path must return scan_errors without raising."""
        result = scan_directory(
            "/this/path/does/not/exist/ever",
            config=NO_TF_CONFIG,
        )
        assert isinstance(result, ScanResult)
        assert len(result.scan_errors) > 0
        assert result.total_files_scanned == 0

    def test_nonexistent_dir_never_raises(self):
        """scan_directory must NEVER raise, even on completely invalid paths."""
        try:
            result = scan_directory("Z:/nonexistent/path/xyzzy", config=NO_TF_CONFIG)
        except Exception as exc:
            pytest.fail(f"scan_directory raised: {exc}")
        assert result is not None

    def test_file_as_input_dir_returns_scan_errors(self, tmp_path: Path):
        """Passing a file path instead of a directory must be handled gracefully."""
        f = tmp_path / "myfile.png"
        f.write_bytes(b"data")
        result = scan_directory(str(f), config=NO_TF_CONFIG)
        assert isinstance(result, ScanResult)
        assert len(result.scan_errors) > 0


# ──────────────────────────────────────────────────────────────────────────────
# Empty directory
# ──────────────────────────────────────────────────────────────────────────────

class TestEmptyDirectory:
    def test_empty_dir_returns_zero_count(self, tmp_path: Path):
        """An empty directory must return a ScanResult with 0 files scanned."""
        result = scan_directory(str(tmp_path), config=NO_TF_CONFIG)
        assert result.total_files_scanned == 0
        assert result.ok_count == 0
        assert result.corrupted_count == 0
        assert not result.scan_errors

    def test_dir_with_only_non_image_files_returns_zero(self, tmp_path: Path):
        """A directory with only .txt files must have 0 image files scanned."""
        (tmp_path / "readme.txt").write_text("hello")
        (tmp_path / "data.csv").write_text("a,b,c")
        result = scan_directory(str(tmp_path), config=NO_TF_CONFIG)
        assert result.total_files_scanned == 0


# ──────────────────────────────────────────────────────────────────────────────
# Mixed directory
# ──────────────────────────────────────────────────────────────────────────────

class TestMixedDirectory:
    def _setup_mixed(self, tmp_path: Path) -> Path:
        """Create a directory with 2 valid PNGs and 2 corrupted files."""
        # Valid
        (tmp_path / "valid1.png").write_bytes(_minimal_valid_png())
        (tmp_path / "valid2.png").write_bytes(_minimal_valid_png())
        # Corrupted: zero-byte
        (tmp_path / "zero.jpg").write_bytes(b"")
        # Corrupted: non-image renamed to .png
        (tmp_path / "fake.png").write_bytes(b"This is not a PNG! " * 50)
        return tmp_path

    def test_correct_total_count(self, tmp_path: Path):
        d = self._setup_mixed(tmp_path)
        result = scan_directory(str(d), config=NO_TF_CONFIG)
        assert result.total_files_scanned == 4

    def test_ok_count_correct(self, tmp_path: Path):
        d = self._setup_mixed(tmp_path)
        result = scan_directory(str(d), config=NO_TF_CONFIG)
        assert result.ok_count == 2

    def test_corrupted_count_correct(self, tmp_path: Path):
        d = self._setup_mixed(tmp_path)
        result = scan_directory(str(d), config=NO_TF_CONFIG)
        assert result.corrupted_count == 2

    def test_corrupted_files_property(self, tmp_path: Path):
        d = self._setup_mixed(tmp_path)
        result = scan_directory(str(d), config=NO_TF_CONFIG)
        bad = result.corrupted_files
        assert len(bad) == 2
        for vr in bad:
            assert vr.status == ValidationStatus.CORRUPTED

    def test_corruption_rate_calculation(self, tmp_path: Path):
        d = self._setup_mixed(tmp_path)
        result = scan_directory(str(d), config=NO_TF_CONFIG)
        assert result.corruption_rate == pytest.approx(50.0, abs=0.1)

    def test_scan_never_raises_in_mixed_dir(self, tmp_path: Path):
        d = self._setup_mixed(tmp_path)
        try:
            result = scan_directory(str(d), config=NO_TF_CONFIG)
        except Exception as exc:
            pytest.fail(f"scan_directory raised: {exc}")
        assert result is not None


# ──────────────────────────────────────────────────────────────────────────────
# Recursive scanning
# ──────────────────────────────────────────────────────────────────────────────

class TestRecursiveScanning:
    def test_recursive_finds_nested_files(self, tmp_path: Path):
        nested = tmp_path / "subdir" / "deep"
        nested.mkdir(parents=True)
        (nested / "img.png").write_bytes(_minimal_valid_png())
        (tmp_path / "root.png").write_bytes(_minimal_valid_png())

        result = scan_directory(str(tmp_path), config=NO_TF_CONFIG, recursive=True)
        assert result.total_files_scanned == 2

    def test_non_recursive_ignores_subdirs(self, tmp_path: Path):
        nested = tmp_path / "subdir"
        nested.mkdir()
        (nested / "img.png").write_bytes(_minimal_valid_png())
        (tmp_path / "root.png").write_bytes(_minimal_valid_png())

        result = scan_directory(str(tmp_path), config=NO_TF_CONFIG, recursive=False)
        assert result.total_files_scanned == 1


# ──────────────────────────────────────────────────────────────────────────────
# ScanResult helpers
# ──────────────────────────────────────────────────────────────────────────────

class TestScanResultHelpers:
    def test_summary_dict_has_required_keys(self, tmp_path: Path):
        result = scan_directory(str(tmp_path), config=NO_TF_CONFIG)
        s = result.summary()
        for key in ("input_dir", "total_scanned", "ok", "corrupted", "corruption_rate_pct"):
            assert key in s, f"Missing key '{key}' in summary"

    def test_zero_files_corruption_rate_is_zero(self, tmp_path: Path):
        result = scan_directory(str(tmp_path), config=NO_TF_CONFIG)
        assert result.corruption_rate == 0.0
