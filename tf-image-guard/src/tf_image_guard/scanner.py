"""
scanner.py — Core scanning engine for tf-image-guard.

Orchestrates directory traversal + parallel image validation:

- Recursively (or non-recursively) walks a directory with ``os.walk``
- Filters files by supported extensions
- Dispatches each file to :func:`validators.validate_image` via a
  ``ThreadPoolExecutor`` for parallel processing
- Wraps EVERY operation in fault-tolerant exception handling so that
  a single bad file, a permission error on a subdirectory, or any
  unexpected OS error never stops the scan
- Provides a live ``tqdm`` progress bar
- Returns a :class:`ScanResult` dataclass with full statistics and per-file results
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, List, Optional

from tqdm import tqdm

from .utils import get_supported_extensions, load_config, safe_execute, setup_logger
from .validators import ValidationResult, ValidationStatus, validate_image


# ──────────────────────────────────────────────────────────────────────────────
# ScanResult dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ScanResult:
    """Aggregated results from a directory scan."""
    input_dir: Path
    total_files_scanned: int = 0
    ok_count: int = 0
    corrupted_count: int = 0
    skipped_count: int = 0
    results: List[ValidationResult] = field(default_factory=list)
    scan_errors: List[str] = field(default_factory=list)  # Non-fatal scan-level errors

    @property
    def corrupted_files(self) -> List[ValidationResult]:
        return [r for r in self.results if r.is_corrupted]

    @property
    def ok_files(self) -> List[ValidationResult]:
        return [r for r in self.results if r.is_ok]

    @property
    def corruption_rate(self) -> float:
        if self.total_files_scanned == 0:
            return 0.0
        return (self.corrupted_count / self.total_files_scanned) * 100

    def summary(self) -> dict:
        return {
            "input_dir": str(self.input_dir),
            "total_scanned": self.total_files_scanned,
            "ok": self.ok_count,
            "corrupted": self.corrupted_count,
            "skipped": self.skipped_count,
            "corruption_rate_pct": round(self.corruption_rate, 2),
            "scan_errors": self.scan_errors,
        }


# ──────────────────────────────────────────────────────────────────────────────
# File discovery
# ──────────────────────────────────────────────────────────────────────────────

def _iter_image_files(
    root_dir: Path,
    supported_extensions: set,
    recursive: bool = True,
    logger: Optional[logging.Logger] = None,
) -> Iterator[Path]:
    """
    Yield all image file paths under ``root_dir``.

    Uses ``os.walk`` which is resilient to most OS errors when used with
    ``onerror`` parameter. Each per-directory error is logged and skipped.
    """
    def _on_error(exc: OSError) -> None:
        msg = f"Cannot access directory: {exc}"
        if logger:
            logger.warning("[SCAN] %s", msg)

    walk = os.walk(root_dir, onerror=_on_error) if recursive else [(root_dir, [], os.listdir(root_dir))]

    for dirpath, _dirnames, filenames in walk:
        for fname in filenames:
            try:
                fpath = Path(dirpath) / fname
                if fpath.suffix.lower() in supported_extensions:
                    yield fpath
            except Exception as exc:  # noqa: BLE001
                if logger:
                    logger.warning("[SCAN] Skipping %s: %s", fname, exc)


# ──────────────────────────────────────────────────────────────────────────────
# Public scanner
# ──────────────────────────────────────────────────────────────────────────────

def scan_directory(
    input_dir: str | Path,
    config: Optional[dict] = None,
    recursive: bool = True,
    workers: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> ScanResult:
    """
    Scan a directory for corrupted images using parallel validation.

    This function NEVER raises. All errors — including invalid paths,
    permission errors, and per-file validation failures — are captured
    in the returned :class:`ScanResult`.

    Parameters
    ----------
    input_dir : str or Path
        Directory to scan.
    config : dict, optional
        Merged configuration dictionary from :func:`utils.load_config`.
    recursive : bool
        If True (default), scan subdirectories.
    workers : int, optional
        Number of parallel threads. Defaults to ``config['workers']`` or 4.
    logger : logging.Logger, optional
        Logger for progress and error messages.
    progress_callback : callable, optional
        Called with (files_processed, total_files) for external progress tracking.

    Returns
    -------
    ScanResult
    """
    if config is None:
        config = {}

    if logger is None:
        logger = setup_logger()

    root = Path(input_dir)
    result = ScanResult(input_dir=root)

    # ── Validate input directory ─────────────────────────────────────────────
    if not root.exists():
        result.scan_errors.append(f"Input directory does not exist: {root}")
        logger.error("[SCAN] Input directory does not exist: %s", root)
        return result

    if not root.is_dir():
        result.scan_errors.append(f"Input path is not a directory: {root}")
        logger.error("[SCAN] Input path is not a directory: %s", root)
        return result

    # ── Discover files ───────────────────────────────────────────────────────
    supported_ext = get_supported_extensions(config)
    num_threads = workers or config.get("workers", 4) or 4

    logger.info("[SCAN] Discovering image files in: %s (recursive=%s)", root, recursive)

    # Collect files first so we can show an accurate progress bar
    try:
        all_files = list(_iter_image_files(root, supported_ext, recursive=recursive, logger=logger))
    except Exception as exc:  # noqa: BLE001
        msg = f"Unexpected error during file discovery: {exc}"
        result.scan_errors.append(msg)
        logger.error("[SCAN] %s", msg)
        all_files = []

    total = len(all_files)
    logger.info("[SCAN] Found %d image file(s) to validate with %d worker(s).", total, num_threads)

    if total == 0:
        logger.info("[SCAN] No image files found. Scan complete.")
        return result

    # ── Parallel validation ──────────────────────────────────────────────────
    processed = 0

    with tqdm(
        total=total,
        desc="Scanning",
        unit="img",
        dynamic_ncols=True,
        colour="cyan",
    ) as pbar:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            future_to_path: dict[Future, Path] = {}

            for fpath in all_files:
                future = executor.submit(
                    _safe_validate_one,
                    fpath,
                    config,
                    logger,
                )
                future_to_path[future] = fpath

            for future in as_completed(future_to_path):
                file_path = future_to_path[future]

                try:
                    vr: ValidationResult = future.result()
                except Exception as exc:  # noqa: BLE001
                    # This should never happen because _safe_validate_one catches
                    # everything, but we guard here as an extra safety net.
                    msg = f"Unexpected future exception for {file_path}: {exc}"
                    result.scan_errors.append(msg)
                    logger.error("[SCAN] %s", msg)
                    vr = ValidationResult(
                        file_path=file_path,
                        status=ValidationStatus.CORRUPTED,
                        error_message=str(exc),
                    )

                result.results.append(vr)
                result.total_files_scanned += 1

                if vr.is_ok:
                    result.ok_count += 1
                elif vr.is_corrupted:
                    result.corrupted_count += 1
                    logger.info(
                        "[CORRUPTED] %-60s layer=%-12s msg=%s",
                        str(file_path)[-60:],
                        vr.error_layer.value,
                        vr.error_message[:120],
                    )
                else:
                    result.skipped_count += 1

                processed += 1
                pbar.update(1)
                pbar.set_postfix(
                    ok=result.ok_count,
                    bad=result.corrupted_count,
                    refresh=False,
                )

                if progress_callback:
                    safe_execute(progress_callback, processed, total)

    logger.info(
        "[SCAN] Complete — %d scanned, %d OK, %d corrupted, %d skipped (%.1f%% corruption rate).",
        result.total_files_scanned,
        result.ok_count,
        result.corrupted_count,
        result.skipped_count,
        result.corruption_rate,
    )
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Worker wrapper (ensures thread-level fault tolerance)
# ──────────────────────────────────────────────────────────────────────────────

def _safe_validate_one(
    path: Path,
    config: dict,
    logger: logging.Logger,
) -> ValidationResult:
    """
    Validate a single image file with full exception isolation.

    Wraps :func:`validators.validate_image` in a try/except so that
    even an unexpected error inside the validator doesn't propagate
    to the thread pool and never stops the scan.
    """
    try:
        return validate_image(path, config=config, logger=logger)
    except Exception as exc:  # noqa: BLE001
        # Last-resort safety net
        logger.error("[SCAN] Unhandled error validating %s: %s", path, exc)
        from .validators import ErrorLayer, ValidationResult, ValidationStatus  # noqa: PLC0415
        return ValidationResult(
            file_path=path,
            status=ValidationStatus.CORRUPTED,
            error_layer=ErrorLayer.FILESYSTEM,
            error_message=f"Unhandled validation error: {exc}",
        )
