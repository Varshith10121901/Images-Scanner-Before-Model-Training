"""
quarantine.py — Move/copy corrupted images to a quarantine directory.

Features:
- Preserves original directory structure inside the quarantine folder
- Generates a ``_manifest.json`` documenting every quarantined file + reason
- Supports ``dry_run`` mode (logs what would happen without touching files)
- Never raises — all file operations are wrapped in safe_execute()
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from .scanner import ScanResult
from .utils import format_size, safe_execute, setup_logger
from .validators import ValidationResult


# ──────────────────────────────────────────────────────────────────────────────
# QuarantineResult dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class QuarantineEntry:
    original_path: str
    quarantine_path: str
    error_layer: str
    error_message: str
    file_size_bytes: Optional[int]
    moved: bool
    move_error: str = ""


@dataclass
class QuarantineResult:
    quarantine_dir: Path
    dry_run: bool
    total_attempted: int = 0
    moved_count: int = 0
    failed_count: int = 0
    entries: List[QuarantineEntry] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# Public quarantine function
# ──────────────────────────────────────────────────────────────────────────────

def quarantine_corrupted(
    scan_result: ScanResult,
    quarantine_dir: str | Path,
    config: Optional[dict] = None,
    dry_run: bool = False,
    logger: Optional[logging.Logger] = None,
) -> QuarantineResult:
    """
    Move all corrupted files from ``scan_result`` into ``quarantine_dir``.

    This function NEVER raises. File move/copy errors are recorded in the
    returned :class:`QuarantineResult`.

    Parameters
    ----------
    scan_result : ScanResult
        The result of a previous :func:`scanner.scan_directory` call.
    quarantine_dir : str or Path
        Destination directory for corrupted files.
    config : dict, optional
        Merged configuration dictionary.
    dry_run : bool
        If True, log actions without moving files.
    logger : logging.Logger, optional
        Logger for progress and error messages.

    Returns
    -------
    QuarantineResult
    """
    if config is None:
        config = {}
    if logger is None:
        logger = setup_logger()

    q_cfg = config.get("quarantine", {})
    preserve_structure: bool = q_cfg.get("preserve_structure", True)
    generate_manifest: bool = q_cfg.get("generate_manifest", True)

    q_dir = Path(quarantine_dir)
    result = QuarantineResult(quarantine_dir=q_dir, dry_run=dry_run)

    corrupted = scan_result.corrupted_files
    if not corrupted:
        logger.info("[QUARANTINE] No corrupted files to quarantine.")
        return result

    # ── Create quarantine directory ──────────────────────────────────────────
    if not dry_run:
        mkdir_result = safe_execute(q_dir.mkdir, parents=True, exist_ok=True)
        if mkdir_result.failed:
            logger.error(
                "[QUARANTINE] Cannot create quarantine directory %s: %s",
                q_dir,
                mkdir_result.error,
            )
            return result

    logger.info(
        "[QUARANTINE] %s%d corrupted file(s) → %s",
        "[DRY-RUN] Would move " if dry_run else "Moving ",
        len(corrupted),
        q_dir,
    )

    for vr in corrupted:
        entry = _quarantine_one(
            vr=vr,
            q_dir=q_dir,
            source_root=scan_result.input_dir,
            preserve_structure=preserve_structure,
            dry_run=dry_run,
            logger=logger,
        )
        result.entries.append(entry)
        result.total_attempted += 1

        if entry.moved or dry_run:
            result.moved_count += 1
        else:
            result.failed_count += 1

    # ── Write manifest ───────────────────────────────────────────────────────
    if generate_manifest and not dry_run:
        _write_manifest(result, logger)

    logger.info(
        "[QUARANTINE] Done — %d moved, %d failed.",
        result.moved_count,
        result.failed_count,
    )
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _quarantine_one(
    vr: ValidationResult,
    q_dir: Path,
    source_root: Path,
    preserve_structure: bool,
    dry_run: bool,
    logger: logging.Logger,
) -> QuarantineEntry:
    """Move (or dry-run) one corrupted file into the quarantine directory."""

    src = vr.file_path

    # Determine destination path
    if preserve_structure:
        try:
            rel = src.relative_to(source_root)
            dest = q_dir / rel
        except ValueError:
            # file is not under source_root — just use filename
            dest = q_dir / src.name
    else:
        dest = q_dir / src.name

    entry = QuarantineEntry(
        original_path=str(src),
        quarantine_path=str(dest),
        error_layer=vr.error_layer.value,
        error_message=vr.error_message,
        file_size_bytes=vr.file_size,
        moved=False,
    )

    if dry_run:
        logger.info("[DRY-RUN] Would move: %s → %s", src, dest)
        entry.moved = True  # Mark as "would move" for counting
        return entry

    # Ensure parent directory exists
    mkdir_r = safe_execute(dest.parent.mkdir, parents=True, exist_ok=True)
    if mkdir_r.failed:
        msg = f"Cannot create parent dir {dest.parent}: {mkdir_r.error}"
        logger.warning("[QUARANTINE] %s", msg)
        entry.move_error = msg
        return entry

    # Move the file
    move_r = safe_execute(shutil.move, str(src), str(dest))
    if move_r.failed:
        msg = f"Move failed: {move_r.error}"
        logger.warning("[QUARANTINE] Cannot move %s: %s", src.name, msg)
        entry.move_error = msg
    else:
        logger.debug("[QUARANTINE] Moved: %s → %s", src, dest)
        entry.moved = True

    return entry


def _write_manifest(result: QuarantineResult, logger: logging.Logger) -> None:
    """Write a JSON manifest of all quarantined files."""
    manifest_path = result.quarantine_dir / "_manifest.json"
    manifest_data = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "quarantine_dir": str(result.quarantine_dir),
        "dry_run": result.dry_run,
        "total_moved": result.moved_count,
        "total_failed": result.failed_count,
        "entries": [
            {
                "original_path": e.original_path,
                "quarantine_path": e.quarantine_path,
                "error_layer": e.error_layer,
                "error_message": e.error_message,
                "file_size": format_size(e.file_size_bytes),
                "moved": e.moved,
                "move_error": e.move_error,
            }
            for e in result.entries
        ],
    }

    write_r = safe_execute(
        _write_json_file,
        manifest_path,
        manifest_data,
    )
    if write_r.failed:
        logger.warning("[QUARANTINE] Could not write manifest: %s", write_r.error)
    else:
        logger.info("[QUARANTINE] Manifest written: %s", manifest_path)


def _write_json_file(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False, default=str)
