"""
utils.py — Shared utilities for tf-image-guard.

Provides:
- setup_logger()     : Configurable dual (console + file) logger
- safe_execute()     : Fault-tolerant wrapper — catches ALL exceptions, never raises
- format_size()      : Human-readable byte sizes
- get_supported_extensions() : Returns the set of supported image extensions
- load_config()      : Loads and merges YAML config with defaults
"""

from __future__ import annotations

import logging
import os
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Set

import yaml

# ──────────────────────────────────────────────────────────────────────────────
# Default configuration (merged with user-supplied YAML)
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_CONFIG: dict[str, Any] = {
    "supported_extensions": [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"],
    "validation": {
        "filesystem_check": True,
        "pillow_decode": True,
        "tensorflow_decode": True,
    },
    "tensorflow": {
        "cpu_only": True,
        "channels": 0,
    },
    "size_limits": {
        "min_bytes": 100,
        "max_bytes": None,
    },
    "workers": 4,
    "quarantine": {
        "preserve_structure": True,
        "generate_manifest": True,
    },
    "report": {
        "format": "console",
        "output_path": None,
        "include_ok_files": False,
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# SafeResult dataclass — returned by safe_execute()
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class SafeResult:
    """Encapsulates the outcome of a fault-tolerant function call."""
    success: bool
    value: Any = None
    error: Optional[Exception] = None
    error_type: str = ""
    traceback_str: str = ""

    @property
    def failed(self) -> bool:
        return not self.success


# ──────────────────────────────────────────────────────────────────────────────
# Fault-tolerant executor
# ──────────────────────────────────────────────────────────────────────────────
def safe_execute(fn: Callable, *args, logger: Optional[logging.Logger] = None, **kwargs) -> SafeResult:
    """
    Call ``fn(*args, **kwargs)`` and catch every possible exception.

    Never raises. Returns a :class:`SafeResult` with success/failure info.

    Parameters
    ----------
    fn : Callable
        The function to call.
    *args :
        Positional arguments forwarded to ``fn``.
    logger : logging.Logger, optional
        If provided, exceptions are logged at DEBUG level.
    **kwargs :
        Keyword arguments forwarded to ``fn``.

    Returns
    -------
    SafeResult
    """
    try:
        result = fn(*args, **kwargs)
        return SafeResult(success=True, value=result)
    except KeyboardInterrupt:
        # Re-raise keyboard interrupt — user explicitly asked to stop
        raise
    except SystemExit:
        # Re-raise SystemExit — intentional exits should propagate
        raise
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        if logger:
            logger.debug("safe_execute caught exception in %s: %s\n%s", fn.__name__, exc, tb)
        return SafeResult(
            success=False,
            error=exc,
            error_type=type(exc).__name__,
            traceback_str=tb,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Logger setup
# ──────────────────────────────────────────────────────────────────────────────
def setup_logger(
    name: str = "tf-image-guard",
    log_file: Optional[str | Path] = None,
    level: int = logging.INFO,
    verbose: bool = False,
) -> logging.Logger:
    """
    Create and return a logger with a console handler and optionally a file handler.

    Parameters
    ----------
    name : str
        Logger name.
    log_file : str or Path, optional
        If provided, logs are also written to this file.
    level : int
        Base logging level. Overridden to DEBUG when ``verbose=True``.
    verbose : bool
        If True, sets level to DEBUG.

    Returns
    -------
    logging.Logger
    """
    if verbose:
        level = logging.DEBUG

    logger = logging.getLogger(name)

    # Avoid duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    logger.setLevel(level)
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(fmt)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    # Optional file handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(fmt)
        file_handler.setLevel(logging.DEBUG)  # Always capture everything in the file
        logger.addHandler(file_handler)

    return logger


# ──────────────────────────────────────────────────────────────────────────────
# Human-readable size formatter
# ──────────────────────────────────────────────────────────────────────────────
def format_size(num_bytes: Optional[int]) -> str:
    """Convert a byte count to a human-readable string (e.g. '4.2 MB')."""
    if num_bytes is None:
        return "unknown"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:,.1f} {unit}"
        num_bytes /= 1024.0  # type: ignore[assignment]
    return f"{num_bytes:.1f} PB"


# ──────────────────────────────────────────────────────────────────────────────
# Supported extensions
# ──────────────────────────────────────────────────────────────────────────────
def get_supported_extensions(config: Optional[dict] = None) -> Set[str]:
    """Return a set of supported image extensions (lower-cased, with leading dot)."""
    if config and "supported_extensions" in config:
        exts = config["supported_extensions"]
    else:
        exts = DEFAULT_CONFIG["supported_extensions"]
    return {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in exts}


# ──────────────────────────────────────────────────────────────────────────────
# Config loader
# ──────────────────────────────────────────────────────────────────────────────
def load_config(config_path: Optional[str | Path] = None) -> dict[str, Any]:
    """
    Load a YAML config file and merge it with :data:`DEFAULT_CONFIG`.

    If ``config_path`` is None or the file is unreadable, the default config
    is returned without raising an error.

    Parameters
    ----------
    config_path : str or Path, optional
        Path to a YAML configuration file.

    Returns
    -------
    dict
        Merged configuration dictionary.
    """
    config = _deep_merge({}, DEFAULT_CONFIG)

    if config_path is None:
        return config

    try:
        path = Path(config_path)
        if not path.is_file():
            return config
        with path.open("r", encoding="utf-8") as fh:
            user_cfg = yaml.safe_load(fh) or {}
        config = _deep_merge(config, user_cfg)
    except Exception:  # noqa: BLE001
        # Config loading failure is non-fatal; fall back to defaults
        pass

    return config


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into ``base`` and return the result."""
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Misc helpers
# ──────────────────────────────────────────────────────────────────────────────
def get_file_size(path: Path) -> Optional[int]:
    """Return file size in bytes, or None if the file is inaccessible."""
    try:
        return os.path.getsize(path)
    except OSError:
        return None
