"""
cli.py — Click-based command-line interface for tf-image-guard.

Commands:
  tf-image-guard scan   → Scan a directory and report corrupted images
  tf-image-guard clean  → Scan + quarantine corrupted images
  tf-image-guard info   → Show version and configuration info

All commands route through the same fault-tolerant scanner core.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import click

from . import __version__
from .reporter import generate_report
from .scanner import scan_directory
from .utils import load_config, setup_logger


# ──────────────────────────────────────────────────────────────────────────────
# Shared CLI options (applied to multiple commands via decorator)
# ──────────────────────────────────────────────────────────────────────────────

_shared_scan_options = [
    click.option(
        "--input-dir", "-i",
        required=True,
        type=click.Path(file_okay=False, dir_okay=True),
        help="Path to the image directory to scan.",
    ),
    click.option(
        "--config", "-c",
        default=None,
        type=click.Path(exists=True, file_okay=True, dir_okay=False),
        help="Path to a YAML configuration file (optional).",
    ),
    click.option(
        "--log", "-l",
        default=None,
        type=click.Path(file_okay=True, dir_okay=False),
        help="Write log output to this file.",
    ),
    click.option(
        "--recursive/--no-recursive", "-r/-R",
        default=True,
        show_default=True,
        help="Recursively scan subdirectories.",
    ),
    click.option(
        "--workers", "-w",
        default=None,
        type=click.IntRange(1, 64),
        metavar="N",
        help="Number of parallel worker threads (default: from config or 4).",
    ),
    click.option(
        "--report-format",
        default="console",
        type=click.Choice(["console", "json", "csv"], case_sensitive=False),
        show_default=True,
        help="Format for the scan report.",
    ),
    click.option(
        "--report-output",
        default=None,
        type=click.Path(file_okay=True, dir_okay=False),
        help="Write report to this file (defaults to stdout).",
    ),
    click.option(
        "--include-ok",
        is_flag=True,
        default=False,
        help="Include valid (non-corrupted) files in the report.",
    ),
    click.option(
        "--verbose", "-v",
        is_flag=True,
        default=False,
        help="Enable verbose/debug output.",
    ),
]


def _add_options(options):
    """Decorator that adds multiple options to a Click command."""
    def decorator(fn):
        for opt in reversed(options):
            fn = opt(fn)
        return fn
    return decorator


# ──────────────────────────────────────────────────────────────────────────────
# Root group
# ──────────────────────────────────────────────────────────────────────────────

@click.group(
    name="tf-image-guard",
    context_settings={"help_option_names": ["-h", "--help"]},
    invoke_without_command=True,
)
@click.version_option(version=__version__, prog_name="tf-image-guard")
@click.pass_context
def main(ctx: click.Context) -> None:
    """
    \b
    tf-image-guard — Fault-tolerant image validator for TensorFlow datasets.

    Scans directories for corrupted images using a 3-layer pipeline:
      1. Filesystem check  (zero-byte, permissions, size limits)
      2. Pillow decode     (truncated, broken headers)
      3. TensorFlow decode (ensures tf.data pipeline compatibility)

    Execution NEVER stops due to a bad file.
    """
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


# ──────────────────────────────────────────────────────────────────────────────
# scan command
# ──────────────────────────────────────────────────────────────────────────────

@main.command("scan")
@_add_options(_shared_scan_options)
def cmd_scan(
    input_dir: str,
    config: Optional[str],
    log: Optional[str],
    recursive: bool,
    workers: Optional[int],
    report_format: str,
    report_output: Optional[str],
    include_ok: bool,
    verbose: bool,
) -> None:
    """
    Scan a directory and report all corrupted images.

    \b
    Examples:
      tf-image-guard scan -i ./images
      tf-image-guard scan -i ./images --report-format json --report-output report.json
      tf-image-guard scan -i ./images -w 8 --include-ok
    """
    logger = setup_logger(log_file=log, verbose=verbose)
    cfg = load_config(config)

    logger.info("tf-image-guard v%s — scan", __version__)

    scan_result = scan_directory(
        input_dir=input_dir,
        config=cfg,
        recursive=recursive,
        workers=workers,
        logger=logger,
    )

    generate_report(
        scan_result=scan_result,
        fmt=report_format,
        output_path=report_output,
        include_ok=include_ok,
        logger=logger,
    )

    # Exit code: 0 = no corruption, 1 = corruption found, 2 = scan error
    if scan_result.scan_errors:
        sys.exit(2)
    elif scan_result.corrupted_count > 0:
        sys.exit(1)
    else:
        sys.exit(0)


# ──────────────────────────────────────────────────────────────────────────────
# clean command
# ──────────────────────────────────────────────────────────────────────────────

@main.command("clean")
@_add_options(_shared_scan_options)
@click.option(
    "--output-dir", "-o",
    default=None,
    type=click.Path(file_okay=False, dir_okay=True),
    help=(
        "Directory to move corrupted files into. "
        "Defaults to <input-dir>/_quarantine."
    ),
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Show what would be moved without actually moving files.",
)
def cmd_clean(
    input_dir: str,
    config: Optional[str],
    log: Optional[str],
    recursive: bool,
    workers: Optional[int],
    report_format: str,
    report_output: Optional[str],
    include_ok: bool,
    verbose: bool,
    output_dir: Optional[str],
    dry_run: bool,
) -> None:
    """
    Scan a directory, then quarantine corrupted images.

    Corrupted files are moved to the output directory. The original
    folder structure is preserved inside the quarantine directory,
    and a _manifest.json is generated.

    \b
    Examples:
      tf-image-guard clean -i ./images
      tf-image-guard clean -i ./images -o ./quarantine --dry-run
      tf-image-guard clean -i ./images -o ./bad_images --report-format json
    """
    from .quarantine import quarantine_corrupted  # noqa: PLC0415

    logger = setup_logger(log_file=log, verbose=verbose)
    cfg = load_config(config)

    logger.info("tf-image-guard v%s — clean%s", __version__, " (dry-run)" if dry_run else "")

    # Determine quarantine directory
    q_dir = Path(output_dir) if output_dir else Path(input_dir) / "_quarantine"

    # Scan
    scan_result = scan_directory(
        input_dir=input_dir,
        config=cfg,
        recursive=recursive,
        workers=workers,
        logger=logger,
    )

    # Quarantine
    quarantine_result = quarantine_corrupted(
        scan_result=scan_result,
        quarantine_dir=q_dir,
        config=cfg,
        dry_run=dry_run,
        logger=logger,
    )

    # Report
    generate_report(
        scan_result=scan_result,
        fmt=report_format,
        output_path=report_output,
        include_ok=include_ok,
        quarantine_result=quarantine_result,
        logger=logger,
    )

    # Exit code
    if scan_result.scan_errors:
        sys.exit(2)
    elif scan_result.corrupted_count > 0:
        sys.exit(1)
    else:
        sys.exit(0)


# ──────────────────────────────────────────────────────────────────────────────
# info command
# ──────────────────────────────────────────────────────────────────────────────

@main.command("info")
@click.option(
    "--config", "-c",
    default=None,
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help="Path to a YAML config file to display resolved settings.",
)
def cmd_info(config: Optional[str]) -> None:
    """
    Show version information and resolved configuration.

    \b
    Examples:
      tf-image-guard info
      tf-image-guard info -c sample_config.yml
    """
    import json  # noqa: PLC0415
    from rich.console import Console  # noqa: PLC0415
    from rich.panel import Panel  # noqa: PLC0415

    cfg = load_config(config)
    console = Console()

    console.print(
        Panel(
            f"[bold]tf-image-guard[/bold] v{__version__}\n"
            f"[dim]A fault-tolerant image validator for TensorFlow datasets.[/dim]",
            border_style="cyan",
            expand=False,
        )
    )

    console.print("\n[bold]Resolved Configuration:[/bold]")
    console.print(json.dumps(cfg, indent=2, default=str))


# ──────────────────────────────────────────────────────────────────────────────
# Entry point for direct script execution
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
