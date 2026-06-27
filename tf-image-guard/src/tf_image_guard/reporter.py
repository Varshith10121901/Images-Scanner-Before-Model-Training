"""
reporter.py — Scan report generation for tf-image-guard.

Supports three output formats:
- console  : Rich-formatted, color-coded terminal output
- json     : Machine-readable full report written to stdout or file
- csv      : Spreadsheet-friendly tabular output

All formatting/IO errors are caught; if a reporter fails, a plain fallback
is printed without stopping execution.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.text import Text

from .quarantine import QuarantineResult
from .scanner import ScanResult
from .utils import format_size, safe_execute, setup_logger

# ──────────────────────────────────────────────────────────────────────────────
# Main dispatcher
# ──────────────────────────────────────────────────────────────────────────────

def generate_report(
    scan_result: ScanResult,
    fmt: str = "console",
    output_path: Optional[str | Path] = None,
    include_ok: bool = False,
    quarantine_result: Optional[QuarantineResult] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Generate and output a scan report.

    Parameters
    ----------
    scan_result : ScanResult
        The result of a scan.
    fmt : str
        Output format: ``console``, ``json``, or ``csv``.
    output_path : str or Path, optional
        If provided, write the report to this file. Otherwise, write to stdout.
    include_ok : bool
        Whether to include OK (non-corrupted) files in the report.
    quarantine_result : QuarantineResult, optional
        If provided, include quarantine summary in the report.
    logger : logging.Logger, optional
        Logger for error messages.
    """
    if logger is None:
        logger = setup_logger()

    fmt = fmt.lower().strip()

    dispatch = {
        "console": _report_console,
        "json":    _report_json,
        "csv":     _report_csv,
    }

    reporter_fn = dispatch.get(fmt)
    if reporter_fn is None:
        logger.warning("[REPORT] Unknown format '%s'. Falling back to console.", fmt)
        reporter_fn = _report_console

    result = safe_execute(
        reporter_fn,
        scan_result,
        output_path,
        include_ok,
        quarantine_result,
    )

    if result.failed:
        # Fallback: plain text summary so the user always gets something
        logger.error("[REPORT] Report generation failed: %s", result.error)
        _fallback_print(scan_result)


# ──────────────────────────────────────────────────────────────────────────────
# Console reporter (Rich)
# ──────────────────────────────────────────────────────────────────────────────

def _report_console(
    scan_result: ScanResult,
    output_path: Optional[Path],
    include_ok: bool,
    quarantine_result: Optional[QuarantineResult],
) -> None:
    console = Console(file=sys.stdout, highlight=False)

    # ── Summary panel ─────────────────────────────────────────────────────
    rate = scan_result.corruption_rate
    rate_colour = "green" if rate == 0 else ("yellow" if rate < 10 else "red")

    summary_lines = [
        f"[bold]Input Directory:[/bold] {scan_result.input_dir}",
        f"[bold]Total Scanned:[/bold]   {scan_result.total_files_scanned:,}",
        f"[bold]✓ OK:[/bold]            [green]{scan_result.ok_count:,}[/green]",
        f"[bold]✗ Corrupted:[/bold]     [red]{scan_result.corrupted_count:,}[/red]",
        f"[bold]⊘ Skipped:[/bold]       [dim]{scan_result.skipped_count:,}[/dim]",
        f"[bold]Corruption Rate:[/bold] [{rate_colour}]{rate:.1f}%[/{rate_colour}]",
    ]

    if quarantine_result:
        summary_lines.append(
            f"[bold]Quarantined:[/bold]     [yellow]{quarantine_result.moved_count:,}[/yellow]"
            + (f" [dim](dry-run)[/dim]" if quarantine_result.dry_run else "")
        )

    console.print(
        Panel(
            "\n".join(summary_lines),
            title="[bold cyan]tf-image-guard — Scan Summary[/bold cyan]",
            border_style="cyan",
            expand=False,
        )
    )

    if scan_result.scan_errors:
        console.print(f"\n[bold red]⚠ Scan-Level Errors ({len(scan_result.scan_errors)}):[/bold red]")
        for err in scan_result.scan_errors:
            console.print(f"  [red]•[/red] {err}")

    # ── Corrupted files table ──────────────────────────────────────────────
    corrupted = scan_result.corrupted_files
    ok_files = scan_result.ok_files

    rows_to_show = corrupted + (ok_files if include_ok else [])

    if rows_to_show:
        table = Table(
            title=f"File Validation Results ({'all files' if include_ok else 'corrupted only'})",
            box=box.ROUNDED,
            show_lines=True,
            header_style="bold magenta",
        )
        table.add_column("Status", style="bold", width=10, justify="center")
        table.add_column("File Path", style="dim", overflow="fold")
        table.add_column("Layer", width=12)
        table.add_column("Size", width=10, justify="right")
        table.add_column("Error Message", overflow="fold")

        for vr in rows_to_show:
            if vr.is_ok:
                status_cell = "[green]✓ OK[/green]"
                layer_cell = ""
                error_cell = ""
            else:
                status_cell = "[red]✗ BAD[/red]"
                layer_cell = f"[yellow]{vr.error_layer.value}[/yellow]"
                error_cell = vr.error_message

            table.add_row(
                status_cell,
                str(vr.file_path),
                layer_cell,
                format_size(vr.file_size),
                error_cell,
            )

        console.print(table)
    else:
        if scan_result.corrupted_count == 0:
            console.print("\n[bold green]🎉 All images passed validation![/bold green]")


# ──────────────────────────────────────────────────────────────────────────────
# JSON reporter
# ──────────────────────────────────────────────────────────────────────────────

def _report_json(
    scan_result: ScanResult,
    output_path: Optional[Path],
    include_ok: bool,
    quarantine_result: Optional[QuarantineResult],
) -> None:
    rows = scan_result.corrupted_files
    if include_ok:
        rows = scan_result.results

    report = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "summary": scan_result.summary(),
        "files": [vr.to_dict() for vr in rows],
    }

    if quarantine_result:
        report["quarantine"] = {
            "quarantine_dir": str(quarantine_result.quarantine_dir),
            "dry_run": quarantine_result.dry_run,
            "moved": quarantine_result.moved_count,
            "failed": quarantine_result.failed_count,
        }

    json_str = json.dumps(report, indent=2, ensure_ascii=False, default=str)

    if output_path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json_str, encoding="utf-8")
    else:
        print(json_str)


# ──────────────────────────────────────────────────────────────────────────────
# CSV reporter
# ──────────────────────────────────────────────────────────────────────────────

def _report_csv(
    scan_result: ScanResult,
    output_path: Optional[Path],
    include_ok: bool,
    quarantine_result: Optional[QuarantineResult],
) -> None:
    rows = scan_result.corrupted_files
    if include_ok:
        rows = scan_result.results

    fieldnames = [
        "file_path",
        "status",
        "error_layer",
        "error_message",
        "file_size_bytes",
        "file_size_human",
        "detected_format",
    ]

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for vr in rows:
        writer.writerow(vr.to_dict())

    csv_str = buf.getvalue()

    if output_path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(csv_str, encoding="utf-8")
    else:
        print(csv_str, end="")


# ──────────────────────────────────────────────────────────────────────────────
# Fallback (plain text — used if Rich or other deps fail)
# ──────────────────────────────────────────────────────────────────────────────

def _fallback_print(scan_result: ScanResult) -> None:
    print(
        f"\n=== tf-image-guard Scan Report ===\n"
        f"Input:     {scan_result.input_dir}\n"
        f"Scanned:   {scan_result.total_files_scanned}\n"
        f"OK:        {scan_result.ok_count}\n"
        f"Corrupted: {scan_result.corrupted_count}\n"
        f"Skipped:   {scan_result.skipped_count}\n"
        f"Rate:      {scan_result.corruption_rate:.1f}%\n"
    )
    for vr in scan_result.corrupted_files:
        print(f"  [CORRUPTED] {vr.file_path} ({vr.error_layer.value}): {vr.error_message}")
