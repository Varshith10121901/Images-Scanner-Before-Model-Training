# -*- coding: utf-8 -*-
"""
third_eye.py — Third Eye AI Terminal Chatbot
Fast image-dataset scanner: checks file type + Pillow header (no TensorFlow).

Run:   python third_eye.py
"""

# ─────────────────────────────────────────────────────────────────────────────
#  WINDOWS UTF-8 FIX  (must be first)
# ─────────────────────────────────────────────────────────────────────────────
import sys, io, os

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    os.system("chcp 65001 >nul 2>&1")

# ─────────────────────────────────────────────────────────────────────────────
#  STANDARD IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import threading
import time
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
#  DEPENDENCY CHECK
# ─────────────────────────────────────────────────────────────────────────────
_missing = []
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import (
        Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
    )
    from rich.text import Text
    from rich.rule import Rule
    from rich.align import Align
    from rich.padding import Padding
    from rich import box
except ImportError:
    _missing.append("rich")

try:
    from PIL import Image, ImageFile, UnidentifiedImageError, ImageTk
    ImageFile.LOAD_TRUNCATED_IMAGES = False
except ImportError:
    _missing.append("Pillow")

if _missing:
    print(f"\n[ERROR] Missing packages: {', '.join(_missing)}")
    print(f"  Install:  pip install {' '.join(_missing)}")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
#  GLOBALS
# ─────────────────────────────────────────────────────────────────────────────
console = Console(force_terminal=True, highlight=False, emoji=True)

VERSION  = "v1.0.0"
AI_NAME  = "Third Eye"

# Accepted image extensions (file-type check — fast, no TF needed)
SUPPORTED_EXT = {
    ".jpg", ".jpeg", ".png", ".bmp", ".gif",
    ".webp", ".tiff", ".tif", ".ico", ".ppm", ".pgm",
}

# ─────────────────────────────────────────────────────────────────────────────
#  LOGO
# ─────────────────────────────────────────────────────────────────────────────

LOGO_ASCII = r"""
  _____ _   _ ___ ____  ____      _______   _______
 |_   _| | | |_ _|  _ \|  _ \   | ____\ \ / / ____|
   | | | |_| || || |_) | | | |  |  _|  \ V /|  _|
   | | |  _  || ||  _ <| |_| |  | |___  | | | |___
   |_| |_| |_|___|_| \_\____/   |_____| |_| |_____|
"""

TAGLINE = "[ AI-Powered Image Dataset Scanner  |  Never stops. Always scans. ]"

# ─────────────────────────────────────────────────────────────────────────────
#  UI HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def print_logo():
    terminal_width = shutil.get_terminal_size((100, 24)).columns
    console.print()
    for line in LOGO_ASCII.strip("\n").split("\n"):
        console.print(Align.center(Text(line, style="bold white")))
    console.print()
    name_banner = Text()
    name_banner.append("  ◉  ", style="bold cyan")
    name_banner.append("THIRD EYE", style="bold white")
    name_banner.append("  AI  ", style="bold cyan")
    console.print(Align.center(name_banner))
    console.print()
    console.print(Align.center(Text(TAGLINE, style="dim white")))
    console.print()
    console.print(Rule(style="dim white"))
    console.print()


def print_welcome():
    lines = [
        f"[bold white]Welcome to [cyan]Third Eye[/cyan] {VERSION}[/bold white]",
        "",
        "[dim]I scan folders and check file types and signatures.[/dim]",
        "[dim]Validation: File-type check  →  Magic bytes signature verify[/dim]",
        "[dim](Ultra-fast mode — no image decoding or TensorFlow required)[/dim]",
        "",
        "  [bold cyan]•[/bold cyan]  Paste a [bold white]folder path[/bold white] and press Enter to scan",
        "  [bold cyan]•[/bold cyan]  Type [bold white]help[/bold white] for commands",
        "  [bold cyan]•[/bold cyan]  Press [bold white]Ctrl+C[/bold white] during a scan to stop it early",
        "  [bold cyan]•[/bold cyan]  Type [bold white]exit[/bold white] to quit",
    ]
    console.print(
        Panel(
            "\n".join(lines),
            title="[bold cyan]◉ Third Eye[/bold cyan]",
            border_style="cyan",
            padding=(1, 3),
        )
    )
    console.print()


def print_help():
    console.print(
        Panel(
            "\n".join([
                "[bold white]Commands[/bold white]\n",
                "  [cyan]<folder_path>[/cyan]  →  Scan all images in that directory",
                "  [cyan]help[/cyan]           →  Show this help",
                "  [cyan]clear[/cyan]          →  Clear screen & show logo",
                "  [cyan]exit / quit[/cyan]    →  Exit Third Eye",
                "",
                "  [bold white]Ctrl+C[/bold white]        →  Stop a running scan and show partial results",
                "",
                "[dim]Examples:[/dim]",
                '  [dim]D:\\datasets\\training_images[/dim]',
                '  [dim]C:\\Users\\you\\Pictures[/dim]',
                '  [dim]/home/user/images[/dim]',
            ]),
            title="[bold cyan]◉ Help[/bold cyan]",
            border_style="dim cyan",
            padding=(1, 3),
        )
    )
    console.print()


def ai_say(message: str):
    console.print(f"[bold cyan]◉ Third Eye:[/bold cyan] {message}")


def ai_think(message: str):
    console.print(f"  [dim cyan]→  {message}[/dim cyan]")


def ai_warn(message: str):
    console.print(f"[bold yellow]⚠  {message}[/bold yellow]")


def ai_error(message: str):
    console.print(f"[bold red]✗  {message}[/bold red]")


def get_input() -> str:
    console.print()
    try:
        return console.input("[bold white]You ›[/bold white] ").strip()
    except (EOFError, KeyboardInterrupt):
        console.print()
        return "__ctrl_c__"
    except Exception:
        return ""


# ─────────────────────────────────────────────────────────────────────────────
#  IMAGE VALIDATION  (fast: extension check + Pillow header only)
# ─────────────────────────────────────────────────────────────────────────────

class ScanEntry:
    __slots__ = ("path", "status", "layer", "reason", "size_bytes", "fmt")

    def __init__(self, path: Path):
        self.path       = path
        self.status     = "ok"        # "ok" | "corrupted"
        self.layer      = ""          # which check caught it
        self.reason     = ""
        self.size_bytes: Optional[int] = None
        self.fmt        = ""

    @property
    def filename(self) -> str:
        return self.path.name

    @property
    def is_ok(self) -> bool:
        return self.status == "ok"

    @property
    def is_corrupted(self) -> bool:
        return self.status == "corrupted"


def _file_size(path: Path) -> Optional[int]:
    try:
        return path.stat().st_size
    except Exception:
        return None


def _human_size(n: Optional[int]) -> str:
    if n is None:
        return "?"
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TB"


def check_image_signature(path: Path) -> tuple[bool, str, str]:
    """
    Checks the file signature (magic bytes) to verify it matches a supported image format.
    Returns: (is_valid, format_name, error_reason)
    """
    try:
        with open(path, "rb") as f:
            header = f.read(12)
    except Exception as e:
        return False, "", f"Could not read file: {e}"

    if len(header) == 0:
        return False, "", "Empty file (0 bytes)"

    ext = path.suffix.lower()

    if ext in (".jpg", ".jpeg"):
        if header.startswith(b"\xff\xd8"):
            return True, "JPEG", ""
        return False, "", "Invalid JPEG signature (missing FF D8)"

    elif ext == ".png":
        if header.startswith(b"\x89PNG\r\n\x1a\n"):
            return True, "PNG", ""
        return False, "", "Invalid PNG signature"

    elif ext == ".gif":
        if header.startswith(b"GIF87a") or header.startswith(b"GIF89a"):
            return True, "GIF", ""
        return False, "", "Invalid GIF signature"

    elif ext == ".bmp":
        if header.startswith(b"BM"):
            return True, "BMP", ""
        return False, "", "Invalid BMP signature"

    elif ext == ".webp":
        if header.startswith(b"RIFF") and header[8:12] == b"WEBP":
            return True, "WEBP", ""
        return False, "", "Invalid WEBP signature"

    elif ext in (".tiff", ".tif"):
        if header.startswith(b"II*\x00") or header.startswith(b"MM\x00*"):
            return True, "TIFF", ""
        return False, "", "Invalid TIFF signature"

    elif ext == ".ico":
        if header.startswith(b"\x00\x00\x01\x00"):
            return True, "ICO", ""
        return False, "", "Invalid ICO signature"

    elif ext in (".ppm", ".pgm"):
        if header.startswith(b"P1") or header.startswith(b"P2") or header.startswith(b"P3") or header.startswith(b"P4") or header.startswith(b"P5") or header.startswith(b"P6") or header.startswith(b"P7"):
            return True, "PPM/PGM", ""
        return False, "", "Invalid PPM/PGM signature"

    return False, "", f"Unsupported extension {ext}"


def validate_image(path: Path) -> ScanEntry:
    """
    Fast validation checking only file type (extension) and magic bytes.
    No heavy image decode or scanning.
    """
    e = ScanEntry(path)
    e.size_bytes = _file_size(path)

    try:
        if not path.exists():
            raise FileNotFoundError("File does not exist")
        if not path.is_file():
            raise ValueError("Path is not a regular file")
        if not os.access(path, os.R_OK):
            raise PermissionError("File is not readable (permission denied)")
    except Exception as exc:
        e.status = "corrupted"
        e.layer  = "file-system"
        e.reason = str(exc)
        return e

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXT:
        e.status = "corrupted"
        e.layer  = "file-type"
        e.reason = f"Non-image file type ({ext or 'no extension'})"
        return e

    # Check magic bytes signature
    is_valid, fmt, reason = check_image_signature(path)
    if not is_valid:
        e.status = "corrupted"
        e.layer  = "signature"
        e.reason = reason
        return e

    e.status = "ok"
    e.fmt = fmt
    return e


def _safe_validate(path: Path, stop_event: threading.Event) -> Optional[ScanEntry]:
    """
    Thread-safe wrapper.
    Returns None if stop_event is set (scan was cancelled).
    Never raises.
    """
    if stop_event.is_set():
        return None
    try:
        return validate_image(path)
    except Exception as exc:
        entry = ScanEntry(path)
        entry.status = "corrupted"
        entry.layer  = "unknown"
        entry.reason = f"Unhandled error: {exc}"
        return entry


# ─────────────────────────────────────────────────────────────────────────────
#  DIRECTORY WALKER
# ─────────────────────────────────────────────────────────────────────────────

def discover_all_files(root: Path, recursive: bool = True) -> list:
    """Return all file paths under root. Never raises."""
    found = []
    try:
        walker = root.rglob("*") if recursive else root.iterdir()
        for p in walker:
            try:
                if p.is_file():
                    if p.suffix.lower() == ".zip":
                        continue
                    if p.name.lower() in ("desktop.ini", "thumbs.db", ".ds_store"):
                        continue
                    found.append(p)
            except Exception:
                continue
    except Exception:
        pass
    return found


# ─────────────────────────────────────────────────────────────────────────────
#  SCAN ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def handle_zip_files(root_dir: Path):
    """
    Finds zip files in root_dir recursively and prompts user whether to extract them.
    Extracted files will be scanned, and original zips are ignored during scanning.
    """
    import zipfile
    zip_paths = []
    try:
        for p in root_dir.rglob("*.zip"):
            if p.is_file():
                zip_paths.append(p)
    except Exception:
        pass

    if not zip_paths:
        return

    ai_warn(f"Found {len(zip_paths)} zip file(s) in the directory.")
    for zip_path in zip_paths:
        try:
            relative_path = zip_path.relative_to(root_dir)
        except ValueError:
            relative_path = zip_path.name
            
        console.print(f"  [bold yellow]•[/bold yellow] [bold white]{relative_path}[/bold white]")
        try:
            ans = console.input("  Do you want to unzip this file? (y/n) › ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            console.print()
            break

        if ans in ("y", "yes"):
            # Extract to directory named after the zip
            dest_dir = zip_path.parent / zip_path.stem
            ai_think(f"Extracting '{zip_path.name}' to '{dest_dir.name}'...")
            try:
                dest_dir.mkdir(exist_ok=True)
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(dest_dir)
                ai_say(f"Successfully extracted [green]{zip_path.name}[/green] to [cyan]{dest_dir.name}[/cyan].")
            except Exception as e:
                ai_error(f"Failed to extract {zip_path.name}: {e}")
    console.print()


def run_scan(folder_str: str) -> Optional[tuple[list, bool]]:
    """
    Scan files in a directory with a live progress bar.
    • Ctrl+C cancels the scan gracefully and shows partial results.
    • Returns (list[ScanEntry], cancelled) or None on bad directory.
    """
    root = Path(folder_str.strip().strip('"').strip("'"))

    # ── Validate directory ───────────────────────────────────────────────────
    if not root.exists():
        ai_error(f"Directory not found: {root}")
        ai_say("Double-check the path and try again.")
        return None

    if not root.is_dir():
        ai_error(f"That is a file, not a folder: {root}")
        ai_say("Please provide a directory path.")
        return None

    # ── Check for zip files first ────────────────────────────────────────────
    handle_zip_files(root)

    # ── Discover files ───────────────────────────────────────────────────────
    ai_think("Searching for files…")
    files = discover_all_files(root, recursive=True)
    total  = len(files)

    if total == 0:
        ai_warn("No files found in that folder.")
        return [], False

    workers = min(8, os.cpu_count() or 4)
    ai_think(
        f"Found [bold white]{total}[/bold white] file(s). "
        f"Verifying file types with {workers} parallel workers…"
    )
    ai_think("[dim]Press [bold]Ctrl+C[/bold] at any time to stop and see partial results.[/dim]")
    console.print()

    # ── Parallel scan with Ctrl+C support ────────────────────────────────────
    results: list = []
    stop_event    = threading.Event()   # Set this to cancel workers
    cancelled     = False

    with Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("[bold cyan]Scanning[/bold cyan]"),
        BarColumn(
            bar_width=38,
            style="dim white",
            complete_style="bold green",
            finished_style="bold green",
        ),
        TextColumn("[bold white]{task.completed}/{task.total}[/bold white]"),
        TextColumn("[green]+{task.fields[ok]}ok[/green] [red]-{task.fields[bad]}bad[/red]"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as prog:
        task = prog.add_task("scan", total=total, ok=0, bad=0)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_path = {
                executor.submit(_safe_validate, p, stop_event): p
                for p in files
            }

            try:
                for future in as_completed(future_to_path):
                    if stop_event.is_set():
                        break

                    try:
                        entry = future.result()
                    except Exception as exc:
                        path  = future_to_path[future]
                        entry = ScanEntry(path)
                        entry.status = "corrupted"
                        entry.layer  = "unknown"
                        entry.reason = str(exc)

                    if entry is None:
                        continue   # Worker was cancelled

                    results.append(entry)
                    ok_n  = sum(1 for r in results if r.is_ok)
                    bad_n = sum(1 for r in results if r.is_corrupted)
                    prog.update(task, advance=1, ok=ok_n, bad=bad_n)

            except KeyboardInterrupt:
                # User pressed Ctrl+C — signal all workers to stop
                stop_event.set()
                cancelled = True
                prog.stop()
                console.print()
                console.print(
                    "[bold yellow]⚠  Scan stopped by user (Ctrl+C). "
                    "Showing partial results…[/bold yellow]"
                )

            finally:
                # Always signal shutdown so threads don't keep running
                stop_event.set()

    return results, cancelled   # May be partial if cancelled


# ─────────────────────────────────────────────────────────────────────────────
#  REPORT RENDERER
# ─────────────────────────────────────────────────────────────────────────────

def render_report(results: list, folder_str: str, partial: bool = False):
    """Print full (or partial) scan report with score, health bar, and tables."""
    total  = len(results)
    ok_n   = sum(1 for r in results if r.is_ok)
    bad_n  = sum(1 for r in results if r.is_corrupted)
    rate   = (bad_n / total * 100) if total > 0 else 0.0
    health = 100.0 - rate

    score_col = "green" if rate == 0 else ("yellow" if rate < 20 else "red")

    # ── Health bar ────────────────────────────────────────────────────────────
    W      = 40
    filled = int(W * ok_n / total) if total else 0
    empty  = W - filled
    bar    = (
        f"[bold green]{'|' * filled}[/bold green]"
        f"[dim red]{'|' * empty}[/dim red]"
    )

    score_str = (
        f"[bold {score_col}]{ok_n}[/bold {score_col}]"
        f"[bold white] / {total}[/bold white]"
        f"  images OK  •  [bold red]{bad_n}[/bold red] corrupted"
    )

    partial_note = "  [bold yellow](Partial results — scan was stopped early)[/bold yellow]" if partial else ""

    console.print()
    title_txt = "[bold cyan]◉  SCAN REPORT[/bold cyan]"
    if partial:
        title_txt += "  [bold yellow](PARTIAL)[/bold yellow]"
    console.print(Rule(title_txt, style="cyan"))
    console.print()

    # ── Summary panel ─────────────────────────────────────────────────────────
    summary = "\n".join(filter(None, [
        f"  [bold white]Directory  :[/bold white] [dim]{folder_str}[/dim]",
        f"  [bold white]Scanned at :[/bold white] [dim]{datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}[/dim]",
        partial_note,
        "",
        f"  [bold white]Score      :[/bold white]  {score_str}",
        f"  [bold white]Health     :[/bold white]  {bar}  [{score_col}]{health:.1f}%[/{score_col}]",
        "",
        f"  [bold green]✓  Good Images :[/bold green]   {ok_n:>6,}",
        f"  [bold red]✗  Corrupted   :[/bold red]   {bad_n:>6,}",
        f"  [bold white]   Scanned     :[/bold white]   {total:>6,}",
        "",
        f"  [bold white]Corruption Rate :[/bold white]  [{score_col}]{rate:.2f}%[/{score_col}]",
    ]))

    panel_title = "[bold cyan]◉  Third Eye — Image Health Score[/bold cyan]"
    if partial:
        panel_title += "  [bold yellow]⚠ PARTIAL[/bold yellow]"

    console.print(
        Panel(summary, title=panel_title, border_style="cyan", padding=(1, 2))
    )
    console.print()

    # ── Verdict ───────────────────────────────────────────────────────────────
    if partial:
        ai_say(
            f"[bold yellow]Scan stopped early.[/bold yellow] "
            f"Of [bold white]{total}[/bold white] files checked so far: "
            f"[bold green]{ok_n} OK[/bold green], [bold red]{bad_n} corrupted[/bold red]."
        )
    elif bad_n == 0:
        ai_say(
            f"[bold green]Perfect score! All {total} images are healthy.[/bold green] "
            "Dataset is clean and ready."
        )
    elif rate < 5:
        ai_say(
            f"[bold green]{ok_n}/{total} images passed.[/bold green] "
            f"[bold yellow]{bad_n} corrupted[/bold yellow] — low impact."
        )
    elif rate < 20:
        ai_say(
            f"[bold yellow]{ok_n}/{total} images passed.[/bold yellow] "
            f"[bold red]{bad_n} corrupted[/bold red] — clean before training."
        )
    else:
        ai_say(
            f"[bold red]High corruption! {bad_n}/{total} files are bad ({rate:.1f}%).[/bold red] "
            "Clean your dataset before use."
        )

    # ── Corrupted files table ─────────────────────────────────────────────────
    corrupted = [r for r in results if r.is_corrupted]
    if corrupted:
        console.print()
        console.print(Rule(
            f"[bold red]Corrupted Files  ({bad_n})[/bold red]", style="dim red"
        ))
        console.print()

        tbl = Table(
            box=box.ROUNDED,
            border_style="dim red",
            header_style="bold red",
            show_lines=True,
            expand=True,
        )
        tbl.add_column("#",      style="dim",       width=5,  justify="right")
        tbl.add_column("File",   style="bold white", overflow="fold")
        tbl.add_column("Check",  style="yellow",     width=12)
        tbl.add_column("Size",   style="dim",        width=10, justify="right")
        tbl.add_column("Reason", style="dim red",    overflow="fold")
        tbl.add_column("Action", style="bold blue", justify="center", width=12)

        for i, entry in enumerate(corrupted, 1):
            tbl.add_row(
                str(i),
                entry.filename,
                entry.layer,
                _human_size(entry.size_bytes),
                entry.reason[:110],
                "[bold white on blue]  View  [/bold white on blue]"
            )
        console.print(tbl)

    # ── OK files (only shown when small set) ──────────────────────────────────
    ok_files = [r for r in results if r.is_ok]
    if ok_files:
        console.print()
        if len(ok_files) <= 30:
            console.print(Rule(
                f"[bold green]Valid Images  ({ok_n})[/bold green]", style="dim green"
            ))
            console.print()
            tbl2 = Table(
                box=box.SIMPLE_HEAD,
                header_style="bold green",
                show_lines=False,
                expand=True,
            )
            tbl2.add_column("#",      style="dim",      width=5, justify="right")
            tbl2.add_column("File",   style="white",    overflow="fold")
            tbl2.add_column("Format", style="dim cyan", width=8)
            tbl2.add_column("Size",   style="dim",      width=10, justify="right")
            tbl2.add_column("Action", style="bold blue", justify="center", width=12)
            for i, entry in enumerate(ok_files, 1):
                tbl2.add_row(
                    str(i),
                    entry.filename,
                    entry.fmt or "?",
                    _human_size(entry.size_bytes),
                    "[bold white on blue]  View  [/bold white on blue]"
                )
            console.print(tbl2)
        else:
            console.print(
                f"  [bold green]✓ {ok_n} valid images[/bold green] "
                "[dim](list hidden — too many to display)[/dim]"
            )

    console.print()
    console.print(Rule(style="dim white"))
    console.print()


# ─────────────────────────────────────────────────────────────────────────────
#  POST-SCAN ACTIONS & IMAGE POPUP VIEWER
# ─────────────────────────────────────────────────────────────────────────────

def show_image_popup(image_path: Path):
    """Pop up a Tkinter window showing the image with a blue close button."""
    try:
        import tkinter as tk
        from PIL import Image, ImageTk
    except ImportError as e:
        ai_error(f"Cannot display image. Tkinter or Pillow is missing: {e}")
        return

    try:
        # Open image using Pillow
        img = Image.open(image_path)
        
        # Resize to fit within screen limits
        max_w, max_h = 800, 600
        w, h = img.size
        if w > max_w or h > max_h:
            ratio = min(max_w / w, max_h / h)
            img = img.resize((int(w * ratio), int(h * ratio)), Image.Resampling.LANCZOS)

        root = tk.Tk()
        root.title(f"Third Eye Viewer - {image_path.name}")
        root.configure(bg="#1a1a1a")
        
        # Keep it on top
        root.attributes("-topmost", True)

        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(img)

        # Title label inside window
        title_lbl = tk.Label(
            root, 
            text=f"Viewing: {image_path.name} ({w}x{h})", 
            fg="white", 
            bg="#1a1a1a", 
            font=("Arial", 10, "italic")
        )
        title_lbl.pack(pady=(10, 5))

        # Display Image
        img_lbl = tk.Label(root, image=photo, bg="#1a1a1a")
        img_lbl.image = photo  # keep reference
        img_lbl.pack(padx=20, pady=10)

        # Close button with a big ✕ cross icon
        def close_viewer():
            root.destroy()

        close_btn = tk.Button(
            root,
            text="  ✕  Close Image  ",
            command=close_viewer,
            bg="#0066cc",  # Blue color
            fg="white",
            activebackground="#004d99",
            activeforeground="white",
            font=("Arial", 11, "bold"),
            bd=0,
            cursor="hand2",
            padx=10,
            pady=6
        )
        close_btn.pack(pady=(5, 15))

        # Bind keys to close
        root.bind("<Escape>", lambda e: close_viewer())
        root.bind("<q>", lambda e: close_viewer())

        # Start tkinter mainloop
        root.mainloop()
    except Exception as e:
        ai_error(f"Failed to show image popup: {e}")


def move_corrupted_files(results: list, scanned_dir: Path, history: list):
    corrupted = [r for r in results if r.is_corrupted]
    if not corrupted:
        ai_say("No corrupted files found to move.")
        return

    target_dir = scanned_dir / "Curropted images"
    try:
        target_dir.mkdir(exist_ok=True)
    except Exception as e:
        ai_error(f"Could not create folder '{target_dir.name}': {e}")
        return

    ai_think(f"Moving {len(corrupted)} file(s) to '{target_dir.name}'...")
    
    moved = []
    moved_count = 0
    for entry in corrupted:
        try:
            # Skip if already inside target folder
            if entry.path.parent == target_dir:
                continue
            
            original_path = entry.path
            dest = target_dir / entry.path.name
            if dest.exists():
                base = dest.stem
                ext = dest.suffix
                idx = 1
                while True:
                    dest = target_dir / f"{base}_{idx}{ext}"
                    if not dest.exists():
                        break
                    idx += 1

            shutil.move(str(original_path), str(dest))
            entry.path = dest
            moved.append((original_path, dest, entry))
            moved_count += 1
        except Exception as e:
            ai_error(f"Failed to move {entry.filename}: {e}")

    if moved:
        history.append(moved)
    ai_say(f"Successfully moved {moved_count} corrupted file(s) to [cyan]{target_dir.name}[/cyan].")


def separate_images(results: list, scanned_dir: Path, history: list):
    ok_files = [r for r in results if r.is_ok]
    if not ok_files:
        ai_say("No valid images found to move.")
        return

    target_dir = scanned_dir / "Images"
    try:
        target_dir.mkdir(exist_ok=True)
    except Exception as e:
        ai_error(f"Could not create folder '{target_dir.name}': {e}")
        return

    ai_think(f"Moving {len(ok_files)} valid image(s) to '{target_dir.name}'...")
    
    moved = []
    moved_count = 0
    for entry in ok_files:
        try:
            if entry.path.parent == target_dir:
                continue
            
            original_path = entry.path
            dest = target_dir / entry.path.name
            if dest.exists():
                base = dest.stem
                ext = dest.suffix
                idx = 1
                while True:
                    dest = target_dir / f"{base}_{idx}{ext}"
                    if not dest.exists():
                        break
                    idx += 1

            shutil.move(str(original_path), str(dest))
            entry.path = dest
            moved.append((original_path, dest, entry))
            moved_count += 1
        except Exception as e:
            ai_error(f"Failed to move {entry.filename}: {e}")

    if moved:
        history.append(moved)
    ai_say(f"Successfully moved {moved_count} valid image(s) to [cyan]{target_dir.name}[/cyan].")


def rollback_last_operation(history: list):
    """Restores files from the last move operation back to their original locations."""
    if not history:
        ai_warn("No move operations to undo / rollback.")
        return

    last_op = history.pop()
    ai_think(f"Undoing last move operation ({len(last_op)} file(s))...")
    
    undone_count = 0
    for original_path, new_path, entry in reversed(last_op):
        try:
            # Create parent folder if deleted
            original_path.parent.mkdir(exist_ok=True, parents=True)
            
            # Move file back
            shutil.move(str(new_path), str(original_path))
            
            # Restore ScanEntry path
            entry.path = original_path
            undone_count += 1
        except Exception as e:
            ai_error(f"Failed to restore {new_path.name} to original location: {e}")

    # Try to clean up empty destination folders
    for _, new_path, _ in last_op:
        try:
            parent = new_path.parent
            if parent.exists() and len(os.listdir(parent)) == 0:
                parent.rmdir()
        except Exception:
            pass

    ai_say(f"Successfully rolled back: restored {undone_count} file(s) to their original paths.")


def post_scan_menu(results: list, root_dir: Path):
    """Interactive options shown to the user after a scan completes."""
    corrupted = [r for r in results if r.is_corrupted]
    ok_files = [r for r in results if r.is_ok]
    
    # Store move operation history for rollback
    history = []

    while True:
        console.print()
        console.print(Rule("[bold cyan]Post-Scan Actions[/bold cyan]", style="dim cyan"))
        console.print()
        console.print("  [bold cyan]1.[/bold cyan] View a file from the report (opens popup)")
        console.print("  [bold cyan]2.[/bold cyan] Move corrupted files to [bold yellow]'Curropted images'[/bold yellow] folder")
        console.print("  [bold cyan]3.[/bold cyan] Separate images (move valid images to [bold green]'Images'[/bold green] folder)")
        console.print("  [bold cyan]4.[/bold cyan] Rollback / Undo last move operation")
        console.print("  [bold cyan]5.[/bold cyan] Done (return to chatbot)")
        console.print()

        try:
            choice = console.input("[bold white]Option (1-5) › [/bold white]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print()
            break

        if choice == "1":
            console.print("\n[dim]To view: enter index (e.g. '3' for corrupted #3, 'v5' for valid #5, or filename):[/dim]")
            try:
                target_str = console.input("[bold white]File to view › [/bold white]").strip()
            except (KeyboardInterrupt, EOFError):
                console.print()
                continue
                
            if not target_str:
                continue

            target_path = None
            if target_str.lower().startswith("v"):
                try:
                    num = int(target_str[1:])
                    if 1 <= num <= len(ok_files):
                        target_path = ok_files[num - 1].path
                    else:
                        ai_error(f"Valid image number '{num}' is out of range.")
                except ValueError:
                    pass
            elif target_str.lower().startswith("c"):
                try:
                    num = int(target_str[1:])
                    if 1 <= num <= len(corrupted):
                        target_path = corrupted[num - 1].path
                    else:
                        ai_error(f"Corrupted image number '{num}' is out of range.")
                except ValueError:
                    pass
            else:
                try:
                    num = int(target_str)
                    if corrupted and 1 <= num <= len(corrupted):
                        target_path = corrupted[num - 1].path
                    elif 1 <= num <= len(ok_files):
                        target_path = ok_files[num - 1].path
                    else:
                        ai_error(f"Number '{num}' is out of range.")
                except ValueError:
                    # Try searching by filename
                    matches = [r for r in results if r.filename.lower() == target_str.lower() or r.path.name.lower() == target_str.lower()]
                    if matches:
                        target_path = matches[0].path
                    else:
                        ai_error(f"No file found matching '{target_str}'.")

            if target_path:
                if not target_path.exists():
                    ai_error(f"File no longer exists: {target_path}")
                else:
                    ai_say(f"Displaying image: [bold white]{target_path.name}[/bold white]")
                    show_image_popup(target_path)

        elif choice == "2":
            move_corrupted_files(results, root_dir, history)

        elif choice == "3":
            separate_images(results, root_dir, history)

        elif choice == "4":
            rollback_last_operation(history)

        elif choice == "5" or choice == "":
            break
        else:
            ai_error("Invalid option. Please choose 1, 2, 3, 4, or 5.")


# ─────────────────────────────────────────────────────────────────────────────
#  CHATBOT LOOP
# ─────────────────────────────────────────────────────────────────────────────

def handle_input(raw: str) -> bool:
    """Process one user message. Returns False to exit, True to continue."""
    cmd = raw.lower().strip()

    # Ctrl+C at the input prompt — just remind, don't exit
    if raw == "__ctrl_c__":
        console.print()
        ai_say("Use [cyan]exit[/cyan] to quit, or send a folder path to scan.")
        return True

    if cmd in ("exit", "quit", "bye", "q", ":q"):
        console.print()
        ai_say("Goodbye! May your datasets always be clean. [cyan]◉[/cyan]")
        console.print()
        return False

    if cmd in ("help", "h", "?"):
        print_help()
        return True

    if cmd in ("clear", "cls"):
        clear_screen()
        print_logo()
        print_welcome()
        return True

    if cmd == "":
        ai_say("Send me a folder path to scan, or type [cyan]help[/cyan].")
        return True

    # ── Path detection ────────────────────────────────────────────────────────
    stripped = raw.strip().strip('"').strip("'")
    p        = Path(stripped)

    is_path = (
        os.sep in raw
        or "/" in raw
        or (len(stripped) >= 2 and stripped[1] == ":")
        or raw.startswith(".")
        or p.exists()
    )

    if is_path:
        console.print()
        ai_say(f"Scanning: [bold white]{stripped}[/bold white]")
        console.print()

        # ── Run scan (Ctrl+C handled inside run_scan) ─────────────────────────
        partial = False
        try:
            scan_out = run_scan(stripped)
            if scan_out is None:
                return True
            results, partial = scan_out
        except KeyboardInterrupt:
            # Shouldn't reach here normally, but be safe
            results  = []
            partial  = True

        if len(results) == 0 and not partial:
            return True   # No files found — warning already printed

        # Detect if it was a partial scan
        render_report(results, stripped, partial=partial)
        post_scan_menu(results, p)
        ai_say("Send another folder path or type [cyan]help[/cyan].")
        return True

    # ── Unknown command ───────────────────────────────────────────────────────
    ai_say(
        f"I don't recognize [bold white]{raw!r}[/bold white]. "
        "Send a folder path or type [cyan]help[/cyan]."
    )
    return True


def main():
    clear_screen()
    print_logo()
    print_welcome()

    console.print(
        Padding(
            "[bold yellow]●[/bold yellow] [dim white]Tip  "
            "Paste a folder path and press Enter. "
            "Ctrl+C stops a running scan.[/dim white]",
            pad=(0, 0, 1, 4),
        )
    )
    console.print()

    running = True
    while running:
        try:
            user_in = get_input()
            running = handle_input(user_in)
        except KeyboardInterrupt:
            # Ctrl+C at the prompt — don't exit, just prompt again
            console.print()
            ai_say("Use [cyan]exit[/cyan] to quit.")
        except Exception as exc:
            # Never crash the chatbot
            ai_error(f"Unexpected error: {exc}")
            ai_think("Recovered. Keep chatting.")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print()
        console.print("[dim white]Exiting Third Eye…[/dim white]")
        sys.exit(0)
