"""Command-line interface for FITS to Zarr conversion.

This module provides the CLI entry point for converting OVRO-LWA FITS files
to Zarr format using the Typer framework.
"""

from __future__ import annotations

import logging
import os
import sys
import warnings
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import Iterator, Optional

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from ovro_lwa_portal.fits_to_zarr_xradio import fix_fits_headers
from ovro_lwa_portal.ingest.core import ConversionConfig, FITSToZarrConverter

__all__ = ["main", "app"]


@contextmanager
def suppress_stderr() -> Iterator[None]:
    """Context manager to temporarily suppress stderr output.

    This is useful for suppressing C++ library warnings from casacore
    that cannot be controlled via Python logging.
    """
    # Save the original stderr file descriptor
    stderr_fd = sys.stderr.fileno()
    old_stderr = os.dup(stderr_fd)

    try:
        # Redirect stderr to /dev/null
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, stderr_fd)
        os.close(devnull)
        yield
    finally:
        # Restore original stderr
        os.dup2(old_stderr, stderr_fd)
        os.close(old_stderr)


console = Console()
app = typer.Typer(
    name="ovro-ingest",
    help="Convert OVRO-LWA FITS files to Zarr format",
    add_completion=False,
)


class LogLevel(str, Enum):
    """Supported logging levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


def _configure_logging(level: LogLevel) -> None:
    """Configure logging based on user preference.

    Parameters
    ----------
    level : LogLevel
        Desired logging level.
    """
    log_level_map = {
        LogLevel.DEBUG: logging.DEBUG,
        LogLevel.INFO: logging.INFO,
        LogLevel.WARNING: logging.WARNING,
        LogLevel.ERROR: logging.ERROR,
    }
    logging.basicConfig(
        level=log_level_map[level],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


@app.command()
def convert(
    input_dir: Path = typer.Argument(
        ...,
        help="Directory containing input FITS files",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    output_dir: Path = typer.Argument(
        ...,
        help="Directory where the Zarr store will be written",
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    zarr_name: str = typer.Option(
        "ovro_lwa_full_lm_only.zarr",
        "--zarr-name",
        "-z",
        help="Name of the output Zarr store",
    ),
    fixed_dir: Optional[Path] = typer.Option(
        None,
        "--fixed-dir",
        "-f",
        help="Directory for storing fixed FITS files (default: OUTPUT_DIR/fixed_fits)",
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    chunk_lm: int = typer.Option(
        1024,
        "--chunk-lm",
        "-c",
        help="Chunk size for l and m spatial dimensions (0 to disable)",
        min=0,
    ),
    rebuild: bool = typer.Option(
        False,
        "--rebuild",
        "-r",
        help="Overwrite existing Zarr store instead of appending",
    ),
    skip_header_fixing: bool = typer.Option(
        False,
        "--skip-header-fixing",
        "-s",
        help="Skip header fixing (assume headers already fixed with fix-headers command)",
    ),
    log_level: LogLevel = typer.Option(
        LogLevel.INFO,
        "--log-level",
        "-l",
        help="Logging verbosity level",
        case_sensitive=False,
    ),
) -> None:
    """Convert OVRO-LWA FITS files to a single Zarr store.

    This command processes FITS image files from the specified INPUT_DIR and
    combines them into an optimized Zarr store at OUTPUT_DIR. Files are grouped
    by observation time and frequency subband.

    \b
    Examples:
        # Basic conversion (fixes headers on-demand)
        ovro-ingest convert /data/fits /data/output

        # Rebuild existing store with verbose logging
        ovro-ingest convert /data/fits /data/output --rebuild --log-level debug

        # Custom Zarr name and chunk size
        ovro-ingest convert /data/fits /data/output \\
            --zarr-name my_data.zarr --chunk-lm 2048

        # Two-step workflow: fix headers first, then convert
        ovro-ingest fix-headers /data/fits /data/fixed_fits
        ovro-ingest convert /data/fits /data/output \\
            --fixed-dir /data/fixed_fits --skip-header-fixing
    """
    _configure_logging(log_level)

    verbose = log_level == LogLevel.DEBUG

    # Build configuration
    config = ConversionConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        zarr_name=zarr_name,
        fixed_dir=fixed_dir,
        chunk_lm=chunk_lm,
        rebuild=rebuild,
        fix_headers_on_demand=not skip_header_fixing,  # Invert the flag
        verbose=verbose,
    )

    # Display configuration
    console.print("\n[bold cyan]OVRO-LWA FITS → Zarr Conversion[/bold cyan]")
    console.print(f"  Input directory:  {input_dir}")
    console.print(f"  Output directory: {output_dir}")
    console.print(f"  Zarr store name:  {zarr_name}")
    console.print(f"  Fixed FITS dir:   {config.fixed_dir}")
    console.print(f"  Chunk size (l,m): {chunk_lm}")
    console.print(f"  Mode:             {'REBUILD' if rebuild else 'APPEND'}")
    console.print(f"  Fix headers:      {'ON-DEMAND' if not skip_header_fixing else 'SKIP (pre-fixed)'}")
    console.print(f"  Log level:        {log_level.value.upper()}\n")

    # Temporarily suppress INFO-level logging during progress display
    # (unless user explicitly requested verbose output)
    if log_level != LogLevel.DEBUG:
        # Suppress our own loggers
        logging.getLogger("ovro_lwa_portal.fits_to_zarr_xradio").setLevel(logging.WARNING)
        logging.getLogger("ovro_lwa_portal.ingest.core").setLevel(logging.WARNING)

        # Suppress external library loggers
        logging.getLogger("viperlog").setLevel(logging.ERROR)  # xradio logging
        logging.getLogger("astropy").setLevel(logging.ERROR)   # astropy logging

        # Suppress astropy warnings
        warnings.filterwarnings("ignore", category=Warning, module="astropy")

    # Create progress display
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Converting...", total=100)

        def progress_callback(stage: str, current: int, total: int, message: str) -> None:
            """Update progress bar based on conversion stage."""
            if total > 0:
                percentage = (current / total) * 100
                progress.update(task, completed=percentage, description=message)

        # Execute conversion (suppress CASA stderr warnings unless in debug mode)
        converter = FITSToZarrConverter(config, progress_callback=progress_callback)

        try:
            if log_level != LogLevel.DEBUG:
                with suppress_stderr():
                    result = converter.convert()
            else:
                result = converter.convert()

            progress.update(task, completed=100, description="Conversion complete")
            console.print(f"\n[bold green]✓[/bold green] Successfully created: {result}")

        except FileNotFoundError as e:
            console.print(f"\n[bold red]✗[/bold red] Error: {e}", style="red")
            console.print(
                "\nNo matching FITS files found. Please check:\n"
                "  • The input directory contains .fits files\n"
                "  • Filenames contain a timestamp (YYYYMMDD_HHMMSS) and frequency (<freq>MHz)"
            )
            raise typer.Exit(code=1) from e

        except RuntimeError as e:
            console.print(f"\n[bold red]✗[/bold red] Conversion failed: {e}", style="red")
            if "lock" in str(e).lower():
                console.print(
                    "\nAnother process may be writing to the same output location.\n"
                    "Please ensure no other conversion processes are running."
                )
            raise typer.Exit(code=1) from e

        except Exception as e:
            console.print(
                f"\n[bold red]✗[/bold red] Unexpected error: {e}",
                style="red",
            )
            if verbose:
                console.print_exception()
            raise typer.Exit(code=1) from e


@app.command()
def fix_headers(
    input_dir: Path = typer.Argument(
        ...,
        help="Directory containing input FITS files",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    fixed_dir: Path = typer.Argument(
        ...,
        help="Directory where fixed FITS files will be written",
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    skip_existing: bool = typer.Option(
        True,
        "--skip-existing/--overwrite",
        help="Skip files that already have fixed versions",
    ),
    log_level: LogLevel = typer.Option(
        LogLevel.INFO,
        "--log-level",
        "-l",
        help="Logging verbosity level",
        case_sensitive=False,
    ),
) -> None:
    """Fix FITS headers ahead of time before conversion.

    This command processes FITS files to ensure they have the necessary headers
    for xradio conversion. It can be run as a separate step before the conversion
    to separate header fixing from the actual Zarr conversion process.

    \b
    Examples:
        # Fix headers for all FITS files
        ovro-ingest fix-headers /data/fits /data/fixed_fits

        # Overwrite existing fixed files
        ovro-ingest fix-headers /data/fits /data/fixed_fits --overwrite

        # Then convert using pre-fixed headers
        ovro-ingest convert /data/fits /data/output \\
            --fixed-dir /data/fixed_fits --skip-header-fixing
    """
    _configure_logging(log_level)

    console.print("\n[bold cyan]OVRO-LWA FITS Header Fixing[/bold cyan]")
    console.print(f"  Input directory:  {input_dir}")
    console.print(f"  Fixed FITS dir:   {fixed_dir}")
    console.print(f"  Skip existing:    {skip_existing}\n")

    # Discover all FITS files
    input_files = sorted(input_dir.glob("*.fits"))
    if not input_files:
        console.print(
            f"[bold red]✗[/bold red] No FITS files found in {input_dir}",
            style="red",
        )
        raise typer.Exit(code=1)

    console.print(f"Found {len(input_files)} FITS file(s)")

    # Temporarily suppress INFO-level logging during progress display
    # (unless user explicitly requested verbose output)
    if log_level != LogLevel.DEBUG:
        # Suppress our own loggers
        logging.getLogger("ovro_lwa_portal.fits_to_zarr_xradio").setLevel(logging.WARNING)

        # Suppress external library loggers
        logging.getLogger("viperlog").setLevel(logging.ERROR)  # xradio logging
        logging.getLogger("astropy").setLevel(logging.ERROR)   # astropy logging

        # Suppress astropy warnings
        warnings.filterwarnings("ignore", category=Warning, module="astropy")

    # Create progress display
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Fixing headers...", total=len(input_files))

        try:
            # Process files one at a time to update progress
            fixed_dir.mkdir(parents=True, exist_ok=True)
            fixed_paths = []

            for i, f in enumerate(sorted(input_files, key=lambda p: p.name)):
                # Update progress with current file
                progress.update(
                    task,
                    completed=i,
                    description=f"Fixing {f.name}..."
                )

                # Fix this single file (suppress CASA stderr warnings unless in debug mode)
                if log_level != LogLevel.DEBUG:
                    with suppress_stderr():
                        result = fix_fits_headers([f], fixed_dir, skip_existing=skip_existing)
                else:
                    result = fix_fits_headers([f], fixed_dir, skip_existing=skip_existing)
                fixed_paths.extend(result)

            progress.update(task, completed=len(input_files), description="Complete")
            console.print(
                f"\n[bold green]✓[/bold green] Successfully fixed {len(fixed_paths)} file(s)"
            )
            console.print(f"Fixed files written to: {fixed_dir}")

        except Exception as e:
            console.print(
                f"\n[bold red]✗[/bold red] Header fixing failed: {e}",
                style="red",
            )
            if log_level == LogLevel.DEBUG:
                console.print_exception()
            raise typer.Exit(code=1) from e


@app.command()
def version() -> None:
    """Display version information."""
    try:
        from ovro_lwa_portal.version import __version__
    except ImportError:
        __version__ = "unknown"

    console.print(f"ovro-ingest version {__version__}")
    console.print("Part of the ovro_lwa_portal package")


@app.callback(invoke_without_command=True)
def callback(
    ctx: typer.Context,
    version_flag: bool = typer.Option(
        False,
        "--version",
        "-v",
        help="Show version and exit",
        is_eager=True,
    ),
) -> None:
    """OVRO-LWA FITS to Zarr conversion tool."""
    if version_flag:
        version()
        raise typer.Exit()

    if ctx.invoked_subcommand is None:
        console.print(
            "[yellow]No command specified. Use --help to see available commands.[/yellow]"
        )
        raise typer.Exit()


def main() -> None:
    """Main entry point for the CLI application."""
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[yellow]Conversion interrupted by user[/yellow]")
        sys.exit(130)


if __name__ == "__main__":
    main()
