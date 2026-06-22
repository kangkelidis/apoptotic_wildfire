"""
Output Management: Timestamped output directory structure.

Creates organized output directories with:
  outputs/
    YYYYMMDD_HHMMSS/
      data/       - Parquet metrics and analysis tables (CSV fallback)
      videos/     - MP4 video recordings
      plots/      - Analysis plots and figures
      summary.txt - Analysis summary report
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

BASE_DIR: str = "outputs"


@dataclass
class OutputPaths:
    """Paths for a timestamped output directory."""
    root: Path
    data: Path
    videos: Path
    plots: Path
    summary: Path

    @property
    def timestamp(self) -> str:
        """Extract timestamp from root directory name."""
        return self.root.name


def create_output_directory(
    suffix: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> OutputPaths:
    """
    Create a timestamped output directory with optional suffix.

    Args:
        suffix: Optional suffix appended as "<timestamp>_<suffix>"
        timestamp: Optional explicit timestamp (YYYYMMDD_HHMMSS)
    """
    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    name = ts if not suffix else f"{ts}_{str(suffix).strip()}"
    root = Path(BASE_DIR) / name
    return output_paths_from_root(root)


def output_paths_from_root(root: str | Path) -> OutputPaths:
    """Build OutputPaths from an existing/new root path and ensure subdirs exist."""
    root = Path(root)
    paths = OutputPaths(
        root=root,
        data=root / "data",
        videos=root / "videos",
        plots=root / "plots",
        summary=root / "summary.txt",
    )
    paths.data.mkdir(parents=True, exist_ok=True)
    paths.videos.mkdir(parents=True, exist_ok=True)
    paths.plots.mkdir(parents=True, exist_ok=True)
    return paths


def find_latest_output(base_dir: str = "outputs") -> Optional[OutputPaths]:
    """
    Find the most recent timestamped output directory.

    Args:
        base_dir: Base directory to search

    Returns:
        OutputPaths for latest directory, or None if none found
    """
    base = Path(base_dir)
    if not base.exists():
        return None

    # Find directories matching timestamp pattern
    dirs = [d for d in base.iterdir() if d.is_dir() and
            _is_timestamp_dir(d.name)]

    if not dirs:
        return None

    # Sort by name (timestamps sort chronologically)
    latest = sorted(dirs, key=lambda d: d.name)[-1]

    return OutputPaths(
        root=latest,
        data=latest / "data",
        videos=latest / "videos",
        plots=latest / "plots",
        summary=latest / "summary.txt",
    )


def find_output_by_timestamp(
    timestamp: str,
) -> Optional[OutputPaths]:
    """
    Find output directory by timestamp prefix.

    Args:
        timestamp: Full or partial timestamp (e.g., "20260127" or "20260127_143000")

    Returns:
        OutputPaths if found, None otherwise
    """
    base = Path(BASE_DIR)
    if not base.exists():
        return None

    # Find directories matching the timestamp prefix
    matches = [d for d in base.iterdir() if d.is_dir() and
               d.name.startswith(timestamp)]

    if not matches:
        return None

    # Return the most recent match
    match = sorted(matches, key=lambda d: d.name)[-1]

    return OutputPaths(
        root=match,
        data=match / "data",
        videos=match / "videos",
        plots=match / "plots",
        summary=match / "summary.txt",
    )


def _is_timestamp_dir(name: str) -> bool:
    """Check if directory name matches timestamp pattern YYYYMMDD_HHMMSS."""
    if len(name) != 15:
        return False
    if name[8] != "_":
        return False
    return name[:8].isdigit() and name[9:].isdigit()
