"""Sentinel-2 multi-date spectral index loader for LPPLS analysis.

Loads per-date spectral index GeoTIFFs and builds temporal stacks.
Reuses conventions from Geoscan Gold 2026 project.

Expected data layout:
    {indices_root}/{tile_id}/{scene_date}/{tile_id}_{date}_{index}.tif
    OR
    {indices_root}/{tile_id}/{scene_dir}/*_{index}.tif

Supported indices: NDVI, BSI, Clay Index, Iron Oxide, NDWI,
                   Ferrous Minerals, Laterite, Gossan, SWIR Alteration.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import structlog
from numpy.typing import NDArray

log = structlog.get_logger()

# Scene directory naming: S2[A|B|C]_{TILE}_{YYYYMMDD}_{idx}_L2A
SCENE_DATE_PATTERN = re.compile(r"(\d{8})")

INDEX_ALIASES: dict[str, list[str]] = {
    "ndvi": ["ndvi"],
    "bsi": ["bsi"],
    "iron_oxide": ["iron_oxide"],
    "clay_index": ["clay_index", "clay"],
    "ndwi": ["ndwi"],
    "ferrous_minerals": ["ferrous_minerals"],
    "laterite": ["laterite"],
    "gossan": ["gossan"],
    "swir_alteration": ["swir_alteration"],
}


@dataclass
class SceneInfo:
    """Metadata for a single Sentinel-2 scene."""

    date: datetime
    date_str: str  # YYYYMMDD
    directory: Path
    available_indices: list[str] = field(default_factory=list)


@dataclass
class TemporalStack:
    """Multi-date stack of a spectral index at a spatial location."""

    index_name: str
    dates: list[datetime]
    days_since_start: NDArray  # float64, days from first observation
    values: NDArray  # (n_dates,) mean values per date
    pixel_row: int
    pixel_col: int
    tile_id: str


def discover_scenes(tile_dir: Path, min_indices: int = 2) -> list[SceneInfo]:
    """Find all scene directories within a tile folder.

    Args:
        tile_dir: Path to tile directory (e.g., .../42UYC/)
        min_indices: Minimum index files to consider scene valid

    Returns:
        Sorted list of SceneInfo (oldest first)
    """
    scenes: list[SceneInfo] = []
    if not tile_dir.is_dir():
        log.warning("tile_dir_not_found", path=str(tile_dir))
        return scenes

    for entry in sorted(tile_dir.iterdir()):
        if not entry.is_dir():
            continue
        # Extract date from directory name
        match = SCENE_DATE_PATTERN.search(entry.name)
        if not match:
            continue
        date_str = match.group(1)
        try:
            date = datetime.strptime(date_str, "%Y%m%d")
        except ValueError:
            continue

        # Find available index files
        tif_files = list(entry.glob("*.tif")) + list(entry.glob("*.tiff"))
        available = []
        for idx_name, aliases in INDEX_ALIASES.items():
            for alias in aliases:
                if any(alias in f.stem.lower() for f in tif_files):
                    available.append(idx_name)
                    break

        if len(available) >= min_indices:
            scenes.append(
                SceneInfo(
                    date=date, date_str=date_str, directory=entry, available_indices=available
                )
            )

    scenes.sort(key=lambda s: s.date)
    log.info("scenes_discovered", tile=tile_dir.name, n_scenes=len(scenes))
    return scenes


def load_index_at_pixel(
    scene: SceneInfo, index_name: str, row: int, col: int, window_size: int = 3
) -> float | None:
    """Load a spectral index value at a specific pixel from a scene.

    Uses a small window (default 3x3) and takes the mean to reduce noise.

    Args:
        scene: Scene to load from
        index_name: Spectral index name (e.g., "ndvi")
        row: Pixel row
        col: Pixel column
        window_size: Side length of averaging window (odd number)

    Returns:
        Mean value in window, or None if loading fails
    """
    try:
        import rasterio
    except ImportError:
        log.warning("rasterio_not_installed", msg="pip install rasterio")
        return None

    aliases = INDEX_ALIASES.get(index_name, [index_name])
    tif_path = None
    for f in scene.directory.glob("*.tif"):
        if any(alias in f.stem.lower() for alias in aliases):
            tif_path = f
            break

    if tif_path is None:
        return None

    try:
        half = window_size // 2
        with rasterio.open(tif_path) as src:
            r0 = max(0, row - half)
            c0 = max(0, col - half)
            r1 = min(src.height, row + half + 1)
            c1 = min(src.width, col + half + 1)
            window = rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0)
            data = src.read(1, window=window).astype(float)
            data[data == src.nodata] = np.nan
            val = float(np.nanmean(data))
            return val if np.isfinite(val) else None
    except Exception as e:
        log.warning("pixel_load_failed", scene=scene.date_str, index=index_name, error=str(e))
        return None


def build_temporal_stack(
    scenes: list[SceneInfo],
    index_name: str,
    row: int,
    col: int,
    tile_id: str = "",
    window_size: int = 3,
) -> TemporalStack | None:
    """Build a time series of spectral index values at a pixel location.

    Args:
        scenes: Sorted list of scenes (oldest first)
        index_name: Spectral index (e.g., "ndvi")
        row, col: Pixel coordinates
        tile_id: Tile identifier for metadata
        window_size: Averaging window for noise reduction

    Returns:
        TemporalStack or None if insufficient data
    """
    dates: list[datetime] = []
    values: list[float] = []

    for scene in scenes:
        if index_name not in scene.available_indices:
            continue
        val = load_index_at_pixel(scene, index_name, row, col, window_size)
        if val is not None:
            dates.append(scene.date)
            values.append(val)

    if len(dates) < 10:
        log.warning(
            "insufficient_temporal_data",
            index=index_name,
            n_dates=len(dates),
            min_required=10,
        )
        return None

    dates_arr = np.array(dates)
    values_arr = np.array(values)

    # Days since first observation
    t0 = dates_arr[0]
    days = np.array([(d - t0).days for d in dates_arr], dtype=float)

    log.info(
        "temporal_stack_built",
        index=index_name,
        n_dates=len(dates),
        span_days=int(days[-1]),
        tile=tile_id,
    )

    return TemporalStack(
        index_name=index_name,
        dates=list(dates_arr),
        days_since_start=days,
        values=values_arr,
        pixel_row=row,
        pixel_col=col,
        tile_id=tile_id,
    )
