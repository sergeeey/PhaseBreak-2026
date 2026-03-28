"""Tests for Sentinel-2 multi-date loader (synthetic/mock data)."""

import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from src.geo.sentinel_loader import (
    SceneInfo,
    TemporalStack,
    discover_scenes,
    SCENE_DATE_PATTERN,
    INDEX_ALIASES,
)


class TestSceneDatePattern:
    def test_standard_format(self):
        m = SCENE_DATE_PATTERN.search("S2C_43UCU_20260108_0_L2A")
        assert m is not None
        assert m.group(1) == "20260108"

    def test_date_only(self):
        m = SCENE_DATE_PATTERN.search("20250615")
        assert m is not None
        assert m.group(1) == "20250615"

    def test_no_date(self):
        m = SCENE_DATE_PATTERN.search("no_date_here")
        assert m is None


class TestIndexAliases:
    def test_core_indices(self):
        assert "ndvi" in INDEX_ALIASES
        assert "bsi" in INDEX_ALIASES
        assert "iron_oxide" in INDEX_ALIASES
        assert "clay_index" in INDEX_ALIASES

    def test_clay_alias(self):
        assert "clay" in INDEX_ALIASES["clay_index"]


class TestDiscoverScenes:
    def test_discovers_valid_scenes(self, tmp_path):
        """Create mock scene directories and verify discovery."""
        dates = ["20250601", "20250615", "20250701"]
        for d in dates:
            scene_dir = tmp_path / f"S2C_42UYC_{d}_0_L2A"
            scene_dir.mkdir()
            (scene_dir / f"42UYC_{d}_ndvi.tif").write_bytes(b"mock")
            (scene_dir / f"42UYC_{d}_bsi.tif").write_bytes(b"mock")
            (scene_dir / f"42UYC_{d}_iron_oxide.tif").write_bytes(b"mock")

        scenes = discover_scenes(tmp_path, min_indices=2)
        assert len(scenes) == 3
        assert all(isinstance(s, SceneInfo) for s in scenes)
        assert scenes[0].date < scenes[-1].date
        assert "ndvi" in scenes[0].available_indices

    def test_skips_insufficient_indices(self, tmp_path):
        scene_dir = tmp_path / "S2C_42UYC_20250601_0_L2A"
        scene_dir.mkdir()
        (scene_dir / "only_one.tif").write_bytes(b"mock")

        scenes = discover_scenes(tmp_path, min_indices=2)
        assert len(scenes) == 0

    def test_empty_directory(self, tmp_path):
        scenes = discover_scenes(tmp_path)
        assert len(scenes) == 0

    def test_nonexistent_directory(self):
        scenes = discover_scenes(Path("/nonexistent"))
        assert len(scenes) == 0

    def test_sorted_by_date(self, tmp_path):
        for d in ["20250901", "20250301", "20250601"]:
            sd = tmp_path / f"scene_{d}"
            sd.mkdir()
            (sd / f"ndvi_{d}.tif").write_bytes(b"mock")
            (sd / f"bsi_{d}.tif").write_bytes(b"mock")

        scenes = discover_scenes(tmp_path, min_indices=2)
        dates = [s.date for s in scenes]
        assert dates == sorted(dates)


class TestTemporalStack:
    def test_dataclass_fields(self):
        ts = TemporalStack(
            index_name="ndvi",
            dates=[datetime(2025, 1, 1), datetime(2025, 2, 1)],
            days_since_start=np.array([0.0, 31.0]),
            values=np.array([0.45, 0.52]),
            pixel_row=100,
            pixel_col=200,
            tile_id="42UYC",
        )
        assert ts.index_name == "ndvi"
        assert len(ts.values) == 2
        assert ts.days_since_start[-1] == 31.0
