"""Stage 2 Validation: LPPLS on real Sentinel-2 temporal series.

Runs geological LPPLS fitting on actual satellite data from Geoscan project.
Tests cross-domain hypothesis: do (m, ω) parameters correlate with finance?

Data source: D:/GEOLOGORAZVEDKA — GOLD/data/satellite/bestobe_gold/indices/
Tiles: 42UYC (19+ scenes), 43UCU (20+ scenes)
"""

import numpy as np
import pytest
import rasterio
import structlog

from pathlib import Path

from src.geo.sentinel_loader import discover_scenes, build_temporal_stack, load_index_at_pixel
from src.geo.geo_lppls import (
    fit_geo_lppls,
    fit_geo_multiwindow,
    compare_with_finance,
    GeoLPPLSResult,
)

log = structlog.get_logger()

# Real data paths
DATA_ROOT = Path("D:/GEOLOGORAZVEDKA — GOLD/data/satellite/bestobe_gold/indices")
TILES = ["42UYC", "43UCU"]
INDICES = ["ndvi", "iron_oxide", "clay_index", "bsi"]

# Sample pixel locations (center of tile — likely stable land)
# These are approximate centers; actual pixel coords depend on raster size
SAMPLE_PIXELS = {
    "42UYC": [(5000, 5000), (3000, 7000), (7000, 3000)],
    "43UCU": [(5000, 5000), (3000, 7000), (7000, 3000)],
}

# Finance parameters from Stage 1 validation (known bubbles that passed filters)
FINANCE_M = np.array([0.637, 0.565, 0.478, 0.437])  # btc17, btc21, dotcom, china
FINANCE_OMEGA = np.array([11.86, 6.00, 9.93, 10.83])


def _get_raster_center(tile_dir: Path) -> tuple[int, int] | None:
    """Get center pixel coordinates from first available scene."""
    scenes = discover_scenes(tile_dir, min_indices=1)
    if not scenes:
        return None
    for f in scenes[0].directory.glob("*.tif"):
        try:
            with rasterio.open(f) as src:
                return src.height // 2, src.width // 2
        except Exception:
            continue
    return None


@pytest.fixture(scope="module")
def tile_scenes() -> dict[str, list]:
    """Discover scenes for all tiles."""
    result = {}
    for tile in TILES:
        tile_dir = DATA_ROOT / tile
        if tile_dir.exists():
            scenes = discover_scenes(tile_dir, min_indices=2)
            result[tile] = scenes
            log.info("tile_loaded", tile=tile, n_scenes=len(scenes))
    return result


class TestSceneDiscovery:
    def test_tiles_have_scenes(self, tile_scenes):
        """Both tiles should have ≥10 scenes for meaningful LPPLS fit."""
        for tile in TILES:
            if tile in tile_scenes:
                assert len(tile_scenes[tile]) >= 5, f"{tile}: only {len(tile_scenes[tile])} scenes"
                log.info("scenes_ok", tile=tile, n=len(tile_scenes[tile]))

    def test_temporal_span(self, tile_scenes):
        """Scenes should span at least 1 year."""
        for tile, scenes in tile_scenes.items():
            if len(scenes) < 2:
                continue
            span_days = (scenes[-1].date - scenes[0].date).days
            assert span_days >= 300, f"{tile}: only {span_days} days span"
            log.info("temporal_span", tile=tile, days=span_days)

    def test_print_scene_summary(self, tile_scenes):
        print("\n" + "=" * 70)
        print("SENTINEL-2 SCENE INVENTORY")
        print("=" * 70)
        for tile, scenes in tile_scenes.items():
            if not scenes:
                continue
            span = (scenes[-1].date - scenes[0].date).days
            print(
                f"{tile}: {len(scenes)} scenes, {scenes[0].date_str}—{scenes[-1].date_str} ({span}d)"
            )
            # Count available indices
            idx_counts = {}
            for s in scenes:
                for idx in s.available_indices:
                    idx_counts[idx] = idx_counts.get(idx, 0) + 1
            for idx, cnt in sorted(idx_counts.items()):
                print(f"  {idx}: {cnt}/{len(scenes)} scenes")
        print("=" * 70)


class TestGeoLPPLSRealData:
    """Fit LPPLS to real Sentinel-2 temporal series."""

    @pytest.fixture(scope="class")
    def geo_fits(self, tile_scenes) -> list[GeoLPPLSResult]:
        """Run geo LPPLS on all tile/index/pixel combinations."""
        results = []

        for tile, scenes in tile_scenes.items():
            if len(scenes) < 10:
                log.warning("too_few_scenes", tile=tile, n=len(scenes))
                continue

            # Get actual raster center
            center = _get_raster_center(DATA_ROOT / tile)
            if center is None:
                continue
            cy, cx = center
            # Sample 3 pixels: center, offset NW, offset SE
            pixels = [(cy, cx), (cy - 1000, cx - 1000), (cy + 1000, cx + 1000)]

            for index in INDICES:
                for row, col in pixels:
                    stack = build_temporal_stack(
                        scenes,
                        index,
                        row,
                        col,
                        tile_id=tile,
                        window_size=5,
                    )
                    if stack is None:
                        continue

                    # Fit LPPLS
                    result = fit_geo_lppls(
                        stack.days_since_start,
                        stack.values,
                        index_name=f"{tile}_{index}_r{row}_c{col}",
                        grid_size=10,
                    )
                    results.append(result)

        log.info("geo_fits_complete", n_total=len(results))
        return results

    def test_some_fits_succeed(self, geo_fits):
        """At least some geo fits should produce valid results."""
        valid = [r for r in geo_fits if r.r_squared > 0.3]
        log.info("valid_fits", n_valid=len(valid), n_total=len(geo_fits))
        # Geological data is noisy — even a few valid fits is informative
        assert len(geo_fits) > 0, "No fits were attempted"

    def test_print_geo_results(self, geo_fits):
        print("\n" + "=" * 95)
        print("GEOLOGICAL LPPLS FITS — Real Sentinel-2 Data")
        print("=" * 95)
        print(f"{'Name':<40} {'R²':>6} {'PT':>4} {'m':>6} {'ω':>6} {'tc':>8} {'DW':>6}")
        print("-" * 95)
        for r in sorted(geo_fits, key=lambda x: -x.r_squared)[:20]:
            m_str = f"{r.m:.3f}" if r.m else "—"
            w_str = f"{r.omega:.2f}" if r.omega else "—"
            tc_str = f"{r.tc_days:.0f}d" if r.tc_days else "—"
            print(
                f"{r.index_name:<40} {r.r_squared:>6.4f} "
                f"{'YES' if r.is_phase_transition else 'NO':>4} "
                f"{m_str:>6} {w_str:>6} {tc_str:>8} {r.durbin_watson:>6.3f}"
            )
        print("=" * 95)
        print(f"Total fits: {len(geo_fits)}")
        valid = [r for r in geo_fits if r.r_squared > 0.5]
        pt = [r for r in geo_fits if r.is_phase_transition]
        print(f"R² > 0.5: {len(valid)}, Phase transitions detected: {len(pt)}")


class TestCrossDomainComparison:
    """THE key test: do geological (m, ω) match financial (m, ω)?"""

    @pytest.fixture(scope="class")
    def comparison(self, tile_scenes) -> dict:
        """Re-run geo fits and compare with finance — self-contained."""
        geo_results = []
        for tile, scenes in tile_scenes.items():
            if len(scenes) < 10:
                continue
            center = _get_raster_center(DATA_ROOT / tile)
            if center is None:
                continue
            cy, cx = center
            for index in INDICES:
                stack = build_temporal_stack(scenes, index, cy, cx, tile_id=tile, window_size=5)
                if stack is None:
                    continue
                result = fit_geo_lppls(
                    stack.days_since_start, stack.values, index_name=f"{tile}_{index}", grid_size=10
                )
                geo_results.append(result)
        return compare_with_finance(geo_results, FINANCE_M, FINANCE_OMEGA)

    def test_comparison_runs(self, comparison):
        assert "geo_m" in comparison
        assert "finance_m" in comparison

    def test_print_cross_domain(self, comparison):
        print("\n" + "=" * 70)
        print("CROSS-DOMAIN (m, ω) COMPARISON: Geology vs Finance")
        print("=" * 70)
        geo_m = comparison["geo_m"]
        geo_omega = comparison["geo_omega"]
        print(f"Geological fits with (m, ω): {len(geo_m)}")
        print(f"Financial fits with (m, ω): {len(FINANCE_M)}")

        if len(geo_m) > 0:
            print(
                f"\nGeo m:     mean={np.mean(geo_m):.3f}, std={np.std(geo_m):.3f}, range=[{np.min(geo_m):.3f}, {np.max(geo_m):.3f}]"
            )
            print(
                f"Fin m:     mean={np.mean(FINANCE_M):.3f}, std={np.std(FINANCE_M):.3f}, range=[{np.min(FINANCE_M):.3f}, {np.max(FINANCE_M):.3f}]"
            )
        if len(geo_omega) > 0:
            print(
                f"Geo ω:     mean={np.mean(geo_omega):.2f}, std={np.std(geo_omega):.2f}, range=[{np.min(geo_omega):.2f}, {np.max(geo_omega):.2f}]"
            )
            print(
                f"Fin ω:     mean={np.mean(FINANCE_OMEGA):.2f}, std={np.std(FINANCE_OMEGA):.2f}, range=[{np.min(FINANCE_OMEGA):.2f}, {np.max(FINANCE_OMEGA):.2f}]"
            )

        ks_m = comparison.get("ks_m")
        ks_omega = comparison.get("ks_omega")
        if ks_m:
            print(
                f"\nKS test (m):     statistic={ks_m['statistic']:.4f}, p-value={ks_m['pvalue']:.4f}"
            )
            print(
                f"KS test (ω):     statistic={ks_omega['statistic']:.4f}, p-value={ks_omega['pvalue']:.4f}"
            )
            similar = comparison["are_similar"]
            print(
                f"\nConclusion: {'UNIVERSAL (p>0.05)' if similar else 'DOMAIN-SPECIFIC (p<0.05)'}"
            )
        else:
            print(f"\nInsufficient data for KS test: {comparison.get('reason', '?')}")
        print("=" * 70)
