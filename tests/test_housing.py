"""Tests for housing price data loader and LPPLS validation."""

import numpy as np
import pandas as pd
import pytest
import structlog

from src.housing.data import (
    HousingDataset,
    HOUSING_BUBBLES,
    HOUSING_CONTROLS,
    load_fhfa_state,
    load_housing_episode,
    make_housing_dataset,
)
from src.lppls.optimizer import LPPLSOptimizer
from src.lppls.confidence import MultiWindowConfidence

log = structlog.get_logger()


class TestHousingDataset:
    def test_episode_definitions(self):
        assert len(HOUSING_BUBBLES) >= 4
        assert len(HOUSING_CONTROLS) >= 3

    def test_bubbles_have_peak(self):
        for name, config in HOUSING_BUBBLES.items():
            assert config["peak_date"] is not None, f"{name} missing peak_date"

    def test_controls_no_peak(self):
        for name, config in HOUSING_CONTROLS.items():
            assert config["peak_date"] is None, f"{name} should have no peak"


class TestFHFALoader:
    def test_load_arizona(self):
        df = load_fhfa_state("AZ", start="2003-01", end="2008-12")
        assert len(df) > 30
        assert "date" in df.columns
        assert "hpi" in df.columns
        assert df["hpi"].dtype == float or df["hpi"].dtype == np.float64

    def test_load_nonexistent_state(self):
        with pytest.raises(ValueError, match="No FHFA data"):
            load_fhfa_state("ZZ")


class TestHousingEpisodes:
    @pytest.fixture(scope="class")
    def episode_results(self) -> dict[str, dict]:
        results = {}
        all_episodes = {**HOUSING_BUBBLES, **HOUSING_CONTROLS}
        for name in all_episodes.keys():  # All episodes
            try:
                ds = load_housing_episode(name)
                # Fit LPPLS
                opt = LPPLSOptimizer(
                    grid_size=10,
                    n_best=5,
                    m_range=(0.1, 0.9),
                    omega_range=(4.0, 15.0),
                )
                model = opt.fit(ds.t, ds.log_price)
                r2 = model.r_squared(ds.t, ds.log_price)
                is_bubble = model.params is not None and model.params.is_bubble and r2 > 0.5
                peak_err = (
                    ds.peak_error_quarters(model.params.tc)
                    if model.params and ds.peak_date
                    else None
                )

                results[name] = {
                    "dataset": ds,
                    "r_squared": r2,
                    "is_bubble": is_bubble,
                    "peak_error_months": peak_err,
                    "m": model.params.m if model.params else None,
                    "omega": model.params.omega if model.params else None,
                    "expected_bubble": name in HOUSING_BUBBLES,
                }
                log.info(
                    "housing_fit",
                    name=name,
                    r2=round(r2, 4),
                    is_bubble=is_bubble,
                    peak_err=peak_err,
                )
            except Exception as e:
                results[name] = {"error": str(e), "expected_bubble": name in HOUSING_BUBBLES}
                log.warning("housing_fit_failed", name=name, error=str(e))
        return results

    def test_episodes_load(self, episode_results):
        loaded = sum(1 for r in episode_results.values() if "error" not in r)
        assert loaded >= 3, f"Only {loaded} episodes loaded"

    def test_print_results(self, episode_results):
        print("\n" + "=" * 85)
        print("HOUSING LPPLS VALIDATION — Phase 1")
        print("=" * 85)
        print(
            f"{'Name':<30} {'Expected':>8} {'Bubble':>7} {'R²':>7} {'PeakErr':>8} {'m':>6} {'ω':>6}"
        )
        print("-" * 85)
        for name, r in episode_results.items():
            if "error" in r:
                print(
                    f"{name:<30} {'BUBBLE' if r['expected_bubble'] else 'CONTROL':>8} ERROR: {r['error'][:30]}"
                )
                continue
            exp = "BUBBLE" if r["expected_bubble"] else "CONTROL"
            bub = "YES" if r["is_bubble"] else "NO"
            pe = f"{r['peak_error_months']}q" if r["peak_error_months"] is not None else "—"
            m = f"{r['m']:.3f}" if r["m"] else "—"
            w = f"{r['omega']:.2f}" if r["omega"] else "—"
            correct = (r["expected_bubble"] and r["is_bubble"]) or (
                not r["expected_bubble"] and not r["is_bubble"]
            )
            print(
                f"{name:<30} {exp:>8} {bub:>7} {r['r_squared']:>7.4f} {pe:>8} {m:>6} {w:>6} {'✓' if correct else '✗':>3}"
            )

        # Summary
        bubbles = {
            n: r
            for n, r in episode_results.items()
            if r.get("expected_bubble") and "error" not in r
        }
        controls = {
            n: r
            for n, r in episode_results.items()
            if not r.get("expected_bubble") and "error" not in r
        }
        tp = sum(1 for r in bubbles.values() if r["is_bubble"])
        fp = sum(1 for r in controls.values() if r["is_bubble"])
        print(f"\nTP={tp}/{len(bubbles)} FP={fp}/{len(controls)}")
        print("=" * 85)
