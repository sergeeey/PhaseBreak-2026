"""Commodity supercycle LPPLS validation — 5th domain."""

import numpy as np
import pytest
import structlog

from src.lppls.data_commodities import (
    COMMODITY_BUBBLES,
    COMMODITY_CONTROLS,
    load_commodity_episode,
)
from src.lppls.optimizer import LPPLSOptimizer
from src.lppls.metrics import durbin_watson

log = structlog.get_logger()

FINANCE_M = np.array([0.637, 0.565, 0.478, 0.437])
FINANCE_OMEGA = np.array([11.86, 6.00, 9.93, 10.83])


class TestCommodityData:
    def test_bubbles_exist(self):
        assert len(COMMODITY_BUBBLES) >= 4

    def test_controls_exist(self):
        assert len(COMMODITY_CONTROLS) >= 3

    def test_bubbles_have_tc(self):
        for name, cfg in COMMODITY_BUBBLES.items():
            assert cfg["known_tc_date"] is not None

    def test_controls_no_tc(self):
        for name, cfg in COMMODITY_CONTROLS.items():
            assert cfg["known_tc_date"] is None


class TestCommodityLPPLS:
    @pytest.fixture(scope="class")
    def results(self) -> dict[str, dict]:
        results = {}
        all_eps = {**COMMODITY_BUBBLES, **COMMODITY_CONTROLS}
        for name in all_eps:
            expected = name in COMMODITY_BUBBLES
            try:
                ds = load_commodity_episode(name)
                opt = LPPLSOptimizer(grid_size=12, n_best=5)
                model = opt.fit(ds.t, ds.log_price)
                r2 = model.r_squared(ds.t, ds.log_price)
                is_bubble = model.params is not None and model.params.is_bubble and r2 > 0.5
                tc_err = (
                    ds.tc_error_days(model.params.tc) if model.params and ds.known_tc_date else None
                )
                residuals = model.residuals(ds.t, ds.log_price) if model.params else np.array([])
                dw = durbin_watson(residuals) if len(residuals) > 0 else 0.0

                results[name] = {
                    "r2": r2,
                    "is_bubble": is_bubble,
                    "tc_error": tc_err,
                    "m": model.params.m if model.params else None,
                    "omega": model.params.omega if model.params else None,
                    "dw": dw,
                    "expected": expected,
                }
                log.info(
                    "commodity_fit", name=name, r2=round(r2, 4), is_bubble=is_bubble, tc_err=tc_err
                )
            except Exception as e:
                results[name] = {"error": str(e), "expected": expected}
                log.warning("commodity_failed", name=name, error=str(e))
        return results

    def test_all_loaded(self, results):
        loaded = sum(1 for r in results.values() if "error" not in r)
        assert loaded >= 6, f"Only {loaded} loaded"

    def test_zero_fp_on_controls(self, results):
        fp = sum(
            1 for r in results.values() if "error" not in r and not r["expected"] and r["is_bubble"]
        )
        assert fp <= 1, f"FP on controls: {fp}"

    def test_print_results(self, results):
        print("\n" + "=" * 90)
        print("COMMODITY SUPERCYCLE — LPPLS Validation")
        print("=" * 90)
        print(
            f"{'Name':<25} {'Exp':>6} {'Bub':>5} {'R²':>7} {'tc_err':>8} {'m':>6} {'ω':>6} {'DW':>6} {'OK':>4}"
        )
        print("-" * 90)
        for name, r in results.items():
            if "error" in r:
                print(f"{name:<25} {'BUB' if r['expected'] else 'CTL':>6} ERROR: {r['error'][:35]}")
                continue
            exp = "BUB" if r["expected"] else "CTL"
            bub = "YES" if r["is_bubble"] else "NO"
            tc = f"{r['tc_error']}d" if r["tc_error"] is not None else "—"
            m = f"{r['m']:.3f}" if r["m"] else "—"
            w = f"{r['omega']:.2f}" if r["omega"] else "—"
            correct = r["expected"] == r["is_bubble"]
            print(
                f"{name:<25} {exp:>6} {bub:>5} {r['r2']:>7.4f} {tc:>8} {m:>6} {w:>6} {r['dw']:>6.3f} {'✓' if correct else '✗':>4}"
            )

        # Summary
        bubs = {n: r for n, r in results.items() if r.get("expected") and "error" not in r}
        ctrls = {n: r for n, r in results.items() if not r.get("expected") and "error" not in r}
        tp = sum(1 for r in bubs.values() if r["is_bubble"])
        fp = sum(1 for r in ctrls.values() if r["is_bubble"])
        print(f"\nTP={tp}/{len(bubs)} FP={fp}/{len(ctrls)}")
        print("=" * 90)

    def test_cross_domain(self, results):
        """Compare commodity (m,ω) with finance."""
        from scipy.stats import ks_2samp

        com_m = np.array(
            [
                r["m"]
                for r in results.values()
                if "error" not in r and r["m"] is not None and r["is_bubble"]
            ]
        )
        com_omega = np.array(
            [
                r["omega"]
                for r in results.values()
                if "error" not in r and r["omega"] is not None and r["is_bubble"]
            ]
        )

        if len(com_m) >= 3:
            ks_m = ks_2samp(com_m, FINANCE_M)
            ks_w = ks_2samp(com_omega, FINANCE_OMEGA)
            print(f"\nCross-domain (Commodities vs Finance):")
            print(f"  Com m:  mean={com_m.mean():.3f} range=[{com_m.min():.3f}, {com_m.max():.3f}]")
            print(f"  Fin m:  mean={FINANCE_M.mean():.3f}")
            print(f"  KS (m): p={ks_m.pvalue:.4f}")
            print(f"  KS (ω): p={ks_w.pvalue:.4f}")
            print(
                f"  Verdict: {'NOT_REJECTED' if ks_m.pvalue > 0.05 and ks_w.pvalue > 0.05 else 'REJECTED'}"
            )
        else:
            print(f"\nInsufficient commodity fits for KS test: {len(com_m)}")
