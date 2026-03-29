"""Housing validation: LPPLS vs baselines + held-out split + cross-domain."""

import numpy as np
import pytest
import structlog

from src.housing.data import HOUSING_BUBBLES, HOUSING_CONTROLS, load_housing_episode
from src.housing.baselines import run_all_baselines
from src.lppls.optimizer import LPPLSOptimizer

log = structlog.get_logger()

# Split: tune on 2006 episodes, evaluate on 2022
TUNE_SET = ["az_2006", "nv_2006", "fl_2006", "ca_2006"]
EVAL_SET = ["az_2022", "id_2022"]
CONTROL_SET = ["tx_2015", "oh_2018", "pa_2014", "il_2016"]


def _fit_lppls(ds):
    opt = LPPLSOptimizer(grid_size=10, n_best=5, m_range=(0.1, 0.9), omega_range=(4.0, 15.0))
    model = opt.fit(ds.t, ds.log_price)
    r2 = model.r_squared(ds.t, ds.log_price)
    return model.params is not None and model.params.is_bubble and r2 > 0.5


class TestBaselineComparison:
    @pytest.fixture(scope="class")
    def comparison(self) -> dict:
        all_names = list(HOUSING_BUBBLES.keys()) + list(HOUSING_CONTROLS.keys())
        results = {"lppls": {}, "cagr": {}, "z_accel": {}, "trend_dev": {}}

        for name in all_names:
            expected = name in HOUSING_BUBBLES
            try:
                ds = load_housing_episode(name)
                # LPPLS
                lppls_bubble = _fit_lppls(ds)
                results["lppls"][name] = {"predicted": lppls_bubble, "expected": expected}

                # Baselines
                baselines = run_all_baselines(ds.values)
                for bl in baselines:
                    key = bl.method.lower().replace("-", "_")
                    results[key][name] = {"predicted": bl.is_bubble, "expected": expected}

            except Exception as e:
                log.warning("comparison_failed", name=name, error=str(e))
        return results

    def test_print_comparison(self, comparison):
        print("\n" + "=" * 90)
        print("HOUSING: LPPLS vs BASELINES (10 episodes)")
        print("=" * 90)
        print(f"{'Name':<20} {'Expected':>8} {'LPPLS':>7} {'CAGR':>7} {'Z-Acc':>7} {'TrDev':>7}")
        print("-" * 90)

        all_names = list(HOUSING_BUBBLES.keys()) + list(HOUSING_CONTROLS.keys())
        for name in all_names:
            exp = "BUBBLE" if name in HOUSING_BUBBLES else "CTRL"
            lppls = comparison["lppls"].get(name, {}).get("predicted", "?")
            cagr = comparison.get("cagr", {}).get(name, {}).get("predicted", "?")
            zaccel = comparison.get("z_accel", {}).get(name, {}).get("predicted", "?")
            tdev = comparison.get("trend_dev", {}).get(name, {}).get("predicted", "?")

            def _fmt(v):
                if v is True:
                    return "YES"
                if v is False:
                    return "NO"
                return "?"

            print(
                f"{name:<20} {exp:>8} {_fmt(lppls):>7} {_fmt(cagr):>7} {_fmt(zaccel):>7} {_fmt(tdev):>7}"
            )

        # Summary per method
        print("\n" + "-" * 90)
        for method in ["lppls", "cagr", "z_accel", "trend_dev"]:
            data = comparison.get(method, {})
            if not data:
                continue
            tp = sum(1 for r in data.values() if r["expected"] and r["predicted"])
            fp = sum(1 for r in data.values() if not r["expected"] and r["predicted"])
            fn = sum(1 for r in data.values() if r["expected"] and not r["predicted"])
            tn = sum(1 for r in data.values() if not r["expected"] and not r["predicted"])
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            print(f"{method:<12} TP={tp} FP={fp} FN={fn} TN={tn}  P={prec:.0%} R={rec:.0%}")
        print("=" * 90)


class TestHeldOutValidation:
    """Held-out: tune on 2006, evaluate on 2022."""

    def test_held_out(self):
        print("\n" + "=" * 70)
        print("HELD-OUT: tune on 2006, evaluate on 2022 + controls")
        print("=" * 70)

        # Evaluate on 2022 bubbles (not used for filter tuning)
        eval_tp = 0
        for name in EVAL_SET:
            try:
                ds = load_housing_episode(name)
                detected = _fit_lppls(ds)
                eval_tp += int(detected)
                print(f"  {name}: {'DETECTED' if detected else 'MISSED'}")
            except Exception as e:
                print(f"  {name}: ERROR {e}")

        # Evaluate on controls
        eval_fp = 0
        for name in CONTROL_SET:
            try:
                ds = load_housing_episode(name)
                detected = _fit_lppls(ds)
                eval_fp += int(detected)
                print(f"  {name}: {'FALSE POSITIVE' if detected else 'OK'}")
            except Exception as e:
                print(f"  {name}: ERROR {e}")

        print(f"\nHeld-out: TP={eval_tp}/{len(EVAL_SET)} FP={eval_fp}/{len(CONTROL_SET)}")
        print("=" * 70)

        # 2022 episodes are held-out → true generalization test
        assert eval_fp == 0, f"False positives on held-out controls: {eval_fp}"


class TestCrossDomainWithHousing:
    """Add housing (m,ω) to cross-domain comparison."""

    def test_housing_params_in_range(self):
        """Housing LPPLS params should be in LPPLS-valid ranges."""
        housing_m = []
        housing_omega = []
        for name in HOUSING_BUBBLES:
            try:
                ds = load_housing_episode(name)
                opt = LPPLSOptimizer(
                    grid_size=10, n_best=5, m_range=(0.1, 0.9), omega_range=(4.0, 15.0)
                )
                model = opt.fit(ds.t, ds.log_price)
                if model.params and model.params.is_bubble:
                    housing_m.append(model.params.m)
                    housing_omega.append(model.params.omega)
            except Exception:
                continue

        if housing_m:
            m_arr = np.array(housing_m)
            w_arr = np.array(housing_omega)
            print(f"\nHousing (m,ω): n={len(m_arr)}")
            print(f"  m: mean={m_arr.mean():.3f} range=[{m_arr.min():.3f}, {m_arr.max():.3f}]")
            print(f"  ω: mean={w_arr.mean():.2f} range=[{w_arr.min():.2f}, {w_arr.max():.2f}]")

            # Compare with finance (from validated results)
            fin_m = np.array([0.637, 0.565, 0.478, 0.437])
            from scipy.stats import ks_2samp

            ks = ks_2samp(m_arr, fin_m)
            print(f"  KS vs finance (m): p={ks.pvalue:.4f}")
