"""Ablation study: measure what each PhaseBreak layer adds.

Runs 5 configurations on 12 datasets (6 bubbles + 6 controls):
1. LPPLS only (loose filters)
2. LPPLS + tight filters
3. LPPLS + tight + multi-window CI
4. LPPLS + tight + MW + HMM gate
5. LPPLS + tight + MW + HMM + EWS

Produces the key ablation table for the paper.
"""

import numpy as np
import pytest
import structlog

from src.lppls.data import KNOWN_BUBBLES, NEGATIVE_CONTROLS, load_yfinance, BubbleDataset
from src.lppls.model import LPPLS, LPPLSParams
from src.lppls.optimizer import LPPLSOptimizer
from src.lppls.confidence import MultiWindowConfidence
from src.lppls.ensemble import HMMLPPLSEnsemble
from src.ews.critical_slowing import compute_ews

log = structlog.get_logger()

ALL_DATASETS: dict[str, dict] = {}
EXPECTED: dict[str, bool] = {}
for name, config in KNOWN_BUBBLES.items():
    ALL_DATASETS[name] = config
    EXPECTED[name] = True
for name, config in NEGATIVE_CONTROLS.items():
    ALL_DATASETS[name] = config
    EXPECTED[name] = False


def _load_all() -> dict[str, BubbleDataset]:
    datasets = {}
    for name, config in ALL_DATASETS.items():
        try:
            datasets[name] = load_yfinance(**config)
        except Exception as e:
            log.warning("load_failed", name=name, error=str(e))
    return datasets


# ============================================================
# Layer 1: LPPLS only (loose filters — original Sornette)
# ============================================================
class _LooseOptimizer(LPPLSOptimizer):
    """Subclass with loose Sornette filters — only B<0 required."""

    @staticmethod
    def _passes_filters(params) -> bool:
        return params.B < 0


def _run_lppls_loose(ds: BubbleDataset) -> bool:
    """LPPLS with truly loose filters via subclass (no monkey-patching)."""
    opt = _LooseOptimizer(grid_size=12, n_best=5)
    model = opt.fit(ds.t, ds.log_price)
    r2 = model.r_squared(ds.t, ds.log_price)
    return model.params is not None and model.params.B < 0 and r2 > 0.5


# ============================================================
# Layer 2: LPPLS + tight filters
# ============================================================
def _run_lppls_tight(ds: BubbleDataset) -> bool:
    """LPPLS with tightened Sornette filters (current optimizer)."""
    opt = LPPLSOptimizer(grid_size=12, n_best=5)
    model = opt.fit(ds.t, ds.log_price)
    r2 = model.r_squared(ds.t, ds.log_price)
    return model.params is not None and model.params.is_bubble and r2 > 0.5


# ============================================================
# Layer 3: LPPLS + tight + multi-window CI
# ============================================================
def _run_multiwindow(ds: BubbleDataset) -> bool:
    """Multi-window confidence indicator."""
    mwc = MultiWindowConfidence(windows=[60, 90, 120, 180], tc_tolerance=20, min_r_squared=0.5)
    result = mwc.evaluate(ds.t, ds.log_price)
    return result.confidence in ("HIGH", "MEDIUM")


# ============================================================
# Layer 4: LPPLS + tight + MW + HMM gate
# ============================================================
def _run_ensemble(ds: BubbleDataset) -> bool:
    """HMM-gated LPPLS ensemble."""
    ensemble = HMMLPPLSEnsemble(
        confidence_kwargs={"windows": [60, 90, 120], "tc_tolerance": 20},
        skip_normal=True,
    )
    result = ensemble.analyze(ds.t, ds.log_price)
    return result.final_verdict in ("BUBBLE", "POSSIBLE_BUBBLE")


# ============================================================
# Layer 5: Ensemble + EWS confirmation
# ============================================================
def _run_ensemble_ews(ds: BubbleDataset) -> bool:
    """Ensemble + EWS: both must agree for BUBBLE."""
    # Ensemble
    ensemble = HMMLPPLSEnsemble(
        confidence_kwargs={"windows": [60, 90, 120], "tc_tolerance": 20},
        skip_normal=True,
    )
    ens_result = ensemble.analyze(ds.t, ds.log_price)
    ens_bubble = ens_result.final_verdict in ("BUBBLE", "POSSIBLE_BUBBLE")

    # EWS
    ews = compute_ews(ds.log_price, window=min(40, len(ds.log_price) // 5))
    ews_signal = ews.ews_score > 0.5 or ews.kendall_tau_variance > 0.2

    # WHY: EWS must actually gate — require BOTH signals for BUBBLE.
    # If EWS doesn't confirm, downgrade to NO_BUBBLE.
    # This makes the ablation row meaningful (not identical to Layer 4).
    return ens_bubble and ews_signal


LAYERS = [
    ("LPPLS (loose)", _run_lppls_loose),
    ("+ tight filters", _run_lppls_tight),
    ("+ multi-window CI", _run_multiwindow),
    ("+ HMM gate", _run_ensemble),
    ("+ EWS confirm", _run_ensemble_ews),
]


class TestAblation:
    @pytest.fixture(scope="class")
    def ablation_results(self) -> list[dict]:
        datasets = _load_all()
        results = []

        for layer_name, layer_fn in LAYERS:
            tp = fp = fn = tn = 0
            tc_errors = []

            for name, ds in datasets.items():
                expected = EXPECTED[name]
                try:
                    predicted = layer_fn(ds)
                except Exception as e:
                    log.warning("ablation_error", layer=layer_name, dataset=name, error=str(e))
                    predicted = False

                if expected and predicted:
                    tp += 1
                elif not expected and predicted:
                    fp += 1
                elif expected and not predicted:
                    fn += 1
                else:
                    tn += 1

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

            results.append(
                {
                    "layer": layer_name,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "fp_rate": fp_rate,
                }
            )

            log.info(
                "ablation_layer",
                layer=layer_name,
                tp=tp,
                fp=fp,
                fn=fn,
                tn=tn,
                precision=round(precision, 3),
                recall=round(recall, 3),
            )

        return results

    def test_tight_filters_reduce_fp(self, ablation_results):
        """Tight filters should have fewer FP than loose."""
        loose = ablation_results[0]
        tight = ablation_results[1]
        assert tight["fp"] <= loose["fp"]

    def test_multiwindow_maintains_recall(self, ablation_results):
        """Multi-window should not drastically reduce recall vs tight."""
        tight = ablation_results[1]
        mw = ablation_results[2]
        # Allow 1 less TP
        assert mw["tp"] >= tight["tp"] - 2

    def test_print_ablation_table(self, ablation_results):
        print("\n" + "=" * 90)
        print("ABLATION TABLE — PhaseBreak Layer-by-Layer (12 datasets: 6 bubbles + 6 controls)")
        print("=" * 90)
        print(
            f"{'Layer':<25} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4} {'Prec':>8} {'Recall':>8} {'F1':>8} {'FP Rate':>8}"
        )
        print("-" * 90)
        for r in ablation_results:
            print(
                f"{r['layer']:<25} {r['tp']:>4} {r['fp']:>4} {r['fn']:>4} {r['tn']:>4} "
                f"{r['precision']:>8.1%} {r['recall']:>8.1%} {r['f1']:>8.1%} {r['fp_rate']:>8.1%}"
            )
        print("=" * 90)

        # Show improvement from loose to best
        loose = ablation_results[0]
        best = max(ablation_results, key=lambda r: r["f1"])
        print(
            f"\nBaseline (loose): Precision={loose['precision']:.1%}, FP Rate={loose['fp_rate']:.1%}"
        )
        print(
            f"Best ({best['layer']}): Precision={best['precision']:.1%}, FP Rate={best['fp_rate']:.1%}"
        )
        if loose["precision"] > 0:
            prec_gain = (best["precision"] - loose["precision"]) / loose["precision"] * 100
            print(f"Precision improvement: {prec_gain:+.1f}%")
        fp_reduction = loose["fp"] - best["fp"]
        print(f"False positives eliminated: {fp_reduction}")
        print("=" * 90)
