"""Cox Proportional Hazards with Doomsday-enhanced features for fraud survival.

Compares:
1. Cox baseline (velocity, amount_std, n_accounts, acceleration)
2. Cox + Doomsday (baseline + doomsday_percentile, remaining_frac, log_remaining)

If C-index improves with Doomsday features → publishable contribution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import structlog

from src.survival.doomsday import fit_weibull_prior, batch_doomsday_features

log = structlog.get_logger()

BASELINE_FEATURES = ["velocity", "amount_std", "n_accounts", "acceleration"]
DOOMSDAY_FEATURES = ["doomsday_pct", "remaining_frac", "log_remaining"]


@dataclass
class SurvivalResult:
    """Result of Cox model comparison."""

    c_index_baseline: float  # concordance without Doomsday
    c_index_doomsday: float  # concordance with Doomsday features
    improvement: float  # absolute improvement in C-index
    improvement_pct: float  # percentage improvement
    n_samples: int
    weibull_shape: float
    weibull_scale: float
    baseline_coefficients: dict[str, float]
    doomsday_coefficients: dict[str, float]


def add_doomsday_features(
    df: pd.DataFrame,
    weibull_shape: float,
    weibull_scale: float,
) -> pd.DataFrame:
    """Add Doomsday-derived features to fraud dataset.

    Requires 'n_observed' column (from generate_observed_snapshot).
    """
    df = df.copy()
    n_obs = df["n_observed"].values
    features = batch_doomsday_features(n_obs, weibull_shape, weibull_scale)
    df["doomsday_pct"] = features[:, 0]
    df["remaining_frac"] = features[:, 1]
    df["log_remaining"] = features[:, 2]
    return df


def fit_and_compare(
    df: pd.DataFrame,
    duration_col: str = "lifetime",
    event_col: str = "detected",
) -> SurvivalResult:
    """Fit baseline and Doomsday-enhanced Cox models, compare C-index.

    Args:
        df: DataFrame with baseline features + Doomsday features
        duration_col: Column with survival time
        event_col: Column with event indicator (1=detected)

    Returns:
        SurvivalResult with comparison metrics
    """
    from lifelines import CoxPHFitter

    # WHY: fit Weibull prior on first 50% (train split), evaluate on full data.
    # Prevents look-ahead bias where prior is informed by test cases.
    n_train = len(df) // 2
    train_lifetimes = df[duration_col].values[:n_train]
    train_events = df[event_col].values[:n_train]
    shape, scale = fit_weibull_prior(train_lifetimes, train_events)

    # Ensure Doomsday features exist
    if "doomsday_pct" not in df.columns:
        df = add_doomsday_features(df, shape, scale)

    # Baseline Cox model
    baseline_cols = [c for c in BASELINE_FEATURES if c in df.columns]
    df_baseline = df[baseline_cols + [duration_col, event_col]].dropna()

    cox_baseline = CoxPHFitter(penalizer=0.01)
    cox_baseline.fit(df_baseline, duration_col=duration_col, event_col=event_col)
    c_baseline = cox_baseline.concordance_index_

    baseline_coefs = {col: round(float(cox_baseline.params_[col]), 4) for col in baseline_cols}

    # Doomsday-enhanced Cox model
    all_cols = baseline_cols + [c for c in DOOMSDAY_FEATURES if c in df.columns]
    df_doomsday = df[all_cols + [duration_col, event_col]].dropna()

    cox_doomsday = CoxPHFitter(penalizer=0.01)
    cox_doomsday.fit(df_doomsday, duration_col=duration_col, event_col=event_col)
    c_doomsday = cox_doomsday.concordance_index_

    doomsday_coefs = {col: round(float(cox_doomsday.params_[col]), 4) for col in all_cols}

    improvement = c_doomsday - c_baseline
    improvement_pct = (improvement / c_baseline) * 100 if c_baseline > 0 else 0

    log.info(
        "cox_comparison",
        c_baseline=round(c_baseline, 4),
        c_doomsday=round(c_doomsday, 4),
        improvement=round(improvement, 4),
        improvement_pct=round(improvement_pct, 2),
    )

    return SurvivalResult(
        c_index_baseline=c_baseline,
        c_index_doomsday=c_doomsday,
        improvement=improvement,
        improvement_pct=improvement_pct,
        n_samples=len(df_doomsday),
        weibull_shape=shape,
        weibull_scale=scale,
        baseline_coefficients=baseline_coefs,
        doomsday_coefficients=doomsday_coefs,
    )
