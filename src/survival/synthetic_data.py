"""Synthetic fraud timeline generator for survival analysis.

Generates realistic fraud scheme lifetimes with:
- Transaction velocity (accelerating before detection)
- Amount distribution (increasing variance over time)
- Detection events (censored vs uncensored)
- Multiple fraud types with different survival characteristics

Used when real KZ fraud data is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import structlog

log = structlog.get_logger()


@dataclass
class FraudDatasetConfig:
    """Configuration for synthetic fraud data generation."""

    n_schemes: int = 500
    seed: int = 42
    # Weibull shape/scale for fraud lifetimes (transactions until detection)
    weibull_shape: float = 1.5  # >1 = increasing hazard (detection gets easier over time)
    weibull_scale: float = 200.0  # median ~165 transactions
    # Censoring rate (fraction of schemes still active / not detected)
    censoring_rate: float = 0.2
    # Feature noise
    noise_level: float = 0.1


def generate_fraud_timelines(config: FraudDatasetConfig | None = None) -> pd.DataFrame:
    """Generate synthetic fraud scheme dataset for survival analysis.

    Each row = one fraud scheme with:
    - lifetime: total transactions until detection (or censoring)
    - detected: 1 = scheme was detected, 0 = still active (censored)
    - velocity: mean transactions per day
    - amount_std: standard deviation of transaction amounts
    - n_accounts: number of accounts involved
    - acceleration: rate of velocity increase over lifetime
    - scheme_type: categorical (0=phishing, 1=card_fraud, 2=money_laundering)

    Returns:
        DataFrame with columns for Cox/Weibull analysis
    """
    cfg = config or FraudDatasetConfig()
    rng = np.random.default_rng(cfg.seed)

    n = cfg.n_schemes

    # WHY Weibull: real fraud detection has increasing hazard
    # (more transactions → more evidence → easier to detect)
    true_lifetimes = rng.weibull(cfg.weibull_shape, size=n) * cfg.weibull_scale

    # Censoring: some schemes still active
    n_censored = int(n * cfg.censoring_rate)
    is_detected = np.ones(n, dtype=int)
    censored_idx = rng.choice(n, size=n_censored, replace=False)
    # Censored schemes: observed time < true lifetime
    censor_times = rng.uniform(0.3, 0.8, size=n_censored) * true_lifetimes[censored_idx]
    observed_lifetimes = true_lifetimes.copy()
    observed_lifetimes[censored_idx] = censor_times
    is_detected[censored_idx] = 0

    # Ensure positive lifetimes
    observed_lifetimes = np.clip(observed_lifetimes, 1.0, None)

    # Features correlated with lifetime
    # velocity: higher velocity → shorter lifetime (more visible)
    velocity = 5.0 + rng.exponential(3.0, size=n)
    velocity += -0.005 * true_lifetimes + rng.normal(0, cfg.noise_level * 2, size=n)
    velocity = np.clip(velocity, 0.5, 50.0)

    # amount_std: higher variance → more suspicious → shorter lifetime
    amount_std = 100 + rng.exponential(200, size=n)
    amount_std += -0.2 * true_lifetimes + rng.normal(0, cfg.noise_level * 50, size=n)
    amount_std = np.clip(amount_std, 10.0, 2000.0)

    # n_accounts: more accounts → more complex → longer lifetime initially
    n_accounts = rng.poisson(5, size=n) + 1
    n_accounts = np.clip(n_accounts, 1, 50)

    # acceleration: how fast velocity increases (early warning signal)
    acceleration = rng.normal(0.02, 0.01, size=n)
    # Short-lived schemes have higher acceleration (desperation)
    acceleration += -0.00005 * true_lifetimes
    acceleration = np.clip(acceleration, -0.05, 0.15)

    # scheme_type: 0=phishing (short), 1=card_fraud (medium), 2=money_laundering (long)
    scheme_type = np.zeros(n, dtype=int)
    scheme_type[true_lifetimes > np.percentile(true_lifetimes, 33)] = 1
    scheme_type[true_lifetimes > np.percentile(true_lifetimes, 66)] = 2
    # Add noise to type assignment
    flip_mask = rng.random(n) < 0.15
    scheme_type[flip_mask] = rng.integers(0, 3, size=flip_mask.sum())

    df = pd.DataFrame(
        {
            "lifetime": np.round(observed_lifetimes, 1),
            "detected": is_detected,
            "velocity": np.round(velocity, 3),
            "amount_std": np.round(amount_std, 2),
            "n_accounts": n_accounts,
            "acceleration": np.round(acceleration, 5),
            "scheme_type": scheme_type,
        }
    )

    log.info(
        "synthetic_fraud_data",
        n_total=n,
        n_detected=int(is_detected.sum()),
        n_censored=n_censored,
        median_lifetime=round(float(np.median(observed_lifetimes)), 1),
    )

    return df


def generate_observed_snapshot(
    df: pd.DataFrame, observation_fraction: float = 0.5, seed: int = 42
) -> pd.DataFrame:
    """Simulate observing schemes at a random point in their lifetime.

    For Doomsday analysis: we need n_observed (current transaction count)
    for each active scheme.

    Args:
        df: Full fraud dataset
        observation_fraction: How far into lifetime we observe (0-1)
        seed: Random seed

    Returns:
        DataFrame with additional column 'n_observed'
    """
    rng = np.random.default_rng(seed)
    df = df.copy()

    # WHY: n_observed must NOT be derived directly from lifetime (information leakage).
    # Instead, generate independently from velocity (observable covariate) + noise.
    # This simulates: "we see N transactions so far" based on observable behavior.
    velocity = df["velocity"].values
    time_window = rng.uniform(10, 50, size=len(df))  # observation window in days
    n_observed = np.round(velocity * time_window * rng.uniform(0.5, 1.5, size=len(df)))
    n_observed = np.clip(n_observed, 1, df["lifetime"].values * 0.9).astype(int)
    df["n_observed"] = np.maximum(n_observed, 1)

    # Observation fraction (how far along are we?)
    df["obs_fraction"] = df["n_observed"] / df["lifetime"]

    return df
