"""Doomsday Argument applied to fraud scheme lifetime prediction.

The Doomsday Argument (Gott 1993):
If you observe a process at random point n out of total N,
then P(n | N) = 1/N (random observer assumption).

Applied to fraud: if we see a fraud scheme at transaction n,
how many total transactions N before detection?

P(N | n) ∝ P(n | N) * P(N) = (1/N) * Prior(N)

Prior = Weibull fitted on historical fraud lifetimes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import structlog
from numpy.typing import NDArray

log = structlog.get_logger()


@dataclass(frozen=True)
class DoomsdayPrediction:
    """Prediction of remaining fraud scheme lifetime."""

    n_observed: int  # current transaction count
    n_median: float  # median predicted total N
    n_lower: float  # 5th percentile (optimistic — scheme ends soon)
    n_upper: float  # 95th percentile (pessimistic — scheme lasts long)
    remaining_median: float  # median remaining transactions
    remaining_lower: float  # 5th percentile remaining
    remaining_upper: float  # 95th percentile remaining
    doomsday_percentile: float  # where n_observed falls in predicted N distribution


def fit_weibull_prior(lifetimes: NDArray, detected: NDArray | None = None) -> tuple[float, float]:
    """Fit Weibull distribution to historical fraud lifetimes.

    Args:
        lifetimes: Array of scheme lifetimes (transactions)
        detected: Boolean array (1=detected, 0=censored). If None, all detected.

    Returns:
        (shape, scale) parameters for Weibull distribution
    """
    from lifelines import WeibullFitter

    wf = WeibullFitter()
    if detected is not None:
        wf.fit(lifetimes, event_observed=detected)
    else:
        wf.fit(lifetimes)

    # lifelines uses lambda_ (scale) and rho_ (shape)
    shape = float(wf.rho_)
    scale = float(wf.lambda_)

    log.info("weibull_prior_fitted", shape=round(shape, 4), scale=round(scale, 2))
    return shape, scale


def weibull_pdf(n: NDArray, shape: float, scale: float) -> NDArray:
    """Weibull PDF: f(n) = (shape/scale) * (n/scale)^(shape-1) * exp(-(n/scale)^shape)."""
    x = n / scale
    return (shape / scale) * np.power(x, shape - 1) * np.exp(-np.power(x, shape))


def doomsday_posterior(
    n_observed: int,
    shape: float,
    scale: float,
    n_max_multiplier: int = 50,
) -> tuple[NDArray, NDArray]:
    """Compute Doomsday posterior P(N | n_observed).

    P(N | n) ∝ (1/N) * Weibull(N; shape, scale)

    Args:
        n_observed: Current transaction count
        shape: Weibull shape parameter
        scale: Weibull scale parameter
        n_max_multiplier: Search up to n_observed * multiplier

    Returns:
        (N_range, posterior) — normalized probability distribution
    """
    n_max = max(n_observed * n_max_multiplier, int(scale * 5))
    N_range = np.arange(n_observed, n_max + 1, dtype=float)

    # Likelihood: random observer assumption
    likelihood = 1.0 / N_range

    # Prior: Weibull
    prior = weibull_pdf(N_range, shape, scale)

    # Posterior ∝ likelihood * prior
    posterior = likelihood * prior
    total = posterior.sum()
    if total > 0:
        posterior = posterior / total
    else:
        posterior = np.ones_like(N_range) / len(N_range)

    return N_range, posterior


def predict_remaining(
    n_observed: int,
    shape: float,
    scale: float,
) -> DoomsdayPrediction:
    """Predict remaining lifetime of a fraud scheme using Doomsday logic.

    Args:
        n_observed: Current transaction count
        shape: Weibull shape parameter
        scale: Weibull scale parameter

    Returns:
        DoomsdayPrediction with median and credible interval
    """
    N_range, posterior = doomsday_posterior(n_observed, shape, scale)

    # CDF for percentile computation
    cdf = np.cumsum(posterior)

    # Find percentiles
    def _percentile(p: float) -> float:
        idx = np.searchsorted(cdf, p)
        idx = min(idx, len(N_range) - 1)
        return float(N_range[idx])

    n_median = _percentile(0.5)
    n_lower = _percentile(0.05)
    n_upper = _percentile(0.95)

    # Doomsday percentile: where does n_observed fall?
    idx_obs = 0  # n_observed is the first element of N_range
    doomsday_pct = float(cdf[idx_obs]) if len(cdf) > 0 else 0.0

    return DoomsdayPrediction(
        n_observed=n_observed,
        n_median=n_median,
        n_lower=n_lower,
        n_upper=n_upper,
        remaining_median=max(0, n_median - n_observed),
        remaining_lower=max(0, n_lower - n_observed),
        remaining_upper=max(0, n_upper - n_observed),
        doomsday_percentile=doomsday_pct,
    )


def batch_doomsday_features(n_observed_arr: NDArray, shape: float, scale: float) -> NDArray:
    """Compute Doomsday features for a batch of observations.

    Returns array of shape (n, 3): [doomsday_percentile, remaining_median_frac, log_remaining]
    Designed as features for Cox model.
    """
    features = np.zeros((len(n_observed_arr), 3))
    for i, n_obs in enumerate(n_observed_arr):
        pred = predict_remaining(int(n_obs), shape, scale)
        features[i, 0] = pred.doomsday_percentile
        features[i, 1] = pred.remaining_median / max(n_obs, 1)  # fraction of observed
        features[i, 2] = np.log1p(pred.remaining_median)  # log remaining
    return features
