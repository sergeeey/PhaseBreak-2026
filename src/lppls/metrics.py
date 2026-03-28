"""Residual analysis and model selection metrics for LPPLS.

Provides AIC, BIC, Durbin-Watson, and residual diagnostics
to evaluate quality of LPPLS fits beyond R².
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def aic(n: int, k: int, sse: float) -> float:
    """Akaike Information Criterion.

    AIC = n * ln(SSE/n) + 2k
    Lower = better. Penalizes model complexity.

    Args:
        n: number of observations
        k: number of parameters (7 for LPPLS)
        sse: sum of squared errors
    """
    if sse <= 0 or n <= 0:
        return float("inf")
    return n * np.log(sse / n) + 2 * k


def bic(n: int, k: int, sse: float) -> float:
    """Bayesian Information Criterion.

    BIC = n * ln(SSE/n) + k * ln(n)
    Stricter penalty than AIC for large n.
    """
    if sse <= 0 or n <= 0:
        return float("inf")
    return n * np.log(sse / n) + k * np.log(n)


def durbin_watson(residuals: NDArray) -> float:
    """Durbin-Watson statistic for autocorrelation in residuals.

    DW ≈ 2: no autocorrelation (good)
    DW < 1.5: positive autocorrelation (bad — LPPLS missed a pattern)
    DW > 2.5: negative autocorrelation (rare for LPPLS)
    """
    diff = np.diff(residuals)
    ss_diff = float(np.sum(diff**2))
    ss_res = float(np.sum(residuals**2))
    if ss_res < 1e-15:
        return 2.0  # perfect fit
    return ss_diff / ss_res


def residual_normality(residuals: NDArray) -> dict[str, float]:
    """Basic normality diagnostics for residuals.

    Returns skewness and kurtosis. For good LPPLS fit:
    - |skewness| < 1.0
    - |kurtosis - 3| < 2.0 (excess kurtosis near 0)
    """
    n = len(residuals)
    if n < 4:
        return {"skewness": 0.0, "kurtosis": 3.0}
    mean = np.mean(residuals)
    std = np.std(residuals, ddof=1)
    if std < 1e-15:
        return {"skewness": 0.0, "kurtosis": 3.0}
    z = (residuals - mean) / std
    skew = float(np.mean(z**3))
    kurt = float(np.mean(z**4))
    return {"skewness": skew, "kurtosis": kurt}
