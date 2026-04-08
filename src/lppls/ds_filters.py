"""DS-LPPLS (Double-Scale LPPLS) — Sornette filters and confidence indicator.

Implements strict physical constraints on LPPLS parameters to eliminate
mathematical noise and false positives.

References:
- Sornette et al. (2015) "DS LPPLS Confidence Indicator"
- Filimonov & Sornette (2013) "A stable and robust calibration scheme"
"""

from __future__ import annotations

import numpy as np
import structlog

from src.lppls.model import LPPLS, LPPLSParams
from src.lppls.optimizer import LPPLSOptimizer

log = structlog.get_logger()


def _log_prices(prices: np.ndarray) -> np.ndarray:
    """Natural log of prices; LPPLS is defined on ln(p(t)), not p(t)."""
    return np.log(np.clip(np.asarray(prices, dtype=float), 1e-300, None))


def _make_optimizer(
    t: np.ndarray,
    *,
    domain: str,
    grid_size: int,
    n_best: int,
    tc_horizon: float,
) -> LPPLSOptimizer:
    """Grid + L-BFGS-B optimizer on ln(price), domain-specific ω search box."""
    t_last = float(t[-1])
    tc_lo = t_last + 1.0
    tc_hi = t_last + float(tc_horizon)
    if domain == "housing":
        omega_range = (4.0, 15.0)
        filter_omega_interior = (4.05, 14.95)
    else:
        omega_range = (6.0, 13.0)
        filter_omega_interior = (5.0, 13.5)
    return LPPLSOptimizer(
        tc_range=(tc_lo, tc_hi),
        m_range=(0.1, 0.9),
        omega_range=omega_range,
        grid_size=grid_size,
        n_best=n_best,
        filter_m_interior=(0.1, 0.87),
        filter_omega_interior=filter_omega_interior,
    )


def _fit_lppls_log_price(
    t: np.ndarray,
    prices: np.ndarray,
    *,
    domain: str = "finance",
    grid_size: int = 10,
    n_best: int = 5,
    tc_horizon: float = 180.0,
) -> tuple[LPPLS | None, float]:
    """Fit LPPLS on ln(price). Returns (model, r²) or (None, 0)."""
    t = np.asarray(t, dtype=float)
    if len(prices) < 30:
        return None, 0.0
    log_price = _log_prices(prices)
    opt = _make_optimizer(
        t, domain=domain, grid_size=grid_size, n_best=n_best, tc_horizon=tc_horizon
    )
    try:
        model = opt.fit(t, log_price)
    except Exception:
        return None, 0.0
    r2 = model.r_squared(t, log_price)
    return model, r2


def _quality_score_lppls(p: LPPLSParams, r_squared: float, domain: str) -> float:
    """Heuristic 0–1 score (legacy structure, ln-price R²)."""
    quality = 0.0
    if r_squared > 0.8:
        quality += 0.3
    elif r_squared > 0.7:
        quality += 0.15
    if p.B < 0:
        quality += 0.2
    if 0.2 < p.m < 0.7:
        quality += 0.2
    elif 0.1 < p.m < 0.9:
        quality += 0.1
    if domain == "housing":
        if 6.0 < p.omega < 13.0:
            quality += 0.15
        elif 4.0 < p.omega < 15.0:
            quality += 0.08
    elif 6.0 < p.omega < 13.0:
        quality += 0.15
    damp = p.damping
    if damp > 1.0:
        quality += 0.15
    elif damp > 0.5:
        quality += 0.05
    return float(min(quality, 1.0))


def _dict_for_ds_window(model: LPPLS, t: np.ndarray, prices: np.ndarray, domain: str) -> dict | None:
    """One-window result for calculate_ds_lppls_confidence."""
    p = model.params
    if p is None:
        return None
    log_price = _log_prices(prices)
    r2 = model.r_squared(t, log_price)
    q = _quality_score_lppls(p, r2, domain)
    verdict = "BUBBLE" if q > 0.5 else "NO_BUBBLE"
    return {
        "tc": float(p.tc),
        "m": float(p.m),
        "omega": float(p.omega),
        "a": float(p.A),
        "b": float(p.B),
        "c": float(p.C),
        "r_squared": float(r2),
        "quality_score": q,
        "verdict": verdict,
    }


def passes_sornette_filters(
    tc: float,
    m: float,
    omega: float,
    a: float,
    b: float,
    c: float,
    t_start: float,
    t_end: float,
    domain: str = "finance",
) -> bool:
    """Strict Sornette filters for LPPLS parameter validation.

    Args:
        tc: Critical time (index)
        m: Exponent (0.1-0.9)
        omega: Log-periodic frequency
        a: Linear parameter A
        b: Nonlinear parameter B (must be < 0 for bubble)
        c: Nonlinear parameter C
        t_start: Start of time window (index)
        t_end: End of time window (index)
        domain: "finance" or "housing" (affects omega/oscillation thresholds)

    Returns:
        True if all filters pass
    """
    # 1. tc должен быть в будущем или совсем недавно (post-crash detection)
    if tc < t_end:
        # Post-crash: tc прошёл, но недавно (< 30 дней для daily, < 2 месяцев для monthly)
        days_past = t_end - tc
        max_past = 30 if domain == "finance" else 60
        if days_past > max_past:
            return False  # tc слишком давно, сигнал устарел

    # 2. Базовые границы параметров
    if not (0.1 <= m <= 0.9):
        return False

    # Omega range зависит от домена
    omega_min = 4.0 if domain == "housing" else 6.0
    omega_max = 15.0 if domain == "housing" else 13.0
    if not (omega_min <= omega <= omega_max):
        return False

    # 3. Минимум лог-периодических осцилляций в окне
    # Формула: (omega / 2π) * ln((tc - t_start) / (tc - t_end))
    try:
        dt_start = tc - t_start
        dt_end = max(tc - t_end, 0.01)  # Avoid division by zero

        if dt_start <= 0 or dt_end <= 0:
            return False  # tc должен быть после t_end

        oscillations = (omega / (2 * np.pi)) * np.log(dt_start / dt_end)

        # Housing имеет меньше точек → ниже порог
        min_oscillations = 1.5 if domain == "housing" else 2.5
        if oscillations < min_oscillations:
            return False

    except (ValueError, ZeroDivisionError):
        return False

    # 4. Условие затухания — каноника как в LPPLSParams.damping: |B|/|C|
    # (c = совместная амплитуда лог-периодического члена в параметризации A,B,C,φ)
    try:
        c_abs = abs(c)
        if c_abs <= 1e-10:
            return False
        damping = abs(b) / c_abs
        min_damping = 0.5 if domain == "housing" else 0.8
        if damping < min_damping:
            return False

    except ZeroDivisionError:
        return False

    # 5. B < 0 для bubble (сверхэкспоненциальный рост)
    if b >= 0:
        return False

    return True


def calculate_ds_lppls_confidence(
    t: np.ndarray,
    prices: np.ndarray,
    fit_func,
    min_window: int = 60,
    step: int = 10,
    domain: str = "finance",
) -> dict:
    """Double-Scale LPPLS Confidence Indicator.

    Scans the time series with multiple overlapping windows and calculates
    the fraction of windows that produce valid LPPLS fits.

    Args:
        t: Time index array
        prices: Price array
        fit_func: Function that fits LPPLS and returns params dict
                  (tc, m, omega, a, b, c, r_squared, quality_score, verdict)
        min_window: Minimum window size (points)
        step: Step size for moving start point
        domain: "finance" or "housing"

    Returns:
        Dict with ds_confidence, median_tc, tc_std, valid_windows, total_windows
    """
    total_windows = 0
    valid_fits = 0
    predictions_tc = []
    quality_scores = []

    t_end = t[-1]

    # Двигаем стартовую точку, создавая окна разной длины
    for start_idx in range(0, len(t) - min_window, step):
        t_window = t[start_idx:]
        p_window = prices[start_idx:]

        total_windows += 1

        # Фитим LPPLS на этом окне
        try:
            params = fit_func(t_window, p_window)
        except Exception:
            continue

        if not params or params.get("verdict") == "NO_BUBBLE":
            continue

        # Проверяем фильтры Sornette
        if passes_sornette_filters(
            tc=params["tc"],
            m=params["m"],
            omega=params["omega"],
            a=params.get("a", 0),
            b=params["b"],
            c=params.get("c", 0),
            t_start=t_window[0],
            t_end=t_window[-1],
            domain=domain,
        ):
            valid_fits += 1
            predictions_tc.append(params["tc"])
            quality_scores.append(params.get("quality_score", 0.5))

    # Расчет DS-LPPLS Confidence Indicator
    ds_confidence = valid_fits / total_windows if total_windows > 0 else 0.0

    # Статистика tc
    median_tc = float(np.median(predictions_tc)) if predictions_tc else None
    tc_std = float(np.std(predictions_tc)) if len(predictions_tc) > 1 else None
    tc_p10 = float(np.percentile(predictions_tc, 10)) if len(predictions_tc) >= 5 else None
    tc_p90 = float(np.percentile(predictions_tc, 90)) if len(predictions_tc) >= 5 else None

    # Средняя quality
    mean_quality = float(np.mean(quality_scores)) if quality_scores else 0.0

    # Вердикт на основе DS confidence
    if ds_confidence >= 0.6 and valid_fits >= 3:
        ds_verdict = "BUBBLE"
    elif ds_confidence >= 0.3 and valid_fits >= 2:
        ds_verdict = "POSSIBLE"
    else:
        ds_verdict = "NO_BUBBLE"

    log.info(
        "ds_lppls_complete",
        ds_confidence=round(ds_confidence, 3),
        valid_windows=valid_fits,
        total_windows=total_windows,
        median_tc=round(median_tc, 1) if median_tc else None,
        ds_verdict=ds_verdict,
    )

    return {
        "ds_confidence": round(ds_confidence, 4),
        "median_tc": median_tc,
        "tc_std": round(tc_std, 1) if tc_std else None,
        "tc_p10": round(tc_p10, 1) if tc_p10 else None,
        "tc_p90": round(tc_p90, 1) if tc_p90 else None,
        "valid_windows": valid_fits,
        "total_windows": total_windows,
        "mean_quality": round(mean_quality, 4),
        "ds_verdict": ds_verdict,
    }


def fit_lppls_simple(t: np.ndarray, prices: np.ndarray, domain: str = "finance") -> dict:
    """Single-window LPPLS on ln(price): grid search + L-BFGS-B (canonical LPPLS)."""
    t = np.asarray(t, dtype=float)
    n = len(prices)
    if n < 30:
        return {"verdict": "NO_BUBBLE", "quality_score": 0.0, "r_squared": 0.0}

    model, _ = _fit_lppls_log_price(
        t, prices, domain=domain, grid_size=10, n_best=5, tc_horizon=180.0
    )
    if model is None or model.params is None:
        return {"verdict": "NO_BUBBLE", "quality_score": 0.0, "r_squared": 0.0}

    p = model.params
    log_price = _log_prices(prices)
    r_squared = float(model.r_squared(t, log_price))

    try:
        dt_start = p.tc - t[0]
        dt_end = max(p.tc - t[-1], 0.01)
        oscillations = (p.omega / (2 * np.pi)) * np.log(dt_start / dt_end)
    except Exception:
        oscillations = 0.0

    damping = p.damping
    quality = _quality_score_lppls(p, r_squared, domain)

    omega_ok = (6.0 < p.omega < 13.0) if domain != "housing" else (4.0 < p.omega < 15.0)

    is_bubble = (
        p.B < 0
        and 0.2 < p.m < 0.7
        and omega_ok
        and damping > 1.0
        and r_squared > 0.8
        and oscillations >= 2.5
        and quality > 0.85
    )

    is_possible = (
        quality > 0.75
        and p.B < 0
        and r_squared > 0.75
        and oscillations >= 2.0
        and p.m > 0.1
    )

    verdict = "BUBBLE" if is_bubble else ("POSSIBLE" if is_possible else "NO_BUBBLE")

    from datetime import datetime, timedelta

    tc_date_str = None
    try:
        tc_date_str = (datetime.now() + timedelta(days=int(p.tc - t[-1]))).strftime("%Y-%m-%d")
    except Exception:
        tc_date_str = f"index_{p.tc:.0f}"

    return {
        "tc_estimate": float(p.tc),
        "tc_date_str": tc_date_str,
        "quality_score": round(quality, 4),
        "r_squared": round(r_squared, 4),
        "m": round(float(p.m), 4),
        "omega": round(float(p.omega), 4),
        "B": round(float(p.B), 4),
        "C": round(float(p.C), 4),
        "A": round(float(p.A), 4),
        "damping": round(float(damping), 4),
        "oscillations": round(float(oscillations), 3),
        "verdict": verdict,
    }


def fit_lppls_with_ds(
    t: np.ndarray,
    prices: np.ndarray,
    domain: str = "finance",
) -> dict:
    """Full LPPLS fit with DS-LPPLS confidence (ln-price, LPPLSOptimizer).

    This is the production-ready replacement for fit_lppls_simple.

    Args:
        t: Time index array
        prices: Price array
        domain: "finance" or "housing"

    Returns:
        Dict with all LPPLS params + DS confidence metrics
    """
    t = np.asarray(t, dtype=float)
    n = len(prices)
    if n < 30:
        return {
            "verdict": "NO_BUBBLE",
            "quality_score": 0.0,
            "r_squared": 0.0,
            "ds_confidence": 0.0,
            "ds_verdict": "NO_BUBBLE",
        }

    model, _ = _fit_lppls_log_price(
        t, prices, domain=domain, grid_size=10, n_best=5, tc_horizon=180.0
    )
    if model is None or model.params is None:
        return {
            "verdict": "NO_BUBBLE",
            "quality_score": 0.0,
            "r_squared": 0.0,
            "ds_confidence": 0.0,
            "ds_verdict": "NO_BUBBLE",
        }

    p = model.params
    r_squared = float(model.r_squared(t, _log_prices(prices)))
    best_tc = float(p.tc)

    def simple_fit_func(t_win: np.ndarray, p_win: np.ndarray):
        mloc, _ = _fit_lppls_log_price(
            t_win,
            p_win,
            domain=domain,
            grid_size=8,
            n_best=3,
            tc_horizon=180.0,
        )
        if mloc is None or mloc.params is None:
            return None
        return _dict_for_ds_window(mloc, t_win, p_win, domain)

    ds_result = calculate_ds_lppls_confidence(
        t=t,
        prices=prices,
        fit_func=simple_fit_func,
        min_window=80 if domain == "finance" else 40,
        step=25,
        domain=domain,
    )

    quality = _quality_score_lppls(p, r_squared, domain)

    oscillations = (p.omega / (2 * np.pi)) * np.log(
        max(best_tc - t[0], 1.0) / max(best_tc - t[-1], 0.01)
    )

    omega_ok = (6.0 < p.omega < 13.0) if domain != "housing" else (4.0 < p.omega < 15.0)

    is_bubble = (
        p.B < 0
        and 0.2 < p.m < 0.7
        and omega_ok
        and p.damping > 1.0
        and r_squared > 0.8
        and oscillations >= 2.5
        and quality > 0.85
    )

    if ds_result["ds_confidence"] > 0.6:
        quality = min(quality + 0.2, 1.0)
    elif ds_result["ds_confidence"] > 0.3:
        quality = min(quality + 0.1, 1.0)

    sornette_pass = passes_sornette_filters(
        tc=best_tc,
        m=p.m,
        omega=p.omega,
        a=p.A,
        b=p.B,
        c=p.C,
        t_start=t[0],
        t_end=t[-1],
        domain=domain,
    )

    if sornette_pass and ds_result["ds_verdict"] == "BUBBLE" and quality > 0.7 and is_bubble:
        verdict = "BUBBLE"
    elif quality > 0.75 and p.B < 0 and r_squared > 0.8 and is_bubble:
        verdict = "POSSIBLE"
    else:
        verdict = "NO_BUBBLE"

    from datetime import datetime, timedelta

    tc_date_str = None
    try:
        last_date = datetime.now()
        days_after = int(best_tc - t[-1])
        tc_date = last_date + timedelta(days=days_after)
        tc_date_str = tc_date.strftime("%Y-%m-%d")
    except Exception:
        tc_date_str = f"index_{best_tc:.0f}"

    return {
        "tc_estimate": best_tc,
        "tc_date_str": tc_date_str,
        "quality_score": round(quality, 4),
        "r_squared": round(r_squared, 4),
        "m": round(float(p.m), 4),
        "omega": round(float(p.omega), 4),
        "B": round(float(p.B), 4),
        "C": round(float(p.C), 4),
        "A": round(float(p.A), 4),
        "verdict": verdict,
        "ds_confidence": ds_result["ds_confidence"],
        "ds_verdict": ds_result["ds_verdict"],
        "ds_median_tc": ds_result["median_tc"],
        "ds_tc_std": ds_result["tc_std"],
        "ds_tc_p10": ds_result["tc_p10"],
        "ds_tc_p90": ds_result["tc_p90"],
        "ds_valid_windows": ds_result["valid_windows"],
        "ds_total_windows": ds_result["total_windows"],
        "sornette_filters_pass": sornette_pass,
    }
