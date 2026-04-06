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

log = structlog.get_logger()


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

    # 4. Условие затухания (Damping condition)
    # Гарантирует, что крах (сингулярность) не перекрывается амплитудой колебаний
    try:
        damping = (m * abs(b)) / (omega * abs(c)) if abs(c) > 1e-10 else np.inf

        # Housing: ниже порог (более шумные данные)
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


def fit_lppls_simple(t: np.ndarray, prices: np.ndarray) -> dict:
    """Fast single-window LPPLS fit (original version)."""
    from scipy.optimize import minimize

    n = len(prices)
    if n < 30:
        return {"verdict": "NO_BUBBLE", "quality_score": 0.0, "r_squared": 0.0}

    def lppls_func(params, t, tc):
        A, B, m, C, omega, phi = params
        dt = tc - t
        dt = np.clip(dt, 0.01, None)
        return A + B * dt**m + C * dt**m * np.cos(omega * np.log(dt) + phi)

    def sse_for_tc(tc, t, prices):
        A0 = prices[-1]
        B0 = -(prices[-1] - prices[0]) / max((tc - t[0]) ** 0.5, 0.01)
        m0 = 0.5
        C0 = (prices[-1] - prices[0]) / 10
        omega0 = 8.0
        phi0 = 0.0
        try:
            result = minimize(
                lambda params: np.sum((lppls_func(params, t, tc) - prices) ** 2),
                [A0, B0, m0, C0, omega0, phi0],
                method="L-BFGS-B",
                bounds=[
                    (-1e10, 1e10),
                    (-1e10, 0),
                    (0.1, 0.9),
                    (-1e10, 1e10),
                    (4.0, 25.0),
                    (-np.pi, np.pi),
                ],
            )
            return result.fun, result.x
        except Exception:
            return np.inf, None

    tc_candidates = np.linspace(t[-1] + 1, t[-1] + 180, 30)
    best_sse, best_params, best_tc = np.inf, None, None

    for tc in tc_candidates:
        sse, params = sse_for_tc(tc, t, prices)
        if sse < best_sse:
            best_sse, best_params, best_tc = sse, params, tc

    if best_params is None:
        return {"verdict": "NO_BUBBLE", "quality_score": 0.0, "r_squared": 0.0}

    ss_tot = np.sum((prices - np.mean(prices)) ** 2)
    r_squared = 1 - best_sse / ss_tot if ss_tot > 0 else 0.0
    A, B, m, C, omega, phi = best_params

    # Calculate oscillations count
    try:
        dt_start = best_tc - t[0]
        dt_end = max(best_tc - t[-1], 0.01)
        oscillations = (omega / (2 * np.pi)) * np.log(dt_start / dt_end)
    except Exception:
        oscillations = 0.0

    # Calculate damping ratio
    damping = abs(B) / max(abs(C), 1e-10)

    # STRICT quality scoring — proven FP rate 7% on random walks
    # WHY: relaxing these thresholds (attempted 2026-04-05) increased FP from 7% to 35%
    # and dropped eval accuracy from 68% to 56%. Meme stocks need RAG, not looser math.
    quality = 0.0
    if r_squared > 0.8:
        quality += 0.3
    elif r_squared > 0.7:
        quality += 0.15
    if B < 0:
        quality += 0.2
    if 0.2 < m < 0.7:
        quality += 0.2
    elif 0.1 < m < 0.9:
        quality += 0.1
    if 6.0 < omega < 13.0:
        quality += 0.15
    if damping > 1.0:
        quality += 0.15
    elif damping > 0.5:
        quality += 0.05

    # STRICT verdict thresholds
    is_bubble = (
        B < 0
        and 0.2 < m < 0.7
        and 6.0 < omega < 13.0
        and damping > 1.0
        and r_squared > 0.8
        and oscillations >= 2.5
        and quality > 0.85
    )

    is_possible = (
        quality > 0.75
        and B < 0
        and r_squared > 0.75
        and oscillations >= 2.0
        and m > 0.1  # WHY: m=0.1 exactly = optimizer stuck at boundary = noise
    )

    verdict = "BUBBLE" if is_bubble else ("POSSIBLE" if is_possible else "NO_BUBBLE")

    from datetime import datetime, timedelta

    tc_date_str = None
    try:
        tc_date_str = (datetime.now() + timedelta(days=int(best_tc - t[-1]))).strftime("%Y-%m-%d")
    except Exception:
        tc_date_str = f"index_{best_tc:.0f}"

    return {
        "tc_estimate": float(best_tc),
        "tc_date_str": tc_date_str,
        "quality_score": round(quality, 4),
        "r_squared": round(r_squared, 4),
        "m": round(float(m), 4),
        "omega": round(float(omega), 4),
        "B": round(float(B), 4),
        "C": round(float(C), 4),
        "A": round(float(A), 4),
        "damping": round(float(damping), 4),
        "oscillations": round(float(oscillations), 3),
        "verdict": verdict,
    }


def fit_lppls_with_ds(
    t: np.ndarray,
    prices: np.ndarray,
    domain: str = "finance",
) -> dict:
    """Full LPPLS fit with DS-LPPLS confidence calculation.

    This is the production-ready replacement for fit_lppls_simple.

    Args:
        t: Time index array
        prices: Price array
        domain: "finance" or "housing"

    Returns:
        Dict with all LPPLS params + DS confidence metrics
    """
    from scipy.optimize import minimize

    n = len(prices)
    if n < 30:
        return {
            "verdict": "NO_BUBBLE",
            "quality_score": 0.0,
            "r_squared": 0.0,
            "ds_confidence": 0.0,
            "ds_verdict": "NO_BUBBLE",
        }

    # ─── Single-window fit ────────────────────────────────────────────

    def lppls_func(params, t, tc):
        A, B, m, C, omega, phi = params
        dt = tc - t
        dt = np.clip(dt, 0.01, None)
        return A + B * dt**m + C * dt**m * np.cos(omega * np.log(dt) + phi)

    def sse_for_tc(tc, t, prices):
        A0 = prices[-1]
        B0 = -(prices[-1] - prices[0]) / max((tc - t[0]) ** 0.5, 0.01)
        m0 = 0.5
        C0 = (prices[-1] - prices[0]) / 10
        omega0 = 8.0
        phi0 = 0.0

        try:
            result = minimize(
                lambda params: np.sum((lppls_func(params, t, tc) - prices) ** 2),
                [A0, B0, m0, C0, omega0, phi0],
                method="L-BFGS-B",
                bounds=[
                    (-1e10, 1e10),
                    (-1e10, 0),
                    (0.1, 0.9),
                    (-1e10, 1e10),
                    (4.0, 25.0),
                    (-np.pi, np.pi),
                ],
            )
            return result.fun, result.x
        except Exception:
            return np.inf, None

    tc_candidates = np.linspace(t[-1] + 1, t[-1] + 180, 30)

    best_sse = np.inf
    best_params = None
    best_tc = None

    for tc in tc_candidates:
        sse, params = sse_for_tc(tc, t, prices)
        if sse < best_sse:
            best_sse = sse
            best_params = params
            best_tc = tc

    if best_params is None or best_tc is None:
        return {
            "verdict": "NO_BUBBLE",
            "quality_score": 0.0,
            "r_squared": 0.0,
            "ds_confidence": 0.0,
            "ds_verdict": "NO_BUBBLE",
        }

    ss_tot = np.sum((prices - np.mean(prices)) ** 2)
    r_squared = 1 - best_sse / ss_tot if ss_tot > 0 else 0.0

    # Optimizer order: [A, B, m, C, omega, phi]
    A, B, m, C, omega, phi = best_params

    # ─── DS-LPPLS Confidence ──────────────────────────────────────────

    def simple_fit_func(t_win, p_win):
        """Wrapper for DS confidence calculation."""
        # Quick fit on this window — fewer candidates for speed
        tc_cands = np.linspace(t_win[-1] + 1, t_win[-1] + 180, 8)
        best_s = np.inf
        best_p = None
        best_t = None

        for tc in tc_cands:
            s, p = sse_for_tc(tc, t_win, p_win)
            if s < best_s:
                best_s = s
                best_p = p
                best_t = tc

        if best_p is None:
            return None

        ss_t = np.sum((p_win - np.mean(p_win)) ** 2)
        r2 = 1 - best_s / ss_t if ss_t > 0 else 0.0
        q = 0.0
        if r2 > 0.7:
            q += 0.3
        if best_p[1] < 0:
            q += 0.2
        if 0.1 < best_p[2] < 0.9:
            q += 0.2
        if 4.0 < best_p[4] < 25.0:
            q += 0.15
        if abs(best_p[1]) / max(abs(best_p[3]), 1e-10) > 0.3:
            q += 0.15

        return {
            "tc": best_t,
            "m": best_p[2],
            "omega": best_p[4],
            "a": best_p[0],
            "b": best_p[1],
            "c": best_p[3],
            "r_squared": r2,
            "quality_score": q,
            "verdict": "BUBBLE" if q > 0.5 else "NO_BUBBLE",
        }

    ds_result = calculate_ds_lppls_confidence(
        t=t,
        prices=prices,
        fit_func=simple_fit_func,
        min_window=80 if domain == "finance" else 40,
        step=25,
        domain=domain,
    )

    # ─── Final verdict (combine single + DS) ──────────────────────────

    # Single-window quality — STRICT thresholds (synced with fit_lppls_simple)
    quality = 0.0
    if r_squared > 0.8:
        quality += 0.3
    elif r_squared > 0.7:
        quality += 0.15
    if B < 0:
        quality += 0.2
    if 0.2 < m < 0.7:
        quality += 0.2
    elif 0.1 < m < 0.9:
        quality += 0.1
    if 6.0 < omega < 13.0:
        quality += 0.15
    damping = abs(B) / max(abs(C), 1e-10)
    if damping > 1.0:
        quality += 0.15
    elif damping > 0.5:
        quality += 0.05

    # Verdict — synced with fit_lppls_simple thresholds
    oscillations = (omega / (2 * np.pi)) * np.log(
        max(best_tc - t[0], 1.0) / max(best_tc - t[-1], 0.01)
    )
    is_bubble = (
        B < 0
        and 0.2 < m < 0.7
        and 6.0 < omega < 13.0
        and damping > 1.0
        and r_squared > 0.8
        and oscillations >= 2.5
        and quality > 0.85
    )

    # DS confidence boost
    if ds_result["ds_confidence"] > 0.6:
        quality = min(quality + 0.2, 1.0)
    elif ds_result["ds_confidence"] > 0.3:
        quality = min(quality + 0.1, 1.0)

    # Combined verdict
    sornette_pass = passes_sornette_filters(
        tc=best_tc,
        m=m,
        omega=omega,
        a=A,
        b=B,
        c=C,
        t_start=t[0],
        t_end=t[-1],
        domain=domain,
    )

    if sornette_pass and ds_result["ds_verdict"] == "BUBBLE" and quality > 0.7 and is_bubble:
        verdict = "BUBBLE"
    elif quality > 0.75 and B < 0 and r_squared > 0.8 and is_bubble:
        verdict = "POSSIBLE"
    else:
        verdict = "NO_BUBBLE"

    # tc date
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
        "tc_estimate": float(best_tc),
        "tc_date_str": tc_date_str,
        "quality_score": round(quality, 4),
        "r_squared": round(r_squared, 4),
        "m": round(float(m), 4),
        "omega": round(float(omega), 4),
        "B": round(float(B), 4),
        "C": round(float(C), 4),
        "A": round(float(A), 4),
        "verdict": verdict,
        # DS-LPPLS metrics
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
