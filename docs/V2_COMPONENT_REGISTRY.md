# PhaseBreak v2 — Component Registry

**Date:** 2026-03-29
**Purpose:** Classify each v2 component as confirmed, exploratory, or diagnostic.

## Classification

| Component | File | Status | Evidence |
|-----------|------|--------|----------|
| Soft scoring | `scoring.py` | **CONFIRMED** | Same detection as hard filters + richer diagnostics. Ablation: no FP change. |
| tc uncertainty (bootstrap) | `uncertainty.py` | **CONFIRMED** | Primary uncertainty method. Produces [p10, p90] intervals. Mean width ~4 days on finance. |
| HMM prior weighting | `stages.py` | **CONFIRMED INTEGRATION, MIXED ACCURACY** | Architectural: reduces compute by skipping LPPLS on NORMAL periods. Accuracy: +1 FP on finance ablation, 0 recall gain. Multi-window fallback + Hurst override partially compensate. Not an accuracy improvement — an operational convenience with accuracy trade-off. |
| Adaptive windows | `windowing.py` | **CONFIRMED** | Correctly selects quarterly windows for housing. Same detection on daily data. No degradation. |
| Pipeline separation (A/B/C) | `stages.py` | **CONFIRMED** | Architectural improvement. Detector separated from science layer. |
| Triple split protocol | `splits.py` | **CONFIRMED** | Train/val/test separation prevents in-sample critique. |
| Adversarial controls | `adversarial_controls.py` | **CONFIRMED** | 6/6 correct: TP=1, TN=5. Zero false positives on hard negatives. |
| EW metrics | `metrics_early_warning.py` | **CONFIRMED** | Lead time, coverage, interval width available for all detections. |
| Conformal prediction | `conformal.py` | **EXPLORATORY** | Secondary uncertainty method. Requires calibration set. Not in default pipeline. |
| Changepoint (CUSUM) | `changepoint.py` | **DIAGNOSTIC ONLY** | Independent signal. Not integrated into verdict. Useful for visual confirmation. |
| Wavelet LPPLS | `wavelet_lppls.py` | **DIAGNOSTIC ONLY** | CWT spectral diagnostics. ω domains incompatible with LPPLS ω. Exploratory research. |
| EWS (critical slowing) | `critical_slowing.py` | **EXPLORATORY** | Independent EWS layer. Weak on real data (documented). Not in v2 pipeline. |
| Council validator | `council_validator.py` | **EXPLORATORY** | Requires Ollama. Not integrated into automated pipeline. |

## Science Layer Status

**FROZEN** — fit logic (`optimizer.py`, `model.py`) unchanged in v2.

Rationale:
- v2 changes affect scoring, uncertainty, and pipeline flow — NOT the LPPLS fit itself
- Parameter extraction (m, ω, tc) is identical to v1
- Cross-domain KS tests, universality analysis remain valid without recomputation
- KS p-values: Finance↔Geology (m=0.085, ω=0.222), Finance↔Housing (m=0.970), Finance↔Commodities (m=0.771, ω=1.000) — all unchanged

## Primary vs Secondary Paths

| Path | Entry Point | Use Case |
|------|-------------|----------|
| **v2 (recommended)** | `run_full_pipeline()` | Production: soft scoring + uncertainty + HMM prior |
| **legacy** | `run_legacy_pipeline()` or `HMMLPPLSEnsemble.analyze()` | Backward compat, ablation baseline |

## Uncertainty Method Decision

| Method | Status | Reason |
|--------|--------|--------|
| **Bootstrap** (`uncertainty.py`) | PRIMARY | Works standalone, no calibration set needed |
| Conformal (`conformal.py`) | SECONDARY | Needs known-error calibration set, coverage guarantee |
