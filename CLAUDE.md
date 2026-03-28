# PhaseBreak — CLAUDE.md

## Project
Universal Phase Transition Detection with Adversarial AI Validation.
Cross-domain: Finance (LPPLS) + Geology (Sentinel-2) + Fraud Survival (Doomsday Bayesian).

## Stack
Python 3.11, NumPy, SciPy, yfinance, lifelines, structlog, pytest.

## Key files
- `src/lppls/model.py` — Core LPPLS equation (Sornette 2003)
- `src/lppls/optimizer.py` — Grid search + L-BFGS-B
- `src/lppls/data.py` — yfinance loader + known bubble datasets
- `tests/test_lppls_model.py` — Unit tests

## Math
LPPLS: `ln(p(t)) = A + B(tc-t)^m + C(tc-t)^m * cos(ω*ln(tc-t) + φ)`
7 params: tc (critical time), m (exponent), ω (frequency), A, B, C1, C2.
OLS for linear (A,B,C1,C2), grid+L-BFGS-B for nonlinear (tc,m,ω).

## Constraints (Sornette filters)
- m ∈ (0.1, 0.9), ω ∈ (6, 13) for finance
- B < 0 for bubble detection
- damping |B|/|C| > 0.5

## Commands
```bash
pytest tests/ -v                    # run tests
python -m notebooks/01_btc_2017     # first experiment
```

## Current stage
Этап 1: LPPLS baseline on known bubbles. GO/NO-GO after 5 days.
