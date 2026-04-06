# PhaseBreak — CLAUDE.md

## Project
Universal Phase Transition Detection with Adversarial AI Validation.
Cross-domain: Finance (LPPLS) + Geology (Sentinel-2) + Fraud Survival (Doomsday Bayesian).

## Stack
Python 3.11, NumPy, SciPy, yfinance, hmmlearn, chernoffpy, lifelines, structlog, pytest.

## Key files
- `src/lppls/model.py` — Core LPPLS equation (Sornette 2003)
- `src/lppls/optimizer.py` — Grid search + L-BFGS-B
- `src/lppls/data.py` — yfinance loader + known bubbles + negative controls
- `src/lppls/confidence.py` — Multi-window DS LPPLS Confidence Indicator
- `src/lppls/regime.py` — HMM regime detection (3 states)
- `src/lppls/ensemble.py` — HMM-gated LPPLS pipeline
- `src/lppls/certified_fit.py` — Richardson extrapolation certified tc bounds
- `tests/test_lppls_model.py` — Unit tests

## Math
LPPLS: `ln(p(t)) = A + B(tc-t)^m + C(tc-t)^m * cos(ω*ln(tc-t) + φ)`
7 params: tc (critical time), m (exponent), ω (frequency), A, B, C1, C2.
OLS for linear (A,B,C1,C2), grid+L-BFGS-B for nonlinear (tc,m,ω).

## Constraints (Sornette filters)
- m ∈ (0.1, 0.9), ω ∈ (6, 13) for finance (optimizer bounds; post-filter accepts 5.0–13.5 to avoid boundary artifacts)
- B < 0 for bubble detection
- damping |B|/|C| > 0.5

## Design Principles (Feynman-inspired)
- Seek INVARIANTS, not entities — compare conserved ratios across domains, not raw values
- Detect SYMMETRY BREAKING — stationarity loss (variance↑, AC1↑) signals regime change
- Hidden states are COMPUTABLE — latent "stress" recovered from observable traces
- No physics jargon without math — "energy" metaphors stay out, equations stay in

## 5 Contributions
1. Multi-window DS LPPLS Confidence Indicator (Sornette 2015, validated)
2. HMM-gated LPPLS ensemble (novel, not in literature)
3. Certified convergence bounds for tc via Richardson extrapolation (ChernoffPy, novel)
4. Critical slowing down layer (independent EWS, model-free)
5. Cross-domain phase transition universality (main thesis)

## Commands
```bash
pytest tests/ -v                    # run tests
python -m notebooks/01_btc_2017     # first experiment
```

## Current stage
Stage 1 + 1.5: DONE. Gate 1 GO, Gate 1.5 KEEP. Precision 80%, Recall 67%.
Stage 2: IN PROGRESS (geology, Sentinel-2 temporal LPPLS).
Next: Stage 3 (fraud survival) + Stage 3.5 (critical slowing down).
Full TZ: docs/TZ_STAGES_3-6.md
