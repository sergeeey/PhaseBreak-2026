# PhaseBreak 2026

**Cross-Domain Phase Transition Detection with Log-Periodic Power Law Analysis**

Framework for detecting phase transitions in financial markets and geological spectral data using LPPLS (Sornette 2003) + Bayesian Survival analysis. Validated on real yfinance and Sentinel-2 satellite data.

## Scientific Hypothesis

> LPPLS parameters (m, ω) from financial bubbles and geological spectral anomalies are drawn from the same distribution — preliminary evidence of universal phase transition signatures.

## Core Math: LPPLS (Sornette 2003)

```
ln(E[p(t)]) = A + B(tc - t)^m + C(tc - t)^m * cos(ω * ln(tc - t) + φ)
```

7 parameters: `tc` (critical time), `m` (exponent), `ω` (frequency), `A`, `B`, `C1`, `C2`

## Domains

| Domain | Data Source | Application |
|--------|-----------|-------------|
| **Finance** | Yahoo Finance, FRED | Bubble/crash detection |
| **Geology** | Sentinel-2 temporal series | Seismic precursor patterns |
| **Fraud** | Transaction timelines | Scheme lifetime prediction (Doomsday Bayesian) |

## Quick Start

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Key Contributions

1. **Multi-window DS LPPLS Confidence Indicator** — fit on multiple overlapping windows, consensus = confident signal (Sornette 2015). Predicts tc with 1-19 day accuracy on known bubbles.
2. **Tightened Sornette filters** — the critical engineering contribution: raises precision from 55% to 100%, eliminates 5/6 false positives.
3. **HMM-gated LPPLS ensemble** — Hidden Markov Model pre-screens regime → LPPLS fits only in bubble state (novel combination, precision 80%).
4. **Doomsday Bayesian survival** — Gott (1993) random observer assumption as Cox PH feature for fraud scheme lifetime prediction. +27% C-index on synthetic data (upper bound).
5. **Cross-domain parameter comparison** — KS tests on (m, ω) between finance and geology: p > 0.05 (cannot reject H0, small sample).

**Exploratory modules** (framework ready, not yet validated):
- Certified convergence bounds (Richardson extrapolation, adapted from ChernoffPy)
- Adversarial AI council (Bull/Bear/Skeptic, heuristic fallback — 0% improvement in current mode)

## Project Status

- [x] LPPLS model + tightened Sornette filters (Stage 1)
- [x] Multi-window confidence indicator (Stage 1)
- [x] HMM-gated ensemble — precision 80%, recall 67% (Stage 1.5)
- [x] Geology — LPPLS on real Sentinel-2 data, KS p>0.05 (Stage 2)
- [x] Fraud survival — Doomsday + Cox on synthetic data (Stage 3)
- [x] Critical Slowing Down — EWS layer (Stage 3.5)
- [x] Adversarial AI council — framework, heuristic mode (Stage 4)
- [x] Cross-domain correlation — KS + Mann-Whitney + bootstrap (Stage 5)
- [x] Paper draft — 7 pages, 3 figures, compiles to PDF (Stage 6)
- [x] Ablation study on 12 real datasets
- [x] 3 Jupyter notebooks (finance, geology, fraud)
- [x] **164 tests passing**

## License

MIT
