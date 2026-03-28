# PhaseBreak 2026

**Universal Phase Transition Detection with Adversarial AI Validation**

Cross-domain framework for detecting phase transitions (bubbles, crashes, seismic precursors, chromatin restructuring) using a unified mathematical apparatus: LPPLS + Bayesian Survival + Logistic Saturation, validated through adversarial multi-agent debate.

## Scientific Hypothesis

> Phase transitions in financial, geological, and biological systems are described by the same mathematical models (log-periodic oscillations near critical time). AI-adversarial validation improves precision of critical time `tc` prediction by 15-30% compared to classical LPPLS fitting.

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

1. **Multi-window DS LPPLS Confidence Indicator** — fit on multiple overlapping windows, consensus = confident signal (Sornette 2015)
2. **HMM-gated LPPLS ensemble** — Hidden Markov Model pre-screens regime → LPPLS fits only in bubble state (novel combination)
3. **Certified convergence bounds for tc** — Richardson extrapolation validates fit stability, detects overfitting (adapted from ChernoffPy/Galkin-Remizov 2025)
4. **Cross-domain universality** — same (m, ω) fingerprint across finance, geology, fraud?

## Project Status

- [x] LPPLS model (fit, predict, R², RMSE)
- [x] Grid search + L-BFGS-B optimizer
- [x] 6 known bubble datasets (BTC 2017/2021, Dot-com, Tesla, Shanghai, S&P500)
- [x] 15 unit tests passing
- [x] 6 negative control datasets with 0/6 false positives
- [x] Certified convergence bounds (Richardson extrapolation)
- [ ] Validation on known bubbles + multi-window confidence (Stage 1)
- [ ] HMM regime detection + ensemble (Stage 1.5)
- [ ] Cross-domain geology (Stage 2)
- [ ] Bayesian survival for fraud (Stage 3)
- [ ] Adversarial AI validation (Stage 4)
- [ ] Cross-domain (m, ω) correlation (Stage 5)
- [ ] Paper (Stage 6)

## License

MIT
