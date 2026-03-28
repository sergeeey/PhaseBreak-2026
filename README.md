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

## Project Status

- [x] LPPLS model (fit, predict, R², RMSE)
- [x] Grid search + L-BFGS-B optimizer
- [x] 6 known bubble datasets (BTC 2017/2021, Dot-com, Tesla, Shanghai, S&P500)
- [x] 15 unit tests passing
- [ ] Validation on known bubbles (Étape 1)
- [ ] Cross-domain geology (Étape 2)
- [ ] Bayesian survival for fraud (Étape 3)
- [ ] Adversarial AI validation (Étape 4)
- [ ] Paper

## License

MIT
