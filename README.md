# PhaseBreak 2026

**Cross-Domain Phase Transition Detection with Log-Periodic Power Law Analysis**

Framework for detecting phase transitions across 5 domains using LPPLS (Sornette 2003), HMM regime detection, and Bayesian Survival analysis. Validated on real yfinance, Sentinel-2 satellite, FHFA housing, and commodity data.

## Scientific Hypothesis

> LPPLS parameters (m, ω) from financial bubbles and geological spectral anomalies are drawn from the same distribution — preliminary evidence of universal phase transition signatures.

## Core Math: LPPLS (Sornette 2003)

```
ln(E[p(t)]) = A + B(tc - t)^m + C(tc - t)^m * cos(ω * ln(tc - t) + φ)
```

7 parameters: `tc` (critical time), `m` (exponent), `ω` (frequency), `A`, `B`, `C1`, `C2`

## Domains (5) — 58 benchmark episodes

| Domain | Data Source | Episodes | v2 Results |
|--------|-----------|----------|------------|
| **Finance** | Yahoo Finance | 11 bubbles + 9 controls | Precision=78%, Recall=64% |
| **Commodities** | Yahoo Finance (futures) | 6 bubbles + 4 controls | Precision=67%, Recall=50% |
| **Housing** | FHFA + Zillow | 10+6 episodes | Precision=67%, Recall=33-50% |
| **Geology** | Sentinel-2 temporal series | Seismic precursor patterns | KS p>0.05 |
| **Fraud** | Transaction timelines | Doomsday Bayesian, C-index +27% | Survival model |

## Domain Maturity Tiers

| Tier | Domains | Meaning |
|------|---------|---------|
| **Supported** | Finance, Commodities | Validated on real data, forward-tested, production-usable |
| **Experimental** | Housing, Epidemics | Works but limited recall, needs more data/tuning |
| **Science-only** | Geology, AI Compute, Landslides | Research validation, not for operational decisions |
| **Separate track** | Fraud | Different model (survival), not LPPLS bubble detection |

## Pipeline Architecture (v2)

```
Layer A: Screening → data quality + HMM regime detection
Layer B: Structural Fit → LPPLS + soft scoring + tc uncertainty + adaptive windows + HMM prior
Layer C: Scientific Inference → cross-domain KS, universality (offline only)
```

**Entry points:**
- `run_full_pipeline()` — v2 recommended path
- `run_legacy_pipeline()` — backward-compatible path
- `HMMLPPLSEnsemble.analyze()` — legacy ensemble

## Web API & Dashboard (NEW)

**FastAPI Server** — REST API for LPPLS detection:
```bash
pip install fastapi uvicorn[standard]
python -m service.server.main
# API: http://localhost:8000
# Docs: http://localhost:8000/docs
```

**React Dashboard** — Visual interface with real-time scanning:
```bash
cd service/frontend
npm install && npm start
# Dashboard: http://localhost:3000
```

**API Endpoints:**
- `POST /api/v1/scan` — Scan multiple tickers
- `GET /api/v1/scan/{ticker}` — Scan single ticker
- `GET /api/v1/scorecard` — Get current predictions
- `GET /api/v1/domains` — List available domains
- `GET /api/v1/benchmark` — Get benchmark summary

See `service/README.md` for full documentation.

## One-Command Replication

```bash
# Linux / macOS / Git Bash
bash reproduce.sh

# Windows PowerShell
powershell -ExecutionPolicy Bypass -File reproduce.ps1
```

Creates a virtual environment, installs all dependencies, runs 268 tests, then reproduces the full v2 benchmark (58 episodes), ablation study, and baseline comparisons. Results saved to `data/v2_results/`.

## Quick Start

```bash
pip install -e ".[dev]"
pytest tests/ -v                           # run all tests
python -m src.benchmark.v2_benchmark       # run official v2 benchmark
python -m src.benchmark.v2_ablation        # run ablation study
```

## Key Contributions

1. **Multi-window DS LPPLS Confidence Indicator** — fit on multiple overlapping windows, consensus = confident signal (Sornette 2015). Predicts tc with 1-19 day accuracy on known bubbles.
2. **Tightened Sornette filters** — the critical engineering contribution: raises precision from 55% to 100%, eliminates 5/6 false positives.
3. **HMM-gated LPPLS ensemble** — Hidden Markov Model pre-screens regime → LPPLS fits only in bubble state (novel combination, precision 80%).
4. **Doomsday Bayesian survival** — Gott (1993) random observer assumption as Cox PH feature for fraud scheme lifetime prediction. +27% C-index on synthetic data (upper bound).
5. **Cross-domain parameter comparison** — KS tests on (m, ω) between finance and geology: p > 0.05 (cannot reject H0, small sample).

## v2 Enhancements (confirmed)

| Component | Status | Impact |
|-----------|--------|--------|
| Soft scoring | Confirmed | Quality score 0-1 instead of binary pass/fail |
| tc uncertainty (bootstrap) | Confirmed | [p10, p90] intervals, mean width ~4 days |
| Adaptive windows | Confirmed | Frequency-aware: daily/quarterly/highvol presets |
| HMM prior weighting | Confirmed (mixed impact) | Reduces compute on normal periods; +1 FP on finance ablation |
| Pipeline separation | Confirmed | Detector ≠ science inference |
| Triple split | Confirmed | Train/val/test per domain |
| Adversarial controls | Confirmed | 6/6 correct (TP=1, TN=5) |
| EW metrics | Confirmed | Lead time, coverage, interval width |

**Exploratory modules** (not in default pipeline):
- Conformal prediction — secondary uncertainty method (needs calibration set)
- Changepoint (CUSUM) — diagnostic only, not integrated into verdict
- Wavelet LPPLS — CWT spectral diagnostics, ω domains incompatible
- EWS (critical slowing down) — weak on real data
- AI council — requires Ollama, heuristic fallback

## v2 Ablation (Finance, 12 episodes)

| Layer | TP | FP | Precision | Recall |
|-------|----|----|-----------|--------|
| Raw LPPLS | 4 | 0 | 100% | 67% |
| + Hard filters | 4 | 0 | 100% | 67% |
| + Soft scoring | 4 | 0 | 100% | 67% |
| + Adaptive windows | 4 | 0 | 100% | 67% |
| Full v2 (+ HMM prior) | 4 | 1 | 80% | 67% |

## Project Status

- [x] LPPLS model + tightened Sornette filters (Stage 1)
- [x] Multi-window confidence indicator (Stage 1)
- [x] HMM-gated ensemble — precision 80%, recall 67% (Stage 1.5)
- [x] Geology — LPPLS on real Sentinel-2 data, KS p>0.05 (Stage 2)
- [x] Fraud survival — Doomsday + Cox on synthetic data (Stage 3)
- [x] Critical Slowing Down — EWS layer (Stage 3.5)
- [x] Adversarial AI council — framework, heuristic mode (Stage 4)
- [x] Cross-domain correlation — KS + Mann-Whitney + bootstrap (Stage 5)
- [x] Paper draft — 7 pages, 3 figures (Stage 6)
- [x] v2 integration — soft scoring, uncertainty, adaptive windows, HMM prior
- [x] v2 benchmark — 58 episodes across 6 categories (finance, commodities, housing, housing_monthly, adversarial, forward)
- [x] v2 ablation — layer-by-layer evaluation
- [x] 5 domains validated (finance, commodities, housing, geology, fraud)
- [x] Forward validation 2024-2025: Nvidia/Nikkei/BTC detected, 0 FP
- [x] **268+ tests passing**

## License

MIT
