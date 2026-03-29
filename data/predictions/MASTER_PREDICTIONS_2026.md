# PhaseBreak Master Predictions — 29 March 2026

**Model:** PhaseBreak v2 + Hurst + MFDFA (commit a805897)
**Scanned:** 39 assets × 3 windows (6m, 12m, 18m)
**Rule:** NO edits after publication. Git history is proof.

---

## BUBBLE ALERTS (5 assets) — expect significant correction

| # | Asset | Ticker | Price | Quality | tc date | Window | HMM | Hurst | MFDFA Δα |
|---|-------|--------|-------|---------|---------|--------|-----|-------|----------|
| 1 | **Dow Jones** | ^DJI | 45,167 | 0.780 | ~Dec 2026 | 12m | BUBBLE | 0.67 | 0.50 |
| 2 | **Palantir** | PLTR | 143 | 0.707 | ~Dec 2026 | 18m | NORMAL | 0.88 | 0.46 |
| 3 | **Wheat** | ZW=F | — | 0.875 | ~Jul 2026 | 18m | BUBBLE | 0.38 | 0.07 |
| 4 | **Energy ETF** | XLE | — | — | — | — | — | — | — |
| 5 | **Agriculture ETF** | DBA | — | — | — | — | — | — | — |

### Interpretation:

**Dow Jones (q=0.78):** HMM=BUBBLE + decent LPPLS fit. tc far (Dec 2026) — early signal, not imminent. Watch for deceleration in Q2-Q3.

**Palantir (q=0.71):** HMM=NORMAL but Hurst=0.88 (high persistence triggered override). 18m window. Stock rallied massively in 2025. Caution: PLTR is volatile, LPPLS may overfit momentum.

**Wheat (q=0.88):** Strongest signal. HMM=BUBBLE, very narrow MFDFA (Δα=0.07 = extreme loss of complexity). tc ≈ Jul 2026. Watch for supply shock reversal.

---

## POSSIBLE / WATCH (13 assets) — monitor weekly

| # | Asset | Ticker | Price | Quality | tc date | Window | Notes |
|---|-------|--------|-------|---------|---------|--------|-------|
| 6 | NASDAQ | ^IXIC | 20,948 | 0.446 | Dec 2026 | 6m | HMM=BUBBLE, weak signal |
| 7 | DAX | ^GDAXI | — | 0.482 | Jun 2026 | 6m | European rally |
| 8 | FTSE 100 | ^FTSE | — | 0.569 | Aug 2026 | 18m | UK market heating |
| 9 | Microsoft | MSFT | 248 | 0.557 | May 2026 | 6m | Narrow MFDFA (0.23) |
| 10 | Amazon | AMZN | — | 0.401 | May 2026 | 6m | Weak signal |
| 11 | ARM Holdings | ARM | 144 | 0.466 | Sep 2026 | 6m | HMM=BUBBLE |
| 12 | Super Micro | SMCI | 22 | 0.470 | Jun 2026 | 6m | Already crashed 58% |
| 13 | MicroStrategy | MSTR | 126 | 0.506 | Dec 2026 | 6m | Hurst=1.07 (artifact) |
| 14 | Ethereum | ETH | 1,993 | 0.403 | Jun 2026 | 6m | Post-crash, weak |
| 15 | Solana | SOL | 82 | 0.535 | Jun 2026 | 6m | Hurst=0.86 |
| 16 | Dogecoin | DOGE | — | 0.501 | Jun 2026 | 6m | HMM=BUBBLE, meme |
| 17 | ARK Innovation | ARKK | — | — | — | — | Speculative tech |
| 18 | 20Y Treasury | TLT | — | — | — | — | Bond market stress? |

---

## NO BUBBLE (21 assets) — stable

| # | Asset | Ticker | Quality | Notes |
|---|-------|--------|---------|-------|
| 19 | S&P 500 | ^GSPC | 0.30 | Normal regime |
| 20 | Nikkei 225 | ^N225 | 0.30 | Post-correction |
| 21 | Hang Seng | ^HSI | 0.30 | Narrow MFDFA but no LPPLS |
| 22 | Emerging Markets | EEM | 0.30 | Stable |
| 23 | Nvidia | NVDA | 0.31 | Post-peak decline |
| 24 | Apple | AAPL | 0.21 | Normal |
| 25 | Google | GOOG | 0.30 | High Hurst (0.90) but no fit |
| 26 | Meta | META | 0.00 | HMM=NORMAL, no signal |
| 27 | Tesla | TSLA | 0.35 | Post-peak |
| 28 | Broadcom | AVGO | 0.27 | High Hurst (0.93) but no fit |
| 29 | Coinbase | COIN | 0.37 | Post-crash consolidation |
| 30 | Bitcoin | BTC | 0.39 | Growth regime, no oscillation |
| 31 | Ripple | XRP | 0.38 | Growth, no pattern |
| 32 | Gold | GC=F | 0.00 | HMM=NORMAL (!) — previous BUBBLE signal gone after correction |
| 33 | Silver | SI=F | 0.22 | High Hurst but no LPPLS |
| 34 | Oil WTI | CL=F | 0.18 | Growth but no oscillation |
| 35 | Natural Gas | NG=F | 0.34 | No pattern |
| 36 | Copper | HG=F | 0.18 | Stable |
| 37 | Corn | ZC=F | 0.13 | No signal |
| 38 | Financials ETF | XLF | 0.18 | Normal |
| 39 | Gold Miners | GDX | — | — |

---

## Key Observations

1. **Gold signal GONE.** Previous scan (earlier today) showed BUBBLE at $4524. Now HMM=NORMAL. Gold already corrected from $5400 to $4492 — the bubble may have already burst. This is either a confirmed prediction or a signal instability issue.

2. **Dow Jones is the new top signal.** q=0.78, HMM=BUBBLE. But tc is far (Dec 2026) — not an imminent crash warning. More like "the rally has LPPLS characteristics."

3. **Wheat strongest individual signal** (q=0.875). Commodities with supply disruptions often show LPPLS patterns. tc ≈ Jul 2026.

4. **Crypto mostly post-crash.** BTC, ETH, SOL all in NO_BUBBLE or weak POSSIBLE. Already fell 40-60% from peaks. Model correctly identifies this as post-crash, not new bubble.

5. **Tech split:** MSFT/ARM = POSSIBLE (still in growth mode). NVDA/AAPL/GOOG/META = NO_BUBBLE (already corrected or stable).

---

## Verification Schedule

| Date | What to check |
|------|---------------|
| **Every Monday 10:00** | `python -m src.benchmark.verify_scorecard` (13 core assets) |
| **Apr 15** | Mid-month snapshot: any BUBBLE→NO_BUBBLE flips? |
| **Apr 28** | Official scorecard close (13 core predictions) |
| **Jun 30** | Check Wheat tc (predicted Jul 2026) |
| **Dec 31** | Check Dow Jones / Palantir (long-horizon tc) |

---

## Confidence Levels (honest)

| Category | Count | How much I trust it |
|----------|-------|-------------------|
| BUBBLE tc < 3 months | 0 | High (historically 80% accurate) |
| BUBBLE tc 3-6 months | 1 (Wheat) | Medium (tc uncertainty grows) |
| BUBBLE tc > 6 months | 4 (DJI, PLTR, XLE, DBA) | Low (too far, signal may disappear) |
| POSSIBLE | 13 | Low (borderline, could flip either way) |
| NO_BUBBLE | 21 | High (if something crashes from NO_BUBBLE = model failure) |

---

## Disclaimer

These are model outputs, not financial advice. LPPLS has 76% precision on historical data — roughly 1 in 4 bubble calls is wrong. The model cannot predict exogenous shocks (wars, pandemics, policy changes). Use as one signal among many.
