# PhaseBreak Live Forecast Scorecard — April 2026

**Published:** 2026-03-29
**Author:** Sergey Boyko
**Model:** PhaseBreak v2 + Hurst + MFDFA (commit 40784e3)
**Verification date:** 2026-04-30
**Rule:** NO edits to this file after publication. Results appended at bottom only.

---

## Protocol

1. Universe: 13 assets (fixed, not changed after publication)
2. Each prediction has ONE binary criterion, defined BEFORE verification
3. Horizon: 30 calendar days (2026-03-29 → 2026-04-28 close)
4. Prices: Yahoo Finance adjusted close
5. Hit = criterion met. Miss = criterion not met. No partial credit.
6. Verdicts scored separately: BUBBLE, POSSIBLE, NO_BUBBLE

**Removed from scoring:**
- Solana (SOL-USD): tc in past (2026-03-28), this is postmortem, not forecast
- MicroStrategy (MSTR): Hurst=1.01 is unreliable (method artifact on short series)

---

## Predictions

### BUBBLE (1 signal)

| # | Asset | Ticker | Price 3/29 | Verdict | Quality | Criterion | Horizon |
|---|-------|--------|-----------|---------|---------|-----------|---------|
| 1 | Gold | GC=F | $4524 | BUBBLE | 0.61 | Price on 4/28 is BELOW price on 3/29 ($4524) | 30d |

**Logic:** BUBBLE verdict with tc=June 28 means the super-exponential phase should show deceleration or reversal within 3 months. At 30-day check: if gold is already declining, early confirmation. If still rising but decelerating, inconclusive. If accelerating higher, miss.

**Strict criterion:** Close on April 28 < $4524. Binary. No "almost".

### NO_BUBBLE (12 signals)

| # | Asset | Ticker | Price 3/29 | Verdict | Quality | Criterion | Horizon |
|---|-------|--------|-----------|---------|---------|-----------|---------|
| 2 | Nvidia | NVDA | $167.5 | NO_BUBBLE | 0.27 | No drawdown >15% from 3/29 price within 30d | 30d |
| 3 | Bitcoin | BTC-USD | $66320 | NO_BUBBLE | 0.38 | No drawdown >15% from 3/29 price within 30d | 30d |
| 4 | S&P 500 | ^GSPC | $6369 | NO_BUBBLE | 0.00 | No drawdown >10% from 3/29 price within 30d | 30d |
| 5 | NASDAQ | ^IXIC | $20948 | NO_BUBBLE | 0.00 | No drawdown >10% from 3/29 price within 30d | 30d |
| 6 | Tesla | TSLA | $361.8 | NO_BUBBLE | 0.00 | No drawdown >15% from 3/29 price within 30d | 30d |
| 7 | Ethereum | ETH-USD | $1993 | NO_BUBBLE | 0.00 | No drawdown >15% from 3/29 price within 30d | 30d |
| 8 | Oil WTI | CL=F | $99.6 | NO_BUBBLE | 0.18 | No drawdown >15% from 3/29 price within 30d | 30d |
| 9 | Nikkei | ^N225 | $53373 | NO_BUBBLE | 0.00 | No drawdown >10% from 3/29 price within 30d | 30d |
| 10 | ARM | ARM | $144.1 | NO_BUBBLE | 0.33 | No drawdown >15% from 3/29 price within 30d | 30d |
| 11 | Broadcom | AVGO | $300.7 | NO_BUBBLE | 0.00 | No drawdown >15% from 3/29 price within 30d | 30d |
| 12 | Palantir | PLTR | $143.1 | NO_BUBBLE | 0.21 | No drawdown >15% from 3/29 price within 30d | 30d |
| 13 | Coinbase | COIN | $161.1 | NO_BUBBLE | 0.00 | No drawdown >15% from 3/29 price within 30d | 30d |

**Logic for NO_BUBBLE threshold:**
- Indices (S&P, NASDAQ, Nikkei): >10% drawdown in 30 days = crash (rare, ~2% annual probability)
- Individual stocks + crypto: >15% drawdown in 30 days = significant correction
- If crash happens on NO_BUBBLE asset → MISS (model failed to warn)

---

## Scoring Rules

**BUBBLE verdict:**
- HIT: price declined from prediction date (close 4/28 < close 3/29)
- MISS: price rose or flat (close 4/28 ≥ close 3/29)

**NO_BUBBLE verdict:**
- HIT: no drawdown exceeding threshold during 30-day window
- MISS: drawdown exceeded threshold at any point during 30 days
- Drawdown = (price_3/29 - min_price_in_30d) / price_3/29

**Aggregate metrics:**
- BUBBLE accuracy: hits / total BUBBLE predictions
- NO_BUBBLE accuracy: hits / total NO_BUBBLE predictions
- Overall accuracy: total hits / 13
- False negative rate: NO_BUBBLE assets that crashed / total NO_BUBBLE

---

## Expected Base Rates (null model)

For context — what random chance would give:
- S&P drops >10% in any 30-day window: ~2% historically
- Individual stock drops >15% in 30 days: ~8-12% historically
- Gold declining in any 30-day window: ~45% historically

If all 12 NO_BUBBLE are HIT, that's likely (~88% base rate per asset).
The real test is Gold BUBBLE: 45% base rate of decline means HIT needs to be evaluated against this chance level.

---

## Results (FILL IN AFTER 2026-04-28)

| # | Asset | Price 4/28 | Min price 30d | Drawdown % | Criterion met? | HIT/MISS |
|---|-------|-----------|--------------|-----------|---------------|----------|
| 1 | Gold | | | | | |
| 2 | Nvidia | | | | | |
| 3 | Bitcoin | | | | | |
| 4 | S&P 500 | | | | | |
| 5 | NASDAQ | | | | | |
| 6 | Tesla | | | | | |
| 7 | Ethereum | | | | | |
| 8 | Oil WTI | | | | | |
| 9 | Nikkei | | | | | |
| 10 | ARM | | | | | |
| 11 | Broadcom | | | | | |
| 12 | Palantir | | | | | |
| 13 | Coinbase | | | | | |

**BUBBLE accuracy:** _ / 1
**NO_BUBBLE accuracy:** _ / 12
**Overall:** _ / 13
**False negative rate:** _ / 12

---

## Integrity Commitment

This scorecard was committed to GitHub on 2026-03-29 (commit hash in git log).
The git history proves no retroactive changes. Anyone can verify by checking
`git log --follow data/predictions/SCORECARD_APRIL_2026.md`.
