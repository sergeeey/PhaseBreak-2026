# Nikkei 2024 Failure Analysis

**Date:** 2026-03-29
**Episode:** ^N225 peaked ~41,580 Jul 2024, crashed 26% to ~31,000 on Aug 5

## Pipeline output: NO_BUBBLE (quality=0.21) — complete miss

## Cascade of 3 failures

### Layer 1: Window sensitivity (ROOT CAUSE)
- 13-month window (Jun 2023 → Jul 2024): includes 4 months flat data → R²=-7383 (garbage)
- **9-month window** (Oct 2023 → Jul 2024): m=0.203, ω=10.95, R²=0.95, quality=**0.849** (excellent!)
- Adaptive windows selected [60,90,120,180] days — none captured the clean 9-month rally

### Layer 2: HMM penalty (decisive)
- HMM classified as GROWTH (not BUBBLE): bubble_prob=0.0
- hmm_weight = 0.70 → quality 0.849 × 0.70 = **0.594**
- BUBBLE threshold = 0.6 → **missed by 0.006 points**

### Layer 3: Hurst too low
- H=0.64 (13m window), H=0.75 (9m window)
- Threshold 0.85 — not triggered. "Stealth bubble" (low vol, steady growth)

## MFDFA correctly flagged it
- Δα=0.121 → bubble_score=1.0 (strong narrow spectrum signal)
- But MFDFA boost limited to 5% — not enough to overcome HMM penalty

## Proposed fixes (ranked by safety)

1. **Bubble-onset-anchored windows** — use changepoint to find rally start, anchor window there
2. **Cap HMM penalty** when raw_quality > 0.8 AND MFDFA confirms (hmm_weight min 0.80)
3. **Multi-window consensus** — fit all windows independently, use voting
4. **Raise MFDFA boost** to 10-15% when strongly confirmed

## tc estimate from 9-month fit
- tc ≈ idx 332-350 → late Sep to mid-Oct 2024
- Actual crash: Aug 5 — about 6-8 weeks early
- Within known LPPLS uncertainty range
