# PhaseBreak New Domains Scorecard — 29 March 2026

**Published:** 2026-03-29
**Model:** PhaseBreak v2 + Hurst + MFDFA (commit da5742a)
**Rule:** NO edits after publication. Git history is proof.

---

## Domain 7: AI COMPUTE — The Singularity Question

| # | Series | Verdict | Quality | R² | tc year | Interpretation |
|---|--------|---------|---------|-----|---------|---------------|
| 1 | **Training FLOP** | **BUBBLE** | 0.836 | 0.997 | **~2038** | Super-exponential growth has finite-time singularity. Current trajectory cannot be sustained. |
| 2 | Investment ($B) | NO_SIGNAL | 0.212 | — | ~2050 | Too few points, no LPPLS pattern. |
| 3 | Electricity (TWh) | NO_SIGNAL | 0.268 | — | ~2027 | Linear growth, not super-exponential. |

**Key prediction: AI training compute growth will hit a wall around 2038.**

This does NOT mean "AI stops" — it means the current scaling law (doubling every ~6 months) will break. Could be: energy limits, economic limits, diminishing returns, or paradigm shift.

**Verification:** Track Epoch AI dataset annually. If FLOP growth decelerates before 2038 → early confirmation. If FLOP growth accelerates → tc shifts later.

---

## Domain 6: EPIDEMICS — COVID Wave Detection

| # | Wave | Country | Verdict | Quality | Expected | Result |
|---|------|---------|---------|---------|----------|--------|
| 1 | Wave 1 (Apr 2020) | US | NO_BUBBLE | 0.000 | Wave | MISS |
| 2 | **Delta (Sep 2021)** | **US** | **BUBBLE** | **0.862** | Wave | **OK** |
| 3 | **Omicron (Jan 2022)** | **US** | **BUBBLE** | **0.655** | Wave | **OK** |
| 4 | Delta (May 2021) | India | NO_BUBBLE | 0.235 | Wave | MISS |
| 5 | Alpha (Jan 2021) | UK | NO_BUBBLE | 0.000 | Wave | MISS |
| 6 | Gamma (Mar 2021) | Brazil | NO_BUBBLE | 0.000 | Wave | MISS |
| 7 | Summer plateau | US | POSSIBLE | 0.549 | Control | MISS (FP) |
| 8 | Stable period | Japan | NO_BUBBLE | 0.210 | Control | OK |

**Score: 3/8 correct (37.5%).** Epidemic detection needs work:
- HMM blocks most waves (cumulative cases don't look like equity bubbles)
- US Delta and Omicron detected with high quality (0.86, 0.66)
- India/UK/Brazil missed — different growth dynamics

**Future application:** If H5N1 or new pandemic emerges, apply LPPLS to cumulative cases with HMM bypass. Lead time could be 1-2 weeks before wave peak.

---

## Domain 8: LANDSLIDES — Displacement Failure Prediction

| # | Episode | Verdict | Quality | tc predicted | tc actual | Result |
|---|---------|---------|---------|-------------|-----------|--------|
| 1 | **Vaiont 1963** (Italy) | **BUBBLE** | 0.608 | 100.9 | 100 | **OK** |
| 2 | **Randa 1991** (Switzerland) | **BUBBLE** | 0.620 | 119.0 | 120 | **OK** |
| 3 | **Asamushi 2015** (Japan) | **BUBBLE** | 0.722 | 154.8 | 150 | **OK** |
| 4 | **Xinmo 2017** (China) | **BUBBLE** | 0.682 | 91.0 | 90 | **OK** |
| 5 | **Bingham Canyon 2013** (USA) | **BUBBLE** | 0.897 | 114.1 | 110 | **OK** |
| 6 | Stable slope (linear) | NO_BUBBLE | 0.000 | — | — | **OK** |
| 7 | Stable slope (seasonal) | NO_BUBBLE | 0.196 | — | — | **OK** |

**Score: 7/7 correct (100%).** tc error: 0.9–4.8 index units.

**Caveat:** These are synthetic episodes calibrated to published LPPLS parameters. Real USGS data validation needed for publication claim.

**Cross-domain parameters:**
- Landslide m = [0.12, 0.33] — lower than finance [0.45, 0.64]
- Landslide ω = [6.9, 10.3] — overlaps with finance [7.5, 11.9]
- m differs across domains, ω may be universal

---

## Summary: 8 Domains Total

| Domain | Episodes | Accuracy | Key Finding |
|--------|----------|----------|-------------|
| Finance | 20 | 75% | Core domain, best validated |
| Commodities | 10 | 60% | Domain-aware gating helps |
| Housing | 16 | 50% | Quarterly data limits recall |
| Geology | Sentinel-2 | — | Spectral LPPLS, KS p>0.05 |
| Fraud | 500 synth | C-index +27% | Doomsday Bayesian |
| **Epidemics** | **8** | **38%** | Needs HMM bypass for cumulative cases |
| **AI Compute** | **3** | **—** | **BUBBLE tc≈2038 (first ever LPPLS)** |
| **Landslides** | **7** | **100%** | Synthetic but calibrated to real params |

---

## What to Track

| When | What | How |
|------|------|-----|
| Annually | AI training FLOP growth rate | Epoch AI dataset — deceleration = early confirmation of tc≈2038 |
| If pandemic | Apply LPPLS to cumulative cases | `src/epidemic/data.py` + bypass HMM |
| If USGS landslide data acquired | Validate on real displacement | Replace synthetic with USGS monitoring data |
