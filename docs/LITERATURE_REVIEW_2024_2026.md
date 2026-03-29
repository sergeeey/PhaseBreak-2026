# LPPLS Literature Review 2024-2026

**Date:** 2026-03-29
**Purpose:** Position PhaseBreak against recent publications. Identify preempted vs novel claims.

## Critical Papers (must cite)

### 1. Deep LPPLS (Nielsen, Sornette, Raissi 2024)
- **Source:** arXiv:2405.12803
- **Key:** Neural network LPPLS calibration applied to finance + 1 rockslide (cross-domain)
- **Preempts:** Cross-domain LPPLS concept (but only 2 domains, no HMM, no adversarial)

### 2. Unified Failure Model (Lei, Sornette 2025) — Nature
- **Source:** Communications Earth & Environment
- **Key:** LPPLS on 109 geohazard events (landslides, rockbursts, glaciers, volcanoes)
- **Preempts:** Geology LPPLS universality (stronger than our 13 Sentinel-2 fits)

### 3. AI-based LPPL Classification (2025) — Nature HSS
- **Key:** 13M+ labeled parameter sets, reliability score, DTCAI metric
- **Doesn't preempt:** Different mechanism (AI classifier vs multi-window confidence)

### 4. LPPLS Survey (Shu & Song 2024) — WIREs
- **Key:** Comprehensive review. Must cite.

## What remains novel in PhaseBreak

| Contribution | Preempted? | Evidence |
|---|---|---|
| HMM-gated LPPLS ensemble | **NO** — no prior work found | Web search verified |
| Adversarial validation / negative controls | **NO** — unique | Web search verified |
| Certified tc bounds (Richardson) | **NO** — unique | Not in any LPPLS paper |
| Soft scoring + bootstrap uncertainty | **NO** — unique combination | AI paper uses NN, not scoring |
| Cross-domain universality | **PARTIALLY** — Sornette group 2024-2025 | Must reframe |

## Required paper reframe

**Before:** "First cross-domain LPPLS universality study"
**After:** "First adversarially-validated multi-domain ensemble framework with HMM gating, uncertainty quantification, and systematic negative controls"

## Competitor repos

| Repo | Stars | vs PhaseBreak |
|---|---|---|
| Boulder lppls | 430 | Fit+filter only. No HMM, no ensemble, no multi-domain |
| DS-LPPLS | <50 | Crypto only |
| Fantazzini/bubble (R) | — | Trust indicator, but no HMM, no cross-domain |

**PhaseBreak fills unique niche:** no existing package combines LPPLS + HMM + uncertainty + multi-domain.
