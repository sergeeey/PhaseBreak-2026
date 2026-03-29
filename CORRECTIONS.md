# PhaseBreak — Corrections & Errata

## Critical fixes (2026-03-28 session)

### 1. Ablation loose filter numbers — commit `0819b8c`
**Incorrect:** commit message states `LPPLS (loose): P=100% R=67% FP=0/6`
**Actual:** P=55%, R=100%, FP=5/6 (83%) — loose filters without tight Sornette bounds.
**Root cause:** test code used optimizer with tight filters internally; "loose" label was misleading.
**Fix:** commit `288445f` — `_LooseOptimizer` subclass truly bypasses tight filters.

### 2. Universality label overstatement — commit `a5e60d7`
**Incorrect:** commit message and code label `VERDICT = UNIVERSAL (6/6 tests)`
**Actual:** KS test with n_fin=4 has very low statistical power; p>0.05 means "cannot reject", not "confirmed universal". Bootstrap robustness: ω robust (86%), m borderline (70%).
**Fix:** paper (commit `ca015ee`, `8f304c4`) uses "Not rejected*" with explicit low-power caveat. Code label remains for backward compatibility but paper is authoritative.

### 3. Fraud C-index information leakage — commit `52deee7`
**Incorrect:** `n_observed = lifetime × fraction` — direct derivation from target variable.
**Fix:** commit `288445f` — `n_observed = velocity × time_window` (observable covariate).
**Impact:** reported +27% C-index is upper bound; actual improvement on external data may be lower.

### 4. EWS ablation dead code — commit `0819b8c`
**Incorrect:** EWS layer always returned True when ensemble said bubble — making ablation row identical to HMM gate.
**Fix:** commit `288445f` — `return ens_bubble AND ews_signal` (real gating).

## Paper vs. commit history

The paper (`paper/main.tex`) is the authoritative source for all claims. Commit messages reflect state at time of commit and may contain preliminary or subsequently corrected results. When in doubt, trust the paper.
