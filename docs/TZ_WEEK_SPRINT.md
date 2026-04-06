# PhaseBreak — Week Sprint TZ (6-12 April 2026)

## Goal
Объединить математическую мощь PhaseBreak 2026 с production-обвязкой Hybrid 2.0.
Результат: единая система 9.5/10, готовая к arXiv + live monitoring.

## Метрики успеха
- [ ] arXiv-ready paper (все числа синхронизированы, tests green)
- [ ] Blind eval accuracy >= 75% (сейчас 71%)
- [ ] FP rate <= 10% на random walks
- [ ] Live predictions зафиксированы в git
- [ ] Monitor работает автоматически

---

## Day 1: Merge + arXiv + HMM import

### 1.1 PhaseBreak 2026: merge fix/arxiv-metrics-sync → main, push
- [x] `git checkout main && git merge fix/arxiv-metrics-sync` ✅ 7cb899e
- [x] `git push origin main` ✅ pushed
- [x] Verify 268/268 tests pass ✅ (267+1 fixed)

### 1.2 Hybrid 2.0: import HMM regime detection
- [x] Copy `src/lppls/regime.py` from 2026 → Hybrid ✅
- [x] Install hmmlearn ✅
- [x] Create `src/lppls/hmm_gate.py` — advisory wrapper ✅
- [x] Wire into `predict_all.py` via `run_lppls_with_hmm()` ✅
- [x] HMM is ADVISORY (not blocker): NORMAL + Q>=0.8 → keep POSSIBLE ✅

### 1.3 Results
- [x] Blind eval with HMM: 22/34 (65%) — same as without (HMM helps live, not historical)
- [x] Live test: KO correctly downgraded, TSLA preserved, Gold=BUBBLE by HMM ✅
- [x] HMM value is in live monitoring (filters noise), not historical eval

---

## Day 2: Crash-oriented eval

### 2.1 Reframe eval set — DONE
- [x] eval_crash.json created: 34 cases, 13 crashed / 21 safe
- [x] crash_eval.py runner: precision 50%, recall 23%, F1 32%
- [x] This is the HONEST baseline — LPPLS catches classic bubbles but misses meme stocks


- [ ] New file: `data/eval_crash.json`
- [ ] Label: `crashed_30pct_within_6months: true/false`
- [ ] Add `actual_drawdown_pct` and `drawdown_date` for each case
- [ ] 34 existing cases + verify against Yahoo Finance data

### 2.2 New eval runner
- [ ] `src/contract/crash_eval.py`:
  - Load eval_crash.json
  - For each case: run LPPLS → if BUBBLE/POSSIBLE, check if actually crashed
  - Metrics: precision (of crash predictions), recall (of actual crashes), lead time
- [ ] Run and document baseline metrics

---

## Day 3: Signal persistence filter

### 3.1 Persistence tracker
- [ ] `src/pipeline/persistence.py`:
  - Load history from `data/signal_history.json`
  - For each ticker: count consecutive scans with BUBBLE/POSSIBLE
  - Signal = CONFIRMED if count >= 3
  - Signal = TENTATIVE if count 1-2
  - Signal = DROPPED if was active, now NO_BUBBLE
- [ ] Wire into monitor_cron.py: only alert on CONFIRMED signals

### 3.2 Test stability
- [ ] Run 3 consecutive scans (different seeds or days)
- [ ] Verify no flip-flop on stable assets (KO, JNJ should stay NO_BUBBLE)
- [ ] Verify real signals persist (if NVDA stays POSSIBLE 3x → CONFIRMED)

---

## Day 4: Hurst exponent + RAG caching

### 4.1 Import Hurst from 2026
- [ ] Add `hurst` package to deps
- [ ] Create `src/lppls/hurst_signal.py`:
  - `compute_hurst(prices) -> float`
  - H > 0.7 = persistent trend (supports bubble hypothesis)
  - H ≈ 0.5 = random walk (weakens bubble hypothesis)
- [ ] Add to verdict_contract as signal #3 (after LPPLS + RAG)

### 4.2 RAG response caching
- [ ] In rag_engine.py: cache LLM responses by (ticker, date_range_hash)
- [ ] TTL: 24 hours (news doesn't change that fast)
- [ ] Storage: `data/rag_cache/` JSON files
- [ ] Expected: repeat scans 3x faster, 3x cheaper

---

## Day 5: Geology + arXiv submission

### 5.1 Geology n>=15 — BLOCKED
- [ ] ~~Download Sentinel-2 scenes~~ — data/geology/ is empty, needs Copernicus account + download
- [x] Verified: geo tests use synthetic data only, no real Sentinel-2 files on disk
- Status: DEFERRED to post-sprint. Paper already acknowledges "larger samples needed"

### 5.2 arXiv readiness — DONE
- [x] Paper metrics match benchmark JSON (78%/67%/76%/61%)
- [x] Figures regenerated with correct data
- [x] 11 citations, 113-line bib file
- [x] plotly removed, test count 268
- [x] Paper ready to submit when endorsement arrives

---

## Day 6: Streamlit dashboard

### 6.1 Minimal dashboard
- [ ] `dashboard.py` (Streamlit, single file):
  - Tab 1: Current signals table (from latest scan JSON)
  - Tab 2: Sector heatmap (from sector_scan output)
  - Tab 3: Signal history (from signal_history.json)
  - Tab 4: Eval results
- [ ] `streamlit run dashboard.py`
- [ ] Add streamlit to optional deps

---

## Day 7: Live monitoring + predictions

### 7.1 Register monitor in Task Scheduler
- [ ] Windows Task Scheduler: run monitor_cron.py every 6 hours
- [ ] Verify Telegram alerts work (or dry-run)

### 7.2 Lock predictions
- [ ] Run full scan on 50+ assets
- [ ] Commit predictions to git (immutable timestamp)
- [ ] Set verification dates:
  - TSLA tc Aug 2026 → check Sep 1
  - NVDA tc Oct 2026 → check Nov 1
  - Any new signals → check tc + 30 days

### 7.3 Final status report
- [ ] Accuracy on crash-eval
- [ ] FP rate
- [ ] Signal count (CONFIRMED / TENTATIVE / NO_BUBBLE)
- [ ] Commit everything

---

## Architecture after sprint

```
predict_all.py
  ├── HMM gate (from 2026) → skip NORMAL assets
  ├── fit_lppls_simple → verdict
  ├── Hurst exponent → persistence signal
  ├── RSS + RAG → semantic context (cached)
  ├── verdict_contract (LPPLS + HMM + Hurst + RAG → final)
  ├── persistence filter → CONFIRMED / TENTATIVE
  ├── sector scan → contagion detection
  └── alerter → Telegram (only CONFIRMED)
```

## Out of scope (NOT THIS WEEK)
- Vector DB / embeddings
- Reddit/Twitter sentiment
- New domains (epidemic, landslide, AI compute)
- Web API (FastAPI)
- Refactoring orchestrator.py
