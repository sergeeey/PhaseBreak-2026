# ТЗ: PhaseBreak — Этапы 3-6

**Дата:** 2026-03-28
**Автор:** Сергей Бойко + Claude Opus 4.6
**Статус проекта:** Stage 2 в процессе, Gate 1 + 1.5 пройдены

---

## Текущее состояние (VERIFIED)

| Этап | Статус | Результат |
|------|--------|-----------|
| 1 | ✅ DONE | LPPLS baseline, multi-window CI, certified bounds. Gate 1 GO: tc<30d на 4/6 |
| 1.5 | ✅ DONE | HMM ensemble tuned. Precision 80%, Recall 67%. 1 borderline FP (S&P 2013 QE) |
| 2 | 🔄 IN PROGRESS | Sentinel-2 temporal LPPLS. KS preliminary: p=0.085 (m), p=0.222 (ω) |
| 3 | ❌ TODO | Fraud survival (Doomsday Bayesian) |
| 4 | ❌ TODO | Adversarial AI validation |
| 5 | ❌ TODO | Cross-domain correlation |
| 6 | ❌ TODO | Paper |

**Codebase:** 7 коммитов, 109 тестов, ~3600 LOC, 4 contributions.

---

## ЭТАП 3 — Fraud Survival (Неделя 5)

### Цель
Применить Doomsday Argument (Bayesian survival analysis) к предсказанию lifetime мошеннических схем. Третий домен для universality thesis.

### Научная гипотеза
> Если мы наблюдаем fraud-схему на её n-й транзакции, Bayesian posterior P(N_total | n) с Weibull prior даёт calibrated prediction оставшегося lifetime. Doomsday-feature улучшает Cox model C-index на ≥15%.

### Входные данные

**Вариант A — Synthetic (основной, для paper):**
- Генерация 500 fraud timelines: Weibull(shape=1.5, scale=200) + covariates
- Covariates: transaction_velocity, amount_std, unique_merchants, time_between_transactions
- Censoring: 30% right-censored (ещё не пойманы)

**Вариант B — Kaggle (дополнительная валидация, если время есть):**
- IEEE-CIS Fraud Detection dataset
- Credit Card Fraud dataset
- Преобразование: fraud_flag → survival time (first fraud → detection delay)

### Что кодировать

#### 3.1 `src/survival/synthetic_data.py`
```
generate_fraud_timelines(n=500, seed=42) → pd.DataFrame
  Колонки: entity_id, lifetime, detected (bool), velocity, amount_std,
           unique_merchants, tx_interval_mean, tx_count

generate_observed_snapshot(timelines, observation_point) → pd.DataFrame
  Добавляет: n_observed (сколько транзакций видим к моменту observation)
```

**Constraints:**
- Weibull distribution для lifetime (shape=1.5, scale=200)
- Covariates коррелированы с lifetime (velocity↑ → lifetime↓)
- Censoring random 30%

#### 3.2 `src/survival/doomsday.py`
```
fit_weibull_prior(historical_lifetimes) → WeibullFitter
  Fit Weibull на исторических данных (observed lifetimes)

doomsday_posterior(n_observed, weibull_prior, N_range) → np.array
  P(N_total | n_observed) = P(n|N) * P(N) / P(n)
  P(n|N) = 1/N (random observer assumption)
  P(N) = weibull_prior.pdf(N)
  Return: normalized posterior over N_range

predict_remaining(n_observed, weibull_prior) → dict
  median_remaining: int
  ci_80: tuple[int, int]  (10th, 90th percentile)
  doomsday_percentile: float  (where are we in the lifecycle? 0-1)
```

#### 3.3 `src/survival/fraud_survival.py`
```
fit_and_compare(df) → dict
  1. Fit CoxPH baseline (velocity, amount_std, unique_merchants, tx_interval)
  2. Add doomsday_percentile as feature → fit CoxPH+Doomsday
  3. Compare: C-index, log-likelihood, AIC
  4. Return: {baseline_cindex, doomsday_cindex, improvement_pct, p_value}
```

### Тесты

```
tests/test_survival.py:
  - test_synthetic_data_shape (500 rows, required columns)
  - test_synthetic_censoring_rate (~30%)
  - test_weibull_prior_fit (shape > 0, scale > 0)
  - test_doomsday_posterior_sums_to_one
  - test_doomsday_posterior_penalizes_large_N
  - test_doomsday_percentile_range (0-1)
  - test_cox_baseline_runs
  - test_cox_doomsday_improvement (C-index ≥ baseline)
  - test_predict_remaining_ci (ci_80[0] < median < ci_80[1])
```

### Gate 3 criteria
| Метрика | GO | NO-GO |
|---------|-----|--------|
| Doomsday C-index improvement | ≥ 10% over baseline Cox | < 5% |
| Doomsday posterior calibration | KS p > 0.05 vs actual | p < 0.01 |
| LPPLS-like parameters extractable | (m, ω) analog exists | No analog |

**NO-GO action:** Drop fraud domain, paper = finance + geology only (2 domains).

### Effort: 8-12 часов

---

## ЭТАП 3.5 — Critical Slowing Down (2-3 часа, параллельно с Этапом 3)

### Цель
Добавить model-independent physical early warning signal. 5-й contribution для paper.

### Научное обоснование
Перед критическим переходом системы замедляют возврат к равновесию: variance растёт, autocorrelation растёт, recovery rate падает. Это универсальный физический предвестник, не зависящий от LPPLS parametric form.

### Что кодировать

#### `src/ews/critical_slowing.py`
```
compute_ews(series, window=50, detrend="linear") → EWSResult
  Features:
    - rolling_variance: np.array
    - rolling_ac1: np.array  (lag-1 autocorrelation)
    - rolling_skewness: np.array
    - recovery_rate: np.array  (1/AR(1) coefficient)

  Derived:
    - kendall_tau_variance: float  (trend in variance, >0 = approaching transition)
    - kendall_tau_ac1: float  (trend in AC1, >0 = slowing down)
    - ews_score: float  (0-1, composite)
    - is_slowing: bool  (tau_var > 0.3 AND tau_ac1 > 0.3)

class EWSResult:
    features: dict[str, np.array]
    kendall_tau_variance: float
    kendall_tau_ac1: float
    ews_score: float
    is_slowing: bool
```

**Формулы:**
```
variance(t) = Var(x[t-w:t])  (rolling window w)
ac1(t) = Corr(x[t-w:t-1], x[t-w+1:t])  (lag-1 autocorrelation)
skewness(t) = Skew(x[t-w:t])
kendall_tau = scipy.stats.kendalltau(time_index, metric)[0]
ews_score = 0.5 * norm(tau_var) + 0.3 * norm(tau_ac1) + 0.2 * norm(tau_skew)
```

### Тесты
```
tests/test_ews.py:
  - test_constant_series_no_slowing (variance flat → tau ≈ 0)
  - test_synthetic_approaching_bifurcation (variance↑ → tau > 0)
  - test_ews_score_range (0-1)
  - test_known_bubbles_show_slowing (BTC 2017, Dotcom → is_slowing=True)
  - test_negative_controls_no_slowing (S&P 2013 → is_slowing=False)
```

### Effort: 2-3 часа

---

## ЭТАП 4 — Adversarial AI Validation (Неделя 6)

### Цель
CogniRouter adversarial council дебатирует каждый phase transition prediction. Measure: precision improvement ≥15%.

### Архитектура

```
LPPLS/HMM/EWS prediction
        ↓
CogniRouter API call (localhost:8000)
        ↓
┌───────────────────────────────────┐
│ Bull Agent (qwen3:8b, temp=0.3)  │ → "Yes, this is a real transition"
│ Bear Agent (gemma2:9b, temp=0.5) │ → "No, this is noise/overfit"
│ Skeptic (qwen3:8b, temp=0.4)     │ → "Check residuals, check R²"
│ Arbiter (qwen3:8b, temp=0.2)     │ → Final verdict + confidence
└───────────────────────────────────┘
        ↓
CouncilVerdict: CONFIRMED / REJECTED / UNCERTAIN
```

### Что кодировать

#### 4.1 `src/validation/council_validator.py`
```
run_council(prediction: dict, evidence: dict) → CouncilVerdict
  1. Format prediction as structured prompt (tc, m, ω, R², confidence, EWS)
  2. POST to CogniRouter /tasks with agent=contrafactual
  3. Parse response → verdict

class CouncilVerdict:
    verdict: "CONFIRMED" | "REJECTED" | "UNCERTAIN"
    confidence: float
    bull_argument: str
    bear_argument: str
    arbiter_reasoning: str
    council_mode: "fast" | "full"
```

**Если CogniRouter недоступен:** fallback к rule-based validation:
- R² < 0.5 → REJECTED
- B > 0 (not bubble) → REJECTED
- EWS + LPPLS agree → CONFIRMED
- EWS + LPPLS disagree → UNCERTAIN

#### 4.2 `src/validation/experiment_runner.py`
```
run_experiment(
    bubbles: list[BubbleDataset],
    controls: list[BubbleDataset],
    use_council: bool = True
) → ExperimentSummary

class ExperimentSummary:
    n_total: int
    tp: int  (true positive: bubble detected correctly)
    fp: int  (false positive: non-bubble flagged)
    fn: int  (false negative: bubble missed)
    tn: int  (true negative: non-bubble correctly ignored)
    precision: float
    recall: float
    f1: float
    precision_improvement: float  (vs without council)
    fp_reduction: float  (vs without council)
```

### Experiment design
```
Dataset: 6 known bubbles + 6 negative controls + 6 random periods (noise) = 18 total

Run 1: LPPLS+HMM ensemble only (no council) → baseline precision/recall
Run 2: LPPLS+HMM+Council (full path) → enhanced precision/recall
Compare: precision_improvement = (precision_run2 - precision_run1) / precision_run1
```

### Тесты
```
tests/test_council.py:
  - test_council_verdict_schema
  - test_fallback_rule_based
  - test_confirmed_on_strong_signal (R²>0.8, B<0, EWS agrees)
  - test_rejected_on_noise (R²<0.3, B>0)
  - test_experiment_runner_counts (tp+fp+fn+tn = total)
```

### Gate 4 criteria
| Метрика | GO | NO-GO |
|---------|-----|--------|
| Precision improvement | ≥ 15% | < 5% |
| False positive reduction | ≥ 20% | < 10% |
| Council adds latency | < 30 sec | > 60 sec |

**NO-GO action:** Drop adversarial angle, paper = multi-window + HMM + cross-domain (без council).

### Effort: 8-12 часов

---

## ЭТАП 5 — Cross-Domain Correlation (Неделя 7-8)

### Цель
THE MAIN FINDING. Параметры (m, ω) фазовых переходов коррелируют между доменами?

### Центральный вопрос
> Существует ли универсальный "fingerprint" фазового перехода, инвариантный относительно домена?

### Что кодировать

#### 5.1 `src/cross_domain/correlation.py`
```
class DomainParams:
    domain: str  ("finance" | "geology" | "fraud")
    m_values: list[float]
    omega_values: list[float]
    n_samples: int

full_cross_domain_analysis(
    finance: DomainParams,
    geology: DomainParams,
    fraud: DomainParams | None  # может быть None если Gate 3 NO-GO
) → CrossDomainReport

class CrossDomainReport:
    # Pairwise KS tests
    ks_m_finance_geo: KSResult  (statistic, p_value)
    ks_omega_finance_geo: KSResult
    ks_m_finance_fraud: KSResult | None
    ks_omega_finance_fraud: KSResult | None
    ks_m_geo_fraud: KSResult | None
    ks_omega_geo_fraud: KSResult | None

    # Mann-Whitney U tests
    mw_m_finance_geo: MWResult
    mw_omega_finance_geo: MWResult

    # Overlap score (0-1)
    parameter_overlap_score: float

    # Verdict
    universality_verdict: "SUPPORTED" | "PARTIAL" | "REJECTED"
    # SUPPORTED: all KS p > 0.05 (distributions not significantly different)
    # PARTIAL: some domains match, others don't
    # REJECTED: all KS p < 0.01
```

#### 5.2 `src/cross_domain/universality.py`
```
robustness_check(
    finance: DomainParams,
    geology: DomainParams,
    n_bootstrap: int = 1000
) → RobustnessResult
  Bootstrap resampling → confidence interval on overlap score

class RobustnessResult:
    overlap_mean: float
    overlap_ci_95: tuple[float, float]
    n_bootstrap: int
    is_robust: bool  (CI does not include 0)
```

### Визуализация (для paper)
```
Figure 1: 2D scatter plot (m, ω) по доменам (цвет = домен)
Figure 2: Violin plots of m distribution per domain
Figure 3: Violin plots of ω distribution per domain
Figure 4: Overlap heatmap (domain × domain)
```

### Тесты
```
tests/test_cross_domain.py:
  - test_identical_distributions_high_pvalue
  - test_different_distributions_low_pvalue
  - test_overlap_score_range (0-1)
  - test_bootstrap_ci_contains_mean
  - test_verdict_logic (SUPPORTED/PARTIAL/REJECTED)
```

### Gate 5 criteria
| Метрика | Verdict |
|---------|---------|
| All KS p > 0.05 | **SUPPORTED** — universality confirmed |
| Some KS p > 0.05 | **PARTIAL** — domain-specific signatures (also publishable) |
| All KS p < 0.01 | **REJECTED** — domains are different (still publishable as negative result) |

**Все 3 verdict'а publishable.** Нет NO-GO — результат информативен в любом случае.

### Effort: 8-12 часов

---

## ЭТАП 6 — Paper (Неделя 9-10)

### Цель
arXiv preprint → workshop/journal submission.

### Target venues (по приоритету)

| Venue | Type | Deadline | Fit |
|-------|------|----------|-----|
| arXiv physics.soc-ph | Preprint | Anytime | 100% |
| Chaos, Solitons & Fractals | Journal | Rolling | Best fit |
| Physica A | Journal | Rolling | Good fit |
| NeurIPS "AI for Science" | Workshop | ~June 2026 | If adversarial works |
| Quantitative Finance | Journal | Rolling | If finance focus |

### Paper structure

```
Title: "PhaseBreak: Cross-Domain Phase Transition Detection
        with Adversarial AI Validation"

1. Introduction (1 page)
   - Phase transitions in complex systems (Sornette, Scheffer)
   - Gap: no cross-domain unified framework with AI validation
   - 5 contributions listed

2. Related Work (0.5 page)
   - LPPLS (Sornette 2003, 2015)
   - EWS / Critical slowing down (Scheffer 2009)
   - HMM for regime detection
   - AI for financial prediction

3. Method (2.5 pages)
   - 3.1 LPPLS model + multi-window DS Confidence Indicator
   - 3.2 HMM-gated ensemble (novel)
   - 3.3 Certified convergence bounds (Galkin-Remizov 2025)
   - 3.4 Critical slowing down layer
   - 3.5 Cross-domain adaptation (geological/fraud bounds)
   - 3.6 Adversarial AI validation (CogniRouter council)

4. Experiments (2.5 pages)
   - 4.1 Finance: 6 bubbles + 6 controls → precision/recall table
   - 4.2 Geology: Sentinel-2 temporal LPPLS → R², (m,ω) comparison
   - 4.3 Fraud: Doomsday survival → C-index improvement
   - 4.4 Adversarial: with/without council → precision gain
   - 4.5 Cross-domain: KS tests, overlap score, universality verdict

5. Results (1 page)
   - Table 1: Finance precision/recall (LPPLS vs HMM+LPPLS vs ensemble+council)
   - Table 2: Cross-domain (m, ω) comparison (KS p-values)
   - Figure 1: (m, ω) scatter by domain
   - Figure 2: ROC curves per method

6. Discussion (0.5 page)
   - Limitations: small sample, curated datasets, LPPLS optimizer instability
   - Implications: universal phase transition theory across domains
   - Future: Koopman/HAVOK, topology, network contagion

7. Conclusion (0.25 page)

References: 25-35 papers
Appendix: hyperparameters, negative control details, ablation table
```

### 5 contributions для paper

| # | Contribution | Status | Evidence |
|---|-------------|--------|----------|
| 1 | Multi-window DS LPPLS CI | ✅ Done | 4/6 bubbles detected, 0 FP |
| 2 | HMM-gated LPPLS ensemble | ✅ Done | Precision 80%, Recall 67% |
| 3 | Certified convergence bounds (Galkin-Remizov) | ✅ Done | ChernoffPy integration |
| 4 | Critical slowing down layer | 🔄 Stage 3.5 | Independent EWS |
| 5 | Cross-domain universality | 🔄 Stage 5 | KS preliminary: p=0.085/0.222 |
| +6 | Adversarial AI validation | 🔄 Stage 4 | If precision +15% → 6th contribution |

### Ablation table (для paper)

```
| Method                    | Precision | Recall | FP Rate | Lead Time |
|---------------------------|-----------|--------|---------|-----------|
| LPPLS only                | ?%        | ?%     | ?%      | ? days    |
| LPPLS + multi-window      | ?%        | ?%     | ?%      | ? days    |
| LPPLS + HMM gate          | 80%       | 67%    | 8.3%    | 1-18 days |
| LPPLS + HMM + EWS         | ?%        | ?%     | ?%      | ? days    |
| LPPLS + HMM + EWS + Council | ?%     | ?%     | ?%      | ? days    |
```

### Effort: 2 недели (writing + figures + review)

---

## ОБЩАЯ TIMELINE

```
Неделя 5:     Stage 3 (fraud) + Stage 3.5 (CSD) параллельно
Неделя 6:     Stage 4 (adversarial validation)
Неделя 7-8:   Stage 5 (cross-domain correlation)
Неделя 9-10:  Stage 6 (paper writing)
```

## BUDGET

| Ресурс | Стоимость |
|--------|-----------|
| Compute | $0 (RTX 5070 Ti + Ollama local) |
| CogniRouter API | $0 (localhost, Ollama) |
| Data | $0 (yfinance, synthetic, Geoscan existing) |
| arXiv | $0 |
| Journal submission | $0 (open access not required for first submission) |
| **Total** | **$0** |

## RISK MATRIX

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Fraud Gate 3 NO-GO | 30% | Medium | Paper = 2 domains (finance + geology) |
| Adversarial Gate 4 NO-GO | 40% | Low | Drop council, keep 5 contributions |
| Universality REJECTED | 20% | Low | Publish as negative result (still valuable) |
| LPPLS optimizer instability | 20% | Medium | Multi-window CI already mitigates |
| Reviewer asks for more domains | 50% | Medium | Add "Future Work" section |
| Paper rejected | 30% | Medium | Revise → resubmit to different venue |
