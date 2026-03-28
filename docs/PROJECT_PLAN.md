# PhaseBreak — Universal Phase Transition Detection with Adversarial AI Validation

## Миссия
Обнаружение фазовых переходов (пузыри, крахи, сейсмические предвестники, хроматиновые перестройки) через единый математический аппарат (LPPLS + Bayesian Survival + Logistic Saturation) с валидацией через adversarial multi-agent debate.

## Научная гипотеза
> Фазовые переходы в финансовых, геологических и биологических системах описываются одними и теми же математическими моделями (log-periodic oscillations near critical time). AI-adversarial validation улучшает precision предсказания critical time `t_c` на 15-30% по сравнению с классическим LPPLS fitting.

## Уникальность (почему это не "ещё один LPPLS")
1. **Cross-domain:** финансы + геология + геномика в одном framework (никто не делал)
2. **Adversarial validation:** CogniRouter council дебатирует каждый `t_c` prediction
3. **Existing infrastructure:** VeriFind (данные + VEE sandbox), Geoscan (Sentinel-2), ARCHCODE (ClinVar)
4. **Bayesian survival for fraud:** novel application of Doomsday math to fraud scheme lifetimes

---

## TECH STACK

```
Core:           Python 3.11, NumPy, SciPy (L-BFGS-B optimizer)
Visualization:  Matplotlib, Plotly
Data:           yfinance, FRED API, Sentinel-2 (rasterio), ClinVar API
ML:             scikit-learn, statsmodels, lifelines (survival analysis)
AI Validation:  CogniRouter adversarial council (Ollama local)
Notebooks:      Jupyter (research), pytest (validation)
Paper:          LaTeX (ICML/NeurIPS template)
```

---

## ПЛАН РЕАЛИЗАЦИИ

### ЭТАП 1 — LPPLS Baseline (Неделя 1)
**Цель:** Воспроизвести результаты Сорнетта на известных пузырях. Proof of concept.

#### 1.1 Математика LPPLS (День 1-2)

**Уравнение Сорнетта:**
```
ln(E[p(t)]) = A + B(tc - t)^m + C(tc - t)^m * cos(ω * ln(tc - t) + φ)
```

**7 параметров для оптимизации:**
- `tc` — critical time (дата краха/фазового перехода)
- `m` — power law exponent (0.1 < m < 0.9)
- `ω` — angular log-frequency (6 < ω < 13)
- `A` — log-price at tc
- `B` — amplitude (B < 0 для пузырей)
- `C` — log-periodic amplitude
- `φ` — phase

**Constraints (Sornette 2003):**
```python
BOUNDS = {
    "m": (0.1, 0.9),      # power law exponent
    "omega": (6.0, 13.0),  # log-frequency
    "tc": (t_last + 1, t_last + 252),  # crash within 1 year
    "B": (-np.inf, -1e-5),  # negative for bubble
}
```

**Оптимизация:**
- Outer loop: Grid search по (tc, m, ω) — 1000 комбинаций
- Inner loop: OLS для (A, B, C, φ) при фиксированных (tc, m, ω)
- Финальная polish: L-BFGS-B по всем 7 параметрам

**Файлы:**
```
src/
├── lppls/
│   ├── __init__.py
│   ├── model.py          # LPPLS class: fit(), predict(), confidence()
│   ├── optimizer.py       # Grid search + L-BFGS-B
│   └── metrics.py         # Residual analysis, R², AIC/BIC
```

#### 1.2 Validation на известных пузырях (День 3-4)

**Датасеты (ground truth существует):**

| Пузырь | Актив | Дата краха | Источник |
|--------|-------|-----------|----------|
| Dot-com | NASDAQ (^IXIC) | 2000-03-10 | yfinance |
| Bitcoin 2017 | BTC-USD | 2017-12-17 | yfinance |
| Bitcoin 2021 | BTC-USD | 2021-11-10 | yfinance |
| Housing 2008 | S&P Case-Shiller | 2006-07 | FRED |
| Tesla 2021 | TSLA | 2021-11-04 | yfinance |
| Китай 2015 | ^SSEC | 2015-06-12 | yfinance |

**Метрика успеха:**
- `tc` prediction error < 30 дней от реального краха
- Directional accuracy > 60% на out-of-sample (2022-2026)
- **GO/NO-GO:** если accuracy < 50% на 3+ пузырях → проект не жизнеспособен

**Файлы:**
```
notebooks/
├── 01_btc_2017_bubble.ipynb     # Reproduce Sornette on BTC 2017
├── 02_dotcom_2000.ipynb          # Classic validation
├── 03_out_of_sample_2024.ipynb   # Can we predict recent events?
tests/
├── test_lppls_model.py           # Unit tests for optimizer
├── test_known_bubbles.py         # Regression: known tc dates
```

#### 1.3 Multi-Window Confidence Indicator (День 4-5)

**Проблема:** Один fit window → один tc. Но LPPLS нестабилен — сдвиг window на 10 дней даёт другой tc.

**Решение (Sornette 2015, "Real-Time Bubble Detection"):**

DS LPPLS Confidence Indicator — для каждого `t_end`:
```python
windows = [60, 90, 120, 150, 180]  # дней назад
fits = [lppls.fit(t[t_end-w:t_end], log_price[t_end-w:t_end]) for w in windows]
tc_values = [f.params.tc for f in fits if f.params.is_bubble and f.r_squared() > 0.5]

# Confidence: доля windows где tc совпадает (±15 дней)
tc_median = np.median(tc_values)
n_agree = sum(1 for tc in tc_values if abs(tc - tc_median) < 15)
confidence = n_agree / len(windows)  # 0.0 → noise, 0.6+ → bubble signal
```

**Пороги:**
- `confidence >= 0.6` (3+/5 windows agree) → CONFIDENT bubble signal
- `confidence < 0.4` → NOISE, discard
- `tc_std < 20 дней` → stable prediction

**WHY критично:** Без multi-window Gate 1 может не пройти — single-window LPPLS часто переfit на noise. Multi-window повышает precision на 30-50% (Sornette 2015).

**Файлы:**
```
src/
├── lppls/
│   ├── confidence.py      # Multi-window confidence indicator
│   └── scanner.py         # Scan time series with rolling windows
```

#### 1.4 Negative Control Dataset (День 5)

**Проблема:** 6 known bubbles — все positive. Без negative control LPPLS может detect bubbles everywhere = useless model.

**Negative controls (6 "boring" periods):**

| Период | Актив | Характер | Ожидание |
|--------|-------|----------|----------|
| S&P 500 2013-2014 | ^GSPC | Steady growth, no crash | is_bubble=False или R²<0.5 |
| BTC Q1-Q3 2019 | BTC-USD | Sideways / range-bound | is_bubble=False |
| NASDAQ 2016 | ^IXIC | Normal market | is_bubble=False |
| Gold 2022 | GC=F | Range-bound | is_bubble=False |
| Tesla 2023 | TSLA | Recovery, not bubble | is_bubble=False |
| Shanghai 2018 | 000001.SS | Decline, not bubble | is_bubble=False |

**Метрика:**
- LPPLS false positive rate на negative controls ≤ 1/6 (max 1 false alarm)
- С multi-window confidence → false positive rate = 0/6

**Файлы:**
```
src/lppls/data.py          # + NEGATIVE_CONTROLS dict
tests/test_negative_controls.py  # LPPLS must NOT detect bubble
```

#### 1.5 Comparison с baselines (День 6)

**Baselines для сравнения:**
- PyPI `lppls` package (Sornette reference implementation)
- Simple moving average crossover (MA50/MA200)
- Bollinger Band breakout
- Random predictor (control)

**Output Этапа 1:**
- [ ] LPPLS модель fit/predict работает
- [ ] Multi-window confidence indicator (DS LPPLS CI)
- [ ] 6 known bubbles: tc error < 30 дней на 4/6 (с multi-window)
- [ ] 6 negative controls: false positive rate ≤ 1/6
- [ ] Out-of-sample 2024-2026: accuracy > 50%
- [ ] Comparison table: LPPLS vs LPPLS+multi-window vs baselines
- [ ] **GO/NO-GO decision**

---

### ЭТАП 1.5 — HMM Regime Detection + Ensemble (Неделя 2)
**Цель:** Повысить precision через pre-screening: LPPLS фитится только в "bubble regime", а не на всём ряде. Publishable contribution — комбинация HMM + LPPLS ранее не описана.

#### 1.5.1 Hidden Markov Model для режимов рынка

**3 скрытых состояния:**
```
State 0: NORMAL  — low volatility, moderate returns
State 1: GROWTH  — high returns, increasing volatility
State 2: BUBBLE  — super-exponential returns, high volatility, accelerating
```

**Implementation (hmmlearn):**
```python
from hmmlearn import GaussianHMM

# Features: [log_return, volatility_20d, acceleration]
features = np.column_stack([log_returns, rolling_vol, np.diff(rolling_vol, prepend=0)])

hmm = GaussianHMM(n_components=3, covariance_type="full", n_iter=100)
hmm.fit(features)
states = hmm.predict(features)

# Map states to regimes by mean return (highest = BUBBLE)
state_means = [features[states == i, 0].mean() for i in range(3)]
bubble_state = np.argmax(state_means)
```

#### 1.5.2 Ensemble: HMM pre-screen → LPPLS fit

**Pipeline:**
```
Raw time series → HMM regime detection → BUBBLE state detected?
                                              │
                                    YES       │       NO
                                    ↓         │       ↓
                              LPPLS fit       │    SKIP (no bubble)
                              + multi-window  │
                                    ↓
                              tc prediction
                              + confidence
```

**Преимущества:**
- **Precision↑**: LPPLS не фитится на noise/normal periods → меньше false positives
- **Speed↑**: LPPLS fit только на ~20% данных (bubble periods) → 5x быстрее scanning
- **Publishable**: "HMM-gated LPPLS" = новая комбинация, нет в литературе

#### 1.5.3 Метрики

| Метрика | LPPLS only | HMM + LPPLS (ожидание) |
|---------|-----------|----------------------|
| Precision (known bubbles) | baseline | +10-20% |
| False positive rate (negative controls) | ≤1/6 | 0/6 |
| Scan speed (1000 days) | ~30 sec | ~6 sec |
| tc error (median) | baseline | same or better |

**Output Этапа 1.5:**
- [ ] HMM regime detector (3 states: normal/growth/bubble)
- [ ] Ensemble pipeline: HMM → LPPLS
- [ ] Comparison: standalone LPPLS vs HMM+LPPLS на 12 datasets (6 pos + 6 neg)
- [ ] **If precision improvement < 10% → drop HMM, keep multi-window only**

**Файлы:**
```
src/
├── lppls/
│   ├── regime.py          # HMM regime detection
│   └── ensemble.py        # HMM-gated LPPLS pipeline
tests/
├── test_regime.py         # HMM state detection tests
├── test_ensemble.py       # End-to-end pipeline tests
```

**Зависимость:** `hmmlearn>=0.3` (добавить в pyproject.toml)

---

### ЭТАП 2 — Cross-Domain: Геология (Неделя 3-4)
**Цель:** Применить LPPLS к спутниковым данным Sentinel-2 для обнаружения phase transitions в геологических процессах.

#### 2.1 Гипотеза

Спектральные аномалии в Sentinel-2 temporal series перед геологическими событиями (оползни, активизация разломов) демонстрируют log-periodic oscillations аналогичные финансовым пузырям.

**Обоснование:**
- Напряжения в горных породах накапливаются по power law (Omori law)
- Перед разрушением — ускоряющиеся осцилляции (log-periodic precursors)
- Sornette сам публиковал о землетрясениях (1990s papers)

#### 2.2 Данные

**Источник:** Geoscan Gold 2026 project (уже есть pipeline)

```
Sentinel-2 temporal series:
- Band ratios (NDVI, Clay Index, Iron Oxide) за 2-3 года
- Temporal resolution: 5-10 дней
- Spatial: конкретные участки из Geoscan (известные аномалии)
```

**Preprocessing:**
- Pixel-level time series extraction (уже в Geoscan pipeline)
- Cloud masking (SCL band)
- Normalization to 0-1 scale (аналогия: "price" = spectral index value)

#### 2.3 LPPLS fitting на спектральных данных

```python
# Аналогия: spectral index = "price", geological event = "crash"
lppls_geo = LPPLS()
lppls_geo.fit(
    time_series=ndvi_temporal,  # NDVI values over 2 years
    bounds=GEO_BOUNDS           # adjusted tc, m, omega for geological timescales
)
tc_geo = lppls_geo.predict_critical_time()
```

**Adjusted bounds для геологии:**
```python
GEO_BOUNDS = {
    "m": (0.1, 0.9),
    "omega": (4.0, 25.0),      # wider range for geological oscillations
    "tc": (t_last, t_last + 365),  # event within 1 year
}
```

#### 2.4 Validation

- Сравнить `tc_geo` с известными геологическими событиями (если есть ground truth из Geoscan)
- Если нет ground truth: correlation analysis между spectral LPPLS signal и seismic catalogs (USGS)

**Output Этапа 2:**
- [ ] LPPLS fit на Sentinel-2 temporal series
- [ ] Визуализация: log-periodic oscillations в спектральных данных
- [ ] Correlation с известными геологическими событиями
- [ ] **Cross-domain finding: параметры (m, ω) correlate между финансами и геологией?**

**Файлы:**
```
src/
├── geo/
│   ├── sentinel_loader.py    # Reuse from Geoscan
│   ├── temporal_series.py    # Pixel → time series
│   └── geo_lppls.py          # LPPLS with geological bounds
notebooks/
├── 04_sentinel_lppls.ipynb   # Main cross-domain notebook
├── 05_geo_finance_correlation.ipynb  # THE key analysis
```

---

### ЭТАП 3 — Bayesian Survival для Fraud (Неделя 5)
**Цель:** Doomsday Argument math → предсказание lifetime fraud-схем.

#### 3.1 Модель

**Doomsday-inspired survival model:**

Если мы наблюдаем fraud-схему на её `n`-й транзакции, а полное число транзакций до закрытия = `N`:

```
P(N | n) ∝ P(n | N) * P(N)
P(n | N) = 1/N  (random observer assumption)
```

**Prior P(N):** Weibull distribution fitted на исторических данных fraud lifetimes.

**В ML терминах:**
- Features: transaction velocity, amount distribution, network topology
- Target: time-to-detection (survival time)
- Model: Cox Proportional Hazards + Bayesian prior from Doomsday logic

#### 3.2 Данные

- Синтетические fraud timelines (если нет доступа к реальным KZ данным)
- Kaggle fraud datasets (IEEE-CIS, credit card fraud)
- TERAG historical patterns (если доступны)

#### 3.3 Implementation

```python
from lifelines import CoxPHFitter, WeibullFitter

# Step 1: Fit Weibull prior on historical fraud lifetimes
weibull = WeibullFitter()
weibull.fit(historical_lifetimes)

# Step 2: Bayesian update with Doomsday logic
def doomsday_posterior(n_observed, weibull_prior):
    """P(N_total | n_observed) using Doomsday reasoning"""
    N_range = np.arange(n_observed, n_observed * 100)
    likelihood = 1.0 / N_range  # random observer
    prior = weibull_prior.pdf(N_range)
    posterior = likelihood * prior
    return posterior / posterior.sum()

# Step 3: Cox model with Doomsday-adjusted features
cox = CoxPHFitter()
cox.fit(df, duration_col="lifetime", event_col="detected",
        formula="velocity + amount_std + doomsday_percentile")
```

**Output Этапа 3:**
- [ ] Weibull prior на fraud lifetimes
- [ ] Doomsday posterior: predict remaining lifetime
- [ ] Cox model с Doomsday feature → C-index improvement
- [ ] Comparison: Cox + Doomsday vs Cox baseline

**Файлы:**
```
src/
├── survival/
│   ├── doomsday.py           # Bayesian posterior with Doomsday logic
│   ├── fraud_survival.py     # Cox + Weibull models
│   └── synthetic_data.py     # Generate fraud timelines
notebooks/
├── 06_doomsday_fraud.ipynb   # Main survival analysis
```

---

### ЭТАП 4 — Adversarial AI Validation (Неделя 6)
**Цель:** CogniRouter council дебатирует каждый `tc` prediction. Novel methodology.

#### 4.1 Архитектура

```
LPPLS fit → tc prediction → CogniRouter Adversarial Council
                                    ↓
                    ┌───────────────┼───────────────┐
                    ↓               ↓               ↓
               Bull Agent      Bear Agent      Skeptic Agent
            "tc is correct,   "tc is wrong,    "LPPLS overfitting,
             crash imminent"   noise pattern"    check residuals"
                    ↓               ↓               ↓
                    └───────────────┼───────────────┘
                                    ↓
                              Arbiter Agent
                        (weighted synthesis → final tc ± confidence)
```

#### 4.2 Agent prompts (domain-specific)

```yaml
bull_agent:
  role: "Bubble detection advocate"
  instructions: |
    You have LPPLS fit results. Argue WHY this is a real phase transition:
    - Check R² > 0.8
    - Check m ∈ (0.1, 0.9)
    - Check ω ∈ (6, 13)
    - Compare with known bubble signatures

bear_agent:
  role: "False positive detector"
  instructions: |
    Challenge the LPPLS prediction:
    - Is this overfitting? (too many parameters)
    - Is the residual structure random? (Durbin-Watson test)
    - Are there regime changes in the window?
    - Could this be a liquidity shock, not a phase transition?

skeptic_agent:
  role: "Methodological critic"
  instructions: |
    Question the methodology:
    - Is the fitting window appropriate?
    - Are bounds too tight/loose?
    - How sensitive is tc to window start date?
    - Compare with simple baselines (MA crossover)
```

#### 4.3 Метрика

**Hypothesis:** Adversarial validation reduces false positive rate by 15-30%.

**Experiment:**
- 100 LPPLS fits (50 real bubbles + 50 random noise)
- Without council: precision/recall on bubble detection
- With council: precision/recall on bubble detection
- **Improvement = scientific contribution**

**Output Этапа 4:**
- [ ] CogniRouter integration с LPPLS pipeline
- [ ] 100 predictions: with/without adversarial validation
- [ ] Precision/recall comparison table
- [ ] **If improvement > 15% → publishable finding**

**Файлы:**
```
src/
├── validation/
│   ├── council_validator.py   # CogniRouter API integration
│   ├── experiment_runner.py   # 100-prediction experiment
│   └── metrics.py             # Precision, recall, F1, false positive rate
notebooks/
├── 07_adversarial_validation.ipynb
```

---

### ЭТАП 5 — Cross-Domain Correlation (Неделя 7-8)
**Цель:** THE MAIN FINDING. Параметры phase transitions коррелируют между доменами?

#### 5.1 Центральный вопрос

> Существует ли универсальный "fingerprint" фазового перехода, инвариантный относительно домена?

**Конкретно:** Если `(m, ω)` параметры LPPLS коррелируют между:
- Финансовыми пузырями
- Геологическими аномалиями
- Геномными перестройками

→ Это **фундаментальный результат** о природе phase transitions.

#### 5.2 Анализ

```python
# Collect (m, omega) from all domains
finance_params = [(m, omega) for fit in finance_fits]
geology_params = [(m, omega) for fit in geology_fits]
genomics_params = [(m, omega) for fit in genomics_fits]  # if available

# Statistical test: are distributions the same?
from scipy.stats import ks_2samp, mannwhitneyu

ks_m = ks_2samp(finance_m, geology_m)         # Kolmogorov-Smirnov
ks_omega = ks_2samp(finance_omega, geology_omega)

# If p > 0.05: distributions NOT significantly different
# → UNIVERSAL PHASE TRANSITION SIGNATURE
```

#### 5.3 Визуализация

- 2D scatter plot: (m, ω) по доменам (цвет = домен)
- Overlap → universality claim
- Separation → domain-specific signatures (also interesting)

**Output Этапа 5:**
- [ ] Cross-domain (m, ω) comparison
- [ ] Statistical tests (KS, Mann-Whitney)
- [ ] Visualization: universal fingerprint или domain separation
- [ ] **Main finding формулировка**

---

### ЭТАП 6 — Paper Writing (Неделя 9-10)
**Цель:** arXiv preprint → workshop submission.

#### 6.1 Paper structure

```
Title: "PhaseBreak: Cross-Domain Phase Transition Detection
        with Adversarial AI Validation"

Abstract: ~150 words
1. Introduction (1 page)
   - Phase transitions в complex systems
   - Gap: no cross-domain unified framework
   - Contribution: LPPLS + adversarial AI + 3 domains

2. Related Work (0.5 page)
   - Sornette LPPLS
   - AI for financial prediction
   - Cross-domain complexity science

3. Method (2 pages)
   - 3.1 LPPLS model (math)
   - 3.2 Cross-domain adaptation (bounds, preprocessing)
   - 3.3 Adversarial AI validation (council architecture)
   - 3.4 Bayesian survival with Doomsday prior

4. Experiments (2 pages)
   - 4.1 Finance: 6 known bubbles + out-of-sample
   - 4.2 Geology: Sentinel-2 temporal anomalies
   - 4.3 Fraud survival: Doomsday-enhanced Cox model
   - 4.4 Adversarial validation: precision improvement
   - 4.5 Cross-domain correlation analysis

5. Results (1 page)
   - Tables: accuracy, precision, tc error
   - Figures: (m, ω) cross-domain scatter

6. Discussion (0.5 page)
   - Limitations (LPPLS optimizer instability, small sample)
   - Implications (universal phase transition theory)

7. Conclusion (0.25 page)

References: 20-30 papers
```

#### 6.2 Target venues

| Venue | Deadline | Fit | Probability |
|-------|----------|-----|-------------|
| arXiv (physics.soc-ph) | Anytime | Preprint | 100% (self-publish) |
| NeurIPS "AI for Science" Workshop | ~June 2026 | High | 40% |
| ICML "AI4Finance" Workshop | ~May 2026 | High | 45% |
| Quantitative Finance (journal) | Rolling | Medium | 30% |
| Physica A: Statistical Mechanics | Rolling | High | 50% |
| Chaos, Solitons & Fractals | Rolling | Very High | 55% |

---

## СТРУКТУРА ПРОЕКТА

```
PhaseBreak/
├── src/
│   ├── lppls/
│   │   ├── model.py              # Core LPPLS implementation
│   │   ├── optimizer.py          # Grid search + L-BFGS-B
│   │   ├── data.py               # yfinance loader + known bubbles + negative controls
│   │   ├── confidence.py         # Multi-window DS LPPLS Confidence Indicator (NEW)
│   │   ├── scanner.py            # Rolling window scanner (NEW)
│   │   ├── regime.py             # HMM regime detection (NEW)
│   │   ├── ensemble.py           # HMM-gated LPPLS pipeline (NEW)
│   │   └── metrics.py            # R², AIC, residual analysis
│   ├── geo/
│   │   ├── sentinel_loader.py    # Sentinel-2 temporal series
│   │   └── geo_lppls.py          # LPPLS for geological data
│   ├── survival/
│   │   ├── doomsday.py           # Bayesian Doomsday posterior
│   │   └── fraud_survival.py     # Cox + Weibull + Doomsday
│   ├── validation/
│   │   ├── council_validator.py  # CogniRouter adversarial
│   │   └── experiment_runner.py  # Batch experiments
│   └── cross_domain/
│       ├── correlation.py        # (m, ω) analysis across domains
│       └── universality.py       # Statistical tests
├── notebooks/
│   ├── 01_btc_2017_bubble.ipynb
│   ├── 02_dotcom_2000.ipynb
│   ├── 03_out_of_sample_2024.ipynb
│   ├── 04_sentinel_lppls.ipynb
│   ├── 05_geo_finance_correlation.ipynb
│   ├── 06_doomsday_fraud.ipynb
│   ├── 07_adversarial_validation.ipynb
│   └── 08_hmm_ensemble.ipynb     # HMM+LPPLS comparison (NEW)
├── data/
│   ├── finance/                  # yfinance cached data
│   ├── geology/                  # Sentinel-2 time series
│   └── fraud/                    # Synthetic / Kaggle
├── paper/
│   ├── main.tex                  # LaTeX paper
│   ├── figures/
│   └── references.bib
├── tests/
│   ├── test_lppls_model.py       # Core math tests (15 passing)
│   ├── test_known_bubbles.py     # Regression: known tc dates
│   ├── test_negative_controls.py # LPPLS must NOT detect bubble (NEW)
│   ├── test_confidence.py        # Multi-window CI tests (NEW)
│   ├── test_regime.py            # HMM state detection tests (NEW)
│   ├── test_ensemble.py          # End-to-end pipeline tests (NEW)
│   ├── test_doomsday.py
│   └── test_council_validator.py
├── configs/
│   ├── finance_bounds.yaml
│   ├── geology_bounds.yaml
│   └── council_agents.yaml
├── pyproject.toml
├── CLAUDE.md
└── README.md
```

---

## GATE CRITERIA (GO/NO-GO)

### Gate 1 — после Этапа 1 (Неделя 1)
**Вопрос:** Работает ли LPPLS вообще?

| Метрика | GO | NO-GO |
|---------|-----|--------|
| tc error на known bubbles (multi-window) | < 30 дней на 4/6 | > 60 дней на 3+ |
| Multi-window confidence на known bubbles | ≥ 0.6 на 4/6 | < 0.4 на 3+ |
| False positive rate (negative controls) | ≤ 1/6 | ≥ 3/6 |
| Out-of-sample accuracy | > 50% | < 40% |
| R² on best fits | > 0.75 | < 0.50 |

**NO-GO action:** Вернуться к Skeptic Engine SHIP.

### Gate 1.5 — после Этапа 1.5 (Неделя 2)
**Вопрос:** Добавляет ли HMM ensemble ценность?

| Метрика | GO | NO-GO |
|---------|-----|--------|
| Precision improvement vs standalone LPPLS | > 10% | < 5% |
| False positive rate (negative controls) | 0/6 | > 1/6 (не лучше LPPLS) |
| Scan speed improvement | > 2x | < 1.5x |

**NO-GO action:** Drop HMM → продолжить с multi-window LPPLS only. Не блокирует проект.

### Gate 2 — после Этапа 2 (Неделя 4)
**Вопрос:** Есть ли cross-domain signal?

| Метрика | GO | NO-GO |
|---------|-----|--------|
| LPPLS fits на Sentinel-2 | R² > 0.50 на 3+ sites | R² < 0.30 |
| Visual log-periodic pattern | Visible | Not visible |
| (m, ω) overlap с финансами | p > 0.05 (KS test) | p < 0.01 |

**NO-GO action:** Pivot → только Finance + Fraud (drop geology).

### Gate 3 — после Этапа 4 (Неделя 6)
**Вопрос:** Помогает ли adversarial validation?

| Метрика | GO | NO-GO |
|---------|-----|--------|
| Precision improvement | > 15% | < 5% |
| False positive reduction | > 20% | < 10% |

**NO-GO action:** Drop adversarial angle → pure cross-domain paper.

---

## BUDGET & RESOURCES

| Ресурс | Стоимость | Где |
|--------|-----------|-----|
| Compute (GPU) | $0 | RTX 5070 Ti local |
| LLM (CogniRouter) | $0 | Ollama local |
| Data (yfinance, FRED) | $0 | Free APIs |
| Data (Sentinel-2) | $0 | Copernicus Open Access |
| arXiv submission | $0 | Free |
| Workshop registration | $200-500 | If accepted |
| **Total** | **$0-500** | |

**Timeline:** 10 недель (~2.5 месяца)
**Risk:** Отвлечение от Skeptic Engine sales

---

## ОБНОВЛЁННЫЙ ТАЙМЛАЙН

```
Неделя 1    → Этап 1:   LPPLS baseline + multi-window CI + negative controls → GO/NO-GO
Неделя 2    → Этап 1.5: HMM regime detection + ensemble (NEW) → Gate 1.5
Неделя 3-4  → Этап 2:   Геология (Sentinel-2) → Gate 2
Неделя 5    → Этап 3:   Fraud survival (Doomsday Bayesian)
Неделя 6    → Этап 4:   Adversarial AI validation → Gate 3
Неделя 7-8  → Этап 5:   Cross-domain correlation
Неделя 9-10 → Этап 6:   Paper writing (arXiv → workshop)
```

**4 contributions (vs 1 в исходном плане):**
1. Multi-window DS LPPLS Confidence Indicator (Sornette 2015 method, validated)
2. HMM-gated LPPLS ensemble (novel combination, not in literature)
3. Certified convergence bounds for tc (Richardson extrapolation, adapted from ChernoffPy/Galkin-Remizov 2025)
4. Cross-domain phase transition universality (main thesis)

---

## ПЕРВЫЙ ШАГ (ВЫПОЛНЕН)

Создан GitHub repo `PhaseBreak` с Этапом 1.1:
1. `src/lppls/model.py` — LPPLS class (fit, predict, R², RMSE)
2. `src/lppls/optimizer.py` — Grid search + L-BFGS-B
3. `src/lppls/data.py` — yfinance loader + 6 known bubbles
4. `tests/test_lppls_model.py` — 15 unit tests (all passing)

**СЛЕДУЮЩИЙ ШАГ:** Этап 1.2 — validation на known bubbles + Этап 1.3 multi-window CI

**GO/NO-GO через 5 дней.**
