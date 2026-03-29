# ТЗ: PhaseBreak v2.0 — Что Реализуем Сейчас, Что Оставляем На Потом

**Дата:** 2026-03-29  
**Статус:** к реализации  
**Назначение:** передать Claude Code жесткий приоритетный план развития методологии без расползания scope.

---

## 1. Цель документа

На текущем этапе проект нужно усиливать **не количеством модулей**, а **качеством фундамента**:

- строже валидация;
- меньше риск overfitting;
- лучше uncertainty;
- понятнее разделение между operational detector и scientific claims;
- сильнее защита перед внешним reviewer.

Этот документ фиксирует:

1. что реализуем **сейчас**;
2. что реализуем **после этого**;
3. что **сознательно не делаем пока**.

---

## 2. Исходная позиция

На момент старта этого ТЗ проект уже имеет:

- 5 доменов: `finance`, `geology`, `housing`, `commodities`, `fraud`
- рабочее LPPLS-ядро
- tightened Sornette filters
- multi-window logic
- HMM-gated pipeline
- EWS exploratory layer
- cross-domain comparison
- growing validation suite

Главный риск следующего этапа:

- превратить сильный и честный research framework в перегруженную систему из слишком большого числа новых модулей;
- потерять ясность, что именно реально улучшает качество;
- получить сложность быстрее, чем доказательность.

Поэтому приоритет смещается в сторону:

- validation protocol,
- soft evidence,
- uncertainty,
- hard negatives,
- clean pipeline separation.

---

## 3. Главная цель PhaseBreak v2.0

Перевести проект из состояния:

`хороший набор работающих исследовательских модулей`

в состояние:

`строгая, калиброванная, защитимая методология раннего обнаружения предкризисных режимов`

Без:

- полной смены ядра,
- тяжелого Bayesian refactor,
- deep learning веток,
- premature production complexity.

---

## 4. Что реализуем сейчас (P0 / текущая фаза)

### 4.1 Разделение pipeline на 3 уровня

Нужно явно развести:

1. `Screening layer`
2. `Structural fit layer`
3. `Scientific inference layer`

#### Цель

Убрать смешение operational detection и scientific interpretation.

#### Новая логика

**Layer A — Screening**

- data quality checks
- HMM posterior / regime prior
- simple baseline indicators
- EWS only as weak auxiliary evidence

**Layer B — Structural fit**

- LPPLS fit
- soft filter scores
- multi-window evidence
- tc distribution / interval output

**Layer C — Scientific inference**

- cross-domain statistics
- ablation
- equivalence-style analysis later
- held-out / walk-forward evaluation

#### Критерий приемки

- архитектура отражена в коде и документации;
- operational verdict больше не смешивается с universality выводами;
- scientific modules не участвуют в online verdict path напрямую.

---

### 4.2 Soft filters вместо только hard thresholds

Текущие фильтры не удалять, но перевести из чистого `pass/fail` в систему скоринга.

#### Нужно реализовать

Для каждого fit считать:

- `score_m`
- `score_B`
- `score_omega`
- `score_damping`
- `score_tc_position`

Итог:

`lppls_quality_score = weighted combination of filter scores`

#### Важно

- current hard filters можно временно оставить как compatibility mode;
- новая логика должна уметь работать в двух режимах:
  - `strict_hard_filters`
  - `soft_filter_scoring`

#### Критерий приемки

- есть отдельный модуль soft scoring;
- тесты сравнивают hard vs soft режимы;
- новый скор не ухудшает FP behavior на текущих controls.

---

### 4.3 tc как распределение, а не как одна точка

Сейчас нужно внедрить uncertainty по `tc` **без full Bayesian LPPLS**.

#### Реализуем сейчас

- bootstrap over windows
- perturbation / residual bootstrap where feasible
- ensemble over multiple valid fits / starts
- summary:
  - `tc_median`
  - `tc_p50_low`
  - `tc_p50_high`
  - `tc_p80_low`
  - `tc_p80_high`

#### Цель

Перейти от:

`tc = one date`

к:

`tc lies in an uncertainty band`

#### Критерий приемки

- каждый LPPLS verdict умеет возвращать interval-like tc summary;
- валидация считает не только tc point error, но и coverage / interval width;
- интерфейс не ломает старые тесты.

---

### 4.4 Triple split + walk-forward validation

Это обязательный фундаментальный апгрейд.

#### Нужно реализовать

Новая схема:

- `train`
- `validation`
- `test`

Для time series:

- `walk-forward validation`
- `rolling-origin evaluation`

#### Цель

Прекратить tuning на evaluation-like данных и явно разделить:

- confirmatory results
- exploratory results

#### Критерий приемки

- хотя бы finance и housing имеют явный split protocol;
- новые thresholds / scoring rules подбираются только на `train/validation`;
- `test` используется один раз для финальной report-оценки.

---

### 4.5 Hard negative controls + adversarial benchmark

Negative controls нужно усилить.

#### Нужно собрать

Новый набор сложных отрицательных примеров:

- flat periods
- normal trend periods
- volatile but non-bubble periods
- exogenous shock periods
- cyclical but non-critical dynamics
- synthetic nulls with matched volatility
- false log-periodic lookalikes

#### Отдельный adversarial benchmark

Включить примеры типа:

- V-shaped recovery without bubble
- monotonic rally without crash
- noisy oscillations with fake frequency
- regime shift without LPPLS structure

#### Критерий приемки

- создан отдельный benchmark suite;
- новый suite входит в detector validation;
- результаты на hard negatives выведены отдельно от обычных controls.

---

### 4.6 Метрики раннего предупреждения

Добавить новые метрики поверх TP / FP / Precision / Recall.

#### Нужно считать

- `lead_time`
- `warning_stability`
- `false_alarm_duration`
- `tc_interval_width`
- `coverage of tc interval`
- `decision_utility` в простой условной постановке

#### Критерий приемки

- метрики считаются минимум для finance, housing, commodities;
- они входят в benchmark summary;
- paper-ready tables могут быть собраны из этих результатов.

---

## 5. Что реализуем после P0 (P1 / следующая фаза)

### 5.1 HMM как probabilistic prior, а не только gate

#### Реализовать после P0

- использовать `P(HMM=bubble)` как множитель или prior для LPPLS score;
- избегать чисто жесткой логики `skip / run only`.

#### Почему не сейчас

Сначала нужен clean soft scoring и новая validation framework.

---

### 5.2 Adaptive multi-scale windows

#### Реализовать после P0

- different window families for different regimes;
- сначала rule-based adaptation;
- только потом сложные state-aware schemes.

#### Почему не сейчас

Если добавить это до clean validation, будет трудно понять, действительно ли оно помогает.

---

### 5.3 Wavelet prototype

#### Реализовать после P0

- только как exploratory spectral validation;
- не как прямую замену LPPLS;
- не сравнивать naive wavelet frequency напрямую с LPPLS omega без корректной постановки.

#### Почему не сейчас

Есть риск методологической ошибки: LPPLS живет в log-time, а wavelet обычно в ordinary time.

---

### 5.4 Meta-calibration layer

#### Реализовать после P0

Входы:

- HMM posterior
- LPPLS quality score
- multi-window agreement
- EWS score
- data quality flags
- domain flags

Выход:

- `P(transition within horizon H)`
- confidence bucket
- false-alarm risk proxy

#### Почему не сейчас

Сначала нужно получить хорошие базовые сигналы и cleaner labels.

---

### 5.5 Better event labeling

#### Реализовать после P0

Вместо одной точки:

- `transition interval`
- `peak zone`
- `onset / climax / unwind`

Особенно важно для:

- housing
- commodities
- slower macro-like series

---

### 5.6 Domain-specific LPPLS modes

#### Реализовать после P0

- `LPPLS-fast`
- `LPPLS-slow`

С отдельными:

- окнами
- допустимым tc error
- priors / score maps
- omega expectations

---

## 6. Что сознательно откладываем (P2 / не сейчас)

Следующие идеи **не реализовывать в этой фазе**, даже если они выглядят технологично.

### 6.1 Full Bayesian LPPLS

Причина:

- очень высокий implementation risk;
- сложная posterior geometry;
- трудно доказать, что gains не иллюзорны;
- слишком тяжелый refactor для текущей стадии.

### 6.2 Transformer hybrid

Причина:

- мало качественно размеченных эпизодов;
- высокий риск narrative drift;
- снизится defendability перед reviewer.

### 6.3 SHAP / heavy explainability tooling

Причина:

- сначала нужна правильная структура сигналов и uncertainty;
- иначе получится псевдообъяснимость.

### 6.4 GPU acceleration

Причина:

- сначала нужен profiler and proven bottleneck;
- это не усиливает scientific credibility напрямую.

### 6.5 Learned stacking / complex meta-learner

Причина:

- пока мало чистых, хорошо размеченных эпизодов;
- высокий риск переобучения на текущем archive.

### 6.6 Regime-dependent learned thresholds

Причина:

- до появления жесткой triple-split validation это слишком похоже на скрытый overfitting.

---

## 7. Обязательная структура реализации

### 7.1 Новые или обновленные модули

Минимально ожидаемые направления:

- `src/lppls/scoring.py`
- `src/lppls/uncertainty.py`
- `src/validation/splits.py`
- `src/validation/metrics_early_warning.py`
- `src/validation/adversarial_controls.py`
- `src/pipeline/stages.py` или аналогичный orchestration module

### 7.2 Тесты

Добавить отдельные тестовые файлы:

- `tests/test_lppls_scoring.py`
- `tests/test_lppls_uncertainty.py`
- `tests/test_validation_splits.py`
- `tests/test_early_warning_metrics.py`
- `tests/test_adversarial_controls.py`

Если путь уже занят существующей логикой, аккуратно расширить текущие tests без слома структуры.

---

## 8. Правила реализации для Claude Code

### 8.1 Не ломать текущее рабочее ядро

- новые режимы должны включаться feature-flag style;
- старый baseline path должен сохраняться;
- все изменения должны быть incremental.

### 8.2 Не объявлять улучшение, пока его нет

Если новый модуль:

- не улучшает precision,
- не улучшает calibration,
- или не делает uncertainty честнее,

то он должен остаться как exploratory mode, без завышения claim.

### 8.3 Не смешивать detector и scientific inference

`universality`, `cross-domain tests`, `bootstrap comparisons` не должны напрямую участвовать в online verdict logic.

---

## 9. Правила обязательного пересчета результатов

### Tier 1 — локальный пересчет

Если меняются:

- документация
- labels
- отчеты

То достаточно:

- targeted tests
- smoke benchmark

### Tier 2 — detector-level пересчет

Если меняются:

- scoring
- uncertainty output
- HMM usage
- signal fusion

Нужно пересчитать:

- finance
- housing
- commodities
- detector summaries
- benchmark tables

### Tier 3 — LPPLS-core пересчет

Если меняются:

- fit logic
- windowing
- thresholding behavior
- tc extraction

Нужно пересчитать:

- finance
- geology
- housing
- commodities
- cross-domain LPPLS parameter pools
- ablation tables
- figure generation where affected

### Tier 4 — scientific inference пересчет

Если меняются:

- cross-domain statistics
- event labeling
- equivalence logic

Нужно пересчитать:

- universality summaries
- bootstrap tables
- paper-level science claims

---

## 10. Definition of Done для текущей фазы

Текущая фаза считается завершенной, если выполнены все пункты ниже:

1. pipeline разделен на `screening / structural fit / scientific inference`
2. soft scoring реализован и протестирован
3. tc uncertainty выдается как interval-style output
4. есть triple split или walk-forward protocol минимум для ключевых доменов
5. есть hard negative / adversarial benchmark
6. есть richer early-warning metrics
7. результаты пересчитаны в нужном объеме
8. документация обновлена без overstating

---

## 11. Краткий приоритетный порядок

### Делать сейчас

1. pipeline separation
2. soft filters / scoring
3. tc uncertainty
4. triple split / walk-forward validation
5. hard negative controls
6. richer early-warning metrics

### Делать потом

7. HMM as probabilistic prior
8. adaptive windows
9. wavelet prototype
10. meta-calibration layer
11. better event labeling
12. LPPLS-fast / LPPLS-slow split

### Не делать сейчас

13. full Bayesian LPPLS
14. transformer hybrid
15. SHAP-heavy explainability
16. GPU acceleration
17. learned stacking
18. regime-dependent thresholds

---

## 12. Финальная установка

PhaseBreak сейчас надо развивать по принципу:

**сначала доказательность и устойчивость, потом усложнение**

а не наоборот.

Claude Code должен реализовывать не "максимум новых идей", а **максимум усиления фундамента**.

