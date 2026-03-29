# ТЗ: PhaseBreak — Следующая Фаза Методологических Улучшений

**Дата:** 2026-03-29  
**Статус:** к реализации  
**Цель документа:** передать Claude Code четкое ТЗ на следующую фазу развития методологии без расползания scope.

---

## 1. Контекст

На момент старта этого ТЗ проект уже имеет:

- 5 доменов: `finance`, `geology`, `housing`, `commodities`, `fraud`
- действующее LPPLS-ядро
- tightened Sornette filters
- multi-window confidence
- HMM-gated logic
- EWS как exploratory layer
- fraud survival branch на synthetic data
- cross-domain comparison через KS / Mann-Whitney / bootstrap
- 182 passing tests

Текущее состояние сильное как `pre-submission research framework`, но следующие улучшения должны быть **строго отфильтрованы по ROI**. Нельзя одновременно тащить тяжелые Bayesian, transformer и UI-ветки.

---

## 2. Главная цель фазы

Усилить **операционную и научную полезность** ядра `PhaseBreak` без разрушения текущей честной методологии.

Нужны 4 практических улучшения:

1. `Conformal prediction / calibrated intervals`
2. `Adaptive multi-scale windows`
3. `Changepoint detector` как новый сигнал
4. `Wavelet prototype` как исследовательская spectral-валидация

---

## 3. Что НЕ делать в этой фазе

Следующие направления **не входят** в данное ТЗ:

- full Bayesian LPPLS / PyMC posterior inference
- transformer hybrid
- SHAP explainability
- GPU acceleration
- learned stacking / meta-learner
- regime-dependent thresholds
- production UI / deployment

Причина: высокий риск, низкий immediate ROI, недостаточный объем данных для устойчивой реализации.

---

## 4. Общие принципы реализации

### 4.1 Scope discipline

- Не ломать текущее рабочее ядро.
- Все новые слои добавлять **опционально** и **поэтапно**.
- Каждый новый модуль должен иметь:
  - отдельный файл в `src/`
  - отдельный набор тестов
  - отдельный маленький benchmark/summary

### 4.2 Honest methodology

- Не писать ожидаемые улучшения как факт, пока они не подтверждены.
- Если новый слой не дает прироста, оставить его как `exploratory` и явно это зафиксировать.
- Не заменять текущий production-like path новым модулем до тех пор, пока новый модуль не выиграл на тестах.

### 4.3 Recompute discipline

Любое изменение должно сопровождаться **пересчетом только нужного слоя результатов**, а не хаотичным повторным прогоном всего подряд.

Правила пересчета описаны в разделе 9.

---

## 5. Архитектурная рамка

### 5.1 Текущий базовый detector path

```text
Raw time series
  -> HMM regime detection
  -> LPPLS fit
  -> tight Sornette filters
  -> multi-window confidence
  -> detector verdict
```

### 5.2 Целевой detector path после этой фазы

```text
Raw time series
  -> HMM regime detection
  -> adaptive window selection
  -> LPPLS fit
  -> tight Sornette filters
  -> multi-window confidence
  -> changepoint signal
  -> conformal calibration / tc interval
  -> optional wavelet diagnostic
  -> detector verdict + interval + diagnostics
```

Ключевое требование:  
`wavelet` и `changepoint` не должны сразу ломать decision logic.  
Сначала они добавляются как **диагностические и сравнительные сигналы**.

---

## 6. План реализации на 4 спринта

---

## Спринт 1 — Conformal Prediction / Calibrated Intervals

### Цель

Перевести детектор из point-estimate / heuristic confidence в формат:

- `tc_estimate`
- `tc_interval`
- `risk_level`
- `calibrated coverage`

### Что реализовать

#### Новый модуль

Создать:

- `src/lppls/conformal.py`

Содержимое:

- функция калибровки интервалов по историческим residual-style errors
- функция построения interval вокруг `tc`
- функция перевода raw confidence в calibrated confidence band

Минимальный API:

```python
fit_tc_calibrator(errors: np.ndarray, alpha: float = 0.1) -> dict
predict_tc_interval(tc_estimate: float, calibrator: dict) -> tuple[float, float]
calibrate_confidence(raw_score: float, bucket_stats: dict) -> float
```

#### Интеграция

- не менять существующий `LPPLSOptimizer`
- не менять существующий `CertifiedFit`
- добавить lightweight wrapper на уровне detector output

#### Тесты

Создать:

- `tests/test_conformal.py`

Проверки:

- интервалы строятся корректно
- lower <= estimate <= upper
- coverage на synthetic calibration set не деградирует ниже target в разумных пределах
- поведение на edge-cases: пустые ошибки, одна точка, infinite values

### Критерии приемки

- новый модуль покрыт тестами
- detector может возвращать `tc_interval`
- не сломаны старые тесты
- есть короткий print summary в одном тесте

### Что пересчитать после спринта

- finance detector outputs
- housing detector outputs
- commodities detector outputs
- notebooks, если там показывается `tc`

Полный cross-domain recompute **не нужен**, если `m, omega` не менялись.

---

## Спринт 2 — Adaptive Multi-Scale Windowing

### Цель

Уйти от жестко зафиксированных окон и сделать выбор окон адаптивным под тип ряда и volatility regime.

### Что реализовать

#### Новый модуль

Создать:

- `src/lppls/windowing.py`

Минимальный API:

```python
select_adaptive_windows(
    series: np.ndarray,
    frequency: str = "daily",
    volatility_regime: str | None = None,
) -> list[int]

infer_volatility_regime(series: np.ndarray) -> str
```

#### Базовая логика

На первом этапе не делать сложный Kalman stack.

Нужно:

- daily / high-vol -> короче окна
- daily / normal -> текущий набор
- quarterly / slow series -> длиннее окна

Примеры:

- finance / commodities: `[60, 90, 120, 180]`
- high-vol commodities / crypto-like: `[30, 45, 60, 90]`
- housing quarterly: `[8, 12, 16, 20]` кварталов

#### Интеграция

- расширить `MultiWindowConfidence`, чтобы он мог принимать:
  - либо явные `windows`
  - либо adaptive mode

Без разрушения текущего API.

#### Тесты

Создать:

- `tests/test_windowing.py`

Проверки:

- окна выбираются детерминированно
- quarterly housing не получает daily windows
- adaptive mode возвращает валидный список окон
- `MultiWindowConfidence` работает и со старыми windows, и с adaptive mode

### Критерии приемки

- старые multi-window тесты проходят
- adaptive windows интегрированы без breaking changes
- есть минимум один domain test, который реально использует adaptive mode

### Что пересчитать после спринта

Это уже изменение fit/window logic, поэтому нужен **полный LPPLS recompute**:

- finance
- geology
- housing
- commodities
- ablation
- held-out housing
- cross-domain `m, omega`
- figures / notebook outputs, если они зависят от tc или param extraction

---

## Спринт 3 — Changepoint Detector

### Цель

Добавить новый независимый signal layer, который детектирует структурный перелом временного ряда и сравнивается с LPPLS.

### Что реализовать

#### Новый модуль

Создать:

- `src/signals/changepoint.py`

Если не хочется нового top-level пакета, допустимо:

- `src/lppls/changepoint.py`

Минимальный API:

```python
detect_changepoint_strength(series: np.ndarray) -> float
detect_recent_changepoint(series: np.ndarray) -> dict
```

Рекомендуемый старт:

- простой offline changepoint baseline
- без streaming/online избыточности
- без тяжелой Bayesian ветки на первом шаге

### Интеграция

Пока changepoint не должен автоматически влиять на final verdict везде.

Сначала нужно:

- считать сигнал
- сравнивать его с LPPLS
- добавить его в ablation / detector comparison

Возможные статусы:

- `LPPLS_ONLY`
- `CP_ONLY`
- `LPPLS_AND_CP`
- `NO_SIGNAL`

#### Тесты

Создать:

- `tests/test_changepoint.py`

Проверки:

- synthetic change detected
- flat/noisy series do not yield strong false signal
- recent changepoint score bounded in `[0,1]`

#### Detector-level comparison

Создать / расширить:

- `tests/test_detector_comparison.py` или встроить в существующие validation tests

Нужно сравнить:

- LPPLS
- LPPLS + HMM
- LPPLS + changepoint diagnostic

### Критерии приемки

- changepoint signal выделен в отдельный модуль
- не ломает старые pipeline paths
- есть сравнение на finance, housing, commodities

### Что пересчитать после спринта

- detector-level validation
- finance / housing / commodities verdict tables
- новый ablation или comparison table

Пересчет cross-domain `m, omega` **не обязателен**, если LPPLS fitting logic не менялась.

---

## Спринт 4 — Wavelet Prototype

### Цель

Построить **исследовательский prototype** spectral-валидации, но не подменять им LPPLS ядро до тех пор, пока gain не подтвержден.

### Важное теоретическое ограничение

Нельзя напрямую сравнивать:

- wavelet frequency in ordinary time
- LPPLS `omega` in log-time

Это разные величины.

Следовательно wavelet-прототип должен использоваться как:

- spectral diagnostic
- ridge / oscillation stability indicator
- auxiliary confidence modifier

А не как прямой “independent omega validator” в лоб.

### Что реализовать

#### Новый модуль

Создать:

- `src/signals/wavelet_lppls.py`

или

- `src/lppls/wavelet.py`

Минимальный API:

```python
extract_wavelet_diagnostics(series: np.ndarray, sampling_rate: float) -> dict
score_oscillation_stability(diag: dict) -> float
compare_lppls_wavelet(lppls_result: dict, wavelet_diag: dict) -> dict
```

#### Scope

На этом этапе:

- prototype only
- применить только к:
  - finance
  - commodities

Не трогать geology / housing до подтверждения пользы.

#### Тесты

Создать:

- `tests/test_wavelet_lppls.py`

Проверки:

- код работает на synthetic oscillatory series
- diagnostic score finite and bounded
- stable behavior on flat series / pure noise

### Критерии приемки

- wavelet layer не ломает основной detector
- есть отдельный benchmark-комментарий: помогает / не помогает
- если gain нет, слой маркируется как exploratory

### Что пересчитать после спринта

- targeted evaluation only:
  - finance
  - commodities

Не делать полный recompute проекта без оснований.

---

## 7. Новые/изменяемые файлы

### Новые файлы

```text
docs/TZ_NEXT_PHASE_METHOD_UPGRADES.md
src/lppls/conformal.py
src/lppls/windowing.py
src/signals/changepoint.py
src/signals/wavelet_lppls.py
tests/test_conformal.py
tests/test_windowing.py
tests/test_changepoint.py
tests/test_wavelet_lppls.py
```

### Возможные изменения в существующих файлах

```text
src/lppls/confidence.py
src/lppls/ensemble.py
src/lppls/__init__.py
README.md
paper/main.tex
paper/generate_figures.py
notebooks/01_finance_bubbles.ipynb
notebooks/04_housing_bubbles.ipynb
```

Если меняются публичные claims, нужно обновлять README и paper только после подтвержденного пересчета.

---

## 8. Acceptance Criteria по всей фазе

Фаза считается успешно выполненной, если:

1. Все новые модули покрыты тестами.
2. Полный suite остается зеленым.
3. Нет деградации текущего честного narrative.
4. Новый detector output становится богаче:
   - interval
   - diagnostics
   - adaptive behavior
5. Хотя бы одно из улучшений дает подтвержденный практический выигрыш.

### Недопустимые исходы

- сломать текущие validated results ради сложной идеи
- переобещать gains без benchmark
- переписать core pipeline слишком рано
- смешать operational detector и scientific cross-domain interpretation

---

## 9. Правила пересчета результатов

### Tier 1 — Локальный пересчет

Нужен после маленьких изменений, не влияющих на fit logic:

- affected unit tests
- один короткий smoke benchmark

### Tier 2 — Detector-level пересчет

Нужен после:

- conformal layer
- changepoint layer

Пересчитывать:

- finance
- housing
- commodities
- detector outputs
- relevant notebooks

### Tier 3 — Full LPPLS recompute

Нужен после:

- adaptive windows
- изменения fit logic
- изменения filters

Пересчитывать:

- finance
- geology
- housing
- commodities
- ablation
- held-out housing
- cross-domain `m, omega`
- figures
- paper metrics

### Tier 4 — Science-layer recompute

Нужен после:

- изменения cross-domain stats
- изменения extraction logic for `m, omega`

Пересчитывать:

- KS / MW / bootstrap tables
- universality summary
- paper claims

---

## 10. Приоритеты для Claude Code

### Делать сейчас

1. `Conformal prediction`
2. `Adaptive windows`
3. `Changepoint detector`
4. `Wavelet prototype`

### Делать потом

1. finance-specific multi-asset correlation
2. simple ensemble over LPPLS + changepoint + HMM
3. regime-dependent constraints

### Не делать в этой фазе

1. Bayesian LPPLS
2. Transformer hybrid
3. SHAP
4. GPU optimization

---

## 11. Ожидаемый итог фазы

После реализации этой фазы проект должен перейти из состояния:

```text
Strong research detector with point-style outputs
```

в состояние:

```text
Better calibrated, interval-aware, diagnostics-rich early warning framework
```

Без разрушения текущего validated ядра.

---

## 12. Финальная инструкция исполнителю

Реализовывать строго поэтапно.

После каждого спринта:

1. прогнать нужный слой пересчета
2. кратко зафиксировать:
   - что изменилось
   - что улучшилось
   - что не улучшилось
3. не объединять experimental claims с validated claims

Если какое-либо улучшение не дает воспроизводимого выигрыша:

- оставить модуль в проекте как `exploratory`
- не включать в основную narrative-цепочку

