# ТЗ: PhaseBreak v2 — Full Integration And Recompute

**Дата:** 2026-03-29  
**Статус:** к реализации  
**Назначение:** довести `PhaseBreak v2` от состояния "новые модули добавлены" до состояния "v2 полностью интегрирован, результаты пересчитаны, claims синхронизированы".

---

## 1. Контекст

На текущий момент в репозитории уже существуют новые v2-модули:

- `src/lppls/conformal.py`
- `src/lppls/scoring.py`
- `src/lppls/uncertainty.py`
- `src/lppls/windowing.py`
- `src/signals/changepoint.py`
- `src/signals/wavelet_lppls.py`
- `src/pipeline/stages.py`
- `src/validation/splits.py`
- `src/validation/adversarial_controls.py`
- `src/validation/metrics_early_warning.py`

Также есть расширенный тестовый набор, и текущий `pytest` проходит.

Но этого еще недостаточно, чтобы честно утверждать:

> "PhaseBreak v2 полностью интегрирован, старые результаты пересчитаны, новая методология стала официальной".

Проблемы текущего состояния:

- часть модулей существует отдельно, но не подключена в основной verdict path;
- нет единого `official v2 benchmark run`;
- старые headline-результаты не пересчитаны как новые официальные результаты;
- ablation не обновлен под v2;
- public artifacts (`README`, `paper`, notebooks) не синхронизированы;
- не зафиксировано, что `confirmed`, а что `exploratory`.

---

## 2. Главная цель

Довести проект до состояния:

1. `v2 pipeline` реально работает как основной или явно обозначенный альтернативный официальный путь;
2. ключевые результаты по доменам пересчитаны через него;
3. `ablation`, `benchmark`, `uncertainty`, `controls`, `claims` синхронизированы;
4. документация и paper отражают реальное состояние проекта.

---

## 3. Что должно быть сделано обязательно

### 3.1 Интеграция P1 в основной pipeline

Нужно реализовать и подключить в operational path:

1. `HMM as probabilistic prior`
2. `adaptive windows` inside real fit path
3. `wavelet prototype` как auxiliary / diagnostic layer
4. `meta-calibration layer` только если успевает быть реализована чисто и без ломки ядра

#### Требования

- это должны быть не standalone utilities, а реальная часть `v2 pipeline`;
- старый pipeline должен оставаться доступным как `legacy`;
- должно быть явно указано, какой path:
  - `legacy`
  - `v2`
  - `recommended`

#### Критерий приемки

- есть единый public entry point для `v2 pipeline`;
- v2 pipeline реально использует:
  - soft scoring
  - tc uncertainty
  - HMM prior
  - adaptive windows
- wavelet, если не дает подтвержденный прирост, остается `diagnostic only`.

---

### 3.2 Единый официальный v2 benchmark run

Нужно создать единый benchmark script / workflow, который пересчитывает официальные результаты.

#### Должны входить домены

- `finance`
- `housing`
- `commodities`
- `geology`
- `fraud` только если конкретный результат зависит от новых модулей
- `adversarial controls`

#### Должны считаться метрики

- TP / FP / FN / TN
- precision / recall / false positive rate
- lead time
- tc point error
- tc interval width
- tc interval coverage
- warning stability
- false alarm duration
- mean quality score

#### Критерий приемки

- есть один воспроизводимый benchmark path;
- результаты сохраняются в явном виде, а не только печатаются в консоль;
- можно повторно получить official tables без ручной сборки по разным тестам.

---

### 3.3 Пересчет headline-results по доменам

Нужно заново получить официальные summary numbers для каждого домена.

#### Finance

Пересчитать:

- TP / FP
- tc error / lead time
- quality score summary
- tc interval coverage

#### Housing

Пересчитать:

- held-out results
- baselines comparison
- interval metrics
- control periods

#### Commodities

Пересчитать:

- TP / FP
- control performance
- uncertainty summaries

#### Geology

Пересчитать:

- fit yield / successful fits
- параметрические пулы, если fit path изменился

#### Fraud

Если новые v2 изменения не затрагивают fraud path напрямую:

- оставить fraud как отдельный трек,
- но проверить, что old results still pass.

#### Критерий приемки

- для каждого домена есть новый v2 summary;
- старые цифры больше не используются как "официальные", если они были получены старым путем.

---

### 3.4 Новый v2 ablation

Нужно построить новую ablation-таблицу.

Минимальные строки:

1. legacy LPPLS
2. + hard filters
3. + soft scoring
4. + tc uncertainty
5. + adaptive windows
6. + HMM prior
7. + optional auxiliary layers

Если модуль не показывает прирост:

- это нужно зафиксировать честно;
- не маскировать как improvement.

#### Критерий приемки

- есть отдельный v2 ablation output;
- видно, какие именно слои реально дают улучшение;
- exploratory layers не смешиваются с confirmed gains.

---

### 3.5 Решение: conformal vs bootstrap uncertainty

Сейчас в проекте есть как минимум две uncertainty-ветки:

- `conformal`
- `bootstrap tc uncertainty`

Нужно зафиксировать:

- что является `primary uncertainty method`
- что является `secondary / exploratory`

#### Критерий приемки

- в pipeline нет двусмысленности;
- в benchmarks и документации используется один primary path;
- второй путь может остаться как comparison mode.

---

### 3.6 Пересчет scientific inference при изменении fit behavior

Если v2 pipeline реально меняет:

- window selection
- accepted fits
- parameter extraction
- tc extraction logic

то надо заново пересчитать scientific layer:

- cross-domain parameter pools
- KS / Mann-Whitney / bootstrap
- universality summaries

Если fit behavior для official science path не меняется:

- оставить science layer как legacy-compatible,
- но это должно быть явно написано.

#### Критерий приемки

- либо science layer пересчитан,
- либо прямо задокументировано, почему он не пересчитывался.

---

### 3.7 Confirmed vs exploratory registry

Нужно явно пометить все ключевые v2-компоненты:

- `confirmed`
- `useful but exploratory`
- `diagnostic only`
- `not integrated into official verdict`

Минимально классифицировать:

- soft scoring
- tc uncertainty
- conformal
- changepoint
- wavelet
- HMM prior
- adaptive windows
- EWS

#### Критерий приемки

- есть один документированный registry;
- paper/README не переобещают модули, которые остались exploratory.

---

### 3.8 Обновление public artifacts

После пересчета надо обновить:

- `README.md`
- `paper/main.tex`
- relevant notebooks
- при необходимости `CORRECTIONS.md`

#### README должен отражать

- реальное число тестов
- актуальное число доменов
- текущий status `legacy vs v2`
- confirmed vs exploratory modules

#### Paper должен отражать

- только те claims, которые реально подтверждены;
- v2 narrative только в той части, которая прошла recompute.

#### Критерий приемки

- код, README и paper говорят одним языком;
- нет расхождения между public claims и actual benchmark outputs.

---

### 3.9 Финальная очистка и фиксация состояния репозитория

Перед финальным статусом нужно:

- понять судьбу новых `docs/*.md`
- проверить `git status`
- не оставлять непонятные незакоммиченные изменения, относящиеся к v2

Не трогать пользовательские внешние изменения без явной необходимости, но финальное состояние по v2 должно быть прозрачным.

#### Критерий приемки

- v2-related changes либо закоммичены, либо сознательно исключены из финального состояния;
- можно четко сказать, что именно вошло в v2 integration release.

---

## 4. Что НЕ нужно делать в этом ТЗ

В это ТЗ не входят:

- full Bayesian LPPLS
- transformer hybrid
- SHAP-heavy explainability
- GPU acceleration
- regime-dependent thresholds
- большой UI / deployment слой

Это отдельные будущие ветки. Сейчас цель — **интеграция, пересчет, синхронизация**, а не дальнейшее распухание scope.

---

## 5. Ожидаемые файлы и артефакты

### Код

Ожидаемо будут изменены или созданы файлы в:

- `src/pipeline/`
- `src/lppls/`
- `src/signals/`
- `src/validation/`

### Тесты

Ожидаемо будут добавлены или обновлены:

- `tests/test_v2_pipeline.py`
- benchmark-oriented tests or validation scripts
- обновленные domain validation tests при необходимости

### Артефакты пересчета

Желательно сохранить:

- benchmark summaries
- ablation summaries
- domain result tables
- optional JSON/CSV outputs for reproducibility

---

## 6. Порядок реализации

### Этап 1

- интегрировать P1 в основной v2 path
- выбрать primary uncertainty method
- определить legacy vs v2 official path

### Этап 2

- прогнать единый v2 benchmark
- пересчитать domain headline-results
- пересчитать v2 ablation

### Этап 3

- пересчитать scientific inference или явно зафиксировать, почему он не менялся
- составить confirmed vs exploratory registry

### Этап 4

- обновить README / paper / notebooks
- проверить git state
- зафиксировать итоговую версию

---

## 7. Definition of Done

Можно честно писать:

> "PhaseBreak v2 fully integrated and recomputed"

только если выполнены все пункты ниже:

1. P1 интегрирован в реальный v2 pipeline
2. есть единый официальный v2 benchmark run
3. headline-results по доменам пересчитаны
4. v2 ablation пересчитан
5. выбран primary uncertainty method
6. science layer пересчитан или явно заморожен с объяснением
7. confirmed vs exploratory registry составлен
8. README и paper синхронизированы
9. финальное состояние репозитория прозрачно

---

## 8. Инструкция для Claude Code

Работать строго по этому приоритету:

1. сначала интеграция;
2. потом пересчет;
3. потом синхронизация claims;
4. только потом косметика.

Не объявлять:

- "всё интегрировано"
- "результаты пересчитаны"
- "v2 official"

пока реально не выполнен `Definition of Done`.

Если какой-то новый слой не дает подтвержденной пользы:

- оставить его как exploratory;
- явно указать это в финальном отчете.

