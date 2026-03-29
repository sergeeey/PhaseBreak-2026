# ТЗ: Audit Remediation Plan for PhaseBreak 2026

**Дата:** 2026-03-29  
**Статус:** к реализации  
**Основание:** полный аудит проекта выявил воспроизводимые проблемы в benchmark discipline, dependency discipline, claim synchronization и official packaging.

---

## 1. Цель

Довести проект до состояния:

- `audit-clean`
- `publication-consistent`
- `reproducible from a clean environment`

Это ТЗ **не про новые фичи**, а про устранение найденных дефектов в:

- official benchmark truth
- зависимости среды
- legacy vs v2 path
- consistency между кодом, README, paper, отчетами
- научными claim'ами

---

## 2. Ключевые проблемы, которые нужно закрыть

### 2.1 Benchmark truth drift

Сейчас в проекте фигурируют разные official цифры:

- `38 episodes`
- `50 episodes`
- `58 episodes`
- разные counts tests (`240+`, `253`)

Это недопустимо для submission-grade package.

### 2.2 Hidden runtime dependencies

Официальный `v2 pipeline` и science-layer зависят от пакетов, которые не зафиксированы как обязательные:

- `hurst`
- `MFDFA`
- `pingouin`
- `bayesian_changepoint_detection`

При их отсутствии код тихо уходит в fallback, меняя benchmark behavior.

### 2.3 HMM prior overstated

В official registry и README `HMM prior` подается как confirmed/recommended, хотя current ablation показывает:

- no recall gain
- extra false positive on finance

Формулировка должна быть приведена в соответствие с evidence.

### 2.4 Legacy path contamination

`legacy` path не должен зависеть от нового screening path и новых optional hooks, если он используется как baseline.

### 2.5 Science layer reproducibility gap

Paper уже использует TOST/equivalence narrative, но:

- `pingouin` не гарантирован,
- `run_full_equivalence` не покрыт тестами,
- часть science claims сильнее, чем текущая доказательная поверхность.

---

## 3. Что нужно сделать за 1 день

### 3.1 Зафиксировать один официальный benchmark truth

Нужно выбрать и зафиксировать:

- official count episodes
- official count tests
- official benchmark scope

Варианты не должны сосуществовать в README/paper/report одновременно.

#### Требование

После выбора official benchmark truth нужно обновить:

- `README.md`
- `paper/main.tex`
- `docs/PROJECT_REPORT.md`
- при необходимости `docs/V2_COMPONENT_REGISTRY.md`
- любые `json/md` summary files, которые цитируются как source of truth

#### Критерий приемки

- во всех публичных артефактах фигурируют одни и те же числа;
- цифры совпадают с реально воспроизводимым benchmark output.

---

### 3.2 Зафиксировать dependency contract

Нужно решить для каждого пакета:

- `обязательный`
- `optional / exploratory`
- `не используется в official path`

Минимально проверить:

- `hurst`
- `MFDFA`
- `pingouin`
- `bayesian_changepoint_detection`

#### Требование

Если модуль влияет на official benchmark / official science claims:

- пакет должен быть объявлен в `pyproject.toml` или соответствующем extra;
- отсутствие пакета должно приводить не к silent fallback, а к явному signal/error в official run.

#### Критерий приемки

- clean environment reproducibly поднимается;
- official benchmark не меняет вывод тихо из-за отсутствующей зависимости.

---

### 3.3 Ослабить HMM prior claim до доказанного уровня

Нужно привести narrative в соответствие с actual ablation.

#### Требование

Если current evidence показывает:

- extra FP
- no recall gain

то `HMM prior` нельзя подавать как unconditional confirmed improvement.

Допустимые варианты:

- `confirmed architectural integration, mixed impact`
- `operationally useful, accuracy-neutral`
- `useful with caveat`

#### Критерий приемки

- registry, README и paper не переобещают эффект HMM prior;
- wording совпадает с actual ablation.

---

## 4. Что нужно сделать за 3 дня

### 4.1 Развести benchmark surfaces

Нужно ввести явные уровни:

1. `official operational benchmark`
2. `science benchmark`
3. `survival benchmark`
4. `extended / experimental benchmark`

#### Пример

**Official operational benchmark**

- finance
- commodities
- housing
- forward
- adversarial

**Science benchmark**

- geology
- cross-domain inference

**Survival benchmark**

- fraud

#### Критерий приемки

- каждый benchmark имеет явный scope;
- исчезает путаница `4 domains`, `5 domains`, `6 categories`, `38/50/58 episodes`.

---

### 4.2 Сделать legacy path реально legacy

`run_legacy_pipeline()` должен быть независим от:

- Hurst override
- MFDFA boost
- новых domain hacks
- новой screening logic, если она меняет baseline

#### Требование

Legacy path = честный baseline.

Если нужен общий pre-check на NaN/short series, он должен быть строго нейтральным и одинаковым для обоих paths.

#### Критерий приемки

- `legacy` действительно отражает старую методологию;
- `v2 vs legacy` comparison становится корректным.

---

### 4.3 Добавить tests for official claims

Нужно добавить/обновить тесты, которые защищают public truth:

- test on official benchmark size
- test on official benchmark schema
- test on claim synchronization
- test on required dependencies for official benchmark
- test on TOST/equivalence path, если он остается в paper

#### Критерий приемки

- нельзя снова случайно разъехаться между code / JSON / README / paper без падения тестов.

---

## 5. Что нужно сделать за неделю

### 5.1 Пересобрать paper под current truth

Paper должен содержать только:

- актуальные benchmark numbers
- актуальный test count
- актуальный benchmark scope
- актуальную cross-domain wording

#### Требование

Убрать все устаревшие цифры и narrative drift.

#### Критерий приемки

- paper компилируется;
- paper numbers совпадают с benchmark outputs;
- paper wording соответствует actual evidence strength.

---

### 5.2 Сделать reproducible benchmark environment

Нужно зафиксировать способ воспроизведения benchmark:

- install command
- optional extras
- exact benchmark run command
- environment metadata

Желательно сохранить рядом с official benchmark:

- Python version
- dependency snapshot
- benchmark timestamp

#### Критерий приемки

- другой человек может поднять окружение и получить те же benchmark artifacts без скрытых зависимостей.

---

### 5.3 Привести science claims к реальной доказательной поверхности

Нужно решить:

- либо science layer fully tested and reproducible,
- либо claims сузить.

Особенно важно для:

- TOST/equivalence
- universality language
- all pairwise tests wording

#### Критерий приемки

- paper и README не утверждают больше, чем реально проверено.

---

## 6. Что НЕ делать в этом ТЗ

Не добавлять:

- новые домены
- новый deep learning
- full Bayesian LPPLS
- UI / dashboard
- новые exploratory layers

Это remediation phase, не expansion phase.

---

## 7. Ожидаемые файлы к изменению

С высокой вероятностью будут изменены:

- `pyproject.toml`
- `README.md`
- `paper/main.tex`
- `docs/PROJECT_REPORT.md`
- `docs/V2_COMPONENT_REGISTRY.md`
- `src/pipeline/stages.py`
- `src/benchmark/v2_benchmark.py`
- `src/benchmark/v2_ablation.py`
- `src/cross_domain/universality.py`
- тесты, связанные с official benchmark / claim synchronization

---

## 8. Definition of Done

Remediation phase считается завершенной, если:

1. есть один официальный benchmark truth
2. все public artifacts используют одни и те же числа
3. hidden dependencies для official paths устранены
4. HMM prior claim ослаблен или подтвержден строго
5. legacy path действительно legacy
6. official claims защищены тестами
7. paper и README синхронизированы с current benchmark outputs

---

## 9. Инструкция для Claude Code

Работать в таком порядке:

1. benchmark truth
2. dependency contract
3. HMM prior claim remediation
4. legacy cleanup
5. tests for official claims
6. README/paper/report sync

Не добавлять новые возможности, если они не нужны для закрытия найденных audit findings.

Если при remediation выяснится, что какой-то claim нельзя сохранить честно:

- сузить claim,
- явно задокументировать ограничение,
- не пытаться сохранить красивую формулировку ценой достоверности.

