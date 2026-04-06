# PhaseBreak — Итоговый отчёт о проведённой работе

**Дата:** 6 апреля 2026
**Период работы:** 5-6 апреля 2026 (2 сессии, ~10 часов)
**Проекты:** PhaseBreak 2026 (academic) + PhaseBreak Hybrid 2.0 (product)

---

## 1. Что было сделано

### Фаза 1: Red Team аудит (5 апреля)

Проведён полный adversarial аудит обоих проектов в 4 параллельных потока:
- **Data loading audit** — поиск моков, кэша, подмены данных
- **Math integrity audit** — параметры, пороги, forced convergence
- **LLM/RAG vulnerability audit** — confirmation bias, answer leakage
- **Test suite audit** — cherry-picking, train/test contamination

#### Результаты по PhaseBreak 2026:
- Математическое ядро **корректно** — LPPLS Sornette 2003, 4 уровня защиты от FP
- Paper metrics **не совпадали** с benchmark data (80% в abstract vs 76% в JSON)
- generate_v2_figures.py **hardcoded** метрики вместо чтения из JSON
- Test count **устарел** (253 в paper vs 268 фактически)
- Council validator имел **answer leakage** (is_bubble в LLM контексте)
- plotly — **мёртвая зависимость**

#### Результаты по PhaseBreak Hybrid 2.0:
- **C↔omega swap bug** в fit_lppls_with_ds (line 397) — параметры перепутаны
- **Answer leakage** в eval_runner — known_outcome передавался в LLM
- **Leading prompts** — "обнаружила пузырь" вместо нейтрального
- **Hardcoded mocks** для NVDA и GME в rag_engine
- **Дубликаты** в eval set (4 кейса x2)
- **COIN** — невалидный кейс (период до IPO)

### Фаза 2: Hardening (5 апреля)

Все найденные баги исправлены:

| Fix | PhaseBreak 2026 | PhaseBreak Hybrid 2.0 |
|-----|-----------------|----------------------|
| Paper metrics | 80%→78% finance, 77%→76% overall | — |
| Figures | Reads from JSON | — |
| Test count | 253→268 | — |
| Council | is_bubble removed, parser fixed | — |
| plotly | Removed | — |
| C↔omega | — | Fixed (line 397-398) |
| Answer leakage | — | blind=True default |
| Leading prompts | — | "проанализировала" (neutral) |
| Hardcoded mocks | — | Generic fallback |
| Eval set | — | Deduped 38→34, COIN fixed |

### Фаза 3: Feature Development (5-6 апреля)

Три спринта для Hybrid 2.0:

**Sprint 1 — RSS Feed Engine:**
- 24 RSS ленты (Bloomberg, Reuters, CNBC, CoinDesk, etc.)
- 17 ticker aliases (NVDA→Nvidia, BTC-USD→Bitcoin, etc.)
- 15-min TTL cache, DuckDuckGo fallback
- Интегрирован в RAG engine как primary source

**Sprint 2 — Telegram Alerts:**
- AlertPayload (Pydantic model) + TelegramAlerter (httpx, no deps)
- should_alert() — verdict in (BUBBLE, POSSIBLE) AND quality > 0.7
- monitor_cron.py — scheduled runner с --dry-run и --loop

**Sprint 3 — Sector Scan:**
- 8 секторов, 59 тикеров
- ThreadPoolExecutor для параллельного анализа
- Sector alert при 3+ bubble signals в одном секторе
- Rich console table + JSON heatmap export

### Фаза 4: Week Sprint (6 апреля)

| Day | Task | Result |
|-----|------|--------|
| 1 | Merge 2026 fixes + HMM import | Pushed 7cb899e, advisory HMM gate |
| 2 | Crash-oriented eval | 34 cases, precision 50%, recall 23% |
| 3 | Persistence filter | TENTATIVE→CONFIRMED after 3 scans |
| 4 | Hurst exponent + RAG cache | H=0.726 confirms TSLA signal |
| 5 | Geology + arXiv | Paper ready, geology BLOCKED (no data) |
| 6 | Streamlit dashboard | 4 tabs: Signals, Sectors, History, Eval |
| 7 | Live monitoring + lock | 2 predictions locked, Task Scheduler set |

---

## 2. Текущее состояние проектов

### PhaseBreak 2026 (Academic)

```
Статус:         arXiv-ready (ждёт endorsement physics.soc-ph)
Коммитов:       32 (на main, pushed)
Tests:          268/268 pass
Benchmark:      58 эпизодов, 6 доменов
Paper metrics:  Precision 76% overall, 78% finance, Recall 61%
```

**Матрица доменов:**

| Домен | Episodes | Precision | Recall | Maturity |
|-------|----------|-----------|--------|----------|
| Finance | 20 | 78% | 64% | PRODUCTION |
| Commodities | 10 | 67% | 67% | PRODUCTION |
| Housing FHFA | 10 | 67% | 33% | MVP |
| Housing Zillow | 6 | 67% | 50% | MVP |
| Adversarial | 6 | 100% | 100% | PRODUCTION |
| Forward 2024-25 | 6 | 100% | 100% | PRODUCTION |

### PhaseBreak Hybrid 2.0 (Product)

```
Статус:         Live monitoring active (daily at 18:00)
Modules:        33 Python files
Pipeline:       yfinance → HMM → LPPLS → Hurst → RAG → Verdict → Persistence → Alert
Signals:        3 mathematical (LPPLS + HMM + Hurst) + 1 semantic (RAG)
Eval:           34 cases, crash precision 50%, recall 23%
Dashboard:      streamlit run dashboard.py
```

**Архитектура:**

```
[Data Layer]
  yfinance API → prices
  24 RSS feeds → news context
  DuckDuckGo → fallback search

[Signal Layer]
  HMM Regime Detector → NORMAL/GROWTH/BUBBLE (advisory gate)
  LPPLS Optimizer → tc, m, omega, quality score
  Hurst Exponent → trend persistence (H > 0.65 = supports bubble)

[Decision Layer]
  Verdict Contract → 5 rules combining LPPLS + HMM + RAG
  Sornette Filters → m ∈ (0.2, 0.7), ω ∈ (6, 13), damping > 1.0

[Operational Layer]
  Persistence Filter → TENTATIVE (1-2 scans) → CONFIRMED (3+)
  Telegram Alerter → only CONFIRMED signals
  Sector Scan → 8 sectors, 59 tickers, contagion detection
  Streamlit Dashboard → 4 tabs (Signals, Sectors, History, Eval)
  Task Scheduler → daily at 18:00
```

---

## 3. Live Predictions (зафиксированы 6 апреля 2026)

### Активные сигналы

| Ticker | Verdict | tc | Quality | Hurst | HMM | Price | Verify by |
|--------|---------|-----|---------|-------|-----|-------|-----------|
| **TSLA** | POSSIBLE | 2 Aug 2026 | 0.85 | 0.726 (persistent) | NORMAL | $360.59 | 1 Sep 2026 |
| **NVDA** | POSSIBLE | 20 Sep 2026 | 0.80 | 0.633 (neutral) | GROWTH | $177.39 | 20 Oct 2026 |

### Интерпретация

**TSLA** — самый сильный сигнал:
- LPPLS quality 0.85 (выше порога 0.75)
- Hurst 0.726 (persistent trend, supports bubble)
- HMM = NORMAL (противоречие — HMM не видит пузырь, но Q >= 0.8 сохраняет сигнал)
- tc = 2 августа 2026 — если коррекция начнётся в июле-августе, модель права

**NVDA** — умеренный сигнал:
- LPPLS quality 0.80
- Hurst 0.633 (borderline, не persistent)
- HMM = GROWTH (не bubble, но растёт)
- tc = 20 сентября 2026 — дальше горизонт, менее надёжно

### Наблюдаемые, но не активные

| Ticker | HMM | Hurst | Note |
|--------|-----|-------|------|
| **Gold** ($4,676) | BUBBLE | 0.744 | LPPLS omega вне range — не пропускает фильтр |
| **S&P 500** | BUBBLE | 0.559 | HMM видит пузырь, LPPLS нет |
| **MSTR** ($120) | BUBBLE | 0.761 | Q=0.50, не проходит порог |

### 18 активов: NO_BUBBLE
AAPL, MSFT, GOOGL, META, AMZN, PLTR, ARM, AVGO, BTC, ETH, SOL, SI=F, CL=F, NASDAQ, DJI — стабильны.

---

## 4. Честные метрики

### Что модель РЕАЛЬНО предсказывает

Crash Eval (34 кейса, "упал >30% за 6 месяцев?"):

| Метрика | Значение |
|---------|----------|
| Crash Precision | **50%** (3 из 6 предсказаний верны) |
| Crash Recall | **23%** (3 из 13 крахов пойманы) |
| F1 | **32%** |
| Accuracy | **62%** |

Модель ловит **классические пузыри** (BTC 2017, China 2015, Silver 2011) но пропускает **мем-стоки** (GME, DOGE, AMC) и **быстрые спайки**. Это ограничение физики LPPLS — модель ищет лог-периодические осцилляции, а мем-стоки = прямая линия вверх.

### FP rate на random walks
- fit_lppls_simple: **7%** (100 random walks, seed=42)
- С HMM gate на реальных данных: ещё ниже (KO, JNJ downgraded)

---

## 5. Все возможности системы

### Команды

```bash
# === PhaseBreak 2026 (Academic) ===
python -m src.cli scan NVDA TSLA BTC-USD       # Сканировать активы
python -m src.benchmark.v2_benchmark            # Полный бенчмарк (58 эпизодов)
python -m src.benchmark.v2_ablation             # Ablation study
python -m pytest tests/ -v                       # 268 тестов

# === PhaseBreak Hybrid 2.0 (Product) ===
python predict_all.py                            # Полный скан всех доменов
python predict_all.py --no-rag                   # Без RAG (быстрее)
python monitor_cron.py --dry-run                 # Мониторинг без Telegram
python monitor_cron.py --loop --interval 6       # Каждые 6 часов
python lock_predictions.py                       # Зафиксировать прогнозы
python -m src.contract.crash_eval                # Crash-oriented eval
python -m src.contract.eval_runner --no-rag      # Blind eval
python -m src.pipeline.sector_scan               # Sector heatmap
streamlit run dashboard.py                       # Web dashboard
```

### Файловая структура Hybrid 2.0

```
PhaseBreak Hybrid 2.0/
├── predict_all.py              # Main pipeline (finance + commodities + crypto + housing)
├── monitor_cron.py             # Scheduled monitor + alerts
├── lock_predictions.py         # Lock predictions to JSON
├── dashboard.py                # Streamlit dashboard (4 tabs)
├── register_task.bat           # Windows Task Scheduler setup
│
├── src/
│   ├── lppls/
│   │   ├── ds_filters.py       # LPPLS fitter + Sornette filters + DS confidence
│   │   ├── regime.py           # HMM 3-state regime detector
│   │   ├── hmm_gate.py         # Advisory HMM gate wrapper
│   │   └── hurst_signal.py     # Hurst exponent (R/S analysis)
│   │
│   ├── hybrid/
│   │   ├── rag_engine.py       # RAG: RSS + DuckDuckGo + OpenAI + 24h cache
│   │   ├── rss_provider.py     # RSS feed aggregation (24 feeds)
│   │   ├── prompts.py          # LLM prompts (neutral, no leakage)
│   │   ├── orchestrator.py     # HybridOrchestrator
│   │   └── housing.py          # Housing data loader (Zillow ZHVI)
│   │
│   ├── pipeline/
│   │   ├── alerter.py          # Telegram alerts (httpx POST)
│   │   ├── persistence.py      # Signal stability (3-scan confirm)
│   │   ├── sector_scan.py      # Parallel sector scan (8 sectors, 59 tickers)
│   │   └── delta_tracker.py    # Prediction delta tracking
│   │
│   └── contract/
│       ├── verdict_contract.py # 5-rule verdict (LPPLS + DS + RAG + Sornette + violations)
│       ├── eval_runner.py      # Eval set runner (blind mode)
│       ├── crash_eval.py       # Crash-oriented eval
│       └── domain_status.py    # Domain maturity registry
│
├── configs/
│   ├── rss_feeds.yaml          # 24 RSS feeds + 17 ticker aliases
│   └── sectors.yaml            # 8 sectors, 59 tickers
│
├── data/
│   ├── eval_set.json                    # 34 eval cases
│   ├── eval_crash.json                  # 34 crash-oriented cases
│   ├── LOCKED_PREDICTIONS_20260406.json # Immutable predictions
│   ├── signal_history.json              # Persistence tracker data
│   ├── live_scan_20260405.json          # Last live scan
│   └── various eval results...
│
├── tests/
│   ├── test_new_modules.py     # 10 tests for RSS, Alerter, Sectors
│   └── test_prompts.py         # Prompt template tests
│
└── docs/
    ├── TZ_WEEK_SPRINT.md       # Sprint plan and status
    └── FINAL_REPORT.md         # This file
```

---

## 6. Сравнение: было vs стало

### PhaseBreak 2026

| Аспект | До аудита | После |
|--------|-----------|-------|
| Paper precision | 80% (stale) | **78% finance / 76% overall** (from JSON) |
| Test count | "253" | **268** |
| Figures | Hardcoded 0.80 | **Reads from benchmark JSON** |
| Council validator | is_bubble leakage | **Fixed: no leakage, debiased parser** |
| plotly dep | In pyproject.toml | **Removed** |
| omega docs | 6-13 only | **Clarified: optimizer 6-13, filter 5.0-13.5** |

### PhaseBreak Hybrid 2.0

| Аспект | До аудита | После |
|--------|-----------|-------|
| Signals | 1 (LPPLS only) | **3 (LPPLS + HMM + Hurst) + RAG** |
| Context | DuckDuckGo scraping | **24 RSS feeds + DDG fallback** |
| Eval | 70% with leakage | **71% blind (with RAG), 65% LPPLS-only** |
| Crash eval | Didn't exist | **Precision 50%, Recall 23%** |
| Monitoring | Manual | **Auto daily at 18:00 + Telegram** |
| Signal stability | Flip-flop | **Persistence filter (3-scan confirm)** |
| Dashboard | None | **Streamlit 4-tab dashboard** |
| Predictions | Ephemeral | **Locked in git JSON** |

---

## 7. Известные ограничения

1. **LPPLS не ловит мем-стоки** — GME, DOGE, AMC имеют m ≈ 0.1 (вертикальный рост без осцилляций). Ограничение физики модели.

2. **Recall 23% на crash prediction** — модель пропускает 77% реальных крахов. Precision-first дизайн: лучше не предсказать, чем предсказать ложно.

3. **RSS не покрывает исторические данные** — для кейсов до 2024 RSS бесполезен (показывает сегодняшние новости). Нужен NewsAPI или GDELT для исторического контекста.

4. **Geology domain заблокирован** — нет Sentinel-2 данных на диске. TOST equivalence n=13 < n=15 recommended.

5. **Signal instability** — 5/5 BUBBLE сигналов от 29 марта исчезли к 5 апреля. Persistence filter смягчает, но не решает.

6. **Нет prospective validation** — прогнозы зафиксированы, но проверка будет только через 4-6 месяцев (TSLA Aug, NVDA Sep).

---

## 8. Что дальше

### Ближайшие проверки
- **1 мая 2026** — месячный чекпоинт: TSLA/NVDA тренд сохраняется?
- **1 августа 2026** — TSLA tc verification (±30 дней)
- **20 сентября 2026** — NVDA tc verification

### Технические улучшения (backlog)
- Hurst как сигнал в verdict_contract (сейчас информационный)
- News volume spike detection (из RSS cache)
- Options data (VIX, put/call ratio) как 4-й сигнал
- pgvector для исторических событий (RAG + embeddings)

### Публикация
- arXiv: paper ready, ждём endorsement
- Geology: нужны Sentinel-2 данные для n>=15
- Replication pack: reproduce.sh создан
