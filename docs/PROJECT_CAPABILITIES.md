# PhaseBreak 2026 — Полное описание возможностей и практической пользы

**Версия:** Unified (academic + product)
**Дата:** 6 апреля 2026
**Размер:** 77 Python модулей, 12,820 строк кода, 278 тестов, 34 коммита

---

## 1. Что это за проект

PhaseBreak — система предсказания финансовых крахов и обнаружения пузырей на основе математической модели LPPLS (Log-Periodic Power Law Singularity) профессора Дидье Сорнетте.

**Главная ценность:** предсказывает не просто "будет ли крах", а **КОГДА** — конкретную дату (tc) с доверительным интервалом.

**Аналогия:** сейсмолог не может предсказать землетрясение по одному датчику. Но если 3 датчика (сейсмограф + деформация грунта + уровень воды в колодцах) показывают одно и то же — вероятность высокая. PhaseBreak делает то же самое: 4 независимых сигнала (LPPLS + HMM + Hurst + RAG) должны согласоваться.

---

## 2. Кому полезен

| Аудитория | Как использовать | Что получает |
|-----------|------------------|-------------|
| **Трейдер / инвестор** | `python predict_all.py` | Список активов с bubble risk, дата возможного краха |
| **Риск-менеджер** | `python -m src.pipeline.sector_scan` | Heatmap секторов — где накапливается системный риск |
| **Финансовый аналитик** | `streamlit run dashboard.py` | Dashboard с историей сигналов и eval метриками |
| **Исследователь** | `python -m src.benchmark.v2_benchmark` | 58 эпизодов, ablation study, TOST universality |
| **Регулятор / центробанк** | `python monitor_cron.py --loop` | Автоматический мониторинг с Telegram-алертами |

---

## 3. Все возможности (по модулям)

### 3.1 Предсказание крахов (Core LPPLS)

**Что делает:** Находит активы с супер-экспоненциальным ростом и лог-периодическими осцилляциями — математическую подпись пузыря.

**Команды:**
```bash
# Сканировать конкретные активы
python -m src.cli scan NVDA TSLA BTC-USD GC=F

# Полный скан всех доменов (45+ активов)
python predict_all.py

# Только финансы / только крипту
python predict_all.py --finance-only
python predict_all.py --crypto-only
```

**Что получаешь:**
- Verdict: BUBBLE / POSSIBLE / NO_BUBBLE
- tc — предсказанная дата краха (±10-30 дней)
- Quality score (0-1) — уверенность модели
- R² — качество математического фита
- Параметры: m (скорость роста), omega (частота осцилляций), damping

**Покрытие:**
- 18 акций (NVDA, TSLA, AAPL, MSFT, GOOGL, META, AMZN, ARM, AVGO, PLTR, MSTR, COIN + индексы)
- 7 commodities (Gold, Silver, Oil, Brent, Natural Gas, Copper, Wheat)
- 8 криптовалют (BTC, ETH, SOL, XRP, ADA, DOGE, AVAX, DOT)
- 10 рынков жилья (New York, Los Angeles, Miami, Phoenix, Austin и др.)

---

### 3.2 Мульти-сигнальный анализ (HMM + Hurst + LPPLS)

**Что делает:** 3 независимых математических сигнала + 1 семантический:

| Сигнал | Метод | Что измеряет |
|--------|-------|-------------|
| **LPPLS** | Sornette equation fitting | Лог-периодические осцилляции перед крахом |
| **HMM** | 3-state Hidden Markov Model | Режим рынка (NORMAL / GROWTH / BUBBLE) |
| **Hurst** | R/S analysis | Persistence тренда (H>0.65 = тренд устойчивый) |
| **RAG** | RSS + LLM (GPT-4o) | Семантический анализ новостей (фундамент vs спекуляция) |

**Логика:** Один сигнал может ошибаться. Если 3 из 4 согласны — вероятность реального пузыря выше.

---

### 3.3 Мониторинг в реальном времени

**Что делает:** Автоматически сканирует рынок по расписанию и шлёт алерты.

**Команды:**
```bash
# Одноразовый прогон (без отправки)
python monitor_cron.py --dry-run

# Мониторинг каждые 6 часов
python monitor_cron.py --loop --interval 6

# Только секторный скан
python monitor_cron.py --sectors-only

# Автоматический запуск (Windows Task Scheduler)
register_task.bat    # → ежедневно в 18:00
```

**Что получаешь:**
- Сканирование 18 активов watchlist
- Sector scan (8 секторов, 59 тикеров)
- Persistence filter — сигнал считается надёжным после 3 подтверждений подряд
- Telegram-алерт при CONFIRMED сигнале (нужен бот)
- Лог в `data/monitor_log.json`

---

### 3.4 Sector Scan (системный риск)

**Что делает:** Параллельно анализирует 59 тикеров в 8 секторах. Если 3+ в одном секторе показывают BUBBLE/POSSIBLE — это sector alert (заражение).

**Команда:**
```bash
python -m src.pipeline.sector_scan --output data/heatmap.json
```

**Секторы:**
- Technology (8): NVDA, AAPL, MSFT, GOOGL, META, AMZN, CRM, ORCL
- Semiconductors (8): NVDA, AMD, AVGO, ARM, INTC, TSM, QCOM, MU
- Financials (8): JPM, BAC, GS, MS, WFC, C, BLK, SCHW
- Energy (8): XOM, CVX, COP, SLB, EOG, MPC, VLO, OXY
- Healthcare (8): UNH, JNJ, LLY, PFE, ABBV, MRK, TMO, ABT
- Consumer (8): TSLA, NKE, SBUX, MCD, HD, LOW, TGT, COST
- Crypto (6): BTC, ETH, SOL, XRP, ADA, DOGE
- Commodities (5): Gold, Silver, Oil, NatGas, Copper

**Heat Score:** 0% (всё спокойно) → 100% (весь сектор в пузыре)

---

### 3.5 RAG Analysis (новостной контекст)

**Что делает:** Собирает новости из 24 RSS лент + DuckDuckGo, отправляет в GPT-4o для анализа. Определяет: рост вызван фундаментальными причинами или спекуляцией.

**Источники:**
- Bloomberg, Reuters, CNBC, MarketWatch, Yahoo Finance, FT, WSJ
- Seeking Alpha, ZeroHedge (analysis)
- Federal Reserve, ECB, IMF (macro)
- CoinDesk, CoinTelegraph, Decrypt (crypto)
- OilPrice, Kitco, Mining.com (commodities)

**RAG добавляет контекст к математике:**
- LPPLS говорит POSSIBLE на NVDA → RAG: "revenue +73% YoY, real AI demand" → HOLD
- LPPLS говорит POSSIBLE на мем-стоке → RAG: "retail frenzy, no fundamentals" → AVOID

---

### 3.6 Verdict Contract (система правил)

**Что делает:** 5 формальных правил комбинирования LPPLS + RAG + HMM сигналов:

1. **DS-LPPLS override:** если multi-window confidence = 0, даже single-window BUBBLE downgrade до POSSIBLE
2. **RAG fundamental override:** если RAG уверен в fundamental growth (>0.8) → BUBBLE → NO_BUBBLE
3. **Both confirm:** если LPPLS + RAG оба видят пузырь → strengthen до BUBBLE
4. **Sornette filter:** если Sornette фильтры не пройдены → BUBBLE → POSSIBLE
5. **RAG speculative upgrade:** если RAG уверен в спекуляции → NO_BUBBLE → POSSIBLE

Каждое решение записывается с причиной. Contract violations логируются.

---

### 3.7 Persistence Filter (стабильность сигналов)

**Что делает:** Решает проблему "flip-flop" — когда сигнал появляется и исчезает между сканами.

**Логика:**
- **TENTATIVE** — сигнал появился 1-2 раза. Наблюдаем, не алертим.
- **CONFIRMED** — сигнал держится 3+ сканов подряд. Надёжный, шлём alert.
- **DROPPED** — был активный сигнал, теперь исчез. Логируем.

**Почему важно:** 5 из 5 BUBBLE сигналов от 29 марта исчезли через неделю. Без persistence filter каждый из них отправил бы ложный alert.

---

### 3.8 Dashboard (визуализация)

**Команда:**
```bash
streamlit run dashboard.py
```

**4 вкладки:**
1. **Signals** — текущие сигналы с таблицей (ticker, price, verdict, quality, HMM, Hurst, tc)
2. **Sectors** — heatmap секторов с progress bars
3. **History** — история сигналов по тикеру, график quality score
4. **Eval** — crash precision/recall/F1, blind eval accuracy

---

### 3.9 Crash Evaluation (честные метрики)

**Что делает:** Измеряет то, что модель реально предсказывает: "упадёт ли актив >30% за 6 месяцев?"

**Команда:**
```bash
python -m src.contract.crash_eval
```

**Текущие метрики (34 кейса):**
- Crash Precision: 50% (3 из 6 предсказаний верны)
- Crash Recall: 23% (3 из 13 крахов пойманы)
- Accuracy: 62%

**Что ловит:** BTC 2017 (-84%), China 2015 (-45%), Silver 2011 (-45%)
**Что пропускает:** мем-стоки (GME, DOGE, AMC) — слишком быстрый рост без осцилляций

---

### 3.10 Academic Pipeline (для arXiv)

**Команды:**
```bash
# Полный бенчмарк (58 эпизодов, 6 доменов)
python -m src.benchmark.v2_benchmark

# Ablation study (вклад каждого компонента)
python -m src.benchmark.v2_ablation

# Сравнение с baselines
python -m src.benchmark.baselines

# TOST universality (finance ↔ commodities)
python -m pytest tests/test_cross_domain.py -v

# Все 278 тестов
python -m pytest tests/ -v
```

**Результаты:**
- 58 benchmark эпизодов: 76% precision, 61% recall
- TOST equivalence: finance ↔ commodities (p=0.031 для m)
- Forward validation 2024-25: 3/3 bubbles, 0 FP
- Adversarial: 100% (6/6 синтетических edge cases)

---

### 3.11 Locked Predictions (prospective validation)

**Что делает:** Фиксирует прогнозы в JSON с timestamp. Через месяцы можно проверить: сбылось или нет.

**Команда:**
```bash
python lock_predictions.py
```

**Текущие locked predictions (6 апреля 2026):**

| Ticker | Verdict | tc | Price | Verify by |
|--------|---------|-----|-------|-----------|
| TSLA | POSSIBLE | 2 Aug 2026 | $360.59 | 1 Sep 2026 |
| NVDA | POSSIBLE | 3 Oct 2026 | $177.39 | 3 Nov 2026 |

---

## 4. Практическая польза (сценарии использования)

### Сценарий 1: "Стоит ли мне сейчас покупать NVDA?"

```bash
python -m src.cli scan NVDA
```
→ POSSIBLE (Q=0.80, tc Oct 2026) — модель видит параболический рост, но Hurst=0.63 (не persistent). RAG: "revenue +73%, real demand".
**Вывод:** рост фундаментальный, но есть vulnerability window в Q3-Q4 2026. HOLD with caution.

### Сценарий 2: "Есть ли пузырь на рынке золота?"

```bash
python -m src.cli scan GC=F
```
→ NO_BUBBLE формально, но HMM=BUBBLE (prob=1.0) + Hurst=0.744. Gold вырос 101% за 2 года, но уже корректируется (-12% от пика).
**Вывод:** не классический пузырь (нет осцилляций), но перегрет. Коррекция до $4,200-4,400 вероятна.

### Сценарий 3: "Какой сектор самый рискованный?"

```bash
python -m src.pipeline.sector_scan
```
→ Heatmap показывает все секторы на 0% (апрель 2026 — рынок спокоен). Но если в tech появятся 3+ POSSIBLE → sector alert.

### Сценарий 4: "Хочу получать алерты о пузырях"

```bash
# 1. Создать Telegram бота через @BotFather
# 2. Добавить в .env:
#    TELEGRAM_BOT_TOKEN=your_token
#    TELEGRAM_CHAT_ID=your_chat_id
# 3. Запустить мониторинг
python monitor_cron.py --loop --interval 6
```
→ Каждые 6 часов: скан рынка → если CONFIRMED сигнал → Telegram alert с тикером, verdict, tc, quality.

### Сценарий 5: "Я исследователь, хочу воспроизвести результаты"

```bash
git clone https://github.com/sergeeey/PhaseBreak-2026.git
cd PhaseBreak-2026
pip install -e ".[dev,v2,science]"
python -m pytest tests/ -v           # 278 тестов
python -m src.benchmark.v2_benchmark  # 58 эпизодов
```

---

## 5. Чего система НЕ может

| Ограничение | Почему | Обходной путь |
|-------------|--------|---------------|
| Не ловит мем-стоки (GME, DOGE) | Рост слишком вертикальный (m≈0.1), нет осцилляций | RAG может поймать через "retail frenzy" в новостях |
| Не различает "почему растёт" | LPPLS видит форму роста, не причину | RAG добавляет фундаментальный/спекулятивный контекст |
| Сигналы нестабильны между сканами | LPPLS чувствителен к последним точкам данных | Persistence filter (3-scan confirm) |
| Recall 23% на crash prediction | Precision-first дизайн | Принять: лучше пропустить крах, чем дать ложный alarm |
| Исторический RAG слаб | RSS даёт сегодняшние новости, не 2017 года | Для исторических кейсов — только LPPLS |
| Не предсказывает exogenous shocks | Войны, пандемии, политика — вне модели | RAG частично покрывает геополитику |

---

## 6. Техническая архитектура (одна картинка)

```
                    ┌─────────────┐
                    │  yfinance   │
                    │  (prices)   │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
        ┌─────▼─────┐ ┌───▼───┐ ┌─────▼─────┐
        │    HMM    │ │ LPPLS │ │   Hurst   │
        │  regime   │ │  fit  │ │ exponent  │
        └─────┬─────┘ └───┬───┘ └─────┬─────┘
              │            │            │
              └────────────┼────────────┘
                           │
                    ┌──────▼──────┐     ┌──────────┐
                    │   Verdict   │◄────│ RAG/LLM  │
                    │  Contract   │     │ (24 RSS  │
                    │  (5 rules)  │     │ + GPT-4o)│
                    └──────┬──────┘     └──────────┘
                           │
                    ┌──────▼──────┐
                    │ Persistence │
                    │   Filter    │
                    │ (3-scan)    │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
        ┌─────▼─────┐ ┌───▼───┐ ┌─────▼─────┐
        │ Telegram  │ │ JSON  │ │ Dashboard │
        │  Alert    │ │  Log  │ │ Streamlit │
        └───────────┘ └───────┘ └───────────┘
```

---

## 7. Все команды (шпаргалка)

```bash
# === ПРОГНОЗЫ ===
python predict_all.py                    # полный скан (45+ активов)
python -m src.cli scan NVDA TSLA GC=F    # конкретные тикеры
python lock_predictions.py               # зафиксировать прогнозы

# === МОНИТОРИНГ ===
python monitor_cron.py --dry-run         # одноразовый прогон
python monitor_cron.py --loop            # каждые 6 часов
python monitor_cron.py --sectors-only    # только секторы
register_task.bat                        # автозапуск Windows

# === АНАЛИЗ ===
python -m src.pipeline.sector_scan       # sector heatmap
python -m src.contract.crash_eval        # crash precision/recall
python -m src.contract.eval_runner       # blind eval (34 cases)
streamlit run dashboard.py               # web dashboard

# === НАУКА ===
python -m src.benchmark.v2_benchmark     # 58 эпизодов
python -m src.benchmark.v2_ablation      # ablation study
python -m pytest tests/ -v               # 278 тестов
```
