# PhaseBreak API Server & Dashboard

FastAPI backend + React dashboard for PhaseBreak LPPLS bubble detection.

## Architecture

```
service/
├── server/              # FastAPI backend
│   ├── main.py         # API endpoints
│   └── requirements.txt
└── frontend/           # React + TypeScript + Tailwind dashboard
    ├── src/
    │   └── App.tsx     # Main dashboard component
    └── package.json
```

## Quick Start

### 1. Start API Server

```bash
# From project root
pip install fastapi uvicorn[standard]
python -m service.server.main

# Or
uvicorn service.server.main:app --reload --port 8000
```

API will be available at: http://localhost:8000
Interactive docs: http://localhost:8000/docs

### 2. Start Dashboard

```bash
cd service/frontend
npm install
npm start
```

Dashboard will be available at: http://localhost:3000

## API Endpoints

### Health Check
```bash
GET /api/v1/health
```

### Scan Assets
```bash
POST /api/v1/scan
{
  "tickers": ["NVDA", "BTC-USD", "SPY"],
  "window_months": 12,
  "domain": "finance"
}
```

### Scan Single Ticker
```bash
GET /api/v1/scan/NVDA?window_months=12&domain=finance
```

### Get Scorecard
```bash
GET /api/v1/scorecard
```

### List Domains
```bash
GET /api/v1/domains
```

### Get Domain Episodes
```bash
GET /api/v1/domains/finance/episodes
```

### Get Signal History
```bash
GET /api/v1/history/NVDA?limit=30
```

### Get Latest Signals
```bash
GET /api/v1/signals
```

### Get Benchmark Summary
```bash
GET /api/v1/benchmark
```

## Dashboard Features

- **Latest Signals Tab**: View most recent bubble detection results with quality scores, tc dates, and HMM regimes
- **Scan Assets Tab**: Real-time scanning of any tickers with live results
- **History Tab**: Track signal evolution over time (requires monitor runs)
- **Benchmark Tab**: Visualize benchmark results with charts (58 episodes, 6 domains)

## Integration with PhaseBreak Pipeline

The API server directly imports from `src.pipeline.stages`:
- `run_full_pipeline()` — v2 recommended path
- Uses same LPPLS optimizer, HMM gating, soft scoring, bootstrap uncertainty
- Results are identical to CLI `python -m src.cli scan`

## Development

### Add New Endpoint

1. Define Pydantic model in `service/server/main.py`
2. Add endpoint function
3. Test with Swagger UI at `/docs`

### Customize Dashboard

1. Edit `service/frontend/src/App.tsx`
2. Add new components in `service/frontend/src/components/`
3. Charts use Recharts library

## Production Deployment

### API Server
```bash
uvicorn service.server.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Frontend
```bash
cd service/frontend
npm run build
# Deploy build/ to static hosting (Netlify, Vercel, etc.)
```

## Environment Variables

Create `.env` file in project root:
```bash
# Optional: configure API server
PORT=8000
HOST=0.0.0.0
LOG_LEVEL=info
```
