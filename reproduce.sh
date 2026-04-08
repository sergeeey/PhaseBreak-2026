#!/bin/bash
# PhaseBreak 2026 — One-Command Replication Pack
# Usage: bash reproduce.sh
# Requirements: Python 3.11+, pip, git
set -e

echo "============================================"
echo "  PhaseBreak 2026 — Full Replication"
echo "============================================"

# Step 1: Create virtual environment
echo ""
echo "[1/5] Creating virtual environment..."
python -m venv .venv
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source .venv/Scripts/activate
else
    source .venv/bin/activate
fi

# Step 2: Install dependencies
echo "[2/5] Installing dependencies..."
pip install --upgrade pip -q
pip install -e ".[dev,v2,science,benchmark]" -q

# Step 3: Run tests
echo "[3/5] Running test suite..."
python -m pytest tests/ -v --tb=short

# Step 4: Run benchmark
echo "[4/5] Running v2 benchmark (58 episodes, ~2 min)..."
python -m src.benchmark.v2_benchmark
python -m src.benchmark.v2_ablation
python -m src.benchmark.baselines

# Step 5: Summary
echo ""
echo "[5/5] Results:"
echo "  - Benchmark: data/v2_results/benchmark_results.json"
echo "  - Ablation:  data/v2_results/ablation.json"
echo "  - Baselines: data/v2_results/baselines.json"
echo ""
echo "============================================"
echo "  Replication complete."
echo "============================================"
