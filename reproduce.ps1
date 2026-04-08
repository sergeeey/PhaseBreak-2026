# PhaseBreak 2026 — One-Command Replication Pack
# Usage: powershell -ExecutionPolicy Bypass -File reproduce.ps1
# Requirements: Python 3.11+, pip, git
$ErrorActionPreference = "Stop"

Write-Host "============================================"
Write-Host "  PhaseBreak 2026 — Full Replication"
Write-Host "============================================"

# Step 1: Create virtual environment
Write-Host ""
Write-Host "[1/5] Creating virtual environment..."
python -m venv .venv
& .\.venv\Scripts\Activate.ps1

# Step 2: Install dependencies
Write-Host "[2/5] Installing dependencies..."
pip install --upgrade pip -q
pip install -e ".[dev,v2,science,benchmark]" -q

# Step 3: Run tests
Write-Host "[3/5] Running test suite..."
python -m pytest tests/ -v --tb=short
if ($LASTEXITCODE -ne 0) { throw "Tests failed" }

# Step 4: Run benchmark
Write-Host "[4/5] Running v2 benchmark (58 episodes, ~2 min)..."
python -m src.benchmark.v2_benchmark
python -m src.benchmark.v2_ablation
python -m src.benchmark.baselines

# Step 5: Summary
Write-Host ""
Write-Host "[5/5] Results:"
Write-Host "  - Benchmark: data/v2_results/benchmark_results.json"
Write-Host "  - Ablation:  data/v2_results/ablation.json"
Write-Host "  - Baselines: data/v2_results/baselines.json"
Write-Host ""
Write-Host "============================================"
Write-Host "  Replication complete."
Write-Host "============================================"
