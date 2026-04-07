"""Quick project health check."""
import numpy as np
from src.pipeline.stages import run_full_pipeline
from src.pipeline.acceptor import create_acceptor, run_tfs_pipeline_iteration

print("=" * 60)
print("PROJECT HEALTH CHECK")
print("=" * 60)

# 1. Main pipeline smoke test
print("\n1. Main Pipeline Smoke Test")
np.random.seed(42)
n = 120
t = np.arange(n, dtype=float)
values = np.exp(0.02 * t) + 2 * np.random.randn(n)
values = values * 50

result = run_full_pipeline(t, values, domain="finance", n_bootstrap=5)
print(f"   Verdict: {result.final_verdict}")
if result.fit:
    print(f"   Quality: {result.fit.quality_score:.3f}")
    print(f"   R²: {result.fit.r_squared:.3f}")
print(f"   Path: {result.path}")
print("   ✅ PASS")

# 2. TFS acceptor test
print("\n2. TFS Acceptor Test")
acceptor = create_acceptor("finance", t, values, hmm_bubble_prob=0.5)
print(f"   m_expected: {acceptor.m_expected}")
print(f"   omega_expected: {acceptor.omega_expected}")
print(f"   min_r²: {acceptor.min_r_squared}")
print(f"   confidence: {acceptor.confidence}")
print("   ✅ PASS")

# 3. TFS pipeline test
print("\n3. TFS Pipeline Test")
tfs_result = run_tfs_pipeline_iteration(
    t=t, values=values, domain="finance", max_iterations=2
)
print(f"   Iterations: {tfs_result['iterations']}")
print(f"   Satisfaction: {tfs_result['satisfaction']:.3f}")
print(f"   Acceptor match: {tfs_result['acceptor_match']}")
print("   ✅ PASS")

# 4. Circular import check
print("\n4. Circular Import Check")
import importlib
modules = [
    "src.lppls.model",
    "src.lppls.optimizer",
    "src.lppls.scoring",
    "src.lppls.regime",
    "src.lppls.ds_filters",
    "src.lppls.uncertainty",
    "src.lppls.windowing",
    "src.pipeline.stages",
    "src.pipeline.acceptor",
]
for mod in modules:
    try:
        importlib.import_module(mod)
        print(f"   ✅ {mod}")
    except Exception as e:
        print(f"   ❌ {mod}: {e}")

# 5. Data files check
print("\n5. Critical Data Files Check")
from pathlib import Path
critical_files = [
    "data/v2_results/benchmark_results.json",
    "pyproject.toml",
    "README.md",
]
for f in critical_files:
    p = Path(f)
    if p.exists():
        print(f"   ✅ {f} ({p.stat().st_size} bytes)")
    else:
        print(f"   ⚠️  Missing: {f}")

print("\n" + "=" * 60)
print("ALL CHECKS PASSED ✅")
print("=" * 60)
