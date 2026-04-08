"""STRESS TEST: Feed random/synthetic data to PhaseBreak pipeline.

Run: python stress_test.py
Expected: All random walks should return NO_BUBBLE.
If ANY returns BUBBLE/POSSIBLE → model is broken.
"""
import numpy as np
from src.pipeline.stages import run_full_pipeline

np.random.seed(42)
N_TESTS = 50
BUBBLE_COUNT = 0

print(f"Running {N_TESTS} stress tests with random data...\n")

for i in range(N_TESTS):
    # Test 1: Pure white noise
    prices = np.cumsum(np.random.randn(200)) + 100
    prices = np.abs(prices) + 1  # ensure positive
    
    t = np.arange(len(prices), dtype=float)
    result = run_full_pipeline(t, prices, domain="finance", n_bootstrap=5)
    
    if result.final_verdict in ("BUBBLE", "POSSIBLE"):
        BUBBLE_COUNT += 1
        print(f"  [{i+1}] FAIL: {result.final_verdict} on WHITE NOISE (quality={result.final_confidence:.3f})")
    elif i < 5:  # Show first 5 passed
        print(f"  [{i+1}] OK: {result.final_verdict} (quality={result.final_confidence:.3f})")

# Test 2: Random walk with drift (smooth growth, no bubble)
for i in range(10):
    prices = 100 + np.cumsum(np.random.randn(200) * 0.5 + 0.3)  # smooth uptrend
    prices = np.abs(prices) + 1
    
    t = np.arange(len(prices), dtype=float)
    result = run_full_pipeline(t, prices, domain="finance", n_bootstrap=5)
    
    if result.final_verdict in ("BUBBLE", "POSSIBLE"):
        BUBBLE_COUNT += 1
        print(f"  [drift-{i+1}] FAIL: {result.final_verdict} on RANDOM WALK DRIFT (quality={result.final_confidence:.3f})")

print(f"\n{'='*60}")
print(f"RESULTS: {BUBBLE_COUNT} false positives out of {N_TESTS + 10} tests")
if BUBBLE_COUNT == 0:
    print("PASS: Model correctly rejects all random data")
else:
    print(f"FAIL: {BUBBLE_COUNT}/{N_TESTS + 10} = {BUBBLE_COUNT/(N_TESTS+10)*100:.1f}% false positive rate on noise")
