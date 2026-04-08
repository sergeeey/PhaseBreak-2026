"""stress_test_random.py — Red Team: random data → should produce NO_BUBBLE."""

import numpy as np
import sys

sys.path.insert(0, "src")

from lppls.optimizer import LPPLSOptimizer

np.random.seed(42)
n_tests = 50
false_positives = 0
fp_details = []

for i in range(n_tests):
    # Random walk (cumulative sum of normal returns)
    returns = np.random.normal(0, 0.02, 252)
    prices = 100 * np.exp(np.cumsum(returns))
    t = np.arange(len(prices), dtype=float)
    log_p = np.log(prices)

    opt = LPPLSOptimizer()
    model = opt.fit(t, log_p)
    params = model.params
    r2 = model.r_squared(t, log_p)

    if params.is_bubble and r2 > 0.5:
        false_positives += 1
        fp_details.append(
            f"  [FP #{false_positives}] trial={i}, tc={params.tc:.1f}, "
            f"m={params.m:.3f}, omega={params.omega:.3f}, R2={r2:.3f}"
        )

for d in fp_details:
    print(d)

fp_rate = false_positives / n_tests
print(f"\n{'=' * 50}")
print(f"FALSE POSITIVE RATE: {false_positives}/{n_tests} = {fp_rate:.1%}")
if fp_rate > 0.10:
    print("VERDICT: FAIL — model hallucinates bubbles in noise")
else:
    print("VERDICT: PASS — noise correctly rejected")
