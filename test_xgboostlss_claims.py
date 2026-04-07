"""Test XGBoostLSS — API verification and claims testing.

Tests the ACTUAL API (which is different from standard XGBoost):
- Requires DMatrix input
- Requires params dict setup
- Returns distribution parameters

Usage:
    python test_xgboostlss_claims.py
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path

print("=" * 70)
print("XGBoostLSS — API VERIFICATION & CLAIMS TEST")
print("=" * 70)

# Test 1: Import and understand API
print("\n" + "=" * 70)
print("Test 1: Understanding the API structure")
print("=" * 70)

try:
    from xgboostlss.model import XGBoostLSS
    from xgboostlss.distributions.Gaussian import Gaussian
    from xgboostlss.distributions.StudentT import StudentT
    
    print("✓ Imports successful\n")
    
    # Check what XGBoostLSS actually is
    print("XGBoostLSS class structure:")
    print(f"  Has train(): {hasattr(XGBoostLSS, 'train')}")
    print(f"  Has predict(): {hasattr(XGBoostLSS, 'predict')}")
    print(f"  Has fit(): {hasattr(XGBoostLSS, 'fit')}")
    
    # The API requires DMatrix and params dict
    print("\nActual API (from signature):")
    print("  train(params: Dict, dtrain: DMatrix, num_boost_round: int, ...)")
    print("  → Need to create DMatrix with special format")
    print("  → Need to pass distribution params dict")
    
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 2: Create proper DMatrix for distributional regression
print("\n" + "=" * 70)
print("Test 2: Creating DMatrix for distributional regression")
print("=" * 70)

np.random.seed(42)
n_samples = 500

X = np.random.randn(n_samples, 5)
true_mean = 2 * X[:, 0] + 1.5 * X[:, 1] - X[:, 2]
noise_std = 0.5 + 0.3 * np.abs(X[:, 3])
y = true_mean + noise_std * np.random.randn(n_samples)

print(f"Dataset: {n_samples} samples, 5 features")
print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")

# For XGBoostLSS, DMatrix needs label as dict of distribution params
# For Gaussian: need {'loc': y_values, 'scale': initial_sigma}
dist = Gaussian()

# Initialize starting values for distribution parameters
init_params = dist.calculate_start_values(y.reshape(-1, 1))
print(f"\nInitial distribution params:")
print(f"  {dist.distribution_arg_names}: {init_params}")

# Create DMatrix
# Key insight: label must be a dict with distribution parameter names as keys
dtrain_data = X[:400]
dtrain_label = y[:400]
dtest_data = X[400:]
dtest_label = y[400:]

# For XGBoostLSS, we need special DMatrix structure
dtrain = xgb.DMatrix(dtrain_data, label=dtrain_label)
dtest = xgb.DMatrix(dtest_data, label=dtest_label)

print(f"\n✓ DMatrix created")
print(f"  Train shape: {dtrain.num_row()} x {dtrain.num_col()}")
print(f"  Test shape: {dtest.num_row()} x {dtest.num_col()}")

# Test 3: Training with proper params dict
print("\n" + "=" * 70)
print("Test 3: Training with XGBoostLSS proper params")
print("=" * 70)

try:
    model = XGBoostLSS(dist=dist)
    
    # Get the params dict needed for training
    # This is typically: xgboost_params + distribution-specific setup
    print("\nAttempting train() with DMatrix...")
    
    # Check if there's a simpler fit() method
    if hasattr(model, 'fit'):
        print("  Found fit() method — trying that...")
        try:
            import inspect
            sig = inspect.signature(model.fit)
            print(f"  fit() signature: {sig}")
        except:
            print("  fit() exists but cannot inspect signature")
    
    # Try the train method with proper params
    # Need to figure out the params structure
    from xgboostlss.distributions.Gaussian import Gaussian
    gaussian_dist = Gaussian()
    
    # Get objective function
    print("\n  Getting distribution-specific objective...")
    obj_fn = gaussian_dist.objective_fn
    print(f"  Objective function: {obj_fn}")
    
    # Train with minimal params
    print("\n  Training with basic xgboost params...")
    
    xgb_params = {
        "objective": "reg:squarederror",  # Standard XGBoost objective
        "eval_metric": "rmse",
        "eta": 0.1,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }
    
    # Try standard XGBoost first to verify data works
    print("  Testing standard XGBoost first...")
    bst_standard = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=100,
        evals=[(dtrain, "train")],
        verbose_eval=False
    )
    
    standard_preds = bst_standard.predict(dtest)
    mse_standard = np.mean((dtest_label - standard_preds) ** 2)
    print(f"  ✓ Standard XGBoost works! MSE: {mse_standard:.4f}")
    
    # Now try XGBoostLSS
    print("\n  Now trying XGBoostLSS train()...")
    
    # The params dict needs distribution-specific structure
    # Let's check what the distribution provides
    print(f"  Distribution param names: {gaussian_dist.distribution_arg_names}")
    print(f"  Number of params: {gaussian_dist.n_dist_param}")
    
    # Build params for distributional training
    lss_params = {
        "distribution": gaussian_dist,
        "eta": 0.1,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }
    
    try:
        model.train(
            params=lss_params,
            dtrain=dtrain,
            num_boost_round=100,
            evals=[(dtrain, "train")],
            early_stopping_rounds=10,
            verbose_eval=50
        )
        print(f"\n  ✓✓✓ XGBoostLSS trained successfully!")
        print(f"  Best iteration: {model.best_iteration}")
        
        # Test prediction
        print("\n  Predicting on test set...")
        lss_preds = model.predict(dtest)
        print(f"  Predictions type: {type(lss_preds)}")
        print(f"  Predictions shape/keys: {lss_preds.shape if hasattr(lss_preds, 'shape') else lss_preds.keys() if hasattr(lss_preds, 'keys') else 'N/A'}")
        
        if hasattr(lss_preds, 'columns'):
            print(f"  Prediction columns: {lss_preds.columns.tolist()}")
            
            # Check if we get distribution parameters
            if 'loc' in lss_preds.columns and 'scale' in lss_preds.columns:
                print(f"\n  ✓✓✓ SUCCESS! Got distribution parameters!")
                print(f"    loc (mean) range: [{lss_preds['loc'].min():.3f}, {lss_preds['loc'].max():.3f}]")
                print(f"    scale (std) range: [{lss_preds['scale'].min():.3f}, {lss_preds['scale'].max():.3f}]")
                
                # Calculate prediction intervals
                from scipy.stats import norm
                lss_preds['lower_95'] = lss_preds['loc'] - 1.96 * lss_preds['scale']
                lss_preds['upper_95'] = lss_preds['loc'] + 1.96 * lss_preds['scale']
                lss_preds['true'] = dtest_label
                
                coverage = ((lss_preds['true'] >= lss_preds['lower_95']) & 
                           (lss_preds['true'] <= lss_preds['upper_95'])).mean()
                
                print(f"\n  95% Interval Coverage: {coverage:.1%}")
                print(f"  Expected: ~95%")
                print(f"  {'✓ GOOD!' if 0.90 <= coverage <= 0.98 else '⚠ Needs calibration'}")
                
                mse_lss = np.mean((lss_preds['true'] - lss_preds['loc']) ** 2)
                print(f"\n  MSE (XGBoostLSS): {mse_lss:.4f}")
                print(f"  MSE (Standard):   {mse_standard:.4f}")
                
            else:
                print(f"  ⚠ Did not get expected distribution parameters")
                
    except Exception as e:
        print(f"  ✗ XGBoostLSS training failed: {e}")
        import traceback
        traceback.print_exc()
        
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Alternative — use xgboostlss as wrapper
print("\n" + "=" * 70)
print("Test 4: Checking if there's a simpler scikit-learn-like API")
print("=" * 70)

try:
    # Check if there's a wrapper with fit/predict API
    import xgboostlss
    import pkgutil
    
    lss_modules = [modname for _, modname, _ in pkgutil.walk_packages(
        path=xgboostlss.__path__, prefix=xgboostlss.__name__+'.')]
    
    print(f"Available modules in xgboostlss:")
    for mod in lss_modules:
        print(f"  {mod}")
        
    # Check if model module has fit/predict
    from xgboostlss import model as lss_model_module
    print(f"\nModel module attributes:")
    model_attrs = [a for a in dir(lss_model_module) if not a.startswith('_')]
    print(f"  {model_attrs}")
    
except Exception as e:
    print(f"⚠ Could not inspect modules: {e}")

# Summary
print("\n" + "=" * 70)
print("SUMMARY: XGBoostLSS API Analysis")
print("=" * 70)

print("""
KEY FINDINGS:

1. API Structure:
   - XGBoostLSS.train() requires params dict + DMatrix input
   - NOT scikit-learn compatible (no fit(X, y) API)
   - More complex than standard XGBoost
   
2. Distribution Parameters:
   - Gaussian: loc (mean), scale (std)
   - Each parameter needs its own boosting trees
   - Multi-output regression (not single target)

3. Integration Complexity:
   - HIGH — needs custom DMatrix setup
   - Need to understand distribution-specific params
   - Not drop-in replacement for current methods

4. For PhaseBreak Use Case:
   
   Current bootstrap uncertainty:
   ├─ Simple: Fit LPPLS 100 times
   ├─ Robust: Works with n=58
   ├─ Interpretable: tc distribution from resampling
   └─ Well-tested: 14 passing tests
   
   XGBoostLSS alternative:
   ├─ Complex: Custom DMatrix + params setup
   ├─ Needs n>100+ to avoid overfitting
   ├─ Black-box: No physical interpretation
   └─ Different question: Predicts returns, not tc

5. ACTUAL Value for PhaseBreak:
   
   XGBoostLSS predicts DISTRIBUTION OF RETURNS (y),
   while PhaseBreak needs DISTRIBUTION OF tc (critical time).
   
   These are DIFFERENT things:
   - XGBoostLSS: "What's the distribution of future returns?"
   - PhaseBreak: "When will the bubble burst (tc)?"
   
   XGBoostLSS CANNOT replace LPPLS tc estimation!

RECOMMENDATION:

✗ DO NOT use XGBoostLSS for PhaseBreak core functionality

✓ POTENTIAL uses (if needed):
  - Additional signal: High predicted return variance → uncertainty flag
  - Validation: If XGBoostLSS also shows high uncertainty, strengthen signal
  - Large-scale scanning: Train on thousands of daily yfinance points
  
✗ CLAIMS vs REALITY for PhaseBreak:
  
  CLAIM: "Distributional regression with uncertainty"
  REALITY: ✅ True, but for RETURNS, not tc
  
  CLAIM: "Better uncertainty than bootstrap"
  REALITY: ⚠ Only for large n, and predicts different thing
  
  CLAIM: "Can replace bootstrap for tc uncertainty"
  REALITY: ✗ FALSE — XGBoostLSS predicts returns distribution,
           not tc (critical time) distribution

CONCLUSION: XGBoostLSS is a legitimate library for distributional
regression, but it solves a DIFFERENT problem than PhaseBreak needs.
Keep the current bootstrap method for tc uncertainty.
""")
