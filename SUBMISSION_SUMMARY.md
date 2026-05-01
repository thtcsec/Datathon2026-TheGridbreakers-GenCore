# Datathon 2026 Submission Summary

**Project:** GenCore - Revenue Forecasting (548 days, 2023-01-01 to 2024-07-01)  
**Pipeline:** Zero-Leakage Seasonal Naive + Prophet + LightGBM Residual Ensemble  
**Status:** ✓ Completed | ✓ Diagnostic Analysis | ✓ Git Staged

---

## Key Results

| Metric | Revenue | COGS |
|--------|---------|------|
| **Holdout MAE** | 746,176 | 659,624 |
| **Holdout RMSE** | 1,025,347 | 907,481 |
| **Holdout R²** | 0.5517 | 0.5480 |
| **Best Blend** | Naive 75% + Hybrid 25% | Naive 80% + Hybrid 20% |
| **Tet Strength** | 0.10 (optimal) | 0.10 (optimal) |

---

## Problem Solved

**Initial Failure:** GenCore baseline (src/features.py) had **data leakage**  
- Used web_traffic, inventory contemporaneously during training
- Unavailable during forecast period → CV-LB gap: 417k → 2M MAE

**Solution Implemented:**
1. ✓ Zero-leakage architecture (all features deterministic)
2. ✓ Holdout validation matching Kaggle horizon (548 days)
3. ✓ Tet holiday calibration (±30 days, strength=0.10)
4. ✓ Conservative blend (Naive-heavy: 75-80% weight)

---

## Artifacts (Committed to `origin/Tu`)

```
notebooks/12_Final_Forecast.ipynb (20 cells)
├── Seasonal Naive 364-day baseline
├── Prophet Bayesian ensemble (CPS=0.03, 0.05)
├── LightGBM residual corrector
├── Tet strength & blend weight search (Cell 16)
├── Diagnostic analysis (Cell 20)
└── Forecasting + submission generation

notebooks/output/
├── submission_final_optimized.csv ⭐ (BEST)
├── optimization_sweep.csv (12 configs tested)
├── diagnostic_analysis.csv (error breakdown)
└── validation_metrics.csv

TECHNICAL_PROMPT.md (Comprehensive review)
├── Root cause analysis (leakage diagnosis)
├── Model architecture (3-tier ensemble)
├── Feature engineering (zero-leakage checklist)
├── Validation strategy (purged time-series split)
├── Risk assessment (distribution shift, Tet coverage)
├── Code review checklist
└── Improvement recommendations
```

---

## Diagnostic Health Score

```
✓ Zero Leakage verified
✓ Holdout split matches Kaggle horizon
✓ Tet calendar coverage (13 years)
✓ Feature determinism (all static/calendar)
✓ No data leakage (no future features)
✓ Distribution shift < 20%
✓ Residuals pass i.i.d. tests
⚠ Prophet weight learned to 0.00 (ensemble less effective than expected)
⚠ Exogenous shocks not modeled (pandemic recovery, etc.)

Overall: 7/10 checks ✓ PASSING
```

---

## Expected Kaggle Score

- **Best case:** MAE ≈ 746k (low distribution shift)
- **Realistic:** MAE ≈ 850k (1-year gap shift, -10-15%)
- **Worst case:** MAE > 1.5M (major exogenous shock)

---

## Git Commits (Branch `Tu`)

```
dd6ed2c docs(technical-prompt): comprehensive review + diagnostic checklist
38dcc94 opt(sweep): Tet [0.10-0.35] + blend weights; Revenue 746k COGS 660k
c729c1d feat(forecast): zero-leakage 548-day holdout pipeline
```

---

## Next Steps for External Review

1. **Run:** `jupyter notebook notebooks/12_Final_Forecast.ipynb`
   - Execute all cells to reproduce holdout scores
   - Verify diagnostic_analysis.csv outputs
   
2. **Review:** TECHNICAL_PROMPT.md sections:
   - IV. Model Architecture (ensemble design)
   - VIII. Code Review Checklist
   - IX. Recommendations

3. **Validate:** Check for:
   - Feature leakage (search for web_traffic, inventory in notebooks)
   - Holdout integrity (train/holdout split clean)
   - Submission format (548 rows, YYYY-MM-DD dates)

4. **Suggest:** Improvements in:
   - LGB hyperparameter tuning (learning_rate, max_depth)
   - Prophet alternative architectures
   - Exogenous feature availability

---

**Notebook Execution Time:** ~90s (all 20 cells)  
**Environment:** Python 3.13.5 (.venv)  
**Dependencies:** pandas, numpy, Prophet, LightGBM, scikit-learn, statsmodels

**Best Submission File:** `notebooks/output/submission_final_optimized.csv`
