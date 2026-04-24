"""Model comparison & evaluation pipeline.

Runs multiple models (baseline, XGBoost, LightGBM) on the **same**
TimeSeriesSplit folds, collects all metrics (MAE, RMSE, R², MAPE, sMAPE),
and exports a tidy comparison CSV.

Public API
----------
* ``compare_models``      – evaluate a list of models via time-series CV.
* ``build_comparison_table`` – end-to-end: baseline + XGB + LGBM → DataFrame.
* ``save_comparison``     – persist the comparison DataFrame as CSV.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.models import _get_estimator
from src.utils import calculate_metrics


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

class MeanBaseline:
    """Predicts the training-set mean for every sample."""

    def __init__(self):
        self._mean = 0.0

    def fit(self, X, y):
        self._mean = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self._mean)

    def __repr__(self):
        return 'MeanBaseline()'


class SeasonalNaiveBaseline:
    """Predicts the last observed value at the same day-of-year.

    Falls back to the global mean when no matching day-of-year exists in
    the training window (e.g. very short history).
    """

    def __init__(self):
        self._lookup: Dict[int, float] = {}
        self._fallback = 0.0

    def fit(self, X, y):
        y_arr = np.asarray(y)
        self._fallback = float(np.mean(y_arr))

        # Use 'dayofyear' if present, else positional index mod 365
        if hasattr(X, 'columns') and 'dayofyear' in X.columns:
            doy = X['dayofyear'].values
        else:
            doy = np.arange(len(y_arr)) % 365

        # Keep the *last* occurrence per day-of-year
        for d, val in zip(doy, y_arr):
            self._lookup[int(d)] = float(val)

        return self

    def predict(self, X):
        if hasattr(X, 'columns') and 'dayofyear' in X.columns:
            doy = X['dayofyear'].values
        else:
            doy = np.arange(len(X)) % 365

        return np.array([self._lookup.get(int(d), self._fallback) for d in doy])

    def __repr__(self):
        return 'SeasonalNaiveBaseline()'


# ---------------------------------------------------------------------------
# Core comparison logic
# ---------------------------------------------------------------------------

def _cv_evaluate(model, X, y, n_splits=5):
    """Run TimeSeriesSplit CV and return mean metrics across folds."""
    tss = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics: List[Dict[str, float]] = []

    for train_idx, valid_idx in tss.split(X):
        X_tr, X_va = X.iloc[train_idx], X.iloc[valid_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[valid_idx]

        model_clone = _clone_model(model)
        model_clone.fit(X_tr, y_tr)
        preds = model_clone.predict(X_va)
        fold_metrics.append(calculate_metrics(y_va, preds))

    if not fold_metrics:
        return {k: np.nan for k in ('MAE', 'RMSE', 'R2', 'MAPE', 'sMAPE')}

    return {
        k: float(np.mean([m[k] for m in fold_metrics]))
        for k in fold_metrics[0]
    }


def _clone_model(model):
    """Cheap clone: re-instantiate sklearn-style estimators via get_params."""
    if hasattr(model, 'get_params'):
        return model.__class__(**model.get_params())
    # For simple baselines, just create a new instance
    return model.__class__()


def compare_models(
    X: pd.DataFrame,
    y: pd.Series,
    target_name: str = 'target',
    models: Optional[Dict[str, Any]] = None,
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Evaluate multiple models via TimeSeriesSplit CV.

    Parameters
    ----------
    X, y      : features / target.
    target_name : label for the target column (used in output).
    models    : ``{name: estimator}`` dict.  If *None*, uses default set
                (MeanBaseline, SeasonalNaive, XGBoost, LightGBM).
    n_splits  : number of folds for TimeSeriesSplit.

    Returns
    -------
    DataFrame with one row per model and columns for each metric.
    """
    if models is None:
        models = {
            'MeanBaseline': MeanBaseline(),
            'SeasonalNaive': SeasonalNaiveBaseline(),
            'XGBoost': _get_estimator('xgboost', random_state=random_state),
            'LightGBM': _get_estimator('lightgbm', random_state=random_state),
        }

    rows: List[Dict[str, Any]] = []

    for name, model in models.items():
        print(f'  Evaluating {name:20s} ...', end='', flush=True)
        cv_metrics = _cv_evaluate(model, X, y, n_splits=n_splits)
        row = {'Model': name, 'Target': target_name, **cv_metrics}
        rows.append(row)
        print(f'  MAE={cv_metrics["MAE"]:.2f}  sMAPE={cv_metrics["sMAPE"]:.2f}%')

    return pd.DataFrame(rows)


def build_comparison_table(
    train_features: pd.DataFrame,
    feature_cols: List[str],
    target_cols: Sequence[str] = ('Revenue', 'COGS'),
    n_splits: int = 5,
    random_state: int = 42,
    extra_models: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """End-to-end comparison: baseline + XGB + LGBM for each target.

    Parameters
    ----------
    train_features : DataFrame with features **and** target columns.
    feature_cols   : columns to use as model input.
    target_cols    : targets to evaluate.
    extra_models   : additional ``{name: estimator}`` to include.

    Returns
    -------
    Concatenated DataFrame across all targets, sorted by Target then MAE.
    """
    X = train_features[feature_cols]
    all_dfs: List[pd.DataFrame] = []

    for target in target_cols:
        print(f'\n{"="*60}')
        print(f'  Comparing models for target: {target}')
        print(f'{"="*60}')

        y = train_features[target]

        models = {
            'MeanBaseline': MeanBaseline(),
            'SeasonalNaive': SeasonalNaiveBaseline(),
            'XGBoost': _get_estimator('xgboost', random_state=random_state),
            'LightGBM': _get_estimator('lightgbm', random_state=random_state),
        }
        if extra_models:
            models.update(extra_models)

        df = compare_models(
            X, y,
            target_name=target,
            models=models,
            n_splits=n_splits,
            random_state=random_state,
        )
        all_dfs.append(df)

    result = pd.concat(all_dfs, ignore_index=True)
    result = result.sort_values(['Target', 'MAE']).reset_index(drop=True)

    # Pretty print
    print(f'\n{"="*60}')
    print('  MODEL COMPARISON SUMMARY')
    print(f'{"="*60}')
    print(result.to_string(index=False, float_format='%.4f'))
    print()

    return result


def save_comparison(
    comparison_df: pd.DataFrame,
    out_path: str = 'output/model_comparison.csv',
) -> str:
    """Save comparison DataFrame to CSV.

    Returns the path written.
    """
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    comparison_df.to_csv(out_path, index=False, float_format='%.4f')
    print(f'📁 Comparison saved → {out_path}')
    return out_path
