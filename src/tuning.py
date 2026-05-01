"""Systematic hyperparameter tuning for XGBoost / LightGBM.

Uses **Optuna** (Bayesian TPE sampler) with **TimeSeriesSplit** validation
to avoid data leakage.  The module exposes three public helpers:

* ``tune_model``         – tune a single model for one target.
* ``tune_and_select_best`` – tune both XGB & LightGBM, pick the winner.
* ``save_tuning_artifacts``– persist model, params, and a human-readable report.
"""

from __future__ import annotations

import json
import logging
import os
import textwrap
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import optuna
from joblib import dump as joblib_dump
from sklearn.model_selection import TimeSeriesSplit

from src.models import _get_estimator
from src.utils import calculate_metrics

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Search‑space definitions
# ---------------------------------------------------------------------------

_XGB_SPACE: Dict[str, Any] = {
    'n_estimators':     ('int',        300, 1500),
    'max_depth':        ('int',          3,   10),
    'learning_rate':    ('float_log', 0.01,  0.3),
    'subsample':        ('float',      0.5,  1.0),
    'colsample_bytree': ('float',      0.5,  1.0),
    'min_child_weight': ('int',          1,   10),
    'reg_alpha':        ('float_log', 1e-8, 10.0),
    'reg_lambda':       ('float_log', 1e-8, 10.0),
}

_LGBM_SPACE: Dict[str, Any] = {
    'n_estimators':     ('int',        300, 1500),
    'max_depth':        ('int',          3,   10),
    'learning_rate':    ('float_log', 0.01,  0.3),
    'subsample':        ('float',      0.5,  1.0),
    'colsample_bytree': ('float',      0.5,  1.0),
    'num_leaves':       ('int',         15,  127),
    'reg_alpha':        ('float_log', 1e-8, 10.0),
    'reg_lambda':       ('float_log', 1e-8, 10.0),
}


def _get_search_space(model_name: str) -> Dict[str, Any]:
    name = model_name.lower()
    if name in {'xgboost', 'xgb'}:
        return _XGB_SPACE
    if name in {'lightgbm', 'lgbm'}:
        return _LGBM_SPACE
    raise ValueError(f'No search space defined for model: {model_name}')


def _sample_params(trial: optuna.Trial, space: Dict[str, Any]) -> Dict[str, Any]:
    """Sample hyper‑parameters from *space* using an Optuna *trial*."""
    params: Dict[str, Any] = {}
    for name, (kind, low, high) in space.items():
        if kind == 'int':
            params[name] = trial.suggest_int(name, low, high)
        elif kind == 'float':
            params[name] = trial.suggest_float(name, low, high)
        elif kind == 'float_log':
            params[name] = trial.suggest_float(name, low, high, log=True)
        else:
            raise ValueError(f'Unknown param kind: {kind}')
    return params


# ---------------------------------------------------------------------------
# Objective (Optuna)
# ---------------------------------------------------------------------------

def _make_objective(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    n_splits: int,
    random_state: int,
    metric: str,
):
    """Return a closure that Optuna can optimise."""
    space = _get_search_space(model_name)

    def objective(trial: optuna.Trial) -> float:
        params = _sample_params(trial, space)
        tss = TimeSeriesSplit(n_splits=n_splits)
        fold_scores: List[float] = []

        for train_idx, valid_idx in tss.split(X):
            X_tr, X_va = X.iloc[train_idx], X.iloc[valid_idx]
            y_tr, y_va = y.iloc[train_idx], y.iloc[valid_idx]

            model = _get_estimator(
                model_name=model_name,
                random_state=random_state,
                **params,
            )

            # XGBoost accepts `verbose` in fit(); LightGBM uses constructor
            # `verbosity` (already set to -1 in _get_estimator defaults).
            fit_kwargs: Dict[str, Any] = {'eval_set': [(X_va, y_va)]}
            if model_name.lower() in {'xgboost', 'xgb'}:
                fit_kwargs['verbose'] = False

            model.fit(X_tr, y_tr, **fit_kwargs)
            preds = model.predict(X_va)
            metrics = calculate_metrics(y_va, preds)
            fold_scores.append(metrics[metric])

        return float(np.mean(fold_scores))

    return objective


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def tune_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str = 'xgboost',
    n_trials: int = 50,
    n_splits: int = 5,
    random_state: int = 42,
    metric: str = 'MAE',
    show_progress_bar: bool = True,
) -> Dict[str, Any]:
    """Tune *model_name* and return best params + CV metrics.

    Parameters
    ----------
    X, y : training features / target.
    model_name : ``'xgboost'`` or ``'lightgbm'``.
    n_trials : number of Optuna trials (default 50).
    n_splits : folds for ``TimeSeriesSplit`` (default 5).
    metric : optimisation target — ``'MAE'`` or ``'RMSE'``.

    Returns
    -------
    dict with keys ``best_params``, ``best_score``, ``model_name``, ``study``.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=random_state),
        study_name=f'{model_name}_tuning',
    )

    objective = _make_objective(
        X, y,
        model_name=model_name,
        n_splits=n_splits,
        random_state=random_state,
        metric=metric,
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=show_progress_bar,
    )

    logger.info(
        '%s best %s = %.4f  (trial %d)',
        model_name, metric, study.best_value, study.best_trial.number,
    )

    return {
        'best_params': study.best_params,
        'best_score': study.best_value,
        'model_name': model_name,
        'metric': metric,
        'study': study,
    }


def tune_and_select_best(
    X: pd.DataFrame,
    y: pd.Series,
    target_name: str = 'target',
    model_names: Sequence[str] = ('xgboost', 'lightgbm'),
    n_trials: int = 50,
    n_splits: int = 5,
    random_state: int = 42,
    metric: str = 'MAE',
    show_progress_bar: bool = True,
) -> Dict[str, Any]:
    """Tune every model in *model_names*, select the best one.

    Returns
    -------
    dict with ``best_model`` (fitted sklearn estimator), ``best_params``,
    ``best_model_name``, ``best_score``, ``all_results``, and CV ``metrics``.
    """
    all_results: Dict[str, Dict] = {}

    for mname in model_names:
        print(f'\n{"="*60}')
        print(f'  Tuning {mname.upper()} for target: {target_name}')
        print(f'  Trials: {n_trials} | CV folds: {n_splits} | Metric: {metric}')
        print(f'{"="*60}')

        result = tune_model(
            X, y,
            model_name=mname,
            n_trials=n_trials,
            n_splits=n_splits,
            random_state=random_state,
            metric=metric,
            show_progress_bar=show_progress_bar,
        )
        all_results[mname] = result
        print(f'  ✓ Best {metric}: {result["best_score"]:.4f}')

    # Pick winner ----------------------------------------------------------
    best_name = min(all_results, key=lambda k: all_results[k]['best_score'])
    best = all_results[best_name]

    # Re‑train final model on all data with best params --------------------
    final_model = _get_estimator(
        model_name=best_name,
        random_state=random_state,
        **best['best_params'],
    )
    final_model.fit(X, y)

    # Full‑data train metrics for reference --------------------------------
    train_preds = final_model.predict(X)
    train_metrics = calculate_metrics(y, train_preds)

    print(f'\n{"="*60}')
    print(f'  ★ WINNER for {target_name}: {best_name.upper()}')
    print(f'    CV {metric}: {best["best_score"]:.4f}')
    print(f'    Train MAE : {train_metrics["MAE"]:.4f}')
    print(f'    Train RMSE: {train_metrics["RMSE"]:.4f}')
    print(f'    Train R²  : {train_metrics["R2"]:.4f}')
    print(f'{"="*60}\n')

    return {
        'best_model': final_model,
        'best_model_name': best_name,
        'best_params': best['best_params'],
        'best_score': best['best_score'],
        'metric': metric,
        'train_metrics': train_metrics,
        'all_results': {
            k: {kk: vv for kk, vv in v.items() if kk != 'study'}
            for k, v in all_results.items()
        },
    }


def run_full_tuning(
    train_features: pd.DataFrame,
    feature_cols: List[str],
    target_cols: Sequence[str] = ('Revenue', 'COGS'),
    model_names: Sequence[str] = ('xgboost', 'lightgbm'),
    n_trials: int = 50,
    n_splits: int = 5,
    random_state: int = 42,
    metric: str = 'MAE',
    out_dir: str = 'output/tuning',
    show_progress_bar: bool = True,
) -> Dict[str, Any]:
    """End‑to‑end tuning pipeline: tune → select → save for each target.

    Parameters
    ----------
    train_features : DataFrame that includes both features and targets.
    feature_cols   : list of column names to use as model input.
    target_cols    : target column(s) to tune for.
    out_dir        : where to save artifacts.

    Returns
    -------
    dict mapping each target to its ``tune_and_select_best`` result.
    """
    X = train_features[feature_cols]
    results: Dict[str, Any] = {}

    for target in target_cols:
        y = train_features[target]
        res = tune_and_select_best(
            X, y,
            target_name=target,
            model_names=model_names,
            n_trials=n_trials,
            n_splits=n_splits,
            random_state=random_state,
            metric=metric,
            show_progress_bar=show_progress_bar,
        )
        results[target] = res

    # Persist everything ---------------------------------------------------
    save_tuning_artifacts(results, out_dir=out_dir)

    return results


# ---------------------------------------------------------------------------
# Artifact persistence
# ---------------------------------------------------------------------------

def save_tuning_artifacts(
    results: Dict[str, Dict[str, Any]],
    out_dir: str = 'output/tuning',
) -> Dict[str, str]:
    """Save models, params JSON, and a human‑readable report.

    Returns a dict of file paths written.
    """
    os.makedirs(out_dir, exist_ok=True)
    paths: Dict[str, str] = {}

    # --- best_params.json -------------------------------------------------
    params_payload: Dict[str, Any] = {
        'timestamp': datetime.now().isoformat(),
    }
    for target, res in results.items():
        params_payload[target] = {
            'model': res['best_model_name'],
            'params': res['best_params'],
            f'cv_{res["metric"]}': res['best_score'],
            'train_metrics': res.get('train_metrics', {}),
            'all_candidates': {
                k: {
                    'best_score': v['best_score'],
                    'best_params': v['best_params'],
                }
                for k, v in res.get('all_results', {}).items()
            },
        }

    params_path = os.path.join(out_dir, 'best_params.json')
    with open(params_path, 'w') as f:
        json.dump(params_payload, f, indent=2, default=str)
    paths['params'] = params_path

    # --- model .joblib files ----------------------------------------------
    for target, res in results.items():
        model_path = os.path.join(out_dir, f'best_model_{target.lower()}.joblib')
        joblib_dump(res['best_model'], model_path)
        paths[f'model_{target}'] = model_path

    # --- tuning_report.txt ------------------------------------------------
    lines: List[str] = [
        '=' * 70,
        '  HYPERPARAMETER TUNING REPORT',
        f'  Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        '=' * 70,
        '',
    ]
    for target, res in results.items():
        lines.append(f'Target: {target}')
        lines.append(f'  Winner     : {res["best_model_name"].upper()}')
        lines.append(f'  CV {res["metric"]:>4s}    : {res["best_score"]:.4f}')

        tm = res.get('train_metrics', {})
        if tm:
            lines.append(f'  Train MAE  : {tm.get("MAE", 0):.4f}')
            lines.append(f'  Train RMSE : {tm.get("RMSE", 0):.4f}')
            lines.append(f'  Train R²   : {tm.get("R2", 0):.4f}')

        lines.append(f'  Best params:')
        for k, v in res['best_params'].items():
            if isinstance(v, float):
                lines.append(f'    {k:25s}: {v:.6g}')
            else:
                lines.append(f'    {k:25s}: {v}')

        # Runner‑up scores
        all_res = res.get('all_results', {})
        if len(all_res) > 1:
            lines.append(f'  All candidates:')
            for mname, mres in all_res.items():
                tag = ' ★' if mname == res['best_model_name'] else ''
                lines.append(
                    f'    {mname:15s} → {res["metric"]} = {mres["best_score"]:.4f}{tag}'
                )
        lines.append('')

    report_path = os.path.join(out_dir, 'tuning_report.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))
    paths['report'] = report_path

    print(f'\n📁 Artifacts saved to: {out_dir}/')
    for label, p in paths.items():
        print(f'   {label:20s} → {p}')

    return paths
