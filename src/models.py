import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit

from src.utils import calculate_metrics


def _get_estimator(model_name='xgboost', random_state=42, **kwargs):
    """Return an estimator instance.

    Any extra ``kwargs`` override the default hyper-parameters, allowing
    the tuning module to inject optimised values while keeping sensible
    defaults for manual / baseline usage.
    """
    name = model_name.lower()

    if name in {'xgboost', 'xgb'}:
        from xgboost import XGBRegressor

        defaults = dict(
            n_estimators=700,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
            n_jobs=-1,
        )
        defaults.update(kwargs)
        return XGBRegressor(**defaults)

    if name in {'lightgbm', 'lgbm'}:
        from lightgbm import LGBMRegressor

        defaults = dict(
            n_estimators=900,
            learning_rate=0.05,
            num_leaves=31,
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
        )
        defaults.update(kwargs)
        return LGBMRegressor(**defaults)

    raise ValueError(f'Unsupported model_name: {model_name}')


def _time_series_cv_metrics(X, y, model_name='xgboost', n_splits=5, random_state=42):
    tss = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics = []

    for train_idx, valid_idx in tss.split(X):
        X_tr, X_va = X.iloc[train_idx], X.iloc[valid_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[valid_idx]

        model = _get_estimator(model_name=model_name, random_state=random_state)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_va)
        fold_metrics.append(calculate_metrics(y_va, preds))

    if not fold_metrics:
        return {'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan}

    return {
        'MAE': float(np.mean([m['MAE'] for m in fold_metrics])),
        'RMSE': float(np.mean([m['RMSE'] for m in fold_metrics])),
        'R2': float(np.mean([m['R2'] for m in fold_metrics])),
    }


def train_and_evaluate(
    train_features,
    feature_cols,
    target_cols=('Revenue', 'COGS'),
    model_name='xgboost',
    n_splits=5,
    random_state=42,
):
    models = {}
    metrics = {}

    X = train_features[feature_cols]

    for target in target_cols:
        y = train_features[target]

        cv_metric = _time_series_cv_metrics(
            X,
            y,
            model_name=model_name,
            n_splits=n_splits,
            random_state=random_state,
        )

        final_model = _get_estimator(model_name=model_name, random_state=random_state)
        final_model.fit(X, y)

        train_pred = final_model.predict(X)
        train_metric = calculate_metrics(y, train_pred)

        models[target] = final_model
        metrics[target] = {
            'cv': cv_metric,
            'train': train_metric,
        }

    return models, metrics


def export_feature_importance(models, feature_cols, out_dir='../output'):
    os.makedirs(out_dir, exist_ok=True)
    paths = {}

    for target, model in models.items():
        if not hasattr(model, 'feature_importances_'):
            continue

        imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=True)

        fig, ax = plt.subplots(figsize=(8, 6))
        imp.tail(20).plot(kind='barh', ax=ax)
        ax.set_title(f'Feature Importance - {target}')
        fig.tight_layout()

        path = os.path.join(out_dir, f'feature_importance_{target.lower()}.png')
        fig.savefig(path, dpi=150)
        plt.close(fig)

        paths[target] = path

    return paths


def generate_submission(
    forecast_features,
    models,
    feature_cols,
    out_path='../output/submission.csv',
    date_col='order_date',
):
    """Generate a submission CSV matching the sample_submission.csv format.

    The output preserves the original date order (no sorting/shuffling)
    and uses plain decimal formatting (no scientific notation).
    """
    if forecast_features is None or forecast_features.empty:
        raise ValueError('forecast_features is empty; cannot generate submission.')

    X_test = forecast_features[feature_cols]
    rev_pred = models['Revenue'].predict(X_test)
    cogs_pred = models['COGS'].predict(X_test)

    submission = pd.DataFrame({
        'Date': pd.to_datetime(forecast_features[date_col]).dt.strftime('%Y-%m-%d'),
        'Revenue': np.maximum(0, rev_pred).round(2),
        'COGS': np.maximum(0, cogs_pred).round(2),
    })

    # Do NOT sort — keep the exact row order from forecast_features.

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    submission.to_csv(out_path, index=False, float_format='%.2f')

    return submission, out_path

