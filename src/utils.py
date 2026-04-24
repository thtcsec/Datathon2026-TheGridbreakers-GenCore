import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _mape(y_true, y_pred, eps=1e-8):
    """Mean Absolute Percentage Error (%).

    Uses a small *eps* floor on the denominator to avoid division-by-zero
    when actual values are exactly 0.
    """
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100)


def _smape(y_true, y_pred, eps=1e-8):
    """Symmetric Mean Absolute Percentage Error (%).

    Bounded in [0, 200], treats over- and under-predictions symmetrically —
    easier to interpret for business stakeholders.
    """
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true) + np.abs(y_pred), eps)
    return float(np.mean(2.0 * np.abs(y_true - y_pred) / denom) * 100)


def calculate_metrics(y_true, y_pred):
    """
    Calculate evaluation metrics: MAE, RMSE, R2, MAPE (%), sMAPE (%).
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = _mape(y_true, y_pred)
    smape = _smape(y_true, y_pred)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape,
        "sMAPE": smape,
    }

def print_metrics(y_true, y_pred, model_name="Model"):
    """
    Calculate and print evaluation metrics.
    """
    metrics = calculate_metrics(y_true, y_pred)
    print(f"--- {model_name} Performance ---")
    print(f"MAE:   {metrics['MAE']:,.2f}")
    print(f"RMSE:  {metrics['RMSE']:,.2f}")
    print(f"R2:    {metrics['R2']:.4f}")
    print(f"MAPE:  {metrics['MAPE']:.2f}%")
    print(f"sMAPE: {metrics['sMAPE']:.2f}%")
    return metrics

