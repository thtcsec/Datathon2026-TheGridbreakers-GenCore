import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_metrics(y_true, y_pred):
    """
    Calculate evaluation metrics: MAE, RMSE, R2.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

def print_metrics(y_true, y_pred, model_name="Model"):
    """
    Calculate and print evaluation metrics.
    """
    metrics = calculate_metrics(y_true, y_pred)
    print(f"--- {model_name} Performance ---")
    print(f"MAE:  {metrics['MAE']:,.2f}")
    print(f"RMSE: {metrics['RMSE']:,.2f}")
    print(f"R2:   {metrics['R2']:.4f}")
    return metrics
