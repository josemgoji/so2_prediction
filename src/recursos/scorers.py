# --- imports básicos ---
import numpy as np
from sklearn.metrics import make_scorer, mean_squared_error


# ---- métricas custom (seguras para series) ----
def mape_safe(y_true, y_pred, eps=1e-8):
    """
    Mean Absolute Percentage Error seguro para series temporales.
    Maneja valores cercanos a cero usando un epsilon.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return np.mean(np.abs((y_true - y_pred) / denom))


def rmse(y_true, y_pred):
    """Root Mean Square Error."""
    return mean_squared_error(y_true, y_pred, squared=False)


# ---- Scorers listos para usar directamente ----
wmape_scorer = make_scorer(mape_safe, greater_is_better=False)
mape_scorer = make_scorer(mape_safe, greater_is_better=False)  # mismo que wmape
rmse_scorer = make_scorer(rmse, greater_is_better=False)
