import numpy as np
from sklearn.metrics import make_scorer, mean_squared_error


def mape_safe(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return np.mean(np.abs((y_true - y_pred) / denom))


def wmape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + eps)


def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)


# Scorers (menor = mejor)
mape_scorer = make_scorer(mape_safe, greater_is_better=False)
wmape_scorer = make_scorer(wmape, greater_is_better=False)
rmse_scorer = make_scorer(rmse, greater_is_better=False)
