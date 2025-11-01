import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, make_scorer


# -----------------------
# Helpers
# -----------------------
def _asfloat1d(a):
    return np.asarray(a, dtype=float).ravel()


# -----------------------
# Métricas base
# -----------------------
def mape_safe(y_true, y_pred, eps=1e-8):
    y_true = _asfloat1d(y_true)
    y_pred = _asfloat1d(y_pred)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)))


def mape_safe_dynamic(y_true, y_pred, rel_floor=0.05, abs_floor=1e-8):
    """
    MAPE con piso dinámico: max(|y|, rel_floor * mediana(|y|), abs_floor).
    Mitiga explosiones cuando y_true ~ 0.
    """
    y_true = _asfloat1d(y_true)
    y_pred = _asfloat1d(y_pred)
    scale = np.median(np.abs(y_true)) if y_true.size else 0.0
    floor = max(abs_floor, rel_floor * scale)
    denom = np.maximum(np.abs(y_true), floor)
    return float(np.mean(np.abs((y_true - y_pred) / denom)))


def wmape(y_true, y_pred, eps=1e-8):
    y_true = _asfloat1d(y_true)
    y_pred = _asfloat1d(y_pred)
    return float(np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + eps))


def rmse(y_true, y_pred):
    y_true = _asfloat1d(y_true)
    y_pred = _asfloat1d(y_pred)
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# -----------------------
# Métricas para tuning
# -----------------------
def mape_overall_metric(y_true, y_pred):
    """MAPE promedio (fracción), versión clásica."""
    return mape_safe(y_true, y_pred)


def mape_overall_metric_dynamic(y_true, y_pred):
    """MAPE promedio (fracción), robusto con piso dinámico."""
    return mape_safe_dynamic(y_true, y_pred)


def mape_step_h_factory(h: int, rel_floor=0.05, abs_floor=1e-8, penalty=1e6):
    """
    Métrica para optimizar un horizonte específico h (1-indexed) durante el search.
    Usa piso dinámico; si el fold es más corto, devuelve una penalización grande.
    """
    idx = h - 1

    def _metric(y_true, y_pred):
        y_true = _asfloat1d(y_true)
        y_pred = _asfloat1d(y_pred)
        if idx >= len(y_true):
            return float(penalty)
        scale = np.median(np.abs(y_true)) if y_true.size else 0.0
        floor = max(abs_floor, rel_floor * scale)
        denom = max(abs(y_true[idx]), floor)
        return float(abs((y_true[idx] - y_pred[idx]) / denom))

    return _metric


# -----------------------
# Step-wise (post-hoc) para validación y test
# -----------------------
def stepwise_mape_from_backtesting(
    preds_df: pd.DataFrame, y_series: pd.Series, rel_floor=0.05, abs_floor=1e-8
) -> pd.Series:
    """
    Calcula MAPE(h) agregando sobre folds de backtesting (fracción).
    'preds_df' viene de backtesting_forecaster con columnas ['pred','fold'].
    """
    preds = preds_df.copy()
    preds["h"] = preds.groupby("fold").cumcount() + 1
    y_true = y_series.loc[preds.index].values
    y_pred = preds["pred"].values
    df_eval = pd.DataFrame({"h": preds["h"].values, "y_true": y_true, "y_pred": y_pred})

    def _mape(g):
        yt = g["y_true"].values
        yp = g["y_pred"].values
        scale = np.median(np.abs(yt)) if len(yt) else 0.0
        floor = max(abs_floor, rel_floor * scale)
        denom = np.maximum(np.abs(yt), floor)
        return float(np.mean(np.abs((yt - yp) / denom)))

    try:
        return df_eval.groupby("h").apply(_mape, include_groups=False)
    except TypeError:
        return df_eval.groupby("h", group_keys=False).apply(_mape)


def stepwise_mape_on_test(
    y_true: pd.Series, y_pred: pd.Series, H: int, rel_floor=0.05, abs_floor=1e-8
) -> pd.Series:
    """
    MAPE(h) en test (fracción), agrupando por posición dentro de bloques de tamaño H (1..H).
    Soporta último bloque parcial.
    """
    n = len(y_pred)
    h = (np.arange(n) % H) + 1
    df = pd.DataFrame(
        {"h": h, "y_true": _asfloat1d(y_true), "y_pred": _asfloat1d(y_pred)}
    )

    def _mape(g):
        yt = g["y_true"].values
        yp = g["y_pred"].values
        scale = np.median(np.abs(yt)) if len(yt) else 0.0
        floor = max(abs_floor, rel_floor * scale)
        denom = np.maximum(np.abs(yt), floor)
        return float(np.mean(np.abs((yt - yp) / denom)))

    try:
        return df.groupby("h").apply(_mape, include_groups=False)
    except TypeError:
        return df.groupby("h", group_keys=False).apply(_mape)


# Scorers (menor = mejor)
mape_scorer = make_scorer(mape_safe, greater_is_better=False)
wmape_scorer = make_scorer(wmape, greater_is_better=False)
rmse_scorer = make_scorer(rmse, greater_is_better=False)
mape_overall_metric_scorer = make_scorer(mape_overall_metric, greater_is_better=False)
mape_step_h_scorer = make_scorer(mape_step_h_factory(h=1), greater_is_better=False)