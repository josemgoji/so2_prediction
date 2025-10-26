# --- imports
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.recursos.data_manager import DataManager
from src.recursos.regressors import LGBMRegressor
from src.recursos.windows_features import (
    FourierWindowFeatures,
    CustomRollingFeatures,
)
from src.recursos.scorers import (
    mape_overall_metric_dynamic,  # tuning robusto
    mape_safe,  # clásico (para referencia)
    wmape,
    rmse,
    stepwise_mape_from_backtesting,
    stepwise_mape_on_test,
)
from src.utils.data_splitter import split_data_by_dates

from skforecast.recursive import ForecasterRecursive
from skforecast.model_selection import (
    TimeSeriesFold,
    grid_search_forecaster,
    backtesting_forecaster,
)

# ===== 1) Cargar datos base =====
df = DataManager().load_data("data/stage/SO2/processed/processed_CEN-TRAF.csv")
df = df.sort_index()
# df = df.asfreq("H")  # si necesitas forzar frecuencia horaria fija

# ===== 2) Configuración columns =====
TARGET_COL = "target"

# ===== 3) Cargar selección desde JSON =====
feat_sel_path = Path(
    "data/stage/SO2/selected/lasso/con_exog/selected_cols_CEN-TRAF_lasso_rf.json"
)
with open(feat_sel_path, "r", encoding="utf-8") as f:
    sel = json.load(f)

selected_lags: list[int] = sel["selected_lags"]
selected_window_features: list[str] = sel["selected_window_features"]
selected_exog: list[str] = sel.get("selected_exog", [])

missing = [c for c in [TARGET_COL] + selected_exog if c not in df.columns]
if missing:
    raise ValueError(f"Faltan columnas en df: {missing}")

# ===== 4) Window features =====
window_features = [
    FourierWindowFeatures(period=24, K=3),
    CustomRollingFeatures(stats=["mean"], window_sizes=[3, 6, 24, 48, 72]),
    CustomRollingFeatures(stats=["min"], window_sizes=[6, 24]),
    CustomRollingFeatures(stats=["max"], window_sizes=[6, 12, 24]),
]

# ===== 5) Split train / val / test =====
y_train, exog_train, y_val, exog_val, y_test, exog_test, y_trainval, exog_trainval = (
    split_data_by_dates(
        df=df,
        target_col=TARGET_COL,
        exog_cols=selected_exog,
        val_months=2,
        test_months=2,
    )
)

# ===== 6) Forecaster recursivo =====
forecaster = ForecasterRecursive(
    regressor=LGBMRegressor(random_state=123, verbose=-1),
    lags=selected_lags,
    window_features=window_features,
)

# ===== 7) Backtesting + Grid Search (tuning robusto) =====
H = 1  # horizonte
cv = TimeSeriesFold(
    steps=H,
    initial_train_size=len(y_train),
    refit=False,
)

param_grid = {
    "n_estimators": [5, 10],
    "max_depth": [5, 10],
}
lags_grid = [selected_lags]

results = grid_search_forecaster(
    forecaster=forecaster,
    y=y_trainval,
    exog=exog_trainval,
    param_grid=param_grid,
    lags_grid=lags_grid,
    cv=cv,
    metric=mape_overall_metric_dynamic,  # <-- métrica robusta para tuning
    return_best=True,
    n_jobs="auto",
    verbose=False,
    show_progress=True,
)
print("Resultados grid (top 5):")
print(results.head())

# ===== 7.1) Validación post-hoc (train+val): curvas MAPE(h) y agregados =====
metric_vals_tv, preds_tv = backtesting_forecaster(
    forecaster=forecaster,
    y=y_trainval,
    exog=exog_trainval,
    cv=cv,
    metric=mape_overall_metric_dynamic,  # cualquier callable válido
    return_predictors=False,
    n_jobs="auto",
    verbose=False,
    show_progress=False,
)

mape_h_tv = stepwise_mape_from_backtesting(preds_tv, y_trainval)  # fracción
mape_overall_tv = mape_overall_metric_dynamic(
    y_trainval.loc[preds_tv.index], preds_tv["pred"]
)
wmape_tv = wmape(y_trainval.loc[preds_tv.index], preds_tv["pred"])

print("\nValidación (train+val):")
print("MAPE(h) % (dinámico):\n", (100 * mape_h_tv).round(2))
print(f"MAPE overall (dinámico) %: {(100 * mape_overall_tv):.2f}")
print(f"WMAPE %: {(100 * wmape_tv):.2f}")

# ===== 8) Fit final y evaluación en test =====
forecaster.fit(y=y_trainval, exog=exog_trainval)
y_pred = forecaster.predict(steps=len(y_test), exog=exog_test)

test_rmse = rmse(y_test, y_pred)
test_mape_overall_dyn = mape_overall_metric_dynamic(y_test, y_pred)
test_wmape = wmape(y_test, y_pred)

print("\nTest:")
print(f"RMSE: {test_rmse:.4f}")
print(f"MAPE overall (dinámico) %: {100 * test_mape_overall_dyn:.2f}")
print(f"WMAPE %: {100 * test_wmape:.2f}")

# (Opcional) curva MAPE(h) en test con H=72
mape_h_test = stepwise_mape_on_test(y_test, y_pred, H=H)
print("MAPE(h) en test % (dinámico):\n", (100 * mape_h_test).round(2))
