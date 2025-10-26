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
    create_default_window_features_generator,
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
    random_search_forecaster,
    backtesting_forecaster,
)

from sklearn.preprocessing import FunctionTransformer

# ===== 1) Cargar datos base =====
df = DataManager().load_data("data/stage/SO2/processed/processed_CEN-TRAF.csv")
df = df.sort_index()
# df = df.asfreq("H")  # si necesitas forzar frecuencia horaria fija

# ===== 2) Configuración columns =====
TARGET_COL = "target"

# ===== 3) Cargar selección desde JSON =====


# ===== 4) Window features =====
window_features = create_default_window_features_generator().get_window_features()

# ===== 5) Split train / val / test =====
y_train, exog_train, y_val, exog_val, y_test, exog_test, y_trainval, exog_trainval = (
    split_data_by_dates(
        df=df,
        target_col=TARGET_COL,
        exog_cols=df.drop(columns="target").columns.tolist(),
        val_months=2,
        test_months=2,
    )
)


# ===== 💡 Cargar archivo de pesos y crear weight_func =====
weights_path = Path("data/stage/SO2/marks/weights_CEN-TRAF.csv")
weights = pd.read_csv(weights_path, parse_dates=["datetime"]).set_index("datetime")[
    "weight"
]


def weight_func(index: pd.DatetimeIndex) -> np.ndarray:
    """
    Devuelve un vector de pesos alineado al índice temporal del fold actual.
    Los huecos o zonas imputadas (weight=0) no influyen en el entrenamiento.
    """
    return weights.reindex(index).fillna(1.0).to_numpy()


# ===== 6) Forecaster recursivo =====
forecaster = ForecasterRecursive(
    regressor=LGBMRegressor(random_state=123, verbose=-1),
    lags=72,
    window_features=window_features,
    weight_func=weight_func,
    transformer_y=FunctionTransformer(
        func=np.log1p, 
        inverse_func=np.expm1
    ),
)

# ===== 7) Backtesting + Random Search (tuning robusto) =====

H = 1  # horizonte 1 paso
cv = TimeSeriesFold(
    steps=H,
    initial_train_size=len(y_train),
    refit=False,
)

# Param grid más amplio
param_distributions = {
    "n_estimators": [200, 400, 800],
    "max_depth": [5, 10, 15, 20],
    "learning_rate": [0.1, 0.05, 0.01],
    "num_leaves": [31, 63, 127],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "min_child_samples": [10, 20, 50],
}

lags_grid = [np.arange(1, 73)]

results = random_search_forecaster(
    forecaster=forecaster,
    y=y_trainval,
    exog=exog_trainval,
    param_distributions=param_distributions,
    lags_grid=lags_grid,
    cv=cv,
    metric=wmape,  # <-- métrica robusta
    n_iter=20,  # probar 20 combinaciones aleatorias
    random_state=123,
    return_best=True,
    n_jobs="auto",
    verbose=False,
    show_progress=True,
)
print("Resultados Random Search (top 5):")
print(results.head())

# ===== 7.1) Validación post-hoc (train+val): métricas robustas =====
metric_vals_tv, preds_tv = backtesting_forecaster(
    forecaster=forecaster,
    y=y_trainval,
    exog=exog_trainval,
    cv=cv,
    metric=wmape,
    return_predictors=False,
    n_jobs="auto",
    verbose=False,
    show_progress=False,
)


mape_overall_tv = wmape(y_trainval.loc[preds_tv.index], preds_tv["pred"])
rmse_tv = rmse(y_trainval.loc[preds_tv.index], preds_tv["pred"])

print("\nValidación (train+val) [unscaled]:")
print(f"WMAPE %: {(100 * mape_overall_tv):.2f}")
print(f"RMSE: {rmse_tv:.4f}")

# ===== 8) Fit final (train+val) y evaluación en test =====
forecaster.fit(y=y_trainval, exog=exog_trainval)
y_pred = forecaster.predict(steps=len(y_test), exog=exog_test)

test_rmse = rmse(y_test, y_pred)
test_wmape = wmape(y_test, y_pred)

print("\nTest [unscaled]:")
print(f"RMSE: {test_rmse:.4f}")
print(f"WMAPE %: {100 * test_wmape:.2f}")
