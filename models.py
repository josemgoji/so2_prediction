# --- imports
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.recursos.data_manager import DataManager
from src.recursos.regressors import (
    create_lgbm_regressor,
    create_xgb_regressor,
    create_rf_regressor,
    create_lasso_regressor,
)
from src.recursos.windows_features import (
    FourierWindowFeatures,
    CustomRollingFeatures,
)
from src.recursos.scorers import (
    mape_overall_metric_dynamic,
    mape_safe,
    wmape,
    rmse,
    stepwise_mape_from_backtesting,
    stepwise_mape_on_test,
)
from src.utils.data_splitter import split_data_by_dates
from src.utils.plot_utils import create_prediction_plots

from skforecast.recursive import ForecasterRecursive
from skforecast.model_selection import (
    TimeSeriesFold,
    grid_search_forecaster,
    random_search_forecaster,
    backtesting_forecaster,
)

from sklearn.preprocessing import FunctionTransformer
from src.constants.parsed_fields import (
    FEATURE_SELECTION_CONFIG,
    REGRESSORS_CONFIG,
    MODEL_RESULTS_CONFIG,
)


def clean_params_for_json(params_dict):
    """Convierte parámetros a tipos serializables en JSON"""
    cleaned = {}
    for key, value in params_dict.items():
        if isinstance(value, (np.integer, np.floating)):
            cleaned[key] = value.item()
        elif isinstance(value, np.ndarray):
            cleaned[key] = value.tolist()
        elif isinstance(value, (list, tuple)):
            cleaned[key] = [
                v.item() if isinstance(v, (np.integer, np.floating)) else v
                for v in value
            ]
        else:
            cleaned[key] = value
    return cleaned


# ===== 💡 Configuración de estación =====
STATION = "CEN-TRAF"  # Opciones: "CEN-TRAF", "GIR-EPM", "ITA-CJUS", "MED-FISC"

print(f"🚀 Ejecutando modelos para la estación: {STATION}")

# ===== 💡 Configuración de regresores =====
# Los regresores se cargan desde parsed_fields.py
USE_EXOG = True  # True para modelo con exógenas, False para sin exógenas

# ===== 1) Cargar datos base =====
df = DataManager().load_data(f"data/stage/SO2/processed/processed_{STATION}.csv")
df = df.sort_index()

# ===== 2) Configuración columns =====
TARGET_COL = "target"

# ===== 3) Cargar selección desde JSON =====
feat_sel_path = Path(
    f"data/stage/SO2/selected/lasso/con_exog/selected_cols_{STATION}_lasso_rf.json"
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

# ===== 6) Escalado robusto y Forecaster recursivo =====
# Escalado log1p para estabilizar variancia (evita explosión de MAPE si hay outliers)
TARGET_COL = "target"

y_train, exog_train, y_val, exog_val, y_test, exog_test, y_trainval, exog_trainval = (
    split_data_by_dates(
        df=df,
        target_col=TARGET_COL,
        exog_cols=selected_exog,
        val_months=2,
        test_months=2,
    )
)

# ===== 💡 Control de pesos para gaps =====
USE_WEIGHTS = True  # Cambiar a False para desactivar los pesos de gaps

# ===== 💡 Cargar archivo de pesos y crear weight_func =====
if USE_WEIGHTS:
    weights_path = Path(f"data/stage/SO2/marks/weights_{STATION}.csv")
    weights = pd.read_csv(weights_path, parse_dates=["datetime"]).set_index("datetime")[
        "weight"
    ]

    def weight_func(index: pd.DatetimeIndex) -> np.ndarray:
        """
        Devuelve un vector de pesos alineado al índice temporal del fold actual.
        Los huecos o zonas imputadas (weight=0) no influyen en el entrenamiento.
        """
        return weights.reindex(index).fillna(1.0).to_numpy()
else:
    weight_func = None

# ===== 6) Configuración común para todos los regresores =====
H = 6  # horizonte 1 paso
cv = TimeSeriesFold(
    steps=H,
    initial_train_size=len(y_train),
    refit=False,
)

# lags_grid = [selected_lags]

# ===== 7) Iterar sobre diferentes regresores =====
all_results = []

# Crear directorio de resultados si no existe
results_dir = (
    Path(MODEL_RESULTS_CONFIG["analytics_dir"]) / MODEL_RESULTS_CONFIG["results_subdir"]
)
results_dir.mkdir(parents=True, exist_ok=True)

# Determinar subdirectorio basado en configuración
exog_status = "con_exog" if USE_EXOG else "sin_exog"
station_results_dir = results_dir / STATION / exog_status
station_results_dir.mkdir(parents=True, exist_ok=True)

for regressor_config in REGRESSORS_CONFIG:
    regressor_name = regressor_config["name"]
    regressor_func_name = regressor_config["regressor_func"]
    param_distributions = regressor_config["params"]

    # Mapear nombre de función a función real
    regressor_func_map = {
        "create_lgbm_regressor": create_lgbm_regressor,
        "create_xgb_regressor": create_xgb_regressor,
        "create_rf_regressor": create_rf_regressor,
        "create_lasso_regressor": create_lasso_regressor,
    }

    regressor_func = regressor_func_map[regressor_func_name]

    print(f"\n{'=' * 60}")
    print(f"🚀 Entrenando modelo: {regressor_name}")
    print(f"{'=' * 60}")

    # Crear regressor base con parámetros por defecto
    base_regressor = regressor_func(
        random_state=FEATURE_SELECTION_CONFIG["random_state"]
    )

    # ===== Forecaster recursivo =====
    forecaster_params = {
        "regressor": base_regressor,
        "lags": selected_lags,
        "window_features": window_features,
        "transformer_y": FunctionTransformer(func=np.log1p, inverse_func=np.expm1),
    }

    # Solo agregar weight_func si está habilitado
    if USE_WEIGHTS:
        forecaster_params["weight_func"] = weight_func

    forecaster = ForecasterRecursive(**forecaster_params)

    # ===== Random Search =====
    try:
        results = random_search_forecaster(
            forecaster=forecaster,
            y=y_trainval,
            exog=exog_trainval,
            param_distributions=param_distributions,
            # lags_grid=lags_grid,
            cv=cv,
            metric=wmape,
            n_iter=10,  # Aumentado para mejor tuning
            random_state=FEATURE_SELECTION_CONFIG["random_state"],
            return_best=True,
            n_jobs=-1,
            verbose=False,
            show_progress=True,
        )

        print(f"\n📊 Resultados Random Search para {regressor_name}:")
        print(results.head())

        # Extraer mejor parámetros correctamente
        if len(results) > 0:
            best_params = results.iloc[0].to_dict()
            # Remover columnas que no son parámetros del modelo
            params_to_remove = ["metric", "metric_std", "metric_mean"]
            best_params = {
                k: v for k, v in best_params.items() if k not in params_to_remove
            }

            # Manejar específicamente el parámetro 'lags' que puede ser problemático
            if "lags" in best_params:
                lags_value = best_params["lags"]
                if hasattr(lags_value, "tolist"):
                    best_params["lags"] = lags_value.tolist()
                elif isinstance(lags_value, np.ndarray):
                    best_params["lags"] = lags_value.tolist()
                else:
                    best_params["lags"] = str(
                        lags_value
                    )  # Convertir a string si es necesario
        else:
            best_params = {}

        # ===== Validación post-hoc (train+val) =====
        metric_vals_tv, preds_tv = backtesting_forecaster(
            forecaster=forecaster,
            y=y_trainval,
            exog=exog_trainval,
            cv=cv,
            metric=wmape,
            return_predictors=True,
            n_jobs=-1,
            verbose=False,
            show_progress=False,
        )

        mape_overall_tv = wmape(y_trainval.loc[preds_tv.index], preds_tv["pred"])
        rmse_tv = rmse(y_trainval.loc[preds_tv.index], preds_tv["pred"])

        # Calcular stepwise MAPE para validación
        stepwise_mape_val = stepwise_mape_from_backtesting(
            preds_tv, y_trainval.loc[preds_tv.index]
        )

        print(f"\n📈 Validación (train+val) - {regressor_name}:")
        print(f"WMAPE %: {(100 * mape_overall_tv):.2f}")
        print(f"RMSE: {rmse_tv:.4f}")
        print(f"Stepwise MAPE: {stepwise_mape_val.to_dict()}")

        # ===== Evaluación en test =====
        forecaster.fit(y=y_trainval, exog=exog_trainval)
        y_pred = forecaster.predict(steps=len(y_test), exog=exog_test)

        test_rmse = rmse(y_test, y_pred)
        test_wmape = wmape(y_test, y_pred)

        # Calcular stepwise MAPE para test
        stepwise_mape_test = stepwise_mape_on_test(y_test, y_pred, H=H)

        print(f"\n🎯 Test - {regressor_name}:")
        print(f"RMSE: {test_rmse:.4f}")
        print(f"WMAPE %: {100 * test_wmape:.2f}")
        print(f"Stepwise MAPE: {stepwise_mape_test.to_dict()}")

        # ===== Crear gráficos de predicciones =====
        try:
            plot_files = create_prediction_plots(
                y_val=y_val,
                preds_val=preds_tv.loc[y_val.index]
                if len(preds_tv) > 0
                else pd.DataFrame(),
                y_test=y_test,
                y_pred_test=y_pred,
                model_name=regressor_name,
                station=STATION,
                save_dir=station_results_dir,
            )
            print(f"📊 Gráficos creados exitosamente para {regressor_name}")
        except Exception as e:
            print(f"⚠️ Error creando gráficos para {regressor_name}: {str(e)}")
            plot_files = {}

        # ===== Guardar modelo entrenado =====
        timestamp_str = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        model_file = station_results_dir / f"{regressor_name}_model_{timestamp_str}.pkl"

        # Guardar el forecaster entrenado
        with open(model_file, "wb") as f:
            pickle.dump(forecaster, f)

        print(f"💾 Modelo entrenado guardado en: {model_file}")

        # Preparar datos para guardar
        result_data = {
            "station": STATION,
            "model_type": regressor_name,
            "use_exog": USE_EXOG,
            "use_weights": USE_WEIGHTS,
            "validation_metrics": {
                "wmape": float(mape_overall_tv),
                "rmse": float(rmse_tv),
                "stepwise_mape": stepwise_mape_val.to_dict(),
            },
            "test_metrics": {
                "wmape": float(test_wmape),
                "rmse": float(test_rmse),
                "stepwise_mape": stepwise_mape_test.to_dict(),
            },
            "best_params": clean_params_for_json(best_params),
            "model_file": str(model_file),
            "plot_files": plot_files,
            "timestamp": pd.Timestamp.now().isoformat(),
        }

        # Guardar resultados individuales
        timestamp_str = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        result_file = station_results_dir / f"{regressor_name}_{timestamp_str}.json"

        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)

        print(f"💾 Resultados guardados en: {result_file}")

        # Guardar resultados para comparación
        all_results.append(
            {
                "regressor": regressor_name,
                "val_wmape": mape_overall_tv,
                "val_rmse": rmse_tv,
                "val_stepwise_mape": stepwise_mape_val.to_dict(),
                "test_wmape": test_wmape,
                "test_rmse": test_rmse,
                "test_stepwise_mape": stepwise_mape_test.to_dict(),
                "best_params": clean_params_for_json(best_params),
                "model_file": str(model_file),
                "plot_files": plot_files,
            }
        )

    except Exception as e:
        print(f"❌ Error entrenando {regressor_name}: {str(e)}")
        all_results.append(
            {
                "regressor": regressor_name,
                "val_wmape": float("inf"),
                "val_rmse": float("inf"),
                "val_stepwise_mape": {},
                "test_wmape": float("inf"),
                "test_rmse": float("inf"),
                "test_stepwise_mape": {},
                "best_params": {},
                "model_file": None,
                "plot_files": {},
                "error": str(e),
            }
        )

# ===== 8) Resumen de resultados =====
print(f"\n{'=' * 80}")
print(f"📋 RESUMEN DE RESULTADOS PARA ESTACIÓN: {STATION}")
print(
    f"🔧 Configuración: {'Con exógenas' if USE_EXOG else 'Sin exógenas'}, {'Con pesos' if USE_WEIGHTS else 'Sin pesos'}"
)
print(f"{'=' * 80}")

results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values("test_wmape")

print("\n🏆 RANKING POR TEST WMAPE:")
for i, (_, row) in enumerate(results_df.iterrows(), 1):
    if row["test_wmape"] != float("inf"):
        print(
            f"{i}. {row['regressor']}: WMAPE = {100 * row['test_wmape']:.2f}%, RMSE = {row['test_rmse']:.4f}"
        )
        # Mostrar stepwise MAPE para el mejor modelo
        if i == 1:
            print(f"   Stepwise MAPE Test: {row['test_stepwise_mape']}")
    else:
        print(f"{i}. {row['regressor']}: ERROR - {row.get('error', 'Unknown error')}")

print(f"\n🥇 MEJOR MODELO: {results_df.iloc[0]['regressor']}")
print(f"Test WMAPE: {100 * results_df.iloc[0]['test_wmape']:.2f}%")
print(f"Test RMSE: {results_df.iloc[0]['test_rmse']:.4f}")
print(f"Test Stepwise MAPE: {results_df.iloc[0]['test_stepwise_mape']}")
print(f"Modelo guardado en: {results_df.iloc[0]['model_file']}")
if results_df.iloc[0]["plot_files"]:
    print("📊 Gráficos guardados:")
    for plot_type, plot_path in results_df.iloc[0]["plot_files"].items():
        print(f"   {plot_type}: {plot_path}")

# ===== 9) Guardar resumen completo =====
summary_data = {
    "station": STATION,
    "configuration": {
        "use_exog": USE_EXOG,
        "use_weights": USE_WEIGHTS,
        "timestamp": pd.Timestamp.now().isoformat(),
    },
    "results_summary": results_df.to_dict("records"),
    "best_model": {
        "name": results_df.iloc[0]["regressor"],
        "test_wmape": float(results_df.iloc[0]["test_wmape"]),
        "test_rmse": float(results_df.iloc[0]["test_rmse"]),
        "test_stepwise_mape": results_df.iloc[0]["test_stepwise_mape"],
        "val_stepwise_mape": results_df.iloc[0]["val_stepwise_mape"],
        "best_params": clean_params_for_json(results_df.iloc[0]["best_params"]),
        "model_file": results_df.iloc[0]["model_file"],
        "plot_files": results_df.iloc[0]["plot_files"],
    },
}

# Guardar resumen completo
timestamp_str = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
summary_file = station_results_dir / f"summary_{timestamp_str}.json"

with open(summary_file, "w", encoding="utf-8") as f:
    json.dump(summary_data, f, indent=2, ensure_ascii=False)

print(f"\n💾 Resumen completo guardado en: {summary_file}")

# Guardar también como CSV para fácil análisis
csv_file = station_results_dir / f"results_comparison_{timestamp_str}.csv"
results_df.to_csv(csv_file, index=False)
print(f"📊 Comparación en CSV guardada en: {csv_file}")

print(f"\n✅ Proceso completado para estación {STATION}")
