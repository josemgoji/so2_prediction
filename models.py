# --- imports
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import optuna

from src.recursos.data_manager import DataManager
from src.recursos.regressors import (
    create_lgbm_regressor,
    create_xgb_regressor,
    create_rf_regressor,
    create_lasso_regressor,
)
from src.recursos.windows_features import (
    # FourierWindowFeatures,
    CustomRollingFeatures,
)
from src.recursos.scorers import (
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


# =============================================================================
# CONFIGURACIÓN INICIAL
# =============================================================================
STATION = "CEN-TRAF"  # Opciones: "CEN-TRAF", "GIR-EPM", "ITA-CJUS", "MED-FISC"
USE_EXOG = True  # True para modelo con exógenas, False para sin exógenas
USE_WEIGHTS = False  # True para usar pesos de gaps, False para desactivar

# Optuna
N_TRIALS = 20  # <-- ajusta según presupuesto
STUDY_STORAGE = None  # ej: "sqlite:///optuna.db" si quieres persistir estudios
STUDY_SAMPLER = optuna.samplers.TPESampler(
    seed=FEATURE_SELECTION_CONFIG["random_state"]
)

print(f"Ejecutando modelos para la estacion: {STATION}")

# =============================================================================
# CARGA Y PREPARACIÓN DE DATOS
# =============================================================================
df = DataManager().load_data(f"data/stage/SO2/processed/processed_{STATION}.csv")
df = df.sort_index()

TARGET_COL = "target"

# Cargar selección de características desde JSON
feat_sel_path = Path(
    f"data/stage/SO2/selected/lasso/con_exog/selected_cols_{STATION}_lasso_rf.json"
)
with open(feat_sel_path, "r", encoding="utf-8") as f:
    sel = json.load(f)

selected_lags: list[int] = sel["selected_lags"]
selected_window_features: list[str] = sel["selected_window_features"]
selected_exog: list[str] = sel.get("selected_exog", [])

# Verificar columnas requeridas
missing = [c for c in [TARGET_COL] + selected_exog if c not in df.columns]
if missing:
    raise ValueError(f"Faltan columnas en df: {missing}")

# =============================================================================
# CONFIGURACIÓN DE CARACTERÍSTICAS TEMPORALES
# =============================================================================
window_features = [
    # FourierWindowFeatures(period=24, K=3),
    CustomRollingFeatures(stats=["mean"], window_sizes=[3, 6, 24, 48, 72]),
    CustomRollingFeatures(stats=["min"], window_sizes=[6, 24]),
    CustomRollingFeatures(stats=["max"], window_sizes=[6, 12, 24]),
]

# =============================================================================
# DIVISIÓN DE DATOS
# =============================================================================
y_train, exog_train, y_val, exog_val, y_test, exog_test, y_trainval, exog_trainval = (
    split_data_by_dates(
        df=df,
        target_col=TARGET_COL,
        exog_cols=selected_exog,
        val_months=2,
        test_months=2,
    )
)

# =============================================================================
# PESOS (opcional)
# =============================================================================
if USE_WEIGHTS:
    weights_path = Path(f"data/stage/SO2/marks/weights_{STATION}.csv")
    weights = pd.read_csv(weights_path, parse_dates=["datetime"]).set_index("datetime")[
        "weight"
    ]

    def weight_func(index: pd.DatetimeIndex) -> np.ndarray:
        return weights.reindex(index).fillna(1.0).to_numpy()
else:
    weight_func = None

# =============================================================================
# CV Y ESTRUCTURA DE RESULTADOS
# =============================================================================
H = 72  # horizonte de predicción (si 1 paso=1h y quieres 72h, usa H=72)
cv = TimeSeriesFold(
    steps=H,
    initial_train_size=len(y_train),
    refit=False,
)

# Carpetas de resultados
results_dir = (
    Path(MODEL_RESULTS_CONFIG["analytics_dir"]) / MODEL_RESULTS_CONFIG["results_subdir"]
)
results_dir.mkdir(parents=True, exist_ok=True)

exog_status = "con_exog" if USE_EXOG else "sin_exog"
station_results_dir = results_dir / STATION / exog_status / f"H{H}"
station_results_dir.mkdir(parents=True, exist_ok=True)

models_dir = station_results_dir / "models"
plots_dir = station_results_dir / "plots"
results_dir_station = station_results_dir / "results"
summary_dir = station_results_dir / "summary"

for subdir in [models_dir, plots_dir, results_dir_station, summary_dir]:
    subdir.mkdir(parents=True, exist_ok=True)

print(f"\nEstructura de carpetas creada para {STATION}:")
print(f"   {station_results_dir}")
print("   -- models/     (modelos entrenados .pkl)")
print("   -- plots/      (graficos de predicciones)")
print("   -- results/    (resultados individuales .json)")
print("   -- summary/    (resumenes y comparaciones)")
print(f"   Configuracion: {exog_status}, Horizonte: {H}")
print("=" * 60)

# =============================================================================
# ENTRENAMIENTO + EVALUACIÓN (Optuna)
# =============================================================================
all_results = []

for regressor_config in REGRESSORS_CONFIG:
    regressor_name = regressor_config["name"]
    regressor_func_name = regressor_config["regressor_func"]
    param_distributions = regressor_config["params"]  # solo listas → categóricos

    print(f"\n{'=' * 60}")
    print(f"Entrenando modelo (Optuna): {regressor_name}")
    print(f"{'=' * 60}")

    regressor_func_map = {
        "create_lgbm_regressor": create_lgbm_regressor,
        "create_xgb_regressor": create_xgb_regressor,
        "create_rf_regressor": create_rf_regressor,
        "create_lasso_regressor": create_lasso_regressor,
    }
    regressor_func = regressor_func_map[regressor_func_name]

    # -----------------------------
    # Objetivo Optuna (minimiza WMAPE)
    # -----------------------------
    def objective(trial: optuna.Trial) -> float:
        # 1) Sugerir hiperparámetros como categóricos (porque son listas)
        trial_params = {
            name: trial.suggest_categorical(name, values)
            for name, values in param_distributions.items()
        }

        # 2) Construir el regressor con seed fija
        base_regressor = regressor_func(
            random_state=FEATURE_SELECTION_CONFIG["random_state"], **trial_params
        )

        # 3) Forecaster con lags seleccionados y window_features
        forecaster = ForecasterRecursive(
            regressor=base_regressor,
            lags=selected_lags,
            window_features=window_features,
            transformer_y=FunctionTransformer(func=np.log1p, inverse_func=np.expm1),
            **({"weight_func": weight_func} if USE_WEIGHTS else {}),
        )

        # 4) Backtesting (si falla, penaliza)
        try:
            metric_vals, _ = backtesting_forecaster(
                forecaster=forecaster,
                y=y_trainval,
                exog=exog_trainval,
                cv=cv,
                metric=wmape,
                return_predictors=False,
                n_jobs=-1,
                verbose=False,
                show_progress=False,
            )
            return float(np.mean(metric_vals))  # Optuna minimiza
        except Exception:
            return 1e6  # penalización

    # -----------------------------
    # Crear estudio y optimizar
    # -----------------------------
    study = optuna.create_study(
        direction="minimize",
        sampler=STUDY_SAMPLER,
        storage=STUDY_STORAGE,
        study_name=f"{STATION}_{regressor_name}_H{H}" if STUDY_STORAGE else None,
        load_if_exists=bool(STUDY_STORAGE),
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    # -----------------------------
    # Re-instanciar con mejores params
    # -----------------------------
    best_params = dict(study.best_trial.params)

    base_regressor_best = regressor_func(
        random_state=FEATURE_SELECTION_CONFIG["random_state"], **best_params
    )

    forecaster = ForecasterRecursive(
        regressor=base_regressor_best,
        lags=selected_lags,  # si deseas optimizar lags, agrégalo a params y cámbialo aquí
        window_features=window_features,
        transformer_y=FunctionTransformer(func=np.log1p, inverse_func=np.expm1),
        **({"weight_func": weight_func} if USE_WEIGHTS else {}),
    )

    # -------------------------------------------------------------------------
    # VALIDACIÓN (train+val) con el mejor forecaster
    # -------------------------------------------------------------------------
    try:
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
        stepwise_mape_val = stepwise_mape_from_backtesting(
            preds_tv, y_trainval.loc[preds_tv.index]
        )

        print(f"\nValidacion (train+val) - {regressor_name}:")
        print(f"WMAPE %: {(100 * mape_overall_tv):.2f}")
        print(f"RMSE: {rmse_tv:.4f}")
        print(f"Stepwise MAPE: {stepwise_mape_val.to_dict()}")

        # ---------------------------------------------------------------------
        # TEST
        # ---------------------------------------------------------------------
        cv_test = TimeSeriesFold(
            steps=H,
            initial_train_size=len(y_trainval),
            refit=False,
        )

        metric_vals_test, preds_test = backtesting_forecaster(
            forecaster=forecaster,
            y=df[TARGET_COL],
            exog=df[selected_exog],
            cv=cv_test,
            metric=wmape,
            return_predictors=True,
            n_jobs=-1,
            verbose=False,
            show_progress=False,
        )

        y_pred = preds_test["pred"]

        common_index = y_test.index.intersection(y_pred.index)
        if len(common_index) > 0:
            y_test_aligned = y_test.loc[common_index]
            y_pred_aligned = y_pred.loc[common_index]
            test_rmse = rmse(y_test_aligned, y_pred_aligned)
            test_wmape = wmape(y_test_aligned, y_pred_aligned)
            stepwise_mape_test = stepwise_mape_on_test(
                y_test_aligned, y_pred_aligned, H=H
            )
        else:
            print("   WARNING: No hay indices comunes para calcular metricas de test")
            test_rmse = float("inf")
            test_wmape = float("inf")
            stepwise_mape_test = pd.Series(dtype=float)

        print(f"\nTest - {regressor_name}:")
        print(f"RMSE: {test_rmse:.4f}")
        print(f"WMAPE %: {100 * test_wmape:.2f}")
        print(f"Stepwise MAPE: {stepwise_mape_test.to_dict()}")

        # ---------------------------------------------------------------------
        # PLOTS + SAVE
        # ---------------------------------------------------------------------
        try:
            plot_files = create_prediction_plots(
                y_val=y_val,
                preds_val=preds_tv.loc[y_val.index]
                if len(preds_tv) > 0
                else pd.DataFrame(),
                y_test=y_test_aligned if len(common_index) > 0 else y_test,
                y_pred_test=y_pred_aligned if len(common_index) > 0 else y_pred,
                model_name=regressor_name,
                station=STATION,
                save_dir=plots_dir,
            )
            print(f"Graficos creados exitosamente para {regressor_name}")
        except Exception as e:
            print(f"ERROR creando graficos para {regressor_name}: {str(e)}")
            plot_files = {}

        # Guardar modelo
        timestamp_str = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        weights_suffix = "w" if USE_WEIGHTS else "nw"
        model_file = (
            models_dir / f"{regressor_name}_model_{weights_suffix}_{timestamp_str}.pkl"
        )

        with open(model_file, "wb") as f:
            pickle.dump(forecaster, f)

        print(f"Modelo entrenado guardado en: {model_file}")

        # Guardar resultados individuales
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
            "optuna": {
                "best_value": float(study.best_value),
                "best_trial": int(study.best_trial.number),
                "n_trials": int(len(study.trials)),
            },
            "model_file": str(model_file),
            "plot_files": plot_files,
            "timestamp": pd.Timestamp.now().isoformat(),
        }

        result_file = (
            results_dir_station
            / f"{regressor_name}_{weights_suffix}_{timestamp_str}.json"
        )

        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)

        print(f"Resultados guardados en: {result_file}")

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
        print(f"ERROR entrenando {regressor_name}: {str(e)}")
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

# =============================================================================
# RESUMEN FINAL
# =============================================================================
print(f"\n{'=' * 80}")
print(f"RESUMEN DE RESULTADOS PARA ESTACION: {STATION}")
print(
    f"Configuracion: {'Con exogenas' if USE_EXOG else 'Sin exogenas'}, {'Con pesos' if USE_WEIGHTS else 'Sin pesos'}"
)
print(f"{'=' * 80}")

results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values("test_wmape")

print("\nRANKING POR TEST WMAPE:")
for i, (_, row) in enumerate(results_df.iterrows(), 1):
    if row["test_wmape"] != float("inf"):
        print(
            f"{i}. {row['regressor']}: WMAPE = {100 * row['test_wmape']:.2f}%, RMSE = {row['test_rmse']:.4f}"
        )
        if i == 1:
            print(f"   Stepwise MAPE Test: {row['test_stepwise_mape']}")
    else:
        print(f"{i}. {row['regressor']}: ERROR - {row.get('error', 'Unknown error')}")

print(f"\nMEJOR MODELO: {results_df.iloc[0]['regressor']}")
print(f"Test WMAPE: {100 * results_df.iloc[0]['test_wmape']:.2f}%")
print(f"Test RMSE: {results_df.iloc[0]['test_rmse']:.4f}")
print(f"Test Stepwise MAPE: {results_df.iloc[0]['test_stepwise_mape']}")
print(f"Modelo guardado en: {results_df.iloc[0]['model_file']}")
if results_df.iloc[0]["plot_files"]:
    print("Graficos guardados:")
    for plot_type, plot_path in results_df.iloc[0]["plot_files"].items():
        print(f"   {plot_type}: {plot_path}")

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

timestamp_str = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
weights_suffix = "w" if USE_WEIGHTS else "nw"
summary_file = summary_dir / f"summary_{weights_suffix}_{timestamp_str}.json"

with open(summary_file, "w", encoding="utf-8") as f:
    json.dump(summary_data, f, indent=2, ensure_ascii=False)

print(f"\nResumen completo guardado en: {summary_file}")

csv_file = summary_dir / f"results_comparison_{weights_suffix}_{timestamp_str}.csv"
results_df.to_csv(csv_file, index=False)
print(f"Comparacion en CSV guardada en: {csv_file}")

print(f"\nProceso completado para estacion {STATION}")
