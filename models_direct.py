# --- imports
import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional

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
from src.utils.results_manager import (
    clean_params_for_json,
    save_individual_result,
    save_summary_and_comparison,
    print_results_summary,
)

from skforecast.direct import ForecasterDirect
from skforecast.model_selection import (
    TimeSeriesFold,
    backtesting_forecaster,
)
from sklearn.model_selection import TimeSeriesSplit

from src.constants.parsed_fields import (
    FEATURE_SELECTION_CONFIG,
    REGRESSORS_CONFIG,
    MODEL_RESULTS_CONFIG,
)


def train_and_evaluate_models(
    station: str,
    horizon: int,
    use_exog: bool = True,
    n_trials: int = 20,
    study_storage: Optional[str] = None,
    val_months: int = 2,
    test_months: int = 2,
) -> Dict[str, Any]:
    """
    Entrena y evalúa modelos Direct para una estación y horizonte dados.

    Parameters:
    -----------
    station : str
        Nombre de la estación (ej: "CEN-TRAF", "GIR-EPM", "ITA-CJUS", "MED-FISC")
    horizon : int
        Horizonte directo (número de pasos hacia adelante que se predicen simultáneamente)
        Ejemplo: horizon=72 predice los próximos 72 pasos usando un modelo por cada paso
    use_exog : bool
        True para modelo con exógenas, False para sin exógenas
    n_trials : int
        Número de trials para Optuna
    study_storage : str, optional
        Storage para Optuna (ej: "sqlite:///optuna.db")
    val_months : int
        Número de meses para validación
    test_months : int
        Número de meses para test

    Returns:
    --------
    dict
        Diccionario con los resultados del entrenamiento
    """
    print(f"\n{'=' * 80}")
    print(f"Entrenando modelos Direct para estacion: {station}, Horizonte: {horizon}")
    print(f"{'=' * 80}")

    # =============================================================================
    # CARGA Y PREPARACIÓN DE DATOS
    # =============================================================================
    df = DataManager().load_data(f"data/stage/SO2/processed/processed_{station}.csv")
    df = df.sort_index()

    TARGET_COL = "target"

    # Cargar selección de características desde JSON
    exog_status_feat = "con_exog" if use_exog else "sin_exog"
    feat_sel_path = Path(
        f"data/stage/SO2/selected/lasso/{exog_status_feat}/selected_cols_{station}_lasso_rf.json"
    )

    if not feat_sel_path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de características seleccionadas: {feat_sel_path}"
        )

    with open(feat_sel_path, "r", encoding="utf-8") as f:
        sel = json.load(f)

    selected_lags: list[int] = sel["selected_lags"]
    selected_exog: list[str] = sel.get("selected_exog", []) if use_exog else []

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
    (
        y_train,
        exog_train,
        y_val,
        exog_val,
        y_test,
        exog_test,
        y_trainval,
        exog_trainval,
    ) = split_data_by_dates(
        df=df,
        target_col=TARGET_COL,
        exog_cols=selected_exog,
        val_months=val_months,
        test_months=test_months,
    )

    # =============================================================================
    # CV Y ESTRUCTURA DE RESULTADOS
    # =============================================================================
    H = horizon  # Horizonte directo

    # Para ForecasterDirect, usamos TimeSeriesSplit similar a models.py
    # pero adaptado para el horizonte H
    tscv = TimeSeriesSplit(
        n_splits=5,
        test_size=H,
        gap=72,
        max_train_size=len(y_train),
    )

    # Carpetas de resultados
    results_dir = (
        Path(MODEL_RESULTS_CONFIG["analytics_dir"])
        / MODEL_RESULTS_CONFIG["results_subdir"]
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    exog_status = "con_exog" if use_exog else "sin_exog"
    station_results_dir = results_dir / station / exog_status / "direct" / f"H{H}"
    station_results_dir.mkdir(parents=True, exist_ok=True)

    models_dir = station_results_dir / "models"
    plots_dir = station_results_dir / "plots"
    results_dir_station = station_results_dir / "results"
    summary_dir = station_results_dir / "summary"

    for subdir in [models_dir, plots_dir, results_dir_station, summary_dir]:
        subdir.mkdir(parents=True, exist_ok=True)

    print(f"\nEstructura de carpetas creada para {station}:")
    print(f"   {station_results_dir}")
    print("   -- models/     (modelos entrenados .pkl)")
    print("   -- plots/      (graficos de predicciones)")
    print("   -- results/    (resultados individuales .json)")
    print("   -- summary/    (resumenes y comparaciones)")
    print(f"   Configuracion: {exog_status}, Horizonte Direct: {H}")
    print("=" * 60)

    # =============================================================================
    # CONFIGURACIÓN OPTUNA
    # =============================================================================
    STUDY_SAMPLER = optuna.samplers.TPESampler(
        seed=FEATURE_SELECTION_CONFIG["random_state"]
    )

    # =============================================================================
    # ENTRENAMIENTO + EVALUACIÓN (Optuna)
    # =============================================================================
    all_results = []

    for regressor_config in REGRESSORS_CONFIG:
        regressor_name = regressor_config["name"]
        regressor_func_name = regressor_config["regressor_func"]
        param_distributions = regressor_config["params"]  # solo listas → categóricos

        print(f"\n{'=' * 60}")
        print(f"Entrenando modelo Direct (Optuna): {regressor_name}")
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

            # 3) ForecasterDirect con lags seleccionados y window_features
            forecaster = ForecasterDirect(
                regressor=base_regressor,
                lags=selected_lags,
                steps=H,
                window_features=window_features,
            )

            # 4) Backtesting (si falla, penaliza)
            try:
                metric_vals, _ = backtesting_forecaster(
                    forecaster=forecaster,
                    y=y_trainval,
                    exog=exog_trainval,
                    cv=tscv,
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
            storage=study_storage,
            study_name=f"{station}_{regressor_name}_direct_H{H}"
            if study_storage
            else None,
            load_if_exists=bool(study_storage),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        # -----------------------------
        # Re-instanciar con mejores params
        # -----------------------------
        best_params = dict(study.best_trial.params)

        base_regressor_best = regressor_func(
            random_state=FEATURE_SELECTION_CONFIG["random_state"], **best_params
        )

        forecaster = ForecasterDirect(
            regressor=base_regressor_best,
            lags=selected_lags,
            steps=H,
            window_features=window_features,
        )

        # -------------------------------------------------------------------------
        # VALIDACIÓN (train+val) con el mejor forecaster
        # -------------------------------------------------------------------------
        try:
            metric_vals_tv, preds_tv = backtesting_forecaster(
                forecaster=forecaster,
                y=y_trainval,
                exog=exog_trainval,
                cv=tscv,
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
            # Para ForecasterDirect predecimos en bloques de tamaño H sobre test
            y_full = pd.concat([y_trainval, y_test])
            exog_full = pd.concat([exog_trainval, exog_test]) if use_exog else None

            cv_test = TimeSeriesFold(
                steps=H,
                initial_train_size=len(y_trainval),
                refit=False,
            )

            metric_vals_test, preds_all = backtesting_forecaster(
                forecaster=forecaster,
                y=y_full,
                exog=exog_full,
                cv=cv_test,
                metric=wmape,
                return_predictors=True,
                n_jobs=-1,
                verbose=False,
                show_progress=False,
            )

            # Filtrar sólo predicciones que caen en el rango de test
            preds_test = preds_all.loc[
                preds_all.index.intersection(y_test.index)
            ].copy()

            if len(preds_test) > 0:
                y_test_aligned = y_test.loc[preds_test.index]
                y_pred_aligned = preds_test["pred"]
                test_rmse = rmse(y_test_aligned, y_pred_aligned)
                test_wmape = wmape(y_test_aligned, y_pred_aligned)
                stepwise_mape_test = stepwise_mape_on_test(
                    y_test_aligned, y_pred_aligned, H=H
                )
            else:
                print(
                    "   WARNING: No hay indices comunes para calcular metricas de test"
                )
                test_rmse = float("inf")
                test_wmape = float("inf")
                stepwise_mape_test = pd.Series(dtype=float)
                y_test_aligned = y_test
                y_pred_aligned = pd.Series(dtype=float)

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
                    y_test=y_test_aligned if len(preds_test) > 0 else y_test,
                    y_pred_test=y_pred_aligned
                    if len(preds_test) > 0
                    else pd.Series(dtype=float),
                    model_name=regressor_name,
                    station=station,
                    save_dir=plots_dir,
                )
                print(f"Graficos creados exitosamente para {regressor_name}")
            except Exception as e:
                print(f"ERROR creando graficos para {regressor_name}: {str(e)}")
                plot_files = {}

            # Guardar modelo
            timestamp_str = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            model_file = models_dir / f"{regressor_name}_direct_{timestamp_str}.pkl"

            with open(model_file, "wb") as f:
                pickle.dump(forecaster, f)

            print(f"Modelo entrenado guardado en: {model_file}")

            # Guardar resultados individuales
            result_data = {
                "station": station,
                "model_type": f"{regressor_name} (Direct)",
                "use_exog": use_exog,
                "horizon": H,
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

            result_file = save_individual_result(
                result_data=result_data,
                results_dir=results_dir_station,
                regressor_name=regressor_name,
                timestamp_str=timestamp_str,
            )

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
            import traceback

            traceback.print_exc()
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
    timestamp_str = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    # Imprimir resumen en consola
    print_results_summary(
        all_results=all_results,
        station=station,
        use_exog=use_exog,
    )

    # Guardar resumen y comparación
    summary_file, csv_file = save_summary_and_comparison(
        all_results=all_results,
        station=station,
        use_exog=use_exog,
        summary_dir=summary_dir,
        timestamp_str=timestamp_str,
    )

    print(f"\nResumen completo guardado en: {summary_file}")
    print(f"Comparacion en CSV guardada en: {csv_file}")
    print(f"\n✅ Proceso completado para estacion {station}, horizonte directo {H}")

    return {
        "station": station,
        "horizon": H,
        "use_exog": use_exog,
        "results": all_results,
        "summary_file": str(summary_file),
        "csv_file": str(csv_file),
        "timestamp": timestamp_str,
    }


# =============================================================================
# EJECUCIÓN DIRECTA (para compatibilidad con ejecución antigua)
# =============================================================================
if __name__ == "__main__":
    # Configuración por defecto para ejecución directa
    STATION = "CEN-TRAF"  # Opciones: "CEN-TRAF", "GIR-EPM", "ITA-CJUS", "MED-FISC"
    USE_EXOG = True  # True para modelo con exógenas, False para sin exógenas
    HORIZON = 72  # horizonte directo (número de pasos a modelar simultáneamente)
    N_TRIALS = 20  # ajusta según presupuesto
    STUDY_STORAGE = None  # ej: "sqlite:///optuna.db" si quieres persistir estudios

    result = train_and_evaluate_models(
        station=STATION,
        horizon=HORIZON,
        use_exog=USE_EXOG,
        n_trials=N_TRIALS,
        study_storage=STUDY_STORAGE,
    )
