# --- imports
import json
import pickle
import os
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd

from sklearn.preprocessing import MinMaxScaler

# Backend de Keras -> torch ANTES de importar keras
os.environ["KERAS_BACKEND"] = "torch"
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from src.recursos.data_manager import DataManager
from src.recursos.scorers import (
    wmape,
    rmse,
    mae,
    mse,
    r2,
    stepwise_wmape_on_test,
)
from src.utils.data_splitter import split_data_by_dates, apply_target_shift
from src.utils.plot_utils import create_prediction_plots
from src.utils.results_manager import (
    clean_params_for_json,
    save_individual_result,
    save_summary_and_comparison,
    print_results_summary,
)

from skforecast.deep_learning import ForecasterRnn
from skforecast.deep_learning import create_and_compile_model

from src.constants.parsed_fields import (
    MODEL_RESULTS_CONFIG,
)

import optuna

import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


def train_and_evaluate_models(
    station: str,
    steps: int,
    use_exog: bool = True,
    n_trials: int = 20,
    study_storage: Optional[str] = None,
    val_months: int = 2,
    test_months: int = 2,
    horizon: int = 0,
    param_grid: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Entrena y evalúa modelos LSTM para una estación y steps dados.

    Parameters:
    -----------
    station : str
        Nombre de la estación (ej: "CEN-TRAF", "GIR-EPM", "ITA-CJUS", "MED-FISC")
    steps : int
        Pasos de predicción (horizonte)
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
    horizon : int, default=0
        Horizonte de shift del target.
        Si horizon > 0, el target se shifteará para predecir desde el paso 'horizon'.
        Ejemplo: horizon=24 significa que en tiempo t predecimos el valor en tiempo t+24.
    param_grid : dict, optional
        Grid de hiperparámetros para búsqueda. Si None, usa valores por defecto.
        Nota: Con Optuna, esto define los rangos/opciones para la optimización.

    Returns:
    --------
    dict
        Diccionario con los resultados del entrenamiento
    """
    print(f"\n{'=' * 80}")
    print(f"Entrenando modelos LSTM para estacion: {station}, Steps: {steps}")
    if horizon > 0:
        print(f"Horizonte de shift: {horizon}")
    print(f"{'=' * 80}")

    # =============================================================================
    # CARGA Y PREPARACIÓN DE DATOS
    # =============================================================================
    df = DataManager().load_data(f"data/stage/SO2/processed/processed_{station}.csv")
    df = df.sort_index()

    TARGET_COL = "target"

    # Aplicar shift al target si horizon > 0
    if horizon > 0:
        print(f"\nAplicando shift al target: horizon={horizon}")
        df = apply_target_shift(df, target_col=TARGET_COL, step=horizon)

    # Cargar selección de características desde JSON
    exog_status_feat = "con_exog" if use_exog else "sin_exog"
    feat_sel_path = Path(
        # f"data/stage/SO2/selected/lasso/{exog_status_feat}/selected_cols_{station}_lasso_lasso.json"
        f"data/stage/SO2/selected/lasso/{exog_status_feat}/selected_cols_CEN-TRAF_lasso_lasso.json"
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
    # Nota: window_features declaradas pero no usadas directamente en LSTM
    # (las características ya están incluidas en los lags y exog seleccionados)

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

    # Convertir las series a DataFrame (requisito de skforecast)
    y_train = y_train.to_frame(name=TARGET_COL)
    y_val = y_val.to_frame(name=TARGET_COL)
    y_test = y_test.to_frame(name=TARGET_COL)
    y_trainval = y_trainval.to_frame(name=TARGET_COL)

    # Chequeo rápido de NaNs
    def assert_no_nans(df_obj, name):
        if df_obj is None:
            return
        if isinstance(df_obj, pd.DataFrame) and df_obj.isnull().any().any():
            n = int(df_obj.isnull().sum().sum())
            raise ValueError(
                f"Se encontraron {n} NaNs en {name}. Imputa/filtra antes de entrenar."
            )
        elif isinstance(df_obj, pd.Series) and df_obj.isnull().any():
            n = int(df_obj.isnull().sum())
            raise ValueError(
                f"Se encontraron {n} NaNs en {name}. Imputa/filtra antes de entrenar."
            )

    assert_no_nans(y_train, "y_train")
    assert_no_nans(y_val, "y_val")
    assert_no_nans(y_test, "y_test")
    if use_exog:
        assert_no_nans(exog_train, "exog_train")
        assert_no_nans(exog_val, "exog_val")
        assert_no_nans(exog_test, "exog_test")

    # =============================================================================
    # ESTRUCTURA DE RESULTADOS
    # =============================================================================
    S = steps  # Pasos de predicción

    # Carpetas de resultados
    results_dir = (
        Path(MODEL_RESULTS_CONFIG["analytics_dir"])
        / MODEL_RESULTS_CONFIG["results_subdir"]
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    exog_status = "con_exog" if use_exog else "sin_exog"
    station_results_dir = results_dir / station / exog_status / "lstm" / f"S{S}"
    if horizon > 0:
        station_results_dir = station_results_dir / f"H{horizon}"
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
    print(f"   Configuracion: {exog_status}, Steps: {S}")
    if horizon > 0:
        print(f"   Horizonte de shift: {horizon}")
    print("=" * 60)

    regressor_name = "LSTM"

    # =============================================================================
    # CONFIGURACIÓN OPTUNA
    # =============================================================================
    # Definir rangos por defecto para hiperparámetros si no se proporciona param_grid
    if param_grid is None:
        param_grid = {
            "recurrent_units": [[100, 50], [128, 64], [64, 32], [150, 75]],
            "dense_units": [[32, 16], [64, 32], [50, 25], [16, 8]],
            "learning_rate": [1e-4, 5e-3, 1e-3, 1e-2, 5e-2, 1e-1],
            "epochs": [5, 8, 10, 15, 20],
            "batch_size": [64, 128, 256],
        }

    # Configurar sampler de Optuna
    STUDY_SAMPLER = optuna.samplers.TPESampler(seed=42)

    # =============================================================================
    # OPTIMIZACIÓN DE HIPERPARÁMETROS CON OPTUNA
    # =============================================================================
    print(f"\n{'=' * 60}")
    print(f"Optimización de hiperparámetros con Optuna: {n_trials} trials")
    print(f"{'=' * 60}")

    def objective(trial: optuna.Trial) -> float:
        """Función objetivo para Optuna: minimiza WMAPE en validación."""
        # Sugerir hiperparámetros
        rec_units = trial.suggest_categorical(
            "recurrent_units", param_grid["recurrent_units"]
        )
        dense_units = trial.suggest_categorical(
            "dense_units", param_grid["dense_units"]
        )
        lr = trial.suggest_categorical("learning_rate", param_grid["learning_rate"])
        epochs = trial.suggest_categorical("epochs", param_grid["epochs"])
        batch_size = trial.suggest_categorical("batch_size", param_grid["batch_size"])

        try:
            # Crear modelo con create_and_compile_model
            temp_model = create_and_compile_model(
                series=y_train,
                levels=[TARGET_COL],
                lags=selected_lags,
                steps=S,
                exog=exog_train if use_exog else None,
                recurrent_layer="LSTM",
                recurrent_units=rec_units,
                dense_units=dense_units,
                compile_kwargs={
                    "optimizer": Adam(learning_rate=lr),
                    "loss": MeanSquaredError(),
                },
            )

            # Crear forecaster con el modelo
            temp_forecaster = ForecasterRnn(
                regressor=temp_model,
                levels=[TARGET_COL],
                lags=selected_lags,
                transformer_series=MinMaxScaler(),
                transformer_exog=MinMaxScaler() if use_exog else None,
                fit_kwargs={
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "verbose": 0,
                    "callbacks": [
                        EarlyStopping(
                            monitor="loss",
                            patience=5,
                            restore_best_weights=True,
                            verbose=0,
                        ),
                        ReduceLROnPlateau(
                            monitor="loss",
                            factor=0.5,
                            patience=3,
                            min_lr=1e-6,
                            verbose=0,
                        ),
                    ],
                    "validation_split": 0.1,
                },
            )

            # Fit (series y exog con mismo índice)
            temp_forecaster.fit(series=y_train, exog=exog_train if use_exog else None)

            # Predicción en validación
            y_pred_val = temp_forecaster.predict(
                steps=S, exog=exog_val if use_exog else None
            )

            # ======== VALIDACIÓN ========
            y_true_val = y_val[TARGET_COL]  # DataFrame → Series
            y_pred_val_s = y_pred_val["pred"]  # tomar solo la columna pred

            # Alinear índices por seguridad
            y_true_val, y_pred_val_s = y_true_val.align(y_pred_val_s, join="inner")

            # Calcular WMAPE (métrica a minimizar)
            val_wmape = wmape(y_true_val, y_pred_val_s)

            return float(val_wmape)  # Optuna minimiza

        except Exception as e:
            print(f"⚠️  Trial falló: {str(e)}")
            return 1e6  # Penalización para trials que fallan

    # Crear estudio y optimizar
    study_name = f"{station}_{regressor_name}_lstm_S{S}"
    if horizon > 0:
        study_name += f"_H{horizon}"

    study = optuna.create_study(
        direction="minimize",
        sampler=STUDY_SAMPLER,
        storage=study_storage,
        study_name=study_name if study_storage else None,
        load_if_exists=bool(study_storage),
    )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Obtener mejores parámetros
    best_params = {
        "recurrent_units": study.best_params["recurrent_units"],
        "dense_units": study.best_params["dense_units"],
        "learning_rate": study.best_params["learning_rate"],
        "epochs": study.best_params["epochs"],
        "batch_size": study.best_params["batch_size"],
    }

    print(f"\n🏆 Mejor combinación encontrada (WMAPE: {study.best_value:.4f}):")
    print(f"   {best_params}")

    # Guardar información del estudio Optuna
    optuna_info = {
        "best_value": float(study.best_value),
        "n_trials": len(study.trials),
        "best_params": best_params,
    }

    # =============================================================================
    # ENTRENAR MODELO FINAL CON MEJORES PARÁMETROS
    # =============================================================================
    print(f"\n{'=' * 60}")
    print("Entrenando modelo final con mejores parámetros...")
    print(f"{'=' * 60}")

    final_model = create_and_compile_model(
        series=y_trainval,
        levels=[TARGET_COL],
        lags=selected_lags,
        steps=S,
        exog=exog_trainval if use_exog else None,
        recurrent_layer="LSTM",
        recurrent_units=best_params["recurrent_units"],
        dense_units=best_params["dense_units"],
        compile_kwargs={
            "optimizer": Adam(learning_rate=best_params["learning_rate"]),
            "loss": MeanSquaredError(),
        },
    )

    final_forecaster = ForecasterRnn(
        regressor=final_model,
        levels=[TARGET_COL],
        lags=selected_lags,
        transformer_series=MinMaxScaler(),
        transformer_exog=MinMaxScaler() if use_exog else None,
        fit_kwargs={
            "epochs": best_params["epochs"],
            "batch_size": best_params["batch_size"],
            "verbose": 1,
            "callbacks": [
                EarlyStopping(
                    monitor="loss", patience=5, restore_best_weights=True, verbose=1
                ),
                ReduceLROnPlateau(
                    monitor="loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
                ),
            ],
            "validation_split": 0.1,
        },
    )

    # Entrenar con train+val
    final_forecaster.fit(series=y_trainval, exog=exog_trainval if use_exog else None)

    # =============================================================================
    # VALIDACIÓN (train+val) CON EL MEJOR FORECASTER
    # =============================================================================
    print(f"\n{'=' * 60}")
    print("Evaluando en validación (train+val)...")
    print(f"{'=' * 60}")

    # Usar trainval completo para evaluación
    y_pred_trainval = final_forecaster.predict(
        steps=S, exog=exog_trainval if use_exog else None
    )

    y_trainval_series = y_trainval[TARGET_COL]
    y_pred_trainval_s = y_pred_trainval["pred"]

    y_trainval_aligned, y_pred_trainval_aligned = y_trainval_series.align(
        y_pred_trainval_s, join="inner"
    )

    mape_overall_tv = wmape(y_trainval_aligned, y_pred_trainval_aligned)
    rmse_tv = rmse(y_trainval_aligned, y_pred_trainval_aligned)
    mae_tv = mae(y_trainval_aligned, y_pred_trainval_aligned)
    mse_tv = mse(y_trainval_aligned, y_pred_trainval_aligned)
    r2_tv = r2(y_trainval_aligned, y_pred_trainval_aligned)

    # Calcular stepwise WMAPE para validación
    stepwise_wmape_val = stepwise_wmape_on_test(
        y_trainval_aligned, y_pred_trainval_aligned, H=S
    )

    print(f"\nValidacion (train+val) - {regressor_name}:")
    print(f"RMSE: {rmse_tv:.4f}")
    print(f"MAE: {mae_tv:.4f}")
    print(f"MSE: {mse_tv:.4f}")
    print(f"R²: {r2_tv:.4f}")
    print(f"WMAPE %: {(100 * mape_overall_tv):.2f}")
    print(f"Stepwise WMAPE: {stepwise_wmape_val.to_dict()}")

    # =============================================================================
    # TEST
    # =============================================================================
    print(f"\n{'=' * 60}")
    print("Evaluando en test...")
    print(f"{'=' * 60}")

    y_pred_test = final_forecaster.predict(
        steps=S, exog=exog_test if use_exog else None
    )

    y_true_test = y_test[TARGET_COL]
    y_pred_test_s = y_pred_test["pred"]

    y_test_aligned, y_pred_test_aligned = y_true_test.align(y_pred_test_s, join="inner")

    test_rmse = rmse(y_test_aligned, y_pred_test_aligned)
    test_mae = mae(y_test_aligned, y_pred_test_aligned)
    test_mse = mse(y_test_aligned, y_pred_test_aligned)
    test_r2 = r2(y_test_aligned, y_pred_test_aligned)
    test_wmape = wmape(y_test_aligned, y_pred_test_aligned)
    stepwise_wmape_test = stepwise_wmape_on_test(
        y_test_aligned, y_pred_test_aligned, H=S
    )

    print(f"\nTest - {regressor_name}:")
    print(f"RMSE: {test_rmse:.4f}")
    print(f"MAE: {test_mae:.4f}")
    print(f"MSE: {test_mse:.4f}")
    print(f"R²: {test_r2:.4f}")
    print(f"WMAPE %: {100 * test_wmape:.2f}")
    print(f"Stepwise WMAPE: {stepwise_wmape_test.to_dict()}")

    # =============================================================================
    # PLOTS + SAVE
    # =============================================================================
    try:
        # Para validación: extraer predicciones que caen en y_val
        if len(y_pred_trainval) > 0:
            common_index_val = y_val.index.intersection(y_pred_trainval.index)
            if len(common_index_val) > 0:
                preds_val_plot = y_pred_trainval.loc[common_index_val]
            else:
                preds_val_plot = pd.DataFrame()
        else:
            preds_val_plot = pd.DataFrame()

        plot_files = create_prediction_plots(
            y_val=y_val[TARGET_COL],
            preds_val=preds_val_plot,
            y_test=y_test_aligned,
            y_pred_test=y_pred_test_aligned,
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
    model_file = models_dir / f"{regressor_name}_model_{timestamp_str}.pkl"

    with open(model_file, "wb") as f:
        pickle.dump(final_forecaster, f)

    print(f"Modelo entrenado guardado en: {model_file}")

    # Guardar resultados individuales
    result_data = {
        "station": station,
        "model_type": regressor_name,
        "use_exog": use_exog,
        "steps": S,
        "horizon": horizon if horizon > 0 else None,
        "validation_metrics": {
            "rmse": float(rmse_tv),
            "mae": float(mae_tv),
            "mse": float(mse_tv),
            "r2": float(r2_tv),
            "wmape": float(mape_overall_tv),
            "stepwise_wmape": stepwise_wmape_val.to_dict(),
        },
        "test_metrics": {
            "rmse": float(test_rmse),
            "mae": float(test_mae),
            "mse": float(test_mse),
            "r2": float(test_r2),
            "wmape": float(test_wmape),
            "stepwise_wmape": stepwise_wmape_test.to_dict(),
        },
        "best_params": clean_params_for_json(best_params),
        "optuna": clean_params_for_json(optuna_info),
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

    # Preparar all_results para save_summary_and_comparison
    all_results = [
        {
            "regressor": regressor_name,
            "val_rmse": rmse_tv,
            "val_mae": mae_tv,
            "val_mse": mse_tv,
            "val_r2": r2_tv,
            "val_wmape": mape_overall_tv,
            "val_stepwise_wmape": stepwise_wmape_val.to_dict(),
            "test_rmse": test_rmse,
            "test_mae": test_mae,
            "test_mse": test_mse,
            "test_r2": test_r2,
            "test_wmape": test_wmape,
            "test_stepwise_wmape": stepwise_wmape_test.to_dict(),
            "best_params": clean_params_for_json(best_params),
            "model_file": str(model_file),
            "plot_files": plot_files,
        }
    ]

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
    print(f"\n✅ Proceso completado para estacion {station}, steps {S}")
    if horizon > 0:
        print(f"   Horizonte de shift: {horizon}")

    return {
        "station": station,
        "steps": S,
        "horizon": horizon if horizon > 0 else None,
        "use_exog": use_exog,
        "results": all_results,
        "optuna_study": optuna_info,
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
    STEPS = 72  # pasos de predicción
    N_TRIALS = 20  # Número de trials para Optuna
    STUDY_STORAGE = None  # ej: "sqlite:///optuna.db" si quieres persistir estudios
    VAL_MONTHS = 2  # Meses para validación
    TEST_MONTHS = 2  # Meses para test
    HORIZON = (
        0  # Horizonte para shift del target (0 = no shift, >0 = shift hacia adelante)
    )

    result = train_and_evaluate_models(
        station=STATION,
        steps=STEPS,
        use_exog=USE_EXOG,
        n_trials=N_TRIALS,
        study_storage=STUDY_STORAGE,
        val_months=VAL_MONTHS,
        test_months=TEST_MONTHS,
        horizon=HORIZON,
    )
