# --- imports
import json
import pickle
import os
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
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
from skforecast.model_selection import backtesting_forecaster_multiseries
from skforecast.model_selection import TimeSeriesFold  # <-- necesario

from src.constants.parsed_fields import MODEL_RESULTS_CONFIG

import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# =============================================================================
# Función principal
# =============================================================================
def train_and_evaluate_models(
    station: str,
    steps: int,
    use_exog: bool = True,
    val_months: int = 2,
    test_months: int = 2,
    horizon: int = 0,
    param_grid: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    print(f"\n{'=' * 80}")
    print(f"Entrenando modelos LSTM para estación: {station}, Steps: {steps}")
    if horizon > 0:
        print(f"Horizonte de shift: {horizon}")
    print(f"{'=' * 80}")

    # =============================================================================
    # CARGA Y PREPARACIÓN DE DATOS
    # =============================================================================
    df = DataManager().load_data(f"data/stage/SO2/processed/processed_{station}.csv")
    df = df.sort_index()

    TARGET_COL = "target"

    if horizon > 0:
        print(f"\nAplicando shift al target: horizon={horizon}")
        df = apply_target_shift(df, target_col=TARGET_COL, step=horizon)

    exog_status_feat = "con_exog" if use_exog else "sin_exog"
    feat_sel_path = Path(
        f"data/stage/SO2/selected/lasso/{exog_status_feat}/selected_cols_{station}_lasso_lasso.json"
    )

    if not feat_sel_path.exists():
        raise FileNotFoundError(f"No se encontró {feat_sel_path}")

    with open(feat_sel_path, "r", encoding="utf-8") as f:
        sel = json.load(f)

    # --- Forzamos lag consistente
    SELECTED_LAGS = 72
    selected_lags = SELECTED_LAGS
    selected_exog = sel.get("selected_exog", []) if use_exog else []

    missing = [c for c in [TARGET_COL] + selected_exog if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en df: {missing}")

    # =============================================================================
    # SPLIT DE DATOS
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

    y_train = y_train.to_frame(name=TARGET_COL)
    y_val = y_val.to_frame(name=TARGET_COL)
    y_test = y_test.to_frame(name=TARGET_COL)
    y_trainval = y_trainval.to_frame(name=TARGET_COL)

    # =============================================================================
    # CHEQUEO NaNs
    # =============================================================================
    def assert_no_nans(df_obj, name):
        if df_obj is None:
            return
        if isinstance(df_obj, pd.DataFrame) and df_obj.isnull().any().any():
            raise ValueError(f"Se encontraron NaNs en {name}")
        elif isinstance(df_obj, pd.Series) and df_obj.isnull().any():
            raise ValueError(f"Se encontraron NaNs en {name}")

    for obj, name in [
        (y_train, "y_train"),
        (y_val, "y_val"),
        (y_test, "y_test"),
        (exog_train, "exog_train" if use_exog else None),
    ]:
        if name:
            assert_no_nans(obj, name)

    # =============================================================================
    # CONFIGURACIÓN RESULTADOS
    # =============================================================================
    S = steps
    exog_status = "con_exog" if use_exog else "sin_exog"

    results_dir = (
        Path(MODEL_RESULTS_CONFIG["analytics_dir"])
        / MODEL_RESULTS_CONFIG["results_subdir"]
    )
    station_results_dir = results_dir / station / exog_status / "lstm" / f"S{S}"
    if horizon > 0:
        station_results_dir = station_results_dir / f"H{horizon}"

    for sub in ["models", "plots", "results", "summary"]:
        (station_results_dir / sub).mkdir(parents=True, exist_ok=True)

    models_dir = station_results_dir / "models"
    plots_dir = station_results_dir / "plots"
    results_dir_station = station_results_dir / "results"
    summary_dir = station_results_dir / "summary"

    regressor_name = "LSTM"

    # =============================================================================
    # BÚSQUEDA DE HIPERPARÁMETROS (grid corto)
    # =============================================================================
    from itertools import product

    if param_grid is None:
        param_grid = {
            "recurrent_units": [[128, 64]],
            "dense_units": [[64, 32]],
            "learning_rate": [0.01],
            "epochs": [1],
            "batch_size": [128],
        }

    print("\n" + "=" * 80)
    print("BÚSQUEDA DE HIPERPARÁMETROS")
    print("=" * 80)

    grid = list(product(*param_grid.values()))
    results_list = []

    # =============================================================================
    # MODO RÁPIDO: Usar primer parámetro del grid directamente (sin búsqueda)
    # =============================================================================
    first_params = grid[0]
    rec_units, dense_units, lr, epochs, batch_size = first_params
    best_params = {
        "recurrent_units": rec_units,
        "dense_units": dense_units,
        "learning_rate": lr,
        "epochs": int(epochs),
        "batch_size": int(batch_size),
    }
    best_combo = {
        "val_wmape": 0.0,  # Placeholder cuando no hay búsqueda
    }

    print("\n" + "=" * 80)
    print("PARÁMETROS SELECCIONADOS (modo rápido)")
    print("=" * 80)
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    # =============================================================================
    # ENTRENAMIENTO MODELO PARA VALIDACIÓN (train)
    # =============================================================================
    # NOTA: Este modelo se entrena con solo 'train' para hacer backtesting en 'val'.
    # Durante el backtesting, skforecast puede re-entrenar el modelo para cada fold
    # del CV (incluso con refit=False), lo cual es normal en modelos RNN/deep learning.
    # Esto explica por qué se ve entrenamiento adicional durante el backtesting.
    print("\n" + "=" * 80)
    print("ENTRENAMIENTO MODELO PARA VALIDACIÓN (train)")
    print("=" * 80)

    val_model = create_and_compile_model(
        series=y_train,
        levels=[TARGET_COL],
        lags=selected_lags,
        steps=S,
        exog=exog_train if use_exog else None,
        recurrent_layer="LSTM",
        recurrent_units=best_params["recurrent_units"],
        dense_units=best_params["dense_units"],
        compile_kwargs={
            "optimizer": Adam(learning_rate=best_params["learning_rate"]),
            "loss": MeanSquaredError(),
        },
    )

    val_forecaster = ForecasterRnn(
        regressor=val_model,
        levels=[TARGET_COL],
        lags=selected_lags,
        transformer_series=MinMaxScaler(),
        transformer_exog=MinMaxScaler() if use_exog else None,
        fit_kwargs={
            "epochs": best_params["epochs"],
            "batch_size": best_params["batch_size"],
            "verbose": 1,
            "validation_split": 0.1,
        },
    )

    #val_forecaster.fit(series=y_train, exog=exog_train if use_exog else None)
    print("✓ Modelo para validación entrenado")

    # Inicializar variables de validación por si hay errores
    y_true_val = pd.Series(dtype=float)
    y_pred_val_s = pd.Series(dtype=float)
    predictions_val = None
    stepwise_wmape_val = {}
    pred_col_val = "pred"

    # =============================================================================
    # BACKTESTING EN VALIDACIÓN con TimeSeriesFold + backtesting_forecaster_multiseries
    # =============================================================================
    print("\n" + "=" * 80)
    print("BACKTESTING EN PERÍODO DE VALIDACIÓN (rolling)")
    print("=" * 80)

    try:
        # 1) Construimos la serie total para que el backtesting comience justo después de train
        series_bt_val = pd.concat([y_train, y_val], axis=0)
        exog_bt_val = pd.concat([exog_train, exog_val], axis=0) if use_exog else None

        # 2) Definimos el CV: sólo validación (initial_train_size = len(train))
        # NOTA: Aunque refit=False, skforecast con modelos RNN puede re-entrenar el modelo
        # para cada fold del CV debido a cómo maneja internamente los modelos de deep learning.
        # Esto es un comportamiento conocido de skforecast con ForecasterRnn.
        cv_val = TimeSeriesFold(
            steps=S,
            initial_train_size=len(y_train),
            refit=False,  # Intenta no re-entrenar, pero puede hacerlo igual con RNN
        )

        # 3) Ejecutamos backtesting sobre VALIDACIÓN
        metrics_bt_val, preds_bt_val = backtesting_forecaster_multiseries(
            forecaster=val_forecaster,
            series=series_bt_val,
            exog=exog_bt_val,
            cv=cv_val,
            levels=[TARGET_COL],
            metric=wmape,
            suppress_warnings=True,
            verbose=False,
        )

        print("\n📊 Métricas de backtesting validación (SKForecast):")
        print(metrics_bt_val)

        # 4) Extraemos predicciones de VALIDACIÓN y métricas clásicas
        pred_col_candidates_val = [
            c for c in ["pred", "y_pred", "prediction"] if c in preds_bt_val.columns
        ]
        true_col_candidates_val = [
            c for c in ["y", "y_true", TARGET_COL] if c in preds_bt_val.columns
        ]

        if not pred_col_candidates_val:
            raise RuntimeError(
                "No se encontró columna de predicción en 'preds_bt_val'."
            )

        if not true_col_candidates_val:
            y_true_aligned_val = series_bt_val.loc[preds_bt_val.index, TARGET_COL]
            preds_bt_val = preds_bt_val.copy()
            preds_bt_val["y_true"] = y_true_aligned_val
            true_col_val = "y_true"
        else:
            true_col_val = true_col_candidates_val[0]

        pred_col_val = pred_col_candidates_val[0]

        # Filtramos explícitamente el rango de VALIDACIÓN
        val_index_mask = preds_bt_val.index >= y_val.index.min()
        y_true_val = preds_bt_val.loc[val_index_mask, true_col_val].astype(float)
        y_pred_val_s = preds_bt_val.loc[val_index_mask, pred_col_val].astype(float)

        print(f"\n✅ Resultados backtesting VALIDACIÓN (recalculados):")
        wmape_val = wmape(y_true_val, y_pred_val_s)
        rmse_val = rmse(y_true_val, y_pred_val_s)
        mae_val = mae(y_true_val, y_pred_val_s)
        mse_val = mse(y_true_val, y_pred_val_s)
        r2_val = r2(y_true_val, y_pred_val_s)

        print(f"   WMAPE: {100 * wmape_val:.2f}%")
        print(f"   RMSE:  {rmse_val:.4f}")
        print(f"   MAE:   {mae_val:.4f}")
        print(f"   MSE:   {mse_val:.4f}")
        print(f"   R²:    {r2_val:.4f}")

        # 5) WMAPE por step (1..S) para validación
        try:
            stepwise_wmape_val_series = stepwise_wmape_on_test(
                y_true_val, y_pred_val_s, H=S
            )
            stepwise_wmape_val = {
                int(k): float(v) for k, v in stepwise_wmape_val_series.items()
            }
            print(f"\n📈 WMAPE por step validación (primeros 10):")
            for step, value in list(stepwise_wmape_val.items())[:10]:
                print(f"   Step {step}: {100 * value:.2f}%")
        except Exception as e:
            print(f"⚠️  No se pudo calcular stepwise_wmape validación: {e}")
            stepwise_wmape_val = {}

        # Actualizar best_combo con las métricas reales de validación
        best_combo["val_wmape"] = (
            float(wmape_val) if wmape_val is not None and wmape_val != np.inf else 0.0
        )

        # Guardar predicciones de validación
        predictions_val = preds_bt_val.copy()
    except Exception as e:
        print(f"⚠️  Error en backtesting de validación: {e}")
        print("   Continuando con valores por defecto para validación")
        wmape_val = None
        rmse_val = None
        mae_val = None
        mse_val = None
        r2_val = None

    # =============================================================================
    # ENTRENAMIENTO MODELO FINAL CON MEJORES PARÁMETROS (train+val)
    # =============================================================================
    print("\n" + "=" * 80)
    print("ENTRENAMIENTO MODELO FINAL (train+val)")
    print("=" * 80)

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
            "validation_split": 0.1,
        },
    )

    # final_forecaster.fit(series=y_trainval, exog=exog_trainval if use_exog else None)

    print("✓ Modelo final entrenado")

    # =============================================================================
    # BACKTESTING EN TEST con TimeSeriesFold + backtesting_forecaster_multiseries
    # =============================================================================
    print("\n" + "=" * 80)
    print("BACKTESTING EN PERÍODO DE TEST (rolling)")
    print("=" * 80)

    # 1) Construimos la serie total para que el backtesting comience justo después de train+val
    series_bt = pd.concat([y_trainval, y_test], axis=0)
    exog_bt = pd.concat([exog_trainval, exog_test], axis=0) if use_exog else None

    # 2) Definimos el CV: sólo test (initial_train_size = len(train+val))
    cv = TimeSeriesFold(
        steps=S,  # = final_forecaster.max_step si prefieres
        initial_train_size=len(y_trainval),
        refit=False,
    )

    # 3) Ejecutamos backtesting sobre TEST
    metrics_bt, preds_bt = backtesting_forecaster_multiseries(
        forecaster=final_forecaster,
        series=series_bt,  # DataFrame con la columna TARGET_COL
        exog=exog_bt,  # exógenas alineadas (o None)
        cv=cv,
        levels=[TARGET_COL],
        metric=wmape,  # usando wmape en lugar de mean_absolute_error
        suppress_warnings=True,
        verbose=False,
    )

    print("\n📊 Métricas de backtesting (SKForecast):")
    print(metrics_bt)

    # 4) Extraemos predicciones de TEST y métricas clásicas
    pred_col_candidates = [
        c for c in ["pred", "y_pred", "prediction"] if c in preds_bt.columns
    ]
    true_col_candidates = [
        c for c in ["y", "y_true", TARGET_COL] if c in preds_bt.columns
    ]

    if not pred_col_candidates:
        raise RuntimeError("No se encontró columna de predicción en 'preds_bt'.")

    if not true_col_candidates:
        # reconstruimos y_true a partir del índice de preds_bt
        y_true_aligned = series_bt.loc[preds_bt.index, TARGET_COL]
        preds_bt = preds_bt.copy()
        preds_bt["y_true"] = y_true_aligned
        true_col = "y_true"
    else:
        true_col = true_col_candidates[0]

    pred_col = pred_col_candidates[0]

    # Filtramos explícitamente el rango de TEST
    test_index_mask = preds_bt.index >= y_test.index.min()
    y_true_test = preds_bt.loc[test_index_mask, true_col].astype(float)
    y_pred_test_s = preds_bt.loc[test_index_mask, pred_col].astype(float)

    print(f"\n✅ Resultados backtesting TEST (recalculados):")
    wmape_test = wmape(y_true_test, y_pred_test_s)
    rmse_test = rmse(y_true_test, y_pred_test_s)
    mae_test = mae(y_true_test, y_pred_test_s)
    mse_test = mse(y_true_test, y_pred_test_s)
    r2_test = r2(y_true_test, y_pred_test_s)

    print(f"   WMAPE: {100 * wmape_test:.2f}%")
    print(f"   RMSE:  {rmse_test:.4f}")
    print(f"   MAE:   {mae_test:.4f}")
    print(f"   MSE:   {mse_test:.4f}")
    print(f"   R²:    {r2_test:.4f}")

    # 5) WMAPE por step (1..S)
    try:
        stepwise_wmape_test_series = stepwise_wmape_on_test(
            y_true_test, y_pred_test_s, H=S
        )
        stepwise_wmape_test = {
            int(k): float(v) for k, v in stepwise_wmape_test_series.items()
        }
        print(f"\n📈 WMAPE por step (primeros 10):")
        for step, value in list(stepwise_wmape_test.items())[:10]:
            print(f"   Step {step}: {100 * value:.2f}%")
    except Exception as e:
        print(f"⚠️  No se pudo calcular stepwise_wmape: {e}")
        stepwise_wmape_test = {}

    # Para compatibilidad con tu pipeline de guardado:
    predictions_test = preds_bt.copy()

    # =============================================================================
    # GUARDAR RESULTADOS
    # =============================================================================
    print("\n" + "=" * 80)
    print("GUARDANDO RESULTADOS")
    print("=" * 80)

    # Guardar modelo
    model_filename = f"forecaster_{regressor_name}_S{S}.pkl"
    model_path = models_dir / model_filename
    with open(model_path, "wb") as f:
        pickle.dump(final_forecaster, f)
    print(f"✓ Modelo guardado en: {model_path}")

    # Guardar predicciones de validación
    if predictions_val is not None:
        pred_filename_val = f"predictions_{regressor_name}_S{S}_validation.csv"
        predictions_val.to_csv(results_dir_station / pred_filename_val)
        print(
            f"✓ Predicciones validación guardadas en: {results_dir_station / pred_filename_val}"
        )

    # Guardar predicciones de test
    if predictions_test is not None:
        pred_filename = f"predictions_{regressor_name}_S{S}.csv"
        predictions_test.to_csv(results_dir_station / pred_filename)
        print(
            f"✓ Predicciones test guardadas en: {results_dir_station / pred_filename}"
        )

    # Crear gráficos
    if predictions_test is not None and y_true_test is not None:
        try:
            # Preparar predicciones de validación como DataFrame con columna 'pred'
            # La función create_prediction_plots espera un DataFrame, no una Series
            if (
                predictions_val is not None
                and len(predictions_val) > 0
                and len(y_pred_val_s) > 0
            ):
                # Filtrar solo las predicciones del período de validación
                val_index_mask_plot = predictions_val.index >= y_val.index.min()
                preds_val_df = predictions_val.loc[val_index_mask_plot].copy()
                # Asegurarse de que tenga la columna 'pred' (puede ser 'pred', 'y_pred', etc.)
                if "pred" not in preds_val_df.columns:
                    # Si la columna de predicción tiene otro nombre, renombrarla o crear 'pred'
                    if pred_col_val in preds_val_df.columns:
                        preds_val_df["pred"] = preds_val_df[pred_col_val]
                    else:
                        # Si no está, crear la columna 'pred' desde y_pred_val_s
                        preds_val_df = pd.DataFrame(
                            {"pred": y_pred_val_s}, index=y_pred_val_s.index
                        )
            else:
                preds_val_df = pd.DataFrame()

            # Usar datos de validación reales del backtesting
            create_prediction_plots(
                y_val=y_true_val if len(y_true_val) > 0 else pd.Series(dtype=float),
                preds_val=preds_val_df if len(preds_val_df) > 0 else pd.DataFrame(),
                y_test=y_true_test,
                y_pred_test=y_pred_test_s,
                model_name=regressor_name,
                station=station,
                save_dir=plots_dir,
            )
            print(f"✓ Gráficos guardados en: {plots_dir}")
        except Exception as e:
            print(f"⚠️  Error al crear gráficos: {e}")
            import traceback

            traceback.print_exc()

    # Preparar resultados finales
    results_dict = {
        "station": station,
        "model": regressor_name,
        "steps": S,
        "horizon": horizon,
        "use_exog": use_exog,
        "n_exog": len(selected_exog),
        "lags": selected_lags,
        "best_params": best_params,
        "metrics_validation": {
            "wmape": float(wmape_val)
            if wmape_val is not None and wmape_val != np.inf
            else None,
            "rmse": float(rmse_val) if rmse_val is not None else None,
            "mae": float(mae_val) if mae_val is not None else None,
            "mse": float(mse_val) if mse_val is not None else None,
            "r2": float(r2_val) if r2_val is not None else None,
        },
        "stepwise_wmape_validation": stepwise_wmape_val,
        "metrics_test": {
            "wmape": float(wmape_test)
            if wmape_test is not None and wmape_test != np.inf
            else None,
            "rmse": float(rmse_test) if rmse_test is not None else None,
            "mae": float(mae_test) if mae_test is not None else None,
            "mse": float(mse_test) if mse_test is not None else None,
            "r2": float(r2_test) if r2_test is not None else None,
        },
        "stepwise_wmape_test": stepwise_wmape_test,
    }

    # Guardar resultados JSON
    results_json = clean_params_for_json(results_dict)
    results_filename = f"results_{regressor_name}_S{S}.json"
    with open(results_dir_station / results_filename, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)
    print(f"✓ Resultados guardados en: {results_dir_station / results_filename}")

    # Guardar resumen
    try:
        timestamp_str = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        save_individual_result(
            result_data=results_dict,
            results_dir=summary_dir,
            regressor_name=regressor_name,
            timestamp_str=timestamp_str,
        )
        print(f"✓ Resumen guardado en: {summary_dir}")
    except Exception as e:
        print(f"⚠️  Error al guardar resumen: {e}")

    print("\n" + "=" * 80)
    print("✅ PROCESO COMPLETADO")
    print("=" * 80)

    return results_dict


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    STATION = "CEN-TRAF"
    USE_EXOG = True
    STEPS = 72
    VAL_MONTHS = 2
    TEST_MONTHS = 2
    HORIZON = 0

    # Grid de hiperparámetros más completo (opcional)
    PARAM_GRID = {
        "recurrent_units": [[128, 64]],
        "dense_units": [[64, 32]],
        "learning_rate": [0.01],
        "epochs": [1],
        "batch_size": [128],
    }

    result = train_and_evaluate_models(
        station=STATION,
        steps=STEPS,
        use_exog=USE_EXOG,
        val_months=VAL_MONTHS,
        test_months=TEST_MONTHS,
        horizon=HORIZON,
        param_grid=PARAM_GRID,  # Usa None para el grid por defecto
    )

    print("\n" + "=" * 80)
    print("RESUMEN FINAL")
    print("=" * 80)
    print(f"Estación: {result['station']}")
    print(f"Modelo: {result['model']}")
    print(f"Steps: {result['steps']}")
    print("\n📊 Métricas de Validación:")
    val_metrics = result["metrics_validation"]
    if val_metrics["wmape"] is not None:
        print(f"   WMAPE: {100 * val_metrics['wmape']:.2f}%")
        print(f"   RMSE:  {val_metrics['rmse']:.4f}")
        print(f"   MAE:   {val_metrics['mae']:.4f}")
        print(f"   MSE:   {val_metrics['mse']:.4f}")
        print(f"   R²:    {val_metrics['r2']:.4f}")
    else:
        print("   Métricas de validación no disponibles")
    print("\n📊 Métricas de Test:")
    test_metrics = result["metrics_test"]
    if test_metrics["wmape"] is not None:
        print(f"   WMAPE: {100 * test_metrics['wmape']:.2f}%")
        print(f"   RMSE:  {test_metrics['rmse']:.4f}")
        print(f"   MAE:   {test_metrics['mae']:.4f}")
        print(f"   MSE:   {test_metrics['mse']:.4f}")
        print(f"   R²:    {test_metrics['r2']:.4f}")
    else:
        print("   Métricas de test no disponibles")
