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
            "epochs": [5],
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
    # ENTRENAMIENTO MODELO FINAL CON MEJORES PARÁMETROS
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

    final_forecaster.fit(series=y_trainval, exog=exog_trainval if use_exog else None)

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

    # Guardar predicciones
    if predictions_test is not None:
        pred_filename = f"predictions_{regressor_name}_S{S}.csv"
        predictions_test.to_csv(results_dir_station / pred_filename)
        print(f"✓ Predicciones guardadas en: {results_dir_station / pred_filename}")

    # Crear gráficos
    if predictions_test is not None and y_true_test is not None:
        try:
            # Para LSTM no tenemos datos de validación, pasamos Series vacías
            y_val_empty = pd.Series(dtype=float)
            preds_val_empty = pd.DataFrame()
            create_prediction_plots(
                y_val=y_val_empty,
                preds_val=preds_val_empty,
                y_test=y_true_test,
                y_pred_test=y_pred_test_s,
                model_name=regressor_name,
                station=station,
                save_dir=plots_dir,
            )
            print(f"✓ Gráficos guardados en: {plots_dir}")
        except Exception as e:
            print(f"⚠️  Error al crear gráficos: {e}")

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
            "wmape": best_combo["val_wmape"],
        },
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
    print(f"WMAPE Validación: {100 * result['metrics_validation']['wmape']:.2f}%")
    print(
        f"WMAPE Test: {100 * result['metrics_test']['wmape']:.2f}%"
        if result["metrics_test"]["wmape"] is not None
        else "WMAPE Test: N/A"
    )
