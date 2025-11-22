# --- baseline_model.py --------------------------------------------------------
# Baseline "naive-por-lag": para horizonte k, predice y[t-k].
# No usa Optuna ni modelos de skforecast. Genera folds manualmente (sin .split()).
# Mantiene tu estructura de guardados, plots y resúmenes.

import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

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
    clean_params_for_json,  # compat
    save_individual_result,
    save_summary_and_comparison,
    print_results_summary,
)

from src.constants.parsed_fields import MODEL_RESULTS_CONFIG


# ---------- Utilidades baseline naive-by-lag (sin modelo skforecast) ----------


def _get_index_freq(idx: pd.DatetimeIndex):
    """Obtiene la frecuencia como offset, con fallbacks robustos."""
    freq = idx.freq if idx.freq is not None else pd.infer_freq(idx)
    if freq is None:
        # último recurso: delta modal
        deltas = np.diff(idx.view("int64"))
        if len(deltas) == 0:
            raise ValueError("No se pudo inferir la frecuencia del índice.")
        step = pd.to_timedelta(pd.Series(deltas).mode().iloc[0])
        return step
    return pd.tseries.frequencies.to_offset(freq)


def _predict_naive_by_lag_from_train_end(
    y: pd.Series, train_end_ts: pd.Timestamp, steps: int
) -> pd.Series:
    """
    Predicciones naive-por-lag:
    [y[train_end - 1*freq], y[train_end - 2*freq], ..., y[train_end - steps*freq]]
    """
    y = y.sort_index()
    off = _get_index_freq(y.index)
    future_index = pd.date_range(start=train_end_ts + off, periods=steps, freq=off)

    preds = []
    for k in range(1, steps + 1):
        ts_past = train_end_ts - k * off
        if ts_past in y.index:
            val = y.loc[ts_past]
        else:
            # Fallback: timestamp <= ts_past más cercano
            prev_idx = y.index[y.index <= ts_past]
            if len(prev_idx) == 0:
                raise ValueError(f"No hay dato disponible para lag {k} (ts {ts_past}).")
            val = y.loc[prev_idx[-1]]
        preds.append(val)

    return pd.Series(preds, index=future_index, name="pred")


def _backtest_naive_by_lag_manual(
    y: pd.Series,
    initial_train_size: int,
    steps: int,
    allow_incomplete_last_fold: bool = True,
) -> pd.DataFrame:
    """
    Backtesting naive-por-lag sin depender de TimeSeriesFold.split():
    - Genera folds manualmente con ventana de test de tamaño `steps`.
    - El último fold puede ser incompleto si `allow_incomplete_last_fold=True`.
    """
    y = y.sort_index()
    n = len(y)
    if initial_train_size <= 0 or initial_train_size >= n:
        raise ValueError("initial_train_size debe estar en (0, len(y)).")

    all_preds = []
    pos = initial_train_size  # inicio del primer test (posición en y)
    while pos < n:
        test_start = pos
        test_end = min(pos + steps, n)  # exclusiv
        steps_fold = test_end - test_start
        if steps_fold <= 0:
            break
        if steps_fold < steps and not allow_incomplete_last_fold:
            break

        # fin del train es el elemento anterior al inicio del test
        train_end_pos = test_start - 1
        train_end_ts = y.index[train_end_pos]

        preds = _predict_naive_by_lag_from_train_end(y, train_end_ts, steps_fold)
        all_preds.append(preds)

        pos += steps  # siguiente bloque de test

    if not all_preds:
        return pd.DataFrame(columns=["pred"])

    preds_concat = pd.concat(all_preds).to_frame(name="pred")
    # Si por alguna razón hay timestamps repetidos, conserva el último
    preds_concat = preds_concat[~preds_concat.index.duplicated(keep="last")]
    return preds_concat.sort_index()


# ------------------------------- Entrenamiento --------------------------------


def train_and_evaluate_models(
    station: str,
    step: int,
    use_exog: bool = True,  # ignorado en baseline (exog=None)
    n_trials: int = 0,  # compat: no se usa
    study_storage: Optional[str] = None,  # compat: no se usa
    val_months: int = 2,
    test_months: int = 2,
    horizon: int = 0,
) -> Dict[str, Any]:
    """
    Baseline 'naive-por-lag' (paso k usa y[t-k]).
    Mantiene estructura de guardado, plots y resumen.
    """
    print(f"\n{'=' * 80}")
    print(f"Entrenando modelos para estacion: {station}, Step: {step}")
    if horizon > 0:
        print(f"Horizonte de shift: {horizon}")
    print(f"{'=' * 80}")

    # ------------------------------ Carga de datos ------------------------------
    df = DataManager().load_data(f"data/stage/SO2/processed/processed_{station}.csv")
    df = df.sort_index()

    TARGET_COL = "target"

    if horizon > 0:
        print(f"\nAplicando shift al target: horizon={horizon}")
        df = apply_target_shift(df, target_col=TARGET_COL, step=horizon)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Falta columna target '{TARGET_COL}' en df.")

    # ----------------------------- División de datos ----------------------------
    (
        y_train,
        _exog_train,
        y_val,
        _exog_val,
        y_test,
        _exog_test,
        y_trainval,
        _exog_trainval,
    ) = split_data_by_dates(
        df=df,
        target_col=TARGET_COL,
        exog_cols=[],  # baseline no usa exógenas
        val_months=val_months,
        test_months=test_months,
    )

    print("Data split summary:")
    print(f"  Total data range: {df.index.min()} to {df.index.max()}")
    print(
        f"  Training: {y_train.index.min()} to {y_train.index.max()} ({len(y_train)} samples)"
    )
    print(
        f"  Validation: {y_val.index.min()} to {y_val.index.max()} ({len(y_val)} samples)"
    )
    print(
        f"  Test: {y_test.index.min()} to {y_test.index.max()} ({len(y_test)} samples)\n"
    )

    # ------------------------- Estructura de resultados -------------------------
    S = step

    results_dir = (
        Path(MODEL_RESULTS_CONFIG["analytics_dir"])
        / MODEL_RESULTS_CONFIG["results_subdir"]
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    exog_status = "con_exog" if use_exog else "sin_exog"

    baseline_dir = "baseline"

    station_results_dir = results_dir / station / exog_status / baseline_dir / f"S{S}"
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
    print(f"   Configuracion: {exog_status}, Step: {S}")
    if horizon > 0:
        print(f"   Horizonte de shift: {horizon}")
    print("=" * 60)

    all_results = []
    model_name = "NaiveByLag"

    print(f"\n{'=' * 60}")
    print(f"Entrenando modelo baseline: {model_name}")
    print(f"{'=' * 60}")

    # ----------------------- Validación (train+val) -----------------------------
    initial_train_size_val = len(y_train)  # como hace TimeSeriesFold
    preds_tv = _backtest_naive_by_lag_manual(
        y=y_trainval,
        initial_train_size=initial_train_size_val,
        steps=S,
        allow_incomplete_last_fold=True,
    )

    y_trainval_aligned = y_trainval.loc[preds_tv.index]
    preds_tv_aligned = preds_tv["pred"]

    mape_overall_tv = wmape(y_trainval_aligned, preds_tv_aligned)
    rmse_tv = rmse(y_trainval_aligned, preds_tv_aligned)
    mae_tv = mae(y_trainval_aligned, preds_tv_aligned)
    mse_tv = mse(y_trainval_aligned, preds_tv_aligned)
    r2_tv = r2(y_trainval_aligned, preds_tv_aligned)

    stepwise_wmape_val = stepwise_wmape_on_test(
        y_trainval_aligned, preds_tv_aligned, H=S
    )

    print(f"\nValidacion (train+val) - {model_name}:")
    print(f"RMSE: {rmse_tv:.4f}")
    print(f"MAE: {mae_tv:.4f}")
    print(f"MSE: {mse_tv:.4f}")
    print(f"R²: {r2_tv:.4f}")
    print(f"WMAPE %: {(100 * mape_overall_tv):.2f}")
    print(f"Stepwise WMAPE: {stepwise_wmape_val.to_dict()}")

    # --------------------------------- Test -------------------------------------
    initial_train_size_test = len(y_trainval)
    preds_test = _backtest_naive_by_lag_manual(
        y=df[TARGET_COL],
        initial_train_size=initial_train_size_test,
        steps=S,
        allow_incomplete_last_fold=True,
    )
    y_pred = preds_test["pred"]

    common_index = y_test.index.intersection(y_pred.index)
    if len(common_index) > 0:
        y_test_aligned = y_test.loc[common_index]
        y_pred_aligned = y_pred.loc[common_index]
        test_rmse = rmse(y_test_aligned, y_pred_aligned)
        test_mae = mae(y_test_aligned, y_pred_aligned)
        test_mse = mse(y_test_aligned, y_pred_aligned)
        test_r2 = r2(y_test_aligned, y_pred_aligned)
        test_wmape = wmape(y_test_aligned, y_pred_aligned)
        stepwise_wmape_test = stepwise_wmape_on_test(
            y_test_aligned, y_pred_aligned, H=S
        )
    else:
        print("   WARNING: No hay indices comunes para calcular metricas de test")
        y_test_aligned = y_test
        y_pred_aligned = y_pred
        test_rmse = float("inf")
        test_mae = float("inf")
        test_mse = float("inf")
        test_r2 = float("-inf")
        test_wmape = float("inf")
        stepwise_wmape_test = pd.Series(dtype=float)

    print(f"\nTest - {model_name}:")
    print(f"RMSE: {test_rmse:.4f}")
    print(f"MAE: {test_mae:.4f}")
    print(f"MSE: {test_mse:.4f}")
    print(f"R²: {test_r2:.4f}")
    print(f"WMAPE %: {100 * test_wmape:.2f}")
    print(f"Stepwise WMAPE: {stepwise_wmape_test.to_dict()}")

    # ---------------------------- Plots + guardado ------------------------------
    try:
        if len(preds_tv) > 0:
            common_index_val = y_val.index.intersection(preds_tv.index)
            preds_val_plot = (
                preds_tv.loc[common_index_val]
                if len(common_index_val) > 0
                else pd.DataFrame()
            )
        else:
            preds_val_plot = pd.DataFrame()

        plot_files = create_prediction_plots(
            y_val=y_val,
            preds_val=preds_val_plot,
            y_test=y_test_aligned if len(common_index) > 0 else y_test,
            y_pred_test=y_pred_aligned if len(common_index) > 0 else y_pred,
            model_name=model_name,
            station=station,
            save_dir=plots_dir,
        )
        print(f"Graficos creados exitosamente para {model_name}")
    except Exception as e:
        print(f"ERROR creando graficos para {model_name}: {str(e)}")
        plot_files = {}

    # Guardar "modelo" liviano (descriptor del baseline)
    timestamp_str = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    model_file = models_dir / f"{model_name}_descriptor_{timestamp_str}.pkl"
    baseline_descriptor = {
        "type": "naive_by_lag",
        "steps": S,
        "note": "Predicción: paso k = y[t-k]. No requiere entrenamiento.",
    }
    with open(model_file, "wb") as f:
        pickle.dump(baseline_descriptor, f)
    print(f"Descriptor del baseline guardado en: {model_file}")

    # Guardar resultados individuales
    result_data = {
        "station": station,
        "model_type": model_name,
        "use_exog": use_exog,
        "step": S,
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
        "best_params": {},
        "optuna": {"best_value": None, "best_trial": None, "n_trials": 0},
        "model_file": str(model_file),
        "plot_files": plot_files,
        "timestamp": pd.Timestamp.now().isoformat(),
    }

    result_file = save_individual_result(
        result_data=result_data,
        results_dir=results_dir_station,
        regressor_name=model_name,
        timestamp_str=timestamp_str,
    )
    print(f"Resultados guardados en: {result_file}")

    all_results.append(
        {
            "regressor": model_name,
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
            "best_params": {},
            "model_file": str(model_file),
            "plot_files": plot_files,
        }
    )

    # -------------------------------- Resumen final -----------------------------
    timestamp_str = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    print_results_summary(
        all_results=all_results,
        station=station,
        use_exog=use_exog,
    )

    summary_file, csv_file = save_summary_and_comparison(
        all_results=all_results,
        station=station,
        use_exog=use_exog,
        summary_dir=summary_dir,
        timestamp_str=timestamp_str,
    )

    print(f"\nResumen completo guardado en: {summary_file}")
    print(f"Comparacion en CSV guardada en: {csv_file}")
    print(f"\n✅ Proceso completado para estacion {station}, step {S}")
    if horizon > 0:
        print(f"   Horizonte de shift: {horizon}")

    return {
        "station": station,
        "step": S,
        "horizon": horizon if horizon > 0 else None,
        "use_exog": use_exog,
        "results": all_results,
        "summary_file": str(summary_file),
        "csv_file": str(csv_file),
        "timestamp": timestamp_str,
    }


if __name__ == "__main__":
    STATION = "CEN-TRAF"
    USE_EXOG = True  # ignorado
    STEP = 72
    N_TRIALS = 0  # ignorado
    STUDY_STORAGE = None
    HORIZON = 0

    _ = train_and_evaluate_models(
        station=STATION,
        step=STEP,
        use_exog=USE_EXOG,
        n_trials=N_TRIALS,
        study_storage=STUDY_STORAGE,
        horizon=HORIZON,
    )
