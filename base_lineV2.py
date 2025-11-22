# --- base_lineV2.py -----------------------------------------------------------
# Baseline "persistencia (last value)" con skforecast y compatibilidad de versiones.
# Para n steps, siempre predice el último valor conocido (recursivo).
# Mantiene tu estructura de guardados, plots y resúmenes.

import pickle
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

from skforecast.recursive import ForecasterEquivalentDate
from skforecast.model_selection import backtesting_forecaster

from src.recursos.data_manager import DataManager
from src.recursos.scorers import wmape, rmse, mae, mse, r2, stepwise_wmape_on_test
from src.utils.data_splitter import split_data_by_dates, apply_target_shift
from src.utils.plot_utils import create_prediction_plots
from src.utils.results_manager import (
    save_individual_result,
    save_summary_and_comparison,
    print_results_summary,
)
from src.constants.parsed_fields import MODEL_RESULTS_CONFIG


def _get_index_freq(idx: pd.DatetimeIndex):
    """Infiere un offset de frecuencia robusto a partir del índice temporal."""
    freq = idx.freq if idx.freq is not None else pd.infer_freq(idx)
    if freq is None:
        deltas = np.diff(idx.view("int64"))
        if len(deltas) == 0:
            raise ValueError("No se pudo inferir la frecuencia del índice.")
        step = pd.to_timedelta(pd.Series(deltas).mode().iloc[0])
        return step
    return pd.tseries.frequencies.to_offset(freq)


def _backtesting_skforecast_compat(
    forecaster,
    y: pd.Series,
    steps: int,
    initial_train_size: int,
    exog=None,
    refit: bool = False,
    fixed_train_size: bool = False,
    return_predict: bool = True,
):
    """
    Ejecuta backtesting_forecaster compatible con varias versiones de skforecast.
    1) Intenta firma moderna (kwargs con steps=...).
    2) Si falla, usa firma antigua por posicionales.
    3) Si también falla, hace backtesting manual (rolling origin) con el forecaster.
    Devuelve: (metric, preds_df) o (None, preds_df).
    """
    try:
        # Firma moderna (skforecast recientes)
        return backtesting_forecaster(
            forecaster=forecaster,
            y=y,
            exog=exog,
            steps=steps,
            metric=None,
            initial_train_size=initial_train_size,
            fixed_train_size=fixed_train_size,
            refit=refit,
            verbose=False,
            n_jobs=1,
            show_progress=False,
            return_predict=return_predict,
        )
    except TypeError:
        try:
            # Firma antigua (posicionales): forecaster, y, exog, steps, metric, initial_train_size, fixed_train_size, refit
            return backtesting_forecaster(
                forecaster,
                y,
                exog,
                steps,
                None,
                initial_train_size,
                fixed_train_size,
                refit,
            )
        except TypeError:
            # Fallback manual (origen rodante) con el forecaster
            y = y.sort_index()
            preds_list = []
            pos = initial_train_size
            while pos < len(y):
                train_end_pos = pos
                forecaster.fit(y=y.iloc[:train_end_pos], exog=None)
                steps_fold = min(steps, len(y) - pos)
                if steps_fold <= 0:
                    break
                pred = forecaster.predict(steps=steps_fold)
                preds_list.append(pred)
                pos += steps
            if preds_list:
                preds_df = pd.concat(preds_list).to_frame("pred").sort_index()
            else:
                preds_df = pd.DataFrame(columns=["pred"])
            return None, preds_df


def train_and_evaluate_models_persistence(
    station: str,
    step: int,
    use_exog: bool = True,  # compat; no se usan exógenas aquí
    n_trials: int = 0,  # compat
    study_storage: Optional[str] = None,  # compat
    val_months: int = 2,
    test_months: int = 2,
    horizon: int = 0,
) -> Dict[str, Any]:
    """
    Baseline 'persistencia' con skforecast.
    Para n steps, siempre predice el último valor conocido (en cada paso).
    """
    print(f"\n{'=' * 80}")
    print(f"Baseline PERSISTENCIA (skforecast) | Estación: {station} | Steps: {step}")
    if horizon > 0:
        print(f"Horizonte de shift (aplicado al target): {horizon}")
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
        exog_cols=[],  # baseline sin exógenas
        val_months=val_months,
        test_months=test_months,
    )

    print("Data split summary:")
    print(f"  Total data range: {df.index.min()} to {df.index.max()}")
    print(
        f"  Training:   {y_train.index.min()} to {y_train.index.max()} (n={len(y_train)})"
    )
    print(
        f"  Validation: {y_val.index.min()}   to {y_val.index.max()}   (n={len(y_val)})"
    )
    print(
        f"  Test:       {y_test.index.min()}  to {y_test.index.max()}  (n={len(y_test)})\n"
    )

    # ------------------------- Estructura de resultados -------------------------
    S = step
    results_base = (
        Path(MODEL_RESULTS_CONFIG["analytics_dir"])
        / MODEL_RESULTS_CONFIG["results_subdir"]
    )
    results_base.mkdir(parents=True, exist_ok=True)

    exog_status = "con_exog" if use_exog else "sin_exog"
    baseline_dir = "baseline_persistence"  # carpeta propia para este baseline

    station_results_dir = results_base / station / exog_status / baseline_dir / f"S{S}"
    if horizon > 0:
        station_results_dir = station_results_dir / f"H{horizon}"
    station_results_dir.mkdir(parents=True, exist_ok=True)

    models_dir = station_results_dir / "models"
    plots_dir = station_results_dir / "plots"
    results_dir_station = station_results_dir / "results"
    summary_dir = station_results_dir / "summary"
    for sub in (models_dir, plots_dir, results_dir_station, summary_dir):
        sub.mkdir(parents=True, exist_ok=True)

    model_name = "PersistenceLastValue"

    # ------------------ Forecaster skforecast: persistencia ---------------------
    off = _get_index_freq(df.index)  # p.ej., 1H si es horario
    forecaster = ForecasterEquivalentDate(offset=off, n_offsets=1)

    # ------------------------- Validación (train+val) ---------------------------
    print(f"\nBacktesting (train+val) con skforecast ({model_name})...")
    metric_val, preds_tv = _backtesting_skforecast_compat(
        forecaster=forecaster,
        y=y_trainval,
        steps=S,
        initial_train_size=len(y_train),
        exog=None,
        refit=False,
        fixed_train_size=False,
        return_predict=True,
    )
    preds_tv = preds_tv.rename(columns={"pred": "pred"}).sort_index()

    # Métricas validación
    y_tv_aligned = y_trainval.loc[preds_tv.index]
    preds_tv_aligned = preds_tv["pred"]
    rmse_tv = rmse(y_tv_aligned, preds_tv_aligned)
    mae_tv = mae(y_tv_aligned, preds_tv_aligned)
    mse_tv = mse(y_tv_aligned, preds_tv_aligned)
    r2_tv = r2(y_tv_aligned, preds_tv_aligned)
    wmape_tv = wmape(y_tv_aligned, preds_tv_aligned)
    stepwise_wmape_val = stepwise_wmape_on_test(y_tv_aligned, preds_tv_aligned, H=S)

    print(
        f"Validación {model_name}: RMSE={rmse_tv:.4f} | MAE={mae_tv:.4f} | "
        f"MSE={mse_tv:.4f} | R²={r2_tv:.4f} | WMAPE%={100 * wmape_tv:.2f}"
    )

    # --------------------------------- Test -------------------------------------
    print(f"\nBacktesting (test) con skforecast ({model_name})...")
    metric_test, preds_test = _backtesting_skforecast_compat(
        forecaster=forecaster,
        y=df[TARGET_COL],
        steps=S,
        initial_train_size=len(y_trainval),
        exog=None,
        refit=False,
        fixed_train_size=False,
        return_predict=True,
    )
    preds_test = preds_test.rename(columns={"pred": "pred"}).sort_index()

    # Alinear y calcular métricas de test
    common_idx = y_test.index.intersection(preds_test.index)
    if len(common_idx) > 0:
        y_test_aligned = y_test.loc[common_idx]
        y_pred_aligned = preds_test.loc[common_idx, "pred"]
    else:
        y_test_aligned = y_test
        y_pred_aligned = preds_test["pred"].reindex(y_test.index).dropna()

    test_rmse = rmse(y_test_aligned, y_pred_aligned)
    test_mae = mae(y_test_aligned, y_pred_aligned)
    test_mse = mse(y_test_aligned, y_pred_aligned)
    test_r2 = r2(y_test_aligned, y_pred_aligned)
    test_wmape = wmape(y_test_aligned, y_pred_aligned)
    stepwise_wmape_test = stepwise_wmape_on_test(y_test_aligned, y_pred_aligned, H=S)

    print(
        f"Test {model_name}: RMSE={test_rmse:.4f} | MAE={test_mae:.4f} | "
        f"MSE={test_mse:.4f} | R²={test_r2:.4f} | WMAPE%={100 * test_wmape:.2f}"
    )

    # ---------------------------- Plots + guardado ------------------------------
    try:
        common_val_idx = y_val.index.intersection(preds_tv.index)
        preds_val_plot = (
            preds_tv.loc[common_val_idx] if len(common_val_idx) > 0 else pd.DataFrame()
        )

        plot_files = create_prediction_plots(
            y_val=y_val,
            preds_val=preds_val_plot,
            y_test=y_test_aligned,
            y_pred_test=y_pred_aligned,
            model_name=model_name,
            station=station,
            save_dir=plots_dir,
        )
        print("Gráficos creados correctamente.")
    except Exception as e:
        print(f"ERROR creando gráficos: {e}")
        plot_files = {}

    # Guardar descriptor (no hay parámetros entrenables)
    timestamp_str = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    model_file = models_dir / f"{model_name}_descriptor_{timestamp_str}.pkl"
    descriptor = {
        "type": "persistence_last_value",
        "steps": S,
        "note": "Siempre predice el último valor observado; sin entrenamiento.",
    }
    with open(model_file, "wb") as f:
        pickle.dump(descriptor, f)

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
            "wmape": float(wmape_tv),
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

    # Resumen y CSV comparación
    all_results = [
        {
            "regressor": model_name,
            "val_rmse": rmse_tv,
            "val_mae": mae_tv,
            "val_mse": mse_tv,
            "val_r2": r2_tv,
            "val_wmape": wmape_tv,
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
    ]

    print_results_summary(all_results=all_results, station=station, use_exog=use_exog)

    timestamp_str2 = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    summary_file, csv_file = save_summary_and_comparison(
        all_results=all_results,
        station=station,
        use_exog=use_exog,
        summary_dir=summary_dir,
        timestamp_str=timestamp_str2,
    )

    print(f"\nResultados guardados en: {result_file}")
    print(f"Resumen completo: {summary_file}")
    print(f"CSV comparación:   {csv_file}")
    print(f"\n✅ Baseline PERSISTENCIA completado | Estación {station} | Steps {S}")
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
        "timestamp": timestamp_str2,
    }


if __name__ == "__main__":
    STATION = "CEN-TRAF"
    STEP = 72
    HORIZON = 0
    _ = train_and_evaluate_models_persistence(
        station=STATION,
        step=STEP,
        use_exog=True,
        n_trials=0,
        study_storage=None,
        horizon=HORIZON,
    )
