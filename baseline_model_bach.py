# --- baseline_batch.py --------------------------------------------------------
"""
Ejecución en batch del baseline naive-por-lag para múltiples estaciones y steps.
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Any

from baseline_model import train_and_evaluate_models


def run_batch_training(
    stations: List[str],
    steps: List[int],
    use_exog: bool = True,  # ignorado
    n_trials: int = 0,  # ignorado
    study_storage: Optional[str] = None,  # ignorado
    val_months: int = 2,
    test_months: int = 2,
    horizon: int = 0,
    batch_name: Optional[str] = None,
) -> Dict[str, Any]:
    all_batch_results: Dict[str, Any] = {}
    total_combinations = len(stations) * len(steps)
    current = 0

    print(f"\n{'=' * 80}")
    print("INICIANDO ENTRENAMIENTO EN BATCH (Baseline naive-por-lag)")
    print(f"{'=' * 80}")
    print(f"Estaciones: {stations}")
    print(f"Steps: {steps}")
    if horizon > 0:
        print(f"Horizonte de shift: {horizon}")
    print(f"Total de combinaciones: {total_combinations}")
    print(
        f"Configuración: {'Con exógenas' if use_exog else 'Sin exógenas'} (baseline ignora exógenas)"
    )
    print(f"{'=' * 80}\n")

    for station in stations:
        for step in steps:
            current += 1
            print(f"\n{'#' * 80}")
            print(f"PROCESANDO: {current}/{total_combinations}")
            print(
                f"Estación: {station} | Step: {step}{' | H=' + str(horizon) if horizon > 0 else ''}"
            )
            print(f"{'#' * 80}")

            key = f"{station}_S{step}" + (f"_H{horizon}" if horizon > 0 else "")

            try:
                result = train_and_evaluate_models(
                    station=station,
                    step=step,
                    use_exog=use_exog,
                    n_trials=n_trials,
                    study_storage=study_storage,
                    val_months=val_months,
                    test_months=test_months,
                    horizon=horizon,
                )
                all_batch_results[key] = {"success": True, "result": result}
                print(f"✅ Completado: {key}")

            except Exception as e:
                import traceback

                traceback.print_exc()
                all_batch_results[key] = {"success": False, "error": str(e)}
                print(f"❌ ERROR en {key}: {e}")

    # ---------------------------- Resúmenes del batch ---------------------------
    timestamp_str = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    exog_suffix = "con_exog" if use_exog else "sin_exog"
    base_dir = Path("data/analytics/model_results")
    batch_dirname = (
        f"batch_summaries_{batch_name}_{timestamp_str}"
        if batch_name
        else f"batch_summaries_{timestamp_str}"
    )
    batch_summary_dir = base_dir / batch_dirname
    batch_summary_dir.mkdir(parents=True, exist_ok=True)

    all_summaries: Dict[str, Any] = {}
    master_rows = []

    for station in stations:
        print(f"\n{'=' * 80}")
        print(f"GENERANDO RESÚMENES PARA ESTACIÓN: {station}")
        print(f"{'=' * 80}")

        station_models: Dict[str, list] = {}

        for step in steps:
            key = f"{station}_S{step}" + (f"_H{horizon}" if horizon > 0 else "")
            if not all_batch_results.get(key, {}).get("success"):
                continue

            result = all_batch_results[key]["result"]
            for model_result in result.get("results", []):
                model_name = model_result.get("regressor", "UnknownModel")
                station_models.setdefault(model_name, []).append(
                    {"step": step, **model_result}
                )

        station_summary_dir = batch_summary_dir / station / exog_suffix
        station_summary_dir.mkdir(parents=True, exist_ok=True)

        for model_name, model_results in station_models.items():
            rows = []
            for mr in sorted(model_results, key=lambda x: x["step"]):
                row = {
                    "station": station,
                    "model": model_name,
                    "step": mr.get("step"),
                    "val_rmse": mr.get("val_rmse", float("inf")),
                    "val_mae": mr.get("val_mae", float("inf")),
                    "val_mse": mr.get("val_mse", float("inf")),
                    "val_r2": mr.get("val_r2", float("-inf")),
                    "val_wmape": mr.get("val_wmape", float("inf")),
                    "test_rmse": mr.get("test_rmse", float("inf")),
                    "test_mae": mr.get("test_mae", float("inf")),
                    "test_mse": mr.get("test_mse", float("inf")),
                    "test_r2": mr.get("test_r2", float("-inf")),
                    "test_wmape": mr.get("test_wmape", float("inf")),
                    "model_file": mr.get("model_file", ""),
                }
                rows.append(row)
                master_rows.append(row)

            df_model = pd.DataFrame(rows).sort_values("step").reset_index(drop=True)
            csv_filename = station_summary_dir / f"{model_name}_all_steps.csv"
            df_model.to_csv(csv_filename, index=False)

            print(f"\n{'─' * 80}")
            print(f"📊 MODELO: {model_name} | ESTACIÓN: {station}")
            print(f"📁 CSV: {csv_filename}")
            print(f"🔢 Steps: {sorted([r['step'] for r in model_results])}")
            print(
                df_model[
                    [
                        "step",
                        "val_rmse",
                        "val_mae",
                        "val_mse",
                        "val_r2",
                        "val_wmape",
                        "test_rmse",
                        "test_mae",
                        "test_mse",
                        "test_r2",
                        "test_wmape",
                    ]
                ].to_string(index=False)
            )
            print(f"{'─' * 80}")

            all_summaries[f"{station}_{model_name}"] = {
                "station": station,
                "model": model_name,
                "csv_file": str(csv_filename),
                "steps": sorted([r["step"] for r in model_results]),
            }

    # Master CSV
    master_csv = None
    if master_rows:
        df_master = pd.DataFrame(master_rows)
        master_csv = batch_summary_dir / "batch_master_results.csv"
        df_master.sort_values(["station", "model", "step"]).to_csv(
            master_csv, index=False
        )
        print(f"\n📚 Master CSV de batch: {master_csv}")
    else:
        print("\n⚠️ No se generó master CSV porque no hubo resultados exitosos.")

    summary = {
        "total_combinations": len(stations) * len(steps),
        "successful": sum(1 for r in all_batch_results.values() if r.get("success")),
        "failed": sum(1 for r in all_batch_results.values() if not r.get("success")),
    }

    batch_summary = {
        "configuration": {
            "stations": stations,
            "steps": steps,
            "horizon": horizon if horizon > 0 else None,
            "use_exog": use_exog,
            "n_trials": n_trials,
            "val_months": val_months,
            "test_months": test_months,
            "timestamp": pd.Timestamp.now().isoformat(),
            "batch_name": batch_name,
        },
        "results": all_batch_results,
        "summaries_by_model": all_summaries,
        "summary": summary,
        "paths": {
            "batch_dir": str(batch_summary_dir),
            "master_csv": str(master_csv) if master_csv else None,
        },
    }

    summary_file = batch_summary_dir / f"batch_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(batch_summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 80}")
    print("RESUMEN FINAL DEL BATCH")
    print(f"{'=' * 80}")
    print(f"Total procesado: {summary['total_combinations']}")
    print(f"Exitosos: {summary['successful']}")
    print(f"Fallidos: {summary['failed']}")
    print(f"📁 Carpeta del batch: {batch_summary_dir}")
    print(f"📄 Resumen JSON: {summary_file}")
    if master_csv:
        print(f"📚 Master CSV: {master_csv}")
    print(f"{'=' * 80}\n")

    return batch_summary


if __name__ == "__main__":
    STATIONS = [
        #"CEN-TRAF",
        "GIR-EPM",
        "ITA-CJUS",
        "MED-FISC",
    ]
    STEPS = list(range(1, 73))  # 1..72
    USE_EXOG = True
    N_TRIALS = 0
    STUDY_STORAGE = None
    VAL_MONTHS = 2
    TEST_MONTHS = 2
    HORIZON = 0
    BATCH_NAME = "naive_by_lag"

    _ = run_batch_training(
        stations=STATIONS,
        steps=STEPS,
        use_exog=USE_EXOG,
        n_trials=N_TRIALS,
        study_storage=STUDY_STORAGE,
        val_months=VAL_MONTHS,
        test_months=TEST_MONTHS,
        horizon=HORIZON,
        batch_name=BATCH_NAME,
    )

