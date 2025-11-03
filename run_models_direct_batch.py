"""
Script para ejecutar entrenamiento de modelos Direct para múltiples estaciones y horizontes.

Este script permite ejecutar el entrenamiento de modelos Direct en batch,
procesando múltiples combinaciones de estaciones y horizontes de forma secuencial.
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Optional

from models_direct import train_and_evaluate_models


def run_batch_training(
    stations: List[str],
    steps_list: List[int],
    use_exog: bool = True,
    n_trials: int = 20,
    study_storage: Optional[str] = None,
    val_months: int = 2,
    test_months: int = 2,
    horizon: int = 0,
) -> dict:
    """
    Ejecuta el entrenamiento de modelos Direct para múltiples estaciones y steps.

    Parameters:
    -----------
    stations : list[str]
        Lista de nombres de estaciones (ej: ["CEN-TRAF", "GIR-EPM"])
    steps_list : list[int]
        Lista de pasos directos (ej: [24, 48, 72]) - número de pasos hacia adelante que se predicen simultáneamente
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
        Horizonte de shift del target (0 = no shift, >0 = shift hacia adelante)

    Returns:
    --------
    dict
        Diccionario con los resultados de todos los entrenamientos
    """
    all_batch_results = {}

    total_combinations = len(stations) * len(steps_list)
    current = 0

    print(f"\n{'=' * 80}")
    print("INICIANDO ENTRENAMIENTO EN BATCH (Direct)")
    print(f"{'=' * 80}")
    print(f"Estaciones: {stations}")
    print(f"Steps: {steps_list}")
    if horizon > 0:
        print(f"Horizonte de shift: {horizon}")
    print(f"Total de combinaciones: {total_combinations}")
    print(f"Configuración: {'Con exógenas' if use_exog else 'Sin exógenas'}")
    print(f"{'=' * 80}\n")

    for station in stations:
        for steps in steps_list:
            current += 1
            print(f"\n{'#' * 80}")
            print(f"PROCESANDO: {current}/{total_combinations}")
            print(f"Estación: {station}, Steps Direct: {steps}")
            if horizon > 0:
                print(f"Horizonte de shift: {horizon}")
            print(f"{'#' * 80}\n")

            try:
                result = train_and_evaluate_models(
                    station=station,
                    steps=steps,
                    use_exog=use_exog,
                    n_trials=n_trials,
                    study_storage=study_storage,
                    val_months=val_months,
                    test_months=test_months,
                    horizon=horizon,
                )

                key = f"{station}_S{steps}"
                if horizon > 0:
                    key = f"{key}_H{horizon}"
                all_batch_results[key] = {
                    "success": True,
                    "result": result,
                }

                print(
                    f"\n✅ Completado: {station} - S{steps}"
                    + (f" - H{horizon}" if horizon > 0 else "")
                )

            except Exception as e:
                print(
                    f"\n❌ ERROR en {station} - S{steps}"
                    + (f" - H{horizon}" if horizon > 0 else "")
                    + f": {str(e)}"
                )
                import traceback

                traceback.print_exc()

                key = f"{station}_S{steps}"
                if horizon > 0:
                    key = f"{key}_H{horizon}"
                all_batch_results[key] = {
                    "success": False,
                    "error": str(e),
                }

    # Organizar resultados por estación y modelo
    timestamp_str = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    exog_suffix = "con_exog" if use_exog else "sin_exog"

    # Directorio para guardar los resúmenes
    batch_summary_dir = Path(
        f"data/analytics/model_results/batch_summaries_direct_{timestamp_str}"
    )
    batch_summary_dir.mkdir(parents=True, exist_ok=True)

    # Procesar por estación
    all_summaries = {}

    for station in stations:
        print(f"\n{'=' * 80}")
        print(f"GENERANDO RESUMENES PARA ESTACIÓN: {station}")
        print(f"{'=' * 80}")

        # Recolectar todos los modelos únicos de esta estación
        station_models = {}

        for steps in steps_list:
            key = f"{station}_S{steps}"
            if horizon > 0:
                key = f"{key}_H{horizon}"
            if key in all_batch_results and all_batch_results[key].get(
                "success", False
            ):
                result = all_batch_results[key]["result"]

                # Para cada modelo en los resultados de este step
                for model_result in result.get("results", []):
                    model_name = model_result.get("regressor")

                    if model_name not in station_models:
                        station_models[model_name] = []

                    # Agregar resultado de este step
                    station_models[model_name].append({"steps": steps, **model_result})

        # Crear CSV por cada modelo para esta estación
        station_summary_dir = batch_summary_dir / station / exog_suffix
        station_summary_dir.mkdir(parents=True, exist_ok=True)

        for model_name, model_results in station_models.items():
            # Crear DataFrame con todos los horizontes de este modelo
            rows = []

            for model_result in sorted(model_results, key=lambda x: x["steps"]):
                row = {
                    "steps": model_result["steps"],
                    "val_rmse": model_result.get("val_rmse", float("inf")),
                    "val_mae": model_result.get("val_mae", float("inf")),
                    "val_mse": model_result.get("val_mse", float("inf")),
                    "val_r2": model_result.get("val_r2", float("-inf")),
                    "val_wmape": model_result.get("val_wmape", float("inf")),
                    "val_stepwise_wmape": str(
                        model_result.get("val_stepwise_wmape", {})
                    ),
                    "test_rmse": model_result.get("test_rmse", float("inf")),
                    "test_mae": model_result.get("test_mae", float("inf")),
                    "test_mse": model_result.get("test_mse", float("inf")),
                    "test_r2": model_result.get("test_r2", float("-inf")),
                    "test_wmape": model_result.get("test_wmape", float("inf")),
                    "test_stepwise_wmape": str(
                        model_result.get("test_stepwise_wmape", {})
                    ),
                    "model_file": model_result.get("model_file", ""),
                }

                rows.append(row)

            # Crear DataFrame y guardar CSV
            df_model = pd.DataFrame(rows)

            # Ordenar por steps
            df_model = df_model.sort_values("steps").reset_index(drop=True)

            csv_filename = station_summary_dir / f"{model_name}_all_steps.csv"
            df_model.to_csv(csv_filename, index=False)

            print(f"\n{'─' * 80}")
            print(f"📊 MODELO: {model_name} | ESTACIÓN: {station}")
            print(f"{'─' * 80}")
            print(f"📁 Archivo CSV: {csv_filename}")
            print(f"🔢 Steps incluidos: {sorted([r['steps'] for r in model_results])}")
            if horizon > 0:
                print(f"   Horizonte de shift: {horizon}")
            print(f"\n📋 TABLA DE RESULTADOS POR STEP:")
            print(f"{'─' * 80}")

            # Seleccionar columnas principales para mostrar
            display_cols = ["steps"]
            if "val_rmse" in df_model.columns:
                display_cols.append("val_rmse")
            if "val_mae" in df_model.columns:
                display_cols.append("val_mae")
            if "val_mse" in df_model.columns:
                display_cols.append("val_mse")
            if "val_r2" in df_model.columns:
                display_cols.append("val_r2")
            if "val_wmape" in df_model.columns:
                display_cols.append("val_wmape")
            if "test_rmse" in df_model.columns:
                display_cols.append("test_rmse")
            if "test_mae" in df_model.columns:
                display_cols.append("test_mae")
            if "test_mse" in df_model.columns:
                display_cols.append("test_mse")
            if "test_r2" in df_model.columns:
                display_cols.append("test_r2")
            if "test_wmape" in df_model.columns:
                display_cols.append("test_wmape")

            # Formatear valores para mejor visualización
            df_display = df_model[display_cols].copy()

            # Formatear WMAPE como porcentaje
            for col in ["val_wmape", "test_wmape"]:
                if col in df_display.columns:
                    df_display[col] = df_display[col].apply(
                        lambda x: f"{100 * x:.2f}%"
                        if isinstance(x, (int, float)) and x != float("inf")
                        else "N/A"
                    )

            # Formatear RMSE, MAE, MSE con 4 decimales
            for col in [
                "val_rmse",
                "test_rmse",
                "val_mae",
                "test_mae",
                "val_mse",
                "test_mse",
            ]:
                if col in df_display.columns:
                    df_display[col] = df_display[col].apply(
                        lambda x: f"{x:.4f}"
                        if isinstance(x, (int, float)) and x != float("inf")
                        else "N/A"
                    )

            # Formatear R² con 4 decimales
            for col in ["val_r2", "test_r2"]:
                if col in df_display.columns:
                    df_display[col] = df_display[col].apply(
                        lambda x: f"{x:.4f}"
                        if isinstance(x, (int, float)) and x != float("-inf")
                        else "N/A"
                    )

            print(df_display.to_string(index=False))
            print(f"{'─' * 80}")

            all_summaries[f"{station}_{model_name}"] = {
                "station": station,
                "model": model_name,
                "csv_file": str(csv_filename),
                "steps": sorted([r["steps"] for r in model_results]),
                "horizon": horizon if horizon > 0 else None,
            }

    # Guardar resumen general del batch en JSON
    batch_summary = {
        "configuration": {
            "stations": stations,
            "steps": steps_list,
            "horizon": horizon,
            "use_exog": use_exog,
            "n_trials": n_trials,
            "val_months": val_months,
            "test_months": test_months,
            "timestamp": pd.Timestamp.now().isoformat(),
        },
        "results": all_batch_results,
        "summaries_by_model": all_summaries,
        "summary": {
            "total_combinations": total_combinations,
            "successful": sum(
                1 for r in all_batch_results.values() if r.get("success", False)
            ),
            "failed": sum(
                1 for r in all_batch_results.values() if not r.get("success", False)
            ),
        },
    }

    summary_file = batch_summary_dir / f"batch_summary_{timestamp_str}.json"

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(batch_summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 80}")
    print(f"RESUMEN FINAL DEL BATCH (Direct)")
    print(f"{'=' * 80}")
    print(f"Total procesado: {total_combinations}")
    print(f"Exitosos: {batch_summary['summary']['successful']}")
    print(f"Fallidos: {batch_summary['summary']['failed']}")
    print(f"\n📁 Resúmenes guardados en: {batch_summary_dir}")
    print(f"📄 Resumen general JSON: {summary_file}")
    print(f"\n📊 CSVs generados por modelo:")
    for key, summary_info in all_summaries.items():
        print(
            f"   - {summary_info['station']} - {summary_info['model']}: {summary_info['csv_file']}"
        )
    print(f"{'=' * 80}\n")

    return batch_summary


if __name__ == "__main__":
    # =============================================================================
    # CONFIGURACIÓN DEL BATCH
    # =============================================================================
    # Lista de estaciones a procesar
    STATIONS = [
        "CEN-TRAF",
        # "GIR-EPM",
        # "ITA-CJUS",
        # "MED-FISC",
    ]

    # Lista de steps a procesar (en horas) - número de pasos que se predicen simultáneamente
    STEPS = list(range(37, 73)) # cambiar 48-61

    # Configuración general
    USE_EXOG = True  # True para modelo con exógenas, False para sin exógenas
    N_TRIALS = 1  # Número de trials para Optuna
    STUDY_STORAGE = None  # ej: "sqlite:///optuna.db" si quieres persistir estudios
    VAL_MONTHS = 2  # Meses para validación
    TEST_MONTHS = 2  # Meses para test
    HORIZON = (
        0  # Horizonte para shift del target (0 = no shift, >0 = shift hacia adelante)
    )

    # Ejecutar batch
    batch_results = run_batch_training(
        stations=STATIONS,
        steps_list=STEPS,
        use_exog=USE_EXOG,
        n_trials=N_TRIALS,
        study_storage=STUDY_STORAGE,
        val_months=VAL_MONTHS,
        test_months=TEST_MONTHS,
        horizon=HORIZON,
    )
