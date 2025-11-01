"""
Utilidades para guardar y gestionar resultados de modelos
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from src.constants.parsed_fields import MODEL_RESULTS_CONFIG


def clean_params_for_json(params_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Convierte parámetros a tipos serializables en JSON"""
    import numpy as np
    
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


def save_individual_result(
    result_data: Dict[str, Any],
    results_dir: Path,
    regressor_name: str,
    timestamp_str: str,
) -> Path:
    """
    Guarda los resultados individuales de un modelo en JSON
    
    Parameters:
    -----------
    result_data : dict
        Diccionario con todos los resultados del modelo
    results_dir : Path
        Directorio donde guardar el resultado
    regressor_name : str
        Nombre del regresor
    timestamp_str : str
        Timestamp para el nombre del archivo
        
    Returns:
    --------
    Path
        Ruta del archivo guardado
    """
    result_file = results_dir / f"{regressor_name}_{timestamp_str}.json"

    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)

    return result_file


def save_summary_and_comparison(
    all_results: List[Dict[str, Any]],
    station: str,
    use_exog: bool,
    summary_dir: Path,
    timestamp_str: str,
) -> tuple[Path, Path]:
    """
    Guarda el resumen completo y la comparación en CSV
    
    Parameters:
    -----------
    all_results : list
        Lista de diccionarios con resultados de todos los modelos
    station : str
        Nombre de la estación
    use_exog : bool
        Si se usaron variables exógenas
    summary_dir : Path
        Directorio donde guardar los archivos
    timestamp_str : str
        Timestamp para los nombres de archivo
        
    Returns:
    --------
    tuple[Path, Path]
        Tupla con (ruta del summary JSON, ruta del CSV de comparación)
    """
    import numpy as np
    
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values("test_wmape")

    # Crear resumen completo
    summary_data = {
        "station": station,
        "configuration": {
            "use_exog": use_exog,
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

    summary_file = summary_dir / f"summary_{timestamp_str}.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    # Guardar CSV de comparación
    csv_file = summary_dir / f"results_comparison_{timestamp_str}.csv"
    results_df.to_csv(csv_file, index=False)

    return summary_file, csv_file


def print_results_summary(
    all_results: List[Dict[str, Any]],
    station: str,
    use_exog: bool,
) -> None:
    """
    Imprime un resumen de los resultados en consola
    
    Parameters:
    -----------
    all_results : list
        Lista de diccionarios con resultados de todos los modelos
    station : str
        Nombre de la estación
    use_exog : bool
        Si se usaron variables exógenas
    """
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values("test_wmape")

    print(f"\n{'=' * 80}")
    print(f"RESUMEN DE RESULTADOS PARA ESTACION: {station}")
    print(
        f"Configuracion: {'Con exogenas' if use_exog else 'Sin exogenas'}"
    )
    print(f"{'=' * 80}")

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

