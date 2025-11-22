"""
Script para consolidar métricas de test y validación de todos los steps y modelos en CSV.

Recorre cada step en data/analytics/model_results/CEN-TRAF/con_exog/
y extrae las métricas de test y validación de cada modelo, guardándolas en CSVs separados.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any


def extract_test_metrics_from_json(json_file: Path) -> Dict[str, Any]:
    """
    Extrae las métricas de test de un archivo JSON de resultados.

    Parameters:
    -----------
    json_file : Path
        Ruta al archivo JSON

    Returns:
    --------
    dict
        Diccionario con step, model_type y métricas de test
    """
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    test_metrics = data.get("test_metrics", {})
    step = data.get("steps") or data.get(
        "step"
    )  # Intentar "steps" primero, luego "step" como fallback
    model_type = data.get("model_type")

    return {
        "step": step,
        "model": model_type,
        "rmse": test_metrics.get("rmse"),
        "mae": test_metrics.get("mae"),
        "mse": test_metrics.get("mse"),
        "r2": test_metrics.get("r2"),
        "wmape": test_metrics.get("wmape"),
        "stepwise_wmape": json.dumps(
            test_metrics.get("stepwise_wmape", {})
        ),  # Convertir a string JSON
    }


def extract_validation_metrics_from_json(json_file: Path) -> Dict[str, Any]:
    """
    Extrae las métricas de validación de un archivo JSON de resultados.

    Parameters:
    -----------
    json_file : Path
        Ruta al archivo JSON

    Returns:
    --------
    dict
        Diccionario con step, model_type y métricas de validación
    """
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    validation_metrics = data.get("validation_metrics", {})
    step = data.get("steps") or data.get(
        "step"
    )  # Intentar "steps" primero, luego "step" como fallback
    model_type = data.get("model_type")

    return {
        "step": step,
        "model": model_type,
        "rmse": validation_metrics.get("rmse"),
        "mae": validation_metrics.get("mae"),
        "mse": validation_metrics.get("mse"),
        "r2": validation_metrics.get("r2"),
        "wmape": validation_metrics.get("wmape"),
        "stepwise_wmape": json.dumps(
            validation_metrics.get("stepwise_wmape", {})
        ),  # Convertir a string JSON
    }


def consolidate_metrics(
    base_dir: Path,
    metric_type: str = "test",  # "test" o "validation"
) -> pd.DataFrame:
    """
    Consolida todas las métricas (test o validación) de todos los steps y modelos.

    Parameters:
    -----------
    base_dir : Path
        Directorio base donde están las carpetas S*
    metric_type : str
        Tipo de métricas a consolidar: "test" o "validation"

    Returns:
    --------
    pd.DataFrame
        DataFrame con todas las métricas consolidadas
    """
    all_results = []

    # Encontrar todas las carpetas S*
    step_dirs = sorted(
        [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("S")],
        key=lambda x: int(x.name[1:]) if x.name[1:].isdigit() else 999999,
    )

    print(f"Encontradas {len(step_dirs)} carpetas de steps")

    # Seleccionar función de extracción según el tipo
    extract_func = (
        extract_test_metrics_from_json
        if metric_type == "test"
        else extract_validation_metrics_from_json
    )

    for step_dir in step_dirs:
        step_name = step_dir.name

        results_dir = step_dir / "results"

        if not results_dir.exists():
            print(f"  ⚠️  {step_name}: No existe carpeta results/")
            continue

        # Buscar todos los archivos JSON
        json_files = list(results_dir.glob("*.json"))

        if len(json_files) == 0:
            print(f"  ⚠️  {step_name}: No se encontraron archivos JSON")
            continue

        print(f"  ✓ {step_name}: {len(json_files)} archivos JSON")

        for json_file in json_files:
            try:
                metrics = extract_func(json_file)
                all_results.append(metrics)
            except Exception as e:
                print(f"    ❌ Error leyendo {json_file.name}: {str(e)}")

    # Crear DataFrame
    if len(all_results) == 0:
        print(f"\n⚠️  No se encontraron resultados de {metric_type} para consolidar")
        return pd.DataFrame()

    df = pd.DataFrame(all_results)

    # Ordenar por step y modelo
    df = df.sort_values(["step", "model"]).reset_index(drop=True)

    return df


def main():
    """Función principal"""
    print("=" * 80)
    print("CONSOLIDACIÓN DE MÉTRICAS DE TEST Y VALIDACIÓN")
    print("=" * 80)

    base_dir = Path("data/analytics/model_results/MED-FISC/con_exog/")

    if not base_dir.exists():
        print(f"❌ Error: No existe el directorio {base_dir}")
        return

    print(f"\n📂 Directorio base: {base_dir}")

    # =============================================================================
    # CONSOLIDAR MÉTRICAS DE TEST
    # =============================================================================
    print("\n" + "=" * 80)
    print("CONSOLIDANDO MÉTRICAS DE TEST...")
    print("=" * 80)

    df_test = consolidate_metrics(base_dir, metric_type="test")

    if not df_test.empty:
        output_file_test = base_dir / "test_metrics_all_steps_direct.csv"
        df_test.to_csv(output_file_test, index=False)

        print("\n" + "=" * 80)
        print(f"✅ CSV de TEST guardado en: {output_file_test}")
        print(f"📊 Total de registros: {len(df_test)}")
        print(f"📈 Steps encontrados: {df_test['step'].nunique()}")
        print(f"🤖 Modelos encontrados: {df_test['model'].unique().tolist()}")
        print("=" * 80)

        # Mostrar muestra de datos
        print("\n📋 Muestra de datos de TEST (primeros 5 registros):")
        print(df_test.head().to_string(index=False))
    else:
        print("\n⚠️  No se pudieron consolidar métricas de TEST")

    # =============================================================================
    # CONSOLIDAR MÉTRICAS DE VALIDACIÓN
    # =============================================================================
    print("\n\n" + "=" * 80)
    print("CONSOLIDANDO MÉTRICAS DE VALIDACIÓN...")
    print("=" * 80)

    df_validation = consolidate_metrics(base_dir, metric_type="validation")

    if not df_validation.empty:
        output_file_validation = base_dir / "validation_metrics_all_steps_direct.csv"
        df_validation.to_csv(output_file_validation, index=False)

        print("\n" + "=" * 80)
        print(f"✅ CSV de VALIDACIÓN guardado en: {output_file_validation}")
        print(f"📊 Total de registros: {len(df_validation)}")
        print(f"📈 Steps encontrados: {df_validation['step'].nunique()}")
        print(f"🤖 Modelos encontrados: {df_validation['model'].unique().tolist()}")
        print("=" * 80)

        # Mostrar muestra de datos
        print("\n📋 Muestra de datos de VALIDACIÓN (primeros 5 registros):")
        print(df_validation.head().to_string(index=False))
    else:
        print("\n⚠️  No se pudieron consolidar métricas de VALIDACIÓN")

    # =============================================================================
    # RESUMEN FINAL
    # =============================================================================
    print("\n\n" + "=" * 80)
    print("RESUMEN FINAL")
    print("=" * 80)

    if not df_test.empty:
        print(f"✅ CSV de TEST: {base_dir / 'test_metrics_all_steps_direct.csv'}")
    if not df_validation.empty:
        print(
            f"✅ CSV de VALIDACIÓN: {base_dir / 'validation_metrics_all_steps_direct.csv'}"
        )

    print("=" * 80)


if __name__ == "__main__":
    main()
