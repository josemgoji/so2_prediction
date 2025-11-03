import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.pipelines.feature_selection_pipeline import FeatureSelector
from src.constants.parsed_fields import (
    FEATURE_SELECTION_CONFIG,
    FEATURE_SELECTION_RUN_CONFIG,
)

# === parámetros ===
STATIONS = ["GIR-EPM", "CEN-TRAF", "ITA-CJUS", "MED-FISC"]
# ====================================

# Crear selector con configuración desde parsed_fields.py
selector = FeatureSelector(**FEATURE_SELECTION_CONFIG)

# Ejecutar pipeline de selección para todas las estaciones
print("INICIANDO PIPELINE DE FEATURE SELECTION")
print("=" * 60)
print(f"Estaciones: {STATIONS}")
print(f"Selector: {FEATURE_SELECTION_CONFIG['selector_type']}")
print(f"Regresor: {FEATURE_SELECTION_CONFIG['regressor_type']}")
print(f"Lags: {FEATURE_SELECTION_CONFIG['lags']}")
print(f"Incluir exógenos: {FEATURE_SELECTION_RUN_CONFIG['include_exog']}")
print("=" * 60)

try:
    # Ejecutar pipeline completo
    results = selector.run_selection_pipeline(
        stations=STATIONS,
        **FEATURE_SELECTION_RUN_CONFIG,
    )

    # Mostrar resumen de resultados
    print("\n" + "=" * 60)
    print("RESUMEN DE RESULTADOS")
    print("=" * 60)

    for station, result in results.items():
        if "error" in result:
            print(f"\n[ERROR] {station}: ERROR")
            print(f"   {result['error']}")
        else:
            print(f"\n[OK] {station}: EXITOSO")
            print(f"   - Lags seleccionados: {result['n_selected_lags']}")
            print(
                f"   - Window features seleccionadas: {result['n_selected_window_features']}"
            )
            print(f"   - Variables exógenas seleccionadas: {result['n_selected_exog']}")
            print(f"   - Total de características: {result['total_features']}")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETADO")
    print("=" * 60)
    print(f"Resultados guardados en: {FEATURE_SELECTION_CONFIG['output_path']}")

except Exception as e:
    print(f"\n[ERROR] ERROR DURANTE EL PIPELINE: {str(e)}")
    import traceback

    traceback.print_exc()
