import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.pipelines.feature_engineering_pipeline import FeatureEngineeringPipeline
from src.constants.parsed_fields import STAGE_DIR, FEATURE_ENGINEERING_CONFIG

# === parámetros ===
STATIONS = ["CEN-TRAF", "GIR-EPM", "ITA-CJUS", "MED-FISC"]
POLLUTANT = "SO2"
# ==================

for STATION in STATIONS:
    try:
        print(
            f"\nRunning feature engineering pipeline: station={STATION} pollutant={POLLUTANT}"
        )

        input_file = STAGE_DIR / "SO2" / "clean" / f"{STATION}.csv"
        pipeline = FeatureEngineeringPipeline()

        results = pipeline.run_complete_pipeline(
            file_path=str(input_file),
            **FEATURE_ENGINEERING_CONFIG,
        )

        output_file = STAGE_DIR / "SO2" / "processed" / f"processed_{STATION}.csv"
        pipeline.save_results(str(output_file))

        summary = pipeline.get_summary()
        print(f"\nResumen del pipeline para {STATION}:")
        print(f"  - Datos originales: {summary['data_info']['raw_shape']}")
        print(f"  - Datos procesados: {summary['data_info']['processed_shape']}")
        print(
            f"  - Nº características creadas: {len(summary['features_info']['feature_engineering_columns'])}"
        )

        # Chequeo rápido de gaps en el index:
        df = results["processed_data"]
        print("\nDeltas de índice más frecuentes (para verificar gaps):")
        print(df.index.to_series().diff().value_counts().head())

        print(f"Done: {STATION}")

    except Exception as e:
        print(f"Error en {STATION}: {e}")
        import traceback

        traceback.print_exc()