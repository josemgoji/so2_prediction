import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.pipelines.data_pipeline import DataPipeline
from src.constants.parsed_fields import STAGE_DIR

# === parámetros ===
STATIONS = ["CEN-TRAF"]
POLLUTANT = "SO2"
POLLUTANT_PATH = "data/raw/Datos_SO2_2021_2024.csv"
METEO_PATH = "data/raw/Datos_Meteo_2021_2024.csv"
# ====================================

for STATION in STATIONS:
    try:
        print(
            f"\nRunning preprocessing pipeline: station={STATION} pollutant={POLLUTANT}"
        )
        pipeline = DataPipeline(stage_dir=STAGE_DIR)
        df = pipeline.run(
            pollutant=POLLUTANT,
            station=STATION,
            pollutant_path=POLLUTANT_PATH,
            meteo_path=METEO_PATH,
        )
        print(f"Done: {STATION}")
    except Exception as e:
        print(f"Error en {STATION}: {e}")
