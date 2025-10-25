from pathlib import Path

# Raíz del proyecto (asumiendo que este archivo vive en: src/constants/parsed_fields.py).
# Si tu estructura difiere, ajusta el número de "parents".
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Directorio de datos y ruta al CSV
DATA_DIR = PROJECT_ROOT / "data"
PATH: Path = DATA_DIR / "raw/Datos_SO2_2021_2024.csv"  # <-- cambia el nombre si aplica
SAVE_PATH: Path = DATA_DIR / "stage/cleaned_data.csv"  # donde guardar el CSV limpio

# Paths for exogenous data
CALIDAD_SO2_PATH: Path = DATA_DIR / "raw/Calidad_SO2_2021_2024.csv"
SO2_PATH: Path = DATA_DIR / "raw/Datos_SO2_2021_2024.csv"
METEO_PATH: Path = DATA_DIR / "raw/Datos_Meteo_2021_2024.csv"
STAGE_DIR: Path = DATA_DIR / "stage"
# Campos del dataset
TARGET: str = "calidad_CEN-TRAF"  # variable objetivo
DATE_COLUMN: str = "datetime"  # nombre de la columna de fecha/hora en el CSV

# Variables para limpieza y análisis (según test.ipynb)
SO2_COL: str = "SO2"
TAIRE_COL: str = "TAire10_SSR"

# Configuración de ubicación para variables solares
LOCATION_CONFIG = {
    "name": "Medellín",
    "region": "Colombia",
    "timezone": "America/Bogota",
    "latitude": 6.2442,
    "longitude": -75.5812,
}

# Configuración por defecto para características de calendario
DEFAULT_CALENDAR_FEATURES = ["month", "week", "day_of_week", "hour"]

# Configuración por defecto para características de temperatura
DEFAULT_TEMP_COLUMNS = ["TAire10_SSR"]
DEFAULT_TEMP_WINDOWS = ["1D", "7D"]
DEFAULT_TEMP_FUNCTIONS = ["mean", "max", "min"]

# Configuración por defecto para trimming
DEFAULT_TRIM_START = 7 * 24  # 168 horas (7 días)
DEFAULT_TRIM_END = 24  # 24 horas (1 día)

# Configuración de características de ventana para variables exógenas
EXOGENOUS_WINDOW_FEATURES = ["mean", "std", "min", "max", "median", "var"]

# Configuración por defecto para características de ventana (WindowFeaturesGenerator)
DEFAULT_WINDOW_PERIOD = 24
DEFAULT_WINDOW_STATS = ["mean", "std", "min", "max"]
DEFAULT_WINDOW_SIZES = [3, 6, 12, 24, 48, 72]
DEFAULT_FOURIER_K = 3
DEFAULT_STL_ROBUST = True
