"""
Configuración centralizada para el proyecto de predicción de SO2.

Este módulo contiene todas las constantes, rutas y configuraciones por defecto
utilizadas en el pipeline de procesamiento de datos y modelado de series temporales.
"""

from pathlib import Path

# =============================================================================
# CONFIGURACIÓN DE RUTAS DEL PROYECTO
# =============================================================================

# Raíz del proyecto (asumiendo que este archivo vive en: src/constants/parsed_fields.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Directorio principal de datos
DATA_DIR = PROJECT_ROOT / "data"

# Directorio de datos procesados (stage)
STAGE_DIR = DATA_DIR / "stage"

# =============================================================================
# RUTAS DE ARCHIVOS DE DATOS
# =============================================================================

# Ruta principal del dataset de SO2 (usado en data_builder.py)
PATH: Path = DATA_DIR / "raw/Datos_SO2_2021_2024.csv"

# Ruta de guardado para datos limpios (usado en data_builder.py)
SAVE_PATH: Path = DATA_DIR / "stage/cleaned_data.csv"

# Rutas para datos exógenos (usado en data_builder.py)
SO2_PATH: Path = DATA_DIR / "raw/Datos_SO2_2021_2024.csv"
METEO_PATH: Path = DATA_DIR / "raw/Datos_Meteo_2021_2024.csv"

# =============================================================================
# CONFIGURACIÓN DE COLUMNAS DEL DATASET
# =============================================================================

# Columna de fecha/hora en los CSVs (usado en data_builder.py)
DATE_COLUMN: str = "datetime"

# Columna de temperatura del aire (usado en outlier_cleaner.py)
TAIRE_COL: str = "TAire10_SSR"

# =============================================================================
# CONFIGURACIÓN DE UBICACIÓN GEOGRÁFICA
# =============================================================================

# Configuración de ubicación para variables solares (usado en feature_engineering.py y feature_engineering_pipeline.py)
LOCATION_CONFIG = {
    "name": "Medellín",
    "region": "Colombia",
    "timezone": "America/Bogota",
    "latitude": 6.2442,
    "longitude": -75.5812,
}

# =============================================================================
# CONFIGURACIÓN DE CARACTERÍSTICAS DE CALENDARIO
# =============================================================================

# Características de calendario por defecto (usado en feature_engineering.py)
DEFAULT_CALENDAR_FEATURES = ["month", "week", "day_of_week", "hour"]

# =============================================================================
# CONFIGURACIÓN DE TRIMMING DE DATOS
# =============================================================================

# Configuración por defecto para trimming (usado en feature_engineering.py y feature_engineering_pipeline.py)
DEFAULT_TRIM_START = 7 * 24  # 168 horas (7 días)
DEFAULT_TRIM_END = 24  # 24 horas (1 día)

# =============================================================================
# CONFIGURACIÓN DE CARACTERÍSTICAS DE VENTANA
# =============================================================================

# Configuración por defecto para características de ventana (usado en windows_features.py)
DEFAULT_WINDOW_PERIOD = 24
DEFAULT_WINDOW_STATS = ["mean", "std", "min", "max"]
DEFAULT_WINDOW_SIZES = [3, 6, 12, 24, 48, 72]  # Para windows_features.py (sin "h")
DEFAULT_FOURIER_K = 3
DEFAULT_STL_ROBUST = True

# Configuración específica para características exógenas de temperatura (usado en feature_engineering.py)
DEFAULT_TEMP_WINDOWS = [
    "3h",
    "6h",
    "12h",
    "24h",
    "48h",
    "72h",
]  # Para rolling de temperatura
DEFAULT_TEMP_FUNCTIONS = ["mean", "std"]  # Para rolling de temperatura

# =============================================================================
# CONFIGURACIÓN DE FEATURE ENGINEERING PIPELINE
# =============================================================================

# Configuración completa del pipeline de feature engineering (usado en run_feature_engineering.py)
FEATURE_ENGINEERING_CONFIG = {
    "temp_columns": ["TAire10_SSR"],
    "calendar_features": DEFAULT_CALENDAR_FEATURES,
    "temp_windows": DEFAULT_TEMP_WINDOWS,
    "temp_functions": DEFAULT_TEMP_FUNCTIONS,
    "trim_start": DEFAULT_TRIM_START,
    "trim_end": DEFAULT_TRIM_END,
}

# =============================================================================
# CONFIGURACIÓN DE FEATURE SELECTION PIPELINE
# =============================================================================

# Configuración de window features para feature selection (usado en run_feature_selection.py)
DEFAULT_WINDOW_FEATURES_PARAMS = {
    "period": DEFAULT_WINDOW_PERIOD,
    "stats": DEFAULT_WINDOW_STATS,
    "window_sizes": DEFAULT_WINDOW_SIZES,
    "fourier_k": DEFAULT_FOURIER_K,
    "stl_robust": DEFAULT_STL_ROBUST,
}

# Configuración completa del pipeline de feature selection (usado en run_feature_selection.py)
FEATURE_SELECTION_CONFIG = {
    "data_path": str(STAGE_DIR / "SO2"),
    "output_path": str(STAGE_DIR / "SO2" / "selected"),
    "file_pattern": "processed_{station}.csv",
    "selector_type": "lasso",
    "regressor_type": "rf",
    "lags": 72,
    "window_features_params": DEFAULT_WINDOW_FEATURES_PARAMS,
    "selector_params": {},
    "regressor_params": {},
    "random_state": 15926,
    # Configuración de organización de archivos
    "organize_by_method": True,  # Organizar por método (lasso/rfecv)
    "organize_by_exog": True,  # Organizar por exógenos (con/sin)
}

# Configuración de ejecución del pipeline (usado en run_feature_selection.py)
FEATURE_SELECTION_RUN_CONFIG = {
    "include_exog": True,
    "select_only": None,
    "force_inclusion": None,
    "subsample": 0.5,
    "save_results": True,
}
