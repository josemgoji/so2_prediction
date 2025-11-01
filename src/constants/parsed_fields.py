"""
Configuración centralizada para el proyecto de predicción de SO2.

Este módulo contiene todas las constantes, rutas y configuraciones por defecto
utilizadas en el pipeline de procesamiento de datos y modelado de series temporales.
"""

from pathlib import Path
import numpy as np

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

# Configuración específica para características exógenas de ventana (usado en feature_engineering.py)
DEFAULT_WINDOW_WINDOWS = [
    "3h",
    "6h",
    "12h",
    "24h",
    "48h",
    "72h",
]  # Para rolling de variables exógenas
DEFAULT_WINDOW_FUNCTIONS = ["mean", "std"]  # Para rolling de variables exógenas

# =============================================================================
# CONFIGURACIÓN DE FEATURE ENGINEERING PIPELINE
# =============================================================================

# Configuración completa del pipeline de feature engineering (usado en run_feature_engineering.py)
FEATURE_ENGINEERING_CONFIG = {
    "window_columns": ["TAire10_SSR"],
    "calendar_features": DEFAULT_CALENDAR_FEATURES,
    "window_windows": DEFAULT_WINDOW_WINDOWS,
    "window_functions": DEFAULT_WINDOW_FUNCTIONS,
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
}

# Configuración completa del pipeline de feature selection (usado en run_feature_selection.py)
FEATURE_SELECTION_CONFIG = {
    "data_path": str(STAGE_DIR / "SO2"),
    "output_path": str(STAGE_DIR / "SO2" / "selected"),
    "file_pattern": "processed_{station}.csv",
    "selector_type": "lasso",
    "regressor_type": "lasso",
    "lags": list(range(1, 73)) + [168, 672],
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

# =============================================================================
# CONFIGURACIÓN DE REGRESORES Y PARÁMETROS
# =============================================================================

# Configuración de regresores para modelado
REGRESSORS_CONFIG = [
    {
        "name": "LGBM",
        "regressor_func": "create_lgbm_regressor",
        "params": {
            "n_estimators": [300, 600, 1000, 1500],
            "learning_rate": [0.10, 0.05, 0.03, 0.02],
            "num_leaves": [31, 63, 127, 255],
            "max_depth": [-1, 10, 15, 20],
            "min_child_samples": [10, 20, 50, 100],
            "feature_fraction": [0.6, 0.8, 1.0],  # (= colsample_bytree)
            "bagging_fraction": [0.6, 0.8, 1.0],  # (= subsample)
            "bagging_freq": [0, 1, 5],
            "lambda_l1": [0.0, 0.1, 1.0, 5.0],
            "lambda_l2": [0.0, 0.1, 1.0, 5.0],
            "min_split_gain": [0.0, 0.1, 0.3],
            "extra_trees": [True, False],
        },
    },
    {
        "name": "XGBoost",
        "regressor_func": "create_xgb_regressor",
        "params": {
            "n_estimators": [300, 600, 1000, 1500],
            "learning_rate": [0.10, 0.05, 0.03, 0.02],
            "max_depth": [3, 5, 7, 10],
            "min_child_weight": [1, 3, 5, 10],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "colsample_bylevel": [0.6, 0.8, 1.0],
            "gamma": [0.0, 0.1, 0.3, 1.0],
            "reg_alpha": [0.0, 0.1, 1.0, 10.0],
            "reg_lambda": [0.1, 1.0, 10.0],
            "tree_method": ["hist"],
            "grow_policy": ["depthwise", "lossguide"],
            "max_leaves": [0, 31, 63, 127],
        },
    },
    {
        "name": "Random Forest",
        "regressor_func": "create_rf_regressor",
        "params": {
            "n_estimators": [300, 600, 1000],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [1, 2, 4, 10],
            "max_features": ["sqrt", "log2", 0.6, 0.8, 1.0],
            "bootstrap": [True],  # en TS conviene mantener bootstrap
            "ccp_alpha": [0.0, 0.0005, 0.001, 0.01],
            "max_leaf_nodes": [None, 1000, 2000],  # opcional
        },
    },
    {
        "name": "Lasso",
        "regressor_func": "create_lasso_regressor",
        "params": {
            "alpha": [0.01, 0.1, 1.0, 10.0],
            "max_iter": [1000, 5000],
            "tol": [1e-4, 1e-3],
        },
    },
]

# # =============================================================================
# # CONFIGURACIÓN DE GUARDADO DE RESULTADOS
# # =============================================================================

# Configuración para guardar resultados de modelos
MODEL_RESULTS_CONFIG = {
    "analytics_dir": str(DATA_DIR / "analytics"),
    "results_subdir": "model_results",
    "include_timestamp": True,
    "save_format": "json",  # json, csv, pickle
    "include_best_params": True,
    "include_validation_metrics": True,
    "include_test_metrics": True,
}
