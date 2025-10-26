#!/usr/bin/env python3
"""
Pipeline de Feature Engineering para series temporales de SO2.
Combina la carga de datos con la generación de características exógenas.
"""

import pandas as pd
from typing import Optional, List, Dict, Any
from pathlib import Path

from ..recursos.data_manager import DataManager
from ..services.feature_engineering import FeatureEngineering
from ..constants.parsed_fields import (
    LOCATION_CONFIG,
    DEFAULT_CALENDAR_FEATURES,
    DEFAULT_TEMP_WINDOWS,
    DEFAULT_TEMP_FUNCTIONS,
    DEFAULT_TRIM_START,
    DEFAULT_TRIM_END,
)


class FeatureEngineeringPipeline:
    """
    Pipeline completo para Feature Engineering de series temporales.

    Esta clase combina:
    - Carga de datos usando DataManager
    - Generación de características exógenas usando FeatureEngineering
    - Guardado de resultados procesados
    """

    def __init__(
        self,
        location_config: Optional[Dict] = None,
        date_column: str = "datetime",
    ):
        """
        Inicializa el pipeline de Feature Engineering.

        Parameters:
        -----------
        location_config : dict, optional
            Configuración de ubicación para variables solares
        date_column : str, default "datetime"
            Nombre de la columna de fecha en los CSVs
        """
        self.location_config = location_config or LOCATION_CONFIG
        self.date_column = date_column

        # Inicializar componentes
        self.data_manager = DataManager(date_column=date_column)
        self.feature_engineering = FeatureEngineering(
            location_config=self.location_config
        )

        # Estado del pipeline
        self.raw_data = None
        self.processed_data = None
        self.feature_engineering_result = None

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Carga datos desde un archivo CSV usando DataManager.

        Parameters:
        -----------
        file_path : str
            Ruta al archivo CSV

        Returns:
        --------
        pd.DataFrame
            DataFrame cargado con datetime como índice
        """
        print(f"🔧 Cargando datos desde: {file_path}")

        # Cargar datos usando DataManager
        self.raw_data = self.data_manager.load_data(file_path)

        print(
            f"✅ Datos cargados: {len(self.raw_data)} filas, {len(self.raw_data.columns)} columnas"
        )
        print(
            f"📅 Rango de fechas: {self.raw_data.index.min()} a {self.raw_data.index.max()}"
        )
        print(f"📊 Columnas disponibles: {list(self.raw_data.columns)}")

        # Mostrar estadísticas básicas
        print("\n📈 Estadísticas básicas:")
        print("   - Valores faltantes por columna:")
        for col in self.raw_data.columns:
            missing = self.raw_data[col].isnull().sum()
            print(f"     {col}: {missing} ({missing / len(self.raw_data) * 100:.1f}%)")

        return self.raw_data

    def create_feature_engineering_features(
        self,
        temp_columns: List[str],
        calendar_features: List[str] = None,
        temp_windows: List[str] = None,
        temp_functions: List[str] = None,
        trim_start: int = DEFAULT_TRIM_START,
        trim_end: int = DEFAULT_TRIM_END,
    ) -> pd.DataFrame:
        """
        Crea características de Feature Engineering usando el servicio FeatureEngineering.

        Parameters:
        -----------
        temp_columns : list
            Lista de columnas de temperatura (requerido)
        calendar_features : list, optional
            Características de calendario a extraer
        temp_windows : list, optional
            Ventanas para características de temperatura
        temp_functions : list, optional
            Funciones para características de temperatura
        trim_start : int, default DEFAULT_TRIM_START
            Filas a eliminar al inicio
        trim_end : int, default DEFAULT_TRIM_END
            Filas a eliminar al final

        Returns:
        --------
        pd.DataFrame
            DataFrame con características de Feature Engineering
        """
        if self.raw_data is None:
            raise ValueError("No hay datos cargados. Ejecuta load_data() primero.")

        print("\n" + "=" * 60)
        print("🧪 CREANDO CARACTERÍSTICAS DE FEATURE ENGINEERING")
        print("=" * 60)

        # Usar valores por defecto si no se especifican
        calendar_features = calendar_features or DEFAULT_CALENDAR_FEATURES
        temp_windows = temp_windows or DEFAULT_TEMP_WINDOWS
        temp_functions = temp_functions or DEFAULT_TEMP_FUNCTIONS

        print(
            f"📍 Ubicación configurada: {self.feature_engineering.location_config['name']}"
        )
        print(f"📅 Características de calendario: {calendar_features}")
        print(f"🌡️ Columnas de temperatura: {temp_columns}")
        print(f"⏰ Ventanas de temperatura: {temp_windows}")
        print(f"📊 Funciones de temperatura: {temp_functions}")

        # Crear todas las características usando FeatureEngineering
        self.feature_engineering_result = self.feature_engineering.create_all_features(
            data=self.raw_data,
            calendar_features=calendar_features,
            temp_columns=temp_columns,
            temp_windows=temp_windows,
            temp_functions=temp_functions,
            trim_start=trim_start,
            trim_end=trim_end,
        )

        print(
            f"✅ Características de Feature Engineering creadas: {list(self.feature_engineering_result.columns)}"
        )
        print(f"📊 Shape: {self.feature_engineering_result.shape}")

        return self.feature_engineering_result

    def run_complete_pipeline(
        self,
        file_path: str,
        temp_columns: List[str],
        calendar_features: List[str] = None,
        temp_windows: List[str] = None,
        temp_functions: List[str] = None,
        trim_start: int = DEFAULT_TRIM_START,
        trim_end: int = DEFAULT_TRIM_END,
    ) -> Dict[str, Any]:
        """
        Ejecuta el pipeline completo de Feature Engineering.

        Parameters:
        -----------
        file_path : str
            Ruta al archivo CSV
        temp_columns : list
            Lista de columnas de temperatura (requerido)
        calendar_features : list, optional
            Características de calendario a extraer
        temp_windows : list, optional
            Ventanas para características de temperatura
        temp_functions : list, optional
            Funciones para características de temperatura
        trim_start : int, default DEFAULT_TRIM_START
            Filas a eliminar al inicio
        trim_end : int, default DEFAULT_TRIM_END
            Filas a eliminar al final

        Returns:
        --------
        dict
            Diccionario con todos los resultados del pipeline
        """
        print("🚀 INICIANDO PIPELINE COMPLETO DE FEATURE ENGINEERING")
        print("=" * 60)

        try:
            # 1. Cargar datos
            raw_data = self.load_data(file_path)

            # 2. Crear características de Feature Engineering
            feature_engineering_result = self.create_feature_engineering_features(
                temp_columns=temp_columns,
                calendar_features=calendar_features,
                temp_windows=temp_windows,
                temp_functions=temp_functions,
                trim_start=trim_start,
                trim_end=trim_end,
            )

            # 3. Asignar resultados finales
            self.processed_data = feature_engineering_result

            # Crear diccionario de resultados
            results = {
                "raw_data": raw_data,
                "feature_engineering_result": feature_engineering_result,
                "processed_data": self.processed_data,
                "pipeline_config": {
                    "location_config": self.location_config,
                },
            }

            print("\n" + "=" * 60)
            print("✅ PIPELINE COMPLETADO EXITOSAMENTE")
            print("=" * 60)
            print(f"📊 Datos originales: {raw_data.shape}")
            print(
                f"📊 Características de Feature Engineering: {feature_engineering_result.shape}"
            )
            print(f"📊 Datos procesados finales: {self.processed_data.shape}")

            return results

        except Exception as e:
            print(f"\n❌ ERROR DURANTE EL PIPELINE: {str(e)}")
            import traceback

            traceback.print_exc()
            raise

    def save_results(self, output_path: str, include_raw: bool = False):
        """
        Guarda los resultados del pipeline en archivos CSV.

        Parameters:
        -----------
        output_path : str
            Ruta base donde guardar los archivos
        include_raw : bool, default False
            Si incluir los datos originales en el guardado
        """
        if self.processed_data is None:
            raise ValueError(
                "No hay datos procesados para guardar. Ejecuta el pipeline primero."
            )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"💾 Guardando resultados en: {output_path}")

        # Guardar datos procesados
        processed_path = output_path.with_suffix(".csv")
        self.data_manager.save(self.processed_data, str(processed_path))
        print(f"✅ Datos procesados guardados en: {processed_path}")

        # Guardar datos originales si se solicita
        if include_raw and self.raw_data is not None:
            raw_path = output_path.with_name(f"{output_path.stem}_raw.csv")
            self.data_manager.save(self.raw_data, str(raw_path))
            print(f"✅ Datos originales guardados en: {raw_path}")

    def get_summary(self) -> Dict[str, Any]:
        """
        Obtiene un resumen de los resultados del pipeline.

        Returns:
        --------
        dict
            Diccionario con resumen de resultados
        """
        summary = {
            "pipeline_status": "completed"
            if self.processed_data is not None
            else "not_run",
            "data_info": {},
            "features_info": {},
        }

        if self.raw_data is not None:
            summary["data_info"]["raw_shape"] = self.raw_data.shape
            summary["data_info"]["raw_columns"] = list(self.raw_data.columns)
            summary["data_info"]["date_range"] = {
                "start": str(self.raw_data.index.min()),
                "end": str(self.raw_data.index.max()),
            }

        if self.feature_engineering_result is not None:
            summary["features_info"]["feature_engineering_shape"] = (
                self.feature_engineering_result.shape
            )
            summary["features_info"]["feature_engineering_columns"] = list(
                self.feature_engineering_result.columns
            )

        if self.processed_data is not None:
            summary["data_info"]["processed_shape"] = self.processed_data.shape

        return summary


def create_default_pipeline() -> FeatureEngineeringPipeline:
    """
    Crea un pipeline de Feature Engineering con valores por defecto.

    Returns:
    --------
    FeatureEngineeringPipeline
        Instancia configurada con valores por defecto
    """
    return FeatureEngineeringPipeline()


def create_custom_pipeline(
    location_config: Optional[Dict] = None,
) -> FeatureEngineeringPipeline:
    """
    Crea un pipeline de Feature Engineering con configuración personalizada.

    Parameters:
    -----------
    location_config : dict, optional
        Configuración de ubicación para variables solares

    Returns:
    --------
    FeatureEngineeringPipeline
        Instancia configurada con parámetros personalizados
    """
    return FeatureEngineeringPipeline(
        location_config=location_config,
    )
