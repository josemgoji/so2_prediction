from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from astral import LocationInfo
from astral.sun import sun
from feature_engine.datetime import DatetimeFeatures
from feature_engine.timeseries.forecasting import WindowFeatures

from ..constants.parsed_fields import (
    LOCATION_CONFIG,
    DEFAULT_CALENDAR_FEATURES,
    DEFAULT_TEMP_COLUMNS,
    DEFAULT_TEMP_WINDOWS,
    DEFAULT_TEMP_FUNCTIONS,
    DEFAULT_TRIM_START,
    DEFAULT_TRIM_END,
    EXOGENOUS_WINDOW_FEATURES,
)


class FeatureEngineering:
    """
    Clase para realizar ingeniería de características en series temporales.

    Esta clase permite extraer diferentes tipos de variables exógenas:
    - Variables de calendario (mes, semana, día de la semana, hora)
    - Variables solares (amanecer, atardecer, horas de luz, período diurno)
    - Variables de temperatura con ventanas móviles
    """

    def __init__(self, location_config: Optional[Dict] = None):
        """
        Inicializa la clase FeatureEngineering.

        Parameters:
        -----------
        location_config : dict, optional
            Configuración de ubicación para variables solares. Si no se proporciona,
            usa la configuración por defecto de LOCATION_CONFIG.
        """
        self.location_config = location_config or LOCATION_CONFIG
        self.location = None
        self._setup_location()

    def _setup_location(self):
        """Configura la información de ubicación para cálculos solares."""
        self.location = LocationInfo(
            name=self.location_config["name"],
            region=self.location_config["region"],
            timezone=self.location_config["timezone"],
            latitude=self.location_config["latitude"],
            longitude=self.location_config["longitude"],
        )

    def extract_calendar_features(
        self,
        data: pd.DataFrame,
        features_to_extract: List[str] = None,
        drop_original: bool = True,
    ) -> pd.DataFrame:
        """
        Extrae variables basadas en el calendario.

        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame con índice datetime
        features_to_extract : list, optional
            Lista de características a extraer. Por defecto: ['month', 'week', 'day_of_week', 'hour']
        drop_original : bool, default True
            Si eliminar las variables originales

        Returns:
        --------
        pd.DataFrame
            DataFrame con las variables de calendario
        """
        if features_to_extract is None:
            features_to_extract = DEFAULT_CALENDAR_FEATURES

        calendar_transformer = DatetimeFeatures(
            variables="index",
            features_to_extract=features_to_extract,
            drop_original=drop_original,
        )

        transformed_data = calendar_transformer.fit_transform(data)
        return transformed_data[features_to_extract]

    def extract_solar_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extrae variables basadas en la luz solar.

        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame con índice datetime

        Returns:
        --------
        pd.DataFrame
            DataFrame con las variables solares
        """
        sunrise_hour = [
            sun(self.location.observer, date=date, tzinfo=self.location.timezone)[
                "sunrise"
            ].hour
            for date in data.index
        ]

        sunset_hour = [
            sun(self.location.observer, date=date, tzinfo=self.location.timezone)[
                "sunset"
            ].hour
            for date in data.index
        ]

        variables_solares = pd.DataFrame(
            {"sunrise_hour": sunrise_hour, "sunset_hour": sunset_hour}, index=data.index
        )

        # Calcular horas de luz del día
        variables_solares["daylight_hours"] = (
            variables_solares["sunset_hour"] - variables_solares["sunrise_hour"]
        )

        # Determinar si es período diurno
        variables_solares["is_daylight"] = np.where(
            (data.index.hour >= variables_solares["sunrise_hour"])
            & (data.index.hour < variables_solares["sunset_hour"]),
            1,
            0,
        )

        return variables_solares

    def extract_temperature_features(
        self,
        data: pd.DataFrame,
        temp_columns: List[str] = None,
        windows: List[str] = None,
        functions: List[str] = None,
        freq: str = "h",
    ) -> pd.DataFrame:
        """
        Extrae variables basadas en temperatura con ventanas móviles.

        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame con índice datetime
        temp_columns : list, optional
            Lista de columnas de temperatura.
        windows : list, optional
            Lista de ventanas temporales.
        functions : list, optional
            Lista de funciones a aplicar.
        freq : str, default 'h'
            Frecuencia de la serie temporal

        Returns:
        --------
        pd.DataFrame
            DataFrame con las variables de temperatura
        """

        # Verificar que las columnas existen
        missing_cols = [col for col in temp_columns if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Las siguientes columnas no existen: {missing_cols}")

        wf_transformer = WindowFeatures(
            variables=temp_columns,
            window=windows,
            functions=functions,
            freq=freq,
        )

        return wf_transformer.fit_transform(data[temp_columns])

    def combine_exogenous_features(
        self,
        calendar_features: pd.DataFrame,
        solar_features: pd.DataFrame,
        temperature_features: pd.DataFrame,
        trim_start: int = DEFAULT_TRIM_START,
        trim_end: int = DEFAULT_TRIM_END,
    ) -> pd.DataFrame:
        """
        Combina todas las variables exógenas en un solo DataFrame.

        Parameters:
        -----------
        calendar_features : pd.DataFrame
            Variables de calendario
        solar_features : pd.DataFrame
            Variables solares
        temperature_features : pd.DataFrame
            Variables de temperatura
        trim_start : int, default DEFAULT_TRIM_START
            Número de filas a eliminar al inicio (por medias móviles)
        trim_end : int, default DEFAULT_TRIM_END
            Número de filas a eliminar al final

        Returns:
        --------
        pd.DataFrame
            DataFrame combinado con todas las variables exógenas
        """
        # Combinar todas las variables
        variables_exogenas = pd.concat(
            [calendar_features, solar_features, temperature_features],
            axis=1,
        )

        # Eliminar valores faltantes al principio y al final
        if trim_start > 0:
            variables_exogenas = variables_exogenas.iloc[trim_start:, :]
        if trim_end > 0:
            variables_exogenas = variables_exogenas.iloc[:-trim_end, :]

        return variables_exogenas

    def create_all_features(
        self,
        data: pd.DataFrame,
        calendar_features: List[str] = None,
        temp_columns: List[str] = None,
        temp_windows: List[str] = None,
        temp_functions: List[str] = None,
        trim_start: int = DEFAULT_TRIM_START,
        trim_end: int = DEFAULT_TRIM_END,
    ) -> pd.DataFrame:
        """
        Método principal que crea todas las variables exógenas de una vez.

        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame con índice datetime
        calendar_features : list, optional
            Características de calendario a extraer
        temp_columns : list, optional
            Columnas de temperatura
        temp_windows : list, optional
            Ventanas para variables de temperatura
        temp_functions : list, optional
            Funciones para variables de temperatura
        trim_start : int, default DEFAULT_TRIM_START
            Filas a eliminar al inicio
        trim_end : int, default DEFAULT_TRIM_END
            Filas a eliminar al final

        Returns:
        --------
        pd.DataFrame
            DataFrame con todas las variables exógenas
        """
        # Extraer cada tipo de variable
        calendar_vars = self.extract_calendar_features(data, calendar_features)
        solar_vars = self.extract_solar_features(data)
        temp_vars = self.extract_temperature_features(
            data, temp_columns, temp_windows, temp_functions
        )

        # Combinar todas las variables
        return self.combine_exogenous_features(
            calendar_vars, solar_vars, temp_vars, trim_start, trim_end
        )
