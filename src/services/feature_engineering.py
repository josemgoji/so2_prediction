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
    DEFAULT_TRIM_START,
    DEFAULT_TRIM_END,
)


class FeatureEngineering:
    """
    Ingeniería de características en series temporales:
    - Calendario (mes, semana, día_semana, hora, etc.)
    - Solares (amanecer, atardecer, horas de luz, bandera diurna)
    - Temperatura con ventanas móviles (freq fija 'h')
    """

    def __init__(self, location_config: Optional[Dict] = None):
        self.location_config = location_config or LOCATION_CONFIG
        self.location = None
        self._setup_location()

    def _setup_location(self):
        """Configura información de ubicación para cálculos solares."""
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
        if features_to_extract is None:
            features_to_extract = DEFAULT_CALENDAR_FEATURES

        calendar_transformer = DatetimeFeatures(
            variables="index",
            features_to_extract=features_to_extract,
            drop_original=drop_original,
        )
        transformed = calendar_transformer.fit_transform(data)
        return transformed[features_to_extract]

    def extract_solar_features(self, data: pd.DataFrame) -> pd.DataFrame:
        # amanecer / atardecer en hora local (solo la hora)
        sunrise_hour = [
            sun(self.location.observer, date=idx, tzinfo=self.location.timezone)[
                "sunrise"
            ].hour
            for idx in data.index
        ]
        sunset_hour = [
            sun(self.location.observer, date=idx, tzinfo=self.location.timezone)[
                "sunset"
            ].hour
            for idx in data.index
        ]
        solar = pd.DataFrame(
            {"sunrise_hour": sunrise_hour, "sunset_hour": sunset_hour}, index=data.index
        )
        solar["daylight_hours"] = solar["sunset_hour"] - solar["sunrise_hour"]
        solar["is_daylight"] = np.where(
            (data.index.hour >= solar["sunrise_hour"])
            & (data.index.hour < solar["sunset_hour"]),
            1,
            0,
        )
        return solar

    def extract_temperature_features(
        self,
        data: pd.DataFrame,
        temp_columns: List[str] = None,
        windows: List[str] = None,
        functions: List[str] = None,
        freq: str = "h",  # fija
    ) -> pd.DataFrame:
        # Validación columnas
        missing = [c for c in (temp_columns or []) if c not in data.columns]
        if missing:
            raise ValueError(f"Las siguientes columnas no existen en 'data': {missing}")

        wf = WindowFeatures(
            variables=temp_columns,
            window=windows,
            functions=functions,
            freq=freq,  # siempre 'h'
        )
        return wf.fit_transform(data[temp_columns])

    def combine_exogenous_features(
        self,
        calendar_features: pd.DataFrame,
        solar_features: pd.DataFrame,
        temperature_features: pd.DataFrame,
        trim_start: int = DEFAULT_TRIM_START,
        trim_end: int = DEFAULT_TRIM_END,
    ) -> pd.DataFrame:
        # El trimming se aplica en el pipeline para mantener consistencia con crudos
        return pd.concat(
            [calendar_features, solar_features, temperature_features], axis=1
        )

    def create_all_features(
        self,
        data: pd.DataFrame,
        calendar_features: List[str] = None,
        temp_columns: List[str] = None,
        temp_windows: List[str] = None,
        temp_functions: List[str] = None,
        trim_start: int = DEFAULT_TRIM_START,
        trim_end: int = DEFAULT_TRIM_END,
        freq: str = "h",  # fija
    ) -> pd.DataFrame:
        # Extraer cada bloque
        cal_vars = self.extract_calendar_features(data, calendar_features)
        solar_vars = self.extract_solar_features(data)
        temp_vars = self.extract_temperature_features(
            data=data,
            temp_columns=temp_columns,
            windows=temp_windows,
            functions=temp_functions,
            freq=freq,  # siempre 'h'
        )
        return self.combine_exogenous_features(
            cal_vars, solar_vars, temp_vars, trim_start, trim_end
        )
