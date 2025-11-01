from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

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

    def extract_stl_features(
        self,
        data: pd.DataFrame,
        period: int,
        robust: bool = True,
        column: str = None,
    ) -> pd.DataFrame:
        """
        Extrae características STL (Seasonal and Trend decomposition using Loess).

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame con la serie temporal
        period : int
            Período estacional para STL
        robust : bool, default True
            Si usar versión robusta de STL
        column : str, optional
            Columna a procesar. Si None, usa la primera columna numérica

        Returns
        -------
        pd.DataFrame
            DataFrame con columnas stl_trend, stl_season, stl_resid
        """
        if column is None:
            # Usar la primera columna numérica
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                raise ValueError("No se encontraron columnas numéricas en el DataFrame")
            column = numeric_cols[0]

        if column not in data.columns:
            raise ValueError(f"La columna '{column}' no existe en el DataFrame")

        y = data[column].astype(float)

        if len(y) >= 2 * period:
            try:
                res = STL(y, period=period, robust=robust).fit()
                trend = res.trend
                seas = res.seasonal
                resid = res.resid
            except Exception:
                trend = seas = resid = np.full(len(y), np.nan)
        else:
            trend = seas = resid = np.full(len(y), np.nan)

        return pd.DataFrame(
            {"stl_trend": trend, "stl_season": seas, "stl_resid": resid},
            index=data.index,
        )

    def extract_window_features(
        self,
        data: pd.DataFrame,
        columns: List[str] = None,
        windows: List[str] = None,
        functions: List[str] = None,
        freq: str = "h",  # fija
    ) -> pd.DataFrame:
        # Validación columnas
        missing = [c for c in (columns or []) if c not in data.columns]
        if missing:
            raise ValueError(f"Las siguientes columnas no existen en 'data': {missing}")

        wf = WindowFeatures(
            variables=columns,
            window=windows,
            functions=functions,
            freq=freq,
        )
        return wf.fit_transform(data[columns])

    def combine_exogenous_features(
        self,
        calendar_features: pd.DataFrame,
        solar_features: pd.DataFrame,
        window_features: pd.DataFrame,
        stl_features: pd.DataFrame = None,
        trim_start: int = DEFAULT_TRIM_START,
        trim_end: int = DEFAULT_TRIM_END,
    ) -> pd.DataFrame:
        # El trimming se aplica en el pipeline para mantener consistencia con crudos
        features_list = [calendar_features, solar_features, window_features]
        if stl_features is not None:
            features_list.append(stl_features)
        return pd.concat(features_list, axis=1)

    def create_all_features(
        self,
        data: pd.DataFrame,
        calendar_features: List[str] = None,
        window_columns: List[str] = None,
        window_windows: List[str] = None,
        window_functions: List[str] = None,
        stl_period: int = None,
        stl_robust: bool = True,
        stl_column: str = None,
        use_stl: bool = False,
        trim_start: int = DEFAULT_TRIM_START,
        trim_end: int = DEFAULT_TRIM_END,
        freq: str = "h",  # fija
    ) -> pd.DataFrame:
        # Extraer cada bloque
        cal_vars = self.extract_calendar_features(data, calendar_features)
        solar_vars = self.extract_solar_features(data)
        window_vars = self.extract_window_features(
            data=data,
            columns=window_columns,
            windows=window_windows,
            functions=window_functions,
            freq=freq,  # siempre 'h'
        )

        # Extraer características STL si se solicita
        stl_vars = None
        if use_stl and stl_period is not None:
            stl_vars = self.extract_stl_features(
                data=data,
                period=stl_period,
                robust=stl_robust,
                column=stl_column,
            )

        return self.combine_exogenous_features(
            cal_vars, solar_vars, window_vars, stl_vars, trim_start, trim_end
        )
