from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from sklearn.preprocessing import PolynomialFeatures

from astral import LocationInfo
from astral.sun import sun
from feature_engine.datetime import DatetimeFeatures
from feature_engine.timeseries.forecasting import WindowFeatures
from feature_engine.creation import CyclicalFeatures

from ..constants.parsed_fields import (
    LOCATION_CONFIG,
    DEFAULT_CALENDAR_FEATURES,
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
        solar = pd.DataFrame(index=data.index)
        solar["daylight_hours"] = [
            sunset - sunrise for sunrise, sunset in zip(sunrise_hour, sunset_hour)
        ]
        solar["is_daylight"] = np.where(
            (data.index.hour >= pd.Series(sunrise_hour, index=data.index))
            & (data.index.hour < pd.Series(sunset_hour, index=data.index)),
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
        freq: str = "h",
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

    def apply_cyclical_encoding(
        self,
        data: pd.DataFrame,
        features_to_encode: List[str] = None,
        max_values: Dict[str, int] = None,
        drop_original: bool = False,
    ) -> pd.DataFrame:
        """
        Aplica codificación cíclica a características temporales y solares.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame con las características a codificar
        features_to_encode : List[str], optional
            Lista de características a codificar cíclicamente.
            Si None, usa valores por defecto.
        max_values : Dict[str, int], optional
            Diccionario con valores máximos para cada variable cíclica.
            Si None, usa valores por defecto.
        drop_original : bool, default False
            Si eliminar las columnas originales después de la codificación

        Returns
        -------
        pd.DataFrame
            DataFrame con características cíclicas codificadas
        """
        if features_to_encode is None:
            features_to_encode = [
                "month",
                "week",
                "day_of_week",
                "hour",
            ]

        if max_values is None:
            max_values = {
                "month": 12,
                "week": 52,
                "day_of_week": 7,
                "hour": 24,
            }

        # Filtrar solo las características que existen en el DataFrame
        available_features = [f for f in features_to_encode if f in data.columns]
        if not available_features:
            return data

        # Filtrar max_values solo para características disponibles
        available_max_values = {
            k: v for k, v in max_values.items() if k in available_features
        }

        cyclical_encoder = CyclicalFeatures(
            variables=available_features,
            max_values=available_max_values,
            drop_original=drop_original,
        )
        return cyclical_encoder.fit_transform(data)

    def apply_polynomial_features(
        self,
        data: pd.DataFrame,
        columns: List[str] = None,
        degree: int = 2,
        interaction_only: bool = True,
        include_bias: bool = False,
    ) -> pd.DataFrame:
        """
        Aplica características polinomiales (interacciones) a variables especificadas.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame con todas las características creadas
        columns : List[str], optional
            Lista de columnas a usar para generar interacciones.
            Si None, usa todas las columnas del DataFrame.
        degree : int, default 2
            Grado del polinomio
        interaction_only : bool, default True
            Si solo generar interacciones (sin términos cuadráticos individuales)
        include_bias : bool, default False
            Si incluir término de sesgo

        Returns
        -------
        pd.DataFrame
            DataFrame original más las interacciones polinomiales
        """
        if data.empty or len(data.columns) == 0:
            return data

        # Determinar columnas a usar
        if columns is None:
            columns_to_use = list(data.columns)
        else:
            # Filtrar solo las columnas que existen en el DataFrame
            columns_to_use = [c for c in columns if c in data.columns]
            if not columns_to_use:
                return data

        # Verificar y manejar NaN antes de aplicar PolynomialFeatures
        # PolynomialFeatures no acepta NaN, así que necesitamos imputar o filtrar
        data_for_poly = data[columns_to_use].copy()

        # Identificar columnas con NaN
        cols_with_nan = data_for_poly.columns[data_for_poly.isna().any()].tolist()

        if cols_with_nan:
            # Imputar NaN: primero forward fill, luego backward fill, finalmente con 0
            for col in cols_with_nan:
                data_for_poly[col] = data_for_poly[col].ffill().bfill().fillna(0)

        # Crear el transformador polinomial con salida pandas
        transformer_poly = PolynomialFeatures(
            degree=degree,
            interaction_only=interaction_only,
            include_bias=include_bias,
        ).set_output(transform="pandas")

        # Aplicar transformación polinomial solo a las columnas seleccionadas
        variables_poly = transformer_poly.fit_transform(data_for_poly)

        # Eliminar las columnas originales (solo mantener las interacciones)
        variables_poly = variables_poly.drop(columns=columns_to_use)

        # Renombrar columnas con prefijo "poly_" y reemplazar espacios con "__"
        variables_poly.columns = [f"poly_{col}" for col in variables_poly.columns]
        variables_poly.columns = variables_poly.columns.str.replace(" ", "__")

        # Verificar que los índices coinciden
        assert all(data.index == variables_poly.index), (
            "Los índices del DataFrame original y las características polinomiales no coinciden"
        )

        # Concatenar el DataFrame original con las interacciones
        result = pd.concat([data, variables_poly], axis=1)

        return result

    def combine_exogenous_features(
        self,
        calendar_features: pd.DataFrame,
        solar_features: pd.DataFrame,
        window_features: pd.DataFrame,
        stl_features: pd.DataFrame = None,
    ) -> pd.DataFrame:
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
        freq: str = "h",
    ) -> pd.DataFrame:
        # 1. Extraer características de calendario
        cal_vars = self.extract_calendar_features(data, calendar_features)

        # 2. Extraer características solares
        solar_vars = self.extract_solar_features(data)

        # Combinar calendario + solares
        combined_features = pd.concat([cal_vars, solar_vars], axis=1)

        # 3. Aplicar codificación cíclica a todas las características hasta ahora
        combined_features = self.apply_cyclical_encoding(combined_features)

        # 4. Extraer características STL si se solicita
        stl_vars = None
        if use_stl and stl_period is not None:
            stl_vars = self.extract_stl_features(
                data=data,
                period=stl_period,
                robust=stl_robust,
                column=stl_column,
            )
            # Agregar características STL
            if stl_vars is not None:
                combined_features = pd.concat([combined_features, stl_vars], axis=1)

        # 5. Extraer características de ventanas móviles
        window_vars = self.extract_window_features(
            data=data,
            columns=window_columns,
            windows=window_windows,
            functions=window_functions,
            freq=freq,
        )
        # Agregar características de ventanas
        combined_features = pd.concat([combined_features, window_vars], axis=1)

        # 6. Aplicar características polinomiales (interacciones) solo a window_columns
        # window_columns se pasa directamente desde feature_engineering_pipeline.py
        # Estas columnas están en el DataFrame original 'data', no en combined_features
        if window_columns:
            # Filtrar solo las columnas que existen en el DataFrame original
            available_window_cols = [c for c in window_columns if c in data.columns]
            if available_window_cols:
                # Aplicar interacciones polinomiales usando el DataFrame original
                # apply_polynomial_features retorna el DataFrame con las interacciones agregadas
                # pero solo queremos las interacciones (columnas con prefijo 'poly_')
                poly_interactions = self.apply_polynomial_features(
                    data[available_window_cols], columns=available_window_cols
                )
                # Extraer solo las columnas de interacciones (sin las originales)
                poly_cols_only = [
                    col for col in poly_interactions.columns if col.startswith("poly_")
                ]
                if poly_cols_only:
                    poly_df = poly_interactions[poly_cols_only].copy()
                    # Asegurar que el índice coincide
                    poly_df = poly_df.reindex(data.index)
                    # Concatenar las interacciones polinomiales al DataFrame combinado
                    combined_features = pd.concat([combined_features, poly_df], axis=1)

        return combined_features
