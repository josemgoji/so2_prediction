from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from statsmodels.tsa.seasonal import STL


class DateTimeFeatures(BaseEstimator, TransformerMixin):
    def __init__(
        self, time_col="ds", country_holidays="CO", drop_time_col=False, tz=None
    ):
        self.time_col = time_col
        self.country_holidays = country_holidays
        self.drop_time_col = drop_time_col
        self.tz = tz

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.time_col] = pd.to_datetime(X[self.time_col])

        if self.tz:
            X[self.time_col] = (
                X[self.time_col]
                .dt.tz_localize(self.tz, nonexistent="shift_forward", ambiguous="NaT")
                .dt.tz_convert(self.tz)
            )

        dt = X[self.time_col]
        X["year"] = dt.dt.year
        X["month"] = dt.dt.month
        X["day"] = dt.dt.day
        X["hour"] = dt.dt.hour
        X["dow"] = dt.dt.weekday  # 0=lunes
        try:
            X["weekofyear"] = dt.dt.isocalendar().week.astype(int)
        except Exception:
            X["weekofyear"] = dt.dt.week.astype(int)
        X["is_weekend"] = (X["dow"] >= 5).astype(int)
        X["is_month_start"] = dt.dt.is_month_start.astype(int)
        X["is_month_end"] = dt.dt.is_month_end.astype(int)

        try:
            import holidays

            hol = holidays.CountryHoliday(self.country_holidays)
            X["is_holiday"] = dt.dt.date.astype("datetime64").isin(hol).astype(int)
        except Exception:
            X["is_holiday"] = 0

        if self.drop_time_col:
            X = X.drop(columns=[self.time_col])

        return X


class LagFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, target_col="y", lags=range(1, 73)):
        self.target_col = target_col
        self.lags = list(lags)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for L in self.lags:
            X[f"{self.target_col}_lag{L}"] = X[self.target_col].shift(L)
        return X


class RollingStats(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        target_col="y",
        windows=(3, 6, 12, 24, 48, 72),
        stats=("mean", "std", "min", "max"),
    ):
        self.target_col = target_col
        self.windows = windows
        self.stats = stats

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for w in self.windows:
            roll = X[self.target_col].shift(1).rolling(w)
            if "mean" in self.stats:
                X[f"{self.target_col}_roll{w}_mean"] = roll.mean()
            if "std" in self.stats:
                X[f"{self.target_col}_roll{w}_std"] = roll.std(ddof=0)
            if "min" in self.stats:
                X[f"{self.target_col}_roll{w}_min"] = roll.min()
            if "max" in self.stats:
                X[f"{self.target_col}_roll{w}_max"] = roll.max()
        return X


class STLFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, target_col="y", period=24, robust=True):
        self.target_col = target_col
        self.period = period
        self.robust = robust

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.target_col not in X.columns:
            print(
                f"Warning: Target column '{self.target_col}' not found, skipping STL decomposition"
            )
            return X

        series = X[self.target_col].astype(float)
        series_filled = series.interpolate(limit_direction="both")

        if len(series_filled.dropna()) < 2 * self.period:
            print(
                f"Warning: Not enough data points for STL decomposition (need at least {2 * self.period}), skipping"
            )
            return X

        try:
            series_values = (
                series_filled.values
                if hasattr(series_filled, "values")
                else series_filled
            )
            if getattr(series_values, "ndim", 1) > 1:
                series_values = series_values.flatten()

            if "datetime" in X.columns:
                datetime_index = pd.to_datetime(X["datetime"])
                series_for_stl = pd.Series(series_values, index=datetime_index)
            else:
                series_for_stl = pd.Series(series_values)

            series_for_stl = series_for_stl.dropna()

            if len(series_for_stl) < 2 * self.period:
                print(
                    f"Warning: Not enough data points for STL decomposition after cleaning (need at least {2 * self.period}), skipping"
                )
                return X

            res = STL(series_for_stl, period=self.period, robust=self.robust).fit()

            X[f"{self.target_col}_stl_trend"] = res.trend.reindex(X.index, fill_value=0)
            X[f"{self.target_col}_stl_season"] = res.seasonal.reindex(
                X.index, fill_value=0
            )
            X[f"{self.target_col}_stl_resid"] = res.resid.reindex(X.index, fill_value=0)

        except Exception as e:
            print(f"Warning: STL decomposition failed: {e}, skipping")

        return X


class ExogenousRollingStats(BaseEstimator, TransformerMixin):
    """
    Crea estadísticas móviles para variables exógenas **pasadas explícitamente**.
    NO detecta columnas automáticamente; usa la whitelist calculada desde el df base.
    """

    def __init__(
        self,
        *,
        exogenous_cols: list[str],  # requerido
        windows: Iterable[int] = (3, 6, 12, 24, 48, 72),
        stats: Iterable[str] = ("mean", "std", "min", "max"),
    ):
        if not isinstance(exogenous_cols, (list, tuple)):
            raise ValueError(
                "ExogenousRollingStats requiere 'exogenous_cols' (lista de nombres de columnas)."
            )
        self.exogenous_cols = list(exogenous_cols)
        self.windows = tuple(windows)
        self.stats = tuple(stats)
        self._fitted_exog: Optional[list[str]] = None

    def fit(self, X, y=None):
        # Congela la intersección entre la whitelist y las columnas realmente presentes en X
        self._fitted_exog = [c for c in self.exogenous_cols if c in X.columns]
        return self

    def transform(self, X):
        X = X.copy()
        if not self._fitted_exog:
            return X

        new_feats = {}
        for col in self._fitted_exog:
            s = X[col]
            for w in self.windows:
                roll = s.rolling(w)
                if "mean" in self.stats:
                    new_feats[f"{col}_roll{w}_mean"] = roll.mean()
                if "std" in self.stats:
                    new_feats[f"{col}_roll{w}_std"] = roll.std(ddof=0)
                if "min" in self.stats:
                    new_feats[f"{col}_roll{w}_min"] = roll.min()
                if "max" in self.stats:
                    new_feats[f"{col}_roll{w}_max"] = roll.max()

        if new_feats:
            new_df = pd.DataFrame(new_feats, index=X.index)
            X = pd.concat([X, new_df], axis=1)

        return X


class FeatureEngineeringService:
    """
    Construye features con datetime, lags, rolling del target, rolling de exógenas (whitelist) y STL.
    """

    def build_features_from_df(
        self,
        df: pd.DataFrame,
        *,
        time_col_name: str = "datetime",
        target_col_name: str = "SO2",
        stl_period: int = 24,
    ) -> pd.DataFrame:
        # Congela el df base y añade columna de tiempo si el índice es datetime
        if isinstance(df.index, pd.DatetimeIndex):
            df_base = df.copy()
            df_base[time_col_name] = df_base.index
        else:
            df_base = df.copy()

        # Whitelist de exógenas a partir del df base (antes de generar lags/rollings/STL)
        exog_base = (
            df_base.select_dtypes(include=[np.number])
            .columns.difference([target_col_name])
            .tolist()
        )

        pipe = Pipeline(
            steps=[
                (
                    "dt",
                    DateTimeFeatures(
                        time_col=time_col_name,
                        country_holidays="CO",
                        drop_time_col=False,
                    ),
                ),
                (
                    "lags",
                    LagFeatures(target_col=target_col_name, lags=range(1, 73)),
                ),  # 72 lags
                (
                    "roll",
                    RollingStats(
                        target_col=target_col_name, windows=(3, 6, 12, 24, 48, 72)
                    ),
                ),
                (
                    "exog_roll",
                    ExogenousRollingStats(
                        exogenous_cols=exog_base,  # usa solo exógenas del df base
                        windows=(3, 6, 12, 24, 48, 72),
                    ),
                ),
                ("stl", STLFeatures(target_col=target_col_name, period=stl_period)),
            ]
        )

        feat = pipe.fit_transform(df_base)
        feat = feat.dropna().reset_index(drop=True)
        return feat

    def build_features_for_last_row_only(
        self,
        hist: pd.DataFrame,
        *,
        time_col_name: str = "datetime",
        target_col_name: str = "SO2",
        stl_period: int = 24,
        max_lag: int = 72,
        rolling_windows: tuple = (3, 6, 12, 24, 48, 72),
    ) -> pd.DataFrame:
        """
        MÉTODO OPTIMIZADO: Calcula features SOLO para la última fila del histórico.

        Asume que hist tiene al menos max_lag + max(rolling_windows) filas para calcular
        correctamente lags y rolling stats.

        Retorna un DataFrame de UNA fila con todas las features calculadas.
        """
        if len(hist) < max_lag:
            raise ValueError(
                f"Se necesitan al menos {max_lag} filas en hist para calcular lags."
            )

        # Resetear índice para trabajar con posiciones
        hist_reset = hist.reset_index(drop=True)
        last_idx = len(hist_reset) - 1

        # Serie del target
        target_series = hist_reset[target_col_name].values

        # 1. Features de datetime (solo última fila)
        last_row = hist_reset.iloc[last_idx : last_idx + 1].copy()
        dt = pd.to_datetime(last_row[time_col_name])

        features = {
            time_col_name: dt.iloc[0],
            target_col_name: target_series[last_idx],
            "year": dt.dt.year.iloc[0],
            "month": dt.dt.month.iloc[0],
            "day": dt.dt.day.iloc[0],
            "hour": dt.dt.hour.iloc[0],
            "dow": dt.dt.weekday.iloc[0],
            "is_weekend": int(dt.dt.weekday.iloc[0] >= 5),
            "is_month_start": int(dt.dt.is_month_start.iloc[0]),
            "is_month_end": int(dt.dt.is_month_end.iloc[0]),
        }

        # Week of year
        try:
            features["weekofyear"] = dt.dt.isocalendar().week.iloc[0]
        except Exception:
            features["weekofyear"] = dt.dt.week.iloc[0]

        # Holidays
        try:
            import holidays

            hol = holidays.CountryHoliday("CO")
            features["is_holiday"] = int(dt.dt.date.iloc[0] in hol)
        except Exception:
            features["is_holiday"] = 0

        # 2. Lags (usar posiciones, no índices)
        for lag in range(1, max_lag + 1):
            idx = last_idx - lag
            if idx >= 0:
                features[f"{target_col_name}_lag{lag}"] = target_series[idx]
            else:
                features[f"{target_col_name}_lag{lag}"] = np.nan

        # 3. Rolling stats del target
        for window in rolling_windows:
            # Para calcular rolling en la última fila, necesitamos las 'window' filas anteriores
            # (excluyendo la última, por eso shift(1))
            start_idx = max(0, last_idx - window)
            window_data = target_series[start_idx:last_idx]  # excluye la última

            if len(window_data) > 0:
                features[f"{target_col_name}_roll{window}_mean"] = np.mean(window_data)
                features[f"{target_col_name}_roll{window}_std"] = np.std(
                    window_data, ddof=0
                )
                features[f"{target_col_name}_roll{window}_min"] = np.min(window_data)
                features[f"{target_col_name}_roll{window}_max"] = np.max(window_data)
            else:
                features[f"{target_col_name}_roll{window}_mean"] = np.nan
                features[f"{target_col_name}_roll{window}_std"] = np.nan
                features[f"{target_col_name}_roll{window}_min"] = np.nan
                features[f"{target_col_name}_roll{window}_max"] = np.nan

        # 4. Rolling stats de exógenas
        exog_cols = [
            c
            for c in hist_reset.columns
            if c not in [time_col_name, target_col_name]
            and hist_reset[c].dtype in [np.float64, np.float32, np.int64, np.int32]
        ]

        for col in exog_cols:
            exog_series = hist_reset[col].values
            for window in rolling_windows:
                start_idx = max(0, last_idx - window + 1)
                window_data = exog_series[start_idx : last_idx + 1]

                if len(window_data) > 0:
                    features[f"{col}_roll{window}_mean"] = np.mean(window_data)
                    features[f"{col}_roll{window}_std"] = np.std(window_data, ddof=0)
                    features[f"{col}_roll{window}_min"] = np.min(window_data)
                    features[f"{col}_roll{window}_max"] = np.max(window_data)
                else:
                    features[f"{col}_roll{window}_mean"] = np.nan
                    features[f"{col}_roll{window}_std"] = np.nan
                    features[f"{col}_roll{window}_min"] = np.nan
                    features[f"{col}_roll{window}_max"] = np.nan

        # 5. STL features (calcular sobre todo el histórico pero solo retornar última fila)
        if len(hist_reset) >= 2 * stl_period:
            try:
                series = hist_reset[target_col_name].astype(float)
                series_filled = series.interpolate(limit_direction="both")

                if "datetime" in hist_reset.columns:
                    datetime_index = pd.to_datetime(hist_reset["datetime"])
                    series_for_stl = pd.Series(
                        series_filled.values, index=datetime_index
                    )
                else:
                    series_for_stl = pd.Series(series_filled.values)

                series_for_stl = series_for_stl.dropna()

                if len(series_for_stl) >= 2 * stl_period:
                    res = STL(series_for_stl, period=stl_period, robust=True).fit()

                    # Obtener valores de la última posición
                    features[f"{target_col_name}_stl_trend"] = res.trend.iloc[-1]
                    features[f"{target_col_name}_stl_season"] = res.seasonal.iloc[-1]
                    features[f"{target_col_name}_stl_resid"] = res.resid.iloc[-1]
                else:
                    features[f"{target_col_name}_stl_trend"] = 0
                    features[f"{target_col_name}_stl_season"] = 0
                    features[f"{target_col_name}_stl_resid"] = 0
            except Exception as e:
                print(f"Warning: STL decomposition failed: {e}")
                features[f"{target_col_name}_stl_trend"] = 0
                features[f"{target_col_name}_stl_season"] = 0
                features[f"{target_col_name}_stl_resid"] = 0
        else:
            features[f"{target_col_name}_stl_trend"] = 0
            features[f"{target_col_name}_stl_season"] = 0
            features[f"{target_col_name}_stl_resid"] = 0

        # Convertir a DataFrame de una fila
        return pd.DataFrame([features])
