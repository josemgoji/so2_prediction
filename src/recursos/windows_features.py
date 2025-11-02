import numpy as np
import pandas as pd

from ..constants.parsed_fields import (
    DEFAULT_WINDOW_PERIOD,
    DEFAULT_WINDOW_STATS,
    DEFAULT_WINDOW_SIZES,
    DEFAULT_FOURIER_K,
    DEFAULT_DELTA_LAGS
)


class CustomRollingFeatures:
    """
    Clase personalizada para calcular características de ventana móvil.
    Calcula todas las estadísticas especificadas para cada tamaño de ventana.

    IMPORTANTE: std/var usan ddof=1 para alinear con pandas.rolling(std/var).
    """

    def __init__(self, stats: list, window_sizes: list):
        self.stats = stats
        self.window_sizes = window_sizes
        self.features_names = []
        for window in window_sizes:
            for stat in stats:
                self.features_names.append(f"rolling_{stat}_{window}")

    def transform_batch(self, y: pd.Series) -> pd.DataFrame:
        result_data = {}
        for window in self.window_sizes:
            rolling = y.rolling(window=window)
            for stat in self.stats:
                col_name = f"rolling_{stat}_{window}"
                if stat == "mean":
                    result_data[col_name] = rolling.mean()
                elif stat == "std":
                    result_data[col_name] = rolling.std(ddof=1)
                elif stat == "min":
                    result_data[col_name] = rolling.min()
                elif stat == "max":
                    result_data[col_name] = rolling.max()
                elif stat == "median":
                    result_data[col_name] = rolling.median()
                elif stat == "var":
                    result_data[col_name] = rolling.var(ddof=1)
                else:
                    result_data[col_name] = rolling.agg(stat)
        return pd.DataFrame(result_data, index=y.index)

    def transform(self, y: np.ndarray) -> np.ndarray:
        res = []
        for w in self.window_sizes:
            if len(y) >= w:
                win = y[-w:]
                for stat in self.stats:
                    if stat == "mean":
                        res.append(float(np.mean(win)))
                    elif stat == "std":
                        res.append(float(np.std(win, ddof=1)))
                    elif stat == "min":
                        res.append(float(np.min(win)))
                    elif stat == "max":
                        res.append(float(np.max(win)))
                    elif stat == "median":
                        res.append(float(np.median(win)))
                    elif stat == "var":
                        res.append(float(np.var(win, ddof=1)))
                    else:
                        res.append(np.nan)
            else:
                for _ in self.stats:
                    res.append(np.nan)
        return np.asarray(res, dtype=float)  # (n_features,)


class DeltaLagFeatures:
    """
    Características de delta: y_t - y_{t-lag} para lags especificados.
    Compatible con skforecast: implementa transform_batch (entrenamiento) y transform (online).
    """

    def __init__(self, lags: list[int]):
        """
        Parameters
        ----------
        lags : list[int]
            Lags (en pasos de tiempo) para calcular deltas. Ej: [1, 6, 24]
        """
        if not isinstance(lags, (list, tuple)) or len(lags) == 0:
            raise ValueError(
                "`lags` debe ser una lista/tupla no vacía de enteros positivos."
            )
        if any((not isinstance(l, int)) or l <= 0 for l in lags):
            raise ValueError("Todos los lags deben ser enteros positivos.")

        self.lags = list(sorted(set(lags)))
        self.features_names = [f"delta_lag_{lag}" for lag in self.lags]

    def transform_batch(self, y: pd.Series) -> pd.DataFrame:
        """
        Calcula para toda la serie los deltas y_t - y_{t-lag}.
        Las primeras `lag` observaciones quedan como NaN.
        """
        data = {}
        for lag in self.lags:
            data[f"delta_lag_{lag}"] = y - y.shift(lag)
        return pd.DataFrame(data, index=y.index)

    def transform(self, y: np.ndarray) -> np.ndarray:
        """
        Devuelve los deltas para el último punto disponible:
        delta_lag_l = y[-1] - y[-1-l], si existe; en caso contrario NaN.
        """
        res = []
        n = len(y)
        if n == 0:
            return np.asarray([np.nan] * len(self.lags), dtype=float)

        current = float(y[-1])
        for lag in self.lags:
            if n > lag:
                res.append(current - float(y[-1 - lag]))
            else:
                res.append(np.nan)
        return np.asarray(res, dtype=float)


class WindowFeaturesGenerator:
    """
    Genera lista de window features para usar en ForecasterRecursive.
    Combina Rolling y DeltaLag (y opcionalmente Fourier si lo integras luego).
    """

    def __init__(
        self,
        period: int,
        stats: list,
        window_sizes: list,
        fourier_k: int,
        delta_lags: list[int] | None = None,
    ):
        self.period = period
        self.stats = stats
        self.window_sizes = window_sizes
        self.fourier_k = fourier_k
        self.delta_lags = delta_lags or []  # p.ej. [1, 6, 24]
        self._create_window_features()

    def _create_window_features(self):
        self.window_features = [
            CustomRollingFeatures(stats=self.stats, window_sizes=self.window_sizes),
        ]
        if len(self.delta_lags) > 0:
            self.window_features.append(DeltaLagFeatures(lags=self.delta_lags))
        # Si agregas Fourier en otra clase, podrías añadirla aquí.

    def get_window_features(self):
        return self.window_features

    def get_feature_names(self):
        feature_names = []
        for feature in self.window_features:
            if hasattr(feature, "features_names"):
                feature_names.extend(feature.features_names)
            elif hasattr(feature, "get_feature_names"):
                feature_names.extend(feature.get_feature_names())
        return feature_names


def create_default_window_features_generator():
    return WindowFeaturesGenerator(
        period=DEFAULT_WINDOW_PERIOD,
        stats=DEFAULT_WINDOW_STATS,
        window_sizes=DEFAULT_WINDOW_SIZES,
        fourier_k=DEFAULT_FOURIER_K,
        delta_lags=DEFAULT_DELTA_LAGS,
    )
