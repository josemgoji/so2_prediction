import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

from ..constants.parsed_fields import (
    DEFAULT_WINDOW_PERIOD,
    DEFAULT_WINDOW_STATS,
    DEFAULT_WINDOW_SIZES,
    DEFAULT_FOURIER_K,
    DEFAULT_STL_ROBUST,
)


class FourierWindowFeatures:
    """
    Custom class para generar características tipo Fourier.
    Compatible con ForecasterRecursive (skforecast>=0.10.0).

    IMPORTANTE: El orden de features_names coincide con el orden que devuelve transform:
    primero todos los 'sin', luego todos los 'cos'.
    """

    def __init__(self, period: int, K: int = 3):
        self.period = period
        self.K = K
        # Orden consistente: sin_1..sin_K, cos_1..cos_K
        self.features_names = [f"fourier_sin_{period}_{k}" for k in range(1, K + 1)] + [
            f"fourier_cos_{period}_{k}" for k in range(1, K + 1)
        ]
        # skforecast requiere este atributo cuando aplica
        self.window_sizes = [period]
        self._t_last = 0

    def transform_batch(self, y: pd.Series) -> pd.DataFrame:
        n = len(y)
        t = np.arange(n, dtype=float)
        data = {}
        # Primero todos los 'sin'
        for k in range(1, self.K + 1):
            data[f"fourier_sin_{self.period}_{k}"] = np.sin(
                2 * np.pi * k * t / self.period
            )
        # Luego todos los 'cos'
        for k in range(1, self.K + 1):
            data[f"fourier_cos_{self.period}_{k}"] = np.cos(
                2 * np.pi * k * t / self.period
            )
        return pd.DataFrame(data, index=y.index)

    def transform(self, y: np.ndarray) -> np.ndarray:
        # ➜ DEVOLVER 1D
        t_next = len(y)
        row = []
        for k in range(1, self.K + 1):
            row.append(np.sin(2 * np.pi * k * t_next / self.period))
        for k in range(1, self.K + 1):
            row.append(np.cos(2 * np.pi * k * t_next / self.period))
        return np.asarray(row, dtype=float)  # (n_features,)


class STLWindowFeatures:
    """
    Custom window feature que calcula descomposición STL causal.
    Compatible con ForecasterRecursive (skforecast>=0.10.0).
    """

    def __init__(self, period: int, window: int | None = None, robust: bool = True):
        self.period = period
        self.window = window or max(2 * period, 3 * period)
        self.robust = robust
        self.features_names = ["stl_trend", "stl_season", "stl_resid"]
        self.window_sizes = [self.window]  # requerido por skforecast

    def transform_batch(self, y: pd.Series) -> pd.DataFrame:
        """
        Batch mode: crea features STL usando solo datos históricos hasta t-1.
        """
        n = len(y)
        y_vals = y.astype(float).values
        trend, seas, resid = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)

        for t in range(n):
            end = t
            start = max(0, end - self.window)
            segment = y_vals[start:end]
            if len(segment) >= 2 * self.period:
                try:
                    res = STL(segment, period=self.period, robust=self.robust).fit()
                    trend[t] = res.trend[-1]
                    seas[t] = res.seasonal[-1]
                    resid[t] = res.resid[-1]
                except Exception:
                    trend[t], seas[t], resid[t] = np.nan, np.nan, np.nan
        return pd.DataFrame(
            {"stl_trend": trend, "stl_season": seas, "stl_resid": resid}, index=y.index
        )

    def transform(self, y: np.ndarray) -> np.ndarray:
        # ➜ DEVOLVER 1D
        if len(y) >= 2 * self.period:
            try:
                res = STL(y, period=self.period, robust=self.robust).fit()
                return np.asarray(
                    [res.trend[-1], res.seasonal[-1], res.resid[-1]], dtype=float
                )
            except Exception:
                pass
        mean_val = np.nanmean(y[-self.period :]) if len(y) > 0 else np.nan
        return np.asarray([mean_val, 0.0, 0.0], dtype=float)


class CustomRollingFeatures:
    """
    Clase personalizada para calcular características de ventana móvil.
    Calcula todas las estadísticas especificadas para cada tamaño de ventana.

    IMPORTANTE: std/var usan ddof=1 para alinear con pandas.rolling(std/var).
    """

    def __init__(self, stats: list, window_sizes: list):
        """
        Parameters
        ----------
        stats : list
            Lista de estadísticas a calcular (p. ej. ['mean', 'std', 'min', 'max'])
        window_sizes : list
            Lista de tamaños de ventana (p. ej. [3, 6, 12, 24])
        """
        self.stats = stats
        self.window_sizes = window_sizes
        self.features_names = []
        for window in window_sizes:
            for stat in stats:
                self.features_names.append(f"rolling_{stat}_{window}")

    def transform_batch(self, y: pd.Series) -> pd.DataFrame:
        """
        Calcula las características de ventana móvil para toda la serie (batch).
        """
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
        # ➜ DEVOLVER 1D
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


class WindowFeaturesGenerator:
    """
    Genera lista de window features para usar en ForecasterRecursive.
    Combina Fourier, (opcional STL) y Rolling.
    """

    def __init__(
        self,
        period: int,
        stats: list,
        window_sizes: list,
        fourier_k: int,
        stl_robust: bool,
        use_stl: bool = False,
    ):
        self.period = period
        self.stats = stats
        self.window_sizes = window_sizes
        self.fourier_k = fourier_k
        self.stl_robust = stl_robust
        self.use_stl = use_stl
        self._create_window_features()

    def _create_window_features(self):
        self.window_features = [
            FourierWindowFeatures(period=self.period, K=self.fourier_k),
            CustomRollingFeatures(stats=self.stats, window_sizes=self.window_sizes),
        ]
        if self.use_stl:
            self.window_features.insert(
                1, STLWindowFeatures(period=self.period, robust=self.stl_robust)
            )

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
        stl_robust=DEFAULT_STL_ROBUST,
        use_stl=False,
    )
