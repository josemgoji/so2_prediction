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
    """

    def __init__(self, period: int, K: int = 3):
        self.period = period
        self.K = K
        self.features_names = [f"fourier_sin_{period}_{k}" for k in range(1, K + 1)] + [
            f"fourier_cos_{period}_{k}" for k in range(1, K + 1)
        ]
        self.window_sizes = [period]  # Skforecast requiere este atributo
        self._t_last = 0

    def transform_batch(self, y: pd.Series) -> pd.DataFrame:
        n = len(y)
        t = np.arange(n, dtype=float)
        data = {}
        for k in range(1, self.K + 1):
            data[f"fourier_sin_{self.period}_{k}"] = np.sin(
                2 * np.pi * k * t / self.period
            )
            data[f"fourier_cos_{self.period}_{k}"] = np.cos(
                2 * np.pi * k * t / self.period
            )
        return pd.DataFrame(data, index=y.index)

    def transform(self, y: np.ndarray) -> np.ndarray:
        """
        Recibe ventana causal de y (incluye predicciones previas) y devuelve
        features Fourier para el siguiente paso.
        """
        t_next = len(y)
        row = []
        for k in range(1, self.K + 1):
            row.append(np.sin(2 * np.pi * k * t_next / self.period))
            row.append(np.cos(2 * np.pi * k * t_next / self.period))
        return np.array(row).reshape(1, -1)


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
        self.window_sizes = [self.window]  # Skforecast requiere este atributo

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
        """
        Step mode: recibe la ventana causal actual y devuelve las 3 features STL
        calculadas solo con esa ventana.
        """
        if len(y) >= 2 * self.period:
            try:
                res = STL(y, period=self.period, robust=self.robust).fit()
                return np.array([[res.trend[-1], res.seasonal[-1], res.resid[-1]]])
            except Exception:
                pass
        # fallback si no hay suficiente data
        mean_val = np.nanmean(y[-self.period :]) if len(y) > 0 else np.nan
        return np.array([[mean_val, 0.0, 0.0]])


class CustomRollingFeatures:
    """
    Clase personalizada para calcular características de ventana móvil.
    Calcula todas las estadísticas especificadas para cada tamaño de ventana.
    """

    def __init__(self, stats: list, window_sizes: list):
        """
        Inicializa la clase CustomRollingFeatures.

        Parameters:
        -----------
        stats : list
            Lista de estadísticas a calcular (ej: ['mean', 'std', 'min', 'max'])
        window_sizes : list
            Lista de tamaños de ventana (ej: [3, 6, 12, 24])
        """
        self.stats = stats
        self.window_sizes = window_sizes
        self.features_names = []

        # Generar nombres de características
        for window in window_sizes:
            for stat in stats:
                self.features_names.append(f"rolling_{stat}_{window}")

    def transform_batch(self, y: pd.Series) -> pd.DataFrame:
        """
        Calcula las características de ventana móvil para toda la serie.

        Parameters:
        -----------
        y : pd.Series
            Serie temporal con índice datetime

        Returns:
        --------
        pd.DataFrame
            DataFrame con las características calculadas
        """
        result_data = {}

        for window in self.window_sizes:
            rolling = y.rolling(window=window)

            for stat in self.stats:
                col_name = f"rolling_{stat}_{window}"
                if stat == "mean":
                    result_data[col_name] = rolling.mean()
                elif stat == "std":
                    result_data[col_name] = rolling.std()
                elif stat == "min":
                    result_data[col_name] = rolling.min()
                elif stat == "max":
                    result_data[col_name] = rolling.max()
                elif stat == "median":
                    result_data[col_name] = rolling.median()
                elif stat == "var":
                    result_data[col_name] = rolling.var()
                else:
                    # Para otras estadísticas, usar agg
                    result_data[col_name] = rolling.agg(stat)

        return pd.DataFrame(result_data, index=y.index)

    def transform(self, y: np.ndarray) -> np.ndarray:
        """
        Calcula las características para un paso específico (para forecasting).

        Parameters:
        -----------
        y : np.ndarray
            Array con la ventana de datos actual

        Returns:
        --------
        np.ndarray
            Array con las características calculadas
        """
        result = []

        for window in self.window_sizes:
            if len(y) >= window:
                window_data = y[-window:]  # Últimos 'window' valores

                for stat in self.stats:
                    if stat == "mean":
                        result.append(np.mean(window_data))
                    elif stat == "std":
                        result.append(np.std(window_data))
                    elif stat == "min":
                        result.append(np.min(window_data))
                    elif stat == "max":
                        result.append(np.max(window_data))
                    elif stat == "median":
                        result.append(np.median(window_data))
                    elif stat == "var":
                        result.append(np.var(window_data))
                    else:
                        # Para otras estadísticas
                        result.append(np.nan)
            else:
                # Si no hay suficientes datos, llenar con NaN
                for stat in self.stats:
                    result.append(np.nan)

        return np.array(result).reshape(1, -1)


class WindowFeaturesGenerator:
    """
    Clase que genera características de ventana para un forecaster recursivo.
    Combina características Fourier, STL y Rolling con parámetros personalizables.
    """

    def __init__(
        self,
        period: int,
        stats: list,
        window_sizes: list,
        fourier_k: int,
        stl_robust: bool,
    ):
        """
        Inicializa el generador de características de ventana.

        Parameters:
        -----------
        period : int
            Período para características Fourier y STL
        stats : list
            Estadísticas para características Rolling
        window_sizes : list
            Tamaños de ventana para características Rolling
        fourier_k : int
            Número de componentes Fourier
        stl_robust : bool
            Si usar STL robusto
        """
        self.period = period
        self.stats = stats
        self.window_sizes = window_sizes
        self.fourier_k = fourier_k
        self.stl_robust = stl_robust

        # Crear las características de ventana
        self._create_window_features()

    def _create_window_features(self):
        """Crea la lista de características de ventana."""
        self.window_features = [
            FourierWindowFeatures(period=self.period, K=self.fourier_k),
            # STLWindowFeatures(period=self.period, robust=self.stl_robust),
            CustomRollingFeatures(
                stats=self.stats,
                window_sizes=self.window_sizes,
            ),
        ]

    def get_window_features(self):
        """
        Retorna la lista de características de ventana configuradas.

        Returns:
        --------
        list
            Lista de objetos de características de ventana
        """
        return self.window_features

    def get_feature_names(self):
        """
        Retorna los nombres de todas las características generadas.

        Returns:
        --------
        list
            Lista de nombres de características
        """
        feature_names = []
        for feature in self.window_features:
            if hasattr(feature, "features_names"):
                feature_names.extend(feature.features_names)
            elif hasattr(feature, "get_feature_names"):
                feature_names.extend(feature.get_feature_names())
        return feature_names


def create_default_window_features_generator():
    """
    Crea un WindowFeaturesGenerator con los valores por defecto de las constantes.

    Returns:
    --------
    WindowFeaturesGenerator
        Instancia configurada con valores por defecto
    """
    return WindowFeaturesGenerator(
        period=DEFAULT_WINDOW_PERIOD,
        stats=DEFAULT_WINDOW_STATS,
        window_sizes=DEFAULT_WINDOW_SIZES,
        fourier_k=DEFAULT_FOURIER_K,
        stl_robust=DEFAULT_STL_ROBUST,
    )
