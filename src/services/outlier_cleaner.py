import pandas as pd
from sklearn.ensemble import IsolationForest
from src.constants.parsed_fields import TAIRE_COL
import numpy as np
from scipy.stats import kurtosis
import matplotlib.pyplot as plt
from src.utils.data_filter import filter_from_first_valid_date


class DataCleaning:
    """
    Métodos para limpiar DataFrames: imputación de nulos y eliminación de outliers univariables y multivariables.
    """

    class OutlierResult:
        """
        Clase interna para encapsular el resultado de la detección de outliers.
        """

        def __init__(
            self,
            df_clean: pd.DataFrame,
            df_with_outliers: pd.DataFrame,
            outlier_labels_uni: np.ndarray = None,
            outlier_labels_multi: np.ndarray = None,
        ):
            self.df = df_clean  # For backward compatibility
            self.df_clean = df_clean
            self.df_with_outliers = df_with_outliers
            self.outlier_labels_uni = outlier_labels_uni
            self.outlier_labels_multi = outlier_labels_multi

        def plot_univariate_outliers(
            self,
            column: str,
            x_label: str = "Index",
            y_label: str = "Value",
            title: str = "Univariate Outliers",
            save_path: str = None,
        ):
            """
            Genera un scatter plot para visualizar outliers univariables.

            Args:
                column (str): Nombre de la columna a graficar.
                x_label (str): Etiqueta para el eje X.
                y_label (str): Etiqueta para el eje Y.
                title (str): Título del gráfico.
                save_path (str): Ruta para guardar el gráfico (opcional).
            """
            plt.figure(figsize=(8, 6))

            # Asegurar que tenemos datos válidos
            df_plot = self.df_with_outliers.dropna(subset=[column, "IF_outlier"])

            if len(df_plot) == 0:
                print(f"No hay datos válidos para graficar la columna {column}")
                return

            plt.plot(
                df_plot.index, df_plot[column], label="Datos originales", color="blue"
            )

            # Filtrar outliers de manera segura
            outlier_mask = df_plot["IF_outlier"] == 1
            if outlier_mask.any():
                outlier_data = df_plot[outlier_mask]
                plt.scatter(
                    outlier_data.index,
                    outlier_data[column],
                    color="red",
                    label="Outliers",
                    alpha=0.7,
                )

            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.title(title)
            plt.legend()
            if save_path:
                plt.savefig(save_path)
                plt.close()
            else:
                plt.show()

        def plot_multivariate_outliers(
            self,
            outlier_col: str = "Curtosis_outlier",
            pollutant: str = "SO2",
            station: str = "",
            save_path: str = None,
        ):
            """
            Genera un scatter plot de Temp vs Pollutant, resaltando outliers multivariables.

            Args:
                outlier_col (str): Nombre de la columna que indica los outliers (por defecto, "Curtosis_outlier").
                pollutant (str): Nombre del contaminante.
                station (str): Nombre de la estación.
                save_path (str): Ruta para guardar el gráfico (opcional).
            """
            # Asegurar que tenemos las columnas necesarias
            required_cols = [outlier_col, TAIRE_COL, "target"]
            df_clean = self.df_with_outliers.dropna(subset=required_cols).copy()

            if len(df_clean) == 0:
                print(f"No hay datos válidos para graficar outliers multivariables")
                return

            plt.figure(figsize=(8, 6))

            # Filtrar outliers e inliers de manera segura
            outlier_mask = df_clean[outlier_col] == 1
            inlier_mask = df_clean[outlier_col] == 0

            if outlier_mask.any():
                outlier_data = df_clean[outlier_mask]
                plt.scatter(
                    outlier_data[TAIRE_COL],
                    outlier_data["target"],
                    color="red",
                    label="Outliers",
                    alpha=0.7,
                )

            if inlier_mask.any():
                inlier_data = df_clean[inlier_mask]
                plt.scatter(
                    inlier_data[TAIRE_COL],
                    inlier_data["target"],
                    color="blue",
                    label="Inliers",
                    alpha=0.7,
                )

            plt.xlabel("Temp")
            plt.ylabel(pollutant)
            plt.legend()
            plt.title(
                f"Scatter Plot Temp vs {pollutant} - Outliers Multivariables {station}"
            )
            if save_path:
                plt.savefig(save_path)
                plt.close()
            else:
                plt.show()

    def __init__(self, contamination: float = 0.01, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.columns_required = ["target", TAIRE_COL]

    def impute_and_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        df_imp = df.copy()
        df_mean = df_imp.fillna(df_imp.rolling(window=3, min_periods=1).mean())
        df_interp = df_mean.interpolate(method="time")
        df_filtered = filter_from_first_valid_date(df_interp, self.columns_required)
        return df_filtered

    def clean_data(self, df: pd.DataFrame) -> OutlierResult:
        """
        Limpia el DataFrame eliminando outliers univariables y multivariables.

        Args:
            df (pd.DataFrame): DataFrame original.

        Returns:
            OutlierResult: Objeto que contiene el DataFrame limpio y los resultados de los outliers.
        """
        # Copia del DataFrame original completo para detección de outliers univariables
        df_uni = df.copy()
        df_uni_clean = df_uni.dropna(subset=["target"]).copy()

        # Detectar outliers univariables en el DataFrame original completo
        iso_forest = IsolationForest(
            contamination=self.contamination, random_state=self.random_state
        )
        outlier_labels_uni = iso_forest.fit_predict(df_uni_clean[["target"]])
        df_uni_clean["IF_outlier"] = 0
        df_uni_clean.loc[df_uni_clean.index[outlier_labels_uni == -1], "IF_outlier"] = 1

        # Imputar y filtrar el DataFrame para detección de outliers multivariables
        df_multi = self.impute_and_filter(df).dropna()

        # Detectar outliers multivariables
        X = df_multi.values  # Ya imputado, no necesita dropna
        initial_projections = [np.random.rand(X.shape[1]) for _ in range(1000)]
        initial_projections = [v / np.linalg.norm(v) for v in initial_projections]
        projected_data = [np.dot(X, v) for v in initial_projections]
        kurtosis_values = [kurtosis(proj) for proj in projected_data]
        best_projection_index = np.argmax(kurtosis_values)
        best_projection = projected_data[best_projection_index]

        cutoff1 = np.percentile(best_projection, 99.5)
        cutoff2 = np.percentile(best_projection, 0.5)
        outlier_labels_multi = (
            (best_projection < cutoff2) | (best_projection > cutoff1)
        ).astype(int)

        df_multi["Curtosis_outlier"] = outlier_labels_multi

        # Crear el DataFrame final como copia del original completo
        df_final = df.copy()

        # Agregar columnas de outliers al DataFrame final uniendo por índices
        df_final["IF_outlier"] = 0
        df_final.loc[df_uni_clean.index, "IF_outlier"] = df_uni_clean["IF_outlier"]
        df_final["Curtosis_outlier"] = 0
        df_final.loc[df_multi.index, "Curtosis_outlier"] = df_multi["Curtosis_outlier"]

        # Crear el DataFrame limpio eliminando outliers
        mask = (df_final["IF_outlier"] == 0) & (df_final["Curtosis_outlier"] == 0)
        df_clean = df_final.loc[mask].copy()

        # Filtrar desde la primera fecha donde hay valores de target
        df_clean = filter_from_first_valid_date(df_clean, ["target"])

        return self.OutlierResult(
            df_clean=df_clean,
            df_with_outliers=df_final,
            outlier_labels_uni=outlier_labels_uni,
            outlier_labels_multi=outlier_labels_multi,
        )
