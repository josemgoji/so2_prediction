#!/usr/bin/env python3
"""
Script para comparar múltiples modelos usando la infraestructura existente.
Incluye opción de con y sin exógenas, usando datos procesados y características seleccionadas.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import json

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.recursos.data_manager import DataManager
from src.recursos.regressors import RandomForestRegressor, LGBMRegressor, RidgeRegressor
from src.recursos.scorers import Scorer
from src.recursos.windows_features import WindowFeaturesGenerator
from src.constants.parsed_fields import STAGE_DIR, DEFAULT_WINDOW_FEATURES_PARAMS
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings("ignore")


class ModelComparator:
    """
    Clase para comparar múltiples modelos con diferentes configuraciones.
    """

    def __init__(self, station: str, use_exogenous: bool = True):
        """
        Inicializa el comparador de modelos.

        Parameters:
        -----------
        station : str
            Nombre de la estación (ej: 'GIR-EPM', 'CEN-TRAF', etc.)
        use_exogenous : bool, default True
            Si usar variables exógenas o solo características de ventana
        """
        self.station = station
        self.use_exogenous = use_exogenous
        self.data_manager = DataManager()

        # Cargar datos y características
        self.data = None
        self.selected_features = None
        self.results = []

    def load_data(self) -> pd.DataFrame:
        """Carga los datos procesados y las características seleccionadas."""
        print(f"Cargando datos para estación: {self.station}")

        # Cargar datos procesados
        processed_file = (
            STAGE_DIR / "SO2" / "processed" / f"processed_{self.station}.csv"
        )
        if not processed_file.exists():
            raise FileNotFoundError(
                f"Archivo procesado no encontrado: {processed_file}"
            )

        self.data = self.data_manager.load_data(str(processed_file))
        print(f"Datos cargados: {self.data.shape}")

        # Cargar características seleccionadas
        exog_suffix = "con_exog" if self.use_exogenous else "sin_exog"
        selected_file = (
            STAGE_DIR
            / "SO2"
            / "selected"
            / "lasso"
            / exog_suffix
            / f"selected_cols_{self.station}_lasso_rf.json"
        )

        if not selected_file.exists():
            raise FileNotFoundError(
                f"Archivo de características seleccionadas no encontrado: {selected_file}"
            )

        with open(selected_file, "r") as f:
            self.selected_features = json.load(f)

        print(f"Características seleccionadas: {len(self.selected_features)}")
        print(f"Características: {self.selected_features[:5]}...")  # Mostrar primeras 5

        return self.data

    def prepare_features(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepara las características para el entrenamiento.

        Returns:
        --------
        Tuple[pd.DataFrame, pd.Series]
            (X, y) donde X son las características e y es el target
        """
        if self.data is None:
            raise ValueError("Debes cargar los datos primero con load_data()")

        # Filtrar características seleccionadas
        available_features = [
            col for col in self.selected_features if col in self.data.columns
        ]
        missing_features = [
            col for col in self.selected_features if col not in self.data.columns
        ]

        if missing_features:
            print(
                f"Advertencia: {len(missing_features)} características no encontradas: {missing_features[:5]}..."
            )

        print(
            f"Usando {len(available_features)} características de {len(self.selected_features)} seleccionadas"
        )

        # Preparar X e y
        X = self.data[available_features].copy()
        y = self.data["target"].copy()

        # Eliminar filas con valores nulos
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]

        print(f"Datos finales: {X.shape[0]} muestras, {X.shape[1]} características")

        return X, y

    def get_models(self) -> Dict:
        """Define los modelos a comparar."""
        return {
            "RandomForest": RandomForestRegressor(),
            "LGBM": LGBMRegressor(),
            "Ridge": RidgeRegressor(),
        }

    def get_param_grids(self) -> Dict:
        """Define los grids de parámetros para cada modelo."""
        return {
            "RandomForest": {
                "n_estimators": [50, 100, 200],
                "max_depth": [5, 10, 15],
                "min_samples_split": [2, 5],
            },
            "LGBM": {
                "n_estimators": [50, 100, 200],
                "max_depth": [5, 10, 15],
                "learning_rate": [0.01, 0.1, 0.2],
            },
            "Ridge": {"alpha": [0.01, 0.1, 1.0, 10.0]},
        }

    def evaluate_model(self, model, X_train, X_test, y_train, y_test) -> Dict:
        """
        Evalúa un modelo y retorna las métricas.

        Parameters:
        -----------
        model : object
            Modelo entrenado
        X_train, X_test : pd.DataFrame
            Datos de entrenamiento y prueba
        y_train, y_test : pd.Series
            Targets de entrenamiento y prueba

        Returns:
        --------
        Dict
            Diccionario con las métricas calculadas
        """
        # Entrenar modelo
        model.fit(X_train, y_train)

        # Hacer predicciones
        y_pred = model.predict(X_test)

        # Calcular métricas
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}

    def run_comparison(
        self, test_size: float = 0.2, random_state: int = 42
    ) -> pd.DataFrame:
        """
        Ejecuta la comparación de modelos.

        Parameters:
        -----------
        test_size : float, default 0.2
            Proporción de datos para prueba
        random_state : int, default 42
            Semilla para reproducibilidad

        Returns:
        --------
        pd.DataFrame
            DataFrame con los resultados de la comparación
        """
        if self.data is None:
            self.load_data()

        # Preparar características
        X, y = self.prepare_features()

        # Dividir datos
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        print(f"\nDivisión de datos:")
        print(f"  Entrenamiento: {len(X_train)} muestras")
        print(f"  Prueba: {len(X_test)} muestras")

        # Obtener modelos y grids
        models = self.get_models()
        param_grids = self.get_param_grids()

        # Almacenar resultados
        results = []

        print(f"\nIniciando comparación de modelos...")
        print(f"Usando exógenas: {self.use_exogenous}")
        print("=" * 60)

        for model_name, model in models.items():
            print(f"\nEvaluando {model_name}...")

            # Obtener grid de parámetros
            param_grid = param_grids[model_name]

            # Hacer grid search simple (evaluar todas las combinaciones)
            best_score = float("inf")
            best_params = None
            best_metrics = None

            # Generar todas las combinaciones de parámetros
            from itertools import product

            param_names = list(param_grid.keys())
            param_values = list(param_grid.values())

            total_combinations = np.prod([len(v) for v in param_values])
            print(f"  Probando {total_combinations} combinaciones de parámetros...")

            for i, param_combination in enumerate(product(*param_values)):
                # Crear diccionario de parámetros
                params = dict(zip(param_names, param_combination))

                # Crear modelo con parámetros
                model_instance = model.__class__(**params)

                # Evaluar modelo
                try:
                    metrics = self.evaluate_model(
                        model_instance, X_train, X_test, y_train, y_test
                    )

                    # Usar RMSE como métrica principal
                    if metrics["rmse"] < best_score:
                        best_score = metrics["rmse"]
                        best_params = params.copy()
                        best_metrics = metrics.copy()

                    # Mostrar progreso cada 10 combinaciones
                    if (i + 1) % 10 == 0 or (i + 1) == total_combinations:
                        print(
                            f"    Progreso: {i + 1}/{total_combinations} - Mejor RMSE: {best_score:.4f}"
                        )

                except Exception as e:
                    print(f"    Error con parámetros {params}: {e}")
                    continue

            # Guardar resultados del mejor modelo
            if best_metrics is not None:
                result = {
                    "model": model_name,
                    "station": self.station,
                    "use_exogenous": self.use_exogenous,
                    "best_params": best_params,
                    **best_metrics,
                }
                results.append(result)

                print(f"  Mejor {model_name}:")
                print(f"    Parámetros: {best_params}")
                print(f"    RMSE: {best_metrics['rmse']:.4f}")
                print(f"    MAE: {best_metrics['mae']:.4f}")
                print(f"    R²: {best_metrics['r2']:.4f}")
            else:
                print(f"  Error: No se pudo evaluar {model_name}")

        # Crear DataFrame de resultados
        df_results = pd.DataFrame(results)

        if not df_results.empty:
            # Ordenar por RMSE
            df_results = df_results.sort_values("rmse").reset_index(drop=True)

            print(f"\n" + "=" * 60)
            print("RESULTADOS FINALES")
            print("=" * 60)
            print(df_results[["model", "rmse", "mae", "r2"]].to_string(index=False))

        return df_results


def main():
    """Función principal para ejecutar la comparación."""

    # Configuración
    STATIONS = ["GIR-EPM", "CEN-TRAF", "ITA-CJUS", "MED-FISC"]
    USE_EXOGENOUS_OPTIONS = [True, False]

    all_results = []

    for station in STATIONS:
        print(f"\n{'=' * 80}")
        print(f"PROCESANDO ESTACIÓN: {station}")
        print(f"{'=' * 80}")

        for use_exog in USE_EXOGENOUS_OPTIONS:
            print(f"\n{'=' * 40}")
            print(f"Configuración: {'Con' if use_exog else 'Sin'} exógenas")
            print(f"{'=' * 40}")

            try:
                # Crear comparador
                comparator = ModelComparator(station=station, use_exogenous=use_exog)

                # Ejecutar comparación
                results = comparator.run_comparison()

                if not results.empty:
                    all_results.append(results)
                    print(
                        f"\n✓ {station} - {'Con' if use_exog else 'Sin'} exógenas completado"
                    )
                else:
                    print(
                        f"\n✗ {station} - {'Con' if use_exog else 'Sin'} exógenas falló"
                    )

            except Exception as e:
                print(
                    f"\n✗ Error en {station} - {'Con' if use_exog else 'Sin'} exógenas: {e}"
                )
                continue

    # Consolidar todos los resultados
    if all_results:
        final_results = pd.concat(all_results, ignore_index=True)

        print(f"\n{'=' * 80}")
        print("RESULTADOS CONSOLIDADOS")
        print(f"{'=' * 80}")

        # Mostrar mejores modelos por estación y configuración
        best_results = (
            final_results.groupby(["station", "use_exogenous"]).first().reset_index()
        )
        print("\nMejores modelos por estación y configuración:")
        print(
            best_results[
                ["station", "use_exogenous", "model", "rmse", "mae", "r2"]
            ].to_string(index=False)
        )

        # Guardar resultados
        output_file = "model_comparison_results.csv"
        final_results.to_csv(output_file, index=False)
        print(f"\nResultados guardados en: {output_file}")

        # Mostrar ranking general
        print(f"\nRanking general (top 10):")
        top_10 = final_results.head(10)
        print(
            top_10[
                ["station", "use_exogenous", "model", "rmse", "mae", "r2"]
            ].to_string(index=False)
        )

    else:
        print("\n✗ No se obtuvieron resultados válidos")


if __name__ == "__main__":
    main()
