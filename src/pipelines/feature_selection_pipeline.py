"""
Clase reutilizable para feature selection que integra DataManager y SkforecastFeatureSelector.
Permite cargar datos enriched y realizar selección de características de forma automatizada.
"""

import os
import json
from typing import Optional, Dict, List, Tuple, Any
import pandas as pd

from ..recursos.data_manager import DataManager
from ..services.feature_selection import SkforecastFeatureSelector
from ..recursos.regressors import create_regressor
from ..recursos.scorers import wmape_scorer


class FeatureSelector:
    """
    Clase reutilizable para feature selection que integra DataManager y SkforecastFeatureSelector.

    Esta clase permite:
    - Cargar datos enriched usando DataManager
    - Realizar selección de características con diferentes algoritmos
    - Guardar resultados de selección (selected_lags, selected_window_features, selected_exog)
    - Configurar parámetros flexibles para diferentes estaciones
    """

    def __init__(
        self,
        data_path: str = "data/stage/SO2",
        output_path: str = "data/stage/SO2/selected",
        selector_type: str = "lasso",
        regressor_type: str = "lgbm",
        lags: int = 48,
        window_features: Optional[List] = None,
        selector_params: Optional[Dict] = None,
        regressor_params: Optional[Dict] = None,
        random_state: int = 15926,
    ):
        """
        Inicializa el FeatureSelector.

        Parameters:
        -----------
        data_path : str, default="data/stage/SO2"
            Ruta base donde están los datos enriched
        output_path : str, default="data/stage/SO2/selected"
            Ruta donde guardar los resultados de selección
        selector_type : str, default="lasso"
            Tipo de selector: "lasso" o "rfecv"
        regressor_type : str, default="lgbm"
            Tipo de regresor: "lgbm", "rf", "lasso", etc.
        lags : int, default=48
            Número de lags a considerar
        window_features : list, optional
            Lista de window features a usar
        selector_params : dict, optional
            Parámetros específicos para el selector
        regressor_params : dict, optional
            Parámetros específicos para el regresor
        random_state : int, default=15926
            Semilla para reproducibilidad
        """
        self.data_path = data_path
        self.output_path = output_path
        self.selector_type = selector_type
        self.regressor_type = regressor_type
        self.lags = lags
        self.window_features = window_features
        self.selector_params = selector_params or {}
        self.regressor_params = regressor_params or {}
        self.random_state = random_state

        # Inicializar componentes
        self.data_manager = DataManager()
        self._setup_selector()

        # Crear directorio de salida si no existe
        os.makedirs(self.output_path, exist_ok=True)

    def _setup_selector(self):
        """Configura el selector de características."""
        # Crear regresor
        regressor = create_regressor(
            regressor_type=self.regressor_type,
            random_state=self.random_state,
            **self.regressor_params,
        )

        # Configurar parámetros del selector
        selector_params = {
            "scorer": wmape_scorer,
            "cv_splits": 3,
            "random_state": self.random_state,
            **self.selector_params,
        }

        # Crear selector
        self.selector = SkforecastFeatureSelector(
            lags=self.lags,
            window_features=self.window_features,
            regressor=regressor,
            selector_type=self.selector_type,
            selector_params=selector_params,
            random_state=self.random_state,
        )

    def load_enriched_data(
        self, station: str, include_exog: bool = False
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Carga datos enriched para una estación específica.

        Parameters:
        -----------
        station : str
            Nombre de la estación (ej: "CEN-TRAF", "GIR-EPM", etc.)
        include_exog : bool, default=False
            Si incluir datos exógenos

        Returns:
        --------
        tuple
            (data_enriched, data_exog) donde data_exog puede ser None
        """
        # Cargar datos enriched
        enriched_path = os.path.join(
            self.data_path, "enrichment", f"enriched_{station}.csv"
        )

        if not os.path.exists(enriched_path):
            raise FileNotFoundError(f"No se encontró el archivo: {enriched_path}")

        data_enriched = self.data_manager.load_data(enriched_path)

        # Cargar datos exógenos si se solicita
        data_exog = None
        if include_exog:
            exog_path = os.path.join(
                self.data_path,
                "enrichment_exogenous",
                f"enriched_{station}_exogenous.csv",
            )

            if os.path.exists(exog_path):
                data_exog = self.data_manager.load_data(exog_path)

        return data_enriched, data_exog

    def prepare_data_for_selection(
        self, data: pd.DataFrame, target_column: str = "target"
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Prepara los datos para la selección de características.

        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame con datos enriched
        target_column : str, default="target"
            Nombre de la columna objetivo

        Returns:
        --------
        tuple
            (y, exog) donde y es la serie objetivo y exog son todas las demás variables
        """
        if target_column not in data.columns:
            raise KeyError(
                f"Columna objetivo '{target_column}' no encontrada en los datos"
            )

        # Separar target y exógenas (todo lo que no sea target es exógeno)
        y = data[target_column].copy()
        exog = data.drop(columns=[target_column]).copy()

        return y, exog

    def select_features_for_station(
        self,
        station: str,
        include_exog: bool = False,
        select_only: Optional[str] = None,
        force_inclusion: Optional[List] = None,
        subsample: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Realiza selección de características para una estación específica.

        Parameters:
        -----------
        station : str
            Nombre de la estación
        include_exog : bool, default=False
            Si incluir datos exógenos en la selección
        select_only : str, optional
            Seleccionar solo tipos específicos: 'lags', 'window_features', 'exog'
        force_inclusion : list, optional
            Características a forzar inclusión
        subsample : float, default=0.5
            Fracción de datos a usar para selección

        Returns:
        --------
        dict
            Diccionario con los resultados de selección
        """
        # Cargar datos
        data_enriched, data_exog = self.load_enriched_data(station, include_exog)

        # Preparar datos
        y, exog = self.prepare_data_for_selection(data_enriched)

        # Combinar exógenas si hay datos exógenos adicionales
        if include_exog and data_exog is not None:
            exog = pd.concat([exog, data_exog], axis=1)

        # Realizar selección
        selected_lags, selected_window_features, selected_exog = (
            self.selector.select_features(
                y=y,
                exog=exog,
                select_only=select_only,
                force_inclusion=force_inclusion,
                subsample=subsample,
                verbose=False,
            )
        )

        # Preparar resultados
        results = {
            "station": station,
            "selector_type": self.selector_type,
            "regressor_type": self.regressor_type,
            "lags": self.lags,
            "selected_lags": selected_lags,
            "selected_window_features": selected_window_features,
            "selected_exog": selected_exog,
            "n_selected_lags": len(selected_lags) if selected_lags else 0,
            "n_selected_window_features": len(selected_window_features)
            if selected_window_features
            else 0,
            "n_selected_exog": len(selected_exog) if selected_exog else 0,
            "total_features": (len(selected_lags) if selected_lags else 0)
            + (len(selected_window_features) if selected_window_features else 0)
            + (len(selected_exog) if selected_exog else 0),
        }

        return results

    def save_selection_results(
        self, results: Dict[str, Any], station: str, suffix: str = ""
    ) -> str:
        """
        Guarda los resultados de selección en archivos JSON.

        Parameters:
        -----------
        results : dict
            Resultados de la selección
        station : str
            Nombre de la estación
        suffix : str, default=""
            Sufijo para el nombre del archivo

        Returns:
        --------
        str
            Ruta del archivo guardado
        """
        # Crear nombre del archivo
        filename = f"selected_cols_{station}_{self.selector_type}_{self.regressor_type}"
        if suffix:
            filename += f"_{suffix}"
        filename += ".json"

        filepath = os.path.join(self.output_path, filename)

        # Guardar resultados
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        return filepath

    def run_selection_pipeline(
        self,
        stations: List[str],
        include_exog: bool = False,
        select_only: Optional[str] = None,
        force_inclusion: Optional[List] = None,
        subsample: float = 0.5,
        save_results: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Ejecuta el pipeline completo de selección para múltiples estaciones.

        Parameters:
        -----------
        stations : list
            Lista de nombres de estaciones
        include_exog : bool, default=False
            Si incluir datos exógenos
        select_only : str, optional
            Seleccionar solo tipos específicos
        force_inclusion : list, optional
            Características a forzar inclusión
        subsample : float, default=0.5
            Fracción de datos a usar
        save_results : bool, default=True
            Si guardar resultados en archivos

        Returns:
        --------
        dict
            Diccionario con resultados por estación
        """
        all_results = {}

        for station in stations:
            try:
                # Realizar selección
                results = self.select_features_for_station(
                    station=station,
                    include_exog=include_exog,
                    select_only=select_only,
                    force_inclusion=force_inclusion,
                    subsample=subsample,
                )

                # Guardar resultados si se solicita
                if save_results:
                    self.save_selection_results(results, station)

                all_results[station] = results

            except Exception as e:
                error_msg = f"Error procesando estación {station}: {str(e)}"
                all_results[station] = {"error": error_msg}

        return all_results

    def get_feature_importance(self) -> Dict[str, Any]:
        """
        Obtiene la importancia de características del selector.

        Returns:
        --------
        dict
            Información de importancia de características
        """
        return self.selector.get_feature_importance()
