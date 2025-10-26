import pandas as pd
import json
from pathlib import Path

from src.services.feature_selection import FeatureSelectionService
from src.recursos.selectors import FeatureSelectorFactory
from src.recursos.scorers import wmape_scorer, rmse_scorer
from src.constants.parsed_fields import (
    SO2_COL,
    STAGE_DIR,
)


class FeatureSelectionPipeline:
    """
    Pipeline que maneja la selección de características y guarda los metadatos
    de las features seleccionadas (selected_lags, selected_window_features, selected_exog)
    en formato JSON para uso posterior.
    """

    def __init__(
        self,
        station: str,
        pollutant: str,
        stage_dir: str = "data/stage",
        use_exogenous: bool = True,
    ):
        self.station = station
        self.pollutant = pollutant
        self.stage_dir = Path(stage_dir)
        self.use_exogenous = use_exogenous
        self.feature_selection = FeatureSelectionService(use_exogenous=use_exogenous)

    def load_enriched_data(self) -> pd.DataFrame:
        """
        Carga los datos enriquecidos del feature engineering pipeline.
        """
        # Definir rutas basadas en si se usan variables exógenas
        exogenous_suffix = "_exogenous" if self.use_exogenous else ""
        enrichment_file = (
            STAGE_DIR
            / f"{self.pollutant}/enrichment{exogenous_suffix}/"
            / f"enriched_{self.station}{exogenous_suffix}.csv"
        )

        if not enrichment_file.exists():
            raise FileNotFoundError(
                f"Archivo enriquecido no encontrado: {enrichment_file}"
            )

        print(f"Loading enriched data for station: {self.station}")
        df = pd.read_csv(enrichment_file)
        print(f"Loaded enriched data shape: {df.shape}")
        return df

    def select_features(
        self,
        df_feat: pd.DataFrame,
        method: str = "rfecv",
        scorer=None,
        lags: int = 48,
        window_features=None,
    ) -> dict:
        """
        Selecciona features usando el sistema de selectores y retorna metadatos.

        Parameters:
        -----------
        df_feat : pd.DataFrame
            DataFrame con features enriquecidas
        method : str, default="rfecv"
            Método de selección: 'rfecv' | 'lasso'
        scorer : callable, default=None
            Scorer de sklearn (ej: wmape_scorer, rmse_scorer, "mae", etc.)
        lags : int, default=48
            Número de lags a considerar
        window_features : list, default=None
            Lista de window features a considerar

        Returns:
        --------
        dict
            Diccionario con metadatos de features seleccionadas
        """
        # Usar scorer por defecto si no se especifica
        if scorer is None:
            scorer = wmape_scorer

        print(f"Using method: {method} with scorer: {scorer}")

        # Preparar X e y
        exclude_cols = {
            "target",
            "datetime",
            SO2_COL,
        }  # excluir target, tiempo y SO2 original si existe
        cols_to_drop = [c for c in df_feat.columns if c in exclude_cols]
        X = df_feat.drop(columns=cols_to_drop)
        y = df_feat["target"].astype(float)

        # Imputar NaNs simples
        if X.isna().any().any():
            med = X.median(numeric_only=True)
            X = X.fillna(med).fillna(-1)

        # Usar el sistema de selectores
        if method in ["rfecv", "lasso"]:
            # Crear selector usando la factory
            selector = FeatureSelectorFactory.create_selector(
                selector_type=method,
                scorer=scorer,
                cv_splits=3,
                n_jobs=-1,
                random_state=123,
            )

            # Ajustar selector
            selector.fit(X, y)

            # Obtener características seleccionadas
            selected_mask = selector.get_support()
            selected_cols = X.columns[selected_mask].tolist()

            # Log adicional para lasso
            if method == "lasso" and hasattr(selector, "best_alpha_"):
                print(
                    f"[lasso] best_alpha={selector.best_alpha_} selected={len(selected_cols)}"
                )

        else:
            raise ValueError("method must be 'rfecv' or 'lasso'")

        print(f"Selected features: {len(selected_cols)}")

        # Categorizar las features seleccionadas
        metadata = self._categorize_selected_features(
            selected_cols, lags, window_features
        )

        # Agregar información adicional
        metadata.update(
            {
                "method": method,
                "scorer": getattr(scorer, "__name__", str(scorer)),
                "total_selected": len(selected_cols),
                "best_alpha": getattr(selector, "best_alpha_", None)
                if method == "lasso"
                else None,
                "use_exogenous": self.use_exogenous,
                "station": self.station,
                "pollutant": self.pollutant,
            }
        )

        return metadata

    def _categorize_selected_features(
        self, selected_cols: list, lags: int, window_features: list = None
    ) -> dict:
        """
        Categoriza las features seleccionadas en lags, window features y exógenas.

        Parameters:
        -----------
        selected_cols : list
            Lista de columnas seleccionadas
        lags : int
            Número de lags considerados
        window_features : list, optional
            Lista de window features consideradas

        Returns:
        --------
        dict
            Diccionario con features categorizadas
        """
        selected_lags = []
        selected_window_features = []
        selected_exog = []

        # Identificar lags (formato: lag_1, lag_2, etc.)
        for col in selected_cols:
            if col.startswith("lag_"):
                try:
                    lag_num = int(col.split("_")[1])
                    if 1 <= lag_num <= lags:
                        selected_lags.append(lag_num)
                except (ValueError, IndexError):
                    pass

        # Identificar window features (formato: feature_window_function)
        window_feature_patterns = [
            "rolling_mean",
            "rolling_std",
            "rolling_min",
            "rolling_max",
            "fourier",
            "stl",
            "trend",
            "seasonal",
            "resid",
        ]

        for col in selected_cols:
            is_window_feature = False
            for pattern in window_feature_patterns:
                if pattern in col.lower():
                    selected_window_features.append(col)
                    is_window_feature = True
                    break

            # Si no es lag ni window feature, es exógena
            if not col.startswith("lag_") and not is_window_feature:
                selected_exog.append(col)

        return {
            "selected_lags": sorted(selected_lags),
            "selected_window_features": selected_window_features,
            "selected_exog": selected_exog,
        }

    def run(
        self,
        method: str = "rfecv",
        scorer=None,
        lags: int = 48,
        window_features=None,
    ) -> dict:
        """
        Ejecuta el pipeline de selección de características completo.
        Retorna los metadatos de features seleccionadas.
        Si el archivo de metadatos ya existe, lo carga.

        Parameters:
        -----------
        method : str, default="rfecv"
            Método de selección: "rfecv" o "lasso"
        scorer : callable, default=None
            Scorer de sklearn (ej: wmape_scorer, rmse_scorer, "mae", etc.)
        lags : int, default=48
            Número de lags a considerar
        window_features : list, default=None
            Lista de window features a considerar

        Returns:
        --------
        dict
            Metadatos de features seleccionadas
        """
        print("=" * 60)
        print("RUNNING FEATURE SELECTION PIPELINE")
        print("=" * 60)

        # Usar scorer por defecto si no se especifica
        if scorer is None:
            scorer = wmape_scorer

        # Crear nombre de archivo basado en el scorer
        scorer_name = getattr(scorer, "__name__", str(scorer))
        if hasattr(scorer, "__name__"):
            scorer_name = scorer.__name__
        elif isinstance(scorer, str):
            scorer_name = scorer
        else:
            scorer_name = "custom"

        # Definir rutas basadas en si se usan variables exógenas
        exogenous_suffix = "_exogenous" if self.use_exogenous else ""
        selected_features_dir = (
            STAGE_DIR / f"{self.pollutant}/selected{exogenous_suffix}/"
        )

        # Verificar si ya existe el JSON de metadatos
        metadata_file = (
            selected_features_dir
            / f"metadata_{self.station}_{method}_{scorer_name}{exogenous_suffix}.json"
        )

        if metadata_file.exists():
            print(
                f"Feature selection metadata already exists, loading: {metadata_file}"
            )
            with open(metadata_file, "r") as f:
                return json.load(f)

        # Cargar datos enriquecidos
        df_feat = self.load_enriched_data()

        # Seleccionar features
        metadata = self.select_features(df_feat, method, scorer, lags, window_features)

        # Guardar metadatos
        selected_features_dir.mkdir(parents=True, exist_ok=True)
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved feature selection metadata: {metadata_file}")

        return metadata

    def save(self, metadata: dict, output_path: str = None) -> None:
        """
        Guarda los metadatos de features seleccionadas.
        """
        if output_path is None:
            # Definir rutas basadas en si se usan variables exógenas
            exogenous_suffix = "_exogenous" if self.use_exogenous else ""
            method = metadata.get("method", "rfecv")
            scorer_name = metadata.get("scorer", "wmape_scorer")
            output_path = str(
                STAGE_DIR
                / f"{self.pollutant}/selected{exogenous_suffix}/"
                / f"metadata_{self.station}_{method}_{scorer_name}{exogenous_suffix}.json"
            )

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved feature selection metadata: {output_file}")

    def load_metadata(
        self, method: str = "rfecv", scorer_name: str = "wmape_scorer"
    ) -> dict:
        """
        Carga metadatos de features seleccionadas previamente guardados.

        Parameters:
        -----------
        method : str, default="rfecv"
            Método de selección usado
        scorer_name : str, default="wmape_scorer"
            Nombre del scorer usado

        Returns:
        --------
        dict
            Metadatos de features seleccionadas
        """
        # Definir rutas basadas en si se usan variables exógenas
        exogenous_suffix = "_exogenous" if self.use_exogenous else ""
        selected_features_dir = (
            STAGE_DIR / f"{self.pollutant}/selected{exogenous_suffix}/"
        )

        metadata_file = (
            selected_features_dir
            / f"metadata_{self.station}_{method}_{scorer_name}{exogenous_suffix}.json"
        )

        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

        with open(metadata_file, "r") as f:
            return json.load(f)
