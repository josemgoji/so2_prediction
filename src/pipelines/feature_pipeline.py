import pandas as pd
import json
from pathlib import Path

from src.services.feature_engineering import FeatureEngineeringService
from src.services.feature_selection import FeatureSelectionService
from src.services.data_manager import DataManager
from src.constants.parsed_fields import (
    SO2_COL,
    STAGE_DIR,
)


class FeaturePipeline:
    """
    Pipeline que lee datos limpios de data/stage/SO2, crea features y selecciona características.
    Puede usar variables exógenas si están habilitadas.
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
        self.file_path = STAGE_DIR / f"{pollutant}/clean/" / f"{station}.csv"
        self.feature_engineering = FeatureEngineeringService()
        self.feature_selection = FeatureSelectionService(use_exogenous=use_exogenous)
        self.data_manager = DataManager()

    def load_data(self) -> pd.DataFrame:
        """
        Carga el CSV limpio y opcionalmente enriquece con datos exógenos.
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {self.file_path}")

        if self.use_exogenous:
            # Cargar datos con variables exógenas completos
            print(f"Loading data with exogenous variables for station: {self.station}")
            df = pd.read_csv(self.file_path)
        else:
            # Cargar solo datos básicos
            print(f"Loading basic data for station: {self.station}")
            df = pd.read_csv(self.file_path, usecols=["datetime", "target"])

        print(f"Loaded data shape: {df.shape}")
        print(f"Target nulls: {df['target'].isnull().sum()}")
        return df

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea features usando FeatureEngineeringService.
        """
        print("Creating features with FeatureEngineeringService...")
        df_feat = self.feature_engineering.build_features_from_df(
            df, time_col_name="datetime", target_col_name="target", stl_period=168
        )
        print(f"Features created: {len(df_feat.columns)} columns")
        return df_feat

    def select_features(
        self, df_feat: pd.DataFrame, method: str = "rfecv", metric: str = "mape"
    ) -> list[str]:
        """
        Selecciona features usando el método especificado.
        method: 'rfecv' | 'lasso' | 'lasso_grid'
        metric: 'wmape' | 'mape' | 'rmse'
        """
        print(f"Using method: {method} (metric: {metric})")

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

        if method == "rfecv":
            selected_cols = self.feature_selection.select_features_rfecv(
                X, y, metric=metric
            )
        elif method == "lasso":
            selected_cols = self.feature_selection.select_features_lasso(
                X, y, metric=metric
            )
        elif method == "lasso_grid":
            selected_cols, best_alpha, coefs, _ = (
                self.feature_selection.select_features_lasso_grid(X, y, metric=metric)
            )
            print(f"[lasso_grid] best_alpha={best_alpha} selected={len(selected_cols)}")
        else:
            raise ValueError("method must be 'rfecv', 'lasso', or 'lasso_grid'")

        print(f"Selected features: {len(selected_cols)}")
        return selected_cols

    def run(self, method: str = "rfecv", metric: str = "mape") -> pd.DataFrame:
        """
        Ejecuta el pipeline completo: carga, features, selección.
        Retorna el DataFrame con features seleccionadas.
        Si el archivo de features seleccionadas ya existe, lo carga.
        Si el archivo de features engineered existe, salta la creación y va directo a selección.
        """
        # Definir rutas basadas en si se usan variables exógenas
        exogenous_suffix = "_exogenous" if self.use_exogenous else ""

        selected_features_dir = (
            STAGE_DIR / f"{self.pollutant}/selected{exogenous_suffix}/"
        )

        # Verificar si ya existe el JSON de features seleccionadas
        selected_cols_file = (
            selected_features_dir
            / f"selected_cols_{self.station}_{method}_{metric}{exogenous_suffix}.json"
        )

        if selected_cols_file.exists():
            print(
                f"Selected features JSON already exists, loading: {selected_cols_file}"
            )
            # Cargar el JSON y crear el DataFrame final
            with open(selected_cols_file, "r") as f:
                selected_cols = json.load(f)

            # Cargar datos enriquecidos
            enrichment_dir = (
                STAGE_DIR / f"{self.pollutant}/enrichment{exogenous_suffix}/"
            )
            enrichment_file = (
                enrichment_dir / f"enriched_{self.station}{exogenous_suffix}.csv"
            )
            df_feat = pd.read_csv(enrichment_file)

            # Crear DataFrame final con features seleccionadas
            final_cols = ["datetime", "target"] + selected_cols
            df_final = df_feat[final_cols].copy()
            return df_final

        # Si no existe, crear features y guardar
        enrichment_dir = STAGE_DIR / f"{self.pollutant}/enrichment{exogenous_suffix}/"
        enrichment_file = (
            enrichment_dir / f"enriched_{self.station}{exogenous_suffix}.csv"
        )

        df = self.load_data()

        if enrichment_file.exists():
            print(f"Enriched features file already exists, loading: {enrichment_file}")
            df_feat = pd.read_csv(enrichment_file)
        else:
            df_feat = self.create_features(df)
            enrichment_dir.mkdir(parents=True, exist_ok=True)
            df_feat.to_csv(enrichment_file, index=False)
            print(f"Saved enriched features: {enrichment_file}")

        selected_cols = self.select_features(df_feat, method, metric)

        # Save selected columns list
        selected_features_dir.mkdir(parents=True, exist_ok=True)
        selected_cols_file = (
            selected_features_dir
            / f"selected_cols_{self.station}_{method}_{metric}{exogenous_suffix}.json"
        )
        with open(selected_cols_file, "w") as f:
            json.dump(selected_cols, f)
        print(f"Saved selected columns: {selected_cols_file}")

        # Crear DataFrame final con datetime, target, y features seleccionadas
        final_cols = ["datetime", "target"] + selected_cols
        df_final = df_feat[final_cols].copy()

        return df_final

    def save(
        self,
        df_final: pd.DataFrame,
        output_dir: str = None,
        method: str = "rfecv",
        metric: str = "mape",
    ):
        """
        Solo guarda el JSON con las columnas seleccionadas, no el CSV.
        """
        # Definir rutas basadas en si se usan variables exógenas
        exogenous_suffix = "_exogenous" if self.use_exogenous else ""

        if output_dir is None:
            output_dir = str(
                STAGE_DIR / f"{self.pollutant}/selected{exogenous_suffix}/"
            )
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Solo guardar el JSON, no el CSV
        print(
            f"JSON ya guardado en: {output_path / f'selected_cols_{self.station}_{method}_{metric}{exogenous_suffix}.json'}"
        )

    def run_with_integrated_selection(
        self, method: str = "lasso_grid", metric: str = "wmape"
    ) -> pd.DataFrame:
        """
        Ejecuta el pipeline usando el nuevo método integrado de FeatureSelectionService.
        Este método usa la funcionalidad completa de datos exógenos y feature engineering.
        """
        print("=" * 60)
        print("RUNNING INTEGRATED FEATURE SELECTION PIPELINE")
        print("=" * 60)

        # Definir rutas basadas en si se usan variables exógenas
        exogenous_suffix = "_exogenous" if self.use_exogenous else ""
        selected_features_dir = (
            STAGE_DIR / f"{self.pollutant}/selected{exogenous_suffix}/"
        )
        selected_file_name = (
            f"features_{self.station}_{method}_{metric}{exogenous_suffix}.csv"
        )
        selected_file_path = selected_features_dir / selected_file_name

        # Si el archivo ya existe, cargarlo
        if selected_file_path.exists():
            print(
                f"Selected features file already exists, loading: {selected_file_path}"
            )
            return pd.read_csv(selected_file_path)

        # Usar el nuevo método integrado
        selected_features, metadata = self.feature_selection.run_complete_pipeline(
            data_path=str(self.file_path),
            output_path=str(selected_file_path),
            method=method,
            target_col="target",
            metric=metric,
            use_exogenous=self.use_exogenous,
        )

        # Guardar metadata adicional
        metadata_file = (
            selected_features_dir
            / f"metadata_{self.station}_{method}_{metric}{exogenous_suffix}.json"
        )
        with open(metadata_file, "w") as f:
            json.dump(
                {
                    "selected_features": selected_features,
                    "method": metadata.get("method"),
                    "metric": metadata.get("metric"),
                    "best_alpha": metadata.get("best_alpha"),
                    "use_exogenous": self.use_exogenous,
                    "station": self.station,
                    "pollutant": self.pollutant,
                },
                f,
                indent=2,
            )

        print(f"Saved metadata: {metadata_file}")

        # Cargar y retornar el resultado
        return pd.read_csv(selected_file_path)
