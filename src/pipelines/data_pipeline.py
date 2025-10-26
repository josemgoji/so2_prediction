import os
import pandas as pd
from pathlib import Path
from src.services.data_builder import DataBuilderService
from src.services.outlier_cleaner import DataCleaning
from src.services.imputation import ImputationService
from src.constants.parsed_fields import STAGE_DIR


class DataPipeline:
    def __init__(self):
        self.dm = DataBuilderService()
        self.cleaner = DataCleaning(contamination=0.01, random_state=42)
        self.imputer = ImputationService()

    def run(self, pollutant: str, station: str, pollutant_path, meteo_path):
        df = self.dm.load(
            target_col=station,
            pollutant_path=pollutant_path,
            meteo_path=meteo_path,
            stage_dir=STAGE_DIR,
            pollutant_name=pollutant,
        )

        # Guardar el DataFrame completo (target + exógenas) para trazabilidad
        merged_folder_path = STAGE_DIR / f"{pollutant}/merged/"
        os.makedirs(merged_folder_path, exist_ok=True)
        self.dm.data_resource.save(
            df, file_path=merged_folder_path / f"merged_{station}.csv"
        )
        print(
            f"DataFrame completo guardado en: {merged_folder_path / f'merged_{station}.csv'}"
        )

        # Limpiar los datos y obtener el resultado
        result = self.cleaner.clean_data(df)

        # Acceder al DataFrame limpio
        df_clean = result.df

        # Crear carpeta para guardar imágenes
        images_folder = Path("images") / f"{pollutant}_{station}"
        images_folder.mkdir(parents=True, exist_ok=True)

        # Graficar outliers univariables
        uni_save_path = (
            images_folder / f"outliers_univariables_{station}_{pollutant}.png"
        )
        result.plot_univariate_outliers(
            column="target",
            x_label="Fecha",
            y_label=pollutant,
            title=f"Outliers Univariables en {station} {pollutant}",
            save_path=str(uni_save_path),
        )

        # Graficar outliers multivariables
        multi_save_path = (
            images_folder / f"outliers_multivariables_{station}_{pollutant}.png"
        )
        result.plot_multivariate_outliers(
            outlier_col="Curtosis_outlier",
            pollutant=pollutant,
            station=station,
            save_path=str(multi_save_path),
        )

        # Imputar valores faltantes
        imputation_result = self.imputer.impute(df_clean)

        # Acceder al DataFrame imputado
        df_imputed = imputation_result.df

        # Graficar imputación
        imp_save_path = images_folder / f"imputation_{station}_{pollutant}.html"
        self.imputer.plot_imputation(
            df_clean,
            df_imputed,
            pollutant=pollutant,
            station=station,
            save_path=str(imp_save_path),
        )

        # Eliminar columnas de outliers antes de guardar
        outlier_columns = [
            col for col in df_imputed.columns if "outlier" in col.lower()
        ]
        if outlier_columns:
            df_imputed = df_imputed.drop(columns=outlier_columns)

        # Crear carpeta para el contaminante y estación
        folder_path = STAGE_DIR / f"{pollutant}/clean/"
        os.makedirs(folder_path, exist_ok=True)

        # Guardar el DataFrame imputado
        self.dm.data_resource.save(df_imputed, file_path=folder_path / f"{station}.csv")

        return df_imputed

    def save(self, df: pd.DataFrame, output_path: str = None) -> None:
        """
        Guarda el DataFrame en la ruta especificada.
        """
        if output_path is None:
            # Usar la ruta por defecto
            output_path = str(STAGE_DIR / "SO2/clean/")

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Resetear índice para guardar como CSV
        df_to_save = df.reset_index()
        df_to_save.to_csv(output_file, index=False)
        print(f"Saved data: {output_file}")
