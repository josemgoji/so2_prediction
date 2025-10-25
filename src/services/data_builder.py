import pandas as pd
from src.constants.parsed_fields import (
    PATH,
    DATE_COLUMN,
    SAVE_PATH,
    SO2_PATH,
    METEO_PATH,
    STAGE_DIR,
)
from src.recursos.data_manager import DataManager as DataManagerResource


class DataBuilderService:
    """
    Servicio para unificar y combinar datos principales con exógenos.
    Utiliza recursos para la carga básica y se enfoca en la lógica de unión y transformación.
    """

    def __init__(
        self, path: str = PATH, date_col: str = DATE_COLUMN, save_path: str = SAVE_PATH
    ) -> None:
        self.path = path
        self.date_col = date_col
        self.save_path = save_path
        self.data_resource = DataManagerResource(date_col)

    def load(
        self,
        target_col: str,
        use_exogenous: bool = True,
        pollutant_path: str = SO2_PATH,
        meteo_path: str = METEO_PATH,
        stage_dir: str = STAGE_DIR,
        pollutant_name: str = "SO2",
        include_other_stations: bool = False,
    ) -> pd.DataFrame:
        """
        Carga datos principales y opcionalmente los combina con datos exógenos.
        Utiliza recursos para la carga básica y se enfoca en la lógica de unión.

        Parameters:
        - target_col: Nombre de la columna target
        - use_exogenous: Si True, carga y combina datos exógenos. Si False, solo procesa el CSV original
        - pollutant_path: Path to the pollutant data CSV
        - meteo_path: Path to the meteorological data CSV
        - stage_dir: Directory to save the processed pickles
        - pollutant_name: Name of the pollutant (e.g., 'SO2')
        - include_other_stations: Si True, incluye datos de contaminantes de otras estaciones. Si False, solo incluye datos meteorológicos
        """
        # Usar el recurso para cargar solo la columna target
        df = self.data_resource.load_target(pollutant_path, target_col, self.date_col)

        if use_exogenous:
            # Mapear nombres de estaciones a prefijos en los datos meteorológicos
            station_mapping = {
                "CEN-TRAF": "CEN-TRAF",
                "GIR-EPM": "V-GIR-EPM",
                "MED-FISC": "MED-FISC",
                "ITA-CJUS": "ITA-CJUS",
            }

            # Obtener el prefijo de la estación objetivo
            station_prefix = station_mapping.get(target_col, target_col)

            # Usar el recurso para cargar datos meteorológicos
            meteo_df = self.data_resource.load_exogenous_data(
                meteo_path, station_prefix
            )

            # Usar el recurso para cargar datos de contaminantes
            pollutant_df = self.data_resource.load_exogenous_data(pollutant_path)

            # Alinear los índices de tiempo - usar solo el rango común
            common_start = max(
                df.index.min(), meteo_df.index.min(), pollutant_df.index.min()
            )
            common_end = min(
                df.index.max(), meteo_df.index.max(), pollutant_df.index.max()
            )

            # Filtrar todos los datasets al rango común
            df_aligned = df.loc[common_start:common_end]
            meteo_aligned = meteo_df.loc[common_start:common_end]
            pollutant_aligned = pollutant_df.loc[common_start:common_end]

            # Combinar datos: target + meteorológicos + contaminantes (opcional)
            merged_df = df_aligned.join(meteo_aligned, how="left")

            # Agregar variables de pollutant según configuración
            if include_other_stations:
                # Agregar variables de pollutant para TODAS las estaciones
                for col in pollutant_aligned.columns:
                    merged_df[f"Pollutant_{pollutant_name}_{col}"] = pollutant_aligned[
                        col
                    ]

            merged_df = merged_df.asfreq("h")
            return merged_df
        else:
            # Solo retornar el DataFrame original sin datos exógenos
            # Ya tiene datetime como índice
            return df
