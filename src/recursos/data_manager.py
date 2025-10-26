import pandas as pd
import numpy as np
from typing import Optional


class DataManager:
    """
    Recurso para cargar y transformar CSVs básicos a time series.
    Se encarga únicamente de la carga y transformación básica de datos.
    """

    def __init__(self, date_column: str = "datetime") -> None:
        """
        Inicializa el DataManager.

        Parameters:
        - date_column: Nombre de la columna de fecha en el CSV
        """
        self.date_column = date_column

    def load_data(
        self, file_path: str, date_column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Carga todo el CSV y convierte la columna datetime especificada a índice.

        Parameters:
        - file_path: Ruta al archivo CSV
        - date_column: Nombre de la columna de fecha (opcional, usa self.date_column por defecto)

        Returns:
        - DataFrame completo con datetime como índice
        """
        # Usar date_column del parámetro o del constructor
        date_col = date_column if date_column is not None else self.date_column

        # Cargar el CSV completo
        df = pd.read_csv(file_path)

        # Validar que la columna de fecha existe
        if date_col not in df.columns:
            raise KeyError(
                f"Columna de fecha '{date_col}' no existe en el CSV. "
                f"Columnas disponibles: {list(df.columns)}"
            )

        # Convertir la columna de fecha a datetime
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

        # Ordenar por fecha y establecer datetime como índice
        df = df.sort_values(date_col)
        df.set_index(date_col, inplace=True)

        # Establecer frecuencia si no está definida
        if df.index.freq is None:
            # Inferir frecuencia automáticamente
            inferred_freq = pd.infer_freq(df.index)
            if inferred_freq is not None:
                df.index.freq = inferred_freq
            else:
                # Si no se puede inferir, verificar si los datos ya están limpios
                # (sin valores faltantes) antes de aplicar asfreq
                if df.isnull().sum().sum() == 0:
                    # Los datos ya están limpios, intentar establecer frecuencia de manera robusta
                    try:
                        # Intentar establecer frecuencia directamente
                        df.index.freq = 'h'
                    except ValueError:
                        # Si falla, crear un nuevo DatetimeIndex con frecuencia
                        try:
                            # Crear un nuevo índice con frecuencia
                            new_index = pd.DatetimeIndex(df.index, freq='h')
                            df.index = new_index
                        except Exception:
                            # Si todo falla, continuar sin frecuencia
                            pass
                else:
                    # Los datos tienen valores faltantes, aplicar asfreq para crear gaps
                    df = df.asfreq("h")
                
                # Asegurar que la frecuencia esté establecida
                if df.index.freq is None:
                    try:
                        # Crear un nuevo índice con frecuencia
                        new_index = pd.DatetimeIndex(df.index, freq='h', copy=False)
                        df.index = new_index
                    except Exception:
                        # Si falla, intentar establecer frecuencia directamente
                        try:
                            df.index.freq = 'h'
                        except Exception:
                            # Si todo falla, usar asfreq para establecer frecuencia
                            try:
                                df = df.asfreq('h', fill_method=None)
                            except Exception:
                                # Si todo falla, usar asfreq con fill_method='ffill'
                                try:
                                    df = df.asfreq('h', fill_method='ffill')
                                except Exception:
                                    # Si todo falla, usar asfreq con fill_method='bfill'
                                    try:
                                        df = df.asfreq('h', fill_method='bfill')
                                    except Exception:
                                        # Si todo falla, usar asfreq con fill_method='nearest'
                                        try:
                                            df = df.asfreq('h', fill_method='nearest')
                                        except Exception:
                                            pass

        return df

    def load_target(
        self, file_path: str, target_column: str, date_column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Carga solo la columna target con datetime como índice.

        Parameters:
        - file_path: Ruta al archivo CSV
        - target_column: Nombre de la columna objetivo
        - date_column: Nombre de la columna de fecha (opcional, usa self.date_column por defecto)

        Returns:
        - DataFrame con solo la columna target y datetime como índice
        """
        # Usar date_column del parámetro o del constructor
        date_col = date_column if date_column is not None else self.date_column

        # Cargar el CSV
        df = pd.read_csv(file_path)

        # Validar que las columnas existen
        if date_col not in df.columns:
            raise KeyError(
                f"Columna de fecha '{date_col}' no existe en el CSV. "
                f"Columnas disponibles: {list(df.columns)}"
            )

        if target_column not in df.columns:
            raise KeyError(
                f"Columna objetivo '{target_column}' no existe en el CSV. "
                f"Columnas disponibles: {list(df.columns)}"
            )

        # Seleccionar solo las columnas necesarias
        df = df[[date_col, target_column]].copy()

        # Convertir a tipos apropiados
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df[target_column] = pd.to_numeric(df[target_column], errors="coerce")

        # Estandarizar nombres de columnas
        df = df.rename(columns={date_col: "datetime", target_column: "target"})

        # Ordenar por fecha y establecer datetime como índice
        df = df.sort_values("datetime")
        df.set_index("datetime", inplace=True)

        # Establecer frecuencia si no está definida
        if df.index.freq is None:
            # Inferir frecuencia automáticamente
            inferred_freq = pd.infer_freq(df.index)
            if inferred_freq is not None:
                df.index.freq = inferred_freq
            else:
                # Si no se puede inferir, crear un nuevo índice con frecuencia
                # basado en el patrón de los datos
                df = df.asfreq("h")

        return df

    def load_exogenous_data(
        self, file_path: str, station_prefix: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Carga datos exógenos con doble encabezado y los procesa.

        Parameters:
        - file_path: Ruta al archivo CSV con datos exógenos
        - station_prefix: Prefijo de la estación para filtrar columnas (opcional)

        Returns:
        - DataFrame procesado con datetime como índice
        """
        # Cargar el CSV con low_memory=False para evitar warnings
        df = pd.read_csv(file_path, low_memory=False)

        if station_prefix:
            # Filtrar columnas que pertenecen a la estación específica
            station_columns = [
                col
                for col in df.columns
                if col.startswith(station_prefix) or col == "est"
            ]

            if len(station_columns) <= 1:  # Solo "est" o ninguna columna
                raise ValueError(
                    f"No se encontraron datos para la estación con prefijo '{station_prefix}'"
                )

            # Crear DataFrame solo con datos de la estación específica
            df = df[station_columns].copy()

            # Obtener nombres de variables de la primera fila (excluyendo "est")
            variable_names = list(df.iloc[0][1:])

            # Crear nombres de columnas limpias
            df.columns = ["datetime"] + variable_names
            df.drop([0, 1], inplace=True)  # Eliminar filas de nombres de variables
            df.reset_index(drop=True, inplace=True)
        else:
            # Procesar todas las estaciones (comportamiento general)
            # Verificar si existe la columna 'est'
            if "est" in df.columns:
                station_columns = [col for col in df.columns if col != "est"]
                df = df[["est"] + station_columns].copy()

                # Cambiar nombres de columnas usando la primera fila
                variable_names = list(df.iloc[0][1:])

                # Crear nombres únicos para evitar duplicados
                unique_columns = []
                column_counts = {}

                for var_name in variable_names:
                    if var_name in column_counts:
                        column_counts[var_name] += 1
                        unique_name = f"{var_name}_{column_counts[var_name]}"
                    else:
                        column_counts[var_name] = 1
                        unique_name = var_name
                    unique_columns.append(unique_name)

                df.columns = ["datetime"] + unique_columns
                df.drop([0, 1], inplace=True)
                df.reset_index(drop=True, inplace=True)
            else:
                # Si no hay columna 'est', asumir que la primera columna es datetime
                # y el resto son variables
                df.columns = ["datetime"] + list(df.columns[1:])
                df.drop([0, 1], inplace=True)
                df.reset_index(drop=True, inplace=True)

        # Convertir datetime y establecer como índice
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.set_index("datetime", inplace=True)

        # Establecer frecuencia si no está definida
        if df.index.freq is None:
            # Inferir frecuencia automáticamente
            inferred_freq = pd.infer_freq(df.index)
            if inferred_freq is not None:
                df.index.freq = inferred_freq
            else:
                # Si no se puede inferir, crear un nuevo índice con frecuencia
                # basado en el patrón de los datos
                df = df.asfreq("h")

        # Convertir todas las columnas a numérico
        df = df.apply(pd.to_numeric, errors="coerce")

        # Eliminar columnas problemáticas si existen
        if "PLiquida_SSR" in df.columns:
            df.drop(columns=["PLiquida_SSR"], inplace=True)

        # Reemplazar 0s con NaN
        df.replace(0, np.nan, inplace=True)

        return df

    def save(self, df: pd.DataFrame, file_path: str, index: bool = True) -> None:
        """
        Guarda un DataFrame en un archivo CSV.

        Parameters:
        - df: DataFrame a guardar
        - file_path: Ruta donde guardar el archivo
        - index: Si True, guarda el índice (por defecto True para mantener datetime)
        """
        df.to_csv(file_path, index=index)
