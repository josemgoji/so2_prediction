#!/usr/bin/env python3
"""
Pipeline de Feature Engineering para series temporales de SO2.
- Frecuencia fija 'h'
- Auto-trim simple por ventana máxima (en horas)
- Imputación de NaNs SOLO en columnas originales (interpolate + ffill + bfill)
- Logs de depuración y verificación de continuidad del índice
"""

import pandas as pd
from typing import Optional, List, Dict, Any
from pathlib import Path

from ..recursos.data_manager import DataManager
from ..services.feature_engineering import FeatureEngineering
from ..constants.parsed_fields import (
    LOCATION_CONFIG,
    DEFAULT_TRIM_START,
    DEFAULT_TRIM_END,
)


def _auto_trim_hours(window_windows: Optional[List[str]]) -> int:
    """
    Devuelve el recorte inicial mínimo en filas para cubrir la ventana máxima.
    Asume ventanas como '3h','6h','24h',... y frecuencia fija horaria.
    """
    if not window_windows:
        return 0
    hours = []
    for w in window_windows:
        wl = w.lower().strip()
        if wl.endswith("h"):
            try:
                hours.append(int(wl[:-1]))
            except ValueError:
                pass
    return max(hours) if hours else 0


class FeatureEngineeringPipeline:
    def __init__(
        self,
        location_config: Optional[Dict] = None,
        date_column: str = "datetime",
    ):
        self.location_config = location_config or LOCATION_CONFIG
        self.date_column = date_column

        self.data_manager = DataManager(date_column=date_column)
        self.feature_engineering = FeatureEngineering(
            location_config=self.location_config
        )

        self.raw_data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[pd.DataFrame] = None
        self.feature_engineering_result: Optional[pd.DataFrame] = None

    # -----------------------------------------------------
    # 1. Carga
    # -----------------------------------------------------
    def load_data(self, file_path: str) -> pd.DataFrame:
        print(f"Cargando datos desde: {file_path}")
        self.raw_data = self.data_manager.load_data(file_path)

        print(
            f"Datos cargados: {len(self.raw_data)} filas, {len(self.raw_data.columns)} columnas"
        )
        print(
            f"Rango de fechas: {self.raw_data.index.min()} a {self.raw_data.index.max()}"
        )
        print(f"Columnas disponibles: {list(self.raw_data.columns)}")

        print("\nEstadísticas básicas:")
        for col in self.raw_data.columns:
            missing = self.raw_data[col].isnull().sum()
            print(f"   - {col}: {missing} ({missing / len(self.raw_data) * 100:.3f}%)")

        return self.raw_data

    # -----------------------------------------------------
    # 2. Creación de features
    # -----------------------------------------------------
    def create_feature_engineering_features(
        self,
        window_columns: List[str],
        calendar_features: List[str] = None,
        window_windows: List[str] = None,
        window_functions: List[str] = None,
        trim_start: int = DEFAULT_TRIM_START,
        trim_end: int = DEFAULT_TRIM_END,
    ) -> pd.DataFrame:
        if self.raw_data is None:
            raise ValueError("No hay datos cargados. Ejecuta load_data() primero.")

        print("\n" + "=" * 70)
        print("CREANDO CARACTERÍSTICAS DE FEATURE ENGINEERING")
        print("=" * 70)

        # --- Frecuencia fija horaria ---
        infer_freq = "h"
        base_delta = pd.Timedelta(hours=1)

        # --- Auto-trim simple en horas ---
        auto_trim = _auto_trim_hours(window_windows)
        eff_trim_start = max(trim_start or 0, auto_trim)

        print(f"Ubicación: {self.feature_engineering.location_config['name']}")
        print(f"Frecuencia: {infer_freq} (delta típico={base_delta})")
        print(f"Window columns: {window_columns}")
        print(f"Window windows: {window_windows}")
        print(f"Window functions: {window_functions}")
        print(f"Trim (user): start={trim_start}, end={trim_end}")
        print(f"Trim (auto por ventanas): start={auto_trim}")
        print(f"Trim (efectivo): start={eff_trim_start}, end={trim_end}")

        # -------------------------------------------------
        # Imputar columnas crudas con NaN (antes de crear features)
        # -------------------------------------------------
        raw_with_patch = self.raw_data.copy()
        cols_with_nan = raw_with_patch.columns[raw_with_patch.isna().any()].tolist()
        if cols_with_nan:
            print(
                f"[INFO] Imputando NaN crudos en columnas originales: {cols_with_nan} "
                f"(interpolate + ffill + bfill)"
            )
            for c in cols_with_nan:
                n_before = raw_with_patch[c].isna().sum()
                raw_with_patch[c] = (
                    raw_with_patch[c]
                    .interpolate(limit=2)  # huecos cortos
                    .ffill()
                    .bfill()
                )
                n_after = raw_with_patch[c].isna().sum()
                print(f"    → {c}: {n_before} → {n_after} NaNs")
        else:
            print("[INFO] No hay NaNs en columnas originales.")

        exog_cols = [c for c in raw_with_patch.columns if c != "target"]
        if exog_cols:
            print(f"[INFO] Aplicando shift de 1h a exógenas: {exog_cols}")
            raw_with_patch[exog_cols] = raw_with_patch[exog_cols].shift(1)
            raw_with_patch = raw_with_patch.dropna() 

        # -------------------------------------------------
        # Crear engineered features (con freq fija 'h')
        # -------------------------------------------------
        feature_engineering_features = self.feature_engineering.create_all_features(
            data=raw_with_patch,
            calendar_features=calendar_features,
            window_columns=window_columns,
            window_windows=window_windows,
            window_functions=window_functions,
            trim_start=eff_trim_start,
            trim_end=trim_end,
            freq=infer_freq,  # fijo 'h'
        )

        # Debug: NaNs en engineered features antes del merge
        nan_pre = feature_engineering_features[
            feature_engineering_features.isna().any(axis=1)
        ]
        print(
            f"[DEBUG] NaNs en engineered features (antes del merge): {len(nan_pre)} filas"
        )
        if not nan_pre.empty:
            for ts, row in nan_pre.iloc[:5].iterrows():
                bad_cols = row.index[row.isna()].tolist()
                print(f"  - {ts}: {bad_cols}")

        # -------------------------------------------------
        # Merge + trimming
        # -------------------------------------------------
        merged = pd.concat([raw_with_patch, feature_engineering_features], axis=1)

        if eff_trim_start > 0:
            merged = merged.iloc[eff_trim_start:, :]
        if trim_end and trim_end > 0:
            merged = merged.iloc[:-trim_end, :]

        self.feature_engineering_result = merged

        # Debug: NaNs después del merge
        nan_post = merged[merged.isna().any(axis=1)]
        print(f"[DEBUG] NaNs tras el merge (todas las columnas): {len(nan_post)} filas")
        if not nan_post.empty:
            print("[DEBUG] Timestamps con NaN restantes (en originales):")
            print(list(nan_post.index[:10]))

        # -------------------------------------------------
        # Dropna solo en engineered features
        # -------------------------------------------------
        eng_cols = feature_engineering_features.columns
        mask_ok = ~merged[eng_cols].isna().any(axis=1)
        dropped = (~mask_ok).sum()
        print(f"Filas eliminadas por NaN en engineered features: {dropped}")
        self.feature_engineering_result = merged.loc[mask_ok]

        # -------------------------------------------------
        # Verificación de continuidad del índice
        # -------------------------------------------------
        diffs = self.feature_engineering_result.index.to_series().diff().dropna()
        if not diffs.eq(pd.Timedelta(hours=1)).all():
            uniq = diffs.value_counts().head(5)
            raise RuntimeError(
                f"[ASSERT] Gap detectado: deltas no constantes.\nTop deltas:\n{uniq}"
            )
        else:
            print("[ASSERT] Índice continuo verificado (1h constante).")

        print(f"Shape engineered features: {feature_engineering_features.shape}")
        print(f"Shape combinado final: {self.feature_engineering_result.shape}")
        print("Datos finales listos para selección de variables / modelado.")

        return self.feature_engineering_result

    # -----------------------------------------------------
    # 3. Orquestador completo
    # -----------------------------------------------------
    def run_complete_pipeline(
        self,
        file_path: str,
        window_columns: List[str],
        calendar_features: List[str] = None,
        window_windows: List[str] = None,
        window_functions: List[str] = None,
        trim_start: int = DEFAULT_TRIM_START,
        trim_end: int = DEFAULT_TRIM_END,
    ) -> Dict[str, Any]:
        print("INICIANDO PIPELINE COMPLETO DE FEATURE ENGINEERING")
        print("=" * 70)

        try:
            raw_data = self.load_data(file_path)
            result = self.create_feature_engineering_features(
                window_columns=window_columns,
                calendar_features=calendar_features,
                window_windows=window_windows,
                window_functions=window_functions,
                trim_start=trim_start,
                trim_end=trim_end,
            )
            self.processed_data = result

            print("\n" + "=" * 70)
            print("PIPELINE COMPLETADO EXITOSAMENTE")
            print("=" * 70)
            print(f"Datos originales: {raw_data.shape}")
            print(f"Datos procesados finales: {self.processed_data.shape}")

            return {
                "raw_data": raw_data,
                "processed_data": self.processed_data,
                "feature_engineering_result": self.feature_engineering_result,
                "pipeline_config": {"location_config": self.location_config},
            }

        except Exception as e:
            print(f"\nERROR DURANTE EL PIPELINE: {str(e)}")
            import traceback

            traceback.print_exc()
            raise

    # -----------------------------------------------------
    # 4. Guardado y resumen
    # -----------------------------------------------------
    def save_results(self, output_path: str, include_raw: bool = False):
        if self.processed_data is None:
            raise ValueError(
                "No hay datos procesados para guardar. Ejecuta el pipeline primero."
            )
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.data_manager.save(
            self.processed_data, str(output_path.with_suffix(".csv"))
        )
        print(f"Datos procesados guardados en: {output_path}")

        if include_raw and self.raw_data is not None:
            raw_path = output_path.with_name(f"{output_path.stem}_raw.csv")
            self.data_manager.save(self.raw_data, str(raw_path))
            print(f"Datos originales guardados en: {raw_path}")

    def get_summary(self) -> Dict[str, Any]:
        summary = {
            "pipeline_status": "completed"
            if self.processed_data is not None
            else "not_run",
            "data_info": {},
            "features_info": {},
        }
        if self.raw_data is not None:
            summary["data_info"]["raw_shape"] = self.raw_data.shape
            summary["data_info"]["raw_columns"] = list(self.raw_data.columns)
        if self.feature_engineering_result is not None:
            summary["features_info"]["feature_engineering_shape"] = (
                self.feature_engineering_result.shape
            )
            summary["features_info"]["feature_engineering_columns"] = list(
                self.feature_engineering_result.columns
            )
        if self.processed_data is not None:
            summary["data_info"]["processed_shape"] = self.processed_data.shape
        return summary


def create_default_pipeline() -> FeatureEngineeringPipeline:
    return FeatureEngineeringPipeline()


def create_custom_pipeline(
    location_config: Optional[Dict] = None,
) -> FeatureEngineeringPipeline:
    return FeatureEngineeringPipeline(location_config=location_config)
