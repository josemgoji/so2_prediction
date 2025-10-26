import os
import numpy as np
import pandas as pd
from pathlib import Path


def save_gap_and_weight_files(
    df: pd.DataFrame,
    pollutant: str,
    station: str,
    stage_dir: Path,
    *,
    target_col: str = "target",
    max_lag: int = 48,
    max_roll: int = 72,
) -> None:
    """
    Marca huecos reales (NaN) en la columna objetivo y guarda:
      - gaps_{station}.csv   → 1 si está en hueco o cerca de uno
      - weights_{station}.csv → 0 para huecos / zona afectada, 1 en caso contrario

    Args:
        df: DataFrame limpio (ya con índice horario y 'target' listo).
        pollutant: Nombre del contaminante (p. ej. 'SO2').
        station: Código o nombre de la estación.
        stage_dir: Carpeta base de los datos intermedios.
        target_col: Columna de la variable objetivo.
        max_lag: Máximo número de lags usados en el modelo.
        max_roll: Ventana máxima usada en rolling features.
    """
    if target_col not in df.columns:
        raise ValueError(f"La columna objetivo '{target_col}' no está en el DataFrame.")

    idx = df.index
    missing_mask = df[target_col].isna()
    pad = max(max_lag, max_roll)

    to_zero = set()
    if missing_mask.any():
        # Detectar runs de NaN contiguos
        is_change = missing_mask.astype(int).diff().fillna(0).ne(0)
        change_pos = np.where(is_change)[0]
        change_pos = np.r_[change_pos, len(missing_mask)]
        starts = change_pos[::2]
        ends = change_pos[1::2] - 1 if len(change_pos) > 1 else starts

        for s, e in zip(starts, ends):
            if s >= len(idx) or e < s:
                continue
            expand_end = min(e + pad, len(idx) - 1)
            expand_idx = idx[s : expand_end + 1]
            to_zero.update(expand_idx)

    to_zero = pd.DatetimeIndex(sorted(to_zero))

    gap_series = pd.Series(0, index=idx, name="is_gap", dtype=int)
    gap_series.loc[to_zero] = 1

    weight_series = pd.Series(1.0, index=idx, name="weight")
    weight_series.loc[to_zero] = 0.0

    # Directorio destino
    marks_dir = Path(stage_dir) / f"{pollutant}/marks/"
    os.makedirs(marks_dir, exist_ok=True)

    # Guardar CSVs
    gaps_path = marks_dir / f"gaps_{station}.csv"
    weights_path = marks_dir / f"weights_{station}.csv"

    pd.DataFrame({"datetime": idx, "is_gap": gap_series.values}).to_csv(
        gaps_path, index=False
    )
    pd.DataFrame({"datetime": idx, "weight": weight_series.values}).to_csv(
        weights_path, index=False
    )

    print(f"[GAP MARKER] Guardados:\n - {gaps_path}\n - {weights_path}")
