#!/usr/bin/env python3
"""
Script para analizar los datos clean y entender por qué asfreq crea gaps.
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.recursos.data_manager import DataManager


def analyze_clean_data():
    """Analiza los datos clean para entender los gaps."""

    # Cargar datos limpios
    dm = DataManager()
    clean_data = dm.load_data("data/stage/SO2/clean/CEN-TRAF.csv")

    print("=== ANÁLISIS DE DATOS CLEAN ===")
    print(f"Shape: {clean_data.shape}")
    print(f"Frecuencia del índice: {clean_data.index.freq}")
    print(f"Rango temporal: {clean_data.index.min()} a {clean_data.index.max()}")
    print(f"Valores nulos: {clean_data.isnull().sum().sum()}")

    # Verificar si hay gaps en el índice
    print(f"\n=== ANÁLISIS DE GAPS EN ÍNDICE ===")
    index_diff = clean_data.index.to_series().diff()
    print(f"Diferencias entre timestamps:")
    print(f"  - Mínima: {index_diff.min()}")
    print(f"  - Máxima: {index_diff.max()}")
    print(f"  - Más común: {index_diff.mode().iloc[0]}")

    # Contar gaps
    gaps = index_diff[index_diff != pd.Timedelta(hours=1)]
    print(f"\nGaps encontrados: {len(gaps)}")
    if len(gaps) > 0:
        print("Primeros 10 gaps:")
        print(gaps.head(10))

        # Mostrar ejemplos de gaps
        print("\nEjemplos de gaps:")
        for i, (idx, gap) in enumerate(gaps.head(5).items()):
            print(f"  {i + 1}. {idx}: gap de {gap}")

    # Probar asfreq
    print(f"\n=== PRUEBA DE ASFREQ ===")
    print('Aplicando asfreq("h")...')
    data_with_freq = clean_data.asfreq("h")
    print(f"Shape después de asfreq: {data_with_freq.shape}")
    print(f"Valores nulos después de asfreq: {data_with_freq.isnull().sum().sum()}")

    if data_with_freq.isnull().sum().sum() > 0:
        print("Valores nulos por columna después de asfreq:")
        null_counts = data_with_freq.isnull().sum()
        for col, count in null_counts[null_counts > 0].items():
            print(f"  {col}: {count}")

        # Mostrar algunos ejemplos de nulos
        print("\nPrimeros 5 índices con valores nulos:")
        null_indices = data_with_freq[data_with_freq.isnull().any(axis=1)].index
        for i, idx in enumerate(null_indices[:5]):
            print(f"  {i + 1}. {idx}")


if __name__ == "__main__":
    analyze_clean_data()

