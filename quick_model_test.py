#!/usr/bin/env python3
"""
Script rápido para probar modelos con tu infraestructura.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.recursos.data_manager import DataManager
from src.recursos.regressors import RandomForestRegressor, LGBMRegressor, RidgeRegressor
from src.constants.parsed_fields import STAGE_DIR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def quick_test(station="GIR-EPM", use_exog=True):
    """
    Prueba rápida de modelos.

    Parameters:
    -----------
    station : str
        Estación a probar
    use_exog : bool
        Si usar exógenas (True) o solo características de ventana (False)
    """
    print(f"🚀 Prueba rápida: {station} - {'Con' if use_exog else 'Sin'} exógenas")
    print("=" * 50)

    # 1. Cargar datos
    data_manager = DataManager()
    processed_file = STAGE_DIR / "SO2" / "processed" / f"processed_{station}.csv"
    data = data_manager.load_data(str(processed_file))
    print(f"✅ Datos cargados: {data.shape}")

    # 2. Cargar características seleccionadas
    exog_suffix = "con_exog" if use_exog else "sin_exog"
    selected_file = (
        STAGE_DIR
        / "SO2"
        / "selected"
        / "lasso"
        / exog_suffix
        / f"selected_cols_{station}_lasso_rf.json"
    )

    with open(selected_file, "r") as f:
        selected_features = json.load(f)
    print(f"✅ Características: {len(selected_features)}")

    # 3. Preparar datos
    available_features = [col for col in selected_features if col in data.columns]
    X = data[available_features].copy()
    y = data["target"].copy()

    # Eliminar nulos
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[mask]
    y = y[mask]
    print(f"✅ Datos limpios: {X.shape[0]} muestras, {X.shape[1]} características")

    # 4. Dividir datos (80% train, 20% test)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # 5. Probar modelos
    models = {
        "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=10),
        "LGBM": LGBMRegressor(n_estimators=100, max_depth=10, learning_rate=0.1),
        "Ridge": RidgeRegressor(alpha=1.0),
    }

    results = []

    for name, model in models.items():
        print(f"\n🔄 Probando {name}...")

        try:
            # Entrenar
            model.fit(X_train, y_train)

            # Predecir
            y_pred = model.predict(X_test)

            # Métricas
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            results.append(
                {
                    "Modelo": name,
                    "RMSE": f"{rmse:.4f}",
                    "MAE": f"{mae:.4f}",
                    "R²": f"{r2:.4f}",
                }
            )

            print(f"   RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")

        except Exception as e:
            print(f"   ❌ Error: {e}")

    # 6. Mostrar resultados
    if results:
        print(f"\n📊 RESULTADOS:")
        print("=" * 50)
        df_results = pd.DataFrame(results)
        print(df_results.to_string(index=False))

        # Mejor modelo
        best_idx = df_results["RMSE"].astype(float).idxmin()
        best_model = df_results.iloc[best_idx]
        print(f"\n🏆 Mejor modelo: {best_model['Modelo']} (RMSE: {best_model['RMSE']})")

    return results


if __name__ == "__main__":
    # Cambiar estos parámetros según lo que quieras probar
    STATION = "GIR-EPM"  # Opciones: GIR-EPM, CEN-TRAF, ITA-CJUS, MED-FISC
    USE_EXOG = True  # True = con exógenas, False = sin exógenas

    results = quick_test(station=STATION, use_exog=USE_EXOG)
