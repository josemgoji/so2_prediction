# --- imports
import json
import pickle
import os
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

# Backend de Keras -> torch ANTES de importar keras
os.environ["KERAS_BACKEND"] = "torch"
import keras
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from src.recursos.data_manager import DataManager

from src.recursos.windows_features import (
    FourierWindowFeatures,
    CustomRollingFeatures,
)
from src.recursos.scorers import (
    wmape,
    rmse,
)
from src.utils.data_splitter import split_data_by_dates
from src.utils.plot_utils import create_prediction_plots

from skforecast.deep_learning import ForecasterRnn
from skforecast.deep_learning import create_and_compile_model

from sklearn.preprocessing import FunctionTransformer
from src.constants.parsed_fields import (
    FEATURE_SELECTION_CONFIG,
    REGRESSORS_CONFIG,
    MODEL_RESULTS_CONFIG,
)

import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))


def clean_params_for_json(params_dict):
    """Convierte parámetros a tipos serializables en JSON."""
    cleaned = {}
    for key, value in params_dict.items():
        if isinstance(value, (np.integer, np.floating)):
            cleaned[key] = value.item()
        elif isinstance(value, np.ndarray):
            cleaned[key] = value.tolist()
        elif isinstance(value, (list, tuple)):
            cleaned[key] = [
                v.item() if isinstance(v, (np.integer, np.floating)) else v
                for v in value
            ]
        else:
            cleaned[key] = value
    return cleaned


def calculate_stepwise_mape(y_true, y_pred, steps):
    """Calcula MAPE por cada paso del horizonte."""
    stepwise_mape = {}
    n_windows = len(y_pred) // steps

    for step in range(1, steps + 1):
        indices = [
            i * steps + (step - 1)
            for i in range(n_windows)
            if i * steps + (step - 1) < len(y_pred)
        ]
        if indices:
            y_true_step = y_true.iloc[indices]
            y_pred_step = y_pred.iloc[indices]

            # Evitar divisiones por cero
            mask = y_true_step != 0
            if mask.sum() > 0:
                mape_step = (
                    np.mean(
                        np.abs(
                            (y_true_step[mask] - y_pred_step[mask]) / y_true_step[mask]
                        )
                    )
                    * 100
                )
                stepwise_mape[f"step_{step}"] = float(mape_step)
            else:
                stepwise_mape[f"step_{step}"] = np.nan

    return stepwise_mape


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ===== Configuración de estación =====
STATION = "GIR-EPM"  # opciones …
print(f"🚀 Ejecutando modelos para la estación: {STATION}")

# ===== Configuración de regresores =====
USE_EXOG = True  # True para usar variables exógenas

# ===== 1) Cargar datos base =====
df = DataManager().load_data(f"data/stage/SO2/processed/processed_{STATION}.csv")
df = df.sort_index()

# ===== 2) Configuración de columnas =====
TARGET_COL = "target"

# ===== 3) Cargar selección desde JSON =====
feat_sel_path = Path(
    f"data/stage/SO2/selected/lasso/con_exog/selected_cols_{STATION}_lasso_rf.json"
)
with open(feat_sel_path, "r", encoding="utf-8") as f:
    sel = json.load(f)
selected_lags = sel["selected_lags"]
selected_window_features = sel["selected_window_features"]
selected_exog = sel.get("selected_exog", [])

missing = [c for c in [TARGET_COL] + selected_exog if c not in df.columns]
if missing:
    raise ValueError(f"Faltan columnas en df: {missing}")

# ===== 4) (Opcional) Window features =====
# Declaradas pero no aplicadas aquí; asumo que ya las aplicaste en tu pipeline previo.
window_features = [
    FourierWindowFeatures(period=24, K=3),
    CustomRollingFeatures(stats=["mean"], window_sizes=[3, 6, 24, 48, 72]),
    CustomRollingFeatures(stats=["min"], window_sizes=[6, 24]),
    CustomRollingFeatures(stats=["max"], window_sizes=[6, 12, 24]),
]

# ===== 5) Split train / val / test =====
y_train, exog_train, y_val, exog_val, y_test, exog_test, y_trainval, exog_trainval = (
    split_data_by_dates(
        df=df,
        target_col=TARGET_COL,
        exog_cols=selected_exog,
        val_months=2,
        test_months=2,
    )
)

# Convertir las series a DataFrame (requisito de skforecast)
y_train = y_train.to_frame(name=TARGET_COL)
y_val = y_val.to_frame(name=TARGET_COL)
y_test = y_test.to_frame(name=TARGET_COL)
y_trainval = y_trainval.to_frame(name=TARGET_COL)


# Chequeo rápido de NaNs
def assert_no_nans(df_obj, name):
    if df_obj is None:
        return
    if df_obj.isnull().any().any():
        n = int(df_obj.isnull().sum().sum())
        raise ValueError(
            f"Se encontraron {n} NaNs en {name}. Imputa/filtra antes de entrenar."
        )


assert_no_nans(y_train, "y_train")
assert_no_nans(y_val, "y_val")
assert_no_nans(y_test, "y_test")
if USE_EXOG:
    assert_no_nans(exog_train, "exog_train")
    assert_no_nans(exog_val, "exog_val")
    assert_no_nans(exog_test, "exog_test")

# Debug: Verificar formas
print(f"\n🔍 Verificando formas de datos:")
print(f"y_train shape: {y_train.shape}")
print(f"exog_train shape: {exog_train.shape if USE_EXOG else 'None'}")
print(f"y_val shape: {y_val.shape}")
print(f"exog_val shape: {exog_val.shape if USE_EXOG else 'None'}")
print(f"Lags seleccionados: {selected_lags}")
print(f"Número de variables exógenas: {len(selected_exog) if USE_EXOG else 0}")

# ===== 6) Configuración de estructura de carpetas =====
USE_WEIGHTS = False  # Eliminado el uso de pesos por incompatibilidad

results_dir = (
    Path(MODEL_RESULTS_CONFIG["analytics_dir"]) / MODEL_RESULTS_CONFIG["results_subdir"]
)
results_dir.mkdir(parents=True, exist_ok=True)

exog_status = "con_exog" if USE_EXOG else "sin_exog"
station_results_dir = results_dir / STATION / exog_status
for sub in ["models", "plots", "results", "summary"]:
    (station_results_dir / sub).mkdir(parents=True, exist_ok=True)

models_dir = station_results_dir / "models"
plots_dir = station_results_dir / "plots"
results_dir_station = station_results_dir / "results"
summary_dir = station_results_dir / "summary"

print(f"\n📁 Estructura de carpetas creada para {STATION} con estado {exog_status}")

regressor_name = "LSTM"

# ===== 7) Configuración común para todos los modelos =====
H = 72  # horizonte en pasos (ajústalo)

# ===== 8) Búsqueda manual de hiperparámetros =====
from itertools import product

param_grid = {
    "recurrent_units": [[100, 50], [128, 64]],
    "dense_units": [[32, 16], [64, 32]],
    "learning_rate": [0.01, 0.001],
    "epochs": [1, 2],
    "batch_size": [64, 128],
}

grid = list(product(*param_grid.values()))

results_list = []

for i, params in enumerate(grid):
    print(f"Training model {i + 1}/{len(grid)} with params: {params}")
    rec_units, dense_units, lr, epochs, batch_size = params

    try:
        # Crear modelo con create_and_compile_model
        temp_model = create_and_compile_model(
            series=y_train,
            levels=[TARGET_COL],
            lags=selected_lags,
            steps=H,
            exog=exog_train if USE_EXOG else None,
            recurrent_layer="LSTM",
            recurrent_units=rec_units,
            dense_units=dense_units,
            compile_kwargs={
                "optimizer": Adam(learning_rate=lr),
                "loss": MeanSquaredError(),
            },
        )

        # Crear forecaster con el modelo
        temp_forecaster = ForecasterRnn(
            regressor=temp_model,
            levels=[TARGET_COL],
            lags=selected_lags,
            transformer_series=MinMaxScaler(),
            transformer_exog=MinMaxScaler() if USE_EXOG else None,
            fit_kwargs={
                "epochs": epochs,
                "batch_size": batch_size,
                "verbose": 0,
                "callbacks": [
                    EarlyStopping(
                        monitor="loss",
                        patience=5,
                        restore_best_weights=True,
                        verbose=0,
                    ),
                    ReduceLROnPlateau(
                        monitor="loss",
                        factor=0.5,
                        patience=3,
                        min_lr=1e-6,
                        verbose=0,
                    ),
                ],
                "validation_split": 0.1,
            },
        )

        # Fit (series y exog con mismo índice)
        temp_forecaster.fit(series=y_train, exog=exog_train if USE_EXOG else None)

        print("starting prediction")
        # Predicción en validación
        y_pred_val = temp_forecaster.predict(
            steps=H, exog=exog_val if USE_EXOG else None
        )
        # ======== VALIDACIÓN ========
        # Extraer solo los valores numéricos
        y_true_val = y_val[TARGET_COL]  # DataFrame → Series
        y_pred_val_s = y_pred_val["pred"]  # tomar solo la columna pred

        # Alinear índices por seguridad
        y_true_val, y_pred_val_s = y_true_val.align(y_pred_val_s, join="inner")

        # Calcular métricas
        val_wmape = wmape(y_true_val, y_pred_val_s)
        val_rmse = rmse(y_true_val, y_pred_val_s)

        print(
            f"✅ Resultado en validación - WMAPE: {100 * val_wmape:.2f}%, RMSE: {val_rmse:.4f}"
        )

        results_list.append(
            {
                "recurrent_units": rec_units,
                "dense_units": dense_units,
                "learning_rate": lr,
                "epochs": epochs,
                "batch_size": batch_size,
                "val_wmape": float(val_wmape),
                "val_rmse": float(val_rmse),
            }
        )

    except Exception as e:
        print(f"❌ Error con esta combinación (rec_units={rec_units}, dense_units={dense_units}, lr={lr}, epochs={epochs}, batch_size={batch_size}): {str(e)}")
        import traceback

        traceback.print_exc()
        continue

# Verificar si se obtuvieron resultados
if len(results_list) == 0:
    raise ValueError(
        "❌ No se pudo entrenar ningún modelo. Verifica los parámetros y datos."
    )

results_df = pd.DataFrame(results_list).sort_values("val_wmape")
print("\n🏆 Mejores combinaciones de hiperparámetros:")
print(results_df.head())

best_combo = results_df.iloc[0].to_dict()
print(f"👉 Mejor combinación: {best_combo}")

# actualizar mejor configuración
best_params = {
    "recurrent_units": best_combo["recurrent_units"],
    "dense_units": best_combo["dense_units"],
    "learning_rate": best_combo["learning_rate"],
    "epochs": int(best_combo["epochs"]),
    "batch_size": int(best_combo["batch_size"]),
}

# ===== 9) Entrenar modelo final con mejores parámetros =====
print(f"\n🎯 Entrenando modelo final con mejores parámetros...")

final_model = create_and_compile_model(
    series=y_trainval,
    lags=selected_lags,
    steps=H,
    exog=exog_trainval if USE_EXOG else None,
    recurrent_layer="LSTM",
    recurrent_units=best_params["recurrent_units"],
    dense_units=best_params["dense_units"],
    compile_kwargs={
        "optimizer": Adam(learning_rate=best_params["learning_rate"]),
        "loss": MeanSquaredError(),
    },
)

final_forecaster = ForecasterRnn(
    regressor=final_model,
    levels=[TARGET_COL],
    lags=selected_lags,
    transformer_series=MinMaxScaler(),
    transformer_exog=MinMaxScaler() if USE_EXOG else None,
    fit_kwargs={
        "epochs": best_params["epochs"],
        "batch_size": best_params["batch_size"],
        "verbose": 1,
        "callbacks": [
            EarlyStopping(
                monitor="loss", patience=5, restore_best_weights=True, verbose=1
            ),
            ReduceLROnPlateau(
                monitor="loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
            ),
        ],
        # Igual que antes: sin validation_data manual
        "validation_split": 0.1,
    },
)

# Entrenar con train+val
final_forecaster.fit(series=y_trainval, exog=exog_trainval if USE_EXOG else None)

# ===== 10) Evaluación en test final =====
y_pred_test = final_forecaster.predict(
    steps=H, exog=exog_test if USE_EXOG else None
)

# ======== TEST ========
y_true_test = y_test[TARGET_COL]
y_pred_test_s = y_pred_test["pred"]

y_true_test, y_pred_test_s = y_true_test.align(y_pred_test_s, join="inner")

test_rmse = rmse(y_true_test, y_pred_test_s)
test_wmape = wmape(y_true_test, y_pred_test_s)

stepwise_mape_test = calculate_stepwise_mape(y_true_test, y_pred_test_s, H)

print(f"\n🎯 Test – {regressor_name}:")
print(f"RMSE: {test_rmse:.4f}")
print(f"WMAPE %: {100 * test_wmape:.2f}")
print(f"Stepwise MAPE: {stepwise_mape_test}")

# ===== 11) Guardado de modelo y resultados =====
timestamp_str = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
model_file = models_dir / f"{regressor_name}_model_{timestamp_str}.pkl"
with open(model_file, "wb") as f:
    pickle.dump(final_forecaster, f)
print(f"💾 Modelo entrenado guardado en: {model_file}")

result_data = {
    "station": STATION,
    "model_type": regressor_name,
    "use_exog": USE_EXOG,
    "validation_metrics": {
        "wmape": float(best_combo["val_wmape"]),
        "rmse": float(best_combo["val_rmse"]),
    },
    "test_metrics": {
        "wmape": float(test_wmape),
        "rmse": float(test_rmse),
        "stepwise_mape": stepwise_mape_test,
    },
    "best_params": clean_params_for_json(best_params),
    "model_file": str(model_file),
    "timestamp": pd.Timestamp.now().isoformat(),
}

result_file = results_dir_station / f"{regressor_name}_{timestamp_str}.json"
with open(result_file, "w", encoding="utf-8") as f:
    json.dump(result_data, f, indent=2, ensure_ascii=False)
print(f"💾 Resultados guardados en: {result_file}")

# ===== 12) Resumen final =====
print(f"\n{'=' * 80}")
print(f"📋 RESUMEN DE RESULTADOS PARA ESTACIÓN: {STATION}")
print(f"🔧 Configuración: {'Con exógenas' if USE_EXOG else 'Sin exógenas'}")
print(f"{'=' * 80}")

print(f"\n🏆 MODELO FINAL:")
print(f"Test WMAPE: {100 * test_wmape:.2f}%")
print(f"Test RMSE: {test_rmse:.4f}")
print(f"Test Stepwise MAPE: {stepwise_mape_test}")
print(f"Modelo guardado en: {model_file}")

summary_data = {
    "station": STATION,
    "configuration": {
        "use_exog": USE_EXOG,
        "timestamp": pd.Timestamp.now().isoformat(),
    },
    "best_model": {
        "name": regressor_name,
        "test_wmape": float(test_wmape),
        "test_rmse": float(test_rmse),
        "test_stepwise_mape": stepwise_mape_test,
        "val_wmape": float(best_combo["val_wmape"]),
        "val_rmse": float(best_combo["val_rmse"]),
        "best_params": clean_params_for_json(best_params),
        "model_file": str(model_file),
    },
    "hyperparameter_search_results": results_df.to_dict("records"),
}

summary_file = summary_dir / f"summary_{timestamp_str}.json"
with open(summary_file, "w", encoding="utf-8") as f:
    json.dump(summary_data, f, indent=2, ensure_ascii=False)

csv_file = summary_dir / f"results_comparison_{timestamp_str}.csv"
results_df.to_csv(csv_file, index=False)

print(f"💾 Resumen completo guardado en: {summary_file}")
print(f"📊 Comparación en CSV guardada en: {csv_file}")
print(f"\n✅ Proceso completado para estación {STATION}")
