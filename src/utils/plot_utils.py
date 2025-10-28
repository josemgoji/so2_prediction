"""
Utilidades para crear gráficos con Plotly
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.offline as pyo
from pathlib import Path


def create_prediction_plots(
    y_val, preds_val, y_test, y_pred_test, model_name, station, save_dir
):
    """
    Crea gráficos de plotly para comparar valores reales vs predicciones
    tanto para validación como para test, y los guarda en PNG y HTML
    
    Parameters:
    -----------
    y_val : pd.Series
        Valores reales de validación
    preds_val : pd.DataFrame
        Predicciones de validación con columna 'pred'
    y_test : pd.Series
        Valores reales de test
    y_pred_test : pd.Series
        Predicciones de test
    model_name : str
        Nombre del modelo
    station : str
        Nombre de la estación
    save_dir : Path
        Directorio donde guardar los gráficos
        
    Returns:
    --------
    dict
        Diccionario con las rutas de los archivos generados
    """
    # Crear directorio para gráficos si no existe
    plots_dir = save_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Crear figura para validación
    fig_val = go.Figure()

    # Trazado para valores reales de validación
    trace_val_real = go.Scatter(
        x=y_val.index,
        y=y_val.values,
        name="Valor Real (Validación)",
        mode="lines",
        line=dict(color="blue", width=2),
    )

    # Trazado para predicciones de validación
    trace_val_pred = go.Scatter(
        x=preds_val.index,
        y=preds_val["pred"].values,
        name="Predicción (Validación)",
        mode="lines",
        line=dict(color="red", width=2, dash="dash"),
    )

    fig_val.add_trace(trace_val_real)
    fig_val.add_trace(trace_val_pred)

    fig_val.update_layout(
        title=f"Valor Real vs Predicciones - Validación - {model_name} - {station}",
        xaxis_title="Fecha y Hora",
        yaxis_title="SO2 (μg/m³)",
        width=1000,
        height=500,
        margin=dict(l=50, r=50, t=60, b=50),
        legend=dict(orientation="h", yanchor="top", y=1.02, xanchor="left", x=0.01),
        hovermode="x unified",
    )

    # Crear figura para test
    fig_test = go.Figure()

    # Trazado para valores reales de test
    trace_test_real = go.Scatter(
        x=y_test.index,
        y=y_test.values,
        name="Valor Real (Test)",
        mode="lines",
        line=dict(color="blue", width=2),
    )

    # Trazado para predicciones de test
    trace_test_pred = go.Scatter(
        x=y_test.index,
        y=y_pred_test.values,
        name="Predicción (Test)",
        mode="lines",
        line=dict(color="red", width=2, dash="dash"),
    )

    fig_test.add_trace(trace_test_real)
    fig_test.add_trace(trace_test_pred)

    fig_test.update_layout(
        title=f"Valor Real vs Predicciones - Test - {model_name} - {station}",
        xaxis_title="Fecha y Hora",
        yaxis_title="SO2 (μg/m³)",
        width=1000,
        height=500,
        margin=dict(l=50, r=50, t=60, b=50),
        legend=dict(orientation="h", yanchor="top", y=1.02, xanchor="left", x=0.01),
        hovermode="x unified",
    )

    # Generar timestamp para nombres de archivo
    timestamp_str = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    # Guardar gráficos en HTML
    html_val_file = (
        plots_dir / f"{model_name}_{station}_validation_{timestamp_str}.html"
    )
    html_test_file = plots_dir / f"{model_name}_{station}_test_{timestamp_str}.html"

    pyo.plot(fig_val, filename=str(html_val_file), auto_open=False)
    pyo.plot(fig_test, filename=str(html_test_file), auto_open=False)

    # Guardar gráficos en PNG
    png_val_file = plots_dir / f"{model_name}_{station}_validation_{timestamp_str}.png"
    png_test_file = plots_dir / f"{model_name}_{station}_test_{timestamp_str}.png"

    fig_val.write_image(str(png_val_file), width=1000, height=500, scale=2)
    fig_test.write_image(str(png_test_file), width=1000, height=500, scale=2)

    print("📊 Gráficos guardados:")
    print(f"   Validación HTML: {html_val_file}")
    print(f"   Validación PNG: {png_val_file}")
    print(f"   Test HTML: {html_test_file}")
    print(f"   Test PNG: {png_test_file}")

    return {
        "validation_html": str(html_val_file),
        "validation_png": str(png_val_file),
        "test_html": str(html_test_file),
        "test_png": str(png_test_file),
    }
