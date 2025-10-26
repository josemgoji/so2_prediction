#!/usr/bin/env python3
"""
Ejemplo de uso del FeatureSelector con window features personalizadas.
"""

from src.pipelines.feature_selection_pipeline import FeatureSelector


def main():
    """Ejemplo de uso del FeatureSelector con window features."""

    # Configuración de window features personalizada
    window_features_params = {
        "period": 24,  # Período para Fourier y STL
        "stats": ["mean", "std", "min", "max"],  # Estadísticas para rolling
        "window_sizes": [3, 6, 12, 24],  # Tamaños de ventana
        "fourier_k": 3,  # Componentes Fourier
        "stl_robust": True,  # STL robusto
    }

    # Crear FeatureSelector con window features personalizadas
    feature_selector = FeatureSelector(
        data_path="data/stage/SO2",
        output_path="data/stage/SO2/selected",
        selector_type="lasso",
        regressor_type="lgbm",
        lags=48,
        window_features_params=window_features_params,  # Usar parámetros personalizados
        random_state=15926,
    )

    # Mostrar información sobre window features configuradas
    print("Window Features configuradas:")
    window_info = feature_selector.get_window_features_info()
    print(
        f"  - Número de tipos de window features: {window_info['window_features_count']}"
    )
    print(f"  - Tipos: {window_info['window_features_types']}")

    # Ejecutar selección para una estación
    stations = ["CEN-TRAF"]

    for station in stations:
        print(f"\nProcesando estación: {station}")

        try:
            # Selección sin exógenos (solo target)
            results_no_exog = feature_selector.select_features_for_station(
                station=station, include_exog=False, subsample=0.3
            )

            print(f"  - Lags seleccionados: {results_no_exog['n_selected_lags']}")
            print(
                f"  - Window features seleccionadas: {results_no_exog['n_selected_window_features']}"
            )
            print(f"  - Exógenas seleccionadas: {results_no_exog['n_selected_exog']}")
            print(f"  - Total características: {results_no_exog['total_features']}")

            # Guardar resultados
            feature_selector.save_selection_results(results_no_exog, station, "no_exog")

            # Selección con exógenos (todas las variables)
            results_with_exog = feature_selector.select_features_for_station(
                station=station, include_exog=True, subsample=0.3
            )

            print(f"  - Lags seleccionados: {results_with_exog['n_selected_lags']}")
            print(
                f"  - Window features seleccionadas: {results_with_exog['n_selected_window_features']}"
            )
            print(f"  - Exógenas seleccionadas: {results_with_exog['n_selected_exog']}")
            print(f"  - Total características: {results_with_exog['total_features']}")

            # Guardar resultados
            feature_selector.save_selection_results(
                results_with_exog, station, "with_exog"
            )

        except Exception as e:
            print(f"  - Error procesando {station}: {str(e)}")


if __name__ == "__main__":
    main()
