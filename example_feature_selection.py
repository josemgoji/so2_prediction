"""
Ejemplo de uso de la clase FeatureSelector para selección de características.
Este script demuestra cómo usar la clase reutilizable para diferentes estaciones.
"""

from src.pipelines.feature_selection_pipeline import FeatureSelector


def main():
    """
    Ejemplo principal de uso de FeatureSelector.
    """

    # Configuración básica
    stations = ["CEN-TRAF", "GIR-EPM", "ITA-CJUS", "MED-FISC"]

    print("=" * 60)
    print("EJEMPLO DE USO: FeatureSelector")
    print("=" * 60)

    # Ejemplo 1: Selección básica con Lasso
    print("\n1. SELECCIÓN BÁSICA CON LASSO")
    print("-" * 40)

    selector_lasso = FeatureSelector(
        data_path="data/stage/SO2",
        output_path="data/stage/SO2/selected",
        selector_type="lasso",
        regressor_type="lgbm",
        lags=48,
    )

    # Ejecutar selección para una estación
    results_lasso = selector_lasso.select_features_for_station(
        station="CEN-TRAF", include_exog=False, subsample=0.3
    )

    # Guardar resultados
    selector_lasso.save_selection_results(results_lasso, "CEN-TRAF", "example")

    # Ejemplo 2: Selección con RFECV
    print("\n2. SELECCIÓN CON RFECV")
    print("-" * 40)

    selector_rfecv = FeatureSelector(
        selector_type="rfecv",
        regressor_type="rf",
        lags=24,  # Menos lags para RFECV (es más lento)
        selector_params={"cv_splits": 3, "step": 2, "min_features_to_select": 5},
        regressor_params={"n_estimators": 50, "max_depth": 10},
    )

    # Ejecutar para una estación
    results_rfecv = selector_rfecv.select_features_for_station(
        station="GIR-EPM",
        include_exog=False,
        subsample=0.2,  # Menor subsample para RFECV
    )

    # Ejemplo 3: Pipeline completo para múltiples estaciones
    print("\n3. PIPELINE COMPLETO PARA MÚLTIPLES ESTACIONES")
    print("-" * 40)

    selector_pipeline = FeatureSelector(
        selector_type="lasso",
        regressor_type="lgbm",
        lags=48,
        selector_params={
            "cv_splits": 3,
            "alphas": [0.001, 0.01, 0.1, 1.0],  # Alphas específicos
        },
        regressor_params={"n_estimators": 100, "max_depth": 8, "learning_rate": 0.1},
    )

    # Ejecutar pipeline completo
    all_results = selector_pipeline.run_selection_pipeline(
        stations=stations, include_exog=False, subsample=0.4, save_results=True
    )

    # Mostrar resumen de resultados
    print("\n4. RESUMEN DE RESULTADOS")
    print("-" * 40)

    for station, results in all_results.items():
        if "error" not in results:
            print(f"{station}:")
            print(f"  - Lags seleccionados: {results['n_selected_lags']}")
            print(f"  - Window features: {results['n_selected_window_features']}")
            print(f"  - Variables exógenas: {results['n_selected_exog']}")
            print(f"  - Total características: {results['total_features']}")
        else:
            print(f"{station}: ERROR - {results['error']}")

    # Ejemplo 4: Selección con datos exógenos
    print("\n5. SELECCIÓN CON DATOS EXÓGENOS")
    print("-" * 40)

    selector_exog = FeatureSelector(
        selector_type="lasso", regressor_type="lgbm", lags=24
    )

    # Ejecutar con datos exógenos
    results_exog = selector_exog.select_features_for_station(
        station="ITA-CJUS",
        include_exog=True,  # Incluir datos exógenos
        select_only=None,  # Seleccionar todos los tipos
        subsample=0.3,
    )

    # Ejemplo 5: Actualización de configuración
    print("\n6. ACTUALIZACIÓN DE CONFIGURACIÓN")
    print("-" * 40)

    # Crear selector inicial
    selector_dynamic = FeatureSelector(
        selector_type="lasso", regressor_type="lgbm", lags=48
    )

    # Crear nuevo selector con configuración diferente
    selector_dynamic = FeatureSelector(
        selector_type="rfecv",
        regressor_type="rf",
        lags=24,
        selector_params={"cv_splits": 2},
        regressor_params={"n_estimators": 30},
    )

    # Usar con nueva configuración
    results_dynamic = selector_dynamic.select_features_for_station(
        station="MED-FISC", subsample=0.2
    )

    print("\n" + "=" * 60)
    print("EJEMPLO COMPLETADO")
    print("=" * 60)
    print("Los resultados se han guardado en:")
    print("- data/stage/SO2/selected/")
    print("\nArchivos generados:")
    print("- selected_cols_CEN-TRAF_lasso_lgbm_example.json")
    print("- selected_cols_[STATION]_[SELECTOR]_[REGRESSOR].json")


if __name__ == "__main__":
    main()
