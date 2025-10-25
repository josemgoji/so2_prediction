# Cambios Realizados al Modelo Recursivo

## Resumen
He adaptado el modelo recursivo (`src/models/train/recursive/model.py`) para que funcione perfectamente con tu estructura de datos y servicios existentes.

## Principales Cambios

### 1. **Integración con tus Servicios**
- **FeatureEngineeringService**: Ahora usa tu `FeatureEngineeringService` en lugar de las transformaciones manuales
- **DataPreparationService**: Integrado con tu `DataPreparationService` para cargar y dividir datos
- **Soporte para exógenas**: Maneja tanto datos con variables exógenas como sin ellas

### 2. **Filtrado por Variables Seleccionadas**
- **JSON Support**: Puede cargar variables seleccionadas desde un archivo JSON
- **CSV Support**: También funciona con archivos CSV directamente
- **Flexibilidad**: Puedes especificar las variables a usar de múltiples formas

### 3. **Parámetros Simplificados**
```python
# Antes (parámetros complejos)
service = GenericRecursiveTunedService(
    lags=tuple(range(1, 73)),
    roll_windows=(3, 6, 12, 24, 48, 72),
    use_stl=True,
    stl_period=24,
    # ... muchos más parámetros
)

# Ahora (simplificado)
service = GenericRecursiveTunedService(
    use_exogenous=True,  # o False
    n_trials=40,
    metric="wmape"
)
```

### 4. **Métodos Actualizados**

#### `fit_with_optuna()`
```python
# Opción 1: Con variables seleccionadas desde JSON
result = service.fit_with_optuna(
    features_csv="data/stage/SO2/enrichment/enriched_MED-FISC.csv",
    selected_features_path="selected_features.json"
)

# Opción 2: Con lista de variables
result = service.fit_with_optuna(
    features_csv="data/stage/SO2/enrichment/enriched_MED-FISC.csv",
    feature_columns=["target_lag1", "target_lag24", "hour", "dow"]
)

# Opción 3: Usar todas las features disponibles
result = service.fit_with_optuna(
    features_csv="data/stage/SO2/enrichment/enriched_MED-FISC.csv"
)
```

#### `evaluate_recursive_on_test()`
```python
eval_results = service.evaluate_recursive_on_test(
    features_csv="data/stage/SO2/enrichment/enriched_MED-FISC.csv",
    trained_model=result.model,
    feature_columns=result.feature_columns
)
```

## Archivos Creados

### 1. **test_recursive_model.py**
Script completo de pruebas que valida:
- ✅ Funcionamiento con datos exógenos
- ✅ Funcionamiento sin datos exógenos  
- ✅ Integración con datos enriched reales
- ✅ Filtrado por variables seleccionadas

### 2. **example_recursive_usage.py**
Ejemplos prácticos de uso:
- Ejemplo con todas las features
- Ejemplo con variables seleccionadas
- Ejemplo sin variables exógenas

## Cómo Usar

### Paso 1: Entrenar el Modelo
```python
from src.models.train.recursive.model import GenericRecursiveTunedService

# Crear servicio
service = GenericRecursiveTunedService(
    use_exogenous=True,  # True si tienes variables exógenas
    n_trials=40,
    metric="wmape"
)

# Entrenar
result = service.fit_with_optuna(
    features_csv="data/stage/SO2/enrichment/enriched_MED-FISC.csv",
    selected_features_path="mi_seleccion.json"  # opcional
)
```

### Paso 2: Evaluar Recursivamente
```python
# Evaluar en test
eval_results = service.evaluate_recursive_on_test(
    features_csv="data/stage/SO2/enrichment/enriched_MED-FISC.csv",
    trained_model=result.model,
    feature_columns=result.feature_columns
)

# Ver métricas
print(f"WMAPE: {eval_results['metrics_block']['wmape']:.4f}")
print(f"RMSE: {eval_results['metrics_block']['rmse']:.4f}")
```

## Compatibilidad

### ✅ **Con Variables Exógenas**
- Usa todas las columnas del CSV enriched
- Aplica feature engineering completo
- Maneja variables meteorológicas y de otras estaciones

### ✅ **Sin Variables Exógenas**  
- Solo usa `datetime` y `target`
- Crea features básicas (lags, rolling, datetime)
- Ideal para comparaciones o cuando no hay exógenas

### ✅ **Con Variables Seleccionadas**
- Filtra por JSON con `selected_features`
- Usa solo las variables que especifiques
- Optimiza el entrenamiento

## Próximos Pasos

1. **Ejecutar pruebas**: `python test_recursive_model.py`
2. **Probar ejemplos**: `python example_recursive_usage.py`
3. **Integrar en tu pipeline**: Usar el modelo en tu flujo de trabajo
4. **Ajustar parámetros**: Modificar `n_trials`, `metric`, etc. según necesites

## Notas Importantes

- **Sin fuga de datos**: El feature engineering se recalcula en cada paso recursivo
- **Métricas acumuladas**: Proporciona métricas por horizonte (1, 2, 3, ..., H)
- **Flexibilidad total**: Funciona con cualquier estructura de datos que tengas
- **Integración perfecta**: Usa exactamente los mismos servicios que tu pipeline actual
