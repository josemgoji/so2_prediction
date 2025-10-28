# 🧠 Integración de Modelo de Deep Learning con PyTorch y GPU

## ✅ Implementación Completada

He integrado exitosamente un modelo de deep learning usando PyTorch con soporte GPU en tu estructura existente de skforecast. Aquí está lo que se ha implementado:

### 📁 Archivos Creados/Modificados:

1. **`src/recursos/deep_learning_regressor.py`** - Nuevo archivo con:
   - `LSTMForecastingModel`: Modelo LSTM personalizado
   - `DeepLearningRegressor`: Wrapper compatible con sklearn/skforecast
   - `create_deep_learning_regressor()`: Función factory
   - `create_and_compile_model()`: Función específica para tu interfaz

2. **`src/recursos/regressors.py`** - Modificado para incluir:
   - `create_deep_learning_regressor_wrapper()`: Wrapper para integración
   - Actualización de la función factory `create_regressor()`

3. **`src/constants/parsed_fields.py`** - Modificado para incluir:
   - Configuración del regresor DeepLearning en `REGRESSORS_CONFIG`

4. **`models_deep_lerning.py`** - Modificado para incluir:
   - Import del nuevo regresor
   - Mapeo en `regressor_func_map`

5. **`example_deep_learning.py`** - Ejemplo de uso específico

### 🚀 Características Implementadas:

#### ✅ Compatibilidad con skforecast
- El modelo es completamente compatible con `ForecasterRecursive`
- Soporte para variables exógenas (`exog`)
- Soporte para window features
- Soporte para weight functions (gaps)

#### ✅ Soporte GPU Automático
- Detección automática de GPU CUDA
- Uso de GPU cuando está disponible
- Fallback a CPU si no hay GPU

#### ✅ Configuración Flexible
- Múltiples capas LSTM configurables
- Capas densas configurables
- Dropout, activaciones, optimizadores configurables
- Early stopping con patience

#### ✅ Interfaz Familiar
- Compatible con la función `create_and_compile_model()` que especificaste
- Parámetros similares a tu ejemplo original
- Integración transparente con tu pipeline existente

### 🔧 Uso del Modelo:

#### Opción 1: Usar con tu estructura existente
```python
# En models_deep_lerning.py, el modelo DeepLearning ya está configurado
# Solo ejecuta el archivo y se entrenará automáticamente
python models_deep_lerning.py
```

#### Opción 2: Usar el ejemplo específico
```python
# Ejemplo dedicado solo para deep learning
python example_deep_learning.py
```

#### Opción 3: Usar directamente la función create_and_compile_model
```python
from src.recursos.deep_learning_regressor import create_and_compile_model

model = create_and_compile_model(
    series=data_exog[['users']],
    levels=['users'],
    lags=72,
    steps=36,
    exog=data_exog[exog_features],
    recurrent_layer="LSTM",
    recurrent_units=[128, 64],
    recurrent_layers_kwargs={"activation": "tanh"},
    dense_units=[64, 32],
    compile_kwargs={'optimizer': 'adam', 'loss': 'mse'},
    model_name="Single-Series-Multi-Step-Exog"
)
```

### 🎯 Configuración Actual:

El modelo está configurado con:
- **GPU**: Detección automática (RTX 3080 detectada ✅)
- **LSTM Layers**: [128, 64] unidades
- **Dense Layers**: [64, 32] unidades
- **Optimizador**: Adam con learning_rate=0.01
- **Loss**: MSE
- **Activación**: tanh
- **Dropout**: 0.2
- **Early Stopping**: patience=10

### 📊 Parámetros para Random Search:

El modelo incluye estos parámetros para optimización:
- `hidden_sizes`: [[128, 64], [64, 32], [256, 128]]
- `dense_units`: [[64, 32], [32, 16], [128, 64]]
- `dropout`: [0.1, 0.2, 0.3]
- `learning_rate`: [0.001, 0.01, 0.05]
- `batch_size`: [16, 32, 64]
- `epochs`: [50, 100, 150]
- `patience`: [5, 10, 15]
- `optimizer`: ["adam", "sgd"]
- `activation`: ["tanh", "relu"]

### 🧪 Pruebas Realizadas:

✅ **Verificación GPU**: PyTorch 2.8.0 con CUDA disponible  
✅ **Creación de modelo**: Funciona correctamente  
✅ **Entrenamiento**: Entrenamiento básico exitoso  
✅ **Predicción**: Predicciones generadas correctamente  
✅ **Integración skforecast**: Compatible con ForecasterRecursive  

### 🚀 Próximos Pasos:

1. **Ejecutar el modelo**: 
   ```bash
   python models_deep_lerning.py
   ```

2. **Comparar resultados**: El modelo DeepLearning se entrenará junto con LGBM y otros

3. **Ajustar parámetros**: Modificar `REGRESSORS_CONFIG` según necesidades

4. **Monitorear GPU**: El modelo usará automáticamente tu RTX 3080

### 💡 Notas Importantes:

- El modelo usa **n_jobs=1** para deep learning (no se beneficia de paralelización)
- Los **epochs** están configurados para entrenamiento completo (100 por defecto)
- El **early stopping** previene overfitting
- La **GPU** se usa automáticamente cuando está disponible
- Compatible con **weight functions** para manejar gaps en datos

¡El modelo de deep learning está listo para usar con tu estructura existente! 🎉
