from skforecast.recursive import ForecasterEquivalentDate
import pandas as pd
from skforecast.model_selection import backtesting_forecaster, TimeSeriesFold   


from src.recursos.data_manager import DataManager
from src.utils.data_splitter import split_data_by_dates
from src.recursos.scorers import wmape, stepwise_mape_on_test, rmse

STATION = "CEN-TRAF"
forecaster = ForecasterEquivalentDate(offset=pd.DateOffset(days=1), n_offsets=1)




df = DataManager().load_data(f"data/stage/SO2/processed/processed_{STATION}.csv")
df = df.sort_index()

# Configuración de columnas
TARGET_COL = "target"

# =============================================================================
# DIVISIÓN DE DATOS EN CONJUNTOS DE ENTRENAMIENTO, VALIDACIÓN Y PRUEBA
# =============================================================================

y_train, exog_train, y_val, exog_val, y_test, exog_test, y_trainval, exog_trainval = (
    split_data_by_dates(
        df=df,
        target_col=TARGET_COL,
        exog_cols=[],
        val_months=2,
        test_months=2,
    )
)

# Entremaiento del forecaster
# ==============================================================================
forecaster.fit(y=y_trainval)
H = 72
cv_test = TimeSeriesFold(
    steps=H,
    initial_train_size=len(y_trainval),
    refit=False,
)

# Realizar backtesting en conjunto de prueba
metric_vals_test, preds_test = backtesting_forecaster(
    forecaster=forecaster,
    y=df[TARGET_COL],
    cv=cv_test,
    metric=wmape,
    n_jobs=-1,
    verbose=False,
    show_progress=False,
)

y_pred = preds_test["pred"]

common_index = y_test.index.intersection(y_pred.index)
if len(common_index) > 0:
    y_test_aligned = y_test.loc[common_index]
    y_pred_aligned = y_pred.loc[common_index]
    test_rmse = rmse(y_test_aligned, y_pred_aligned)
    test_wmape = wmape(y_test_aligned, y_pred_aligned)
    stepwise_mape_test = stepwise_mape_on_test(
        y_test_aligned, y_pred_aligned, H=H
    )