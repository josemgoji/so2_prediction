import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import optuna
import joblib

from src.utils.metrics import rmse, mape_safe, wmape
from src.services.data_preparation import DataPreparationService
from src.services.feature_engineering import DateTimeFeatures, LagFeatures, RollingStats


@dataclass
class TrainResult:
    model: RandomForestRegressor
    best_params: Dict[str, Any]
    metric: str
    val_metric: float
    test_rmse: float
    test_mape: float
    test_wmape: float
    study: optuna.Study


class RandomForestTrainingService:
    """
    Entrena un RandomForestRegressor optimizado con Optuna usando un CSV de features seleccionadas.
    Internamente divide en train/val/test (últimos 2 meses test, anteriores 2 val, resto train).
    - Minimiza la métrica especificada en validación (criterio de Optuna).
    - Métricas disponibles: "wmape", "mape", "rmse"
    - Reentrena con train+valid y evalúa en test.
    """

    def __init__(
        self,
        random_state: int = 42,
        n_trials: int = 50,
        timeout: Optional[int] = None,
        n_jobs: int = -1,
        model_output_path: Optional[str] = None,
        verbosity: int = 0,
        metric: str = "wmape"
    ):
        self.random_state = random_state
        self.n_trials = n_trials
        self.timeout = timeout
        self.n_jobs = n_jobs
        self.model_output_path = model_output_path
        self.verbosity = verbosity
        self.metric = metric

    # --------- Helpers ----------
    def _get_metric_function(self):
        """Get the metric function based on the specified metric name."""
        metric_functions = {
            "wmape": wmape,
            "mape": mape_safe,
            "rmse": rmse
        }
        if self.metric not in metric_functions:
            raise ValueError(f"Metric '{self.metric}' not supported. Available metrics: {list(metric_functions.keys())}")
        return metric_functions[self.metric]


    def _build_objective(self, X_tr, y_tr, X_val, y_val):
        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 10, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
                "max_features": trial.suggest_categorical("max_features", ["auto", "sqrt", "log2"]),
                "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
                "random_state": self.random_state,
                "n_jobs": self.n_jobs,
                "verbose": self.verbosity,
            }

            model = RandomForestRegressor(**params)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_tr, y_tr)

            y_val_pred = model.predict(X_val)
            # Optuna optimiza (minimiza) la métrica especificada
            metric_func = self._get_metric_function()
            return metric_func(y_val, y_val_pred)

        return objective

    # --------- API principal ----------
    def fit_from_features_csv(
        self,
        features_csv: str | Path,
        target_col: str = "target",
        save_model: bool = True
    ) -> TrainResult:
        # 1) Cargar y dividir datos
        data_prep = DataPreparationService()
        train_df, val_df, test_df = data_prep.load_and_split(features_csv)

        # 2) X/y
        X_tr, y_tr = data_prep.prepare_features_and_target(train_df, target_col)
        X_val, y_val = data_prep.prepare_features_and_target(val_df, target_col)
        X_te, y_te = data_prep.prepare_features_and_target(test_df, target_col)

        # 3) Optuna (minimiza la métrica especificada)
        study = optuna.create_study(direction="minimize", study_name=f"rf_opt_{self.metric}")
        study.optimize(
            self._build_objective(X_tr, y_tr, X_val, y_val),
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=False,
        )
        best_params = study.best_params
        best_val_metric = study.best_value

        # 4) Refit con train+val con los mejores params
        X_trval = pd.concat([X_tr, X_val], axis=0)
        y_trval = pd.concat([y_tr, y_val], axis=0)

        final_model = RandomForestRegressor(
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=self.verbosity,
            **best_params
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Refit final completo
            final_model.fit(X_trval, y_trval)

        # 5) Evaluación en test
        y_hat = final_model.predict(X_te)
        test_rmse = rmse(y_te, y_hat)
        test_mape_ = mape_safe(y_te, y_hat)
        test_wmape_ = wmape(y_te, y_hat)

        # 6) Guardado opcional
        if save_model:
            out_path = Path(self.model_output_path or "models/best_rf.pkl")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(
                {
                    "model": final_model,
                    "best_params": best_params,
                    "metrics": {
                        "selection_metric": self.metric,
                        f"val_{self.metric}": best_val_metric,
                        "test_rmse": test_rmse,
                        "test_mape": test_mape_,
                        "test_wmape": test_wmape_,
                    },
                    "feature_columns": list(X_tr.columns),
                    "target_col": target_col,
                },
                out_path
            )

        return TrainResult(
            model=final_model,
            best_params=best_params,
            metric=self.metric,
            val_metric=best_val_metric,
            test_rmse=test_rmse,
            test_mape=test_mape_,
            test_wmape=test_wmape_,
            study=study
        )

    def predict_future(self, model: RandomForestRegressor, last_df: pd.DataFrame, n_steps: int = 72) -> List[float]:
        """
        Predict future values recursively for n_steps ahead.

        Parameters:
        - model: Trained RandomForestRegressor
        - last_df: DataFrame with the last known row, including 'datetime', 'target', and all feature columns
        - n_steps: Number of steps to predict (default 72 for 3 days assuming hourly)

        Returns:
        - List of predicted values
        """
        predictions = []
        target_history = [last_df['target'].iloc[0]]

        for step in range(1, n_steps + 1):
            # Create future datetime
            future_dt = last_df['datetime'].iloc[0] + pd.Timedelta(hours=step)
            new_row = pd.DataFrame({'datetime': [future_dt], 'target': [0.0]})

            # Add datetime features
            dt_feat = DateTimeFeatures(time_col='datetime', country_holidays='CO', drop_time_col=False)
            new_row = dt_feat.fit_transform(new_row)

            # Add lag features
            for lag in range(1, 73):
                if lag <= len(target_history):
                    new_row[f'target_lag{lag}'] = target_history[-lag]
                else:
                    new_row[f'target_lag{lag}'] = last_df['target'].iloc[0]  # fallback

            # Add rolling stats
            for w in [3, 6, 12, 24, 48, 72]:
                if len(target_history) >= w:
                    roll_series = pd.Series(target_history[-w:])
                    new_row[f'target_roll{w}_mean'] = roll_series.mean()
                    new_row[f'target_roll{w}_std'] = roll_series.std(ddof=0)
                    new_row[f'target_roll{w}_min'] = roll_series.min()
                    new_row[f'target_roll{w}_max'] = roll_series.max()
                else:
                    # Use last known rolling stats
                    new_row[f'target_roll{w}_mean'] = last_df.get(f'target_roll{w}_mean', last_df['target'].iloc[0])
                    new_row[f'target_roll{w}_std'] = last_df.get(f'target_roll{w}_std', 0.0)
                    new_row[f'target_roll{w}_min'] = last_df.get(f'target_roll{w}_min', last_df['target'].iloc[0])
                    new_row[f'target_roll{w}_max'] = last_df.get(f'target_roll{w}_max', last_df['target'].iloc[0])

            # Add STL features (use last known)
            new_row['target_stl_trend'] = last_df.get('target_stl_trend', last_df['target'].iloc[0])
            new_row['target_stl_season'] = last_df.get('target_stl_season', 0.0)
            new_row['target_stl_resid'] = last_df.get('target_stl_resid', 0.0)

            # Prepare features for prediction
            X_pred = new_row.drop(columns=['datetime', 'target'])
            # Ensure feature order matches training (assume it does)

            pred = model.predict(X_pred)[0]
            predictions.append(pred)
            target_history.append(pred)

        return predictions