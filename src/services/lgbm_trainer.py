import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import pandas as pd
from lightgbm import LGBMRegressor
from lightgbm import early_stopping as lgb_early_stopping, log_evaluation as lgb_log_evaluation
import optuna
import joblib

from src.utils.metrics import rmse, mape_safe, wmape


@dataclass
class TrainResult:
    model: LGBMRegressor
    best_params: Dict[str, Any]
    metric: str
    val_metric: float
    test_rmse: float
    test_mape: float
    test_wmape: float
    study: optuna.Study


class LightGBMTrainingService:
    """
    Entrena un LGBMRegressor optimizado con Optuna usando CSVs ya separados:
      train.csv, val.csv, test.csv
    - Minimiza la métrica especificada en validación (criterio de Optuna).
    - Métricas disponibles: "wmape", "mape", "rmse"
    - Usa métricas internas de LightGBM para logging/early stopping (callbacks v4+).
    - Reentrena con train+valid y evalúa en test.
    """

    def __init__(
        self,
        random_state: int = 42,
        n_trials: int = 50,
        timeout: Optional[int] = None,
        early_stopping_rounds: int = 100,
        n_jobs: int = -1,
        model_output_path: Optional[str] = None,
        verbosity: int = -1,
        metric: str = "wmape"
    ):
        self.random_state = random_state
        self.n_trials = n_trials
        self.timeout = timeout
        self.early_stopping_rounds = early_stopping_rounds
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

    @staticmethod
    def _read_csv(path: str | Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        if "datetime" in df.columns:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        return df

    @staticmethod
    def _xy(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
        if target_col not in df.columns:
            raise ValueError(f"target_col '{target_col}' no existe en el DataFrame.")
        drop_cols = [target_col]
        if "datetime" in df.columns:
            drop_cols.append("datetime")
        X = df.drop(columns=drop_cols)
        y = df[target_col]
        return X, y

    def _fit_with_callbacks(self, model: LGBMRegressor, X_tr, y_tr, X_val, y_val):
        """Entrenamiento usando solo callbacks de LightGBM v4+ (sin compatibilidad v3)."""
        callbacks = []
        if self.early_stopping_rounds and self.early_stopping_rounds > 0:
            callbacks.append(lgb_early_stopping(self.early_stopping_rounds, verbose=False))
        callbacks.append(lgb_log_evaluation(period=0))
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=callbacks,
        )

    def _build_objective(self, X_tr, y_tr, X_val, y_val):
        def objective(trial: optuna.Trial) -> float:
            # Map metric names to LightGBM internal metrics
            lgb_metric_map = {
                "wmape": "mape",  # Use MAPE as closest approximation
                "mape": "mape",
                "rmse": "rmse"
            }

            params = {
                "objective": "regression",
                "metric": lgb_metric_map.get(self.metric, "mape"),  # métrica interna informativa
                "random_state": self.random_state,
                "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 255),
                "max_depth": trial.suggest_int("max_depth", -1, 16),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 200),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
                "n_jobs": self.n_jobs,
                "verbosity": self.verbosity,
            }

            model = LGBMRegressor(**params)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._fit_with_callbacks(model, X_tr, y_tr, X_val, y_val)

            y_val_pred = model.predict(X_val)
            # Optuna optimiza (minimiza) la métrica especificada
            metric_func = self._get_metric_function()
            return metric_func(y_val, y_val_pred)

        return objective

    # --------- API principal ----------
    def fit_from_splits(
        self,
        train_csv: str | Path,
        val_csv: str | Path,
        test_csv: str | Path,
        target_col: str = "target",
        save_model: bool = True
    ) -> TrainResult:
        # 1) Cargar splits
        train_df = self._read_csv(train_csv)
        val_df = self._read_csv(val_csv)
        test_df = self._read_csv(test_csv)

        # 2) X/y
        X_tr, y_tr = self._xy(train_df, target_col)
        X_val, y_val = self._xy(val_df, target_col)
        X_te, y_te = self._xy(test_df, target_col)

        # 3) Optuna (minimiza la métrica especificada)
        study = optuna.create_study(direction="minimize", study_name=f"lgbm_opt_{self.metric}")
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

        # Map metric names to LightGBM internal metrics
        lgb_metric_map = {
            "wmape": "mape",  # Use MAPE as closest approximation
            "mape": "mape",
            "rmse": "rmse"
        }

        final_model = LGBMRegressor(
            objective="regression",
            metric=lgb_metric_map.get(self.metric, "mape"),
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbosity=self.verbosity,
            **best_params
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Refit final completo (sin early stopping)
            final_model.fit(X_trval, y_trval)

        # 5) Evaluación en test
        y_hat = final_model.predict(X_te)
        test_rmse = rmse(y_te, y_hat)
        test_mape_ = mape_safe(y_te, y_hat)
        test_wmape_ = wmape(y_te, y_hat)

        # 6) Guardado opcional
        if save_model:
            out_path = Path(self.model_output_path or "models/best_lgbm.pkl")
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

