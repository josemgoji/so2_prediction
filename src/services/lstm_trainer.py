import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import optuna
import joblib
from src.utils.metrics import rmse, mape_safe, wmape


# --------- Métricas ----------
def rmse_local(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def mape_safe_local(y_true, y_pred, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)))

def wmape_local(y_true, y_pred, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    return float(np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + eps))


@dataclass
class TrainResult:
    model: Sequential
    best_params: Dict[str, Any]
    metric: str
    val_metric: float
    test_rmse: float
    test_mape: float
    test_wmape: float
    study: optuna.Study
    scaler: MinMaxScaler


class LSTMTrainingService:
    """
    Entrena un modelo LSTM optimizado con Optuna usando CSVs ya separados:
      train.csv, val.csv, test.csv
    - Minimiza la métrica especificada en validación (criterio de Optuna).
    - Métricas disponibles: "wmape", "mape", "rmse"
    - Usa normalización MinMax para los datos.
    - Reentrena con train+valid y evalúa en test.
    - Compatible con datos de series temporales.
    """

    def __init__(
        self,
        random_state: int = 42,
        n_trials: int = 50,
        timeout: Optional[int] = None,
        early_stopping_patience: int = 20,
        reduce_lr_patience: int = 10,
        epochs: int = 200,
        batch_size: int = 32,
        sequence_length: int = 60,
        model_output_path: Optional[str] = None,
        verbosity: int = 1,
        metric: str = "wmape"
    ):
        self.random_state = random_state
        self.n_trials = n_trials
        self.timeout = timeout
        self.early_stopping_patience = early_stopping_patience
        self.reduce_lr_patience = reduce_lr_patience
        self.epochs = epochs
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.model_output_path = model_output_path
        self.verbosity = verbosity
        self.metric = metric
        
        # Set random seeds for reproducibility
        np.random.seed(random_state)
        tf.random.set_seed(random_state)

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

    def _create_sequences(self, data: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM input."""
        X, y = [], []
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i])
            y.append(target[i])
        return np.array(X), np.array(y)

    def _prepare_data(self, X: pd.DataFrame, y: pd.Series, scaler: Optional[MinMaxScaler] = None) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
        """Prepare data for LSTM training."""
        # Combine features and target for scaling
        data = pd.concat([X, y], axis=1)
        
        if scaler is None:
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data)
        else:
            scaled_data = scaler.transform(data)
        
        # Separate features and target
        X_scaled = scaled_data[:, :-1]
        y_scaled = scaled_data[:, -1]
        
        # Create sequences
        X_seq, y_seq = self._create_sequences(X_scaled, y_scaled)
        
        return X_seq, y_seq, scaler

    def _build_lstm_model(self, params: Dict[str, Any], input_shape: Tuple[int, int]) -> Sequential:
        """Build LSTM model with given parameters."""
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(
            units=params['lstm_units_1'],
            return_sequences=True,
            input_shape=input_shape
        ))
        model.add(Dropout(params['dropout_1']))
        
        # Second LSTM layer (if specified)
        if params.get('lstm_units_2', 0) > 0:
            model.add(LSTM(
                units=params['lstm_units_2'],
                return_sequences=params.get('lstm_units_3', 0) > 0
            ))
            model.add(Dropout(params['dropout_2']))
        
        # Third LSTM layer (if specified)
        if params.get('lstm_units_3', 0) > 0:
            model.add(LSTM(units=params['lstm_units_3']))
            model.add(Dropout(params['dropout_3']))
        
        # Dense layers
        for i, units in enumerate(params.get('dense_units', [50])):
            model.add(Dense(units, activation='relu'))
            if params.get(f'dense_dropout_{i}', 0) > 0:
                model.add(Dropout(params[f'dense_dropout_{i}']))
        
        # Output layer
        model.add(Dense(1))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=params['learning_rate']),
            loss='mse',
            metrics=['mae']
        )
        
        return model

    def _build_objective(self, X_tr, y_tr, X_val, y_val, scaler: MinMaxScaler):
        def objective(trial: optuna.Trial) -> float:
            # RED PEQUEÑA PARA PRUEBAS RÁPIDAS
            params = {
                'lstm_units_1': trial.suggest_int('lstm_units_1', 16, 64),      # Reducido de 32-256
                'lstm_units_2': trial.suggest_int('lstm_units_2', 0, 32),       # Reducido de 0-128
                'lstm_units_3': trial.suggest_int('lstm_units_3', 0, 16),       # Reducido de 0-64
                'dropout_1': trial.suggest_float('dropout_1', 0.1, 0.3),        # Reducido de 0.1-0.5
                'dropout_2': trial.suggest_float('dropout_2', 0.1, 0.3),        # Reducido de 0.1-0.5
                'dropout_3': trial.suggest_float('dropout_3', 0.1, 0.3),        # Reducido de 0.1-0.5
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-2, log=True),  # Rango más pequeño
                'dense_units': trial.suggest_categorical('dense_units', [[16], [32], [16, 8], [32, 16]]),  # Reducido
            }

            # RED RECOMENDADA PARA PRODUCCIÓN (comentada):
            # params = {
            #     'lstm_units_1': trial.suggest_int('lstm_units_1', 64, 256),
            #     'lstm_units_2': trial.suggest_int('lstm_units_2', 0, 128),
            #     'lstm_units_3': trial.suggest_int('lstm_units_3', 0, 64),
            #     'dropout_1': trial.suggest_float('dropout_1', 0.1, 0.5),
            #     'dropout_2': trial.suggest_float('dropout_2', 0.1, 0.5),
            #     'dropout_3': trial.suggest_float('dropout_3', 0.1, 0.5),
            #     'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            #     'dense_units': trial.suggest_categorical('dense_units', [[32], [64], [32, 16], [64, 32], [128, 64]]),
            # }
            
            # Add dense dropout parameters based on dense_units
            dense_units = params['dense_units']
            for i in range(len(dense_units)):
                params[f'dense_dropout_{i}'] = trial.suggest_float(f'dense_dropout_{i}', 0.0, 0.2)  # Reducido de 0.3
            
            # Build model
            input_shape = (X_tr.shape[1], X_tr.shape[2])
            model = self._build_lstm_model(params, input_shape)
            
            # Callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=self.early_stopping_patience,
                    restore_best_weights=True,
                    verbose=0
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=self.reduce_lr_patience,
                    min_lr=1e-6,
                    verbose=0
                )
            ]
            
            # Train model
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                history = model.fit(
                    X_tr, y_tr,
                    validation_data=(X_val, y_val),
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    callbacks=callbacks,
                    verbose=0
                )
            
            # Predict and evaluate
            y_val_pred = model.predict(X_val, verbose=0).flatten()
            
            # Inverse transform predictions and actual values
            # Create dummy data for inverse transform
            dummy_data = np.zeros((len(y_val_pred), X_tr.shape[2] + 1))
            dummy_data[:, -1] = y_val_pred
            y_val_pred_original = scaler.inverse_transform(dummy_data)[:, -1]
            
            dummy_data[:, -1] = y_val
            y_val_original = scaler.inverse_transform(dummy_data)[:, -1]
            
            # Calculate metric
            metric_func = self._get_metric_function()
            return metric_func(y_val_original, y_val_pred_original)

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

        # 3) Prepare data with scaling
        X_tr_seq, y_tr_seq, scaler = self._prepare_data(X_tr, y_tr)
        X_val_seq, y_val_seq, _ = self._prepare_data(X_val, y_val, scaler)
        X_te_seq, y_te_seq, _ = self._prepare_data(X_te, y_te, scaler)

        # 4) Optuna optimization
        study = optuna.create_study(direction="minimize", study_name=f"lstm_opt_{self.metric}")
        study.optimize(
            self._build_objective(X_tr_seq, y_tr_seq, X_val_seq, y_val_seq, scaler),
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=False,
        )
        best_params = study.best_params
        best_val_metric = study.best_value

        # 5) Refit with train+val using best parameters
        X_trval = pd.concat([X_tr, X_val], axis=0)
        y_trval = pd.concat([y_tr, y_val], axis=0)
        X_trval_seq, y_trval_seq, _ = self._prepare_data(X_trval, y_trval, scaler)

        # Build final model
        input_shape = (X_trval_seq.shape[1], X_trval_seq.shape[2])
        final_model = self._build_lstm_model(best_params, input_shape)

        # Train final model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            callbacks = [
                EarlyStopping(
                    monitor='loss',
                    patience=self.early_stopping_patience,
                    restore_best_weights=True,
                    verbose=0
                ),
                ReduceLROnPlateau(
                    monitor='loss',
                    factor=0.5,
                    patience=self.reduce_lr_patience,
                    min_lr=1e-6,
                    verbose=0
                )
            ]
            
            final_model.fit(
                X_trval_seq, y_trval_seq,
                epochs=self.epochs,
                batch_size=self.batch_size,
                callbacks=callbacks,
                verbose=0
            )

        # 6) Evaluate on test set
        y_hat_scaled = final_model.predict(X_te_seq, verbose=0).flatten()
        
        # Inverse transform predictions and actual values
        dummy_data = np.zeros((len(y_hat_scaled), X_te_seq.shape[2] + 1))
        dummy_data[:, -1] = y_hat_scaled
        y_hat = scaler.inverse_transform(dummy_data)[:, -1]
        
        dummy_data[:, -1] = y_te_seq
        y_te_original = scaler.inverse_transform(dummy_data)[:, -1]
        
        test_rmse = rmse(y_te_original, y_hat)
        test_mape_ = mape_safe(y_te_original, y_hat)
        test_wmape_ = wmape(y_te_original, y_hat)

        # 7) Save model
        if save_model:
            out_path = Path(self.model_output_path or "models/best_lstm.pkl")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save model and scaler separately
            model_path = out_path.with_suffix('.h5')
            scaler_path = out_path.with_suffix('_scaler.pkl')
            
            final_model.save(model_path)
            joblib.dump(scaler, scaler_path)
            
            # Save metadata
            joblib.dump(
                {
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
                    "sequence_length": self.sequence_length,
                    "model_path": str(model_path),
                    "scaler_path": str(scaler_path),
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
            study=study,
            scaler=scaler
        )
