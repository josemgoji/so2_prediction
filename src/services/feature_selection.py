import numpy as np
import pandas as pd
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from typing import Tuple

from src.utils.metrics import mape_safe, wmape, rmse
from .data_preparation import DataPreparationService
from .feature_engineering import FeatureEngineeringService


class FeatureSelectionService:
    """
    Servicio para selección de características usando RFECV y LassoCV.
    Soporta datos exógenos y feature engineering integrado.
    """

    def __init__(self, random_state: int = 15926, use_exogenous: bool = True):
        """
        Initialize the FeatureSelectionService.

        Parameters:
        - random_state: Random state for reproducibility
        - use_exogenous: Whether to use exogenous data for feature selection
        """
        self.random_state = random_state
        self.use_exogenous = use_exogenous
        self.data_preparation_service = DataPreparationService(
            use_exogenous=use_exogenous
        )
        self.feature_engineering_service = FeatureEngineeringService()

    def load_and_prepare_data(
        self, data_path: str, target_col: str = "target", use_exogenous: bool = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load data and prepare features using the same pipeline as DataPreparationService.

        Parameters:
        - data_path: Path to the data CSV file
        - target_col: Name of the target column
        - use_exogenous: Whether to use exogenous data (overrides instance setting if provided)

        Returns:
        - Tuple of (X, y) where X contains features and y is the target
        """
        # Use parameter if provided, otherwise use instance setting
        use_exog = use_exogenous if use_exogenous is not None else self.use_exogenous

        # Load the data
        df = pd.read_csv(data_path)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)

        # Prepare features and target using the same logic as DataPreparationService
        X, y = self.data_preparation_service.prepare_features_and_target(
            df, target_col=target_col, use_exogenous=use_exog
        )

        return X, y

    def select_features_from_data(
        self,
        data_path: str,
        method: str = "lasso_grid",
        target_col: str = "target",
        metric: str = "wmape",
        use_exogenous: bool = None,
        **kwargs,
    ) -> Tuple[list[str], dict]:
        """
        Complete feature selection pipeline: load data, prepare features, and select features.

        Parameters:
        - data_path: Path to the data CSV file
        - method: Feature selection method ('rfecv', 'lasso', 'lasso_grid')
        - target_col: Name of the target column
        - metric: Metric to optimize ('wmape', 'mape', 'rmse')
        - use_exogenous: Whether to use exogenous data (overrides instance setting if provided)
        - **kwargs: Additional parameters for the specific feature selection method

        Returns:
        - Tuple of (selected_features, metadata) where metadata contains method-specific info
        """
        print(f"Loading and preparing data from {data_path}...")
        X, y = self.load_and_prepare_data(data_path, target_col, use_exogenous)

        print(f"Data shape: {X.shape}, Target shape: {y.shape}")
        print(
            f"Using exogenous data: {use_exogenous if use_exogenous is not None else self.use_exogenous}"
        )

        # Select features based on method
        if method == "rfecv":
            selected_features = self.select_features_rfecv(
                X, y, metric=metric, **kwargs
            )
            metadata = {"method": "rfecv", "metric": metric}
        elif method == "lasso":
            selected_features = self.select_features_lasso(
                X, y, metric=metric, **kwargs
            )
            metadata = {"method": "lasso", "metric": metric}
        elif method == "lasso_grid":
            selected_features, best_alpha, coefs, grid_search = (
                self.select_features_lasso_grid(X, y, metric=metric, **kwargs)
            )
            metadata = {
                "method": "lasso_grid",
                "metric": metric,
                "best_alpha": best_alpha,
                "coefficients": coefs,
                "grid_search": grid_search,
            }
        else:
            raise ValueError(
                f"Unknown method: {method}. Use 'rfecv', 'lasso', or 'lasso_grid'"
            )

        print(
            f"Selected {len(selected_features)} features out of {X.shape[1]} total features"
        )

        return selected_features, metadata

    def save_selected_features(
        self,
        data_path: str,
        selected_features: list[str],
        output_path: str,
        target_col: str = "target",
        use_exogenous: bool = None,
    ) -> None:
        """
        Save selected features to a CSV file for use in model training.

        Parameters:
        - data_path: Path to the original data CSV file
        - selected_features: List of selected feature names
        - output_path: Path where to save the selected features CSV
        - target_col: Name of the target column
        - use_exogenous: Whether to use exogenous data (overrides instance setting if provided)
        """
        print("Loading data and preparing features...")
        X, y = self.load_and_prepare_data(data_path, target_col, use_exogenous)

        # Create DataFrame with selected features and target
        selected_df = X[selected_features].copy()
        selected_df[target_col] = y

        # Add datetime column if available in original data
        df_original = pd.read_csv(data_path)
        if "datetime" in df_original.columns:
            df_original["datetime"] = pd.to_datetime(df_original["datetime"])
            df_original = df_original.sort_values("datetime").reset_index(drop=True)

            # Align datetime with processed data (after feature engineering and dropna)
            # We need to account for the fact that feature engineering may drop some rows
            # Get the datetime values that correspond to the processed data
            datetime_aligned = (
                df_original["datetime"].iloc[: len(selected_df)].reset_index(drop=True)
            )
            selected_df["datetime"] = datetime_aligned

        # Save to CSV
        selected_df.to_csv(output_path, index=False)
        print(f"Selected features saved to {output_path}")
        print(f"Saved {len(selected_features)} features + target + datetime")

    def run_complete_pipeline(
        self,
        data_path: str,
        output_path: str,
        method: str = "lasso_grid",
        target_col: str = "target",
        metric: str = "wmape",
        use_exogenous: bool = None,
        **kwargs,
    ) -> Tuple[list[str], dict]:
        """
        Complete feature selection pipeline: load data, select features, and save results.

        Parameters:
        - data_path: Path to the data CSV file
        - output_path: Path where to save the selected features CSV
        - method: Feature selection method ('rfecv', 'lasso', 'lasso_grid')
        - target_col: Name of the target column
        - metric: Metric to optimize ('wmape', 'mape', 'rmse')
        - use_exogenous: Whether to use exogenous data (overrides instance setting if provided)
        - **kwargs: Additional parameters for the specific feature selection method

        Returns:
        - Tuple of (selected_features, metadata) where metadata contains method-specific info
        """
        print("=" * 60)
        print("FEATURE SELECTION PIPELINE")
        print("=" * 60)

        # Step 1: Select features
        selected_features, metadata = self.select_features_from_data(
            data_path=data_path,
            method=method,
            target_col=target_col,
            metric=metric,
            use_exogenous=use_exogenous,
            **kwargs,
        )

        # Step 2: Save selected features
        self.save_selected_features(
            data_path=data_path,
            selected_features=selected_features,
            output_path=output_path,
            target_col=target_col,
            use_exogenous=use_exogenous,
        )

        print("=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 60)

        return selected_features, metadata

    def select_features_rfecv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        metric: str = "mape",
        step: int = 2,
        cv_splits: int = 3,
        n_jobs: int = -1,
    ) -> list[str]:
        """
        Selección de características usando RFECV con RandomForest.
        metric: 'mape' or 'rmse'
        """
        rf = RandomForestRegressor(
            n_estimators=10, max_depth=5, n_jobs=n_jobs, random_state=self.random_state
        )

        tscv = TimeSeriesSplit(n_splits=cv_splits)

        if metric == "mape":
            scorer = make_scorer(wmape, greater_is_better=False)
        elif metric == "rmse":
            scorer = make_scorer(rmse, greater_is_better=False)
        else:
            raise ValueError("metric must be 'mape' or 'rmse'")

        selector = RFECV(
            estimator=rf,
            step=step,
            cv=tscv,
            scoring=scorer,
            n_jobs=n_jobs,
        )

        selector.fit(X, y)
        selected_mask = selector.support_
        selected_cols = X.columns[selected_mask].tolist()

        return selected_cols

    def select_features_lasso(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        metric: str = "mape",
        alphas: np.ndarray = None,
        cv_splits: int = 3,
        n_jobs: int = -1,
    ) -> list[str]:
        """
        Selección de características usando LassoCV (optimiza MSE internamente).
        """
        if alphas is None:
            alphas = np.logspace(-5, 0, 150)  # reducido para LassoCV

        # Estandarizar
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X), index=X.index, columns=X.columns
        )

        tscv = TimeSeriesSplit(n_splits=cv_splits)

        lasso_cv = LassoCV(
            alphas=alphas,
            cv=tscv,
            max_iter=1000,
            n_jobs=n_jobs,
            random_state=self.random_state,
        )

        lasso_cv.fit(X_scaled, y)

        # Mejor modelo
        coefs = pd.Series(lasso_cv.coef_, index=X.columns)
        selected_cols = coefs[coefs.abs() > 1e-8].index.tolist()

        return selected_cols

    def select_features_lasso_grid(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        metric: str = "wmape",
        alphas: np.ndarray | None = None,
        cv_splits: int = 3,
        n_jobs: int = -1,
        max_iter: int = 5000,
        tol: float = 1e-3,
    ):
        """
        Selección de características con Lasso usando GridSearchCV + métrica custom.
        Devuelve: (selected_cols, best_alpha, coefs_series, grid_search)
        """
        if alphas is None:
            # Rango más conservador de alphas para mejor convergencia
            alphas = np.logspace(-4, 0, 100)

        # scorer según la métrica
        if metric.lower() == "wmape" or metric.lower() == "mape":
            scorer = make_scorer(mape_safe, greater_is_better=False)
        elif metric.lower() == "rmse":
            scorer = make_scorer(rmse, greater_is_better=False)
        else:
            raise ValueError("metric must be 'wmape'/'mape' or 'rmse'")

        # Pipeline para evitar leakage
        pipe = Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                (
                    "lasso",
                    Lasso(max_iter=max_iter, tol=tol, random_state=self.random_state),
                ),
            ]
        )

        # Grid de alphas (en el paso 'lasso' del pipeline)
        param_grid = {"lasso__alpha": alphas}

        tscv = TimeSeriesSplit(n_splits=cv_splits)

        gs = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring=scorer,
            cv=tscv,
            n_jobs=n_jobs,
            refit=True,  # reentrena con el mejor alpha sobre todo X,y
            verbose=0,
        )

        # Suprimir warnings de convergencia durante el grid search
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
            gs.fit(X, y)

        best_alpha = float(gs.best_params_["lasso__alpha"])
        best_pipe = gs.best_estimator_
        lasso: Lasso = best_pipe.named_steps["lasso"]

        # Verificar convergencia del mejor modelo
        if not lasso.n_iter_ < max_iter:
            print(
                f"Warning: Lasso no convergió completamente (iteraciones: {lasso.n_iter_}/{max_iter})"
            )

        # Coefs en el espacio escalado (sirven para selección 0 vs ≠ 0)
        coefs = pd.Series(lasso.coef_, index=X.columns)

        # Selección: coeficientes no nulos (umbral pequeño para estabilidad numérica)
        selected_cols = coefs[coefs.abs() > 1e-8].index.tolist()

        return selected_cols, best_alpha, coefs, gs
