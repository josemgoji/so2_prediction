# --- imports básicos ---
import warnings
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestRegressor


# ---- Selector estilo scikit para usar dentro de skforecast.select_features ----
class LassoGridSelector(BaseEstimator):
    """
    Selector de características con Lasso + GridSearchCV y métrica custom.
    Compatible con skforecast.feature_selection.select_features (requiere get_support()).
    """

    def __init__(
        self,
        alphas=None,
        scorer=None,  # scorer directo de sklearn
        cv_splits: int = 3,
        n_jobs: int = -1,
        max_iter: int = 5000,
        tol: float = 1e-3,
        random_state: int | None = 123,
        coef_threshold: float = 1e-7,
    ):
        self.alphas = alphas
        self.scorer = scorer
        self.cv_splits = cv_splits
        self.n_jobs = n_jobs
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.coef_threshold = coef_threshold

        # atributos tras fit()
        self.best_alpha_ = None
        self.best_estimator_ = None
        self.support_mask_ = None
        self.feature_names_in_ = None
        self.coefs_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        if self.alphas is None:

            self.alphas = np.logspace(-6, -3, 10)

        self.feature_names_in_ = np.array(X.columns)

        pipe = Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                (
                    "lasso",
                    Lasso(
                        max_iter=self.max_iter,
                        tol=self.tol,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )

        param_grid = {"lasso__alpha": self.alphas}
        tscv = TimeSeriesSplit(n_splits=self.cv_splits)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
            gs = GridSearchCV(
                estimator=pipe,
                param_grid=param_grid,
                scoring=self.scorer,
                cv=tscv,
                n_jobs=self.n_jobs,
                refit=True,
                verbose=0,
            )
            gs.fit(X, y)

        self.best_alpha_ = float(gs.best_params_["lasso__alpha"])
        self.best_estimator_ = gs.best_estimator_

        # Coeficientes del mejor Lasso (en espacio escalado; válidos para selección 0 vs ≠ 0)
        lasso: Lasso = self.best_estimator_.named_steps["lasso"]
        self.coefs_ = pd.Series(lasso.coef_, index=self.feature_names_in_)

        # Soporte: |coef| > umbral
        self.support_mask_ = self.coefs_.abs().values > self.coef_threshold
        return self

    def get_support(self, indices: bool = False):
        if self.support_mask_ is None:
            raise RuntimeError("Call fit() before get_support().")
        if indices:
            return np.nonzero(self.support_mask_)[0]
        return self.support_mask_

    # Opcional, por compatibilidad con algunos flujos de sklearn
    def transform(self, X: pd.DataFrame):
        if self.support_mask_ is None:
            raise RuntimeError("Call fit() before transform().")
        keep_cols = self.feature_names_in_[self.support_mask_]
        return X.loc[:, keep_cols]
    
    def get_feature_names_out(self, input_features=None):
        """
        Get output feature names for transformation.
        Required by skforecast.
        """
        if self.support_mask_ is None:
            raise RuntimeError("Call fit() before get_feature_names_out().")
        return self.feature_names_in_[self.support_mask_]


# ---- Factory para seleccionar entre Lasso y RFECV ----
class FeatureSelectorFactory:
    """
    Factory para crear selectores de características.
    Permite cambiar fácilmente entre Lasso y RFECV según un parámetro.
    """

    @staticmethod
    def create_selector(
        selector_type: str = "lasso",  # "lasso" o "rfecv"
        scorer=None,  # scorer directo de sklearn
        cv_splits: int = 3,
        n_jobs: int = -1,
        random_state: int | None = 123,
        # Parámetros específicos de Lasso
        alphas=None,
        max_iter: int = 5000,
        tol: float = 1e-3,
        coef_threshold: float = 1e-8,
        # Parámetros específicos de RFECV
        estimator=None,
        step: int = 1,
        min_features_to_select: int = 1,
        verbose: int = 0,
    ):
        """
        Crea un selector de características según el tipo especificado.

        Parameters:
        -----------
        selector_type : str, default="lasso"
            Tipo de selector: "lasso" o "rfecv"
        scorer : callable, default=None
            Scorer de sklearn (ej: wmape_scorer, rmse_scorer, "mae", etc.)
        cv_splits : int, default=3
            Número de splits para validación cruzada
        n_jobs : int, default=-1
            Número de jobs para paralelización
        random_state : int, default=123
            Semilla para reproducibilidad

        # Parámetros específicos de Lasso
        alphas : array-like, optional
            Valores de alpha para GridSearch en Lasso
        max_iter : int, default=5000
            Máximo de iteraciones para Lasso
        tol : float, default=1e-3
            Tolerancia para convergencia de Lasso
        coef_threshold : float, default=1e-8
            Umbral para considerar coeficientes no nulos

        # Parámetros específicos de RFECV
        estimator : estimator, optional
            Estimador base para RFECV (por defecto RandomForest)
        step : int, default=1
            Número de características a eliminar en cada paso
        min_features_to_select : int, default=1
            Mínimo de características a seleccionar
        verbose : int, default=0
            Verbosidad para RFECV

        Returns:
        --------
        selector : BaseEstimator
            Selector de características (LassoGridSelector o RFECV)
        """

        if selector_type.lower() == "lasso":
            return LassoGridSelector(
                alphas=alphas,
                scorer=scorer,
                cv_splits=cv_splits,
                n_jobs=n_jobs,
                max_iter=max_iter,
                tol=tol,
                random_state=random_state,
                coef_threshold=coef_threshold,
            )

        elif selector_type.lower() == "rfecv":
            # Configurar estimador por defecto si no se proporciona
            if estimator is None:
                estimator = RandomForestRegressor(
                    n_estimators=50,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=random_state,
                    n_jobs=n_jobs,
                )

            # Crear TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=cv_splits)

            return RFECV(
                estimator=estimator,
                step=step,
                cv=tscv,
                scoring=scorer,
                n_jobs=n_jobs,
                min_features_to_select=min_features_to_select,
                verbose=verbose,
            )

        else:
            raise ValueError("selector_type must be 'lasso' or 'rfecv'")
