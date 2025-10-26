"""
Regressors configuration for feature selection and forecasting.
Contains pre-configured regressors for different use cases.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor


# ---- LGBM Regressors ----
def create_lgbm_regressor(
    n_estimators: int = 900,
    max_depth: int = 7,
    random_state: int = 15926,
    verbose: int = -1,
    **kwargs,
) -> LGBMRegressor:
    """
    Creates a configured LGBMRegressor for forecasting.

    Parameters:
    -----------
    n_estimators : int, default=900
        Number of boosting rounds
    max_depth : int, default=7
        Maximum tree depth
    random_state : int, default=15926
        Random seed for reproducibility
    verbose : int, default=-1
        Verbosity level
    **kwargs
        Additional parameters for LGBMRegressor

    Returns:
    --------
    LGBMRegressor
        Configured LGBM regressor
    """
    return LGBMRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        verbose=verbose,
        **kwargs,
    )


def create_lgbm_regressor_fast(
    n_estimators: int = 100,
    max_depth: int = 5,
    random_state: int = 15926,
    verbose: int = -1,
    **kwargs,
) -> LGBMRegressor:
    """
    Creates a fast LGBMRegressor for feature selection.

    Parameters:
    -----------
    n_estimators : int, default=100
        Number of boosting rounds (reduced for speed)
    max_depth : int, default=5
        Maximum tree depth (reduced for speed)
    random_state : int, default=15926
        Random seed for reproducibility
    verbose : int, default=-1
        Verbosity level
    **kwargs
        Additional parameters for LGBMRegressor

    Returns:
    --------
    LGBMRegressor
        Configured fast LGBM regressor
    """
    return LGBMRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        verbose=verbose,
        **kwargs,
    )


# ---- Random Forest Regressors ----
def create_rf_regressor(
    n_estimators: int = 100,
    max_depth: int = 10,
    min_samples_split: int = 5,
    min_samples_leaf: int = 2,
    random_state: int = 15926,
    n_jobs: int = -1,
    **kwargs,
) -> RandomForestRegressor:
    """
    Creates a configured RandomForestRegressor.

    Parameters:
    -----------
    n_estimators : int, default=100
        Number of trees in the forest
    max_depth : int, default=10
        Maximum depth of the tree
    min_samples_split : int, default=5
        Minimum number of samples required to split an internal node
    min_samples_leaf : int, default=2
        Minimum number of samples required to be at a leaf node
    random_state : int, default=15926
        Random seed for reproducibility
    n_jobs : int, default=-1
        Number of jobs to run in parallel
    **kwargs
        Additional parameters for RandomForestRegressor

    Returns:
    --------
    RandomForestRegressor
        Configured Random Forest regressor
    """
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=n_jobs,
        **kwargs,
    )


def create_rf_regressor_fast(
    n_estimators: int = 50,
    max_depth: int = 8,
    min_samples_split: int = 10,
    min_samples_leaf: int = 4,
    random_state: int = 15926,
    n_jobs: int = -1,
    **kwargs,
) -> RandomForestRegressor:
    """
    Creates a fast RandomForestRegressor for feature selection.

    Parameters:
    -----------
    n_estimators : int, default=50
        Number of trees in the forest (reduced for speed)
    max_depth : int, default=8
        Maximum depth of the tree (reduced for speed)
    min_samples_split : int, default=10
        Minimum number of samples required to split an internal node
    min_samples_leaf : int, default=4
        Minimum number of samples required to be at a leaf node
    random_state : int, default=15926
        Random seed for reproducibility
    n_jobs : int, default=-1
        Number of jobs to run in parallel
    **kwargs
        Additional parameters for RandomForestRegressor

    Returns:
    --------
    RandomForestRegressor
        Configured fast Random Forest regressor
    """
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=n_jobs,
        **kwargs,
    )


# ---- Linear Regressors ----
def create_lasso_regressor(
    alpha: float = 1.0,
    max_iter: int = 5000,
    tol: float = 1e-3,
    random_state: int = 15926,
    **kwargs,
) -> Lasso:
    """
    Creates a configured Lasso regressor.

    Parameters:
    -----------
    alpha : float, default=1.0
        Regularization strength
    max_iter : int, default=5000
        Maximum number of iterations
    tol : float, default=1e-3
        Tolerance for stopping criterion
    random_state : int, default=15926
        Random seed for reproducibility
    **kwargs
        Additional parameters for Lasso

    Returns:
    --------
    Lasso
        Configured Lasso regressor
    """
    return Lasso(
        alpha=alpha, max_iter=max_iter, tol=tol, random_state=random_state, **kwargs
    )


def create_ridge_regressor(
    alpha: float = 1.0,
    max_iter: int = 5000,
    tol: float = 1e-3,
    random_state: int = 15926,
    **kwargs,
) -> Ridge:
    """
    Creates a configured Ridge regressor.

    Parameters:
    -----------
    alpha : float, default=1.0
        Regularization strength
    max_iter : int, default=5000
        Maximum number of iterations
    tol : float, default=1e-3
        Tolerance for stopping criterion
    random_state : int, default=15926
        Random seed for reproducibility
    **kwargs
        Additional parameters for Ridge

    Returns:
    --------
    Ridge
        Configured Ridge regressor
    """
    return Ridge(
        alpha=alpha, max_iter=max_iter, tol=tol, random_state=random_state, **kwargs
    )


def create_elastic_net_regressor(
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    max_iter: int = 5000,
    tol: float = 1e-3,
    random_state: int = 15926,
    **kwargs,
) -> ElasticNet:
    """
    Creates a configured ElasticNet regressor.

    Parameters:
    -----------
    alpha : float, default=1.0
        Regularization strength
    l1_ratio : float, default=0.5
        Mixing parameter (0=Ridge, 1=Lasso)
    max_iter : int, default=5000
        Maximum number of iterations
    tol : float, default=1e-3
        Tolerance for stopping criterion
    random_state : int, default=15926
        Random seed for reproducibility
    **kwargs
        Additional parameters for ElasticNet

    Returns:
    --------
    ElasticNet
        Configured ElasticNet regressor
    """
    return ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
        **kwargs,
    )


# ---- XGBoost Regressor ----
def create_xgb_regressor(
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    random_state: int = 15926,
    n_jobs: int = -1,
    **kwargs,
) -> XGBRegressor:
    """
    Creates a configured XGBRegressor.

    Parameters:
    -----------
    n_estimators : int, default=100
        Number of boosting rounds
    max_depth : int, default=6
        Maximum tree depth
    learning_rate : float, default=0.1
        Boosting learning rate
    random_state : int, default=15926
        Random seed for reproducibility
    n_jobs : int, default=-1
        Number of parallel threads
    **kwargs
        Additional parameters for XGBRegressor

    Returns:
    --------
    XGBRegressor
        Configured XGBoost regressor
    """
    return XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        n_jobs=n_jobs,
        **kwargs,
    )


# ---- Factory function ----
def create_regressor(regressor_type: str = "lgbm", **kwargs):
    """
    Factory function to create regressors by type.

    Parameters:
    -----------
    regressor_type : str, default="lgbm"
        Type of regressor: "lgbm", "lgbm_fast", "rf", "rf_fast",
                          "lasso", "ridge", "elastic_net", "xgb"
    **kwargs
        Additional parameters for the regressor

    Returns:
    --------
    Regressor
        Configured regressor instance
    """
    regressor_type = regressor_type.lower()

    if regressor_type == "lgbm":
        return create_lgbm_regressor(**kwargs)
    elif regressor_type == "lgbm_fast":
        return create_lgbm_regressor_fast(**kwargs)
    elif regressor_type == "rf":
        return create_rf_regressor(**kwargs)
    elif regressor_type == "rf_fast":
        return create_rf_regressor_fast(**kwargs)
    elif regressor_type == "lasso":
        return create_lasso_regressor(**kwargs)
    elif regressor_type == "ridge":
        return create_ridge_regressor(**kwargs)
    elif regressor_type == "elastic_net":
        return create_elastic_net_regressor(**kwargs)
    elif regressor_type == "xgb":
        return create_xgb_regressor(**kwargs)
    else:
        raise ValueError(
            f"Unknown regressor_type: {regressor_type}. "
            f"Available types: lgbm, lgbm_fast, rf, rf_fast, lasso, ridge, elastic_net, xgb"
        )


# ---- Default configurations ----
DEFAULT_REGRESSORS = {
    "lgbm_forecasting": create_lgbm_regressor,
    "lgbm_fast": create_lgbm_regressor_fast,
    "rf_forecasting": create_rf_regressor,
    "rf_fast": create_rf_regressor_fast,
    "lasso": create_lasso_regressor,
    "ridge": create_ridge_regressor,
    "elastic_net": create_elastic_net_regressor,
    "xgb": create_xgb_regressor,
}
