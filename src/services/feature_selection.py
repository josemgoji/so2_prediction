"""
Feature selection service using skforecast approach.
Implements feature selection with RollingFeatures and RFECV selector.
"""

import pandas as pd

from skforecast.recursive import ForecasterRecursive
from skforecast.feature_selection import select_features

from ..recursos.selectors import FeatureSelectorFactory
from ..recursos.scorers import wmape_scorer

class SkforecastFeatureSelector:
    """
    Feature selector using skforecast approach with RollingFeatures and RFECV.
    """

    def __init__(
        self,
        lags: int = 48,
        window_features=None,  # Lista de window features
        regressor=None,  # Regressor pasado directamente
        selector_type: str = "rfecv",  # "lasso" o "rfecv"
        selector_params: dict = None,
        random_state: int = 15926,
    ):
        """
        Initialize the SkforecastFeatureSelector.

        Parameters:
        -----------
        lags : int, default=48
            Number of lags to use
        window_features : list, default=None
            Lista de window features (Fourier, STL, Rolling)
        regressor : estimator, default=None
            Regressor instance to use (debe ser pasado directamente)
        selector_type : str, default="rfecv"
            Type of selector: "lasso" o "rfecv"
        selector_params : dict, default=None
            Parameters for the selector
        random_state : int, default=15926
            Random seed for reproducibility
        """
        self.lags = lags
        self.window_features = window_features
        self.regressor = regressor
        self.selector_type = selector_type
        self.selector_params = selector_params or {}
        self.random_state = random_state

        # Set default selector parameters
        self.selector_params.setdefault("scorer", wmape_scorer)
        self.selector_params.setdefault("cv_splits", 3)
        self.selector_params.setdefault("random_state", random_state)

        # Create forecaster
        self.forecaster = ForecasterRecursive(
            regressor=self.regressor,
            lags=self.lags,
            window_features=self.window_features,
        )

        # Create selector usando tu FeatureSelectorFactory
        self.selector = FeatureSelectorFactory.create_selector(
            selector_type=self.selector_type,
            estimator=self.regressor,
            **self.selector_params,
        )

    def select_features(
        self,
        y: pd.Series,
        exog: pd.DataFrame = None,
        select_only: str = None,
        force_inclusion: list = None,
        subsample: float = 0.5,
        verbose: bool = True,
    ):
        """
        Perform feature selection using skforecast approach.

        Parameters:
        -----------
        y : pd.Series
            Target time series
        exog : pd.DataFrame, default=None
            Exogenous variables
        select_only : str, default=None
            Select only specific features: 'lags', 'window_features', 'exog'
        force_inclusion : list, default=None
            Features to force inclusion
        subsample : float, default=0.5
            Fraction of data to use for selection
        verbose : bool, default=True
            Whether to print progress

        Returns:
        --------
        tuple
            (selected_lags, selected_window_features, selected_exog)
        """

        selected_lags, selected_window_features, selected_exog = select_features(
            forecaster=self.forecaster,
            selector=self.selector,
            y=y,
            exog=exog,
            select_only=select_only,
            force_inclusion=force_inclusion,
            subsample=subsample,
            random_state=self.random_state,
            verbose=verbose,
        )

        return selected_lags, selected_window_features, selected_exog

    def get_feature_importance(self):
        """
        Get feature importance from the selector.

        Returns:
        --------
        dict
            Feature importance information
        """
        if hasattr(self.selector, "feature_importances_"):
            return {
                "feature_importances": self.selector.feature_importances_,
                "n_features": self.selector.n_features_,
                "support": self.selector.support_,
            }
        else:
            return {"message": "Feature importance not available for this selector"}
