import pandas as pd
from typing import Tuple
from .feature_engineering import FeatureEngineeringService


class DataPreparationService:
    """
    Service for preparing selected features data: loading, splitting into train/validation/test sets.
    Test and validation sets are each 2 months long, taken from the end of the data.
    Can optionally use exogenous data for enrichment.
    """

    def __init__(self, use_exogenous: bool = True):
        """
        Initialize the DataPreparationService.

        Parameters:
        - use_exogenous: Whether to use exogenous data for enrichment
        Note: Feature engineering is ALWAYS enabled for better model performance
        """
        self.use_exogenous = use_exogenous
        self.use_feature_engineering = True  # Always enabled
        self.feature_engineering_service = FeatureEngineeringService()

    def load_and_split(
        self, selected_features_path: str, use_exogenous: bool = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Loads the CSV and splits into train, validation, and test sets.
        - Test: last 2 months
        - Validation: previous 2 months before test
        - Train: everything before validation

        NO filtra columnas aquí - cada modelo maneja su propio filtrado.
        Solo hace la división temporal.

        Parameters:
        - selected_features_path: Path to the CSV file
        - use_exogenous: Ignorado - solo para compatibilidad

        Returns:
        - Tuple of (train_df, val_df, test_df) con TODAS las columnas
        """
        # Load the data
        df = pd.read_csv(selected_features_path)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)

        # Verificar que tiene las columnas básicas
        if "target" not in df.columns:
            raise KeyError("Target column 'target' not found in the dataset")

        # Determine the end date
        end_date = df["datetime"].max()

        # Test: last 2 months
        test_start = end_date - pd.DateOffset(months=2)
        test_df = df[df["datetime"] >= test_start].copy()

        # Validation: previous 2 months before test
        val_end = test_start
        val_start = val_end - pd.DateOffset(months=2)
        val_df = df[(df["datetime"] >= val_start) & (df["datetime"] < val_end)].copy()

        # Train: everything before validation
        train_df = df[df["datetime"] < val_start].copy()

        print(
            f"Train set: {len(train_df)} samples from {train_df['datetime'].min()} to {train_df['datetime'].max()}"
        )
        print(
            f"Validation set: {len(val_df)} samples from {val_df['datetime'].min()} to {val_df['datetime'].max()}"
        )
        print(
            f"Test set: {len(test_df)} samples from {test_df['datetime'].min()} to {test_df['datetime'].max()}"
        )
        print(f"Total columns available: {len(df.columns)}")

        return train_df, val_df, test_df
