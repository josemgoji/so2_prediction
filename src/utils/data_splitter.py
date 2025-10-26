import pandas as pd
from typing import Tuple, Optional


def split_data_by_dates(
    df: pd.DataFrame,
    target_col: str,
    exog_cols: list,
    val_start_date: Optional[str] = None,
    test_start_date: Optional[str] = None,
    val_months: int = 2,
    test_months: int = 2,
) -> Tuple[
    pd.Series,
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
]:
    """
    Split time series data into train, validation, and test sets based on dates.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with datetime index containing the time series data
    target_col : str
        Name of the target column
    exog_cols : list
        List of exogenous feature column names
    val_start_date : str, optional
        Start date for validation set (format: 'YYYY-MM-DD'). If None, calculated automatically
    test_start_date : str, optional
        Start date for test set (format: 'YYYY-MM-DD'). If None, calculated automatically
    val_months : int, default=2
        Number of months for validation set
    test_months : int, default=2
        Number of months for test set

    Returns:
    --------
    Tuple containing:
    - y_train, exog_train: Training data
    - y_val, exog_val: Validation data
    - y_test, exog_test: Test data
    - y_trainval, exog_trainval: Combined training+validation data for grid search
    """

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")

    # Sort by index to ensure chronological order
    df = df.sort_index()

    # Extract target and exogenous variables
    y = df[target_col]
    exog = df[exog_cols]

    # Calculate dates if not provided
    if val_start_date is None:
        # Calculate validation start date: test_months before the end
        test_end = df.index[-1]
        test_start = test_end - pd.DateOffset(months=test_months)
        val_start = test_start - pd.DateOffset(months=val_months)
    else:
        val_start = pd.to_datetime(val_start_date)
        test_start = val_start + pd.DateOffset(months=val_months)

    if test_start_date is not None:
        test_start = pd.to_datetime(test_start_date)

    # Define end dates
    val_end = test_start - pd.Timedelta(hours=1)
    test_end = test_start + pd.DateOffset(months=test_months) - pd.Timedelta(hours=1)

    # Ensure dates are within data range
    val_start = max(val_start, df.index[0])
    val_end = min(val_end, df.index[-1])
    test_start = max(test_start, df.index[0])
    test_end = min(test_end, df.index[-1])

    # Split the data
    y_train = y.loc[: val_start - pd.Timedelta(hours=1)]
    exog_train = exog.loc[: val_start - pd.Timedelta(hours=1)]

    y_val = y.loc[val_start:val_end]
    exog_val = exog.loc[val_start:val_end]

    y_test = y.loc[test_start:test_end]
    exog_test = exog.loc[test_start:test_end]

    # Combined train+val for grid search
    y_trainval = y.loc[:val_end]
    exog_trainval = exog.loc[:val_end]

    # Print split information
    print(f"Data split summary:")
    print(f"  Total data range: {df.index[0]} to {df.index[-1]}")
    print(
        f"  Training: {y_train.index[0]} to {y_train.index[-1]} ({len(y_train)} samples)"
    )
    print(f"  Validation: {y_val.index[0]} to {y_val.index[-1]} ({len(y_val)} samples)")
    print(f"  Test: {y_test.index[0]} to {y_test.index[-1]} ({len(y_test)} samples)")

    return (
        y_train,
        exog_train,
        y_val,
        exog_val,
        y_test,
        exog_test,
        y_trainval,
        exog_trainval,
    )


def split_data_by_percentage(
    df: pd.DataFrame,
    target_col: str,
    exog_cols: list,
    train_pct: float = 0.70,
    val_pct: float = 0.15,
) -> Tuple[
    pd.Series,
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
]:
    """
    Split time series data into train, validation, and test sets based on percentages.
    This is the original method from models.py for backward compatibility.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with datetime index containing the time series data
    target_col : str
        Name of the target column
    exog_cols : list
        List of exogenous feature column names
    train_pct : float, default=0.70
        Percentage of data for training (70%)
    val_pct : float, default=0.15
        Percentage of data for validation (15%)
        Test will be the remaining 15%

    Returns:
    --------
    Tuple containing:
    - y_train, exog_train: Training data
    - y_val, exog_val: Validation data
    - y_test, exog_test: Test data
    - y_trainval, exog_trainval: Combined training+validation data for grid search
    """

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")

    # Sort by index to ensure chronological order
    df = df.sort_index()

    # Calculate split indices
    n = len(df)
    n_train = int(n * train_pct)
    n_val = int(n * val_pct)

    end_train = df.index[n_train - 1]
    end_val = df.index[n_train + n_val - 1]

    # Extract target and exogenous variables
    y = df[target_col]
    exog = df[exog_cols]

    # Split the data
    y_train = y.loc[:end_train]
    exog_train = exog.loc[:end_train]

    y_val = y.loc[end_train + pd.Timedelta(hours=1) : end_val]
    exog_val = exog.loc[end_train + pd.Timedelta(hours=1) : end_val]

    y_test = y.loc[end_val + pd.Timedelta(hours=1) :]
    exog_test = exog.loc[end_val + pd.Timedelta(hours=1) :]

    # Combined train+val for grid search
    y_trainval = y.loc[:end_val]
    exog_trainval = exog.loc[:end_val]

    # Print split information
    print(f"Data split summary (percentage-based):")
    print(f"  Total data range: {df.index[0]} to {df.index[-1]}")
    print(
        f"  Training: {y_train.index[0]} to {y_train.index[-1]} ({len(y_train)} samples)"
    )
    print(f"  Validation: {y_val.index[0]} to {y_val.index[-1]} ({len(y_val)} samples)")
    print(f"  Test: {y_test.index[0]} to {y_test.index[-1]} ({len(y_test)} samples)")

    return (
        y_train,
        exog_train,
        y_val,
        exog_val,
        y_test,
        exog_test,
        y_trainval,
        exog_trainval,
    )
