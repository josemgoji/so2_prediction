import pandas as pd

def filter_from_first_valid_date(df: pd.DataFrame, columns_required: list) -> pd.DataFrame:
    first_valid_date = df[columns_required].dropna().index.min()
    return df[df.index >= first_valid_date]