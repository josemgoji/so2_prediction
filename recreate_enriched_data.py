#!/usr/bin/env python3
"""
Script para recrear los datos enriched con solo 48 lags en lugar de 72.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from statsmodels.tsa.seasonal import STL
import holidays

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.recursos.data_manager import DataManager

# Clases del notebook FI_V2.ipynb pero con 48 lags
class DateTimeFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, time_col='ds', country_holidays='CO', drop_time_col=False, tz=None):
        self.time_col = time_col
        self.country_holidays = country_holidays
        self.drop_time_col = drop_time_col
        self.tz = tz

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.time_col] = pd.to_datetime(X[self.time_col])
        if self.tz:
            X[self.time_col] = X[self.time_col].dt.tz_localize(self.tz, nonexistent='shift_forward', ambiguous='NaT').dt.tz_convert(self.tz)

        dt = X[self.time_col]
        X['year'] = dt.dt.year
        X['month'] = dt.dt.month
        X['day'] = dt.dt.day
        X['hour'] = dt.dt.hour
        X['dow'] = dt.dt.weekday      # 0=lunes
        X['weekofyear'] = dt.dt.isocalendar().week.astype(int)
        X['is_weekend'] = (X['dow'] >= 5).astype(int)
        X['is_month_start'] = dt.dt.is_month_start.astype(int)
        X['is_month_end'] = dt.dt.is_month_end.astype(int)
        
        try:
            hol = holidays.CountryHoliday(self.country_holidays)
            X['is_holiday'] = dt.dt.date.astype('datetime64').isin(hol).astype(int)
        except Exception:
            X['is_holiday'] = 0

        if self.drop_time_col:
            X = X.drop(columns=[self.time_col])
        return X

class LagFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, target_col='y', lags=range(1, 49)):  # Cambiado a 48 lags
        self.target_col = target_col
        self.lags = list(lags)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for L in self.lags:
            X[f'{self.target_col}_lag{L}'] = X[self.target_col].shift(L)
        return X

class RollingStats(BaseEstimator, TransformerMixin):
    def __init__(self, target_col='y', windows=(3, 6, 12, 24, 48, 72), stats=('mean','std','min','max')):
        self.target_col = target_col
        self.windows = windows
        self.stats = stats

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for w in self.windows:
            roll = X[self.target_col].shift(1).rolling(w)
            if 'mean' in self.stats: X[f'{self.target_col}_roll{w}_mean'] = roll.mean()
            if 'std'  in self.stats: X[f'{self.target_col}_roll{w}_std']  = roll.std(ddof=0)
            if 'min'  in self.stats: X[f'{self.target_col}_roll{w}_min']  = roll.min()
            if 'max'  in self.stats: X[f'{self.target_col}_roll{w}_max']  = roll.max()
        return X

class STLFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, target_col='y', period=24, robust=True, enabled=True):
        self.target_col = target_col
        self.period = period
        self.robust = robust
        self.enabled = enabled

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if not self.enabled:
            return X
        series = X[self.target_col].astype(float)
        series_filled = series.interpolate(limit_direction='both')
        res = STL(series_filled, period=self.period, robust=self.robust).fit()
        X[f'{self.target_col}_stl_trend'] = res.trend
        X[f'{self.target_col}_stl_season'] = res.seasonal
        X[f'{self.target_col}_stl_resid'] = res.resid
        return X

def build_features_from_df(df, time_col_name='datetime', target_col_name='target',
                           stl_period=24, use_stl=True, max_lags=48):
    """
    Construye características con un número limitado de lags.
    """
    if isinstance(df.index, pd.DatetimeIndex):
        df_ = df.copy()
        df_[time_col_name] = df_.index
    else:
        df_ = df.copy()

    pipe = Pipeline(steps=[
        ("dt",   DateTimeFeatures(time_col=time_col_name, country_holidays="CO", drop_time_col=False)),
        ("lags", LagFeatures(target_col=target_col_name, lags=range(1, max_lags + 1))),  # Solo 48 lags
        ("roll", RollingStats(target_col=target_col_name, windows=(3,6,12,24,48,72))),   # ajustable
        ("stl",  STLFeatures(target_col=target_col_name, period=stl_period, enabled=use_stl))
    ])

    feat = pipe.fit_transform(df_)
    
    feat = feat.dropna().reset_index(drop=True)
    return feat

def recreate_enriched_data(station="CEN-TRAF", max_lags=48):
    """
    Recrea los datos enriched con solo 48 lags.
    """
    print(f"Recreando datos enriched para {station} con {max_lags} lags...")
    
    # Cargar datos clean
    dm = DataManager()
    clean_path = f"data/stage/SO2/clean/{station}.csv"
    df_clean = dm.load_data(clean_path)
    
    print(f"Datos clean cargados: {df_clean.shape}")
    print(f"Columnas: {list(df_clean.columns)}")
    
    # Crear características
    df_enriched = build_features_from_df(
        df_clean, 
        time_col_name='datetime', 
        target_col_name='target', 
        stl_period=24, 
        use_stl=True,
        max_lags=max_lags
    )
    
    print(f"Datos enriched creados: {df_enriched.shape}")
    
    # Verificar lags creados
    lag_cols = [col for col in df_enriched.columns if 'target_lag' in col]
    print(f"Lags creados: {len(lag_cols)}")
    print(f"Primeros 5 lags: {lag_cols[:5]}")
    print(f"Últimos 5 lags: {lag_cols[-5:]}")
    
    # Guardar datos enriched
    enriched_path = f"data/stage/SO2/enrichment/enriched_{station}.csv"
    df_enriched.to_csv(enriched_path, index=False)
    print(f"Datos enriched guardados en: {enriched_path}")
    
    return df_enriched

if __name__ == "__main__":
    # Recrear datos para CEN-TRAF
    df_enriched = recreate_enriched_data("CEN-TRAF", max_lags=48)
    print("¡Datos enriched recreados exitosamente!")

