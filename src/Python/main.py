# main.py

import os
import numpy as np
import pandas as pd

from mlforecast import MLForecast
from mlforecast.lag_transforms import (
    RollingMean, ExpandingMean
)
from mlforecast.utils import PredictionIntervals
from sklearn.preprocessing import FunctionTransformer
from mlforecast.target_transforms import GlobalSklearnTransformer
# from mlforecast.target_transforms import LocalStandardScaler, LocalMinMaxScaler, Differences

from sklearn.linear_model import LinearRegression

from src.Python.utils import *

pd.set_option("display.max_rows", 4)
os.environ['NIXTLA_ID_AS_COL'] = '1'

# Parameters --------------------------------------------------------------

dataset_name = 'm5'
frequency = 'daily'
model_name = 'LinearRegression'
freq = 'D'
min_series_length = 365 * 1
horizon = 28
test_window = horizon * 2 
retrain_window = 7
levels = [60, 70, 80, 85, 90, 95, 99]
intervals = PredictionIntervals(h = horizon, n_windows = 4, method = 'conformal_distribution')
Log1p = FunctionTransformer(func = np.log1p, inverse_func = np.expm1)

# Load data ---------------------------------------------------------------

data = get_data(
    path = 'data/m5/',
    name_list = [dataset_name, frequency, 'prep'],
    ext = '.parquet',
    min_series_length = min_series_length
)

# Global ML Models --------------------------------------------------------

# * Linear Regression -----------------------------------------------------

models = [LinearRegression()]
engine = MLForecast(
    models = models,
    freq = freq, 
    num_threads = 1,
    target_transforms = [GlobalSklearnTransformer(Log1p)],
    lags = [1] + [7 * (i+1) for i in range(8)],
    lag_transforms = {
        1: [ExpandingMean()],
        7: [RollingMean(7), RollingMean(14), RollingMean(28)],
        14: [RollingMean(7), RollingMean(14), RollingMean(28)],
        28: [RollingMean(7), RollingMean(14), RollingMean(28)],
    },
    date_features = ['year', 'quarter', 'month', 'week', 'dayofweek', 'day']
)
in_sample_df, out_sample_df, time_df = retrain_ml_model(
    data = data,
    engine = engine,
    test_window = test_window,
    horizon = horizon,
    retrain_window = retrain_window,
    levels = levels,
    intervals = intervals,
    static_features = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
    store_in_sample_results = False
)


