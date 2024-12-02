# Model testing

import os
import numpy as np
import pandas as pd

from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean, ExpandingMean
from mlforecast.utils import PredictionIntervals
# from sklearn.preprocessing import FunctionTransformer
# from mlforecast.target_transforms import GlobalSklearnTransformer
# from mlforecast.target_transforms import LocalStandardScaler, LocalMinMaxScaler, Differences
from utilsforecast.plotting import plot_series

from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor
# from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor
# from catboost import CatBoostRegressor

from src.Python.utils import *

pd.set_option("display.max_rows", 4)
os.environ['NIXTLA_ID_AS_COL'] = '1'


# Parameters --------------------------------------------------------------

# parameters for file management
dataset_name = 'm5'
frequency = 'daily'
model_name = 'LinearRegression'
# model_name = 'XGBRegressor'
# model_name = 'RandomForestRegressor'
# model_name = 'LGBMRegressor'
# model_name = 'CatBoostRegressor'

# the frequency of the data
freq = 'D'
# the minimum length of each series
min_series_length = 365 * 2
# the forecasting horizon
horizon = 28
# the length of the test window
test_window = horizon * 2 # 28 * 13 = last year
# the window for retraining (ex. 7 means retraining every 7 periods)
retrain_window = 7
# the confidence levels for prediction intervals
levels = [50, 60, 70, 80, 90, 95, 99]
# the type of conformal inference method
# NOTE: 
# - n_windows * h should be less than the count of data elements in your time series sequence.
# - n_windows should be at least 2 or more
# - n_windows / retrain_window should be an integer
# - method = 'conformal_distribution' or 'conformal_error'
intervals = PredictionIntervals(h = horizon, n_windows = 4, method = 'conformal_error')
# define ad hoc target transformations
# Log1p = FunctionTransformer(func = np.log1p, inverse_func = np.expm1)

# check how many times the model will be retrained
get_retrain_ids(test_window, horizon, retrain_window)
len(get_retrain_ids(test_window, horizon, retrain_window))


# Global ML Models --------------------------------------------------------

# * Linear Regression -----------------------------------------------------

# define the models
models = [LinearRegression(n_jobs = -1)]
# models = [XGBRegressor(n_jobs = -1)]
# models = [RandomForestRegressor(n_jobs = -1)]
# models = [LGBMRegressor(n_jobs = -1)]
# models = [CatBoostRegressor()]

# instantiate the MLForecast class
engine = MLForecast(
    models = models,
    freq = freq, 
    num_threads = 1,
    # target_transforms = [GlobalSklearnTransformer(Log1p)],
    lags = [1] + [7 * (i+1) for i in range(8)],
    lag_transforms = {
        1: [RollingMean(7), RollingMean(14), RollingMean(30), ExpandingMean()],
        7: [RollingMean(7), RollingMean(14), RollingMean(30)],
        14: [RollingMean(7), RollingMean(14), RollingMean(30)],
        30: [RollingMean(7), RollingMean(14), RollingMean(30)]
    },
    date_features = [
        'year', 'quarter', 'month', 'week', 
        'dayofweek', 'day', is_weekend
    ]
)
# engine.preprocess(
#     train_df, 
#     static_features = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
# )

# fit and predict with retraining
# np.random.seed(1992)
retrain_ml_model(
    dataset_name = dataset_name,
    frequency = frequency,
    engine = engine,
    test_window = test_window,
    horizon = horizon,
    retrain_window = retrain_window,
    levels = levels,
    intervals = intervals,
    static_features = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
    min_series_length = min_series_length,
    samples = 100,
    store_in_sample_results = False,
    combine_results = False,
    ext = '.parquet'
)


# Combine results ---------------------------------------------------------

combine_and_save_files(
    path_to_read = f'results/{dataset_name}/{model_name}/{retrain_window}/preds/tmp/',
    path_to_write = f'results/{dataset_name}/{model_name}/{retrain_window}/preds/',
    name_list = [
        dataset_name, frequency, 'outsample', model_name, str(retrain_window)
    ],
    ext = '.parquet'
)

# Load data ---------------------------------------------------------------

out_sample_df = load_data(
    path = f'results/{dataset_name}/{model_name}/{retrain_window}/preds/', 
    name_list = [dataset_name, frequency, 'outsample', model_name, retrain_window],
    ext = '.parquet'
)
out_sample_df

time_df = load_data(
    path = f'results/{dataset_name}/{model_name}/{retrain_window}/time/', 
    name_list = [dataset_name, frequency, 'time', model_name, retrain_window],
    ext = '.parquet'
)
time_df
