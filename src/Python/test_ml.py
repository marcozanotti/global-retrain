# Model testing

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
from utilsforecast.plotting import plot_series

from src.Python.utils import *

pd.set_option("display.max_rows", 4)
os.environ['NIXTLA_ID_AS_COL'] = '1'


# Parameters --------------------------------------------------------------

# the frequency of the data
freq = 'D'
# the minimum length of each series
min_series_length = 365 * 1
# the forecasting horizon
horizon = 28
# the length of the test window
test_window = horizon * 3 # 28 * 13 = last year
# the window for retraining (ex. 7 means retraining every 7 periods)
retrain_window = 7
# the confidence levels for prediction intervals
levels = [60, 70, 80, 85, 90, 95, 99] # [60, 70, 80, 85, 90, 95, 99]
# the type of conformal inference method
# NOTE: n_windows * h should be less than the count of data elements in your time series sequence.
# NOTE: n_windows should be at least 2 or more.
# NOTE: n_windows / retrain_window should be an integer.
# NOTE: method = 'conformal_distribution' or 'conformal_error'.
intervals = PredictionIntervals(h = horizon, n_windows = 4, method = 'conformal_distribution')

# check how many times the model will be retrained
get_retrain_ids(test_window, horizon, retrain_window)
len(get_retrain_ids(test_window, horizon, retrain_window))


# Load data ---------------------------------------------------------------

np.random.seed(1992)
# just a sample of 100 time series
data = get_data(
    file = 'data/m5/m5_daily_prep.parquet', 
    min_series_length = min_series_length,
    samples = 100
)


# Global ML Models --------------------------------------------------------

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.neural_network import MLPRegressor

# define target transformation
Log1p = FunctionTransformer(func = np.log1p, inverse_func = np.expm1)


# * Linear Regression -----------------------------------------------------

# define the models
# model_params = {
#     'verbose': -1,
#     'num_threads': 4,
#     'force_col_wise': True,
#     'num_leaves': 256,
#     'n_estimators': 50,
# }
# models = [LGBMRegressor(**model_params)]
models = [LinearRegression()]

# instantiate the MLForecast class
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
# engine.preprocess(train_df, static_features = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'])

# fit and predict with retraining
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
in_sample_df
out_sample_df
time_df
time_df.iloc[get_retrain_ids(test_window, horizon, retrain_window)]
time_df['total_sample_time'].sum() # total computing time in seconds


plot_series(
    out_sample_df.query('sample == 7').drop(columns = ['sample', 'y'], axis = 1),
    level = [90],
    max_ids = 4, 
    max_insample_length = horizon * 5, 
    engine = 'plotly'
).show()

# NOTE: the in_sample_df stores only fitting values for the re-training
# times. So plots with fitted values can be done only every retrain period.
plot_series(
    in_sample_df.query('sample == 7').drop(columns = ['sample'], axis = 1),
    out_sample_df.query('sample == 7').drop(columns = ['sample', 'y'], axis = 1),
    level = [90],
    max_ids = 4, 
    max_insample_length = horizon * 5, 
    engine = 'plotly'
).show()