# Model testing

import os
import time
import pandas as pd
import numpy as np
from pytimetk import glimpse

from mlforecast import MLForecast
from mlforecast.lag_transforms import (
    RollingMean, ExpandingMean
)
from mlforecast.utils import PredictionIntervals
from utilsforecast.plotting import plot_series

from src.Python.utils import *

pd.set_option("display.max_rows", 4)
os.environ['NIXTLA_ID_AS_COL'] = '1'


# Import data -------------------------------------------------------------

# download_dataset('m5')
np.random.seed(1992)
m5_train_df, m5_test_df = get_dataset('m5', samples = 8) # just a sample of 8 time series


# Parameters --------------------------------------------------------------

# define the frequency of the data
freq = 'D'
# define the forecasting horizon
horizon = 28
# define the length of the test window
test_window = horizon * 2 # 28 * 13 = last year
# define the window for retraining
retrain_window = 7
# define the confidence levels
levels = [90] # [60, 70, 80, 85, 90, 95, 99]
# define the conformal inference method
intervals = PredictionIntervals(h = horizon, n_windows = 4, method = 'conformal_distribution')
# NOTE: n_windows * h should be less than the count of data elements in your time series sequence.
# NOTE: Also value of n_windows should be at least 2 or more.
# NOTE: n_windows / retrain_window should be an integer.
# NOTE: method = 'conformal_distribution' or 'conformal_error'.

get_retrain_ids(test_window, horizon, retrain_window)


# Prepare data ------------------------------------------------------------
# combine train and test dataframes
data = combine_train_test(m5_train_df, m5_test_df)
data = get_static_features(data, 'm5')
plot_series(data).show()

train_df, test_df = split_train_test(data, test_window)
plot_series(train_df).show()
plot_series(test_df).show()


# Global ML Models --------------------------------------------------------

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.neural_network import MLPRegressor


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
    # target_transforms = [Log1p, LocalStandardScaler()],
    lags = [1] + [7 * (i+1) for i in range(8)],
    lag_transforms = {
        1: [ExpandingMean()],
        7: [RollingMean(7), RollingMean(14), RollingMean(28)],
        14: [RollingMean(7), RollingMean(14), RollingMean(28)],
        28: [RollingMean(7), RollingMean(14), RollingMean(28)],
    },
    date_features = ['year', 'quarter', 'month', 'week', 'dayofweek', 'day']
)
# engine.preprocess(train_df, static_features = [])
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
    static_features = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
)
in_sample_df
out_sample_df
time_df
time_df.iloc[get_retrain_ids(test_window, horizon, retrain_window)]
time_df['total_sample_time'].sum() # total computing time in seconds

plot_series(
    in_sample_df.query('sample == 7').drop(columns = ['sample'], axis = 1),
    out_sample_df.query('sample == 7').drop(columns = ['sample', 'y'], axis = 1),
    level = [90],
    max_ids = 4, 
    max_insample_length = horizon * 5, 
    engine = 'plotly'
).show()
# NOTE: the in_sample_df stores only fitting values for the re-training
# times. So plots with fitted values can be done only every retrain period.
