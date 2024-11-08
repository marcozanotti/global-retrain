# Model testing

import os
import time
import pandas as pd
import numpy as np
from pytimetk import glimpse

from statsforecast import StatsForecast
from statsforecast.utils import ConformalIntervals
from mlforecast import MLForecast
from mlforecast.lag_transforms import (
    RollingMean, ExpandingMean
)
from utilsforecast.plotting import plot_series

from src.Python.utils import *

pd.set_option("display.max_rows", 4)
os.environ['NIXTLA_ID_AS_COL'] = '1'


# Import data -------------------------------------------------------------

# download_dataset('m5')
np.random.seed(1992)
train_df, test_df = get_dataset('m5', samples = 8) # just a sample of 8 time series


# Parameters --------------------------------------------------------------

# define the frequency of the data
freq = 'D'
# define the forecasting horizon
horizon = 28
# define the length of the test window
test_window = horizon * 3 # 28 * 13 = last year
# define the window for retraining
retrain_window = 7
# define the confidence levels
levels = [90] # [60, 70, 80, 85, 90, 95, 99]


# Prepare data ------------------------------------------------------------
# combine train and test dataframes
data = combine_train_test(train_df, test_df)
train_df, test_df = split_train_test(data, test_window)
plot_series(data).show()
plot_series(train_df).show()
plot_series(test_df).show()


# Local Models ------------------------------------------------------------

from statsforecast.models import (
    SeasonalNaive,
    AutoETS
)

# define the models
models = [
    AutoETS(season_length = 7)
]
fallback_model = SeasonalNaive(season_length = 7)

# fit and predict with retraining
preds_df, actual_df, time_df, tot_time = retrain_ets_model(
    data = data,
    models = models,
    fallback_model = fallback_model,  
    freq = freq, 
    levels = levels,
    test_window = test_window,
    horizon = horizon,
    retrain_window = retrain_window
)
preds_df.glimpse()
actual_df.glimpse()
time_df
tot_time


plot_series(
    actual_df.query('sample == 28').drop(columns = ['sample'], axis = 1),
    preds_df.query('sample == 28').drop(columns = ['sample'], axis = 1),
    level = [90],
    max_ids = 4, 
    max_insample_length = horizon * 5, 
    engine = 'plotly'
).show()




# Global Models -----------------------------------------------------------

from statsforecast.models import SeasonalNaive
from sklearn.linear_model import LinearRegression

# define the models
models = [
    LinearRegression()
]
fallback_model = SeasonalNaive(season_length = 7)

# fit and predict with retraining
# fit and predict with retraining
in_sample_df, out_sample_df, time_df, tot_time = retrain_ml_model(
    data = data,
    models = models,
    fallback_model = fallback_model,  
    freq = freq, 
    levels = levels,
    test_window = test_window,
    horizon = horizon,
    retrain_window = retrain_window
)
in_sample_df
out_sample_df
time_df
tot_time

plot_series(
    actual_df.query('sample == 28').drop(columns = ['sample'], axis = 1),
    preds_df.query('sample == 28').drop(columns = ['sample'], axis = 1),
    level = [90],
    max_ids = 4, 
    max_insample_length = horizon * 5, 
    engine = 'plotly'
).show()