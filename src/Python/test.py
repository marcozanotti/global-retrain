# Model testing

import os
import pandas as pd
import numpy as np
from pytimetk import glimpse

from statsforecast import StatsForecast
from statsforecast.utils import ConformalIntervals
from utilsforecast.plotting import plot_series

from src.Python.utils import *

pd.set_option("display.max_rows", 4)
os.environ['NIXTLA_ID_AS_COL'] = '1'


# Import data -------------------------------------------------------------

# download_dataset('m5')
np.random.seed(1992)
train_df, test_df = get_dataset('m5', samples = 10) # just a sample of 10 time series
train_df.glimpse()
test_df.glimpse()


# Plotting data -----------------------------------------------------------
# plot_series(train_df).show()


# Parameters --------------------------------------------------------------

# define the frequency of the data
freq = 'D'
# define the forecasting horizon
horizon = 28
# define the confidence levels
levels = [60, 70, 80, 85, 90, 95, 99]


# Local Models ------------------------------------------------------------

from statsforecast.models import (
    SeasonalNaive,
    AutoETS
)

# define the models
models = [
    SeasonalNaive(season_length = 7),
    AutoETS(season_length = 7)
]
fallback_model = SeasonalNaive(season_length = 7)

# fit and predict with retraining
preds_df = retrain_model(
    models = models,
    fallback_model = fallback_model,  
    train_df = train_df, 
    test_df = test_df, 
    freq = freq, 
    levels = levels, 
    horizon = horizon,
    retrain_window = 1
)


plot_series(
    train_df, preds_df,
    level = [80, 90],
    max_ids = 4, 
    max_insample_length = horizon * 5, 
    engine = 'plotly'
).show()




# Global Models -----------------------------------------------------------

