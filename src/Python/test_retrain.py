
import os
import pandas as pd
from mlforecast.utils import PredictionIntervals
from src.Python.utils import *

pd.set_option("display.max_rows", 4)
os.environ['NIXTLA_ID_AS_COL'] = '1'


# Parameters --------------------------------------------------------------

dataset_name = 'm5'
frequency = 'daily'
model_names = ['LinearRegression', 'Lasso'] 
# [
# 'LinearRegression', 'Lasso', 'Ridge', 
# 'RandomForestRegressor', 
# 'XGBRegressor', 'LGBMRegressor', 'CatBoostRegressor'
# ]
retrain_scenarios = [7, 14, 30] # retrain_scenarios = [7, 14, 21, 30, 60, 90, 120, 150, 180, 364]
horizon = 28
test_window = horizon * 2 # test_window  = 28 * 13
intervals = PredictionIntervals(h = horizon, n_windows = 4, method = 'conformal_error')
static_features = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
min_series_length = 365 * 2

for scenario in retrain_scenarios:
    print(get_retrain_ids(test_window, horizon, scenario))


# Fitting -----------------------------------------------------------------

retrain_model(
    model_names = model_names,
    retrain_scenarios = retrain_scenarios,
    dataset_name = dataset_name,
    frequency = frequency,
    test_window = test_window,
    horizon = horizon,
    intervals = intervals,
    static_features = static_features,
    min_series_length = min_series_length,
    samples = 100
)