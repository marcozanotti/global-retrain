
import os
from time import gmtime, strftime
import pandas as pd
from mlforecast.utils import PredictionIntervals
from src.Python.utils import *

pd.set_option("display.max_rows", 4)
os.environ['NIXTLA_ID_AS_COL'] = '1'


# Parameters --------------------------------------------------------------

dataset_name = 'm5'
frequency = 'daily'
horizon = 28
test_window = horizon * 2 # test_window  = 28 * 13
retrain_scenarios = [30] # retrain_scenarios = [7, 14, 21, 30, 60, 90, 120, 150, 180, 364]

model_names = ['LinearRegression', 'Lasso'] 
# model_names = [
#     'LinearRegression', 'Lasso', 'Ridge', 
#     'RandomForestRegressor', 
#     'XGBRegressor', 'LGBMRegressor', 'CatBoostRegressor'
# ]
model_params = None
model_params = {
    'LinearRegression': {
        'n_jobs': -1
    },
    'Lasso': {
        'alpha': 0.5
    }
}

intervals = PredictionIntervals(h = horizon, n_windows = 4, method = 'conformal_error')
static_features = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
min_series_length = 365 * 2
samples = 100

for scenario in retrain_scenarios:
    print(get_retrain_ids(test_window, horizon, scenario))

# logging configuration
log_file = create_file_name(
    name_list = [dataset_name, frequency, strftime("%Y%m%d_%H%M%S", gmtime())], 
    ext = '.log'
)
configure_logging(log_file)
logger = create_logger()


# Fitting -----------------------------------------------------------------

retrain_model(
    dataset_name = dataset_name,
    frequency = frequency,
    test_window = test_window,
    horizon = horizon,
    retrain_scenarios = retrain_scenarios,
    model_names = model_names,
    model_params = model_params,
    intervals = intervals,
    static_features = static_features,
    min_series_length = min_series_length,
    samples = samples
)

# stop logging
stop_logger(logger)


# Combine results ---------------------------------------------------------

model_name = 'LinearRegression'
retrain_window = 14
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