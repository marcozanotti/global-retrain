
import os
import pandas as pd
from src.Python.utils import *

pd.set_option("display.max_rows", 4)
os.environ['NIXTLA_ID_AS_COL'] = '1'

cfg = get_config('config/retrain_config.yaml')

configure_logging('config/log_config.yaml', [cfg['dataset_name'], cfg['frequency']])
logger = create_logger()


# Fitting -----------------------------------------------------------------

retrain_model(
    dataset_name = cfg['dataset_name'],
    frequency = cfg['frequency'],
    test_window = cfg['test_window'],
    horizon = cfg['horizon'],
    retrain_scenarios = cfg['retrain_scenarios'],
    model_names = cfg['model_names'],
    model_params = cfg['model_params'],
    levels = cfg['levels'],
    static_features = cfg['static_features'],
    min_series_length = cfg['min_series_length'],
    samples = cfg['samples'],
    store_in_sample_results = cfg['store_in_sample_results'],
    combine_results = cfg['combine_results'],
    ext = cfg['ext']
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