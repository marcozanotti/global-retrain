
import os
from src.Python.utils.utilities import *
from src.Python.utils.evaluate_forecasts import evaluate_model

os.environ['NIXTLA_ID_AS_COL'] = '1'
cfg = get_config('config/eval_config.yaml')
configure_logging(
    config_file = 'config/log_config.yaml', 
    name_list = [cfg['dataset_name'], cfg['frequency'], 'eval']
)
logger = create_logger()

evaluate_model(config = cfg)

stop_logger(logger)



# Evaluation --------------------------------------------------------------

import numpy as np
from src.Python.utils.collect_data import load_data
from src.Python.utils.evaluate_forecasts import aggregate_data

# eval_df.shape[0] = n_series * n_retrain_scenarios = 30.000 * 10
eval_df = load_data(
    path_list = ['results', cfg['dataset_name'], cfg['model_names'][0], 'evaluation'],
    name_list = [cfg['dataset_name'], cfg['frequency'], cfg['model_names'][0], 'eval'],
    ext = cfg['ext']
)
eval_df_agg = aggregate_data(
    data = eval_df,
    group_columns = ['method', 'test_window', 'horizon', 'retrain_window'],
    drop_columns = ['unique_id'],
    aggregate_function = np.mean
)

# time_df.shape[0] = n_samples * n_retrain_scenarios = 365 * 10
time_df = load_data(
    path_list = ['results', cfg['dataset_name'], cfg['model_names'][0], 'time'],
    name_list = [cfg['dataset_name'], cfg['frequency'], cfg['model_names'][0], 'time'],
    ext = cfg['ext']
)
time_df_agg = aggregate_data(
    data = time_df,
    group_columns = ['method', 'test_window', 'horizon', 'retrain_window'],
    drop_columns = ['sample'],
    aggregate_function = np.sum
)


# Plotting ----------------------------------------------------------------

# plot_series(
#     forecasts_df = out_sample_df \
#         .query('sample == 7') \
#         .drop(columns = ['sample', 'y', 'test_window', 'horizon', 'retrain_window'], axis = 1),
#     level = [90],
#     max_ids = 4, 
#     max_insample_length = horizon * 5, 
#     engine = 'plotly'
# ).show()

# # NOTE: the in_sample_df stores only fitting values for the re-training
# # times. So plots with fitted values can be done only every retrain period.
# plot_series(
#     in_sample_df.query('sample == 7').drop(columns = ['sample'], axis = 1),
#     out_sample_df.query('sample == 7').drop(columns = ['sample', 'y'], axis = 1),
#     level = [90],
#     max_ids = 4, 
#     max_insample_length = horizon * 5, 
#     engine = 'plotly'
# ).show()

