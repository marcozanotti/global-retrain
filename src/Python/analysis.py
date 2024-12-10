
# Retrain

# if cfg['combine_results']:

#     if cfg['store_in_sample_results']:
#         # combine and save the insample tmp files
#         combine_and_save_files(
#             path_list_to_read = ['results', dataset_name, model_name, retrain_window, 'insample', 'tmp'],
#             path_list_to_write = ['results', dataset_name, model_name, retrain_window, 'insample'],
#             name_list = [dataset_name, frequency, model_name, retrain_window, 'insample'],
#             ext = ext
#         )
    
#     # combine and save the outsample tmp files
#     combine_and_save_files(
#         path_list_to_read = ['results', dataset_name, model_name, retrain_window, 'outsample', 'tmp'],
#         path_list_to_write = ['results', dataset_name, model_name, retrain_window, 'outsample'],
#         name_list = [dataset_name, frequency, model_name, retrain_window, 'outsample'],
#         ext = ext
#     )


# Evaluation --------------------------------------------------------------

import numpy as np
from src.Python.utils.collect_data import load_data
from src.Python.utils.evaluate_forecasts import aggregate_data

from src.Python.utils.utilities import get_config
cfg = get_config('config/eval_config.yaml')

# eval_df.shape[0] = n_series * n_retrain_scenarios = 30.000 * 10
eval_df = load_data(
    path_list = ['results', cfg['dataset_name'], cfg['frequency'], cfg['model_names'][0], 'evaluation'],
    name_list = [cfg['dataset_name'], cfg['frequency'], cfg['model_names'][0], 'eval', cfg['evaluation_sample_type']],
    ext = cfg['ext']
)
eval_df_agg = aggregate_data(
    data = eval_df,
    group_columns = ['method', 'test_window', 'horizon', 'retrain_window'],
    drop_columns = ['unique_id'],
    function_name = 'mean',
    adjust_metrics = True
)

# time_df.shape[0] = n_samples * n_retrain_scenarios = 365 * 10
time_df = load_data(
    path_list = ['results', cfg['dataset_name'], cfg['frequency'], cfg['model_names'][0], 'time'],
    name_list = [cfg['dataset_name'], cfg['frequency'], cfg['model_names'][0], 'time'],
    ext = cfg['ext']
)
time_df_agg = aggregate_data(
    data = time_df,
    group_columns = ['method', 'test_window', 'horizon', 'retrain_window'],
    drop_columns = ['sample'],
    function_name = 'sum',
    adjust_metrics = True,
    retrain_scenarios = cfg['retrain_scenarios']
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

