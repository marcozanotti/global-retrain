
import os
from src.Python.utils.evaluate_forecasts import evaluate_model
from src.Python.utils.utilities import *


from functools import partial
from utilsforecast.losses import bias, mae, mse, rmse, mase, msse, rmsse
metrics = [
    bias, 
    mae, mse, rmse, 
    partial(mase, seasonality = 7),
    partial(msse, seasonality = 7),
    partial(rmsse, seasonality = 7)
]


os.environ['NIXTLA_ID_AS_COL'] = '1'
cfg = get_config('config/eval_config.yaml')
configure_logging(
    config_file = 'config/log_config.yaml', 
    name_list = [cfg['dataset_name'], cfg['frequency'], 'eval']
)
logger = create_logger()

evaluate_model(config = cfg)

stop_logger(logger)










# Load data ---------------------------------------------------------------

# a sample of 100 time series
np.random.seed(1992)
data = get_data(
    path_list = ['data', dataset_name],
    name_list = [dataset_name, frequency, 'prep'],
    ext = '.parquet',
    min_series_length = min_series_length,
    samples = 100
)

out_sample_df = load_data(
    path_list = ['results', dataset_name, model_name, retrain_window, 'preds', 'tmp'],
    name_list = [dataset_name, frequency, 'outsample', model_name, retrain_window, 0]
)

time_df = load_data(
    path_list = ['results', dataset_name, model_name, retrain_window, 'time'], 
    name_list = [dataset_name, frequency, 'time', model_name, retrain_window]
)

# Combine results ---------------------------------------------------------

combine_and_save_files(
    path_list_to_read = ['results', dataset_name, model_name, retrain_window, 'preds', 'tmp'],
    path_list_to_write = ['results', dataset_name, model_name, retrain_window, 'preds'],
    name_list = [dataset_name, frequency, 'outsample', model_name, retrain_window],
    ext = '.parquet'
)


# Evaluation --------------------------------------------------------------

evaluation_df = out_sample_df \
    .evaluate_point_forecasts(metrics = metrics, train_df = data) \
    .aggregate_data(
        group_columns = ['sample', 'method', 'test_window', 'horizon','retrain_window'],
        drop_columns = ['unique_id']
    )

time_df \
    .aggregate_data(
        group_columns = ['method', 'test_window', 'horizon', 'retrain_window'],
        drop_columns = ['sample'],
        aggregate_function = np.sum
    )


# Evaluate the whole model ------------------------------------------------

eval_df = evaluate_model(
    model_name = model_name,
    analysis_type = 'outsample',
    dataset_name = dataset_name, 
    frequency = frequency, 
    metrics = metrics, 
    train_df = data
)

eval_df_agg = evaluate_model(
    model_name = model_name,
    analysis_type = 'outsample',
    dataset_name = dataset_name, 
    frequency = frequency, 
    metrics = metrics, 
    train_df = data, 
    group_columns = ['sample', 'method', 'test_window', 'horizon', 'retrain_window'],
    drop_columns = ['unique_id']
)

time_df = evaluate_model(
    model_name = model_name,
    analysis_type = 'time',
    dataset_name = dataset_name, 
    frequency = frequency, 
    metrics = metrics, 
    train_df = data
)

time_df_agg = evaluate_model(
    model_name = model_name,
    analysis_type = 'time',
    dataset_name = dataset_name, 
    frequency = frequency, 
    metrics = metrics, 
    train_df = data, 
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

