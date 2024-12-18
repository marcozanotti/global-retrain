
from plotnine import ggplot, aes, geom_line, labs, theme
from src.Python.utils.collect_data import load_data
from src.Python.utils.evaluate_forecasts import aggregate_data


# Evaluation --------------------------------------------------------------

dataset_name = 'm5'
frequency = 'daily'
evaluation_type = 'overlap'
ext = '.parquet'

# eval_df.shape[0] = n_series * n_retrain_scenarios * n_models = 30.000 * 10 * 10
eval_df = load_data(
    path_list = ['results', dataset_name, frequency, 'evaluation'],
    name_list = [dataset_name, frequency, 'eval', evaluation_type],
    ext = ext
)
eval_df_agg = aggregate_data(
    data = eval_df,
    group_columns = ['method', 'retrain_window'],
    drop_columns = ['unique_id', 'test_window', 'horizon'],
    function_name = 'mean',
    adjust_metrics = True
)

# time_df.shape[0] = n_samples * n_retrain_scenarios * n_models = 365 * 10 * 10
time_df = load_data(
    path_list = ['results', dataset_name, frequency, 'evaluation'],
    name_list = [dataset_name, frequency, 'time'],
    ext = ext
)
time_df_agg = aggregate_data(
    data = time_df,
    group_columns = ['method', 'retrain_window'],
    drop_columns = ['sample', 'test_window', 'horizon'],
    function_name = 'sum',
    adjust_metrics = True
)


# Table and Plot ----------------------------------------------------------

metric = 'mae'
time_metric = 'total_sample_time'

# evaluation table
eval_df_agg[['method', 'retrain_window', metric]] \
    .pivot_table(index = 'method', columns = 'retrain_window', values = metric) \
    .reset_index()
# time table
time_df_agg[['method', 'retrain_window', time_metric]] \
    .pivot_table(index = 'method', columns = 'retrain_window', values = time_metric) \
    .reset_index()


# evaluation plot
(
    ggplot(eval_df_agg, aes(x = 'retrain_window', y = 'mae', color = 'method'))
    + geom_line()
    + labs(
        title = f'{dataset_name.upper()}: {metric.upper()} by Retrain Window',
        x = 'Retrain Window',
        y = metric.upper()
    )
    + theme(legend_position = 'bottom')
)

# time plot
(
    ggplot(time_df_agg, aes(x = 'retrain_window', y = time_metric, color = 'method'))
    + geom_line()
    + labs(
        title = f'{dataset_name.upper()}: Computing Time by Retrain Window',
        x = 'Retrain Window',
        y = 'Computing Time'
    )
    + theme(legend_position = 'bottom')
)
