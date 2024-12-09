
import gc
import numpy as np
import pandas as pd
import pandas_flavor as pf
from functools import partial
from utilsforecast.losses import bias, mae, mse, rmse, mase, msse, rmsse
from utilsforecast.evaluation import evaluate
from src.Python.utils.collect_data import (
    get_file_name, get_data, load_data, save_data, combine_and_save_files
)
from src.Python.utils.set_engine import get_frequency
from src.Python.utils.fit_models import get_retrain_ids

import logging
module_logger = logging.getLogger('evaluate_forecasts')

@pf.register_dataframe_method
def aggregate_data(data, group_columns, drop_columns = None, aggregate_function = np.mean, adjust_metrics = True):

    """Function to aggregate evaluation metrics.

    Args:
        data (pd.DataFrame): dataframe to aggregate.
        group_columns (list): list of columns to group by.
        aggregate_function (str, optional): function to use to aggregate. 
        Defaults to 'mean'.
    
    Returns:
        pd.DataFrame: dataframe with aggregated data.
    """

    data_agg = data.copy()

    module_logger.info('Aggregating data...')
    if drop_columns is not None:
        data_agg.drop(columns = drop_columns, inplace = True)

    data_agg = data_agg \
        .groupby(group_columns) \
        .agg(aggregate_function) \
        .reset_index()
    
    if adjust_metrics:
        if 'rmse' in data_agg.columns:
            data_agg['rm_mse'] = np.sqrt(data_agg['mse'])
        if 'msse' in data_agg.columns:
            data_agg['rm_msse'] = np.sqrt(data_agg['msse'])
        if 'total_fit_time' in data_agg.columns:
            # tw, h, rw = data_agg['test_window'][0], data_agg['horizon'][0], data_agg['retrain_window'][0]
            # ids = list(range(0, (tw - h + 1), rw)) # same as get_retrain_ids()
            ids = get_retrain_ids(data_agg['test_window'][0], data_agg['horizon'][0], data_agg['retrain_window'][0])
            data_agg['total_fit_time'] = aggregate_function(data['total_fit_time'][ids])

    return data_agg

def get_metrics(metric_names, frequency = None):

    """Function to get the evaluation metrics.

    Args:
        metric_names (list): list of evaluation metric names.
    
    Returns:
        list: list of evaluation metrics.
    """

    module_logger.info('Defining evaluation metrics...')
    freq = get_frequency(frequency)[1]

    metrics = []
    if 'bias' in metric_names:
        metrics.append(bias)
    if 'mae' in metric_names:
        metrics.append(mae)
    if 'mse' in metric_names:
        metrics.append(mse)
    if 'rmse' in metric_names:
        metrics.append(rmse)
    if 'mase' in metric_names:
        metrics.append(partial(mase, seasonality = freq))
    if 'msse' in metric_names:
        metrics.append(partial(msse, seasonality = freq))
    if 'rmsse' in metric_names:
        metrics.append(partial(rmsse, seasonality = freq))

    return metrics

@pf.register_dataframe_method
def evaluate_point_forecasts(
    out_sample_df, 
    metrics = [bias, mae, mse, rmse], 
    train_df = None
):

    """Function to evaluate the point forecasts.
    
    Args:
        out_sample_df (pd.DataFrame): dataframe with columns 'unique_id', 'ds', 'y', 'fcst'.
        metrics (list): list of evaluation metrics.
        train_df (pd.DataFrame, optional): training data in the Nixtla's format. 
        Defaults to None.

    Returns:
        pd.DataFrame: dataframe with evaluation results for each metric.
    """

    module_logger.info('Evaluating point forecasts...')
    samples = list(out_sample_df['sample'].unique())
    # n_samples = len(samples)

    eval_df = pd.DataFrame()

    for s in samples:

        # module_logger.info(f'Samlple {s} of {n_samples}...')
        eval_df_tmp = evaluate(
            out_sample_df[out_sample_df['sample'] == s], 
            metrics = metrics,
            models = ['fcst'],
            train_df = train_df,
            id_col = 'unique_id'        
        ) \
            .pivot(index = 'unique_id', columns = 'metric', values = 'fcst') \
            .reset_index()
        eval_df_tmp['sample'] = s
        eval_df = pd.concat([eval_df, eval_df_tmp], axis = 0)

    eval_df['method'] = out_sample_df['method'][0]
    eval_df['test_window'] = out_sample_df['test_window'][0]
    eval_df['horizon'] = out_sample_df['horizon'][0]
    eval_df['retrain_window'] = out_sample_df['retrain_window'][0]

    return eval_df

@pf.register_dataframe_method
def evaluate_interval_forecasts(
    out_sample_df, 
    metrics = [bias, mae, mse, rmse], 
    train_df = None
):

    """Function to evaluate the interval forecasts.
    
    Args:
        out_sample_df (pd.DataFrame): dataframe with columns 'unique_id', 'ds', 'y', 'fcst'.
        metrics (list): list of evaluation metrics.
        train_df (pd.DataFrame, optional): training data in the Nixtla's format. 
        Defaults to None.

    Returns:
        pd.DataFrame: dataframe with evaluation results for each metric.
    """

    module_logger.info('Evaluating interval forecasts...')
    samples = list(out_sample_df['sample'].unique())
    # n_samples = len(samples)

    eval_df = pd.DataFrame()

    for s in samples:

        # module_logger.info(f'Samlple {s} of {n_samples}...')
        eval_df_tmp = evaluate(
            out_sample_df[out_sample_df['sample'] == s], 
            metrics = metrics,
            models = ['fcst'],
            train_df = train_df,
            id_col = 'unique_id'        
        ) \
            .pivot(index = 'unique_id', columns = 'metric', values = 'fcst') \
            .reset_index()
        eval_df_tmp['sample'] = s
        eval_df = pd.concat([eval_df, eval_df_tmp], axis = 0)

    eval_df['method'] = out_sample_df['method'][0]
    eval_df['test_window'] = out_sample_df['test_window'][0]
    eval_df['horizon'] = out_sample_df['horizon'][0]
    eval_df['retrain_window'] = out_sample_df['retrain_window'][0]

    return eval_df

def evaluate_model(config):

    """Function to evaluate a specific model.

    Args:
        config (dict): configuration dictionary.
    """

    module_logger.info('===============================================================')

    seed = config['seed']
    dataset_name = config['dataset_name']
    frequency = config['frequency']
    retrain_scenarios = config['retrain_scenarios']
    model_names = config['model_names']
    eval_type = config['evaluation_type']
    eval_samples = config['evaluation_samples']
    min_series_length = config['min_series_length']
    samples = config['samples']
    ext = config['ext']

    metrics = get_metrics(config['metrics'], frequency)

    # load the dataset
    if samples is not None:
        np.random.seed(seed)
    data = get_data(
        path_list = ['data', dataset_name],
        name_list = [dataset_name, frequency, 'prep'],
        ext = '.parquet',
        min_series_length = min_series_length,
        samples = samples
    )

    for m in model_names:

        module_logger.info('---------------------------- START ----------------------------')
        module_logger.info(f'[ Model name: {m} ]')

        for rs in retrain_scenarios:

            module_logger.info(f'Evaluate predictions for retrain scenario: {rs}')
            eval_df_retrain = pd.DataFrame() # eval_df_retrain.shape[0] = 30.000 * 365 = 11.000.000
            file_names_tmp = get_file_name(
                path_list = ['results', dataset_name, frequency, m, rs, 'outsample', 'tmp'], 
                name_list = None,
                ext = ext
            )
            file_names_tmp.sort(key = lambda x: int("".join([i for i in x if i.isdigit()])))
            
            for i in range(len(file_names_tmp)):
                eval_df_tmp = load_data(
                    path_list = ['results', dataset_name, frequency, m, rs, 'outsample', 'tmp'],
                    name_list = [file_names_tmp[i]],
                    ext = ext
                )

                if eval_samples == 'nooverlap':
                    if i == 0: 
                        ds_tmp = set(eval_df_tmp['ds'].unique())
                        ds_to_keep = set([eval_df_tmp['ds'].max()]) # keep only last date to be consistent among samples
                        ds_to_remove = ds_tmp - ds_to_keep
                        eval_df_tmp = eval_df_tmp.loc[eval_df_tmp['ds'].isin(ds_to_keep)]
                        ds_to_remove = ds_to_remove | ds_to_keep # update ds_to_remove
                    else:
                        ds_tmp = set(eval_df_tmp['ds'].unique())
                        ds_to_keep = ds_tmp - ds_to_remove
                        eval_df_tmp = eval_df_tmp.loc[eval_df_tmp['ds'].isin(ds_to_keep)]
                        ds_to_remove = ds_to_remove | ds_to_keep # update ds_to_remove

                eval_df_tmp.reset_index(drop = True, inplace = True)

                if eval_type == 'point':
                    eval_df_tmp = evaluate_point_forecasts(
                        out_sample_df = eval_df_tmp, 
                        metrics = metrics, 
                        train_df = data
                    )
                elif eval_type == 'interval':
                    eval_df_tmp = evaluate_interval_forecasts(
                        out_sample_df = eval_df_tmp, 
                        metrics = metrics, 
                        train_df = data
                    )
                else:
                    raise ValueError(f'Invalid evaluation type {eval_type}.')

                eval_df_retrain = pd.concat([eval_df_retrain, eval_df_tmp], axis = 0)
                del eval_df_tmp
                if (i % 10) == 0:
                    gc.collect()

            # eval_df_agg_by_id.shape[0] = 30.000
            eval_df_agg_by_id_tmp = aggregate_data(
                data = eval_df_retrain,
                group_columns = ['method', 'test_window', 'horizon', 'retrain_window', 'unique_id'],
                drop_columns = ['sample'],
                aggregate_function = np.mean
            )
            del eval_df_retrain
            save_data(
                eval_df_agg_by_id_tmp,
                path_list = ['results', dataset_name, frequency, m, 'evaluation', 'byretrain'],
                name_list = [dataset_name, frequency, m, rs, 'eval', eval_type, eval_samples],
                ext = ext
            )
            del eval_df_agg_by_id_tmp

        # combine and save evaluation results
        combine_and_save_files(
            path_list_to_read = ['results', dataset_name, frequency, m, 'evaluation', 'byretrain'],
            path_list_to_write = ['results', dataset_name, frequency, m, 'evaluation'],
            name_list = [dataset_name, frequency, m, 'eval', eval_type, eval_samples],
            ext = ext
        )
        # combine and save time results
        combine_and_save_files(
            path_list_to_read = ['results', dataset_name, frequency, m, 'time', 'byretrain'],
            path_list_to_write = ['results', dataset_name, frequency, m, 'time'],
            name_list = [dataset_name, frequency, m, 'time'],
            ext = ext
        )

        module_logger.info('----------------------------- END -----------------------------')

    module_logger.info('===============================================================')

    return

