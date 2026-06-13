import sys
sys.path.insert(0, 'src/Python/utils')
import gc
import numpy as np
import pandas as pd
from utilities import (
    create_file_path, create_file_name, get_file_name, 
    save_data, load_data, combine_and_save_files
)
from evaluate_forecasts import get_metrics, aggregate_data, evaluate_forecasts, get_metric_type
from collect_data import get_data

import logging
module_logger = logging.getLogger('evaluate_forecasts')

def get_stability_metrics(metric_type = 'point'):

    if metric_type == 'point':
        stab_met = {
            'bias': 'stab_bias', 
            'mae': 'mac', 
            'mase': 'masc',
            'rmse': 'rmsc',
            'rmsse': 'rmssc', 
            'smape': 'smapc'
        }
    else:
        stab_met = {
            'smape': 'smqpc',
            'rmsse': 'rmssqc'
        }
    return stab_met  

def evaluate_point_stability(out_sample_0_df, out_sample_1_df, point_metrics_dict, train_df):

    """Function to evaluate the stability of a single point forecast.

    Args:
        out_sample_0_df (pd.DataFrame): out-of-sample dataframe with forecasts from the first model fit.
        out_sample_1_df (pd.DataFrame): out-of-sample dataframe with forecasts from the second model fit.
        point_metrics_dict (dict): dictionary of point metric functions to evaluate.
        train_df (pd.DataFrame): training dataframe.

    Returns:
        pd.DataFrame: dataframe with stability metrics for the evaluated point forecast.
    """

    module_logger.info('Evaluate stability of point forecasts...')

    out_sample_0_df = out_sample_0_df[['unique_id', 'ds', 'fcst']]
    out_sample_0_df = out_sample_0_df.rename({'fcst': 'y'}, axis = 1)
    out_sample_0_df = out_sample_0_df.reset_index(drop = True)

    quantile_columns = [col for col in out_sample_1_df.columns if '-lo-' in col or '-hi-' in col]
    out_sample_1_df = out_sample_1_df.drop(columns = ['y'] + quantile_columns, inplace = False)
    out_sample_1_df = out_sample_1_df.reset_index(drop = True)

    out_sample_df = out_sample_0_df.merge(out_sample_1_df, how = 'inner', on = ['unique_id', 'ds'])
    # nobs = out_sample_df.shape[0] / len(out_sample_df['unique_id'].unique())
    # module_logger.info(f'Evaluation based on {nobs} observations')

    point_metrics = list(point_metrics_dict.values())

    # Temporarily disable logs from evaluate_forecasts
    logger_ef = logging.getLogger('evaluate_forecasts')
    _prev_disabled = logger_ef.disabled
    logger_ef.disabled = True
    try:
        stab_point_df = evaluate_forecasts(
            out_sample_df = out_sample_df,
            metrics = point_metrics, 
            train_df = train_df
        )
    finally:
        logger_ef.disabled = _prev_disabled

    stab_point_df.rename(get_stability_metrics('point'), axis = 1, inplace = True)

    return stab_point_df

def evaluate_probabilistic_stability(out_sample_0_df, out_sample_1_df, prob_metrics_dict, levels, train_df):

    """Function to evaluate the stability of probabilistic forecasts.

    Args:
        out_sample_0_df (pd.DataFrame): out-of-sample dataframe with forecasts from the first model fit.
        out_sample_1_df (pd.DataFrame): out-of-sample dataframe with forecasts from the second model fit.
        prob_metrics_dict (dict): dictionary of probabilistic metric functions to evaluate.
        levels (list): list of quantile levels to evaluate.
        train_df (pd.DataFrame): training dataframe.

    Returns:
        pd.DataFrame: dataframe with stability metrics for the evaluated probabilistic forecast.
    """

    module_logger.info('Evaluate stability of probabilistic forecasts...')

    quantile_columns_0 = [col for col in out_sample_0_df.columns if ('-lo-' in col or '-hi-' in col) and col.split('-')[-1] in [str(l) for l in levels]]
    quantile_columns_1 = [col for col in out_sample_1_df.columns if ('-lo-' in col or '-hi-' in col) and col.split('-')[-1] in [str(l) for l in levels]]
    if set(quantile_columns_0) != set(quantile_columns_1):
        module_logger.error('The quantile columns in the two dataframes are different. Please check the input dataframes.')
        raise ValueError('The quantile columns in the two dataframes are different. Please check the input dataframes.')
    
    stab_prob_df = pd.DataFrame()
    stab_prob_df['unique_id'] = out_sample_1_df['unique_id'].unique()
    stab_prob_df['sample'] = out_sample_1_df['sample']
    stab_prob_df['method'] = out_sample_1_df['method']
    stab_prob_df['test_window'] = out_sample_1_df['test_window']
    stab_prob_df['horizon'] = out_sample_1_df['horizon']
    stab_prob_df['retrain_window'] = out_sample_1_df['retrain_window']

    for col in quantile_columns_0:

        lvl = col.split('-')[-1]
        lvl_type = col.split('-')[-2]

        out_sample_0_df_tmp = out_sample_0_df[['unique_id', 'ds', col]]
        out_sample_0_df_tmp = out_sample_0_df_tmp.rename({col: 'y'}, axis = 1)
        out_sample_0_df_tmp = out_sample_0_df_tmp.reset_index(drop = True)

        out_sample_1_df_tmp = out_sample_1_df.drop(columns = 'fcst', inplace = False)
        out_sample_1_df_tmp = out_sample_1_df_tmp.rename({col: 'fcst'}, axis = 1)
        quantile_columns_tmp = [col for col in out_sample_1_df_tmp.columns if '-lo-' in col or '-hi-' in col]
        out_sample_1_df_tmp = out_sample_1_df_tmp.drop(columns = ['y'] + quantile_columns_tmp, inplace = False)
        out_sample_1_df_tmp = out_sample_1_df_tmp.reset_index(drop = True)

        out_sample_df_tmp = out_sample_0_df_tmp.merge(out_sample_1_df_tmp, how = 'inner', on = ['unique_id', 'ds'])

        prob_metrics = list(prob_metrics_dict.values())

        # Temporarily disable logs from evaluate_forecasts
        logger_ef = logging.getLogger('evaluate_forecasts')
        _prev_disabled = logger_ef.disabled
        logger_ef.disabled = True
        try:
            stab_prob_df_tmp = evaluate_forecasts(
                out_sample_df = out_sample_df_tmp,
                metrics = prob_metrics, 
                train_df = train_df
            )
        finally:
            logger_ef.disabled = _prev_disabled

        stab_prob_df_tmp.rename(get_stability_metrics('prob'), axis = 1, inplace = True)
        stab_prob_df_tmp = stab_prob_df_tmp.rename({col: f'{col}-{lvl_type}-{lvl}' for col in stab_prob_df_tmp.columns if col in prob_metrics_dict.keys()}, axis = 1)
        stab_prob_df = stab_prob_df.merge(stab_prob_df_tmp, how = 'left', on = ['unique_id', 'sample', 'method', 'test_window', 'horizon', 'retrain_window'])

        del out_sample_0_df_tmp, out_sample_1_df_tmp, out_sample_df_tmp, stab_prob_df_tmp

    # compute the average stability across quantile levels for each metric
    for met_col in prob_metrics_dict.keys():
        stab_prob_df[met_col] = stab_prob_df[[col for col in stab_prob_df.columns if met_col in col]].mean(axis = 1)

    return stab_prob_df

def evaluate_model_stability(config):

    """Function to evaluate a specific model stability.

    Args:
        config (dict): configuration dictionary.
    """

    module_logger.info('===============================================================')

    # dataset parameters
    dataset_name = config['dataset']['dataset_name']
    frequency = config['dataset']['frequency']
    min_series_length = config['dataset']['min_series_length']
    max_series_length = config['dataset']['max_series_length']
    samples = config['dataset']['samples']
    ext = config['dataset']['ext']
    seed = config['dataset']['seed']
    # fitting parameters    
    retrain_scenarios = config['fitting']['retrain_scenarios']
    levels = config['fitting']['levels']
    combine_only = config['fitting']['combine_only']
    # model parameters
    model_names = config['model_names']
    # evaluation parameters
    eval_freq = config['evaluation']['evaluation_frequency']
    metrics = config['evaluation']['metrics']
    skip =  config['evaluation']['skip']
    
    # load the dataset
    if samples is not None:
        np.random.seed(seed)
    train_df = get_data(
        path_list = ['data', dataset_name],
        name_list = [dataset_name, frequency, 'prep'],
        ext = '.parquet',
        min_series_length = min_series_length,
        max_series_length = max_series_length,
        samples = samples
    )
    train_df = train_df[['unique_id', 'ds', 'y']]

    # separate metrics into point and probabilistic metrics into two dictionaries and get the metric functions
    point_metrics_dict = {}
    prob_metrics_dict = {}
    for met in metrics:
        mt = get_metric_type(met)
        if mt == 'point':
            point_metrics_dict[met] = get_metrics([met], eval_freq)[0]
        else:
            prob_metrics_dict[met] = get_metrics([met], eval_freq)[0]

    
    for m in model_names:

        module_logger.info('---------------------------- START ----------------------------')
        module_logger.info(f'[ Model name: {m} ]')

        if not combine_only:

            for rs in retrain_scenarios:

                module_logger.info(f'Evaluate predictions for retrain scenario: {rs}')

                stab_df_retrain = pd.DataFrame() 

                file_names_tmp = get_file_name(
                    path_list = ['results', dataset_name, frequency, m, rs, 'outsample', 'tmp'], 
                    name_list = None,
                    ext = ext
                )
                file_names_tmp.sort(key = lambda x: int("".join([i for i in x if i.isdigit()])))

                for i in range(len(file_names_tmp) - skip):

                    out_sample_0_df = load_data(
                        path_list = ['results', dataset_name, frequency, m, rs, 'outsample', 'tmp'],
                        name_list = [file_names_tmp[i]],
                        ext = ext
                    )
                    out_sample_1_df = load_data(
                        path_list = ['results', dataset_name, frequency, m, rs, 'outsample', 'tmp'],
                        name_list = [file_names_tmp[i + skip]],
                        ext = ext
                    )    

                    # Point stability evaluation
                    stab_point_df_tmp = evaluate_point_stability(
                        out_sample_0_df = out_sample_0_df,
                        out_sample_1_df = out_sample_1_df,
                        point_metrics_dict = point_metrics_dict,
                        train_df = train_df
                    )
                    
                    # Probabilistic stability evaluation
                    stab_prob_df_tmp = evaluate_probabilistic_stability(
                        out_sample_0_df = out_sample_0_df,
                        out_sample_1_df = out_sample_1_df,
                        prob_metrics_dict = prob_metrics_dict,
                        levels = levels,
                        train_df = train_df
                    )

                    stab_df_tmp = stab_point_df_tmp.merge(
                        stab_prob_df_tmp, how = 'left', 
                        on = ['unique_id', 'sample', 'method', 'test_window', 'horizon', 'retrain_window']
                    )
                    stab_df_retrain = pd.concat([stab_df_retrain, stab_df_tmp], axis = 0)

                    del out_sample_0_df, out_sample_1_df, stab_point_df_tmp, stab_prob_df_tmp, stab_df_tmp
                    if (i % 10) == 0:
                        gc.collect()

                stab_df_agg_by_id_tmp = aggregate_data(
                    data = stab_df_retrain,
                    group_columns = ['method', 'test_window', 'horizon', 'retrain_window', 'unique_id'],
                    drop_columns = ['sample'],
                    function_name = 'mean',
                    adjust_metrics = False
                )
                del stab_df_retrain
                save_data(
                    stab_df_agg_by_id_tmp,
                    path_list = ['results', dataset_name, frequency, m, 'stability', 'byretrain'],
                    name_list = [dataset_name, frequency, m, rs, 'stab'],
                    ext = ext
                )
                del stab_df_agg_by_id_tmp

        # combine and save evaluation results
        combine_and_save_files(
            path_list_to_read = ['results', dataset_name, frequency, m, 'stability', 'byretrain'],
            path_list_to_write = ['results', dataset_name, frequency, m, 'stability'],
            name_list = [dataset_name, frequency, m, 'stab'],
            ext = ext
        )

        module_logger.info('----------------------------- END -----------------------------')

    module_logger.info('===============================================================')

    return

def evaluate_dataset_stability(config):

    """Function to evaluate all models for a specific dataset.

    Args:
        config (dict): configuration dictionary.
    """
    
    module_logger.info('===============================================================')
    module_logger.info('---------------------------- START ----------------------------')

    dataset_names = config['dataset']['dataset_names']
    frequencies = config['dataset']['frequencies']
    ext = config['dataset']['ext']
    model_names = config['model_names']

    for i in range(len(dataset_names)):

        dataset_name_tmp = dataset_names[i]
        freq_tmp = frequencies[i]
        module_logger.info(f'[ Dataset: {dataset_name_tmp} | Frequency: {freq_tmp} ]')

        # get file paths and names of stability and time samples
        stab_f_list = []
        for m in model_names:
            stab_f_list += [
                create_file_path(
                    path_list = ['results', dataset_name_tmp, freq_tmp, m, 'stability']
                ) + 
                create_file_name(
                    name_list = [dataset_name_tmp, freq_tmp, m, 'stab'],
                    ext = ext
                )
            ]

        # combine and save stability results
        combine_and_save_files(
            path_list_to_read = None,
            path_list_to_write = ['results', dataset_name_tmp, freq_tmp, 'stability'],
            name_list = [dataset_name_tmp, freq_tmp, 'stab'],
            ext = ext,  
            files_to_read = stab_f_list
        )

    module_logger.info('----------------------------- END -----------------------------')
    module_logger.info('===============================================================')

    return

