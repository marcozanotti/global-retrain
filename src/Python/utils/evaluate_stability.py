
import sys
sys.path.insert(0, 'src/Python/utils')
import gc
import numpy as np
import pandas as pd
import pandas_flavor as pf
from functools import partial
from utilsforecast.losses import (
    bias, mae, mse, rmse, mase, msse, rmsse,
    quantile_loss, mqloss, coverage, calibration, scaled_crps
)
from utilsforecast.evaluation import evaluate
from utilities import (
    create_file_path, create_file_name, get_file_name, 
    save_data, load_data, combine_and_save_files, get_frequency
)
from collect_data import get_data
from fit_models import get_retrain_ids
from evaluate_forecasts import get_aggregate_function, aggregate_data

import logging
module_logger = logging.getLogger('evaluate_forecasts')

def get_stability_metrics(metric_names, frequency = None):

    """Function to get the stability metrics.

    Args:
        metric_names (list): list of stability metric names.
    
    Returns:
        list: list of stability metrics.
    """

    module_logger.info('Defining stability metrics...')
    freq = get_frequency(frequency)[1]

    metrics = []
    if 'smapc' in metric_names:
        metrics.append(smapc)
    if 'mac' in metric_names:
        metrics.append(mac)
    if 'rmsc' in metric_names:
        metrics.append(rmsc)

    return metrics

@pf.register_dataframe_method
def evaluate_forecasts_stability(
    out_sample_df, 
    metrics = [smapc], 
    train_df = None,
    levels = None
):

    """Function to evaluate the forecasts' stability.
    
    Args:
        out_sample_df (pd.DataFrame): dataframe with columns 'unique_id', 'ds', 'y', 'fcst'.
        metrics (list): list of evaluation metrics.
        train_df (pd.DataFrame, optional): training data in the Nixtla's format. 
        Defaults to None.

    Returns:
        pd.DataFrame: dataframe with evaluation results for each metric.
    """

    module_logger.info('Evaluating forecasts stability...')
    samples = list(out_sample_df['sample'].unique())
    # n_samples = len(samples)

    stab_df = pd.DataFrame()

    for s in samples:

        # module_logger.info(f'Samlple {s} of {n_samples}...')
        stab_df_tmp = evaluate(
            out_sample_df[out_sample_df['sample'] == s], 
            metrics = metrics,
            models = ['fcst'],
            train_df = train_df,
            id_col = 'unique_id',
            level = levels   
        ) \
            .pivot(index = 'unique_id', columns = 'metric', values = 'fcst') \
            .reset_index()
        stab_df_tmp['sample'] = s
        stab_df = pd.concat([stab_df, stab_df_tmp], axis = 0)

    stab_df['method'] = out_sample_df['method'][0]
    stab_df['test_window'] = out_sample_df['test_window'][0]
    stab_df['horizon'] = out_sample_df['horizon'][0]
    stab_df['retrain_window'] = out_sample_df['retrain_window'][0]

    return stab_df

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
    samples = config['dataset']['samples']
    ext = config['dataset']['ext']
    seed = config['dataset']['seed']
    # fitting parameters    
    retrain_scenarios = config['fitting']['retrain_scenarios']
    levels = config['fitting']['levels']
    # model parameters
    model_names = config['model_names']
    # evaluation parameters
    eval_freq = config['evaluation']['evaluation_frequency']
    metrics = get_stability_metrics(config['evaluation']['metrics'], eval_freq)

    # load the dataset
    if samples is not None:
        np.random.seed(seed)
    train_df = get_data(
        path_list = ['data', dataset_name],
        name_list = [dataset_name, frequency, 'prep'],
        ext = '.parquet',
        min_series_length = min_series_length,
        samples = samples
    )
    train_df = train_df[['unique_id', 'ds', 'y']]

    for m in model_names:

        module_logger.info('---------------------------- START ----------------------------')
        module_logger.info(f'[ Model name: {m} ]')

        for rs in retrain_scenarios:

            module_logger.info(f'Evaluate predictions for retrain scenario: {rs}')
            stab_df_retrain = pd.DataFrame() 
            file_names_tmp = get_file_name(
                path_list = ['results', dataset_name, frequency, m, rs, 'outsample', 'tmp'], 
                name_list = None,
                ext = ext
            )
            file_names_tmp.sort(key = lambda x: int("".join([i for i in x if i.isdigit()])))
            
            for i in range(len(file_names_tmp)):

                stab0_df_tmp = load_data(
                    path_list = ['results', dataset_name, frequency, m, rs, 'outsample', 'tmp'],
                    name_list = [file_names_tmp[i]],
                    ext = ext
                )
                stab0_df_tmp.reset_index(drop = True, inplace = True)

                stab1_df_tmp = load_data(
                    path_list = ['results', dataset_name, frequency, m, rs, 'outsample', 'tmp'],
                    name_list = [file_names_tmp[i + 1]],
                    ext = ext
                )                
                stab1_df_tmp.reset_index(drop = True, inplace = True)

                stab_df_tmp = evaluate_forecasts_stability(
                    out_sample_df = eval_df_tmp, #FIXME: unire stab0 e stab1
                    metrics = metrics, 
                    train_df = train_df,
                    levels = levels
                )
                stab_df_retrain = pd.concat([stab_df_retrain, stab_df_tmp], axis = 0)
                del stab_df_tmp
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
    eval_sample_type = config['evaluation']['evaluation_sample_type']

    for i in range(len(dataset_names)):

        dataset_name_tmp = dataset_names[i]
        freq_tmp = frequencies[i]
        module_logger.info(f'[ Dataset: {dataset_name_tmp} | Frequency: {freq_tmp} ]')

        # get file paths and names of evaluation and time samples
        eval_f_list = []
        time_f_lst = []
        for m in model_names:
            eval_f_list += [
                create_file_path(
                    path_list = ['results', dataset_name_tmp, freq_tmp, m, 'evaluation']
                ) + 
                create_file_name(
                    name_list = [dataset_name_tmp, freq_tmp, m, 'eval', eval_sample_type],
                    ext = ext
                )
            ]
            time_f_lst += [
                create_file_path(
                    path_list = ['results', dataset_name_tmp, freq_tmp, m, 'time']
                ) + 
                create_file_name(
                    name_list = [dataset_name_tmp, freq_tmp, m, 'time'],
                    ext = ext
                )
            ]        

        # combine and save evaluation results
        combine_and_save_files(
            path_list_to_read = None,
            path_list_to_write = ['results', dataset_name_tmp, freq_tmp, 'evaluation'],
            name_list = [dataset_name_tmp, freq_tmp, 'eval', eval_sample_type],
            ext = ext,  
            files_to_read = eval_f_list
        )
        # combine and save time results
        combine_and_save_files(
            path_list_to_read = None,
            path_list_to_write = ['results', dataset_name_tmp, freq_tmp, 'evaluation'],
            name_list = [dataset_name_tmp, freq_tmp, 'time'],
            ext = ext,
            files_to_read = time_f_lst
        )

    module_logger.info('----------------------------- END -----------------------------')
    module_logger.info('===============================================================')

    return

