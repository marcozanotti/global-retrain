import sys
sys.path.insert(0, 'src/Python/utils')
import gc
import numpy as np
import pandas as pd
from utilities import (
    get_file_name, save_data, combine_and_save_files
)
from collect_data import get_data
from evaluate_forecasts import get_metrics, evaluate_forecasts, aggregate_data

import logging
module_logger = logging.getLogger('predictions')

def combine_model_predictions(config):

    """Function to combine model predictions.

    Args:
        config (dict): configuration dictionary.
    """

    module_logger.info('===============================================================')

    # dataset parameters
    dataset_name = config['dataset']['dataset_name']
    frequency = config['dataset']['frequency']
    ext = config['dataset']['ext']
    # fitting parameters    
    retrain_scenarios = config['fitting']['retrain_scenarios']
    combine_only =config['fitting']['combine_only']
    # model parameters
    model_names = config['model_names']

    for m in model_names:

        module_logger.info('---------------------------- START ----------------------------')
        module_logger.info(f'[ Model name: {m} ]')

        if not combine_only:

            for rs in retrain_scenarios:

                f_list_tmp = get_file_name(
                    path_list = ['results', dataset_name, frequency, m, rs, 'outsample', 'tmp'], 
                    name_list = None,
                    ext = ext, remove_ext=False, add_path=True
                )
                f_list_tmp.sort(key = lambda x: int("".join([i for i in x if i.isdigit()])))

                module_logger.info(f'Combining predictions for retrain scenario: {rs}')
                combine_and_save_files(
                    path_list_to_read = None,
                    path_list_to_write = ['results', dataset_name, frequency, m, 'preds', 'byretrain'],
                    name_list = [dataset_name, frequency, m, rs, 'outsample'],
                    ext = ext,
                    files_to_read = f_list_tmp
                )

        f_list_tmp = get_file_name(
            path_list = ['results', dataset_name, frequency, m, 'preds', 'byretrain'], 
            name_list = None,
            ext = ext, remove_ext=False, add_path=True
        )
        f_list_tmp.sort(key = lambda x: int("".join([i for i in x if i.isdigit()])))
        
        combine_and_save_files(
            path_list_to_read = None,
            path_list_to_write = ['results', dataset_name, frequency, m, 'preds'],
            name_list = [dataset_name, frequency, m, 'preds'],
            ext = ext,
            files_to_read = f_list_tmp
        )

        module_logger.info('----------------------------- END -----------------------------')

    module_logger.info('===============================================================')

    return

def combine_dataset_predictions(config):

    """Function to combine dataset predictions.

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

        # get file paths and names of evaluation and time samples
        preds_f_list = []
        for m in model_names:
            preds_f_list += get_file_name(
                path_list = ['results', dataset_name_tmp, freq_tmp, m, 'preds'], 
                name_list = [dataset_name_tmp, freq_tmp, m, 'preds'],
                ext = ext, remove_ext=False, add_path=True
            )

        combine_and_save_files(
            path_list_to_read = None,
            path_list_to_write = ['results', dataset_name_tmp, freq_tmp, 'preds'],
            name_list = [dataset_name_tmp, freq_tmp, 'preds'],
            ext = ext,
            files_to_read = preds_f_list
        )

    module_logger.info('===============================================================')

    return

def evaluate_predictions(config):

    """Function to evaluate predictions.

    Args:
        config (dict): configuration dictionary.
    """

    module_logger.info('===============================================================')
    module_logger.info('---------------------------- START ----------------------------')

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
    # combine_only = config['fitting']['combine_only']
    # model parameters
    model_names = config['model_names']
    # evaluation parameters
    eval_freq = config['evaluation']['evaluation_frequency']
    metrics = get_metrics(config['evaluation']['metrics'], eval_freq)
    eval_sample_type = config['evaluation']['evaluation_sample_type']
    weighted_metrics = config['evaluation']['weighted_metrics']
    weights = config['evaluation']['weights']

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

    preds_df = get_data(
        path_list = ['results', dataset_name, frequency, 'preds'],
        name_list = [dataset_name, frequency, 'preds'],
        ext = ext
    )

    eval_df = pd.DataFrame()

    for rs in retrain_scenarios:

        module_logger.info('---------------------------- START ----------------------------')
        module_logger.info(f'[ Retrain scenario: {rs} ]')

        for m in model_names:

            module_logger.info(f'[ Model name: {m} ]')
            if preds_df[preds_df['method'] == m].empty:
                module_logger.warning(f'No predictions found for model {m}. Skipping evaluation for this model.')
                continue
            preds_df_tmp = preds_df[preds_df['retrain_window'] == rs]
            preds_df_tmp = preds_df_tmp[preds_df_tmp['method'] == m]
            preds_df_tmp.reset_index(drop = True, inplace = True)

            eval_df_tmp = evaluate_forecasts(
                out_sample_df = preds_df_tmp, 
                metrics = metrics, 
                train_df = train_df,
                levels = levels, 
                weighted_metrics = config['evaluation']['weighted_metrics'],
                weights = config['evaluation']['weights']
            )
            eval_df_agg_tmp = aggregate_data(
                data = eval_df_tmp,
                group_columns = ['method', 'test_window', 'horizon', 'retrain_window', 'unique_id'],
                drop_columns = ['sample'],
                function_name = 'mean',
                adjust_metrics = True
            )

            eval_df = pd.concat([eval_df, eval_df_agg_tmp], axis = 0)
            del preds_df_tmp, eval_df_tmp, eval_df_agg_tmp
            gc.collect()
        
        module_logger.info('----------------------------- END -----------------------------')

    save_data(
        eval_df,
        path_list = ['results', dataset_name, frequency, 'preds'],
        name_list = [dataset_name, frequency, 'preds', 'eval', eval_sample_type],
        ext = ext
    )

    module_logger.info('===============================================================')

    return