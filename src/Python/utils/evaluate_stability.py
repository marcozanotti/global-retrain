
import sys
sys.path.insert(0, 'src/Python/utils')
import gc
import numpy as np
import pandas as pd
import pandas_flavor as pf
from utilsforecast.losses import bias, mae, rmse, smape
from utilsforecast.evaluation import evaluate
from utilities import (
    create_file_path, create_file_name, get_file_name, 
    save_data, load_data, combine_and_save_files, get_frequency
)
from collect_data import get_data
from fit_models import get_retrain_ids
from evaluate_forecasts import get_metrics, get_aggregate_function, aggregate_data, evaluate_forecasts

import logging
module_logger = logging.getLogger('evaluate_forecasts')


def get_stability_metrics():
    stab_met = {'bias': 'stability_bias', 'mae': 'mac', 'rmse': 'rmsc', 'smape': 'smapc'}
    return stab_met    

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
    metrics = get_metrics(config['evaluation']['metrics'])

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
            
            for i in range(len(file_names_tmp) - 1):

                stab0_df_tmp = load_data(
                    path_list = ['results', dataset_name, frequency, m, rs, 'outsample', 'tmp'],
                    name_list = [file_names_tmp[i]],
                    ext = ext
                )
                stab0_df_tmp = stab0_df_tmp[['unique_id', 'ds', 'fcst']]
                stab0_df_tmp.rename({'fcst': 'y'}, axis = 1, inplace = True)
                stab0_df_tmp.reset_index(drop = True, inplace = True)

                stab1_df_tmp = load_data(
                    path_list = ['results', dataset_name, frequency, m, rs, 'outsample', 'tmp'],
                    name_list = [file_names_tmp[i + 1]],
                    ext = ext
                )    
                stab1_df_tmp.drop('y', axis = 1, inplace = True)            
                stab1_df_tmp.reset_index(drop = True, inplace = True)
                
                stab_df_tmp = stab1_df_tmp.merge(stab0_df_tmp, how = 'inner', on = ['unique_id', 'ds'])

                stab_df_tmp = evaluate_forecasts(
                    out_sample_df = stab_df_tmp,
                    metrics = metrics, 
                    train_df = None,
                    levels = levels
                )
                stab_df_tmp.rename(get_stability_metrics(), axis = 1, inplace = True)
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

