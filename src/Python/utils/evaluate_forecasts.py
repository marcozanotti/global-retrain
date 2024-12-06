
import gc
import numpy as np
import pandas as pd
import pandas_flavor as pf
from utilsforecast.losses import bias, mae, mse, rmse
from utilsforecast.evaluation import evaluate
from src.Python.utils.collect_data import *

import logging
module_logger = logging.getLogger('evaluate_forecasts')

@pf.register_dataframe_method
def evaluate_point_forecasts(
    out_sample_df, 
    metrics = [bias, mae, mse, rmse], 
    train_df = None
):

    """Function to evaluate the point forecast.
    
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
    n_samples = len(samples)

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

def evaluate_model(
    config,
    model_name, 
    analysis_type,
    dataset_name, 
    frequency, 
    metrics = [bias, mae, mse, rmse], 
    train_df = None, 
    group_columns = None,
    drop_columns = None,
    aggregate_function = np.mean,
    ext = '.parquet'
):

    """Function to evaluate a specific model.

    Args:
        config (dict): configuration dictionary.

    Returns:
        pd.DataFrame: dataframe with evaluation results for each metric.
    """

    module_logger.info('===============================================================')

    seed = config['seed']
    dataset_name = config['dataset_name']
    frequency = config['frequency']
    retrain_scenarios = config['retrain_scenarios']
    model_names = config['model_names']
    metrics = config['metrics']
    min_series_length = config['min_series_length']
    samples = config['samples']
    ext = config['ext']

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

    # FIXME: set_metrics

    for m in model_names:

        module_logger.info('---------------------------- START ----------------------------')
        
        model_type = get_model_type(m)
        module_logger.info(f'[ Model type: {model_type} | Model name: {m} ]')

        for rs in retrain_scenarios:
            
            rs = 7

            module_logger.info('Evaluating outsample results...')
            eval_df = pd.DataFrame() # nrow(eval_df) = 30.000 * 365 = 11.000.000
            i = 0
            file_names_tmp = get_file_name(
                path_list = ['results', dataset_name, m, rs, 'outsample', 'tmp'], 
                name_list = None,
                ext = ext
            )
            for f in file_names_tmp:
                eval_df_tmp = load_data(
                    path_list = ['results', dataset_name, m, rs, 'outsample', 'tmp'],
                    name_list = [f],
                    ext = ext
                ) \
                    .evaluate_point_forecasts(metrics = metrics, train_df = data)
                eval_df = pd.concat([eval_df, eval_df_tmp], axis = 0)
                del eval_df_tmp
                i += 1
                if (i % 10) == 0:
                    gc.collect()

            # nrow(eval_df_agg_by_id) = 30.000
            eval_df_agg_by_id = eval_df \
                    .aggregate_data(
                        group_columns = ['method', 'test_window', 'horizon', 'retrain_window', 'unique_id'],
                        drop_columns = ['sample'],
                        aggregate_function = np.mean
                    )

            # nrow(eval_df_agg_rs) = 1
            eval_df_agg_rs = eval_df_agg_by_id \
                .aggregate_data(
                    group_columns = ['method', 'test_window', 'horizon', 'retrain_window'],
                    drop_columns = ['unique_id'],
                    aggregate_function = np.mean
                )

        
        module_logger.info('----------------------------- END -----------------------------')

    module_logger.info('===============================================================')

    return


