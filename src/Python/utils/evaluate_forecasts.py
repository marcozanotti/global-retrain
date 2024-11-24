
import os
import numpy as np
import pandas as pd
import pandas_flavor as pf

from utilsforecast.losses import bias, mae, mse, rmse
from utilsforecast.evaluation import evaluate

from utils import *


@pf.register_dataframe_method
def evaluate_point_forecasts(
    out_sample_df, 
    metrics = [bias, mae, mse, rmse], 
    train_df = None
):

    """Function to evaluate the point forecast.
    
    Args:
        forecasts_df (pd.DataFrame): dataframe with columns 'unique_id', 'ds', 'y', 'fcst'.
        metrics (list): list of evaluation metrics.
        train_df (pd.DataFrame, optional): training data in the Nixtla's format. 
        Defaults to None.
        id_col (str, optional): column name for unique_id. Defaults to 'unique_id'.

    Returns:
        pd.DataFrame: dataframe with evaluation results for each metric.
    """

    print('Evaluating point forecasts...')
    fcst_df = out_sample_df.copy()
    samples = list(fcst_df['sample'].unique())
    n_samples = len(samples)

    eval_df = pd.DataFrame()

    for s in samples:

        # print(f'Samlple {s} of {n_samples}...')
        fcst_df_tmp = fcst_df[fcst_df['sample'] == s]
        eval_df_tmp = evaluate(
            fcst_df_tmp, 
            metrics = metrics,
            models = ['fcst'],
            train_df = train_df,
            id_col = 'unique_id'        
        ) \
            .pivot(index = 'unique_id', columns ='metric', values = 'fcst') \
            .reset_index()
        eval_df_tmp['sample'] = s
        eval_df = pd.concat([eval_df, eval_df_tmp], axis = 0)

    eval_df['method'] = out_sample_df['method'][0]
    eval_df['test_window'] = out_sample_df['test_window'][0]
    eval_df['horizon'] = out_sample_df['horizon'][0]
    eval_df['retrain_window'] = out_sample_df['retrain_window'][0]

    return eval_df

def evaluate_model(
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
        model_name (string): name of the model.
        dataset_name (string): name of the dataset (e.g., 'm5', 'm4').
        frequency (string): frequency of the data (e.g., 'daily', 'weekly').
        metrics (list): list of evaluation metrics.
        train_df (pd.DataFrame, optional): training data in the Nixtla's format. 
        Defaults to None.
        group_columns (list, optional): list of columns to group by. Defaults to None.
        drop_columns (list, optional): list of columns to drop. Defaults to None.
        aggregate_function (callable, optional): function to use for aggregation.
        ext (str, optional): extension of the data files. Defaults to '.parquet'.
    
    Returns:
        pd.DataFrame: dataframe with evaluation results for each metric.
    """

    print(f'Evaluating {model_name} model on {dataset_name} dataset...')

    files = os.listdir(f'results/{dataset_name}/')
    file_names = list(
        filter(
            lambda x:
                model_name in x 
                    and frequency in x
                    and analysis_type in x 
                    and ext in x, 
                files
        )
    )
    file_names = [s.replace(ext, "") for s in file_names]

    if analysis_type == 'outsample':
        print(f'Evaluating outsample results...')
    elif analysis_type == 'time':
        print(f'Evaluating time results...')
    else:
        raise ValueError(f'Invalid analysis type {analysis_type}.')
        
    res_df = pd.DataFrame()
    for f in file_names:
        df_tmp = load_data(
            path = f'results/{dataset_name}/',
            name_list = [f],
            ext = ext
        )
        if analysis_type == 'outsample':
            df_tmp = evaluate_point_forecasts(
                df_tmp,
                metrics = metrics,
                train_df = train_df
            )

        if group_columns is not None:
            df_tmp = aggregate_data(
                df_tmp, 
                group_columns, 
                drop_columns,
                aggregate_function
            )

        res_df = pd.concat([res_df, df_tmp], axis = 0)
            
    return res_df

