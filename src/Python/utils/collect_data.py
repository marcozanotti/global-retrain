
import pandas as pd
import numpy as np

# from datasetsforecast.m3 import M3
# from datasetsforecast.m4 import M4
# M3.download('data')
# M4.download('data') 

def download_dataset(dataset_name, frequency = None, save = True):
    """Function to download and save different time series datasets.

    Args:
        dataset_name (string): Name of the dataset (e.g., 'm5', 'm4').
        frequency (string, optional): The frequency of the data (e.g., 'daily', 'weekly'). 
        Defaults to None.
        save (bool, optional): Whether to save data or not. Train and test detasets
        are saved in data/_dataset_name/ as .parquet files. Defaults to True.

    Returns:
        pd.DataFrame: training and test dataframes.
    """

    print(f'Downloading {dataset_name} train and test data...')
    if dataset_name == 'm5':
        frequency = 'daily'
        train_df = pd.read_parquet('https://m5-benchmarks.s3.amazonaws.com/data/train/target.parquet') \
            .rename(columns = {'item_id': 'unique_id', 'timestamp': 'ds', 'demand': 'y'})
        test_df = pd.read_parquet('https://m5-benchmarks.s3.amazonaws.com/data/test/target.parquet') \
            .rename(columns = {'item_id': 'unique_id', 'timestamp': 'ds', 'demand': 'y'})
    else:
        raise(f'Unknown dataset {dataset_name}')

    if save:
        train_name = f'data/{dataset_name}/train_{frequency}.parquet'
        test_name = f'data/{dataset_name}/test_{frequency}.parquet'
        train_df.to_parquet(train_name)
        test_df.to_parquet(test_name)

    return train_df, test_df


def get_dataset(dataset_name, frequency = None, samples = None):
    """Function to load saved datasets.

    Args:
        dataset_name (string): Name of the dataset (e.g., 'm5', 'm4').
        frequency (string, optional): The frequency of the data (e.g., 'daily', 'weekly'). 
        Defaults to None.

    Returns:
        pd.DataFrame: training and test dataframes.
    """

    if dataset_name == 'm5':
        frequency = 'daily'

    train_name = f'data/{dataset_name}/train_{frequency}.parquet'
    test_name = f'data/{dataset_name}/test_{frequency}.parquet'

    print(f'Reading {dataset_name} train data...')
    train_df = pd.read_parquet(train_name)
    train_df['ds'] = pd.to_datetime(train_df['ds'])
    train_df['unique_id'] = train_df['unique_id'].astype(str)

    print(f'Reading {dataset_name} test data...')
    test_df = pd.read_parquet(test_name)
    test_df['ds'] = pd.to_datetime(test_df['ds'])
    test_df['unique_id'] = test_df['unique_id'].astype(str)

    if samples is not None:
        print(f'Sampling {samples} series from train and test data...')
        uids = train_df['unique_id'].unique()
        # np.random.seed(0)
        sample_uids = np.random.choice(uids, size = samples, replace = False)
        train_df = train_df[train_df['unique_id'] \
            .isin(sample_uids)] \
            .reset_index(drop = True)
        test_df = test_df[test_df['unique_id'] \
            .isin(sample_uids)] \
            .reset_index(drop = True)

    return train_df, test_df


