
import numpy as np
import pandas as pd
import pandas_flavor as pf

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

@pf.register_dataframe_method
def remove_series(data, min_series_length):

    """Function to remove series from the data based on their length.

    Args:
        data (pd.DataFrame): Input dataframe in Nixtla's format.
        min_series_length (int): Minimum length of series to be kept.

    Returns:
        pd.DataFrame: dataframe with series removed.
    """

    print('Removing series...')
    series_length = data.groupby('unique_id')['y'].count()
    remove_ids = series_length[series_length < min_series_length].index.tolist()
    res_df = data[~data['unique_id'].isin(remove_ids)]

    n_series = len(series_length)
    n_series_to_remove = len(remove_ids)
    p_series_to_remove = n_series_to_remove / n_series * 100
    print(f'Removed {n_series_to_remove} series out of {n_series} ({p_series_to_remove:.1f}%)')

    return res_df

@pf.register_dataframe_method
def get_static_features(data, dataset_name):

    """Function to add static features to the data.

    Args:
        data (pd.DataFrame): Input dataframe in Nixtla's format.
        dataset_name (string): Name of the dataset (e.g., 'm5', 'm4').
    
    Returns:
        pd.DataFrame: dataframe with static features added.
    """

    if dataset_name == 'm5':
        
        static_df = data['unique_id'].apply(lambda x: pd.Series(str(x).split("_")))
        
        static_df['item_id'] = static_df[0] + "_" + static_df[1] + "_" + static_df[2]
        static_df['item_id'] = static_df['item_id'].astype('category').cat.codes
        static_df['dept_id'] = static_df[0] + "_" + static_df[1]
        static_df['dept_id'] = static_df['dept_id'].astype('category').cat.codes
        static_df['cat_id'] = static_df[0]
        static_df['cat_id'] = static_df['cat_id'].astype('category').cat.codes
        static_df['store_id'] = static_df[3] + "_" + static_df[4]
        static_df['store_id'] = static_df['store_id'].astype('category').cat.codes
        static_df['state_id'] = static_df[3]
        static_df['state_id'] = static_df['state_id'].astype('category').cat.codes
        
        static_df = static_df.drop(columns = [0, 1, 2, 3, 4], axis = 1)
        res_df = pd.concat([data, static_df], axis = 1)

    else:
        raise(f'Unknown dataset {dataset_name}')

    return res_df
