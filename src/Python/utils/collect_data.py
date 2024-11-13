
import os
import numpy as np
import pandas as pd
import pandas_flavor as pf

# from datasetsforecast.m3 import M3
# from datasetsforecast.m4 import M4
# M3.download('data')
# M4.download('data') 

def download_data(dataset_name, frequency, save = True):

    """Function to download and save different time series datasets.

    Args:
        dataset_name (string): Name of the dataset (e.g., 'm5', 'm4').
        frequency (string, optional): The frequency of the data (e.g., 'daily', 'weekly'). 
        save (bool, optional): Whether to save data or not. Train and test detasets
        are saved in data/_dataset_name/ as .parquet files. Defaults to True.

    Returns:
        pd.DataFrame: training and test dataframes.
    """

    print(f'Downloading {dataset_name} train and test data...')
    if dataset_name == 'm5':
        train_df = pd.read_parquet('https://m5-benchmarks.s3.amazonaws.com/data/train/target.parquet') \
            .rename(columns = {'item_id': 'unique_id', 'timestamp': 'ds', 'demand': 'y'})
        test_df = pd.read_parquet('https://m5-benchmarks.s3.amazonaws.com/data/test/target.parquet') \
            .rename(columns = {'item_id': 'unique_id', 'timestamp': 'ds', 'demand': 'y'})
    else:
        raise(f'Unknown dataset {dataset_name}')

    if save:
        print(f'Saving {dataset_name} train and test data...')
        train_name = f'data/{dataset_name}/train_{frequency}.parquet'
        test_name = f'data/{dataset_name}/test_{frequency}.parquet'
        train_df.to_parquet(train_name)
        test_df.to_parquet(test_name)

    return train_df, test_df

@pf.register_dataframe_method
def combine_train_test(train_df, test_df):

    """Function to combine train and test dataframes.

    Args:
        train_df (pd.DataFrame): training data in the Nixtla's format.
        test_df (pd.DataFrame): test data in the Nixtla format.

    Returns:
        pd.DataFrame: combined train and test dataframes.
    """

    print('Combining train and test data...')
    combined_df = pd.concat([train_df, test_df], axis = 0, ignore_index = True)
    combined_df = combined_df.sort_values(by = ['unique_id', 'ds']).reset_index(drop = True)

    return combined_df

@pf.register_dataframe_method
def remove_series(data, min_series_length):

    """Function to remove series from the data based on their length.

    Args:
        data (pd.DataFrame): Input dataframe in Nixtla's format.
        min_series_length (int): Minimum length of series to be kept.

    Returns:
        pd.DataFrame: dataframe with series removed.
    """

    print(f'Removing series shorter than {min_series_length}...')
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

    print(f'Extracting static features from {dataset_name} dataset...')

    # get splitted unique ids
    static_df = data['unique_id'] \
        .drop_duplicates() \
        .apply(lambda x: pd.Series(str(x).split("_"))) \
        .reset_index(drop = True)
    
    if dataset_name == 'm5':

        static_df['unique_id'] = static_df[0] + "_" \
            + static_df[1] + "_" \
            + static_df[2] + "_" \
            + static_df[3] + "_" \
            + static_df[4]
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

    else:
        raise(f'Unknown dataset {dataset_name}')

    res_df = pd.merge(data, static_df, how = 'left', on = 'unique_id')

    return res_df

@pf.register_dataframe_method
def sampling_data(data, samples = 1000):

    """Function to sample dataframes.

    Args:
        data (pd.DataFrame): Input dataframe in Nixtla's format.
        samples (int, optional): Number of samples to be taken. Defaults to 1000.
    
    Returns:
        pd.DataFrame: sampled dataframe.
    """

    print(f'Sampling {samples} series from data...')
    ids = data['unique_id'].unique()
    sample_ids = np.random.choice(ids, size = samples, replace = False)
    res_df = data[data['unique_id'] \
        .isin(sample_ids)] \
        .reset_index(drop = True)

    return res_df

def prepare_data(dataset_name, frequency, static_features = True, save = True):

    """Function to prepare saved datasets.

    Args:
        dataset_name (string): Name of the dataset (e.g., 'm5', 'm4').
        frequency (string, optional): The frequency of the data (e.g., 'daily', 'weekly'). 
        static_features (bool, optional): Whether to include static features. Defaults to True.
        save (bool, optional): Whether to save the processed dataset. Defaults to False.

    Returns:
        pd.DataFrame: full dataframe.
    """

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

    res_df = combine_train_test(train_df, test_df)

    if static_features:
        res_df = get_static_features(res_df, dataset_name)

    if save:
        print('Saving processed dataset...')
        res_df.to_parquet(f'data/{dataset_name}/{dataset_name}_{frequency}_prep.parquet')

    return res_df

def get_data(file, min_series_length = None, samples = None):

    """Function to load the data.

    Args:
        file (string): Path to the dataset file.
        min_series_length (int, optional): Minimum length of series to be included. 
        Defaults to None.
        samples (int, optional): Number of samples to be included. Defaults to None.

    Returns:
        pd.Dataframe: The data.
    """

    file_name = os.path.split(file)[1].removesuffix('.parquet')

    print(f'Reading {file_name} dataset...')
    res_df = pd.read_parquet(file)

    if min_series_length is not None:
        res_df = remove_series(res_df, min_series_length)

    if samples is not None:
        res_df = sampling_data(res_df, samples)        

    return res_df






