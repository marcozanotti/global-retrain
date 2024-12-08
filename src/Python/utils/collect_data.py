
import os
import numpy as np
import pandas as pd
import pandas_flavor as pf
import pyarrow.parquet as pq

import logging
module_logger = logging.getLogger('collect_data')

# from datasetsforecast.m3 import M3
# from datasetsforecast.m4 import M4
# M3.download('data')
# M4.download('data') 

def create_file_path(path_list):
    """Function to create a file path from a list of directories.

    Args:
        path_list (list): List of directories to be joined.
    
    Returns:
        string: The file path.
    """

    if len(path_list) == 1:
        path = path_list[0]
    else:
        path = '/'.join(str(x) for x in path_list) + '/'

    return path

def create_file_name(name_list, ext = None):

    """Function to create a file name from a list of names.

    Args:
        name_list (list): List of names to be used to create the file name.
        ext (str, optional): Extension of the file. Defaults to '.parquet'.
    
    Returns:
        string: The file name.
    """
    if len(name_list) == 1:
        file_name = name_list[0]
    else:
        file_name = '_'.join(str(x) for x in name_list)
    
    if ext is not None:
        file_name += ext

    return file_name

def get_file_name(path_list, name_list = None, ext = '.parquet', remove_ext = True):

    """Function to get file names from a given path.

    Args:
        path_list (list): List of directories to be joined.
        name_list (list): List of names to be used to filter the files. Defaults to None.
        ext (str, optional): Extension of the files. Defaults to '.parquet'.
        remove_ext (bool, optional): Whether to remove the extension from the file names. 
        Defaults to True.
    
    Returns:
        list: List of file names.
    """

    path = create_file_path(path_list)
    file_names = os.listdir(path)
    if name_list is not None:
        for n in name_list:
            file_names = list(filter(lambda x: str(n) in x, file_names))
    if remove_ext:
        file_names = [s.replace(ext, "") for s in file_names]

    return file_names

def combine_and_save_files(path_list_to_read, path_list_to_write, name_list, ext = '.parquet'):

    """Function to combine and save multiple files into a single file.

    Args:
        path_list_to_read (list): List of directories to be joined for reading.
        path_list_to_write (list): List of directories to be joined for writing.
        name_list (list): List of names to be used to filter the files.
        ext (str, optional): Extension of the files. Defaults to '.parquet'.
    """

    module_logger.info('Combining and saving files...')
    path_to_read = create_file_path(path_list_to_read)
    path_to_write = create_file_path(path_list_to_write)
    files = get_file_name(
        path_list = [path_to_read], name_list = name_list,
        ext = ext, remove_ext = False
    )
    write_file_name = create_file_name(name_list = name_list, ext = ext) 

    if ext == '.parquet':
        if files:
            schema = pq.ParquetFile(path_to_read + files[0]).schema_arrow
            with pq.ParquetWriter(path_to_write + write_file_name, schema = schema) as writer:
                for f in files:
                    writer.write_table(pq.read_table(path_to_read + f, schema = schema))
    else:
        raise ValueError(f'Unsupported extension {ext}')

    return

def remove_file(path_list, name_list, ext = '.parquet'):

    """Function to remove files from a given path.

    Args:
        path_list (list): List of directories to be joined.
        name_list (list): List of names to be used to filter the files.
        ext (str, optional): Extension of the files. Defaults to '.parquet'.
    """

    module_logger.info('Removing files...')
    path_to_remove = create_file_path(path_list)
    files = get_file_name(
        path_list = [path_to_remove], name_list = name_list,
        ext = ext, remove_ext = False
    )
    for f in files:
        os.remove(f'{path_to_remove}{f}')

    return

@pf.register_dataframe_method
def save_data(data, path_list, name_list, ext = '.parquet'):

    """Function to save dataframes.

    Args:
        data (pd.DataFrame): Data to be saved.
        path_list (list): List of directories to be joined.
        name_list (list): List of names to be used to create the file name.
        ext (string, optional): File extension (default is '.parquet').
    """
    
    path = create_file_path(path_list)
    if not os.path.exists(path):
        os.makedirs(path)

    file_name = create_file_name(name_list)
    module_logger.info(f'Saving {file_name} dataset...')

    if ext == '.parquet':
        data.to_parquet(f'{path}{file_name}{ext}')
    elif ext == '.csv':
        data.to_csv(f'{path}{file_name}{ext}')
    else:
        raise(f'Unsupported file extension {ext}. Only .parquet and .csv are allowed')

def load_data(path_list, name_list, ext = '.parquet'):

    """Function to load the data.

    Args:
        path_list (list): List of directories to be joined.
        name_list (list): List of names to be used to create the file name.
        ext (string, optional): File extension (default is '.parquet').

    Returns:
        pd.Dataframe: The data.
    """

    path = create_file_path(path_list)
    file_name = create_file_name(name_list)
    module_logger.info(f'Loading {file_name} dataset...')

    if ext == '.parquet':
        res_df = pd.read_parquet(f'{path}{file_name}{ext}')
    elif ext == '.csv':
        res_df = pd.read_csv(f'{path}{file_name}{ext}')
    else:
        raise(f'Unsupported file extension {ext}. Only .parquet and .csv are allowed')

    return res_df

def download_data(dataset_name, frequency, save = True, ext = '.parquet'):

    """Function to download and save different time series datasets.

    Args:
        dataset_name (string): Name of the dataset (e.g., 'm5', 'm4').
        frequency (string, optional): The frequency of the data (e.g., 'daily', 'weekly'). 
        save (bool, optional): Whether to save data or not. Train and test detasets
        are saved in data/_dataset_name/ as .parquet files. Defaults to True.
        ext (string, optional): File extension (default is '.parquet').

    Returns:
        pd.DataFrame: training and test dataframes.
    """

    if dataset_name == 'm5':
        module_logger.info(f'Downloading {dataset_name} train dataset...')
        train_df = pd.read_parquet('https://m5-benchmarks.s3.amazonaws.com/data/train/target.parquet') \
            .rename(columns = {'item_id': 'unique_id', 'timestamp': 'ds', 'demand': 'y'})
        module_logger.info(f'Downloading {dataset_name} test dataset...')
        test_df = pd.read_parquet('https://m5-benchmarks.s3.amazonaws.com/data/test/target.parquet') \
            .rename(columns = {'item_id': 'unique_id', 'timestamp': 'ds', 'demand': 'y'})
    else:
        raise(f'Unknown dataset {dataset_name}')

    if save:
        save_data(
            train_df, path = f'data/{dataset_name}/', 
            name_list = [dataset_name, frequency, 'train'],
            ext = ext
        )
        save_data(
            test_df, path = f'data/{dataset_name}/', 
            name_list = [dataset_name, frequency, 'test'],
            ext = ext
        )

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

    module_logger.info('Combining train and test data...')
    combined_df = pd.concat([train_df, test_df], axis = 0, ignore_index = True)
    combined_df.sort_values(by = ['unique_id', 'ds'], inplace = True)
    combined_df.reset_index(drop = True, inplace = True)

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

    module_logger.info(f'Removing series shorter than {min_series_length} observaions...')
    series_length = data.groupby('unique_id')['y'].count()
    remove_ids = series_length[series_length < min_series_length].index.tolist()
    res_df = data[~data['unique_id'].isin(remove_ids)]

    n_series = len(series_length)
    n_series_to_remove = len(remove_ids)
    p_series_to_remove = n_series_to_remove / n_series * 100
    tot_series = n_series - n_series_to_remove
    module_logger.info(f'Removed {n_series_to_remove} series out of {n_series} ({p_series_to_remove:.1f}%).')
    module_logger.info(f'The final dataset contains {tot_series} series.')

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

    module_logger.info(f'Extracting static features from {dataset_name} dataset...')

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

    module_logger.info(f'Sampling {samples} series from data...')
    ids = data['unique_id'].unique()
    sample_ids = np.random.choice(ids, size = samples, replace = False)
    res_df = data[data['unique_id'] \
        .isin(sample_ids)] \
        .reset_index(drop = True)

    return res_df

def get_xregs_data(path_list, name_list, dataset_name, ext = '.parquet'):

    """Function to get external regressors (xregs) for the specified dataset.

    Args:
        path_list (list): List of directories to be joined.
        name_list (list): List of file names for external regressors.
        dataset_name (string): Name of the dataset (e.g., 'm5', 'm4').
        ext (string, optional): Extension of the external regressors files. Defaults to '.parquet'.
    
    Returns:
        pd.DataFrame: dataframe with external regressors.
    """

    module_logger.info(f'Extracting external regressors for {dataset_name} dataset...')
    
    if dataset_name == 'm5':

        ext = '.csv'
        xreg_df = load_data(
            path_list = path_list, 
            name_list = name_list, 
            ext = ext
        )
        xreg_df['event'] = np.where(xreg_df['event_name_1'].isna(), 0, 1)
        xreg_df['event'] = xreg_df['event'].astype(int)
        xreg_df['date'] = pd.to_datetime(xreg_df['date'])
        xreg_df.rename(columns = {'date': 'ds'}, inplace = True)
        xreg_df = xreg_df[['ds', 'event']]

    else:
        raise ValueError(f'Unknown dataset {dataset_name}.')

    return xreg_df

def prepare_data(dataset_name, frequency, static_features = True, xregs = True, save = True, ext = '.parquet'):

    """Function to prepare saved datasets.

    Args:
        dataset_name (string): Name of the dataset (e.g., 'm5', 'm4').
        frequency (string, optional): The frequency of the data (e.g., 'daily', 'weekly'). 
        static_features (bool, optional): Whether to include static features. Defaults to True.
        xregs (bool, optional): Whether to include external regressors. Defaults to True.
        save (bool, optional): Whether to save the processed dataset. Defaults to False.
        ext (string, optional): Extension of the saved files. Defaults to '.parquet'.

    Returns:
        pd.DataFrame: full dataframe.
    """

    train_df = load_data(
        path_list = ['data', dataset_name], 
        name_list = [dataset_name, frequency, 'train'], 
        ext = ext
    )
    train_df['ds'] = pd.to_datetime(train_df['ds'])
    train_df['unique_id'] = train_df['unique_id'].astype(str)

    test_df = load_data(
        path_list = ['data', dataset_name], 
        name_list = [dataset_name, frequency, 'test'],
        ext = ext
    )
    test_df['ds'] = pd.to_datetime(test_df['ds'])
    test_df['unique_id'] = test_df['unique_id'].astype(str)

    res_df = combine_train_test(train_df, test_df)

    if static_features:
        res_df = get_static_features(res_df, dataset_name)

    if xregs:
        xreg_df = get_xregs_data(
            path_list = ['data', dataset_name],  
            name_list = [dataset_name, 'xregs'], 
            dataset_name = dataset_name,
            ext = ext
        )
        res_df = pd.merge(res_df, xreg_df, how = 'left', on = 'ds')

    if save:
        save_data(
            data = res_df, 
            path_list = ['data', dataset_name], 
            name_list = [dataset_name, frequency, 'prep'],
            ext = ext
        )

    return res_df

def get_data(path_list, name_list, ext = '.parquet', min_series_length = None, samples = None):

    """Function to get the data.

    Args:
        path_list (list): List of directories to be joined.
        name_list (list): List of names to be used to create the file name.
        ext (string, optional): File extension (default is '.parquet').
        min_series_length (int, optional): Minimum length of series to be included. 
        Defaults to None.
        samples (int, optional): Number of samples to be included. Defaults to None.

    Returns:
        pd.Dataframe: The data.
    """

    res_df = load_data(path_list, name_list, ext)

    if min_series_length is not None:
        res_df = remove_series(res_df, min_series_length)

    if samples is not None:
        res_df = sampling_data(res_df, samples)        

    return res_df
