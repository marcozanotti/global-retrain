import sys
sys.path.insert(0, 'src/Python/utils')
import numpy as np
import pandas as pd
import pandas_flavor as pf
from utilities import save_data, load_data, get_frequency, get_dataset_frequency
from pytimetk import get_timeseries_signature

import logging
module_logger = logging.getLogger('collect_data')


def download_data(dataset_name, save = True, ext = '.parquet'):

    """Function to download and save different time series datasets.

    Args:
        dataset_name (string): Name of the dataset (e.g., 'm5', 'm4').
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
    
    elif dataset_name == 'vn1':

        module_logger.info(f'Downloading {dataset_name} train dataset...')
        # Phase 0 datset
        train_df0 = pd.read_csv('data/vn1/Phase 0 - Sales.csv')
        train_df0['unique_id'] = train_df0['Client'].astype(str) + '_' \
            + train_df0['Warehouse'].astype(str) + '_' \
            + train_df0['Product'].astype(str)
        train_df0.drop(columns = ['Client', 'Warehouse', 'Product'], axis = 1, inplace = True)
        # Phase 1 dataset
        train_df1 = pd.read_csv('data/vn1/Phase 1 - Sales.csv')
        train_df1['unique_id'] = train_df1['Client'].astype(str) + '_' \
            + train_df1['Warehouse'].astype(str) + '_' \
            + train_df1['Product'].astype(str)
        train_df1.drop(columns = ['Client', 'Warehouse', 'Product'], axis = 1, inplace = True)
        train_df = train_df0 \
            .merge(train_df1, how = 'left', on = 'unique_id') \
            .melt(id_vars = 'unique_id', var_name = 'ds', value_name  = 'y')

        module_logger.info(f'Downloading {dataset_name} test dataset...')
        test_df = pd.read_csv('data/vn1/Phase 2 - Sales.csv')
        test_df['unique_id'] = test_df['Client'].astype(str) + '_' \
            + test_df['Warehouse'].astype(str) + '_' \
            + test_df['Product'].astype(str)
        test_df.drop(columns = ['Client', 'Warehouse', 'Product'], axis = 1, inplace = True)
        test_df = test_df.melt(id_vars = 'unique_id', var_name = 'ds', value_name  = 'y')

    elif dataset_name == 'm4':

        # if other datasets are needed (like Hourly, Weekly, etc), add all here
        # creating a single train and test df
        
        module_logger.info(f'Downloading {dataset_name} train dataset...')      
        train_df = pd.read_csv('data/m4/Daily-train.csv')
        train_df.columns = ['unique_id'] + list(range(1, train_df.shape[1]))
        train_df = pd.melt(train_df, id_vars = ['unique_id'], var_name = 'ds', value_name = 'y')
        train_df = train_df.dropna()
        train_df['ds'] = train_df['ds'].astype('int')
        train_df = train_df.sort_values(['unique_id', 'ds']).reset_index(drop = True)

        module_logger.info(f'Downloading {dataset_name} test dataset...')
        test_df = pd.read_csv('data/m4/Daily-test.csv')
        test_df.columns = ['unique_id'] + list(range(1, test_df.shape[1]))
        test_df = pd.melt(test_df, id_vars = ['unique_id'], var_name = 'ds', value_name = 'y')
        test_df = test_df.dropna()
        test_df['ds'] = test_df['ds'].astype('int')
        len_train = train_df.groupby('unique_id').agg({'ds': 'max'}).reset_index()
        len_train.columns = ['unique_id', 'len_serie']
        test_df = test_df.merge(len_train, on = ['unique_id'])
        test_df['ds'] = test_df['ds'] + test_df['len_serie']
        test_df.drop('len_serie', axis = 1, inplace = True)
        test_df = test_df.sort_values(['unique_id', 'ds']).reset_index(drop = True)
    
    elif dataset_name == 'hapag_region':

        module_logger.info(f'Downloading {dataset_name} train dataset...')
        train_df = pd.read_csv('data/hapag_region/hapag_region.csv', header = 1, sep = ';', quotechar = '"')
        train_df = train_df[:-1] # drop last row with ###EndofFile### value
        train_df['unique_id'] = train_df['Name'].str.replace(';', '_')
        train_df = train_df[['unique_id'] + [col for col in train_df.columns if 'Date:' in col]]
        train_df.columns = [col.replace('Date:', '') for col in train_df.columns]
        train_df = train_df.melt(id_vars = ['unique_id'], var_name = 'ds', value_name = 'y')
        train_df['ds'] = pd.to_datetime(train_df['ds'], format = '%Y-%m-%d')
        train_df = train_df.sort_values(['unique_id', 'ds']).reset_index(drop = True)

        # split into train and test with last 5 years as test
        test_df = train_df[train_df['ds'] >= '2020-01-01'].copy()
        train_df = train_df[train_df['ds'] < '2020-01-01'].copy()

    else:

        raise(f'Unknown dataset {dataset_name}')

    if save:
        save_data(
            data = train_df, 
            path_list = ['data', dataset_name], 
            name_list = [dataset_name, 'train'],
            ext = ext
        )
        save_data(
            data = test_df, 
            path_list = ['data', dataset_name], 
            name_list = [dataset_name, 'test'],
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
def aggregate_data_by_frequency(data, dataset_name, frequency, drop_firstlast = False):

    """Function to aggregate dataframes by frequency.

    Args:
        data (pd.DataFrame): Input dataframe in Nixtla's format.
        dataset_name (string): Name of the dataset (e.g., 'm5', 'm4').
        frequency (string): The frequency of the data (e.g., 'daily', 'weekly').
    
    Returns:
        pd.DataFrame: dataframe aggregated by frequency.
    """

    def drop_first_last(df):
        return df.iloc[1:-1]

    freq_list = get_frequency(frequency)
    freq = freq_list[0]
    freq_df = get_frequency(get_dataset_frequency(dataset_name))[0]

    if freq != freq_df:
        module_logger.info(f'Aggregating {dataset_name} dataset into {frequency} frequency...')

        if 'unique_id' in data.columns:
            data_agg = data \
                .set_index('ds') \
                .groupby(['unique_id', pd.Grouper(freq = freq_list[2])]) \
                .sum() \
                .reset_index()

            if drop_firstlast:
                data_agg = data_agg \
                    .groupby('unique_id') \
                    .apply(drop_first_last, include_groups = False) \
                    .reset_index() \
                    .drop(columns = ['level_1'], axis = 1)

        else:
            data_agg = data \
                .set_index('ds') \
                .groupby([pd.Grouper(freq = freq_list[2])]) \
                .sum() \
                .reset_index()

            if drop_firstlast:
                data_agg = data_agg \
                    .apply(drop_first_last)

    else:
        data_agg = data      

    return data_agg

@pf.register_dataframe_method
def remove_series(data, min_series_length):

    """Function to remove series from the data based on their length.

    Args:
        data (pd.DataFrame): Input dataframe in Nixtla's format.
        min_series_length (int): Minimum length of series to be kept.    

    Returns:
        pd.DataFrame: dataframe with series removed.
    """

    series_length = data.groupby('unique_id')['y'].count()

    module_logger.info(f'Removing series shorter than {min_series_length} observations...')
    remove_ids = series_length[series_length < min_series_length].index.tolist()
    res_df = data[~data['unique_id'].isin(remove_ids)]

    n_series = len(data['unique_id'].unique())
    n_series_final = len(res_df['unique_id'].unique())
    n_series_to_remove = n_series - n_series_final
    p_series_to_remove = n_series_to_remove / n_series * 100
    module_logger.info(f'Removed {n_series_to_remove} series out of {n_series} ({p_series_to_remove:.1f}%).')
    module_logger.info(f'The final dataset contains {n_series_final} series.')

    return res_df

@pf.register_dataframe_method
def filter_series(data, max_series_length):

    """Function to remove series from the data based on their length.

    Args:
        data (pd.DataFrame): Input dataframe in Nixtla's format.
        max_series_length (int): Maximum length of series to be kept.    

    Returns:
        pd.DataFrame: dataframe with series removed.
    """

    module_logger.info(f'Filtering series longer than {max_series_length} observations...')
    res_df = data \
        .sort_values(['unique_id', 'ds']) \
        .groupby('unique_id') \
        .tail(n = max_series_length) \
        .reset_index(drop = True)

    n_obs = len(data)
    n_obs_final = len(res_df)
    n_obs_to_remove = n_obs - n_obs_final
    p_obs_to_remove = n_obs_to_remove / n_obs * 100
    module_logger.info(f'Removed {n_obs_to_remove} observations out of {n_obs} ({p_obs_to_remove:.1f}%).')
    module_logger.info(f'The final dataset contains {n_obs_final} observations.')

    return res_df

@pf.register_dataframe_method
def get_static_features(data, dataset_name):

    """Function to add static features to the data.

    Args:
        data (pd.DataFrame): Input dataframe in Nixtla's format.
        dataset_name (string): Name of the dataset (e.g., 'm5', 'm4').
        normalize (bool, optional): Whether to normalize the static features. Defaults to True.
    
    Returns:
        pd.DataFrame: dataframe with static features added.
    """

    module_logger.info(f'Extracting static features from {dataset_name} dataset...')

    def create_mapping_df(static_df, original_column_names, code_column_names, variable_names = None):

        if variable_names is None:
            variable_names = code_column_names

        mapping_dfs = pd.DataFrame()

        for original_col, code_col, var_name in zip(original_column_names, code_column_names, variable_names):
            mapping_df = static_df[[original_col, code_col]].drop_duplicates().sort_values(code_col)
            mapping_df.columns = ['value', 'code']
            mapping_df['variable'] = var_name
            mapping_dfs = pd.concat([mapping_dfs, mapping_df], axis = 0)
        
        mapping_dfs = mapping_dfs.reset_index(drop = True)

        return mapping_dfs

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
        mapping_dfs = create_mapping_df(
            static_df, original_column_names = [0, 1, 2, 3, 4], 
            code_column_names = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
        )
        static_df.drop(columns = [0, 1, 2, 3, 4], axis = 1, inplace = True)

    elif dataset_name == 'vn1':
        static_df['unique_id'] = static_df[0] + "_" + static_df[1] + "_" + static_df[2]
        static_df['client'] = static_df[0].astype('category').cat.codes
        static_df['warehouse'] = static_df[1].astype('category').cat.codes
        static_df['product'] = static_df[2].astype('category').cat.codes
        mapping_dfs = create_mapping_df(static_df, [0, 1, 2], ['client', 'warehouse', 'product'])
        static_df.drop(columns = [0, 1, 2], axis = 1, inplace = True)

    elif dataset_name == 'm4':
        static_df = pd.read_csv('data/m4/M4-info.csv')[['M4id', 'category']]
        static_df.columns = ['unique_id', 'category']
        static_df['category'] = static_df['category'].astype('category').cat.codes
        mapping_dfs = None

    elif dataset_name == 'hapag_region' or dataset_name == 'hapag':

        static_df['unique_id'] = static_df[0] + "_" \
            + static_df[1] + "_" \
            + static_df[2] + "_" \
            + static_df[3]
        static_df['geoscope'] = static_df[0].astype('category').cat.codes
        static_df['eqtype'] = static_df[1].astype('category').cat.codes
        static_df['georelated'] = static_df[2].astype('category').cat.codes
        static_df['balance'] = static_df[3].astype('category').cat.codes
        mapping_dfs = create_mapping_df(static_df, [0, 1, 2, 3], ['geoscope', 'eqtype', 'georelated', 'balance'])
        static_df.drop(columns = [0, 1, 2, 3], axis = 1, inplace = True)

    else:
        raise(f'Unknown dataset {dataset_name}')

    if mapping_dfs is not None:
        save_data(mapping_dfs, ['data', dataset_name], [dataset_name, 'static_features_mapping'], ext = '.csv')

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

def get_xregs_data(path_list, name_list, dataset_name, frequency, ext = '.parquet'):

    """Function to get external regressors (xregs) for the specified dataset.

    Args:
        path_list (list): List of directories to be joined.
        name_list (list): List of file names for external regressors.
        dataset_name (string): Name of the dataset (e.g., 'm5', 'm4').
        frequency (string): The frequency of the data (e.g., 'daily', 'weekly').
        ext (string, optional): Extension of the external regressors files. Defaults to '.parquet'.
    
    Returns:
        pd.DataFrame: dataframe with external regressors.
    """

    module_logger.info(f'Extracting external regressors for {dataset_name} dataset...')
    
    if dataset_name == 'm5':

        ext = '.csv'
        xregs_df = load_data(
            path_list = path_list, 
            name_list = name_list, 
            ext = ext
        )
        xregs_df['event'] = np.where(xregs_df['event_name_1'].isna(), 0, 1)
        xregs_df['event'] = xregs_df['event'].astype(int)
        xregs_df['date'] = pd.to_datetime(xregs_df['date'])
        xregs_df.rename(columns = {'date': 'ds'}, inplace = True)
        xregs_df = xregs_df[['ds', 'event']]
        xregs_df = aggregate_data_by_frequency(xregs_df, dataset_name, frequency, drop_firstlast = False)
        xregs_df['event'] = np.where(xregs_df['event'] == 0, 0, 1)

    elif dataset_name == 'vn1':

        raise ValueError(f'Xregs are not available for dataset {dataset_name}.')

    elif dataset_name == 'm4':
        
        raise ValueError(f'Xregs are not available for dataset {dataset_name}.')
    
    elif dataset_name == 'hapag_region' or dataset_name == 'hapag':

        ext = '.csv'
        xregs_df = load_data(
            path_list = path_list, 
            name_list = name_list, 
            kwargs = {'header': 1, 'sep': ';', 'quotechar': '"'},
            ext = ext            
        )
        xregs_df = xregs_df[:-1]
        xregs_df = xregs_df[['Name'] + [col for col in xregs_df.columns if 'Date:' in col]]
        xregs_df = xregs_df.sort_values(['Name']).reset_index(drop = True)
        xregs_df['xregs'] = ['xreg' + str(i) for i in range(1, len(xregs_df) + 1)]
        xregs_df['xregs'] = xregs_df['xregs'].apply(lambda x: x if int(x[4:]) >= 10 else 'xreg0' + x[4:])

        # create a new dataframe with Name and xregs columns to keep the mapping between them
        xregs_mapping_df = xregs_df[['Name', 'xregs']].copy()
        xregs_mapping_df['Name'] = xregs_mapping_df['Name'].str.lstrip('_')
        xregs_mapping_df['Name'] = xregs_mapping_df['Name'].str.lstrip('_')
        
        xregs_df = xregs_df.drop(columns = ['Name'])
        xregs_df.columns = [col.replace('Date:', '') for col in xregs_df.columns]
        xregs_df = xregs_df.melt(id_vars = ['xregs'], var_name = 'ds', value_name = 'value')
        xregs_df = xregs_df.pivot(index = 'ds', columns = 'xregs', values = 'value').reset_index()
        xregs_df['ds'] = pd.to_datetime(xregs_df['ds'], format = '%Y-%m-%d')
        xregs_df = xregs_df.sort_values(['ds']).reset_index(drop = True)

        # save xregs mapping dataframe
        save_data(
            data = xregs_mapping_df, 
            path_list = path_list, 
            name_list = [dataset_name, 'xregs_mapping'],
            ext = '.csv'
        )

    else:
        raise ValueError(f'Unknown dataset {dataset_name}.')

    return xregs_df

def create_time_trend(ds, trend_type='1'):

    """
    Create one or multiple time trends from a single series/column of dates (or index values).

    Args:
        ds (pd.Series or array-like): sequence of dates or ordinal values.
        trend_type (str or list): single trend type or list of trend types.
            Supported values: '1', '2', '3', ..., 'exponential', 'exp', 'log', 'ln', 'sqrt'.

    Returns:
        pd.DataFrame: dataframe with columns ['ds', trend_...].
    """

    # remove duplicates from ds and sort it
    ds = pd.Series(ds).drop_duplicates().reset_index(drop=True)
    tmp = pd.DataFrame({'ds': ds})
    tmp = tmp.sort_values('ds').reset_index(drop=True).copy()
    t = np.arange(len(tmp)).astype(float)

    if isinstance(trend_type, (str, int)):
        trend_types = [trend_type]
    else:
        trend_types = list(trend_type)

    def build_series(trend_key):
        key = str(trend_key).strip().lower()

        if key.isdigit():
            return t ** int(key), f'trend{key}'
        if key in ('exponential', 'exp'):
            t_scaled = t / (t.max() if t.max() > 0 else 1.0)
            return np.exp(t_scaled), 'trendexp'
        if key in ('log', 'ln'):
            return np.log(t + 1.0), 'trendlog'
        if key == 'sqrt':
            return np.sqrt(t), 'trendsqrt'

        raise ValueError(f'Unknown trend_type: {trend_key}')

    out = tmp[['ds']].copy()

    for trend_key in trend_types:
        series, col_name = build_series(trend_key)
        out[col_name] = series

    return out

def create_fourier_terms(ds, fourier):
    
    """
    Create Fourier terms for a given frequency and number of harmonics.

    Args:
        ds (pd.Series or array-like): sequence of dates or ordinal values.
        fourier (dict): Dictionary specifying the number of Fourier terms to add for each period. Keys are period strings (e.g., '7' for weekly seasonality) and values are the number of harmonics (K).
    Returns:
        pd.DataFrame: dataframe with columns ['ds', 'fourier1_sin', 'fourier1_cos', ..., 'fourierK_sin', 'fourierK_cos'].
    """

    ds = pd.Series(ds).drop_duplicates().reset_index(drop=True)
    tmp = pd.DataFrame({'ds': ds})
    tmp = tmp.sort_values('ds').reset_index(drop=True).copy()
    t = np.arange(len(tmp)).astype(float)

    for period, K in fourier.items():
        period = float(period)

        for k in range(1, K + 1):
            tmp[f'fourier{int(period)}sin{k}'] = np.sin(2 * np.pi * k * t / period)
            tmp[f'fourier{int(period)}cos{k}'] = np.cos(2 * np.pi * k * t / period)

    return tmp

def create_calendar_features(ds, feature_list = None):

    """Function to create calendar features from a date column.

    Args:
        ds (pd.Series or array-like): sequence of dates.
        feature_list (list, optional): List of calendar features to be created. Supported values include 'year', 'month', 'day', 'dayofweek', 'weekofyear', etc. If None, all available features will be created. Defaults to None.
    Returns:
        pd.DataFrame: dataframe with calendar features.
    """

    ds = pd.Series(ds).drop_duplicates().reset_index(drop=True)
    calendar_df = get_timeseries_signature(ds)
    calendar_df.columns = [col.replace('ds_', '', 1) for col in calendar_df.columns]
    if feature_list is not None:
        calendar_df = calendar_df[['ds'] + feature_list]

    return calendar_df

def create_one_hot_features(data, columns, drop_columns = True):

    """Function to create one-hot features from a categorical column.

    Args:
        data (pd.DataFrame): Input dataframe in Nixtla's format.
        columns (list): List of column names to be one-hot encoded.
        drop_columns (bool, optional): Whether to drop the original columns after encoding. Defaults to True.
    Returns:
        pd.DataFrame: dataframe with one-hot features.
    """

    for column in columns:
        one_hot_df = pd.get_dummies(data[column], prefix = column, drop_first=True, prefix_sep='_d')
        data = pd.concat([data, one_hot_df], axis = 1)
        if drop_columns:
            data.drop(columns = [column], axis = 1, inplace = True)

    return data

def normalize_features(data, columns, type='min-max'):

    """Function to normalize features.

    Args:
        data (pd.DataFrame): Input dataframe in Nixtla's format.
        columns (list): List of column names to be normalized.
        type (str, optional): Type of normalization to be applied. Supported values: 'min-max', 'z-score', 'robust'. Defaults to 'min-max'.
    
    Returns:
        pd.DataFrame: dataframe with normalized features.
    """

    for column in columns:
        column_new = f'{column}_n'
        if type == 'min-max':
            data[column_new] = (data[column] - data[column].min()) / (data[column].max() - data[column].min())
        elif type == 'z-score':
            data[column_new] = (data[column] - data[column].mean()) / data[column].std()
        elif type == 'robust':
            median = data[column].median()
            mad = (data[column] - median).abs().median()
            data[column_new] = (data[column] - median) / mad
        else:
            raise ValueError(f'Unknown normalization type: {type}')

    return data

def prepare_data(
    dataset_name, 
    frequency, 
    static_features = True, 
    xregs = True, 
    trend = None, 
    fourier = None,
    calendar_features = None,
    features_to_normalize = None,
    features_to_one_hot = None,
    save = True, 
    ext = '.parquet'
):

    """Function to prepare saved datasets.

    Args:
        dataset_name (string): Name of the dataset (e.g., 'm5', 'm4').
        frequency (string, optional): The frequency of the data (e.g., 'daily', 'weekly').
        static_features (bool, optional): Whether to include static features. Defaults to True.
        xregs (bool, optional): Whether to include external regressors. Defaults to True.
        trend (str or list, optional): Type(s) of time trend to add. Supported values: '1', '2', '3', ..., 'exponential', 'exp', 'log', 'ln', 'sqrt'. Defaults to None.
        fourier (dict, optional): Dictionary specifying the number of Fourier terms to add for each period. Keys are period strings (e.g., '7' for weekly seasonality) and values are the number of harmonics (K). Defaults to None.
        calendar_features (list, optional): List of calendar features to be created. Supported values include 'year', 'month', 'day', 'dayofweek', 'weekofyear', etc. If None, no calendar features will be created. Defaults to None.
        features_to_normalize (list, optional): List of feature names to be normalized. Defaults to None.
        features_to_one_hot (list, optional): List of feature names to be one-hot encoded. Defaults to None.
        save (bool, optional): Whether to save the processed dataset. Defaults to False.
        ext (string, optional): Extension of the saved files. Defaults to '.parquet'.

    Returns:
        pd.DataFrame: full dataframe.
    """

    train_df = load_data(
        path_list = ['data', dataset_name], 
        name_list = [dataset_name, 'train'], 
        ext = ext
    )
    train_df['unique_id'] = train_df['unique_id'].astype(str)

    test_df = load_data(
        path_list = ['data', dataset_name], 
        name_list = [dataset_name, 'test'],
        ext = ext
    )
    test_df['unique_id'] = test_df['unique_id'].astype(str)

    if dataset_name != 'm4':
        train_df['ds'] = pd.to_datetime(train_df['ds'])
        test_df['ds'] = pd.to_datetime(test_df['ds'])
    else:
        train_df['ds'] = train_df['ds'].astype('int')
        test_df['ds'] = test_df['ds'].astype('int')

    res_df = combine_train_test(train_df, test_df)
    del train_df, test_df

    if dataset_name == 'm4':
        res_df = res_df[res_df['unique_id'].str.contains(get_frequency(frequency)[0])]
    else:
        res_df = aggregate_data_by_frequency(res_df, dataset_name, frequency, drop_firstlast = True)
        
    if static_features:
        res_df = get_static_features(res_df, dataset_name)

    if xregs:
        xregs_df = get_xregs_data(
            path_list = ['data', dataset_name],  
            name_list = [dataset_name, 'xregs'], 
            dataset_name = dataset_name,
            frequency = frequency, 
            ext = ext
        )
        res_df = pd.merge(res_df, xregs_df, how = 'left', on = 'ds')

    if trend is not None:
        trend_df = create_time_trend(res_df['ds'], trend_type = trend)
        res_df = pd.merge(res_df, trend_df, how = 'left', on = 'ds')

    if fourier is not None:
        fourier_df = create_fourier_terms(res_df['ds'], fourier = fourier)
        res_df = pd.merge(res_df, fourier_df, how = 'left', on = 'ds')

    if calendar_features is not None:
        calendar_df = create_calendar_features(res_df['ds'], feature_list = calendar_features)
        res_df = pd.merge(res_df, calendar_df, how = 'left', on = 'ds')

    if features_to_normalize is not None:
        res_df = normalize_features(res_df, columns = features_to_normalize, type = 'min-max')
    
    if features_to_one_hot is not None:
        res_df = create_one_hot_features(res_df, columns = features_to_one_hot, drop_columns = False)

    if save:
        save_data(
            data = res_df, 
            path_list = ['data', dataset_name], 
            name_list = [dataset_name, frequency, 'prep'],
            ext = ext
        )

    return res_df

def get_data(path_list, name_list, min_series_length = None, max_series_length = None, samples = None, ext = '.parquet'):

    """Function to get the data.

    Args:
        path_list (list): List of directories to be joined.
        name_list (list): List of names to be used to create the file name.
        min_series_length (int, optional): Minimum length of series to be included. 
        Defaults to None.
        max_series_length (int, optional): Maximum length of series to be included. 
        Defaults to None.
        samples (int, optional): Number of samples to be included. Defaults to None.
        ext (string, optional): File extension (default is '.parquet').

    Returns:
        pd.Dataframe: The data.
    """

    res_df = load_data(path_list, name_list, ext)

    if min_series_length is not None:
        res_df = remove_series(res_df, min_series_length)

    if max_series_length is not None:
        res_df = filter_series(res_df, max_series_length)

    if samples is not None:
        res_df = sampling_data(res_df, samples)

    res_df.sort_values(by = ['unique_id', 'ds'], inplace = True)  

    return res_df
