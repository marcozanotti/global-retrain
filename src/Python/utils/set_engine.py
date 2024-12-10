
import os
from mlforecast import MLForecast
from neuralforecast import NeuralForecast
from mlforecast.lag_transforms import RollingMean, ExpandingMean
# from sklearn.preprocessing import FunctionTransformer
# from mlforecast.target_transforms import GlobalSklearnTransformer
# from mlforecast.target_transforms import LocalStandardScaler, LocalMinMaxScaler, Differences
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from neuralforecast.models import MLP, KAN, RNN, GRU, LSTM, TCN, NBEATS, NHITS, DeepAR
from src.Python.utils.custom_feats import is_weekend

import logging
module_logger = logging.getLogger('set_engine')


def get_frequency(frequency):
    """Function to get the frequency of the dataset.

    Args:
        frequency (str): frequency of the dataset.
    
    Returns:
        str: frequency.
    """

    module_logger.info('Defining frequency...')

    if frequency == 'hourly':
        freq = ['H', 24]
    elif frequency == 'daily':
        freq = ['D', 7]
    elif frequency == 'weekly':
        freq = ['W', 52]
    elif frequency == 'monthly':
        freq = ['M',12]
    elif frequency == 'quarterly':
        freq = ['Q', 4]
    elif frequency == 'yearly':
        freq = ['Y', 1]
    else:
        raise ValueError(f'Invalid frequency: {frequency}')

    return freq

def get_target_transforms(model_name):
    """Function to get the target transforms for the dataset.

    Args:
        model_name (str): name of the model.
    
    Returns:
        list: target transforms.
    """

    module_logger.info('Defining target trasformations...')

    # Log1p = FunctionTransformer(func = np.log1p, inverse_func = np.expm1)
    # target_transforms = [GlobalSklearnTransformer(Log1p)],

    if model_name == 'LinearRegression':

        target_transforms = None
    
    elif model_name == 'Lasso':

        target_transforms = None
    
    elif model_name == 'Ridge':

        target_transforms = None
    
    elif model_name == 'RandomForestRegressor':

        target_transforms = None
    
    elif model_name == 'XGBRegressor':

        target_transforms = None
    
    elif model_name == 'LGBMRegressor':

        target_transforms = None
    
    elif model_name == 'CatBoostRegressor':

        target_transforms = None

    elif model_name == 'MLP':

        target_transforms = None
    
    else:
        raise ValueError(f'Invalid model: {model_name}')

    return target_transforms

def get_lags(dataset_name, frequency):
    """Function to get the lags for the dataset.

    Args:
        dataset_name (str): name of the dataset.
        frequency (str): frequency of the dataset.
    
    Returns:
        list: lags.
    """

    module_logger.info('Defining lags...')

    if dataset_name == 'm5':

        if frequency == 'daily':

            lags = [1] + [7 * (i+1) for i in range(8)]
        
        else:
            raise ValueError(f'Invalid frequency: {frequency}')        

    else:
        raise ValueError(f'Invalid dataset: {dataset_name}')

    return lags

def get_lag_transforms(dataset_name, frequency):
    """Function to get the lag transforms for the dataset.

    Args:
        dataset_name (str): name of the dataset.
        frequency (str): frequency of the dataset.
    
    Returns:
        dict: lag transforms.
    """

    module_logger.info('Defining lag trasformations...')

    if dataset_name == 'm5':

        if frequency == 'daily':

            lag_transforms = {
                1: [RollingMean(7), RollingMean(14), RollingMean(30), ExpandingMean()],
                7: [RollingMean(7), RollingMean(14), RollingMean(30)],
                14: [RollingMean(7), RollingMean(14), RollingMean(30)],
                30: [RollingMean(7), RollingMean(14), RollingMean(30)]
            }

        else:
            raise ValueError(f'Invalid frequency: {frequency}')

    else:
        raise ValueError(f'Invalid dataset: {dataset_name}')

    return lag_transforms

def get_date_features(dataset_name, frequency):
    """Function to get the date features for the dataset.

    Args:
        dataset_name (str): name of the dataset.
        frequency (str): frequency of the dataset.
    
    Returns:
        list: date features.
    """

    module_logger.info('Defining date features...')

    if dataset_name == 'm5':

        if frequency == 'daily':

            date_features = ['year', 'quarter', 'month', 'week', 'dayofweek', 'day', is_weekend]

        else:
            raise ValueError(f'Invalid frequency: {frequency}')

    else:
        raise ValueError(f'Invalid dataset: {dataset_name}')

    return date_features

def get_model_type(model_name):
    """Function to get the model type for the dataset.

    Args:
        model_name (str): name of the model.
    
    Returns:
        str: model type.
    """
    
    sf = ['ETS', 'ARIMA']
    ml = [
        'LinearRegression', 'Lasso', 'Ridge', 
        'RandomForestRegressor', 
        'XGBRegressor', 'LGBMRegressor', 'CatBoostRegressor' 
    ]
    dl = ['MLP', 'KAN', 'RNN', 'LSTM', 'GRU', 'TCN', 'NBEATS', 'NHITS', 'DeepAR']

    if model_name in sf:
        model_type = 'sf'
    elif model_name in ml:
        model_type ='ml'
    elif model_name in dl:
        model_type = 'dl'
    else:
        raise ValueError(f'Invalid model: {model_name}')

    return model_type

def get_default_model_params(model_name):
    """Function to get the default parameters for the models.

    Args:
        model_name (str): name of the model.
    
    Returns:
        dict: default model parameters.
    """

    module_logger.info('Defining default model parameters...')

    if model_name == 'LinearRegression':

        model_params = {
            model_name: {'n_jobs': -1}
        }
    
    elif model_name == 'Lasso':

        model_params = {
            model_name: {}
        }
    
    elif model_name == 'Ridge':

        model_params = {
            model_name: {}
        }
    
    elif model_name == 'RandomForestRegressor':

        model_params = {
            model_name: {
                'n_estimators': 100,
                'n_jobs': -1
            }
        }
    
    elif model_name == 'XGBRegressor':

        model_params = {
            model_name: {
                'n_estimators': 100,
                'n_jobs': -1
            }
        }
    
    elif model_name == 'LGBMRegressor':

        model_params = {
            model_name: {
                'n_estimators': 100,
                'n_jobs': -1
            }
        }

    elif model_name == 'CatBoostRegressor':

        model_params = {
            model_name: {
                'n_estimators': 100,
                'thread_count': -1
            }
        }
    
    elif model_name == 'MLP':

        model_params = {
            model_name: {
                'h': 7,
                'input_size': 2,
                'max_steps': 20
            }
        }
    
    else:
        raise ValueError(f'Invalid model: {model_name}')

    return model_params

def set_model(model_name, model_params = None):
    """Function to get the models for the dataset.

    Args:
        model_name (str): name of the model.
        model_params (dict, optional): parameters for the model. Defaults to None.
    
    Returns:
        list: model.
    """

    module_logger.info('Defining the model...')
    if model_params is None:
        model_params = get_default_model_params(model_name)[model_name]
    module_logger.info(f'Model parameters: {model_params}')

    if model_name == 'LinearRegression':
        model = [LinearRegression(**model_params)]
    elif model_name == 'Lasso':
        model = [Lasso(**model_params)]
    elif model_name == 'Ridge':
        model = [Ridge(**model_params)]
    elif model_name == 'RandomForestRegressor':
        model = [RandomForestRegressor(**model_params)]
    elif model_name == 'XGBRegressor':
        model = [XGBRegressor(**model_params)]
    elif model_name == 'LGBMRegressor':
        model = [LGBMRegressor(**model_params)]
    elif model_name == 'CatBoostRegressor':
        model = [CatBoostRegressor(**model_params)]
    elif model_name == 'MLP':
        model = [MLP(**model_params)]
    elif model_name == 'KAN':
        model = [KAN(**model_params)]
    elif model_name == 'RNN':
        model = [RNN(**model_params)]
    elif model_name == 'LSTM':
        model = [LSTM(**model_params)]
    elif model_name == 'GRU':
        model = [GRU(**model_params)]
    elif model_name == 'TCN':
        model = [TCN(**model_params)]
    elif model_name == 'NBEATS':
        model = [NBEATS(**model_params)]
    elif model_name == 'NHITS':
        model = [NHITS(**model_params)]
    elif model_name == 'DeepAR':
        model = [DeepAR(**model_params)]
    else:
        raise ValueError(f'Invalid model: {model_name}')

    return model

def set_engine(model_name, dataset_name, frequency, model_params = None):

    """Function to set the engine based on the model name.

    Args:
        model_name (str): name of the model.
        dataset_name (str): name of the dataset.
        frequency (str): frequency of the dataset.
        model_params (dict, optional): parameters for the model. Defaults to None.
    
    Returns:
        StatsForecast: StatsForecast engine.
        MLForecast: MLForecast engine.
    """

    module_logger.info('Setting the engine...')
    model_type = get_model_type(model_name)
    model = set_model(model_name, model_params)
    freq = get_frequency(frequency)
    target_transforms = get_target_transforms(model_name)
    lags = get_lags(dataset_name, frequency)
    lag_transforms = get_lag_transforms(dataset_name, frequency)
    date_features = get_date_features(dataset_name, frequency)

    if model_type =='sf':

        raise ValueError(f'Not yet implemented for model {model_name}')

    elif model_type == 'ml':

        engine = MLForecast(
            models = model,
            freq = freq[0], 
            num_threads = os.cpu_count(),
            target_transforms = target_transforms,
            lags = lags,
            lag_transforms = lag_transforms,
            date_features = date_features
        )
    
    elif model_type == 'dl':

        engine = NeuralForecast(
            models = model, 
            freq = freq[0],
            local_scaler_type = target_transforms
        )
        
    else:
        raise ValueError(f'Invalid model: {model_name}')

    return engine

