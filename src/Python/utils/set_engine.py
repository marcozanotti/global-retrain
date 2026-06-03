from pytorch_lightning import trainer
import sys
sys.path.insert(0, 'src/Python/utils')
import os
import pandas_flavor as pf
from pandas.api.types import is_numeric_dtype
from statsforecast import StatsForecast
from statsforecast.models import Naive, SeasonalNaive, WindowAverage, AutoETS, AutoARIMA
from mlforecast import MLForecast
from neuralforecast import NeuralForecast
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from neuralforecast.models import MLP, LSTM, TCN, NBEATSx, NHITS
from mlforecast.auto import AutoMLForecast, AutoModel, AutoXGBoost, AutoLightGBM
from neuralforecast.auto import AutoMLP, AutoNBEATSx
import optuna
from pytorch_lightning.loggers import CSVLogger

# NOTE: feature and transform functions must be imported to be used with eval('fun_name')
# from sklearn.preprocessing import FunctionTransformer
# from mlforecast.target_transforms import GlobalSklearnTransformer
from mlforecast.target_transforms import LocalRobustScaler#, LocalStandardScaler, LocalMinMaxScaler, Differences
from mlforecast.lag_transforms import RollingMean, ExpandingMean, RollingStd
from neuralforecast.losses.pytorch import MAE, MSE, RMSE, MAPE, SMAPE
from custom_feats import is_weekend
from utilities import get_frequency

import logging
module_logger = logging.getLogger('set_engine')
optuna.logging.set_verbosity(optuna.logging.ERROR)

def get_loss_function(loss):
    """Function to get the loss function.

    Args:
        loss (str): name of the loss function.
    
    Returns:
        function: loss function.
    """

    module_logger.info('Defining loss function...')

    if loss is not None:
        if loss == 'MAE':
            loss = MAE()
        elif loss == 'MSE':
            loss = MSE()
        elif loss == 'RMSE':
            loss = RMSE()
        elif loss == 'MAPE':
            loss = MAPE()
        elif loss == 'SMAPE':
            loss = SMAPE()
    else:
        loss = MAE()

    return loss

def get_target_transforms(target_transforms):
    """Function to get the target transforms for the dataset.

    Args:
        target_transforms (list): list of target transformations.
    
    Returns:
        list: target transforms.
    """

    module_logger.info('Defining target trasformations...')
    # Log1p = FunctionTransformer(func = np.log1p, inverse_func = np.expm1)
    # target_transforms = [GlobalSklearnTransformer(Log1p)],
    
    if target_transforms is not None:
        for i in range(len(target_transforms)):
            try:
                fun_tmp = eval(target_transforms[i])
            except:
                fun_tmp = 'error'
            if fun_tmp != 'error':
                target_transforms[i] = fun_tmp

    return target_transforms

def get_lags(feature_list):
    """Function to get the lags for the dataset.

    Args:
        feature_list (list): list of features.
    
    Returns:
        list: lags.
    """

    module_logger.info('Defining lags...')
    return feature_list

def get_lag_transforms(feature_list):
    """Function to get the lag transforms for the dataset.

    Args:
        feature_list (list): list of features.
    
    Returns:
        dict: lag transforms.
    """

    module_logger.info('Defining lag trasformations...')
    if feature_list is not None:
        for k in feature_list.keys():
            for i in range(len(feature_list[k])):
                try:
                    fun_tmp = eval(feature_list[k][i])
                except:
                    fun_tmp = 'error'
                if fun_tmp != 'error':
                    feature_list[k][i] = fun_tmp

    return feature_list

def get_date_features(feature_list):
    """Function to get the date features for the dataset.

    Args:
        feature_list (list): list of features.
    
    Returns:
        list: date features.
    """

    module_logger.info('Defining date features...')

    if feature_list is not None:
        for i in range(len(feature_list)):
            try:
                fun_tmp = eval(feature_list[i])
            except:
                fun_tmp = 'error'
            if fun_tmp != 'error':
                feature_list[i] = fun_tmp

    return feature_list

def get_model_type(model_name):
    """Function to get the model type for the dataset.

    Args:
        model_name (str): name of the model.
    
    Returns:
        str: model type.
    """
    
    sf = ['Naive', 'SeasonalNaive', 'WindowAverage', 'ETS', 'ARIMA']
    ml = [
        'LinearRegression', 'Lasso', 'Ridge', 'ElasticNet',
        'RandomForestRegressor', 
        'XGBRegressor', 'LGBMRegressor', 'CatBoostRegressor' 
    ]
    dl = ['MLP', 'LSTM', 'TCN', 'NBEATSx', 'NHITS']
    automl = ['AutoXGBoost', 'AutoLightGBM']
    autodl = ['AutoMLP', 'AutoNBEATSx']

    if model_name in sf:
        model_type = 'sf'
    elif model_name in ml:
        model_type ='ml'
    elif model_name in dl:
        model_type = 'dl'
    elif model_name in automl:
        model_type = 'automl'
    elif model_name in autodl:
        model_type = 'autodl'
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

    if model_name == 'Naive':

        model_params = {
            model_name: {}
        }
    
    elif model_name == 'SeasonalNaive':

        model_params = {
            model_name: {
                'season_length': 1
            }
        }
    
    elif model_name == 'WindowAverage':
        
        model_params = {
            model_name: {
                'window_size': 1
            }
        }

    elif model_name == 'ETS':

        model_params = {
            model_name: {
                'season_leangth': 1,
                'model': 'ZZZ'
            }
        }
    
    elif model_name == 'ARIMA':

        model_params = {
            model_name: {
                'season_leangth': 1
            }
        }
    
    elif model_name == 'LinearRegression':

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
    
    elif model_name == 'ElasticNet':
        
        model_params = {
            model_name: {
                'l1_ratio': 0.5
            }
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
                'h': 28,
                'input_size': 2,
                'max_steps': 100,
                'early_stop_patience_steps': 10
            }
        }

    elif model_name == 'LSTM':

        model_params = {
            model_name: {
                'h': 28,
                'max_steps': 100,
                'early_stop_patience_steps': 10
            }
        }
    
    elif model_name == 'TCN':

        model_params = {
            model_name: {
                'h': 28,
                'max_steps': 100,
                'early_stop_patience_steps': 10
            }
        }
    
    elif model_name == 'NBEATSx':

        model_params = {
            model_name: {
                'h': 28,
                'input_size': 7,
                'max_steps': 100,
                'early_stop_patience_steps': 10
            }
        }

    elif model_name == 'NHITS':

        model_params = {
            model_name: {
                'h': 28,
                'input_size': 7,
                'max_steps': 100,
                'early_stop_patience_steps': 10 
            }
        }

    elif model_name == 'AutoXGBoost':

        model_params = {
            model_name: {}
        }
    
    elif model_name == 'AutoLightGBM':

        model_params = {
            model_name: {}
        }
    
    elif model_name == 'AutoMLP':

        model_params = {
            model_name: {
                'h': 28
            }
        }

    elif model_name == 'AutoNBEATSx':

        model_params = {
            model_name: {
                'h': 28
            }
        }    
    
    else:
        raise ValueError(f'Invalid model: {model_name}')

    return model_params

def get_automodel_config(model_config):
    """Function to get the automl config for the model.

    Args:
        model_params (dict): parameters for the model.

    Returns:
        function: automl config function.
    """

    def automl_config(trial: optuna.Trial):
        evaluated_config = {}
        for key, value in model_config.items():
            if isinstance(value, str) and value.startswith('trial.'):
                evaluated_config[key] = eval(value)
            else:
                evaluated_config[key] = value
        return evaluated_config

    return automl_config

def set_model(model_name, model_params = None):
    """Function to get the models for the dataset.

    Args:
        model_name (str): name of the model.
        model_params (dict, optional): parameters for the model. Defaults to None.
    
    Returns:
        list: model.
    """

    module_logger.info('Defining the model...')
    model_type = get_model_type(model_name)
    # create a copy of the model parameters to avoid modifying the original one
    model_params = model_params.copy()
    if model_params is None:
        model_params = get_default_model_params(model_name)[model_name]
    module_logger.info(f'Model parameters: {model_params}')


    if model_type == 'sf':
        
        if model_name == 'Naive':
            model = [Naive(**model_params)]
        elif model_name == 'SeasonalNaive':
            model = [SeasonalNaive(**model_params)]
        elif model_name == 'WindowAverage':
            model = [WindowAverage(**model_params)]
        elif model_name == 'ETS':
            model = [AutoETS(**model_params)]
        elif model_name == 'ARIMA':
            model = [AutoARIMA(**model_params)]
        else:
            raise ValueError(f'Invalid model: {model_name}')

    elif model_type == 'ml':

        if model_name == 'LinearRegression':
            model = [LinearRegression(**model_params)]
        elif model_name == 'Lasso':
            model = [Lasso(**model_params)]
        elif model_name == 'Ridge':
            model = [Ridge(**model_params)]
        elif model_name == 'ElasticNet':
            model = [ElasticNet(**model_params)]
        elif model_name == 'RandomForestRegressor':
            model = [RandomForestRegressor(**model_params)]
        elif model_name == 'XGBRegressor':
            model = [XGBRegressor(**model_params)]
        elif model_name == 'LGBMRegressor':
            model = [LGBMRegressor(**model_params)]
        elif model_name == 'CatBoostRegressor':
            model = [CatBoostRegressor(**model_params)]
        else:
            raise ValueError(f'Invalid model: {model_name}')

    elif model_type == 'dl':

        if 'loss' in model_params.keys():
            model_params['loss'] = get_loss_function(model_params['loss'])
        if 'valid_loss' in model_params.keys():
            model_params['valid_loss'] = get_loss_function(model_params['valid_loss'])

        if 'log_to_csv' in model_params.keys():
            trainer_kwargs = {"logger": CSVLogger("logs/")}
            model_params.pop('log_to_csv')
        else:
            trainer_kwargs = {}

        if model_name == 'MLP':
            model = [MLP(**model_params, **trainer_kwargs)]
        elif model_name == 'LSTM':
            model = [LSTM(**model_params, **trainer_kwargs)]
        elif model_name == 'TCN':
            model = [TCN(**model_params, **trainer_kwargs)]
        elif model_name == 'NBEATSx':
            model = [NBEATSx(**model_params, **trainer_kwargs)]
        elif model_name == 'NHITS':
            model = [NHITS(**model_params, **trainer_kwargs)]
        else:
            raise ValueError(f'Invalid model: {model_name}')
    
    elif model_type == 'automl':

        if model_name == 'AutoXGBoost':

            if model_params == {}:
                model = [AutoXGBoost(**model_params)]
            else:
                automl_config = get_automodel_config(model_params)
                model = {model_name: AutoModel(model = XGBRegressor(), config = automl_config)}

        elif model_name == 'AutoLightGBM':

            if model_params == {}:
                model = [AutoLightGBM(**model_params)]
            else:
                automl_config = get_automodel_config(model_params)
                model = {model_name: AutoModel(model = LGBMRegressor(), config = automl_config)}

        else:
            raise ValueError(f'Invalid model: {model_name}')
    
    elif model_type == 'autodl':

        h = model_params['h']
        model_params['backend'] = 'optuna'
        model_params['refit_with_val'] = False
        if 'loss' in model_params.keys():
            model_params['loss'] = get_loss_function(model_params['loss'])
        if 'valid_loss' in model_params.keys():
            model_params['valid_loss'] = get_loss_function(model_params['valid_loss'])
        
        if 'log_to_csv' in model_params.keys():
            trainer_kwargs = {"logger": CSVLogger("logs/")}
            model_params.pop('log_to_csv')
        else:
            trainer_kwargs = {}
        
        if model_name == 'AutoMLP':

            if model_params['config'] == {}:
                # if no config is provided, we set the default one with the specified horizon and backend
                autodl_config = AutoMLP.get_default_config(h = h, backend=model_params['backend'])
                model_params['config'] = autodl_config
                model = [AutoMLP(**model_params, **trainer_kwargs)]
            else:
                automl_config = get_automodel_config(model_params['config'])
                model_params['config'] = automl_config
                model = [AutoMLP(**model_params, **trainer_kwargs)]

        elif model_name == 'AutoNBEATSx':

            if model_params['config'] == {}:
                # if no config is provided, we set the default one with the specified horizon and backend
                autodl_config = AutoNBEATSx.get_default_config(h = h, backend=model_params['backend'])
                model_params['config'] = autodl_config
                model = [AutoNBEATSx(**model_params, **trainer_kwargs)]
            else:
                automl_config = get_automodel_config(model_params['config'])
                model_params['config'] = automl_config
                model = [AutoNBEATSx(**model_params, **trainer_kwargs)]

        else:
            raise ValueError(f'Invalid model: {model_name}')

    else:
        raise ValueError(f'Invalid model type: {model_type}')

    return model

def set_engine(model_name, frequency, features, target_transforms = None, model_params = None):

    """Function to set the engine based on the model name.

    Args:
        model_name (str): name of the model.
        frequency (str): frequency of the dataset.
        features (dict): features of the dataset.
        target_transforms (list, optional): target transformations. Defaults to None.
        model_params (dict, optional): parameters for the model. Defaults to None.
    
    Returns:
        StatsForecast: StatsForecast engine.
        MLForecast: MLForecast engine.
    """

    module_logger.info('Setting the engine...')
    model_type = get_model_type(model_name)
    model = set_model(model_name, model_params)
    freq = get_frequency(frequency)[0]
    seas_len = get_frequency(frequency)[1]
    target_transforms = get_target_transforms(target_transforms = target_transforms)
    static_features = features['static']
    lags = get_lags(feature_list = features['lags'])
    lag_transforms = get_lag_transforms(feature_list = features['lag_transforms'])
    date_features = get_date_features(feature_list = features['date'])

    if model_type =='sf':

        engine = StatsForecast(
            models = model,
            freq = freq,
            n_jobs = os.cpu_count()
        )

    elif model_type == 'ml':

        engine = MLForecast(
            models = model,
            freq = freq, 
            num_threads = os.cpu_count(),
            target_transforms = target_transforms,
            lags = lags,
            lag_transforms = lag_transforms,
            date_features = date_features
        )
    
    elif model_type == 'dl':

        engine = NeuralForecast(
            models = model, 
            freq = freq,
            local_scaler_type = target_transforms
        )

    elif model_type == 'automl':

        def init_config(trial: optuna.Trial):
            return {
                'target_transforms': target_transforms,
                'lags': lags,
                'lag_transforms': lag_transforms,
                'date_features': date_features
            }
        def fit_config(trial: optuna.Trial):
            return {'static_features': static_features}

        engine = AutoMLForecast(
            models = model,
            freq = freq,
            season_length = seas_len,
            init_config = init_config,
            fit_config = fit_config,
            num_threads = os.cpu_count(),
            reuse_cv_splits = True
        )
    
    elif model_type == 'autodl':

        engine = NeuralForecast(
            models = model, 
            freq = freq,
            local_scaler_type = target_transforms
        )
        
    else:
        raise ValueError(f'Invalid model: {model_name}')

    return engine

@pf.register_dataframe_method
def add_data_features(data, frequency, features, remove_static = False):

    """Function to add features to the data.

    Args:
        data (pd.DataFrame): Input dataframe in Nixtla's format.
        frequency (string): The frequency of the data (e.g., 'daily', 'weekly').
        features (dict): Dictionary containing feature details.
        forced_frequency (num, optional): value of the frequency to force. Defaults to None. 
    
    Returns:
        pd.DataFrame: dataframe with date features added.
    """

    module_logger.info('Adding features to the dataset...')

    freq = get_frequency(frequency)[0]
    lags = get_lags(feature_list = features['lags'])
    lag_transforms = get_lag_transforms(feature_list = features['lag_transforms'])
    date_features = get_date_features(feature_list = features['date'])
    static_features = features['static']

    data_feat = MLForecast(
        models = [],
        freq = freq, 
        lags = lags,
        lag_transforms = lag_transforms,
        date_features = date_features
    ).preprocess(df = data, static_features = static_features)

    if remove_static:
        data_feat = data_feat.drop(static_features, axis = 1)

    return data_feat
