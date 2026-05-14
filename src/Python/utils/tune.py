import sys
sys.path.insert(0, 'src/Python/utils')
import time
import numpy as np
import pandas as pd
from utilities import save_data, get_dataset_frequency
from collect_data import get_data
from set_engine import get_model_type, set_engine
from fit_models import get_prediction_intervals, split_train_test

import logging
module_logger = logging.getLogger('tune_models')

def fit_automl_model(
    train_df, 
    dataset_name,
    frequency,
    model_name,
    engine,
    n_models,
    valid_window,
    step_size,
    valid_loss,
    horizon,
    features,
    intervals,
    levels,
    store_in_sample_results,
    save_model,
    ext
):

    """ Function to fit an AutoML model based on the provided configuration.
    
        Args:
        train_df (pd.DataFrame): Training dataframe.
        dataset_name (str): Name of the dataset.
        frequency (str): Frequency of the time series data.
        model_name (str): Name of the model to fit.
        engine: Engine object for fitting the model.
        n_models (int): Number of models to fit.
        valid_window (int): Size of the validation window.
        step_size (int): Step size between each cross validation window.
        valid_loss (function): Function that takes the validation and train dataframes and produces a float. If None will use the average SMAPE across series.
        horizon (int): Forecasting horizon.
        features (dict): Dictionary containing feature information.
        intervals (list): List of intervals for prediction intervals.
        levels (list): List of confidence levels for prediction intervals.
        store_in_sample_results (bool): Whether to store in-sample results.
        save_model (bool): Whether to save the fitted model.
        ext (str): File extension for saving results and models.

        Returns:
        None
    """

    module_logger.info('---------------------------------------------------------------')

    # define the model name
    # model_name = get_model_name(engine)
    module_logger.info(f'[ Tuning Model: {model_name} ]')

    # intervals
    pred_intervals = get_prediction_intervals(intervals, model_class = 'ml')

    # define the fitting times
    n_valid_windows = (valid_window - horizon) // step_size + 1

    # define static features
    # static_features = features['static']

    module_logger.info(f'Train dataset contains: {list(train_df.columns)}...')

    start_time = time.time()
    auto_mlf = engine.fit(
        df = train_df,
        num_samples = n_models, # Number of trials to run.
        n_windows = n_valid_windows, # Number of windows to evaluate.
        h = horizon,
        step_size = step_size, # Step size between each cross validation window. If None it will be equal to h.
        loss = valid_loss, # Function that takes the validation and train dataframes and produces a float. If None will use the average SMAPE across series.
        input_size = None, # Maximum training samples per serie in each window. If None, will use an expanding window.
        refit = False, # Retrain model for each cross validation window.
        prediction_intervals = pred_intervals,
        fitted = store_in_sample_results
    )
    end_time = time.time()
    # X_df = test_df.drop(columns = ['y'] + features['static'])
    # auto_mlf.predict(horizon, level=levels, X_df = X_df)

    # Extract the best trial information
    best_trial = auto_mlf.results_[model_name].best_trial
    best_loss = best_trial.value
    best_params = best_trial.user_attrs['config']['model_params']
    init_params = best_trial.user_attrs['config']['mlf_init_params']
    fit_params = best_trial.user_attrs['config']['mlf_fit_params']
    best_valid_plan = {
            'valid_size': valid_window,
            'n_valid_windows': n_valid_windows,
            'step_size': step_size
        }
    # best_loss_by_folds = best_trial.intermediate_values
    tuning_results = {
        'model_name': model_name,
        'validation_plan': best_valid_plan,
        'best_loss': best_loss,
        'best_params': best_params,
        'init_params': init_params,
        'fit_params': fit_params,
        # 'best_loss_by_folds': best_loss_by_folds
    }
    module_logger.info(f'Tuning results: {tuning_results}')
    tuning_results_df = pd.DataFrame({
        'model_name': [model_name],
        'validation_plan': [{
            'valid_size': valid_window,
            'n_valid_windows': n_valid_windows,
            'step_size': step_size
        }],
        'best_loss': [best_loss],
        'best_params': [best_params],
        'init_params': [init_params],
        'fit_params': [fit_params]
    })

    # save tuning results to file
    save_data(
        data = tuning_results_df,
        path_list = ['results', dataset_name, frequency, 'tuning'],
        name_list = [model_name, f"{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"],
        ext = '.json'
    )

    tot_time = end_time - start_time
    module_logger.info(f'Total computing time: {tot_time:.1f} seconds')

    return

def fit_autodl_model(
    train_df, 
    dataset_name,
    frequency,
    model_name,
    engine,
    n_models,
    valid_window,
    step_size,
    valid_loss,
    horizon,
    features,
    intervals,
    levels,
    store_in_sample_results,
    save_model,
    ext
):

    """ Function to fit an AutoDL model based on the provided configuration.
    
        Args:
        train_df (pd.DataFrame): Training dataframe.
        dataset_name (str): Name of the dataset.
        frequency (str): Frequency of the time series data.
        model_name (str): Name of the model to fit.
        engine: Engine object for fitting the model.
        n_models (int): Number of models to fit.
        valid_window (int): Size of the validation window.
        step_size (int): Step size between each cross validation window.
        valid_loss (function): Function that takes the validation and train dataframes and produces a float. If None will use the average SMAPE across series.
        horizon (int): Forecasting horizon.
        features (dict): Dictionary containing feature information.
        intervals (list): List of intervals for prediction intervals.
        levels (list): List of confidence levels for prediction intervals.
        store_in_sample_results (bool): Whether to store in-sample results.
        save_model (bool): Whether to save the fitted model.
        ext (str): File extension for saving results and models.

        Returns:
        None
    """

    module_logger.info('---------------------------------------------------------------')

    # define the model name
    # model_name = get_model_name(engine)
    module_logger.info(f'[ Tuning Model: {model_name} ]')

    # intervals
    pred_intervals = get_prediction_intervals(intervals, model_class = 'ml')

    # define the fitting times
    n_valid_windows = (valid_window - horizon) // step_size + 1

    # define static features
    # static_features = features['static']

    module_logger.info(f'Train dataset contains: {list(train_df.columns)}...')

    start_time = time.time()
    auto_mlf = engine.fit(
        df = train_df,
        num_samples = n_models, # Number of trials to run.
        n_windows = n_valid_windows, # Number of windows to evaluate.
        h = horizon,
        step_size = step_size, # Step size between each cross validation window. If None it will be equal to h.
        loss = valid_loss, # Function that takes the validation and train dataframes and produces a float. If None will use the average SMAPE across series.
        input_size = None, # Maximum training samples per serie in each window. If None, will use an expanding window.
        refit = False, # Retrain model for each cross validation window.
        prediction_intervals = pred_intervals,
        fitted = store_in_sample_results
    )
    end_time = time.time()
    # X_df = test_df.drop(columns = ['y'] + features['static'])
    # auto_mlf.predict(horizon, level=levels, X_df = X_df)

    # Extract the best trial information
    best_trial = auto_mlf.results_[model_name].best_trial
    best_loss = best_trial.value
    best_params = best_trial.user_attrs['config']['model_params']
    init_params = best_trial.user_attrs['config']['mlf_init_params']
    fit_params = best_trial.user_attrs['config']['mlf_fit_params']
    best_valid_plan = {
            'valid_size': valid_window,
            'n_valid_windows': n_valid_windows,
            'step_size': step_size
        }
    # best_loss_by_folds = best_trial.intermediate_values
    tuning_results = {
        'model_name': model_name,
        'validation_plan': best_valid_plan,
        'best_loss': best_loss,
        'best_params': best_params,
        'init_params': init_params,
        'fit_params': fit_params,
        # 'best_loss_by_folds': best_loss_by_folds
    }
    module_logger.info(f'Tuning results: {tuning_results}')
    tuning_results_df = pd.DataFrame({
        'model_name': [model_name],
        'validation_plan': [{
            'valid_size': valid_window,
            'n_valid_windows': n_valid_windows,
            'step_size': step_size
        }],
        'best_loss': [best_loss],
        'best_params': [best_params],
        'init_params': [init_params],
        'fit_params': [fit_params]
    })

    # save tuning results to file
    save_data(
        data = tuning_results_df,
        path_list = ['results', dataset_name, frequency, 'tuning'],
        name_list = [model_name, f"{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"],
        ext = '.json'
    )

    tot_time = end_time - start_time
    module_logger.info(f'Total computing time: {tot_time:.1f} seconds')

    return

def fit_auto_model(config):

    """Function to fit an AutoML or AutoDL model based on the provided configuration.
    
        Args:
        config (dict): Configuration parameters.
    """

    module_logger.info('===============================================================')

    # dataset parameters
    dataset_name = config['dataset']['dataset_name']
    frequency = config['dataset']['frequency']
    min_series_length = config['dataset']['min_series_length']
    max_series_length = config['dataset']['max_series_length']
    samples = config['dataset']['samples']
    ext = config['dataset']['ext']
    seed = config['dataset']['seed']
    # fitting parameters
    test_window = config['fitting']['test_window']
    horizon = config['fitting']['horizon']
    intervals = config['fitting']['intervals']
    levels = config['fitting']['levels']
    store_in_sample_results = config['fitting']['store_in_sample_results']
    n_models = config['fitting']['n_models']
    valid_window = config['fitting']['valid_window']
    step_size = config['fitting']['step_size']
    valid_loss = config['fitting']['valid_loss']
    # model parameters
    model_names = config['model_names']
    model_params = config['model_params']
    target_transforms = config['target_transforms']
    features = config['features']
    save_model = config['save_model']

    # load the dataset
    if samples is not None:
        np.random.seed(seed)
    data = get_data(
        path_list = ['data', dataset_name],
        name_list = [dataset_name, frequency, 'prep'],
        ext = '.parquet',
        min_series_length = min_series_length,
        max_series_length = max_series_length,
        samples = samples
    )
    data = data[['unique_id', 'ds', 'y'] + features['static'] + features['xregs']] 
    # split the data into train and test dataframes
    train_df, test_df = split_train_test(data, test_window)
    del data

    for m in model_names:

        module_logger.info('---------------------------- START ----------------------------')
            
        model_type = get_model_type(m)
        module_logger.info(f'[ Model type: {model_type} | Model name: {m} ]')

        if model_params is None:
            engine_tmp = set_engine(m, get_dataset_frequency(dataset_name), features, target_transforms, model_params)
        else:
            engine_tmp = set_engine(m, get_dataset_frequency(dataset_name), features, target_transforms, model_params[m])

        if model_type == 'automl':

            fit_automl_model(
                train_df = train_df, 
                dataset_name = dataset_name,
                frequency = frequency,
                model_name = m,
                engine = engine_tmp,
                n_models = n_models,
                valid_window = valid_window,
                step_size = step_size,
                valid_loss = valid_loss,
                horizon = horizon,
                features = features,
                intervals = intervals,
                levels = levels,
                store_in_sample_results = store_in_sample_results,
                save_model = save_model,
                ext = ext
            )

        elif model_type == 'autodl':

            fit_autodl_model(
                train_df = train_df, 
                dataset_name = dataset_name,
                frequency = frequency,
                model_name = m,
                engine = engine_tmp,
                valid_window = valid_window,
                horizon = horizon,
                features = features,
                intervals = intervals,
                levels = levels,
                store_in_sample_results = store_in_sample_results,
                save_model = save_model,
                ext = ext
            )    
                
        else:
            raise ValueError('Not yet implemented.')
            
        module_logger.info('----------------------------- END -----------------------------')
        
    module_logger.info('===============================================================')

    return