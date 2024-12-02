
import gc
import time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from statsforecast import StatsForecast
from src.Python.utils.collect_data import *
from src.Python.utils.set_engine import *


def split_train_test(data, test_window):

    """Function to split the data into train and test dataframes.

    Args:
        data (pd.DataFrame): data in the Nixtla's format.
        test_window (int): length of the test window.

    Returns:
        pd.DataFrame: training and test dataframes.
    """
    
    print('Splitting data into train and test...')
    train_df = data \
        .groupby('unique_id') \
        .head(-test_window) \
        .sort_values(by = ['unique_id', 'ds']) \
        .reset_index(drop = True)
    test_df = data \
        .groupby('unique_id') \
        .tail(test_window) \
        .sort_values(by = ['unique_id', 'ds']) \
        .reset_index(drop = True)

    return train_df, test_df

def get_retrain_ids(test_window, horizon, retrain_window = 1):

    """Function to get the retrain ids.

    Args:
        test_window (int): length of the test window.
        retrain_window (int): window for retraining. Defaults to 1.
    
    Returns:
        list: list of retrain ids.
    """
    res = list(range(0, (test_window - horizon + 1), retrain_window))
    return res

def get_model_name(engine):

    """Function to get the model name based on the engine.

    Args:
        engine (str): engine used for fitting the model.
    
    Returns:
        str: model name.
    """

    model_names = list(engine.models.keys())
    if len(model_names) > 1:
        raise(f'Please specify only one model at a time')
    else:
        model_name = model_names[0]
        
    return model_name

def extact_fitted_and_residuals(fitted_model, train_df):

    """Function to extract fitted values and residuals from
    the fitted model for each time series.

    Args:
        fitted_model (StatsForecast): fitted StatsForecast model.
        train_df (pd.DataFrame): training data in the Nixtla's format.

    Returns:
        pd.DataFrame: dataframe of fitted values and residuals
    """

    fitted_res = pd.DataFrame()
    residuals_res = pd.DataFrame()
    
    # estract fitted values and residuals for each time series
    for i in range(0, len(fitted_model.fitted_)): 

        fitted_tmp = fitted_model.fitted_[i][0].model_['fitted']
        fitted_res = pd.concat([fitted_res, pd.DataFrame(fitted_tmp)], axis = 0)

        residuals_tmp = fitted_model.fitted_[i][0].model_['residuals']
        residuals_res = pd.concat([residuals_res, pd.DataFrame(residuals_tmp)], axis = 0)
    
    fitted_res = fitted_res.reset_index(drop = True).rename(columns = {0: 'fitted'})
    residuals_res = residuals_res.reset_index(drop = True).rename(columns = {0:'residuals'})
    res = pd.concat([train_df, fitted_res, residuals_res], axis = 1)

    return res

def extract_model_parameters(fitted_model, in_sample_df):
    """Function to extract model parameters from the fitted model.

    Args:
        fitted_model (StatsForecast): fitted StatsForecast model.
        train_df (pd.DataFrame): training data in the Nixtla's format.

    Returns:
        pd.DataFrame: dataframe of model parameters
    """

    params_df = in_sample_df[['sample', 'unique_id']].drop_duplicates().reset_index(drop = True)
    sample_value = params_df['sample'][0]

    model_params = pd.DataFrame()
    
    # extract model parameters for each time series
    for i in range(0, len(fitted_model.fitted_)):
        # params_dict_tmp = {
        #     'method': fitted_model.fitted_[i][0].model_['method'],
        #     'components': fitted_model.fitted_[i][0].model_['components'],
        #     'params': fitted_model.fitted_[i][0].model_['par']
        # }
        method_tmp = fitted_model.fitted_[i][0].model_['method']
        components_tmp = fitted_model.fitted_[i][0].model_['components']
        params_tmp = fitted_model.fitted_[i][0].model_['par']

        model_params_tmp = pd.DataFrame([method_tmp, components_tmp, params_tmp])
        model_params_tmp.insert(0, 'name', ['method', 'components', 'params'])
        model_params_tmp.insert(0, 'sample', sample_value)
        model_params_tmp = model_params_tmp \
            .pivot(columns = 'name', values = 0, index = 'sample') \
            .reset_index(drop = True)
        model_params = pd.concat([model_params, model_params_tmp], axis = 0)

    model_params = model_params.reset_index(drop = True)    
    params_df = pd.concat([params_df, model_params], axis = 1)

    return params_df

def retrain_ets_model(
    data,
    models, 
    fallback_model, 
    freq, 
    levels, 
    test_window,
    horizon, 
    retrain_window = 1
):

    """Function to retrain the model and predict with retrained model.

    Args:
        data (pd.DataFrame): training and testing data in the Nixtla's format.
        models (list): list of Nixtla models.
        fallback_model (Nixtla model): fallback model for errors.
        freq (str): frequency of the data (e.g., 'daily', 'weekly').
        levels (list): confidence levels for the predictions.
        test_window (int): length of the test window.
        horizon (int): forecasting horizon.
        retrain_window (int, optional): window for retraining. Defaults to 1.

    Returns:
        pd.DataFrame: predictions made by the retrained models.
    """

    # FIXME: usare direttamente AutoETS con metodi .fit .predict .forward .predict_fitted_values
    # TODO: implement retrain window parameter
    # TODO: implement conformal inference for intervals
    # TODO: save model parameters for retraining

    # split the data into train and test dataframes
    train_df, test_df = split_train_test(data, test_window)

    # instantiate the StatsForecast class
    eng = StatsForecast(
        models = models, 
        freq = freq, 
        n_jobs = -1, 
        fallback_model = fallback_model
    )
    
    retrain_ids = list(range(0, test_window, retrain_window))
    n_refit = int(np.round(test_window / retrain_window, 0))
    
    # initialize the dataframes
    in_sample_df = pd.DataFrame()
    params_df = pd.DataFrame()
    out_sample_df = pd.DataFrame()
    time_df = pd.DataFrame()
    
    start_time = time.time()

    for i in range(test_window - horizon):
        
        print(f'Step {i + 1} of {test_window - horizon}')

        # define the training data
        train_df_tmp = combine_train_test(train_df, test_df.groupby('unique_id').head(i))
        
        # define the xreg data
        # FIXME: deve slittare non espandersi
        xreg_df_tmp = test_df \
            .groupby('unique_id') \
            .head(i + horizon) \
            .reset_index(drop = True) # \ .drop(columns = ['y'], axis = 1)

        # define the conformal inference method
        # intervals_tmp = ConformalIntervals(h = (horizon - i), n_windows = 2, method = 'conformal_distribution')
        
        if i == 0:

            # fit the models
            print('Fitting...')
            start_fit_time = time.time()
            fit_tmp = eng.fit(df = train_df_tmp) # prediction_intervals = intervals_tmp
            end_fit_time = time.time()

            # predict out-of-sample with the models
            print('Predicting...')
            start_predict_time = time.time()
            preds_df_tmp = fit_tmp.predict(h = horizon, level = levels)
            end_predict_time = time.time()
        
        else:

            if i in retrain_ids:
                print('Re-training...')
                print('Refitting 1 of n_refit')

            else:
                print('Predicting with pre-trained model...')
        

        # add sample information
        in_sample_df_tmp = train_df_tmp
        in_sample_df_tmp.insert(0, 'sample', (i + 1))
        out_sample_df_tmp = preds_df_tmp
        out_sample_df_tmp.insert(0, 'sample', (i + 1))

        # extract in-sample results from the models
        print('Extracting fitted values and residuals...')
        in_sample_df_tmp = extact_fitted_and_residuals(fit_tmp, in_sample_df_tmp)
        in_sample_df = pd.concat([in_sample_df, train_df_tmp], axis = 0)        

        # extract model parameters for each series
        print('Extracting model parameters...')
        params_df_tmp = extract_model_parameters(fit_tmp, in_sample_df_tmp)
        params_df = pd.concat([params_df, params_df_tmp], axis = 0)

        # storing computing time information for each sample
        time_df_tmp = pd.DataFrame({
            'sample': [i + 1],
            'start_fit_time': start_fit_time,
            'end_fit_time': end_fit_time,
            'total_fit_time': [end_fit_time - start_fit_time],
            'start_predict_time': start_predict_time,
            'end_predict_time': end_predict_time,
            'total_predict_time': [end_predict_time - start_predict_time]
        })
        time_df = pd.concat([time_df, time_df_tmp], axis = 0)
        
        # add actual out-of-sample to results
        out_sample_df = pd.concat([out_sample_df, out_sample_df_tmp], axis = 0)
        out_sample_df = out_sample_df.merge(xreg_df_tmp)

    end_time = time.time()
    tot_time = end_time - start_time
    print(f'Total computing time: {tot_time:.3f} seconds')

    return in_sample_df, params_df, out_sample_df, time_df, tot_time

def retrain_ml_model(
    dataset_name,
    frequency,
    engine, 
    test_window,
    horizon, 
    retrain_window = 1,
    levels = [50, 60, 70, 80, 90, 95, 99],
    intervals = None, 
    static_features = [],
    min_series_length = None,
    samples = None,
    store_in_sample_results = False,
    combine_results = False,
    ext = '.parquet'
):

    """Function to retrain the ML model and predict with retrained model.

    Args:
        dataset_name (str): name of the dataset (e.g., 'm5', 'm4').
        frequency (str): frequency of the data (e.g., 'daily', 'weekly').
        engine (MLForecast class): ML model engine.
        test_window (int): length of the test window.
        horizon (int): forecasting horizon.
        retrain_window (int, optional): window for retraining. Defaults to 1.
        levels (list): confidence levels for the predictions. Defaults to
        [60, 70, 80, 85, 90, 95, 99].
        intervals (PredictionIntervals, optional): conformal inference method. 
        Defaults to PredictionIntervals(h = horizon, n_windows = 2).
        static_features (list, optional): static features to include in the model.
        Defaults to [].
        min_series_length (int, optional): minimum length of the series.
        Defaults to None.
        samples (int, optional): number of samples to generate. Defaults to None.
        store_in_sample_results (bool, optional): store in-sample results.
        Defaults to False.
        ext (str, optional): file extension for storing results. Defaults to '.parquet'.

    Returns:
        pd.DataFrame: predictions made by the retrained models.
    """

    print('===============================================================')
    print('---------------------------- START ----------------------------')
    
    start_time = time.time()

    # define the model name
    model_name = get_model_name(engine)
    print(f'---- Model: {model_name} ---- Retrain Window: {retrain_window} ----')

    # load the dataset
    data = get_data(
        path = 'data/m5/',
        name_list = [dataset_name, frequency, 'prep'],
        ext = '.parquet',
        min_series_length = min_series_length,
        samples = samples
    )

    # split the data into train and test dataframes
    train_df, test_df = split_train_test(data, test_window)
    del data

    # define the fitting times
    fitting_ids = get_retrain_ids(test_window, horizon, retrain_window)
    n_fitting = len(fitting_ids) # int(np.round(test_window / retrain_window, 0))
    n_loops = test_window - horizon + 1
    
    # initialize the time dataframe (the only auto-incremental df with save at the end)
    time_df = pd.DataFrame()

    for i in range(n_loops):
        
        print(f'Step {i + 1} of {n_loops}')

        # define the training data
        train_df_tmp = combine_train_test(train_df, test_df.groupby('unique_id').head(i))

        # define the testing data
        test_df_tmp = test_df.groupby('unique_id').head(i + horizon)
        test_df_tmp.reset_index(drop = True, inplace = True)
        ds_to_remove = train_df_tmp["ds"].unique()
        test_df_tmp = test_df_tmp.loc[~test_df_tmp['ds'].isin(ds_to_remove)]
        # remove static features because they are used in fitting only
        test_df_tmp.drop(columns = static_features, axis = 1, inplace = True)
        
        if i in fitting_ids:

            # re-train the model
            print(f'Fitting: t = {i}, {int(i / retrain_window + 1)} of {n_fitting}...')
            start_fit_time = time.time()

            if intervals == None:
                fit_tmp = engine.fit(
                    df = train_df_tmp, 
                    static_features = static_features,
                    fitted = store_in_sample_results
                )
            else:
                fit_tmp = engine.fit(
                    df = train_df_tmp, 
                    static_features = static_features,
                    fitted = store_in_sample_results,
                    prediction_intervals = intervals
                )

            end_fit_time = time.time()

            # predict out-of-sample with the models
            print('Predicting...')
            start_predict_time = time.time()
            out_sample_df_tmp = fit_tmp.predict(h = horizon, level = levels, X_df = test_df_tmp)
            end_predict_time = time.time()

            tot_sample_time = end_predict_time - start_fit_time

            if store_in_sample_results:

                # extract in-sample results from the model only when fitting
                print('Extracting fitted values...')
                in_sample_df_tmp = fit_tmp.fcst_fitted_values_
                # add additional columns to the dataframes for tracking parameters
                in_sample_df_tmp['sample'] = i
                in_sample_df_tmp['method'] = model_name
                in_sample_df_tmp['test_window'] = test_window
                in_sample_df_tmp['horizon'] = horizon
                in_sample_df_tmp['retrain_window'] = retrain_window
                # format column names and reset index values
                in_sample_df_tmp = in_sample_df_tmp.rename(columns = {model_name: 'fit'})
                in_sample_df_tmp.reset_index(drop = True, inplace = True)
                # save to file
                save_data(
                    out_sample_df_tmp, 
                    path = f'results/{dataset_name}/{model_name}/{retrain_window}/fit/tmp/',
                    name_list = [
                        dataset_name, frequency, 'insample', model_name, str(retrain_window), i
                    ],
                    ext = ext
                )
                del in_sample_df_tmp
                
        else:

            # update the mlforecast object with the new data 
            # NOTE: fundamental to roll predictions without fitting !!!!!
            engine.update(train_df_tmp.groupby('unique_id').tail(1))

            print('Predicting with pre-trained model...')
            start_predict_time = time.time()
            out_sample_df_tmp = fit_tmp.predict(h = horizon, level = levels, X_df = test_df_tmp)
            end_predict_time = time.time()

            tot_sample_time = end_predict_time - start_predict_time
        
        # add additional columns to the dataframes for tracking parameters
        out_sample_df_tmp['sample'] = i
        out_sample_df_tmp['method'] = model_name
        out_sample_df_tmp['test_window'] = test_window
        out_sample_df_tmp['horizon'] = horizon
        out_sample_df_tmp['retrain_window'] = retrain_window
        # add actual out-of-sample to results
        out_sample_df_tmp = out_sample_df_tmp.merge(
            test_df_tmp, how = 'left', on = ['unique_id', 'ds'], copy = False
        )
        # format column names and reset index values
        out_sample_df_tmp.columns = out_sample_df_tmp.columns.str.replace(model_name, 'fcst')
        out_sample_df_tmp.reset_index(drop = True, inplace = True)
        # save to file
        save_data(
            out_sample_df_tmp, 
            path = f'results/{dataset_name}/{model_name}/{retrain_window}/preds/tmp/',
            name_list = [
                dataset_name, frequency, 'outsample', model_name, str(retrain_window), i
            ],
            ext = ext
        ) 
        
        # store computing time information for each sample
        time_df_tmp = pd.DataFrame({
            'sample': i,
            'total_fit_time': [end_fit_time - start_fit_time],
            'total_predict_time': [end_predict_time - start_predict_time],
            'total_sample_time': tot_sample_time
        })
        time_df = pd.concat([time_df, time_df_tmp], axis = 0)

        del train_df_tmp, ds_to_remove, test_df_tmp, out_sample_df_tmp, time_df_tmp
        if (i % 10) == 0:
            gc.collect() # call gc once every 10 iterations to avoid overhead

    
    if combine_results:

        if store_in_sample_results:
            # combine and save the insample tmp files
            combine_and_save_files(
                path_to_read = f'results/{dataset_name}/{model_name}/{retrain_window}/fit/tmp/',
                path_to_write = f'results/{dataset_name}/{model_name}/{retrain_window}/fit/',
                name_list = [
                    dataset_name, frequency, 'insample', model_name, str(retrain_window)
                ],
                ext = ext
            )
    
        # combine and save the outsample tmp files
        combine_and_save_files(
            path_to_read = f'results/{dataset_name}/{model_name}/{retrain_window}/preds/tmp/',
            path_to_write = f'results/{dataset_name}/{model_name}/{retrain_window}/preds/',
            name_list = [
                dataset_name, frequency, 'outsample', model_name, str(retrain_window)
            ],
            ext = ext
        )

    # add additional columns to the dataframes for tracking parameters
    time_df['method'] = model_name
    time_df['test_window'] = test_window
    time_df['horizon'] = horizon
    time_df['retrain_window'] = retrain_window
    time_df.reset_index(drop = True, inplace = True)
    # save to file
    save_data(
        time_df, 
        path = f'results/{dataset_name}/{model_name}/{retrain_window}/time/',
        name_list = [
            dataset_name, frequency, 'time', model_name, str(retrain_window)
        ],
        ext = ext
    )

    end_time = time.time()
    tot_time = end_time - start_time
    print(f'Total computing time: {tot_time:.1f} seconds')

    print('----------------------------- END -----------------------------')
    print('===============================================================')

    return

def retrain_model(
    model_names,
    retrain_scenarios,
    dataset_name,
    frequency,
    test_window,
    horizon, 
    levels = [50, 60, 70, 80, 90, 95, 99],
    intervals = None, 
    static_features = [],
    min_series_length = None,
    samples = None,
    store_in_sample_results = False,
    combine_results = False,
    ext = '.parquet'
):

    """Function to retrain the models and predict with retrained models
    for different retraining scenarios.

    Args:
        model_names (list): names of the ML models to be used.
        retrain_scenarios (list): scenarios for retraining.
        dataset_name (str): name of the dataset (e.g., 'm5', 'm4').
        frequency (str): frequency of the data (e.g., 'daily', 'weekly').
        engine (MLForecast class): ML model engine.
        test_window (int): length of the test window.
        horizon (int): forecasting horizon.
        levels (list): confidence levels for the predictions. Defaults to
        [60, 70, 80, 85, 90, 95, 99].
        intervals (PredictionIntervals, optional): conformal inference method. 
        Defaults to PredictionIntervals(h = horizon, n_windows = 2).
        static_features (list, optional): static features to include in the model.
        Defaults to [].
        min_series_length (int, optional): minimum length of the series.
        Defaults to None.
        samples (int, optional): number of samples to generate. Defaults to None.
        store_in_sample_results (bool, optional): store in-sample results.
        Defaults to False.
        ext (str, optional): file extension for storing results. Defaults to '.parquet'.

    Returns:
        pd.DataFrame: predictions made by the retrained models.
    """

    for m in model_names:
        
        model_type = get_model_type(m)
        engine_tmp = set_engine(m, dataset_name, frequency)

        for scenario in retrain_scenarios:

            if model_type == 'ml':

                retrain_ml_model(
                    dataset_name = dataset_name,
                    frequency = frequency,
                    engine = engine_tmp,
                    test_window = test_window,
                    horizon = horizon,
                    retrain_window = scenario,
                    levels = levels,
                    intervals = intervals,
                    static_features = static_features,
                    min_series_length = min_series_length,
                    samples = samples,
                    store_in_sample_results = store_in_sample_results,
                    combine_results = combine_results,
                    ext = ext
                )
            
            else:
                raise ValueError('Not yet implemented.')

    return