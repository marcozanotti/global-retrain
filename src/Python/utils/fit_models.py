
import time
import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from src.Python.utils.collect_data import combine_train_test


def split_train_test(data, test_window):

    """Function to split the data into train and test dataframes.

    Args:
        data (pd.DataFrame): data in the Nixtla's format.
        test_window (int): length of the test window.

    Returns:
        pd.DataFrame: training and test dataframes.
    """

    train_df = data \
        .groupby('unique_id') \
        .head(-test_window) \
        .sort_values(by = ['unique_id', 'ds']).reset_index(drop = True)
    test_df = data \
        .groupby('unique_id') \
        .tail(test_window) \
        .sort_values(by = ['unique_id', 'ds']).reset_index(drop = True)

    return train_df, test_df

def get_retrain_ids(test_window, horizon, retrain_window = 1):

    """Function to get the retrain ids.

    Args:
        test_window (int): length of the test window.
        retrain_window (int): window for retraining. Defaults to 1.
    
    Returns:
        list: list of retrain ids.
    """

    return list(range(0, (test_window - horizon + 1), retrain_window))

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
    data,
    engine, 
    test_window,
    horizon, 
    retrain_window = 1,
    levels = [60, 70, 80, 85, 90, 95, 99],
    intervals = None, 
    static_features = [],
    store_in_sample_results = False
):

    """Function to retrain the ML model and predict with retrained model.

    Args:
        data (pd.DataFrame): training and testing data in the Nixtla's format.
        engine (MLForecast class): ML model engine.
        test_window (int): length of the test window.
        horizon (int): forecasting horizon.
        retrain_window (int, optional): window for retraining. Defaults to 1.
        levels (list): confidence levels for the predictions. Defaults to
        [60, 70, 80, 85, 90, 95, 99].
        intervals (PredictionIntervals, optional): conformal inference method. 
        Defaults to PredictionIntervals(h = horizon, n_windows = 2).

    Returns:
        pd.DataFrame: predictions made by the retrained models.
    """

    print('===============================================================')
    print('---------------------------- START ----------------------------')
    
    # define the model name
    model_name = get_model_name(engine)
    print(model_name)

    # split the data into train and test dataframes
    train_df, test_df = split_train_test(data, test_window)

    # define the fitting times
    fitting_ids = get_retrain_ids(test_window, horizon, retrain_window)
    n_fitting = len(fitting_ids) # int(np.round(test_window / retrain_window, 0))
    
    # initialize the dataframes
    in_sample_df = pd.DataFrame()
    out_sample_df = pd.DataFrame()
    time_df = pd.DataFrame()
    
    start_time = time.time()

    for i in range(test_window - horizon + 1):
        
        print(f'Step {i + 1} of {test_window - horizon + 1}')

        # define the training data
        train_df_tmp = combine_train_test(train_df, test_df.groupby('unique_id').head(i))

        # define the testing data
        test_df_tmp = test_df.groupby('unique_id').head(i + horizon).reset_index(drop = True)
        ds_to_remove = train_df_tmp["ds"].drop_duplicates()
        test_df_tmp = test_df_tmp.loc[~test_df_tmp['ds'].isin(ds_to_remove)]
        # remove static features because they are used in fitting only
        X_df_tmp = test_df_tmp.drop(columns = static_features, axis = 1)
        
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
            preds_df_tmp = fit_tmp.predict(h = horizon, level = levels)
            end_predict_time = time.time()

            tot_sample_time = end_predict_time - start_fit_time

            if store_in_sample_results:
                # extract in-sample results from the model only when fitting
                print('Extracting fitted values...')
                in_sample_df_tmp = fit_tmp.fcst_fitted_values_ \
                    .rename(columns = {model_name: 'fit'}) \
                    .copy()
                in_sample_df_tmp['sample'] = i
                in_sample_df = pd.concat([in_sample_df, in_sample_df_tmp], axis = 0)
                
        else:

            # update the mlforecast object with the new data 
            # NOTE: fundamental to roll predictions without fitting !!!!!
            update_df = train_df_tmp.groupby('unique_id').tail(1)
            engine.update(update_df)

            print('Predicting with pre-trained model...')
            start_predict_time = time.time()
            preds_df_tmp = fit_tmp.predict(h = horizon, X_df = X_df_tmp, level = levels)
            end_predict_time = time.time()

            tot_sample_time = end_predict_time - start_predict_time
        
        # add actual out-of-sample to results
        out_sample_df_tmp = preds_df_tmp.copy()
        out_sample_df_tmp['sample'] = i
        out_sample_df_tmp = out_sample_df_tmp.merge(X_df_tmp)
        out_sample_df = pd.concat([out_sample_df, out_sample_df_tmp], axis = 0)
        
        # store computing time information for each sample
        time_df_tmp = pd.DataFrame({
            'sample': i,
            'total_fit_time': [end_fit_time - start_fit_time],
            'total_predict_time': [end_predict_time - start_predict_time],
            'total_sample_time': tot_sample_time
        })
        time_df = pd.concat([time_df, time_df_tmp], axis = 0)

    # format column names
    out_sample_df.columns = out_sample_df.columns.str.replace(model_name, 'fcst')
    
    # add additional columns to the dataframes for tracking parameters
    in_sample_df['method'] = model_name
    in_sample_df['test_window'] = test_window
    in_sample_df['horizon'] = horizon
    in_sample_df['retrain_window'] = retrain_window
    out_sample_df['method'] = model_name
    out_sample_df['test_window'] = test_window
    out_sample_df['horizon'] = horizon
    out_sample_df['retrain_window'] = retrain_window
    time_df['method'] = model_name
    time_df['test_window'] = test_window
    time_df['horizon'] = horizon
    time_df['retrain_window'] = retrain_window

    # reset indexes
    in_sample_df = in_sample_df.reset_index(drop = True)
    out_sample_df = out_sample_df.reset_index(drop = True)
    time_df = time_df.reset_index(drop = True)

    end_time = time.time()
    tot_time = end_time - start_time
    print(f'Total computing time: {tot_time:.1f} seconds')

    print('----------------------------- END -----------------------------')
    print('===============================================================')

    return in_sample_df, out_sample_df, time_df
