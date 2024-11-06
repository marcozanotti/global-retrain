
import time
import numpy as np
import pandas as pd
from statsforecast import StatsForecast

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


def combine_train_test(train_df, test_df):
    """Function to combine train and test dataframes.

    Args:
        train_df (pd.DataFrame): training data in the Nixtla format.
        test_df (pd.DataFrame): test data in the Nixtla format.

    Returns:
        pd.DataFrame: combined train and test dataframes.
    """

    combined_df = pd.concat([train_df, test_df], axis = 0, ignore_index = True)
    combined_df = combined_df.sort_values(by = ['unique_id', 'ds']).reset_index(drop = True)

    return combined_df


def retrain_model(
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

    # TODO: implement conformal inference for intervals
    # TODO: save model parameters for retraining
    # TODO: implement retrain window parameter

    # split the data into train and test dataframes
    train_df, test_df = split_train_test(data, test_window)

    # instantiate the StatsForecast class
    eng = StatsForecast(
        models = models, 
        freq = freq, 
        n_jobs = -1, 
        fallback_model = fallback_model
    )
    
    n_fit = int(np.round(test_window / retrain_window, 0))
    retrain_ids = list(range(0, test_window, retrain_window))
    
    # initialize the dataframes
    time_df = pd.DataFrame()
    actual_df = pd.DataFrame()
    preds_df = pd.DataFrame()
    
    start_time = time.time()

    for i in range(test_window - horizon):
        
        print(f'Step {i + 1} of {test_window - horizon}')

        # define the training data
        train_df_tmp = combine_train_test(train_df, test_df.groupby('unique_id').head(i))
        
        # define the xreg data
        xreg_df_tmp = test_df \
            .groupby('unique_id') \
            .head(i + horizon) \
            .drop(columns = ['y'], axis = 1)

        # define the conformal inference method
        # intervals_tmp = ConformalIntervals(h = (horizon - i), n_windows = 2, method = 'conformal_distribution')
        
        if i in retrain_ids:
            print('Re-training...')
        else:
            print('Using pre-trained models...')

        # fit the models
        print('Fitting...')
        start_fit_time = time.time()
        fit_tmp = eng.fit(df = train_df_tmp) # prediction_intervals = intervals_tmp
        end_fit_time = time.time()

        # predict with the models
        print('Predicting...')
        start_predict_time = time.time()
        preds_df_tmp = fit_tmp.predict(h = horizon, level = levels)
        end_predict_time = time.time()
        preds_df_tmp.insert(0, 'sample', (i + 1))
        preds_df = pd.concat([preds_df, preds_df_tmp], axis = 0)
        
        # storing training data for each sample
        train_df_tmp.insert(0, 'sample', (i + 1))
        actual_df = pd.concat([actual_df, train_df_tmp], axis = 0)

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

    end_time = time.time()
    tot_time = end_time - start_time
    print(f'Total computing time: {tot_time:.3f} seconds')

    return preds_df, actual_df, time_df, tot_time
