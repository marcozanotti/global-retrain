
import pandas as pd
from statsforecast import StatsForecast


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
    models, 
    fallback_model, 
    train_df, 
    test_df, 
    freq, 
    levels, 
    horizon, 
    retrain_window = 1
):

    """Function to retrain the model and predict with retrained model.

    Args:
        models (list): list of Nixtla models.
        fallback_model (Nixtla model): fallback model for errors.
        train_df (pd.DataFrame): training data in the Nixtla format.
        test_df (pd.DataFrame): test data in the Nixtla format.
        freq (str): frequency of the data (e.g., 'daily', 'weekly').
        levels (list): confidence levels for the predictions.
        horizon (int): forecasting horizon.
        retrain_window (int, optional): window for retraining. Defaults to 1.

    Returns:
        pd.DataFrame: predictions made by the retrained models.
    """

    # TODO: implement conformal inference for intervals
    # TODO: save model parameters for retraining
    # TODO: implement retrain window parameter

    # combine train and test dataframes
    df = combine_train_test(train_df, test_df)

    # instantiate the StatsForecast class
    eng = StatsForecast(
        models = models, 
        freq = freq, 
        n_jobs = -1, 
        fallback_model = fallback_model
    )
    
    # n_fit = int(np.round(horizon / retrain_window, 0))
    # range(1, horizon + 1, retrain_window)
    
    # initialize the predictions dataframe
    preds_df = pd.DataFrame()
    
    for i in range(horizon):
        
        print(f'Step {i}')

        # define the training data
        train_df_tmp = df.groupby('unique_id').head(-(horizon - i))
        
        # define the xreg data
        test_idx_tmp = train_df_tmp.groupby('unique_id').tail(1).index + 1
        xreg_df_tmp = df.iloc[test_idx_tmp, ] \
            .drop(columns = ['y'], axis = 1)

        # define the conformal inference method
        # intervals_tmp = ConformalIntervals(h = (horizon - i), n_windows = 2, method = 'conformal_distribution')
        
        # fit the models
        print('Fitting models...')
        fit_tmp = eng.fit(df = train_df_tmp) # prediction_intervals = intervals_tmp
        
        # predict with the models
        print('Predicting...')
        preds_df_tmp = fit_tmp.predict(h = (horizon - i), level = levels) \
            .reset_index()
        preds_df_tmp.insert(0, 'sample', (i + 1))
        preds_df = pd.concat([preds_df, preds_df_tmp], axis = 0)


    return preds_df
