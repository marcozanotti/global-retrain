import time
import pandas as pd
# pip install git+https://github.com/knightianuncertainty/regimes
from regimes import BaiPerronTest

import logging
module_logger = logging.getLogger('structural_breaks')

def get_breaks(df, nobs = None, max_breaks=5, selection='bic'):

    """ Detect structural breaks in time series data using the Bai-Perron test.
        
        Args:
        df (pd.DataFrame): Input DataFrame containing 'unique_id', 'ds', and 'y' columns.
        nobs (int, optional): Number of observations to consider for break detection. If None, all observations are used. Default is None.
        max_breaks (int, optional): Maximum number of breaks to detect. Default is 5.
        selection (str, optional): Method for selecting the number of breaks. Options are 'bic', 'lwz', or 'sequential'. Default is 'bic'.
        
        Returns:
        pd.DataFrame: DataFrame containing detected break points and their confidence intervals for each unique_id.
    """

    module_logger.info('---------------------------- START ----------------------------')
    start_time = time.time()

    n_ids = df['unique_id'].nunique()
    ids = df['unique_id'].unique()
    res_df = pd.DataFrame()

    for i in range(n_ids):

        id = ids[i]
        df_tmp = df[df['unique_id'] == id].sort_values('ds')
        ds = df_tmp['ds'].values
        y = df_tmp['y'].values

        if nobs is not None: # keep only the last nobs observations
            ds = ds[-nobs:]
            y = y[-nobs:]

        module_logger.info(f'Processing unique_id: {id} with {len(y)} observations ({i+1} of {n_ids})...')

        test = BaiPerronTest(y)
        res = test.fit(max_breaks=max_breaks, selection=selection) # 'bic', 'lwz', 'sequential'
        
        break_df = pd.DataFrame({
            'unique_id': id,
            'ds': ds[res.break_indices],
            'break': 'break'
        })
        break_low_df = pd.DataFrame({
            'unique_id': id,
            'ds': ds[[tup[0] for tup in res.break_ci.values()]],
            'break': 'low'
        })
        break_up_df = pd.DataFrame({
            'unique_id': id,
            'ds': ds[[tup[1] for tup in res.break_ci.values()]],
            'break': 'up'
        })

        res_df = pd.concat([res_df, break_df, break_low_df, break_up_df], ignore_index=True)

        del id, df_tmp, ds, y, test, res, break_df, break_low_df, break_up_df
    
    res_df = res_df.sort_values(['unique_id', 'ds']).reset_index(drop=True)
    end_time = time.time()
    tot_time = end_time - start_time
    module_logger.info(f'Total computing time: {tot_time:.1f} seconds')

    module_logger.info('---------------------------- END ----------------------------')

    return res_df
