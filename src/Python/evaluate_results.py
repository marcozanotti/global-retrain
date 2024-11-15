
import os
import pandas as pd

from functools import partial
from utilsforecast.losses import bias, mae, mse, rmse, mase

from src.Python.utils import *

pd.set_option("display.max_rows", 4)
os.environ['NIXTLA_ID_AS_COL'] = '1'


# Evaluation --------------------------------------------------------------

out_sample_df = pd.read_parquet('results/m5/m5_daily_outsample_LinearRegression_56_28_7.parquet') \
    .reset_index(drop = True)
metrics = [bias, mae, mse, rmse, partial(mase, seasonality = 7)]

evaluation_df = evaluate_point_forecasts(out_sample_df, metrics = metrics, train_df = data)
evaluation_df_agg = aggregate_data(
    evaluation_df, 
    group_columns = ['sample', 'method', 'test_window', 'horizon','retrain_window']
)

time_df = pd.read_parquet('results/m5/m5_daily_time_LinearRegression_56_28_7.parquet') \
    .reset_index(drop = True)
time_df
aggregate_data(
    time_df, 
    group_columns = ['method', 'test_window', 'horizon','retrain_window'],
    aggregate_function = 'sum'
)


# Plotting ----------------------------------------------------------------



# TODO:
# - RMSSE
# - generalizzare su ciascun file