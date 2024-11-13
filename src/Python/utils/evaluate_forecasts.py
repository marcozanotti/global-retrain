
import os
import pandas as pd

dataset_name = 'm5'
frequency = 'daily'
analysis = 'outsample'
model = 'LinearRegression'

ext = '.parquet'

files = os.listdir(f'results/{dataset_name}/')
files

list(filter(lambda x:ext in x, files))
list(filter(lambda x:analysis in x, files))
list(filter(lambda x:model in x, files))

list(filter(lambda x:model in x and analysis in x, files))

load_data(
    path = f'results/{dataset_name}/',
    name_list = [],
    ext = '.parquet'
)


out_sample_df = pd.read_parquet('results/m5/m5_daily_outsample_LinearRegression_56_28_7.parquet')
out_sample_df

# TODO:
# - impostare valutazione point forecast per singolo file
# - usare ME, MAE, RMSE, MASE, RMSSE
# - generalizzare su ciascun file


