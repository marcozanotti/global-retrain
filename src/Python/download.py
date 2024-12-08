
import pandas as pd
from utils.collect_data import *

pd.set_option("display.max_rows", 4)

dataset_name = 'm5'
frequency = 'daily'

download_data(dataset_name, frequency, save = True)
prepare_data(dataset_name, frequency, static_features = True, save = True)

# check
# full dataset
data = get_data(
    path = 'data/m5/', 
    name_list = [dataset_name, frequency, 'prep'],
    ext = '.parquet'
)
# dataset with series of minimum 365 days length)
data = get_data(
    path = 'data/m5/', 
    name_list = [dataset_name, frequency, 'prep'],
    ext = '.parquet', 
    min_series_length = 365 * 2
)
# sampled dataset
data_sample = get_data(
    path = 'data/m5/', 
    name_list = [dataset_name, frequency, 'prep'],
    ext = '.parquet', 
    samples = 5
)
