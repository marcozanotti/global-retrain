
import pandas as pd
from src.Python.utils import *

pd.set_option("display.max_rows", 4)


# Parameters --------------------------------------------------------------

dataset_name = 'm5'
frequency = 'daily'


# Download data -----------------------------------------------------------

download_data(dataset_name, frequency, save = True)


# Prepare data ------------------------------------------------------------

prepare_data(dataset_name, frequency, static_features = True, save = True)


# Load data ---------------------------------------------------------------

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
    min_series_length = 365
)

# sampled dataset
data_sample = get_data(
    path = 'data/m5/', 
    name_list = [dataset_name, frequency, 'prep'],
    ext = '.parquet', 
    samples = 5
)
