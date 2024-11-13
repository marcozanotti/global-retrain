
import pandas as pd
from src.Python.utils import *

pd.set_option("display.max_rows", 4)


# Download data -----------------------------------------------------------

download_data('m5', frequency = 'daily', save = True)


# Prepare data ------------------------------------------------------------

prepare_data(
    dataset_name = 'm5', 
    frequency = 'daily', 
    static_features = True,
    save = True
)


# Load data ---------------------------------------------------------------

# full dataset
data = get_data('data/m5/m5_daily_prep.parquet')

# dataset with series of minimum 365 days length)
data = get_data('data/m5/m5_daily_prep.parquet', min_series_length = 365)

# sampled dataset
data_sample = get_data('data/m5/m5_daily_prep.parquet', samples = 1000)
