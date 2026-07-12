import sys
sys.path.insert(0, 'src/Python/utils')
from utilities import configure_logging, create_logger, stop_logger, load_data, save_data
from structural_breaks import get_breaks
import pandas as pd

configure_logging(
    config_file = 'config/log_config.yaml', 
    name_list = ['hapag_region', 'weekly', 'breaks']
)
logger = create_logger()
data = load_data(['data/hapag_region/'], ['hapag_region_weekly_prep'])
# xreg_cols = [col for col in data.columns if col.startswith('xreg')]
# data['xreg'] = data[xreg_cols].apply(lambda row: 1 if (row > 0).any() else 0, axis=1)
# data = data[['unique_id', 'ds', 'y', 'xreg']]
data = data[['unique_id', 'ds', 'y']]
data['ds'] = pd.to_datetime(data['ds'])
nobs = 52*5
criterion = 'bic' # 'bic', 'lwz', 'sequential'
res = get_breaks(df=data, nobs=nobs, max_breaks=10, selection=criterion, exog=None) 
stop_logger(logger)

# add sample information
test_window = 104
sample_df = data[data['ds'] >= (data['ds'].max() - pd.DateOffset(weeks=(test_window-1)))][['ds']] \
    .drop_duplicates() \
    .reset_index(drop=True)
sample_df['sample'] = sample_df.index.astype(int)
res = res.merge(sample_df, on='ds', how='left')
res['sample'] = res['sample'].fillna(-1).astype(int)

data_br = data.merge(res, on=['unique_id', 'ds'], how='left')
data_br['break'] = data_br['break'].fillna('no break')
save_data(
    data_br, 
    path_list=['results/hapag_region/weekly/breaks/'], 
    name_list=[f'hapag_region_weekly_breaks_full_{nobs}_{criterion}']
)

data_br_light = res[res['break'] == 'break'][['unique_id', 'ds', 'sample']]
data_br_light.reset_index(drop=True, inplace=True)
save_data(
    data_br_light, 
    path_list=['results/hapag_region/weekly/breaks/'], 
    name_list=[f'hapag_region_weekly_breaks_{nobs}_{criterion}']
)
