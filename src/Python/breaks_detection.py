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
data = data[['unique_id', 'ds', 'y']]
data['ds'] = pd.to_datetime(data['ds'])
res = get_breaks(df=data, nobs=104, max_breaks=5, selection='bic')
stop_logger(logger)

data_br = data.merge(res, on=['unique_id', 'ds'], how='left')
data_br['break'] = data_br['break'].fillna('no break')
save_data(data_br, path_list=['data/hapag_region/'], name_list=['hapag_region_weekly_breaks_104'])
