import sys
sys.path.insert(0, 'src/Python/utils')
import os
from utilities import (
    get_config, configure_logging, create_logger, stop_logger
)
from fit_models import retrain_model

os.environ['NIXTLA_ID_AS_COL'] = '1'


config = get_config('config/fit/hapag_weekly_retrain_sf.yaml')
# config = get_config('config/fit/vn1_weekly_retrain_sf.yaml')
# config = get_config('config/fit/m5_daily_retrain_sf.yaml')
# config = get_config('config/fit/m4_daily_retrain_sf.yaml')
configure_logging(
    config_file = 'config/log_config.yaml', 
    name_list = [
        config['dataset']['dataset_name'], 
        config['dataset']['frequency'], 
        'retrain', 'sf'
    ]
)
logger = create_logger()
retrain_model(config = config)
stop_logger(logger)
