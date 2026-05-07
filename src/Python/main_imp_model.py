import sys
sys.path.insert(0, 'src/Python/utils')
import os
from utilities import (
    get_config, configure_logging, create_logger, stop_logger
)
from importance import compute_feature_importance

os.environ['NIXTLA_ID_AS_COL'] = '1'
config = get_config('config/importance/imp_hapag_weekly_config.yaml')

configure_logging(
    config_file = 'config/log_config.yaml', 
    name_list = [
        config['dataset']['dataset_name'], 
        config['dataset']['frequency'], 
        'importance'
    ]
)
logger = create_logger()

compute_feature_importance(config = config)

stop_logger(logger)
