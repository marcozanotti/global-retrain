
import sys
sys.path.insert(0, 'src/Python/utils')
import os
from utilities import (
    get_config, configure_logging, create_logger, stop_logger
)
from evaluate_stability import evaluate_model_stability

os.environ['NIXTLA_ID_AS_COL'] = '1'
config = get_config('config/stab/vn1_weekly.yaml')
# config = get_config('config/stab/m5_daily.yaml')
# config = get_config('config/stab/m4_daily.yaml')
configure_logging(
    config_file = 'config/log_config.yaml', 
    name_list = [
        config['dataset']['dataset_name'], 
        config['dataset']['frequency'], 
        'stability'
    ]
)
logger = create_logger()

evaluate_model_stability(config = config)

stop_logger(logger)
