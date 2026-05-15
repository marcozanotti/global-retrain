import sys
sys.path.insert(0, 'src/Python/utils')
import os
from utilities import (
    get_config, configure_logging, create_logger, stop_logger
)
from evaluate_forecasts import evaluate_model

os.environ['NIXTLA_ID_AS_COL'] = '1'


# config = get_config('config/eval/vn1_weekly_eval.yaml')
# config = get_config('config/eval/m5_daily_eval.yaml')
# config = get_config('config/eval/m4_daily_eval.yaml')
config = get_config('config/eval/hapag_weekly_eval.yaml')
configure_logging(
    config_file = 'config/log_config.yaml', 
    name_list = [
        config['dataset']['dataset_name'], 
        config['dataset']['frequency'], 
        'evaluation'
    ]
)
logger = create_logger()
evaluate_model(config = config)
stop_logger(logger)
