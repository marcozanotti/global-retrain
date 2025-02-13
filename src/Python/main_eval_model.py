
import os
from src.Python.utils.utilities import (
    get_config, configure_logging, create_logger, stop_logger
)
from src.Python.utils.evaluate_forecasts import evaluate_model

os.environ['NIXTLA_ID_AS_COL'] = '1'
# config = get_config('config/eval/eval_vn1_monthly_config.yaml')
# config = get_config('config/eval/eval_vn1_weekly_config.yaml')
# config = get_config('config/eval/eval_m5_monthly_config.yaml')
# config = get_config('config/eval/eval_m5_weekly_config.yaml')
config = get_config('config/eval/eval_m5_daily_config.yaml')
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
