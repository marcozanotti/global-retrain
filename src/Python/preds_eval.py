import sys
sys.path.insert(0, 'src/Python/utils')
import os
from utilities import (
    get_config, configure_logging, create_logger, stop_logger
)
from predictions import evaluate_predictions

os.environ['NIXTLA_ID_AS_COL'] = '1'
config = get_config('config/preds/hapag_weekly_eval.yaml')

configure_logging(
    config_file = 'config/log_config.yaml', 
    name_list = [
        config['dataset']['dataset_name'], 
        config['dataset']['frequency'], 
        'predictions_evaluation'
    ]
)
logger = create_logger()

evaluate_predictions(config = config)

stop_logger(logger)
