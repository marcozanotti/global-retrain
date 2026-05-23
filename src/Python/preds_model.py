import sys
sys.path.insert(0, 'src/Python/utils')
import os
from utilities import (
    get_config, configure_logging, create_logger, stop_logger
)
from predictions import combine_model_predictions, evaluate_model_predictions

os.environ['NIXTLA_ID_AS_COL'] = '1'


config = get_config('config/preds/hapag_weekly_preds.yaml')
configure_logging(
    config_file = 'config/log_config.yaml', 
    name_list = [
        config['dataset']['dataset_name'], 
        config['dataset']['frequency'], 
        'predictions'
    ]
)
logger = create_logger()
combine_model_predictions(config = config)
evaluate_model_predictions(config = config)
stop_logger(logger)
