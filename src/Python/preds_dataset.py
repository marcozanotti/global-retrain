import sys
sys.path.insert(0, 'src/Python/utils')
import os
from utilities import (
    get_config, configure_logging, create_logger, stop_logger
)
from predictions import combine_dataset_predictions

os.environ['NIXTLA_ID_AS_COL'] = '1'


config = get_config('config/preds/dataset_preds.yaml')
configure_logging(
    config_file = 'config/log_config.yaml', 
    name_list = [
        config['dataset']['dataset_names'], 
        config['dataset']['frequencies'], 
        'predictions', 'dataset'
    ]
)
logger = create_logger()
combine_dataset_predictions(config = config)
stop_logger(logger)
