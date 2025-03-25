
import sys
sys.path.insert(0, 'src/Python/utils')
import os
from utilities import (
    get_config, configure_logging, create_logger, stop_logger
)
from evaluate_stability import evaluate_dataset_stability

os.environ['NIXTLA_ID_AS_COL'] = '1'
config = get_config('config/stab/stab_dataset_config.yaml')
configure_logging(
    config_file = 'config/log_config.yaml', 
    name_list = [
        config['dataset']['dataset_names'], 
        config['dataset']['frequencies'], 
        'stability', 'dataset'
    ]
)
logger = create_logger()

evaluate_dataset_stability(config = config)

stop_logger(logger)
