
import os
from src.Python.utils.utilities import *
from src.Python.utils.evaluate_forecasts import evaluate_model

os.environ['NIXTLA_ID_AS_COL'] = '1'
config = get_config('config/eval_config.yaml')
configure_logging(
    config_file = 'config/log_config.yaml', 
    name_list = [
        config['dataset']['dataset_name'], 
        config['dataset']['frequency'], 
        'eval'
    ]
)
logger = create_logger()

evaluate_model(config = config)

stop_logger(logger)
