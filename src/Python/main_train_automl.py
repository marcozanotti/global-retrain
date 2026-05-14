import sys
sys.path.insert(0, 'src/Python/utils')
import os
from utilities import (
    get_config, configure_logging, create_logger, stop_logger
)
from fit_models import fit_auto_model
import optuna

os.environ['NIXTLA_ID_AS_COL'] = '1'
optuna.logging.set_verbosity(optuna.logging.ERROR)

config = get_config('config/fit/TEST_hapag_tuning_ml.yaml')

configure_logging(
    config_file = 'config/log_config.yaml', 
    name_list = [
        config['dataset']['dataset_name'], 
        config['dataset']['frequency'], 
        'tuning'
    ]
)
logger = create_logger()

fit_auto_model(config = config)

stop_logger(logger)
