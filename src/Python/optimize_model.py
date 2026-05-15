import sys
sys.path.insert(0, 'src/Python/utils')
import os
from utilities import (
    get_config, configure_logging, create_logger, stop_logger
)
from tune_models import fit_auto_model
import optuna

os.environ['NIXTLA_ID_AS_COL'] = '1'
optuna.logging.set_verbosity(optuna.logging.ERROR)
config = get_config('config/tune/hapag_weekly_tune_ml.yaml')
# config = get_config('config/tune/hapag_weekly_tune_dl.yaml')
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
