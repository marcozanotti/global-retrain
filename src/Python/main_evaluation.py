
import os
from src.Python.utils.utilities import *
from src.Python.utils.evaluate_forecasts import evaluate_model

os.environ['NIXTLA_ID_AS_COL'] = '1'
cfg = get_config('config/eval_config.yaml')
configure_logging(
    config_file = 'config/log_config.yaml', 
    name_list = [cfg['dataset_name'], cfg['frequency'], 'eval']
)
logger = create_logger()

evaluate_model(config = cfg)

stop_logger(logger)
