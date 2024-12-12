import os
from src.Python.utils.utilities import *
from src.Python.utils.fit_models import retrain_model

os.environ['NIXTLA_ID_AS_COL'] = '1'
cfg = get_config('config/retrain_config.yaml')
configure_logging(
    config_file = 'config/log_config.yaml', 
    name_list = [cfg['dataset_name'], cfg['frequency'], 'retrain']
)
logger = create_logger()

retrain_model(config = cfg)

stop_logger(logger)
