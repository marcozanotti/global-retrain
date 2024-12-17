import os
import logging
from src.Python.utils.utilities import (
    get_config, configure_logging, create_logger, stop_logger
)
from src.Python.utils.fit_models import retrain_model

os.environ['NIXTLA_ID_AS_COL'] = '1'
config = get_config('config/retrain_dl_config.yaml')

configure_logging(
    config_file = 'config/log_config.yaml', 
    name_list = [
        config['dataset']['dataset_name'], 
        config['dataset']['frequency'], 
        'retrain', 'dl'
    ]
)
logger = create_logger()
logging.getLogger('pytorch_lightning.utilities').setLevel(logging.ERROR)
logging.getLogger('lightning_fabric.utilities').setLevel(logging.ERROR)

retrain_model(config = config)

stop_logger(logger)


