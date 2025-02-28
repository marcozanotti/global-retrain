
import sys
sys.path.insert(0, 'src/Python/utils')
import os
import logging
from utilities import (
    get_config, configure_logging, create_logger, stop_logger
)
from fit_models import retrain_model

os.environ['NIXTLA_ID_AS_COL'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# config = get_config('config/fit/TEST_retrain_dl_vn1_monthly_config.yaml')
# config = get_config('config/fit/retrain_dl_vn1_monthly_config.yaml')
# config = get_config('config/fit/retrain_dl_vn1_weekly_config.yaml')
# config = get_config('config/fit/retrain_dl_m5_monthly_config.yaml')
# config = get_config('config/fit/retrain_dl_m5_weekly_config.yaml')
config = get_config('config/fit/retrain_dl_m5_daily_config.yaml')
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


