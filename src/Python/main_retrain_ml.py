import os
from src.Python.utils.utilities import (
    get_config, configure_logging, create_logger, stop_logger
)
from src.Python.utils.fit_models import retrain_model

os.environ['NIXTLA_ID_AS_COL'] = '1'
# config = get_config('config/retrain_ml_config.yaml')
# config = get_config('config/vn1_monthly_retrain_ml_config.yaml')
# config = get_config('config/vn1_weekly_retrain_ml_config.yaml')
# config = get_config('config/m5_monthly_retrain_ml_config.yaml')
# config = get_config('config/m5_weekly_retrain_ml_config.yaml')
config = get_config('config/m5_daily_retrain_ml_config.yaml')
configure_logging(
    config_file = 'config/log_config.yaml', 
    name_list = [
        config['dataset']['dataset_name'], 
        config['dataset']['frequency'], 
        'retrain', 'ml'
    ]
)
logger = create_logger()

retrain_model(config = config)

stop_logger(logger)
