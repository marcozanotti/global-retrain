import sys
sys.path.insert(0, 'src/Python/utils')
import os
import logging
from utilities import (
    get_config, configure_logging, create_logger, stop_logger
)
from fit_models import retrain_model
from predictions import combine_model_predictions, evaluate_model_predictions, combine_dataset_predictions

os.environ['NIXTLA_ID_AS_COL'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

configure_logging(
    config_file = 'config/log_config.yaml', 
    name_list = ['hapag_region', 'weekly', 'experiment']
)
logger = create_logger()

logging.getLogger('pytorch_lightning.utilities').setLevel(logging.ERROR)
logging.getLogger('lightning_fabric.utilities').setLevel(logging.ERROR)

config_retrain_ml = get_config('config/fit/hapag_weekly_retrain_ml.yaml')
retrain_model(config = config_retrain_ml)

# config_retrain_dl = get_config('config/fit/hapag_weekly_retrain_dl.yaml')
# retrain_model(config = config_retrain_dl)

config_preds_eval = get_config('config/preds/hapag_weekly_preds.yaml')
config_combine_preds = get_config('config/preds/dataset_preds.yaml')

combine_model_predictions(config = config_preds_eval)
evaluate_model_predictions(config = config_preds_eval)

combine_dataset_predictions(config = config_combine_preds)

stop_logger(logger)
