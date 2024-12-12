import os
from src.Python.utils.utilities import *
from src.Python.utils.fit_ensembles import fit_ensembles

os.environ['NIXTLA_ID_AS_COL'] = '1'
config = get_config('config/ensemble_config.yaml')
configure_logging(
    config_file = 'config/log_config.yaml', 
    name_list = [
        config['dataset']['dataset_name'], 
        config['dataset']['frequency'], 
        'retrain'
    ]
)
logger = create_logger()

fit_ensembles(config = config)

stop_logger(logger)
