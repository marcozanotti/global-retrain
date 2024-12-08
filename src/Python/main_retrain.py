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

# if cfg['combine_results']:

#     if cfg['store_in_sample_results']:
#         # combine and save the insample tmp files
#         combine_and_save_files(
#             path_list_to_read = ['results', dataset_name, model_name, retrain_window, 'insample', 'tmp'],
#             path_list_to_write = ['results', dataset_name, model_name, retrain_window, 'insample'],
#             name_list = [dataset_name, frequency, model_name, retrain_window, 'insample'],
#             ext = ext
#         )
    
#     # combine and save the outsample tmp files
#     combine_and_save_files(
#         path_list_to_read = ['results', dataset_name, model_name, retrain_window, 'outsample', 'tmp'],
#         path_list_to_write = ['results', dataset_name, model_name, retrain_window, 'outsample'],
#         name_list = [dataset_name, frequency, model_name, retrain_window, 'outsample'],
#         ext = ext
#     )