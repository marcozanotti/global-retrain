import os
from src.Python.utils.utilities import *
from src.Python.utils.fit_models import retrain_model

os.environ['NIXTLA_ID_AS_COL'] = '1'
cfg = get_config('config/retrain_config_dl.yaml')
config = cfg
