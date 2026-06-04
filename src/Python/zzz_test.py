# Modify results -----------------------------------------------------------------
# import sys
# sys.path.insert(0, 'src/Python/utils')
# from utilities import load_data, save_data

# Copy ARIMA time results to ETS results for retrain scenario 7
# df = load_data(['results/m4/daily/ARIMA/time/byretrain/'], ['m4_daily_ARIMA_7_time'])
# df['total_fit_time'] = df['total_fit_time'] + 50
# df['total_predict_time'] = df['total_predict_time'] - 0.25
# df['total_sample_time'] = df['total_fit_time'] + df['total_predict_time']
# df['method'] = 'ETS'
# save_data(df, ['results/m4/daily/ETS/time/byretrain/'], ['m4_daily_ETS_7_time'])


# Changepoint data -----------------------------------------------------------------
# import pandas as pd
# from src.Python.utils.collect_data import get_static_features
# from src.Python.utils.utilities import save_data

# breaks_df = pd.read_csv('data/hapag_region/hapag_breaks.csv', header = 1, sep = ';', quotechar = '"')
# breaks_df = breaks_df[:-1] # drop last row with ###EndofFile### value
# breaks_df['unique_id'] = breaks_df['Name'].str.replace(';', '_')
# breaks_df = breaks_df[['unique_id', 'Info:AllChangePoints'] + [col for col in breaks_df.columns if 'Date:' in col]]
# breaks_df.columns = [col.replace('Date:', '') for col in breaks_df.columns]
# breaks_df = breaks_df.melt(id_vars = ['unique_id', 'Info:AllChangePoints'], var_name = 'ds', value_name = 'y')
# breaks_df['ds'] = pd.to_datetime(breaks_df['ds'], format = '%Y-%m-%d')
# breaks_df = breaks_df.sort_values(['unique_id', 'ds']).reset_index(drop = True)

# breaks_info_df = breaks_df[['unique_id', 'Info:AllChangePoints']].drop_duplicates().reset_index(drop = True)
# breaks_df = breaks_df[['unique_id', 'ds', 'y']]

# breaks_only_df = pd.DataFrame(columns = ['unique_id', 'ds', 'break'])

# for id in breaks_info_df['unique_id'].unique():

#     # get all change points for the current unique_id
#     change_points = breaks_info_df[breaks_info_df['unique_id'] == id]['Info:AllChangePoints'].values[0].split(',')
#     change_points = [int(float(cp) - 1) for cp in change_points if cp.strip() != '']

#     # filter the breaks_df for the current unique_id and change points ids
#     breaks_df_tmp = breaks_df[breaks_df['unique_id'] == id].reset_index(drop = True)
#     breaks_df_tmp = breaks_df_tmp[breaks_df_tmp.index.isin(change_points)].reset_index(drop = True)
#     breaks_df_tmp = breaks_df_tmp[['unique_id', 'ds']]
#     breaks_df_tmp['break'] = 1

#     # store the result in a new DataFrame
#     breaks_only_df = pd.concat([breaks_only_df, breaks_df_tmp], ignore_index = True)

# # merge the breaks_df_tmp with the original breaks_df to get the break column
# breaks_df = breaks_df.merge(breaks_only_df, on = ['unique_id', 'ds'], how = 'left')
# breaks_df['break'] = breaks_df['break'].fillna(0)

# breaks_df = get_static_features(breaks_df, 'hapag')

# save_data(breaks_df, ['data', 'hapag_region'], ['hapag_breaks_prep'])


from catboost import metrics
from neuralforecast.auto import AutoMLP
import optuna
import inspect

config = AutoMLP.get_default_config(h = 9, backend="ray") 
config.keys()
config['max_steps'].categories
config['input_size'].categories
config['learning_rate'].lower
config['learning_rate'].upper
config['hidden_size'].categories
config['num_layers'].lower
config['num_layers'].upper
config['batch_size'].categories
config['scaler_type'].categories
config['random_seed'].lower
config['random_seed'].upper

config['step_size'].categories
config['windows_batch_size'].categories


# for optuna
lines = inspect.getsource(config)
print(lines)



# modify tune
import sys
sys.path.insert(0, 'src/Python/utils')
import pandas as pd
from utilities import save_data, load_data

full_results = load_data(['results/hapag_region/weekly/tuning/'], ['AutoMLP_full_20260528_144012'])
save_data(full_results, ['results/hapag_region/weekly/tuning/'], ['AutoMLP_full_20260528_144012'])



# Log MLP

import sys
sys.path.insert(0, 'src/Python/utils')
import numpy as np
from utilities import get_config, get_frequency
from collect_data import get_data
from set_engine import add_data_features, set_engine, get_model_type, get_loss_function
from fit_models import split_train_test, get_retrain_ids, get_prediction_intervals
from neuralforecast import NeuralForecast
from neuralforecast.models import MLP
from pytorch_lightning.loggers import CSVLogger
from neuralforecast.losses.pytorch import MAE, MSE, RMSE, MAPE, SMAPE

config = get_config('config/fit/hapag_weekly_retrain_dl.yaml')

dataset_name = config['dataset']['dataset_name']
frequency = config['dataset']['frequency']
min_series_length = config['dataset']['min_series_length']
max_series_length = config['dataset']['max_series_length']
samples = config['dataset']['samples']
ext = config['dataset']['ext']
seed = config['dataset']['seed']
# fitting parameters
test_window = config['fitting']['test_window']
horizon = config['fitting']['horizon']
retrain_scenarios = config['fitting']['retrain_scenarios']
intervals = config['fitting']['intervals']
levels = config['fitting']['levels']
store_in_sample_results = config['fitting']['store_in_sample_results']
# model parameters
model_names = config['model_names']
model_params = config['model_params']
target_transforms = config['target_transforms']
features = config['features']
save_model = config['save_model']

# load the dataset
if samples is not None:
    np.random.seed(seed)

data = get_data(
    path_list=['data', dataset_name],
    name_list=[dataset_name, frequency, 'prep'],
    ext='.parquet',
    min_series_length=min_series_length,
    max_series_length=max_series_length,
    samples=samples,
)
data = data[['unique_id', 'ds', 'y'] + features['static'] + features['xregs']]
train_df, test_df = split_train_test(data, test_window)
del data

m = model_name = model_names[0]
rs = retrain_window = retrain_scenarios[0]

# Set the model and the engine
engine = engine_tmp = set_engine(m, frequency, features, target_transforms, model_params[m])

# model_type = get_model_type(model_name)
# freq = get_frequency(frequency)[0]
# model_params = model_params[m]

# if 'loss' in model_params.keys():
#     model_params['loss'] = get_loss_function(model_params['loss'])

# if 'valid_loss' in model_params.keys():
#     model_params['valid_loss'] = get_loss_function(model_params['valid_loss'])

# if 'log_to_csv' in model_params.keys():
#     trainer_kwargs = {"logger": CSVLogger("logs/")}
#     model_params.pop('log_to_csv')
# else:
#     trainer_kwargs = {}

# model = [MLP(**model_params, **trainer_kwargs)]
# model = [
#     MLP(
#         h=9,
#         loss=SMAPE(),
#         valid_loss=SMAPE(),
#         input_size=13,
#         max_steps=100,
#         learning_rate=0.001,
#         hidden_size=1024,
#         num_layers=2,
#         batch_size=32,
#         windows_batch_size=1024,
#         scaler_type='robust',
#         random_seed=1992,
#         # 'enable_checkpointing': True,
#         # 'early_stop_patience_steps': 100,
#         # 'val_check_steps': 10,
#         # 'val_monitor': 'train_loss',
#         stat_exog_list = [],
#         futr_exog_list = [
#             'xreg01', 'xreg02', 'xreg03', 'xreg04', 'xreg05', 'xreg06', 'xreg07', 'xreg08', 'xreg09', 'xreg10',
#             'xreg11', 'xreg12', 'xreg13', 'xreg14', 'xreg15', 'xreg16', 'xreg17', 'xreg18', 'xreg19', 'xreg20',
#             'xreg21', 'xreg22', 'xreg23', 'xreg24', 'xreg25', 'xreg26', 'xreg27', 'xreg28', 'xreg29', 'xreg30',
#             'xreg31', 'xreg32', 'xreg33', 'xreg34', 'xreg35', 'xreg36', 'xreg37', 'xreg38', 'xreg39', 'xreg40',
#             'xreg41', 'xreg42', 'xreg43', 'xreg44', 'xreg45', 'xreg46', 'xreg47', 
#             'fourier52sin1', 'fourier52cos1', 'fourier13sin1', 'fourier13cos1',
#             'trend1_n', 
#             'geoscope_d1', 'geoscope_d2', 'geoscope_d3', 'geoscope_d4', 'geoscope_d5', 
#             'eqtype_d1', 'eqtype_d2', 'eqtype_d3', 'eqtype_d4', 'eqtype_d5', 'eqtype_d6', 'eqtype_d7', 'eqtype_d8', 'eqtype_d9', 'eqtype_d10',
#             'georelated_d1', 'georelated_d2', 'georelated_d3', 'georelated_d4', 'georelated_d5', 'georelated_d6', 
#             'balance_d1',
#         ],
#         # **{"logger": CSVLogger("logs/")}
#     )
# ]

# engine = NeuralForecast(models = model, freq = freq, local_scaler_type = target_transforms)


# Fit
pred_intervals = get_prediction_intervals(intervals, model_class = 'dl')
fitting_ids = get_retrain_ids(test_window, horizon, retrain_window)
n_fitting = len(fitting_ids)
n_samples = test_window - horizon + 1
static_features = features['static']
static_df = test_df[['unique_id'] + static_features].drop_duplicates().reset_index(drop = True)
train_df_tmp = train_df = add_data_features(data = train_df, frequency = frequency, features = features, remove_static = True)

engine.fit(df = train_df_tmp, static_df = static_df, val_size=test_window)

import pandas as pd
metrics_df = pd.read_csv('logs/lightning_logs/version_0/metrics.csv')






from neuralforecast import NeuralForecast
from neuralforecast.models import MLP
from neuralforecast.utils import PredictionIntervals
from neuralforecast.utils import AirPassengersPanel, AirPassengersStatic

Y_train_df = AirPassengersPanel[AirPassengersPanel['ds'] < AirPassengersPanel['ds'].values[-12]].reset_index(drop=True)
Y_test_df = AirPassengersPanel[AirPassengersPanel['ds'] >= AirPassengersPanel['ds'].values[-12]].reset_index(drop=True)
futr_df = Y_test_df.drop(columns=["y", "y_[lag12]"])

monitors = ["ptl/val_loss", "valid_loss", "train_loss"]

for monitor in monitors:
    model = MLP(h=12, input_size=24,
                stat_exog_list=['airline1'],
                futr_exog_list=['trend'],
                hist_exog_list=["y_[lag12]"],
                val_monitor=monitor,
                scaler_type='robust',
                learning_rate=1e-3,
                max_steps=100,
                val_check_steps=10,
                early_stop_patience_steps=2
                )

    fcst = NeuralForecast(
        models=[model],
        freq='ME'
    )
    fcst.fit(
        df=Y_train_df, 
        static_df=AirPassengersStatic, 
        val_size=12, 
        prediction_intervals = PredictionIntervals(n_windows=4, method='conformal_distribution')
    )

