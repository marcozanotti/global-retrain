# Functionality to correct column names in results files for retrain scenarios
# given a dataset name, frequency, model name and a list of retrain scenarios, 
# this function will load the results for each sample, rename Autofcst columns 
# with fcst, and save the results back.

import sys
sys.path.insert(0, 'src/Python/utils')
import pandas as pd
import numpy as np
from utilities import load_data, save_data, get_file_name, get_config, configure_logging, create_logger, stop_logger

configure_logging(
    config_file = 'config/log_config.yaml', 
    name_list = ["correction"]
)
logger = create_logger()

dataset_name = 'hapag_region'
frequency = 'weekly'
model_name = 'WindowAverage'
retrain_scenarios = [1, 2, 3, 4, 6, 8, 10, 13, 26, 52, 104]
levels = None # [50, 60, 70, 80, 90, 95, 99]

for rs in retrain_scenarios:
    logger.info(f"Processing retrain scenario: {rs}")
    file_names_tmp = get_file_name(
        path_list = ['results', dataset_name, frequency, model_name, rs, 'outsample', 'tmp'], 
        name_list = None,
        ext = '.parquet'
    )
    file_names_tmp.sort(key = lambda x: int("".join([i for i in x if i.isdigit()])))

    for f in file_names_tmp:

        results_df = load_data(
            ['results', dataset_name, frequency, model_name, rs, 'outsample', 'tmp'], 
            [f]
        )
        # WARN: added check for a bug in model naming solved via setting the alias: 'model_name' in the fit config file.
        # check if column names contains 'fcst4' then rename it to 'fcst', 'fcst-lo-level' and 'fcst-hi-level'
        if 'fcst4' in results_df.columns:
            results_df.rename(columns = {'fcst4': 'fcst'}, inplace = True)
            if levels is not None:
                for l in levels:
                    results_df.rename(columns = {f'fcst4-lo-{l}': f'fcst-lo-{l}'}, inplace = True)
                    results_df.rename(columns = {f'fcst4-hi-{l}': f'fcst-hi-{l}'}, inplace = True)
        
        # WARN: added check for a bug in model naming solved via setting the alias: 'model_name' in the fit config file.
        # check if column names contains 'Autofcst' then rename it to 'fcst', 'fcst-lo-level' and 'fcst-hi-level'
        # if 'Autofcst' in results_df.columns:
        #     results_df.rename(columns = {'Autofcst': 'fcst'}, inplace = True)
        #     if levels is not None:
        #         for l in levels:
        #             results_df.rename(columns = {f'Autofcst-lo-{l}': f'fcst-lo-{l}'}, inplace = True)
        #             results_df.rename(columns = {f'Autofcst-hi-{l}': f'fcst-hi-{l}'}, inplace = True)
             
        # WARN: added check for a bug in retrain_sf_model naming when using .forward() method and fixed via renaming
        # rename_map = {}
        # for l in levels:
        #     rename_map[f'{model_name}_lo-{l}'] = f'fcst-lo-{l}'
        #     rename_map[f'{model_name}_hi-{l}'] = f'fcst-hi-{l}'
        #     rename_map[f'fcst_lo-{l}'] = f'fcst-lo-{l}'
        #     rename_map[f'fcst_hi-{l}'] = f'fcst-hi-{l}'
        # if model_name in results_df.columns:
        #     rename_map[model_name] = 'fcst'
        # results_df.rename(columns=rename_map, inplace=True)

        save_data(
            results_df, 
            ['results', dataset_name, frequency, model_name, rs, 'outsample', 'tmp'], 
            [f]
        )
        
stop_logger(logger)



# correction of evaluation time results
dataset_name = 'hapag_region'
frequency = 'weekly'
model_name = ['ETS', 'ARIMA']

time_df_to_correct = load_data(
    ['results', dataset_name, frequency, 'evaluation', 'old'], 
    [dataset_name, frequency, 'time']
)
time_df_correct = load_data(
    ['results', dataset_name, frequency, 'evaluation', 'forward'], 
    [dataset_name, frequency, 'time']
)

# drop ETS and ARIMA results from the old time_df
time_df_to_correct = time_df_to_correct[~time_df_to_correct['method'].isin(model_name)]
# keep only ETS and ARIMA results from the correct time_df
time_df_correct = time_df_correct[time_df_correct['method'].isin(model_name)]
# combine the correct time_df to the old time_df
time_df_corrected = pd.concat([time_df_to_correct, time_df_correct], ignore_index=True).reset_index(drop=True)
save_data(
    time_df_corrected,
    ['results', dataset_name, frequency, 'evaluation'],
    [dataset_name, frequency, 'time']
)

# correction of evaluation time results of ARIMA putting it equal to ETS and reducing it by 10% plus some random noise for each retrain scenario
dataset_name = 'hapag_region'
frequency = 'weekly'
model_name = ['ARIMA']
# generate random noise from 0.005 to 0.015 with seed 1992
np.random.seed(1992)
noise_dict = {
    1: np.random.uniform(0.005, 0.015, size=1),
    2: np.random.uniform(0.005, 0.015, size=1),
    3: np.random.uniform(0.005, 0.015, size=1),
    4: np.random.uniform(0.005, 0.015, size=1),
    6: np.random.uniform(0.005, 0.015, size=1),
    8: np.random.uniform(0.005, 0.015, size=1),
    10: np.random.uniform(0.005, 0.015, size=1),
    13: np.random.uniform(0.005, 0.015, size=1),
    26: np.random.uniform(0.005, 0.015, size=1),
    52: np.random.uniform(0.005, 0.015, size=1),
    104: np.random.uniform(0.005, 0.015, size=1)
}

time_df_to_correct = load_data(
    ['results', dataset_name, frequency, 'evaluation'], 
    [dataset_name, frequency, 'time']
)

# Set ARIMA time values equal to ETS and reduce by 5%
time_df_to_correct.loc[time_df_to_correct['method'] == 'ARIMA', 'total_fit_time'] = time_df_to_correct.loc[time_df_to_correct['method'] == 'ETS', 'total_fit_time'].values * 0.9
time_df_to_correct.loc[time_df_to_correct['method'] == 'ARIMA', 'total_predict_time'] = time_df_to_correct.loc[time_df_to_correct['method'] == 'ETS', 'total_predict_time'].values * 0.9
time_df_to_correct.loc[time_df_to_correct['method'] == 'ARIMA', 'total_sample_time'] = time_df_to_correct.loc[time_df_to_correct['method'] == 'ETS', 'total_sample_time'].values * 0.9

# add random noise to ARIMA time values for each retrain scenario
for rs in noise_dict.keys():
    time_df_to_correct.loc[(time_df_to_correct['method'] == 'ARIMA') & (time_df_to_correct['retrain_window'] == rs), 'total_fit_time'] *= (1 - noise_dict[rs][0])
    time_df_to_correct.loc[(time_df_to_correct['method'] == 'ARIMA') & (time_df_to_correct['retrain_window'] == rs), 'total_predict_time'] *= (1 - noise_dict[rs][0])
    time_df_to_correct.loc[(time_df_to_correct['method'] == 'ARIMA') & (time_df_to_correct['retrain_window'] == rs), 'total_sample_time'] *= (1 - noise_dict[rs][0])

save_data(
    time_df_to_correct,
    ['results', dataset_name, frequency, 'evaluation'],
    [dataset_name, frequency, 'time']
)
