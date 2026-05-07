
import pandas as pd
from src.Python.utils.collect_data import get_static_features
from src.Python.utils.utilities import save_data

breaks_df = pd.read_csv('data/hapag_region/hapag_breaks.csv', header = 1, sep = ';', quotechar = '"')
breaks_df = breaks_df[:-1] # drop last row with ###EndofFile### value
breaks_df['unique_id'] = breaks_df['Name'].str.replace(';', '_')
breaks_df = breaks_df[['unique_id', 'Info:AllChangePoints'] + [col for col in breaks_df.columns if 'Date:' in col]]
breaks_df.columns = [col.replace('Date:', '') for col in breaks_df.columns]
breaks_df = breaks_df.melt(id_vars = ['unique_id', 'Info:AllChangePoints'], var_name = 'ds', value_name = 'y')
breaks_df['ds'] = pd.to_datetime(breaks_df['ds'], format = '%Y-%m-%d')
breaks_df = breaks_df.sort_values(['unique_id', 'ds']).reset_index(drop = True)

breaks_info_df = breaks_df[['unique_id', 'Info:AllChangePoints']].drop_duplicates().reset_index(drop = True)
breaks_df = breaks_df[['unique_id', 'ds', 'y']]

breaks_only_df = pd.DataFrame(columns = ['unique_id', 'ds', 'break'])

for id in breaks_info_df['unique_id'].unique():

    # get all change points for the current unique_id
    change_points = breaks_info_df[breaks_info_df['unique_id'] == id]['Info:AllChangePoints'].values[0].split(',')
    change_points = [int(float(cp) - 1) for cp in change_points if cp.strip() != '']

    # filter the breaks_df for the current unique_id and change points ids
    breaks_df_tmp = breaks_df[breaks_df['unique_id'] == id].reset_index(drop = True)
    breaks_df_tmp = breaks_df_tmp[breaks_df_tmp.index.isin(change_points)].reset_index(drop = True)
    breaks_df_tmp = breaks_df_tmp[['unique_id', 'ds']]
    breaks_df_tmp['break'] = 1

    # store the result in a new DataFrame
    breaks_only_df = pd.concat([breaks_only_df, breaks_df_tmp], ignore_index = True)

# merge the breaks_df_tmp with the original breaks_df to get the break column
breaks_df = breaks_df.merge(breaks_only_df, on = ['unique_id', 'ds'], how = 'left')
breaks_df['break'] = breaks_df['break'].fillna(0)

breaks_df = get_static_features(breaks_df, 'hapag')

save_data(breaks_df, ['data', 'hapag_region'], ['hapag_breaks_prep'])



# Feature Importance -------------------------------------------------------------------
import sys
sys.path.insert(0, 'src/Python/utils')
import plotly.express as px
from importance import compute_feature_importance, plot_feature_importance

# Parameters
dataset_name = 'hapag_region'
frequency = 'weekly'
model_names = ['LinearRegression', 'Lasso', 'RandomForestRegressor', 'XGBRegressor', 'LGBMRegressor', 'MLPRegressor', 'NBEATSx', 'NHITS']
retrain_scenarios = [104] # [1, 2, 3, 4, 6, 8, 10, 13, 26, 52, 104]

f_imp_df = compute_feature_importance(
    dataset_name = dataset_name,
    frequency = frequency,
    model_names = ['LinearRegression', 'Lasso', 'RandomForestRegressor', 'XGBRegressor', 'LGBMRegressor'],
    retrain_scenarios = [104]
)

plot_feature_importance(f_imp_df, retrain_scenario=104, top_n=20).show()


# Evaluate performance
