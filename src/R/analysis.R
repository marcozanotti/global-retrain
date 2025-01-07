
install.packages('tidyverse')
install.packages('DT')
install.packages('patchwork')
install.packages('reticulate')

library(tidyverse)
library(DT)
library(patchwork)
library(reticulate)

reticulate::source_python('src/Python/utils/utilities.py')
source('src/R/utils.R')



# Parameters --------------------------------------------------------------

dataset_name = 'vn1'
frequency = 'monthly'
evaluation_type = 'overlap'
ext = '.parquet'

retrain_scenarios = c(
  1, 
  2, 
  3,
  4, 
  5,
  6,
  9,
  12,
  15,
  18
)
retrain_scenarios = c(
  1, 
  2, 
  3,
  4, 
  6,
  8,
  10,
  13,
  26,
  52
)

metric = 'rm_mse' # 'bias', 'mae', 'mse', 'rmse', 'mase', 'msse', 'rmsse', 'mql', 'cov', 'scrps'
time_metric = 'total_sample_time'

model_type_levels = c('SF', 'ML', 'DL', 'ENS')
method_levels = list(
  'LR' = 'LinearRegression',
  'RF' = 'RandomForestRegressor',
  'XGB' = 'XGBRegressor',
  'LGBM' = 'LGBMRegressor',
  'CatBoost' = 'CatBoostRegressor'
)



# Load data ---------------------------------------------------------------

# eval_df.shape[0] = n_series * n_retrain_scenarios * n_models = 30.000 * 10 * 10
eval_df = load_data(
    path_list = c('results', dataset_name, frequency, 'evaluation'),
    name_list = c(dataset_name, frequency, 'eval', evaluation_type),
    ext = ext
)

# time_df.shape[0] = n_samples * n_retrain_scenarios * n_models = 365 * 10 * 10
time_df = load_data(
  path_list = c('results', dataset_name, frequency, 'evaluation'),
  name_list = c(dataset_name, frequency, 'time'),
  ext = ext
)



# Aggregate data ----------------------------------------------------------

eval_df_agg = aggregate_data(
  data = eval_df,
  group_columns = c('method', 'retrain_window'),
  drop_columns = c('unique_id', 'test_window', 'horizon'),
  function_name = 'mean',
  adjust_metrics = TRUE
) |> 
  tibble::as_tibble()

time_df_agg = aggregate_data(
  data = time_df,
  group_columns = c('method', 'retrain_window'),
  drop_columns = c('sample', 'test_window', 'horizon'),
  function_name = 'sum',
  adjust_metrics = TRUE
) |> 
  tibble::as_tibble()



# Tables & Plots ----------------------------------------------------------

# evaluation table
eval_df_agg |> 
  dplyr::select(dplyr::all_of(c('method', 'retrain_window', metric))) |> 
  tidyr::pivot_wider(names_from = 'retrain_window', values_from = metric) |> 
  dplyr::mutate(
    type = get_model_type(method), 
    type = factor(type, levels = model_type_levels, ordered = TRUE),
    .before = 'method'
  ) |> 
  dplyr::mutate(method = factor(method, levels = method_levels, ordered = TRUE)) |> 
  dplyr::arrange(type, method) |> 
  dplyr::rename_with(stringr::str_to_title) |> 
  dt_table(
    title = '',
    caption = '',
    digits = 3
  )

# time table
time_df_agg |> 
  dplyr::select(dplyr::all_of(c('method', 'retrain_window', time_metric))) |> 
  tidyr::pivot_wider(names_from = 'retrain_window', values_from = time_metric) |> 
  dplyr::mutate(
    type = get_model_type(method), 
    type = factor(type, levels = model_type_levels, ordered = TRUE),
    .before = 'method',
  ) |> 
  dplyr::mutate(method = factor(method, levels = method_levels, ordered = TRUE)) |> 
  dplyr::arrange(type, method) |> 
  dplyr::rename_with(stringr::str_to_title) |> 
  dt_table(
    title = '',
    caption = '',
    digits = 3
  )


# plots
g_eval <- eval_df_agg |> 
  dplyr::mutate(
    type = get_model_type(method),
    type = factor(type, levels = model_type_levels, ordered = TRUE),
    .before = 'method',
  ) |> 
  dplyr::mutate(method = factor(method, levels = method_levels, ordered = TRUE)) |> 
  dplyr::arrange(type, method) |> 
  ggplot(aes_string(x = 'retrain_window', y = metric, color = 'method')) +
  geom_line(linewidth = 2) + 
  scale_x_continuous(breaks = retrain_scenarios) +
  labs(
    title = toupper(dataset_name), 
    x = 'Retrain Scenario', y = toupper(metric),
    color = 'Method'
  ) + 
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

g_time <- time_df_agg |> 
  dplyr::mutate(
    type = get_model_type(method), .before = 'method',
    type = factor(type, levels = model_type_levels, ordered = TRUE)
  ) |> 
  dplyr::mutate(method = factor(method, levels = method_levels, ordered = TRUE)) |> 
  dplyr::arrange(type, method) |> 
  ggplot(aes_string(x = 'retrain_window', y = time_metric, color = 'method')) +
  geom_line(linewidth = 2) + 
  scale_x_continuous(breaks = retrain_scenarios) +
  labs(
    title = toupper(dataset_name), 
    x = 'Retrain Scenario', y = 'Computing Time',
    color = 'Method'
  ) + 
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

combined <- g_eval + g_time + plot_layout(guides = "collect") & theme(legend.position = "bottom")
combined
