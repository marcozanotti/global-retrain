# Test
reticulate::use_condaenv('global_retrain')

library(tidyverse)
library(greybox)
library(DT)
library(patchwork)
library(reticulate)

source('src/R/utils.R')
reticulate::source_python('src/Python/utils/utilities.py')


# VN1 Data
vn1_df <- load_data(c('data', 'vn1'), list('vn1_weekly_prep'))

vn1_df |>
  summarise(
    start_date = as.Date(min(ds)),
    end_date = as.Date(max(ds)),
    clients = n_distinct(client),
    warehouses = n_distinct(warehouse),
    products = n_distinct(product),
    product_ids = n_distinct(unique_id),
    n = n()
  )


vn1_res_df <- load_data(
  c('results', 'vn1', 'weekly', 'evaluation'),
  list('vn1_weekly_eval_overlap')
)

vn1_res_df |>
  as_tibble() |>
  select(method, retrain_window, unique_id, rmsse) |>
  filter(
    !method %in%
      c(
        'EnsembleMean2A',
        'EnsembleMean3A',
        'EnsembleMean4A',
        'EnsembleMean5A',
        'ARIMA'
      )
  ) |>
  mutate(rmsse_low = ifelse(rmsse < 1, 1, 0)) |>
  group_by(method, retrain_window) |>
  count(rmsse_low) |>
  ungroup() |>
  filter(rmsse_low == 1) |>
  select(method, retrain_window, n) |>
  pivot_wider(names_from = retrain_window, values_from = n)

vn1_res_df |>
  as_tibble() |>
  select(method, retrain_window, unique_id, rmsse) |>
  filter(
    !method %in%
      c(
        'EnsembleMean2A',
        'EnsembleMean3A',
        'EnsembleMean4A',
        'EnsembleMean5A',
        'ARIMA'
      )
  ) |>
  # clean_outliers(.metric = c('rmsse'), q = c(0.000, 0.995)) |>
  compute_relative_metrics_by_series(type = 'evaluation') |>
  ggplot(aes(x = retrain_window, y = rmsse, group = unique_id)) +
  geom_line() +
  facet_wrap(~method, scales = 'free_y') +
  theme_bw()
