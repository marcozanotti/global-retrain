# Test
reticulate::use_condaenv('global_retrain')

library(tidyverse)
library(greybox)
library(DT)
library(patchwork)
library(reticulate)

source('src/R/utils.R')
reticulate::source_python('src/Python/utils/utilities.py')


# M5 Data
m5_df <- load_data(c('data', 'm5'), list('m5_daily_prep'))
head(m5_df)

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


# Analyze predictions
reticulate::use_condaenv('global_retrain')

library(tidyverse)
library(reticulate)

source('src/R/utils.R')
reticulate::source_python('src/Python/utils/utilities.py')

# Hapag Data
hapag_preds_df <- load_data(
  c('results', 'hapag_region', 'weekly', 'preds'),
  list('hapag_region_weekly_preds')
) |> as_tibble()

series_names <- c('A_40HC_E_E')
series_names <- c('L_40RE_A_E')

p <- hapag_preds_df |>
  filter(unique_id %in% series_names) |>
  ggplot(aes(x = ds)) +
  geom_line(aes(y = y), color = 'black') +
  geom_line(aes(y = fcst), color = 'red') +
  facet_wrap(~method, scales = 'free_y') +
  theme_bw()
plotly::ggplotly(p)
