
# install.packages('tidyverse')
# install.packages('tsutils')
# install.packages('DT')
# install.packages('patchwork')
# install.packages('reticulate')

library(tidyverse)
library(tsutils)
library(DT)
library(patchwork)
library(reticulate)

source('src/R/utils.R')
reticulate::source_python('src/Python/utils/utilities.py')


config = get_config('config/analyse_config.yaml')
res <- analyse_results(config)



# Test
source('src/R/utils.R')
res <- load('results/analysis/relative_overlap_results_20250108_142040.RData')
res <- analysis_results

names(res)
names(res$m5_monthly)
res$m5_monthly$eval_df_agg
res$m5_monthly$time_df_agg
res$m5_monthly$tab_time
res$m5_monthly$g_eval$rmse
res$m5_monthly$g_eval_comb$bias

res$m5_monthly$eval_df_agg |> 
  plot_retrain_results(
    metric = 'mae', metric_label = 'MAE', 
    title = 'M5 MONTHLY', smooth = FALSE
  )

res$m5_monthly$eval_df_agg |> 
  plot_retrain_results(
    metric = 'mase', metric_label = 'MASE', 
    title = 'M5 MONTHLY', smooth = FALSE
  )

res$m5_monthly$eval_df_agg |> 
  plot_retrain_results(
    metric = 'rmse', metric_label = 'RMSE', 
    title = 'M5 MONTHLY', smooth = FALSE
  )

res$m5_monthly$eval_df_agg |> 
  plot_retrain_results(
    metric = 'rmsse', metric_label = 'RMSSE', 
    title = 'M5 MONTHLY', smooth = FALSE
  )

res$m5_monthly$eval_df_agg |> 
  plot_retrain_results(
    metric = 'mqloss', metric_label = 'MQLOSS', 
    title = 'M5 MONTHLY', smooth = FALSE
  )



x <- eval_df_tmp |> 
  dplyr::filter(method == 'LinearRegression') |> 
  dplyr::filter(retrain_window == 1) |> 
  dplyr::pull(m)
y <- eval_df_tmp |> 
  dplyr::filter(method == 'LinearRegression') |> 
  dplyr::filter(retrain_window == 2) |> 
  dplyr::pull(m)

wilcox.test(x = x, y = y, alternative = "two.sided", mu = 0, paired = FALSE, digits.ranks = 5)
wilcox.test(x = x, y = y, alternative = "less", mu = 0, paired = FALSE, digits.ranks = 5)


n_series = length(unique(eval_df_tmp$unique_id))
n_scn <- length(retrain_scn_tmp)
x <- eval_df_tmp |> 
  dplyr::filter(method == 'LinearRegression') |>
  dplyr::select(dplyr::all_of(c('retrain_window', m))) |> 
  dplyr::mutate(id = rep(1:n_series, n_scn), .before = 1) |> 
  tidyr::pivot_wider(names_from = 'retrain_window', values_from = m) |> 
  dplyr::select(-id)

tsutils::nemenyi(data = x, conf.level = 0.95, plottype = "vmcb")
tsutils::nemenyi(data = x, conf.level = 0.95, plottype = "vline")

y <- greybox::rmcb(data = x, level = 0.95, outplot = "none")
p <- y |> 
  plot(outplot = "mcb", main = "") # mcb or line
