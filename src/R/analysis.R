
# install.packages('tidyverse')
# install.packages('greybox')
# install.packages('DT')
# install.packages('patchwork')
# install.packages('reticulate')

library(tidyverse)
library(greybox)
library(DT)
library(patchwork)
library(reticulate)

source('src/R/utils.R')
reticulate::source_python('src/Python/utils/utilities.py')


config = get_config('config/analyse_config.yaml')
res <- analyse_results(config)



# Ensemble analysis ----
source('src/R/utils.R')
analysis_file_name <- 'results/analysis/absolute_overlap_results_20250212_144529.RData'
res <- load(analysis_file_name)
res <- analysis_results
rm(analysis_results)

dataset_names <- c(
  'm5_monthly', 
  'm5_weekly', 
  # 'm5_daily', 
  'vn1_monthly', 
  'vn1_weekly'
)

# top 3 by accuracy
res_top_acc <- vector('list', length(dataset_names))
for (i in seq_along(dataset_names)) {

  res_top_acc[[i]] <- res[[dataset_names[i]]][['eval_df_agg']] |> 
    dplyr::slice_min(retrain_window) |> 
    dplyr::arrange(rmsse) |> 
    dplyr::slice_head(n = 3) |> 
    dplyr::select(method)

}
res_top_acc <- res_top_acc |> 
  dplyr::bind_cols() |> 
  purrr::set_names(c('M5 MONTHLY', 'M5 WEEKLY', 'VN1 MONTHLY', 'VN1 WEEKLY'))


# top 3 by computation time
res_top_time <- vector('list', length(dataset_names))
for (i in seq_along(dataset_names)) {

  res_top_time[[i]] <- res[[dataset_names[i]]][['time_df_agg']] |> 
    dplyr::slice_min(retrain_window) |> 
    dplyr::arrange(total_sample_time) |> 
    dplyr::slice_head(n = 3) |> 
    dplyr::select(method)

}
res_top_time <- res_top_time |> 
  dplyr::bind_cols() |> 
  purrr::set_names(c('M5 MONTHLY', 'M5 WEEKLY', 'VN1 MONTHLY', 'VN1 WEEKLY'))

res_top_acc
res_top_time




# Friedman - Nemenyi Test ----
source('src/R/utils.R')
analysis_file_name <- 'results/analysis/absolute_overlap_results_20250212_144529.RData'
res <- load(analysis_file_name)
res <- analysis_results
rm(analysis_results)

metrics <- unique(res$m5_monthly$test_res$metric)
unique(res$m5_monthly$test_res$method)
met <- "LR"
for (m in metrics) {
  g <- plot_test_results(
    data = res$m5_monthly$test_res,
    .method = met, 
    .metric = m, 
    metric_label = toupper(gsub("_", " ", m)),
    title = toupper(paste("M5", "MONTHLY", "- Nemenyi Test -", met))
  )
  print(g)
}



# Test ----
source('src/R/utils.R')
analysis_file_name <- 'results/analysis/absolute_overlap_results_20250212_144529.RData'
res <- load(analysis_file_name)
res <- analysis_results
rm(analysis_results)

names(res)
names(res$m5_monthly)
res$m5_monthly$eval_df_agg
res$m5_monthly$time_df_agg
res$m5_monthly$tab_time
res$m5_monthly$g_eval$rmsse
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


