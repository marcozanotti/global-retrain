
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



# Test
source('src/R/utils.R')
res <- load('results/analysis/relative_overlap_results_20250116_162923.RData')
res <- analysis_results
rm(analysis_results)

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
