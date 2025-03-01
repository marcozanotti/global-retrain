# Empirical Analysis of Results

library(tidyverse)
library(DT)
library(patchwork)

source('src/R/utils.R')


# Parameters --------------------------------------------------------------

analysis_file_name <- 'results/analysis/relative_overlap_results_20250301_121204.RData'
analysis_file_name <- 'results/analysis/absolute_overlap_results_20250301_120522.RData'

metrics <- c('rmsse', 'mqloss', 'rmse', 'scaled_crps')

dataset_name <- 'm5_daily'

models <- c(
	'LR',
	# 'RF',
	'XGBoost',
	'LGBM',
	'CatBoost',
	'MLP',
	# 'LSTM',
	'TCN',
	'NBEATSx',
	'NHITS'
)


# Load & prepare data -----------------------------------------------------

res <- load(analysis_file_name)
res <- analysis_results
rm(analysis_results)

res_data <- res[[dataset_name]] 


# Analysis ----------------------------------------------------------------

# * Time table ------------------------------------------------------------

res_data$tab_time

# * Time plot -------------------------------------------------------------

res_data$g_time

# * Evaluation table ------------------------------------------------------

for (m in metrics) {
	print(res_data$tab_eval[[m]])
}

# * Evaluation plot -------------------------------------------------------

for (m in metrics) {
	print(res_data$g_eval[[m]])
}

# * Evaluation plot (combined) --------------------------------------------

for (m in metrics) {
	print(res_data$g_eval_comb[[m]])
}

# * Friedman-Nemenyi test -------------------------------------------------

for (mod in models) {
	for (m in metrics) {
		g <- plot_test_results(
			data = res_data$test_res,
			.method = mod, 
			.metric = m, 
			metric_label = toupper(gsub("_", " ", m)),
			title = toupper(paste(stringr::str_replace_all(toupper(dataset_name), "_", " "), "- Nemenyi Test -", mod))
		)
		print(g)
	}
}

# for (m in metrics) {
# 	res_data$test_res |> 
# 		dplyr::filter(method %in% models, metric == m) |> 
# 		dplyr::arrange(retrain_window) |> 
# 		dplyr::select(method, retrain_window, mean) |> 
# 		tidyr::pivot_wider(names_from = retrain_window, values_from = mean)
# }
