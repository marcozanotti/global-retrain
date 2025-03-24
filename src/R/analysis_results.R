# Empirical Analysis of Results

library(tidyverse)
library(DT)
library(patchwork)

source('src/R/utils.R')


# Load & prepare data -----------------------------------------------------

analysis_file_name <- 'results/analysis/absolute_overlap_results_20250324_112601.RData'
analysis_file_name <- 'results/analysis/relative_overlap_results_20250324_112525.RData'

res <- load(analysis_file_name)
res <- analysis_results
rm(analysis_results)


# Parameters --------------------------------------------------------------

metrics <- c('rmsse', 'mqloss')
dataset_name <- 'm5_daily'
dataset_name2 <- 'vn1_weekly'
models <- c(
	'LR',
	'RF',
	'XGBoost',
	'LGBM',
	'CatBoost',
	'MLP',
	'LSTM',
	'TCN',
	'NBEATSx',
	'NHITS'
)
n_skus = 200000 * 5000
cost_per_hour = 3.5


# Analysis ----------------------------------------------------------------

res_data <- res[[dataset_name]]
res_data2 <- res[[dataset_name2]]

# * Time table ------------------------------------------------------------

res_data$tab_time

# * Time plot -------------------------------------------------------------

res_data$g_time

res_data$g_time + ggplot2::theme(legend.position = "bottom") +
	res_data2$g_time + ggplot2::theme(legend.position = "bottom")


# * Evaluation table ------------------------------------------------------

for (m in metrics) {
	print(res_data$tab_eval[[m]])
}

# * Evaluation plot -------------------------------------------------------

for (m in metrics) {
	print(
		res_data$g_eval[[m]] + ggplot2::theme(legend.position = "bottom") +
			res_data2$g_eval[[m]] + ggplot2::theme(legend.position = "bottom")
	)
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
			title = toupper(paste(stringr::str_replace_all(toupper(dataset_name), "_.*", " "), "- Nemenyi Test -", mod))
		)
		print(g)
	}
}

for (m in metrics) {
	g <- plot_test_results_facet(
		data = res_data$test_res,
		.facet = 'method', 
		.metric = m, 
		metric_label = toupper(gsub("_", " ", m)),
		title = toupper(paste(stringr::str_replace_all(toupper(dataset_name), "_.*", ""), "- Nemenyi Test"))
	)
	print(g)
}

# * Cost Analysis ---------------------------------------------------------

cost_res <- res_data$time_df_agg |> 
	cost_analysis(
		dataset_name = dataset_name, 
		time_var = 'total_sample_time', 
		n_skus = n_skus, 
		cost_per_hour = cost_per_hour
	)

cost_res$tab_cost
cost_res$tab_sav
cost_res$tab_savperc
cost_res$g_cost
cost_res$g_sav
cost_res$g_savperc
cost_res$g_cost_comb

x <- res_data$time_df_agg |> 
	select(method, retrain_window, total_sample_time) |>
	pivot_wider(names_from = retrain_window, values_from = total_sample_time)
x[4, 2:ncol(x)] |> round(0) |> paste(collapse = ' & ')



