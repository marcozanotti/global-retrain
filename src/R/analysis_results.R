# Empirical Analysis of Results

library(tidyverse)
library(DT)
library(patchwork)

source('src/R/utils.R')


# Load & prepare data -----------------------------------------------------

analysis_file_name <- 'results/analysis/absolute_evaltimestabcost_overlap_20250417_102106.RData'
analysis_file_name <- 'results/analysis/relative_evaltimestabcost_overlap_20250417_101712.RData'

res <- load(analysis_file_name)
res <- analysis_results
rm(analysis_results)


# Parameters --------------------------------------------------------------

dataset_name1 <- 'm5_daily'
dataset_name2 <- 'vn1_weekly'
models_type <- 'ML_DL'
models <- c(
	'LR',
	'RF',
	'XGBoost',
	'LGBM',
	'CatBoost',
	'MLP',
	'LSTM',
	'TCN',
	'NBEATSx'
)
models <- c(
	'Ens2A',
	'Ens2T',
	'Ens3A',
	'Ens3T',
	'Ens4A',
	'Ens4T',
	'Ens5A',
	'Ens5T'
)


# Analysis ----------------------------------------------------------------

# * Evaluation ------------------------------------------------------------

eval_res1 <- res[[dataset_name1]][['evaluation']][['results']][[models_type]]
eval_res2 <- res[[dataset_name2]][['evaluation']][['results']][[models_type]]
eval_metrics <- c('bias', 'rmsse', 'mqloss')

# ** Tables ---------------------------------------------------------------

for (m in eval_metrics) {
	print(eval_res1$tables[[m]])
	print(eval_res2$tables[[m]])
}

# ** Plots -----------------------------------------------------------------

for (m in eval_metrics) {
	print(
		eval_res1$plots[[m]] + ggplot2::theme(legend.position = "bottom") +
			eval_res2$plots[[m]] + ggplot2::theme(legend.position = "bottom")
	)
}

# ** Tests -----------------------------------------------------------------

for (m in eval_metrics) {
	g1 <- plot_test_results_facet(
		data = eval_res1$tests[[m]],
		.facet = 'method', 
		.metric = m, 
		metric_label = toupper(gsub("_", " ", m)),
		title = toupper(paste(stringr::str_replace_all(toupper(dataset_name1), "_.*", ""), "- Nemenyi Test"))
	)
	print(g1)
	g2 <- plot_test_results_facet(
		data = eval_res2$tests[[m]],
		.facet = 'method', 
		.metric = m, 
		metric_label = toupper(gsub("_", " ", m)),
		title = toupper(paste(stringr::str_replace_all(toupper(dataset_name2), "_.*", ""), "- Nemenyi Test"))
	)
	print(g2)
}

# for (mod in models) {
# 	for (m in eval_metrics) {
# 		g <- plot_test_results(
# 			data = eval_res1$tests[[m]],
# 			.method = mod, 
# 			.metric = m, 
# 			metric_label = toupper(gsub("_", " ", m)),
# 			title = toupper(paste(stringr::str_replace_all(toupper(dataset_name), "_.*", " "), "- Nemenyi Test -", mod))
# 		)
# 		print(g)
# 		g <- plot_test_results(
# 			data = eval_res2$tests[[m]],
# 			.method = mod, 
# 			.metric = m, 
# 			metric_label = toupper(gsub("_", " ", m)),
# 			title = toupper(paste(stringr::str_replace_all(toupper(dataset_name), "_.*", " "), "- Nemenyi Test -", mod))
# 		)
# 		print(g)
# 	}
# }


# * Time ------------------------------------------------------------------

time_res1 <- res[[dataset_name1]][['time']][['results']][[models_type]]
time_res2 <- res[[dataset_name2]][['time']][['results']][[models_type]]
time_metrics <- c('total_sample_time')

# ** Tables ---------------------------------------------------------------

for (m in time_metrics) {
	print(time_res1$tables[[m]])
	print(time_res2$tables[[m]])
}

# ** Plots -----------------------------------------------------------------

for (m in time_metrics) {
	print(
		time_res1$plots[[m]] + ggplot2::theme(legend.position = "bottom") +
			time_res2$plots[[m]] + ggplot2::theme(legend.position = "bottom")
	)
}

# ** Tests -----------------------------------------------------------------

for (m in time_metrics) {
	g1 <- plot_test_results_facet(
		data = time_res1$tests[[m]],
		.facet = 'method', 
		.metric = m, 
		metric_label = toupper(gsub("_", " ", m)),
		title = toupper(paste(stringr::str_replace_all(toupper(dataset_name1), "_.*", ""), "- Nemenyi Test"))
	)
	print(g1)
	g2 <- plot_test_results_facet(
		data = time_res2$tests[[m]],
		.facet = 'method', 
		.metric = m, 
		metric_label = toupper(gsub("_", " ", m)),
		title = toupper(paste(stringr::str_replace_all(toupper(dataset_name2), "_.*", ""), "- Nemenyi Test"))
	)
	print(g2)
}


# * Stability -------------------------------------------------------------

stab_res1 <- res[[dataset_name1]][['stability']][['results']][[models_type]]
stab_res2 <- res[[dataset_name2]][['stability']][['results']][[models_type]]
stab_metrics <- c('stability_bias', 'mac', 'rmsc', 'smapc', 'mqlossc')

# ** Tables ---------------------------------------------------------------

for (m in stab_metrics) {
	print(stab_res1$tables[[m]])
	print(stab_res2$tables[[m]])
}

# ** Plots -----------------------------------------------------------------

for (m in stab_metrics) {
	print(
		stab_res1$plots[[m]] + ggplot2::theme(legend.position = "bottom") +
			stab_res2$plots[[m]] + ggplot2::theme(legend.position = "bottom")
	)
}

# ** Tests -----------------------------------------------------------------

for (m in stab_metrics) {
	g1 <- plot_test_results_facet(
		data = stab_res1$tests[[m]],
		.facet = 'method', 
		.metric = m, 
		metric_label = toupper(gsub("_", " ", m)),
		title = toupper(paste(stringr::str_replace_all(toupper(dataset_name1), "_.*", ""), "- Nemenyi Test"))
	)
	print(g1)
	g2 <- plot_test_results_facet(
		data = stab_res2$tests[[m]],
		.facet = 'method', 
		.metric = m, 
		metric_label = toupper(gsub("_", " ", m)),
		title = toupper(paste(stringr::str_replace_all(toupper(dataset_name2), "_.*", ""), "- Nemenyi Test"))
	)
	print(g2)
}


# * Cost -------------------------------------------------------------

cost_res1 <- res[[dataset_name1]][['cost']][['results']][[models_type]]
cost_res2 <- res[[dataset_name2]][['cost']][['results']][[models_type]]
cost_metrics <- c('cost', 'savings', 'savings_perc')

# ** Tables ---------------------------------------------------------------

for (m in cost_metrics) {
	print(cost_res1$tables[[m]])
	print(cost_res2$tables[[m]])
}

# ** Plots -----------------------------------------------------------------

# for (m in cost_metrics) {
# 	print(
# 		cost_res1$plots[[m]] + ggplot2::theme(legend.position = "bottom") +
# 			cost_res2$plots[[m]] + ggplot2::theme(legend.position = "bottom")
# 	)
# }

cost_res1$plots[['cost']] + cost_res1$plots[['savings_perc']] +
	patchwork::plot_layout(guides = "collect") & ggplot2::theme(legend.position = "bottom")

cost_res2$plots[['cost']] + cost_res2$plots[['savings_perc']] +
	patchwork::plot_layout(guides = "collect") & ggplot2::theme(legend.position = "bottom")

# ** Tests -----------------------------------------------------------------

for (m in cost_metrics) {
	g1 <- plot_test_results_facet(
		data = cost_res1$tests[[m]],
		.facet = 'method', 
		.metric = m, 
		metric_label = toupper(gsub("_", " ", m)),
		title = toupper(paste(stringr::str_replace_all(toupper(dataset_name1), "_.*", ""), "- Nemenyi Test"))
	)
	print(g1)
	g2 <- plot_test_results_facet(
		data = cost_res2$tests[[m]],
		.facet = 'method', 
		.metric = m, 
		metric_label = toupper(gsub("_", " ", m)),
		title = toupper(paste(stringr::str_replace_all(toupper(dataset_name2), "_.*", ""), "- Nemenyi Test"))
	)
	print(g2)
}

