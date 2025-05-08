# Empirical Analysis of Results

library(tidyverse)
library(DT)
library(patchwork)

source('src/R/utils.R')


# NOTE:
# plot andamenti 1120x525
# plot test 730x635
# plot test doppio 1120x525
# plot overall 800x800 or 900x800


# Load & prepare data -----------------------------------------------------

analysis_file_name <- 'results/analysis/absolute_evaltimestabcost_overlap_20250508_114025.RData'
analysis_file_name <- 'results/analysis/relative_evaltimestabcost_overlap_20250508_111549.RData'

res <- load(analysis_file_name)
res <- analysis_results
rm(analysis_results)



# Parameters --------------------------------------------------------------

dataset_name1 <- 'm5_daily'
dataset_name2 <- 'vn1_weekly'

models_type <- 'ML_DL'
models_type <- 'ENSACC_ENSTIME'



# Analysis ----------------------------------------------------------------

# =========================================================================
# * Evaluation ------------------------------------------------------------
# =========================================================================

eval_res1 <- res[[dataset_name1]][['evaluation']][['results']][[models_type]]
eval_res2 <- res[[dataset_name2]][['evaluation']][['results']][[models_type]]
eval_metrics <- c('rmsse', 'mqloss')

# ** Tables ---------------------------------------------------------------

for (m in eval_metrics) {
	print(eval_res1$tables[[m]])
	print(eval_res2$tables[[m]])
}

# latex
for (m in eval_metrics) {
	cat(paste(dataset_name1, m, "\n\n"))
	print(xtable::xtable(eval_res1$tables[[m]]$x$data, digits = 3), include.rownames = FALSE)
	cat("\n\n")
	cat(paste(dataset_name2, m, "\n\n"))
	print(xtable::xtable(eval_res2$tables[[m]]$x$data, digits = 3), include.rownames = FALSE)
	cat("\n\n")
}

# ** Plots -----------------------------------------------------------------

for (m in eval_metrics) {
	print(
		eval_res1$plots[[m]] + eval_res2$plots[[m]] +
			patchwork::plot_layout(guides = "collect") & ggplot2::theme(legend.position = "bottom")
	)
}

# ** Tests -----------------------------------------------------------------

for (m in eval_metrics) {
	g1 <- plot_test_results_facet(
		data = eval_res1$tests[[m]],
		.metric = m, 
		by = "retrain_window", 
		metric_label = toupper(gsub("_", " ", m)),
		title = toupper(paste(stringr::str_replace_all(toupper(dataset_name1), "_.*", ""), "- Nemenyi Test"))
	)
	print(g1)
	g2 <- plot_test_results_facet(
		data = eval_res2$tests[[m]],
		.metric = m, 
		by = "retrain_window", 
		metric_label = toupper(gsub("_", " ", m)),
		title = toupper(paste(stringr::str_replace_all(toupper(dataset_name2), "_.*", ""), "- Nemenyi Test"))
	)
	print(g2)
}

for (m in eval_metrics) {
	g1 <- plot_test_results(
		data = eval_res1$tests[[m]],
		.metric = m, 
		by = "method", 
		.retrain_window = 7,
		metric_label = toupper(gsub("_", " ", m)),
		title = toupper(paste(stringr::str_replace_all(toupper(dataset_name1), "_.*", " "), "- Nemenyi Test"))
	) + ggplot2::labs(x = "")
	g2 <- plot_test_results(
		data = eval_res2$tests[[m]],
		.metric = m, 
		by = "method", 
		.retrain_window = 1,
		metric_label = toupper(gsub("_", " ", m)),
		title = toupper(paste(stringr::str_replace_all(toupper(dataset_name2), "_.*", " "), "- Nemenyi Test"))
	) + ggplot2::labs(x = "")
	g3 <- g1 + g2 +	patchwork::plot_layout(guides = "collect")
	print(g3)
}



# =========================================================================
# * Time ------------------------------------------------------------------
# =========================================================================

time_res1 <- res[[dataset_name1]][['time']][['results']][[models_type]]
time_res2 <- res[[dataset_name2]][['time']][['results']][[models_type]]
time_metrics <- c('total_sample_time')

# ** Tables ---------------------------------------------------------------

for (m in time_metrics) {
	print(time_res1$tables[[m]])
	print(time_res2$tables[[m]])
}

# latex
for (m in time_metrics) {
	cat(paste(dataset_name1, m, "\n\n"))
	print(xtable::xtable(time_res1$tables[[m]]$x$data, digits = 3), include.rownames = FALSE)
	cat("\n\n")
	cat(paste(dataset_name2, m, "\n\n"))
	print(xtable::xtable(time_res2$tables[[m]]$x$data, digits = 3), include.rownames = FALSE)
	cat("\n\n")
}

# ** Plots -----------------------------------------------------------------

for (m in time_metrics) {
	print(
		time_res1$plots[[m]] + time_res2$plots[[m]] +
			patchwork::plot_layout(guides = "collect") & ggplot2::theme(legend.position = "bottom")
	)
}

# ** Tests -----------------------------------------------------------------

for (m in time_metrics) {
	g1 <- plot_test_results_facet(
		data = time_res1$tests[[m]],
		.metric = m,
		by = 'retrain_window',  
		metric_label = toupper(gsub("_", " ", m)),
		title = toupper(paste(stringr::str_replace_all(toupper(dataset_name1), "_.*", ""), "- Nemenyi Test"))
	)
	print(g1)
	g2 <- plot_test_results_facet(
		data = time_res2$tests[[m]],
		.metric = m, 
		by = 'retrain_window', 
		metric_label = toupper(gsub("_", " ", m)),
		title = toupper(paste(stringr::str_replace_all(toupper(dataset_name2), "_.*", ""), "- Nemenyi Test"))
	)
	print(g2)
}

for (m in time_metrics) {
	g1 <- plot_test_results(
		data = time_res1$tests[[m]],
		.metric = m, 
		by = "method", 
		.retrain_window = 7,
		metric_label = toupper(gsub("_", " ", m)),
		title = toupper(paste(stringr::str_replace_all(toupper(dataset_name1), "_.*", " "), "- Nemenyi Test"))
	) + ggplot2::labs(x = "", y = "Computing Time")
	g2 <- plot_test_results(
		data = time_res2$tests[[m]],
		.metric = m, 
		by = "method", 
		.retrain_window = 1,
		metric_label = toupper(gsub("_", " ", m)),
		title = toupper(paste(stringr::str_replace_all(toupper(dataset_name2), "_.*", " "), "- Nemenyi Test"))
	) + ggplot2::labs(x = "", y = "Computing Time")
	g3 <- g1 + g2 +	patchwork::plot_layout(guides = "collect")
	print(g3)
}



# =========================================================================
# * Stability -------------------------------------------------------------
# =========================================================================

stab_res1 <- res[[dataset_name1]][['stability']][['results']][[models_type]]
stab_res2 <- res[[dataset_name2]][['stability']][['results']][[models_type]]
stab_metrics <- c('smapc', 'mqlossc')

# ** Tables ---------------------------------------------------------------

for (m in stab_metrics) {
	print(stab_res1$tables[[m]])
	print(stab_res2$tables[[m]])
}

# latex
for (m in stab_metrics) {
	cat(paste(dataset_name1, m, "\n\n"))
	print(xtable::xtable(stab_res1$tables[[m]]$x$data, digits = 3), include.rownames = FALSE)
	cat("\n\n")
	cat(paste(dataset_name2, m, "\n\n"))
	print(xtable::xtable(stab_res2$tables[[m]]$x$data, digits = 3), include.rownames = FALSE)
	cat("\n\n")
}

# ** Plots -----------------------------------------------------------------

for (m in stab_metrics) {
	print(
		stab_res1$plots[[m]] + stab_res2$plots[[m]] +
			patchwork::plot_layout(guides = "collect") & ggplot2::theme(legend.position = "bottom")
	)
}

# ** Tests -----------------------------------------------------------------

for (m in stab_metrics) {
	g1 <- plot_test_results_facet(
		data = stab_res1$tests[[m]],
		.metric = m, 
		by = 'retrain_window', 
		metric_label = toupper(gsub("_", " ", m)),
		title = toupper(paste(stringr::str_replace_all(toupper(dataset_name1), "_.*", ""), "- Nemenyi Test"))
	)
	print(g1)
	g2 <- plot_test_results_facet(
		data = stab_res2$tests[[m]],
		.metric = m, 
		by = 'retrain_window', 
		metric_label = toupper(gsub("_", " ", m)),
		title = toupper(paste(stringr::str_replace_all(toupper(dataset_name2), "_.*", ""), "- Nemenyi Test"))
	)
	print(g2)
}

for (m in stab_metrics) {
	g1 <- plot_test_results(
		data = stab_res1$tests[[m]],
		.metric = m, 
		by = "method", 
		.retrain_window = 7,
		metric_label = toupper(gsub("_", " ", m)),
		title = toupper(paste(stringr::str_replace_all(toupper(dataset_name1), "_.*", " "), "- Nemenyi Test"))
	) + ggplot2::labs(x = "")
	g2 <- plot_test_results(
		data = stab_res2$tests[[m]],
		.metric = m, 
		by = "method", 
		.retrain_window = 1,
		metric_label = toupper(gsub("_", " ", m)),
		title = toupper(paste(stringr::str_replace_all(toupper(dataset_name2), "_.*", " "), "- Nemenyi Test"))
	) + ggplot2::labs(x = "")
	g3 <- g1 + g2 +	patchwork::plot_layout(guides = "collect")
	print(g3)
}



# =========================================================================
# * Cost ------------------------------------------------------------------
# =========================================================================

cost_res1 <- res[[dataset_name1]][['cost']][['results']][[models_type]]
cost_res2 <- res[[dataset_name2]][['cost']][['results']][[models_type]]
cost_metrics <- c('cost', 'savings', 'savings_perc')

# ** Tables ---------------------------------------------------------------

for (m in cost_metrics) {
	print(cost_res1$tables[[m]])
	print(cost_res2$tables[[m]])
}

# latex
for (m in cost_metrics) {
	cat(paste(dataset_name1, m, "\n\n"))
	print(xtable::xtable(cost_res1$tables[[m]]$x$data, digits = 3), include.rownames = FALSE)
	cat("\n\n")
	cat(paste(dataset_name2, m, "\n\n"))
	print(xtable::xtable(cost_res2$tables[[m]]$x$data, digits = 3), include.rownames = FALSE)
	cat("\n\n")
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

for (m in 'cost') {
	g1 <- plot_test_results_facet(
		data = cost_res1$tests[[m]],
		.metric = m, 
		by = 'retrain_window', 
		metric_label = toupper(gsub("_", " ", m)),
		title = toupper(paste(stringr::str_replace_all(toupper(dataset_name1), "_.*", ""), "- Nemenyi Test"))
	)
	print(g1)
	g2 <- plot_test_results_facet(
		data = cost_res2$tests[[m]],
		.metric = m, 
		by = 'retrain_window', 
		metric_label = toupper(gsub("_", " ", m)),
		title = toupper(paste(stringr::str_replace_all(toupper(dataset_name2), "_.*", ""), "- Nemenyi Test"))
	)
	print(g2)
}

for (m in 'cost') {
	g1 <- plot_test_results(
		data = cost_res1$tests[[m]],
		.metric = m, 
		by = "method", 
		.retrain_window = 7,
		metric_label = toupper(gsub("_", " ", m)),
		title = toupper(paste(stringr::str_replace_all(toupper(dataset_name1), "_.*", " "), "- Nemenyi Test"))
	) + ggplot2::labs(x = "")
	g2 <- plot_test_results(
		data = cost_res2$tests[[m]],
		.metric = m, 
		by = "method", 
		.retrain_window = 1,
		metric_label = toupper(gsub("_", " ", m)),
		title = toupper(paste(stringr::str_replace_all(toupper(dataset_name2), "_.*", " "), "- Nemenyi Test"))
	) + ggplot2::labs(x = "")
	g3 <- g1 + g2 +	patchwork::plot_layout(guides = "collect")
	print(g3)
}



# =========================================================================
# * Ensemble Comparisons --------------------------------------------------
# =========================================================================

models_types <- c('ML_DL', 'ENSACC_ENSTIME')


# Overall Results ---------------------------------------------------------

# * Evaluation ------------------------------------------------------------

eval_res1 <- res[[dataset_name1]][['evaluation']][['results']]
eval_res2 <- res[[dataset_name2]][['evaluation']][['results']]
eval_metrics <- c('rmsse', 'mqloss')

# ** Tables ---------------------------------------------------------------
for (m in eval_metrics) {
	cat(paste(m, "\n\n"))
	print(
		xtable::xtable(
			dplyr::bind_rows(
				eval_res2[[models_types[[1]]]]$tables[[m]]$x$data |> dplyr::select(1:2),
				eval_res2[[models_types[[2]]]]$tables[[m]]$x$data |> dplyr::select(1:2)
			) |>
				dplyr::left_join(
					dplyr::bind_rows(
						eval_res1[[models_types[[1]]]]$tables[[m]]$x$data |> dplyr::select(1:2),
						eval_res1[[models_types[[2]]]]$tables[[m]]$x$data |> dplyr::select(1:2)
					),
					by = 'Method'
				) |> 
				dplyr::rename('M5' = `7`, 'VN1' = `1`) |>
				dplyr::relocate("M5", .after = "Method"),	
			digits = 3
		), 
		include.rownames = FALSE
	)
	cat("\n\n")
}

# ** Plots ----------------------------------------------------------------
for (m in eval_metrics) {
	g1 <- plot_compared_results(
		dplyr::bind_rows(
			eval_res1[[models_types[[1]]]]$plots[[m]]$data,
			eval_res1[[models_types[[2]]]]$plots[[m]]$data
		), 
		metric = m, .retrain_window = 7, title = "M5"
	)
	g2 <- plot_compared_results(
		dplyr::bind_rows(
			eval_res2[[models_types[[1]]]]$plots[[m]]$data,
			eval_res2[[models_types[[2]]]]$plots[[m]]$data
		), 
		metric = m, .retrain_window = 1, title = "VN1"
	)
	g3 <- g1 / g2 + 
		patchwork::plot_layout(guides = "collect") & ggplot2::theme(legend.position = "right")
	print(g3)
}

# ** Tests ----------------------------------------------------------------


# * Time ------------------------------------------------------------------
time_res1 <- res[[dataset_name1]][['time']][['results']]
time_res2 <- res[[dataset_name2]][['time']][['results']]
time_metrics <- c('total_sample_time')

# ** Tables ---------------------------------------------------------------
for (m in time_metrics) {
	cat(paste(m, "\n\n"))
	print(
		xtable::xtable(
			dplyr::bind_rows(
				time_res2[[models_types[[1]]]]$tables[[m]]$x$data |> dplyr::select(1:2),
				time_res2[[models_types[[2]]]]$tables[[m]]$x$data |> dplyr::select(1:2)
			) |>
				dplyr::left_join(
					dplyr::bind_rows(
						time_res1[[models_types[[1]]]]$tables[[m]]$x$data |> dplyr::select(1:2),
						time_res1[[models_types[[2]]]]$tables[[m]]$x$data |> dplyr::select(1:2)
					),
					by = 'Method'
				) |> 
				dplyr::rename('M5' = `7`, 'VN1' = `1`) |>
				dplyr::relocate("M5", .after = "Method"),	
			digits = 0
		), 
		include.rownames = FALSE
	)
	cat("\n\n")
}

# ** Plots ----------------------------------------------------------------
for (m in time_metrics) {
	g1 <- plot_compared_results(
		dplyr::bind_rows(
			time_res1[[models_types[[1]]]]$plots[[m]]$data,
			time_res1[[models_types[[2]]]]$plots[[m]]$data
		), 
		metric = m, .retrain_window = 7, title = "M5"
	)
	g2 <- plot_compared_results(
		dplyr::bind_rows(
			time_res2[[models_types[[1]]]]$plots[[m]]$data,
			time_res2[[models_types[[2]]]]$plots[[m]]$data
		), 
		metric = m, .retrain_window = 1, title = "VN1"
	)
	g3 <- g1 / g2 + 
		patchwork::plot_layout(guides = "collect") & ggplot2::theme(legend.position = "right")
	print(g3)
}


# * Stability -------------------------------------------------------------

stab_res1 <- res[[dataset_name1]][['stability']][['results']]
stab_res2 <- res[[dataset_name2]][['stability']][['results']]
stab_metrics <- c('smapc', 'mqlossc')

# ** Tables ---------------------------------------------------------------
for (m in stab_metrics) {
	cat(paste(m, "\n\n"))
	print(
		xtable::xtable(
			dplyr::bind_rows(
				stab_res2[[models_types[[1]]]]$tables[[m]]$x$data |> dplyr::select(1:2),
				stab_res2[[models_types[[2]]]]$tables[[m]]$x$data |> dplyr::select(1:2)
			) |>
				dplyr::left_join(
					dplyr::bind_rows(
						stab_res1[[models_types[[1]]]]$tables[[m]]$x$data |> dplyr::select(1:2),
						stab_res1[[models_types[[2]]]]$tables[[m]]$x$data |> dplyr::select(1:2)
					),
					by = 'Method'
				) |> 
				dplyr::rename('M5' = `7`, 'VN1' = `1`) |>
				dplyr::relocate("M5", .after = "Method"),	
			digits = 3
		), 
		include.rownames = FALSE
	)
	cat("\n\n")
}

# ** Plots ----------------------------------------------------------------
for (m in stab_metrics) {
	g1 <- plot_compared_results(
		dplyr::bind_rows(
			stab_res1[[models_types[[1]]]]$plots[[m]]$data,
			stab_res1[[models_types[[2]]]]$plots[[m]]$data
		), 
		metric = m, .retrain_window = 7, title = "M5"
	)
	g2 <- plot_compared_results(
		dplyr::bind_rows(
			stab_res2[[models_types[[1]]]]$plots[[m]]$data,
			stab_res2[[models_types[[2]]]]$plots[[m]]$data
		), 
		metric = m, .retrain_window = 1, title = "VN1"
	)
	g3 <- g1 / g2 + 
		patchwork::plot_layout(guides = "collect") & ggplot2::theme(legend.position = "right")
	print(g3)
}


# * Cost ------------------------------------------------------------------

cost_res1 <- res[[dataset_name1]][['cost']][['results']]
cost_res2 <- res[[dataset_name2]][['cost']][['results']]
cost_metrics <- c('cost')

# ** Tables ---------------------------------------------------------------
for (m in cost_metrics) {
	cat(paste(m, "\n\n"))
	print(
		xtable::xtable(
			dplyr::bind_rows(
				cost_res2[[models_types[[1]]]]$tables[[m]]$x$data |> dplyr::select(1:2),
				cost_res2[[models_types[[2]]]]$tables[[m]]$x$data |> dplyr::select(1:2)
			) |>
				dplyr::left_join(
					dplyr::bind_rows(
						cost_res1[[models_types[[1]]]]$tables[[m]]$x$data |> dplyr::select(1:2),
						cost_res1[[models_types[[2]]]]$tables[[m]]$x$data |> dplyr::select(1:2)
					),
					by = 'Method'
				) |> 
				dplyr::rename('M5' = `7`, 'VN1' = `1`) |>
				dplyr::relocate("M5", .after = "Method"),	
			digits = 0
		), 
		include.rownames = FALSE
	)
	cat("\n\n")
}

# ** Plots ----------------------------------------------------------------
for (m in cost_metrics) {
	g1 <- plot_compared_results(
		dplyr::bind_rows(
			cost_res1[[models_types[[1]]]]$plots[[m]]$data,
			cost_res1[[models_types[[2]]]]$plots[[m]]$data
		), 
		metric = m, .retrain_window = 7, title = "M5"
	)
	g2 <- plot_compared_results(
		dplyr::bind_rows(
			cost_res2[[models_types[[1]]]]$plots[[m]]$data,
			cost_res2[[models_types[[2]]]]$plots[[m]]$data
		), 
		metric = m, .retrain_window = 1, title = "VN1"
	)
	g3 <- g1 / g2 + 
		patchwork::plot_layout(guides = "collect") & ggplot2::theme(legend.position = "bottom")
	print(g3)
}



# Retraining Results ------------------------------------------------------

# * Evaluation ------------------------------------------------------------

eval_res1 <- res[[dataset_name1]][['evaluation']][['results']]
eval_res2 <- res[[dataset_name2]][['evaluation']][['results']]
eval_metrics <- c('rmsse', 'mqloss')

for (m in eval_metrics) {
	print(
		eval_res1[[models_types[1]]]$plots[[m]] + ggplot2::theme(legend.position = "bottom") +
		eval_res1[[models_types[2]]]$plots[[m]] + ggplot2::theme(legend.position = "bottom")
	)
	print(
		eval_res2[[models_types[1]]]$plots[[m]] + ggplot2::theme(legend.position = "bottom") +
		eval_res2[[models_types[2]]]$plots[[m]] + ggplot2::theme(legend.position = "bottom")
	)
}

# * Time ------------------------------------------------------------------

time_res1 <- res[[dataset_name1]][['time']][['results']]
time_res2 <- res[[dataset_name2]][['time']][['results']]
time_metrics <- c('total_sample_time')

for (m in time_metrics) {
	print(
		time_res1[[models_types[1]]]$plots[[m]] + ggplot2::theme(legend.position = "bottom") +
		time_res1[[models_types[2]]]$plots[[m]] + ggplot2::theme(legend.position = "bottom")
	)
	print(
		time_res2[[models_types[1]]]$plots[[m]] + ggplot2::theme(legend.position = "bottom") +
		time_res2[[models_types[2]]]$plots[[m]] + ggplot2::theme(legend.position = "bottom")
	)
}

# * Stability -------------------------------------------------------------

stab_res1 <- res[[dataset_name1]][['stability']][['results']]
stab_res2 <- res[[dataset_name2]][['stability']][['results']]
stab_metrics <- c('smapc', 'mqlossc')

for (m in stab_metrics) {
	print(
		stab_res1[[models_types[1]]]$plots[[m]] + ggplot2::theme(legend.position = "bottom") +
		stab_res1[[models_types[2]]]$plots[[m]] + ggplot2::theme(legend.position = "bottom")
	)
	print(
		stab_res2[[models_types[1]]]$plots[[m]] + ggplot2::theme(legend.position = "bottom") +
		stab_res2[[models_types[2]]]$plots[[m]] + ggplot2::theme(legend.position = "bottom")
	)
}

# * Cost ------------------------------------------------------------------

cost_res1 <- res[[dataset_name1]][['cost']][['results']]
cost_res2 <- res[[dataset_name2]][['cost']][['results']]
cost_metrics <- c('cost', 'savings', 'savings_perc')

for (m in cost_metrics) {
	print(
		cost_res1[[models_types[1]]]$plots[[m]] + ggplot2::theme(legend.position = "bottom") +
		cost_res1[[models_types[2]]]$plots[[m]] + ggplot2::theme(legend.position = "bottom")
	)
	print(
		cost_res2[[models_types[1]]]$plots[[m]] + ggplot2::theme(legend.position = "bottom") +
		cost_res2[[models_types[2]]]$plots[[m]] + ggplot2::theme(legend.position = "bottom")
	)
}
