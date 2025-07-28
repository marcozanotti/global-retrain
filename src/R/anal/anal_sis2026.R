# Empirical Analysis of Results
reticulate::use_condaenv('global_retrain')

library(tidyverse)
library(greybox)
library(DT)
library(patchwork)
library(reticulate)

source('src/R/utils.R')
reticulate::source_python('src/Python/utils/utilities.py')

# NOTE:
# plot andamenti 1120x525
# plot test 730x635
# plot test doppio 1120x525
# plot overall 800x800 or 900x800



# =========================================================================
# =========================================================================
# Global Retrain ----------------------------------------------------------
# =========================================================================
# =========================================================================

# Load & prepare data -----------------------------------------------------

# run twice, one for absolute and one for relative
analysis_file_name <- 'docs/sis2026/absolute_evaltimestabcost_overlap_20250727_213312.RData'
analysis_file_name <- 'docs/sis2026/relative_evaltimestabcost_overlap_20250727_213237.RData'

res <- load(analysis_file_name)
res <- analysis_results
rm(analysis_results)



# Parameters --------------------------------------------------------------

dataset_name1 <- 'm4_daily'

models_type <- 'ML_DL'
models_type <- 'ENSACC'



# Analysis ----------------------------------------------------------------

# =========================================================================
# * Evaluation ------------------------------------------------------------
# =========================================================================

eval_res1 <- res[[dataset_name1]][['evaluation']][['results']][[models_type]]
eval_metrics <- c('rmsse', 'scaled_mqloss')

# ** Tables ---------------------------------------------------------------

for (m in eval_metrics) {
	cat(paste(dataset_name1, m, "\n\n"))
	print(xtable::xtable(eval_res1$tables[[m]]$x$data, digits = 3), include.rownames = FALSE)
	cat("\n\n")
}

# ** Plots -----------------------------------------------------------------

for (m in eval_metrics) {
	print(
		eval_res1$plots[[m]] +
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
}

# =========================================================================
# * Time ------------------------------------------------------------------
# =========================================================================

time_res1 <- res[[dataset_name1]][['time']][['results']][[models_type]]
time_metrics <- c('total_sample_time')

# ** Tables ---------------------------------------------------------------

for (m in time_metrics) {
	cat(paste(dataset_name1, m, "\n\n"))
	print(xtable::xtable(time_res1$tables[[m]]$x$data, digits = 3), include.rownames = FALSE)
	cat("\n\n")
}

# ** Plots -----------------------------------------------------------------

for (m in time_metrics) {
	print(
		time_res1$plots[[m]] +
			patchwork::plot_layout(guides = "collect") & ggplot2::theme(legend.position = "bottom")
	)
}

# =========================================================================
# * Stability -------------------------------------------------------------
# =========================================================================

stab_res1 <- res[[dataset_name1]][['stability']][['results']][[models_type]]
stab_metrics <- c('smapc', 'mqlossc')

# ** Tables ---------------------------------------------------------------

for (m in stab_metrics) {
	cat(paste(dataset_name1, m, "\n\n"))
	print(xtable::xtable(stab_res1$tables[[m]]$x$data, digits = 3), include.rownames = FALSE)
	cat("\n\n")
}

# ** Plots -----------------------------------------------------------------

for (m in stab_metrics) {
	print(
		stab_res1$plots[[m]] +
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
}

# =========================================================================
# * Cost ------------------------------------------------------------------
# =========================================================================

cost_res1 <- res[[dataset_name1]][['cost']][['results']][[models_type]]
cost_metrics <- c('cost', 'savings_perc')

# ** Tables ---------------------------------------------------------------

for (m in cost_metrics) {
	cat(paste(dataset_name1, m, "\n\n"))
	print(xtable::xtable(cost_res1$tables[[m]]$x$data, digits = 3), include.rownames = FALSE)
	cat("\n\n")
}

# ** Plots -----------------------------------------------------------------

cost_res1$plots[['cost']] + cost_res1$plots[['savings_perc']] +
	patchwork::plot_layout(guides = "collect") & ggplot2::theme(legend.position = "bottom")



# =========================================================================
# * Optimal Retraining Scenario -------------------------------------------
# =========================================================================

config = get_config('config/anal/anal_sis2026_config.yaml')
opt_freq <- analyze_optimal_frequency(config, adjust = 2)

opt_freq$m4_daily$evaluation$results$ML_DL$plots$rmsse$overall
opt_freq$m4_daily$evaluation$results$ML_DL$plots$scaled_mqloss$overall





# =========================================================================
# =========================================================================
# Global Stability --------------------------------------------------------
# =========================================================================
# =========================================================================

# =========================================================================
# * Ensemble Comparisons --------------------------------------------------
# =========================================================================

models_types <- c('ML_DL', 'ENSACC')

eval_res1 <- res[[dataset_name1]][['evaluation']][['results']]
eval_metrics <- c('rmsse', 'scaled_mqloss')
time_res1 <- res[[dataset_name1]][['time']][['results']]
time_metrics <- c('total_sample_time')
stab_res1 <- res[[dataset_name1]][['stability']][['results']]
stab_metrics <- c('smapc', 'mqlossc')
cost_res1 <- res[[dataset_name1]][['cost']][['results']]
cost_metrics <- c('cost', 'savings_perc')


# Overall Results ---------------------------------------------------------

# ** Tables ---------------------------------------------------------------
for (m in eval_metrics) {
	cat(paste(m, "\n\n"))
	print(
		xtable::xtable(
			dplyr::bind_rows(
				eval_res1[[models_types[[1]]]]$tables[[m]]$x$data |> dplyr::select(1:2),
			  eval_res1[[models_types[[2]]]]$tables[[m]]$x$data |> dplyr::select(1:2)
			) |> 
				dplyr::rename('M4' = `7`) |>
				dplyr::relocate("M4", .after = "Method"),	
			digits = 3
		), 
		include.rownames = FALSE
	)
	cat("\n\n")
}

for (m in time_metrics) {
	cat(paste(m, "\n\n"))
	print(
		xtable::xtable(
			dplyr::bind_rows(
				time_res1[[models_types[[1]]]]$tables[[m]]$x$data |> dplyr::select(1:2),
				time_res1[[models_types[[2]]]]$tables[[m]]$x$data |> dplyr::select(1:2)
			) |> 
			dplyr::rename('M4' = `7`) |>
			dplyr::relocate("M4", .after = "Method"),	
			digits = 0
		), 
		include.rownames = FALSE
	)
	cat("\n\n")
}

for (m in stab_metrics) {
	cat(paste(m, "\n\n"))
	print(
		xtable::xtable(
			dplyr::bind_rows(
				stab_res1[[models_types[[1]]]]$tables[[m]]$x$data |> dplyr::select(1:2),
				stab_res1[[models_types[[2]]]]$tables[[m]]$x$data |> dplyr::select(1:2)
			) |> 
				dplyr::rename('M4' = `7`) |>
				dplyr::relocate("M4", .after = "Method"),	
			digits = 3
		), 
		include.rownames = FALSE
	)
	cat("\n\n")
}

for (m in cost_metrics) {
	cat(paste(m, "\n\n"))
	print(
		xtable::xtable(
			dplyr::bind_rows(
				cost_res1[[models_types[[1]]]]$tables[[m]]$x$data |> dplyr::select(1:2),
				cost_res1[[models_types[[2]]]]$tables[[m]]$x$data |> dplyr::select(1:2)
			) |> 
				dplyr::rename('M4' = `7`) |>
				dplyr::relocate("M4", .after = "Method"),	
			digits = 0
		), 
		include.rownames = FALSE
	)
	cat("\n\n")
}

# * Evaluation - Stability ------------------------------------------------

metrics <- list(c('rmsse', 'smapc'), c('scaled_mqloss', 'mqlossc'))
for (m in metrics) {
	g1 <- dplyr::bind_rows(
		eval_res1[[models_types[[1]]]]$plots[[m[1]]]$data,
		eval_res1[[models_types[[2]]]]$plots[[m[1]]]$data
	) |> 
		dplyr::left_join(
			dplyr::bind_rows(
				stab_res1[[models_types[[1]]]]$plots[[m[2]]]$data,
				stab_res1[[models_types[[2]]]]$plots[[m[2]]]$data,
			),
			by = c('type', 'method', 'retrain_window')
		) |> 
		plot_scatter_results(
			metrics = m, .retrain_window = 7, title = "M4",
		)
	print(g1)
}

metrics <- list(c('rmsse', 'scaled_mqloss'))
for (m in metrics) {
	g1 <- dplyr::bind_rows(
		eval_res1[[models_types[[1]]]]$plots[[m[1]]]$data,
		eval_res1[[models_types[[2]]]]$plots[[m[1]]]$data,
	) |> 
		plot_scatter_results(
			metrics = m, .retrain_window = 7, title = "M4",
		)
	print(g1)
}

metrics <- list(c('smapc', 'mqlossc'))
for (m in metrics) {
	g1 <- dplyr::bind_rows(
		stab_res1[[models_types[[1]]]]$plots[[m[1]]]$data,
		stab_res1[[models_types[[2]]]]$plots[[m[1]]]$data,
	) |> 
		plot_scatter_results(
			metrics = m, .retrain_window = 7, title = "M4",
		)
	print(g1)
}


# Retraining Results ------------------------------------------------------

# ** Plots ----------------------------------------------------------------

for (m in eval_metrics) {
	print(
		eval_res1[[models_types[1]]]$plots[[m]] + ggplot2::theme(legend.position = "bottom") +
			eval_res1[[models_types[2]]]$plots[[m]] + ggplot2::theme(legend.position = "bottom")
	)
}

for (m in time_metrics) {
	print(
		time_res1[[models_types[1]]]$plots[[m]] + ggplot2::theme(legend.position = "bottom") +
			time_res1[[models_types[2]]]$plots[[m]] + ggplot2::theme(legend.position = "bottom")
	)
}

for (m in stab_metrics) {
	print(
		stab_res1[[models_types[1]]]$plots[[m]] + ggplot2::theme(legend.position = "bottom") +
			stab_res1[[models_types[2]]]$plots[[m]] + ggplot2::theme(legend.position = "bottom")
	)
}

for (m in cost_metrics) {
	print(
		cost_res1[[models_types[1]]]$plots[[m]] + ggplot2::theme(legend.position = "bottom") +
			cost_res1[[models_types[2]]]$plots[[m]] + ggplot2::theme(legend.position = "bottom")
	)
}

