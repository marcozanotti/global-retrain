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


# Load & prepare data -----------------------------------------------------

# run twice, one for absolute and one for relative
analysis_file_name <- 'docs/iifsas_retrain/absolute_evaltimestabcost_overlap_20260128_165006.RData'
analysis_file_name <- 'docs/iifsas_retrain/relative_evaltimestabcost_overlap_20260128_170356.RData'

res <- load(analysis_file_name)
res <- analysis_results
rm(analysis_results)



# Analysis ----------------------------------------------------------------

# * Parameters ------------------------------------------------------------

dataset_name1 <- 'm4_daily'
dataset_name2 <- 'm5_daily'
dataset_name3 <- 'vn1_weekly'

models_type <- 'SF'
models_type <- 'ML_DL'
models_type <- 'ENSACC'



# =========================================================================
# * Evaluation ------------------------------------------------------------
# =========================================================================

eval_res1 <- res[[dataset_name1]][['evaluation']][['results']][[models_type]]
eval_res2 <- res[[dataset_name2]][['evaluation']][['results']][[models_type]]
eval_res3 <- res[[dataset_name3]][['evaluation']][['results']][[models_type]]
eval_metrics <- c('rmsse', 'scaled_mqloss')

# ** Tables ---------------------------------------------------------------
for (m in eval_metrics) {
	cat(paste(dataset_name1, m, "\n\n"))
	print(xtable::xtable(eval_res1$tables[[m]]$x$data, digits = 3), include.rownames = FALSE)
	cat("\n\n")
	cat(paste(dataset_name2, m, "\n\n"))
	print(xtable::xtable(eval_res2$tables[[m]]$x$data, digits = 3), include.rownames = FALSE)
	cat("\n\n")
	cat(paste(dataset_name3, m, "\n\n"))
	print(xtable::xtable(eval_res3$tables[[m]]$x$data, digits = 3), include.rownames = FALSE)
	cat("\n\n")
}

# ** Plots -----------------------------------------------------------------
for (m in eval_metrics) {
	print(
		eval_res1$plots[[m]] + ggplot2::guides(col = FALSE) +
          	eval_res2$plots[[m]] + ggplot2::guides(col = FALSE) +
          	eval_res3$plots[[m]] +
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
  	g3 <- plot_test_results_facet(
		data = eval_res3$tests[[m]],
		.metric = m, 
		by = "retrain_window", 
		metric_label = toupper(gsub("_", " ", m)),
		title = toupper(paste(stringr::str_replace_all(toupper(dataset_name3), "_.*", ""), "- Nemenyi Test"))
	)
	print(g3)
}



# =========================================================================
# * Time ------------------------------------------------------------------
# =========================================================================

time_res1 <- res[[dataset_name1]][['time']][['results']][[models_type]]
time_res2 <- res[[dataset_name2]][['time']][['results']][[models_type]]
time_res3 <- res[[dataset_name3]][['time']][['results']][[models_type]]
time_metrics <- c('total_sample_time')

# ** Tables ---------------------------------------------------------------
for (m in time_metrics) {
	cat(paste(dataset_name1, m, "\n\n"))
	print(xtable::xtable(time_res1$tables[[m]]$x$data, digits = 3), include.rownames = FALSE)
	cat("\n\n")
	cat(paste(dataset_name2, m, "\n\n"))
	print(xtable::xtable(time_res2$tables[[m]]$x$data, digits = 3), include.rownames = FALSE)
	cat("\n\n")
  	cat(paste(dataset_name3, m, "\n\n"))
	print(xtable::xtable(time_res3$tables[[m]]$x$data, digits = 3), include.rownames = FALSE)
	cat("\n\n")
}

# ** Plots -----------------------------------------------------------------
for (m in time_metrics) {
	print(
		time_res1$plots[[m]] + ggplot2::guides(col = FALSE) +
            time_res2$plots[[m]] + ggplot2::guides(col = FALSE) +
          	time_res3$plots[[m]] +
			patchwork::plot_layout(guides = "collect") & ggplot2::theme(legend.position = "bottom")
	)
}



# =========================================================================
# * Cost ------------------------------------------------------------------
# =========================================================================

cost_res1 <- res[[dataset_name1]][['cost']][['results']][[models_type]]
cost_res2 <- res[[dataset_name2]][['cost']][['results']][[models_type]]
cost_res3 <- res[[dataset_name3]][['cost']][['results']][[models_type]]
cost_metrics <- c('cost', 'savings_perc')

# ** Tables ---------------------------------------------------------------
for (m in cost_metrics) {
	cat(paste(dataset_name1, m, "\n\n"))
	print(xtable::xtable(cost_res1$tables[[m]]$x$data, digits = 3), include.rownames = FALSE)
	cat("\n\n")
	cat(paste(dataset_name2, m, "\n\n"))
	print(xtable::xtable(cost_res2$tables[[m]]$x$data, digits = 3), include.rownames = FALSE)
	cat("\n\n")
  	cat(paste(dataset_name3, m, "\n\n"))
	print(xtable::xtable(cost_res3$tables[[m]]$x$data, digits = 3), include.rownames = FALSE)
	cat("\n\n")
}

# ** Plots -----------------------------------------------------------------
cost_res1$plots[['cost']] + cost_res1$plots[['savings_perc']] +
	patchwork::plot_layout(guides = "collect") & ggplot2::theme(legend.position = "bottom")

cost_res2$plots[['cost']] + cost_res2$plots[['savings_perc']] +
	patchwork::plot_layout(guides = "collect") & ggplot2::theme(legend.position = "bottom")

cost_res3$plots[['cost']] + cost_res3$plots[['savings_perc']] +
	patchwork::plot_layout(guides = "collect") & ggplot2::theme(legend.position = "bottom")



# =========================================================================
# * Stability -------------------------------------------------------------
# =========================================================================

stab_res1 <- res[[dataset_name1]][['stability']][['results']][[models_type]]
stab_res2 <- res[[dataset_name2]][['stability']][['results']][[models_type]]
stab_res3 <- res[[dataset_name3]][['stability']][['results']][[models_type]]
stab_metrics <- c('smapc', 'smqc')

# ** Tables ---------------------------------------------------------------
for (m in stab_metrics) {
	cat(paste(dataset_name1, m, "\n\n"))
	print(xtable::xtable(stab_res1$tables[[m]]$x$data, digits = 3), include.rownames = FALSE)
	cat("\n\n")
	cat(paste(dataset_name2, m, "\n\n"))
	print(xtable::xtable(stab_res2$tables[[m]]$x$data, digits = 3), include.rownames = FALSE)
	cat("\n\n")
	cat(paste(dataset_name3, m, "\n\n"))
	print(xtable::xtable(stab_res3$tables[[m]]$x$data, digits = 3), include.rownames = FALSE)
	cat("\n\n")
}

# ** Plots -----------------------------------------------------------------
for (m in stab_metrics) {
	print(
		stab_res1$plots[[m]] + ggplot2::guides(col = FALSE) + 
            stab_res2$plots[[m]] + ggplot2::guides(col = FALSE) + 
            stab_res3$plots[[m]] +
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
	g3 <- plot_test_results_facet(
		data = stab_res3$tests[[m]],
		.metric = m, 
		by = 'retrain_window', 
		metric_label = toupper(gsub("_", " ", m)),
		title = toupper(paste(stringr::str_replace_all(toupper(dataset_name3), "_.*", ""), "- Nemenyi Test"))
	)
	print(g3)
}



# =========================================================================
# * Optimal Retraining Scenario -------------------------------------------
# =========================================================================

config = get_config('config/anal/anal_iifsas_retrain_config.yaml')
opt_freq <- analyze_optimal_frequency(config, adjust = 2)

# Evaluation
opt_freq$m4_daily$evaluation$results$ML_DL$plots$rmsse$overall +
    opt_freq$m5_daily$evaluation$results$ML_DL$plots$rmsse$overall +
	opt_freq$vn1_weekly$evaluation$results$ML_DL$plots$rmsse$overall

opt_freq$m4_daily$evaluation$results$ML_DL$plots$scaled_mqloss$overall +
    opt_freq$m5_daily$evaluation$results$ML_DL$plots$scaled_mqloss$overall +
	opt_freq$vn1_weekly$evaluation$results$ML_DL$plots$scaled_mqloss$overall

opt_freq$m4_daily$evaluation$results$ML_DL$plots$rmsse$bymethod
opt_freq$m4_daily$evaluation$results$ML_DL$plots$scaled_mqloss$bymethod

opt_freq$m5_daily$evaluation$results$ML_DL$plots$rmsse$bymethod
opt_freq$m5_daily$evaluation$results$ML_DL$plots$scaled_mqloss$bymethod

opt_freq$vn1_weekly$evaluation$results$ML_DL$plots$rmsse$bymethod
opt_freq$vn1_weekly$evaluation$results$ML_DL$plots$scaled_mqloss$bymethod

# Stability
opt_freq$m4_daily$stability$results$ML_DL$plots$smapc$overall +
    opt_freq$m5_daily$stability$results$ML_DL$plots$smapc$overall +
	opt_freq$vn1_weekly$stability$results$ML_DL$plots$smapc$overall

opt_freq$m4_daily$stability$results$ML_DL$plots$smqc$overall +
    opt_freq$m5_daily$stability$results$ML_DL$plots$smqc$overall +
	opt_freq$vn1_weekly$stability$results$ML_DL$plots$smqc$overall

opt_freq$m4_daily$stability$results$ML_DL$plots$smapc$bymethod
opt_freq$m4_daily$stability$results$ML_DL$plots$smqc$bymethod

opt_freq$m5_daily$stability$results$ML_DL$plots$smapc$bymethod
opt_freq$m5_daily$stability$results$ML_DL$plots$smqc$bymethod

opt_freq$vn1_weekly$stability$results$ML_DL$plots$smapc$bymethod
opt_freq$vn1_weekly$stability$results$ML_DL$plots$smqc$bymethod



# =========================================================================
# * Ensemble Comparisons --------------------------------------------------
# =========================================================================

models_types <- c('ML_DL', 'ENSACC')

# Overall Results ---------------------------------------------------------

# * Evaluation ------------------------------------------------------------
eval_res1 <- res[[dataset_name1]][['evaluation']][['results']]
eval_res2 <- res[[dataset_name2]][['evaluation']][['results']]
eval_res3 <- res[[dataset_name3]][['evaluation']][['results']]
eval_metrics <- c('rmsse', 'scaled_mqloss')

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

# * Stability -------------------------------------------------------------
stab_res1 <- res[[dataset_name1]][['stability']][['results']]
stab_res2 <- res[[dataset_name2]][['stability']][['results']]
stab_metrics <- c('smapc', 'smqc')

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

# * Evaluation - Stability ------------------------------------------------

models_types <- c('SF', 'ML_DL', 'ENSACC')
eval_res1 <- res[[dataset_name1]][['evaluation']][['results']]
eval_res2 <- res[[dataset_name2]][['evaluation']][['results']]
eval_res3 <- res[[dataset_name3]][['evaluation']][['results']]
stab_res1 <- res[[dataset_name1]][['stability']][['results']]
stab_res2 <- res[[dataset_name2]][['stability']][['results']]
stab_res3 <- res[[dataset_name3]][['stability']][['results']]
metrics <- list(c('rmsse', 'smapc'), c('scaled_mqloss', 'smqc'))

for (m in metrics) {
	g1 <- dplyr::bind_rows(
		eval_res1[[models_types[[1]]]]$plots[[m[1]]]$data,
		eval_res1[[models_types[[2]]]]$plots[[m[1]]]$data,
		eval_res1[[models_types[[3]]]]$plots[[m[1]]]$data |> dplyr::filter(!type %in% c('ENSTIME'))
	) |> 
		dplyr::left_join(
			dplyr::bind_rows(
				stab_res1[[models_types[[1]]]]$plots[[m[2]]]$data,
				stab_res1[[models_types[[2]]]]$plots[[m[2]]]$data,
				stab_res1[[models_types[[3]]]]$plots[[m[2]]]$data
			),
			by = c('type', 'method', 'retrain_window')
		) |> 
		plot_scatter_results(metrics = m, .retrain_window = 7, title = "M4")
	g2 <- dplyr::bind_rows(
		eval_res2[[models_types[[1]]]]$plots[[m[1]]]$data,
		eval_res2[[models_types[[2]]]]$plots[[m[1]]]$data,
		eval_res2[[models_types[[3]]]]$plots[[m[1]]]$data |> dplyr::filter(!type %in% c('ENSTIME'))
	) |> 
		dplyr::left_join(
			dplyr::bind_rows(
				stab_res2[[models_types[[1]]]]$plots[[m[2]]]$data,
				stab_res2[[models_types[[2]]]]$plots[[m[2]]]$data,
				stab_res2[[models_types[[3]]]]$plots[[m[2]]]$data
			),
			by = c('type', 'method', 'retrain_window')
		) |> 
		plot_scatter_results(metrics = m, .retrain_window = 7, title = "M5")
	g3 <- dplyr::bind_rows(
		eval_res3[[models_types[[1]]]]$plots[[m[1]]]$data,
		eval_res3[[models_types[[2]]]]$plots[[m[1]]]$data,
		eval_res3[[models_types[[3]]]]$plots[[m[1]]]$data |> dplyr::filter(!type %in% c('ENSTIME'))
	) |> 
		dplyr::left_join(
			dplyr::bind_rows(
				stab_res3[[models_types[[1]]]]$plots[[m[2]]]$data,
				stab_res3[[models_types[[2]]]]$plots[[m[2]]]$data,
				stab_res3[[models_types[[3]]]]$plots[[m[2]]]$data
			),
			by = c('type', 'method', 'retrain_window')
		) |> 
		plot_scatter_results(metrics = m, .retrain_window = 1, title = "VN1")
	g <- g1 + g2 + g3 +
		patchwork::plot_layout(guides = "collect") & ggplot2::theme(legend.position = "bottom")
	print(g)
}
