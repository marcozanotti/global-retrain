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
analysis_file_name <- 'docs/global_stability/absolute_evaltimestab_overlap_20250911_141355.RData'
analysis_file_name <- 'docs/global_stability/relative_evaltimestab_overlap_20250911_142718.RData'

res <- load(analysis_file_name)
res <- analysis_results
rm(analysis_results)



# Parameters --------------------------------------------------------------

dataset_name1 <- 'm5_daily'
dataset_name2 <- 'vn1_weekly'

models_type <- 'ML_DL'
models_type <- 'ENSACC'



# Analysis ----------------------------------------------------------------

# =========================================================================
# * Stability -------------------------------------------------------------
# =========================================================================

stab_res1 <- res[[dataset_name1]][['stability']][['results']][[models_type]]
stab_res2 <- res[[dataset_name2]][['stability']][['results']][[models_type]]
stab_metrics <- c('smapc', 'smqc')

# ** Tables ---------------------------------------------------------------
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

# =========================================================================
# * Ensemble Comparisons --------------------------------------------------
# =========================================================================

models_types <- c('ML_DL', 'ENSACC')

# Overall Results ---------------------------------------------------------

# * Evaluation ------------------------------------------------------------
eval_res1 <- res[[dataset_name1]][['evaluation']][['results']]
eval_res2 <- res[[dataset_name2]][['evaluation']][['results']]
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

metrics <- list(c('rmsse', 'smapc'), c('scaled_mqloss', 'smqc'))
for (m in metrics) {
	g1 <- dplyr::bind_rows(
		eval_res1[[models_types[[1]]]]$plots[[m[1]]]$data,
		eval_res1[[models_types[[2]]]]$plots[[m[1]]]$data |> dplyr::filter(!type %in% c('ENSTIME')),
	) |> 
		dplyr::left_join(
			dplyr::bind_rows(
				stab_res1[[models_types[[1]]]]$plots[[m[2]]]$data,
				stab_res1[[models_types[[2]]]]$plots[[m[2]]]$data,
			),
			by = c('type', 'method', 'retrain_window')
		) |> 
		plot_scatter_results(
			metrics = m, .retrain_window = 7, title = "M5",
		)
	g2 <- dplyr::bind_rows(
		eval_res2[[models_types[[1]]]]$plots[[m[1]]]$data,
		eval_res2[[models_types[[2]]]]$plots[[m[1]]]$data |> dplyr::filter(!type %in% c('ENSTIME')),
	) |> 
		dplyr::left_join(
			dplyr::bind_rows(
				stab_res2[[models_types[[1]]]]$plots[[m[2]]]$data,
				stab_res2[[models_types[[2]]]]$plots[[m[2]]]$data,
			),
			by = c('type', 'method', 'retrain_window')
		) |> 
		plot_scatter_results(
			metrics = m, .retrain_window = 1, title = "VN1",
		)
	g3 <- g1 + g2 + 
		patchwork::plot_layout(guides = "collect") & ggplot2::theme(legend.position = "bottom")
	print(g3)
}
