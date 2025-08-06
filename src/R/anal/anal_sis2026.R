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
analysis_file_name <- 'docs/sis2026/absolute_evaltimestabcost_overlap_20250805_164302.RData'
analysis_file_name <- 'docs/sis2026/relative_evaltimestabcost_overlap_20250805_164708.RData'

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

eval_res1$plots[[eval_metrics[1]]] + eval_res1$plots[[eval_metrics[2]]] +
	patchwork::plot_layout(guides = "collect") & ggplot2::theme(legend.position = "bottom")


# =========================================================================
# * Optimal Retraining Scenario -------------------------------------------
# =========================================================================

config = get_config('config/anal/anal_sis2026_config.yaml')
opt_freq <- analyze_optimal_frequency(config, adjust = 2)

opt_freq$m4_daily$evaluation$results$ML_DL$plots$rmsse$overall +
	ggplot2::labs(title = "M4 - Point Forecasting") +
	opt_freq$m4_daily$evaluation$results$ML_DL$plots$scaled_mqloss$overall +
	ggplot2::labs(title = "M4 - Probabilistic Forecasting") +
	patchwork::plot_layout(guides = "collect") & ggplot2::theme(legend.position = "bottom")



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
stab_res1 <- res[[dataset_name1]][['stability']][['results']]
stab_metrics <- c('masc', 'rmssc', 'smapc', 'smqc')



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
