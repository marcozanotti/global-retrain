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
# plot andamenti 700x700
# plot test 750x650

# Load & prepare data -----------------------------------------------------

analysis_file_name <- 'docs/new_frontiers/relative_evaltimecost_overlap_20260316_114315.RData'

res <- load(analysis_file_name)
res <- analysis_results
rm(analysis_results)


# Analysis ----------------------------------------------------------------

# * Parameters ------------------------------------------------------------

df_nms <- c('m4_daily', 'm5_daily', 'vn1_weekly')
mod_tps <- c('ML_DL')
eval_metrics <- c('rmsse', 'scaled_mqloss')
time_metrics <- c('total_sample_time')
cost_metrics <- c('cost', 'savings_perc')


# =========================================================================
# * Evaluation ------------------------------------------------------------
# =========================================================================

# ** Plots ----------------------------------------------------------------
for (k in mod_tps) {
	eval_res1 <- res[[df_nms[1]]][['evaluation']][['results']][[k]]
	eval_res2 <- res[[df_nms[2]]][['evaluation']][['results']][[k]]
	eval_res3 <- res[[df_nms[3]]][['evaluation']][['results']][[k]]
	em1 <- eval_metrics[1]
	em2 <- eval_metrics[2]
	print(
		((eval_res1$plots[[em1]] +
			ggplot2::guides(col = FALSE) +
			ggplot2::labs(x = NULL)) +
			(eval_res1$plots[[em2]] +
				ggplot2::guides(col = FALSE) +
				ggplot2::labs(x = NULL))) /
			((eval_res2$plots[[em1]] +
				ggplot2::guides(col = FALSE) +
				ggplot2::labs(x = NULL)) +
				(eval_res2$plots[[em2]] +
					ggplot2::guides(col = FALSE) +
					ggplot2::labs(x = NULL))) /
			((eval_res3$plots[[em1]]) +
				(eval_res3$plots[[em2]])) +
			patchwork::plot_layout(guides = "collect") &
			ggplot2::theme(legend.position = "bottom")
	)
}


# =========================================================================
# * Time ------------------------------------------------------------------
# =========================================================================

# ** Plots ----------------------------------------------------------------
for (k in mod_tps) {
	time_res1 <- res[[df_nms[1]]][['time']][['results']][[k]]
	time_res2 <- res[[df_nms[2]]][['time']][['results']][[k]]
	time_res3 <- res[[df_nms[3]]][['time']][['results']][[k]]
	tm <- time_metrics[1]
	print(
		(
			# (
			# 	time_res1$plots[[tm]]
			# ) +
			(time_res2$plots[[tm]] +
				ggplot2::labs(title = 'Daily') +
				ggplot2::guides(col = FALSE)) +
				(time_res3$plots[[tm]] +
					ggplot2::labs(title = 'Weekly'))
		) +
			patchwork::plot_layout(guides = "collect") &
			ggplot2::theme(legend.position = "bottom")
	)
}

# =========================================================================
# * Cost ------------------------------------------------------------------
# =========================================================================

# ** Plots ----------------------------------------------------------------
for (k in mod_tps) {
	cost_res1 <- res[[df_nms[1]]][['cost']][['results']][[k]]
	cost_res2 <- res[[df_nms[2]]][['cost']][['results']][[k]]
	cost_res3 <- res[[df_nms[3]]][['cost']][['results']][[k]]
	cm1 <- cost_metrics[1]
	cm2 <- cost_metrics[2]
	print(
		# ((cost_res1$plots[[cm1]] +
		# 	ggplot2::guides(col = FALSE) +
		# 	ggplot2::labs(x = NULL)) +
		# 	(cost_res1$plots[[cm2]] +
		# 		ggplot2::guides(col = FALSE) +
		# 		ggplot2::labs(x = NULL))) /
		((cost_res2$plots[[cm1]] +
			ggplot2::guides(col = FALSE) +
			ggplot2::labs(x = NULL, title = 'Daily')) +
			(cost_res3$plots[[cm1]] +
				ggplot2::guides(col = FALSE) +
				ggplot2::labs(x = NULL, title = 'Weekly'))) /
			((cost_res2$plots[[cm2]] +
				ggplot2::guides(col = FALSE) +
				ggplot2::labs(title = NULL)) +
				(cost_res3$plots[[cm2]] + ggplot2::labs(title = NULL))) +
			patchwork::plot_layout(guides = "collect") &
			ggplot2::theme(legend.position = "bottom")
	)
}
