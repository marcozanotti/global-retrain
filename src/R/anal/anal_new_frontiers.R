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
analysis_file_name <- 'docs/new_frontiers/absolute_evaltimecost_overlap_20260317_150702.RData'

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

for (k in mod_tps) {
	cost_res2 <- res[[df_nms[2]]][['cost']][['results']][[k]]
	cost_res3 <- res[[df_nms[3]]][['cost']][['results']][[k]]
	cm1 <- cost_metrics[1]
	print(
		((cost_res2$plots[[cm1]] +
			ggplot2::guides(col = FALSE) +
			ggplot2::labs(title = 'Daily')) +
			(cost_res3$plots[[cm1]] +
				ggplot2::labs(title = 'Weekly'))) +
			patchwork::plot_layout(guides = "collect") &
			ggplot2::theme(legend.position = "bottom")
	)
}

for (k in mod_tps) {
	cost_res2 <- res[[df_nms[2]]][['cost']][['results']][[k]]
	cost_res3 <- res[[df_nms[3]]][['cost']][['results']][[k]]
	cm2 <- cost_metrics[2]
	print(
		((cost_res2$plots[[cm2]] +
			ggplot2::guides(col = FALSE) +
			ggplot2::labs(title = 'Daily')) +
			(cost_res3$plots[[cm2]] + ggplot2::labs(title = 'Weekly'))) +
			patchwork::plot_layout(guides = "collect") &
			ggplot2::theme(legend.position = "bottom")
	)
}


# =========================================================================
# * Environment -----------------------------------------------------------
# =========================================================================

cost_per_hour <- 3.5
energy_coef <- 0.5 # energy coefficient at full utilization is approximately 0.18-0.30 kWh
pue <- 1.54 # power usage effectiveness (PUE)
carbon_intensity <- 0.38 # kg CO2 per kWh

cost_data_daily <- res[[df_nms[2]]][['cost']]$data |>
	dplyr::group_by(retrain_window) |>
	dplyr::summarise('average' = mean(.data[['cost']]), .groups = 'drop') |>
	dplyr::mutate(type = 'Daily', method = 'Daily', .before = 1) |>
	purrr::set_names(c('type', 'method', 'retrain_window', 'cost')) |>
	dplyr::mutate(
		ct_hours = cost / cost_per_hour,
		energy_kwh = ct_hours * energy_coef * pue,
		carbon_kg = energy_kwh * carbon_intensity,
		energy_mwh = energy_kwh / 1000,
		carbon_tons = carbon_kg / 1000
	)
cost_data_weekly <- res[[df_nms[3]]][['cost']]$data |>
	dplyr::group_by(retrain_window) |>
	dplyr::summarise('average' = mean(.data[['cost']]), .groups = 'drop') |>
	dplyr::mutate(type = 'Weekly', method = 'Weekly', .before = 1) |>
	purrr::set_names(c('type', 'method', 'retrain_window', 'cost')) |>
	dplyr::mutate(
		ct_hours = cost / cost_per_hour,
		energy_kwh = ct_hours * energy_coef * pue,
		carbon_kg = energy_kwh * carbon_intensity,
		energy_mwh = energy_kwh / 1000,
		carbon_tons = carbon_kg / 1000
	)
cost_data <- dplyr::bind_rows(cost_data_daily, cost_data_weekly)

cost_data_daily |>
	dplyr::select(method, retrain_window, energy_mwh, carbon_tons)
cost_data_weekly |>
	dplyr::select(method, retrain_window, energy_mwh, carbon_tons)

p_daily <- cost_data_daily |>
	plot_retrain_results(
		metric = 'carbon_tons',
		metric_label = 'Carbon Emissions (tons CO2)',
		title = 'Daily'
	) +
	ggplot2::scale_y_continuous(
		breaks = scales::pretty_breaks(n = 10),
		labels = function(x) scales::number(x, accuracy = 1)
	) +
	ggplot2::theme(legend.position = "none")

p_weekly <- cost_data_weekly |>
	plot_retrain_results(
		metric = 'carbon_tons',
		metric_label = 'Carbon Emissions (tons CO2)',
		title = 'Weekly'
	) +
	ggplot2::scale_y_continuous(
		breaks = scales::pretty_breaks(n = 10),
		labels = function(x) scales::number(x, accuracy = 0.1)
	) +
	ggplot2::theme(legend.position = "none")

p <- p_daily + p_weekly + patchwork::plot_layout(guides = "collect")
