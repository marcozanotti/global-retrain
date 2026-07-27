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
p_gpu <- 0.300 # NVIDIA V100 SXM2, rated power (kW)
p_cpu <- 3 / 14 * 0.135 # 6 vCPU = 3 physical cores of a 14-core Xeon E5-2690 v4 (135 W)
p_mem <- 112 * 0.392 / 1000 # 112 GB at 0.392 W/GB
energy_coef <- (p_gpu + p_cpu + p_mem)
pue <- 1.54 # Uptime Institute Global Data Center Survey 2025
carbon_intensity_us <- 0.379 # EPA eGRID, US national average (kg CO2e/kWh)
carbon_intensity_eu <- 0.255 # EEA, EU average -- CONFIRM REFERENCE YEAR

cost_data_daily <- res[[df_nms[2]]][['cost']]$data |>
	dplyr::group_by(retrain_window) |>
	dplyr::summarise('average' = mean(.data[['cost']]), .groups = 'drop') |>
	dplyr::mutate(type = 'Daily', method = 'Daily', .before = 1) |>
	purrr::set_names(c('type', 'method', 'retrain_window', 'cost')) |>
	dplyr::mutate(
		ct_hours = cost / cost_per_hour,
		energy_kwh = ct_hours * energy_coef * pue,
		energy_mwh = energy_kwh / 1000,
		carbon_kg_us = energy_kwh * carbon_intensity_us,
		carbon_kg_eu = energy_kwh * carbon_intensity_eu,
		carbon_tons_us = carbon_kg_us / 1000,
		carbon_tons_eu = carbon_kg_eu / 1000
	)
cost_data_weekly <- res[[df_nms[3]]][['cost']]$data |>
	dplyr::group_by(retrain_window) |>
	dplyr::summarise('average' = mean(.data[['cost']]), .groups = 'drop') |>
	dplyr::mutate(type = 'Weekly', method = 'Weekly', .before = 1) |>
	purrr::set_names(c('type', 'method', 'retrain_window', 'cost')) |>
	dplyr::mutate(
		ct_hours = cost / cost_per_hour,
		energy_kwh = ct_hours * energy_coef * pue,
		energy_mwh = energy_kwh / 1000,
		carbon_kg_us = energy_kwh * carbon_intensity_us,
		carbon_kg_eu = energy_kwh * carbon_intensity_eu,
		carbon_tons_us = carbon_kg_us / 1000,
		carbon_tons_eu = carbon_kg_eu / 1000
	)
cost_data <- dplyr::bind_rows(cost_data_daily, cost_data_weekly)

cost_data_daily |> dplyr::select(-type, -method)
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
p


# SIMULATION --------------------------------------------------------
set.seed(1234)
n_sim <- 10000

draws <- tibble::tibble(
	sim = seq_len(n_sim),
	p_kw = runif(n_sim, 0.18, 0.30), # implies GPU utilisation ~0.36-0.76
	pue = runif(n_sim, 1.12, 1.54), # best-in-class to global average
	ci = runif(n_sim, 0.15, 0.45) # EU average through US average, with margin
)

# ---- full version: keeps every draw, lets you plot per-scenario densities ----
sim_daily <- cost_data_daily |>
	dplyr::select(retrain_window, cost) |>
	tidyr::crossing(draws) |>
	dplyr::mutate(
		ct_hours = cost / cost_per_hour,
		energy_mwh = ct_hours * p_kw * pue / 1000,
		carbon_t = energy_mwh * ci
	)

bands_daily <- sim_daily |>
	dplyr::group_by(retrain_window) |>
	dplyr::summarise(
		energy_lo = quantile(energy_mwh, 0.05),
		energy_md = median(energy_mwh),
		energy_hi = quantile(energy_mwh, 0.95),
		carbon_lo = quantile(carbon_t, 0.05),
		carbon_md = median(carbon_t),
		carbon_hi = quantile(carbon_t, 0.95),
		.groups = 'drop'
	)

# ---- shortcut: identical result, 10,000x fewer rows -------------------------
mult <- draws |>
	dplyr::summarise(
		e_lo = quantile(p_kw * pue, 0.05),
		e_md = median(p_kw * pue),
		e_hi = quantile(p_kw * pue, 0.95),
		c_lo = quantile(p_kw * pue * ci, 0.05),
		c_md = median(p_kw * pue * ci),
		c_hi = quantile(p_kw * pue * ci, 0.95)
	)

bands_fast <- cost_data_daily |>
	dplyr::transmute(
		retrain_window,
		ct_hours = cost / cost_per_hour,
		energy_lo = ct_hours * mult$e_lo / 1000,
		energy_md = ct_hours * mult$e_md / 1000,
		energy_hi = ct_hours * mult$e_hi / 1000,
		carbon_lo = ct_hours * mult$c_lo / 1000,
		carbon_md = ct_hours * mult$c_md / 1000,
		carbon_hi = ct_hours * mult$c_hi / 1000
	)

# ---- which parameter drives the spread (analytic for a multiplicative model) --
draws |>
	dplyr::summarise(P = var(log(p_kw)), PUE = var(log(pue)), I = var(log(ci))) |>
	tidyr::pivot_longer(
		dplyr::everything(),
		names_to = 'parameter',
		values_to = 'var_log'
	) |>
	dplyr::mutate(share = var_log / sum(var_log))


library(ggplot2)

# regenerate bands with unnamed quantiles (safe version)
bands_daily <- sim_daily |>
	dplyr::group_by(retrain_window) |>
	dplyr::summarise(
		energy_lo = as.numeric(quantile(energy_mwh, 0.05)),
		energy_md = as.numeric(median(energy_mwh)),
		energy_hi = as.numeric(quantile(energy_mwh, 0.95)),
		carbon_lo = as.numeric(quantile(carbon_t, 0.05)),
		carbon_md = as.numeric(median(carbon_t)),
		carbon_hi = as.numeric(quantile(carbon_t, 0.95)),
		.groups = 'drop'
	) |>
	dplyr::arrange(retrain_window)

baseline_r <- min(bands_daily$retrain_window)

tab <- bands_daily |>
	dplyr::mutate(
		reduction = 100 * (1 - energy_md / energy_md[retrain_window == baseline_r]),
		col_r = as.character(retrain_window),
		col_energy = sprintf('%.1f [%.1f, %.1f]', energy_md, energy_lo, energy_hi),
		col_carbon = sprintf('%.2f [%.2f, %.2f]', carbon_md, carbon_lo, carbon_hi),
		col_red = ifelse(
			retrain_window == baseline_r,
			'---',
			sprintf('%.1f', reduction)
		)
	)

latex_table <- c(
	'\\begin{table}[htbp]',
	'    \\centering',
	'    \\caption{Estimated annual energy consumption and carbon emissions of the daily',
	'    retail forecasting operation described in Chapter~\\ref{ch1:chapter}, by retraining',
	'    scenario. Figures are medians over 10{,}000 Monte Carlo draws of the power',
	'    coefficient, power usage effectiveness and grid carbon intensity, with 90\\%',
	'    intervals in brackets. Reductions are computed against the weekly benchmark and',
	'    are invariant to the sampled parameters.}',
	'    \\label{tab:env_results}',
	'    \\begin{tabular}{rccc}',
	'        \\toprule',
	'        $r$ & Energy (MWh) & Carbon (t\\,CO$_2$e) & Reduction (\\%) \\\\',
	'        \\midrule',
	paste0(
		'        ',
		tab$col_r,
		' & ',
		tab$col_energy,
		' & ',
		tab$col_carbon,
		' & ',
		tab$col_red,
		' \\\\'
	),
	'        \\bottomrule',
	'    \\end{tabular}',
	'\\end{table}'
)
cat(latex_table, sep = '\n')

thesis_red <- rgb(0.631, 0.094, 0.094)

plot_data <- dplyr::bind_rows(
	bands_daily |>
		dplyr::transmute(
			retrain_window,
			metric = 'Energy (MWh)',
			lo = energy_lo,
			md = energy_md,
			hi = energy_hi
		),
	bands_daily |>
		dplyr::transmute(
			retrain_window,
			metric = 'Carbon (tonnes CO2e)',
			lo = carbon_lo,
			md = carbon_md,
			hi = carbon_hi
		)
) |>
	dplyr::mutate(
		metric = factor(metric, levels = c('Energy (MWh)', 'Carbon (tonnes CO2e)')),
		r_fct = factor(retrain_window, levels = sort(unique(retrain_window)))
	)

p <- ggplot(plot_data, aes(x = r_fct, group = 1)) +
	geom_ribbon(aes(ymin = lo, ymax = hi), fill = thesis_red, alpha = 0.18) +
	geom_line(aes(y = md), colour = thesis_red, linewidth = 0.6) +
	geom_point(aes(y = md), colour = thesis_red, size = 1.5) +
	facet_wrap(~metric, scales = 'free_y') +
	scale_y_continuous(limits = c(0, NA), expand = expansion(mult = c(0, 0.05))) +
	labs(x = 'Retraining scenario (r)', y = NULL) +
	theme_bw(base_size = 10, base_family = 'serif') +
	theme(
		panel.grid.minor = element_blank(),
		strip.background = element_rect(fill = 'grey92', colour = 'grey70'),
		strip.text = element_text(size = 9)
	)


# =========================================================================
# * ISF 2026 --------------------------------------------------------------
# =========================================================================

# 700x400
for (k in mod_tps) {
	df_nm <- df_nms[2]
	time_resx <- res[[df_nm]][['time']][['results']][[k]]
	cost_resx <- res[[df_nm]][['cost']][['results']][[k]]
	tm <- time_metrics[1]
	cm1 <- cost_metrics[1]
	cm2 <- cost_metrics[2]
	print(
		(time_resx$plots[[tm]] +
			ggplot2::labs(title = 'Relative Computing Time', y = '') +
			ggplot2::guides(col = FALSE)) +
			(cost_resx$plots[[cm1]] +
				ggplot2::labs(title = 'Absolute Costs ($)', y = '')) +
			(cost_resx$plots[[cm2]] + ggplot2::labs(title = 'Savings (%)', y = '')) +
			patchwork::plot_layout(guides = "collect") &
			ggplot2::theme(legend.position = "bottom")
	)
}
