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

# run twice, one for absolute and one for relative
analysis_file_name <- 'docs/drift_retrain/absolute_evaltimecost_overlap_20260629_001122.RData'
analysis_file_name <- 'docs/drift_retrain/relative_evaltimecost_overlap_20260629_001102.RData'

res <- load(analysis_file_name)
res <- analysis_results
rm(analysis_results)


# Analysis ----------------------------------------------------------------

# * Parameters ------------------------------------------------------------

df_nm <- 'hapag_region_weekly'
mod_tps <- c('SF', 'ML_DL')
eval_metrics <- c('rmsse', 'scaled_mqloss')
time_metrics <- c('total_sample_time')
cost_metrics <- c('cost', 'savings_perc')


# =========================================================================
# * Evaluation ------------------------------------------------------------
# =========================================================================

# ** Tables ---------------------------------------------------------------
for (i in seq_along(eval_metrics)) {
	em <- eval_metrics[i]
	tab_eval <- dplyr::bind_rows(
		res[[df_nm]][['evaluation']][['results']][[mod_tps[1]]]$tables[[
			em
		]]$x$data,
		res[[df_nm]][['evaluation']][['results']][[mod_tps[2]]]$tables[[
			em
		]]$x$data
	)
	cat(paste(df_nm, em, "\n\n"))
	print(xtable::xtable(tab_eval, digits = 3), include.rownames = FALSE)
}

# ** Plots -----------------------------------------------------------------
for (k in mod_tps) {
	eval_res <- res[[df_nm]][['evaluation']][['results']][[k]]
	em1 <- eval_metrics[1]
	em2 <- eval_metrics[2]
	print(
		(eval_res$plots[[em1]] + ggplot2::labs(title = NULL)) +
			(eval_res$plots[[em2]] + ggplot2::labs(title = NULL)) +
			patchwork::plot_layout(guides = "collect") &
			ggplot2::theme(legend.position = "bottom")
	)
}

# ** Tests -----------------------------------------------------------------

# 1000x700 horizontal
for (k in mod_tps) {
	eval_res <- res[[df_nm]][['evaluation']][['results']][[k]]
	em1 <- eval_metrics[1]
	em1_lbl <- toupper(gsub(
		"_",
		" ",
		ifelse(em1 == "scaled_mqloss", "smql", em1)
	))
	em2 <- eval_metrics[2]
	em2_lbl <- toupper(gsub(
		"_",
		" ",
		ifelse(em2 == "scaled_mqloss", "smql", em2)
	))
	ge1 <- plot_test_results_facet(
		data = eval_res$tests[[em1]],
		.metric = em1,
		by = "retrain_window",
		metric_label = em1_lbl,
		title = paste(em1_lbl, "- Nemenyi Test")
	)
	ge2 <- plot_test_results_facet(
		data = eval_res$tests[[em2]],
		.metric = em2,
		by = "retrain_window",
		metric_label = em2_lbl,
		title = paste(em2_lbl, "- Nemenyi Test")
	)
	if (k == "SF") {
		g <- ge1 / ge2
		print(g)
	} else {
		print(ge1)
		print(ge2)
	}
}

# 750x650 vertical
for (k in mod_tps) {
	eval_res <- res[[df_nm]][['evaluation']][['results']][[k]]
	em1 <- eval_metrics[1]
	em1_lbl <- toupper(gsub(
		"_",
		" ",
		ifelse(em1 == "scaled_mqloss", "smql", em1)
	))
	em2 <- eval_metrics[2]
	em2_lbl <- toupper(gsub(
		"_",
		" ",
		ifelse(em2 == "scaled_mqloss", "smql", em2)
	))
	ge1 <- plot_test_results_facet(
		data = eval_res$tests[[em1]],
		.metric = em1,
		by = "retrain_window",
		metric_label = em1_lbl,
		title = paste(em1_lbl, "- Nemenyi Test")
	)
	ge2 <- plot_test_results_facet(
		data = eval_res$tests[[em2]],
		.metric = em2,
		by = "retrain_window",
		metric_label = em2_lbl,
		title = paste(em2_lbl, "- Nemenyi Test")
	)
	if (k == "SF") {
		g <- ge1 + ge2
		print(g)
	} else {
		print(ge1)
		print(ge2)
	}
}


# =========================================================================
# * Time ------------------------------------------------------------------
# =========================================================================

# ** Tables ---------------------------------------------------------------
for (i in seq_along(time_metrics)) {
	tm <- time_metrics[i]
	tab_time <- dplyr::bind_rows(
		res[[df_nm]][['time']][['results']][[mod_tps[1]]]$tables[[tm]]$x$data,
		res[[df_nm]][['time']][['results']][[mod_tps[2]]]$tables[[tm]]$x$data
	)
	cat(paste(df_nm, tm, "\n\n"))
	print(xtable::xtable(tab_time, digits = 3), include.rownames = FALSE)
}

# ** Plots -----------------------------------------------------------------
for (k in mod_tps) {
	time_res <- res[[df_nm]][['time']][['results']][[k]]
	tm <- time_metrics[1]
	print(time_res$plots[[tm]] + ggplot2::labs(title = NULL))
}


# =========================================================================
# * Cost ------------------------------------------------------------------
# =========================================================================

# ** Tables ---------------------------------------------------------------
for (i in seq_along(cost_metrics)) {
	cm <- cost_metrics[i]
	tab_cost <- dplyr::bind_rows(
		res[[df_nm]][['cost']][['results']][[mod_tps[1]]]$tables[[cm]]$x$data,
		res[[df_nm]][['cost']][['results']][[mod_tps[2]]]$tables[[cm]]$x$data
	)
	cat(paste(df_nm, cm, "\n\n"))
	print(xtable::xtable(tab_cost, digits = 3), include.rownames = FALSE)
}

# ** Plots -----------------------------------------------------------------
cost_res1 <- res[[df_nm]][['cost']][['results']][[mod_tps[1]]]
cost_res2 <- res[[df_nm]][['cost']][['results']][[mod_tps[2]]]

(cost_res1$plots[['cost']] + ggplot2::labs(title = NULL)) +
	(cost_res1$plots[['savings_perc']] + ggplot2::labs(title = NULL)) +
	patchwork::plot_layout(guides = "collect") &
	ggplot2::theme(legend.position = "bottom")

(cost_res2$plots[['cost']] + ggplot2::labs(title = NULL)) +
	(cost_res2$plots[['savings_perc']] + ggplot2::labs(title = NULL)) +
	patchwork::plot_layout(guides = "collect") &
	ggplot2::theme(legend.position = "bottom")


# =========================================================================
# * Environmental ---------------------------------------------------------
# =========================================================================

cost_per_hour <- 3.5
energy_coef <- 0.5 # energy coefficient at full utilization is approximately 0.18-0.30 kWh
pue <- 1.54 # power usage effectiveness (PUE)
carbon_intensity <- 0.38 # kg CO2 per kWh

cost_data <- res[[df_nm]][['cost']]$data |>
	dplyr::filter(type != 'SF') |>
	dplyr::group_by(retrain_window) |>
	dplyr::summarise('average' = mean(.data[['cost']]), .groups = 'drop') |>
	dplyr::mutate(type = 'x', method = 'x', .before = 1) |>
	purrr::set_names(c('type', 'method', 'retrain_window', 'cost')) |>
	dplyr::mutate(
		ct_hours = cost / cost_per_hour,
		energy_kwh = ct_hours * energy_coef * pue,
		carbon_kg = energy_kwh * carbon_intensity,
		energy_mwh = energy_kwh / 1000,
		carbon_tons = carbon_kg / 1000
	)

# ** Tables ---------------------------------------------------------------
tab_env <- cost_data |>
	dplyr::select(retrain_window, energy_mwh, carbon_tons) |>
	tidyr::pivot_longer(
		cols = c(energy_mwh, carbon_tons),
		names_to = 'metric',
		values_to = 'value'
	) |>
	dplyr::mutate(
		metric = dplyr::case_when(
			metric == 'energy_mwh' ~ 'Energy (MWh)',
			metric == 'carbon_tons' ~ 'Carbon Emissions (tons CO2)',
			TRUE ~ metric
		)
	) |>
	tidyr::pivot_wider(names_from = retrain_window, values_from = value)
print(xtable::xtable(tab_env, digits = 3), include.rownames = FALSE)

# ** Plots -----------------------------------------------------------------
cost_data |>
	dplyr::select(retrain_window, energy_mwh, carbon_tons) |>
	tidyr::pivot_longer(
		cols = c(energy_mwh, carbon_tons),
		names_to = 'metric',
		values_to = 'value'
	) |>
	dplyr::mutate(
		metric = dplyr::case_when(
			metric == 'energy_mwh' ~ 'Energy (MWh)',
			metric == 'carbon_tons' ~ 'Carbon Emissions (tons CO2)',
			TRUE ~ metric
		)
	) |>
	dplyr::mutate(retrain_window = as.factor(retrain_window)) |>
	ggplot2::ggplot(ggplot2::aes(
		x = retrain_window,
		y = value,
		fill = metric
	)) +
	ggplot2::geom_bar(stat = 'identity', position = 'dodge') +
	ggplot2::labs(
		x = 'Retrain Scenario (r)',
		y = '',
		title = 'Environmental Impact of Retraining Scenarios',
	) +
	ggplot2::theme_minimal() +
	ggplot2::scale_fill_manual(values = c('#129f4d', '#dc7114')) +
	ggplot2::theme(
		plot.title = ggplot2::element_text(hjust = 0.5),
		legend.position = 'bottom'
	) +
	ggplot2::guides(fill = ggplot2::guide_legend(title = 'Metric'))


# =========================================================================
# * Optimal Retraining Scenario -------------------------------------------
# =========================================================================

config <- get_config('config/anal/anal_drift_retrain_config.yaml')
opt_freq <- analyze_optimal_frequency(config, adjust = 2)

em1 <- eval_metrics[1]
em1_lbl <- toupper(gsub(
	"_",
	" ",
	ifelse(em1 == "scaled_mqloss", "smql", em1)
))
em2 <- eval_metrics[2]
em2_lbl <- toupper(gsub(
	"_",
	" ",
	ifelse(em2 == "scaled_mqloss", "smql", em2)
))

# SF
(opt_freq[[df_nm]][['evaluation']][['results']][[mod_tps[1]]][['plots']][[
	em1
]][[
	'overall'
]] +
	ggplot2::labs(title = em1_lbl)) +
	(opt_freq[[df_nm]][['evaluation']][['results']][[mod_tps[1]]][['plots']][[
		em2
	]][['overall']] +
		ggplot2::labs(title = em2_lbl))

# ML_DL
(opt_freq[[df_nm]][['evaluation']][['results']][[mod_tps[2]]][['plots']][[
	em1
]][[
	'overall'
]] +
	ggplot2::labs(title = em1_lbl)) +
	(opt_freq[[df_nm]][['evaluation']][['results']][[mod_tps[2]]][['plots']][[
		em2
	]][['overall']] +
		ggplot2::labs(title = em2_lbl))


# =========================================================================
# * Analysis by Groups ----------------------------------------------------
# =========================================================================

config <- get_config('config/anal/anal_drift_retrain_config.yaml')
group_names <- c('breaks') # group_name %in% c('ABC', 'XYZ', 'breaks', 'breaks_multi')
group_res <- analyze_groups(config, group_names = group_names)
lvl <- group_res[[df_nm]][['evaluation']][['data']][['group']] |> unique()

# ** Tables ---------------------------------------------------------------
for (l in lvl) {
	for (i in seq_along(eval_metrics)) {
		em <- eval_metrics[i]
		tab_eval <- dplyr::bind_rows(
			group_res[[df_nm]][['evaluation']][['results']][[mod_tps[1]]]$tables[[
				em
			]]$x$data,
			group_res[[df_nm]][['evaluation']][['results']][[mod_tps[2]]]$tables[[
				em
			]]$x$data
		)
		tab_eval <- tab_eval |>
			dplyr::filter(Group == l) |>
			dplyr::select(-Group)
		cat(paste(df_nm, em, l, "\n\n"))
		print(xtable::xtable(tab_eval, digits = 3), include.rownames = FALSE)
	}
}

# ** Plots -----------------------------------------------------------------
for (k in mod_tps) {
	eval_res <- group_res[[df_nm]][['evaluation']][['results']][[k]]
	em1 <- eval_metrics[1]
	em2 <- eval_metrics[2]
	print(
		(eval_res$plots[[em1]] + ggplot2::labs(title = NULL)) /
			(eval_res$plots[[em2]] + ggplot2::labs(title = NULL)) +
			patchwork::plot_layout(guides = "collect") &
			ggplot2::theme(legend.position = "bottom")
	)
}

# ** Tests -----------------------------------------------------------------

# 1000x700 horizontal
for (l in lvl) {
	cat(paste("Group:", l, "\n"))
	for (k in mod_tps) {
		eval_res <- group_res[[df_nm]][['evaluation']][['results']][[k]]
		em1 <- eval_metrics[1]
		em1_lbl <- toupper(gsub(
			"_",
			" ",
			ifelse(em1 == "scaled_mqloss", "smql", em1)
		))
		em2 <- eval_metrics[2]
		em2_lbl <- toupper(gsub(
			"_",
			" ",
			ifelse(em2 == "scaled_mqloss", "smql", em2)
		))
		ge1 <- plot_test_results_facet(
			data = eval_res$tests[[em1]] |> dplyr::filter(group == l),
			.metric = em1,
			by = "retrain_window",
			metric_label = em1_lbl,
			title = paste(em1_lbl, "- Nemenyi Test")
		)
		ge2 <- plot_test_results_facet(
			data = eval_res$tests[[em2]] |> dplyr::filter(group == l),
			.metric = em2,
			by = "retrain_window",
			metric_label = em2_lbl,
			title = paste(em2_lbl, "- Nemenyi Test")
		)
		if (k == "SF") {
			g <- ge1 / ge2
			print(g)
		} else {
			print(ge1)
			print(ge2)
		}
	}
}


# =========================================================================
# * Optimal Retraining Scenario by Groups ---------------------------------
# =========================================================================

config <- get_config('config/anal/anal_drift_retrain_config.yaml')
group_names <- c('ABC', 'breaks') # group_name %in% c('ABC', 'XYZ', 'breaks', 'breaks_multi')
opt_freq_groups <- analyze_optimal_frequency(
	config,
	adjust = 2,
	group_names = group_names
)

em1 <- eval_metrics[1]
em1_lbl <- toupper(gsub(
	"_",
	" ",
	ifelse(em1 == "scaled_mqloss", "smql", em1)
))
em2 <- eval_metrics[2]
em2_lbl <- toupper(gsub(
	"_",
	" ",
	ifelse(em2 == "scaled_mqloss", "smql", em2)
))

# SF
((opt_freq_groups[[df_nm]][['evaluation']][['results']][[mod_tps[1]]][[
	'plots'
]][[
	em1
]][[
	'overall'
]] +
	ggplot2::labs(title = em1_lbl)) +
	(opt_freq_groups[[df_nm]][['evaluation']][['results']][[mod_tps[1]]][[
		'plots'
	]][[
		em2
	]][['overall']] +
		ggplot2::labs(title = em2_lbl))) +
	patchwork::plot_layout(guides = "collect") &
	ggplot2::theme(legend.position = "bottom")

# ML_DL
((opt_freq_groups[[df_nm]][['evaluation']][['results']][[mod_tps[2]]][[
	'plots'
]][[
	em1
]][[
	'overall'
]] +
	ggplot2::labs(title = em1_lbl)) +
	(opt_freq_groups[[df_nm]][['evaluation']][['results']][[mod_tps[2]]][[
		'plots'
	]][[
		em2
	]][['overall']] +
		ggplot2::labs(title = em2_lbl))) +
	patchwork::plot_layout(guides = "collect") &
	ggplot2::theme(legend.position = "bottom")
