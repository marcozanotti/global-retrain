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
analysis_file_name <- 'docs/drift_retrain/absolute_evalstabtimecost_overlap_20260712_100211.RData'
analysis_file_name <- 'docs/drift_retrain/relative_evalstabtimecost_overlap_20260712_100156.RData'
analysis_file_name <- 'docs/drift_retrain/absolute_'

res <- load(analysis_file_name)
res <- analysis_results
rm(analysis_results)


# Analysis ----------------------------------------------------------------

# * Parameters ------------------------------------------------------------

df_nm <- 'hapag_region_weekly'
mod_tps <- c('SF', 'ML_DL')
eval_metrics <- c('rmsse', 'scaled_mqloss')
stab_metrics <- c('smapc', 'smqpc')
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
eval_data <- res[[df_nm]][['evaluation']]$data |>
	dplyr::mutate(type = ifelse(type == "SF", "Local", "Global"))
em1 <- eval_metrics[1]
em2 <- eval_metrics[2]
(plot_retrain_results(
	data = eval_data,
	metric = em1,
	metric_label = "RMSSE",
	by_type = TRUE
) +
	plot_retrain_results(
		data = eval_data,
		metric = em2,
		metric_label = "SMQL",
		by_type = TRUE
	)) +
	patchwork::plot_layout(guides = "collect") &
	ggplot2::theme(legend.position = "bottom")

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
		g <- ((ge1 + ggplot2::facet_wrap(~method, ncol = 1, scales = "fixed")) +
			(ge2 + ggplot2::facet_wrap(~method, ncol = 1, scales = "fixed"))) +
			patchwork::plot_layout(guides = "collect") &
			ggplot2::theme(legend.position = "bottom")
		print(g)
	}
}


# =========================================================================
# * Optimal Retraining Scenario -------------------------------------------
# =========================================================================

config <- get_config('config/anal/anal_drift_retrain_config.yaml')
opt_freq <- analyze_optimal_frequency(config, adjust = 2)

em1 <- eval_metrics[1]
em2 <- eval_metrics[2]

opt_freq_data_em1 <- opt_freq[[df_nm]][['evaluation']][['data']] |>
	dplyr::mutate(
		type = factor(
			ifelse(type == "SF", "Local", "Global"),
			levels = c("Global", "Local")
		)
	) |>
	dplyr::select(dplyr::all_of(c(
		'type',
		'method',
		'retrain_window',
		'unique_id',
		em1
	))) |>
	dplyr::group_by(type, method, unique_id) |>
	dplyr::arrange(type, method, unique_id, .data[[em1]]) |>
	dplyr::slice_head(n = 1) |>
	dplyr::ungroup()
opt_freq_data_em2 <- opt_freq[[df_nm]][['evaluation']][['data']] |>
	dplyr::mutate(
		type = factor(
			ifelse(type == "SF", "Local", "Global"),
			levels = c("Global", "Local")
		)
	) |>
	dplyr::select(dplyr::all_of(c(
		'type',
		'method',
		'retrain_window',
		'unique_id',
		em2
	))) |>
	dplyr::group_by(type, method, unique_id) |>
	dplyr::arrange(type, method, unique_id, .data[[em2]]) |>
	dplyr::slice_head(n = 1) |>
	dplyr::ungroup()

(plot_optimal_retrain_results(
	data = opt_freq_data_em1,
	metric = em1,
	title = 'RMSSE',
	overall_only = FALSE,
	adjust = 2,
	by_type = TRUE
) +
	ggplot2::theme(legend.position = "bottom")) +
	(plot_optimal_retrain_results(
		data = opt_freq_data_em2,
		metric = em2,
		title = 'SMQL',
		overall_only = FALSE,
		adjust = 2,
		by_type = TRUE
	) +
		ggplot2::theme(legend.position = "bottom")) +
	patchwork::plot_layout(guides = "collect") &
	ggplot2::theme(legend.position = "bottom")


# =========================================================================
# * Time, Cost, and Environment -------------------------------------------
# =========================================================================

# ** Tables ---------------------------------------------------------------
# time tables
for (i in seq_along(time_metrics)) {
	tm <- time_metrics[i]
	tab_time <- dplyr::bind_rows(
		res[[df_nm]][['time']][['results']][[mod_tps[1]]]$tables[[tm]]$x$data,
		res[[df_nm]][['time']][['results']][[mod_tps[2]]]$tables[[tm]]$x$data
	)
	cat(paste(df_nm, tm, "\n\n"))
	print(xtable::xtable(tab_time, digits = 3), include.rownames = FALSE)
}

# cost tables
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
time_data <- res[[df_nm]][['time']]$data |>
	dplyr::mutate(type = ifelse(type == "SF", "Local", "Global"))
cost_data <- res[[df_nm]][['cost']]$data |>
	dplyr::mutate(type = ifelse(type == "SF", "Local", "Global"))
tm1 <- time_metrics[1]
cm2 <- cost_metrics[2]
(plot_retrain_results(
	data = time_data,
	metric = tm1,
	metric_label = "Computating Time",
	by_type = TRUE
) +
	plot_retrain_results(
		data = cost_data,
		metric = cm2,
		metric_label = "Savings (%)",
		by_type = TRUE
	)) +
	patchwork::plot_layout(guides = "collect") &
	ggplot2::theme(legend.position = "bottom")


# =========================================================================
# * Pareto Accuracy-Cost --------------------------------------------------
# =========================================================================

metrics <- c('rmsse', 'cost', 'scaled_mqloss')

acc_data <- res[[df_nm]][['evaluation']]$data
cost_data <- res[[df_nm]][['cost']]$data
pareto_data <- dplyr::left_join(
	acc_data,
	cost_data,
	by = c('type', 'method', 'retrain_window')
) |>
	dplyr::mutate(retrain_window = as.factor(retrain_window)) |>
	dplyr::select(type, method, retrain_window, dplyr::all_of(metrics)) |>
	dplyr::group_by(method) |>
	dplyr::mutate(
		dplyr::across(
			cost,
			~ (.x - min(.x)) / (max(.x) - min(.x)) * 100
		)
	) |>
	dplyr::ungroup()

metrics_plot <- c('rmsse', 'cost')
params <- purrr::map(metrics_plot, ~ get_table_plot_params(.x, 'absolute'))
names(params) <- c('x', 'y')

pareto_data |>
	ggplot2::ggplot(
		ggplot2::aes(
			x = .data[[metrics_plot[1]]],
			y = .data[[metrics_plot[2]]],
			# color = .data[['retrain_window']]
		)
	) +
	ggplot2::geom_point() +
	ggrepel::geom_text_repel(
		ggplot2::aes(label = .data[['retrain_window']]),
		col = 'black',
		vjust = -0.25
	) +
	# ggplot2::scale_x_continuous(labels = params$x$scaling_fun) +
	# ggplot2::scale_y_continuous(labels = params$y$scaling_fun) +
	# ggplot2::scale_color_manual(values = colors_lbls_2) +
	ggplot2::labs(
		title = paste0('Pareto Frontier: ', params$x$label, ' - Cost'),
		x = params$x$label,
		y = 'Normalized Cost'
	) +
	ggplot2::theme_bw() +
	ggplot2::theme(
		plot.title = ggplot2::element_text(hjust = 0.5),
		legend.position = "bottom",
		axis.text.x = ggplot2::element_text(angle = 45, hjust = 1)
	) +
	ggplot2::facet_wrap(~method, ncol = 2, scales = "fixed")


# =========================================================================
# * Analysis by Groups ----------------------------------------------------
# =========================================================================

config <- get_config('config/anal/anal_drift_retrain_config.yaml')

group_names <- c('breaks') # 'ABC', 'XYZ', 'breaks', 'breaks_multi', if c('ABC', 'breaks') then cartesian product

group_levels <- NULL
# group_levels <- c(
# 	'A - No breaks',
# 	'A - Breaks',
# 	'B - No breaks',
# 	'B - Breaks',
# 	'C - No breaks',
# 	'C - Breaks'
# )

# breaks_params <- NULL
breaks_params <- c('260', 'lwz') # only for breaks and breaks_multi, otherwise NULL

group_res <- analyze_groups(
	config,
	group_names = group_names,
	group_levels = group_levels,
	breaks_params = breaks_params
)

analysis <- 'evaluation' # 'evaluation', 'stability', 'evaluation_prepost'
anal_res <- group_res[[df_nm]][[analysis]][['results']]
lvl <- group_res[[df_nm]][[analysis]][['data']][['aggregated']][['group']] |>
	unique()

# ** Tables ---------------------------------------------------------------
# for (l in lvl) {
# 	for (i in seq_along(eval_metrics)) {
# 		em <- eval_metrics[i]
# 		tab_eval <- dplyr::bind_rows(
# 			anal_res[[mod_tps[1]]]$tables[[em]]$x$data,
# 			anal_res[[mod_tps[2]]]$tables[[em]]$x$data
# 		)
# 		tab_eval <- tab_eval |>
# 			dplyr::filter(Group == l) |>
# 			dplyr::select(-Group)
# 		cat(paste(df_nm, em, l, "\n\n"))
# 		print(xtable::xtable(tab_eval, digits = 3), include.rownames = FALSE)
# 	}
# }

# ** Plots -----------------------------------------------------------------
groups_plot_data <- group_res[[df_nm]][[analysis]][['data']][['aggregated']] |>
	dplyr::mutate(
		type = factor(
			ifelse(type == "SF", "Local", "Global"),
			levels = c("Global", "Local")
		)
	)
em1 <- eval_metrics[1]
em2 <- eval_metrics[2]

# 2 levels
# for (k in mod_tps) {
# 	anal_res_k <- anal_res[[k]]
# 	if (analysis %in% c('evaluation', 'evaluation_prepost')) {
# 		m1 <- eval_metrics[1]
# 		m2 <- eval_metrics[2]
# 	} else if (analysis == 'stability') {
# 		m1 <- stab_metrics[1]
# 		m2 <- stab_metrics[2]
# 	}
# 	print(
# 		(anal_res_k$plots[[m1]] + ggplot2::labs(title = NULL)) /
# 			(anal_res_k$plots[[m2]] + ggplot2::labs(title = NULL)) +
# 			patchwork::plot_layout(guides = "collect") &
# 			ggplot2::theme(legend.position = "bottom")
# 	)
# }
((plot_retrain_results(
	data = groups_plot_data,
	metric = em1,
	metric_label = "RMSSE",
	by_type = TRUE,
	group_col = "group"
)) /
	(plot_retrain_results(
		data = groups_plot_data,
		metric = em2,
		metric_label = "SMQL",
		by_type = TRUE,
		group_col = "group"
	))) +
	patchwork::plot_layout(guides = "collect") &
	ggplot2::theme(legend.position = "bottom")

((plot_retrain_results_differences(
	data = groups_plot_data |>
		dplyr::mutate(type = factor(type, levels = c("Local", "Global"))),
	metric = em1,
	scaling_fun = function(x) scales::percent(x, accuracy = 1), # function(x) scales::number(x, accuracy = 0.001)
	metric_label = "Differences in RMSSE (%)",
	aggregate_by = "type",
	group_col = "group",
	reference_level = "No breaks",
	metric_type = "relative"
)) /
	(plot_retrain_results_differences(
		data = groups_plot_data |>
			dplyr::mutate(type = factor(type, levels = c("Local", "Global"))),
		metric = em2,
		scaling_fun = function(x) scales::percent(x, accuracy = 1),
		metric_label = "Differences in SMQL (%)",
		aggregate_by = "type",
		group_col = "group",
		reference_level = "No breaks",
		metric_type = "relative"
	))) +
	patchwork::plot_layout(guides = "collect") &
	ggplot2::theme(legend.position = "bottom")


# 3 levels
# for (k in mod_tps) {
# 	anal_res_k <- anal_res[[k]]
# 	if (analysis %in% c('evaluation', 'evaluation_prepost')) {
# 		m1 <- eval_metrics[1]
# 		m2 <- eval_metrics[2]
# 	} else if (analysis == 'stability') {
# 		m1 <- stab_metrics[1]
# 		m2 <- stab_metrics[2]
# 	}
# 	print(
# 		(anal_res_k$plots[[m1]] +
# 			ggplot2::labs(title = NULL) +
# 			ggplot2::facet_wrap(~group, ncol = 3, scales = "fixed")) /
# 			(anal_res_k$plots[[m2]] +
# 				ggplot2::labs(title = NULL) +
# 				ggplot2::facet_wrap(~group, ncol = 3, scales = "fixed")) +
# 			patchwork::plot_layout(guides = "collect") &
# 			ggplot2::theme(legend.position = "bottom")
# 	)
# }
((plot_retrain_results(
	data = groups_plot_data,
	metric = em1,
	metric_label = "RMSSE",
	by_type = TRUE,
	group_col = "group"
) +
	ggplot2::facet_wrap(~group, ncol = 3, scales = "fixed")) /
	(plot_retrain_results(
		data = groups_plot_data,
		metric = em2,
		metric_label = "SMQL",
		by_type = TRUE,
		group_col = "group"
	) +
		ggplot2::labs(title = NULL) +
		ggplot2::facet_wrap(~group, ncol = 3, scales = "fixed"))) +
	patchwork::plot_layout(guides = "collect") &
	ggplot2::theme(legend.position = "bottom")

((plot_retrain_results_differences(
	data = groups_plot_data |>
		dplyr::mutate(type = factor(type, levels = c("Local", "Global"))),
	metric = em1,
	scaling_fun = function(x) scales::percent(x, accuracy = 1), # function(x) scales::number(x, accuracy = 0.001)
	metric_label = "Differences in RMSSE",
	aggregate_by = "type",
	group_col = "group",
	reference_level = "No breaks",
	metric_type = "relative"
)) /
	(plot_retrain_results_differences(
		data = groups_plot_data |>
			dplyr::mutate(type = factor(type, levels = c("Local", "Global"))),
		metric = em2,
		scaling_fun = function(x) scales::percent(x, accuracy = 1),
		metric_label = "Differences in SMQL",
		aggregate_by = "type",
		group_col = "group",
		reference_level = "No breaks",
		metric_type = "relative"
	))) +
	patchwork::plot_layout(guides = "collect") &
	ggplot2::theme(legend.position = "bottom")

# pre-post
((plot_retrain_results_differences(
	data = groups_plot_data |>
		dplyr::mutate(type = factor(type, levels = c("Local", "Global"))),
	metric = em1,
	scaling_fun = function(x) scales::percent(x, accuracy = 1), # function(x) scales::number(x, accuracy = 0.001)
	metric_label = "Differences in RMSSE",
	aggregate_by = "type",
	group_col = "group",
	reference_level = "T0",
	metric_type = "relative"
)) /
	(plot_retrain_results_differences(
		data = groups_plot_data |>
			dplyr::mutate(type = factor(type, levels = c("Local", "Global"))),
		metric = em2,
		scaling_fun = function(x) scales::percent(x, accuracy = 1),
		metric_label = "Differences in SMQL",
		aggregate_by = "type",
		group_col = "group",
		reference_level = "T0",
		metric_type = "relative"
	))) +
	patchwork::plot_layout(guides = "collect") &
	ggplot2::theme(legend.position = "bottom")

# 6 levels
# for (k in mod_tps) {
# 	anal_res_k <- anal_res[[k]]
# 	if (analysis %in% c('evaluation', 'evaluation_prepost')) {
# 		m1 <- eval_metrics[1]
# 		m2 <- eval_metrics[2]
# 	} else if (analysis == 'stability') {
# 		m1 <- stab_metrics[1]
# 		m2 <- stab_metrics[2]
# 	}
# 	print(
# 		(anal_res_k$plots[[m1]] +
# 			ggplot2::labs(title = NULL) +
# 			ggplot2::facet_wrap(~group, ncol = 2, scales = "fixed")) +
# 			patchwork::plot_layout(guides = "collect") &
# 			ggplot2::theme(legend.position = "bottom")
# 	)
# 	print(
# 		(anal_res_k$plots[[m2]] +
# 			ggplot2::labs(title = NULL) +
# 			ggplot2::facet_wrap(~group, ncol = 2, scales = "fixed")) +
# 			patchwork::plot_layout(guides = "collect") &
# 			ggplot2::theme(legend.position = "bottom")
# 	)
# }
(plot_retrain_results(
	data = groups_plot_data,
	metric = em1,
	metric_label = "RMSSE",
	by_type = TRUE,
	group_col = "group"
) +
	ggplot2::facet_wrap(~group, ncol = 2, scales = "fixed")) +
	ggplot2::theme(legend.position = "bottom")

(plot_retrain_results(
	data = groups_plot_data,
	metric = em2,
	metric_label = "SMQL",
	by_type = TRUE,
	group_col = "group"
) +
	ggplot2::labs(title = NULL) +
	ggplot2::facet_wrap(~group, ncol = 2, scales = "fixed")) +
	ggplot2::theme(legend.position = "bottom")

plot_retrain_results_differences(
	data = groups_plot_data |>
		dplyr::mutate(type = factor(type, levels = c("Local", "Global"))),
	metric = em1,
	scaling_fun = function(x) scales::percent(x, accuracy = 1),
	metric_label = "Differences in RMSSE",
	aggregate_by = "type",
	group_col = "group",
	reference_level = "A - No breaks",
	metric_type = "relative"
) +
	ggplot2::labs(title = NULL) +
	ggplot2::facet_wrap(~group, ncol = 2, scales = "fixed") +
	ggplot2::theme(legend.position = "bottom")

plot_retrain_results_differences(
	data = groups_plot_data |>
		dplyr::mutate(type = factor(type, levels = c("Local", "Global"))),
	metric = em2,
	scaling_fun = function(x) scales::percent(x, accuracy = 1),
	metric_label = "Differences in SMQL",
	aggregate_by = "type",
	group_col = "group",
	reference_level = "A - No breaks",
	metric_type = "relative"
) +
	ggplot2::labs(title = NULL) +
	ggplot2::facet_wrap(~group, ncol = 2, scales = "fixed") +
	ggplot2::theme(legend.position = "bottom")

# ** Tests -----------------------------------------------------------------
group_test_data <- group_res[[df_nm]][[analysis]][['data']][['raw']] |>
	dplyr::mutate(
		type = factor(
			ifelse(type == "SF", "Local", "Global"),
			levels = c("Global", "Local")
		)
	) |>
	aggregate_data(
		group_columns = c('type', 'retrain_window', 'unique_id', 'group'),
		drop_columns = c('method', 'test_window', 'horizon'),
		function_name = 'mean',
		adjust_metrics = TRUE
	) |>
	dplyr::select(dplyr::all_of(c(
		'type',
		'retrain_window',
		'group',
		eval_metrics
	)))


# breaks
group_test_res <- test_group_differences(
	data = group_test_data,
	group_col = 'group',
	type_col = 'type',
	scenario_col = 'retrain_window',
	metrics = eval_metrics,
	ref_group = "No breaks",
	conf_level = 0.95,
	p_adjust = "holm",
	dunn_p_adjust = "holm",
	dunn_always = FALSE,
	alpha = 0.05
)
tabs <- report_group_tests(group_test_res)
names(tabs)
invisible(lapply(tabs, cat, "\n\n")) # print all

# breaks multi
group_test_res <- test_group_differences(
	data = group_test_data,
	group_col = 'group',
	type_col = 'type',
	scenario_col = 'retrain_window',
	metrics = eval_metrics,
	ref_group = "No breaks",
	conf_level = 0.95,
	p_adjust = "holm",
	dunn_p_adjust = "holm",
	dunn_always = FALSE,
	alpha = 0.05
)
tabs <- report_group_tests(group_test_res)
invisible(lapply(tabs, cat, "\n\n")) # print all

# pre-post
group_test_res <- test_group_differences(
	data = group_test_data,
	group_col = 'group',
	type_col = 'type',
	scenario_col = 'retrain_window',
	metrics = eval_metrics,
	ref_group = "T0",
	conf_level = 0.95,
	p_adjust = "holm",
	dunn_p_adjust = "holm",
	dunn_always = TRUE,
	alpha = 0.05
)
tabs <- report_group_tests(group_test_res)
invisible(lapply(tabs, cat, "\n\n")) # print all


# =========================================================================
# * Optimal Retraining Scenario by Groups ---------------------------------
# =========================================================================

config <- get_config('config/anal/anal_drift_retrain_config.yaml')

group_names <- c('breaks') # 'ABC', 'XYZ', 'breaks', 'breaks_multi', if c('ABC', 'breaks') then cartesian product

# breaks_params <- NULL
breaks_params <- c('260', 'lwz') # only for breaks and breaks_multi, otherwise NULL

opt_freq_groups <- analyze_optimal_frequency(
	config,
	adjust = 2,
	group_names = group_names,
	breaks_params = breaks_params
)

analysis <- 'evaluation_prepost' # 'evaluation', 'stability', 'evaluation_prepost'
em1 <- eval_metrics[1]
em2 <- eval_metrics[2]

opt_freq_data_groups_em1 <- opt_freq_groups[[df_nm]][[analysis]][['data']] |>
	dplyr::mutate(
		type = factor(
			ifelse(type == "SF", "Local", "Global"),
			levels = c("Global", "Local")
		)
	) |>
	dplyr::select(dplyr::all_of(c(
		'type',
		'method',
		'retrain_window',
		'unique_id',
		'group',
		em1
	))) |>
	dplyr::group_by(type, method, group, unique_id) |>
	dplyr::arrange(type, method, group, unique_id, .data[[em1]]) |>
	dplyr::slice_head(n = 1) |>
	dplyr::ungroup()
opt_freq_data_groups_em2 <- opt_freq_groups[[df_nm]][[analysis]][['data']] |>
	dplyr::mutate(
		type = factor(
			ifelse(type == "SF", "Local", "Global"),
			levels = c("Global", "Local")
		)
	) |>
	dplyr::select(dplyr::all_of(c(
		'type',
		'method',
		'retrain_window',
		'unique_id',
		'group',
		em2
	))) |>
	dplyr::group_by(type, method, group, unique_id) |>
	dplyr::arrange(type, method, group, unique_id, .data[[em2]]) |>
	dplyr::slice_head(n = 1) |>
	dplyr::ungroup()

(plot_optimal_retrain_results(
	data = opt_freq_data_groups_em1,
	metric = em1,
	title = 'RMSSE',
	overall_only = FALSE,
	adjust = 2,
	group_col = 'group',
	by_type = TRUE
) +
	ggplot2::theme(legend.position = "bottom")) +
	(plot_optimal_retrain_results(
		data = opt_freq_data_groups_em2,
		metric = em2,
		title = 'SMQL',
		overall_only = FALSE,
		adjust = 2,
		group_col = 'group',
		by_type = TRUE
	) +
		ggplot2::theme(legend.position = "bottom")) +
	patchwork::plot_layout(guides = "collect") &
	ggplot2::theme(legend.position = "bottom")
