
get_retrain_ids <- function(test_window, horizon, retrain_window = 1) {
  res = seq.int(from = 0, to = (test_window - horizon + 1), by = retrain_window)
  return(res)
}

get_aggregate_function <- function(function_name) {

  if (function_name == 'mean') {
    return(mean)
  } else if (function_name == 'median') {
    return(median)
  } else if(function_name == 'std') {
    return(sd)
  } else if (function_name == 'max') {
    return(max)
  } else if(function_name =='min') {
    return(min)
  } else if(function_name == 'sum') {
    return(sum)
  } else {
    stop('Invalid aggregate function')
  }
  
}

aggregate_data <- function(
  data, 
  group_columns, 
  drop_columns = NULL, 
  function_name = 'median', 
  adjust_metrics = False
) {

  data_agg = data

  if (!is.null(drop_columns)) {
    data_agg <- data_agg |> 
      dplyr::select(-dplyr::any_of(drop_columns))
  }

  agg_fun <- get_aggregate_function(function_name)

  data_agg <- data_agg |> 
    dplyr::group_by(!!!rlang::syms(group_columns)) |> 
    dplyr::summarise(dplyr::across(where(is.numeric), agg_fun), .groups = 'drop')

  if (adjust_metrics) {

    if ('mse' %in% names(data_agg))
      data_agg[['rmse']] = sqrt(data_agg[['mse']])

    if ('msse' %in% names(data_agg))
      data_agg[['rmsse']] = sqrt(data_agg[['msse']])
    
    if ('total_fit_time' %in% names(data_agg)) {
      model_names <- unique(data_agg[['method']])
      retrain_scenarios <- unique(data_agg[['retrain_window']])
      total_fit_time <- c()
      for (m in model_names) {
        for (rs in retrain_scenarios) {
          data_tmp <- data |> dplyr::filter(method == m & retrain_window == rs)
          ids_tmp <- get_retrain_ids(
            data_tmp[['test_window']][1], 
            data_tmp[['horizon']][1], 
            data_tmp[['retrain_window']][1]
          )
          tot_fit_time_tmp <- data_tmp[['total_fit_time']][ids_tmp]
          total_fit_time <- c(total_fit_time, agg_fun(tot_fit_time_tmp))
        }
      }
      data_agg[['total_fit_time']] = total_fit_time
    }

  }

  return(data_agg)

}

get_model_type <- function(model_name) {

  sf <- c('ETS', 'ARIMA')
  ml <- c(
    'LinearRegression', 'Lasso', 'Ridge', 
    'RandomForestRegressor', 
    'XGBRegressor', 'LGBMRegressor', 'CatBoostRegressor' 
  )
  dl <- c('MLP', 'LSTM', 'TCN', 'NBEATSx', 'NHITS')

  model_type <- dplyr::case_when(
    model_name %in% sf ~ 'SF',
    model_name %in% ml ~ 'ML',
    model_name %in% dl ~ 'DL',
    TRUE ~ 'ENS'
  )

  return(model_type)

}

get_model_name_abbr <- function(model_name) {

  model_name_abbr <- dplyr::case_when(
    model_name == 'LinearRegression' ~ 'LR',
    model_name == 'RandomForestRegressor' ~ 'RF',
    model_name == 'XGBRegressor' ~ 'XGBoost',
    model_name == 'LGBMRegressor' ~ 'LGBM',
    model_name == 'CatBoostRegressor' ~ 'CatBoost',
    model_name == 'MLP' ~ 'MLP',
    model_name == 'LSTM' ~ 'LSTM',
    model_name == 'TCN' ~ 'TCN',
    model_name == 'NBEATSx' ~ 'NBEATSx',
    model_name == 'NHITS' ~ 'NHITS',
    model_name == 'EnsembleMean2A' ~ 'Ens2A',
    model_name == 'EnsembleMean2T' ~ 'Ens2T',
    model_name == 'EnsembleMean3A' ~ 'Ens3A',
    model_name == 'EnsembleMean3T' ~ 'Ens3T',
    model_name == 'EnsembleMean4A' ~ 'Ens4A',
    model_name == 'EnsembleMean4T' ~ 'Ens4T',
    model_name == 'EnsembleMean5A' ~ 'Ens5A',
    model_name == 'EnsembleMean5T' ~ 'Ens5T',
    TRUE ~ model_name
  )
  return(model_name_abbr)

}

recode_data <- function(data, model_type_levels, model_names_abbr) {

  data_recoded <- data |>  
    dplyr::mutate(
      type = get_model_type(method),
      type = factor(type, levels = model_type_levels, ordered = TRUE),
      .before = 'method',
    ) |> 
    dplyr::mutate(
      method = factor(get_model_name_abbr(method), levels = model_names_abbr, ordered = TRUE)
    ) |> 
    dplyr::arrange(type, method)
  return(data_recoded)

}

dt_table <- function(data, title = "", caption = "", rownames = FALSE, digits = 2, format = 'numeric') {
	
	p_len <- nrow(data)
	
	if (format == 'dollar') {
		res_data <- data |>
			dplyr::mutate(dplyr::across(where(is.numeric), ~ round(.x, digits = digits))) |>  
			dplyr::mutate(dplyr::across(where(is.numeric), ~ scales::dollar(.x, big.mark = ",", decimal.mark = '.')))
	} else if (format == 'percent') {
		res_data <- data |>
			dplyr::mutate(dplyr::across(where(is.numeric), ~ round(.x, digits = digits))) |>  
			dplyr::mutate(dplyr::across(where(is.numeric), ~ scales::percent(.x, scale = 1)))
	} else {
		res_data <- data |>
			dplyr::mutate(dplyr::across(where(is.numeric), ~ round(.x, digits = digits)))
	}
	
	res <- res_data |> 
		DT::datatable(
			extensions = "Buttons",
			# filter = "top", # for filtering enable searching and lenghtChange
			options = list(
				pageLength = p_len,
				paging = FALSE,
				searching = FALSE,
				ordering = FALSE,
				lenghtChange = FALSE,
				autoWidth = FALSE,
				dom = "Bfrtip",
				buttons = c("copy", "print", "csv", "excel", "pdf"),
				drawCallback = DT::JS(
					c(
						"function(settings){",
						"  var datatable = settings.oInstance.api();",
						"  var table = datatable.table().node();",
						paste0("  var caption = '", caption, "'"),
						"  $(table).append('<caption style=\"caption-side: bottom\">' + caption + '</caption>');",
						"}"
					)
				)
			),
			rownames = rownames,
			caption = title
		)
	return(res)
	
}

compute_relative_metrics <- function(data, type) {

  reference_data <- data |> 
    dplyr::group_by(method) |> 
    dplyr::slice_min(retrain_window) |> 
    dplyr::ungroup() |> 
    dplyr::select(-dplyr::any_of(c('type', 'retrain_window'))) |> 
    dplyr::rename_with(~ stringr::str_c(.x, "_ref"))

  if (type == 'evaluation') {

    relative_data <- data |> 
      dplyr::left_join(reference_data, by = c("method" = "method_ref")) |>
      dplyr::mutate(
        bias = abs(bias) / abs(bias_ref),
        # coverage_level50 = coverage_level50 / coverage_level50_ref,
        # coverage_level60 = coverage_level60 / coverage_level60_ref,
        # coverage_level70 = coverage_level70 / coverage_level70_ref,
        # coverage_level80 = coverage_level80 / coverage_level80_ref,
        # coverage_level90 = coverage_level90 / coverage_level90_ref,
        # coverage_level95 = coverage_level95 / coverage_level95_ref,
        # coverage_level99 = coverage_level99 / coverage_level99_ref,
        mae = mae / mae_ref,
        mase = mase / mase_ref,
        mqloss = mqloss / mqloss_ref,
        mse = mse / mse_ref,
        msse = msse / msse_ref,
        rmse = rmse / rmse_ref,
        rmsse = rmsse / rmsse_ref,
        scaled_crps = scaled_crps / scaled_crps_ref
      ) |> 
      dplyr::select(-dplyr::ends_with("_ref"))

  } else if (type == 'time') {

    relative_data <- data |> 
      dplyr::left_join(reference_data, by = c("method" = "method_ref")) |> 
      dplyr::mutate(
        total_fit_time = total_fit_time / total_fit_time_ref,
        total_predict_time = total_predict_time / total_predict_time_ref,
        total_sample_time = total_sample_time / total_sample_time_ref
      ) |> 
      dplyr::select(-dplyr::ends_with("_ref"))

  } else if (type == 'stability') {

    relative_data <- data |> 
      dplyr::left_join(reference_data, by = c("method" = "method_ref")) |>
      dplyr::mutate(
        stability_bias = abs(stability_bias) / abs(stability_bias_ref),
        mac = mac / mac_ref,
        mqlossc = mqlossc / mqlossc_ref,
        rmsc = rmsc / rmsc_ref,
        smapc = smapc / smapc_ref
      ) |> 
      dplyr::select(-dplyr::ends_with("_ref"))

  } else if (type == 'cost') {

    relative_data <- data |> 
      dplyr::left_join(reference_data, by = c("method" = "method_ref")) |> 
      dplyr::mutate(
  			cost_perc = cost / cost_ref * 100,
        savings = cost_ref - cost,
  			savings_perc = (cost_ref - cost) / cost_ref * 100
      ) |> 
      dplyr::select(-dplyr::ends_with("_ref"))

  } else {
    stop(paste0('Unknown type ', type))
  }

  return(relative_data)
        
}

table_retrain_results <- function(data, metric, title = "", digits = 2, format = 'numeric') {

  cat("Creating table...\n")  
  tab <- data |> 
    dplyr::select(c('method', 'retrain_window', dplyr::all_of(metric))) |> 
    tidyr::pivot_wider(names_from = 'retrain_window', values_from = metric) |> 
    dplyr::rename_with(stringr::str_to_title) |> 
    dt_table(title = title, caption = '', digits = digits, format = format)
  return(tab)

}

plot_retrain_results <- function(data, metric, metric_label = "", title = "", smooth = FALSE) {

  cat("Creating plot...\n")
  data_plot <- data |> 
    dplyr::mutate(retrain_window = factor(retrain_window, ordered = TRUE))
  
  if (metric %in% c('cost', 'savings', 'savings_perc')) {
  	
  	data_mean <- data_plot |>
  		dplyr::filter(method == 'Average')
  	data_plot <- data_plot |> 
  		dplyr::filter(method != 'Average') |>
  		dplyr::mutate(
  			method = factor(
  				method, 
  				levels = c(
  					'LR', 'RF', 'XGBoost', 'LGBM', 'CatBoost', 
  					'MLP', 'LSTM', 'TCN', 'NBEATSx', 'NHITS'
  				), 
  				ordered = TRUE
  			)
  		)
  	g <- data_plot |> 
  		ggplot2::ggplot(
  			ggplot2::aes(
  				x = .data[['retrain_window']], 
  				y = .data[[metric]], 
  				color = .data[['method']],
  				linetype = .data[['method']],
  				group = .data[['method']]
  			)
  		) +
  		ggplot2::geom_point(size = 2) +
  		ggplot2::geom_line(linewidth = 1) +  
  		ggplot2::labs(
  			title = title, 
  			x = 'Retrain Scenario', y = metric_label,
  			color = 'Method', linetype = 'Method', group = 'Method'
  		) + 
  		ggplot2::theme_minimal() +
  		ggplot2::theme(plot.title = ggplot2::element_text(hjust = 0.5)) +
  		ggplot2::geom_point(
  			data = data_mean, 
  			mapping = ggplot2::aes(linetype = NULL),
  			col = 'darkred', size = 1
  		) +
  		ggplot2::geom_line(
  			data = data_mean, 
  			mapping = ggplot2::aes(linetype = NULL),
  			col = 'darkred', linewidth = 0.5, linetype = 1,
  		)
  	
  } else {
  	
  	if (smooth) {
  		g <- data_plot |> 
  			ggplot2::ggplot(
  				ggplot2::aes(
  					x = .data[['retrain_window']], 
  					y = .data[[metric]], 
  					color = .data[['method']],
            linetype = .data[['method']],
            group = .data[['method']]
  				)
  			) +
  			ggplot2::geom_smooth(
  				method = 'lm', formula = 'y ~ log(x)', linewidth = 1, se = FALSE
  			) +
  			ggplot2::labs(
  				title = title, 
  				x = 'Retrain Scenario', y = metric_label,
  				color = 'Method', linetype = 'Method'
  			) + 
  			ggplot2::theme_minimal() +
  			ggplot2::theme(plot.title = ggplot2::element_text(hjust = 0.5))
  	} else {
  		g <- data_plot |> 
  			ggplot2::ggplot(
  				ggplot2::aes(
  					x = .data[['retrain_window']], 
  					y = .data[[metric]], 
  					color = .data[['method']],
  					linetype = .data[['method']],
            group = .data[['method']]
  				)
  			) +
  			ggplot2::geom_point(size = 2) +
  			ggplot2::geom_line(linewidth = 1) + 
  			ggplot2::labs(
  				title = title, 
  				x = 'Retrain Scenario', y = metric_label,
  				color = 'Method', linetype = 'Method'
  			) + 
  			ggplot2::theme_minimal() +
  			ggplot2::theme(plot.title = ggplot2::element_text(hjust = 0.5))
  	}
  	
  }

  return(g)

}

test_differences <- function(data, .method, .metric) {

  cat(paste0("Testing differences in ", .metric, " for ", .method, "...\n"))
  
  data_test <- data |>
    dplyr::filter(method == .method)

  if (nrow(data_test) == 0) {

    return(NULL)

  } else {

    min_n_series <- min(table(data_test$retrain_window))
    n_scn <- length(unique(data_test$retrain_window))
    data_test <- data_test |> 
      dplyr::group_by(retrain_window) |> 
      dplyr::slice_sample(n = min_n_series) |> # obtain homogenous samples for each retrain window
      dplyr::ungroup() |> 
      dplyr::select(dplyr::all_of(c('retrain_window', .metric))) |> 
      dplyr::mutate(id = rep(1:min_n_series, n_scn), .before = 1) |> 
      tidyr::pivot_wider(names_from = 'retrain_window', values_from = .metric) |> 
      dplyr::select(-id)
  
    test_res <- greybox::rmcb(data = data_test, level = 0.95, outplot = "none")
    data_test <- tibble::tibble(
      'method' = .method,
      'metric' = .metric, 
      'retrain_window' = as.integer(names(test_res$mean)),
      'mean' = test_res$mean,
      'lower' = test_res$interval[, 1],
      'upper' = test_res$interval[, 2],
      'pvalue' = test_res$p.value
    )

    return(data_test)

  }

}

extract_significance <- function(p_value) {
	
  cat("Extracting significance...\n")
	res <- dplyr::case_when(
		p_value < 0.001 ~ "***", 
		p_value >= 0.001 & p_value < 0.01 ~ "**",
		p_value >= 0.01 & p_value < 0.05 ~ "*",
		p_value >= 0.05 & p_value < 0.1 ~ ".",
		TRUE ~ ""
	)
	return(res)
	
}

plot_test_results <- function(data, .method, .metric, metric_label = "", title = "") {

  cat("Creating plot...\n")
  data_plot <- data |> 
  	dplyr::filter(method == .method, metric == .metric) |> 
  	dplyr::mutate(retrain_window = factor(retrain_window, ordered = TRUE))
  data_min <- data_plot |> dplyr::slice_min(mean)
  retrain_scenarios <- sort(unique(data[["retrain_window"]]))

  g <- data_plot |> 
    ggplot2::ggplot(ggplot2::aes(x = retrain_window, y = mean)) +
    ggplot2::geom_errorbar(
      ggplot2::aes(ymin = lower, ymax = upper), 
      col = 'lightblue', width = 0.1, linewidth = 1
    ) +
    ggplot2::geom_point(size = 2, col = 'lightblue') +
    ggplot2::geom_point(data = data_min, size = 2, col = 'red') +
    ggplot2::geom_hline(yintercept = data_min$lower, col = 'gray', linetype = 2) +
    ggplot2::geom_hline(yintercept = data_min$upper, col = 'gray', linetype = 2) +
    ggplot2::labs(title = title, x = 'Retrain Scenario', y = metric_label) + 
    ggplot2::theme_minimal() +
    ggplot2::theme(plot.title = ggplot2::element_text(hjust = 0.5))

  return(g)

}

plot_test_results_facet <- function(data, .facet, .metric, metric_label = "", title = "") {
	
	cat("Creating plot...\n")
	method_lvls <- c('LR', 'RF', 'XGBoost', 'LGBM', 'CatBoost', 'MLP', 'LSTM', 'TCN', 'NBEATSx', 'NHITS')
	data_plot <- data |> 
		dplyr::filter(metric == .metric) |> 
		dplyr::mutate(retrain_window = factor(retrain_window, ordered = TRUE)) |> 
		dplyr::mutate(method = factor(method, levels = method_lvls, ordered = TRUE))
	data_min <- data_plot |> 
		dplyr::group_by(method) |> 
		dplyr::slice_min(mean) |> 
		dplyr::ungroup()
	retrain_scenarios <- sort(unique(data[["retrain_window"]]))
	
	g <- data_plot |> 
		ggplot2::ggplot(ggplot2::aes(x = retrain_window, y = mean)) +
		ggplot2::geom_errorbar(
			ggplot2::aes(ymin = lower, ymax = upper), 
			col = 'lightblue', width = 0.1, linewidth = 1
		) +
		ggplot2::geom_point(size = 1, col = 'lightblue') +
		ggplot2::geom_point(data = data_min, size = 1, col = 'red') +
		ggplot2::geom_hline(data = data_min, mapping = ggplot2::aes(yintercept = lower), col = 'gray', linetype = 2) +
		ggplot2::geom_hline(data = data_min, mapping = ggplot2::aes(yintercept = upper), col = 'gray', linetype = 2) +
		ggplot2::labs(title = title, x = 'Retrain Scenario', y = metric_label) + 
		ggplot2::facet_wrap(~ .data[[.facet]], ncol = 2, scales = 'free_y') +
		ggplot2::theme_bw() +
		ggplot2::theme(plot.title = ggplot2::element_text(hjust = 0.5))
	
	return(g)
	
}

plot_distribution_results <- function(data, metric, metric_label = "", title = "") {
	
	cat("Creating plot...\n")
	retrain_scenario <- sort(unique(data[["retrain_window"]]))
	
	g <- data |> 
		ggplot2::ggplot(
			ggplot2::aes(
				# x = .data[['retrain_window']], 
				x = .data[[metric]], 
				color = .data[['retrain_window']] |> factor(levels = retrain_scenario, ordered = TRUE)
			)
		) +
		ggplot2::geom_density() +
		ggplot2::facet_wrap(~ .data[['method']], nrow = 4, ncol = 2) + 
		# ggplot2::scale_x_continuous(breaks = retrain_scenario) +
		ggplot2::labs(
			title = title, 
			x = 'Retrain Scenario', y = metric_label,
			color = 'Retrain Scenarios'
		) + 
		# ggplot2::theme_minimal() +
		ggplot2::theme(plot.title = ggplot2::element_text(hjust = 0.5), legend.position = "bottom")
	
	return(g)
	
}

clean_outliers <- function(data, .metric, q = c(0.001, 0.999)) {
	
	cat("Removing outliers...\n")
	
	q_funs <- list(
		'qlow' = function(x) { round(quantile(x, q[1]), 3) },
		'qhigh' = function(x) { round(quantile(x, q[2]), 3) }
	)
	qs_df <- data |> 
		dplyr::summarise(dplyr::across(.metric, .fns = q_funs)) |> 
		tidyr::pivot_longer(cols = dplyr::everything()) |>
		tidyr::separate(name, into = c('metric', 'q'), sep = "_") |>
		tidyr::pivot_wider(names_from = q, values_from = value)
	
	data_cln <- data
	for (i in 1:nrow(qs_df)) {
		m <- qs_df[['metric']][i]
		q_low <- qs_df[['qlow']][i]
		q_high <- qs_df[['qhigh']][i]
		cat(paste0("Metric: ", m, ", Q Low: ", q_low, ", Q High: ", q_high, "\n"))
		data_cln <- data_cln |> 
			dplyr::filter(dplyr::between(.data[[m]], q_low, q_high))
	}
	
	n_full <- nrow(data)
	n_filtered <- nrow(data_cln)
	cat(
		paste0(
			"Removed ", n_full - n_filtered, " (", 
			round((n_full - n_filtered) / n_full * 100, 1), 
			"%) of results\n"
		)
	)
	
	return(data_cln)
	
}

compute_costs <- function(
  data, 
  time_var, 
  n_skus, 
  dataset_n_skus, 
  cost_per_hour = 3.5, 
  add_average = FALSE
) {
	
  if (add_average) {
    data_mean <- data |> 
      aggregate_data(
        group_columns = c('retrain_window'),
        drop_columns = c('type', 'method', 'sample', 'test_window', 'horizon'),
        function_name = 'mean',
        adjust_metrics = FALSE
      ) |> 
      dplyr::mutate(method = 'Average', .before = 1) |> 
      dplyr::mutate(type = NA_character_, .before = 1)
    data_cost <- data |> dplyr::bind_rows(data_mean)
  } else {
    data_cost <- data
  }

  data_cost <- data_cost |>  
		dplyr::mutate(
			ct_per_sku = .data[[time_var]] / dataset_n_skus,
			ct_hour = .data[[time_var]] / 60 / 60,
			ct_hour_per_sku = ct_hour / dataset_n_skus,
			ct_hour_tot = ct_hour_per_sku * n_skus,
			cost = ct_hour_tot * cost_per_hour
		) |> 
		dplyr::select(dplyr::all_of(c('type', 'method', 'retrain_window', 'cost'))) |> 
    compute_relative_metrics(type = 'cost')
	
	return (data_cost)
	
}

flatten_list <- function(list) {
  return(purrr::map_chr(list, ~ paste0(.x, collapse = "_")))
}

create_results_list <- function(dataset_names, analysis_types, data_types, model_types, final_analyses) {

  dataset_names <- flatten_list(dataset_names)
  analysis_types <- flatten_list(analysis_types)
  data_types <- flatten_list(data_types)
  model_types <- flatten_list(model_types)
  final_analyses <- flatten_list(final_analyses)

  # level 1 list of dataset names 
  res_list <- vector("list", length(dataset_names)) |> 
    purrr::set_names(dataset_names)

  for (i in seq_along(res_list)) {
    res_list[[i]] <- vector("list", length(analysis_types)) |> 
      purrr::set_names(analysis_types)
  }
  for (i in seq_along(res_list)) {
    for (j in seq_along(res_list[[i]])) {
      res_list[[i]][[j]] <- vector("list", length(data_types)) |> 
        purrr::set_names(data_types)
    }
  }
  for (i in seq_along(res_list)) {
    for (j in seq_along(res_list[[i]])) {
      res_list[[i]][[j]][[2]] <- vector("list", length(model_types)) |> 
        purrr::set_names(model_types)
    }
  }
  for (i in seq_along(res_list)) {
    for (j in seq_along(res_list[[i]])) {
      for (k in seq_along(res_list[[i]][[j]])) {
        res_list[[i]][[j]][[2]][[k]] <- vector("list", length(final_analyses)) |> 
          purrr::set_names(final_analyses)
      }
    }
  }

  gc()
  return(res_list)

}

get_table_plot_params <- function(analysis_type, analysis_metric) {

  if (analysis_type == 'cost') {
    digits <- 0
    if (analysis_metric == 'savings_perc') {
      format <- 'percent'
    } else {
      format <- 'dollar'
    }
  } else {
    digits <- 3
    format <- 'numeric'
  }

  if (analysis_metric == 'total_sample_time') {
    label <- 'Computing Time'
  } else if (analysis_metric == 'cost') {
    label <- 'Cost ($)'
  } else if (analysis_metric == 'savings') {
    label <- 'Savings ($)'
  } else if (analysis_metric == 'savings_perc') {
    label <- 'Savings (%)'
  } else {
    label <- toupper(analysis_metric) 
  }

  res <- list(
    'digits' = digits,
    'format' = format,
    'label' = label
  )
  return(res)

}

analyse_results <- function(config) {

  # analysis config
  analysis_types <- config$analysis$types
  analysis_method <- config$analysis$method
  analysis_sample_type <- config$analysis$sample_type
  # dataset config
  dataset_names <- config$dataset$dataset_names
  frequencies <- config$dataset$frequencies
  dataset_names_full <- paste(dataset_names, frequencies, sep = "_")
  retrain_scenarios <- purrr::map(frequencies, get_retrain_scenarios) |> 
    purrr::set_names(dataset_names_full)
  ext <- config$dataset$ext
  # model config
  model_types <- config$models$types
  model_names <- config$models$model_names
  model_names_abbr <- config$models$model_names_abbr
  model_type_levels <- c('SF', 'ML', 'DL', 'ENS')
  # evaluation, time, stability and cost config
  eval_metrics <- config$evaluation_params$metrics
  eval_outlier_cleaning_metrics <- config$evaluation_params$outlier_cleaning_metrics
  time_metrics <- config$time_params$metrics
  stab_metrics <- config$stability_params$metrics
  stab_outlier_cleaning_metrics <- config$stability_params$outlier_cleaning_metrics
  cost_metrics <- config$cost_params$metrics
  cost_time_var <- config$cost_params$time_var
  cost_n_skus <- config$cost_params$n_skus
  cost_per_hour <- config$cost_params$cost_per_hour
  cost_datasets_n_skus <- as.list(unlist(config$cost_params$cost_datasets_n_skus))
  
  # final analysis names
  data_types <- c('data', 'results')
  final_analyses <- c('tables', 'plots', 'tests')  
  analysis_results <- create_results_list(
    dataset_names_full, analysis_types, data_types, model_types, final_analyses
  )
  
  for (dn in dataset_names_full) {

    cat(paste0("*************** Analysing ", dn, " ***************\n"))
    dataset_name_tmp <- unlist(strsplit(dn, "_"))[1]
    freq_tmp <- unlist(strsplit(dn, "_"))[2]
    retrain_scn_tmp <- retrain_scenarios[[dn]]

    for (at in analysis_types) {

      cat(paste0("--- [ Analysis: ", at, " ] ---\n"))

      if (at == "evaluation") {

        cat("Loading and preparing the evaluation data...\n")
        anal_df_tmp = load_data(
          path_list = c('results', dataset_name_tmp, freq_tmp, 'evaluation'),
          name_list = c(dataset_name_tmp, freq_tmp, 'eval', analysis_sample_type),
          ext = ext
        ) |> 
          tibble::as_tibble() |> 
          dplyr::filter(retrain_window %in% retrain_scn_tmp) |> 
          dplyr::filter(method %in% model_names) |> 
          recode_data(model_type_levels, model_names_abbr) |> 
          clean_outliers(.metric = eval_outlier_cleaning_metrics, q = c(0.001, 0.999))
        anal_df_agg_tmp <- anal_df_tmp |> 
          aggregate_data(
            group_columns = c('type', 'method', 'retrain_window'),
            drop_columns = c('unique_id', 'test_window', 'horizon'),
            function_name = 'mean',
            adjust_metrics = TRUE
          )
        # set the metrics to be used
        anal_metrics <- eval_metrics

      } else if (at == "time") {

        cat("Loading and preparing the time data...\n")
        anal_df_tmp = load_data(
          path_list = c('results', dataset_name_tmp, freq_tmp, 'evaluation'),
          name_list = c(dataset_name_tmp, freq_tmp, 'time'),
          ext = ext
        ) |> 
          tibble::as_tibble() |> 
          dplyr::filter(retrain_window %in% retrain_scn_tmp) |> 
          dplyr::filter(method %in% model_names) |> 
          recode_data(model_type_levels, model_names_abbr)
        anal_df_agg_tmp <- anal_df_tmp |> 
          aggregate_data(
            group_columns = c('type', 'method', 'retrain_window'),
            drop_columns = c('sample', 'test_window', 'horizon'),
            function_name = 'sum',
            adjust_metrics = TRUE
          )
        # set the metrics to be used
        anal_metrics <- time_metrics

      } else if (at == "stability") {

        cat("Loading and preparing the stability data...\n")
        anal_df_tmp = load_data(
          path_list = c('results', dataset_name_tmp, freq_tmp, 'stability'),
          name_list = c(dataset_name_tmp, freq_tmp, 'stab'),
          ext = ext
        ) |> 
          tibble::as_tibble() |> 
          dplyr::filter(retrain_window %in% retrain_scn_tmp) |> 
          dplyr::filter(method %in% model_names) |> 
          recode_data(model_type_levels, model_names_abbr) |> 
          clean_outliers(.metric = stab_outlier_cleaning_metrics, q = c(0.001, 0.999))
        anal_df_agg_tmp <- anal_df_tmp |> 
          aggregate_data(
            group_columns = c('type', 'method', 'retrain_window'),
            drop_columns = c('unique_id', 'test_window', 'horizon'),
            function_name = 'mean',
            adjust_metrics = TRUE
          )
        # set the metrics to be used
        anal_metrics <- stab_metrics

      } else if (at == "cost") {

        cat("Loading and preparing the time data for cost analysis...\n")
        cost_dataset_n_skus_tmp <- cost_datasets_n_skus[[dn]]

        anal_df_tmp = load_data(
          path_list = c('results', dataset_name_tmp, freq_tmp, 'evaluation'),
          name_list = c(dataset_name_tmp, freq_tmp, 'time'),
          ext = ext
        ) |> 
          tibble::as_tibble() |> 
          dplyr::filter(retrain_window %in% retrain_scn_tmp) |> 
          dplyr::filter(method %in% model_names) |> 
          recode_data(model_type_levels, model_names_abbr)
        anal_df_agg_tmp <- anal_df_tmp |> 
          aggregate_data(
            group_columns = c('type', 'method', 'retrain_window'),
            drop_columns = c('sample', 'test_window', 'horizon'),
            function_name = 'sum',
            adjust_metrics = TRUE
          ) |> 
          compute_costs(
            time_var = cost_time_var, 
            n_skus = cost_n_skus,
            dataset_n_skus = cost_dataset_n_skus_tmp,
            cost_per_hour = cost_per_hour, 
            add_average = FALSE
          )
        # set the metrics to be used
        anal_metrics <- cost_metrics

      } else {
        stop(paste0("Unknown analysis type: ", at))
      }

      # absolute or relative analysis
      if (analysis_method == 'relative' & at != 'cost') {
        cat("Compute relative metrics...\n")
        anal_df_agg_tmp <- compute_relative_metrics(anal_df_agg_tmp, type = at)
      }

      # store data of the analysis
      analysis_results[[dn]][[at]][['data']] <- anal_df_agg_tmp

      for (mt in model_types) {

        cat(paste0("--- [ Model Types: ", paste0(mt, collapse = ", "), " ] ---\n"))
        # filter datasets
        anal_df_mt_tmp <- anal_df_tmp |> dplyr::filter(type %in% mt)
        anal_df_agg_mt_tmp <- anal_df_agg_tmp |> dplyr::filter(type %in% mt)
        model_names_abbr_mt_tmp <- unique(as.character(anal_df_mt_tmp$method))

        # analysis tables, plots and tests
        anal_tab <- anal_plot <- anal_test <- vector("list", length(anal_metrics)) |> 
          purrr::set_names(anal_metrics)

        for (am in anal_metrics) {

          cat(paste0("Creating evaluation table, plot and tests for ", toupper(am), "...\n"))
          tp_par <- get_table_plot_params(at, am)

          anal_tab[[am]] <- table_retrain_results(
            anal_df_agg_mt_tmp, 
            metric = am,
            title = toupper(paste(dataset_name_tmp, '-', tp_par$label)),
            digits = tp_par$digits,
            format = tp_par$format
          )

          anal_plot[[am]] <- plot_retrain_results(
            anal_df_agg_mt_tmp, 
            metric = am, 
            metric_label = tp_par$label,
            title = toupper(dataset_name_tmp)
          )

          if (at != 'cost') {
            anal_test[[am]] <- model_names_abbr_mt_tmp |> 
              purrr::map(~ test_differences(anal_df_mt_tmp, .method = .x, .metric = am)) |> 
              dplyr::bind_rows()
          } else {
            anal_test[[am]] <- NULL
          }

        }

        # store results
        mt_name <- paste0(mt, collapse = "_")
        analysis_results[[dn]][[at]][['results']][[mt_name]][['tables']] <- anal_tab 
        analysis_results[[dn]][[at]][['results']][[mt_name]][['plots']] <- anal_plot 
        analysis_results[[dn]][[at]][['results']][[mt_name]][['tests']] <- anal_test 

      }
      
    }

  }

  cat("Saving results...\n")
  file_name <- paste0(
    analysis_method,
    "_",
    stringr::str_sub_all(analysis_types, start = 1, end = 4) |> 
      unlist() |> 
      paste0(collapse = ""),  
    "_",
    analysis_sample_type, 
    "_", 
    Sys.time() |> 
      as.character() |> 
      stringr::str_remove_all("\\..*") |> 
      stringr::str_replace_all("(-)|(:)", "") |> 
      stringr::str_replace_all(" ", "_"),
    ".RData"
  )
  save(analysis_results, file = paste0('results/analysis/', file_name))
  cat("Done!\n")

  return(invisible(NULL))

}
