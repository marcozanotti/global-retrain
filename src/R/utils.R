
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
    model_name == 'EnsembleMeanTOP3ACC' ~ 'ENSacc',
    model_name == 'EnsembleMeanTOP3TIME' ~ 'ENStime',
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

compute_relative_metrics <- function(data) {

  reference_data <- data |> 
    dplyr::group_by(method) |> 
    dplyr::slice_min(retrain_window) |> 
    dplyr::ungroup() |> 
    dplyr::select(-dplyr::any_of(c('type', 'retrain_window'))) |> 
    dplyr::rename_with(~ stringr::str_c(.x, "_ref"))

  if (any(stringr::str_detect(names(data), "cost"))) {
  	
  	relative_data <- data |> 
  		dplyr::left_join(reference_data, by = c("method" = "method_ref")) |> 
  		dplyr::mutate(
  			cost_perc = cost / cost_ref * 100,
  			savings = cost_ref - cost,
  			savings_perc = (cost_ref - cost) / cost_ref * 100
  		) |> 
  		dplyr::select(-dplyr::ends_with("_ref"))
  	
  } else if (any(stringr::str_detect(names(data), "_time"))) {

    relative_data <- data |> 
      dplyr::left_join(reference_data, by = c("method" = "method_ref")) |> 
      dplyr::mutate(
        total_fit_time = total_fit_time / total_fit_time_ref,
        total_predict_time = total_predict_time / total_predict_time_ref,
        total_sample_time = total_sample_time / total_sample_time_ref
      ) |> 
      dplyr::select(-dplyr::ends_with("_ref"))

  } else {

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

analyse_results <- function(config) {

  # dataset config
  dataset_names <- config$dataset$dataset_names
  frequencies <- config$dataset$frequencies
  retrain_scenarios <- purrr::map(frequencies, get_retrain_scenarios)
  ext <- config$dataset$ext
  # model config
  model_names <- config$models$model_names
  model_names_abbr <- config$models$model_names_abbr
  model_type_levels <- c('SF', 'ML', 'DL', 'ENS')
  # evaluation config
  eval_metrics <- config$evaluation$metrics
  time_metric <- config$evaluation$time_metric
  eval_type <- config$evaluation$evaluation_sample_type
  analysis_type <- config$evaluation$analysis_type
  outlier_cleaning_metrics <- config$evaluation$outlier_cleaning_metrics

  analysis_results <- vector("list", length(dataset_names))
  names(analysis_results) <- paste(dataset_names, frequencies, sep = "_")

  for (i in seq_along(dataset_names)) {

    dataset_name_tmp <- dataset_names[i]
    freq_tmp <- frequencies[i]
    retrain_scn_tmp <- retrain_scenarios[[i]]
    cat(paste0("Analysing ", dataset_name_tmp, " ", freq_tmp, "...\n"))

    # load, aggregate and prepare data
    # eval_df.shape[0] = n_series * n_retrain_scenarios * n_models = 30.000 * 10 * 10
    # time_df.shape[0] = n_samples * n_retrain_scenarios * n_models = 365 * 10 * 10
    cat("Loading and preparing the evaluation data...\n")
    eval_df_tmp = load_data(
      path_list = c('results', dataset_name_tmp, freq_tmp, 'evaluation'),
      name_list = c(dataset_name_tmp, freq_tmp, 'eval', eval_type),
      ext = ext
    ) |> 
      tibble::as_tibble() |> 
      dplyr::filter(retrain_window %in% retrain_scn_tmp) |> 
      dplyr::filter(method %in% model_names) |> 
      recode_data(model_type_levels, model_names_abbr) |> 
    	clean_outliers(.metric = outlier_cleaning_metrics, q = c(0.001, 0.999))
    eval_df_agg_tmp <- eval_df_tmp |> 
      aggregate_data(
        group_columns = c('type', 'method', 'retrain_window'),
        drop_columns = c('unique_id', 'test_window', 'horizon'),
        function_name = 'mean',
        adjust_metrics = TRUE
      )

    cat("Loading and preparing the time data...\n")
    time_df_tmp = load_data(
      path_list = c('results', dataset_name_tmp, freq_tmp, 'evaluation'),
      name_list = c(dataset_name_tmp, freq_tmp, 'time'),
      ext = ext
    ) |> 
      tibble::as_tibble() |> 
      dplyr::filter(retrain_window %in% retrain_scn_tmp) |> 
      dplyr::filter(method %in% model_names) |> 
      recode_data(model_type_levels, model_names_abbr)
    time_df_agg_tmp <- time_df_tmp |> 
      aggregate_data(
        group_columns = c('method', 'retrain_window'),
        drop_columns = c('sample', 'test_window', 'horizon'),
        function_name = 'sum',
        adjust_metrics = TRUE
      )

    if (analysis_type == 'relative') {
      cat("Compute relative metrics...\n")
      eval_df_agg_tmp <- compute_relative_metrics(eval_df_agg_tmp)
      time_df_agg_tmp <- compute_relative_metrics(time_df_agg_tmp)
    }

    # time table and plot
    tab_time <- table_retrain_results(
      time_df_agg_tmp, 
      metric = time_metric,
      title = toupper(paste(dataset_name_tmp, freq_tmp, '-', gsub("_", " ", time_metric))),
      digits = 3
    )
    g_time <- plot_retrain_results(
      time_df_agg_tmp, 
      metric = time_metric, metric_label = 'Computing Time',
      title = toupper(paste(dataset_name_tmp, freq_tmp))
    )

    # evaluation tables and plots
    tab_eval <- vector("list", length(eval_metrics))
    names(tab_eval) <- eval_metrics
    g_eval <- vector("list", length(eval_metrics))
    names(g_eval) <- eval_metrics
    g_eval_comb <- vector("list", length(eval_metrics))
    names(g_eval_comb) <- eval_metrics

    for (m in eval_metrics) {
      cat(paste0("Creating evaluation table, plot and tests for ", toupper(m), "...\n"))
      tab_eval[[m]] <- table_retrain_results(
        eval_df_agg_tmp, 
        metric = m,
        title = toupper(paste(dataset_name_tmp, freq_tmp, '-', gsub("_", " ", m))),
        digits = 3
      )
      g_eval[[m]] <- plot_retrain_results(
        eval_df_agg_tmp, 
        metric = m, metric_label = toupper(gsub("_", " ", m)),
        title = toupper(paste(dataset_name_tmp, freq_tmp))
      )
      cat("Combining evaluation and time plot...\n")
      g_eval_comb[[m]] <- g_eval[[m]] + g_time + 
        patchwork::plot_layout(guides = "collect") & ggplot2::theme(legend.position = "bottom")
    }

    # testing differences
    test_res <- model_names_abbr |> 
      purrr::map(
        ~ purrr::map2(
          .x, eval_metrics, 
          ~ test_differences(eval_df_tmp, .method = .x, .metric = .y)
        )
      ) |> 
      dplyr::bind_rows()

    analysis_results[[i]] <- list(
      "eval_df_agg" = eval_df_agg_tmp,
      "time_df_agg" = time_df_agg_tmp,
      "tab_time" = tab_time,
      "g_time" = g_time,
      "tab_eval" = tab_eval,
      "g_eval" = g_eval,
      "g_eval_comb" = g_eval_comb,
      "test_res" = test_res
    )
    
  }

  cat("Saving results...\n")
  file_name <- paste0(
    analysis_type, "_", eval_type, "_results_", 
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

compute_costs <- function(data, dataset_name, time_var, n_skus, cost_per_hour = 3.5) {
	
	if (dataset_name == 'm5_daily') {
		dataset_n_skus <- 28298
	} else {
		dataset_n_skus <- 15053
	}
	
	data_cost <- data |> 
		dplyr::mutate(
			ct_per_sku = .data[[time_var]] / dataset_n_skus,
			ct_hour = .data[[time_var]] / 60 / 60,
			ct_hour_per_sku = ct_hour / dataset_n_skus,
			ct_hour_tot = ct_hour_per_sku * n_skus,
			cost = ct_hour_tot * cost_per_hour
		) |> 
		dplyr::select(dplyr::all_of(c('method', 'retrain_window', 'cost'))) |> 
		compute_relative_metrics()
	
	return (data_cost)
	
}

cost_analysis <- function(data, dataset_name, time_var, n_skus, cost_per_hour) {
	
	data_mean <- data |> 
		dplyr::group_by(retrain_window) |> 
		dplyr::summarise(
			total_fit_time = mean(total_fit_time), 
			total_predict_time = mean(total_predict_time), 
			total_sample_time = mean(total_sample_time),
			method = 'Average'
		) |> 
		dplyr::relocate('method', .before = 1)
	
	data_cost <- data |> 
		dplyr::bind_rows(data_mean) |>
		compute_costs(
			dataset_name = dataset_name, 
			time_var = 'total_sample_time', 
			n_skus = n_skus,
			cost_per_hour = cost_per_hour
		)
	
	# costs table
	tab_cost <- data_cost |> 
		table_retrain_results(metric = 'cost', title = '', digits = 0, format = 'dollar')
	# savings table
	tab_sav <- data_cost |> 
		table_retrain_results(metric = 'savings', title = '', digits = 0, format = 'dollar')
	# savings table perc
	tab_savperc <- data_cost |> 
		table_retrain_results(metric = 'savings_perc', title = '', digits = 0, format = 'percent')
	
	# plots
	g_cost <- plot_retrain_results(
		data = data_cost, 
		metric = 'cost', 
		metric_label = 'Cost ($)',
		title = toupper(stringr::str_replace_all(toupper(dataset_name), "_.*", ""))
	)
	g_sav <- plot_retrain_results(
		data = data_cost, 
		metric = 'savings', 
		metric_label = 'Savings ($)',
		title = toupper(stringr::str_replace_all(toupper(dataset_name), "_.*", ""))
	)
	g_savperc <- plot_retrain_results(
		data = data_cost, 
		metric = 'savings_perc', 
		metric_label = 'Savings (%)',
		title = toupper(stringr::str_replace_all(toupper(dataset_name), "_.*", ""))
	)
	
	cat("Combining evaluation and time plot...\n")
	g_cost_comb <- g_cost + g_savperc + 
		patchwork::plot_layout(guides = "collect") & ggplot2::theme(legend.position = "bottom")
	
	res_list <- list(
		'tab_cost' = tab_cost,
		'tab_sav' = tab_sav,
		'tab_savperc' = tab_savperc,
		'g_cost' = g_cost,
		'g_sav' = g_sav,
		'g_savperc' = g_savperc,
		'g_cost_comb' = g_cost_comb
	)
	return(res_list)
	
}
