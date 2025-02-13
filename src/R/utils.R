
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

dt_table <- function(data, title = "", caption = "", rownames = FALSE, digits = 3) {
	
	p_len <- nrow(data)
	res <- data |>
    dplyr::mutate(dplyr::across(where(is.numeric), ~ round(.x, digits = digits))) |> 
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

  if (any(stringr::str_detect(names(data), "_time"))) {

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

table_retrain_results <- function(data, metric, title = "") {

  cat("Creating table...\n")  
  tab <- data |> 
    dplyr::select(c('method', 'retrain_window', dplyr::all_of(metric))) |> 
    tidyr::pivot_wider(names_from = 'retrain_window', values_from = metric) |> 
    dplyr::rename_with(stringr::str_to_title) |> 
    dt_table(title = title, caption = '', digits = 3)
  return(tab)

}

plot_retrain_results <- function(data, metric, metric_label = "", title = "", smooth = FALSE) {

  cat("Creating plot...\n")
  retrain_scenario <- sort(unique(data[["retrain_window"]]))

  if (smooth) {
    g <- data |> 
      ggplot2::ggplot(
        ggplot2::aes(
          x = .data[['retrain_window']], 
          y = .data[[metric]], 
          color = .data[['method']]
        )
      ) +
      ggplot2::geom_smooth(
        method = 'lm', formula = 'y ~ log(x)', linewidth = 2, se = FALSE
      ) +
      ggplot2::scale_x_continuous(breaks = retrain_scenario) +
      ggplot2::labs(
        title = title, 
        x = 'Retrain Scenario', y = metric_label,
        color = 'Method'
      ) + 
      ggplot2::theme_minimal() +
      ggplot2::theme(plot.title = ggplot2::element_text(hjust = 0.5))
  } else {
    g <- data |> 
      ggplot2::ggplot(
        ggplot2::aes(
          x = .data[['retrain_window']], 
          y = .data[[metric]], 
          color = .data[['method']]
        )
      ) +
      ggplot2::geom_point(size = 10, shape = 18) +
      ggplot2::geom_line(linewidth = 2) + 
      ggplot2::scale_x_continuous(breaks = retrain_scenario) +
      ggplot2::labs(
        title = title, 
        x = 'Retrain Scenario', y = metric_label,
        color = 'Method'
      ) + 
      ggplot2::theme_minimal() +
      ggplot2::theme(plot.title = ggplot2::element_text(hjust = 0.5))
  }

  return(g)

}

test_differences <- function(data, .method, .metric) {

  cat(paste0("Testing differences in ", .metric, " for ", .method, "...\n"))
  n_series = length(unique(data$unique_id))
  n_scn <- length(unique(data$retrain_window))
  
  data_test <- data |>
    dplyr::filter(method == .method) |>
    dplyr::select(dplyr::all_of(c('retrain_window', .metric))) |> 
    dplyr::mutate(id = rep(1:n_series, n_scn), .before = 1) |> 
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
  data_plot <- data |> dplyr::filter(method == .method, metric == .metric)
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
    ggplot2::scale_x_continuous(breaks = retrain_scenarios) +
    ggplot2::labs(title = title, x = 'Retrain Scenario', y = metric_label) + 
    ggplot2::theme_minimal() +
    ggplot2::theme(plot.title = ggplot2::element_text(hjust = 0.5))

  return(g)

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
      recode_data(model_type_levels, model_names_abbr) #|> 
      # dplyr::mutate(rmse = rm_mse, rmsse = rm_msse) |> 
      # dplyr::select(-dplyr::any_of(c("rm_mse", "rm_msse")))
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
      title = toupper(paste(dataset_name_tmp, freq_tmp, '-', gsub("_", " ", time_metric)))
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
        title = toupper(paste(dataset_name_tmp, freq_tmp, '-', gsub("_", " ", m)))
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

  return(invisible(analysis_results))

}
