
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
      data_agg[['rm_mse']] = sqrt(data_agg[['mse']])

    if ('msse' %in% names(data_agg))
      data_agg[['rm_msse']] = sqrt(data_agg[['msse']])
    
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
