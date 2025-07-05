# Ensemble analysis

source('src/R/utils.R')
analysis_file_name <- 'results/analysis/absolute_evaltimestabcost_overlap_20250508_114025.RData'
res <- load(analysis_file_name)
res <- analysis_results
rm(analysis_results)

dataset_names <- c('m5_daily', 'vn1_weekly')
top_n = 5

# top by accuracy
res_top_acc <- vector('list', length(dataset_names))
for (i in seq_along(dataset_names)) {
	
	res_top_acc[[i]] <- res[[dataset_names[i]]][['evaluation']]$data |>
		dplyr::filter(type %in% c('ML', 'DL')) |> 
		dplyr::slice_min(retrain_window) |> 
		dplyr::arrange(rmsse) |> 
		dplyr::slice_head(n = top_n) |> 
		dplyr::select(method)
	
}
res_top_acc <- res_top_acc |> 
	dplyr::bind_cols() |> 
	purrr::set_names(c('M5 DAILY', 'VN1 WEEKLY'))

# top by computation time
res_top_time <- vector('list', length(dataset_names))
for (i in seq_along(dataset_names)) {
	
	res_top_time[[i]] <- res[[dataset_names[i]]][['time']]$data |> 
		dplyr::filter(type %in% c('ML', 'DL')) |> 
		dplyr::slice_min(retrain_window) |> 
		dplyr::arrange(total_sample_time) |> 
		dplyr::slice_head(n = top_n) |> 
		dplyr::select(method)
	
}
res_top_time <- res_top_time |> 
	dplyr::bind_cols() |> 
	purrr::set_names(c('M5 DAILY', 'VN1 WEEKLY'))

res_top_acc
res_top_time
