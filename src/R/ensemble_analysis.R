# Ensemble analysis

source('src/R/utils.R')
analysis_file_name <- 'results/analysis/absolute_overlap_results_20250212_144529.RData'
res <- load(analysis_file_name)
res <- analysis_results
rm(analysis_results)

dataset_names <- c(
	'm5_monthly', 
	'm5_weekly', 
	# 'm5_daily', 
	'vn1_monthly', 
	'vn1_weekly'
)

# top 3 by accuracy
res_top_acc <- vector('list', length(dataset_names))
for (i in seq_along(dataset_names)) {
	
	res_top_acc[[i]] <- res[[dataset_names[i]]][['eval_df_agg']] |> 
		dplyr::slice_min(retrain_window) |> 
		dplyr::arrange(rmsse) |> 
		dplyr::slice_head(n = 3) |> 
		dplyr::select(method)
	
}
res_top_acc <- res_top_acc |> 
	dplyr::bind_cols() |> 
	purrr::set_names(c('M5 MONTHLY', 'M5 WEEKLY', 'VN1 MONTHLY', 'VN1 WEEKLY'))

# top 3 by computation time
res_top_time <- vector('list', length(dataset_names))
for (i in seq_along(dataset_names)) {
	
	res_top_time[[i]] <- res[[dataset_names[i]]][['time_df_agg']] |> 
		dplyr::slice_min(retrain_window) |> 
		dplyr::arrange(total_sample_time) |> 
		dplyr::slice_head(n = 3) |> 
		dplyr::select(method)
	
}
res_top_time <- res_top_time |> 
	dplyr::bind_cols() |> 
	purrr::set_names(c('M5 MONTHLY', 'M5 WEEKLY', 'VN1 MONTHLY', 'VN1 WEEKLY'))

res_top_acc
res_top_time