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
# plot andamenti 1100x1100
# plot test 750x650



# Load & prepare data -----------------------------------------------------

# run twice, one for absolute and one for relative
analysis_file_name <- 'docs/iifsas_retrain/absolute_evaltimestabcost_overlap_20260128_165006.RData'
analysis_file_name <- 'docs/iifsas_retrain/relative_evaltimestabcost_overlap_20260128_170356.RData'

res <- load(analysis_file_name)
res <- analysis_results
rm(analysis_results)



# Analysis ----------------------------------------------------------------

# * Parameters ------------------------------------------------------------

df_nms <- c('m4_daily', 'm5_daily', 'vn1_weekly')
mod_tps <- c('SF', 'ML_DL')
eval_metrics <- c('rmsse', 'scaled_mqloss')
stab_metrics <- c('smapc', 'smqc')



# =========================================================================
# * Evaluation & Stability ------------------------------------------------
# =========================================================================

# ** Tables ---------------------------------------------------------------
for (i in seq_along(eval_metrics)) {
	em <- eval_metrics[i]
  	sm <- stab_metrics[i]
    for (j in seq_along(df_nms)) {
      	df_nm <- df_nms[j]
      	tab_eval <- dplyr::bind_rows(
			res[[df_nm]][['evaluation']][['results']][[mod_tps[1]]]$tables[[em]]$x$data,
			res[[df_nm]][['evaluation']][['results']][[mod_tps[2]]]$tables[[em]]$x$data
		)
        tab_stab <- dplyr::bind_rows(
			res[[df_nm]][['stability']][['results']][[mod_tps[1]]]$tables[[sm]]$x$data,
			res[[df_nm]][['stability']][['results']][[mod_tps[2]]]$tables[[sm]]$x$data
		)
      	cat(paste(df_nms[j], em, "\n\n"))
		print(xtable::xtable(tab_eval, digits = 3), include.rownames = FALSE)
		cat("\n\n")
        cat(paste(df_nms[j], sm, "\n\n"))
		print(xtable::xtable(tab_stab, digits = 3), include.rownames = FALSE)
		cat("\n\n")
	}
}

# ** Plots -----------------------------------------------------------------
for (k in mod_tps) {
	eval_res1 <- res[[df_nms[1]]][['evaluation']][['results']][[k]]
	eval_res2 <- res[[df_nms[2]]][['evaluation']][['results']][[k]]
	eval_res3 <- res[[df_nms[3]]][['evaluation']][['results']][[k]]
	stab_res1 <- res[[df_nms[1]]][['stability']][['results']][[k]]
	stab_res2 <- res[[df_nms[2]]][['stability']][['results']][[k]]
	stab_res3 <- res[[df_nms[3]]][['stability']][['results']][[k]]
	for (i in seq_along(eval_metrics)) {
		em <- eval_metrics[i]
		sm <- stab_metrics[i]
		print(
			(
				(eval_res1$plots[[em]] + ggplot2::guides(col = FALSE)) +
				(stab_res1$plots[[sm]] + ggplot2::guides(col = FALSE)) 
			) /
			(
				(eval_res2$plots[[em]] + ggplot2::guides(col = FALSE)) +
				(stab_res2$plots[[sm]] + ggplot2::guides(col = FALSE))
			) /
			(
				(eval_res3$plots[[em]]) +
				(stab_res3$plots[[sm]])
			) +
			patchwork::plot_layout(guides = "collect") & ggplot2::theme(legend.position = "bottom")
		)
	}
}

# ** Tests -----------------------------------------------------------------

# for SF models
for (k in mod_tps[1]) {
	for (nm in df_nms) {
      	eval_res <- res[[nm]][['evaluation']][['results']][[k]]
      	stab_res <- res[[nm]][['stability']][['results']][[k]]
      	df_nm <- stringr::str_replace_all(toupper(nm), "_.*", "")
        g_list <- vector("list", length(eval_metrics))
      	for (i in seq_along(eval_metrics)) {
			em <- eval_metrics[i]
			em_nm <- toupper(gsub("_", " ", ifelse(em == "scaled_mqloss", "smql", em))) 
			sm <- stab_metrics[i]
			sm_nm <- toupper(gsub("_", " ", sm))
			ge <- plot_test_results_facet(
				data = eval_res$tests[[em]], .metric = em, by = "retrain_window", 
				metric_label = em_nm, title = paste(df_nm, "-", em_nm, "- Nemenyi Test")
			)
			gs <- plot_test_results_facet(
				data = stab_res$tests[[sm]], .metric = sm, by = "retrain_window", 
				metric_label = sm_nm, title = paste(df_nm, "-", sm_nm, "- Nemenyi Test")
			)
			g_list[[i]] <- ge + gs
		}
		g <- g_list[[1]] / g_list[[2]]
        print(g)
	}
}

# for ML_DL models
for (k in mod_tps[2]) {
  	eval_res1 <- res[[df_nms[1]]][['evaluation']][['results']][[k]]
	eval_res2 <- res[[df_nms[2]]][['evaluation']][['results']][[k]]
	eval_res3 <- res[[df_nms[3]]][['evaluation']][['results']][[k]]
	stab_res1 <- res[[df_nms[1]]][['stability']][['results']][[k]]
	stab_res2 <- res[[df_nms[2]]][['stability']][['results']][[k]]
	stab_res3 <- res[[df_nms[3]]][['stability']][['results']][[k]]
	for (i in seq_along(eval_metrics)) {
		em <- eval_metrics[i]
		em_nm <- toupper(gsub("_", " ", ifelse(em == "scaled_mqloss", "smql", em))) 
		sm <- stab_metrics[i]
		sm_nm <- toupper(gsub("_", " ", sm))
		d1_nm <- stringr::str_replace_all(toupper(df_nms[1]), "_.*", "")
		d2_nm <- stringr::str_replace_all(toupper(df_nms[2]), "_.*", "")
		d3_nm <- stringr::str_replace_all(toupper(df_nms[3]), "_.*", "")
		ge1 <- plot_test_results_facet(
			data = eval_res1$tests[[em]], .metric = em, by = "retrain_window", 
			metric_label = em_nm, title = paste(d1_nm, "-", em_nm, "- Nemenyi Test")
		)
		ge2 <- plot_test_results_facet(
			data = eval_res2$tests[[em]], .metric = em, by = "retrain_window", 
			metric_label = em_nm, title = paste(d2_nm, "-", em_nm, "- Nemenyi Test")
		)
		ge3 <- plot_test_results_facet(
			data = eval_res3$tests[[em]], .metric = em, by = "retrain_window", 
			metric_label = em_nm, title = paste(d3_nm, "-", em_nm, "- Nemenyi Test")
		)
		gs1 <- plot_test_results_facet(
			data = stab_res1$tests[[sm]], .metric = sm, by = "retrain_window", 
			metric_label = sm_nm, title = paste(d1_nm, "-", sm_nm, "- Nemenyi Test")
		)
		gs2 <- plot_test_results_facet(
			data = stab_res2$tests[[sm]], .metric = sm, by = "retrain_window", 
			metric_label = sm_nm, title = paste(d2_nm, "-", sm_nm, "- Nemenyi Test")
		)
		gs3 <- plot_test_results_facet(
			data = stab_res3$tests[[sm]], .metric = sm, by = "retrain_window", 
			metric_label = sm_nm, title = paste(d3_nm, "-", sm_nm, "- Nemenyi Test")
		)
		print(ge1)
		print(ge2)
		print(ge3)
		print(gs1)
		print(gs2)
		print(gs3)
	}
}



# =========================================================================
# * Optimal Retraining Scenario -------------------------------------------
# =========================================================================

config = get_config('config/anal/anal_iifsas_retrain_config.yaml')
opt_freq <- analyze_optimal_frequency(config, adjust = 2)

# Evaluation
opt_freq$m4_daily$evaluation$results$ML_DL$plots$rmsse$overall +
    opt_freq$m5_daily$evaluation$results$ML_DL$plots$rmsse$overall +
	opt_freq$vn1_weekly$evaluation$results$ML_DL$plots$rmsse$overall

opt_freq$m4_daily$evaluation$results$ML_DL$plots$scaled_mqloss$overall +
    opt_freq$m5_daily$evaluation$results$ML_DL$plots$scaled_mqloss$overall +
	opt_freq$vn1_weekly$evaluation$results$ML_DL$plots$scaled_mqloss$overall

opt_freq$m4_daily$evaluation$results$ML_DL$plots$rmsse$bymethod
opt_freq$m4_daily$evaluation$results$ML_DL$plots$scaled_mqloss$bymethod

opt_freq$m5_daily$evaluation$results$ML_DL$plots$rmsse$bymethod
opt_freq$m5_daily$evaluation$results$ML_DL$plots$scaled_mqloss$bymethod

opt_freq$vn1_weekly$evaluation$results$ML_DL$plots$rmsse$bymethod
opt_freq$vn1_weekly$evaluation$results$ML_DL$plots$scaled_mqloss$bymethod

# Stability
opt_freq$m4_daily$stability$results$ML_DL$plots$smapc$overall +
    opt_freq$m5_daily$stability$results$ML_DL$plots$smapc$overall +
	opt_freq$vn1_weekly$stability$results$ML_DL$plots$smapc$overall

opt_freq$m4_daily$stability$results$ML_DL$plots$smqc$overall +
    opt_freq$m5_daily$stability$results$ML_DL$plots$smqc$overall +
	opt_freq$vn1_weekly$stability$results$ML_DL$plots$smqc$overall

opt_freq$m4_daily$stability$results$ML_DL$plots$smapc$bymethod
opt_freq$m4_daily$stability$results$ML_DL$plots$smqc$bymethod

opt_freq$m5_daily$stability$results$ML_DL$plots$smapc$bymethod
opt_freq$m5_daily$stability$results$ML_DL$plots$smqc$bymethod

opt_freq$vn1_weekly$stability$results$ML_DL$plots$smapc$bymethod
  opt_freq$vn1_weekly$stability$results$ML_DL$plots$smqc$bymethod
