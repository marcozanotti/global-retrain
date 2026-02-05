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
analysis_file_name <- 'docs/iifsas_retrain/absolute_evalstab_overlap_20260204_174211.RData'
analysis_file_name <- 'docs/iifsas_retrain/relative_evalstab_overlap_20260204_171927.RData'

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
				(eval_res1$plots[[em]] + ggplot2::guides(col = FALSE) + ggplot2::labs(x = NULL)) +
				(stab_res1$plots[[sm]] + ggplot2::guides(col = FALSE) + ggplot2::labs(x = NULL)) 
			) /
			(
				(eval_res2$plots[[em]] + ggplot2::guides(col = FALSE) + ggplot2::labs(x = NULL)) +
				(stab_res2$plots[[sm]] + ggplot2::guides(col = FALSE) + ggplot2::labs(x = NULL))
			) /
			(
				(eval_res3$plots[[em]]) +
				(stab_res3$plots[[sm]])
			) +
			patchwork::plot_layout(guides = "collect") & ggplot2::theme(legend.position = "bottom")
		)
	}
}

# 600 x 400
{
  	eval_res2 <- res[[df_nms[2]]][['evaluation']][['results']][["SF"]]
	stab_res2 <- res[[df_nms[2]]][['stability']][['results']][["SF"]]
  	(
		(
			eval_res2$plots[["rmsse"]] + 
				ggplot2::theme(
					plot.title = ggplot2::element_blank(),
					axis.title.x = ggplot2::element_blank(),
					axis.text.x = ggplot2::element_blank(),
					axis.ticks.x = ggplot2::element_blank()
				)
		) +
		(
			stab_res2$plots[["smapc"]] + 
				ggplot2::theme(
					plot.title = ggplot2::element_blank(),
					axis.title.x = ggplot2::element_blank(),
					axis.text.x = ggplot2::element_blank(),
					axis.ticks.x = ggplot2::element_blank()
				)
		)
	) /
	(
		(eval_res2$plots[["scaled_mqloss"]] + ggplot2::labs(title = NULL)) +
		(stab_res2$plots[["smqc"]] + ggplot2::labs(title = NULL))
	) +
	patchwork::plot_layout(guides = "collect") & 
        patchwork::plot_annotation('ETS - M5', theme = ggplot2::theme(plot.title = ggplot2::element_text(hjust = 0.5))) &
        ggplot2::theme(legend.position = "none") &
		ggplot2::scale_color_manual(values = c("#3b3b3b"))
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
opt_freq <- analyze_optimal_frequency_combined(config, adjust = 2)

(
	(opt_freq$m4_daily$ML_DL[[1]]$overall + ggplot2::labs(x = NULL)) + 
    (opt_freq$m4_daily$ML_DL[[2]]$overall + ggplot2::labs(x = NULL))
) /
(
	(opt_freq$m5_daily$ML_DL[[1]]$overall + ggplot2::labs(x = NULL)) +
	(opt_freq$m5_daily$ML_DL[[2]]$overall + ggplot2::labs(x = NULL))
) /
(
	opt_freq$vn1_weekly$ML_DL[[1]]$overall +
	opt_freq$vn1_weekly$ML_DL[[2]]$overall
) +
patchwork::plot_layout(guides = "collect") &  ggplot2::theme(legend.position = "bottom")







