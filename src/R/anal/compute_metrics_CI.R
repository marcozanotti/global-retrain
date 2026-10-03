# Compute metrics CI
reticulate::use_condaenv('global_retrain')

library(tidyverse)
library(greybox)
library(DT)
library(patchwork)
library(reticulate)

source('src/R/utils.R')
reticulate::source_python('src/Python/utils/utilities.py')
config <- get_config('config/anal/anal_global_stability_config.yaml')


boot_ci <- function(
  df,
  metrics,
  B = 1000,
  conf = c(0.025, 0.975),
  seed = 1992
) {
  set.seed(seed)
  cells <- dplyr::distinct(df, type, method, retrain_window)
  res <- vector("list", length = nrow(cells))
  for (i in seq_len(nrow(cells))) {
    cat(paste0("Computing CI for cell: ", i, "/", nrow(cells), "...\n"))
    d <- dplyr::semi_join(
      df,
      cells[i, ],
      by = c('type', 'method', 'retrain_window')
    )
    res_m <- vector("list", length = length(metrics))
    for (m in metrics) {
      # cat(paste0("Computing CI for metric: ", m, "...\n"))
      x <- d[[m]] # one value per series
      boot_means <- replicate(B, mean(sample(x, replace = TRUE)))
      res_m[[m]] <- tibble::tibble(
        cells[i, ],
        metric = m,
        n_series = length(x),
        mean = mean(x),
        ci_lower = quantile(boot_means, conf[1], names = FALSE),
        ci_upper = quantile(boot_means, conf[2], names = FALSE)
      )
    }
    res[[i]] <- dplyr::bind_rows(res_m)
  }
  ci_df <- dplyr::bind_rows(res)
  return(ci_df)
}


# analysis config
analysis_name <- config$analysis$name
analysis_types <- config$analysis$types
analysis_method <- config$analysis$method
analysis_sample_type <- config$analysis$sample_type
# dataset config
dataset_names <- config$dataset$dataset_names
frequencies <- config$dataset$frequencies
dataset_names_full <- paste(dataset_names, frequencies, sep = "_")
retrain_scenarios <- purrr::map(dataset_names, get_retrain_scenarios) |>
  purrr::set_names(dataset_names_full)
ext <- config$dataset$ext
# model config
model_types <- config$models$types
model_names <- config$models$model_names
model_names_abbr <- config$models$model_names_abbr
model_type_levels <- unlist(model_types) # c('SF', 'ML', 'DL', 'ENSACC', 'ENSTIME')
# evaluation, time, stability and cost config
eval_metrics <- config$evaluation_params$metrics
eval_outlier_cleaning_metrics <- config$evaluation_params$outlier_cleaning_metrics
eval_outlier_cleaning_quantiles <- config$evaluation_params$outlier_cleaning_quantiles
time_metrics <- config$time_params$metrics
stab_metrics <- config$stability_params$metrics
stab_outlier_cleaning_metrics <- config$stability_params$outlier_cleaning_metrics
stab_outlier_cleaning_quantiles <- config$stability_params$outlier_cleaning_quantiles
cost_metrics <- config$cost_params$metrics
cost_time_var <- config$cost_params$time_var
cost_n_skus <- config$cost_params$n_skus
cost_per_hour <- unlist(config$cost_params$cost_per_hour)
cost_datasets_n_skus <- as.list(unlist(
  config$cost_params$cost_datasets_n_skus
))
adjust_time_for_sf <- config$cost_params$adjust_time_for_sf
n_skus_sf <- config$cost_params$n_skus_sf
n_cores <- config$cost_params$n_cores
exclude_series <- config$exclude_series

# final analysis names
res_data <- vector("list", length = length(dataset_names))
names(res_data) <- dataset_names_full

for (i in seq_along(dataset_names)) {
  dn <- dataset_names_full[i]
  cat(paste0("*************** Analysing ", dn, " ***************\n"))
  dataset_name_tmp <- dataset_names[i]
  freq_tmp <- frequencies[i]
  retrain_scn_tmp <- retrain_scenarios[[dn]]

  res_anal_types <- vector("list", length = length(analysis_types))
  names(res_anal_types) <- analysis_types

  for (at in analysis_types) {
    cat(paste0("--- [ Analysis: ", at, " ] ---\n"))

    if (at == "evaluation") {
      cat("Loading and preparing the evaluation data...\n")
      anal_df_tmp <- load_data(
        path_list = c('results', dataset_name_tmp, freq_tmp, 'evaluation'),
        name_list = c(
          dataset_name_tmp,
          freq_tmp,
          'eval',
          analysis_sample_type
        ),
        ext = ext
      ) |>
        tibble::as_tibble()
      if (!is.null(exclude_series)) {
        cat("Excluding series from the analysis...\n")
        anal_df_tmp <- anal_df_tmp |>
          dplyr::filter(!unique_id %in% exclude_series)
      }
      anal_df_tmp <- anal_df_tmp |>
        dplyr::filter(retrain_window %in% retrain_scn_tmp) |>
        dplyr::filter(method %in% model_names) |>
        recode_data(model_type_levels, model_names_abbr)
      if (!is.null(eval_outlier_cleaning_metrics)) {
        anal_df_tmp <- anal_df_tmp |>
          clean_outliers(
            .metric = eval_outlier_cleaning_metrics,
            q = eval_outlier_cleaning_quantiles
          )
      }
      # anal_df_agg_tmp <- anal_df_tmp |>
      #   aggregate_data(
      #     group_columns = c('type', 'method', 'retrain_window'),
      #     drop_columns = c('unique_id', 'test_window', 'horizon'),
      #     function_name = 'mean',
      #     adjust_metrics = TRUE
      #   )
      # set the metrics to be used
      anal_metrics <- eval_metrics
    } else if (at == "stability") {
      cat("Loading and preparing the stability data...\n")
      anal_df_tmp <- load_data(
        path_list = c('results', dataset_name_tmp, freq_tmp, 'stability'),
        name_list = c(dataset_name_tmp, freq_tmp, 'stab'),
        ext = ext
      ) |>
        tibble::as_tibble()
      if (!is.null(exclude_series)) {
        cat("Excluding series from the analysis...\n")
        anal_df_tmp <- anal_df_tmp |>
          dplyr::filter(!unique_id %in% exclude_series)
      }
      anal_df_tmp <- anal_df_tmp |>
        dplyr::filter(retrain_window %in% retrain_scn_tmp) |>
        dplyr::filter(method %in% model_names) |>
        recode_data(model_type_levels, model_names_abbr) |>
        dplyr::filter(type != 'ENSTIME') # remove ensemble time from stability analysis
      if (!is.null(stab_outlier_cleaning_metrics)) {
        anal_df_tmp <- anal_df_tmp |>
          clean_outliers(
            .metric = stab_outlier_cleaning_metrics,
            q = stab_outlier_cleaning_quantiles
          )
      }
      # anal_df_agg_tmp <- anal_df_tmp |>
      #   aggregate_data(
      #     group_columns = c('type', 'method', 'retrain_window'),
      #     drop_columns = c('unique_id', 'test_window', 'horizon'),
      #     function_name = 'mean',
      #     adjust_metrics = TRUE
      #   )
      # set the metrics to be used
      anal_metrics <- stab_metrics
    } else {
      stop(paste0("Unknown analysis type: ", at))
    }

    if ("rmsse" %in% anal_metrics) {
      # replace rmsse with msse for CI computation
      anal_metrics[anal_metrics == "rmsse"] <- "msse"
    }
    anal_df_tmp_selected <- anal_df_tmp |>
      dplyr::select(
        dplyr::all_of(
          c('type', 'method', 'retrain_window', 'unique_id', anal_metrics)
        )
      )

    ci_df <- boot_ci(
      df = anal_df_tmp,
      metrics = anal_metrics,
      B = 10000,
      conf = c(0.025, 0.975),
      seed = 1992
    )

    if ("msse" %in% anal_metrics) {
      # replace msse with rmsse for CI computation
      ci_df <- ci_df |>
        dplyr::mutate(
          metric = dplyr::if_else(metric == "msse", "rmsse", metric),
          mean = dplyr::if_else(metric == "rmsse", sqrt(mean), mean),
          ci_lower = dplyr::if_else(
            metric == "rmsse",
            sqrt(ci_lower),
            ci_lower
          ),
          ci_upper = dplyr::if_else(metric == "rmsse", sqrt(ci_upper), ci_upper)
        )
    }

    res_anal_types[[at]] <- ci_df
  }

  res_data[[dn]] <- dplyr::bind_rows(res_anal_types)
}


saveRDS(res_data, file = paste0('docs/global_stability/', 'ci_res.rds'))
ci_res <- readRDS("docs/global_stability/ci_res.rds")

d_name <- "m5_daily"
ci_df <- ci_res[[d_name]]

for (m in unique(ci_df$metric)) {
  df_m <- dplyr::filter(ci_df, metric == m)
  p <- ggplot(df_m, aes(x = factor(retrain_window), y = mean)) +
    geom_errorbar(aes(ymin = ci_lower, ymax = ci_upper), width = 0.3) +
    geom_point() +
    facet_wrap(~method, scales = 'free_y') +
    labs(
      title = toupper(m),
      x = 'Retrain window',
      y = 'Mean with 95% bootstrap CI'
    ) +
    theme_bw()
  print(p)
}

metric_labels <- c(
  rmsse = 'RMSSE',
  scaled_mqloss = 'SMQL',
  smapc = 'sMAPC',
  smqpc = 'sMQPC'
)
for (m in unique(ci_df$metric)) {
  tab <- ci_df |>
    dplyr::filter(metric == m) |>
    dplyr::arrange(method, retrain_window) |>
    dplyr::mutate(ci = sprintf('[%.3f, %.3f]', ci_lower, ci_upper)) |>
    dplyr::select(method, retrain_window, ci) |>
    tidyr::pivot_wider(names_from = retrain_window, values_from = ci)

  rows <- apply(tab, 1, function(x) paste0(paste(x, collapse = ' & '), ' \\\\'))
  tex <- c(
    '\\begin{table}[!ht]',
    '\\centering',
    paste0(
      '\\caption{',
      metric_labels[m],
      ': 95\\% bootstrap confidence intervals for the mean across series.}'
    ),
    paste0('\\label{tab:ci_', m, '}'),
    '\\begin{adjustbox}{max width=\\textwidth}',
    paste0('\\begin{tabular}{l', strrep('c', ncol(tab) - 1), '}'),
    '\\toprule',
    paste0('Method & ', paste(names(tab)[-1], collapse = ' & '), ' \\\\'),
    '\\midrule',
    rows,
    '\\bottomrule',
    '\\end{tabular}',
    '\\end{adjustbox}',
    '\\end{table}'
  )
  cat(tex)
  cat("\n\n")
  #writeLines(tex, paste0('ci_', m, '.tex'))
}


metric_labels <- c(
  rmsse = 'RMSSE',
  scaled_mqloss = 'SMQL',
  smapc = 'sMAPC',
  smqpc = 'sMQPC'
)
tex <- c()

for (dn in names(ci_res)) {
  ci_df <- ci_res[[dn]]
  ds_label <- toupper(sub('_.*', '', dn)) # m5_daily -> M5

  for (m in unique(ci_df$metric)) {
    tab <- ci_df |>
      dplyr::filter(metric == m) |>
      dplyr::arrange(method, retrain_window) |>
      dplyr::mutate(ci = sprintf('[%.3f, %.3f]', ci_lower, ci_upper)) |>
      dplyr::select(method, retrain_window, ci) |>
      tidyr::pivot_wider(names_from = retrain_window, values_from = ci)

    rows <- apply(tab, 1, function(x) {
      paste0(paste(x, collapse = ' & '), ' \\\\')
    })
    tex <- c(
      tex,
      paste0('% ', ds_label, ' ', metric_labels[m]),
      '\\begin{table}[!ht]',
      '\\centering',
      '\\begin{adjustbox}{max width=\\textwidth}',
      paste0('\\begin{tabular}{l', strrep('c', ncol(tab) - 1), '}'),
      '\\toprule',
      paste0('Method & ', paste(names(tab)[-1], collapse = ' & '), ' \\\\'),
      '\\midrule',
      rows,
      '\\bottomrule',
      '\\end{tabular}',
      '\\end{adjustbox}',
      paste0(
        '\\caption{',
        ds_label,
        ' 95\\% Confidence Interval values for ',
        metric_labels[m],
        ' across series.}'
      ),
      paste0('\\label{tab:ci_', tolower(ds_label), '_', m, '}'),
      '\\end{table}',
      ''
    )
  }
}

writeLines(tex, 'ci_tables.tex')
