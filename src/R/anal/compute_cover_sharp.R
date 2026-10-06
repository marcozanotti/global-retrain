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


# analysis config
analysis_name <- config$analysis$name
analysis_types <- "evaluation"
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
eval_metrics <- c("cov", "sharp")
eval_outlier_cleaning_metrics <- config$evaluation_params$outlier_cleaning_metrics
eval_outlier_cleaning_quantiles <- config$evaluation_params$outlier_cleaning_quantiles
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
  # if (!is.null(exclude_series)) {
  #   cat("Excluding series from the analysis...\n")
  #   anal_df_tmp <- anal_df_tmp |>
  #     dplyr::filter(!unique_id %in% exclude_series)
  # }
  anal_df_tmp <- anal_df_tmp |>
    dplyr::filter(retrain_window %in% retrain_scn_tmp) |>
    dplyr::filter(method %in% model_names) |>
    recode_data(model_type_levels, model_names_abbr)
  # if (!is.null(eval_outlier_cleaning_metrics)) {
  #   anal_df_tmp <- anal_df_tmp |>
  #     clean_outliers(
  #       .metric = eval_outlier_cleaning_metrics,
  #       q = eval_outlier_cleaning_quantiles
  #     )
  # }
  anal_df_agg_tmp <- anal_df_tmp |>
    aggregate_data(
      group_columns = c('type', 'method', 'retrain_window'),
      drop_columns = c('unique_id', 'test_window', 'horizon'),
      function_name = 'mean',
      adjust_metrics = TRUE
    )

  res_data[[dn]] <- anal_df_agg_tmp
}

# LaTeX tables ------------------------------------------------------------
levels_ci <- c(50, 60, 70, 80, 90, 95, 99)
cov_cols <- paste0("coverage_level", levels_ci)
sharp_cols <- paste0("scaled_sharpness_level", levels_ci)

tex_lines <- c()
for (ds in names(res_data)) {
  df <- res_data[[ds]] |> arrange(method, retrain_window)

  header <- c(
    "\\begin{table}[htbp]",
    "\\centering",
    "\\begin{adjustbox}{max width=\\textwidth, max totalheight=0.9\\textheight}",
    paste0("\\begin{tabular}{ll", strrep("r", 2 * length(levels_ci)), "}"),
    "\\toprule",
    paste0(
      " & & \\multicolumn{",
      length(levels_ci),
      "}{c}{Coverage}",
      " & \\multicolumn{",
      length(levels_ci),
      "}{c}{Scaled sharpness} \\\\"
    ),
    paste0(
      "\\cmidrule(lr){3-",
      2 + length(levels_ci),
      "}",
      "\\cmidrule(lr){",
      3 + length(levels_ci),
      "-",
      2 + 2 * length(levels_ci),
      "}"
    ),
    paste0(
      "Method & Window & ",
      paste(c(levels_ci, levels_ci), collapse = " & "),
      " \\\\"
    ),
    "\\midrule"
  )

  body <- c()
  methods <- unique(df$method)
  for (m in methods) {
    df_m <- df |> filter(method == m)
    for (i in 1:nrow(df_m)) {
      method_cell <- ifelse(
        i == 1,
        paste0("\\multirow{", nrow(df_m), "}{*}{", m, "}"),
        ""
      )
      values <- sprintf("%.3f", unlist(df_m[i, c(cov_cols, sharp_cols)]))
      body <- c(
        body,
        paste0(
          method_cell,
          " & ",
          df_m$retrain_window[i],
          " & ",
          paste(values, collapse = " & "),
          " \\\\"
        )
      )
    }
    if (m != methods[length(methods)]) body <- c(body, "\\midrule")
  }

  footer <- c(
    "\\bottomrule",
    "\\end{tabular}",
    "\\end{adjustbox}",
    paste0(
      "\\caption{Empirical coverage and scaled sharpness by method and retrain window ",
      "at nominal levels 50--99\\% (",
      gsub("_", " ", ds),
      ").}"
    ),
    paste0("\\label{tab:covsharp_", ds, "}"),
    "\\end{table}",
    ""
  )

  tex_lines <- c(tex_lines, header, body, footer)
}

cat(tex_lines)
writeLines(tex_lines, "coverage_sharpness_tables.tex")


# Plots -------------------------------------------------------------------
levels_ci <- c(50, 60, 70, 80, 90, 95, 99)
for (ds in names(res_data)) {
  df_long <- res_data[[ds]] |>
    pivot_longer(
      cols = c(
        starts_with("coverage_level"),
        starts_with("scaled_sharpness_level")
      ),
      names_to = c("metric", "level"),
      names_pattern = "(.*)_level(.*)",
      values_to = "value"
    ) |>
    mutate(
      level = as.numeric(level) / 100,
      retrain_window = factor(
        retrain_window,
        levels = sort(unique(retrain_window))
      )
    )

  # Coverage: empirical vs nominal, dashed line = perfect calibration
  p_cov <- df_long |>
    filter(metric == "coverage") |>
    ggplot(aes(x = level, y = value, color = retrain_window)) +
    geom_abline(
      slope = 1,
      intercept = 0,
      linetype = "dashed",
      color = "grey40"
    ) +
    geom_line() +
    geom_point(size = 1) +
    facet_wrap(~method) +
    expand_limits(x = 1, y = 1) +
    scale_x_continuous(breaks = levels_ci / 100, labels = levels_ci) +
    labs(
      x = "Nominal level (%)",
      y = "Empirical coverage",
      color = "Retrain window"
    ) +
    theme_bw()
  print(p_cov)

  # Scaled sharpness: mean interval width by nominal level
  p_sharp <- df_long |>
    filter(metric == "scaled_sharpness") |>
    ggplot(aes(x = level, y = value, color = retrain_window)) +
    geom_line() +
    geom_point(size = 1) +
    facet_wrap(~method, scales = "free_y") +
    scale_x_continuous(breaks = levels_ci / 100, labels = levels_ci) +
    labs(
      x = "Nominal level (%)",
      y = "Scaled sharpness",
      color = "Retrain window"
    ) +
    theme_bw()
  print(p_sharp)
}
