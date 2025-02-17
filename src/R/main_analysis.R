
# install.packages('tidyverse')
# install.packages('greybox')
# install.packages('DT')
# install.packages('patchwork')
# install.packages('reticulate')

library(tidyverse)
library(greybox)
library(DT)
library(patchwork)
library(reticulate)

source('src/R/utils.R')
reticulate::source_python('src/Python/utils/utilities.py')


config = get_config('config/analyse_config.yaml')
res <- analyse_results(config)
