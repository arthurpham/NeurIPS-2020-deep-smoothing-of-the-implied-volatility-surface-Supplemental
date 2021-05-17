rm(list = ls())

## Load packages
p <- c("tidyverse", "lubridate", "patchwork", "ggthemes", "tensorflow")
sapply(p, require, character.only = TRUE)

## Load functions
files <- c(
  "bsm", "plot", "ivsmoother_models", "ivsmoother_fit", "tensorflow"
)
purrr::walk(paste("R/utils_", files, ".R", sep = ""), source)

## Tensorflow
sess <- reset_tf_session(gpu_mem_frac = 0.1)

# ## Load train data
df_train <- read_csv("data/train_data.csv") 

## visual check
df_train %>% plot_totvar()

## Dictionnaries
di_train <- get_ivsmoother_dict(df_train)$di

# Get ATM data
df_atm <- df_train %>% 
  group_by(ttm) %>%
  filter(abs(logm) == min(abs(logm)))

# Compute extrapolation function for the ATM total variance
w_atm_fun <- get_w_atm(df_atm)

## The default controls for the model and fit
ivsmoother_controls <- get_ivsmoother_controls(w_atm_fun = w_atm_fun)
# ivsmoother_controls$prior <- "bs"
fit_controls <- get_fit_controls()

## Fit
model <- get_ivsmoother(ivsmoother_controls)
system.time(output <- init_and_train(model, di_train, sess, fit_controls))

## Same figure as in the paper (scenario 12)
p <- plot_fit(df_train %>% mutate(name = "Data"), model, sess)
p