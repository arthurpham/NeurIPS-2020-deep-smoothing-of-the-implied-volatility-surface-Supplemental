rm(list = ls())
cores <- 40 # for parallel pre-processing
optpr_file <- here::here("data", "spx", "opprcd.csv") # option prices file from OptionMetrics extracted from WRDS
secpr_file <- here::here("data", "spx", "secprd.csv") # index prices file from OptionMetrics extracted from WRDS
output_file <- here::here("data", "spx", "optall.rds") # for the results

## Load packages
packages <- c("here", "tidyverse", "lubridate", "furrr")
purrr::walk(packages, require, character.only = TRUE)
plan(multisession, workers = cores)

## Load files
files <- c(
  "bsm", "data_preprocess"
)
paste("utils_", files, ".R", sep = "") %>%
  (function(x) here::here("R", x)) %>%
  walk(source)

## Read index and option data
## (*) Filter monthlies overlapping with weeklies
df <- read_csv(optpr_file) %>%
  left_join(read_csv(secpr_file) %>% select(date, close), by = "date") %>%
  select(date, exdate, close, strike_price, best_bid, best_offer, 
         symbol, open_interest, cp_flag) %>% 
  transmute(
    date = date %>% num_to_date(),
    exdate = exdate %>% num_to_date(),
    ttm = as.numeric(exdate - date) / 365.25,
    close = close,
    strike = strike_price * 1e-3,
    mid = 0.5 * (best_offer + best_bid),
    best_bid = best_bid,
    best_offer = best_offer,
    spread = best_offer - best_bid,
    weekly = ifelse(substr(symbol, 4, 4) =="W", TRUE, FALSE),
    open_interest = open_interest,
    cp_flag = cp_flag) %>%
  drop_na() %>%
  arrange(date, exdate, strike, cp_flag, !weekly) %>% # (*) remove the ! to select the monthlies instead
  distinct(date, exdate, strike, cp_flag, .keep_all = TRUE) 

## Extract forward using the Put-Call parity
system.time(df_fw <- df %>%
              group_by(date, exdate) %>%              
              select(cp_flag, ttm, strike, mid, close) %>%
              pivot_wider(names_from = cp_flag, values_from = mid) %>%
              group_split() %>%
              future_map_dfr(extract_forward, .progress = TRUE) %>%
              fill_na_forward() %>%
              select(-c(ttm, close)))


## Merge both data frames and keep only out-of-the-money options
df <- right_join(df, df_fw, by = c("date", "exdate")) %>% 
  mutate(logm = log(strike / forward)) %>%
  filter((cp_flag == "P" & logm <= 0) | (cp_flag == "C" & logm > 0))

## Compute implied volatilities
safe_iv <- possibly(
  function(K, ttm, rf, q, price, cp_flag) {
    gbsm_iv(1.0, K, ttm, rf * ttm, q * ttm, price, cp_flag)
  },
  NA_real_,
  quiet = TRUE
)
system.time(df <- df %>%
              mutate(iv_mid = future_pmap_dbl(
                list(K = strike / close, ttm, rf, q, price = mid / close, cp_flag),
                safe_iv, .progress = TRUE),
                iv_best_bid = future_pmap_dbl(
                  list(K = strike / close, ttm, rf, q, price = best_bid / close, cp_flag),
                  safe_iv, .progress = TRUE),
                iv_best_offer = future_pmap_dbl(
                  list(K = strike / close, ttm, rf, q, price = best_offer / close, cp_flag),
                  safe_iv, .progress = TRUE)
              ))

## Save
write_rds(df, output_file)
