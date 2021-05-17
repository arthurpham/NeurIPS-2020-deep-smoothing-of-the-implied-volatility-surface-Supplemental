## To convert optionmetrics date format to R
num_to_date <- function(x) make_date(year = x %/% 10000,
                                     month = (x %% 10000) %/% 100,
                                     day = x %% 100)

## Extract the forward price using the Put-Call parity
## Known problems:
##  - can obtain negative div yields (q) and interest rates (rf) 
##    values for short maturities
##  - term structure changes direction too frequently contango/backwardation
##  - on some dates there are not enough options (e.g 1998-09-18 and 1 day ttm)
## Important:
##  - we only care about (rf-q)*ttm, not the individual accuracy of rf/q !!!
extract_forward <- function(df) {
  ttm <- df$ttm[1]
  close <- df$close[1]
  forward <- q <- rf <- NA
  try({
    fit <- df %>%
      mutate(
        y = -(C - P)/close, 
        x = strike/close
      ) %>% 
      filter(abs(x-1) < 0.075) %>% 
      lm(y ~ x, data = .)
    rf <- -log(coef(fit)[2]) / ttm
    q <- -log(-coef(fit)[1]) / ttm
    forward <- close * exp((rf-q) * ttm)
  }, silent=TRUE)
  tibble(
    date = df$date[1],
    exdate = df$exdate[1],
    q = q,
    rf = rf,
    forward = forward,
    close = close,
    ttm = ttm
  )
}

## Fill missing q, rf, forward
fill_na_forward <- function(df) {
  
  ## Dates with and without missing data
  df_na <- df %>% filter(is.na(q))
  df_nona <- df %>% anti_join(df_na)
  
  ## Fill NA values
  df_na <- df_na %>% 
    group_by(date) %>% 
    group_split() %>%
    map_dfr(
      function(data) {
        df_tmp <- df_nona %>% filter(date == data$date[1])
        q_fun <- approxfun(df_tmp$ttm, df_tmp$q, rule=2)
        rf_fun <- approxfun(df_tmp$ttm, df_tmp$rf, rule=2)
        data %>%
          mutate(q = q_fun(ttm),
                 rf = rf_fun(ttm),
                 forward = close * exp((rf-q) * ttm))
      }
    )
  
  ## Replace NA values
  df_nona %>% 
    bind_rows(df_na) %>% 
    arrange(date, exdate)
}