plot_one <- function(scenario_number, results, df, scenarios, sess) {
  
  # Tensorflow
  sess <- reset_tf_session(gpu_mem_frac = 0.1)
  
  # Get ATM data
  df_atm <- df %>%
    group_by(ttm) %>%
    filter(abs(logm) == min(abs(logm)))
  
  # Compute extrapolation function for the ATM total variance
  w_atm_fun <- get_w_atm(df_atm)
  
  ## The default controls for the model and fit
  ivsmoother_controls <- get_ivsmoother_controls(w_atm_fun = w_atm_fun)
  fit_controls <- get_fit_controls()
  
  ## Fit
  ivsmoother_controls$w_atm_fun <- get_w_atm(df_atm)
  fit_controls <- get_fit_controls()
  
  ## Set the penalty scenario and result
  ivsmoother_controls$prior <- scenarios %>%
    dplyr::filter(scenario == scenario_number) %>%
    pull(prior)
  fit_controls$penalty <- scenarios %>%
    dplyr::filter(scenario == scenario_number) %>%
    dplyr::select(fit, c4, c5, c6, atm) %>%
    as.numeric()
  data_controls$logm_prop <- scenarios %>%
    dplyr::filter(scenario == scenario_number) %>%
    pull(logm_prop)
  
  model <- get_ivsmoother(ivsmoother_controls)
  reset_prior(model, sess, results$weights, results$params)
  df <- df %>% mutate(name = "Data")
  df_train <- df %>% semi_join(results$df_train %>% select(ttm, logm)) %>%
    mutate(train = TRUE)
  df_test <- df %>% semi_join(results$df_test %>% select(ttm, logm)) %>%
    mutate(train = FALSE)
  
  
  plot_fit(bind_rows(df_train, df_test), model, sess) +
    plot_annotation(
      title = paste0("Prior = ", str_to_upper(model$prior_model$prior),
                     ", Lambda = ", fit_controls$penalty[2])
    )
}

get_fit <- function(df, model, sess){
  
  di <- get_ivsmoother_dict(df)$di
  
  nn <- df %>%
    pull(name) %>%
    unique() %>%
    str_to_title()
  
  df_fit <- df %>%
    mutate(w_prior = sess$run(model$preds$w_prior,
                              feed_dict = di) %>% as.numeric(),
           w_hat = sess$run(model$preds$w_hat,
                            feed_dict = di) %>% as.numeric()) %>% 
    dplyr::select(-c(name, iv, put, call)) %>%
    pivot_longer(c(w, w_prior, w_hat), values_to = "w") %>%
    mutate(
      call = NA, 
      put = NA,
      iv = (w / ttm)**0.5,
      ttm = round(ttm, 2),
      name = factor(name, 
                    levels = c("w", "w_prior", "w_hat"),
                    labels = c(nn, "Prior", "Prior x NN Model")))
  
  return(df_fit)
}

plot_fit <- function(df, model, sess, maturity = "ttm") {
  
  df_fit <- get_fit(df, model, sess)
  
  ttm <- df_fit %>% pull(ttm) %>% unique()
  sel_slices <- ttm[map_int(c(1/12, 2/12, 1, 2), function(t) which.min(abs(ttm - t)))]
  

  
  p1 <- df_fit %>% 
    plot_totvar(maturity = maturity, 
                title = "Data vs Predictions: Total Variance", 
                train = FALSE) + 
    labs(color = "Time to maturity")
  
  df_fit <- df_fit %>% 
    dplyr::filter(ttm %in% sel_slices) 
  if (maturity != "ttm") {
    p1 <- p1  +
      guides(color = guide_legend(ncol = 2, byrow = TRUE))
  } else {
    df_fit <- df_fit %>%
      mutate(ttm = factor(ttm, labels = paste("Time to maturity = ", 
                                              c("1 month", "2 months", 
                                                "1 year", "2 years"))))
  }
  
  p2 <- df_fit %>%
    plot_impvol(maturity = maturity, nrow = 1,
                title = "Data vs Predictions: Implied Volatility",
                train = FALSE)
  
  df_extended <- get_ivsmoother_dict(df)$ttm_logm %>%
    filter(type == "c4c5") %>%
    pull(ttm_logm) %>%
    (function(x) x[[1]]) %>% 
    mutate(m = exp(logm),
           w = 0, 
           iv = 0, 
           expiry = ttm)
  
  di <- get_ivsmoother_dict(df_extended)$di
  
  df_extended <- df_extended %>%
    mutate(w_prior = sess$run(model$preds$w_prior,
                              feed_dict = di) %>% as.numeric(),
           ann_output = sess$run(model$preds$output,
                            feed_dict = di) %>% as.numeric() / 
             (2 * sess$run(model$prior_model$params$scale, feed_dict = di)) %>% as.numeric(),
           w_hat = sess$run(model$preds$w_hat,
                            feed_dict = di) %>% as.numeric()) %>%  
    dplyr::select(-c(iv, w)) %>%
    pivot_longer(c(w_prior, ann_output, w_hat), values_to = "w") %>%
    mutate(
      call = NA, 
      put = NA,
      iv = (w / ttm)**0.5,
      name = factor(name, 
                    levels = c("w_prior", "ann_output", "w_hat"),
                    labels = c("Prior", "Scaled NN Model", "Prior x NN Model")))
  
  p3 <- df_extended %>% 
    plot_totvar(title = "Predictions on an Extended Grid") +
    labs(color = "Time to maturity")
  
  (p1 / p2 / p3) * theme_minimal()
    #(theme_minimal() + theme(legend.position = "bottom"))
}

plot_totvar <- function(df, title = "",
                        maturity = "expiry",
                        y = "Total Variance",
                        train = TRUE) {
  df <- df %>% 
    mutate(maturity = df %>% pull(!!maturity))
  p <- df %>%
  ggplot(aes(x = logm, y = w, color = maturity, group = maturity)) +
    geom_line()
  if (train == FALSE) {
    p <- p   +
      geom_point(data = df %>% dplyr::filter(train == TRUE), 
                 size = 0.8)
  }
  p +
    facet_wrap(~name) +
    labs(
      x = "Log-Moneyness",
      y = y,
      title = title,
      color = maturity
    )
}

plot_impvol <- function(df, title = "", maturity = "expiry", nrow = NULL, train = TRUE) {
    df <- df %>% 
    mutate(maturity = df %>% pull(!!maturity))
    
    p <- df %>%
    ggplot(aes(x = logm, y = iv, color = name)) +
    geom_line()
    
    if (train == FALSE) {
      p <- p   +
        geom_point(data = df %>% dplyr::filter(train == TRUE), size = 0.8)
    }
    p +
    facet_wrap(~maturity, nrow = nrow) +
    labs(
      x = "Log-Moneyness",
      y = "Implied Volatility",
      title = title,
      color = "Data / Model"
    )
}

plot_callput <- function(df, title = "") {
  ggplot(data = df, aes(x = K)) +
    geom_line(aes(y = call, colour = "call")) +
    geom_line(aes(y = put, colour = "put")) +
    xlab("Moneyness") +
    ylab("Option Price") +
    facet_wrap(~expiry) +
    ggtitle(title)
}

plot_put <- function(df, title = "") {
  ggplot(data = df, aes(x = m, y = put, color = name)) +
    geom_line() +
    xlab("Moneyness") +
    ylab("Option Price") +
    facet_wrap(~expiry) +
    ggtitle(title)
}

plot_price <- function(df, title = "") {

  ## SLOW !!!
  df <- df %>%
    rowwise() %>%
    mutate(price = bsm_price(S = 1, K = m, tau = ttm, r = 0, d = 0, sig = iv, type = ifelse(m < 1, "P", "C"))) %>%
    ungroup()

  ggplot(data = df, aes(x = m, y = price, group = expiry)) +
    geom_line(aes(color = expiry)) +
    xlab("Moneyness") +
    ylab("OTM Option Price") +
    facet_wrap(~name) +
    ggtitle(title)
}
