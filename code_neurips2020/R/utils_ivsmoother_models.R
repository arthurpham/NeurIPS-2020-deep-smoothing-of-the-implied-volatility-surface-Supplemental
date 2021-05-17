get_ivsmoother_controls <- function(neurons_vec = rep(40L, 4L),
                                    activation = "softPlus",
                                    prior = "svi",
                                    phi_fun = "power_law",
                                    w_atm_fun = NULL,
                                    spread = FALSE) {
  list(
    neurons_vec = neurons_vec,
    activation = activation,
    prior = prior,
    phi_fun = phi_fun,
    w_atm_fun = w_atm_fun,
    spread = spread
  )
}

get_ivsmoother <- function(controls = get_ivsmoother_controls()) {
  
  neurons_vec <- controls$neurons_vec
  activation <- controls$activation
  prior <- controls$prior
  phi_fun <- controls$phi_fun
  w_atm_fun <- controls$w_atm_fun
  spread <- controls$spread
  
  afun <- switch(activation,
                 "relu" = tf$nn$relu,
                 "softPlus" = tf$nn$softplus
  )
  
  ## Placeholders
  if (substr(tf$version$VERSION, 1, 1) != "2") {
    stop("The code has been upgraded to work with TF 2.0")
  }
  
  ## Total variance and implied volatility
  w <- tf$compat$v1$placeholder("float", shape = shape(NULL, 1L), name = "w")
  iv <- tf$compat$v1$placeholder("float", shape = shape(NULL, 1L), name = "iv")
  if (spread) {
    iv_spread <- tf$compat$v1$placeholder("float", shape = shape(NULL, 1L), name = "iv_spread")
  } else {
    iv_spread <- NULL
  }
  
  ## ttm and logm for
  ## fit = observations,
  ## c4c5 = calendar (C4) & butterfly (C5) arbitrage,
  ## c6 = large-moneyness C6),
  ## atm = atm totvar when using a prior
  ttm_logm <- tibble(
    type = c("fit", "c4c5", "c6", "atm"),
    ttm = map(type, get_placeholder, name = "ttm"),
    logm = map(type, get_placeholder, name = "logm")
  )
  
  ## Input layer
  layers <- ttm_logm %>%
    mutate(input = pmap(list(ttm, logm, type), function(ttm, logm, type) {
      tf$concat(list(ttm, logm), 1L, name = get_name("input", type))
    })) %>%
    dplyr::select(-c(ttm, logm))
  
  ## parameters
  weights <- list()
  if (!is.null(neurons_vec)) {
    ## Number of layers
    n_input <- 2L
    n_layer <- length(neurons_vec)
    for (i in c(1:n_layer, 0)) {
      ## Bias and weight names
      b_name <- paste0("b", i)
      W_name <- paste0("W", i)
      
      ## Bias and weight dimensions
      ## TODO: recheck and cleanup
      n_to <- neurons_vec[i]
      b_mean <- 0.0
      W_mean <- 0.0
      if (i == 0) {
        n_to <- 1L ## override
        n_from <- as.integer(tail(neurons_vec, 1))
        ## The last layer starts at 0!!!
        b_stddev <- 1.0 / sqrt(n_from + n_to)
        W_stddev <- 1.0 / sqrt(n_from + n_to)
      } else {
        n_from <- as.integer(n_to)
        n_to <- as.integer(neurons_vec[i])
        if (i == 1) {
          n_from <- as.integer(n_input)
        }
        b_stddev <- 1 / sqrt(n_from + n_to)
        W_stddev <- 1 / sqrt(n_from + n_to)
      }
      assign(b_name, tf$Variable(tf$random$normal(list(1L, n_to),
                                                  mean = b_mean,
                                                  stddev = b_stddev
      ),
      name = b_name
      ))
      assign(W_name, tf$Variable(tf$random$normal(list(n_from, n_to),
                                                  mean = W_mean,
                                                  stddev = W_stddev
      ),
      name = W_name
      ))
      
      ## Save variables
      weights[[b_name]] <- get(b_name)
      weights[[W_name]] <- get(W_name)
    }
    
    ## Add layers
    next_input <- layers %>% pull(input)
    for (i in seq_along(neurons_vec)) {
      layer_name <- paste("layer", i, sep = "_")
      layers <- layers %>%
        mutate(!!layer_name := map2(next_input, type, get_layer,
                                    i = i, envir = environment(), afun = afun,
                                    layer_name = layer_name
        ))
      next_input <- layers %>% pull(!!layer_name)
    }
  } else {
    next_input <- layers %>% pull(input)
    weights <- list(NULL)
  }
  
  
  ## Total variance
  if (is.null(prior)) {
    
    if (is.null(neurons_vec)) {
      stop("neurons_vec and prior can't both be empty")
    }
    
    ## Final layers is > 0
    layers <- layers %>%
      mutate(output = map2(next_input, type, function(input, type) {
        name_variable(
          var = tf$matmul(input, tf$exp(W0)) + tf$exp(b0),
          name = "output",
          type = type
        )
      }))
    
    ## Multiply by ttm for w_hat
    layers <- layers %>%
      mutate(w_hat = pmap(
        list(ttm_logm %>% pull(ttm), output, type),
        function(ttm, output, type) {
          name_variable(
            var = ttm * output,
            name = "w_hat",
            type = type
          )
        }
      ))
    
    prior_model <- NULL
  } else {
    
    ## Get the prior model
    prior_model <- get_prior(ttm_logm, afun, w_atm_fun, prior, phi_fun)
    
    if (!is.null(neurons_vec)) {
      
      ## Get the scale parameter
      scale_trans <- tf$Variable(tf$constant(log(exp(1) - 1),
                                             shape = shape(1L, 1L)
      ),
      name = "scale_trans"
      )
      scale <- tf$nn$softplus(scale_trans, name = "scale")
      prior_model$params$scale <- scale # because not sure where else to save that
      prior_model$params$scale_trans <- scale_trans # because not sure where else to save that
      prior_model$var_list <- c(prior_model$var_list, "scale_trans")
      
      ## Get the final layer
      layers <- layers %>%
        mutate(output = map2(next_input, type, function(input, type) {
          name_variable(
            var = scale * (1 + (1 - 1e-3) * tf$nn$tanh(tf$matmul(input, W0) + b0)),
            name = "output",
            type = type
          )
        }))
    } else {
      scale <- tf$constant(1.0, shape = shape(1L, 1L), name = "scale")
      layers <- layers %>% mutate(
        output = map(type, function(type) 
          name_variable(var = scale,
                        name = "output",
                        type = type))
      )
    }
    
    
    
    
    ## Multiply ANN and prior for w_hat
    layers <- layers %>%
      mutate(w_hat = pmap(
        list(prior_model$w$w_prior, output, type),
        function(w, output, type) {
          name_variable(
            var = output * w,
            name = "w_hat",
            type = type
          )
        }
      ))
  }
  
  ## Put all variables together and get the implied volatility
  variables <- ttm_logm %>%
    left_join(layers, by = "type") %>%
    mutate(iv_hat = list(ttm, w_hat, type) %>%
             pmap(function(ttm, w_hat, type) {
               name_variable(
                 var = (w_hat / ttm)^0.5,
                 name = "iv_hat", type = type
               )
             }))
  
  ## Compute the losses
  losses <- variables %>%
    mutate(loss = list(type, ttm, logm, output, w_hat, iv_hat) %>%
             pmap(get_loss, w = w, iv = iv, iv_spread = iv_spread, 
                  prior_model = prior_model, prior = prior)) %>%
    dplyr::select(type, loss)
  
  ## For predictions
  preds <- variables %>%
    left_join(prior_model$w, by = "type") %>%
    dplyr::filter(type == "fit") %>%
    dplyr::select(-type) %>%
    flatten()
  
  ## Returns everything
  model <- list(
    losses = losses,
    variables = variables,
    preds = preds,
    weights = weights,
    prior_model = prior_model,
    controls = controls
  )
  
  return(model)
}

get_name <- function(var, type) {
  str_c(var, type, sep = "_")
}

get_placeholder <- function(name, type) {
  tf$compat$v1$placeholder("float",
                           shape = shape(NULL, 1L),
                           name = get_name(name, type)
  )
}

name_variable <- function(var, name, type) tf$identity(var, name = get_name(name, type))

get_layer <- function(var, type, i, envir, afun, layer_name) {
  
  ## Bias and weight names
  b_name <- paste0("b", i)
  W_name <- paste0("W", i)
  
  ## Compute layer output
  afun(tf$matmul(var, get(W_name, envir = envir)) + get(b_name, envir = envir),
       name = str_c(layer_name, type, sep = "_")
  )
}


get_w_atm <- function(df_atm) {
  
  tfp <- reticulate::import("tensorflow_probability")
  
  ttm_atm <- df_atm %>% pull(ttm)
  w_atm <- df_atm %>% pull(w)
  
  min_ttm <- 0
  max_ttm <- max(ttm_atm)
  ttm_grid <- seq(min_ttm, max_ttm, 1e-2)
  
  if (all(diff(w_atm) >= 0)) {
    ttm_atm <- c(0, ttm_atm)
    w_atm <- c(0, w_atm)
    fit <- spline(ttm_atm, w_atm, xout = ttm_grid)
  } else { ## fallback if data has problems
    require(scam)
    fit <- scam(log(w_atm) ~ s(ttm_atm, k = 10, bs = "mpi"))
    fit$y <- exp(predict(fit, newdata = data.frame(ttm_atm = ttm_grid)))
  }
  
  y <- tf$constant(fit$y, shape = shape(length(fit$y)), dtype = "float32")
  
  function(ttm) {
    tfp$math$interp_regular_1d_grid(
      x = ttm,
      x_ref_min = min_ttm,
      x_ref_max = max_ttm,
      y_ref = y,
      fill_value = "extrapolate"
    )
  }
}

get_prior <- function(ttm_logm, afun, w_atm_fun,
                      prior = "svi",
                      phi_fun = "power_law") {
  if (is.null(w_atm_fun)) {
    stop("A function for the ATM total variance should be provided!")
  }
  
  ## Prepares output
  output <- list(
    prior = prior,
    phi_fun = phi_fun
  )
  
  ## Adds atm variance
  ttm_logm <- ttm_logm %>%
    mutate(w_atm = map(ttm, w_atm_fun))
  
  
  if (prior == "bs") {
    
    ## With Black-Scholes, we are actually only do ATM totvar matching
    ttm_logm <- ttm_logm %>%
      mutate(w_prior = w_atm)
    
    params <- list()
    var_list <- c()
    # sigma_trans <- tf$Variable(tf$zeros(shape = shape(1L, 1L)), name = "sigma_trans")
    # sigma <- tf$nn$softplus(sigma_trans, name = "sigma")
    # 
    # ttm_logm <- ttm_logm %>%
    #   mutate(w_prior = map2(ttm, type, function(ttm, type) {
    #     name_variable(tf$pow(sigma, 2.0) * ttm, name = "w_prior", type = type)
    #   }))
    # 
    # params <- list(sigma = sigma)
    # var_list <- "sigma_trans"
  } else if (prior == "svi") {
    
    ## Rho parameter
    rho_trans <- tf$Variable(tf$zeros(shape = shape(1L, 1L)),
                             name = "rho_trans"
    )
    rho <- tf$tanh(rho_trans, name = "rho")
    
    params <- list(rho = rho, rho_trans = rho_trans)
    var_list <- "rho_trans"
    
    if (phi_fun == "heston_like") { ## Heston-like parametrization
      
      lambda_trans <- tf$Variable(tf$zeros(shape = shape(1L, 1L)),
                                  name = "lambda_trans"
      )
      lambda <- tf$exp(lambda_trans, name = "lambda")
      
      phi_fun <- function(w_atm) {
        1 / (lambda * w_atm) * (1 - (1 - tf$exp(-lambda * w_atm)) / (lambda * w_atm))
      }
      
      params$lambda <- lambda
      params$lambda_trans <- lambda_trans 
      var_list <- c(var_list, "lambda_trans")
    } else if (phi_fun == "power_law") { ## Power-law parameterization
      
      eta_trans <- tf$Variable(tf$zeros(shape = shape(1L, 1L)),
                               name = "eta_trans"
      )
      eta <- tf$exp(eta_trans, name = "eta")
      
      gamma_trans <- tf$Variable(tf$zeros(shape = shape(1L, 1L)),
                                 name = "gamma_trans"
      )
      gamma <- tf$nn$sigmoid(gamma_trans, name = "gamma")
      
      phi_fun <- function(w_atm) {
        eta / (tf$pow(w_atm, gamma) * tf$pow(1 + w_atm, 1 - gamma))
      }
      
      params <- modifyList(
        params,
        list(eta = eta, gamma = gamma, 
             eta_trans = eta_trans, gamma_trans = gamma_trans)
      )
      var_list <- c(var_list, c("eta_trans", "gamma_trans"))
    } else {
      stop("Incorrect function for phi")
    }
    
    ## SSVI total variance
    w_svi <- function(logm, w_atm, phi) {
      w_atm / 2 * (1 + rho * phi * logm +
                     tf$sqrt(tf$square(phi * logm + rho) + 1 - tf$square(rho)))
    }
    
    ttm_logm <- ttm_logm %>%
      mutate(
        phi = map(w_atm, phi_fun),
        phi = map2(phi, type, name_variable, name = "phi"),
        w_prior = pmap(list(logm, w_atm, phi), w_svi),
        w_prior = map2(w_prior, type, name_variable, name = "w_prior"),
        iv_prior = list(ttm, w_prior, type) %>%
          pmap(function(ttm, w_prior, type) {
            name_variable(
              var = (w_prior / ttm)^0.5,
              name = "iv_prior", type = type
            )
          })
      )
  } else {
    stop("prior not implemented!")
  }
  
  output <- modifyList(
    output,
    list(
      w = ttm_logm %>% dplyr::select(-c(ttm, logm)),
      params = params,
      var_list = var_list
    )
  )
  return(output)
}


get_loss <- function(type, ttm, logm, output, w_hat, iv_hat, w, iv,
                     prior_model, prior, pdf, iv_spread) {
  switch(type,
         fit = get_loss_fit(
           w = w, w_hat = w_hat, iv = iv, iv_hat = iv_hat, 
           iv_spread = iv_spread, prior_model = prior_model
         ),
         c4c5 = get_loss_arb(w = w_hat, ttm = ttm, logm = logm),
         c6 = get_loss_arb(w = w_hat, ttm = ttm, logm = logm),
         atm = get_loss_atm(ann_output = output, prior = prior),
         stop("loss type unknown!")
  )
}


get_loss_fit <- function(w, w_hat, iv, iv_hat, iv_spread, prior_model) {
  
  # if (!is.null(prior_model)) {
  #   prior_model <- prior_model$w %>% filter(type == "fit")
  #   w_prior <- prior_model$w_prior[[1]]
  #   iv_prior <- prior_model$iv_prior[[1]]
  # }

  ## total variance error
  l_fit_w_rmse <- name_variable(tf$reduce_mean(1e-6 + (w - w_hat)^2)^0.5,
                                name = "l",
                                type = "fit_w_rmse"
  )
  l_fit_w_mape <- tf$reduce_mean(tf$abs(tf$divide(
    tf$subtract(w_hat, w),
    (w + 1e-6)
  )),
  name = "l_fit_w_mape"
  )
  
  ## implied volatility error
  l_fit_iv_mape <- tf$reduce_mean(tf$abs(tf$divide(
    tf$subtract(iv_hat, iv),
    (iv + 1e-6)
  )),
  name = "l_fit_iv_mape"
  )

  if (is.null(iv_spread)) {
    l_fit_iv_rmse <- name_variable(tf$reduce_mean(1e-6 + (iv - iv_hat)^2)^0.5,
                                   name = "l",
                                   type = "fit_iv_rmse"
    )
    l_fit_iv <- tf$add(l_fit_iv_rmse, l_fit_iv_mape, name = "l_fit_iv")
  } else {
    # l_fit_iv_rmse <- name_variable(tf$reduce_mean(1e-6 + (iv - iv_hat)^2)^0.5,
    #                                name = "l",
    #                                type = "fit_iv_rmse"
    # )
    # l_fit_iv <- tf$add(l_fit_iv_rmse, l_fit_iv_mape, name = "l_fit_iv")
    l_fit_iv_rmse <- name_variable(tf$reduce_mean(1e-6 + tf$divide(tf$abs(iv - iv_hat), 1 + iv_spread)),
                                   name = "l",
                                   type = "fit_iv_rmse"
    )
    l_fit_iv <- name_variable(l_fit_iv_rmse, name = "l", type = "fit_iv")
  }
  
  
  l_fit_w <- tf$add(l_fit_w_rmse, l_fit_w_mape, name = "l_fit_w")
  
  return(list(
    l_fit_w_rmse = l_fit_w_rmse,
    l_fit_w_mape = l_fit_w_mape,
    l_fit_w = l_fit_w,
    l_fit_iv_rmse = l_fit_iv_rmse,
    l_fit_iv_mape = l_fit_iv_mape,
    l_fit_iv = l_fit_iv
  ))
}

get_loss_arb <- function(w, ttm, logm) {
  
  ## Gradients
  dvdt <- tf$gradients(ys = w, xs = ttm)[[1]]
  dvdm <- tf$gradients(ys = w, xs = logm)[[1]]
  d2vdm2 <- tf$gradients(ys = dvdm, xs = logm)[[1]]
  
  ## Calendar arbitrage (C4)
  l_c4 <- tf$reduce_mean(tf$nn$relu(tf$negative(dvdt)), name = "l_c4")
  
  ## Butterfly arbitrage (C5)
  g_k <- name_variable((1 - logm * dvdm / (2 * w))^2 -
                         dvdm^2 / 4 * (1 / w + 1 / 4) + d2vdm2 / 2,
                       name = "g",
                       type = "k"
  )
  l_c5 <- tf$reduce_mean(tf$nn$relu(tf$negative(g_k)), name = "l_c5")
  
  ## Large-moneyness behavior (C6)
  l_c6 <- tf$reduce_mean(tf$abs(d2vdm2), name = "l_c6")
  # l_c6 <- tf$reduce_mean(tf$nn$relu(w / (tf$abs(logm) * ttm) - 2), name = "l_c6")
  
  ## The output
  l_arb <- list(l_c4 = l_c4, l_c5 = l_c5, l_c6 = l_c6,
                g_k = g_k, dvdt = dvdt, dvdm = dvdm, d2vdm2 = d2vdm2)
  
  return(l_arb)
}


get_loss_atm <- function(ann_output, prior) {
  if (is.null(prior)) {
    l_atm <- tf$constant(0.0, name = "l_atm")
  } else if (prior %in% c("svi", "bs")) {
    l_atm <- name_variable(tf$reduce_mean(1e-6 + (ann_output - 1.0)^2)^0.5,
                           name = "l",
                           type = "atm"
    )
  } else {
    stop("unknown prior")
  }
  
  return(list(l_atm = l_atm))
}

get_rnpdf_locvol <- function(preds) {
  
  ## Extract variables
  w <- preds$w_hat
  ttm <- preds$ttm
  logm <- preds$logm
  
  ## Gradients and g_k
  losses <- get_loss_arb(w, ttm, logm)
  dvdt <- losses$dvdt
  dvdm <- losses$dvdm
  d2vdm2 <- losses$d2vdm2
  g_k <- losses$g_k
  
  ## Compute local volatility using Dupire's formula
  locvol <- tf$divide(dvdt, 1 - (logm / w) * dvdm + 0.25 * (
    -0.25 + 1 / w + logm^2 / w^2) * (dvdm)^2 + 0.5 * d2vdm2)
  
  ## Compute the risk-neutral pdf
  d_minus <- (-logm / w^0.5 - 0.5 * w^0.5)
  rnpdf <- tf$exp(-d_minus^2 / 2) * g_k / tf$sqrt(2 * pi * w)
  
  return(list(locvol = locvol, rnpdf = rnpdf))
}

load_trained_variables <- function(m, variables, session) {
  
  ## Load values
  if (!is.null(m$weights[[1]])) {
    walk2(m$weights, variables$weights, function(tensor, value) tensor$load(value, session))
  }
  if (!is.null(m$prior_model)) {
    walk2(m$prior_model$params[m$prior_model$var_list], variables$params, 
          function(tensor, value) tensor$load(matrix(value), session))
  }
  
  return(m)
}

save_trained_variables <- function(m, session) {
  
  ## Save trained variables
  trained_var <- list()
  if (!is.null(m$weights[[1]])) {
    weights <- m$weights
    trained_var$weights_trained <- lapply(weights, session$run)
  }
  if (!is.null(m$prior_model)) {
    params <- m$prior_model$params[m$prior_model$var_list]
    trained_var$params_trained <- lapply(params, session$run)
  }
  
  return(trained_var)
}

get_ivsmoother_dict <- function(df, 
                                types = c("fit", "c4c5", "c6", "atm"),
                                iv_spread = FALSE,
                                ...) {
  args <- list(...)
  if ("ttm" %in% names(args)) {
    ttm <- args$ttm
    args[["ttm"]] <- NULL
  } else {
    ttm <- df$ttm
  }
  
  if ("logm" %in% names(args)) {
    logm <- args$logm
    args[["logm"]] <- NULL
  } else {
    logm <- df$logm
  }
  
  
  ttm_logm <- tibble(
    type = types,
    ttm_logm = map(type, get_ttm_logm,
                   ttm = ttm, logm = logm, args = args
    )
  )
  
  # Rename/flatten/convert for tensorflow
  di <- map2(types, ttm_logm$ttm_logm, function(type, df) {
    ttm_type <- paste(get_name("ttm", type), ":0", sep = "")
    logm_type <- paste(get_name("logm", type), ":0", sep = "")
    df %>%
      rename(
        !!ttm_type := ttm,
        !!logm_type := logm
      )
  }) %>%
    flatten() %>%
    map(as.matrix)
  
  di[["iv:0"]] <- as.matrix(df$iv)
  di[["w:0"]] <- as.matrix(df$w)
  if(iv_spread) {
    di[["iv_spread:0"]] <- as.matrix(df$iv_spread)
  }
  
  return(list(ttm_logm = ttm_logm, di = dict(di)))
}

get_ttm_logm <- function(type, ttm, logm, args) {
  
  # Deal with ttm
  ttm_type <- get_name("ttm", type)
  expand <- TRUE # To know whether expanding is needed
  if (ttm_type %in% names(args)) {
    ttm <- args[[ttm_type]]
  } else {
    ttm <- switch(type,
                  "fit" = ttm,
                  "c4c5" = get_logspace_ttm(max(ttm) + 1),
                  "c6" = unique(ttm),
                  "atm" = unique(ttm)
    )
    if (type == "fit") {
      expand <- FALSE
    }
  }
  
  # Deal with ttm
  logm_type <- get_name("logm", type)
  if (logm_type %in% names(args)) {
    logm <- args[[logm_type]]
  } else {
    logm <- switch(type,
                   "fit" = logm,
                   "c4c5" = get_powerspace_logm(min(logm), max(logm)),
                   "c6" = c(6, 4, 4, 6) * rep(c(min(logm), max(logm)), each = 2),
                   "atm" = 0
    )
  }
  
  # Expand when needed
  if (expand) {
    output <- crossing(ttm = ttm, logm = logm)
  } else {
    output <- tibble(ttm = ttm, logm = logm)
  }
  
  return(output)
}

get_logspace_ttm <- function(ttm_max) {
  seq(log(1/365), log(ttm_max), length.out = 100) %>% exp()
}

get_powerspace_logm <- function(logm_min, logm_max){
  seq(-(-logm_min * 2)**(1/3), (logm_max* 2)**(1/3), length.out = 100)**3
}