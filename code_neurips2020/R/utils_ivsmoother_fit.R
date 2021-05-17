## Controls used for the fits
get_fit_controls <- function(train_ini = TRUE,
                             var_ini_min = -0.25,
                             var_ini_max = 0.25,
                             var_toload = NULL,
                             iter_max = 4e3,
                             var_pred = "iv",
                             penalty = c(fit = 1, c4 = 10, c5 = 10, c6 = 10, atm = 0.1),
                             learning_rate = 0.01,
                             tol_abs = 25e-4,
                             tol_rel = 1e-2,
                             patience = 5e2,
                             n_restart = 4,
                             gpu_mem_frac = 1,
                             tf_seed = 1,
                             verbose = TRUE) {
  controls <- list(
    train_ini = train_ini,
    var_toload = var_toload,
    iter_max = iter_max,
    var_pred = var_pred,
    penalty = penalty,
    learning_rate = learning_rate,
    init = list(min = var_ini_min, max = var_ini_max),
    tol_abs = tol_abs,
    tol_rel = tol_rel,
    patience = patience,
    n_restart = n_restart,
    verbose = verbose,
    gpu_mem_frac = gpu_mem_frac
  )

  return(controls)
}


init_and_train <- function(model,
                           feed_dict,
                           session,
                           controls = get_fit_controls(),
                           var_list = NULL) {
  ## Parameters
  train_ini <- controls$train_ini
  var_toload <- controls$var_toload
  iter_max <- controls$iter_max
  tol_abs <- controls$tol_abs
  tol_rel <- controls$tol_rel
  patience <- controls$patience
  learning_rate <- tf$constant(controls$learning_rate)
  lr_stdout <- controls$learning_rate
  n_restart <- controls$n_restart
  verbose <- controls$verbose

  ## Loss function
  loss <- get_total_loss(model, controls)

  ## Initialization and optimizer
  if (train_ini) {
    generator <- tf$compat$v1$train$AdamOptimizer(learning_rate = controls$learning_rate)
    optimizer <- generator$minimize(loss$total, var_list)
    init <- tf$compat$v1$global_variables_initializer()
    session$run(init)
    controls[["init"]] <- init
    controls[["generator"]] <- generator
    controls[["optimizer"]] <- optimizer
  } else {
    if (is.null(controls$generator)) {
      stop("the generator is missing")
    }
    generator <- controls$generator
    optimizer <- generator$minimize(loss$total, var_list)
    controls[["optimizer"]] <- optimizer
  }
  
  ## Load variables (warm start)
  if (!is.null(var_toload)) {
    load_trained_variables(model, var_toload, session)
  }

  ## GD optim
  last_cost <- best_cost <- session$run(loss$total, feed_dict = feed_dict)
  iter <- 0
  if (verbose) {
    make_verbose <- function() {
      loss_fit <- session$run(loss$fit, feed_dict = feed_dict)
      loss_arb <- session$run(loss$arb, feed_dict = feed_dict)
      loss_atm <- session$run(loss$atm, feed_dict = feed_dict)
      # map_dbl(loss, sess$run, feed_dict = feed_dict)
      cat(
        "iter =", iter,
        "loss =", last_cost,
        "loss fit = ", loss_fit,
        "loss arb = ", loss_arb,
        "loss atm = ", loss_atm,
        "learning rate =", lr_stdout, "\n"
      )
    }
    make_verbose()
  }
  rel_cost <- 1
  cost_history <- c()
  learning_rate_history <- lr_stdout
  best_iter <- iter <- counter <- 1
  conv <- "iter_max"
  best_var <- save_trained_variables(model, session)

  while (iter <= iter_max) {
    session$run(optimizer, feed_dict = feed_dict)

    current_cost <- session$run(loss$total, feed_dict = feed_dict)
    if (is.nan(current_cost)) {
      break
    }
    cost_history <- c(cost_history, current_cost)
    rel_cost <- c(rel_cost, log(current_cost / best_cost))

    # Check if we improve
    if (iter > patience && current_cost < best_cost) {
      best_iter <- iter
      best_cost <- current_cost
      best_var <- save_trained_variables(model, session)
    }

    # Early stopping
    if (current_cost < tol_abs) {
      conv <- "tol_abs"
      break
    }
    if (iter > 4 * patience &&
      all(tail(rel_cost, 4 * patience) > -tol_rel)) {
      conv <- "tol_rel"
      break
    }
    # Learning rate scheduler
    if (counter > patience &&
      all(tail(rel_cost, patience) > -tol_rel)) {
      learning_rate <- learning_rate * 0.5
      lr_stdout <- lr_stdout * 0.5
      counter <- 1
    }
    learning_rate_history <- c(learning_rate_history, lr_stdout)
    last_cost <- current_cost

    if (verbose & ((iter == 1) || ((iter %% 500) == 0))) {
      make_verbose()
    }
    iter <- iter + 1
    counter <- counter + 1
  }

  ## Use the best parameters
  load_trained_variables(model, best_var, session)
  if (verbose) {
    make_verbose()
  }

  if (conv != "tol_abs" & n_restart > 0 & (iter_max - iter - patience) > 0) {
    controls_restart <- controls
    controls_restart$n_restart <- n_restart - 1
    controls_restart$train_ini <- FALSE
    controls_restart$var_toload <- NULL
    controls_restart$iter_max <- iter_max - iter
    output_restart <- init_and_train(model, feed_dict, session, controls_restart)

    output <- list(
      metrics = list(
        cost_history = c(cost_history, output_restart$metrics$history),
        learning_rate_history = c(
          learning_rate_history,
          output_restart$metrics$learning_rate_history
        )
      ),
      n_restart = 1 + output_restart$n_restart,
      best_cost = output_restart$best_cost,
      best_iter = iter + output_restart$best_iter,
      conv = output_restart$conv,
      controls = controls,
      loss = loss
    )
  } else{
    output <- list(
      metrics = list(
        cost_history = cost_history,
        learning_rate_history = learning_rate_history
      ),
      n_restart = 0,
      best_cost = best_cost,
      best_iter = best_iter,
      conv = conv,
      controls = controls,
      loss = loss
    )
  }

  return(output)
}

get_total_loss <- function(model, controls) {

  ## penalty
  losses <- deframe(model$losses)
  penalty <- controls$penalty

  ## losses
  if (controls$var_pred == "w") {
    l_fit <- losses$fit$l_fit_w
  } else if (controls$var_pred == "iv") {
    l_fit <- losses$fit$l_fit_iv
  } else {
    stop("unknown var_pred")
  }

  ## losses
  l_c4 <- losses$c4c5$l_c4 + losses$c6$l_c4
  l_c5 <- losses$c4c5$l_c5 + losses$c6$l_c5
  l_c6 <- losses$c6$l_c6
  l_atm <- losses$atm$l_atm
  l_arb <- l_c4 + l_c5 + l_c6

  ## Compute total loss
  l_total <- penalty[1] * l_fit + penalty[2] * l_c4 +
    penalty[3] * l_c5 + penalty[4] * l_c6 + penalty[5] * l_atm

  return(list(
    total = l_total,
    fit = l_fit,
    arb = l_arb,
    c4 = l_c4,
    c5 = l_c5,
    c6 = l_c6,
    atm = l_atm
  ))
}

# get_fit <- function(session, model, df) {
#   y <- model$preds$w_hat
#   preds <- session$run(y, feed_dict = make_dict_sc(df))
#   df_fit <- df %>%
#     mutate(call = NA, put = NA) %>%
#     mutate(w = preds) %>%
#     mutate(iv = (w / ttm)^0.5) %>%
#     select(-call, -put) %>%
#     mutate(name = "fit")
# 
#   return(df_fit)
# }
# 
# get_losses <- function(session, model, df) {
#   res <- map_dbl(
#     model$losses %>% unlist(),
#     function(loss) session$run(loss, feed_dict = make_dict_sc(df))
#   )
#   names(res) <- model$losses %>%
#     unlist() %>%
#     names() %>%
#     map_chr(function(x) {
#       split <- strsplit(x, "[.]")[[1]]
#       if (length(split) == 1) {
#         return(split)
#       } else {
#         return(split[2])
#       }
#     })
#   return(res)
# }
# 
# get_n_param <- function(session) {
#   tf$compat$v1$trainable_variables() %>%
#     sapply(function(X) {
#       X %>%
#         session$run() %>%
#         dim() %>%
#         prod()
#     }) %>%
#     sum()
# }
