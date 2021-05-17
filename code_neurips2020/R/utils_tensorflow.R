## To reset a tf session
reset_tf_session <- function(gpu_mem_frac = 1, seed = 1, verbose = FALSE) {
  
  if (exists("sess")) {
    sess$close()
  }

  gpu_options <- tf$compat$v1$GPUOptions(per_process_gpu_memory_fraction = gpu_mem_frac)
  tf$compat$v1$reset_default_graph()
  sess <- tf$compat$v1$Session(config = tf$compat$v1$ConfigProto(gpu_options = gpu_options))
  tf$compat$v1$set_random_seed(seed)
  tf$compat$v1$disable_eager_execution()
  
  return(sess)
}

## To constrain the fraction of gpu memory used by a tensorflow session (for
## keras)
contrain_gpu_memory <- function(memory_fraction) {
  gpu_options <- tf$GPUOptions(per_process_gpu_memory_fraction = memory_fraction)
  config <- tf$ConfigProto(gpu_options = gpu_options)
  k_set_session(tf$Session(config = config))
}

## A progress bar that can be used in purrr::map and purrr:map_*
spawn_progressbar <- function(x, .name = .pb, .times = 1) {
  .name <- substitute(.name)
  n <- nrow(x) * .times
  eval(substitute(.name <<- dplyr::progress_estimated(n)))
  x
}
