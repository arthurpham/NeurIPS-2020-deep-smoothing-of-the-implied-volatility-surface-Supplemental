"""
TensorFlow session management utilities.

Converted from R/utils_tensorflow.R
"""

import tensorflow as tf


def reset_tf_session(gpu_mem_frac: float = 1.0, seed: int = 1, verbose: bool = False):
    """
    Reset TensorFlow session for reproducible training.

    Uses TF1 compatibility mode for numerical equivalence with R code.

    Parameters
    ----------
    gpu_mem_frac : float
        Fraction of GPU memory to use (0.0 to 1.0)
    seed : int
        Random seed for reproducibility
    verbose : bool
        Whether to print session info

    Returns
    -------
    tf.compat.v1.Session
        Configured TensorFlow session
    """
    # Disable eager execution for TF1-style graph mode
    tf.compat.v1.disable_eager_execution()

    # Reset the default graph
    tf.compat.v1.reset_default_graph()

    # Set random seed before graph construction
    tf.compat.v1.set_random_seed(seed)

    # Configure GPU options
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=gpu_mem_frac)
    config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)

    # Allow memory growth to avoid OOM errors
    config.gpu_options.allow_growth = True

    # Create session
    sess = tf.compat.v1.Session(config=config)

    if verbose:
        print(f"TensorFlow session created with GPU memory fraction: {gpu_mem_frac}")
        print(f"Random seed: {seed}")
        print(f"TensorFlow version: {tf.__version__}")

    return sess


def get_trainable_variable_count(session):
    """
    Get the total count of trainable parameters.

    Parameters
    ----------
    session : tf.compat.v1.Session
        TensorFlow session

    Returns
    -------
    int
        Total number of trainable parameters
    """
    total = 0
    for var in tf.compat.v1.trainable_variables():
        shape = session.run(var).shape
        total += int(np.prod(shape))
    return total
