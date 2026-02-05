"""
Training loop with Adam optimizer, early stopping, and multi-restart.

Converted from R/utils_ivsmoother_fit.R
"""

from typing import Dict, Any, Optional, List
import numpy as np
import tensorflow as tf

from .models import FitControls, load_trained_variables, save_trained_variables


def get_total_loss(model: Dict, controls: FitControls) -> Dict[str, tf.Tensor]:
    """
    Compute the total weighted loss.

    Parameters
    ----------
    model : Dict
        Model dictionary with losses
    controls : FitControls
        Fit controls with penalty weights

    Returns
    -------
    Dict
        Dictionary with total, fit, arb, c4, c5, c6, atm losses
    """
    losses = model["losses"]
    penalty = controls.penalty

    # Select fit loss based on var_pred
    if controls.var_pred == "w":
        l_fit = losses["fit"]["l_fit_w"]
    elif controls.var_pred == "iv":
        l_fit = losses["fit"]["l_fit_iv"]
    else:
        raise ValueError(f"Unknown var_pred: {controls.var_pred}")

    # Arbitrage losses
    l_c4 = losses["c4c5"]["l_c4"] + losses["c6"]["l_c4"]
    l_c5 = losses["c4c5"]["l_c5"] + losses["c6"]["l_c5"]
    l_c6 = losses["c6"]["l_c6"]
    l_atm = losses["atm"]["l_atm"]
    l_arb = l_c4 + l_c5 + l_c6

    # Total loss with penalties
    l_total = (
        penalty[0] * l_fit
        + penalty[1] * l_c4
        + penalty[2] * l_c5
        + penalty[3] * l_c6
        + penalty[4] * l_atm
    )

    return {
        "total": l_total,
        "fit": l_fit,
        "arb": l_arb,
        "c4": l_c4,
        "c5": l_c5,
        "c6": l_c6,
        "atm": l_atm,
    }


def init_and_train(
    model: Dict,
    feed_dict: Dict,
    session: tf.compat.v1.Session,
    controls: FitControls = None,
    var_list: Optional[List] = None,
) -> Dict[str, Any]:
    """
    Initialize and train the model with Adam optimizer.

    Parameters
    ----------
    model : Dict
        Model dictionary
    feed_dict : Dict
        Feed dictionary for training data
    session : tf.compat.v1.Session
        TensorFlow session
    controls : FitControls
        Training controls
    var_list : List or None
        Variables to optimize (None = all trainable)

    Returns
    -------
    Dict
        Training output with metrics, best_cost, best_iter, conv, controls, loss
    """
    if controls is None:
        controls = FitControls()

    # Extract parameters
    train_ini = controls.train_ini
    var_toload = controls.var_toload
    iter_max = controls.iter_max
    tol_abs = controls.tol_abs
    tol_rel = controls.tol_rel
    patience = controls.patience
    n_restart = controls.n_restart
    verbose = controls.verbose

    # Compute total loss
    loss = get_total_loss(model, controls)

    # Learning rate variable for scheduling
    lr_value = controls.learning_rate

    # Initialize or reuse optimizer
    if train_ini:
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=controls.learning_rate)
        train_op = optimizer.minimize(loss["total"], var_list=var_list)
        init = tf.compat.v1.global_variables_initializer()
        session.run(init)
        controls.optimizer = optimizer
        controls.train_op = train_op
    else:
        if not hasattr(controls, "optimizer"):
            raise ValueError("Optimizer is missing for warm restart")
        optimizer = controls.optimizer
        train_op = optimizer.minimize(loss["total"], var_list=var_list)
        controls.train_op = train_op

    # Load variables (warm start)
    if var_toload is not None:
        load_trained_variables(model, var_toload, session)

    # Verbose logging function
    def make_verbose(iter_num, lr):
        loss_total = session.run(loss["total"], feed_dict=feed_dict)
        loss_fit = session.run(loss["fit"], feed_dict=feed_dict)
        loss_arb = session.run(loss["arb"], feed_dict=feed_dict)
        loss_atm = session.run(loss["atm"], feed_dict=feed_dict)
        print(
            f"iter = {iter_num:5d}  "
            f"loss = {loss_total:.6f}  "
            f"loss fit = {loss_fit:.6f}  "
            f"loss arb = {loss_arb:.6f}  "
            f"loss atm = {loss_atm:.6f}  "
            f"learning rate = {lr:.6f}"
        )

    # Initial state
    best_cost = session.run(loss["total"], feed_dict=feed_dict)
    last_cost = best_cost

    if verbose:
        make_verbose(0, lr_value)

    rel_cost = [1.0]
    cost_history = []
    learning_rate_history = [lr_value]
    best_iter = 0
    iter_num = 1
    counter = 1
    conv = "iter_max"
    best_var = save_trained_variables(model, session)

    # Training loop
    while iter_num <= iter_max:
        session.run(train_op, feed_dict=feed_dict)

        current_cost = session.run(loss["total"], feed_dict=feed_dict)

        if np.isnan(current_cost):
            conv = "nan"
            break

        cost_history.append(current_cost)
        rel_cost.append(np.log(current_cost / best_cost) if best_cost > 0 else 0)

        # Check if we improve (after patience iterations)
        if iter_num > patience and current_cost < best_cost:
            best_iter = iter_num
            best_cost = current_cost
            best_var = save_trained_variables(model, session)

        # Early stopping: absolute tolerance
        if current_cost < tol_abs:
            conv = "tol_abs"
            break

        # Early stopping: relative tolerance (no improvement for 4*patience)
        if iter_num > 4 * patience:
            if all(r > -tol_rel for r in rel_cost[-4 * patience :]):
                conv = "tol_rel"
                break

        # Learning rate scheduler: decay by 0.5 if no improvement for patience iters
        if counter > patience:
            if all(r > -tol_rel for r in rel_cost[-patience :]):
                lr_value *= 0.5
                counter = 1

        learning_rate_history.append(lr_value)
        last_cost = current_cost

        if verbose and (iter_num == 1 or iter_num % 500 == 0):
            make_verbose(iter_num, lr_value)

        iter_num += 1
        counter += 1

    # Load best parameters
    load_trained_variables(model, best_var, session)

    if verbose:
        print("--- Best model loaded ---")
        make_verbose(best_iter, lr_value)

    # Multi-restart if not converged
    if conv != "tol_abs" and n_restart > 0 and (iter_max - iter_num - patience) > 0:
        # Create new controls for restart
        controls_restart = FitControls(
            train_ini=False,
            var_toload=None,
            iter_max=iter_max - iter_num,
            var_pred=controls.var_pred,
            penalty=controls.penalty,
            learning_rate=lr_value,  # Use current (possibly decayed) LR
            tol_abs=tol_abs,
            tol_rel=tol_rel,
            patience=patience,
            n_restart=n_restart - 1,
            verbose=verbose,
        )
        controls_restart.optimizer = controls.optimizer

        if verbose:
            print(f"\n=== Restart {controls.n_restart - n_restart + 1} ===")

        output_restart = init_and_train(
            model, feed_dict, session, controls_restart, var_list
        )

        output = {
            "metrics": {
                "cost_history": cost_history + output_restart["metrics"]["cost_history"],
                "learning_rate_history": learning_rate_history
                + output_restart["metrics"]["learning_rate_history"],
            },
            "n_restart": 1 + output_restart["n_restart"],
            "best_cost": output_restart["best_cost"],
            "best_iter": iter_num + output_restart["best_iter"],
            "conv": output_restart["conv"],
            "controls": controls,
            "loss": loss,
        }
    else:
        output = {
            "metrics": {
                "cost_history": cost_history,
                "learning_rate_history": learning_rate_history,
            },
            "n_restart": 0,
            "best_cost": best_cost,
            "best_iter": best_iter,
            "conv": conv,
            "controls": controls,
            "loss": loss,
        }

    return output
