#!/usr/bin/env python3 -u
"""
Example script for Deep Smoothing IV surface model.

Equivalent to R-ex/run_example.R from the original R implementation.

Usage:
    cd code_python
    python examples/run_example.py
"""

import os
import sys

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*NotOpenSSLWarning.*')
warnings.filterwarnings('ignore', category=DeprecationWarning)

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from deep_smoothing.tf_utils import reset_tf_session
from deep_smoothing.models import (
    IVSmootherControls,
    FitControls,
    get_ivsmoother,
    get_w_atm,
    get_ivsmoother_dict,
)
from deep_smoothing.training import init_and_train
from deep_smoothing.plotting import plot_fit, plot_totvar
from deep_smoothing.data_utils import load_train_data, get_atm_data


def main():
    """Run the Deep Smoothing example."""
    print("=" * 60)
    print("Deep Smoothing: Neural Network for IV Surface Modeling")
    print("Python Implementation")
    print("=" * 60)

    # Change to script directory to find data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.dirname(script_dir))
    print(f"\nWorking directory: {os.getcwd()}")

    # Initialize TensorFlow session
    print("\n--- Initializing TensorFlow Session ---")
    sess = reset_tf_session(gpu_mem_frac=0.1, seed=1, verbose=True)

    # Load training data
    print("\n--- Loading Training Data ---")
    df_train = load_train_data("data/train_data.csv")
    print(f"Loaded {len(df_train)} observations")
    print(f"TTM range: [{df_train['ttm'].min():.4f}, {df_train['ttm'].max():.4f}]")
    print(f"LogM range: [{df_train['logm'].min():.4f}, {df_train['logm'].max():.4f}]")

    # Visual check: plot total variance
    print("\n--- Creating Initial Plot ---")
    df_train["name"] = "Data"
    df_train["train"] = True

    # Get ATM data for the w_atm interpolation function
    print("\n--- Computing ATM Total Variance Function ---")
    df_atm = get_atm_data(df_train)
    print(f"ATM data points: {len(df_atm)}")
    w_atm_fun = get_w_atm(df_atm)

    # Create model controls
    print("\n--- Creating Model ---")
    ivsmoother_controls = IVSmootherControls(
        neurons_vec=[40, 40, 40, 40],
        activation="softplus",
        prior="svi",
        phi_fun="power_law",
        w_atm_fun=w_atm_fun,
    )
    print(f"Neurons: {ivsmoother_controls.neurons_vec}")
    print(f"Prior: {ivsmoother_controls.prior}")
    print(f"Phi function: {ivsmoother_controls.phi_fun}")

    # Create fit controls
    fit_controls = FitControls(
        iter_max=4000,
        learning_rate=0.01,
        penalty=(1.0, 10.0, 10.0, 10.0, 0.1),
        patience=500,
        n_restart=4,
        tol_abs=0.0025,
        verbose=True,
    )
    print(f"\nPenalties: fit={fit_controls.penalty[0]}, c4={fit_controls.penalty[1]}, "
          f"c5={fit_controls.penalty[2]}, c6={fit_controls.penalty[3]}, "
          f"atm={fit_controls.penalty[4]}")

    # Build the model
    model = get_ivsmoother(ivsmoother_controls)

    # Create feed dictionary
    print("\n--- Creating Feed Dictionary ---")
    di_train = get_ivsmoother_dict(df_train)["di"]
    print(f"Feed dictionary keys: {list(di_train.keys())}")

    # Train the model
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)

    import time
    start_time = time.time()
    output = init_and_train(model, di_train, sess, fit_controls)
    elapsed = time.time() - start_time

    print(f"\n--- Training Complete ---")
    print(f"Time elapsed: {elapsed:.2f} seconds")
    print(f"Best iteration: {output['best_iter']}")
    print(f"Best loss: {output['best_cost']:.6f}")
    print(f"Convergence: {output['conv']}")
    print(f"Number of restarts: {output['n_restart']}")

    # Generate plots
    print("\n--- Generating Plots ---")

    # Plot similar to the paper (scenario 12)
    fig = plot_fit(df_train, model, sess, figsize=(14, 14))

    # Add title with model info
    prior_name = model["prior_model"]["prior"].upper() if model["prior_model"] else "None"
    fig.suptitle(
        f"Prior = {prior_name}, Lambda = {fit_controls.penalty[1]}",
        y=1.02,
        fontsize=14,
    )

    # Save and show plot
    plt.savefig("examples/output_plot.png", dpi=150, bbox_inches="tight")
    print("Saved plot to examples/output_plot.png")

    # Plot loss history
    fig_loss, ax = plt.subplots(figsize=(10, 4))
    ax.plot(output["metrics"]["cost_history"], linewidth=0.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Total Loss")
    ax.set_title("Training Loss History")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    plt.savefig("examples/loss_history.png", dpi=150, bbox_inches="tight")
    print("Saved loss history to examples/loss_history.png")

    # Show predictions summary
    print("\n--- Predictions Summary ---")
    w_hat = sess.run(model["preds"]["w_hat"], feed_dict=di_train).flatten()
    iv_hat = sess.run(model["preds"]["iv_hat"], feed_dict=di_train).flatten()

    w_actual = df_train["w"].values
    iv_actual = df_train["iv"].values

    w_rmse = np.sqrt(np.mean((w_actual - w_hat) ** 2))
    iv_rmse = np.sqrt(np.mean((iv_actual - iv_hat) ** 2))
    w_mape = np.mean(np.abs((w_actual - w_hat) / (w_actual + 1e-6))) * 100
    iv_mape = np.mean(np.abs((iv_actual - iv_hat) / (iv_actual + 1e-6))) * 100

    print(f"Total Variance - RMSE: {w_rmse:.6f}, MAPE: {w_mape:.2f}%")
    print(f"Implied Vol    - RMSE: {iv_rmse:.6f}, MAPE: {iv_mape:.2f}%")

    # Export fitted values
    print("\n--- Exporting Fitted Values ---")
    df_output = df_train.copy()
    df_output["w_hat"] = w_hat
    df_output["iv_hat"] = iv_hat

    w_prior = sess.run(model["preds"]["w_prior"], feed_dict=di_train).flatten()
    df_output["w_prior"] = w_prior
    df_output["iv_prior"] = np.sqrt(w_prior / df_output["ttm"])

    df_output.to_csv("examples/fitted_values.csv", index=False)
    print("Saved fitted values to examples/fitted_values.csv")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)

    # Show plots (only if running interactively)
    try:
        plt.show()
    except Exception:
        pass

    return model, output, df_train


if __name__ == "__main__":
    model, output, df_train = main()
