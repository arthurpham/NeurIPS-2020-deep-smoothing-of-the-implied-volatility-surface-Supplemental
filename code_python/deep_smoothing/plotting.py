"""
Visualization utilities for IV surface plots.

Converted from R/utils_plot.R
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
import tensorflow as tf

from .models import get_ivsmoother_dict


def get_fit(
    df: pd.DataFrame, model: Dict, session: tf.compat.v1.Session
) -> pd.DataFrame:
    """
    Get fitted values from the model.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with ttm, logm, w, iv columns
    model : Dict
        Model dictionary
    session : tf.compat.v1.Session
        TensorFlow session

    Returns
    -------
    pd.DataFrame
        DataFrame with data, prior, and model predictions
    """
    di = get_ivsmoother_dict(df)["di"]

    # Get predictions
    w_prior = session.run(model["preds"]["w_prior"], feed_dict=di).flatten()
    w_hat = session.run(model["preds"]["w_hat"], feed_dict=di).flatten()

    # Get original name or default
    original_name = df["name"].iloc[0] if "name" in df.columns else "Data"
    original_name = str(original_name).title()

    # Create long-format DataFrame
    n = len(df)
    result_rows = []

    for i in range(n):
        row_base = {
            "ttm": round(df["ttm"].iloc[i], 2),
            "logm": df["logm"].iloc[i],
            "m": df["m"].iloc[i] if "m" in df.columns else np.exp(df["logm"].iloc[i]),
            "quote_date": df["quote_date"].iloc[i] if "quote_date" in df.columns else None,
            "expiry": df["expiry"].iloc[i] if "expiry" in df.columns else None,
            "call": np.nan,
            "put": np.nan,
        }

        # Original data
        row_data = row_base.copy()
        row_data["w"] = df["w"].iloc[i]
        row_data["iv"] = df["iv"].iloc[i]
        row_data["name"] = original_name
        row_data["train"] = df["train"].iloc[i] if "train" in df.columns else True
        result_rows.append(row_data)

        # Prior prediction
        row_prior = row_base.copy()
        row_prior["w"] = w_prior[i]
        row_prior["iv"] = np.sqrt(w_prior[i] / row_base["ttm"]) if row_base["ttm"] > 0 else np.nan
        row_prior["name"] = "Prior"
        row_prior["train"] = False
        result_rows.append(row_prior)

        # Model prediction
        row_model = row_base.copy()
        row_model["w"] = w_hat[i]
        row_model["iv"] = np.sqrt(w_hat[i] / row_base["ttm"]) if row_base["ttm"] > 0 else np.nan
        row_model["name"] = "Prior x NN Model"
        row_model["train"] = False
        result_rows.append(row_model)

    df_fit = pd.DataFrame(result_rows)

    # Set name as categorical with proper order
    df_fit["name"] = pd.Categorical(
        df_fit["name"],
        categories=[original_name, "Prior", "Prior x NN Model"],
        ordered=True,
    )

    return df_fit


def plot_fit(
    df: pd.DataFrame,
    model: Dict,
    session: tf.compat.v1.Session,
    figsize: Tuple[int, int] = (14, 14),
) -> plt.Figure:
    """
    Create 3-panel plot comparing data and model predictions.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with ttm, logm, w, iv columns
    model : Dict
        Model dictionary
    session : tf.compat.v1.Session
        TensorFlow session
    figsize : Tuple[int, int]
        Figure size

    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    df_fit = get_fit(df, model, session)

    # Get unique TTMs for slice selection
    ttm_unique = df_fit["ttm"].unique()
    target_ttms = [1 / 12, 2 / 12, 1, 2]
    sel_slices = [ttm_unique[np.argmin(np.abs(ttm_unique - t))] for t in target_ttms]

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0.3)

    # Panel 1: Total Variance vs logm, faceted by series
    ax1_container = fig.add_subplot(gs[0])
    ax1_container.set_visible(False)

    names = df_fit["name"].cat.categories.tolist()
    axes1 = []
    for idx, name in enumerate(names):
        ax = fig.add_subplot(gs[0], position=[0.05 + idx * 0.31, 0.68, 0.28, 0.28])
        axes1.append(ax)
        df_subset = df_fit[df_fit["name"] == name]

        # Color by ttm
        ttm_values = df_subset["ttm"].unique()
        colors = cm.viridis(np.linspace(0, 1, len(ttm_values)))
        ttm_to_color = dict(zip(sorted(ttm_values), colors))

        for ttm_val in sorted(ttm_values):
            df_ttm = df_subset[df_subset["ttm"] == ttm_val].sort_values("logm")
            ax.plot(df_ttm["logm"], df_ttm["w"], color=ttm_to_color[ttm_val], linewidth=0.8)

        # Overlay training points for Data
        if name == names[0]:
            df_train = df_subset[df_subset["train"] == True]
            if len(df_train) > 0:
                for ttm_val in sorted(df_train["ttm"].unique()):
                    df_ttm = df_train[df_train["ttm"] == ttm_val]
                    ax.scatter(
                        df_ttm["logm"],
                        df_ttm["w"],
                        color=ttm_to_color.get(ttm_val, "gray"),
                        s=8,
                        alpha=0.7,
                    )

        ax.set_xlabel("Log-Moneyness")
        ax.set_ylabel("Total Variance")
        ax.set_title(name)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Data vs Predictions: Total Variance", y=0.98, fontsize=12)

    # Panel 2: IV vs logm, faceted by TTM (4 selected slices)
    df_slices = df_fit[df_fit["ttm"].isin(sel_slices)]
    ttm_labels = {
        sel_slices[0]: "1 month",
        sel_slices[1]: "2 months",
        sel_slices[2]: "1 year",
        sel_slices[3]: "2 years",
    }

    for idx, ttm_val in enumerate(sel_slices):
        ax = fig.add_subplot(1, 4, idx + 1, position=[0.05 + idx * 0.24, 0.38, 0.20, 0.22])
        df_ttm = df_slices[df_slices["ttm"] == ttm_val]

        colors_series = {"Data": "C0", names[0]: "C0", "Prior": "C1", "Prior x NN Model": "C2"}
        for name in names:
            df_name = df_ttm[df_ttm["name"] == name].sort_values("logm")
            color = colors_series.get(name, "gray")
            ax.plot(df_name["logm"], df_name["iv"], label=name, color=color, linewidth=1.2)

        # Training points
        df_train = df_ttm[(df_ttm["name"] == names[0]) & (df_ttm["train"] == True)]
        if len(df_train) > 0:
            ax.scatter(df_train["logm"], df_train["iv"], color="C0", s=12, zorder=5)

        ax.set_xlabel("Log-Moneyness")
        ax.set_ylabel("Implied Volatility")
        ax.set_title(f"TTM = {ttm_labels.get(ttm_val, str(ttm_val))}")
        ax.grid(True, alpha=0.3)

        if idx == 0:
            ax.legend(loc="upper right", fontsize=8)

    # Panel 3: Extended grid predictions
    di_info = get_ivsmoother_dict(df)
    df_extended = pd.DataFrame(di_info["ttm_logm"]["c4c5"])
    df_extended["m"] = np.exp(df_extended["logm"])
    df_extended["w"] = 0
    df_extended["iv"] = 0
    df_extended["expiry"] = df_extended["ttm"]

    di_ext = get_ivsmoother_dict(df_extended)["di"]

    w_prior_ext = session.run(model["preds"]["w_prior"], feed_dict=di_ext).flatten()
    w_hat_ext = session.run(model["preds"]["w_hat"], feed_dict=di_ext).flatten()
    output_ext = session.run(model["preds"]["output"], feed_dict=di_ext).flatten()

    # Get scale for normalization
    scale = session.run(model["prior_model"]["params"]["scale"], feed_dict=di_ext)
    ann_output_ext = output_ext / (2 * float(scale))

    # Build extended DataFrame
    n_ext = len(df_extended)
    result_ext = []

    for i in range(n_ext):
        row_base = {
            "ttm": df_extended["ttm"].iloc[i],
            "logm": df_extended["logm"].iloc[i],
            "m": df_extended["m"].iloc[i],
        }

        # Prior
        row_prior = row_base.copy()
        row_prior["w"] = w_prior_ext[i]
        row_prior["iv"] = np.sqrt(w_prior_ext[i] / row_base["ttm"]) if row_base["ttm"] > 0 else np.nan
        row_prior["name"] = "Prior"
        result_ext.append(row_prior)

        # Scaled NN
        row_nn = row_base.copy()
        row_nn["w"] = ann_output_ext[i]
        row_nn["iv"] = np.nan  # Not meaningful
        row_nn["name"] = "Scaled NN Model"
        result_ext.append(row_nn)

        # Model
        row_model = row_base.copy()
        row_model["w"] = w_hat_ext[i]
        row_model["iv"] = np.sqrt(w_hat_ext[i] / row_base["ttm"]) if row_base["ttm"] > 0 else np.nan
        row_model["name"] = "Prior x NN Model"
        result_ext.append(row_model)

    df_ext = pd.DataFrame(result_ext)
    df_ext["name"] = pd.Categorical(
        df_ext["name"],
        categories=["Prior", "Scaled NN Model", "Prior x NN Model"],
        ordered=True,
    )

    ext_names = ["Prior", "Scaled NN Model", "Prior x NN Model"]
    for idx, name in enumerate(ext_names):
        ax = fig.add_subplot(1, 3, idx + 1, position=[0.05 + idx * 0.31, 0.05, 0.28, 0.25])
        df_name = df_ext[df_ext["name"] == name]

        ttm_values = sorted(df_name["ttm"].unique())
        colors = cm.viridis(np.linspace(0, 1, len(ttm_values)))
        ttm_to_color = dict(zip(ttm_values, colors))

        for ttm_val in ttm_values:
            df_ttm = df_name[df_name["ttm"] == ttm_val].sort_values("logm")
            ax.plot(df_ttm["logm"], df_ttm["w"], color=ttm_to_color[ttm_val], linewidth=0.5)

        ax.set_xlabel("Log-Moneyness")
        ax.set_ylabel("Total Variance")
        ax.set_title(name)
        ax.grid(True, alpha=0.3)

    plt.figtext(0.5, 0.32, "Predictions on an Extended Grid", ha="center", fontsize=12)

    return fig


def plot_totvar(
    df: pd.DataFrame,
    title: str = "",
    maturity: str = "ttm",
    y_label: str = "Total Variance",
    train: bool = True,
    figsize: Tuple[int, int] = (12, 4),
) -> plt.Figure:
    """
    Plot total variance vs log-moneyness, faceted by series.

    Parameters
    ----------
    df : pd.DataFrame
        Data with ttm, logm, w, name columns
    title : str
        Plot title
    maturity : str
        Column to use for maturity coloring
    y_label : str
        Y-axis label
    train : bool
        Whether to show training points
    figsize : Tuple[int, int]
        Figure size

    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, df["name"].nunique(), figsize=figsize, sharey=True)
    if df["name"].nunique() == 1:
        axes = [axes]

    names = df["name"].unique()
    maturity_values = sorted(df[maturity].unique())
    colors = cm.viridis(np.linspace(0, 1, len(maturity_values)))
    mat_to_color = dict(zip(maturity_values, colors))

    for ax, name in zip(axes, names):
        df_name = df[df["name"] == name]

        for mat_val in maturity_values:
            df_mat = df_name[df_name[maturity] == mat_val].sort_values("logm")
            ax.plot(df_mat["logm"], df_mat["w"], color=mat_to_color[mat_val], linewidth=0.8)

        if train and "train" in df.columns:
            df_train = df_name[df_name["train"] == True]
            for mat_val in df_train[maturity].unique():
                df_mat = df_train[df_train[maturity] == mat_val]
                ax.scatter(df_mat["logm"], df_mat["w"], color=mat_to_color.get(mat_val, "gray"), s=8)

        ax.set_xlabel("Log-Moneyness")
        ax.set_ylabel(y_label)
        ax.set_title(name)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    return fig


def plot_impvol(
    df: pd.DataFrame,
    title: str = "",
    maturity: str = "ttm",
    nrow: Optional[int] = None,
    train: bool = True,
    figsize: Tuple[int, int] = (12, 4),
) -> plt.Figure:
    """
    Plot implied volatility vs log-moneyness, faceted by maturity.

    Parameters
    ----------
    df : pd.DataFrame
        Data with ttm, logm, iv, name columns
    title : str
        Plot title
    maturity : str
        Column to use for faceting
    nrow : int or None
        Number of rows in facet grid
    train : bool
        Whether to show training points
    figsize : Tuple[int, int]
        Figure size

    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    maturity_values = sorted(df[maturity].unique())
    n_facets = len(maturity_values)

    if nrow is None:
        nrow = 1
    ncol = int(np.ceil(n_facets / nrow))

    fig, axes = plt.subplots(nrow, ncol, figsize=figsize, sharey=True)
    if n_facets == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    names = df["name"].unique()
    colors_series = {name: f"C{i}" for i, name in enumerate(names)}

    for ax, mat_val in zip(axes, maturity_values):
        df_mat = df[df[maturity] == mat_val]

        for name in names:
            df_name = df_mat[df_mat["name"] == name].sort_values("logm")
            ax.plot(
                df_name["logm"],
                df_name["iv"],
                label=name,
                color=colors_series[name],
                linewidth=1.2,
            )

        if train and "train" in df.columns:
            df_train = df_mat[df_mat["train"] == True]
            for name in df_train["name"].unique():
                df_name = df_train[df_train["name"] == name]
                ax.scatter(
                    df_name["logm"],
                    df_name["iv"],
                    color=colors_series.get(name, "gray"),
                    s=12,
                    zorder=5,
                )

        ax.set_xlabel("Log-Moneyness")
        ax.set_ylabel("Implied Volatility")
        ax.set_title(f"{maturity} = {mat_val:.4f}")
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for ax in axes[n_facets:]:
        ax.set_visible(False)

    # Add legend to first axis
    axes[0].legend(loc="upper right", fontsize=8)

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    return fig
