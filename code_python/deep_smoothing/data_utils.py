"""
Data preprocessing utilities.

Converted from R/utils_data_preprocess.R
"""

import pandas as pd
import numpy as np
from typing import Tuple


def load_train_data(filepath: str = None) -> pd.DataFrame:
    """
    Load training data from CSV.

    Parameters
    ----------
    filepath : str
        Path to the training data CSV. If None, uses the default location
        in the R code directory.

    Returns
    -------
    pd.DataFrame
        Training data
    """
    if filepath is None:
        # Use the data from the R code directory
        import os
        code_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        filepath = os.path.join(code_dir, "code_neurips2020", "data", "train_data.csv")

    df = pd.read_csv(filepath)
    return df


def get_atm_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract ATM (at-the-money) data points.

    Selects the observation with minimum |logm| for each ttm.

    Parameters
    ----------
    df : pd.DataFrame
        Data with ttm and logm columns

    Returns
    -------
    pd.DataFrame
        ATM data subset
    """
    df_atm = df.copy()
    df_atm["abs_logm"] = df_atm["logm"].abs()

    # Get index of minimum |logm| for each ttm
    idx = df_atm.groupby("ttm")["abs_logm"].idxmin()
    df_atm = df_atm.loc[idx].drop(columns=["abs_logm"])

    return df_atm.reset_index(drop=True)


def train_test_split_by_logm(
    df: pd.DataFrame, logm_prop: float = 1.0, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/test by sampling log-moneyness values.

    Parameters
    ----------
    df : pd.DataFrame
        Data with logm column
    logm_prop : float
        Proportion of unique logm values to include in training
    seed : int
        Random seed

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (train_df, test_df)
    """
    np.random.seed(seed)

    unique_logm = df["logm"].unique()
    n_train = int(len(unique_logm) * logm_prop)

    train_logm = np.random.choice(unique_logm, size=n_train, replace=False)

    df_train = df[df["logm"].isin(train_logm)].copy()
    df_test = df[~df["logm"].isin(train_logm)].copy()

    return df_train, df_test


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived columns if missing.

    Parameters
    ----------
    df : pd.DataFrame
        Data

    Returns
    -------
    pd.DataFrame
        Data with added columns
    """
    df = df.copy()

    # Add moneyness if missing
    if "m" not in df.columns and "logm" in df.columns:
        df["m"] = np.exp(df["logm"])

    # Add total variance if missing
    if "w" not in df.columns and "iv" in df.columns and "ttm" in df.columns:
        df["w"] = df["iv"] ** 2 * df["ttm"]

    # Add implied volatility if missing
    if "iv" not in df.columns and "w" in df.columns and "ttm" in df.columns:
        df["iv"] = np.sqrt(df["w"] / df["ttm"])

    return df
