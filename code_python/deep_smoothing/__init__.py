"""
Deep Smoothing: Neural Network for Implied Volatility Surface Modeling

Python implementation of the NeurIPS 2020 paper "Deep Smoothing" - a neural network
approach for modeling implied volatility surfaces that enforces absence of arbitrage
and respects parametric priors from financial models.
"""

from .bsm import gbsm_price, gbsm_iv, bsm_price, bsm_iv, gbsm_greek
from .tf_utils import reset_tf_session
from .models import (
    IVSmootherControls,
    FitControls,
    get_ivsmoother,
    get_w_atm,
    get_ivsmoother_dict,
)
from .training import init_and_train
from .plotting import plot_fit, plot_totvar, plot_impvol

__version__ = "1.0.0"
__all__ = [
    "gbsm_price",
    "gbsm_iv",
    "bsm_price",
    "bsm_iv",
    "gbsm_greek",
    "reset_tf_session",
    "IVSmootherControls",
    "FitControls",
    "get_ivsmoother",
    "get_w_atm",
    "get_ivsmoother_dict",
    "init_and_train",
    "plot_fit",
    "plot_totvar",
    "plot_impvol",
]
