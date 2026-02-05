"""
Black-Scholes-Merton pricing and implied volatility functions.

Converted from R/utils_bsm.R
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from typing import Literal


def gbsm_price(
    S: float,
    K: float,
    tau: float,
    Ir: float,
    Id: float,
    sig: float,
    option_type: Literal["C", "P"] = "C",
) -> float:
    """
    Generalized Black-Scholes-Merton option pricing.

    Parameters
    ----------
    S : float
        Spot price
    K : float
        Strike price
    tau : float
        Time to maturity (in years)
    Ir : float
        Integrated risk-free rate (r * tau)
    Id : float
        Integrated dividend yield (d * tau)
    sig : float
        Volatility
    option_type : str
        'C' for call, 'P' for put

    Returns
    -------
    float
        Option price
    """
    d1 = (np.log(S / K) + Ir - Id + 0.5 * sig**2 * tau) / (sig * np.sqrt(tau))
    d2 = d1 - sig * np.sqrt(tau)

    if option_type == "C":
        value = np.exp(-Id) * S * norm.cdf(d1) - K * np.exp(-Ir) * norm.cdf(d2)
    elif option_type == "P":
        value = K * np.exp(-Ir) * norm.cdf(-d2) - np.exp(-Id) * S * norm.cdf(-d1)
    else:
        raise ValueError(f"Unknown option type: {option_type}")

    return value


def gbsm_iv(
    S: float,
    K: float,
    tau: float,
    Ir: float,
    Id: float,
    price: float,
    option_type: Literal["C", "P"] = "C",
    lower: float = 1e-4,
    upper: float = 9.99,
) -> float:
    """
    Compute implied volatility given parameters, type, and market price.

    Parameters
    ----------
    S : float
        Spot price
    K : float
        Strike price
    tau : float
        Time to maturity (in years)
    Ir : float
        Integrated risk-free rate (r * tau)
    Id : float
        Integrated dividend yield (d * tau)
    price : float
        Market price of the option
    option_type : str
        'C' for call, 'P' for put
    lower : float
        Lower bound for IV search
    upper : float
        Upper bound for IV search

    Returns
    -------
    float
        Implied volatility, or np.nan if not found
    """

    def objective(sig):
        return gbsm_price(S, K, tau, Ir, Id, sig, option_type) - price

    try:
        iv = brentq(objective, lower, upper, xtol=np.finfo(float).eps / 2, maxiter=100000)
        return iv
    except (ValueError, RuntimeError):
        return np.nan


def gbsm_greek(
    S: float,
    K: float,
    tau: float,
    Ir: float,
    Id: float,
    sig: float,
    option_type: Literal["C", "P"] = "C",
    name: Literal["delta", "vega"] = "delta",
) -> float:
    """
    Compute option Greeks.

    Parameters
    ----------
    S : float
        Spot price
    K : float
        Strike price
    tau : float
        Time to maturity (in years)
    Ir : float
        Integrated risk-free rate (r * tau)
    Id : float
        Integrated dividend yield (d * tau)
    sig : float
        Volatility
    option_type : str
        'C' for call, 'P' for put
    name : str
        'delta' or 'vega'

    Returns
    -------
    float
        Greek value
    """
    d1 = (np.log(S / K) + Ir - Id + 0.5 * sig**2 * tau) / (sig * np.sqrt(tau))

    if name == "delta":
        if option_type == "C":
            return np.exp(-Id) * norm.cdf(d1)
        elif option_type == "P":
            return -np.exp(-Id) * norm.cdf(-d1)
    elif name == "vega":
        return S * np.exp(-Id) * norm.pdf(d1) * np.sqrt(tau)
    else:
        raise ValueError(f"Unknown greek: {name}")


def bsm_price(
    S: float,
    K: float,
    tau: float,
    r: float,
    d: float,
    sig: float,
    option_type: Literal["C", "P"] = "C",
) -> float:
    """
    Black-Scholes-Merton option pricing wrapper.

    Parameters
    ----------
    S : float
        Spot price
    K : float
        Strike price
    tau : float
        Time to maturity (in years)
    r : float
        Risk-free rate (annualized)
    d : float
        Dividend yield (annualized)
    sig : float
        Volatility
    option_type : str
        'C' for call, 'P' for put

    Returns
    -------
    float
        Option price
    """
    return gbsm_price(S, K, tau, r * tau, d * tau, sig, option_type)


def bsm_iv(
    S: float,
    K: float,
    tau: float,
    r: float,
    d: float,
    price: float,
    option_type: Literal["C", "P"] = "C",
) -> float:
    """
    Black-Scholes-Merton implied volatility wrapper.

    Parameters
    ----------
    S : float
        Spot price
    K : float
        Strike price
    tau : float
        Time to maturity (in years)
    r : float
        Risk-free rate (annualized)
    d : float
        Dividend yield (annualized)
    price : float
        Market price of the option
    option_type : str
        'C' for call, 'P' for put

    Returns
    -------
    float
        Implied volatility
    """
    return gbsm_iv(S, K, tau, r * tau, d * tau, price, option_type)


def bsm_greek(
    S: float,
    K: float,
    tau: float,
    r: float,
    d: float,
    sig: float,
    option_type: Literal["C", "P"] = "C",
    name: Literal["delta", "vega"] = "delta",
) -> float:
    """
    Black-Scholes-Merton Greek wrapper.

    Parameters
    ----------
    S : float
        Spot price
    K : float
        Strike price
    tau : float
        Time to maturity (in years)
    r : float
        Risk-free rate (annualized)
    d : float
        Dividend yield (annualized)
    sig : float
        Volatility
    option_type : str
        'C' for call, 'P' for put
    name : str
        'delta' or 'vega'

    Returns
    -------
    float
        Greek value
    """
    return gbsm_greek(S, K, tau, r * tau, d * tau, sig, option_type, name)
