"""
Neural network architecture, priors, and loss functions for IV surface smoothing.

Converted from R/utils_ivsmoother_models.R
"""

from dataclasses import dataclass, field
from typing import List, Optional, Callable, Dict, Any, Tuple
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp


@dataclass
class IVSmootherControls:
    """Controls for IV smoother model architecture."""

    neurons_vec: List[int] = field(default_factory=lambda: [40, 40, 40, 40])
    activation: str = "softplus"
    prior: Optional[str] = "svi"  # "svi", "bs", or None
    phi_fun: str = "power_law"  # "power_law" or "heston_like"
    w_atm_fun: Optional[Callable] = None
    spread: bool = False


@dataclass
class FitControls:
    """Controls for model fitting/training."""

    train_ini: bool = True
    var_ini_min: float = -0.25
    var_ini_max: float = 0.25
    var_toload: Optional[Dict] = None
    iter_max: int = 4000
    var_pred: str = "iv"  # "iv" or "w"
    penalty: Tuple[float, ...] = (1.0, 10.0, 10.0, 10.0, 0.1)  # fit, c4, c5, c6, atm
    learning_rate: float = 0.01
    tol_abs: float = 0.0025
    tol_rel: float = 0.01
    patience: int = 500
    n_restart: int = 4
    gpu_mem_frac: float = 1.0
    tf_seed: int = 1
    verbose: bool = True


def get_w_atm(df_atm: pd.DataFrame) -> Callable:
    """
    Create ATM total variance interpolation function.

    Parameters
    ----------
    df_atm : pd.DataFrame
        DataFrame with 'ttm' and 'w' columns for ATM data

    Returns
    -------
    Callable
        Function that takes ttm tensor and returns interpolated w_atm
    """
    ttm_atm = df_atm["ttm"].values
    w_atm = df_atm["w"].values

    # Sort by ttm
    sort_idx = np.argsort(ttm_atm)
    ttm_atm = ttm_atm[sort_idx]
    w_atm = w_atm[sort_idx]

    min_ttm = 0.0
    max_ttm = float(np.max(ttm_atm))
    ttm_grid = np.arange(min_ttm, max_ttm + 1e-2, 1e-2)

    # Check if w_atm is monotonically increasing
    if np.all(np.diff(w_atm) >= 0):
        # Prepend (0, 0) for proper interpolation
        ttm_atm = np.concatenate([[0], ttm_atm])
        w_atm = np.concatenate([[0], w_atm])
        # Use scipy for initial interpolation to get grid values
        from scipy.interpolate import interp1d

        interp_func = interp1d(ttm_atm, w_atm, kind="cubic", fill_value="extrapolate")
        y_grid = interp_func(ttm_grid).astype(np.float32)
    else:
        # Fallback: use log transformation with monotonic spline
        from scipy.interpolate import UnivariateSpline

        spline = UnivariateSpline(ttm_atm, np.log(w_atm), k=3, s=0)
        y_grid = np.exp(spline(ttm_grid)).astype(np.float32)

    # Create TF constant for the grid
    y_ref = tf.constant(y_grid, dtype=tf.float32)

    def w_atm_function(ttm):
        """Interpolate ATM total variance at given ttm values."""
        # Squeeze if needed and reshape
        ttm_flat = tf.reshape(ttm, [-1])
        result = tfp.math.interp_regular_1d_grid(
            x=ttm_flat,
            x_ref_min=min_ttm,
            x_ref_max=max_ttm,
            y_ref=y_ref,
            fill_value="extrapolate",
        )
        return tf.reshape(result, tf.shape(ttm))

    return w_atm_function


def _get_name(var: str, type_: str) -> str:
    """Create named variable identifier."""
    return f"{var}_{type_}"


def _get_placeholder(name: str, type_: str) -> tf.Tensor:
    """Create a TF1-style placeholder."""
    return tf.compat.v1.placeholder(
        tf.float32, shape=[None, 1], name=_get_name(name, type_)
    )


def _name_variable(var: tf.Tensor, name: str, type_: str) -> tf.Tensor:
    """Name a tensor for debugging."""
    return tf.identity(var, name=_get_name(name, type_))


def get_ivsmoother(controls: IVSmootherControls) -> Dict[str, Any]:
    """
    Build the IV smoother neural network model.

    Parameters
    ----------
    controls : IVSmootherControls
        Model architecture controls

    Returns
    -------
    Dict
        Model containing losses, variables, predictions, weights, and prior_model
    """
    neurons_vec = controls.neurons_vec
    activation = controls.activation
    prior = controls.prior
    phi_fun = controls.phi_fun
    w_atm_fun = controls.w_atm_fun
    spread = controls.spread

    # Activation function
    if activation == "relu":
        afun = tf.nn.relu
    elif activation == "softplus":
        afun = tf.nn.softplus
    else:
        raise ValueError(f"Unknown activation: {activation}")

    # Placeholders for total variance and implied volatility targets
    w = tf.compat.v1.placeholder(tf.float32, shape=[None, 1], name="w")
    iv = tf.compat.v1.placeholder(tf.float32, shape=[None, 1], name="iv")
    iv_spread = None
    if spread:
        iv_spread = tf.compat.v1.placeholder(tf.float32, shape=[None, 1], name="iv_spread")

    # Create placeholders for ttm and logm for each type
    types = ["fit", "c4c5", "c6", "atm"]
    ttm_logm = {}
    for t in types:
        ttm_logm[t] = {
            "ttm": _get_placeholder("ttm", t),
            "logm": _get_placeholder("logm", t),
        }

    # Input layers: concatenate ttm and logm
    inputs = {}
    for t in types:
        inputs[t] = tf.concat(
            [ttm_logm[t]["ttm"], ttm_logm[t]["logm"]], axis=1, name=_get_name("input", t)
        )

    # Initialize weights
    weights = {}
    if neurons_vec is not None:
        n_input = 2
        n_layer = len(neurons_vec)

        # Create weights for hidden layers and output layer
        for i in list(range(1, n_layer + 1)) + [0]:
            b_name = f"b{i}"
            W_name = f"W{i}"

            if i == 0:
                # Output layer
                n_to = 1
                n_from = neurons_vec[-1]
            else:
                # Hidden layers
                n_to = neurons_vec[i - 1]
                n_from = n_input if i == 1 else neurons_vec[i - 2]

            stddev = 1.0 / np.sqrt(n_from + n_to)

            weights[b_name] = tf.Variable(
                tf.random.normal([1, n_to], mean=0.0, stddev=stddev),
                name=b_name,
            )
            weights[W_name] = tf.Variable(
                tf.random.normal([n_from, n_to], mean=0.0, stddev=stddev),
                name=W_name,
            )

    # Build hidden layers
    layers = {t: {} for t in types}
    if neurons_vec is not None:
        for t in types:
            current_input = inputs[t]
            for i in range(1, len(neurons_vec) + 1):
                layer_output = afun(
                    tf.matmul(current_input, weights[f"W{i}"]) + weights[f"b{i}"],
                    name=_get_name(f"layer_{i}", t),
                )
                layers[t][f"layer_{i}"] = layer_output
                current_input = layer_output
            layers[t]["final_hidden"] = current_input
    else:
        for t in types:
            layers[t]["final_hidden"] = inputs[t]

    # Build output and w_hat
    prior_model = None
    outputs = {}
    w_hats = {}

    if prior is None:
        if neurons_vec is None:
            raise ValueError("neurons_vec and prior can't both be empty")

        # No prior: direct output with exp activation
        for t in types:
            output = tf.matmul(layers[t]["final_hidden"], tf.exp(weights["W0"])) + tf.exp(
                weights["b0"]
            )
            outputs[t] = _name_variable(output, "output", t)
            w_hats[t] = _name_variable(
                ttm_logm[t]["ttm"] * outputs[t], "w_hat", t
            )
    else:
        # Get prior model
        prior_model = _get_prior(ttm_logm, afun, w_atm_fun, prior, phi_fun)

        if neurons_vec is not None:
            # Scale parameter
            scale_trans = tf.Variable(
                tf.constant(np.log(np.e - 1), shape=[1, 1], dtype=tf.float32),
                name="scale_trans",
            )
            scale = tf.nn.softplus(scale_trans, name="scale")
            prior_model["params"]["scale"] = scale
            prior_model["params"]["scale_trans"] = scale_trans
            prior_model["var_list"].append("scale_trans")

            # Output: scale * (1 + (1 - 1e-3) * tanh(linear))
            for t in types:
                linear = tf.matmul(layers[t]["final_hidden"], weights["W0"]) + weights["b0"]
                output = scale * (1.0 + (1.0 - 1e-3) * tf.nn.tanh(linear))
                outputs[t] = _name_variable(output, "output", t)
        else:
            scale = tf.constant(1.0, shape=[1, 1], dtype=tf.float32, name="scale")
            for t in types:
                outputs[t] = _name_variable(scale, "output", t)

        # w_hat = output * w_prior
        for t in types:
            w_hats[t] = _name_variable(
                outputs[t] * prior_model["w"][t]["w_prior"], "w_hat", t
            )

    # Compute iv_hat = sqrt(w_hat / ttm)
    iv_hats = {}
    for t in types:
        iv_hats[t] = _name_variable(
            tf.pow(w_hats[t] / ttm_logm[t]["ttm"], 0.5), "iv_hat", t
        )

    # Compute losses
    losses = {}
    losses["fit"] = _get_loss_fit(w, w_hats["fit"], iv, iv_hats["fit"], iv_spread, prior_model)
    losses["c4c5"] = _get_loss_arb(w_hats["c4c5"], ttm_logm["c4c5"]["ttm"], ttm_logm["c4c5"]["logm"])
    losses["c6"] = _get_loss_arb(w_hats["c6"], ttm_logm["c6"]["ttm"], ttm_logm["c6"]["logm"])
    losses["atm"] = _get_loss_atm(outputs["atm"], prior)

    # Predictions for fit type
    preds = {
        "ttm": ttm_logm["fit"]["ttm"],
        "logm": ttm_logm["fit"]["logm"],
        "output": outputs["fit"],
        "w_hat": w_hats["fit"],
        "iv_hat": iv_hats["fit"],
    }
    if prior_model is not None:
        preds["w_prior"] = prior_model["w"]["fit"]["w_prior"]
        preds["iv_prior"] = prior_model["w"]["fit"].get("iv_prior")

    # Variables dict
    variables = {
        "ttm_logm": ttm_logm,
        "inputs": inputs,
        "outputs": outputs,
        "w_hats": w_hats,
        "iv_hats": iv_hats,
    }

    model = {
        "losses": losses,
        "variables": variables,
        "preds": preds,
        "weights": weights,
        "prior_model": prior_model,
        "controls": controls,
        "placeholders": {"w": w, "iv": iv, "iv_spread": iv_spread},
    }

    return model


def _get_prior(
    ttm_logm: Dict,
    afun: Callable,
    w_atm_fun: Callable,
    prior: str,
    phi_fun: str,
) -> Dict[str, Any]:
    """
    Build the prior model (SVI or BS).

    Parameters
    ----------
    ttm_logm : Dict
        Dictionary with ttm and logm placeholders for each type
    afun : Callable
        Activation function (unused but kept for compatibility)
    w_atm_fun : Callable
        Function to compute ATM total variance
    prior : str
        Prior type: "svi" or "bs"
    phi_fun : str
        Phi function type: "power_law" or "heston_like"

    Returns
    -------
    Dict
        Prior model with w, params, and var_list
    """
    if w_atm_fun is None:
        raise ValueError("A function for the ATM total variance should be provided!")

    output = {"prior": prior, "phi_fun": phi_fun}

    # Compute w_atm for each type
    w_atm = {}
    for t in ttm_logm:
        w_atm[t] = w_atm_fun(ttm_logm[t]["ttm"])

    if prior == "bs":
        # Black-Scholes: just ATM variance matching
        w_dict = {}
        for t in ttm_logm:
            w_dict[t] = {"w_atm": w_atm[t], "w_prior": w_atm[t]}

        output["w"] = w_dict
        output["params"] = {}
        output["var_list"] = []

    elif prior == "svi":
        # SVI prior with rho parameter
        rho_trans = tf.Variable(
            tf.zeros([1, 1], dtype=tf.float32), name="rho_trans"
        )
        rho = tf.tanh(rho_trans, name="rho")

        params = {"rho": rho, "rho_trans": rho_trans}
        var_list = ["rho_trans"]

        if phi_fun == "heston_like":
            # Heston-like parametrization
            lambda_trans = tf.Variable(
                tf.zeros([1, 1], dtype=tf.float32), name="lambda_trans"
            )
            lambda_ = tf.exp(lambda_trans, name="lambda")

            def phi_function(w_atm_val):
                return (
                    1.0
                    / (lambda_ * w_atm_val)
                    * (1.0 - (1.0 - tf.exp(-lambda_ * w_atm_val)) / (lambda_ * w_atm_val))
                )

            params["lambda"] = lambda_
            params["lambda_trans"] = lambda_trans
            var_list.append("lambda_trans")

        elif phi_fun == "power_law":
            # Power-law parametrization
            eta_trans = tf.Variable(
                tf.zeros([1, 1], dtype=tf.float32), name="eta_trans"
            )
            eta = tf.exp(eta_trans, name="eta")

            gamma_trans = tf.Variable(
                tf.zeros([1, 1], dtype=tf.float32), name="gamma_trans"
            )
            gamma = tf.nn.sigmoid(gamma_trans, name="gamma")

            def phi_function(w_atm_val):
                return eta / (
                    tf.pow(w_atm_val, gamma) * tf.pow(1.0 + w_atm_val, 1.0 - gamma)
                )

            params["eta"] = eta
            params["gamma"] = gamma
            params["eta_trans"] = eta_trans
            params["gamma_trans"] = gamma_trans
            var_list.extend(["eta_trans", "gamma_trans"])
        else:
            raise ValueError(f"Incorrect function for phi: {phi_fun}")

        # SSVI total variance formula
        def w_svi(logm, w_atm_val, phi):
            return (
                w_atm_val
                / 2.0
                * (
                    1.0
                    + rho * phi * logm
                    + tf.sqrt(tf.square(phi * logm + rho) + 1.0 - tf.square(rho))
                )
            )

        # Compute w_prior for each type
        w_dict = {}
        for t in ttm_logm:
            phi = phi_function(w_atm[t])
            phi = _name_variable(phi, "phi", t)
            w_prior = w_svi(ttm_logm[t]["logm"], w_atm[t], phi)
            w_prior = _name_variable(w_prior, "w_prior", t)
            iv_prior = tf.pow(w_prior / ttm_logm[t]["ttm"], 0.5)
            iv_prior = _name_variable(iv_prior, "iv_prior", t)

            w_dict[t] = {
                "w_atm": w_atm[t],
                "phi": phi,
                "w_prior": w_prior,
                "iv_prior": iv_prior,
            }

        output["w"] = w_dict
        output["params"] = params
        output["var_list"] = var_list

    else:
        raise ValueError(f"Prior not implemented: {prior}")

    return output


def _get_loss_fit(
    w: tf.Tensor,
    w_hat: tf.Tensor,
    iv: tf.Tensor,
    iv_hat: tf.Tensor,
    iv_spread: Optional[tf.Tensor],
    prior_model: Optional[Dict],
) -> Dict[str, tf.Tensor]:
    """
    Compute fitting losses.

    Parameters
    ----------
    w : tf.Tensor
        Target total variance
    w_hat : tf.Tensor
        Predicted total variance
    iv : tf.Tensor
        Target implied volatility
    iv_hat : tf.Tensor
        Predicted implied volatility
    iv_spread : tf.Tensor or None
        IV spread for weighted loss
    prior_model : Dict or None
        Prior model (unused but kept for compatibility)

    Returns
    -------
    Dict
        Dictionary of loss tensors
    """
    eps = 1e-6

    # Total variance RMSE
    l_fit_w_rmse = _name_variable(
        tf.pow(tf.reduce_mean(eps + tf.square(w - w_hat)), 0.5),
        "l",
        "fit_w_rmse",
    )

    # Total variance MAPE
    l_fit_w_mape = tf.reduce_mean(
        tf.abs(tf.divide(tf.subtract(w_hat, w), w + eps)), name="l_fit_w_mape"
    )

    # IV MAPE
    l_fit_iv_mape = tf.reduce_mean(
        tf.abs(tf.divide(tf.subtract(iv_hat, iv), iv + eps)), name="l_fit_iv_mape"
    )

    if iv_spread is None:
        # IV RMSE
        l_fit_iv_rmse = _name_variable(
            tf.pow(tf.reduce_mean(eps + tf.square(iv - iv_hat)), 0.5),
            "l",
            "fit_iv_rmse",
        )
        l_fit_iv = tf.add(l_fit_iv_rmse, l_fit_iv_mape, name="l_fit_iv")
    else:
        # Spread-weighted IV loss
        l_fit_iv_rmse = _name_variable(
            tf.reduce_mean(eps + tf.divide(tf.abs(iv - iv_hat), 1.0 + iv_spread)),
            "l",
            "fit_iv_rmse",
        )
        l_fit_iv = _name_variable(l_fit_iv_rmse, "l", "fit_iv")

    l_fit_w = tf.add(l_fit_w_rmse, l_fit_w_mape, name="l_fit_w")

    return {
        "l_fit_w_rmse": l_fit_w_rmse,
        "l_fit_w_mape": l_fit_w_mape,
        "l_fit_w": l_fit_w,
        "l_fit_iv_rmse": l_fit_iv_rmse,
        "l_fit_iv_mape": l_fit_iv_mape,
        "l_fit_iv": l_fit_iv,
    }


def _get_loss_arb(
    w: tf.Tensor, ttm: tf.Tensor, logm: tf.Tensor
) -> Dict[str, tf.Tensor]:
    """
    Compute arbitrage penalty losses.

    Parameters
    ----------
    w : tf.Tensor
        Total variance
    ttm : tf.Tensor
        Time to maturity
    logm : tf.Tensor
        Log-moneyness

    Returns
    -------
    Dict
        Dictionary of loss tensors and gradients
    """
    # Compute gradients
    dvdt = tf.gradients(ys=w, xs=ttm)[0]
    dvdm = tf.gradients(ys=w, xs=logm)[0]
    d2vdm2 = tf.gradients(ys=dvdm, xs=logm)[0]

    # Calendar arbitrage (C4): dw/dt >= 0
    l_c4 = tf.reduce_mean(tf.nn.relu(tf.negative(dvdt)), name="l_c4")

    # Butterfly arbitrage (C5): g_k >= 0
    g_k = _name_variable(
        tf.square(1.0 - logm * dvdm / (2.0 * w))
        - tf.square(dvdm) / 4.0 * (1.0 / w + 0.25)
        + d2vdm2 / 2.0,
        "g",
        "k",
    )
    l_c5 = tf.reduce_mean(tf.nn.relu(tf.negative(g_k)), name="l_c5")

    # Large-moneyness behavior (C6): smoothness penalty
    l_c6 = tf.reduce_mean(tf.abs(d2vdm2), name="l_c6")

    return {
        "l_c4": l_c4,
        "l_c5": l_c5,
        "l_c6": l_c6,
        "g_k": g_k,
        "dvdt": dvdt,
        "dvdm": dvdm,
        "d2vdm2": d2vdm2,
    }


def _get_loss_atm(ann_output: tf.Tensor, prior: Optional[str]) -> Dict[str, tf.Tensor]:
    """
    Compute ATM regularization loss.

    Parameters
    ----------
    ann_output : tf.Tensor
        Neural network output at ATM
    prior : str or None
        Prior type

    Returns
    -------
    Dict
        Dictionary with l_atm tensor
    """
    if prior is None:
        l_atm = tf.constant(0.0, name="l_atm")
    elif prior in ["svi", "bs"]:
        # Encourage NN output to be close to 1 at ATM (so prior dominates)
        l_atm = _name_variable(
            tf.pow(tf.reduce_mean(1e-6 + tf.square(ann_output - 1.0)), 0.5),
            "l",
            "atm",
        )
    else:
        raise ValueError(f"Unknown prior: {prior}")

    return {"l_atm": l_atm}


def get_ivsmoother_dict(
    df: pd.DataFrame,
    types: List[str] = ["fit", "c4c5", "c6", "atm"],
    iv_spread: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """
    Create feed dictionaries for the model.

    Parameters
    ----------
    df : pd.DataFrame
        Training data with ttm, logm, iv, w columns
    types : List[str]
        Types of data to create
    iv_spread : bool
        Whether to include iv_spread in the dictionary
    **kwargs
        Optional overrides for ttm and logm values

    Returns
    -------
    Dict
        Dictionary with 'ttm_logm' and 'di' (feed dictionary)
    """
    ttm = kwargs.get("ttm", df["ttm"].values)
    logm = kwargs.get("logm", df["logm"].values)

    ttm_logm = {}
    for t in types:
        ttm_logm[t] = _get_ttm_logm(t, ttm, logm, kwargs)

    # Build feed dictionary
    di = {}
    for t in types:
        ttm_key = f"ttm_{t}:0"
        logm_key = f"logm_{t}:0"
        di[ttm_key] = ttm_logm[t]["ttm"].reshape(-1, 1).astype(np.float32)
        di[logm_key] = ttm_logm[t]["logm"].reshape(-1, 1).astype(np.float32)

    # Add targets
    di["iv:0"] = df["iv"].values.reshape(-1, 1).astype(np.float32)
    di["w:0"] = df["w"].values.reshape(-1, 1).astype(np.float32)

    if iv_spread and "iv_spread" in df.columns:
        di["iv_spread:0"] = df["iv_spread"].values.reshape(-1, 1).astype(np.float32)

    return {"ttm_logm": ttm_logm, "di": di}


def _get_ttm_logm(
    type_: str, ttm: np.ndarray, logm: np.ndarray, kwargs: Dict
) -> Dict[str, np.ndarray]:
    """
    Get ttm and logm arrays for a specific type.

    Parameters
    ----------
    type_ : str
        Data type: "fit", "c4c5", "c6", or "atm"
    ttm : np.ndarray
        Time to maturity array
    logm : np.ndarray
        Log-moneyness array
    kwargs : Dict
        Optional overrides

    Returns
    -------
    Dict
        Dictionary with 'ttm' and 'logm' arrays
    """
    ttm_key = f"ttm_{type_}"
    logm_key = f"logm_{type_}"
    expand = True

    # Get ttm values
    if ttm_key in kwargs:
        ttm_vals = kwargs[ttm_key]
    else:
        if type_ == "fit":
            ttm_vals = ttm
            expand = False
        elif type_ == "c4c5":
            ttm_vals = _get_logspace_ttm(np.max(ttm) + 1)
        elif type_ == "c6":
            ttm_vals = np.unique(ttm)
        elif type_ == "atm":
            ttm_vals = np.unique(ttm)

    # Get logm values
    if logm_key in kwargs:
        logm_vals = kwargs[logm_key]
    else:
        if type_ == "fit":
            logm_vals = logm
        elif type_ == "c4c5":
            logm_vals = _get_powerspace_logm(np.min(logm), np.max(logm))
        elif type_ == "c6":
            # Extreme moneyness points
            logm_vals = np.array(
                [6 * np.min(logm), 4 * np.min(logm), 4 * np.max(logm), 6 * np.max(logm)]
            )
        elif type_ == "atm":
            logm_vals = np.array([0.0])

    # Expand grid if needed
    if expand:
        ttm_grid, logm_grid = np.meshgrid(ttm_vals, logm_vals, indexing="ij")
        return {"ttm": ttm_grid.flatten(), "logm": logm_grid.flatten()}
    else:
        return {"ttm": np.array(ttm_vals), "logm": np.array(logm_vals)}


def _get_logspace_ttm(ttm_max: float, n: int = 100) -> np.ndarray:
    """Create log-spaced time to maturity grid."""
    return np.exp(np.linspace(np.log(1 / 365), np.log(ttm_max), n))


def _get_powerspace_logm(logm_min: float, logm_max: float, n: int = 100) -> np.ndarray:
    """Create power-spaced log-moneyness grid."""
    # Cube root spacing for better resolution near ATM
    return np.linspace(
        -np.power(-logm_min * 2, 1 / 3), np.power(logm_max * 2, 1 / 3), n
    ) ** 3


def load_trained_variables(
    model: Dict, variables: Dict, session: tf.compat.v1.Session
) -> Dict:
    """
    Load trained variable values into the model.

    Parameters
    ----------
    model : Dict
        Model dictionary
    variables : Dict
        Dictionary with 'weights' and 'params' values
    session : tf.compat.v1.Session
        TensorFlow session

    Returns
    -------
    Dict
        Model (unchanged)
    """
    if model["weights"]:
        for name, tensor in model["weights"].items():
            if name in variables.get("weights", {}):
                tensor.load(variables["weights"][name], session)

    if model["prior_model"] is not None:
        var_list = model["prior_model"]["var_list"]
        for name in var_list:
            if name in variables.get("params", {}):
                value = variables["params"][name]
                if np.isscalar(value):
                    value = np.array([[value]], dtype=np.float32)
                model["prior_model"]["params"][name].load(value, session)

    return model


def save_trained_variables(model: Dict, session: tf.compat.v1.Session) -> Dict:
    """
    Save trained variable values from the model.

    Parameters
    ----------
    model : Dict
        Model dictionary
    session : tf.compat.v1.Session
        TensorFlow session

    Returns
    -------
    Dict
        Dictionary with 'weights' and 'params' values
    """
    trained_var = {}

    if model["weights"]:
        trained_var["weights"] = {
            name: session.run(tensor) for name, tensor in model["weights"].items()
        }

    if model["prior_model"] is not None:
        var_list = model["prior_model"]["var_list"]
        trained_var["params"] = {
            name: session.run(model["prior_model"]["params"][name]) for name in var_list
        }

    return trained_var
