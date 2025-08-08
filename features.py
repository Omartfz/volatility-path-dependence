"""
Feature engineering for the TSPL volatility model.

This module implements the time‑shifted power‑law (TSPL) kernels and
functions to compute the trend and volatility features used in the
path‑dependent volatility model of Guyon & Lekeufack (2023).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def tspl_kernel(alpha: float, delta: float, T: int = 300) -> np.ndarray:
    """Construct a normalized TSPL kernel of length `T+1`.

    The kernel is defined as K[tau] = 1 / (tau + delta)**alpha for
    tau=0..T, normalized to sum to one.  The parameters have the
    following interpretations:

    - `alpha` controls the memory decay (larger values decay faster).
    - `delta` shifts the kernel to delay the response (delta >= 0).

    Parameters
    ----------
    alpha : float
        Exponent controlling the decay rate. Must be positive.
    delta : float
        Time shift parameter. Must be non‑negative.
    T : int
        Maximum lag (inclusive). The kernel has length T+1.

    Returns
    -------
    np.ndarray
        Normalized kernel array of shape (T+1,).
    """
    tau = np.arange(0, T + 1, dtype=float)
    weights = 1.0 / (tau + delta)**alpha
    weights /= weights.sum()
    return weights


def compute_tspl_features(
    returns: pd.Series,
    alpha1: float,
    delta1: float,
    alpha2: float,
    delta2: float,
    T: int = 300
) -> Tuple[pd.Series, pd.Series]:
    """Compute TSPL features R1 and R2 using vectorized convolution.

    Parameters
    ----------
    returns : pd.Series
        Series of daily returns (no NaN).  Should be long enough to
        support at least `T+1` observations.
    alpha1, delta1 : float
        Parameters for the trend kernel (R1).
    alpha2, delta2 : float
        Parameters for the volatility kernel (R2).
    T : int
        Maximum lag in days (the kernels have length `T+1`).

    Returns
    -------
    (pd.Series, pd.Series)
        Two series R1 and R2 aligned to the original index starting at
        `returns.index[T:]`.
    """
    # Build kernels
    k1 = tspl_kernel(alpha1, delta1, T)
    k2 = tspl_kernel(alpha2, delta2, T)
    arr = returns.values.astype(float)
    # Check length
    if len(arr) < len(k1):
        raise ValueError("Return series too short for specified T")
    # Convolution directly with the kernels to compute a causal sliding dot product.
    #
    # We want the features R1 and R2 at time t to be the weighted sum of the
    # current return and the past T returns: R[t] = sum_{tau=0..T} r[t - tau] * K[tau].
    #
    # The numpy convolution ``np.convolve(arr, kernel)`` with ``mode='valid'``
    # produces exactly this causal correlation: the element at index i in
    # the output corresponds to the sum over arr[i : i+T+1] weighted by kernel
    # reversed in index, which matches the desired ordering without manually
    # reversing the kernel. Using the kernels directly avoids shifting the
    # weights and aligns with the formulation in the original notebook.
    r1_full = np.convolve(arr, k1, mode='valid')
    r2_full = np.convolve(arr**2, k2, mode='valid')
    # Build index
    new_index = returns.index[T:]
    R1 = pd.Series(r1_full, index=new_index, name='R1')
    R2 = pd.Series(r2_full, index=new_index, name='R2')
    return R1, R2
