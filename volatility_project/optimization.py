"""
Parameter optimisation utilities for the TSPL model.

This module provides functions to calibrate the parameters of the TSPL
kernels (\alpha_1, \delta_1, \alpha_2, \delta_2) by minimising an
objective function on a set of time series splits.  It uses
``scipy.optimize.minimize`` and the existing cross validation
infrastructure.

Example
-------

>>> from volatility_project.data import load_market_data
>>> from volatility_project.evaluate import generate_time_splits
>>> from volatility_project.optimization import optimize_tspl_parameters
>>> df = load_market_data('^GSPC', '^VIX')
>>> returns, vix = df['returns'], df['vix']
>>> splits = generate_time_splits(df.index, [
...    ('2000-01-01','2015-12-31','2016-01-01','2020-12-31'),
...    ('2005-01-01','2015-12-31','2016-01-01','2020-12-31')
... ])
>>> params, rmse = optimize_tspl_parameters(returns, vix, splits)

The returned ``params`` tuple contains the calibrated parameters and
``rmse`` is the mean RMSE across the splits.
"""

from __future__ import annotations

from typing import List, Tuple, Callable, Optional

import numpy as np
from scipy.optimize import minimize

from .evaluate import cross_validate_tspl
from .features import compute_tspl_features
from .models import TSPLModel
import pandas as pd

def optimize_tspl_parameters(
    returns: np.ndarray | 'pd.Series',
    vix: np.ndarray | 'pd.Series',
    splits: List[Tuple['pd.Index', 'pd.Index']],
    initial_guess: Tuple[float, float, float, float] = (3.0, 20.0, 1.0, 0.1),
    bounds: Optional[List[Tuple[float, Optional[float]]]] = None,
    smoothing_span: int = 20,
    method: str = 'Powell',
    options: Optional[dict] = None,
) -> Tuple[Tuple[float, float, float, float], float]:
    """Optimise TSPL kernel parameters on time series splits.

    This function searches for the values of (\alpha_1, \delta_1, \alpha_2, \delta_2)
    that minimise the mean RMSE of a TSPL model across multiple
    train/test splits.  It wraps :func:`scipy.optimize.minimize` and
    relies on the existing :func:`cross_validate_tspl` for model evaluation.

    Parameters
    ----------
    returns, vix : array like
        Series of returns and corresponding VIX values.
    splits : list of (train_index, test_index)
        Train/test indices produced by :func:`generate_time_splits`.
    initial_guess : tuple of floats, optional
        Starting values for (\alpha_1, \delta_1, \alpha_2, \delta_2).
    bounds : list of tuples, optional
        Bounds for the parameters.  Each element should be a 2 tuple
        ``(lower, upper)``.  Use ``None`` for no upper bound.
    smoothing_span : int, optional
        Span for exponential moving average applied to R₂ before taking
        the square root.  Passed to the TSPL model.
    method : str, optional
        Optimisation method passed to :func:`scipy.optimize.minimize`.
    options : dict, optional
        Additional options for the optimiser.

    Returns
    -------
    (params, rmse) : tuple
        ``params`` is the best found tuple (\alpha_1, \delta_1, \alpha_2, \delta_2),
        and ``rmse`` is the corresponding mean RMSE across the splits.
    """

    if bounds is None:
        bounds = [
            (10e-4, float('inf')),   # alpha1
            (10e-4, float('inf')),  # delta1
            (10e-4, float('inf')),   # alpha2
            (10e-4, float('inf'))   # delta2
        ]

    def objective(params: np.ndarray) -> float:
        model_factory = lambda: TSPLModel(smoothing_span=smoothing_span)
        try:
            mean_rmse, _ = cross_validate_tspl(
                returns,
                vix,
                tuple(params),
                splits,
                compute_tspl_features,
                model_factory,
            )
            return mean_rmse
        except Exception:
            return np.inf

    result = minimize(
        objective,
        x0=np.array(initial_guess, dtype=float),
        bounds=bounds,
        method=method,
        options=options or {},
    )
    best_params = tuple(float(x) for x in result.x)
    best_rmse = float(result.fun)
    return best_params, best_rmse