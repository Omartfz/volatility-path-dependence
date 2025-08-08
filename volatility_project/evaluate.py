"""
Evaluation and cross‑validation utilities.

This module provides helper functions to split data into training and
test sets based on dates and to perform simple time‑series
cross‑validation.  These functions are generic and can be used with
any model exposing `fit` and `score` methods.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


def train_test_split(dates: pd.Index, split_date: str) -> Tuple[pd.Index, pd.Index]:
    """Split a date index into training and test subsets around `split_date`.

    Parameters
    ----------
    dates : pd.Index
        Index of dates.
    split_date : str
        Date at which to split.  The training set includes all dates
        strictly before the split date; the test set includes all dates
        on or after the split date.

    Returns
    -------
    (train_index, test_index) : Tuple[pd.Index, pd.Index]
        Indices for the training and test sets.
    """
    split_ts = pd.to_datetime(split_date)
    train_idx = dates[dates < split_ts]
    test_idx = dates[dates >= split_ts]
    return train_idx, test_idx


def generate_time_splits(
    dates: pd.Index,
    windows: List[Tuple[str, str, str, str]]
) -> List[Tuple[pd.Index, pd.Index]]:
    """Generate train/test indices for a list of time windows.

    Each element of ``windows`` must be a 4‑tuple
    (train_start, train_end, test_start, test_end).  The training set
    consists of all dates from ``train_start`` up to and including
    ``train_end``; the test set includes dates from ``test_start`` up to
    and including ``test_end``.  Dates outside these ranges are
    excluded.

    Returns a list of tuples (train_index, test_index).
    """
    splits = []
    for (train_start, train_end, test_start, test_end) in windows:
        train_start_ts = pd.to_datetime(train_start)
        train_end_ts = pd.to_datetime(train_end)
        test_start_ts = pd.to_datetime(test_start)
        test_end_ts = pd.to_datetime(test_end)
        mask_train = (dates >= train_start_ts) & (dates <= train_end_ts)
        mask_test = (dates >= test_start_ts) & (dates <= test_end_ts)
        train_idx = dates[mask_train]
        test_idx = dates[mask_test]
        splits.append((train_idx, test_idx))
    return splits



def cross_validate_tspl(
    returns: pd.Series,
    vix: pd.Series,
    alphas_deltas: Tuple[float, float, float, float],
    splits: List[Tuple[pd.Index, pd.Index]],
    compute_features,
    model_factory,  
) -> Tuple[float, float]:
    """
    Cross-validate TSPL parameters on predefined splits.

    Parameters
    ----------
    returns, vix : pd.Series
        Returns and VIX with the same date index.
    alphas_deltas : tuple[float, float, float, float]
        (alpha1, delta1, alpha2, delta2).
    splits : list[tuple[pd.Index, pd.Index]]
        (train_index, test_index) pairs.
    compute_features : callable
        Returns (R1, R2) from returns and params.
    model_factory : callable
        Returns a fresh model exposing .fit/.score.

    Returns
    -------
    tuple[float, float]
        (mean_rmse, mean_r2) across splits.
    """

    alpha1, delta1, alpha2, delta2 = alphas_deltas
    rmses: List[float] = []
    r2s: List[float] = []


    R1, R2 = compute_features(returns, alpha1, delta1, alpha2, delta2)

    for train_idx, test_idx in splits:

        train_idx_aligned = train_idx.intersection(R1.index)
        test_idx_aligned = test_idx.intersection(R1.index)


        model = model_factory()


        model.fit(
            R1.loc[train_idx_aligned],
            R2.loc[train_idx_aligned],
            vix.loc[train_idx_aligned],
        )


        rmse, r2 = model.score(
            (R1.loc[test_idx_aligned], R2.loc[test_idx_aligned]),
            vix.loc[test_idx_aligned],
        )
        rmses.append(rmse)
        r2s.append(r2)

    return float(np.mean(rmses)), float(np.mean(r2s))
