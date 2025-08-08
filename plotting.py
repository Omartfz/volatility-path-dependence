"""
Visualization utilities for the volatility forecasting project.

This module groups together functions for plotting the data, kernels and
model outputs, inspired by the visualizations in the original notebook.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Tuple

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.nonparametric.smoothers_lowess import lowess

def plot_index_vs_vix(df: pd.DataFrame) -> None:
    """Plot the underlying index close and VIX on dual axes.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns 'index_close' and 'vix' indexed by date.

    Produces a matplotlib figure with two y‑axes.
    """
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Index Close", color='tab:blue')
    ax1.plot(df.index, df['index_close'], color='tab:blue', label='Index Close')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel("VIX", color='tab:red')
    ax2.plot(df.index, df['vix'], color='tab:red', alpha=0.7, label='VIX')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    fig.tight_layout()
    fig.suptitle("Underlying Index vs VIX", y=1.02)
    plt.show()


def plot_tspl_kernels(k1: np.ndarray, k2: np.ndarray) -> None:
    """Visualize two TSPL kernels on the same plot.

    Parameters
    ----------
    k1, k2 : np.ndarray
        Kernel arrays of length T+1.
    """
    tau = np.arange(len(k1))
    plt.figure(figsize=(8, 4))
    plt.plot(tau, k1, label='K1 (trend)')
    plt.plot(tau, k2, label='K2 (volatility)')
    plt.xlabel("Lag (days)")
    plt.ylabel("Weight")
    plt.title("TSPL kernels")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_real_vs_pred(dates: pd.Index, y_real: pd.Series, y_pred: pd.Series) -> None:
    """Plot real vs predicted VIX time series on the same axes.
    """
    plt.figure(figsize=(12, 5))
    plt.plot(dates, y_real, label='Real VIX', color='black')
    plt.plot(dates, y_pred, label='Predicted VIX', color='green', alpha=0.7)
    plt.xlabel("Date")
    plt.ylabel("VIX")
    plt.title("Real vs Predicted VIX")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_scatter_real_pred(y_real: pd.Series, y_pred: pd.Series) -> None:
    """Scatter plot of real VIX vs predicted VIX with identity line.
    """
    plt.figure(figsize=(5, 5))
    plt.scatter(y_pred, y_real, alpha=0.4, color='purple')
    min_val = min(y_real.min(), y_pred.min())
    max_val = max(y_real.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--')
    plt.xlabel("Predicted VIX")
    plt.ylabel("Real VIX")
    plt.title("Scatter: Real vs Predicted VIX")
    plt.grid(True)
    plt.show()


def plot_relationship(feature: pd.Series, y: pd.Series, feature_name: str, y_name: str = "VIX") -> None:
    """Plot relationship between a feature and the VIX.

    Parameters
    ----------
    feature : pd.Series
        Feature values aligned with y.
    y : pd.Series
        Target values.
    feature_name : str
        Name of the feature for labeling.
    y_name : str, optional
        Name of the target variable.
    """
    lowess_fit = lowess(y, feature, frac=0.15)
    
    plt.figure(figsize=(12, 5))
    plt.plot(lowess_fit[:, 0], lowess_fit[:, 1], color="red", linewidth=2, label="LOWESS")
    plt.scatter(feature, y, alpha=0.4, color='teal')
    plt.xlabel(feature_name)
    plt.ylabel(y_name)
    plt.title(f"{y_name} vs {feature_name}")
    plt.grid(True)
    plt.show()


def plot_residuals(y_real: pd.Series, y_pred: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Plot real VIX vs residuals and return residual series.

    Returns
    -------
    residuals : pd.Series
        Series of residuals aligned with y_real.
    """
    fig, ax1 = plt.subplots(figsize=(14, 4))
    ax1.plot(y_real.index, y_real, label="Real VIX", color="black")
    ax1.set_ylabel("Real VIX", color="black")
    ax1.tick_params(axis='y', labelcolor="black")


    ax2 = ax1.twinx()
    
    residuals_test = y_real - y_pred

    ax2.plot(y_real.index, residuals_test, color='blue', alpha=0.6, label="Residuals")
    ax2.axhline(0, color='gray', linestyle='--')
    ax2.set_ylabel("Residuals (VIX - predicted VIX)", color="blue")
    ax2.tick_params(axis='y', labelcolor="blue")


    fig.suptitle("Real VIX vs Residuals (test set)")
    fig.tight_layout()
    fig.legend(loc="upper right")
    plt.grid(True)
    plt.show()
    return residuals_test


def plot_histogram_residuals(residuals: pd.Series) -> None:
    """Plot histogram of residuals.
    """
    plt.figure(figsize=(6, 4))
    plt.hist(residuals.dropna(), bins=30, color='skyblue', edgecolor='black')
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.title("Histogram of Residuals")
    plt.grid(True)
    plt.show()


def plot_autocorrelation_residuals(residuals: pd.Series, lags: int = 40) -> None:
    """Plot autocorrelation function of residuals.
    """
    plt.figure(figsize=(6, 4))
    plot_acf(residuals.dropna(), lags=lags, alpha=0.05)
    plt.title("Autocorrelation of Residuals")
    plt.xlabel("Lag")
    plt.ylabel("ACF")
    plt.grid(True)
    plt.show()


def plot_realized_vs_pred(returns: pd.Series, vix_pred: pd.Series, window: int = 30) -> None:
    """Compare realized volatility (rolling std) with predicted VIX.
    """
    realized_vol = returns.rolling(window).std() * np.sqrt(252)
    common = realized_vol.index.intersection(vix_pred.index)
    rv = realized_vol.loc[common]
    vp = vix_pred.loc[common]
    plt.figure(figsize=(12, 4))
    plt.plot(rv.index, rv, label="Realized Volatility", color='blue')
    plt.plot(vp.index, vp, label="Predicted VIX", color='orange')
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.title("Realized Volatility vs Predicted VIX")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Scatter
    plt.figure(figsize=(5, 5))
    plt.scatter(vp, rv, alpha=0.3, color='green')
    min_val = min(rv.min(), vp.min())
    max_val = max(rv.max(), vp.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--')
    plt.xlabel("Predicted VIX")
    plt.ylabel("Realized Volatility")
    plt.title("Scatter: Predicted VIX vs Realized Volatility")
    plt.grid(True)
    plt.show()