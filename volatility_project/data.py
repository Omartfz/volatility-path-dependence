"""
Data loading and preprocessing utilities for volatility forecasting.

This module provides functions to download daily closing prices for
underlying equity indices and their corresponding implied volatility
indices via Yahoo Finance.  Data are cached locally to avoid
repeated downloads and basic preprocessing (return computation) is
performed.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple, Optional

import pandas as pd
import numpy as np
try:
    import yfinance as yf  # type: ignore
    _HAS_YFINANCE = True
except ImportError:
    _HAS_YFINANCE = False

def _download_series(ticker: str, start: str, end: str) -> pd.Series:
    """Download daily close prices for a ticker from Yahoo Finance.

    Parameters
    ----------
    ticker : str
        Ticker symbol (e.g. '^GSPC' for S&P 500).
    start : str
        Start date in YYYY-MM-DD format.
    end : str
        End date in YYYY-MM-DD format (exclusive).

    Returns
    -------
    pd.Series
        Series of daily closing prices indexed by date.
    """
    if not _HAS_YFINANCE:
        raise ImportError(
            "yfinance is required for downloading data but is not installed. "
            "Please install yfinance or provide cached CSV files in the cache_dir."
        )
    data = yf.download(ticker, start=start, end=end, progress=False)[['Close']]
    data.columns = ['close']
    return data['close']


def load_market_data(
    index_ticker: str,
    vix_ticker: str,
    start: str = '2000-01-01',
    end: str = '2025-01-01',
    cache_dir: str = 'data'
) -> pd.DataFrame:
    """Load or download index and VIX closing prices and compute returns.

    If cached CSV files exist for the requested tickers and date range,
    they are loaded; otherwise data are downloaded from Yahoo Finance
    and cached.  The returned DataFrame contains the following
    columns:

    - `index_close`: closing price of the underlying index
    - `vix`: closing value of the corresponding volatility index
    - `returns`: daily arithmetic returns of the index (pct change)

    Parameters
    ----------
    index_ticker : str
        Ticker for the underlying index.
    vix_ticker : str
        Ticker for the volatility index (e.g. '^VIX').
    start : str
        Start date for the data (inclusive).
    end : str
        End date for the data (exclusive).
    cache_dir : str
        Directory where downloaded data are cached.

    Returns
    -------
    pd.DataFrame
        DataFrame with daily prices and returns.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(exist_ok=True)
    # File names include date range to avoid collisions
    index_file = cache_path / f"{index_ticker.replace('^','')}_{start}_{end}.csv"
    vix_file = cache_path / f"{vix_ticker.replace('^','')}_{start}_{end}.csv"
    # Load or download index data
    if index_file.exists():
        index_series = pd.read_csv(index_file, index_col=0, parse_dates=True)['close']
    else:
        index_series = _download_series(index_ticker, start, end)
        index_series.to_csv(index_file)
    # Load or download VIX data
    if vix_file.exists():
        vix_series = pd.read_csv(vix_file, index_col=0, parse_dates=True)['close']
    else:
        vix_series = _download_series(vix_ticker, start, end)
        vix_series.to_csv(vix_file)
    # Merge on index
    df = pd.concat([
        index_series.rename('index_close'),
        vix_series.rename('vix')
    ], axis=1, join='inner').sort_index()
    # Compute returns (drop first NaN)
    df['returns'] = df['index_close'].pct_change()
    df.dropna(inplace=True)
    return df
