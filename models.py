"""
Model definitions for volatility forecasting.

This module defines a base class and several concrete models for
implied volatility prediction:

* `TSPLModel`: linear regression on TSPL features (trend and
  volatility kernels).
* `RealizedVolBaseline`: regression on 30‑day realized volatility.
* `HARModel`: heterogeneous autoregressive model using realized
  volatility at daily, weekly and monthly frequencies.

Each model exposes `fit` and `score` methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


class BaseModel(ABC):
    """Abstract base class for volatility models."""

    @abstractmethod
    def fit(self, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, *args, **kwargs):
        pass

    def score(self, X, y) -> Tuple[float, float]:
        """Compute RMSE and R² on given input and target.

        Parameters
        ----------
        X : any
            Model input (depends on the concrete model).
        y : array‑like
            True target values.

        Returns
        -------
        rmse : float
            Root mean squared error.
        r2 : float
            Coefficient of determination (R²).
        """
        y_pred = self.predict(X)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        return rmse, r2


class TSPLModel(BaseModel):
    """Linear regression on TSPL features R1 and R2."""

    def __init__(self, smoothing_span: int = 20) -> None:
        """Initialize the TSPL model.

        Parameters
        ----------
        smoothing_span : int, optional
            Span for exponential moving average applied to R2 before
            taking the square root.  A value of 0 disables smoothing.
        """
        self.model = LinearRegression()
        self.smoothing_span = smoothing_span

    def fit(self, R1: pd.Series, R2: pd.Series, y: pd.Series) -> None:
        """Fit linear regression on R1 and sqrt(R₂).

        Any rows where the feature vectors contain NaNs or where the
        target is NaN are dropped before fitting, as scikit‑learn’s
        ``LinearRegression`` does not handle NaN values.

        Parameters
        ----------
        R1, R2 : pd.Series
            TSPL features aligned on the same index as the target.
        y : pd.Series
            VIX series aligned with features.
        """
        # Apply smoothing to R2 if requested
        if self.smoothing_span > 0:
            R2_smooth = R2.ewm(span=self.smoothing_span, adjust=False).mean()
        else:
            R2_smooth = R2
        # Construct feature matrix
        X = np.vstack([R1.values, np.sqrt(R2_smooth.values)]).T
        # Align target
        y_arr = y.values
        # Drop rows with NaNs in X or y
        mask = (~np.isnan(X).any(axis=1)) & (~np.isnan(y_arr))
        X_clean = X[mask]
        y_clean = y_arr[mask]
        self.model.fit(X_clean, y_clean)

    def predict(self, inputs: Tuple[pd.Series, pd.Series] | np.ndarray) -> np.ndarray:
        """Predict VIX from TSPL features.

        Parameters
        ----------
        inputs : tuple of (R1_series, R2_series) or 2D numpy array

        Returns
        -------
        np.ndarray
            Predicted VIX values.
        """
        if isinstance(inputs, tuple):
            R1, R2 = inputs
            if self.smoothing_span > 0:
                R2_smooth = R2.ewm(span=self.smoothing_span, adjust=False).mean()
            else:
                R2_smooth = R2
            X = np.vstack([R1.values, np.sqrt(R2_smooth.values)]).T
        else:
            X = inputs
        # Drop rows with NaNs to avoid issues during prediction
        if isinstance(X, np.ndarray):
            mask = ~np.isnan(X).any(axis=1)
            X_clean = X[mask]
            y_pred = self.model.predict(X_clean)
            # Pad the prediction with NaNs where features were NaN to preserve length
            preds = np.full(X.shape[0], np.nan)
            preds[mask] = y_pred
            return preds
        return self.model.predict(X)

    def score(self, inputs: Tuple[pd.Series, pd.Series], y: pd.Series) -> Tuple[float, float]:
        """Compute RMSE and R² on given input and target, handling NaNs.

        Rows with NaNs in the features or in the target are removed prior to
        scoring.

        Parameters
        ----------
        inputs : tuple of (R1_series, R2_series)
            TSPL feature series for the evaluation period.
        y : pd.Series
            True target (VIX) values for the evaluation period.

        Returns
        -------
        rmse : float
            Root mean squared error.
        r2 : float
            Coefficient of determination (R²).
        """
        R1, R2 = inputs
        if self.smoothing_span > 0:
            R2_smooth = R2.ewm(span=self.smoothing_span, adjust=False).mean()
        else:
            R2_smooth = R2
        X = np.vstack([R1.values, np.sqrt(R2_smooth.values)]).T
        y_arr = y.values
        # Drop rows with NaNs in features or target
        mask = (~np.isnan(X).any(axis=1)) & (~np.isnan(y_arr))
        X_clean = X[mask]
        y_clean = y_arr[mask]
        y_pred = self.model.predict(X_clean)
        rmse = np.sqrt(mean_squared_error(y_clean, y_pred))
        r2 = r2_score(y_clean, y_pred)
        return rmse, r2


class RealizedVolBaseline(BaseModel):
    """Baseline model regressing the VIX on 30‑day realized volatility."""

    def __init__(self, window: int = 30) -> None:
        self.window = window
        self.model = LinearRegression()

    def fit(self, returns: pd.Series, y: pd.Series) -> None:
        """Fit regression on realized volatility.

        The model is trained on the intersection of the return and VIX
        indices.  Realized volatility is computed on the training
        returns and aligned with the VIX series.  Missing values
        resulting from the rolling window are dropped.

        Parameters
        ----------
        returns : pd.Series
            Series of daily returns aligned with y.
        y : pd.Series
            VIX series aligned with returns.
        """
        # Compute realized volatility on training returns
        rv = returns.rolling(self.window).std() * np.sqrt(252)
        common = rv.index.intersection(y.index)
        # Align and drop NaNs
        X = rv.loc[common].values.reshape(-1, 1)
        y_aligned = y.loc[common].values
        mask = ~np.isnan(X[:, 0])
        X = X[mask]
        y_aligned = y_aligned[mask]
        self.model.fit(X, y_aligned)
        # Keep a reference to the common index for information, but predictions
        # will recompute rolling vol on the fly
        self.train_index = common

    def predict(self, returns: pd.Series, y_index: Optional[pd.Index] = None) -> np.ndarray:
        """Predict VIX from realized volatility.

        Parameters
        ----------
        returns : pd.Series
            Returns series on which to compute realized volatility.
        y_index : pd.Index, optional
            Optional index of dates for which predictions should be
            returned.  If provided, realized volatility is aligned to
            these dates.  Otherwise predictions are made for all dates
            where the rolling window produces a value.

        Returns
        -------
        np.ndarray
            Predicted VIX values aligned with ``y_index`` if provided.
        """
        rv = returns.rolling(self.window).std() * np.sqrt(252)
        if y_index is not None:
            idx = rv.index.intersection(y_index)
        else:
            idx = rv.index
        # Drop NaNs resulting from rolling window
        X = rv.loc[idx].values.reshape(-1, 1)
        mask = ~np.isnan(X[:, 0])
        X = X[mask]
        preds = self.model.predict(X)
        return preds

    def score(self, returns: pd.Series, y: pd.Series) -> Tuple[float, float]:
        """Compute RMSE and R² on test data.

        This overrides the default `BaseModel.score` to allow
        alignment of realized volatility with the target series.
        """
        # Compute realized vol on test returns
        rv = returns.rolling(self.window).std() * np.sqrt(252)
        common = rv.index.intersection(y.index)
        X = rv.loc[common].values.reshape(-1, 1)
        y_aligned = y.loc[common].values
        mask = ~np.isnan(X[:, 0])
        X = X[mask]
        y_aligned = y_aligned[mask]
        y_pred = self.model.predict(X)
        rmse = np.sqrt(mean_squared_error(y_aligned, y_pred))
        r2 = r2_score(y_aligned, y_pred)
        return rmse, r2


class HARModel(BaseModel):
    """Heterogeneous autoregressive model of the VIX.

    The HAR model regresses the VIX on realized volatilities computed
    over daily, weekly and monthly horizons (1, 5 and 22 days).  It
    captures persistence at multiple time scales.
    """

    def __init__(self, horizons: Tuple[int, int, int] = (1, 5, 22)) -> None:
        self.horizons = horizons
        self.model = LinearRegression()
        self.index = None

    def _compute_har_features(self, returns: pd.Series) -> pd.DataFrame:
        rv = returns.pow(2).rolling(self.horizons[0]).mean().pow(0.5) * np.sqrt(252)
        features = []
        names = []
        for h in self.horizons:
            # Mean realized volatility over the horizon
            vol_h = returns.pow(2).rolling(h).mean().pow(0.5) * np.sqrt(252)
            features.append(vol_h)
            names.append(f'RV_{h}')
        feat_df = pd.concat(features, axis=1)
        feat_df.columns = names
        return feat_df

    def fit(self, returns: pd.Series, y: pd.Series) -> None:
        feat_df = self._compute_har_features(returns)
        common = feat_df.index.intersection(y.index)
        X = feat_df.loc[common].values
        y_aligned = y.loc[common].values
        # drop rows with NaNs
        mask = ~np.isnan(X).any(axis=1)
        X = X[mask]
        y_aligned = y_aligned[mask]
        self.model.fit(X, y_aligned)
        # Store the training index to be used if no y_index is provided in predict
        self.train_index = common[mask]

    def predict(self, returns: pd.Series, y_index: Optional[pd.Index] = None) -> np.ndarray:
        """Predict VIX from returns using the HAR model.

        Parameters
        ----------
        returns : pd.Series
            Returns series on which to compute features.
        y_index : pd.Index, optional
            Optional index of dates to return predictions for. If provided,
            features are aligned to this index; otherwise the model predicts
            on the index used during training.

        Returns
        -------
        np.ndarray
            Predicted VIX values.
        """
        feat_df = self._compute_har_features(returns)
        if y_index is not None:
            idx = feat_df.index.intersection(y_index)
        else:
            idx = feat_df.index.intersection(self.train_index)
        X = feat_df.loc[idx].values
        # Drop rows with NaNs
        mask = ~np.isnan(X).any(axis=1)
        X = X[mask]
        return self.model.predict(X)

    def score(self, returns: pd.Series, y: pd.Series) -> Tuple[float, float]:
        """Compute RMSE and R² for HAR model on test data.
        """
        feat_df = self._compute_har_features(returns)
        idx = feat_df.index.intersection(y.index)
        X = feat_df.loc[idx].values
        y_aligned = y.loc[idx].values
        mask = ~np.isnan(X).any(axis=1)
        X = X[mask]
        y_aligned = y_aligned[mask]
        y_pred = self.model.predict(X)
        rmse = np.sqrt(mean_squared_error(y_aligned, y_pred))
        r2 = r2_score(y_aligned, y_pred)
        return rmse, r2
