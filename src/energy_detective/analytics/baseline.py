"""Weather-adjusted baseline: predicts expected load from weather + calendar features.

Quality is reported with CV(RMSE) and NMBE and compared with ASHRAE Guideline 14
hourly thresholds (CV(RMSE) <= 30%, |NMBE| <= 10%).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from ..config import GL14_HOURLY


def make_features(index: pd.DatetimeIndex, weather: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    w = weather.reindex(index)
    c = calendar.reindex(index)
    temp = w["temp_c"].interpolate(limit_direction="both")
    rh = w["rh"].interpolate(limit_direction="both")
    X = pd.DataFrame(index=index)
    X["hour"] = index.hour
    X["dow"] = index.dayofweek
    X["doy_sin"] = np.sin(2 * np.pi * index.dayofyear / 365.25)
    X["doy_cos"] = np.cos(2 * np.pi * index.dayofyear / 365.25)
    X["temp"] = temp.to_numpy()
    X["cdh"] = np.maximum(temp.to_numpy() - 22, 0)
    X["temp_3h"] = temp.rolling(3, min_periods=1).mean().to_numpy()
    X["rh"] = rh.to_numpy()
    for col in ("is_weekend", "is_holiday", "is_break"):
        X[col] = c[col].fillna(0).astype(int).to_numpy()
    return X


def cv_rmse(y, yhat) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(100 * np.sqrt(np.mean((y - yhat) ** 2)) / np.mean(y))


def nmbe(y, yhat) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(100 * np.sum(yhat - y) / np.sum(y))


class WeatherAdjustedBaseline:
    """Gradient-boosted regressor + per-hour robust residual scale."""

    def __init__(self, holdout_days: int = 90, random_state: int = 0):
        self.holdout_days = holdout_days
        self.random_state = random_state
        self.model = None
        self.sigma_by_hour = np.full(24, 0.05)
        self.quality: dict = {}

    def _new(self):
        return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08,
                                             max_leaf_nodes=31, random_state=self.random_state)

    def fit(self, y: pd.Series, X: pd.DataFrame) -> "WeatherAdjustedBaseline":
        n_hold = self.holdout_days * 24
        if len(y) > 2 * n_hold:
            tr, ho = slice(0, len(y) - n_hold), slice(len(y) - n_hold, len(y))
            m = self._new().fit(X.iloc[tr], y.iloc[tr])
            pred = np.maximum(m.predict(X.iloc[ho]), 1e-3)
            yy = y.iloc[ho].to_numpy()
            rel = (yy - pred) / pred
            hours = X.iloc[ho]["hour"].to_numpy()
            for h in range(24):
                r = rel[hours == h]
                if len(r) >= 10:
                    mad = np.median(np.abs(r - np.median(r)))
                    self.sigma_by_hour[h] = max(1.4826 * mad, 0.02)
            self.quality = {"cv_rmse": cv_rmse(yy, pred), "nmbe": nmbe(yy, pred), "n_holdout": int(len(yy))}
        self.model = self._new().fit(X, y)
        self.quality["gl14_hourly_pass"] = bool(
            self.quality.get("cv_rmse", 1e9) <= GL14_HOURLY["cv_rmse"]
            and abs(self.quality.get("nmbe", 1e9)) <= GL14_HOURLY["nmbe"])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.maximum(self.model.predict(X), 1e-3)
