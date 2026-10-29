"""Anomaly detection on top of baselines, plus two comparison detectors."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Incident:
    i0: int
    i1: int            # inclusive
    start: pd.Timestamp
    end: pd.Timestamp  # inclusive hour

    @property
    def hours(self) -> int:
        return self.i1 - self.i0 + 1


def _runs(flags: np.ndarray):
    runs, i, n = [], 0, len(flags)
    while i < n:
        if flags[i]:
            j = i
            while j + 1 < n and flags[j + 1]:
                j += 1
            runs.append([i, j])
            i = j + 1
        else:
            i += 1
    return runs


def incidents_from_flags(flags: np.ndarray, index: pd.DatetimeIndex, min_run: int = 3,
                         fill_gap: int = 1, merge_gap: int = 18,
                         strong: np.ndarray | None = None) -> list[Incident]:
    """Turn hourly flags into incidents.

    1. join runs separated by <= fill_gap hours,
    2. keep runs with length >= min_run (or containing a `strong` hour),
    3. merge kept runs separated by <= merge_gap hours into one incident.
    """
    flags = np.asarray(flags, bool)
    runs = _runs(flags)
    joined = []
    for r in runs:
        if joined and r[0] - joined[-1][1] - 1 <= fill_gap:
            joined[-1][1] = r[1]
        else:
            joined.append(r[:])
    kept = []
    for a, b in joined:
        ok = (b - a + 1) >= min_run or (strong is not None and np.asarray(strong)[a:b + 1].any())
        if ok:
            kept.append([a, b])
    merged = []
    for r in kept:
        if merged and r[0] - merged[-1][1] - 1 <= merge_gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r[:])
    return [Incident(a, b, index[a], index[b]) for a, b in merged]


class ResidualDetector:
    """Flags hours where the relative residual exceeds k per-hour robust sigmas."""

    def __init__(self, k: float = 3.5, k_strong: float = 8.0, min_run: int = 3,
                 fill_gap: int = 1, merge_gap: int = 18, drop_frac: float = 0.5):
        self.k, self.k_strong, self.drop_frac = k, k_strong, drop_frac
        self.min_run, self.fill_gap, self.merge_gap = min_run, fill_gap, merge_gap

    def zscores(self, y, pred, sigma_by_hour, hours) -> np.ndarray:
        rel = (np.asarray(y, float) - pred) / pred
        return rel / sigma_by_hour[np.asarray(hours)]

    def detect(self, y, pred, sigma_by_hour, index: pd.DatetimeIndex):
        z = self.zscores(y, pred, sigma_by_hour, index.hour)
        z = np.nan_to_num(z, nan=0.0)
        rel = np.nan_to_num((np.asarray(y, float) - pred) / pred, nan=0.0)
        # Over-consumption is flagged at k sigmas. Under-consumption (savings,
        # holidays) is not an alarm unless it is a large drop (zero/shutdown).
        big_drop = rel < -self.drop_frac
        flags = (z > self.k) | big_drop
        strong = (z > self.k_strong) | big_drop
        return z, flags, incidents_from_flags(flags, index, self.min_run, self.fill_gap,
                                              self.merge_gap, strong)


class FixedThresholdDetector:
    """Baseline 1: flag load above mean + n_std * std of the training period."""

    def __init__(self, n_std: float = 3.0):
        self.n_std = n_std
        self.threshold = None

    def fit(self, y_train: pd.Series):
        self.threshold = float(y_train.mean() + self.n_std * y_train.std())
        return self

    def flags(self, y: pd.Series) -> np.ndarray:
        return (y.to_numpy() > self.threshold)


class SeasonalProfileDetector:
    """Baseline 2: weather-blind statistical baseline (median by month/daytype/hour)."""

    def __init__(self, k: float = 3.5):
        self.k = k
        self.profile = None
        self.sigma = None

    @staticmethod
    def _keys(index):
        daytype = (index.dayofweek >= 5).astype(int)
        return list(zip(index.month, daytype, index.hour))

    def fit(self, y_train: pd.Series):
        keys = self._keys(y_train.index)
        df = pd.DataFrame({"k": keys, "y": y_train.to_numpy()})
        self.profile = df.groupby("k")["y"].median().to_dict()
        pred = np.array([self.profile[k] for k in keys])
        rel = (y_train.to_numpy() - pred) / np.maximum(pred, 1e-3)
        hours = y_train.index.hour.to_numpy()
        self.sigma = np.full(24, 0.1)
        for h in range(24):
            r = rel[hours == h]
            self.sigma[h] = max(1.4826 * np.median(np.abs(r - np.median(r))), 0.02)
        return self

    def flags(self, y: pd.Series) -> np.ndarray:
        keys = self._keys(y.index)
        pred = np.array([self.profile.get(k, np.nan) for k in keys], float)
        pred = np.where(np.isnan(pred), np.nanmedian(pred), pred)
        z = (y.to_numpy() - pred) / np.maximum(pred, 1e-3) / self.sigma[y.index.hour.to_numpy()]
        return np.nan_to_num(z) > self.k
