"""Synthetic campus electricity simulator with labelled anomalies.

Why synthetic: real campus smart-meter data may not be available. The simulator
provides ground-truth anomaly *causes* so detection and diagnosis can be
evaluated. Results on this data validate the pipeline, NOT real-world accuracy.
The diagnosis rules and the simulator are written by the same authors, so the
cause-accuracy numbers are optimistic by construction (see README, Limitations).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ACADEMIC = [0.05] * 7 + [0.2] + [1.0] * 10 + [0.6] * 4 + [0.2, 0.1]
LIBRARY = [0.08] * 7 + [0.3] + [1.0] * 14 + [0.4, 0.15]
DORM = [0.7] * 6 + [0.5] * 2 + [0.3] * 9 + [0.9] * 7


@dataclass(frozen=True)
class BuildingSpec:
    code: str
    name: str
    base_kw: float
    plug_kw: float
    cool_kw_per_c: float
    profile: tuple
    weekend_factor: float
    break_factor: float
    holiday_factor: float

    @property
    def peak_kw(self) -> float:
        return self.base_kw + self.plug_kw + self.cool_kw_per_c * 12


BUILDINGS = {
    "ENG": BuildingSpec("ENG", "工學院", 80, 150, 20, tuple(ACADEMIC), 0.2, 0.3, 0.15),
    "LIB": BuildingSpec("LIB", "圖書館", 60, 110, 14, tuple(LIBRARY), 0.7, 0.5, 0.4),
    "DORM": BuildingSpec("DORM", "宿舍", 50, 90, 12, tuple(DORM), 0.9, 0.4, 0.6),
}

# Simplified Taiwan holiday list (not exhaustive; for simulation only).
HOLIDAYS = pd.to_datetime([
    "2024-01-01", "2024-02-28", "2024-04-04", "2024-04-05", "2024-05-01", "2024-06-10",
    "2024-09-17", "2024-10-10", "2025-01-01", "2025-02-28", "2025-04-04", "2025-04-05",
    "2025-05-01", "2025-05-31", "2025-10-06", "2025-10-10",
])


@dataclass
class CampusData:
    meter: pd.DataFrame           # hourly kW, one column per building code
    weather: pd.DataFrame         # temp_c, rh
    calendar: pd.DataFrame        # is_weekend, is_holiday, is_break
    events: pd.DataFrame          # building, start, end (exclusive), name
    truth: pd.DataFrame = field(default_factory=pd.DataFrame)   # injected anomalies
    mask: pd.DataFrame = field(default_factory=pd.DataFrame)    # per-hour anomaly labels
    buildings: dict = field(default_factory=lambda: dict(BUILDINGS))


def make_calendar(idx: pd.DatetimeIndex) -> pd.DataFrame:
    d = idx.normalize()
    md = idx.month * 100 + idx.day
    is_break = ((md >= 113) & (md <= 219)) | ((md >= 621) & (md <= 907))
    return pd.DataFrame({
        "is_weekend": (idx.dayofweek >= 5).astype(int),
        "is_holiday": d.isin(HOLIDAYS).astype(int),
        "is_break": np.asarray(is_break).astype(int),
    }, index=idx)


def make_weather(idx: pd.DatetimeIndex, rng: np.random.Generator) -> pd.DataFrame:
    doy = idx.dayofyear.to_numpy()
    hour = idx.hour.to_numpy()
    seasonal = 23.5 + 7.5 * np.sin(2 * np.pi * (doy - 110) / 365)
    daily = 4.0 * np.sin(2 * np.pi * (hour - 9) / 24)
    noise = np.zeros(len(idx))
    e = rng.normal(0, 0.6, len(idx))
    for i in range(1, len(idx)):
        noise[i] = 0.93 * noise[i - 1] + e[i]
    temp = seasonal + daily + noise
    rh = np.clip(75 - 0.8 * (temp - 24) + rng.normal(0, 4, len(idx)), 40, 100)
    return pd.DataFrame({"temp_c": temp, "rh": rh}, index=idx)


def _components(spec: BuildingSpec, idx, weather, cal):
    level = np.ones(len(idx))
    level = np.where(cal["is_break"].to_numpy() == 1, spec.break_factor, level)
    level = np.where(cal["is_weekend"].to_numpy() == 1, np.minimum(level, spec.weekend_factor), level)
    level = np.where(cal["is_holiday"].to_numpy() == 1, np.minimum(level, spec.holiday_factor), level)
    occ = np.array(spec.profile)[idx.hour.to_numpy()] * level
    temp = weather["temp_c"].to_numpy()
    rh = weather["rh"].to_numpy()
    cool = spec.cool_kw_per_c * np.maximum(temp - 22, 0) * (0.25 + 0.75 * np.minimum(occ, 1.0))
    cool = cool * np.clip(1 + 0.006 * (rh - 70), 0.8, 1.2)
    return np.full(len(idx), spec.base_kw), spec.plug_kw * occ, cool


def _plan_events(rng, spec, idx):
    events = []
    for year in sorted(set(idx.year)):
        days = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
        weekends = [d for d in days if d.dayofweek >= 5 and d not in HOLIDAYS]
        for k in rng.choice(len(weekends), size=8, replace=False):
            d = weekends[int(k)]
            events.append({"building": spec.code, "start": d + pd.Timedelta(hours=9),
                           "end": d + pd.Timedelta(hours=17),
                           "name": "scheduled_event",
                           "extra_kw": float(rng.uniform(0.10, 0.18) * spec.peak_kw)})
    return events


def _overlaps(blocked, s, e, margin_h=24):
    m = pd.Timedelta(hours=margin_h)
    return any(not (e + m <= bs or s >= be + m) for bs, be in blocked)


def _plan_anomalies(rng, spec, test_start, test_end, blocked, counts):
    plans = []
    blocked = list(blocked)

    def place(kind, cause, dur_h, lo, hi, start_hour=None, **params):
        for _ in range(300):
            span_days = max((hi - lo).days - dur_h // 24, 1)
            day = lo + pd.Timedelta(days=int(rng.integers(0, span_days)))
            h = start_hour if start_hour is not None else int(rng.integers(0, 24))
            s = day + pd.Timedelta(hours=h)
            e = s + pd.Timedelta(hours=dur_h)
            if e > hi or _overlaps(blocked, s, e):
                continue
            blocked.append((s, e))
            plans.append({"building": spec.code, "kind": kind, "cause": cause,
                          "start": s, "end": e, **params})
            return

    for _ in range(counts.get("after_hours", 0)):
        place("after_hours", "hvac_after_hours", 8, test_start, test_end, start_hour=22,
              frac=float(rng.uniform(0.18, 0.28)))
    for _ in range(counts.get("chiller", 0)):
        place("chiller", "chiller_inefficiency", 24 * int(rng.integers(7, 11)),
              pd.Timestamp(test_start.year, 5, 1), pd.Timestamp(test_start.year, 9, 20),
              start_hour=0, mult=float(rng.uniform(1.30, 1.45)))
    for _ in range(counts.get("leak", 0)):
        place("leak", "base_load_leak", 24 * int(rng.integers(2, 6)), test_start, test_end,
              frac=float(rng.uniform(0.08, 0.14)))
    for _ in range(counts.get("meter", 0)):
        sub = str(rng.choice(["flatline", "zero", "spike"]))
        dur = {"flatline": int(rng.integers(8, 25)), "zero": int(rng.integers(4, 13)),
               "spike": int(rng.integers(1, 3))}[sub]
        place("meter_" + sub, "meter_fault", dur, test_start, test_end)
    return plans


def generate_dataset(seed: int = 0, start="2024-01-01", end="2025-12-31 23:00",
                     inject_from="2025-01-01", counts=None) -> CampusData:
    """Hourly campus data. Anomalies are injected only on/after `inject_from`."""
    counts = counts or {"after_hours": 3, "chiller": 1, "leak": 2, "meter": 3}
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq="h")
    weather = make_weather(idx, rng)
    cal = make_calendar(idx)
    meter = pd.DataFrame(index=idx)
    all_events, all_truth = [], []
    mask = pd.DataFrame(False, index=idx, columns=list(BUILDINGS))
    test_start = pd.Timestamp(inject_from)
    test_end = idx[-1] + pd.Timedelta(hours=1)

    for code, spec in BUILDINGS.items():
        base, plug, cool = _components(spec, idx, weather, cal)
        events = _plan_events(rng, spec, idx)
        all_events += events
        ev_extra = np.zeros(len(idx))
        for ev in events:
            ev_extra[(idx >= ev["start"]) & (idx < ev["end"])] += ev["extra_kw"]
        noise = rng.lognormal(0, 0.025, len(idx))
        blocked = [(ev["start"].normalize(), ev["end"].normalize() + pd.Timedelta(days=1))
                   for ev in events]
        plans = _plan_anomalies(rng, spec, test_start, test_end, blocked, counts)

        cool_mult = np.ones(len(idx))
        extra = np.zeros(len(idx))
        for p in plans:
            sel = (idx >= p["start"]) & (idx < p["end"])
            if p["kind"] in ("after_hours", "leak"):
                extra[sel] += p["frac"] * spec.peak_kw
            elif p["kind"] == "chiller":
                cool_mult[sel] *= p["mult"]
        y_clean = (base + plug + cool + ev_extra) * noise
        y = (base + plug + cool * cool_mult + ev_extra + extra) * noise
        for p in plans:
            sel = np.where((idx >= p["start"]) & (idx < p["end"]))[0]
            if p["kind"] == "meter_flatline":
                y[sel] = y[sel[0] - 1]
            elif p["kind"] == "meter_zero":
                y[sel] = 0.0
            elif p["kind"] == "meter_spike":
                y[sel] = y[sel] * 5.0
        rel = np.abs(y - y_clean) / np.maximum(y_clean, 1e-6)
        m = rel > 0.05
        mask[code] = m
        for p in plans:
            sel = (idx >= p["start"]) & (idx < p["end"])
            n_lab = int(m[sel].sum())
            if n_lab >= 1:
                all_truth.append({"building": code, "kind": p["kind"], "cause": p["cause"],
                                  "start": p["start"], "end": p["end"], "labeled_hours": n_lab})
        meter[code] = y

    return CampusData(meter=meter, weather=weather, calendar=cal,
                      events=pd.DataFrame(all_events), truth=pd.DataFrame(all_truth), mask=mask)
