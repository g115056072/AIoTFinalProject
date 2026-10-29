"""CSV loaders. Real campus data plugs in through the same schema.

meter.csv    : timestamp, <one column per building code>   (kW, hourly)
weather.csv  : timestamp, temp_c, rh
calendar.csv : timestamp, is_weekend, is_holiday, is_break  (optional; derived if missing)
events.csv   : building, start, end, name                   (optional)
truth.csv    : building, kind, cause, start, end            (optional, evaluation only)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .simulator import CampusData, BUILDINGS, make_calendar


def save_dataset(data: CampusData, out_dir) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data.meter.rename_axis("timestamp").to_csv(out / "meter.csv")
    data.weather.rename_axis("timestamp").to_csv(out / "weather.csv")
    data.calendar.rename_axis("timestamp").to_csv(out / "calendar.csv")
    data.events.to_csv(out / "events.csv", index=False)
    if len(data.truth):
        data.truth.to_csv(out / "truth.csv", index=False)


def load_csv_dataset(in_dir) -> CampusData:
    d = Path(in_dir)
    meter = pd.read_csv(d / "meter.csv", index_col=0, parse_dates=True)
    weather = pd.read_csv(d / "weather.csv", index_col=0, parse_dates=True)
    cal_path = d / "calendar.csv"
    calendar = (pd.read_csv(cal_path, index_col=0, parse_dates=True)
                if cal_path.exists() else make_calendar(meter.index))
    ev_path = d / "events.csv"
    events = (pd.read_csv(ev_path, parse_dates=["start", "end"])
              if ev_path.exists() else pd.DataFrame(columns=["building", "start", "end", "name"]))
    truth_path = d / "truth.csv"
    truth = (pd.read_csv(truth_path, parse_dates=["start", "end"])
             if truth_path.exists() else pd.DataFrame())
    return CampusData(meter=meter,
                      weather=weather.reindex(meter.index).interpolate(limit=6),
                      calendar=calendar.reindex(meter.index).ffill(),
                      events=events, truth=truth,
                      buildings={c: BUILDINGS[c] for c in meter.columns if c in BUILDINGS})
