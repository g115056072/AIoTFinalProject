"""Tools the agent can call. Every number in a report comes from one of these."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..analytics.baseline import WeatherAdjustedBaseline, make_features
from ..analytics.detector import Incident, ResidualDetector
from ..data.quality import quality_report
from ..data.simulator import CampusData
from ..diagnosis.rules import Diagnosis, diagnose, incident_features
from ..knowledge.retriever import KBHit, KnowledgeBase


@dataclass
class Analysis:
    code: str
    df: pd.DataFrame                      # y, pred, z, temp
    incidents: list[Incident]
    baseline_quality: dict
    data_quality: dict


@dataclass
class Finding:
    code: str
    incident: Incident
    kind: str                             # "anomaly" | "expected_event"
    diagnosis: Diagnosis | None = None
    event_name: str | None = None
    kb_hits: list[KBHit] = field(default_factory=list)


class ToolBox:
    def __init__(self, data: CampusData, train_end: str, kb: KnowledgeBase | None = None,
                 detector: ResidualDetector | None = None):
        self.data = data
        self.train_end = pd.Timestamp(train_end)
        self.kb = kb or KnowledgeBase()
        self.detector = detector or ResidualDetector()
        self._baselines: dict[str, WeatherAdjustedBaseline] = {}
        self.trace: list[str] = []

    def log(self, msg: str) -> None:
        self.trace.append(msg)

    # --- data tools -------------------------------------------------------
    def get_meter_data(self, code: str, start, end) -> pd.Series:
        s = self.data.meter[code].loc[start:end]
        self.log(f"get_meter_data({code}, {pd.Timestamp(start):%Y-%m-%d %H:%M}, {pd.Timestamp(end):%Y-%m-%d %H:%M}) -> {len(s)} 筆")
        return s

    def fit_baseline(self, code: str) -> WeatherAdjustedBaseline:
        if code not in self._baselines:
            y = self.data.meter[code].loc[:self.train_end].dropna()
            X = make_features(y.index, self.data.weather, self.data.calendar)
            self._baselines[code] = WeatherAdjustedBaseline().fit(y, X)
            q = self._baselines[code].quality
            self.log(f"fit_baseline({code}) -> CV(RMSE)={q['cv_rmse']:.1f}%, NMBE={q['nmbe']:+.1f}%")
        return self._baselines[code]

    # --- analytics tools --------------------------------------------------
    def analyze(self, code: str, start, end, pad_hours: int = 24) -> Analysis:
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        lo = max(start - pd.Timedelta(hours=pad_hours), self.data.meter.index[0])
        hi = min(end + pd.Timedelta(hours=pad_hours), self.data.meter.index[-1])
        y = self.get_meter_data(code, lo, hi)
        model = self.fit_baseline(code)
        X = make_features(y.index, self.data.weather, self.data.calendar)
        pred = model.predict(X)
        z, flags, incs = self.detector.detect(y.to_numpy(), pred, model.sigma_by_hour, y.index)
        df = pd.DataFrame({"y": y.to_numpy(), "pred": pred, "z": z, "temp": X["temp"].to_numpy()},
                          index=y.index)
        incs = [i for i in incs if i.end >= start and i.start <= end]
        dq = quality_report(y)
        self.log(f"detect_anomalies({code}) -> {len(incs)} 個事件；資料品質警告 {len(dq['warnings'])} 項")
        return Analysis(code, df, incs, model.quality, dq)

    def check_calendar(self, code: str, inc: Incident, df: pd.DataFrame) -> str | None:
        ev = self.data.events
        if ev is None or len(ev) == 0:
            return None
        mine = ev[ev["building"] == code]
        hrs = df.index[inc.i0:inc.i1 + 1]
        if (df["y"].iloc[inc.i0:inc.i1 + 1] - df["pred"].iloc[inc.i0:inc.i1 + 1]).mean() <= 0:
            return None
        for _, e in mine.iterrows():
            inside = ((hrs >= e["start"] - pd.Timedelta(hours=1)) & (hrs < e["end"] + pd.Timedelta(hours=1))).mean()
            if inside >= 0.5:
                self.log(f"check_calendar({code}) -> 事件與行事曆活動「{e['name']}」重疊 {inside:.0%}")
                return str(e["name"])
        return None

    def diagnose_incident(self, code: str, inc: Incident, df: pd.DataFrame) -> Diagnosis:
        f = incident_features(df["y"].to_numpy(), df["pred"].to_numpy(), df["temp"].to_numpy(),
                              df.index, inc.i0, inc.i1)
        d = diagnose(f)
        self.log(f"diagnose({code}, {inc.start:%m-%d %H:%M}) -> {d.cause}（{d.confidence}, score={d.score:.2f}）")
        return d

    def retrieve_knowledge(self, cause: str, k: int = 3) -> list[KBHit]:
        hits = self.kb.retrieve(cause.replace("_", " "), cause=cause, k=k)
        hits = [h for h in hits if cause in self.kb.get(h.id)["cause"]]
        self.log(f"retrieve_knowledge({cause}) -> {[h.id for h in hits]}")
        return hits

    def build_findings(self, a: Analysis) -> list[Finding]:
        out = []
        for inc in a.incidents:
            ev = self.check_calendar(a.code, inc, a.df)
            if ev:
                out.append(Finding(a.code, inc, "expected_event", event_name=ev,
                                   kb_hits=self.retrieve_knowledge("scheduled_event", k=1)))
                continue
            d = self.diagnose_incident(a.code, inc, a.df)
            hits = [] if d.cause == "unexplained" else self.retrieve_knowledge(d.cause, k=1)
            out.append(Finding(a.code, inc, "anomaly", diagnosis=d, kb_hits=hits))
        return out
