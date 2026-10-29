"""Evaluation on synthetic data: 4 detection methods + cause diagnosis.

M1 fixed threshold (mean + 3 std of training load)
M2 weather-blind seasonal profile (month x daytype x hour median)
M3 weather-adjusted baseline residual detector (no calendar, no knowledge)
M4 Energy Detective = M3 + calendar explanation (+ diagnosis)

Matching: a true anomaly is detected if any predicted incident overlaps it
(with a tolerance); a predicted incident is a true positive if it overlaps any
true anomaly. Everything is incident-level (not hour-level).
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from ..analytics.detector import (FixedThresholdDetector, ResidualDetector,
                                  SeasonalProfileDetector, incidents_from_flags)
from ..agent.tools import ToolBox
from ..config import CAUSES, TRAIN_END
from ..data.simulator import CampusData, generate_dataset

TEST_START = "2025-01-01"
METHODS = ["M1_fixed_threshold", "M2_seasonal_profile", "M3_weather_baseline", "M4_energy_detective"]


def _overlap(inc_start, inc_end, t_start, t_end, tol_h: int = 3) -> bool:
    tol = pd.Timedelta(hours=tol_h)
    return inc_start <= t_end + tol and inc_end >= t_start - tol


def match_incidents(truth: pd.DataFrame, incs: list, events: pd.DataFrame, code: str):
    """Return counts and the truth->incident index mapping."""
    tp_truth, mapping = 0, {}
    for ti, t in truth.reset_index(drop=True).iterrows():
        hits = [k for k, i in enumerate(incs) if _overlap(i.start, i.end, t["start"], t["end"])]
        if hits:
            tp_truth += 1
            mapping[ti] = hits
    tp_pred, fp_event, fp_other = 0, 0, 0
    ev = events[events["building"] == code] if len(events) else events
    for i in incs:
        if any(_overlap(i.start, i.end, t["start"], t["end"]) for _, t in truth.iterrows()):
            tp_pred += 1
        else:
            near_event = len(ev) and any(_overlap(i.start, i.end, e["start"], e["end"], 1) for _, e in ev.iterrows())
            if near_event:
                fp_event += 1
            else:
                fp_other += 1
    return {"n_truth": len(truth), "tp_truth": tp_truth, "n_pred": len(incs), "tp_pred": tp_pred,
            "fp_event": int(fp_event), "fp_other": int(fp_other)}, mapping


def evaluate_data(data: CampusData, k: float = 3.5) -> dict:
    counts = {m: Counter() for m in METHODS}
    diag_pairs: list[tuple[str, str]] = []
    tools = ToolBox(data, TRAIN_END, detector=ResidualDetector(k=k))
    test_end = data.meter.index[-1]
    for code in data.meter.columns:
        y_all = data.meter[code]
        y_tr, y_te = y_all.loc[:TRAIN_END], y_all.loc[TEST_START:]
        truth = data.truth[data.truth["building"] == code]
        idx = y_te.index

        fixed = FixedThresholdDetector().fit(y_tr)
        inc1 = incidents_from_flags(fixed.flags(y_te), idx, min_run=3)
        seas = SeasonalProfileDetector(k=k).fit(y_tr)
        inc2 = incidents_from_flags(seas.flags(y_te), idx, min_run=3)

        an = tools.analyze(code, TEST_START, test_end, pad_hours=0)
        inc3 = an.incidents
        findings = tools.build_findings(an)
        anomalies = [f for f in findings if f.kind == "anomaly"]
        inc4 = [f.incident for f in anomalies]

        for name, incs in zip(METHODS, [inc1, inc2, inc3, inc4]):
            c, mapping = match_incidents(truth, incs, data.events, code)
            counts[name].update(c)
            if name == "M4_energy_detective":
                for ti, hits in mapping.items():
                    t = truth.reset_index(drop=True).iloc[ti]
                    best = max(hits, key=lambda h: anomalies[h].incident.hours)
                    diag_pairs.append((t["cause"], anomalies[best].diagnosis.cause))
    return {"counts": {m: dict(c) for m, c in counts.items()}, "diag_pairs": diag_pairs}


def prf(c: dict):
    p = c["tp_pred"] / c["n_pred"] if c["n_pred"] else 0.0
    r = c["tp_truth"] / c["n_truth"] if c["n_truth"] else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def run(seeds=range(10), k: float = 3.5) -> dict:
    per_seed = defaultdict(list)
    pooled = {m: Counter() for m in METHODS}
    pairs: list[tuple[str, str]] = []
    n_building_years = 0
    for s in seeds:
        data = generate_dataset(s)
        out = evaluate_data(data, k=k)
        n_building_years += data.meter.shape[1]
        for m in METHODS:
            pooled[m].update(out["counts"][m])
            per_seed[m].append(prf(out["counts"][m]))
        pairs += out["diag_pairs"]

    summary = {}
    for m in METHODS:
        P, R, F = prf(pooled[m])
        arr = np.array(per_seed[m])
        c = pooled[m]
        summary[m] = {
            "precision": P, "recall": R, "f1": F,
            "f1_seed_mean": float(arr[:, 2].mean()), "f1_seed_std": float(arr[:, 2].std()),
            "false_alarms_per_building_year": (c["n_pred"] - c["tp_pred"]) / n_building_years,
            "fp_on_scheduled_events": c["fp_event"], "fp_other": c["fp_other"],
            "n_pred": c["n_pred"], "n_truth": c["n_truth"], "detected": c["tp_truth"],
        }
    truth_counts = Counter(t for t, _ in pairs)
    acc = float(np.mean([t == p for t, p in pairs])) if pairs else 0.0
    majority = max(truth_counts.values()) / len(pairs) if pairs else 0.0
    per_cause = {}
    for cz in CAUSES:
        rows = [(t, p) for t, p in pairs if t == cz]
        per_cause[cz] = {"n": len(rows), "acc": float(np.mean([t == p for t, p in rows])) if rows else None}
    labels = CAUSES + ["unexplained"]
    confusion = {t: {p: sum(1 for a, b in pairs if a == t and b == p) for p in labels} for t in CAUSES}
    return {"seeds": list(seeds), "k": k, "n_building_years": n_building_years, "detection": summary,
            "diagnosis": {"n": len(pairs), "top1_accuracy": acc, "majority_baseline": majority,
                          "uniform_baseline": 1 / len(CAUSES), "per_cause": per_cause,
                          "confusion": confusion}}
