"""Evidence-based diagnosis of an incident.

The features are computed by tools (numbers), the scores come from transparent
rules, and the knowledge base supplies explanations and actions. The LLM (if
used) only rephrases this structured evidence; it does not compute anything.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ACTIONS = {
    "hvac_after_hours": [
        "查核建築自動化系統（BAS）夜間排程與手動覆寫紀錄",
        "夜間現場巡檢，確認空調、通風與照明運轉狀態",
        "確認設定值是否在活動後恢復為預設排程",
    ],
    "chiller_inefficiency": [
        "檢視冰水主機 COP、供回水溫差與冷卻水溫度",
        "檢查熱交換器結垢、冷媒量與冷卻水塔運作",
        "比對同期相似建築的單位面積用電，確認是否為個別設備問題",
    ],
    "base_load_leak": [
        "以分電表或現場巡檢找出新增的常態負載（實驗設備、伺服器、加熱器）",
        "確認是否有設備未依規定關閉",
        "必要時檢查迴路漏電",
    ],
    "meter_fault": [
        "先確認電表與通訊狀態，補齊或標記該段資料",
        "資料問題確認前，不要據此調整空調或設備",
    ],
    "unexplained": [
        "補充分項電表、設備運轉紀錄或現場資訊後再判斷",
    ],
}


@dataclass
class Diagnosis:
    cause: str
    score: float
    confidence: str
    evidence: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    features: dict = field(default_factory=dict)
    candidates: list = field(default_factory=list)   # [(cause, score)]


def incident_features(y, pred, temp, index, i0, i1) -> dict:
    y = np.asarray(y, float)
    pred = np.asarray(pred, float)
    temp = np.asarray(temp, float)
    sl = slice(i0, i1 + 1)
    ys, ps, ts = y[sl], pred[sl], temp[sl]
    hours = index[sl].hour.to_numpy()
    resid = ys - ps
    n = len(ys)
    prev = y[i0 - 1] if i0 > 0 else np.nan
    ext = np.concatenate([[prev], ys]) if i0 > 0 else ys
    flat_frac = float(np.mean(np.diff(ext) == 0)) if len(ext) > 1 else 0.0
    zero_frac = float(np.mean(ys <= 0.05 * ps))
    night = (hours >= 22) | (hours <= 5)
    day = (hours >= 9) & (hours <= 17)
    pos = np.clip(resid, 0, None)
    night_frac = float(pos[night].sum() / pos.sum()) if pos.sum() > 0 else 0.0
    day_mean = float(resid[day].mean()) if day.any() else np.nan
    deep_night = hours <= 5
    night_mean = float(resid[deep_night].mean()) if deep_night.any() else np.nan
    cdh = np.maximum(ts - 22, 0)
    if n >= 24 and np.std(resid) > 0 and np.std(cdh) > 0:
        temp_corr = float(np.corrcoef(resid, cdh)[0, 1])
    else:
        temp_corr = float("nan")
    return {
        "hours": int(n), "mean_resid_kw": float(resid.mean()),
        "mean_rel": float(resid.mean() / ps.mean()), "pos_frac": float(np.mean(resid > 0)),
        "flat_frac": flat_frac, "zero_frac": zero_frac,
        "max_rel_ratio": float(np.max(ys / ps)), "night_frac": night_frac,
        "day_mean_kw": day_mean, "night_mean_kw": night_mean, "temp_corr": temp_corr,
        "excess_kwh": float(resid.sum()),
    }


def _confidence(top: float, second: float) -> str:
    if top >= 0.8 and top - second >= 0.2:
        return "high"
    if top >= 0.6:
        return "medium"
    return "low"


def diagnose(features: dict) -> Diagnosis:
    f = features
    scores: dict[str, float] = {}
    evidence: dict[str, list] = {}

    # meter / data faults first: they invalidate every other interpretation
    ev = []
    s = 0.0
    if f["flat_frac"] >= 0.6 and f["hours"] >= 6:
        s = 0.95
        ev.append(f"{f['flat_frac']:.0%} 的相鄰小時讀數完全相同（疑似凍結值）")
    if f["zero_frac"] >= 0.5:
        s = max(s, 0.95)
        ev.append(f"{f['zero_frac']:.0%} 的時數讀數接近 0（遠低於預期）")
    if f["max_rel_ratio"] >= 3 and f["hours"] <= 4:
        s = max(s, 0.85)
        ev.append(f"短時間內讀數達預期的 {f['max_rel_ratio']:.1f} 倍（疑似突波）")
    if s:
        scores["meter_fault"], evidence["meter_fault"] = s, ev

    positive = f["mean_rel"] > 0 and f["pos_frac"] > 0.6
    if positive and f["night_frac"] >= 0.7 and f["hours"] <= 16:
        scores["hvac_after_hours"] = 0.6 + 0.4 * f["night_frac"]
        evidence["hvac_after_hours"] = [
            f"增量的 {f['night_frac']:.0%} 發生在 22:00–05:59",
            f"平均超出基線 {f['mean_resid_kw']:.0f} kW（{f['mean_rel']:.0%}）",
            f"持續 {f['hours']} 小時，屬夜間單次事件",
        ]

    if positive and f["pos_frac"] > 0.8 and f["hours"] >= 24 and f["night_mean_kw"] > 0 \
            and not np.isnan(f["day_mean_kw"]):
        ratio = f["day_mean_kw"] / f["night_mean_kw"]
        tc = f["temp_corr"]
        if 0.5 <= ratio <= 2.0 and (np.isnan(tc) or tc < 0.5):
            scores["base_load_leak"] = 0.8
            evidence["base_load_leak"] = [
                f"日間增量 {round(f['day_mean_kw'])} kW 與深夜增量 {round(f['night_mean_kw'])} kW 接近（比值 {ratio:.1f}）",
                "增量與氣溫關係不明顯" if np.isnan(tc) else f"增量與冷卻度時的相關係數僅 {tc:.2f}",
                f"持續 {f['hours']} 小時",
            ]

    if positive and f["hours"] >= 36:
        tc = 0.0 if np.isnan(f["temp_corr"]) else f["temp_corr"]
        dm, nm = f["day_mean_kw"], f["night_mean_kw"]
        day_heavy = (not np.isnan(dm)) and (np.isnan(nm) or nm <= 0 or dm > 1.8 * nm)
        if day_heavy or tc >= 0.5:
            scores["chiller_inefficiency"] = 0.7 + 0.2 * max(min(tc, 1.0), 0.0)
            evidence["chiller_inefficiency"] = [
                f"增量集中在日間（日間 {round(dm)} kW，深夜 {0 if np.isnan(nm) else round(nm)} kW）",
                f"增量與冷卻度時相關係數 {tc:.2f}",
                f"持續 {f['hours']} 小時（約 {f['hours'] / 24:.1f} 天）",
            ]

    scores["unexplained"] = 0.3
    evidence["unexplained"] = ["現有特徵無法對應已知原因型態"]
    # Design principle: a likely data fault invalidates every equipment explanation,
    # so a strong meter-fault signal takes precedence over other candidates.
    if scores.get("meter_fault", 0) >= 0.85:
        scores["meter_fault"] = 1.0
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    top, top_s = ranked[0]
    second_s = ranked[1][1] if len(ranked) > 1 else 0.0
    return Diagnosis(cause=top, score=float(top_s), confidence=_confidence(top_s, second_s),
                     evidence=evidence[top], actions=ACTIONS[top], features=f,
                     candidates=[(c, float(s)) for c, s in ranked])
