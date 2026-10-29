"""Planner: parses a natural-language query into a structured plan.

Deliberately rule-based so it works offline and is testable. An LLM planner can
replace `parse_query` as long as it returns the same `Plan`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from ..config import BUILDING_KEYWORDS


@dataclass
class Plan:
    buildings: list[str]
    start: pd.Timestamp
    end: pd.Timestamp
    intent: str
    steps: list[str]
    assumptions: list[str]


DATE_RE = re.compile(r"(\d{4})[-/年](\d{1,2})[-/月](\d{1,2})日?")


def _dates(query: str) -> list[pd.Timestamp]:
    return [pd.Timestamp(int(y), int(m), int(d)) for y, m, d in DATE_RE.findall(query)]


def parse_query(query: str, now: pd.Timestamp, available: list[str]) -> Plan:
    q = query.lower()
    assumptions: list[str] = []
    codes = [c for c in available
             if any(k in q for k in (kw.lower() for kw in BUILDING_KEYWORDS.get(c, [])))]
    if not codes:
        codes = list(available)
        assumptions.append("未指定建築，分析全部建築")

    today = now.normalize()
    dates = _dates(query)
    if len(dates) >= 2:
        start, end = dates[0], dates[1] + pd.Timedelta(hours=23)
    elif len(dates) == 1:
        start, end = dates[0], dates[0] + pd.Timedelta(hours=23)
    elif "昨晚" in query or "last night" in q:
        start, end = today - pd.Timedelta(hours=6), today + pd.Timedelta(hours=8)
        assumptions.append("「昨晚」解讀為前一日 18:00 至當日 08:00")
    elif "昨天" in query or "yesterday" in q:
        start, end = today - pd.Timedelta(days=1), today - pd.Timedelta(hours=1)
    elif "上週" in query or "上周" in query or "last week" in q:
        this_monday = today - pd.Timedelta(days=today.dayofweek)
        start, end = this_monday - pd.Timedelta(days=7), this_monday - pd.Timedelta(hours=1)
    elif "上個月" in query or "last month" in q:
        first = today.replace(day=1)
        start, end = (first - pd.Timedelta(days=1)).replace(day=1), first - pd.Timedelta(hours=1)
    else:
        start, end = today - pd.Timedelta(days=7), today - pd.Timedelta(hours=1)
        assumptions.append("未指定時間，預設分析最近 7 天")

    intent = "diagnose" if any(w in query for w in ("為何", "為什麼", "原因", "why", "診斷")) else "summary"
    steps = ["解析意圖與參數", "取得電表與氣象資料並檢查品質", "以天氣校正基線計算殘差",
             "偵測異常事件", "比對行事曆活動", "檢索領域知識並診斷原因", "彙整報告與證據"]
    return Plan(codes, start, end, intent, steps, assumptions)
