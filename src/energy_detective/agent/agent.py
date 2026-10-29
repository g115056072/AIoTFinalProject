"""Energy Detective agent: plan -> retrieve -> analyze -> diagnose -> report."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..config import TRAIN_END
from ..data.simulator import CampusData
from .llm import TemplateExplainer
from .planner import Plan, parse_query
from .tools import Analysis, Finding, ToolBox


@dataclass
class AgentResult:
    query: str
    plan: Plan
    analyses: dict[str, Analysis]
    findings: list[Finding]
    explanations: dict[int, str] = field(default_factory=dict)   # index in findings -> text
    trace: list[str] = field(default_factory=list)
    report_md: str = ""


class EnergyDetectiveAgent:
    def __init__(self, data: CampusData, train_end: str = TRAIN_END, explainer=None,
                 now: pd.Timestamp | None = None):
        self.data = data
        self.tools = ToolBox(data, train_end)
        self.explainer = explainer or TemplateExplainer()
        self.now = now if now is not None else data.meter.index[-1] + pd.Timedelta(hours=1)

    def ask(self, query: str) -> AgentResult:
        self.tools.trace = []
        plan = parse_query(query, self.now, list(self.data.meter.columns))
        self.tools.log(f"plan: buildings={plan.buildings}, {plan.start:%Y-%m-%d %H:%M} ~ {plan.end:%Y-%m-%d %H:%M}")
        analyses, findings = {}, []
        for code in plan.buildings:
            a = self.tools.analyze(code, plan.start, plan.end)
            analyses[code] = a
            findings += self.tools.build_findings(a)
        result = AgentResult(query, plan, analyses, findings)
        for i, f in enumerate(findings):
            result.explanations[i] = self.explainer.explain(finding_payload(f))
        result.trace = list(self.tools.trace)
        from ..report.report import build_markdown
        result.report_md = build_markdown(result, self.data)
        return result


def finding_payload(f: Finding) -> dict:
    p = {"building": f.code, "start": str(f.incident.start), "end": str(f.incident.end),
         "kind": f.kind, "event_name": f.event_name}
    if f.diagnosis:
        d = f.diagnosis
        p["diagnosis"] = {"cause": d.cause, "confidence": d.confidence, "score": d.score,
                          "evidence": d.evidence, "actions": d.actions}
    return p
