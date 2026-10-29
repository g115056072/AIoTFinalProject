"""Explanation layer.

TemplateExplainer is deterministic and offline (default; used in tests and eval).
AnthropicExplainer rephrases the SAME structured evidence in natural language.
It is told not to add facts; if the SDK, API key or network is unavailable, the
agent silently falls back to the template.
"""
from __future__ import annotations

import json
import os

from ..config import CAUSE_LABELS_ZH


class TemplateExplainer:
    name = "template"

    def explain(self, finding_payload: dict) -> str:
        p = finding_payload
        if p["kind"] == "expected_event":
            return (f"此段用電增加與行事曆活動「{p['event_name']}」重疊，視為預期變化，"
                    "不需故障處置。")
        d = p["diagnosis"]
        label = CAUSE_LABELS_ZH.get(d["cause"], d["cause"])
        conf = {"high": "高", "medium": "中", "low": "低"}[d["confidence"]]
        return f"最可能原因：{label}（信心：{conf}）。依據：" + "；".join(d["evidence"]) + "。"


class AnthropicExplainer:
    name = "anthropic"

    def __init__(self, model: str | None = None):
        self.model = model or os.getenv("ED_LLM_MODEL", "claude-sonnet-4-6")
        self.fallback = TemplateExplainer()

    def explain(self, finding_payload: dict) -> str:
        try:
            import anthropic  # optional dependency
            if not os.getenv("ANTHROPIC_API_KEY"):
                raise RuntimeError("no api key")
            client = anthropic.Anthropic()
            msg = client.messages.create(
                model=self.model, max_tokens=400,
                system=("你是建築能源診斷助理。只能使用提供的 JSON 證據改寫成簡潔的繁體中文說明，"
                        "不可新增證據中沒有的數字或原因；信心低時要明說不確定。"),
                messages=[{"role": "user", "content": json.dumps(finding_payload, ensure_ascii=False)}],
            )
            return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
        except Exception:
            return self.fallback.explain(finding_payload)
