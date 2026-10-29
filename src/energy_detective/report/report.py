"""Markdown report and plot for an AgentResult."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..config import CAUSE_LABELS_ZH, GL14_HOURLY  # noqa: E402


def build_markdown(result, data) -> str:
    p = result.plan
    names = {c: data.buildings[c].name if c in data.buildings else c for c in p.buildings}
    L = [f"# Energy Detective 診斷報告", "",
         f"- 問題：{result.query}",
         f"- 分析範圍：{', '.join(names.values())}；{p.start:%Y-%m-%d %H:%M} ～ {p.end:%Y-%m-%d %H:%M}"]
    for a in p.assumptions:
        L.append(f"- 假設：{a}")
    L += ["", "## 基線品質與資料品質", ""]
    for code, an in result.analyses.items():
        q = an.baseline_quality
        ok = "符合" if q.get("gl14_hourly_pass") else "未符合"
        L.append(f"- **{names[code]}**：基線 CV(RMSE)={q['cv_rmse']:.1f}%、NMBE={q['nmbe']:+.1f}%"
                 f"（{ok} ASHRAE Guideline 14 逐時準則：CV(RMSE)≤{GL14_HOURLY['cv_rmse']:.0f}%、|NMBE|≤{GL14_HOURLY['nmbe']:.0f}%）")
        for w in an.data_quality["warnings"]:
            L.append(f"  - 資料警告：{w}")
    L += ["", "## 偵測結果", ""]
    if not result.findings:
        L.append("此範圍內未偵測到超出天氣校正基線的異常。")
    for i, f in enumerate(result.findings):
        inc = f.incident
        an = result.analyses[f.code]
        seg = an.df.iloc[inc.i0:inc.i1 + 1]
        excess = float((seg["y"] - seg["pred"]).sum())
        L += [f"### {i + 1}. {names[f.code]}　{inc.start:%m-%d %H:%M} ～ {inc.end:%m-%d %H:%M}（{inc.hours} 小時）", "",
              f"- 累計超出基線：{excess:,.0f} kWh（基線 {seg['pred'].sum():,.0f} kWh）"]
        if f.kind == "expected_event":
            L.append(f"- 判斷：**預期變化**（行事曆活動「{f.event_name}」）")
        else:
            d = f.diagnosis
            L.append(f"- 判斷：**{CAUSE_LABELS_ZH[d.cause]}**（信心：{d.confidence}，分數 {d.score:.2f}）")
            L.append("- 證據：" + "；".join(d.evidence))
            alts = [f"{CAUSE_LABELS_ZH[c]} {s:.2f}" for c, s in d.candidates[1:3]]
            if alts:
                L.append("- 其他候選：" + "、".join(alts))
            L.append("- 建議處置：")
            L += [f"  {j + 1}. {a}" for j, a in enumerate(d.actions)]
        L.append(f"- 說明：{result.explanations.get(i, '')}")
        for h in f.kb_hits[:2]:
            tag = "" if h.verified else "（來源待驗證）"
            L.append(f"- 知識庫：{h.title}［{h.source}］{tag}")
        L.append("")
    L += ["## 限制", "",
          "- 診斷為規則式推論，非確定結論；建議經現場查核後再採取行動。",
          "- 基線是估計值，存在不確定性；異常判定取決於閾值設定。",
          "", "## 工具呼叫紀錄", ""]
    L += [f"{k + 1}. `{t}`" for k, t in enumerate(result.trace)]
    return "\n".join(L)


def plot_result(result, out_path) -> Path:
    n = max(len(result.analyses), 1)
    fig, axes = plt.subplots(n, 1, figsize=(11, 3.2 * n), squeeze=False)
    for ax, (code, an) in zip(axes[:, 0], result.analyses.items()):
        ax.plot(an.df.index, an.df["y"], lw=1.0, label="actual")
        ax.plot(an.df.index, an.df["pred"], lw=1.0, ls="--", label="weather-adjusted baseline")
        for f in result.findings:
            if f.code != code:
                continue
            color = "tab:green" if f.kind == "expected_event" else "tab:red"
            ax.axvspan(f.incident.start, f.incident.end, color=color, alpha=0.2)
        ax.set_ylabel(f"{code} kW")
        ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out
