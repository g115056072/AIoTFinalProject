"""Reproduce the evaluation: python scripts/run_eval.py [--seeds 10] [--k 3.5]

Final results use seeds 0..N-1. Seeds 100-102 were used only while developing
the detector, so no hyper-parameter was tuned on the reported seeds.
"""
import _bootstrap  # noqa: F401
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from energy_detective.config import CAUSE_LABELS_ZH, CAUSES
from energy_detective.evaluation.evaluate import METHODS, run

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=10)
ap.add_argument("--start", type=int, default=0, help="first seed")
ap.add_argument("--k", type=float, default=3.5)
ap.add_argument("--out", default="results")
a = ap.parse_args()

res = run(range(a.start, a.start + a.seeds), k=a.k)
out = Path(a.out)
out.mkdir(exist_ok=True)
(out / f"metrics_seeds{a.start}-{a.start + a.seeds - 1}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))

D, G = res["detection"], res["diagnosis"]
L = ["# 評估結果（合成資料）", "",
     f"- 隨機種子：{res['seeds'][0]}–{res['seeds'][-1]}（共 {len(res['seeds'])} 組，每組 3 棟建築、1 年測試期，共 {res['n_building_years']} 棟年）",
     f"- 殘差閾值 k = {res['k']}",
     "- **注意**：資料為模擬器產生，診斷規則與模擬器由同一組作者撰寫，數字用於驗證流程，不代表真實場域準確度。", "",
     "## 事件層級偵測", "",
     "| 方法 | Precision | Recall | F1 | 每棟年誤報數 | 誤報中屬於已排定活動 |",
     "|---|---|---|---|---|---|"]
names = {"M1_fixed_threshold": "M1 固定門檻", "M2_seasonal_profile": "M2 季節剖面（不含天氣）",
         "M3_weather_baseline": "M3 天氣校正基線", "M4_energy_detective": "M4 Energy Detective（+行事曆）"}
for m in METHODS:
    d = D[m]
    L.append(f"| {names[m]} | {d['precision']:.2f} | {d['recall']:.2f} | {d['f1']:.2f} "
             f"(逐組 {d['f1_seed_mean']:.2f}±{d['f1_seed_std']:.2f}) | {d['false_alarms_per_building_year']:.1f} | {d['fp_on_scheduled_events']} |")
L += ["", f"真實異常事件總數：{D['M4_energy_detective']['n_truth']}。", "",
      "## 原因診斷（僅統計已被偵測到的事件）", "",
      f"- 樣本數：{G['n']}；Top-1 準確率：**{G['top1_accuracy']:.1%}**",
      f"- 對照：多數類別 {G['majority_baseline']:.1%}、均勻隨機 {G['uniform_baseline']:.1%}", "",
      "| 真實原因 | 樣本數 | Top-1 準確率 |", "|---|---|---|"]
for c in CAUSES:
    pc = G["per_cause"][c]
    L.append(f"| {CAUSE_LABELS_ZH[c]} | {pc['n']} | {'-' if pc['acc'] is None else format(pc['acc'], '.1%')} |")
L += ["", "### 混淆矩陣（列＝真實，欄＝預測）", "",
      "| | " + " | ".join(CAUSES + ["unexplained"]) + " |", "|---|" + "---|" * (len(CAUSES) + 1)]
for t in CAUSES:
    L.append(f"| {t} | " + " | ".join(str(G["confusion"][t][p]) for p in CAUSES + ["unexplained"]) + " |")
(out / f"summary_seeds{a.start}-{a.start + a.seeds - 1}.md").write_text("\n".join(L), encoding="utf-8")

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
x = np.arange(len(METHODS)); w = 0.27
for j, (key, lab) in enumerate([("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")]):
    ax[0].bar(x + (j - 1) * w, [D[m][key] for m in METHODS], w, label=lab)
ax[0].set_xticks(x); ax[0].set_xticklabels(["M1", "M2", "M3", "M4"]); ax[0].set_ylim(0, 1.05)
ax[0].set_title("Incident-level detection"); ax[0].legend(fontsize=8)
cm = np.array([[G["confusion"][t][p] for p in CAUSES + ["unexplained"]] for t in CAUSES])
ax[1].imshow(cm, cmap="Blues")
ax[1].set_xticks(range(5)); ax[1].set_xticklabels(["after\nhours", "chiller", "leak", "meter", "unexpl."], fontsize=8)
ax[1].set_yticks(range(4)); ax[1].set_yticklabels(["after hours", "chiller", "leak", "meter"], fontsize=8)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax[1].text(j, i, cm[i, j], ha="center", va="center", fontsize=9)
ax[1].set_title("Diagnosis confusion (true x predicted)")
fig.tight_layout(); fig.savefig(out / f"evaluation_seeds{a.start}-{a.start + a.seeds - 1}.png", dpi=140)
print("\n".join(L))
