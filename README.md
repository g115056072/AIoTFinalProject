# Energy Detective

結合建築能源領域知識的校園用電異常診斷 AI Agent。

管理者問：「工學院上週為何用電偏高？」Agent 會取得電表與氣象資料、建立天氣校正基線、偵測異常、比對行事曆、檢索領域知識，最後回報**最可能原因、證據、信心與建議處置**，而不只是「用電變高了」。

> **狀態**：期末專題實作版。目前以**模擬資料**驗證流程；真實校園電表資料可透過相同 CSV 格式接入（見下方）。

---

## 快速開始

```bash
pip install -r requirements.txt          # numpy, pandas, scikit-learn, matplotlib
python -m unittest discover -s tests -v  # 21 項測試
python scripts/demo.py --query "2025-05-01 到 2025-05-20 工學院 為何用電偏高"
python scripts/run_eval.py --seeds 10    # 重現評估（約 1 分鐘）
```

- `demo.py` 輸出 `results/demo/report.md`（診斷報告）與 `report.png`（實際 vs 基線圖）。
- 不給 `--query` 時，自動選一個模擬異常事件當範例。
- `--data DIR` 讀取自己的 CSV 資料；`--llm` 用 Anthropic API 改寫說明文字（需 `ANTHROPIC_API_KEY`，失敗時自動退回模板，數字與結論不由 LLM 產生）。

## 與 Proposal 架構的對應

| Proposal 模組 | 實作位置 |
|---|---|
| ① 使用者情境／② 查詢輸入 | `scripts/demo.py`、`agent/agent.py::ask` |
| ③ Planner & Reasoning | `agent/planner.py`（意圖、建築、時間範圍解析，可替換為 LLM planner） |
| ④ Data Acquisition | `data/loaders.py`、`data/quality.py`、`data/simulator.py` |
| ⑤ Analytics Engine | `analytics/baseline.py`（天氣校正基線）、`analytics/detector.py`（異常偵測） |
| ⑥ Domain Knowledge | `knowledge/kb.json`、`knowledge/retriever.py`、`diagnosis/rules.py` |
| ⑦ Decision / Recommendation | `diagnosis/rules.py`、`agent/tools.py::build_findings` |
| ⑧ Output & Interaction | `report/report.py`、`agent/llm.py` |

## 方法摘要

1. **資料品質檢查**：缺漏、零值、凍結值。
2. **天氣校正基線**：梯度提升樹，特徵為時段、星期、季節、氣溫、冷卻度時、濕度、假日／學期假期。保留最後 90 天估計逐時殘差尺度，並以 ASHRAE Guideline 14 逐時準則（CV(RMSE) ≤ 30%、|NMBE| ≤ 10%）檢查基線品質。
3. **異常偵測**：相對殘差超過每小時穩健標準差的 3.5 倍（只針對用電偏高；偏低僅在大幅掉落時才算，例如零讀數），連續 ≥3 小時（或極端值）成為事件，間隔 ≤18 小時合併。
4. **行事曆比對**：與已登記活動重疊的增量視為預期變化，不報警。
5. **原因診斷**：以事件特徵（夜間占比、日夜增量比、與氣溫相關、持續時間、凍結／零值／突波）做透明規則評分；電表異常訊號強時優先於設備原因。
6. **知識庫**：每筆條目有來源與 `verified` 旗標；未驗證者在報告中標示「來源待驗證」。

## 評估結果（模擬資料）

設定：訓練期 2024（無異常），測試期 2025，每組 3 棟建築、共 27 個注入異常（夜間空調、冰水主機效率下降、基載異常、電表異常各類）與測試期內 24 個已排定活動（不應報警）。事件層級指標；種子 0–9 共 30 棟年、270 個真實異常。

| 方法 | Precision | Recall | F1 | 每棟年誤報數 |
|---|---|---|---|---|
| M1 固定門檻 | 0.13 | 0.09 | 0.10 | 11.5 |
| M2 季節剖面（不含天氣） | 0.22 | 0.74 | 0.34 | 28.6 |
| M3 天氣校正基線 | 0.51 | 0.97 | 0.67 | 8.9 |
| **M4 Energy Detective（＋行事曆）** | **0.87** | **0.97** | **0.92** | **1.4** |

原因診斷 Top-1 準確率（僅已偵測事件）：**97.0%**（多數類別基準 34.2%、隨機 25%）。

完整結果：`results/summary_seeds0-9.md`、`results/metrics_seeds0-9.json`、`results/evaluation_seeds0-9.png`。

**種子紀律**：種子 100–102 僅在開發偵測器時使用；最終種子 0–9 報告上表。看過 0–9 的錯誤分析後做過兩項調整（電表異常優先、事件合併間隔 12→18 小時），因此另以**沒看過的種子 10–19** 驗證：M4 F1 0.92、診斷準確率 98.5%（`results/summary_seeds10-19.md`）。

## 限制（請務必閱讀）

- **資料是模擬的**：模擬器與診斷規則由同一組作者撰寫，診斷準確率偏樂觀，只能說明流程可運作，不能代表真實場域表現。真實資料的雜訊、感測問題、操作行為遠比模擬複雜。
- 目前的原因類別只有四種，真實建築還有更多型態；規則沒涵蓋的會回報「原因不明」。
- 診斷為規則式推論，不是確定結論，建議經現場查核後再行動。
- 知識庫多數條目為作者領域筆記，**尚待補上正式文獻來源**（`verified: false`）。
- 低解析度電表（讀數量化）可能被誤判為凍結值。
- 氣象站與建築的距離偏差、基線誤差對閾值的影響尚未做敏感度分析。
- 未與「無領域知識的 LLM」做實際比較（需要 API 與評審標註，列為後續工作）。

## 使用自己的資料

準備 CSV（見 `src/energy_detective/data/loaders.py`）：

```
meter.csv    : timestamp, <建築代碼欄位…>   逐時 kW
weather.csv  : timestamp, temp_c, rh
calendar.csv : timestamp, is_weekend, is_holiday, is_break   （可省略，會自動推導）
events.csv   : building, start, end, name                    （可省略）
```

建築代碼目前對應 `ENG / LIB / DORM`（`data/simulator.py::BUILDINGS` 與 `config.py::BUILDING_KEYWORDS`），新增建築時在這兩處加入即可。氣象資料可自中央氣象署開放資料平台匯出為上述格式。

## 專案結構

```
src/energy_detective/
  data/         simulator, loaders, quality
  analytics/    baseline, detector
  knowledge/    kb.json, retriever
  diagnosis/    rules
  agent/        planner, tools, agent, llm
  report/       report
  evaluation/   evaluate
scripts/        demo, run_eval, make_sample_data
tests/          21 項單元／端到端測試
results/        評估輸出與示範報告
data/sample/    範例 CSV（模擬，seed 0）
DESIGN.md       系統設計文件
```

## 後續工作

1. 接入真實校園電表與中央氣象署資料。
2. 補齊知識庫的正式文獻來源，擴充 HVAC 故障型態。
3. 與無領域知識的 LLM 做對照實驗，並加入使用者（設施管理人員）評估。
4. 基線不確定性區間、氣象站距離敏感度分析。
5. 互動式介面（Streamlit）與回饋學習（案例記憶）。

## 參考文獻

1. UNEP & GlobalABC, *Global Status Report for Buildings and Construction 2025–2026*.
2. Falouz et al., *An Agentic AI Pipeline for Appliance-Level Energy Anomaly Detection and LLM-Driven Recommendations*, 2026. (arXiv/alphaXiv 2606.28467)
3. *Leveraging Generative AI and Semantic Data for Improved Operation of a Real-Life Building*, EIA Nordic 2025 (SINTEF).
4. Lin & coauthor, *Physics-Informed Large Language Models for HVAC Anomaly Detection with Autonomous Rule Generation*, arXiv:2510.17146, NeurIPS 2025 UrbanAI Workshop.
5. Miller et al., *The Building Data Genome Project 2, energy meter data from the ASHRAE Great Energy Predictor III competition*, Scientific Data, 2020 (arXiv:2006.02273).
6. *Advances in Calibration of Building Energy Models*, NREL, 2018 ASHRAE Building Performance Analysis Conference.
7. Burman et al., SimBuild 2020 (ASHRAE Guideline 14 calibration tolerances).

## AI 使用聲明

本專案由 Claude 協助設計與撰寫程式、測試與文件。所有結果皆由 `scripts/run_eval.py` 實際執行產生；作者需能解釋架構、設計理由與驗證方式。
