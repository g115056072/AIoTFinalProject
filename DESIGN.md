# DESIGN.md — Energy Detective

> 系統設計文件。隨實作持續維護；每次設計決策變更請更新第 12 節「變更紀錄」。
> 狀態：**已實作並以模擬資料驗證**；真實資料驗證尚未進行。

---

## 1. 目標與範圍

### 1.1 問題
偵測到用電異常不等於知道原因。同樣的用電飆升，可能來自天氣、營運（課表／活動）、設備故障或電表資料問題，處置完全不同。本系統回答：**「這個異常是什麼？為什麼？該怎麼辦？」**

### 1.2 使用者
設施管理人員、節能工程師、學校行政單位。

### 1.3 範圍
| 範圍內 | 範圍外 |
|---|---|
| 建築（棟別）層級用電異常偵測與原因診斷 | 設備自動控制（不直接操作空調） |
| 天氣校正基線、殘差分析 | 即時毫秒級保護 |
| 領域知識庫檢索與規則推理 | 電價／契約容量最佳化 |
| 自然語言查詢與報告 | 帳務與電費計算 |

### 1.4 設計原則
1. **工具負責計算，LLM 只負責改寫**：所有數字與結論來自分析工具與規則；LLM（可選）只把結構化證據改寫成文字，被禁止新增證據。
2. **每個結論可追溯**：報告附證據、候選原因、知識庫來源與完整工具呼叫紀錄。
3. **誠實表達不確定性**：基線品質、資料警告、信心等級、「原因不明」都是一等公民。
4. **資料問題優先於設備推論**：強烈的電表異常訊號會覆蓋其他候選原因。
5. **人在迴路**：Agent 提供診斷建議，最終決策由人負責。

---

## 2. 系統架構

| # | 模組 | 職責 | 實作 |
|---|---|---|---|
| ① | User / Application Scenario | 設施管理人員查詢「某棟某時段為何用電偏高」 | — |
| ② | Query / Input | 自然語言問題 | `scripts/demo.py`、`EnergyDetectiveAgent.ask` |
| ③ | Planner & Reasoning | 解析建築、時間範圍、意圖並產生步驟 | `agent/planner.py`（規則式；介面可替換為 LLM planner） |
| ④ | Data Acquisition | 電表、氣象、日曆、活動；品質檢查 | `data/loaders.py`、`data/quality.py`、`data/simulator.py` |
| ⑤ | Analytics Engine | 天氣校正基線、殘差、異常偵測 | `analytics/baseline.py`、`analytics/detector.py` |
| ⑥ | Domain Knowledge | 知識庫檢索、診斷規則 | `knowledge/`、`diagnosis/rules.py` |
| ⑦ | Decision / Recommendation | 候選原因排序、信心、處置建議 | `diagnosis/rules.py`、`agent/tools.py` |
| ⑧ | Output & Interaction | Markdown 報告、圖、說明文字 | `report/report.py`、`agent/llm.py` |

### 2.1 資料流
```
Query → Planner(parse) → get_meter_data → quality check → fit_baseline(cached)
→ residual z → flags → incidents → check_calendar
   ├─ 與活動重疊 → expected_event（不報警）
   └─ 否 → incident_features → diagnose → retrieve_knowledge
→ Explainer → Report(+plot) → user
```

---

## 3. 資料設計

### 3.1 輸入結構
| 檔案 | 欄位 | 說明 |
|---|---|---|
| meter.csv | timestamp, 各建築代碼 | 逐時 kW |
| weather.csv | timestamp, temp_c, rh | 氣溫、濕度 |
| calendar.csv（可省略） | timestamp, is_weekend, is_holiday, is_break | 缺少時自動推導 |
| events.csv（可省略） | building, start, end, name | 已排定活動 |
| truth.csv（僅評估） | building, kind, cause, start, end | 注入異常標註 |

### 3.2 模擬資料（目前使用）
- 3 棟建築（工學院、圖書館、宿舍），逐時，2024–2025。
- 負載 = 基載 + 插座／照明（隨佔用率）+ 冷卻（隨氣溫、濕度、佔用率）+ 雜訊（對數常態 σ=2.5%）。
- 2024 為「正常」訓練期；2025 注入異常（每棟：夜間空調 3、冰水主機 1、基載異常 2、電表異常 3）與每年 8 個週末活動。
- 異常與活動互不重疊（前後各留 24 小時）。
- **限制**：模擬器與診斷規則同源，結果僅驗證流程。

### 3.3 資料品質規則
缺漏率 > 5% 降低信心並警示；零值比例 > 1%、連續 ≥ 6 小時相同讀數警示。

---

## 4. 分析設計

### 4.1 天氣校正基線
- 模型：`HistGradientBoostingRegressor`（300 次迭代，學習率 0.08）。
- 特徵：時段、星期、年週期（sin/cos）、氣溫、冷卻度時（max(T−22,0)）、3 小時平均氣溫、濕度、週末／假日／學期假期旗標。
- 保留最後 90 天估計：(a) CV(RMSE)、NMBE，與 ASHRAE Guideline 14 逐時準則比較；(b) 每小時相對殘差的穩健標準差（MAD×1.4826，下限 2%）。再以全部訓練資料重訓。
- 注意：基線**不含**活動資訊，活動由行事曆比對層處理。

### 4.2 異常偵測
- z = 相對殘差 / 該時段殘差尺度。偏高：z > 3.5；偏低：僅在相對殘差 < −50%（零讀數、停機）時算異常。
- 連續 ≥ 3 小時成為事件（z > 8 或大幅掉落的單點也保留）；間隔 1 小時的片段先接合；事件間隔 ≤ 18 小時合併。
- 18 小時的理由：只在白天出現的異常（如冷卻效率下降）夜間缺口約 14–16 小時，需合併成同一事件才能看出多日型態。此值在看過示範案例後由 12 調整，並以未見過的種子驗證。

### 4.3 行事曆比對
事件與同一棟建築已登記活動（前後 1 小時）重疊 ≥ 50% 且增量為正 → 標記為預期變化。

### 4.4 比較方法
- M1 固定門檻：訓練期平均 + 3 倍標準差。
- M2 季節剖面：月份 × 平日／假日 × 時段的中位數，不含天氣。
- M3 天氣校正基線殘差（無行事曆、無診斷）。
- M4 Energy Detective：M3 + 行事曆比對 + 診斷。

---

## 5. 診斷與領域知識設計

### 5.1 事件特徵（由工具計算）
持續時數、平均增量（kW、相對值）、正增量比例、相鄰小時相同讀數比例、近零讀數比例、最大倍數、夜間（22:00–05:59）增量占比、日間／深夜平均增量、增量與冷卻度時的相關係數。

### 5.2 規則
| 原因 | 主要條件 |
|---|---|
| meter_fault | 凍結比例 ≥ 60%；或近零比例 ≥ 50%；或 ≤ 4 小時且最大倍數 ≥ 3。分數 ≥ 0.85 時優先 |
| hvac_after_hours | 增量為正、夜間占比 ≥ 70%、持續 ≤ 16 小時 |
| base_load_leak | 增量為正、持續 ≥ 24 小時、日間／深夜增量比 0.5–2、與氣溫相關 < 0.5 |
| chiller_inefficiency | 增量為正、持續 ≥ 36 小時、日間增量 > 1.8 倍深夜 或 與氣溫相關 ≥ 0.5 |
| unexplained | 以上皆不符（分數 0.3） |

信心：分數 ≥ 0.8 且領先第二名 ≥ 0.2 → high；分數 ≥ 0.6 → medium；其餘 low。

### 5.3 知識庫
- `knowledge/kb.json`：每筆含 title、keywords、text、cause、source、url、verified。
- 檢索：TF-IDF（字元 2–3 gram）；與診斷原因相符的條目優先。可換成向量資料庫，介面不變。
- 已驗證來源：ASHRAE Guideline 14 準則（NREL、SimBuild）、UNEP/GlobalABC 建築部門占比（經新聞轉述）。其餘為作者領域筆記（`verified: false`），報告中標示「來源待驗證」。

### 5.4 LLM 的角色
`TemplateExplainer`（預設、離線、可測）。`AnthropicExplainer` 把同一份結構化證據改寫成文字，系統提示禁止新增數字或原因；失敗時退回模板。

---

## 6. Agent 設計

### 6.1 工具
`get_meter_data`、`fit_baseline`（快取）、`analyze`（偵測）、`check_calendar`、`diagnose_incident`、`retrieve_knowledge`、`build_findings`。每次呼叫寫入 trace，最終隨報告輸出。

### 6.2 Pseudocode（對應 `EnergyDetectiveAgent.ask`）
```
Algorithm 1: Energy Detective Agent
Input : User Query Q
Output: Diagnosis Report R

01: plan <- ParseQuery(Q)                      // buildings, time range, intent
02: for each building b in plan do
03:     y <- GetMeterData(b, range); dq <- QualityReport(y)
04:     model <- FitBaseline(b)                // cached, trained on normal period
05:     pred <- model.Predict(features(weather, calendar))
06:     z <- (y - pred) / pred / sigma_hour
07:     incidents <- Detect(z)                 // k=3.5, min run 3h, merge gap 18h
08:     for each incident i in incidents do
09:         if OverlapsScheduledEvent(i) then mark expected; continue
10:         f <- IncidentFeatures(y, pred, weather, i)
11:         d <- Diagnose(f)                   // rule scores, meter fault first
12:         K <- RetrieveKnowledge(d.cause)
13:         explanation <- Explainer(d, K)     // template or LLM rewording
14: R <- BuildReport(findings, baseline quality, data warnings, trace)
15: return R
```

### 6.3 例外處理
| 情況 | 處置 |
|---|---|
| 資料缺漏／品質低 | 警示並降低信心 |
| 未偵測到異常 | 回報正常並附基線品質 |
| 與活動重疊 | 標記預期變化，不報警 |
| 規則無法對應 | 回報「原因不明」，列出需補充資料 |
| LLM 不可用 | 退回模板說明 |
| 時間範圍／建築未指定 | 以假設補齊並在報告揭露 |

---

## 7. 評估設計與結果（模擬資料）

- 指標：事件層級 Precision／Recall／F1、每棟年誤報數；診斷 Top-1 準確率（僅已偵測事件）與混淆矩陣。
- 種子紀律：100–102 開發用；0–9 最終；10–19 為調整後的未見驗證。
- 結果（種子 0–9）：M1 F1 0.10、M2 0.34、M3 0.67、**M4 0.92**（Precision 0.87、Recall 0.97、每棟年誤報 1.4）；診斷準確率 **97.0%**。未見種子 10–19：M4 F1 0.92、診斷 98.5%。
- 詳見 `results/`。

### 評估的效度限制
1. 模擬器與規則同源，診斷準確率偏樂觀。
2. 注入異常的強度由作者設定，與真實分佈未必相符。
3. 未與無領域知識的 LLM 比較；未做使用者研究。

---

## 8. 風險與限制

| 風險 | 影響 | 因應 |
|---|---|---|
| 真實資料行為與模擬不同 | 精度下降 | 接入校內／公開資料重新評估，必要時調整規則與閾值 |
| 規則只涵蓋四種原因 | 其他故障被標為不明 | 擴充知識庫與規則；保留「原因不明」 |
| 知識庫來源不足 | 建議可信度有限 | 補文獻來源；未驗證者在報告中標示 |
| 低解析度電表被誤判為凍結值 | 誤判電表異常 | 依電表解析度調整凍結判斷 |
| 氣象站距離、基線誤差 | 閾值失準 | 做敏感度分析並揭露 |
| LLM 幻覺 | 說明不可信 | LLM 僅改寫、不計算；預設不用 LLM |

---

## 9. 專案結構與執行

見 `README.md`。測試：`python -m unittest discover -s tests`（21 項）。重現評估：`python scripts/run_eval.py --seeds 10`。

## 10. 里程碑

| 階段 | 狀態 |
|---|---|
| M1 資料與模擬器、品質檢查 | 完成 |
| M2 基線與比較方法 | 完成 |
| M3 Agent、知識庫、診斷、報告 | 完成（知識庫來源待補） |
| M4 模擬資料評估 | 完成 |
| M5 真實資料驗證、與無領域知識 LLM 比較、使用者評估 | 待辦 |

## 11. AI 使用紀錄
本設計與實作由 Claude 協助。作者須能解釋架構、設計理由與驗證方式。

## 12. 變更紀錄

| 日期 | 變更 | 原因 |
|---|---|---|
| 2026-10-28 | 建立初版（設計稿） | Proposal 階段 |
| 2026-10-29 | 依實作更新：模組對應、方法細節、評估結果 | 完成第一版實作 |
| 2026-10-29 | 偵測改為偏高單邊＋大幅掉落例外 | 避免把「用電偏低」當告警 |
| 2026-10-29 | 電表異常訊號強時優先於設備原因 | 夜間突波被誤判為夜間空調；符合「資料問題優先」原則；以未見種子驗證 |
| 2026-10-29 | 事件合併間隔 12→18 小時 | 日間型異常被夜間缺口切段，導致診斷失敗；以未見種子驗證 |
