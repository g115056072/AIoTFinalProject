"""Shared constants. Keyword map lets the planner resolve building names in queries."""

BUILDING_KEYWORDS = {
    "ENG": ["工學院", "engineering", "eng"],
    "LIB": ["圖書館", "library", "lib"],
    "DORM": ["宿舍", "dorm"],
}

# ASHRAE Guideline 14 acceptance criteria for calibrated models (hourly / monthly).
# Source: knowledge/kb.json entry "gl14_criteria".
GL14_HOURLY = {"cv_rmse": 30.0, "nmbe": 10.0}
GL14_MONTHLY = {"cv_rmse": 15.0, "nmbe": 5.0}

TRAIN_END = "2024-12-31 23:00"

CAUSES = ["hvac_after_hours", "chiller_inefficiency", "base_load_leak", "meter_fault"]

CAUSE_LABELS_ZH = {
    "hvac_after_hours": "非營運時段空調／設備運轉",
    "chiller_inefficiency": "冷卻系統（冰水主機）效率下降",
    "base_load_leak": "基載異常（設備未關閉／漏電）",
    "meter_fault": "電表或資料異常",
    "scheduled_event": "已排定活動（預期變化）",
    "unexplained": "原因不明（證據不足）",
}
