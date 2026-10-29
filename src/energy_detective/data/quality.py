"""Data quality checks that run before analytics (principle: be honest about data)."""
from __future__ import annotations

import pandas as pd


def quality_report(y: pd.Series, flat_run: int = 6) -> dict:
    n = len(y)
    missing = float(y.isna().mean()) if n else 1.0
    zeros = float((y.fillna(1) <= 0).mean()) if n else 0.0
    same = (y.diff() == 0).to_numpy()
    longest = cur = 0
    for v in same:
        cur = cur + 1 if v else 0
        longest = max(longest, cur)
    warnings = []
    if missing > 0.05:
        warnings.append(f"資料缺漏率 {missing:.1%} 超過 5%，信心降低")
    if zeros > 0.01:
        warnings.append(f"零值比例 {zeros:.1%}，可能為電表或通訊問題")
    if longest >= flat_run:
        warnings.append(f"出現連續 {longest + 1} 小時相同讀數（疑似凍結值）")
    return {"n": n, "missing_frac": missing, "zero_frac": zeros,
            "longest_flat_run": int(longest), "warnings": warnings,
            "low_quality": missing > 0.05}
