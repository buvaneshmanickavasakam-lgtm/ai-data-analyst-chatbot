"""IQR (Tukey fence) outlier detection."""
import pandas as pd


def iqr_outliers(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 4:
        return None
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = (s < lo) | (s > hi)
    return {"count": int(mask.sum()), "pct": float(mask.mean() * 100), "lower": float(lo), "upper": float(hi),
            "q1": float(q1), "q3": float(q3), "n": len(s), "values": s[mask]}


def outlier_table(df, cols):
    rows = []
    for c in cols:
        r = iqr_outliers(df[c])
        if r:
            rows.append({"Column": c, "Outliers": r["count"], "Outlier %": round(r["pct"], 2),
                         "Lower fence": round(r["lower"], 3), "Upper fence": round(r["upper"], 3)})
    return pd.DataFrame(rows)
