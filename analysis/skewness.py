import pandas as pd


def interpret(s):
    a = abs(s)
    side = "right (positive)" if s > 0 else "left (negative)"
    if a < 0.5:
        return "approximately symmetric"
    return f"{'moderately' if a < 1 else 'highly'} skewed to the {side}"


def skew_table(df, cols):
    rows = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(s) >= 3 and s.nunique() > 1:
            sk = float(s.skew())
            rows.append({"Column": c, "Skewness": round(sk, 3), "Interpretation": interpret(sk)})
    return pd.DataFrame(rows)
