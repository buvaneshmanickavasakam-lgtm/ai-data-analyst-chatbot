def duplicate_info(df):
    mask = df.duplicated(keep="first")
    n = int(mask.sum())
    return {"count": n, "pct": n / max(len(df), 1) * 100, "examples": df[mask].head(10)}
