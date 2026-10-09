import pandas as pd


def cardinality_table(df):
    n = max(len(df), 1)
    u = df.nunique(dropna=True)
    t = pd.DataFrame({"Column": u.index, "Unique": u.values, "Unique %": (u.values / n * 100).round(2)})
    return t.sort_values("Unique", ascending=False).reset_index(drop=True)
