import pandas as pd


def missing_table(df):
    m = df.isna().sum()
    t = pd.DataFrame({"Column": m.index, "Missing": m.values,
                      "Missing %": (m.values / max(len(df), 1) * 100).round(2)})
    return t.sort_values("Missing", ascending=False).reset_index(drop=True)
