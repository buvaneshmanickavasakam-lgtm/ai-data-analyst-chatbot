import numpy as np
import pandas as pd


def strength(r):
    a = abs(r)
    word = "very weak" if a < 0.2 else "weak" if a < 0.4 else "moderate" if a < 0.6 else "strong" if a < 0.8 else "very strong"
    return f"{word} {'positive' if r > 0 else 'negative'}"


def pair_corr(df, a, b):
    d = df[[a, b]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(d) < 3 or d[a].nunique() < 2 or d[b].nunique() < 2:
        return None, len(d)
    return float(d[a].corr(d[b])), len(d)


def corr_matrix(df, cols):
    cols = [c for c in cols if df[c].nunique() > 1]
    return df[cols].apply(pd.to_numeric, errors="coerce").corr()


def top_pairs(corr, n=10):
    rows, cols = [], list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = corr.iloc[i, j]
            if not np.isnan(v):
                rows.append({"Column A": cols[i], "Column B": cols[j], "Correlation": round(float(v), 3),
                             "Strength": strength(v)})
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    return t.reindex(t["Correlation"].abs().sort_values(ascending=False).index).head(n).reset_index(drop=True)
