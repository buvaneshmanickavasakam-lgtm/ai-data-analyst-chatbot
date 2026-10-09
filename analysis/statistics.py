import pandas as pd


def describe_numeric(df, cols):
    if not cols:
        return pd.DataFrame()
    t = df[cols].apply(pd.to_numeric, errors="coerce").describe().T
    t.index.name = "Column"
    return t.rename(columns={"50%": "median"}).reset_index()


def describe_categorical(df, cols):
    rows = []

    for c in cols:
        vc = df[c].value_counts()

        rows.append({
            "Column": str(c),
            "Unique": int(df[c].nunique()),
            "Top value": str(vc.index[0]) if len(vc) else None,
            "Top count": int(vc.iloc[0]) if len(vc) else 0
        })

    return pd.DataFrame(
        rows,
        columns=["Column", "Unique", "Top value", "Top count"]
    )
