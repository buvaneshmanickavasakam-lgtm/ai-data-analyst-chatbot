"""Turn the dataset profile into small text documents (schema/context only; never used for calculation)."""
import pandas as pd


def build_documents(df, profile):
    docs = [{"source": "overview",
             "text": f"Dataset has {profile['n_rows']} rows and {profile['n_cols']} columns. "
                     f"Numerical columns: {', '.join(profile['numerical']) or 'none'}. "
                     f"Categorical columns: {', '.join(profile['categorical']) or 'none'}. "
                     f"Datetime columns: {', '.join(profile['datetime']) or 'none'}. "
                     f"Boolean columns: {', '.join(profile['boolean']) or 'none'}. "
                     f"Text columns: {', '.join(profile['text']) or 'none'}. "
                     f"Identifier columns: {', '.join(profile['identifier']) or 'none'}."}]
    for c, i in profile["columns"].items():
        t = f"Column '{c}' is {i['kind']} (dtype {i['dtype']}), {i['unique']} unique values, {i['missing']} missing ({i['missing_pct']}%). "
        s = df[c].dropna()
        if len(s):
            if i["kind"] == "numerical":
                t += f"Range {s.min()} to {s.max()}, mean {s.mean():.4g}, median {s.median():.4g}."
            elif i["kind"] == "datetime":
                t += f"Dates from {s.min():%Y-%m-%d} to {s.max():%Y-%m-%d}."
            elif i["kind"] in ("categorical", "boolean"):
                t += "Most frequent values: " + ", ".join(map(str, s.value_counts().index[:8])) + "."
            elif i["kind"] == "text":
                t += "Free text, e.g. " + str(s.iloc[0])[:80]
        docs.append({"source": f"column:{c}", "text": t})
    return docs
