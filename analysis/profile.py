"""Automatic dataset profiling and column-type classification."""
import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype
from .utils import split_tokens

ID_TOKENS = {"id", "no", "num", "number", "code", "key", "uuid", "guid", "sku"}
BOOL_STR = {"yes", "no", "true", "false", "y", "n", "t", "f"}
KINDS = ("numerical", "categorical", "datetime", "boolean", "text", "identifier")


def classify_column(s, name):
    nn = s.dropna()
    n = len(nn)
    if n == 0:
        return "categorical"
    toks = split_tokens(name)
    hint = bool(toks) and toks[-1] in ID_TOKENS
    nun = nn.nunique()
    ratio = nun / n
    if is_bool_dtype(s):
        return "boolean"
    if is_datetime64_any_dtype(s):
        return "datetime"
    if is_numeric_dtype(s):
        if nun <= 2 and set(nn.unique()) <= {0, 1}:
            return "boolean"
        intlike = bool((nn % 1 == 0).all())
        if hint and intlike:
            return "identifier"
        if intlike and ratio == 1 and n >= 20 and nn.is_monotonic_increasing:
            return "identifier"
        return "numerical"
    sv = nn.astype(str).str.strip().str.lower()
    if nun <= 2 and set(sv.unique()) <= BOOL_STR:
        return "boolean"
    if hint and nun >= 20:
        return "identifier"
    smp = sv.sample(min(n, 20000), random_state=0)
    avg_len, avg_words = smp.str.len().mean(), smp.str.split().str.len().mean()
    if avg_len > 50 or (ratio > 0.5 and avg_words >= 3):
        return "text"
    return "categorical"


def build_profile(df):
    cols, groups = {}, {k: [] for k in KINDS}
    n = len(df)
    for c in df.columns:
        kind = classify_column(df[c], c)
        miss = int(df[c].isna().sum())
        cols[c] = {"kind": kind, "dtype": str(df[c].dtype), "missing": miss,
                   "missing_pct": round(miss / max(n, 1) * 100, 2), "unique": int(df[c].nunique())}
        groups[kind].append(c)
    return {"n_rows": n, "n_cols": df.shape[1], "columns": cols, **groups,
            "missing_total": int(df.isna().sum().sum()), "duplicates": int(df.duplicated().sum()),
            "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2)}


def column_report(df, profile):
    n = max(len(df), 1)
    rows = []
    for c, i in profile["columns"].items():
        sk = None
        if i["kind"] == "numerical":
            s = pd.to_numeric(df[c], errors="coerce").dropna()
            sk = round(float(s.skew()), 3) if len(s) > 2 and s.nunique() > 1 else None
        rows.append({"Column": c, "Type": i["kind"], "Dtype": i["dtype"], "Missing": i["missing"],
                     "Missing %": i["missing_pct"], "Unique": i["unique"],
                     "Unique %": round(i["unique"] / n * 100, 2), "Skewness": sk})
    return pd.DataFrame(rows)
