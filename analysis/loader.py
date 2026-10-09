"""Dataset loading + light type coercion."""
import io
import warnings
import pandas as pd
from .utils import name_tokens

SUPPORTED = ("csv", "xlsx", "xls")


def load_dataset(data: bytes, filename: str, sheet=None):
    """Return (DataFrame, sheet_names, loaded_sheet). Raises ValueError for bad files."""
    ext = filename.lower().rsplit(".", 1)[-1]
    if ext not in SUPPORTED:
        raise ValueError("Unsupported file type. Please upload a CSV or Excel file.")
    sheets, loaded, df = [], None, None
    if ext == "csv":
        for enc in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                df = pd.read_csv(io.BytesIO(data), encoding=enc, sep=None, engine="python")
                break
            except UnicodeDecodeError:
                continue
    else:
        xl = pd.ExcelFile(io.BytesIO(data))
        sheets = xl.sheet_names
        loaded = sheet if sheet in sheets else sheets[0]
        df = xl.parse(loaded)
    if df is None or df.empty or df.shape[1] == 0:
        raise ValueError("The uploaded file contains no data.")
    df.columns = [str(c).strip() for c in df.columns]
    empty_unnamed = [c for c in df.columns if c.startswith("Unnamed") and df[c].isna().all()]
    df = df.drop(columns=empty_unnamed)
    return coerce_types(df), sheets, loaded


def coerce_types(df):
    """Convert numeric-looking and date-looking text columns to proper dtypes."""
    for c in df.columns:
        s = df[c]
        if not (s.dtype == object or pd.api.types.is_string_dtype(s)):
            continue
        nn = s.dropna()
        if nn.empty:
            continue
        sample = nn.sample(min(len(nn), 1000), random_state=0).astype(str).str.strip()
        if sample.str.match(r"^0\d").any():  # leading zeros -> identifier-like text
            continue
        num = pd.to_numeric(sample.str.replace(",", "", regex=False), errors="coerce")
        if num.notna().mean() >= 0.95:
            df[c] = pd.to_numeric(s.astype(str).str.replace(",", "", regex=False).str.strip(), errors="coerce")
            continue
        hint = any(t in ("date", "time", "day", "month", "year", "timestamp") for t in name_tokens(c))
        if hint or sample.str.contains(r"[-/:]").mean() > 0.9:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ok = pd.to_datetime(sample, errors="coerce").notna().mean() >= 0.9
                if ok:
                    df[c] = pd.to_datetime(s, errors="coerce")
    return df


def add_derived_columns(df):
    """If no revenue/sales column exists but Quantity and Price do, add Revenue = Quantity * Price."""
    notes = []
    toks = {c: set(name_tokens(c)) for c in df.columns}
    if any(t & {"revenue", "sale", "amount", "total"} for t in toks.values()):
        return df, notes
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    q = [c for c in num if toks[c] & {"quantity", "qty"}]
    p = [c for c in num if "price" in toks[c]]
    if q and p:
        df = df.copy()
        df["Revenue"] = df[q[0]] * df[p[0]]
        notes.append(f"Added derived column **Revenue** = {q[0]} × {p[0]} (no revenue/sales column existed).")
    return df, notes
