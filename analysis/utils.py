"""Small shared helpers: tokenising column names, number formatting, labels."""
import re
import numpy as np

AGG_WORD = {"mean": "average", "sum": "total", "median": "median", "min": "minimum",
            "max": "maximum", "std": "standard deviation", "count": "number of rows",
            "nunique": "number of unique"}
FREQ = {"day": ("D", "daily"), "week": ("W", "weekly"), "month": ("MS", "monthly"),
        "quarter": ("QS", "quarterly"), "year": ("YS", "yearly")}


def split_tokens(name):
    """'CustomerID' -> ['customer','id'], 'unit_price' -> ['unit','price']."""
    return [t.lower() for t in re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+", str(name))]


def stem(t):
    if t.endswith("ies") and len(t) > 4:
        return t[:-3] + "y"
    if t.endswith("s") and not t.endswith("ss") and len(t) > 3:
        return t[:-1]
    return t


def name_tokens(name):
    return [stem(t) for t in split_tokens(name)]


def fmt_num(x):
    if x is None:
        return "N/A"
    if isinstance(x, (int, np.integer)):
        return f"{int(x):,}"
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if np.isnan(x):
        return "N/A"
    if abs(x) < 1e15 and x == int(x):
        return f"{int(x):,}"
    if abs(x) >= 1000:
        return f"{x:,.2f}"
    if abs(x) >= 0.01:
        return f"{x:,.3f}".rstrip("0").rstrip(".")
    return f"{x:.4g}"


def agg_label(agg, col):
    return f"{AGG_WORD.get(agg, agg)} {col}"
