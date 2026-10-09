"""Controlled analysis engine.

The question parser (rule-based or LLM) produces a *plan* (a small dict). This module validates the
plan against the dataset and executes it with Pandas/NumPy. The LLM never computes numbers.
Every handler returns {"ok", "text", "table", "viz"}.
"""
import re
from collections import Counter
import numpy as np
import pandas as pd
from .utils import fmt_num, agg_label, FREQ
from . import outliers as ol, skewness as sk, correlation as co, duplicates as du, missing_values as mv
from . import statistics as stt

NOT_IN_DATA = "I cannot find that information in the uploaded dataset."
UNRELATED = "I can only answer questions based on the currently uploaded dataset."
UNRELIABLE = "I cannot determine this reliably from the available data."
NO_CHART = "I cannot generate a meaningful chart for that question."


def col_not_found(term):
    return f"I cannot find a column related to '{term}' in this dataset."


def R(text, table=None, viz=None, ok=True):
    return {"ok": ok, "text": text, "table": table, "viz": viz}


def fail(text):
    return R(text, ok=False)


def apply_filters(df, filters):
    if not filters:
        return df, ""
    d, parts = df, []
    for c, v in filters:
        d = d[d[c].astype(str).str.lower() == str(v).lower()]
        parts.append(f"{c} = {v}")
    return d, f"*Filtered to {', '.join(parts)} ({len(d):,} rows).*"


def need_numeric(p, c):
    if c not in p["columns"]:
        return col_not_found(c)
    k = p["columns"][c]["kind"]
    if k != "numerical":
        return f"'{c}' is a {k} column, so a numerical calculation on it is not meaningful."
    return None


def _num(d, c):
    return pd.to_numeric(d[c], errors="coerce")


def _miss_note(d, c):
    m = int(d[c].isna().sum())
    return f", ignoring {m:,} missing value(s)" if m else ""


# ------------------------------------------------------------------ dataset level
def h_row_count(d, p, plan, ctx):
    return R(f"The dataset has **{len(d):,} rows** and **{p['n_cols']} columns**.")


def h_columns(d, p, plan, ctx):
    t = pd.DataFrame([{"Column": c, "Type": i["kind"], "Missing": i["missing"]} for c, i in p["columns"].items()])
    return R(f"The dataset has **{p['n_cols']} columns**:", t)


def h_overview(d, p, plan, ctx):
    kinds = ", ".join(f"{len(p[k])} {k}" for k in ("numerical", "categorical", "datetime", "boolean", "text", "identifier") if p[k])
    t = pd.DataFrame([{"Column": c, "Type": i["kind"], "Unique": i["unique"], "Missing": i["missing"]} for c, i in p["columns"].items()])
    txt = (f"The dataset has **{p['n_rows']:,} rows** and **{p['n_cols']} columns** ({kinds}). "
           f"It contains **{p['missing_total']:,} missing cells** and **{p['duplicates']:,} duplicate rows**.")
    return R(txt, t)


def h_missing(d, p, plan, ctx):
    cols = [c for c in (plan.get("columns") or []) if c in p["columns"]]
    t = mv.missing_table(d)
    if cols:
        t = t[t["Column"].isin(cols)]
        parts = [f"'{r[0]}' has {int(r[1]):,} missing value(s) ({r[2]}%)" for r in t.itertuples(index=False)]
        return R("; ".join(parts) + ".", t.reset_index(drop=True),
                 {"kind": "bar", "x": "Column", "y": "Missing", "data": t, "title": "Missing values"})
    nz = t[t["Missing"] > 0].reset_index(drop=True)
    if nz.empty:
        return R("There are **no missing values** in this dataset.")
    names = ", ".join(f"'{c}'" for c in nz["Column"].head(10))
    return R(f"**{len(nz)} of {d.shape[1]} columns** contain missing values ({int(nz['Missing'].sum()):,} missing cells in total): {names}"
             f"{'…' if len(nz) > 10 else ''}.", nz, {"kind": "bar", "x": "Column", "y": "Missing", "data": nz.head(25), "title": "Missing values by column"})


def h_duplicates(d, p, plan, ctx):
    r = du.duplicate_info(d)
    if r["count"] == 0:
        return R("There are **no duplicate rows** in this dataset.")
    return R(f"There are **{r['count']:,} duplicate rows** ({r['pct']:.2f}% of {len(d):,} rows). Examples:", r["examples"])


# ------------------------------------------------------------------ numeric column analysis
def h_outliers(d, p, plan, ctx):
    c = plan.get("column")
    if not c:
        cols = p["numerical"]
        if not cols:
            return fail("There are no numerical columns, so I cannot detect outliers.")
        t = ol.outlier_table(d, cols)
        if t.empty:
            return fail(UNRELIABLE)
        return R("Outliers per numerical column (IQR method, 1.5×IQR fences):", t,
                 {"kind": "bar", "x": "Column", "y": "Outliers", "data": t, "title": "Outliers per numerical column"})
    err = need_numeric(p, c)
    if err:
        return fail(err)
    r = ol.iqr_outliers(d[c])
    if r is None:
        return fail(UNRELIABLE + " (too few valid values)")
    if r["count"] == 0:
        txt = f"No outliers found in **{c}** (IQR method; fences {fmt_num(r['lower'])} to {fmt_num(r['upper'])})."
        tab = None
    else:
        v = r["values"]
        txt = (f"**{r['count']:,} outliers** found in **{c}** ({r['pct']:.2f}% of {r['n']:,} values). Normal range by the IQR method: "
               f"{fmt_num(r['lower'])} to {fmt_num(r['upper'])}. Outlier values range from {fmt_num(v.min())} to {fmt_num(v.max())}.")
        tab = pd.DataFrame({c: v.sort_values(key=lambda x: (x - r['q1']).abs(), ascending=False).head(10).values})
    return R(txt, tab, {"kind": "box", "y": c, "title": f"Outliers in {c}"})


def h_skewness(d, p, plan, ctx):
    c = plan.get("column")
    cols = [c] if c else p["numerical"]
    if c:
        err = need_numeric(p, c)
        if err:
            return fail(err)
    t = sk.skew_table(d, cols)
    if t.empty:
        return fail(UNRELIABLE)
    if c:
        r = t.iloc[0]
        return R(f"**{c}** has skewness **{r['Skewness']}** — {r['Interpretation']}.", None, {"kind": "hist", "x": c})
    return R("Skewness of numerical columns:", t, {"kind": "bar", "x": "Column", "y": "Skewness", "data": t, "title": "Skewness by column"})


def h_stat(d, p, plan, ctx):
    c, agg = plan.get("column"), plan.get("agg") or "mean"
    err = need_numeric(p, c)
    if err:
        return fail(err)
    s = _num(d, c).dropna()
    if len(s) < (2 if agg == "std" else 1):
        return fail(UNRELIABLE)
    val = getattr(s, agg)()
    label = agg_label(agg, c)
    return R(f"The {label} is **{fmt_num(val)}** (based on {len(s):,} valid values{_miss_note(d, c)}).", None, {"kind": "hist", "x": c})


def h_top_n(d, p, plan, ctx):
    column = plan.get("column")
    entity_col = plan.get("entity_col")
    n = plan.get("n")
    rank = plan.get("rank")

    if column not in p["columns"]:
        return fail(col_not_found(column))

    if entity_col not in d.columns:
        return fail(col_not_found(entity_col))

    err = need_numeric(p, column)
    if err:
        return fail(err)

    if not isinstance(n, int) or n < 1:
        return fail("Please specify a valid number of records.")

    ascending = rank == "asc"

    # Sort the actual rows; do not group or aggregate customers.
    result = d.copy()
    result[column] = _num(result, column)
    result = result.dropna(subset=[column])
    result = result.sort_values(
        column,
        ascending=ascending,
        kind="stable",
    )

    result = result[[entity_col, column]].head(n).reset_index(drop=True)

    if result.empty:
        return fail(UNRELIABLE)

    direction = "lowest" if ascending else "highest"
    return R(
        f"Showing the {min(n, len(result))} records with the "
        f"{direction} {column}:",
        result,
    )



def h_top_n(d, p, plan, ctx):
    column = plan.get("column")
    entity_col = plan.get("entity_col")
    n = plan.get("n")
    rank = plan.get("rank")

    # Validate the requested columns.
    if column not in d.columns:
        return fail(col_not_found(column or "that"))

    if entity_col not in d.columns:
        return fail(col_not_found(entity_col or "that"))

    # Validate the ranking column is numeric.
    err = need_numeric(p, column)
    if err:
        return fail(err)

    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        return fail("Please specify a valid number of records.")

    if rank not in ("asc", "desc"):
        return fail("Please specify whether you want the highest or lowest records.")

    # Apply the same filters as other query handlers.
    result, filter_err = apply_filters(d, plan.get("filters"))
    if filter_err:
        return fail(filter_err)

    # Convert the ranking column safely and exclude missing values.
    result = result.copy()
    result[column] = _num(result, column)
    result = result.dropna(subset=[column])

    if result.empty:
        return fail("No valid records were found for this query.")


    # Stable sort makes ties deterministic where possible.
    result = result.sort_values(
        by=column,
        ascending=(rank == "asc"),
        kind="stable",
    )

    # Keep the identifying column and the ranking value.
    result = result[[entity_col, column]].head(n).reset_index(drop=True)

    direction = "lowest" if rank == "asc" else "highest"
    return R(
        f"Showing the {len(result)} records with the {direction} {column}:",
        result,
    )


def h_describe_column(d, p, plan, ctx):
    c = plan.get("column")
    if c not in p["columns"]:
        return fail(col_not_found(c))
    kind = p["columns"][c]["kind"]
    if kind == "datetime":
        s = d[c].dropna()
        if s.empty:
            return fail(UNRELIABLE)
        return R(f"**{c}** ranges from **{s.min():%Y-%m-%d}** to **{s.max():%Y-%m-%d}** ({(s.max() - s.min()).days:,} days, {len(s):,} valid values).")
    if kind != "numerical":
        return h_value_counts(d, p, {**plan, "column": c}, ctx)
    s = _num(d, c).dropna()
    if len(s) < 3:
        return fail(UNRELIABLE)
    t = stt.describe_numeric(d, [c])
    extra = ""
    if s.nunique() > 1:
        extra += f" Skewness is {s.skew():.2f} ({sk.interpret(float(s.skew()))})."
    o = ol.iqr_outliers(s)
    if o:
        extra += f" {o['count']:,} outliers by the IQR rule."
    return R(f"Summary of **{c}**: mean {fmt_num(s.mean())}, median {fmt_num(s.median())}, min {fmt_num(s.min())}, max {fmt_num(s.max())}.{extra}",
             t, {"kind": "hist", "x": c})


def h_correlation(d, p, plan, ctx):
    cols = plan.get("columns") or []

    # Explicit request for a correlation matrix / heatmap
    if plan.get("chart_type") == "heatmap" or len(cols) < 2:
        num = [
            c for c in p["numerical"]
            if c in d.columns and d[c].nunique() > 1
        ]

        if len(num) < 2:
            return fail(
                "I need at least two varying numerical columns "
                "to generate a correlation heatmap."
            )

        m = co.corr_matrix(d, num)
        t = co.top_pairs(m)

        txt = "Correlation matrix of numerical columns computed."
        if not t.empty:
            a = t.iloc[0]
            txt = (
                f"Strongest correlation: **{a['Column A']}** and "
                f"**{a['Column B']}** (r = {a['Correlation']}, "
                f"{a['Strength']})."
            )

        return R(
            txt,
            t,
            {"kind": "heatmap", "columns": num}
        )

    # Pairwise correlation: retain the existing scatter-plot behavior
    a, b = cols[:2]

    for c in (a, b):
        err = need_numeric(p, c)
        if err:
            return fail(err)

    r, n = co.pair_corr(d, a, b)

    if r is None:
        return fail(
            UNRELIABLE +
            " (not enough valid, varying values)."
        )

    return R(
        f"The Pearson correlation between **{a}** and **{b}** "
        f"is **{r:.3f}** ({co.strength(r)}), based on "
        f"{n:,} complete pairs. Correlation does not imply causation.",
        None,
        {"kind": "scatter", "x": a, "y": b}
    )

# ------------------------------------------------------------------ grouping / counting
def h_groupby(d, p, plan, ctx):
    g, agg, val, ent = plan.get("group_col"), plan.get("agg") or "count", plan.get("value_col"), plan.get("entity_col")
    if g not in p["columns"]:
        return fail(col_not_found(g))
    gk = p["columns"][g]["kind"]
    if gk == "identifier" or (gk == "numerical" and p["columns"][g]["unique"] > 30):
        return fail(f"Grouping by '{g}' is not meaningful (it has too many distinct values / is an identifier).")
    if agg == "nunique":
        if ent not in p["columns"]:
            return fail(col_not_found(ent))
        s = d.groupby(g, observed=True)[ent].nunique()
        label = f"number of unique {ent}"
    elif agg == "count":
        s = d.groupby(g, observed=True).size()
        label = "number of rows"
    else:
        err = need_numeric(p, val)
        if err:
            return fail(err)
        tmp = pd.DataFrame({g: d[g], val: _num(d, val)})
        s = tmp.groupby(g, observed=True)[val].agg(agg)
        label = agg_label(agg, val)
    s = s.dropna()
    if s.empty:
        return fail(UNRELIABLE)
    rank, n = plan.get("rank"), plan.get("n")
    asc = rank == "asc"
    s = s.sort_values(ascending=asc, kind="stable")
    t = s.reset_index()
    t.columns = [g, label]
    top = t.iloc[0]
    
    # Preserve existing relationship behavior and support explicit chart requests.
    chart_type = plan.get("chart_type")

    if plan.get("relation") or chart_type == "box":
        if not val:
            return fail(
                "A box plot needs a numerical column to compare across categories."
            )

        err = need_numeric(p, val)
        if err:
            return fail(err)

        return R(
            f"{val} by {g}: distribution across categories.",
            t.head(30),
            {
                "kind": "box",
                "x": g,
                "y": val,
                "title": f"{val} by {g}",
            },
        )

    if rank:
        word = "lowest" if asc else "highest"
        tied = t[t[label] == top[label]][g].tolist()
        if len(tied) > 1:
            txt = f"{len(tied)} values of {g} tie for the {word} {label} (**{fmt_num(top[label])}**): {', '.join(map(str, tied[:6]))}."
        else:
            txt = f"**{top[g]}** has the {word} {label}: **{fmt_num(top[label])}**."
        if plan.get("inherited"):
            txt += f" _(Using “{label}” from your previous question.)_"
    else:
        txt = f"{label.capitalize()} by **{g}**:"
    show = t.head(n or (10 if rank else 50))
    
    # Keep bar charts as the default; honor supported explicit chart types.
    supported_chart_types = {
        "bar", "line", "hist", "box", "scatter",
        "heatmap", "pie", "grouped_bar", "stacked_bar",
    }

    if chart_type in supported_chart_types:
        if chart_type == "box":
            # Use raw filtered rows for a true box plot.
            viz = {
                "kind": "box",
                "x": g,
                "y": val,
                "title": f"{val} by {g}",
            }
        else:
            viz = {
                "kind": chart_type,
                "x": g,
                "y": label,
                "data": t.head(n or 15),
                "title": f"{label.capitalize()} by {g}",
            }
    else:
        viz = {
            "kind": "bar",
            "x": g,
            "y": label,
            "data": t.head(n or 15),
            "title": f"{label.capitalize()} by {g}",
        }

    return R(txt, show, viz)


def h_count_unique(d, p, plan, ctx):
    c = plan.get("column")
    if c not in p["columns"]:
        return fail(col_not_found(c))
    n = d[c].nunique()
    return R(f"There are **{n:,} unique** values in '{c}' ({int(d[c].notna().sum()):,} non-missing rows).")


def h_value_counts(d, p, plan, ctx):
    c = plan.get("column")
    if c not in p["columns"]:
        return fail(col_not_found(c))
    vc = d[c].value_counts(dropna=True)
    if vc.empty:
        return fail(UNRELIABLE)
    asc = plan.get("rank") == "asc"
    if asc:
        vc = vc.sort_values(ascending=True, kind="stable")
    t = pd.DataFrame({c: vc.index, "Count": vc.values, "Percent": (vc.values / vc.sum() * 100).round(2)})
    word = "least common" if asc else "most common"
    txt = f"The {word} value of **{c}** is **{t.iloc[0][c]}** ({int(t.iloc[0]['Count']):,} rows, {t.iloc[0]['Percent']}%)."
    return R(txt, t.head(plan.get("n") or 15), {"kind": "bar", "x": c, "y": "Count", "data": t.head(plan.get("n") or 15), "title": f"{c}: frequency"})


STOPWORDS = set("the and for are but not you all any can had her was one our out has have this that with from they will would there their what about which when your been were them than then into only also very just its it's i'm don't did does too more most some such".split())


def h_text_words(d, p, plan, ctx):
    c = plan.get("column")
    if c not in p["columns"]:
        return fail(col_not_found(c))
    words = Counter()
    for t in d[c].dropna().astype(str).head(200000):
        words.update(w for w in re.findall(r"[a-z']{3,}", t.lower()) if w not in STOPWORDS)
    if not words:
        return fail(UNRELIABLE)
    t = pd.DataFrame(words.most_common(20), columns=["Word", "Count"])
    return R(f"Most frequent words in **{c}** (common stop-words removed). This is a word-frequency count, not topic classification:",
             t, {"kind": "bar", "x": "Word", "y": "Count", "data": t, "title": f"Most frequent words in {c}"})


def h_timeseries(d, p, plan, ctx):
    dc, val, agg = plan.get("date_col"), plan.get("value_col"), plan.get("agg") or ("sum" if plan.get("value_col") else "count")
    if dc not in p["columns"] or p["columns"][dc]["kind"] != "datetime":
        return fail(f"I cannot find a usable date/time column for a time-based analysis.")
    s = d[[dc] + ([val] if val else [])].dropna(subset=[dc])
    if s.empty:
        return fail(UNRELIABLE)
    freq = plan.get("freq")
    if freq not in FREQ:
        span = (s[dc].max() - s[dc].min()).days
        freq = "month" if span >= 180 else "day"
    rule, word = FREQ[freq]
    ts = s.set_index(dc)
    if val:
        err = need_numeric(p, val)
        if err:
            return fail(err)
        ser = _num(ts, val).resample(rule).agg(agg)
        label = agg_label(agg, val)
    else:
        ser = ts.iloc[:, 0:0].assign(_n=1)["_n"].resample(rule).sum()
        label = "number of rows"
    if agg in ("mean", "median", "min", "max", "std"):
        ser = ser.dropna()
    if len(ser) < 2:
        return fail(UNRELIABLE + " (fewer than two time periods).")
    t = ser.reset_index()
    t.columns = ["Period", label]
    hi, lo = t.loc[t[label].idxmax()], t.loc[t[label].idxmin()]
    pf = "%Y-%m" if freq == "month" else "%Y-%m-%d" if freq in ("day", "week") else "%Y" if freq == "year" else "%Y-%m"
    txt = (f"{word.capitalize()} {label} over {len(t)} periods. Peak: **{hi['Period']:{pf}}** ({fmt_num(hi[label])}); "
           f"lowest: **{lo['Period']:{pf}}** ({fmt_num(lo[label])}).")
    return R(txt, t, {"kind": "line", "x": "Period", "y": label, "data": t, "title": f"{word.capitalize()} {label}"})


def h_churn_rate(d, p, plan, ctx):
    c = plan.get("column")
    if c not in p["columns"]:
        return fail(col_not_found(c))
    sv = d[c].dropna().astype(str).str.strip().str.lower()
    pos = {"yes", "true", "1", "y", "t", "churned", "left", "1.0"}
    uniq = set(sv.unique())
    if sv.empty or not (uniq & pos) or len(uniq) > 3:
        return fail(f"{UNRELIABLE} I could not tell which value of '{c}' means 'churned' (values: {', '.join(sorted(uniq)[:6])}).")
    rate = sv.isin(pos).mean() * 100
    vc = sv.value_counts()
    t = pd.DataFrame({c: vc.index, "Count": vc.values, "Percent": (vc.values / vc.sum() * 100).round(2)})
    return R(f"The churn rate is **{rate:.2f}%** ({int(sv.isin(pos).sum()):,} of {len(sv):,} rows with a recorded '{c}' value).", t,
             {"kind": "bar", "x": c, "y": "Count", "data": t, "title": f"{c}: churned vs retained"})


def h_explain(d, p, plan, ctx):
    if not ctx or not ctx.get("ok") or not ctx.get("plan"):
        return fail("There is no previous analysis for me to break down. Ask a question first.")
    lp, t = ctx["plan"], ctx.get("table")
    if lp.get("intent") != "groupby" or t is None or t.empty or lp.get("relation"):
        return fail("I can only break down grouped results (e.g. “which X has the highest Y”). The data alone cannot tell me the cause of other results.")
    g = lp["group_col"]
    top = t.iloc[0][g]
    d2, _ = apply_filters(d, lp.get("filters"))
    m = d2[g].astype(str) == str(top)
    lines = [f"**{top}** accounts for {int(m.sum()):,} of {len(d2):,} rows ({m.mean() * 100:.1f}%)."]
    v, agg = lp.get("value_col"), lp.get("agg")
    if v and agg in ("sum", "mean", "median"):
        x = _num(d2, v)
        lines.append(f"Within {top}, the average {v} per row is {fmt_num(x[m].mean())} vs {fmt_num(x.mean())} overall.")
        if agg == "sum" and x.sum():
            lines.append(f"{top} contributes {x[m].sum() / x.sum() * 100:.1f}% of total {v}.")
    ent = lp.get("entity_col")
    if ent and ent in d2:
        lines.append(f"{top} has {d2.loc[m, ent].nunique():,} unique {ent} out of {d2[ent].nunique():,} overall.")
    lines.append("_These figures describe what the data shows; the data alone cannot establish the underlying cause._")
    return R("Breakdown of the previous result:\n\n- " + "\n- ".join(lines))


def h_not_found(d, p, plan, ctx):
    return fail(col_not_found(plan.get("term") or "that"))


def h_unrelated(d, p, plan, ctx):
    return fail(UNRELATED)


def h_clarify(d, p, plan, ctx):
    return fail(f"{UNRELIABLE} Please say what to measure, e.g. “average <numeric column> by {plan.get('group_col', 'category')}”.")


def h_churn_predict(d, p, plan, ctx):
    return fail(f"{UNRELIABLE} Predicting which individual customers will churn needs a trained model, which this analyst does not "
                "build. I can report the churn rate, or compare groups, e.g. “average <column> by churn”.")


HANDLERS = {
    "row_count": h_row_count, "columns": h_columns, "overview": h_overview, "missing": h_missing,
    "duplicates": h_duplicates, "outliers": h_outliers, "skewness": h_skewness, "stat": h_stat,
    "describe_column": h_describe_column, "correlation": h_correlation, "groupby": h_groupby,
    "count_unique": h_count_unique, "value_counts": h_value_counts, "text_words": h_text_words,
    "timeseries": h_timeseries, "churn_rate": h_churn_rate, "churn_predict": h_churn_predict,
    "explain": h_explain, "column_not_found": h_not_found, "unrelated": h_unrelated, "clarify": h_clarify,"top_n": h_top_n,
}


def execute(df, profile, plan, ctx=None):
    """Validate + run a plan. Returns result dict (always includes the plan)."""
    fn = HANDLERS.get(plan.get("intent"))
    if fn is None:
        return {**fail(NOT_IN_DATA), "plan": plan}
    d, note = apply_filters(df, plan.get("filters"))
    if d.empty:
        return {**fail("No rows match the requested filter, so I cannot answer this."), "plan": plan}
    try:
        r = fn(d, profile, plan, ctx)
    except Exception:  # never surface a guess; fail closed
        r = fail(UNRELIABLE)
    if note and r["ok"]:
        r["text"] = note + "\n\n" + r["text"]
    if r.get("viz") is not None:
        r["viz"]["filters"] = plan.get("filters")
    r["plan"] = plan
    return r
