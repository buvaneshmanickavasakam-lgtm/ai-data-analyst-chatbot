
"""Visualization Decision Engine:
Picks the chart that fits the question, columns, types and data shape.
Existing chart functions are preserved; fallbacks are attempted only when needed.
"""

import pandas as pd

from analysis.query_engine import apply_filters, NO_CHART
from analysis import outliers as ol

from .bar_chart import bar, grouped_bar
from .line_chart import line
from .histogram import histogram
from .box_plot import box
from .scatter_plot import scatter
from .heatmap import heatmap
from .piechart import create_pie_chart


def _infer(profile, cols):
    """Infer a suitable chart when the query engine does not specify one."""
    cols = [c for c in cols if c in profile["columns"]]

    num = [
        c for c in cols
        if profile["columns"][c]["kind"] == "numerical"
    ]
    cat = [
        c for c in cols
        if profile["columns"][c]["kind"] in ("categorical", "boolean")
    ]
    dt = [
        c for c in cols
        if profile["columns"][c]["kind"] == "datetime"
    ]

    if len(num) >= 3 and not cat and not dt:
        return "heatmap", {"columns": num}
    if dt and num:
        return "line", {"x": dt[0], "y": num[0]}
    if len(num) == 2 and not cat:
        return "scatter", {"x": num[0], "y": num[1]}
    if cat and num:
        return "box", {"x": cat[0], "y": num[0]}
    if len(cat) >= 2:
        return "grouped_bar", {"x": cat[0], "color": cat[1]}
    if len(num) == 1:
        return "hist", {"x": num[0]}
    if len(cat) == 1:
        return "count", {"x": cat[0]}

    return None, {}


def _make_chart(df, profile, viz, kind):
    """Generate one chart without applying fallback rules."""
    cols = profile["columns"]
    title = viz.get("title")

    if kind == "pie":
        x = viz.get("x")
        if not x or x not in df.columns:
            return None, NO_CHART
        fig = create_pie_chart(df, x)
        return (fig, "pie chart") if fig is not None else (None, NO_CHART)

    if kind == "bar":
        x, y = viz.get("x"), viz.get("y")
        if x not in df.columns or y not in df.columns:
            return None, NO_CHART
        fig = bar(df, x, y, title or f"{y} by {x}")
        return (fig, "bar chart") if fig is not None else (None, NO_CHART)

    if kind == "count":
        x = viz.get("x")
        if not x or x not in df.columns:
            return None, NO_CHART
        vc = df[x].value_counts().reset_index()
        vc.columns = [x, "Count"]
        fig = bar(vc, x, "Count", title or f"Count of {x}")
        return (fig, "bar chart of category frequencies") if fig is not None else (None, NO_CHART)

    if kind == "line":
        x, y = viz.get("x"), viz.get("y")
        if x not in df.columns or y not in df.columns:
            return None, NO_CHART
        fig = line(df, x, y, title or f"{y} over time")
        return (fig, "line chart over time") if fig is not None else (None, NO_CHART)

    if kind == "hist":
        x = viz.get("x")
        if not x or x not in df.columns:
            return None, NO_CHART
        o = ol.iqr_outliers(df[x])
        if o and o["count"] > 0:
            fig = histogram(df, x, f"Distribution of {x}", "box")
            desc = "histogram with box-plot margin (outliers present)"
        else:
            fig = histogram(df, x, f"Distribution of {x}")
            desc = "histogram"
        return (fig, desc) if fig is not None else (None, NO_CHART)

    if kind == "box":
        y, x = viz.get("y"), viz.get("x")
        if not y or y not in df.columns:
            return None, NO_CHART
        if x is not None and x not in df.columns:
            return None, NO_CHART
        chart_title = title or (
            f"{y} by {x}" if x else f"Distribution of {y}"
        )
        fig = box(df, y, x, chart_title)
        desc = "box plot" + (
            " (distribution per category)"
            if x else " (outliers shown as points)"
        )
        return (fig, desc) if fig is not None else (None, NO_CHART)

    if kind == "scatter":
        x, y = viz.get("x"), viz.get("y")
        if (
            x not in cols or y not in cols
            or x not in df.columns or y not in df.columns
            or cols[x]["kind"] != "numerical"
            or cols[y]["kind"] != "numerical"
        ):
            return None, NO_CHART
        fig = scatter(
            df, x, y, title or f"{y} vs {x}", viz.get("color")
        )
        return (fig, "scatter plot with linear fit") if fig is not None else (None, NO_CHART)

    if kind == "heatmap":
        requested = viz.get("columns", profile["numerical"])
        num = [
            c for c in requested
            if c in df.columns and c in cols
            and cols[c]["kind"] == "numerical"
        ]
        if len(num) < 2:
            return None, NO_CHART
        fig, note = heatmap(df, num)
        return (fig, "correlation heatmap" + note) if fig is not None else (None, NO_CHART)

    if kind in ("grouped_bar", "stacked_bar"):
        x, color = viz.get("x"), viz.get("color")
        if (
            not x or not color
            or x not in df.columns or color not in df.columns
        ):
            return None, NO_CHART

        # The existing grouped-bar function supports the stacked option.
        # If stacked rendering fails, try the ordinary grouped bar.
        try:
            fig = grouped_bar(
                df, x, color,
                title or f"{x} by {color}",
                kind == "stacked_bar"
            )
            if fig is not None:
                return fig, (
                    "stacked bar chart" if kind == "stacked_bar"
                    else "grouped bar chart"
                )
        except Exception:
            if kind != "stacked_bar":
                raise

        if kind == "stacked_bar":
            fig = grouped_bar(
                df, x, color, title or f"{x} by {color}", False
            )
            if fig is not None:
                return fig, "grouped bar chart (fallback)"

    return None, NO_CHART


def _fallback_kinds(kind, viz, profile):
    """Return ordered, conservative fallback choices."""
    x, y = viz.get("x"), viz.get("y")
    cols = profile["columns"]

    if kind == "pie":
        return ["count", "bar"] if x else []

    if kind == "stacked_bar":
        return ["grouped_bar", "bar"]

    if kind == "grouped_bar":
        return ["bar", "count"]

    if kind == "line":
        if y and y in cols and cols[y]["kind"] == "numerical":
            return ["bar", "box", "hist"]
        return ["count", "bar"]

    if kind == "scatter":
        if y and y in cols and cols[y]["kind"] == "numerical":
            return ["box", "hist"]
        return ["hist", "count"]

    if kind == "hist":
        return ["box", "count", "bar"]

    if kind == "heatmap":
        return ["scatter", "bar"]

    if kind == "box":
        return ["hist", "count", "bar"]

    if kind == "bar":
        return ["count", "pie"]

    return []


def _fallback_viz(kind, viz, profile):
    """Adapt chart settings for an alternative chart."""
    out = dict(viz)
    cols = profile["columns"]
    x, y = out.get("x"), out.get("y")

    if kind == "count":
        category = x if x in cols else None
        if category and cols[category]["kind"] in ("categorical", "boolean"):
            out.update({"kind": "count", "x": category})
            return out
        return None

    if kind == "bar":
        if x in cols and y in cols:
            out["kind"] = "bar"
            return out
        category = next(
            (c for c in out.get("columns", [])
             if c in cols and cols[c]["kind"] in ("categorical", "boolean")),
            None
        )
        if category:
            out.update({"kind": "count", "x": category})
            return out
        return None

    if kind == "pie":
        category = x if x in cols else None
        if category and cols[category]["kind"] in ("categorical", "boolean"):
            out.update({"kind": "pie", "x": category})
            return out
        return None

    if kind == "hist":
        numeric = y if y in cols and cols[y]["kind"] == "numerical" else (
            x if x in cols and cols[x]["kind"] == "numerical" else None
        )
        if numeric:
            out.update({"kind": "hist", "x": numeric})
            return out
        return None

    if kind == "box":
        numeric = y if y in cols and cols[y]["kind"] == "numerical" else (
            x if x in cols and cols[x]["kind"] == "numerical" else None
        )
        if numeric:
            category = x if x in cols and cols[x]["kind"] in ("categorical", "boolean") else None
            out.update({"kind": "box", "x": category, "y": numeric})
            return out
        return None

    if kind == "scatter":
        requested = out.get("columns", [])
        numeric = [
            c for c in requested
            if c in cols and cols[c]["kind"] == "numerical"
        ]
        if len(numeric) >= 2:
            out.update({"kind": "scatter", "x": numeric[0], "y": numeric[1]})
            return out
        return None

    return None


def decide_chart(df, profile, viz):
    """Return (figure, description), using conservative fallbacks if needed."""
    if not viz:
        return None, NO_CHART

    viz = dict(viz)
    data = viz.get("data")
    base = (
        data
        if data is not None
        else apply_filters(df, viz.get("filters"))[0]
    )

    kind = viz.get("kind")
    if kind in (None, "auto"):
        kind, extra = _infer(profile, viz.get("columns", []))
        viz.update(extra)

    if kind is None:
        return None, NO_CHART

    try:
        fig, desc = _make_chart(base, profile, viz, kind)
        if fig is not None:
            return fig, desc
    except Exception as e:
        print(f"Chart generation error ({kind}): {e}")

    # Try fallbacks only after the requested chart fails.
    for fallback_kind in _fallback_kinds(kind, viz, profile):
        fallback_viz = _fallback_viz(fallback_kind, viz, profile)
        if fallback_viz is None:
            continue

        try:
            fig, desc = _make_chart(base, profile, fallback_viz, fallback_kind)
            if fig is not None:
                return fig, f"{desc} (fallback for {kind})"
        except Exception as e:
            print(f"Fallback chart error ({fallback_kind}): {e}")

    return None, NO_CHART
