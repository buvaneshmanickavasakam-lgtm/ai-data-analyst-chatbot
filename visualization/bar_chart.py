import pandas as pd
import plotly.express as px
from .visualization_utils import finalize, trunc, big_numbers


def bar(data, x, y, title, max_items=25):
    d = data.head(max_items).copy()
    d["_label"] = d[x].map(trunc)
    long = d["_label"].str.len().max() > 12 or len(d) > 8
    if long:  # horizontal bars read better with many / long labels
        fig = px.bar(d, x=y, y="_label", orientation="h", hover_data={x: True, "_label": False},
                     labels={"_label": x}, text_auto=".3s" if d[y].abs().max() >= 1e4 else True)
        fig.update_yaxes(autorange="reversed", title=x, categoryorder="array", categoryarray=d["_label"].tolist())
        big_numbers(fig, "x", d[y].abs().max())
        h = max(320, min(900, 120 + 28 * len(d)))
    else:
        fig = px.bar(d, x="_label", y=y, hover_data={x: True, "_label": False}, labels={"_label": x},
                     text_auto=".3s" if d[y].abs().max() >= 1e4 else True)
        big_numbers(fig, "y", d[y].abs().max())
        h = 420
    fig.update_traces(textposition="outside", cliponaxis=False)
    return finalize(fig, title, h, legend=False)


def grouped_bar(df, x, color, title, stacked=False, max_levels=12):
    ct = pd.crosstab(df[x], df[color])
    ct = ct.loc[ct.sum(axis=1).sort_values(ascending=False).index[:max_levels], ct.sum().sort_values(ascending=False).index[:8]]
    long = ct.reset_index().melt(id_vars=x, value_name="Count", var_name=color)
    long[x] = long[x].map(trunc)
    fig = px.bar(long, x=x, y="Count", color=color, barmode="stack" if stacked else "group")
    return finalize(fig, title, 460)
