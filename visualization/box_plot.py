import plotly.express as px
from .visualization_utils import finalize, sample_df, trunc


def box(df, y, x=None, title="", max_groups=15):
    d = sample_df(df[[y] + ([x] if x else [])].dropna(), 50000).copy()
    if x:
        top = d[x].value_counts().index[:max_groups]
        d = d[d[x].isin(top)]
        d[x] = d[x].astype(str).map(trunc)
        fig = px.box(d, x=x, y=y, points="outliers")
    else:
        fig = px.box(d, y=y, points="outliers")
    return finalize(fig, title, 460, legend=False)
