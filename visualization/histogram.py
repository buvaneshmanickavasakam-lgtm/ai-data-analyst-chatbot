import plotly.express as px
from .visualization_utils import finalize, sample_df, big_numbers


def histogram(df, x, title, marginal=None):
    d = sample_df(df[[x]].dropna(), 100000)
    fig = px.histogram(d, x=x, nbins=min(60, max(10, int(d[x].nunique() ** 0.5 * 2))), marginal=marginal)
    fig.update_layout(bargap=0.05)
    big_numbers(fig, "x", d[x].abs().max())
    return finalize(fig, title, 460 if marginal else 400, legend=False)
