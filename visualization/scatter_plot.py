import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from .visualization_utils import finalize, sample_df


def scatter(df, x, y, title, color=None):
    d = df[[x, y] + ([color] if color else [])].dropna()
    fit = None
    if len(d) > 2 and d[x].nunique() > 1:
        m, b = np.polyfit(d[x], d[y], 1)
        xs = np.array([d[x].min(), d[x].max()])
        fit = (xs, m * xs + b)
    d = sample_df(d, 5000)
    fig = px.scatter(d, x=x, y=y, color=color, opacity=0.6)
    if fit is not None:
        fig.add_trace(go.Scatter(x=fit[0], y=fit[1], mode="lines", name="Linear fit", line=dict(color="#E45756", width=2)))
    return finalize(fig, title, 460, legend=fit is not None or color is not None)
